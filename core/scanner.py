import gc
import numpy as np
from tqdm import tqdm
from utils import helpers
import jax
import jax.numpy as jnp

def run_scan(args, coords, model, params, state, wt_seq):
    target_indices = helpers.find_track_indices(model, args.tissue, args.assay)
    
    indices = list(range(0, len(wt_seq) - args.mutation_size, args.step_size))
    if args.exclude_gene_body and 'gene_start_rel' in coords:
        g_start = coords['gene_start_rel']
        g_end = coords['gene_end_rel']
        indices = [i for i in indices if not (i < g_end and (i + args.mutation_size) > g_start)]

    print("Compiling JIT Kernel...")
    @jax.jit
    def fast_predict(p, s, inputs, o, n):
        # 1. Input Translation: Match the BFloat16 weights
        inputs_bf16 = jnp.array(inputs, dtype=jnp.bfloat16)
        
        # 2. Heavy Math: Runs safely in VRAM without overflow
        preds = model._predict(p, s, inputs_bf16, o, negative_strand_mask=n, strand_reindexing=None)
        
        # 3. Output Translation: Cast back to Float32 so standard NumPy can read it!
        return jax.tree_util.tree_map(
            lambda x: x.astype(jnp.float32) if getattr(x, 'dtype', None) == jnp.bfloat16 else x, 
            preds
        )

    print("Calculating Baseline (JIT Optimized)...")
    
    wt_encoded = helpers.one_hot_encode(wt_seq)[jnp.newaxis, ...]
    if args.batch_size > 1:
        pad = jnp.tile(wt_encoded, (args.batch_size - 1, 1, 1))
        batch_input = jnp.concatenate([wt_encoded, pad], axis=0)
    else:
        batch_input = wt_encoded

    base_preds = fast_predict(
        params, state, batch_input, 
        jnp.zeros((args.batch_size,), dtype=jnp.int32), 
        jnp.zeros((args.batch_size,), dtype=bool)
    )
    
    head_key = helpers.get_head_key(base_preds, args.assay)
    print(f"   [i] Selected Head: '{head_key}'")
    
    base_data = np.array(base_preds[head_key])
    
    if target_indices == "ALL":
        print(f"   [i] Blind Mode: Tracking ALL {base_data.shape[-1]} tracks in this head.")
        base_relevant = base_data[0, :, :]
    else:
        base_relevant = base_data[0, :, target_indices]
    
    if args.agg_mode == 'mean': baseline_val = float(np.nanmean(base_relevant))
    else: baseline_val = float(np.nanmax(base_relevant))
        
    print(f"   [i] Global Baseline: {baseline_val:.4f}")
    
    if np.isnan(baseline_val):
        print("   [!] CRITICAL WARNING: Baseline is still NaN. Hardware incompatibility remains.")
    
    del base_preds, batch_input
    gc.collect()

    print("Generating Masked Sequences (Single-Threaded)...")
    masked_seqs = []
    for i in tqdm(indices):
        masked_seqs.append(helpers.mask_sequence((wt_seq, i, args.mutation_size)))

    print("Scan Started...")
    results = []
    tss_offset = coords.get('tss', 0)
    
    for i in tqdm(range(0, len(masked_seqs), args.batch_size)):
        if i % 10 == 0: gc.collect()
        
        batch_seqs = masked_seqs[i : i + args.batch_size]
        curr_bs = len(batch_seqs)
        
        if curr_bs < args.batch_size:
            batch_seqs += [batch_seqs[-1]] * (args.batch_size - curr_bs)
            
        batch_arr = jnp.stack([helpers.one_hot_encode(s) for s in batch_seqs])
        
        preds = fast_predict(
            params, state, batch_arr, 
            jnp.zeros((args.batch_size,), dtype=jnp.int32), 
            jnp.zeros((args.batch_size,), dtype=bool)
        )
        
        raw_data = np.array(preds[head_key])
        del preds
        
        for j in range(curr_bs):
            if target_indices == "ALL":
                curr_tracks = raw_data[j, :, :]
            else:
                curr_tracks = raw_data[j, :, target_indices]
                
            curr_val = np.nanmean(curr_tracks)
            impact = baseline_val - curr_val
            
            pos = coords['start'] + indices[i+j]
            results.append({
                'chrom': coords['chrom'], 'start': pos, 'end': pos + args.mutation_size,
                'impact': impact, 'distance_to_tss': pos - tss_offset
            })

    results.sort(key=lambda x: x['impact'], reverse=True)
    return results