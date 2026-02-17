import gc
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from utils import helpers
import jax
import jax.numpy as jnp

def run_scan(args, coords, model, params, state, wt_seq):
    # 1. Setup Tracks
    target_indices = helpers.find_track_indices(model, args.tissue, args.assay)
    
    # 2. Prepare Data
    indices = list(range(0, len(wt_seq) - args.mutation_size, args.step_size))
    
    # Gene Body Exclusion Logic
    if args.exclude_gene_body and 'gene_start_rel' in coords:
        g_start = coords['gene_start_rel']
        g_end = coords['gene_end_rel']
        original_len = len(indices)
        indices = [i for i in indices if not (i < g_end and (i + args.mutation_size) > g_start)]
        print(f"Excluded gene body. Scannable variants: {len(indices)} (was {original_len})")

    # 3. COMPILE JIT KERNEL (The Memory Fix)
    print("Compiling JIT Kernel...")
    
    # We define the JIT function locally to capture 'model' safely
    @jax.jit
    def fast_predict(p, s, inputs, o, n):
        return model._predict(p, s, inputs, o, negative_strand_mask=n, strand_reindexing=None)

    # 4. CALCULATE BASELINE (USING JIT)
    # We replaced the helper call with this JIT-safe block
    print("Calculating Baseline (JIT Optimized)...")
    
    # Prepare single-item batch for baseline
    wt_encoded = helpers.one_hot_encode(wt_seq)[jnp.newaxis, ...] # Shape (1, 1Mb, 4)
    
    # Pad to match batch_size if necessary (JAX requires consistent shapes)
    if args.batch_size > 1:
        pad_amt = args.batch_size - 1
        padding = jnp.tile(wt_encoded, (pad_amt, 1, 1))
        batch_input = jnp.concatenate([wt_encoded, padding], axis=0)
    else:
        batch_input = wt_encoded

    # Run Inference
    base_preds = fast_predict(
        params, state, 
        batch_input, 
        jnp.zeros((args.batch_size,), dtype=jnp.int32), 
        jnp.zeros((args.batch_size,), dtype=bool)
    )
    
    # Extract Baseline Value
    head_key = helpers.get_head_key(base_preds, args.assay)
    base_data = np.array(base_preds[head_key]) # Move to CPU
    
    # We only care about the first item (the real WT sequence)
    base_relevant = base_data[0, :, target_indices]
    
    if args.agg_mode == 'mean':
        baseline_val = float(np.nanmean(base_relevant))
    else:
        baseline_val = float(np.nanmax(base_relevant))
        
    print(f"Global Baseline: {baseline_val:.4f}")
    
    # Cleanup VRAM immediately
    del base_preds
    del batch_input
    gc.collect()

    # 5. GENERATE VARIANTS
    print("Generating Masked Sequences...")
    with ProcessPoolExecutor() as exc:
        tasks = [(wt_seq, i, args.mutation_size) for i in indices]
        masked_seqs = list(tqdm(exc.map(helpers.mask_sequence, tasks), total=len(indices)))

    print("Scan Started...")
    results = []
    tss_offset = coords.get('tss', 0)
    chrom = coords['chrom']
    
    for i in tqdm(range(0, len(masked_seqs), args.batch_size)):
        if i % 10 == 0: gc.collect()
        
        batch_seqs = masked_seqs[i : i + args.batch_size]
        curr_bs = len(batch_seqs)
        
        # Padding for JIT stability
        if curr_bs < args.batch_size:
            batch_seqs += [batch_seqs[-1]] * (args.batch_size - curr_bs)
            
        batch_arr = jnp.stack([helpers.one_hot_encode(s) for s in batch_seqs])
        
        # Run Inference
        preds = fast_predict(
            params, state, 
            batch_arr, 
            jnp.zeros((args.batch_size,), dtype=jnp.int32), 
            jnp.zeros((args.batch_size,), dtype=bool)
        )
        
        # Move to CPU & Free VRAM
        raw_data = np.array(preds[head_key])
        del preds
        
        for j in range(curr_bs):
            relevant_tracks = raw_data[j, :, target_indices] 
            curr_val = np.nanmean(relevant_tracks)
            impact = baseline_val - curr_val
            
            pos = coords['start'] + indices[i+j]
            dist = pos - tss_offset
            
            results.append({
                'chrom': chrom, 'start': pos, 'end': pos + args.mutation_size,
                'impact': impact, 'distance_to_tss': dist
            })

    results.sort(key=lambda x: x['impact'], reverse=True)
    return results