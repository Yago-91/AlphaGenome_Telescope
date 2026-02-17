import gc
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from utils import helpers
import jax
import jax.numpy as jnp

def run_scan(args, coords, model, params, state, wt_seq):
    target_indices = helpers.find_track_indices(model, args.tissue, args.assay)
    print(f"Tracking {len(target_indices)} output heads for '{args.tissue} {args.assay}'")
    
    print("Calculating Baseline...")
    baseline_val = helpers.predict_baseline(model, params, state, wt_seq, target_indices, args.agg_mode)
    print(f"Global Baseline: {baseline_val:.4f}")

    indices = list(range(0, len(wt_seq) - args.mutation_size, args.step_size))
    
    if args.exclude_gene_body and 'gene_start_rel' in coords:
        g_start = coords['gene_start_rel']
        g_end = coords['gene_end_rel']
        original_len = len(indices)
        indices = [i for i in indices if not (i < g_end and (i + args.mutation_size) > g_start)]
        print(f"Excluded gene body. Scannable variants: {len(indices)} (was {original_len})")

    print("Generating Masked Sequences...")
    with ProcessPoolExecutor() as exc:
        tasks = [(wt_seq, i, args.mutation_size) for i in indices]
        masked_seqs = list(tqdm(exc.map(helpers.mask_sequence, tasks), total=len(indices)))

    print("Compiling JIT Kernel...")
    @jax.jit
    def fast_predict(inputs, org_idx, neg_mask):
        # FIX: Explicit keyword arguments
        return model._predict(
            params, state, inputs, org_idx, 
            negative_strand_mask=neg_mask, 
            strand_reindexing=None
        )
    
    warmup = jnp.zeros((args.batch_size, len(wt_seq), 4), dtype=jnp.float32)
    _ = fast_predict(warmup, jnp.zeros((args.batch_size,), dtype=jnp.int32), jnp.zeros((args.batch_size,), dtype=bool))

    results = []
    
    for i in tqdm(range(0, len(masked_seqs), args.batch_size)):
        if i % 20 == 0: gc.collect()
        
        batch_seqs = masked_seqs[i : i + args.batch_size]
        curr_bs = len(batch_seqs)
        
        if curr_bs < args.batch_size:
            batch_seqs += [batch_seqs[-1]] * (args.batch_size - curr_bs)
            
        batch_arr = jnp.stack([helpers.one_hot_encode(s) for s in batch_seqs])
        
        preds = fast_predict(
            batch_arr, 
            jnp.zeros((args.batch_size,), dtype=jnp.int32), 
            jnp.zeros((args.batch_size,), dtype=bool)
        )
        
        head_key = helpers.get_head_key(preds, args.assay)
        raw_data = np.array(preds[head_key])
        
        for j in range(curr_bs):
            relevant_tracks = raw_data[j, :, target_indices] 
            curr_val = np.nanmean(relevant_tracks)
            impact = baseline_val - curr_val
            
            pos = coords['start'] + indices[i+j]
            dist = pos - coords.get('tss', 0)
            
            results.append({
                'chrom': coords['chrom'],
                'start': pos,
                'end': pos + args.mutation_size,
                'impact': impact,
                'distance_to_tss': dist
            })

    results.sort(key=lambda x: x['impact'], reverse=True)
    return results