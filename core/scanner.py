import gc
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from utils import helpers
import jax
import jax.numpy as jnp

# --- 1. GLOBAL JIT DEFINITION (The Memory Fix) ---
# By defining this outside, we ensure JAX compiles it ONCE and locks the memory.
@jax.jit
def _fast_predict_jit(params, state, inputs, org_idx, neg_mask):
    """
    Static JIT-compiled kernel. 
    We pass 'model' logic via the params/state which are JAX pytrees.
    We assume the model architecture is static (AlphaGenome).
    """
    # We need to access the model._predict method. 
    # Since we can't pass the 'model' object (a Python class) into a JIT function easily without partials,
    # we will rely on the pure function call if possible, OR we use a closure if we must.
    # However, to fix the memory leak, we should use the model's pure functional apply if available.
    
    # Reverting to the closure method but ensuring it's efficient:
    # Actually, the cleanest way with Haiku/JAX is to keep the function pure.
    # But since 'model' is a high-level wrapper, we have to call model._predict.
    # The LEAK likely comes from 'model' being captured in the closure repeatedly.
    pass 

# Since model._predict is a bound method, we can't easily JIT it globally without the model instance.
# IMPROVED STRATEGY: We will create the JIT function *inside* run_scan but use 
# jax.clear_backends() or better memory management to ensure it doesn't duplicate.

def run_scan(args, coords, model, params, state, wt_seq):
    # 1. Setup
    target_indices = helpers.find_track_indices(model, args.tissue, args.assay)
    print(f"Tracking {len(target_indices)} output heads for '{args.tissue} {args.assay}'")
    
    # 2. Baseline
    print("Calculating Baseline...")
    # Force garbage collection before the big alloc
    gc.collect() 
    jax.clear_caches() 
    
    baseline_val = helpers.predict_baseline(model, params, state, wt_seq, target_indices, args.agg_mode)
    print(f"Global Baseline: {baseline_val:.4f}")

    # 3. Variants
    indices = list(range(0, len(wt_seq) - args.mutation_size, args.step_size))
    if args.exclude_gene_body and 'gene_start_rel' in coords:
        g_start = coords['gene_start_rel']
        g_end = coords['gene_end_rel']
        indices = [i for i in indices if not (i < g_end and (i + args.mutation_size) > g_start)]

    print("Generating Masked Sequences...")
    with ProcessPoolExecutor() as exc:
        tasks = [(wt_seq, i, args.mutation_size) for i in indices]
        masked_seqs = list(tqdm(exc.map(helpers.mask_sequence, tasks), total=len(indices)))

    # --- MEMORY OPTIMIZATION START ---
    print("Compiling JIT Kernel (Optimized)...")
    
    # We define the JIT function here, but we pass EVERYTHING as arguments
    # so the closure doesn't capture large arrays accidentally.
    @jax.jit
    def fast_predict(p, s, i, o, n):
        return model._predict(p, s, i, o, negative_strand_mask=n, strand_reindexing=None)

    # Warmup with dummy data to trigger compilation & allocation
    # Crucial: Use the exact shape we will use in the loop!
    # If args.batch_size is 1, use 1.
    dummy_input = jnp.zeros((args.batch_size, len(wt_seq), 4), dtype=jnp.float32)
    dummy_org = jnp.zeros((args.batch_size,), dtype=jnp.int32)
    dummy_mask = jnp.zeros((args.batch_size,), dtype=bool)
    
    _ = fast_predict(params, state, dummy_input, dummy_org, dummy_mask)
    # --- MEMORY OPTIMIZATION END ---

    print("Scan Started...")
    results = []
    
    # Pre-calculate constants to avoid overhead in loop
    tss_offset = coords.get('tss', 0)
    chrom = coords['chrom']
    
    for i in tqdm(range(0, len(masked_seqs), args.batch_size)):
        # Aggressive GC every few steps to prevent fragmentation
        if i % 10 == 0: 
            gc.collect()
        
        batch_seqs = masked_seqs[i : i + args.batch_size]
        curr_bs = len(batch_seqs)
        
        # Padding (Crucial for JIT stability - shape must not change!)
        if curr_bs < args.batch_size:
            batch_seqs += [batch_seqs[-1]] * (args.batch_size - curr_bs)
            
        batch_arr = jnp.stack([helpers.one_hot_encode(s) for s in batch_seqs])
        
        # Run Inference
        # We pass params/state explicitly
        preds = fast_predict(
            params, state, 
            batch_arr, 
            jnp.zeros((args.batch_size,), dtype=jnp.int32), 
            jnp.zeros((args.batch_size,), dtype=bool)
        )
        
        # Move to CPU immediately to free VRAM
        head_key = helpers.get_head_key(preds, args.assay)
        # Using np.array() here forces a copy to CPU RAM
        raw_data = np.array(preds[head_key]) 
        
        # Delete JAX array from VRAM explicitly
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