import os
import gc
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from utils import helpers

# JAX Setup
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform" 
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_CUDA_FORCE_BINDING"] = "true"
import jax
import jax.numpy as jnp
from alphagenome_research.model import dna_model, augmentation

# Monkey Patch
def no_op(*args, **kwargs): return args[0]
augmentation.reverse_complement = no_op

def run_scan(args, coords):
    # 1. Initialize Model
    print("Loading Model...")
    model = dna_model.create_from_huggingface('all_folds')
    params = getattr(model, '_params', None) or getattr(model, 'params', None)
    state = getattr(model, '_state', None) or getattr(model, 'state', None)

    # 2. Get Sequence
    wt_seq = helpers.fetch_sequence(model, coords)
    
    # 3. Define Target Tracks (The "Filter")
    # We find the indices for the requested Tissue/Assay
    target_indices = helpers.find_track_indices(model, args.tissue, args.assay)
    print(f"Found {len(target_indices)} tracks matching '{args.tissue}' + '{args.assay}'")
    
    # 4. Calculate Baseline (Eager Mode)
    print("Calculating Baseline...")
    baseline_val = helpers.predict_baseline(model, params, state, wt_seq, target_indices, args.agg_mode)
    print(f"Global Baseline Impact: {baseline_val:.4f}")

    # 5. Generate Mutations
    print("Generating Variants...")
    indices = list(range(0, len(wt_seq) - args.mutation_size, args.step_size))
    
    # Gene Body Exclusion Logic
    if args.exclude_gene_body and 'gene_start_rel' in coords:
        g_start = coords['gene_start_rel']
        g_end = coords['gene_end_rel']
        # Filter out indices that overlap the gene body
        indices = [i for i in indices if not (i < g_end and (i + args.mutation_size) > g_start)]
        print(f"Excluded gene body regions. {len(indices)} variants remain.")

    # Masking
    with ProcessPoolExecutor() as exc:
        tasks = [(wt_seq, i, args.mutation_size) for i in indices]
        masked_seqs = list(tqdm(exc.map(helpers.mask_sequence, tasks), total=len(indices)))

    # 6. JIT Compilation
    print("Compiling JIT Kernel...")
    @jax.jit
    def fast_predict(inputs, org_idx, neg_mask):
        return model._predict(params, state, inputs, org_idx, negative_strand_mask=neg_mask, strand_reindexing=None)
    
    # Warmup
    warmup = jnp.zeros((args.batch_size, len(wt_seq), 4), dtype=jnp.float32)
    _ = fast_predict(warmup, jnp.zeros((args.batch_size,), dtype=jnp.int32), jnp.zeros((args.batch_size,), dtype=bool))
    print("Scan Started.")

    # 7. The Loop
    results = []
    # We force Organism Index 0 (Human) for now. 
    # TODO: Make configurable if needed, but 0 is standard for hg38.
    
    for i in tqdm(range(0, len(masked_seqs), args.batch_size)):
        if i % 20 == 0: gc.collect()
        
        batch_seqs = masked_seqs[i : i + args.batch_size]
        curr_bs = len(batch_seqs)
        
        # Padding
        if curr_bs < args.batch_size:
            batch_seqs += [batch_seqs[-1]] * (args.batch_size - curr_bs)
            
        batch_arr = jnp.stack([helpers.one_hot_encode(s) for s in batch_seqs])
        
        # Run Inference
        preds = fast_predict(
            batch_arr, 
            jnp.zeros((args.batch_size,), dtype=jnp.int32), 
            jnp.zeros((args.batch_size,), dtype=bool)
        )
        
        # Extract & Aggregage
        # preds is a Dict. We need to extract the specific assay head (e.g. 'RNA')
        # Note: Model output keys are Enums usually, we convert to string to match.
        head_key = helpers.get_head_key(preds, args.assay)
        raw_data = np.array(preds[head_key]) # Shape: (Batch, 896, Num_Tracks)
        
        for j in range(curr_bs):
            # 1. Select specific tracks (spleen)
            relevant_tracks = raw_data[j, :, target_indices] 
            
            # 2. Average across bins? Or target bins only?
            # For this version, we average the whole window as requested by user originally,
            # BUT if we had gene coords, we could slice [:, target_bins] here.
            # Let's do Global Mean for robustness in this version.
            
            curr_val = np.nanmean(relevant_tracks) # Mean across bins AND tracks
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

    # Sort by impact
    results.sort(key=lambda x: x['impact'], reverse=True)
    return results