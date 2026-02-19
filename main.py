import os
import sys

# --- 1. STRICT MEMORY & PRECISION CONFIGURATION ---
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import argparse
import jax
import jax.numpy as jnp
from utils import helpers
from core import scanner, inspector
from alphagenome_research.model import dna_model, augmentation

# Monkey Patch to prevent augmentation errors during inference
def no_op(*args, **kwargs): return args[0]
augmentation.reverse_complement = no_op

def main():
    parser = argparse.ArgumentParser(description="AlphaGenome Telescope: Unbiased Regulatory Discovery")
    
    # --- ARGUMENTS ---
    target = parser.add_argument_group("Target")
    target.add_argument("--gene", type=str, help="Target Gene Symbol (e.g., TFEB)")
    target.add_argument("--region", type=str, help="Manual coordinates (chr:start-end)")
    target.add_argument("--window_size", type=int, default=1048576) 
    target.add_argument("--exclude_gene_body", action="store_true", default=False)

    scan_params = parser.add_argument_group("Scan")
    scan_params.add_argument("--mutation_size", type=int, default=2000)
    scan_params.add_argument("--step_size", type=int, default=1000)
    scan_params.add_argument("--batch_size", type=int, default=1)
    
    filters = parser.add_argument_group("Filters")
    filters.add_argument("--tissue", type=str, default="Spleen")
    filters.add_argument("--assay", type=str, default="RNA")
    filters.add_argument("--agg_mode", type=str, default='mean', choices=['mean', 'max'])
    filters.add_argument("--inspect_tracks", type=str, default="")
    
    out = parser.add_argument_group("Output")
    out.add_argument("--out_dir", type=str, default="./results")
    out.add_argument("--top_hits_scan", type=int, default=20)
    out.add_argument("--top_hits_inspect", type=int, default=5)
    out.add_argument("--no_inspect", action="store_true")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"--- AlphaGenome Telescope Initialized ---")
    
    try:
        coords = helpers.resolve_coordinates(args)
    except Exception as e:
        print(f"[!] Error resolving coordinates: {e}")
        sys.exit(1)
        
    print(f"Analysis Window: {coords['chrom']}:{coords['start']}-{coords['end']} ({coords['strand']})")

    print("Loading AlphaGenome Model into VRAM...")
    model = dna_model.create_from_huggingface('all_folds')
    
    raw_params = getattr(model, '_params', None) or getattr(model, 'params', None)
    raw_state = getattr(model, '_state', None) or getattr(model, 'state', None)
    
    # --- THE NATIVE BFLOAT16 CASTING ---
    print("Casting model to native BFloat16 (TPU Precision) to prevent NaN overflow...")
    
    # Safely convert all floating point weights to bfloat16 natively in JAX
    params = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16) if getattr(x, 'dtype', None) in (jnp.float32, jnp.float16) else x, 
        raw_params
    )
    state = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16) if getattr(x, 'dtype', None) in (jnp.float32, jnp.float16) else x, 
        raw_state
    )
    
    print("Fetching Wild-Type Sequence...")
    wt_seq = helpers.fetch_sequence(model, coords)

    print("\n>>> PHASE 1: SCANNING")
    scan_results = scanner.run_scan(args, coords, model, params, state, wt_seq)
    
    helpers.save_scan_results(scan_results, args, coords)
    
    if not args.no_inspect:
        print("\n>>> PHASE 2: INSPECTING (Unbiased Discovery)")
        top_hits = scan_results[:args.top_hits_inspect]
        if len(top_hits) > 0:
            inspection_data = inspector.inspect_hits(top_hits, args, model, params, state, coords, wt_seq)
            helpers.save_inspection_report(inspection_data, args)
        else:
            print("   [!] No significant hits found to inspect.")
    
    print(f"\n>>> DONE. Results saved in {args.out_dir}")

if __name__ == "__main__":
    main()