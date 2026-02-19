import os
import sys

# --- 1. STRICT MEMORY & PRECISION CONFIGURATION ---
# Must be set before importing JAX or TensorFlow
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Disable TensorFloat32 to prevent NaN overflow on high-end GPUs
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "float32"

import argparse
import haiku as hk
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
    target.add_argument("--exclude_gene_body", action="store_true", default=False, 
                        help="Skip mutations inside the gene body.")

    scan_params = parser.add_argument_group("Scan")
    scan_params.add_argument("--mutation_size", type=int, default=2000, help="Size of deletion/mutation in bp")
    scan_params.add_argument("--step_size", type=int, default=1000, help="Step size for sliding window")
    scan_params.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
    
    filters = parser.add_argument_group("Filters")
    filters.add_argument("--tissue", type=str, default="Spleen", help="Target tissue context")
    filters.add_argument("--assay", type=str, default="RNA", help="Target assay (RNA, ATAC, DNASE, CAGE)")
    filters.add_argument("--agg_mode", type=str, default='mean', choices=['mean', 'max'])
    filters.add_argument("--inspect_tracks", type=str, default="", help="Specific tracks to force-check")
    
    out = parser.add_argument_group("Output")
    out.add_argument("--out_dir", type=str, default="./results")
    out.add_argument("--top_hits_scan", type=int, default=20)
    out.add_argument("--top_hits_inspect", type=int, default=5)
    out.add_argument("--no_inspect", action="store_true", help="Skip the transcription factor inspection phase")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"--- AlphaGenome Telescope Initialized ---")
    
    try:
        coords = helpers.resolve_coordinates(args)
    except Exception as e:
        print(f"[!] Error resolving coordinates: {e}")
        sys.exit(1)
        
    print(f"Analysis Window: {coords['chrom']}:{coords['start']}-{coords['end']} ({coords['strand']})")

    # --- THE BFLOAT16 FIX ---
    print("Loading AlphaGenome Model into VRAM (BFloat16 Precision)...")
    
    # BFloat16 prevents Infinity/NaN overflows while saving massive VRAM
    policy = hk.mixed_precision.MixedPrecisionPolicy(
        param_dtype=jnp.bfloat16, 
        compute_dtype=jnp.bfloat16, 
        output_dtype=jnp.float32
    )
    
    with hk.mixed_precision.policy_scope(policy):
        model = dna_model.create_from_huggingface('all_folds')
    
    params = getattr(model, '_params', None) or getattr(model, 'params', None)
    state = getattr(model, '_state', None) or getattr(model, 'state', None)
    
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