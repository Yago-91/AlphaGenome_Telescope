import os
import sys

# --- 1. CRITICAL MEMORY & JAX CONFIG ---
# Must be set before importing JAX or TensorFlow
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import argparse
from utils import helpers
from core import scanner, inspector
from alphagenome_research.model import dna_model, augmentation

# Monkey Patch to prevent augmentation errors during inference
def no_op(*args, **kwargs): return args[0]
augmentation.reverse_complement = no_op

def main():
    parser = argparse.ArgumentParser(description="AlphaGenome Telescope: Unbiased Regulatory Discovery")
    
    # --- ARGUMENTS ---
    
    # Target Group
    target = parser.add_argument_group("Target")
    target.add_argument("--gene", type=str, help="Target Gene Symbol (e.g., TFEB)")
    target.add_argument("--region", type=str, help="Manual coordinates (chr:start-end)")
    # Default window is 1Mb (Required for AlphaGenome architecture)
    target.add_argument("--window_size", type=int, default=1048576) 
    # Default is False (Scan everything). If flag is present, it becomes True.
    target.add_argument("--exclude_gene_body", action="store_true", default=False, 
                        help="Skip mutations inside the gene body (useful for enhancer-only scans).")

    # Scan Parameters
    scan_params = parser.add_argument_group("Scan")
    scan_params.add_argument("--mutation_size", type=int, default=2000, help="Size of deletion/mutation in bp")
    scan_params.add_argument("--step_size", type=int, default=1000, help="Step size for sliding window")
    scan_params.add_argument("--batch_size", type=int, default=1, help="Inference batch size (Keep low for 1Mb window)")
    
    # Filters
    filters = parser.add_argument_group("Filters")
    filters.add_argument("--tissue", type=str, default="Spleen", help="Target tissue context")
    filters.add_argument("--assay", type=str, default="RNA", help="Target assay (RNA, ATAC, DNASE, CAGE)")
    filters.add_argument("--agg_mode", type=str, default='mean', choices=['mean', 'max'], help="How to aggregate multiple tracks")
    filters.add_argument("--inspect_tracks", type=str, default="", help="Comma-separated list of specific tracks to force-check")
    
    # Output
    out = parser.add_argument_group("Output")
    out.add_argument("--out_dir", type=str, default="./results")
    out.add_argument("--top_hits_scan", type=int, default=20)
    out.add_argument("--top_hits_inspect", type=int, default=5)
    out.add_argument("--no_inspect", action="store_true", help="Skip the transcription factor inspection phase")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # --- 2. GLOBAL INITIALIZATION ---
    print(f"--- AlphaGenome Telescope Initialized ---")
    
    # A. Resolve Coordinates (Using MyGene.info)
    try:
        coords = helpers.resolve_coordinates(args)
    except Exception as e:
        print(f"[!] Error resolving coordinates: {e}")
        sys.exit(1)
        
    print(f"Analysis Window: {coords['chrom']}:{coords['start']}-{coords['end']} ({coords['strand']})")

    # B. Load Model (ONCE)
    print("Loading AlphaGenome Model into VRAM...")
    # NOTE: We use standard loading (Float32) to avoid NaN instability.
    # The JIT optimizer in scanner.py handles memory efficiency.
    model = dna_model.create_from_huggingface('all_folds')
    
    # Extract params/state for JAX functional calls
    params = getattr(model, '_params', None) or getattr(model, 'params', None)
    state = getattr(model, '_state', None) or getattr(model, 'state', None)
    
    # C. Fetch Sequence (ONCE)
    print("Fetching Wild-Type Sequence...")
    wt_seq = helpers.fetch_sequence(model, coords)

    # --- 3. PHASE 1: SCANNING ---
    print("\n>>> PHASE 1: SCANNING")
    
    # Pass the model and sequence to the scanner
    scan_results = scanner.run_scan(args, coords, model, params, state, wt_seq)
    
    # Save Results
    top_scan = scan_results[:args.top_hits_scan]
    helpers.save_scan_results(scan_results, args, coords)
    
    # --- 4. PHASE 2: INSPECTION ---
    if not args.no_inspect:
        print("\n>>> PHASE 2: INSPECTING (Unbiased Discovery)")
        top_hits = scan_results[:args.top_hits_inspect]
        
        if len(top_hits) > 0:
            # Pass the same model/seq to inspector
            inspection_data = inspector.inspect_hits(top_hits, args, model, params, state, coords, wt_seq)
            helpers.save_inspection_report(inspection_data, args)
        else:
            print("   [!] No significant hits found to inspect.")
    
    print(f"\n>>> DONE. Results saved in {args.out_dir}")

if __name__ == "__main__":
    main()