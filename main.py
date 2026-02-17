import argparse
import os
import sys
from utils import helpers
from core import scanner, inspector
from alphagenome_research.model import dna_model, augmentation

# Monkey Patch (Apply once globally)
def no_op(*args, **kwargs): return args[0]
augmentation.reverse_complement = no_op

def main():
    parser = argparse.ArgumentParser(description="AlphaGenome Telescope: Unbiased Regulatory Discovery")
    
    # --- ARGUMENTS (Same as before) ---
    # (Keep your existing argument definitions here for brevity)
    # ... [Copy the argparse block from the previous main.py] ...
    
    # For context, here are the simplified groups if you need to copy-paste:
    target = parser.add_argument_group("Target")
    target.add_argument("--gene", type=str)
    target.add_argument("--region", type=str)
    target.add_argument("--window_size", type=int, default=196608)
    target.add_argument("--exclude_gene_body", action="store_true", default=True)

    scan_params = parser.add_argument_group("Scan")
    scan_params.add_argument("--mutation_size", type=int, default=2000)
    scan_params.add_argument("--step_size", type=int, default=1000)
    scan_params.add_argument("--batch_size", type=int, default=1)
    
    filters = parser.add_argument_group("Filters")
    filters.add_argument("--tissue", type=str, default="Spleen")
    filters.add_argument("--assay", type=str, default="RNA")
    filters.add_argument("--agg_mode", type=str, default='mean')
    filters.add_argument("--inspect_tracks", type=str, default="")
    
    out = parser.add_argument_group("Output")
    out.add_argument("--out_dir", type=str, default="./results")
    out.add_argument("--top_hits_scan", type=int, default=20)
    out.add_argument("--top_hits_inspect", type=int, default=5)
    out.add_argument("--no_inspect", action="store_true")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # --- 1. GLOBAL INITIALIZATION (The Efficiency Fix) ---
    print(f"--- AlphaGenome Telescope Initialized ---")
    
    # A. Resolve Coordinates
    coords = helpers.resolve_coordinates(args)
    print(f"Analysis Window: {coords['chrom']}:{coords['start']}-{coords['end']} ({coords['strand']})")

    # B. Load Model (ONCE)
    print("Loading AlphaGenome Model into VRAM...")
    model = dna_model.create_from_huggingface('all_folds')
    params = getattr(model, '_params', None) or getattr(model, 'params', None)
    state = getattr(model, '_state', None) or getattr(model, 'state', None)
    
    # C. Fetch Sequence (ONCE)
    print("Fetching Wild-Type Sequence...")
    wt_seq = helpers.fetch_sequence(model, coords)

    # --- 2. RUN SCANNER ---
    print("\n>>> PHASE 1: SCANNING")
    # PASS THE MODEL AND SEQ DOWN
    scan_results = scanner.run_scan(args, coords, model, params, state, wt_seq)
    helpers.save_scan_results(scan_results, args, coords)
    
    # --- 3. RUN INSPECTOR ---
    if not args.no_inspect:
        print("\n>>> PHASE 2: INSPECTING (Unbiased Discovery)")
        top_hits = scan_results[:args.top_hits_inspect]
        
        # PASS THE SAME MODEL AND SEQ DOWN
        inspection_data = inspector.inspect_hits(top_hits, args, model, params, state, coords, wt_seq)
        helpers.save_inspection_report(inspection_data, args)
    
    print(f"\n>>> DONE. Results saved in {args.out_dir}")

if __name__ == "__main__":
    main()