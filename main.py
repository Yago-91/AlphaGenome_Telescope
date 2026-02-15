import argparse
import os
import sys
from utils import helpers
from core import scanner, inspector

# AlphaGenome Telescope: Enhancer Discovery & Characterization Tool

def main():
    parser = argparse.ArgumentParser(description="AlphaGenome Telescope: Enhancer Discovery & Characterization Tool")
    
    # --- 1. TARGET DEFINITION ---
    target = parser.add_argument_group("Target Definition")
    target.add_argument("--gene", type=str, help="Target gene name (e.g., TFEB). Auto-fetches coordinates.")
    target.add_argument("--region", type=str, help="Manual region (chr6:41700000-41800000).")
    target.add_argument("--window_size", type=int, default=196608, help="Total window size to analyze (Default: ~196kb). Max: 1048576.")
    target.add_argument("--exclude_gene_body", action="store_true", default=True, help="Skip mutations inside the gene body (Default: True).")

    # --- 2. SCANNING PARAMETERS ---
    scan_params = parser.add_argument_group("Scanning Parameters")
    scan_params.add_argument("--mutation_size", type=int, default=2000, help="Size of the mask/mutation in bp (Default: 2000).")
    scan_params.add_argument("--step_size", type=int, default=1000, help="Step size for sliding window in bp (Default: 1000).")
    scan_params.add_argument("--batch_size", type=int, default=1, help="GPU Batch size (Default: 1).")
    
    # --- 3. BIOLOGICAL FILTERS ---
    filters = parser.add_argument_group("Biological Filters")
    filters.add_argument("--tissue", type=str, default="Spleen", help="Primary tissue for impact analysis (e.g., 'Spleen').")
    filters.add_argument("--assay", type=str, default="RNA", help="Assay for impact analysis (Default: RNA).")
    filters.add_argument("--agg_mode", type=str, choices=['mean', 'max'], default='mean', help="How to combine multiple tracks (Default: mean).")
    filters.add_argument("--inspect_tracks", type=str, default="DNASE,H3K27ac,H3K4me3,CTCF", help="Comma-separated list of extra tracks to inspect.")
    
    # --- 4. OUTPUT OPTIONS ---
    out = parser.add_argument_group("Output Options")
    out.add_argument("--out_dir", type=str, default="./results", help="Output directory.")
    out.add_argument("--top_hits_scan", type=int, default=20, help="Number of hits to report in CSV (Default: 20).")
    out.add_argument("--top_hits_inspect", type=int, default=5, help="Number of top hits to deep-dive inspect (Default: 5).")
    out.add_argument("--no_inspect", action="store_true", help="Skip inspection phase (Scan only).")

    args = parser.parse_args()

    # --- SETUP ---
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"--- AlphaGenome Telescope Initialized ---")
    print(f"Target: {args.gene if args.gene else args.region}")
    print(f"Tissue: {args.tissue} | Assay: {args.assay}")

    # 1. RESOLVE COORDINATES
    coords = helpers.resolve_coordinates(args)
    print(f"Analysis Window: {coords['chrom']}:{coords['start']}-{coords['end']} ({coords['strand']})")

    # 2. RUN SCANNER (Impact Analysis)
    print("\n>>> PHASE 1: SCANNING FOR REGULATORY ELEMENTS")
    scan_results = scanner.run_scan(args, coords)
    
    # Save Scan Results
    helpers.save_scan_results(scan_results, args, coords)
    
    # 3. RUN INSPECTOR (Mechanism Analysis)
    if not args.no_inspect:
        print("\n>>> PHASE 2: INSPECTING TOP HITS")
        top_hits = scan_results[:args.top_hits_inspect]
        inspection_data = inspector.inspect_hits(top_hits, args, coords)
        helpers.save_inspection_report(inspection_data, args)
    
    print(f"\n>>> DONE. Results saved in {args.out_dir}")

if __name__ == "__main__":
    main()