import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

# --- 1. COORDINATE & SEQUENCE HANDLING ---

def resolve_coordinates(args):
    """
    Resolves target arguments (--gene or --region) into exact genomic coordinates.
    """
    # Hardcoded TFEB for stability in this version. 
    # In a future V2, we can add 'pyensembl' to fetch any gene dynamically.
    if args.gene and args.gene.upper() == "TFEB":
        tss = 41704353
        half_window = args.window_size // 2
        return {
            'chrom': 'chr6',
            'start': tss - half_window,
            'end': tss + half_window,
            'strand': '-',
            'tss': tss,
            'gene_start_rel': tss - 5000,   # Approx start of gene body relative to TSS
            'gene_end_rel': tss + 45000     # Approx end of gene body
        }
    elif args.region:
        # Parse format "chr6:1000-2000"
        try:
            chrom, span = args.region.split(':')
            start, end = map(int, span.split('-'))
            return {
                'chrom': chrom,
                'start': start,
                'end': end,
                'strand': '+', # Default to + for manual regions
                'tss': start   # Default TSS to start of region
            }
        except ValueError:
            raise ValueError("Invalid Region Format. Use 'chr:start-end' (e.g., chr6:41000-42000)")
    else:
        raise ValueError("No Target Specified. Use --gene TFEB or --region.")

def fetch_sequence(model, coords):
    """
    Extracts the One-Hot encoded sequence from the model's Fasta extractor.
    """
    # 1. Find the Human extractor key
    keys = list(model._fasta_extractors.keys())
    # Heuristic: Find key with '9606' (TaxID) or 'sapiens'
    hum_key = next((k for k in keys if "9606" in str(k) or "SAPIENS" in str(k).upper()), None)
    
    if not hum_key:
        print("WARNING: Could not identify Human Fasta Extractor. Using first available.")
        hum_key = keys[0]

    # 2. Create Interval and Extract
    from alphagenome.data import genome
    interval = genome.Interval(coords['chrom'], coords['start'], coords['end'])
    seq = model._fasta_extractors[hum_key].extract(interval)
    return seq

def one_hot_encode(seq):
    """
    Converts a string sequence (ACGT) to a (L, 4) one-hot numpy array.
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, char in enumerate(seq.upper()):
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return jnp.array(arr)

def mask_sequence(args):
    """
    Worker function for ProcessPoolExecutor.
    Replaces a specific window with 'N's.
    """
    seq, start_idx, length = args
    s_list = list(seq)
    # Ensure we don't go out of bounds
    end_idx = min(start_idx + length, len(s_list))
    s_list[start_idx : end_idx] = ['N'] * (end_idx - start_idx)
    return "".join(s_list)

# --- 2. MODEL INTERACTION & PREDICTION ---

def find_track_indices(model, tissue, assay):
    """
    Finds the output indices corresponding to the requested tissue and assay.
    """
    # Get all track names
    all_tracks = get_track_metadata(model)
    
    # Filter
    indices = []
    for i, name in enumerate(all_tracks):
        # Case-insensitive match
        if tissue.lower() in name.lower() and assay.lower() in name.lower():
            indices.append(i)
            
    if not indices:
        print(f"WARNING: No tracks found for '{tissue}' + '{assay}'. using first 5 tracks as fallback.")
        return list(range(5))
        
    return indices

def get_head_key(preds, assay_name):
    """
    Finds the correct key in the prediction dictionary (e.g. 'human' vs 'mouse').
    """
    keys = list(preds.keys())
    # Try to find a key that looks like the assay or 'human'
    # AlphaGenome usually returns keys like 'human_head' or just 'human'
    # Enformer returns 'human' and 'mouse'
    target = next((k for k in keys if 'human' in str(k).lower()), None)
    if not target:
        target = keys[0]
    return target

def predict_baseline(model, params, state, seq, track_indices, agg_mode):
    """
    Runs a single prediction on the WT sequence to get the baseline value.
    """
    enc = one_hot_encode(seq)[jnp.newaxis, ...] # Add batch dim
    
    # Run in Eager Mode (disable_jit) for safety on single call
    with jax.disable_jit():
        preds = model._predict(params, state, enc, jnp.array([0]), jnp.array([False]), None)
    
    head_key = get_head_key(preds, 'human')
    # Output shape: (1, 896, 5313)
    data = np.array(preds[head_key])
    
    # Select specific tracks
    selected_data = data[0, :, track_indices]
    
    # Aggregation
    if agg_mode == 'mean':
        return float(np.nanmean(selected_data))
    elif agg_mode == 'max':
        return float(np.nanmax(selected_data))
    return float(np.nanmean(selected_data))

def predict_all_tracks(model, params, state, seq):
    """
    Runs prediction and returns the raw output for ALL 5,313 tracks.
    Used by the Unbiased Inspector.
    """
    enc = one_hot_encode(seq)[jnp.newaxis, ...]
    
    with jax.disable_jit():
        # Passing None for strand_reindexing to get raw output
        raw_out = model._predict(params, state, enc, jnp.array([0]), jnp.array([False]), None)
    
    head_key = get_head_key(raw_out, 'human')
    # Return shape: (896, Num_Tracks)
    return np.array(raw_out[head_key][0])

# --- 3. METADATA & INSPECTION UTILS ---

def get_track_metadata(model):
    """
    Retrieves the list of track names (e.g., 'CAGE:Spleen') from the model config.
    """
    # 1. Try standard Enformer attribute
    if hasattr(model, 'track_names'):
        return model.track_names
    
    # 2. Try AlphaGenome config
    if hasattr(model, 'config') and hasattr(model.config, 'target_names'):
        return model.config.target_names
    
    # 3. Try hidden attribute
    if hasattr(model, '_track_names'):
        return model._track_names

    print("WARNING: Could not find track names in model metadata. Returning dummy IDs.")
    # Return dummy list if all else fails
    return [f"Track_{i}" for i in range(5313)]

def clean_track_name(raw_name):
    """
    Cleans up complex track names for the CSV report.
    Example: 'CHIP:CTCF:human_spleen_sample1' -> 'CTCF'
    """
    # Simple heuristic parsing
    parts = raw_name.replace(':', ' ').replace('_', ' ').split()
    
    # Common TFs and marks to prioritize
    vocab = ['CTCF', 'PU.1', 'SPI1', 'H3K27AC', 'H3K4ME3', 'H3K4ME1', 'H3K27ME3', 'POL2', 'DNASE', 'ATAC']
    
    for word in parts:
        if word.upper() in vocab:
            return word.upper()
            
    # If no common vocab found, return the second part (usually the factor name)
    # e.g. "CHIP:CEBPB:..." -> CEBPB
    if ':' in raw_name:
        try:
            return raw_name.split(':')[1]
        except:
            pass
            
    return raw_name

# --- 4. FILE WRITERS ---

def save_scan_results(results, args, coords):
    """
    Saves the mutagenesis scan to CSV and BedGraph.
    """
    df = pd.DataFrame(results)
    
    # CSV
    csv_name = f"{args.gene if args.gene else 'Region'}_{args.tissue}_{args.assay}_scan.csv"
    csv_path = os.path.join(args.out_dir, csv_name)
    df.to_csv(csv_path, index=False)
    print(f"   [+] CSV Saved: {csv_path}")
    
    # BedGraph
    bg_name = csv_name.replace('.csv', '.bedgraph')
    bg_path = os.path.join(args.out_dir, bg_name)
    
    with open(bg_path, 'w') as f:
        # Header for IGV
        f.write(f"track type=bedGraph name='{args.gene} {args.tissue} Impact' description='Impact Score (Baseline - Mutant)' visibility=full autoScale=on color=255,0,0\n")
        for r in results:
            # BedGraph format: chrom start end value
            f.write(f"{r['chrom']}\t{r['start']}\t{r['end']}\t{r['impact']:.5f}\n")
            
    print(f"   [+] BedGraph Saved: {bg_path}")

def save_inspection_report(data, args):
    """
    Saves the Unbiased Inspector report to CSV.
    """
    if not data:
        print("   [!] No inspection data generated.")
        return

    df = pd.DataFrame(data)
    
    csv_name = f"{args.gene if args.gene else 'Region'}_inspection_report.csv"
    csv_path = os.path.join(args.out_dir, csv_name)
    
    # Reorder columns for readability if they exist
    cols = ['location', 'type', 'impact_on_gene', 'top_regulatory_signals']
    # Add any extra columns that might be there
    existing_cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]
    
    df = df[existing_cols]
    df.to_csv(csv_path, index=False)
    print(f"   [+] Inspection Report Saved: {csv_path}")