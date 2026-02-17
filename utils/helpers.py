import os
import sys
import requests
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import mygene

# --- 1. COORDINATE & SEQUENCE HANDLING ---

def resolve_coordinates(args):
    """
    Dynamically fetches coordinates for ANY gene symbol using MyGene.info (hg38).
    """
    # Case A: Manual Region (e.g., "chr6:100-200")
    if args.region:
        try:
            chrom, span = args.region.split(':')
            start, end = map(int, span.split('-'))
            return {
                'chrom': chrom,
                'start': start,
                'end': end,
                'strand': '+', # Default to + for regions
                'tss': start   # Default TSS to start
            }
        except ValueError:
            raise ValueError("Invalid Region Format. Use 'chr:start-end'")

    # Case B: Gene Symbol (e.g., "TFEB", "MYC")
    elif args.gene:
        print(f"   [i] Querying MyGene.info for '{args.gene}' coordinates (hg38)...")
        mg = mygene.MyGeneInfo()
        
        # Query for genomic position
        results = mg.query(args.gene, scopes='symbol', fields='genomic_pos', species='human')
        
        if not results or 'hits' not in results or len(results['hits']) == 0:
            raise ValueError(f"Gene '{args.gene}' not found in database.")
            
        # Get the first hit (usually the best match)
        hit = results['hits'][0]
        if 'genomic_pos' not in hit:
             # sometimes genomic_pos is a list if multiple locations exists, take the first
             if isinstance(hit.get('genomic_pos'), list):
                 gpos = hit['genomic_pos'][0]
             else:
                 raise ValueError(f"No genomic coordinates found for {args.gene}.")
        else:
            gpos = hit['genomic_pos']
            if isinstance(gpos, list): gpos = gpos[0]

        # Extract Info
        chrom = f"chr{gpos['chr']}"
        start_bp = gpos['start']
        end_bp = gpos['end']
        strand_val = gpos['strand'] # 1 or -1
        
        # LOGIC: Calculate TSS based on Strand
        # Ensembl/MyGene 'start' is always the smallest coordinate.
        # If Strand is + (1), TSS is 'start'.
        # If Strand is - (-1), TSS is 'end'.
        if strand_val == 1:
            tss = start_bp
            strand_sym = '+'
        else:
            tss = end_bp
            strand_sym = '-'
            
        print(f"       -> Found {args.gene} on {chrom} ({strand_sym} strand). TSS: {tss}")

        # Calculate the 1Mb Input Window (Centered on TSS)
        half_window = args.window_size // 2
        window_start = tss - half_window
        window_end = tss + half_window
        
        # Define Gene Body (for Exclusion)
        # We add a buffer to be safe
        gene_buffer = 1000
        gene_start_rel = (start_bp - window_start) - gene_buffer
        gene_end_rel = (end_bp - window_start) + gene_buffer

        return {
            'chrom': chrom,
            'start': window_start,
            'end': window_end,
            'strand': strand_sym,
            'tss': tss,
            'gene_start_rel': gene_start_rel,
            'gene_end_rel': gene_end_rel
        }

    else:
        raise ValueError("No Target Specified. Use --gene SYMBOL or --region.")

def fetch_sequence(model, coords):
    keys = list(model._fasta_extractors.keys())
    hum_key = next((k for k in keys if "9606" in str(k) or "SAPIENS" in str(k).upper()), None)
    if not hum_key:
        hum_key = keys[0]

    from alphagenome.data import genome
    interval = genome.Interval(coords['chrom'], coords['start'], coords['end'])
    seq = model._fasta_extractors[hum_key].extract(interval)
    return seq

def one_hot_encode(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, char in enumerate(seq.upper()):
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return jnp.array(arr)

def mask_sequence(args):
    seq, start_idx, length = args
    s_list = list(seq)
    end_idx = min(start_idx + length, len(s_list))
    s_list[start_idx : end_idx] = ['N'] * (end_idx - start_idx)
    return "".join(s_list)

# --- 2. MODEL INTERACTION ---

def find_track_indices(model, tissue, assay):
    all_tracks = get_track_metadata(model)
    indices = []
    for i, name in enumerate(all_tracks):
        if tissue.lower() in name.lower() and assay.lower() in name.lower():
            indices.append(i)
    
    if not indices:
        print(f"WARNING: No tracks found for '{tissue}' + '{assay}'. (Searched {len(all_tracks)} tracks).")
        print(f"Sample tracks: {all_tracks[:3]}")
        # Return first 5 as emergency fallback to prevent crash, but user should know.
        return list(range(5))
    return indices

def get_head_key(preds, assay_name):
    keys = list(preds.keys())
    # Prefer 'human' key
    target = next((k for k in keys if 'human' in str(k).lower()), keys[0])
    return target

def predict_baseline(model, params, state, seq, track_indices, agg_mode):
    enc = one_hot_encode(seq)[jnp.newaxis, ...]
    with jax.disable_jit():
        preds = model._predict(
            params, state, enc, jnp.array([0]), 
            negative_strand_mask=jnp.array([False]), 
            strand_reindexing=None
        )
    head_key = get_head_key(preds, 'human')
    data = np.array(preds[head_key])
    selected_data = data[0, :, track_indices]
    
    if agg_mode == 'mean': return float(np.nanmean(selected_data))
    elif agg_mode == 'max': return float(np.nanmax(selected_data))
    return float(np.nanmean(selected_data))

def predict_all_tracks(model, params, state, seq):
    enc = one_hot_encode(seq)[jnp.newaxis, ...]
    with jax.disable_jit():
        raw_out = model._predict(
            params, state, enc, jnp.array([0]), 
            negative_strand_mask=jnp.array([False]), 
            strand_reindexing=None
        )
    head_key = get_head_key(raw_out, 'human')
    return np.array(raw_out[head_key][0])

# --- 3. METADATA HANDLING (The Fix) ---

def download_official_tracks():
    """Downloads the Enformer track list from a reliable source."""
    url = "https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_human.txt"
    dest = "human_track_names.txt"
    
    if not os.path.exists(dest):
        print("   [i] Downloading official track metadata...")
        try:
            r = requests.get(url)
            with open(dest, 'w') as f:
                f.write(r.text)
        except Exception as e:
            print(f"   [!] Download failed: {e}")
            return []
            
    # Read file (Format: index \t identifier \t description)
    tracks = []
    with open(dest, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1:
                # Combine identifier and description for better matching
                # e.g., "CNhs12345 Spleen RNA-seq"
                tracks.append(" ".join(parts))
    return tracks

def get_track_metadata(model):
    # 1. Try internal attributes
    if hasattr(model, 'track_names'): return model.track_names
    if hasattr(model, 'config') and hasattr(model.config, 'target_names'): return model.config.target_names
    
    # 2. Try fetching from file (The robust fix)
    tracks = download_official_tracks()
    if tracks:
        return tracks

    print("WARNING: Could not find ANY track names. Using IDs.")
    # Assuming Enformer standard 5313
    return [f"Track_{i}" for i in range(5313)]

def clean_track_name(raw_name):
    parts = raw_name.replace(':', ' ').replace('_', ' ').split()
    vocab = ['CTCF', 'PU.1', 'SPI1', 'H3K27AC', 'H3K4ME3', 'POL2', 'DNASE', 'ATAC']
    for word in parts:
        if word.upper() in vocab: return word.upper()
    return raw_name

# --- 4. FILE WRITERS ---

def save_scan_results(results, args, coords):
    df = pd.DataFrame(results)
    csv_name = f"{args.gene if args.gene else 'Region'}_{args.tissue}_{args.assay}_scan.csv"
    csv_path = os.path.join(args.out_dir, csv_name)
    df.to_csv(csv_path, index=False)
    print(f"   [+] CSV Saved: {csv_path}")
    
    bg_name = csv_name.replace('.csv', '.bedgraph')
    bg_path = os.path.join(args.out_dir, bg_name)
    with open(bg_path, 'w') as f:
        f.write(f"track type=bedGraph name='{args.gene} Impact' color=255,0,0\n")
        for r in results:
            f.write(f"{r['chrom']}\t{r['start']}\t{r['end']}\t{r['impact']:.5f}\n")
    print(f"   [+] BedGraph Saved: {bg_path}")

def save_inspection_report(data, args):
    if not data: return
    df = pd.DataFrame(data)
    csv_path = os.path.join(args.out_dir, f"{args.gene if args.gene else 'Region'}_inspection_report.csv")
    df.to_csv(csv_path, index=False)
    print(f"   [+] Inspection Report Saved: {csv_path}")