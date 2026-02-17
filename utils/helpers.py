import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

# --- 1. COORDINATE & SEQUENCE HANDLING ---

def resolve_coordinates(args):
    if args.gene and args.gene.upper() == "TFEB":
        tss = 41704353
        half_window = args.window_size // 2
        return {
            'chrom': 'chr6',
            'start': tss - half_window,
            'end': tss + half_window,
            'strand': '-',
            'tss': tss,
            'gene_start_rel': tss - 5000,
            'gene_end_rel': tss + 45000
        }
    elif args.region:
        try:
            chrom, span = args.region.split(':')
            start, end = map(int, span.split('-'))
            return {
                'chrom': chrom,
                'start': start,
                'end': end,
                'strand': '+',
                'tss': start
            }
        except ValueError:
            raise ValueError("Invalid Region Format. Use 'chr:start-end'")
    else:
        raise ValueError("No Target Specified. Use --gene TFEB or --region.")

def fetch_sequence(model, coords):
    keys = list(model._fasta_extractors.keys())
    hum_key = next((k for k in keys if "9606" in str(k) or "SAPIENS" in str(k).upper()), None)
    
    if not hum_key:
        print("WARNING: Could not identify Human Fasta Extractor. Using first available.")
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

# --- 2. MODEL INTERACTION & PREDICTION ---

def find_track_indices(model, tissue, assay):
    all_tracks = get_track_metadata(model)
    
    indices = []
    for i, name in enumerate(all_tracks):
        if tissue.lower() in name.lower() and assay.lower() in name.lower():
            indices.append(i)
            
    if not indices:
        print(f"WARNING: No tracks found for '{tissue}' + '{assay}'. using first 5 tracks as fallback.")
        return list(range(5))
        
    return indices

def get_head_key(preds, assay_name):
    keys = list(preds.keys())
    target = next((k for k in keys if 'human' in str(k).lower()), None)
    if not target:
        target = keys[0]
    return target

def predict_baseline(model, params, state, seq, track_indices, agg_mode):
    enc = one_hot_encode(seq)[jnp.newaxis, ...]
    
    with jax.disable_jit():
        # FIX: Explicit keyword arguments
        preds = model._predict(
            params, state, enc, jnp.array([0]), 
            negative_strand_mask=jnp.array([False]), 
            strand_reindexing=None
        )
    
    head_key = get_head_key(preds, 'human')
    data = np.array(preds[head_key])
    selected_data = data[0, :, track_indices]
    
    if agg_mode == 'mean':
        return float(np.nanmean(selected_data))
    elif agg_mode == 'max':
        return float(np.nanmax(selected_data))
    return float(np.nanmean(selected_data))

def predict_all_tracks(model, params, state, seq):
    enc = one_hot_encode(seq)[jnp.newaxis, ...]
    
    with jax.disable_jit():
        # FIX: Explicit keyword arguments
        raw_out = model._predict(
            params, state, enc, jnp.array([0]), 
            negative_strand_mask=jnp.array([False]), 
            strand_reindexing=None
        )
    
    head_key = get_head_key(raw_out, 'human')
    return np.array(raw_out[head_key][0])

# --- 3. METADATA & INSPECTION UTILS ---

def get_track_metadata(model):
    # Debugging: Print available attributes to find where track names are
    # keys = dir(model)
    # print(f"DEBUG: Model attributes: {[k for k in keys if 'track' in k or 'name' in k]}")

    if hasattr(model, 'track_names'):
        return model.track_names
    if hasattr(model, 'config') and hasattr(model.config, 'target_names'):
        return model.config.target_names
    if hasattr(model, '_track_names'):
        return model._track_names
    
    # Try looking in the generator config if it exists
    if hasattr(model, 'generator') and hasattr(model.generator, 'track_names'):
        return model.generator.track_names

    print("WARNING: Could not find track names in model metadata. Returning dummy IDs.")
    return [f"Track_{i}" for i in range(5313)]

def clean_track_name(raw_name):
    parts = raw_name.replace(':', ' ').replace('_', ' ').split()
    vocab = ['CTCF', 'PU.1', 'SPI1', 'H3K27AC', 'H3K4ME3', 'H3K4ME1', 'H3K27ME3', 'POL2', 'DNASE', 'ATAC']
    
    for word in parts:
        if word.upper() in vocab:
            return word.upper()
    
    if ':' in raw_name:
        try:
            return raw_name.split(':')[1]
        except:
            pass
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
        f.write(f"track type=bedGraph name='{args.gene} {args.tissue} Impact' description='Impact Score (Baseline - Mutant)' visibility=full autoScale=on color=255,0,0\n")
        for r in results:
            f.write(f"{r['chrom']}\t{r['start']}\t{r['end']}\t{r['impact']:.5f}\n")
    print(f"   [+] BedGraph Saved: {bg_path}")

def save_inspection_report(data, args):
    if not data:
        print("   [!] No inspection data generated.")
        return
    df = pd.DataFrame(data)
    csv_name = f"{args.gene if args.gene else 'Region'}_inspection_report.csv"
    csv_path = os.path.join(args.out_dir, csv_name)
    cols = ['location', 'type', 'impact_on_gene', 'top_regulatory_signals']
    existing_cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]
    df = df[existing_cols]
    df.to_csv(csv_path, index=False)
    print(f"   [+] Inspection Report Saved: {csv_path}")