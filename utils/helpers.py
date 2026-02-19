import os
import requests
import json
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import mygene

# --- 1. COORDINATE & SEQUENCE HANDLING ---
def resolve_coordinates(args):
    if args.region:
        try:
            chrom, span = args.region.split(':')
            start, end = map(int, span.split('-'))
            return {'chrom': chrom, 'start': start, 'end': end, 'strand': '+', 'tss': start}
        except ValueError: raise ValueError("Invalid Region Format.")
    elif args.gene:
        print(f"   [i] Querying MyGene.info for '{args.gene}' coordinates (hg38)...")
        mg = mygene.MyGeneInfo()
        results = mg.query(args.gene, scopes='symbol', fields='genomic_pos', species='human')
        if not results or 'hits' not in results or not results['hits']:
            raise ValueError(f"Gene '{args.gene}' not found.")
        hit = results['hits'][0]
        gpos = hit.get('genomic_pos', {})
        if isinstance(gpos, list): gpos = gpos[0]
        if not gpos: raise ValueError(f"No coordinates for {args.gene}")

        chrom = f"chr{gpos['chr']}"
        tss = gpos['start'] if gpos['strand'] == 1 else gpos['end']
        strand = '+' if gpos['strand'] == 1 else '-'
        print(f"       -> Found {args.gene} on {chrom} ({strand} strand). TSS: {tss}")
        
        half = args.window_size // 2
        return {
            'chrom': chrom, 'start': tss - half, 'end': tss + half, 
            'strand': strand, 'tss': tss,
            'gene_start_rel': (gpos['start'] - (tss - half)) - 1000,
            'gene_end_rel': (gpos['end'] - (tss - half)) + 1000
        }
    raise ValueError("No Target Specified.")

def fetch_sequence(model, coords):
    keys = list(model._fasta_extractors.keys())
    hum_key = next((k for k in keys if "9606" in str(k) or "SAPIENS" in str(k).upper()), keys[0])
    from alphagenome.data import genome
    return model._fasta_extractors[hum_key].extract(genome.Interval(coords['chrom'], coords['start'], coords['end']))

def one_hot_encode(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, char in enumerate(seq.upper()):
        if char in mapping: arr[i, mapping[char]] = 1.0
    return jnp.array(arr)

def mask_sequence(args):
    seq, start_idx, length = args
    s_list = list(seq)
    end = min(start_idx + length, len(s_list))
    s_list[start_idx : end] = ['N'] * (end - start_idx)
    return "".join(s_list)

# --- 2. MULTI-HEAD SELECTION ---
def get_head_key(preds, assay_name):
    keys = list(preds.keys())
    assay = assay_name.upper()
    
    mapping = {
        'RNA': ['RNA', 'SEQ'],
        'CAGE': ['CAGE'],
        'ATAC': ['ATAC'],
        'DNASE': ['DNASE'],
        'CHIP': ['CHIP', 'TF'],
        'HISTONE': ['HISTONE']
    }
    
    keywords = mapping.get(assay, [assay])
    
    for k in keys:
        k_str = str(k).upper()
        if all(kw in k_str for kw in keywords):
            return k
            
    best_k = keys[0]
    max_dim = 0
    for k, v in preds.items():
        if hasattr(v, 'shape') and v.shape[-1] > max_dim:
            max_dim = v.shape[-1]
            best_k = k
            
    print(f"   [!] WARNING: Could not find exact head for '{assay}'. Fallback to largest: '{best_k}'")
    return best_k

def find_track_indices(model, tissue, assay):
    # AlphaGenome open source metadata not available, defaulting to Blind Mode.
    return "ALL" 

# --- 3. WRITERS ---
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

# Unused legacy stubs
def predict_baseline(model, params, state, seq, track_indices, agg_mode): pass
def predict_all_tracks(model, params, state, seq): pass
def get_track_metadata(model): return []
def clean_track_name(raw_name): return raw_name