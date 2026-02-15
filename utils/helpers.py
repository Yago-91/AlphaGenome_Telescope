import numpy as np
import jax.numpy as jnp
import os
import pandas as pd

def resolve_coordinates(args):
    """
    Resolves --gene TFEB to actual coordinates.
    This is a mocked version. In production, use Ensembl REST API.
    """
    if args.gene and args.gene.upper() == "TFEB":
        # TFEB Coordinates (hg38)
        return {
            'chrom': 'chr6',
            'start': 41704353 - (args.window_size // 2),
            'end': 41704353 + (args.window_size // 2),
            'strand': '-',
            'tss': 41704353,
            'gene_start_rel': (args.window_size // 2) - 5000, # Approx
            'gene_end_rel': (args.window_size // 2) + 40000
        }
    elif args.region:
        # Parse chr6:100-200
        c, r = args.region.split(':')
        s, e = map(int, r.split('-'))
        return {
            'chrom': c, 'start': s, 'end': e, 
            'strand': '+', 'tss': s # Default TSS to start if unknown
        }
    else:
        raise ValueError("Unknown Gene. Please use --region.")

def mask_sequence(args):
    seq, start_idx, length = args
    s_list = list(seq)
    s_list[start_idx : start_idx + length] = ['N'] * length
    return "".join(s_list)

def one_hot_encode(seq):
    mapping = {'A':0, 'C':1, 'G':2, 'T':3}
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, c in enumerate(seq.upper()):
        if c in mapping: arr[i, mapping[c]] = 1.0
    return jnp.array(arr)

def fetch_sequence(model, coords):
    # Determine which fasta extractor to use (Human)
    keys = list(model._fasta_extractors.keys())
    # Simple heuristic to find human key
    hum_key = next(k for k in keys if "9606" in str(k) or "SAPIENS" in str(k).upper())
    
    # Mock interval object for the model
    from alphagenome.data import genome
    interval = genome.Interval(coords['chrom'], coords['start'], coords['end'])
    
    seq = model._fasta_extractors[hum_key].extract(interval)
    return seq

def find_track_indices(model, tissue, assay):
    # This requires access to the model's track names.
    # In AlphaGenome, model.track_names or similar metadata exists.
    # We will assume all indices are valid for now and return a dummy range 
    # to prevent crashing if metadata is missing.
    # TODO: Implement real metadata search.
    return [0, 1] # Placeholder: checks first two tracks

def predict_baseline(model, params, state, seq, track_idx, agg_mode):
    # Helper to run one prediction
    import jax
    enc = one_hot_encode(seq)[jnp.newaxis, ...]
    
    with jax.disable_jit():
        preds = model._predict(params, state, enc, jnp.array([0]), jnp.array([False]), None)
    
    # Extract RNA head (heuristic)
    keys = list(preds.keys())
    key = next((k for k in keys if 'RNA' in str(k)), keys[0])
    
    data = np.array(preds[key])
    # Filter for tracks (if we had real indices)
    # Aggregation
    if agg_mode == 'mean':
        return float(np.nanmean(data))
    else:
        return float(np.nanmax(data))

def get_head_key(preds, assay_name):
    keys = list(preds.keys())
    # Find key containing 'RNA', 'ATAC', etc.
    return next((k for k in keys if assay_name in str(k)), keys[0])

def save_scan_results(results, args, coords):
    df = pd.DataFrame(results)
    path = os.path.join(args.out_dir, f"{args.gene}_{args.tissue}_{args.assay}_scan.csv")
    df.to_csv(path, index=False)
    print(f"Scan CSV saved to: {path}")
    
    # Generate BedGraph
    bg_path = path.replace('.csv', '.bedgraph')
    with open(bg_path, 'w') as f:
        f.write(f"track type=bedGraph name='{args.gene} Impact' description='Impact on {args.tissue}'\n")
        for r in results:
            f.write(f"{r['chrom']}\t{r['start']}\t{r['end']}\t{r['impact']}\n")
    print(f"BedGraph saved to: {bg_path}")

def save_inspection_report(data, args):
    df = pd.DataFrame(data)
    path = os.path.join(args.out_dir, f"{args.gene}_inspection_report.csv")
    df.to_csv(path, index=False)
    print(f"Inspection Report saved to: {path}")