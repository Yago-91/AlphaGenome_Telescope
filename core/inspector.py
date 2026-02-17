import numpy as np
from utils import helpers

def inspect_hits(top_hits, args, model, params, state, coords, wt_seq):
    """
    Unbiased discovery using the pre-loaded model.
    """
    print(f"--- Running Inspector on {len(top_hits)} regions ---")
    
    # 1. Get Full Predictions (Using passed model/params)
    print("Generating full-track regulatory map...")
    # Pass params/state explicitly to avoid reloading logic in helper
    full_preds = helpers.predict_all_tracks(model, params, state, wt_seq) 
    
    # 2. Get Metadata
    track_names = helpers.get_track_metadata(model)
    
    inspection_report = []
    
    for hit in top_hits:
        rel_pos = hit['start'] - coords['start']
        bin_idx = int(rel_pos / 128)
        bin_idx = min(max(bin_idx, 0), full_preds.shape[0] - 1)
        
        track_values = full_preds[bin_idx, :]
        sorted_indices = np.argsort(track_values)[::-1]
        
        top_factors = []
        seen_factors = set()
        
        # Forced Inclusions (if user asked for specific tracks)
        forced_tracks = [t.strip().upper() for t in args.inspect_tracks.split(',')] if args.inspect_tracks else []
        
        # Scan ranked list
        for idx in sorted_indices:
            if len(top_factors) >= 10: break
            
            raw_name = track_names[idx]
            score = float(track_values[idx])
            
            # Check forced tracks
            is_forced = any(f in raw_name.upper() for f in forced_tracks)
            
            # Heuristics for "Interesting" tracks
            is_regulatory = any(x in raw_name.upper() for x in ['CHIP', 'TF', 'BINDING', 'DNASE', 'ATAC', 'H3K'])
            is_expression = any(x in raw_name.upper() for x in ['RNA', 'CAGE'])
            
            if is_forced or (is_regulatory and not is_expression):
                factor_name = helpers.clean_track_name(raw_name)
                if factor_name not in seen_factors:
                    top_factors.append(f"{factor_name} ({score:.2f})")
                    seen_factors.add(factor_name)

        hit_type = "Promoter" if abs(hit['distance_to_tss']) < 2000 else "Distal"
            
        inspection_report.append({
            'location': f"{hit['chrom']}:{hit['start']}-{hit['end']}",
            'impact_on_gene': hit['impact'],
            'type': hit_type,
            'top_regulatory_signals': "; ".join(top_factors)
        })
        
    return inspection_report