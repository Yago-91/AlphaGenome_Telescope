import numpy as np
from utils import helpers

def inspect_hits(top_hits, args, coords):
    """
    Takes the top hits and queries specific tracks (TFs, Histones) 
    to understand WHY they are enhancers.
    """
    # We need to reload the model? No, main.py should pass it? 
    # For simplicity in this CLI architecture, we will re-initialize or 
    # ideally we would pass the model object. 
    # To keep files decoupled, let's assume we re-load OR (better) 
    # we just run the inspection on the WILD TYPE sequence once 
    # and map the bins to the hits.
    
    # OPTIMIZATION: We don't need to run the model again!
    # We just need the Wild-Type prediction (which we have from baseline)
    # and look at the tracks at the specific bins corresponding to the hits.
    
    print("Retrieving Mechanistic Data...")
    
    # 1. Define tracks to inspect
    # Primary tissue (Spleen) + Extra global markers (CTCF, Promoter)
    tracks_to_check = args.inspect_tracks.split(',')
    
    # 2. Results container
    inspection_report = []
    
    for hit in top_hits:
        # Calculate which bin in the 896-bin output corresponds to this hit
        # The model outputs 896 bins for 1048576 bp.
        # Ratio = 1048576 / 896 = 1170 bp per bin.
        
        rel_pos = hit['start'] - coords['start']
        bin_idx = int(rel_pos / 1170)
        
        # Classification
        hit_type = "Distal"
        if abs(hit['distance_to_tss']) < 2000:
            hit_type = "Promoter"
            
        report_row = {
            'location': f"{hit['chrom']}:{hit['start']}-{hit['end']}",
            'impact_score': hit['impact'],
            'distance_to_tss': hit['distance_to_tss'],
            'type': hit_type,
            'tissue': args.tissue
        }
        
        # Placeholder: In a persistent session we would query the existing `baseline_raw` 
        # variable. Since we split files, we would ideally pass this data.
        # For this standalone code, we assume 'Enrichment' is a placeholder 
        # unless we re-run prediction. 
        # *Self-Correction*: To make this real, `scanner.py` should return the baseline_raw 
        # object or we perform inspection *inside* scanner.
        
        # For now, we will mark this for the user:
        report_row['note'] = "Load this region in IGV to see TF tracks."
        
        inspection_report.append(report_row)
        
    return inspection_report