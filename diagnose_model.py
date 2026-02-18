import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.numpy as jnp
import numpy as np
from alphagenome_research.model import dna_model, augmentation

# Monkey patch to avoid augmentation errors
def no_op(*args, **kwargs): return args[0]
augmentation.reverse_complement = no_op

def inspect():
    print("--- ALPHAGENOME MODEL DIAGNOSTIC ---")
    print("[1] Loading Model...")
    model = dna_model.create_from_huggingface('all_folds')
    
    params = getattr(model, '_params', None) or getattr(model, 'params', None)
    state = getattr(model, '_state', None) or getattr(model, 'state', None)
    
    print("\n[2] inspecting Configuration...")
    if hasattr(model, 'config'):
        print(f"    Config Object Found: {type(model.config)}")
        # Try to print relevant config keys
        keys = dir(model.config)
        for k in keys:
            if not k.startswith('_'):
                val = getattr(model.config, k)
                if isinstance(val, (int, str, list, dict)):
                    print(f"    - {k}: {val}")
    else:
        print("    [!] No 'config' attribute found on model wrapper.")

    print("\n[3] Inspecting Embeddings (The Organism Clue)...")
    # We look for the embedding layer to see how many organisms it handles.
    # Usually located in 'stem' or 'embedding' params.
    found_embedding = False
    for k, v in params.items():
        if 'embed' in k.lower() or 'organism' in k.lower():
            # v is a dict of parameters for that layer
            for sub_k, sub_v in v.items():
                if hasattr(sub_v, 'shape'):
                    print(f"    - Parameter '{k}/{sub_k}': Shape {sub_v.shape}")
                    if sub_v.shape[0] == 1:
                        print("      -> CONCLUSION: Model supports only ONE organism (Index 0).")
                    elif sub_v.shape[0] > 1:
                        print(f"      -> CONCLUSION: Model supports {sub_v.shape[0]} organisms (Indices 0-{sub_v.shape[0]-1}).")
                    found_embedding = True
    
    if not found_embedding:
        print("    [!] Could not locate organism embedding weights.")

    print("\n[4] Running Dummy Inference (To Mapping Outputs)...")
    # Create a tiny dummy input (Sequence length 196,608 is minimal for some blocks, 
    # but let's try 1Mb to be safe as that is the standard window)
    SEQ_LEN = 1048576 
    dummy_seq = np.zeros((1, SEQ_LEN, 4), dtype=np.float32)
    
    # Try Index 0 first
    print("    Running with Organism Index [0]...")
    try:
        preds = model._predict(params, state, dummy_seq, 
                             jnp.zeros((1,), dtype=jnp.int32), # Index 0
                             negative_strand_mask=jnp.array([False]), 
                             strand_reindexing=None)
        
        print("\n[5] MODEL OUTPUT MAP (The Truth):")
        for k, v in preds.items():
            if hasattr(v, 'shape'):
                print(f"    Key: '{k}' | Shape: {v.shape}")
                # Heuristic for Human vs Mouse
                if v.shape[-1] == 5313:
                    print("         -> MATCH: Matches Enformer Human Track Count!")
                elif v.shape[-1] == 1664:
                    print("         -> MATCH: Matches Enformer Mouse Track Count!")
                elif v.shape[-1] == 768:
                    print("         -> INFO: Likely RNA-Seq Head (AlphaGenome specific)")
    except Exception as e:
        print(f"    [!] Crash on Index 0: {e}")

    print("\n-------------------------------------------")

if __name__ == "__main__":
    inspect()