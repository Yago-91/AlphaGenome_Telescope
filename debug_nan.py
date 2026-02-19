import os
# Strict memory rules
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.numpy as jnp
import numpy as np
from alphagenome_research.model import dna_model, augmentation
from utils import helpers
from argparse import Namespace

# Monkey patch
def no_op(*args, **kwargs): return args[0]
augmentation.reverse_complement = no_op

print("========================================")
print("        NaN ISOLATION DIAGNOSTIC        ")
print("========================================")

# 1. BIOLOGY TEST: FASTA Integrity
print("\n[1] Testing FASTA Sequence Integrity...")
args = Namespace(gene="TFEB", region=None, window_size=1048576)
coords = helpers.resolve_coordinates(args)

print("Loading Model...")
model = dna_model.create_from_huggingface('all_folds')

print(f"Extracting sequence for {coords['chrom']}:{coords['start']}-{coords['end']}...")
seq = helpers.fetch_sequence(model, coords)
seq_len = len(seq)

# Count Ns safely (handles both 'N' and 'n')
n_count = seq.upper().count('N')

print(f"  -> Length extracted: {seq_len}")
print(f"  -> 'N' (Unknown base) count: {n_count} ({(n_count/max(1, seq_len))*100:.2f}%)")
print(f"  -> Sample (first 50bp): {seq[:50]}")

if n_count > seq_len * 0.5:
    print("\n[!] DIAGNOSIS FOUND: The FASTA Extractor is failing.")
    print("    Reason: It returned mostly 'N's. This happens when the chromosome name")
    print("    is wrong (e.g., the FASTA uses '6' but we asked for 'chr6').")
    exit()

# 2. HARDWARE TEST: Math Precision
print("\n[2] Testing GPU Math Precision (Bypassing Biology)...")
print("  -> Generating perfect 'ACGT' dummy sequence...")
dummy_str = "ACGT" * (1048576 // 4)
dummy_enc = helpers.one_hot_encode(dummy_str)[jnp.newaxis, ...]

params = getattr(model, '_params', None) or getattr(model, 'params', None)
state = getattr(model, '_state', None) or getattr(model, 'state', None)

@jax.jit
def fast_predict(p, s, inputs):
    return model._predict(p, s, inputs, jnp.zeros((1,), dtype=jnp.int32), jnp.zeros((1,), dtype=bool), None)

print("  -> Running JIT Inference on Dummy Sequence (Float32)...")
try:
    preds = fast_predict(params, state, dummy_enc)
    key = helpers.get_head_key(preds, 'RNA')
    val = float(np.nanmean(np.array(preds[key])))
    print(f"  -> RESULT: {val}")
    
    if np.isnan(val):
        print("\n[!] DIAGNOSIS FOUND: Hardware Math Overflow.")
        print("    Reason: Float32 attention is overflowing on your GPU.")
        print("    We will need to enforce Google's native BFloat16 policy.")
    else:
        print("\n[+] Math is stable! The hardware is functioning perfectly.")
        
except Exception as e:
    print(f"  -> CRASH: {e}")

print("\n========================================")