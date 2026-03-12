"""
Sovereign MLX port of prepare.py.
Handles data prep (via huggingface-cli), tokenizer training, and MLX-native utilities.
"""

import os
import sys
import time
import math
import argparse
import pickle
import subprocess
from multiprocessing import Pool

import pyarrow.parquet as pq
import rustbpe
import tiktoken
import mlx.core as mx
import numpy as np

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048       # context length
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
EVAL_TOKENS = 50 * 512   # Balanced for fast evaluation on Mac GPUs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
VOCAB_SIZE = 8192

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Data download (Using huggingface-cli)
# ---------------------------------------------------------------------------

def download_data():
    """Download TinyStories parquet shards using huggingface-cli."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Data: Downloading TinyStories parquet files to {DATA_DIR} using huggingface-cli...")
    
    # Download all parquet files from the 'data' directory of the dataset
    cmd = f"huggingface-cli download roneneldan/TinyStories --include 'data/*.parquet' --local-dir {DATA_DIR} --local-dir-use-symlinks False"
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"Data: Download complete.")
    except subprocess.CalledProcessError as e:
        print(f"Data: Download failed with error: {e}")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def list_parquet_files():
    """Return sorted list of parquet file paths in the data directory."""
    # TinyStories parquet files are inside a 'data' subdirectory after download
    search_dir = os.path.join(DATA_DIR, "data")
    if not os.path.exists(search_dir):
        search_dir = DATA_DIR 
        
    files = sorted(f for f in os.listdir(search_dir) if f.endswith(".parquet"))
    return [os.path.join(search_dir, f) for f in files]

def text_iterator(max_chars=1_000_000_000, doc_cap=10_000):
    """Yield documents from training split (all shards except the first)."""
    parquet_paths = list_parquet_files()
    if not parquet_paths: return
    val_path = parquet_paths[0]
    parquet_paths = [p for p in parquet_paths if p != val_path]
    nchars = 0
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return

def train_tokenizer():
    """Train BPE tokenizer using rustbpe, save as tiktoken pickle."""
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.npy")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    parquet_files = list_parquet_files()
    if not parquet_files:
        print("Tokenizer: no data found. Download data first.")
        sys.exit(1)

    print("Tokenizer: training BPE tokenizer...")
    t0 = time.time()
    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)

    # Build tiktoken encoding from trained merges
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=tokenizer.get_pattern(),
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        token_bytes_list.append(0 if token_str in set(SPECIAL_TOKENS) else len(token_str.encode("utf-8")))
    
    np.save(token_bytes_path, np.array(token_bytes_list, dtype=np.int32))
    print(f"Tokenizer: trained in {time.time() - t0:.1f}s")

# ---------------------------------------------------------------------------
# Runtime utilities (MLX Native)
# ---------------------------------------------------------------------------

class Tokenizer:
    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None: ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids: row.insert(0, prepend_id)
        return ids

    def decode(self, ids):
        if hasattr(ids, "tolist"): ids = ids.tolist()
        return self.enc.decode(ids)

def get_token_bytes():
    path = os.path.join(TOKENIZER_DIR, "token_bytes.npy")
    return mx.array(np.load(path))

def _document_batches(split, tokenizer_batch_size=128):
    parquet_paths = list_parquet_files()
    if not parquet_paths: return
    val_path = parquet_paths[0]
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
    else:
        parquet_paths = [val_path]
    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], epoch
        epoch += 1

def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    
    while True:
        rows = []
        for _ in range(B):
            row = []
            while len(row) < row_capacity:
                while len(doc_buffer) < buffer_size:
                    doc_batch, epoch = next(batches)
                    doc_buffer.extend(tokenizer.encode(doc_batch, prepend=bos_token))
                
                remaining = row_capacity - len(row)
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    if len(doc) <= remaining and len(doc) > best_len:
                        best_idx, best_len = i, len(doc)
                
                if best_idx >= 0:
                    row.extend(doc_buffer.pop(best_idx))
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row.extend(doc[:remaining])
            rows.append(row)
        
        arr = mx.array(rows)
        yield arr[:, :-1], arr[:, 1:], epoch

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_bpb(model, tokenizer, batch_size):
    token_bytes = get_token_bytes()
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    
    for _ in range(steps):
        x, y, _ = next(val_loader)
        from train import loss_fn_eval
        loss_flat = loss_fn_eval(model, x, y) 
        
        y_flat = y.reshape(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        
        total_nats += mx.sum(loss_flat * mask).item()
        total_bytes += mx.sum(nbytes).item()
        
    return total_nats / (math.log(2) * total_bytes)

if __name__ == "__main__":
    download_data()
    train_tokenizer()
    print("Done! Ready to train with MLX.")
