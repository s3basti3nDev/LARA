"""
Dataset utilities for LARA experiments.

Supports:
  - shakespeare : Tiny Shakespeare (fast to download, good for unit tests)
  - fineweb     : FineWeb-Edu 10B token subset (high-quality, real benchmark)

Both are converted to a flat binary (.bin) file of uint16 token IDs.
"""
import os
import urllib.request
import numpy as np
import torch
from torch.utils.data import Dataset


# ──────────────────────────────────────────────────────────────
# Tiny Shakespeare
# ──────────────────────────────────────────────────────────────

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


def prepare_shakespeare(data_dir: str = "data"):
    os.makedirs(data_dir, exist_ok=True)
    raw_path = os.path.join(data_dir, "shakespeare.txt")
    train_path = os.path.join(data_dir, "shakespeare_train.bin")
    val_path = os.path.join(data_dir, "shakespeare_val.bin")

    if os.path.exists(train_path):
        return train_path, val_path

    print("Downloading Tiny Shakespeare...")
    urllib.request.urlretrieve(SHAKESPEARE_URL, raw_path)

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    with open(raw_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = enc.encode_ordinary(text)
    tokens = np.array(tokens, dtype=np.uint16)
    split = int(0.9 * len(tokens))

    tokens[:split].tofile(train_path)
    tokens[split:].tofile(val_path)
    print(f"  train: {split:,} tokens  |  val: {len(tokens)-split:,} tokens")
    return train_path, val_path


# ──────────────────────────────────────────────────────────────
# FineWeb-Edu (HuggingFace — 10B token subset, high quality)
# ──────────────────────────────────────────────────────────────

def prepare_fineweb(data_dir: str = "data", sample: str = "sample-10BT"):
    """
    Download and tokenize FineWeb-Edu from HuggingFace.
    Requires: pip install datasets tiktoken
    Uses the 10B-token sample by default (~20GB download).
    For a quick test, pass sample='sample-350BT' (smaller) — actually use
    'sample-10BT' for the 10B version.
    """
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, f"fineweb_{sample}_train.bin")
    val_path   = os.path.join(data_dir, f"fineweb_{sample}_val.bin")

    if os.path.exists(train_path) and os.path.exists(val_path):
        print("FineWeb-Edu déjà préparé.")
        return train_path, val_path

    print(f"Téléchargement FineWeb-Edu ({sample}) depuis HuggingFace...")
    from datasets import load_dataset
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=sample,
                      split="train", streaming=True)

    def tokenize(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(eot)
        return ids

    # Stream and write train / val (.bin)
    val_tokens = 10_000_000   # ~10M tokens for validation
    train_tokens_written = 0
    val_tokens_written   = 0

    train_f = open(train_path, "wb")
    val_f   = open(val_path,   "wb")

    for doc in ds:
        ids = tokenize(doc)
        arr = np.array(ids, dtype=np.uint16)
        if val_tokens_written < val_tokens:
            arr.tofile(val_f)
            val_tokens_written += len(ids)
        else:
            arr.tofile(train_f)
            train_tokens_written += len(ids)
        if train_tokens_written % 100_000_000 == 0 and train_tokens_written > 0:
            print(f"  {train_tokens_written/1e9:.1f}B train tokens written...")

    train_f.close()
    val_f.close()
    print(f"  train: {train_tokens_written/1e9:.2f}B tokens  |  val: {val_tokens_written/1e6:.1f}M tokens")
    return train_path, val_path


# ──────────────────────────────────────────────────────────────
# Generic binary token dataset
# ──────────────────────────────────────────────────────────────

class TokenDataset(Dataset):
    """
    Memory-mapped dataset over a flat binary file of uint16 token IDs.
    Returns (input_ids, target_ids) pairs of length block_size.
    """

    def __init__(self, bin_path: str, block_size: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = torch.from_numpy(
            self.data[idx: idx + self.block_size + 1].astype(np.int64)
        )
        return chunk[:-1], chunk[1:]


def get_dataloaders(config, num_workers: int = 0):
    """Return (train_loader, val_loader) for the dataset named in config."""
    if config.dataset == "shakespeare":
        train_bin, val_bin = prepare_shakespeare(config.data_dir)
    elif config.dataset == "fineweb":
        train_bin, val_bin = prepare_fineweb(config.data_dir)
    elif config.dataset == "local":
        import importlib.util, sys as _sys
        spec = importlib.util.spec_from_file_location(
            "make_local_dataset",
            os.path.join(os.path.dirname(__file__), "make_local_dataset.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        train_bin, val_bin = mod.prepare_local(config.data_dir)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}.")

    from torch.utils.data import DataLoader, RandomSampler

    train_ds = TokenDataset(train_bin, config.block_size)
    val_ds = TokenDataset(val_bin, config.block_size)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              sampler=RandomSampler(train_ds, replacement=True),
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    return train_loader, val_loader
