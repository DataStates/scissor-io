#!/usr/bin/env python3
"""
gen_dlio_dataset.py: Generate data files in DLIO indexed_binary format.

Creates the exact same on-disk layout as DLIO's IndexedBinaryGenerator:
  {outdir}/train/img_{i:0Nd}_of_{N}.indexed_binary          : raw sample bytes
  {outdir}/train/img_{i:0Nd}_of_{N}.indexed_binary.off.idx  : uint64 offsets
  {outdir}/train/img_{i:0Nd}_of_{N}.indexed_binary.sz.idx   : uint64 sizes

The format is compatible with:
  - DLIO's IndexedBinaryReader (get_sample uses these index files)
  - gen_trace_dlio.py (reads the same index files to build IO traces)
  - Megatron-LM indexed datasets (same binary layout, different naming)

Usage:
    python gen_dlio_dataset.py --config config/dlio/bert_packed.yaml \\
                               --outdir /tmp/bench_data
    python gen_dlio_dataset.py --config config/dlio/resnet50_dist.yaml \\
                               --outdir /tmp/bench_data
"""

import argparse
import math
import os
import struct
import sys

import numpy as np
import yaml


BLOCK_SIZE = 1 << 20    # 1 MB write chunk for data files


def _zpad(i: int, total: int) -> str:
    """Zero-pad i to the same width as total, matching DLIO's add_padding()."""
    return str(i).zfill(len(str(total)))


def file_paths(outdir: str, num_files: int, fmt: str = "indexed_binary"):
    """Yield (data_path, off_path, sz_path) for each training file."""
    nd = len(str(num_files))
    train_dir = os.path.join(outdir, "train")
    for i in range(num_files):
        base = os.path.join(train_dir,
                            f"img_{str(i).zfill(nd)}_of_{num_files}.{fmt}")
        yield base, base + ".off.idx", base + ".sz.idx"


def generate_file(data_path: str, off_path: str, sz_path: str,
                  num_samples: int, sample_size: int,
                  file_id: int, seed: int = 42, skip: bool = False):
    """
    Write one indexed_binary data file and its two index files.

    Index format (matching DLIO's IndexedBinaryGenerator exactly):
      .off.idx: struct.pack('Q' * n, offset_0, offset_1, ...)  (uint64 LE)
      .sz.idx:  struct.pack('Q' * n, size_0, size_1, ...)      (uint64 LE)

    For fixed-size samples, offset_i = i * sample_size and size_i = sample_size.
    """
    total_size = num_samples * sample_size

    if skip and os.path.exists(data_path) and os.path.getsize(data_path) == total_size:
        print(f"  [skip] {os.path.basename(data_path)}")
        return

    print(f"  [write] {os.path.basename(data_path)}  "
          f"({num_samples} × {sample_size//1024} KB = "
          f"{total_size/(1<<20):.1f} MB) ...", end="", flush=True)

    rng = np.random.default_rng(seed + file_id)

    # Write data file (back-to-back samples, no header, pure DLIO format)
    with open(data_path, "wb") as f:
        written = 0
        while written < total_size:
            chunk = min(BLOCK_SIZE, total_size - written)
            f.write(rng.bytes(chunk))
            written += chunk
        f.flush()
        os.fsync(f.fileno())

    # Write offset index: uint64 LE, matching struct.pack('Q'*n, ...)
    offsets = np.arange(num_samples, dtype=np.uint64) * sample_size
    with open(off_path, "wb") as f:
        f.write(struct.pack(f"<{num_samples}Q", *offsets))

    # Write size index: all equal for fixed-size samples
    sizes = np.full(num_samples, sample_size, dtype=np.uint64)
    with open(sz_path, "wb") as f:
        f.write(struct.pack(f"<{num_samples}Q", *sizes))

    print(" done")


def main():
    parser = argparse.ArgumentParser(
        description="Generate DLIO indexed_binary dataset for bench_dlio.py")
    parser.add_argument("--config",  required=True,
                        help="DLIO-format workload YAML (e.g. config/dlio/bert_packed.yaml)")
    parser.add_argument("--outdir",  required=True,
                        help="Root output directory. Files go in {outdir}/train/")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files that already exist with the correct size")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ds           = cfg["dataset"]
    num_files    = ds["num_files_train"]
    num_samples  = ds["num_samples_per_file"]
    sample_size  = ds["record_length_bytes"]
    fmt          = ds.get("format", "indexed_binary")
    total_gb     = num_files * num_samples * sample_size / (1 << 30)

    print(f"[gen_dlio_dataset] Workload: {cfg['model']['name']}")
    print(f"[gen_dlio_dataset] Files:    {num_files} × {num_samples} samples × "
          f"{sample_size//1024} KB = {total_gb:.2f} GB")

    train_dir = os.path.join(args.outdir, "train")
    os.makedirs(train_dir, exist_ok=True)

    for data_path, off_path, sz_path in file_paths(args.outdir, num_files, fmt):
        fid = int(os.path.basename(data_path).split("_")[1])
        generate_file(data_path, off_path, sz_path,
                      num_samples, sample_size, fid,
                      seed=args.seed, skip=args.skip_existing)

    # Write manifest (sorted list of data file paths, one per line)
    manifest_path = os.path.join(args.outdir, "manifest.txt")
    paths = sorted(p for p, _, _ in file_paths(args.outdir, num_files, fmt))
    with open(manifest_path, "w") as f:
        for p in paths:
            f.write(os.path.abspath(p) + "\n")

    print(f"\n[gen_dlio_dataset] Manifest → {manifest_path}")
    print(f"[gen_dlio_dataset] Done.")


if __name__ == "__main__":
    main()
