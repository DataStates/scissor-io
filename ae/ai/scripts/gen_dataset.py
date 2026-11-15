#!/usr/bin/env python3
"""
gen_dataset.py (Phase 2): Synthetic Dataset Creator
Creates binary data files of the exact sizes specified in the config.
Each file is filled with pseudo-random bytes (reproducible per file_id).
Files are pre-allocated to avoid filesystem fragmentation.

Usage:
    python gen_dataset.py --config config/resnet50.yaml --outdir /data/bench
"""

import argparse
import os
import struct
import sys
import numpy as np
import yaml


# File header (64 bytes)
FILE_MAGIC   = b"AIOD"           # AI IO Dataset
FILE_VERSION = 1
HEADER_FMT   = "<4sII56s"        # magic(4)+version(4)+file_id(4)+pad(52)
HEADER_SIZE  = struct.calcsize(HEADER_FMT)
BLOCK_SIZE   = 1 << 20           # 1 MB write block


def create_file(file_id: int, file_size_b: int, outdir: str,
                seed: int = 42) -> str:
    """Create a single synthetic data file."""
    path = os.path.join(outdir, f"data_{file_id:04d}.bin")

    if os.path.exists(path) and os.path.getsize(path) == file_size_b:
        print(f"  [skip] {path} already exists ({file_size_b/(1<<30):.1f} GB)")
        return path

    print(f"  [create] {path} ({file_size_b/(1<<30):.2f} GB) ...",
          end="", flush=True)

    rng = np.random.default_rng(seed + file_id)

    with open(path, "wb") as f:
        # Write header
        header = struct.pack(HEADER_FMT,
                             FILE_MAGIC,
                             FILE_VERSION,
                             file_id,
                             b"\x00" * 52)
        f.write(header)

        # Fill remaining space with pseudo-random blocks
        written = HEADER_SIZE
        while written < file_size_b:
            to_write = min(BLOCK_SIZE, file_size_b - written)
            # Generate random bytes block
            block = rng.bytes(to_write)
            f.write(block)
            written += to_write

        # Ensure exact size (pad if needed)
        if written < file_size_b:
            f.write(b"\x00" * (file_size_b - written))

        # Sync to disk
        f.flush()
        os.fsync(f.fileno())

    actual = os.path.getsize(path)
    print(f" done ({actual/(1<<30):.2f} GB)")
    return path


def verify_file(path: str, expected_size: int) -> bool:
    """Quick sanity check on file."""
    if not os.path.exists(path):
        return False
    actual = os.path.getsize(path)
    if actual != expected_size:
        print(f"  [WARN] {path}: size {actual} != expected {expected_size}")
        return False
    # Read and check header magic
    with open(path, "rb") as f:
        header_bytes = f.read(HEADER_SIZE)
    magic = struct.unpack_from("4s", header_bytes)[0]
    return magic == FILE_MAGIC


def main():
    parser = argparse.ArgumentParser(
        description="Create synthetic dataset files from YAML config")
    parser.add_argument("--config",  required=True,
                        help="Path to workload YAML config")
    parser.add_argument("--outdir",  required=True,
                        help="Output directory for data files")
    parser.add_argument("--verify",  action="store_true",
                        help="Verify existing files instead of creating")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ds          = cfg["dataset"]
    n_files     = ds["n_files"]
    file_size_b = int(ds["file_size_gb"] * (1 << 30))
    seed        = ds.get("random_seed", 42)

    os.makedirs(args.outdir, exist_ok=True)

    total_gb = n_files * ds["file_size_gb"]
    print(f"\n[gen_dataset] Workload: {cfg['workload']['name']}")
    print(f"[gen_dataset] Files: {n_files} × {ds['file_size_gb']} GB "
          f"= {total_gb} GB total")
    print(f"[gen_dataset] Output: {args.outdir}\n")

    # Check available disk space
    statvfs = os.statvfs(args.outdir)
    avail_gb = (statvfs.f_bavail * statvfs.f_frsize) / (1 << 30)
    if avail_gb < total_gb * 1.1:
        print(f"[ERROR] Insufficient disk space: {avail_gb:.1f} GB available, "
              f"{total_gb:.1f} GB needed")
        sys.exit(1)
    print(f"[gen_dataset] Disk available: {avail_gb:.1f} GB")

    paths = []
    if args.verify:
        print("[gen_dataset] Verifying existing files ...")
        all_ok = True
        for fid in range(n_files):
            path = os.path.join(args.outdir, f"data_{fid:04d}.bin")
            ok   = verify_file(path, file_size_b)
            status = "OK" if ok else "FAIL"
            print(f"  {path}: {status}")
            if not ok:
                all_ok = False
        print(f"\n[gen_dataset] Verification: {'PASSED' if all_ok else 'FAILED'}")
        sys.exit(0 if all_ok else 1)

    print("[gen_dataset] Creating dataset files ...")
    for fid in range(n_files):
        path = create_file(fid, file_size_b, args.outdir, seed)
        paths.append(path)

    # Write file manifest (list of paths → read by bench.c)
    manifest_path = os.path.join(args.outdir, "manifest.txt")
    with open(manifest_path, "w") as f:
        for path in paths:
            f.write(path + "\n")
    print(f"\n[gen_dataset] Manifest → {manifest_path}")
    print(f"[gen_dataset] Done. {n_files} files × "
          f"{ds['file_size_gb']} GB = {total_gb} GB created.")


if __name__ == "__main__":
    main()
