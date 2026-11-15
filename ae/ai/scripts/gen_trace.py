#!/usr/bin/env python3
"""
gen_trace.py (Phase 1): Trace Generator
Generates a reproducible IO access trace (CSV + binary) from a YAML config.

Usage:
    python gen_trace.py --config config/resnet50.yaml
    python gen_trace.py --config config/bert.yaml
    python gen_trace.py --config config/unet3d.yaml

Access patterns (set via trace.access_pattern in the YAML):

  uniform   (default / current behaviour)
    Each sample's file and offset are drawn independently and uniformly at
    random over the full dataset.  Produces zero spatial locality, the worst
    case for SCISSOR.  Useful as a baseline.

  epoch
    Simulates a standard shuffled epoch: a fixed permutation of all
    (file, sample_index) pairs is generated once (reproducibly), then steps
    advance through it sequentially.  Multiple DataLoader workers each handle
    a contiguous slice of the permutation, so all workers access their slice
    in near-sequential file order.  This produces the moderate intra-worker
    locality seen in PyTorch/TensorFlow DataLoaders with shuffle=True.

    Config key: trace.n_workers  (default: from compute.n_workers)

  window
    Each step draws all batch_size samples from a randomly-placed contiguous
    window of trace.window_size_mb bytes within each file.  The window is
    chosen once per step and shared across all samples, so every sample in
    the batch is within window_size_mb of every other sample in the same
    file.  This models a shuffle-buffer-based dataloader (WebDataset,
    TFRecord) where the buffer covers a fixed-size chunk of the stream.

    Config key: trace.window_size_mb  (required for window pattern)

  shard
    The file address space is divided into trace.n_shards equal-sized shards.
    Each step selects trace.shards_per_step shards at random and draws all
    batch_size samples exclusively from those shards.  This models
    multi-worker training where each worker pre-fetches one shard at a time
    (common in DLIO / MLPerf storage benchmarks and HDF5-based scientific
    workloads where data is stored in spatially-coherent chunks).

    Config keys: trace.n_shards        (default: 8)
                 trace.shards_per_step (default: 2)
"""

import argparse
import csv
import os
import struct
import sys
import numpy as np
import yaml


# Binary trace record format (little-endian, 24 bytes per record):
#   uint32  step
#   uint16  file_id
#   uint16  _pad
#   uint64  offset          (bytes, 4096-aligned)
#   uint32  length          (bytes, 4096-aligned)
#   float   ideal_start_s   (ideal wall-clock issue time)
TRACE_RECORD_FMT  = "<IHHQIf"
TRACE_RECORD_SIZE = struct.calcsize(TRACE_RECORD_FMT)
TRACE_MAGIC       = b"AIOT"
TRACE_VERSION     = 1


def align_up(value: int, align: int) -> int:
    return (value + align - 1) & ~(align - 1)


# ── Access-pattern generators ─────────────────────────────────────────────────

def _sample_lengths(rng, batch_size: int, mean_sample_b: int,
                    sample_std_kb: float) -> np.ndarray:
    """Draw batch_size sample lengths, 4 KB-aligned, min 4 KB."""
    if sample_std_kb > 0:
        raw = rng.normal(mean_sample_b, sample_std_kb * 1024,
                         size=batch_size).astype(np.int64)
        raw = np.clip(raw, 4096, mean_sample_b * 4)
        return np.array([align_up(int(l), 4096) for l in raw], dtype=np.int64)
    return np.full(batch_size, mean_sample_b, dtype=np.int64)


def _build_records_uniform(rng, cfg: dict) -> list[dict]:
    """
    Pure i.i.d. uniform random offsets and file IDs.

    Every request is drawn independently from the full [0, file_size) range.
    This is the original behaviour and serves as the baseline with zero spatial
    locality, the worst case for SCISSOR's coalescing.
    """
    ds = cfg["dataset"]
    tr = cfg["trace"]

    n_files     = ds["n_files"]
    file_size_b = int(ds["file_size_gb"] * (1 << 30))
    mean_b      = align_up(int(ds["sample_size_kb"] * 1024), 4096)
    std_kb      = ds.get("sample_size_std_kb", 0)
    n_steps     = tr["n_steps"]
    batch_size  = tr["batch_size"]
    max_off     = file_size_b - mean_b * 2

    records, t = [], 0.0
    for step in range(n_steps):
        file_ids = rng.integers(0, n_files, size=batch_size)
        offsets  = (rng.integers(0, max(1, max_off // 4096), size=batch_size)
                    .astype(np.int64) * 4096)
        lengths  = _sample_lengths(rng, batch_size, mean_b, std_kb)

        for i in range(batch_size):
            off = int(offsets[i])
            l   = int(lengths[i])
            if off + l > file_size_b:
                off = align_up(max(0, file_size_b - l - mean_b), 4096)
            records.append({"step": step, "file_id": int(file_ids[i]),
                            "offset": off, "length": l, "ideal_start_s": t})
        t += _advance_clock(rng, cfg)
    return records


def _build_records_epoch(rng, cfg: dict) -> list[dict]:
    """
    Epoch-based shuffled access.

    A single random permutation of all (file, sample_index) pairs is generated
    once per epoch.  Steps advance through it sequentially.  n_workers parallel
    DataLoader workers each take every n_workers-th element of the permutation,
    producing a strided-sequential file access: intra-worker requests are spaced
    (n_workers × mean_sample_size) apart, which is modest locality compared to
    pure random but much better than uniform.

    This models PyTorch DataLoader with shuffle=True and persistent_workers=True,
    which is standard for ImageNet / BERT training.
    """
    ds = cfg["dataset"]
    tr = cfg["trace"]
    cmp = cfg["compute"]

    n_files          = ds["n_files"]
    file_size_b      = int(ds["file_size_gb"] * (1 << 30))
    samples_per_file = ds.get("samples_per_file",
                               int(file_size_b // (ds["sample_size_kb"] * 1024)))
    mean_b           = align_up(int(ds["sample_size_kb"] * 1024), 4096)
    std_kb           = ds.get("sample_size_std_kb", 0)
    n_steps          = tr["n_steps"]
    batch_size       = tr["batch_size"]
    n_workers        = tr.get("n_workers", cmp.get("n_workers", 4))

    # Build full epoch as list of (file_id, sample_index) pairs then shuffle
    epoch = [(f, s) for f in range(n_files)
             for s in range(samples_per_file)]
    idx   = rng.permutation(len(epoch))

    # Stride the permutation across n_workers: each worker takes every n_workers-th
    # element.  This groups consecutive (within a worker) samples into nearsequential
    # file positions, mimicking the real DataLoader prefetch pattern.
    strided = []
    for w in range(n_workers):
        strided.extend(idx[w::n_workers])
    # Slice into steps
    pos = 0
    stride_epoch = [epoch[i] for i in strided]

    spacing_b = align_up(mean_b, 4096)   # samples stored back-to-back

    records, t = [], 0.0
    for step in range(n_steps):
        for _ in range(batch_size):
            if pos >= len(stride_epoch):
                # wrap to new epoch shuffle
                idx2 = rng.permutation(len(epoch))
                strided2 = []
                for w in range(n_workers):
                    strided2.extend(idx2[w::n_workers])
                stride_epoch = [epoch[i] for i in strided2]
                pos = 0

            file_id, sample_idx = stride_epoch[pos]
            pos += 1

            # File offset = sample_index × mean_sample_size (sequential layout)
            base_off = sample_idx * spacing_b
            if base_off + mean_b > file_size_b:
                base_off = file_size_b - mean_b

            length = _sample_lengths(rng, 1, mean_b, std_kb)[0]
            off    = align_up(min(base_off, file_size_b - int(length)), 4096)

            records.append({"step": step, "file_id": file_id,
                            "offset": int(off), "length": int(length),
                            "ideal_start_s": t})
        t += _advance_clock(rng, cfg)
    return records


def _build_records_window(rng, cfg: dict) -> list[dict]:
    """
    Shuffle-buffer (sliding window) locality.

    Each step picks a random window of size window_size_mb within each file
    and draws all its samples exclusively from that window.  Every pair of
    samples within the same step and file is at most window_size_mb apart,
    creating strong intra-batch spatial locality.

    This models WebDataset / TFRecord pipelines with a shuffle_buffer that
    buffers a fixed-size chunk of the stream before shuffling.  It is also the
    pattern produced by HDF5 datasets read with a fixed chunk-cache, since the
    cache naturally retains the most recently accessed file region.

    window_size_mb controls the locality radius:
      small (<1 MB)  → very strong locality, many samples coalesce easily
      large (>file)  → degenerates to uniform random (no locality)
    """
    ds = cfg["dataset"]
    tr = cfg["trace"]

    n_files     = ds["n_files"]
    file_size_b = int(ds["file_size_gb"] * (1 << 30))
    mean_b      = align_up(int(ds["sample_size_kb"] * 1024), 4096)
    std_kb      = ds.get("sample_size_std_kb", 0)
    n_steps     = tr["n_steps"]
    batch_size  = tr["batch_size"]

    window_mb = tr.get("window_size_mb")
    if window_mb is None:
        raise ValueError("trace.window_size_mb is required for access_pattern: window")
    window_b = int(window_mb * (1 << 20))
    window_b = max(window_b, mean_b * 2)   # must fit at least 2 samples

    records, t = [], 0.0
    for step in range(n_steps):
        # One independent window per file per step
        max_win_start = max(0, file_size_b - window_b)
        win_starts = (rng.integers(0, max(1, max_win_start // 4096),
                                   size=n_files).astype(np.int64) * 4096)

        file_ids = rng.integers(0, n_files, size=batch_size)
        lengths  = _sample_lengths(rng, batch_size, mean_b, std_kb)

        for i in range(batch_size):
            fid  = int(file_ids[i])
            l    = int(lengths[i])
            wstart = int(win_starts[fid])
            wend   = wstart + window_b
            # Draw offset uniformly within the window, aligned
            max_off = max(wstart, wend - l)
            off = int(rng.integers(wstart // 4096,
                                   max(wstart // 4096 + 1, max_off // 4096 + 1))
                      * 4096)
            if off + l > file_size_b:
                off = file_size_b - l
            off = align_up(max(0, off), 4096)
            records.append({"step": step, "file_id": fid,
                            "offset": off, "length": l, "ideal_start_s": t})
        t += _advance_clock(rng, cfg)
    return records


def _build_records_shard(rng, cfg: dict) -> list[dict]:
    """
    Shard-based locality.

    Each file is partitioned into n_shards equal-sized regions.  Every step
    selects shards_per_step shards at random and draws all batch_size samples
    exclusively from those shards.  Samples within the same shard are at most
    (file_size / n_shards) apart.

    This models:
      - DLIO / MLPerf Storage benchmark patterns where workers pre-fetch one
        shard at a time (strong within-shard locality, zero cross-shard).
      - HDF5 scientific workloads (UNet3D / Cosmoflow) where a mini-batch
        consists of spatially adjacent 3D sub-volumes from the same domain
        partition, equivalent to drawing from a small number of file shards.
      - Multi-worker data pipelines where each worker owns a disjoint file
        partition and prefetches it linearly.

    n_shards and shards_per_step together control locality strength:
      n_shards=8, shards_per_step=1  → strongest locality (1/8 of file per step)
      n_shards=8, shards_per_step=8  → degenerates to uniform random
    """
    ds = cfg["dataset"]
    tr = cfg["trace"]

    n_files          = ds["n_files"]
    file_size_b      = int(ds["file_size_gb"] * (1 << 30))
    mean_b           = align_up(int(ds["sample_size_kb"] * 1024), 4096)
    std_kb           = ds.get("sample_size_std_kb", 0)
    n_steps          = tr["n_steps"]
    batch_size       = tr["batch_size"]
    n_shards         = tr.get("n_shards",         8)
    shards_per_step  = tr.get("shards_per_step",  2)
    shards_per_step  = min(shards_per_step, n_shards)

    shard_size_b = align_up(file_size_b // n_shards, 4096)

    records, t = [], 0.0
    for step in range(n_steps):
        # Pick shards_per_step shards without replacement
        chosen_shards = rng.choice(n_shards, size=shards_per_step, replace=False)
        file_ids      = rng.integers(0, n_files,   size=batch_size)
        shard_assigns = rng.integers(0, shards_per_step, size=batch_size)
        lengths       = _sample_lengths(rng, batch_size, mean_b, std_kb)

        for i in range(batch_size):
            fid   = int(file_ids[i])
            shard = int(chosen_shards[shard_assigns[i]])
            l     = int(lengths[i])

            shard_start = shard * shard_size_b
            shard_end   = min((shard + 1) * shard_size_b, file_size_b)
            max_off     = max(shard_start, shard_end - l)

            if max_off <= shard_start:
                off = shard_start
            else:
                off = int(rng.integers(shard_start // 4096,
                                       max_off // 4096 + 1) * 4096)

            if off + l > file_size_b:
                off = file_size_b - l
            off = align_up(max(0, off), 4096)

            records.append({"step": step, "file_id": fid,
                            "offset": off, "length": l, "ideal_start_s": t})
        t += _advance_clock(rng, cfg)
    return records


def _advance_clock(rng, cfg: dict) -> float:
    """Sample one compute-time interval from the configured distribution."""
    cmp   = cfg["compute"]
    mu    = cmp["preprocess_mean_ms"] / 1000.0
    sigma = cmp["preprocess_std_ms"]  / 1000.0
    dist  = cmp.get("distribution", "normal")

    if dist == "gamma":
        k     = (mu / sigma) ** 2
        theta = sigma ** 2 / mu
        return float(rng.gamma(k, theta))
    elif dist == "uniform":
        return float(rng.uniform(max(0.0, mu - 3 * sigma), mu + 3 * sigma))
    else:
        return float(max(0.0, rng.normal(mu, sigma)))


# ── IO and summary ────────────────────────────────────────────────────────────

def write_csv(records: list[dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["step", "file_id", "offset", "length",
                           "ideal_start_s"])
        writer.writeheader()
        writer.writerows(records)
    print(f"[gen_trace] CSV  → {path}  ({len(records)} records)")


def write_binary(records: list[dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(TRACE_MAGIC)
        f.write(struct.pack("<II", TRACE_VERSION, len(records)))
        for r in records:
            f.write(struct.pack(TRACE_RECORD_FMT,
                                r["step"], r["file_id"], 0,
                                r["offset"], r["length"],
                                r["ideal_start_s"]))
    print(f"[gen_trace] BIN  → {path}  ({len(records)} records, "
          f"{TRACE_RECORD_SIZE} B/record)")


def print_summary(records: list[dict], cfg: dict, pattern: str):
    steps       = sorted(set(r["step"] for r in records))
    total_bytes = sum(r["length"] for r in records)
    sizes       = [r["length"] for r in records]
    offsets     = [r["offset"] for r in records]

    # Per-step locality: mean nearest-neighbour distance within same file
    # (lower = stronger locality)
    nn_dists = []
    for step in steps[:min(10, len(steps))]:   # sample first 10 steps
        by_file: dict[int, list[int]] = {}
        for r in records:
            if r["step"] == step:
                by_file.setdefault(r["file_id"], []).append(r["offset"])
        for offs in by_file.values():
            if len(offs) < 2:
                continue
            s = sorted(offs)
            nn_dists.extend(b - a for a, b in zip(s, s[1:]))

    print(f"\n{'='*60}")
    print(f"  Trace Summary: {cfg['workload']['name']}")
    print(f"{'='*60}")
    print(f"  Access pattern: {pattern}")
    print(f"  Steps:          {len(steps)}")
    print(f"  Total records:  {len(records)}")
    print(f"  Records/step:   {len(records)//max(1,len(steps))}")
    print(f"  Total data:     {total_bytes/(1<<30):.2f} GB")
    print(f"  Sample size:    min={min(sizes)//1024} KB  "
          f"mean={int(np.mean(sizes))//1024} KB  "
          f"max={max(sizes)//1024} KB")
    print(f"  Offset range:   0 to {max(offsets)//(1<<30):.1f} GB")
    print(f"  Files used:     {cfg['dataset']['n_files']}")
    if nn_dists:
        print(f"  Intra-step NN gap (mean): "
              f"{int(np.mean(nn_dists))//1024} KB  "
              f"median: {int(np.median(nn_dists))//1024} KB")
        print(f"  (gap between nearest same-file neighbours within a step,")
        print(f"   lower = stronger locality = more SCISSOR coalescing)")
    print(f"{'='*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

PATTERN_BUILDERS = {
    "uniform": _build_records_uniform,
    "epoch":   _build_records_epoch,
    "window":  _build_records_window,
    "shard":   _build_records_shard,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate IO access trace from YAML config")
    parser.add_argument("--config", required=True,
                        help="Path to workload YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pattern = cfg["trace"].get("access_pattern", "uniform")
    if pattern not in PATTERN_BUILDERS:
        sys.exit(f"[gen_trace] Unknown access_pattern '{pattern}'. "
                 f"Valid: {list(PATTERN_BUILDERS)}")

    print(f"[gen_trace] Workload: {cfg['workload']['name']}")
    print(f"[gen_trace] Access pattern: {pattern}")
    print(f"[gen_trace] Generating trace ...")

    rng     = np.random.default_rng(cfg["trace"]["random_seed"])
    records = PATTERN_BUILDERS[pattern](rng, cfg)

    write_csv(records,    cfg["trace"]["output_csv"])
    write_binary(records, cfg["trace"]["output_bin"])
    print_summary(records, cfg, pattern)


if __name__ == "__main__":
    main()
