#!/usr/bin/env python3
"""
gen_trace_dlio.py (Option B): DLIO-based Trace Extractor
Converts DLIO indexed_binary dataset + YAML config into a binary IO trace
compatible with the ai-io-bench engine, without running the benchmark.

The output format is identical to gen_trace.py:
  - Binary: AIOT header + records of (step, file_id, pad, offset, length, ideal_start_s)
  - CSV:    one row per record

Usage:
    python gen_trace_dlio.py \\
        --dlio-config /path/to/dlio/workload.yaml \\
        --data-dir    /path/to/dlio/data \\
        --output-bin  traces/dlrm.bin \\
        --output-csv  traces/dlrm.csv

    # Override number of epochs or seed:
    python gen_trace_dlio.py --dlio-config dlrm.yaml --data-dir data \\
        --output-bin out.bin --epochs 2 --seed 42

How it works:
  1. Parse DLIO YAML config (without Hydra) to extract:
       dataset.format, num_files_train, num_samples_per_file, record_length_bytes,
       reader.batch_size, reader.sample_shuffle, reader.file_shuffle,
       train.computation_time, dataset.seed, dataset.seed_change_epoch
  2. Discover training files in {data_dir}/train/*.indexed_binary (sorted).
  3. For each training file, read its .off.idx and .sz.idx index files
     to get per-sample (offset, length) exactly as written by
     IndexedBinaryGenerator.
  4. Replicate DLIO's get_global_map_index() shuffle logic (single rank,
     comm_size=1) to reproduce the exact sample access order DLIO would use.
  5. Emit one trace record per sample per step, with ideal_start_s derived
     from train.computation_time.

Limitations:
  - Single-rank only (comm_size=1, my_rank=0). Multi-rank sharding is not
    modelled. The trace represents one rank's view of the full dataset.
  - file_shuffle for the INDEX sampler is not applied by DLIO itself (only
    the ITERATIVE sampler uses file_shuffle), and this script matches that.
  - Only indexed_binary format is supported.
"""

import argparse
import csv
import glob
import math
import os
import struct
import sys

import numpy as np
import yaml


# ── Binary trace format (must match gen_trace.py and the C++ engine) ─────────
TRACE_RECORD_FMT  = "<IHHQIf"
TRACE_RECORD_SIZE = struct.calcsize(TRACE_RECORD_FMT)
TRACE_MAGIC       = b"AIOT"
TRACE_VERSION     = 1


# ── DLIO config parsing ───────────────────────────────────────────────────────

def _nested_get(d: dict, *keys, default=None):
    """Walk nested dict with dotted key path."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def load_dlio_config(yaml_path: str) -> dict:
    """
    Parse a DLIO workload YAML and return a normalised flat config dict.

    Only the fields required for trace extraction are extracted. Everything
    else is ignored.  Defaults match DLIO's ConfigArguments defaults.
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    fmt = _nested_get(raw, "dataset", "format", default="indexed_binary")
    if fmt != "indexed_binary":
        sys.exit(f"[gen_trace_dlio] Only indexed_binary format is supported, but "
                 f"config specifies '{fmt}'.")

    # record_dims (list) takes priority over record_length_bytes (scalar).
    # DLIO's IndexedBinaryGenerator uses record_dims when provided, computing
    # sample_size = prod(record_dims) * record_element_bytes.  This avoids
    # the sqrt-truncation that happens when only record_length_bytes is given.
    import math
    record_dims   = _nested_get(raw, "dataset", "record_dims", default=None)
    record_elem_b = _nested_get(raw, "dataset", "record_element_bytes", default=1)
    if record_dims is not None:
        dims = record_dims if isinstance(record_dims, list) else [record_dims]
        record_length = int(math.prod(dims)) * int(record_elem_b)
    else:
        record_length = _nested_get(raw, "dataset", "record_length_bytes", default=None)

    return {
        "data_folder":         _nested_get(raw, "dataset", "data_folder", default="data"),
        "num_files_train":     _nested_get(raw, "dataset", "num_files_train",     default=1),
        "num_samples_per_file":_nested_get(raw, "dataset", "num_samples_per_file",default=1),
        "record_length_bytes": record_length,
        "batch_size":          _nested_get(raw, "reader",  "batch_size",          default=1),
        # sample_shuffle: "random" or "off" (DLIO uses Shuffle enum, treat any non-off as random)
        "sample_shuffle":      _nested_get(raw, "reader",  "sample_shuffle",      default="off"),
        "file_shuffle":        _nested_get(raw, "reader",  "file_shuffle",        default="off"),
        "computation_time_s":  _nested_get(raw, "train",   "computation_time",    default=0.0),
        "epochs":              _nested_get(raw, "train",   "epochs",              default=1),
        "seed":                _nested_get(raw, "dataset", "seed",                default=123),
        "seed_change_epoch":   _nested_get(raw, "dataset", "seed_change_epoch",   default=True),
    }


# ── File discovery ────────────────────────────────────────────────────────────

def discover_train_files(data_dir: str, num_files_train: int) -> list[str]:
    """
    Find training files under {data_dir}/train/*.indexed_binary (sorted).

    Falls back to {data_dir}/training/ then {data_dir}/ if the standard
    subdirectory is absent.
    """
    candidates = [
        os.path.join(data_dir, "train"),
        os.path.join(data_dir, "training"),
        data_dir,
    ]
    for folder in candidates:
        pattern = os.path.join(folder, "*.indexed_binary")
        files = sorted(glob.glob(pattern))
        if files:
            if len(files) < num_files_train:
                print(f"[gen_trace_dlio] WARNING: found {len(files)} files in "
                      f"{folder} but config says {num_files_train}. "
                      f"Using all found files.", file=sys.stderr)
            return files[:num_files_train]

    sys.exit(f"[gen_trace_dlio] No *.indexed_binary files found under {data_dir}. "
             f"Run DLIO with generate_data=True first.")


# ── Index file loading ────────────────────────────────────────────────────────

def load_index(filepath: str, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Read .off.idx and .sz.idx files for a data file.

    Returns (offsets, sizes) as uint64 numpy arrays of length num_samples,
    exactly as written by IndexedBinaryGenerator.
    """
    off_path = filepath + ".off.idx"
    sz_path  = filepath + ".sz.idx"

    for p in (off_path, sz_path):
        if not os.path.exists(p):
            sys.exit(f"[gen_trace_dlio] Index file not found: {p}\n"
                     f"  Run DLIO with generate_data=True to create index files.")

    offsets = np.fromfile(off_path, dtype=np.int64, count=num_samples)
    sizes   = np.fromfile(sz_path,  dtype=np.int64, count=num_samples)

    if len(offsets) < num_samples or len(sizes) < num_samples:
        sys.exit(f"[gen_trace_dlio] Index file for {filepath} has fewer entries "
                 f"than expected ({len(offsets)} < {num_samples}).")

    return offsets, sizes


# ── Shuffle replication (mirrors DLIO get_global_map_index, comm_size=1) ─────

def build_epoch_sample_order(cfg: dict, epoch: int) -> list[tuple[str, int]]:
    """
    Reproduce DLIO's get_global_map_index() for a single rank (comm_size=1).

    Returns list of (filepath, sample_index_within_file) in access order.
    The list length equals total_samples = num_files_train × num_samples_per_file.
    """
    file_list      = cfg["_file_list"]
    n_samples      = cfg["num_samples_per_file"]
    total_samples  = len(file_list) * n_samples
    shuffle_on     = str(cfg["sample_shuffle"]).lower() not in ("off", "false", "none", "0")

    sample_list = np.arange(total_samples, dtype=np.int64)

    if shuffle_on:
        seed = cfg["seed"] + epoch if cfg["seed_change_epoch"] else cfg["seed"]
        np.random.seed(seed)
        np.random.shuffle(sample_list)

    result = []
    for global_idx in sample_list:
        file_idx    = int(global_idx) // n_samples
        sample_idx  = int(global_idx) % n_samples
        result.append((file_list[file_idx], sample_idx))

    return result


# ── Trace generation ──────────────────────────────────────────────────────────

def generate_records(cfg: dict, epochs: int) -> list[dict]:
    """
    Walk the DLIO sample order (all epochs) and emit one trace record per sample.

    step numbering is global across epochs (step 0 = first step of epoch 0,
    step K = last step of last epoch).
    """
    batch_size    = cfg["batch_size"]
    compute_s     = cfg["computation_time_s"]
    index_cache   = cfg["_index_cache"]   # filepath -> (offsets, sizes)
    file_id_map   = cfg["_file_id_map"]   # filepath -> int file_id

    records  = []
    global_step = 0
    t = 0.0

    for epoch in range(epochs):
        access_order = build_epoch_sample_order(cfg, epoch)
        total_samples = len(access_order)
        n_steps = math.ceil(total_samples / batch_size)

        for step_in_epoch in range(n_steps):
            batch_start = step_in_epoch * batch_size
            batch_end   = min(batch_start + batch_size, total_samples)
            batch       = access_order[batch_start:batch_end]

            for filepath, sample_idx in batch:
                offsets, sizes = index_cache[filepath]
                offset = int(offsets[sample_idx])
                length = int(sizes[sample_idx])
                fid    = file_id_map[filepath]
                records.append({
                    "step":         global_step,
                    "file_id":      fid,
                    "offset":       offset,
                    "length":       length,
                    "ideal_start_s": t,
                })

            t += compute_s
            global_step += 1

    return records


# ── Output ────────────────────────────────────────────────────────────────────

def write_csv(records: list[dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["step", "file_id", "offset", "length", "ideal_start_s"])
        writer.writeheader()
        writer.writerows(records)
    print(f"[gen_trace_dlio] CSV  → {path}  ({len(records)} records)")


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
    print(f"[gen_trace_dlio] BIN  → {path}  ({len(records)} records, "
          f"{TRACE_RECORD_SIZE} B/record)")


def print_summary(records: list[dict], cfg: dict):
    steps   = sorted(set(r["step"] for r in records))
    sizes   = [r["length"] for r in records]
    offsets = [r["offset"] for r in records]
    total_b = sum(sizes)

    # Intra-step nearest-neighbour gap (locality metric)
    nn_dists = []
    for step in steps[:min(10, len(steps))]:
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
    print(f"  Trace Summary: DLIO indexed_binary")
    print(f"{'='*60}")
    print(f"  Files:          {len(cfg['_file_list'])}")
    print(f"  Samples/file:   {cfg['num_samples_per_file']}")
    print(f"  Batch size:     {cfg['batch_size']}")
    print(f"  Epochs:         {cfg['_epochs']}")
    print(f"  Steps:          {len(steps)}")
    print(f"  Total records:  {len(records)}")
    print(f"  Total data:     {total_b/(1<<30):.3f} GB")
    print(f"  Sample size:    min={min(sizes)//1024} KB  "
          f"mean={int(np.mean(sizes))//1024} KB  "
          f"max={max(sizes)//1024} KB")
    print(f"  Offset range:   0 to {max(offsets)/(1<<30):.3f} GB")
    shuffle_on = str(cfg["sample_shuffle"]).lower() not in ("off", "false", "none", "0")
    print(f"  Sample shuffle: {'ON (random)' if shuffle_on else 'OFF'}")
    if nn_dists:
        print(f"  Intra-step NN gap (mean):   {int(np.mean(nn_dists))//1024} KB")
        print(f"  Intra-step NN gap (median): {int(np.median(nn_dists))//1024} KB")
        print(f"  (lower = stronger locality = more SCISSOR coalescing)")
    print(f"{'='*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert DLIO indexed_binary dataset to ai-io-bench trace")
    parser.add_argument("--dlio-config", required=True,
                        help="Path to DLIO workload YAML (e.g. dlrm.yaml)")
    parser.add_argument("--data-dir", required=True,
                        help="Root directory of generated DLIO data "
                             "(overrides dataset.data_folder in YAML)")
    parser.add_argument("--output-bin", required=True,
                        help="Output binary trace file path")
    parser.add_argument("--output-csv", default=None,
                        help="Output CSV trace file path (optional)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (overrides train.epochs)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides dataset.seed)")
    args = parser.parse_args()

    print(f"[gen_trace_dlio] Config:   {args.dlio_config}")
    print(f"[gen_trace_dlio] Data dir: {args.data_dir}")

    cfg = load_dlio_config(args.dlio_config)

    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.seed is not None:
        cfg["seed"] = args.seed

    epochs = cfg["epochs"]
    n_samples = cfg["num_samples_per_file"]

    # Discover files
    file_list = discover_train_files(args.data_dir, cfg["num_files_train"])
    print(f"[gen_trace_dlio] Found {len(file_list)} training files")

    # Load index files
    print(f"[gen_trace_dlio] Loading index files ...")
    index_cache = {}
    for fp in file_list:
        offsets, sizes = load_index(fp, n_samples)
        index_cache[fp] = (offsets, sizes)

    file_id_map = {fp: i for i, fp in enumerate(file_list)}

    # Attach derived state to cfg for convenience
    cfg["_file_list"]   = file_list
    cfg["_index_cache"] = index_cache
    cfg["_file_id_map"] = file_id_map
    cfg["_epochs"]      = epochs

    shuffle_on = str(cfg["sample_shuffle"]).lower() not in ("off", "false", "none", "0")
    print(f"[gen_trace_dlio] Epochs: {epochs}  Batch: {cfg['batch_size']}  "
          f"Shuffle: {'ON' if shuffle_on else 'OFF'}  Seed: {cfg['seed']}")
    print(f"[gen_trace_dlio] Generating trace ...")

    records = generate_records(cfg, epochs)

    write_binary(records, args.output_bin)
    if args.output_csv:
        write_csv(records, args.output_csv)

    print_summary(records, cfg)


if __name__ == "__main__":
    main()
