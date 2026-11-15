#!/usr/bin/env python3
"""
sample_trace.py: Post-DLIO trace selection filter.

Reads a full AIOT binary trace (produced by gen_trace_dlio.py) and applies
a realistic data-selection algorithm before IO replay:

  clustered        Divides each file's offset range into n_clusters equal-width
                   segments.  A target_cluster_fraction of segments per file are
                   "target" (high density = intra_keep_rate), and the rest are
                   "non-target" (low density = inter_keep_rate).
                   Models: ImageNet per-class coreset (SSP / Moderate Coreset),
                   NLP domain-weighted selection (DSIR / perplexity pruning).

  temporal_window  Retains only records from the high-offset portion of each
                   file (recent temporal window), then subsamples within the
                   window at within_keep_rate.
                   (Kept for completeness, not used by the primary benchmark
                   workloads. See config/dlio/ for active workload configs.)

Selected records are re-sorted by (file_id, offset) within each new step,
matching the ordering a DataLoader uses when draining a per-worker read queue.
This preserves the spatial locality that SCISSOR exploits.

Standalone usage:
    # BERT-Large clustered selection
    python3 sample_trace.py \\
        --input  traces/dlio/bert_clustered_trace.bin \\
        --output traces/dlio/bert_clustered_sampled.bin \\
        --mode   clustered --n-clusters 16 \\
        --intra-keep 0.85 --inter-keep 0.15 --target-fraction 0.50 \\
        --batch-size 256 --compute-time 0.005

    # ResNet-50 per-class coreset selection
    python3 sample_trace.py \\
        --input  traces/dlio/resnet50_clustered_trace.bin \\
        --output traces/dlio/resnet50_clustered_sampled.bin \\
        --mode   clustered --n-clusters 64 \\
        --intra-keep 0.60 --inter-keep 0.60 --target-fraction 1.0 \\
        --batch-size 128 --compute-time 0.090

    The --compute-time value must match train.computation_time in the
    corresponding config/dlio/*.yaml (CPU data transformation time, seconds).

API usage from bench_dlio.py:
    from sample_trace import apply_selection
    apply_selection(input_bin, output_bin, selection_cfg,
                    batch_size, compute_time_s, output_csv, seed)
"""

import argparse
import csv
import os
import struct
import sys

import numpy as np

# ── Binary trace format (must match gen_trace_dlio.py and the C++ engine) ─────
TRACE_RECORD_FMT  = "<IHHQIf"
TRACE_RECORD_SIZE = struct.calcsize(TRACE_RECORD_FMT)
TRACE_MAGIC       = b"AIOT"
TRACE_VERSION     = 1


# ── IO helpers ────────────────────────────────────────────────────────────────

def read_trace(path: str) -> list[dict]:
    """Read an AIOT binary trace and return list of record dicts."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != TRACE_MAGIC:
            sys.exit(f"[sample_trace] Not a valid AIOT trace: {path}")
        _version, n_records = struct.unpack("<II", f.read(8))
        records = []
        for _ in range(n_records):
            raw = f.read(TRACE_RECORD_SIZE)
            if len(raw) < TRACE_RECORD_SIZE:
                break
            step, file_id, _pad, offset, length, ideal_start_s = \
                struct.unpack(TRACE_RECORD_FMT, raw)
            records.append({
                "step":          step,
                "file_id":       file_id,
                "offset":        offset,
                "length":        length,
                "ideal_start_s": ideal_start_s,
            })
    return records


def write_trace(records: list[dict], bin_path: str,
                csv_path: str | None = None) -> None:
    """Write records to AIOT binary (and optional CSV) trace."""
    os.makedirs(os.path.dirname(bin_path) or ".", exist_ok=True)
    with open(bin_path, "wb") as f:
        f.write(TRACE_MAGIC)
        f.write(struct.pack("<II", TRACE_VERSION, len(records)))
        for r in records:
            f.write(struct.pack(TRACE_RECORD_FMT,
                                r["step"], r["file_id"], 0,
                                r["offset"], r["length"],
                                r["ideal_start_s"]))
    print(f"[sample_trace] BIN → {bin_path}  ({len(records):,} records)")

    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["step", "file_id", "offset", "length",
                               "ideal_start_s"])
            writer.writeheader()
            writer.writerows(records)
        print(f"[sample_trace] CSV → {csv_path}")


# ── Selection modes ───────────────────────────────────────────────────────────

def _select_clustered(records: list[dict], cfg: dict,
                      rng: np.random.Generator) -> list[dict]:
    """
    Divide each file's offset space into n_clusters equal-width segments.
    A fraction target_cluster_fraction of segments are "target" (high density
    = intra_keep_rate), and the rest are "non-target" (low density = inter_keep_rate).

    Cluster assignment is deterministic per file (seeded by file_id) so that
    the same file always maps to the same target/non-target cluster layout
    across multiple runs.
    """
    n_clusters   = int(cfg["n_clusters"])
    intra_keep   = float(cfg["intra_keep_rate"])
    inter_keep   = float(cfg["inter_keep_rate"])
    target_frac  = float(cfg.get("target_cluster_fraction", 0.50))

    by_file: dict[int, list[dict]] = {}
    for r in records:
        by_file.setdefault(r["file_id"], []).append(r)

    selected: list[dict] = []
    for file_id, file_records in sorted(by_file.items()):
        offsets     = [r["offset"] for r in file_records]
        min_offset  = min(offsets)
        max_offset  = max(offsets)
        span        = max(max_offset - min_offset, 1)
        cluster_span = span / n_clusters

        # Deterministic target cluster set for this file
        n_target = max(1, round(n_clusters * target_frac))
        file_rng = np.random.default_rng([rng.integers(2**31), file_id])
        target_clusters = set(
            file_rng.choice(n_clusters, n_target, replace=False).tolist()
        )

        for r in file_records:
            cluster_idx = min(
                int((r["offset"] - min_offset) / cluster_span),
                n_clusters - 1,
            )
            keep_rate = intra_keep if cluster_idx in target_clusters else inter_keep
            if rng.random() < keep_rate:
                selected.append(r)

    return selected


def _select_temporal_window(records: list[dict], cfg: dict,
                            rng: np.random.Generator) -> list[dict]:
    """
    Keep only records whose offset falls in the high-offset portion of each
    file (the "recent temporal window"), then subsample within the window.

    window_fraction = 0.58 means the top 58 % of each file's offset range is
    the window (analogous to the last 14 of 24 Criteo training days).
    within_keep_rate = 0.30 applies a further 30 % random subsampling inside
    the window (analogous to within-day popularity / hard-negative filtering).
    """
    window_fraction = float(cfg["window_fraction"])
    within_keep     = float(cfg["within_keep_rate"])

    by_file: dict[int, list[dict]] = {}
    for r in records:
        by_file.setdefault(r["file_id"], []).append(r)

    selected: list[dict] = []
    for file_id, file_records in sorted(by_file.items()):
        offsets    = [r["offset"] for r in file_records]
        min_offset = min(offsets)
        max_offset = max(offsets)
        span       = max_offset - min_offset
        threshold  = min_offset + span * (1.0 - window_fraction)

        for r in file_records:
            if r["offset"] >= threshold and rng.random() < within_keep:
                selected.append(r)

    return selected


# ── Re-batching ───────────────────────────────────────────────────────────────

def _rebatch(selected: list[dict], batch_size: int,
             compute_time_s: float) -> list[dict]:
    """
    Re-group selected records into fixed-size steps of batch_size, sorted
    within each step by (file_id, offset).

    Sorting by (file_id, offset) within a step groups spatially adjacent
    records together, giving SCISSOR the opportunity to coalesce them, which
    is exactly what a DataLoader does when draining its per-worker read queue.

    ideal_start_s increases by compute_time_s after each step.
    """
    sorted_records = sorted(selected, key=lambda r: (r["file_id"], r["offset"]))

    batched: list[dict] = []
    t = 0.0
    for step_idx in range(0, len(sorted_records), batch_size):
        batch = sorted_records[step_idx: step_idx + batch_size]
        for r in batch:
            batched.append({
                "step":          step_idx // batch_size,
                "file_id":       r["file_id"],
                "offset":        r["offset"],
                "length":        r["length"],
                "ideal_start_s": t,
            })
        t += compute_time_s

    return batched


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(n_orig: int, selected: list[dict],
                   batched: list[dict], mode: str) -> None:
    n_sel   = len(selected)
    n_steps = 1 + max((r["step"] for r in batched), default=-1)
    total_b = sum(r["length"] for r in batched)

    # Intra-step nearest-neighbour gap (first 20 steps, per file)
    gaps: list[int] = []
    seen_steps = sorted(set(r["step"] for r in batched))[:20]
    for step in seen_steps:
        by_file: dict[int, list[int]] = {}
        for r in batched:
            if r["step"] == step:
                by_file.setdefault(r["file_id"], []).append(r["offset"])
        for off_list in by_file.values():
            s = sorted(off_list)
            gaps.extend(b - a for a, b in zip(s, s[1:]))

    print(f"\n{'='*60}")
    print(f"  Sample Selection Summary")
    print(f"{'='*60}")
    print(f"  Mode:                {mode}")
    print(f"  Original records:    {n_orig:>12,}")
    print(f"  Selected records:    {n_sel:>12,}  "
          f"({100 * n_sel / max(n_orig, 1):.1f}% kept)")
    print(f"  Steps after rebatch: {n_steps:>12,}")
    print(f"  Total data to read:  {total_b / (1 << 30):>11.3f} GB")
    if gaps:
        mean_gap   = int(np.mean(gaps))
        median_gap = int(np.median(gaps))
        print(f"  Intra-step NN gap (mean):   "
              f"{mean_gap:>8,} B  ({mean_gap // 1024} KB)")
        print(f"  Intra-step NN gap (median): "
              f"{median_gap:>8,} B  ({median_gap // 1024} KB)")
        print(f"  → SCISSOR bridges gaps up to its regressor threshold,")
        print(f"    pread/io_uring issue one syscall per record regardless.")
    print(f"{'='*60}\n")


# ── Public API ────────────────────────────────────────────────────────────────

def apply_selection(input_bin: str, output_bin: str,
                    selection_cfg: dict,
                    batch_size: int, compute_time_s: float,
                    output_csv: str | None = None,
                    seed: int = 42) -> list[dict]:
    """
    Read full trace, apply selection filter, rebatch, write sampled trace.

    Parameters
    ----------
    input_bin      : path to full AIOT binary trace from gen_trace_dlio.py
    output_bin     : path for the sampled AIOT binary trace
    selection_cfg  : dict, must contain "mode" plus mode-specific keys
    batch_size     : records per step in the output trace
    compute_time_s : seconds of simulated compute between steps
    output_csv     : optional CSV output path
    seed           : global RNG seed (cluster assignments also seeded by file_id)

    Returns the list of batched record dicts written to output_bin.
    """
    rng = np.random.default_rng(seed)

    records = read_trace(input_bin)
    print(f"[sample_trace] Read {len(records):,} records from {input_bin}")

    mode = selection_cfg.get("mode", "none")
    if mode == "clustered":
        selected = _select_clustered(records, selection_cfg, rng)
    elif mode == "temporal_window":
        selected = _select_temporal_window(records, selection_cfg, rng)
    elif mode == "none":
        selected = list(records)
    else:
        sys.exit(f"[sample_trace] Unknown selection mode: {mode!r}. "
                 f"Choose: clustered | temporal_window | none")

    if not selected:
        sys.exit("[sample_trace] No records survived selection: "
                 "check keep rates and window fraction.")

    batched = _rebatch(selected, batch_size, compute_time_s)
    _print_summary(len(records), selected, batched, mode)
    write_trace(batched, output_bin, output_csv)
    return batched


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Post-DLIO trace selection filter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",        required=True,
                   help="Input AIOT binary trace (full, from gen_trace_dlio.py)")
    p.add_argument("--output",       required=True,
                   help="Output AIOT binary trace (sampled)")
    p.add_argument("--output-csv",   default=None,
                   help="Optional CSV output path")
    p.add_argument("--mode",         required=True,
                   choices=["clustered", "temporal_window"],
                   help="Selection mode")
    p.add_argument("--batch-size",   type=int,   default=256,
                   help="Records per step in the output trace")
    p.add_argument("--compute-time", type=float, default=0.1,
                   help="Simulated compute time between steps (seconds)")
    p.add_argument("--seed",         type=int,   default=42)
    # clustered
    p.add_argument("--n-clusters",       type=int,   default=16,
                   help="[clustered] Number of offset segments per file")
    p.add_argument("--intra-keep",       type=float, default=0.85,
                   help="[clustered] Keep rate inside target clusters")
    p.add_argument("--inter-keep",       type=float, default=0.15,
                   help="[clustered] Keep rate inside non-target clusters")
    p.add_argument("--target-fraction",  type=float, default=0.50,
                   help="[clustered] Fraction of clusters designated target")
    # temporal_window
    p.add_argument("--window-fraction",  type=float, default=0.58,
                   help="[temporal_window] Fraction of each file to keep (from end)")
    p.add_argument("--within-keep",      type=float, default=0.30,
                   help="[temporal_window] Sub-sample rate inside the window")
    args = p.parse_args()

    if args.mode == "clustered":
        sel_cfg: dict = {
            "mode":                    "clustered",
            "n_clusters":              args.n_clusters,
            "intra_keep_rate":         args.intra_keep,
            "inter_keep_rate":         args.inter_keep,
            "target_cluster_fraction": args.target_fraction,
        }
    else:
        sel_cfg = {
            "mode":             "temporal_window",
            "window_fraction":  args.window_fraction,
            "within_keep_rate": args.within_keep,
        }

    apply_selection(
        input_bin      = args.input,
        output_bin     = args.output,
        selection_cfg  = sel_cfg,
        batch_size     = args.batch_size,
        compute_time_s = args.compute_time,
        output_csv     = args.output_csv,
        seed           = args.seed,
    )


if __name__ == "__main__":
    main()
