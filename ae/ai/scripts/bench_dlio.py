#!/usr/bin/env python3
"""
bench_dlio.py: IO engine benchmark for the SCISSOR evaluation.

Runs Phases 3 to 4 of the pipeline.  Phases 1 to 2.5 (data generation and trace
preparation) are handled by prepare_dlio.py and must be completed first.

Pipeline:
  Phase 3  Copy data from PFS to SSD (rsync), drop page cache (vmtouch or
           /proc), run bench binary for each engine × N runs.
  Phase 4  Aggregate JSON results and produce two publication figures.

Cache eviction:
  Without sudo, pass --vmtouch-dir pointing to the directory that contains
  the vmtouch binary.  The benchmark evicts only the SSD data directory
  for the workload under test before each run:
      vmtouch -eq <ssd-root>/<workload>/train/
      sync
  With root or passwordless sudo, /proc/sys/vm/drop_caches is used instead.

Usage:
    # --pfs-root, --ssd-root and --vmtouch-dir default to $PFS_ROOT, $SSD_ROOT
    # and $VMTOUCH_DIR.

    # Copy from PFS to SSD and benchmark (typical first run on a compute node):
    python3 scripts/bench_dlio.py \\
        --pfs-root   $PFS_ROOT \\
        --ssd-root   $SSD_ROOT \\
        --traces-dir $PFS_ROOT/traces \\
        --vmtouch-dir /path/to/vmtouch/bin \\
        --runs 5 --gap-sweep

    # Data already on SSD (same node, subsequent runs, skip rsync):
    python3 scripts/bench_dlio.py \\
        --ssd-root   $SSD_ROOT \\
        --traces-dir $PFS_ROOT/traces \\
        --vmtouch-dir /path/to/vmtouch/bin \\
        --no-copy --runs 5 --gap-sweep

    # Plot only from existing results:
    python3 scripts/bench_dlio.py --no-run
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Local shared module
sys.path.insert(0, str(Path(__file__).parent))
from _dlio_common import (
    CFG_DIR, TRACES_DIR, SRC,
    DEFAULT_WORKLOADS, WORKLOAD_LABELS,
    ENGINES, GAP_VARIANTS, GAP_SWEEP_VALUES, PATTERN_NOTES,
    GAP_SWEEP_FIXED_BATCH_VALUES, FIXED_BATCH_MB,
    BATCH_SWEEP_MB_VALUES, FIXED_GAP_BY_WORKLOAD,
    write_manifest, run_cmd, load_result, throughput_gbps,
    ROOT,
)


# ── Cache eviction ────────────────────────────────────────────────────────────

def drop_caches(data_dir: str, vmtouch_dir: str | None) -> None:
    """
    Evict page cache for data_dir.

    If vmtouch_dir is set, uses:
        vmtouch -eq <data_dir>
        sync
    Otherwise falls back to /proc/sys/vm/drop_caches (requires root/sudo).
    """
    if vmtouch_dir:
        vmtouch_bin = os.path.join(vmtouch_dir, "vmtouch")
        os.system(f"{vmtouch_bin} -eq {data_dir}")
        os.system("sync")
    elif os.geteuid() == 0:
        os.system("sync; echo 3 > /proc/sys/vm/drop_caches")
    else:
        rc = os.system(
            "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null")
        if rc != 0:
            print("    [cache] SKIPPED: provide --vmtouch-dir, root, or passwordless sudo.",
                  file=sys.stderr)


# ── Data copy: PFS → SSD ──────────────────────────────────────────────────────

def copy_to_ssd(workload: str, pfs_root: str, ssd_root: str,
                num_files: int) -> str:
    """
    Rsync workload data from PFS to SSD.  Returns the SSD workload directory.
    Writes a manifest pointing to SSD paths after the copy.
    """
    pfs_train = os.path.join(pfs_root, workload, "train")
    ssd_outdir = os.path.join(ssd_root, workload)
    ssd_train  = os.path.join(ssd_outdir, "train")
    os.makedirs(ssd_train, exist_ok=True)

    print(f"\n[copy] {workload}")
    print(f"       {pfs_train}")
    print(f"    →  {ssd_train}")
    cmd = ["rsync", "--archive", "--info=progress2",
           f"{pfs_train}/", f"{ssd_train}/"]
    run_cmd(cmd, label=f"rsync {workload}")

    write_manifest(ssd_outdir, num_files)
    return ssd_outdir


# ── Phase 3: Benchmark runs ───────────────────────────────────────────────────

def phase_bench(workload: str, ssd_data_dir: str, trace_bin: str,
                results_dir: str, bench_bin: str, n_runs: int,
                cfg: dict, gap_sweep: bool,
                gap_batch_sweep: bool, batch_sweep: bool,
                vmtouch_dir: str | None) -> dict[str, list[float]]:
    """
    Run the bench binary for each engine × n_runs.
    Returns a dict of engine_key → list of throughput values (GB/s).
    """
    manifest = os.path.join(ssd_data_dir, "manifest.txt")
    if not os.path.exists(manifest):
        print(f"  [WARN] manifest not found: {manifest}", file=sys.stderr)
        return {}

    ssd_train_dir = os.path.join(ssd_data_dir, "train")
    bm      = cfg.get("benchmark", {})
    qd      = bm.get("queue_depth", 128)
    o_dir   = bm.get("o_direct", True)
    tr_cfg  = cfg.get("train", {})
    mean_ms = float(tr_cfg.get("computation_time", 0.1)) * 1000
    std_ms  = mean_ms * 0.1

    os.makedirs(results_dir, exist_ok=True)

    engine_keys = [e[0] for e in ENGINES]
    if gap_sweep:
        engine_keys += [v[0] for v in GAP_VARIANTS]
    throughputs: dict[str, list[float]] = {k: [] for k in engine_keys}

    # Derive chunk_size from the workload's record size so that:
    #   - SCISSOR sees exactly 1 chunk per record (not N×4KB chunks)
    #   - force_gap units are in record-widths, matching inter-record gap structure
    #   - io_uring submits 1 SQE per record, keeping QD records in flight at once
    # record_dims is the logical shape of one sample in bytes (e.g. [163840] for
    # ResNet-50 160 KB records, [8192] for BERT 8 KB records).  For multi-dim
    # tensors, chunk_size = product of all dims.
    ds = cfg.get("dataset", {})
    _dims = ds.get("record_dims", ds.get("record_length_bytes", 65536))
    _dims = _dims if isinstance(_dims, list) else [_dims]
    chunk_size = 1
    for d in _dims:
        chunk_size *= int(d)
    # O_DIRECT requires file offsets AND transfer sizes to be multiples of 4096.
    # For DLIO indexed_binary, offset of record i = i × record_size, so offsets
    # are aligned iff record_size is a multiple of 4096.  We abort here if that
    # is not the case rather than silently producing wrong results or EINVAL.
    if chunk_size % 4096 != 0:
        raise ValueError(
            f"[bench_dlio] record_size={chunk_size} B is not a multiple of 4096. "
            "O_DIRECT requires 4 KB-aligned offsets and transfer sizes. "
            "Check dataset.record_dims in the workload YAML."
        )

    def _run_engine(eng_name: str, force_gap: int | None = None,
                    force_batch_mb: int | None = None) -> None:
        for run in range(1, n_runs + 1):
            # eng_name is always the full result key (e.g. "scissor_sr_g4")
            suffix   = f"_{eng_name}"
            out_json = os.path.join(results_dir, f"{workload}{suffix}_r{run}.json")
            print(f"\n  [{eng_name}] run {run}/{n_runs}")

            drop_caches(ssd_train_dir, vmtouch_dir)

            # mmap does not support O_DIRECT
            use_o_direct  = o_dir and eng_name != "mmap"
            o_direct_flag = ["--o-direct"] if use_o_direct else []

            # When force_gap is set, eng_name encodes the base engine plus gap
            # (e.g. "scissor_g4" or "scissor_sr_g4").  Extract the base engine
            # name to pass to --engine. bench.cpp then renames it to eng_name.
            if force_gap is not None:
                base_engine = "scissor_sr" if eng_name.startswith("scissor_sr") \
                              else "scissor"
            else:
                base_engine = eng_name

            cmd = [bench_bin,
                   "--trace",           trace_bin,
                   "--manifest",        manifest,
                   "--engine",          base_engine,
                   "--queue-depth",     str(qd),
                   "--chunk-size",      str(chunk_size),
                   "--compute-mean-ms", str(mean_ms),
                   "--compute-std-ms",  str(std_ms),
                   "--compute-dist",    "normal",
                   "--report-json",     out_json,
                   "--verbose",
                   *o_direct_flag]
            if force_gap is not None:
                cmd += ["--force-gap", str(force_gap)]
            if force_batch_mb is not None:
                cmd += ["--force-batch-mb", str(force_batch_mb)]

            ok = run_cmd(cmd, label=f"{workload}/{eng_name}/run{run}")
            if ok:
                result = load_result(out_json)
                if result:
                    t = throughput_gbps(result)
                    if t is not None:
                        throughputs[eng_name].append(t)

    # Base engines are skipped whenever any sweep mode is active: sweeps are
    # standalone controlled experiments that vary only one parameter at a time.
    run_baselines = not (gap_sweep or gap_batch_sweep or batch_sweep)

    if run_baselines:
        for eng_name, *_ in ENGINES:
            _run_engine(eng_name)

    if gap_sweep:
        scissor_bases = [e[0] for e in ENGINES if e[0].startswith("scissor")]
        for base in scissor_bases:
            for gap_val in GAP_SWEEP_VALUES:
                _run_engine(f"{base}_g{gap_val}", force_gap=gap_val)
            throughputs[f"{base}_gauto"] = throughputs.get(base, [])

    # Gap sweep with fixed batch (FIXED_BATCH_MB, GAP_SWEEP_FIXED_BATCH_VALUES)
    if gap_batch_sweep:
        for gap_val in GAP_SWEEP_FIXED_BATCH_VALUES:
            key = f"scissor_sr_b{FIXED_BATCH_MB}mb_g{gap_val}"
            throughputs.setdefault(key, [])
            _run_engine(key, force_gap=gap_val, force_batch_mb=FIXED_BATCH_MB)

    # Batch sweep with fixed gap (FIXED_GAP_BY_WORKLOAD, BATCH_SWEEP_MB_VALUES)
    if batch_sweep:
        fixed_gap = FIXED_GAP_BY_WORKLOAD.get(workload, 1)
        for batch_mb in BATCH_SWEEP_MB_VALUES:
            key = f"scissor_sr_g{fixed_gap}_b{batch_mb}mb"
            throughputs.setdefault(key, [])
            _run_engine(key, force_gap=fixed_gap, force_batch_mb=batch_mb)

    return throughputs


# ── Phase 4: Load results from disk ──────────────────────────────────────────

def load_results_from_disk(workloads: list[str],
                           results_dir: str) -> dict[str, dict[str, list[float]]]:
    all_keys = [e[0] for e in ENGINES] + [v[0] for v in GAP_VARIANTS]
    data: dict[str, dict[str, list[float]]] = {wl: {} for wl in workloads}
    for wl in workloads:
        for key in all_keys:
            pattern = os.path.join(results_dir, f"{wl}_{key}_r*.json")
            vals    = []
            for path in sorted(glob.glob(pattern)):
                r = load_result(path)
                if r:
                    t = throughput_gbps(r)
                    if t is not None:
                        vals.append(t)
            data[wl][key] = vals
        if not data[wl].get("scissor_gauto"):
            data[wl]["scissor_gauto"] = data[wl].get("scissor", [])
    return data


# ── Plot 1: Throughput comparison ─────────────────────────────────────────────

def plot_throughput(all_results: dict[str, dict[str, list[float]]],
                    workloads: list[str], outdir: str) -> None:
    n_wl      = len(workloads)
    n_engines = len(ENGINES)
    bar_w     = 0.22
    group_gap = 0.12
    group_w   = n_engines * bar_w + group_gap
    x_centers = np.arange(n_wl) * group_w

    fig, ax = plt.subplots(figsize=(max(7, n_wl * 3.2), 4.5))
    max_y   = 0.0

    for ei, (eng_name, eng_label, fc, hatch, ec) in enumerate(ENGINES):
        offset = (ei - n_engines / 2 + 0.5) * bar_w
        x_pos  = x_centers + offset
        means, errs = [], []
        for wl in workloads:
            vals = all_results.get(wl, {}).get(eng_name, [])
            if vals:
                m = float(np.mean(vals))
                e = float(np.std(vals)) if len(vals) > 1 else 0.0
            else:
                m, e = 0.0, 0.0
            means.append(m)
            errs.append(e)
            max_y = max(max_y, m + e)

        bars = ax.bar(x_pos, means, width=bar_w,
                      color=fc, hatch=hatch, edgecolor=ec, linewidth=0.8,
                      yerr=errs, capsize=3, error_kw={"elinewidth": 1.2, "capsize": 3},
                      label=eng_label, zorder=3)
        y_top = max(max_y, 0.5)
        for bar, m in zip(bars, means):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + y_top * 0.015,
                        f"{m:.2f}", ha="center", va="bottom",
                        fontsize=7, rotation=60)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([WORKLOAD_LABELS.get(wl, wl) for wl in workloads],
                       fontsize=9)
    ax.set_ylabel("Throughput (GB/s)", fontsize=11)
    y_top = max(max_y * 1.4, 0.5)
    ax.set_ylim(0, y_top)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    for xi, wl in enumerate(workloads):
        note = PATTERN_NOTES.get(wl, "")
        ax.text(x_centers[xi], -y_top * 0.18, note,
                ha="center", va="top", fontsize=7, color="#444444",
                style="italic")

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_title(
        "IO Engine Throughput: DLIO Workloads with Realistic Data Selection\n"
        "(clustered selection → spatially correlated gaps → SCISSOR coalescing advantage)",
        fontsize=10, pad=10,
    )
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = os.path.join(outdir, f"bench_dlio_throughput.{ext}")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"[plot] {out_path}")
    plt.close(fig)


# ── Plot 2: SCISSOR gap analysis ──────────────────────────────────────────────

def plot_gap_analysis(all_results: dict[str, dict[str, list[float]]],
                      workloads: list[str], outdir: str) -> None:
    """One subplot per workload: throughput vs. SCISSOR coalescing gap threshold."""
    has_data = any(
        any(all_results.get(wl, {}).get(v[0], []) for v in GAP_VARIANTS)
        for wl in workloads
    )
    if not has_data:
        print("[plot] No gap sweep results, skipping gap analysis plot.")
        return

    n_wl = len(workloads)
    fig, axes = plt.subplots(1, n_wl,
                             figsize=(max(5 * n_wl, 8), 4.5),
                             sharey=False)
    if n_wl == 1:
        axes = [axes]

    for ax, wl in zip(axes, workloads):
        wl_data  = all_results.get(wl, {})
        x_labels, means, errs, fcs, hatches, ecs = [], [], [], [], [], []

        for eng_key, x_label, fc, hatch, ec in GAP_VARIANTS:
            vals = wl_data.get(eng_key, [])
            m = float(np.mean(vals)) if vals else 0.0
            e = float(np.std(vals))  if len(vals) > 1 else 0.0
            x_labels.append(x_label)
            means.append(m); errs.append(e)
            fcs.append(fc); hatches.append(hatch); ecs.append(ec)

        x_pos = np.arange(len(x_labels))
        bar_w = 0.6
        max_y = max((m + e for m, e in zip(means, errs)), default=0.5)

        for i, (m, e, fc, h, ec) in enumerate(zip(means, errs, fcs, hatches, ecs)):
            err_kw = {"error_kw": {"elinewidth": 1.2, "capsize": 3}} if e > 0 else {}
            ax.bar(x_pos[i], m, width=bar_w,
                   color=fc, hatch=h, edgecolor=ec, linewidth=0.8,
                   yerr=(e if e > 0 else None), **err_kw, zorder=3)
            if m > 0:
                ax.text(x_pos[i], m + max_y * 0.03,
                        f"{m:.2f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_xlabel("Gap threshold", fontsize=10)
        ax.set_ylabel("Throughput (GB/s)", fontsize=10)
        ax.set_ylim(0, max(max_y * 1.35, 0.5))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(WORKLOAD_LABELS.get(wl, wl), fontsize=10)

    fig.suptitle(
        "SCISSOR Gap Threshold Analysis: Throughput vs. Coalescing Gap",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = os.path.join(outdir, f"bench_dlio_gap_analysis.{ext}")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"[plot] {out_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SCISSOR IO engine benchmark (phases 3 to 4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--workloads", nargs="+", default=DEFAULT_WORKLOADS,
                        help=f"Workload config names (default: {DEFAULT_WORKLOADS})")
    parser.add_argument("--pfs-root", default=os.environ.get("PFS_ROOT"),
                        help="PFS directory where prepare_dlio.py wrote the data "
                             "(default: $PFS_ROOT). "
                             "Required unless --no-copy or --no-run.")
    parser.add_argument("--ssd-root", default=os.environ.get("SSD_ROOT"),
                        help="SSD directory where data is copied for benchmarking "
                             "(default: $SSD_ROOT). "
                             "Required unless --no-run.")
    parser.add_argument("--traces-dir", default=None,
                        help=f"Directory containing the prepared traces "
                             f"(default: {TRACES_DIR}).")
    parser.add_argument("--vmtouch-dir", default=os.environ.get("VMTOUCH_DIR"),
                        help="Directory containing the vmtouch binary "
                             "(default: $VMTOUCH_DIR). "
                             "Used instead of /proc/sys/vm/drop_caches.")
    parser.add_argument("--bench", default=str(SRC / "bench"),
                        help="Path to the bench binary.")
    parser.add_argument("--runs", type=int, default=3,
                        help="Benchmark repetitions per engine per workload.")
    parser.add_argument("--gap-sweep", action="store_true",
                        help="Also run scissor_g{0,1,2,4,8,16} for gap analysis plot.")
    parser.add_argument("--gap-batch-sweep", action="store_true",
                        help=f"Run scissor_sr with fixed batch={FIXED_BATCH_MB} MB "
                             f"varying gap over {GAP_SWEEP_FIXED_BATCH_VALUES}.")
    parser.add_argument("--batch-sweep", action="store_true",
                        help=f"Run scissor_sr with fixed gap (per workload optimum) "
                             f"varying batch over {BATCH_SWEEP_MB_VALUES} MB.")
    parser.add_argument("--no-copy", action="store_true",
                        help="Skip rsync and assume data already at --ssd-root.")
    parser.add_argument("--no-run", action="store_true",
                        help="Skip phase 3 and only plot from existing results.")
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "dlio"),
                        help="Directory for JSON result files.")
    parser.add_argument("--outdir", default=str(ROOT / "plots"),
                        help="Directory for output figures.")
    args = parser.parse_args()

    traces_dir = Path(args.traces_dir) if args.traces_dir else TRACES_DIR

    workloads = [w for w in args.workloads if w in WORKLOAD_LABELS]
    unknown   = [w for w in args.workloads if w not in WORKLOAD_LABELS]
    if unknown:
        print(f"[WARN] Unknown workloads ignored: {unknown}\n"
              f"       Known: {sorted(WORKLOAD_LABELS)}", file=sys.stderr)
    if not workloads:
        sys.exit("[bench_dlio] No valid workloads specified.")

    if not args.no_run:
        if not os.path.exists(args.bench):
            sys.exit(f"[bench_dlio] bench binary not found: {args.bench}\n"
                     f"  Run: cd {SRC} && make")
        if args.ssd_root is None:
            sys.exit("[bench_dlio] --ssd-root is required unless --no-run.")
        if not args.no_copy and args.pfs_root is None:
            sys.exit("[bench_dlio] --pfs-root is required unless --no-copy or --no-run.")

    all_results: dict[str, dict[str, list[float]]] = {}

    for wl in workloads:
        cfg_path = CFG_DIR / f"{wl}.yaml"
        if not cfg_path.exists():
            print(f"[WARN] Config not found: {cfg_path}", file=sys.stderr)
            continue
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh)

        sel_mode = cfg.get("selection", {}).get("mode", "none")
        print(f"\n{'='*60}")
        print(f"  Workload : {wl}  (selection: {sel_mode})")
        print(f"{'='*60}")

        if args.no_run:
            all_results = load_results_from_disk(workloads, args.results_dir)
            break

        num_files = cfg["dataset"]["num_files_train"]

        # Copy PFS → SSD (or just write manifest if --no-copy)
        if not args.no_copy:
            ssd_data_dir = copy_to_ssd(wl, args.pfs_root, args.ssd_root, num_files)
        else:
            ssd_data_dir = os.path.join(args.ssd_root, wl)
            write_manifest(ssd_data_dir, num_files)

        # Locate the replay trace (sampled or full)
        sampled = traces_dir / f"{wl}_sampled.bin"
        full    = traces_dir / f"{wl}_trace.bin"
        if sampled.exists():
            replay_trace = str(sampled)
        elif full.exists():
            print(f"  [WARN] Sampled trace not found, falling back to full trace.",
                  file=sys.stderr)
            replay_trace = str(full)
        else:
            print(f"  [SKIP] {wl}: no trace found in {traces_dir}", file=sys.stderr)
            continue

        # Phase 3: benchmark
        results = phase_bench(
            workload        = wl,
            ssd_data_dir    = ssd_data_dir,
            trace_bin       = replay_trace,
            results_dir     = args.results_dir,
            bench_bin       = args.bench,
            n_runs          = args.runs,
            cfg             = cfg,
            gap_sweep       = args.gap_sweep,
            gap_batch_sweep = args.gap_batch_sweep,
            batch_sweep     = args.batch_sweep,
            vmtouch_dir     = args.vmtouch_dir,
        )
        all_results[wl] = results

    # Phase 4: plot
    plot_throughput(all_results, workloads, args.outdir)
    plot_gap_analysis(all_results, workloads, args.outdir)

    # Summary table
    all_keys = [e[0] for e in ENGINES]
    if args.gap_sweep or args.no_run:
        all_keys += [v[0] for v in GAP_VARIANTS]
    if args.gap_batch_sweep:
        all_keys += [f"scissor_sr_b{FIXED_BATCH_MB}mb_g{g}"
                     for g in GAP_SWEEP_FIXED_BATCH_VALUES]
    if args.batch_sweep:
        seen = set(all_keys)
        for wl in workloads:
            fg = FIXED_GAP_BY_WORKLOAD.get(wl, 1)
            for b in BATCH_SWEEP_MB_VALUES:
                k = f"scissor_sr_g{fg}_b{b}mb"
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

    print(f"\n{'='*60}")
    print(f"{'Workload':<25} {'Engine':<15} {'Mean GB/s':>10} {'Runs':>6}")
    print(f"{'='*60}")
    for wl in workloads:
        for key in all_keys:
            vals  = all_results.get(wl, {}).get(key, [])
            label = next((e[1] for e in ENGINES if e[0] == key),
                         next((v[1] for v in GAP_VARIANTS if v[0] == key), key))
            if vals:
                print(f"{wl:<25} {label:<15} {np.mean(vals):>10.3f} {len(vals):>6}")
            else:
                print(f"{wl:<25} {label:<15} {'N/A':>10} {'0':>6}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
