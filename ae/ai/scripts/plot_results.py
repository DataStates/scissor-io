#!/usr/bin/env python3
"""
plot_results.py: Generate comparison charts from benchmark JSON results.

Handles multiple runs per configuration.  Files are expected to follow:
    {workload}_{engine}_r{N}.json

Single-run files ({workload}_{engine}.json) are also accepted for
backward compatibility.

The engine name is read from the JSON "engine" field, not the filename.

Usage:
    python3 scripts/plot_results.py --results-dir results/ --outdir plots/
"""

import argparse
import json
import os
import glob
import re
import sys
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[plot] matplotlib/numpy not available, printing text summary only")


# ── Engine display metadata ───────────────────────────────────────────────────
# Colors and hatches match evaluation_plots.ipynb:
#   MMap:    #1f77b4  //    black edge
#   IOUring: #ff7f0e  oo    black edge
#   SCISSOR: #2ca02c  xxx   black edge

MAIN_ENGINES = ["mmap", "libaio", "io_uring", "scissor_gauto"]

COLORS = {
    "mmap":          "#1f77b4",
    "libaio":        "#aec7e8",   # lighter blue, kept for completeness
    "io_uring":      "#ff7f0e",
    "scissor_gauto": "#2ca02c",
    # Gap sweep variants: sequential greens (light → dark)
    "scissor_g0":    "#c7e9c0",
    "scissor_g1":    "#74c476",
    "scissor_g2":    "#31a354",
    "scissor_g3":    "#006d2c",
}

HATCHES = {
    "mmap":          "//",
    "libaio":        "..",
    "io_uring":      "oo",
    "scissor_gauto": "xxx",
    "scissor_g0":    "xxx",
    "scissor_g1":    "//",
    "scissor_g2":    "\\\\",
    "scissor_g3":    "..",
}

LABELS = {
    "mmap":          "MMap",
    "libaio":        "libaio",
    "io_uring":      "IOUring",
    "scissor_gauto": "SCISSOR\n(auto)",
    "scissor_g0":    "0 B",
    "scissor_g1":    "1 MB",
    "scissor_g2":    "2 MB",
    "scissor_g3":    "3 MB",
}

METRICS = [
    ("throughput_gbs",  "Throughput (GB/s)", False),
    ("iops",            "IOPS",              False),
    ("io_wait_p50_ms",  "IO Wait P50 (ms)",  True),
    ("io_wait_p95_ms",  "IO Wait P95 (ms)",  True),
    ("io_wait_p99_ms",  "IO Wait P99 (ms)",  True),
]

GAP_METRICS = [
    ("io_wait_p50_ms", "IO Wait P50 (ms)", True),
    ("io_wait_p95_ms", "IO Wait P95 (ms)", True),
    ("throughput_gbs", "Throughput (GB/s)", False),
]

# Gap value labels for x-axis (force_gap integer → human-readable bytes/KB/MB)
# force_gap is passed directly to the SCISSOR Loader as the gap threshold in MB.
# gap=0 → 0 B (exact adjacency required), gap ≥ 1 → N MB gap allowed.
def gap_x_label(engine: str) -> str:
    """Return human-readable gap threshold label for x-axis."""
    if engine == "scissor_gauto":
        return "SCISSOR\nauto"
    m = re.match(r"scissor_g(\d+)$", engine)
    if not m:
        return engine
    val = int(m.group(1))
    if val == 0:
        return "0 B"
    mb = val
    if mb >= 1024:
        return f"{mb // 1024} GB"
    return f"{mb} MB"


def engine_label(engine: str) -> str:
    return LABELS.get(engine, engine)


def is_scissor_variant(engine: str) -> bool:
    return engine.startswith("scissor")


def gap_sort_key(engine: str) -> int:
    if engine == "scissor_gauto":
        return 9999
    m = re.match(r"scissor_g(\d+)$", engine)
    return int(m.group(1)) if m else -1


# ── Result loading ────────────────────────────────────────────────────────────

def load_results(results_dir: str) -> dict:
    """
    Load all JSON result files.

    Returns {workload: {engine: [run_dict, run_dict, ...]}}

    Each run_dict is the raw parsed JSON from one bench run.
    Multiple _r{N} files for the same (workload, engine) are collected
    into the list, and single-run files (no _r suffix) produce a list of one.
    """
    data = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            try:
                result = json.load(f)
            except json.JSONDecodeError:
                print(f"[plot] Skipping malformed JSON: {path}")
                continue

        engine = result.get("engine", "")
        if not engine:
            print(f"[plot] No 'engine' field in {path}, skipping")
            continue

        base = os.path.basename(path).replace(".json", "")
        base_no_run = re.sub(r"_r\d+$", "", base)
        suffix = "_" + engine
        if base_no_run.endswith(suffix):
            workload = base_no_run[: -len(suffix)]
        else:
            workload = base_no_run.rsplit("_", 1)[0]

        data.setdefault(workload, {}).setdefault(engine, []).append(result)

    return data


# ── Per-metric aggregation across runs ────────────────────────────────────────

def agg_runs(runs: list) -> dict:
    """
    Compute mean and sample std across a list of run dicts.

    Returns a dict with keys like "throughput_gbs", "throughput_gbs_std", etc.
    With a single run std is 0 (no error bars shown).
    """
    result = {"n_runs": len(runs)}
    for key, _, _ in METRICS:
        vals = [r.get(key, 0.0) for r in runs]
        if HAS_MPL:
            result[key]          = float(np.mean(vals))
            result[key + "_std"] = float(np.std(vals, ddof=min(1, len(vals) - 1)))
        else:
            mean = sum(vals) / len(vals)
            var  = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
            result[key]          = mean
            result[key + "_std"] = var ** 0.5
    return result


# ── Text summary ──────────────────────────────────────────────────────────────

def print_text_table(data: dict) -> None:
    print("\n" + "=" * 96)
    print(f"  {'Workload':<12} {'Engine':<18} {'Runs':>4}  "
          f"{'Throughput (GB/s)':>18}  {'P50 ms':>14}  {'P95 ms':>14}  {'P99 ms':>14}")
    print("-" * 96)
    for workload in sorted(data):
        engines = sorted(
            data[workload],
            key=lambda e: (is_scissor_variant(e), gap_sort_key(e), e))
        for engine in engines:
            runs = data[workload][engine]
            a    = agg_runs(runs)
            tput = f"{a['throughput_gbs']:.2f} ± {a['throughput_gbs_std']:.2f}"
            p50  = f"{a['io_wait_p50_ms']:.2f} ± {a['io_wait_p50_ms_std']:.2f}"
            p95  = f"{a['io_wait_p95_ms']:.2f} ± {a['io_wait_p95_ms_std']:.2f}"
            p99  = f"{a['io_wait_p99_ms']:.2f} ± {a['io_wait_p99_ms_std']:.2f}"
            print(f"  {workload:<12} {engine:<18} {a['n_runs']:>4}  "
                  f"{tput:>18}  {p50:>14}  {p95:>14}  {p99:>14}")
    print("=" * 96 + "\n")


# ── Shared bar-chart helper ───────────────────────────────────────────────────

def _bar_chart(ax, engines: list, agg_by_engine: dict,
               key: str, label: str, lower_better: bool,
               baseline_engine=None, baseline_agg=None,
               use_gap_labels: bool = False) -> None:
    means  = [agg_by_engine[e][key]          for e in engines]
    stds   = [agg_by_engine[e][key + "_std"] for e in engines]
    colors = [COLORS.get(e, "#888888")        for e in engines]
    htchs  = [HATCHES.get(e, "")             for e in engines]
    xlbls  = ([gap_x_label(e) for e in engines] if use_gap_labels
               else [engine_label(e) for e in engines])

    bars = ax.bar(xlbls, means,
                  color=colors, hatch=htchs, edgecolor="black", linewidth=0.8,
                  yerr=stds if any(s > 0 for s in stds) else None,
                  error_kw=dict(elinewidth=1.2, capsize=4, ecolor="#333333"))

    for bar, mean, std in zip(bars, means, stds):
        label_text = (f"{mean:.2f}\n±{std:.2f}" if std > 0 else f"{mean:.2f}")
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.04,
                label_text, ha="center", va="bottom",
                fontsize=7, fontweight="bold")

    if baseline_engine and baseline_agg:
        bval = baseline_agg[key]
        bstd = baseline_agg[key + "_std"]
        ax.axhline(bval, color=COLORS.get(baseline_engine, "#555"),
                   linestyle="--", linewidth=1.2,
                   label=f"{engine_label(baseline_engine)} = {bval:.2f} ± {bstd:.2f}")
        ax.legend(fontsize=7)

    ax.set_ylabel(label)
    ax.tick_params(axis="x", rotation=15)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda x, _: f"{x:,.0f}" if x >= 1000 else f"{x:.2f}"))
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    all_vals = means + ([baseline_agg[key]] if baseline_agg else [])
    ax.set_ylim(0, max(all_vals) * 1.30 if max(all_vals) > 0 else 1)
    ax.set_facecolor("#fff8f8" if lower_better else "#f8fff8")


# ── Main 3-engine comparison (mmap / io_uring / scissor_gauto) ───────────────

def plot_comparison(data: dict, outdir: str) -> None:
    """Bar charts: mmap / libaio / io_uring / scissor_gauto per workload."""
    if not HAS_MPL:
        return

    for workload, engines_data in data.items():
        engines = [e for e in MAIN_ENGINES if e in engines_data]
        if not engines:
            continue
        agg = {e: agg_runs(engines_data[e]) for e in engines}

        fig, axes = plt.subplots(1, len(METRICS),
                                 figsize=(5 * len(METRICS), 5))
        fig.suptitle(f"IO Engine Comparison: {workload}", fontsize=14,
                     fontweight="bold")

        for ax, (key, label, lower_better) in zip(axes, METRICS):
            _bar_chart(ax, engines, agg, key, label, lower_better)
            ax.set_title(label, fontsize=11)

        n = max(agg[e]["n_runs"] for e in engines)
        fig.text(0.5, 0.01, f"n = {n} runs per engine, error bars = ±1 SD",
                 ha="center", fontsize=8, color="#555")
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        path = os.path.join(outdir, f"{workload}_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[plot] Saved: {path}")


# ── Scissor gap sweep: throughput + latency vs. gap threshold ─────────────────

def plot_gap_sweep(data: dict, outdir: str) -> None:
    """
    Bar charts: scissor_g0/g1/g2/g3/gauto per workload.
    X-axis labels show the actual gap threshold in B / KB / MB.
    Dashed baseline = best available non-scissor engine.
    """
    if not HAS_MPL:
        return

    BASELINE_PREF = ["io_uring", "libaio", "mmap"]

    for workload, engines_data in data.items():
        variants = sorted(
            [e for e in engines_data if is_scissor_variant(e)],
            key=gap_sort_key)
        if len(variants) < 2:
            continue

        agg      = {e: agg_runs(engines_data[e]) for e in variants}
        baseline = next((e for e in BASELINE_PREF if e in engines_data), None)
        b_agg    = agg_runs(engines_data[baseline]) if baseline else None

        fig, axes = plt.subplots(1, len(GAP_METRICS),
                                 figsize=(5 * len(GAP_METRICS), 5))
        fig.suptitle(f"SCISSOR Gap Threshold: {workload}", fontsize=14,
                     fontweight="bold")

        for ax, (key, label, lower_better) in zip(axes, GAP_METRICS):
            _bar_chart(ax, variants, agg, key, label, lower_better,
                       baseline_engine=baseline, baseline_agg=b_agg,
                       use_gap_labels=True)
            ax.set_title(label, fontsize=11)
            ax.set_xlabel("Gap threshold", fontsize=10)

        n = max(agg[e]["n_runs"] for e in variants)
        fig.text(0.5, 0.01, f"n = {n} runs per gap, error bars = ±1 SD",
                 ha="center", fontsize=8, color="#555")
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        path = os.path.join(outdir, f"{workload}_scissor_gap_sweep.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[plot] Saved: {path}")


# ── Per-step IO wait timeline ─────────────────────────────────────────────────

def plot_timelines(data: dict, outdir: str) -> None:
    """
    Per-step IO wait: mean line ± shaded ±1 SD band across runs.
    Non-scissor engines first, scissor variants on top.
    """
    if not HAS_MPL:
        return

    for workload, engines_data in data.items():
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_title(f"IO Wait per Step: {workload}", fontsize=12,
                     fontweight="bold")

        order = (sorted([e for e in engines_data if not is_scissor_variant(e)])
                 + sorted([e for e in engines_data if is_scissor_variant(e)],
                          key=gap_sort_key))

        for engine in order:
            runs = engines_data[engine]
            step_vals = defaultdict(list)
            for run_data in runs:
                for s in run_data.get("steps", []):
                    step_vals[s["step"]].append(s["io_wait_ms"])

            if not step_vals:
                continue

            steps = sorted(step_vals)
            means = np.array([np.mean(step_vals[s]) for s in steps])
            stds  = np.array([np.std(step_vals[s],
                                     ddof=min(1, len(step_vals[s]) - 1))
                              for s in steps])

            color = COLORS.get(engine, "#888")
            ax.plot(steps, means, label=engine_label(engine),
                    color=color, linewidth=1.2, alpha=0.9)
            if stds.max() > 0:
                ax.fill_between(steps, means - stds, means + stds,
                                color=color, alpha=0.12)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("IO Wait (ms)")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3, linestyle="--")
        path = os.path.join(outdir, f"{workload}_io_timeline.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[plot] Saved: {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot benchmark comparison charts")
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing JSON result files")
    parser.add_argument("--outdir", default="plots",
                        help="Output directory for plots")
    args = parser.parse_args()

    data = load_results(args.results_dir)
    if not data:
        print(f"[plot] No results found in {args.results_dir}/")
        sys.exit(1)

    total_runs = sum(len(runs)
                     for wl in data.values()
                     for runs in wl.values())
    print(f"[plot] Loaded {total_runs} result files across "
          f"{len(data)} workload(s)")

    os.makedirs(args.outdir, exist_ok=True)

    print_text_table(data)
    plot_comparison(data, args.outdir)
    plot_gap_sweep(data, args.outdir)
    plot_timelines(data, args.outdir)
    print("[plot] Done.")


if __name__ == "__main__":
    main()
