#!/usr/bin/env python3
"""
prepare_dlio.py: Data preparation pipeline for the SCISSOR benchmark.

Runs the expensive phases that need large storage (PFS) and DLIO access:
  Phase 1    Generate DLIO indexed_binary dataset files.
  Phase 2    Extract binary IO trace (gen_trace_dlio.py).
  Phase 2.5  Apply data-selection filter (sample_trace.py).

Outputs written to disk:
  --data-root/<workload>/train/*.indexed_binary   : dataset files
  --data-root/<workload>/manifest.txt             : file manifest
  --traces-dir/<workload>_trace.bin               : full IO trace
  --traces-dir/<workload>_trace.csv               : full IO trace (CSV)
  --traces-dir/<workload>_sampled.bin             : filtered trace
  --traces-dir/<workload>_sampled.csv             : filtered trace (CSV)

The traces directory (--traces-dir) must be on a filesystem accessible
from the machine that runs bench_dlio.py.  If the generation and benchmark
nodes share a PFS (e.g. Lustre), the default location inside the repo is
sufficient.  Otherwise, copy the traces/ directory manually, or point
--traces-dir at a shared path.

Usage:
    # Generate data + traces for all primary workloads
    # (--data-root defaults to $PFS_ROOT):
    python3 scripts/prepare_dlio.py \\
        --data-root  $PFS_ROOT

    # Single workload, custom traces directory:
    python3 scripts/prepare_dlio.py \\
        --workloads  bert_clustered \\
        --data-root  $PFS_ROOT \\
        --traces-dir $PFS_ROOT/traces

    # Override the DLIO binary and its Hydra config path:
    python3 scripts/prepare_dlio.py \\
        --data-root      $PFS_ROOT \\
        --dlio-bin       /path/to/dlio_benchmark \\
        --dlio-config-path /path/to/dlio_benchmark/dlio_benchmark/configs

    # Skip data generation (DLIO files already exist), re-run trace only:
    python3 scripts/prepare_dlio.py \\
        --data-root $PFS_ROOT --no-generate

    # Skip all generation, re-run selection filter only:
    python3 scripts/prepare_dlio.py \\
        --data-root $PFS_ROOT --no-generate --no-trace
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

# Local shared module
sys.path.insert(0, str(Path(__file__).parent))
from _dlio_common import (
    CFG_DIR, TRACES_DIR, DLIO_CFGDIR,
    DEFAULT_WORKLOADS, WORKLOAD_LABELS,
    _dlio_base_workload, _record_dims_arg,
    _discover_train_files, write_manifest,
    run_cmd, find_dlio_bin,
    SCRIPT_DIR,
)


# ── Phase 1: Data generation ──────────────────────────────────────────────────

def phase_generate(workload: str, cfg: dict, data_root: str,
                   dlio_bin: str, dlio_config_path: Path) -> bool:
    """
    Generate the DLIO indexed_binary dataset.
    Returns True on success (or if files already exist), False on failure.
    """
    ds     = cfg["dataset"]
    reader = cfg.get("reader", {})
    outdir = os.path.abspath(os.path.join(data_root, workload))

    existing = _discover_train_files(outdir, ds["num_files_train"])
    if len(existing) == ds["num_files_train"]:
        print(f"\n[generate] {workload}: {len(existing)} files already exist, skipping")
        write_manifest(outdir, ds["num_files_train"])
        return True

    print(f"\n[generate] {workload} → {outdir}")
    os.makedirs(outdir, exist_ok=True)

    base_wl = _dlio_base_workload(cfg)
    cmd = [
        dlio_bin,
        "--config-path", str(dlio_config_path),
        "--config-name", "config",
        f"workload={base_wl}",
        "++workload.workflow.generate_data=True",
        "++workload.workflow.train=False",
        f"++workload.dataset.data_folder={outdir}",
        f"++workload.dataset.num_files_train={ds['num_files_train']}",
        f"++workload.dataset.num_samples_per_file={ds['num_samples_per_file']}",
        _record_dims_arg(cfg),
        f"++workload.reader.batch_size={reader.get('batch_size', 1)}",
        # DLIO converts each generation buffer to Python integers (about 80 B
        # of memory per byte), so the 2 GB default exhausts memory for large files
        "++workload.dataset.generation_buffer_size=16777216",
    ]
    if not run_cmd(cmd, label=f"dlio_benchmark generate {workload}"):
        print(f"  [SKIP] {workload}: DLIO data generation failed", file=sys.stderr)
        return False

    write_manifest(outdir, ds["num_files_train"])
    return True


# ── Phase 2: Trace extraction ─────────────────────────────────────────────────

def phase_trace(workload: str, data_root: str, traces_dir: Path) -> str | None:
    """
    Extract the full IO trace from the DLIO dataset.
    Returns the path to the full trace binary, or None on failure.
    """
    cfg_path = CFG_DIR / f"{workload}.yaml"
    data_dir = os.path.abspath(os.path.join(data_root, workload))
    traces_dir.mkdir(parents=True, exist_ok=True)

    bin_path = str(traces_dir / f"{workload}_trace.bin")
    csv_path = str(traces_dir / f"{workload}_trace.csv")

    print(f"\n[trace] {workload}")
    ok = run_cmd(
        [sys.executable, str(SCRIPT_DIR / "gen_trace_dlio.py"),
         "--dlio-config", str(cfg_path),
         "--data-dir",    data_dir,
         "--output-bin",  bin_path,
         "--output-csv",  csv_path],
        label=f"gen_trace_dlio {workload}",
    )
    if not ok or not os.path.exists(bin_path):
        print(f"  [WARN] Trace not produced for {workload}", file=sys.stderr)
        return None
    return bin_path


# ── Phase 2.5: Trace selection filter ────────────────────────────────────────

def phase_sample(workload: str, full_trace_bin: str,
                 cfg: dict, traces_dir: Path) -> str:
    """
    Apply the data-selection filter defined in the workload config.
    If mode=none or no [selection] section, returns the full trace unchanged.
    Returns the path to the trace to replay.
    """
    sel_cfg = cfg.get("selection", {})
    mode    = sel_cfg.get("mode", "none")
    if mode == "none":
        return full_trace_bin

    from sample_trace import apply_selection  # noqa: PLC0415

    reader         = cfg.get("reader", {})
    train          = cfg.get("train", {})
    batch_size     = int(reader.get("batch_size", 256))
    compute_time_s = float(train.get("computation_time", 0.1))
    seed           = int(sel_cfg.get("seed", 42))

    sampled_bin = str(traces_dir / f"{workload}_sampled.bin")
    sampled_csv = str(traces_dir / f"{workload}_sampled.csv")

    print(f"\n[sample] {workload}  mode={mode}")
    apply_selection(
        input_bin      = full_trace_bin,
        output_bin     = sampled_bin,
        selection_cfg  = sel_cfg,
        batch_size     = batch_size,
        compute_time_s = compute_time_s,
        output_csv     = sampled_csv,
        seed           = seed,
    )
    return sampled_bin


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DLIO data generation and trace preparation for the SCISSOR benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--workloads", nargs="+", default=DEFAULT_WORKLOADS,
                        help=f"Workload config names (default: {DEFAULT_WORKLOADS})")
    parser.add_argument("--data-root", default=os.environ.get("PFS_ROOT"),
                        help="Root directory for generated datasets (PFS path, "
                             "default: $PFS_ROOT).")
    parser.add_argument("--traces-dir", default=None,
                        help=("Directory for trace files "
                              f"(default: {TRACES_DIR}).  "
                              "Must be accessible from the benchmark node."))
    parser.add_argument("--dlio-bin", default=None,
                        help="Path to dlio_benchmark binary (default: auto-detect from PATH).")
    parser.add_argument("--dlio-config-path", default=None,
                        help=("Path to dlio_benchmark Hydra configs/ directory "
                              f"(default: auto-detected as {DLIO_CFGDIR})."))
    parser.add_argument("--no-generate", action="store_true",
                        help="Skip Phase 1 and reuse existing DLIO data files.")
    parser.add_argument("--no-trace", action="store_true",
                        help="Skip Phase 2 and reuse existing full traces.")
    parser.add_argument("--no-sample", action="store_true",
                        help="Skip Phase 2.5 and reuse existing sampled traces.")
    args = parser.parse_args()

    if args.data_root is None:
        sys.exit("[prepare_dlio] --data-root (or $PFS_ROOT) is required.")

    traces_dir = Path(args.traces_dir) if args.traces_dir else TRACES_DIR
    dlio_config_path = Path(args.dlio_config_path) if args.dlio_config_path else DLIO_CFGDIR
    if dlio_config_path is None:
        sys.exit("[prepare_dlio] DLIO Hydra configs not found: install dlio_benchmark, "
                 "set DLIO_DIR, or pass --dlio-config-path.")

    workloads = [w for w in args.workloads if w in WORKLOAD_LABELS]
    unknown   = [w for w in args.workloads if w not in WORKLOAD_LABELS]
    if unknown:
        print(f"[WARN] Unknown workloads ignored: {unknown}\n"
              f"       Known: {sorted(WORKLOAD_LABELS)}", file=sys.stderr)
    if not workloads:
        sys.exit("[prepare_dlio] No valid workloads specified.")

    dlio_bin = find_dlio_bin(args.dlio_bin)
    print(f"[prepare_dlio] dlio_benchmark : {dlio_bin}")
    print(f"[prepare_dlio] DLIO config    : {dlio_config_path}")
    print(f"[prepare_dlio] data-root      : {args.data_root}")
    print(f"[prepare_dlio] traces-dir     : {traces_dir}")

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

        # Phase 1
        if not args.no_generate:
            if not phase_generate(wl, cfg, args.data_root, dlio_bin, dlio_config_path):
                print(f"  [SKIP] {wl}: aborting after generation failure.",
                      file=sys.stderr)
                continue
        else:
            outdir = os.path.abspath(os.path.join(args.data_root, wl))
            write_manifest(outdir, cfg["dataset"]["num_files_train"])

        # Phase 2
        if not args.no_trace:
            full_trace = phase_trace(wl, args.data_root, traces_dir)
            if full_trace is None:
                print(f"  [SKIP] {wl}: aborting after trace failure.", file=sys.stderr)
                continue
        else:
            full_trace = str(traces_dir / f"{wl}_trace.bin")
            if not os.path.exists(full_trace):
                print(f"  [SKIP] {wl}: trace not found: {full_trace}", file=sys.stderr)
                continue

        # Phase 2.5
        if not args.no_sample:
            phase_sample(wl, full_trace, cfg, traces_dir)
        else:
            sampled = traces_dir / f"{wl}_sampled.bin"
            if not sampled.exists():
                print(f"  [WARN] Sampled trace not found: {sampled}. "
                      f"bench_dlio.py will fall back to full trace.", file=sys.stderr)

    print(f"\n[prepare_dlio] Done.  Traces in: {traces_dir}")
    print(f"[prepare_dlio] Pass --pfs-root {args.data_root} "
          f"--traces-dir {traces_dir} to bench_dlio.py.")


if __name__ == "__main__":
    main()
