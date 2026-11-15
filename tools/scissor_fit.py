#!/usr/bin/env python3
"""
scissor_fit.py: Fit IO and compute models from profiling data and write models.conf.

Usage:
  python3 scissor_fit.py \\
      --profile   profile.csv \\
      --storage   ssd|pfs|hybrid \\
      [--bench-jsons results/dlio/bert_pread_r1.json ...] \\
      [--workload-yamls config/dlio/bert_clustered.yaml ...] \\
      [--output   models.conf]
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Built-in Polaris defaults (used for non-calibrated tiers and fallbacks).
# ---------------------------------------------------------------------------
BUILTIN_IO_PARAMS = {
    "pfs":    (4.29e+00, -3.33e-02, 7.26e-06, 4.54e-01,  1.37e-01),
    "ssd":    (1.13e+01, -5.58e-03, 1.82e-05, 1.64e-01,  7.48e-02),
    "hybrid": (4.26e+00, -3.47e-02, 4.35e-06, 3.86e-01, -2.72e-02),
}

BUILTIN_COMPUTE = {
    "slope": 0.748,
    "intercept": -17.0366,
}


# ---------------------------------------------------------------------------
# IO throughput model: T = C * N^alpha * exp(-beta*N) * S^gamma * exp(-delta*S)
#   N = IOP count, S = total size in GiB, T = throughput in GiB/s
# ---------------------------------------------------------------------------
def throughput_model(data, C, alpha, beta, gamma, delta):
    """Vectorised throughput model for curve_fit."""
    import numpy as np
    iop, s = data
    return (C
            * np.power(np.clip(iop, 1, None), alpha)
            * np.exp(-beta * iop)
            * np.power(np.clip(s, 1e-9, None), gamma)
            * np.exp(-delta * s))


# ---------------------------------------------------------------------------
# Load profile CSV.
# ---------------------------------------------------------------------------
def load_profile_csv(csv_path):
    """Return (iop_array, size_gib_array, throughput_array) from profile CSV."""
    import numpy as np

    rows = []
    with open(csv_path, newline="") as fh:
        # Skip header line.
        header = fh.readline()
        _ = header  # noqa
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            try:
                n_iop       = float(parts[0])
                total_bytes = float(parts[2])
                throughput  = float(parts[4])
            except ValueError:
                continue
            total_gib = total_bytes / (1024.0 ** 3)
            rows.append((n_iop, total_gib, throughput))

    if not rows:
        return np.array([]), np.array([]), np.array([])

    arr = np.array(rows)
    return arr[:, 0], arr[:, 1], arr[:, 2]


# ---------------------------------------------------------------------------
# Load bench JSON files (pread engine only).
# ---------------------------------------------------------------------------
def load_bench_jsons(json_paths):
    """Return list of (n_iops, total_gib, throughput_gib_s) from pread JSON files."""
    import numpy as np

    records = []
    for json_path in json_paths:
        try:
            with open(json_path) as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  Warning: cannot read {json_path}: {exc}", file=sys.stderr)
            continue

        engine = data.get("engine", "")
        if not engine.startswith("pread"):
            continue

        for step in data.get("steps", []):
            io_wait_ms = step.get("io_wait_ms", 0.0)
            reqs       = step.get("reqs", 0)
            raw_bytes  = step.get("bytes", 0)

            # Exclude noisy micro-measurements.
            if io_wait_ms <= 1.0 or reqs <= 1:
                continue

            total_gib   = raw_bytes / (1024.0 ** 3)
            elapsed_s   = io_wait_ms / 1000.0
            throughput   = total_gib / elapsed_s if elapsed_s > 0 else 0.0

            if throughput <= 0 or total_gib <= 0:
                continue

            records.append((float(reqs), total_gib, throughput))

    if not records:
        return np.array([]), np.array([]), np.array([])

    arr = np.array(records)
    return arr[:, 0], arr[:, 1], arr[:, 2]


# ---------------------------------------------------------------------------
# Fit the IO model.
# ---------------------------------------------------------------------------
def fit_io_model(iop_arr, size_gib_arr, throughput_arr, storage_tier):
    """
    Fit the IO throughput model using scipy.optimize.curve_fit.
    Returns (params_tuple, r_squared, mae, n_points).
    Falls back to built-in defaults on failure.
    """
    try:
        from scipy.optimize import curve_fit
        import numpy as np
    except ImportError:
        print("Error: scipy and numpy are required. Install with:", file=sys.stderr)
        print("  pip install scipy numpy", file=sys.stderr)
        sys.exit(1)

    defaults = BUILTIN_IO_PARAMS[storage_tier]

    # Filter valid rows.
    mask = (throughput_arr > 0) & (size_gib_arr > 0) & (iop_arr >= 1)
    n_iop_filt    = iop_arr[mask]
    n_size_filt   = size_gib_arr[mask]
    n_tput_filt   = throughput_arr[mask]
    n_points      = int(mask.sum())

    if n_points < 5:
        print(f"  Warning: only {n_points} valid points after filtering, "
              "falling back to built-in defaults.", file=sys.stderr)
        return defaults, float("nan"), float("nan"), n_points

    p0 = list(defaults)

    try:
        popt, _ = curve_fit(
            throughput_model,
            (n_iop_filt, n_size_filt),
            n_tput_filt,
            p0=p0,
            maxfev=10000,
        )
    except RuntimeError as exc:
        print(f"  Warning: curve_fit failed ({exc}), "
              "falling back to built-in defaults.", file=sys.stderr)
        return defaults, float("nan"), float("nan"), n_points

    # Compute fit quality.
    predicted = throughput_model((n_iop_filt, n_size_filt), *popt)
    ss_res = np.sum((n_tput_filt - predicted) ** 2)
    ss_tot = np.sum((n_tput_filt - np.mean(n_tput_filt)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae       = float(np.mean(np.abs(n_tput_filt - predicted)))

    if not math.isnan(r_squared) and r_squared < 0.5:
        print(f"  Warning: IO model R²={r_squared:.3f} < 0.5, "
              "falling back to built-in defaults.", file=sys.stderr)
        return defaults, r_squared, mae, n_points

    return tuple(float(x) for x in popt), r_squared, mae, n_points


# ---------------------------------------------------------------------------
# Load workload YAMLs and fit compute model.
# ---------------------------------------------------------------------------
def fit_compute_model(yaml_paths):
    """
    Load computation_time and work_bytes from YAML files, fit log-log regression.
    Returns (slope, intercept, r_squared, n_points).
    """
    try:
        import yaml
        import numpy as np
    except ImportError as exc:
        print(f"  Warning: cannot import {exc.name}, "
              "using built-in compute defaults.", file=sys.stderr)
        return BUILTIN_COMPUTE["slope"], BUILTIN_COMPUTE["intercept"], float("nan"), 0

    records = []
    for yaml_path in yaml_paths:
        try:
            with open(yaml_path) as fh:
                cfg = yaml.safe_load(fh)
        except (OSError, Exception) as exc:
            print(f"  Warning: cannot read {yaml_path}: {exc}", file=sys.stderr)
            continue

        try:
            comp_time = float(cfg["train"]["computation_time"])
        except (KeyError, TypeError, ValueError):
            continue

        try:
            batch_size = int(cfg["reader"]["batch_size"])
        except (KeyError, TypeError, ValueError):
            continue

        # Determine sample_bytes.
        try:
            record_dims = cfg["dataset"]["record_dims"]
            sample_bytes = math.prod(record_dims)
        except (KeyError, TypeError):
            try:
                sample_bytes = int(cfg["dataset"]["record_length_bytes"])
            except (KeyError, TypeError, ValueError):
                continue

        work_bytes = batch_size * sample_bytes
        if work_bytes <= 0 or comp_time <= 0:
            continue
        records.append((float(work_bytes), comp_time))

    n_points = len(records)
    if n_points < 2:
        print(f"  Note: only {n_points} YAML points, "
              "using built-in compute defaults.", file=sys.stderr)
        return BUILTIN_COMPUTE["slope"], BUILTIN_COMPUTE["intercept"], float("nan"), n_points

    arr       = np.array(records)
    ln_x      = np.log(arr[:, 0])
    ln_y      = np.log(arr[:, 1])
    coeffs    = np.polyfit(ln_x, ln_y, 1)
    slope     = float(coeffs[0])
    intercept = float(coeffs[1])

    # R² for log-log fit.
    predicted_ln_y = slope * ln_x + intercept
    ss_res    = np.sum((ln_y - predicted_ln_y) ** 2)
    ss_tot    = np.sum((ln_y - np.mean(ln_y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return slope, intercept, r_squared, n_points


# ---------------------------------------------------------------------------
# Write models.conf.
# ---------------------------------------------------------------------------
def write_models_conf(output_path,
                      storage_tier,
                      io_params_fitted,
                      io_r2, io_mae, io_n,
                      slope, intercept, compute_r2, compute_n,
                      timestamp_str):
    """Write a complete models.conf with all three storage tiers."""
    with open(output_path, "w") as fh:
        fh.write("# scissor/models.conf\n")
        fh.write(f"# Generated by scissor_fit.py at {timestamp_str}\n")
        fh.write(f"# Calibrated storage tier : {storage_tier}\n")
        fh.write(f"# IO model: N={io_n}, R²={io_r2:.4f}, MAE={io_mae:.4f} GiB/s\n"
                 if not math.isnan(io_r2) else
                 f"# IO model: N={io_n} (fit failed, using defaults)\n")
        fh.write(f"# Compute model: N={compute_n}, R²={compute_r2:.4f}\n"
                 if not math.isnan(compute_r2) else
                 f"# Compute model: N={compute_n} (using defaults)\n")
        fh.write("\n")

        fh.write("[compute]\n")
        fh.write(f"slope     = {slope:.6f}\n")
        fh.write(f"intercept = {intercept:.6f}\n")
        fh.write("\n")

        for tier in ("pfs", "ssd", "hybrid"):
            if tier == storage_tier:
                params = io_params_fitted
                comment = f"# Fitted from calibration data (R²={io_r2:.4f})" \
                          if not math.isnan(io_r2) else \
                          "# Fallback to built-in defaults (fit failed)"
            else:
                params = BUILTIN_IO_PARAMS[tier]
                comment = "# Built-in Polaris defaults"

            params_str = ", ".join(f"{v:.6e}" for v in params)
            fh.write(f"[{tier}]\n")
            fh.write(f"{comment}\n")
            fh.write(f"io_params = {params_str}\n")
            fh.write("gaps      = 0, 1, 2, 4, 8, 16\n")
            fh.write("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fit IO and compute models from profiling data and write models.conf."
    )
    parser.add_argument("--profile",        required=True,
                        help="Path to calibration CSV from scissor-profile --calibrate")
    parser.add_argument("--storage",        required=True, choices=("pfs", "ssd", "hybrid"),
                        help="Storage tier being calibrated")
    parser.add_argument("--bench-jsons",    nargs="*", default=[],
                        help="Additional pread benchmark JSON files")
    parser.add_argument("--workload-yamls", nargs="*", default=[],
                        help="Workload YAML files for compute model fitting")
    parser.add_argument("--output",         default="models.conf",
                        help="Output models.conf path (default: models.conf)")
    args = parser.parse_args()

    # Check for scipy/numpy early.
    try:
        import numpy as np  # noqa: F401
        from scipy.optimize import curve_fit  # noqa: F401
    except ImportError as exc:
        print(f"Error: required package not found: {exc.name}", file=sys.stderr)
        print("Install with: pip install scipy numpy", file=sys.stderr)
        sys.exit(1)

    # Validate profile CSV exists.
    profile_path = Path(args.profile)
    if not profile_path.exists():
        print(f"Error: profile CSV not found: {args.profile}", file=sys.stderr)
        sys.exit(1)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # -----------------------------------------------------------------------
    # IO model fitting
    # -----------------------------------------------------------------------
    print(f"Loading profile CSV: {args.profile}")
    iop_arr, size_gib_arr, tput_arr = load_profile_csv(args.profile)
    print(f"  {len(iop_arr)} rows from CSV")

    # Load optional bench JSONs.
    if args.bench_jsons:
        print(f"Loading bench JSONs ({len(args.bench_jsons)} files) ...")
        extra_iop, extra_size, extra_tput = load_bench_jsons(args.bench_jsons)
        if len(extra_iop) > 0:
            import numpy as np
            iop_arr      = np.concatenate([iop_arr, extra_iop])
            size_gib_arr = np.concatenate([size_gib_arr, extra_size])
            tput_arr     = np.concatenate([tput_arr, extra_tput])
            print(f"  {len(extra_iop)} additional rows from bench JSONs")

    print(f"Fitting IO model for tier '{args.storage}' ...")
    io_params_fitted, io_r2, io_mae, io_n = fit_io_model(
        iop_arr, size_gib_arr, tput_arr, args.storage
    )
    if not math.isnan(io_r2):
        print(f"  N={io_n}, R²={io_r2:.4f}, MAE={io_mae:.4f} GiB/s")
        print(f"  Fitted params (C,alpha,beta,gamma,delta): "
              f"{', '.join(f'{v:.4e}' for v in io_params_fitted)}")
    else:
        print(f"  N={io_n}: using built-in defaults")

    # -----------------------------------------------------------------------
    # Compute model fitting
    # -----------------------------------------------------------------------
    print(f"Fitting compute model from {len(args.workload_yamls)} YAML(s) ...")
    slope, intercept, compute_r2, compute_n = fit_compute_model(args.workload_yamls)
    if not math.isnan(compute_r2):
        print(f"  N={compute_n}, R²={compute_r2:.4f}, slope={slope:.4f}, intercept={intercept:.4f}")
    else:
        print(f"  N={compute_n}: using built-in defaults "
              f"(slope={slope:.4f}, intercept={intercept:.4f})")

    # -----------------------------------------------------------------------
    # Write models.conf
    # -----------------------------------------------------------------------
    print(f"Writing models.conf to: {args.output}")
    write_models_conf(
        args.output,
        args.storage,
        io_params_fitted,
        io_r2, io_mae, io_n,
        slope, intercept, compute_r2, compute_n,
        timestamp_str,
    )
    print("Done.")

    # Print the written file.
    print(f"\n{'='*60}")
    print(f"Contents of {args.output}:")
    print('='*60)
    with open(args.output) as fh:
        print(fh.read())


if __name__ == "__main__":
    main()
