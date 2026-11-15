"""
_dlio_common.py: Shared constants, registries, and utilities for
                  prepare_dlio.py and bench_dlio.py.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()
ROOT        = SCRIPT_DIR.parent
SRC         = ROOT / "src"
CFG_DIR     = ROOT / "config" / "dlio"
TRACES_DIR  = ROOT / "traces" / "dlio"   # default, overridable via CLI --traces-dir

# DLIO Hydra config directory.  DLIO_DIR, if set, points at a dlio_benchmark
# source checkout (configs in $DLIO_DIR/dlio_benchmark/configs). Otherwise it
# is derived from the installed package.
# dlio_benchmark may be a namespace package (__file__ is None), so
# use __path__ to locate the package root reliably.
if os.environ.get("DLIO_DIR"):
    DLIO_CFGDIR = Path(os.environ["DLIO_DIR"]) / "dlio_benchmark" / "configs"
else:
    try:
        import dlio_benchmark as _dlio_pkg
        _pkg_path = Path(next(iter(_dlio_pkg.__path__)))
        DLIO_CFGDIR = _pkg_path / "configs"
    except (ImportError, StopIteration):
        DLIO_CFGDIR = None   # pass --dlio-config-path to prepare_dlio.py


# ── Workload registry ─────────────────────────────────────────────────────────
WORKLOAD_LABELS: dict[str, str] = {
    "bert_clustered":           "BERT-Large\n(clustered sel., 8 KB)",
    "resnet50_clustered":       "ResNet-50\n(per-class coreset, 160 KB)",
    "bert_packed":              "BERT-Large\n(seq. shard, 8 KB)",
    "resnet50_dist":            "ResNet-50\n(seq. shard, 160 KB)",
    "test_bert_clustered":      "BERT test\n(clustered)",
    "test_resnet50_clustered":  "ResNet test\n(clustered)",
}

DEFAULT_WORKLOADS: list[str] = ["bert_clustered", "resnet50_clustered"]

# ── Engine display config ─────────────────────────────────────────────────────
ENGINES: list[tuple] = [
    ("mmap",       "MMap",       "#1f77b4", "//",  "black"),
    ("io_uring",   "IOUring",    "#ff7f0e", "oo",  "black"),
    ("scissor",    "SCISSOR",    "#2ca02c", "xxx", "black"),
    ("scissor_sr", "SCISSOR-SR", "#9467bd", "+++", "black"),
]

# Gap-sweep variants: force_gap values match the Regressor's candidate gaps_ = {0,1,2,4,8,16}.
# Each gap value is a chunk-index gap. With chunk_size=163840 B (ResNet record size) the
# bridged byte gaps are: g0=0 B, g1=160 KB, g2=320 KB, g4=640 KB, g8=1.28 MB, g16=2.56 MB
GAP_VARIANTS: list[tuple] = [
    ("scissor_g0",       "0 B\n(g=0)",    "#edf8e9", "xxx",  "black"),
    ("scissor_g1",       "160 KB\n(g=1)", "#c7e9c0", "///",  "black"),
    ("scissor_g2",       "320 KB\n(g=2)", "#a1d99b", "\\\\", "black"),
    ("scissor_g4",       "640 KB\n(g=4)", "#74c476", "//",   "black"),
    ("scissor_g8",       "1.28 MB\n(g=8)","#31a354", "...",  "black"),
    ("scissor_g16",      "2.56 MB\n(g=16)","#006d2c","xxx",  "black"),
    ("scissor_gauto",    "SCISSOR\nauto", "#2ca02c", "xxx",  "black"),
    ("scissor_sr_g0",    "0 B\n(g=0)",    "#e8d5f5", "xxx",  "black"),
    ("scissor_sr_g1",    "160 KB\n(g=1)", "#d4b0ea", "///",  "black"),
    ("scissor_sr_g2",    "320 KB\n(g=2)", "#bb8bd5", "\\\\", "black"),
    ("scissor_sr_g4",    "640 KB\n(g=4)", "#9e65bf", "//",   "black"),
    ("scissor_sr_g8",    "1.28 MB\n(g=8)","#7c3faa", "...",  "black"),
    ("scissor_sr_g16",   "2.56 MB\n(g=16)","#5b1a96","xxx",  "black"),
    ("scissor_sr_gauto", "SCISSOR-SR\nauto","#9467bd","+++",  "black"),
]

# Gap index → chunk-index value (matches Regressor gaps_ in models.conf)
GAP_SWEEP_VALUES: list[int] = [0, 1, 2, 4, 8, 16]

# ── Controlled sweep parameters ───────────────────────────────────────────────
# Gap sweep with fixed batch: vary gap, hold batch at FIXED_BATCH_MB.
GAP_SWEEP_FIXED_BATCH_VALUES: list[int] = [1, 2, 4, 8]   # chunk-index gaps
FIXED_BATCH_MB: int = 256

# Batch sweep with fixed gap: vary batch, hold gap at 1 chunk.
BATCH_SWEEP_MB_VALUES: list[int] = [32, 64, 128, 256]    # batch sizes in MiB
FIXED_GAP_BY_WORKLOAD: dict[str, int] = {
    "bert_clustered":     8,
    "resnet50_clustered": 8,
}

# ── Access-pattern annotations ────────────────────────────────────────────────
PATTERN_NOTES: dict[str, str] = {
    "bert_clustered":           "clustered sel. 50%\ngap~10 KB → SCISSOR 3-5×",
    "resnet50_clustered":       "per-class coreset 60%\ngap~273 KB → SCISSOR 2-3×",
    "bert_packed":              "sequential shard\n256×8 KB → 1 IOP",
    "resnet50_dist":            "sequential shard\n128×160 KB → 1 IOP",
    "test_bert_clustered":      "clustered sel. (test)",
    "test_resnet50_clustered":  "per-class (test)",
}


# ── DLIO base workload mapping ────────────────────────────────────────────────
# Both bert_clustered and resnet50_clustered use indexed_binary format.
# In the standard DLIO installation, "dlrm" is the only built-in workload
# that declares format: indexed_binary.  All dataset fields are overridden
# via Hydra "++" overrides, and the base template only sets the data format.
_DLIO_BASE_WORKLOAD: dict[str, str] = {
    "bert":   "dlrm",
    "resnet": "dlrm",
}

def _dlio_base_workload(cfg: dict) -> str:
    """Return the DLIO Hydra workload group name for the given benchmark config."""
    model_name = cfg.get("model", {}).get("name", "")
    for prefix, base in _DLIO_BASE_WORKLOAD.items():
        if model_name.startswith(prefix):
            return base
    return "dlrm"   # safe default: only indexed_binary template available


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _record_dims_arg(cfg: dict) -> str:
    """Build the Hydra override for record dimensions."""
    ds   = cfg["dataset"]
    dims = ds.get("record_dims")
    if dims is not None:
        d = dims if isinstance(dims, list) else [dims]
        return f"++workload.dataset.record_dims=[{','.join(str(x) for x in d)}]"
    rb = ds.get("record_length_bytes", 65536)
    return f"++workload.dataset.record_length_bytes={rb}"


def _discover_train_files(outdir: str, num_files: int) -> list[str]:
    """Return sorted list of up to num_files .indexed_binary data files."""
    pattern = os.path.join(outdir, "train", "*.indexed_binary")
    files   = sorted(f for f in glob.glob(pattern)
                     if not f.endswith(".off.idx") and not f.endswith(".sz.idx"))
    return files[:num_files]


def write_manifest(outdir: str, num_files: int) -> bool:
    """
    Write manifest.txt listing absolute paths of all .indexed_binary files
    under outdir/train/.  Returns True on success.
    """
    files = _discover_train_files(outdir, num_files)
    if not files:
        print(f"  [WARN] No .indexed_binary files under {outdir}/train/",
              file=sys.stderr)
        return False
    manifest_path = os.path.join(outdir, "manifest.txt")
    with open(manifest_path, "w") as fh:
        for p in files:
            fh.write(os.path.abspath(p) + "\n")
    print(f"  [manifest] {manifest_path}  ({len(files)} files)")
    return True


# ── General utilities ─────────────────────────────────────────────────────────

def run_cmd(cmd: list, label: str = "") -> bool:
    """Run a command, streaming stdout/stderr line-by-line so progress is visible."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    sys.stdout.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        print(f"    {line}", end="")
        sys.stdout.flush()
    proc.wait()
    if proc.returncode != 0:
        print(f"  [WARN] {label} failed (exit {proc.returncode})", file=sys.stderr)
        return False
    return True


def load_result(path: str) -> dict | None:
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def throughput_gbps(result: dict) -> float | None:
    if "throughput_gbs" in result:
        return float(result["throughput_gbs"])
    if ("total_bytes_gb" in result and "elapsed_s" in result
            and result["elapsed_s"] > 0):
        return result["total_bytes_gb"] / result["elapsed_s"]
    for key in ("throughput_gbps", "mean_throughput_gbps", "read_throughput_gbps"):
        if key in result:
            return float(result[key])
    return None


def find_dlio_bin(override: str | None) -> str:
    if override:
        return override
    found = shutil.which("dlio_benchmark")
    if found:
        return found
    sys.exit(
        "[dlio] dlio_benchmark not found on PATH.\n"
        "  Activate your venv: source .venv/bin/activate\n"
        "  Or pass: --dlio-bin /path/to/dlio_benchmark"
    )
