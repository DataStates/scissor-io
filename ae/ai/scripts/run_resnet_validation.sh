#!/usr/bin/env bash
# run_resnet_validation.sh: End-to-end validation of the resnet50_clustered
# pipeline: prepare data, extract trace, apply selection filter, benchmark all
# engines + gap sweep, and print a summary table.
#
# Usage:
#   bash run_resnet_validation.sh --data-root /path/to/data [OPTIONS]
#
# Required:
#   --data-root DIR     Directory where dataset files will be generated.
#                       Must have enough free space (see workload config).
#
# Optional:
#   --workload NAME     Workload config name (default: test_resnet50_clustered).
#                       The corresponding YAML must exist in config/dlio/.
#   --venv PATH         Path to the virtualenv activate script.
#                       (default: auto-detected from PATH / common locations)
#   --vmtouch PATH      Path to the vmtouch binary.  When provided, page cache
#                       is evicted with: vmtouch -eq <data_dir>
#                       Falls back to: sudo tee /proc/sys/vm/drop_caches
#   --log FILE          Log file path (default: /tmp/resnet_validation.log).
#   --no-prepare        Skip phases 1 to 2.5 (reuse existing data and traces).
#   -h, --help          Show this message.
#
# Cache eviction (in priority order):
#   1. vmtouch -eq <dir>           if --vmtouch is given and the binary exists
#   2. sudo tee /proc/sys/vm/drop_caches   if passwordless sudo is available
#   3. Warning printed, run continues without cache eviction
#
# To grant passwordless cache-drop without full sudo, add via: sudo visudo
#   <your-user> ALL=(ALL) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches
#
# Examples:
#   # Minimal (vmtouch available):
#   bash run_resnet_validation.sh \
#       --data-root /tmp/scissor_data \
#       --vmtouch   /usr/local/bin/vmtouch
#
#   # Reuse existing data, skip prepare:
#   bash run_resnet_validation.sh \
#       --data-root /tmp/scissor_data \
#       --no-prepare

set -euo pipefail

# ── Locate this script's directory (works regardless of where you call it from)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# ── Derived paths (no hardcoded machine paths below this block) ───────────────
BENCH="$REPO_ROOT/src/bench"
RESULTS_DIR="$REPO_ROOT/results/dlio"
CONFIG_DIR="$REPO_ROOT/config/dlio"

# ── Defaults ──────────────────────────────────────────────────────────────────
DATA_ROOT=""
WORKLOAD="test_resnet50_clustered"
VENV_ACTIVATE=""
VMTOUCH_BIN=""
LOG="/tmp/resnet_validation.log"
SKIP_PREPARE=false
COMPUTE_MEAN_MS_OVERRIDE=""  # empty = read from YAML

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
    sed -n '2,/^set -euo/{ /^set -euo/d; s/^# \{0,1\}//; p }' "${BASH_SOURCE[0]}"
    exit 0
}

_require_value() {
    # Guard against empty $DATA_DIR / unset variables being passed as option values.
    # Usage: _require_value --flag "$value"
    if [[ -z "$2" || "$2" == --* ]]; then
        echo "ERROR: $1 requires a non-empty value (got: '${2:-<empty>}')." >&2
        echo "       If you passed a shell variable, check that it is set first." >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)  _require_value "$1" "$2"; DATA_ROOT="$2";     shift 2 ;;
        --workload)   _require_value "$1" "$2"; WORKLOAD="$2";       shift 2 ;;
        --venv)       _require_value "$1" "$2"; VENV_ACTIVATE="$2";  shift 2 ;;
        --vmtouch)    _require_value "$1" "$2"; VMTOUCH_BIN="$2";    shift 2 ;;
        --log)        _require_value "$1" "$2"; LOG="$2";            shift 2 ;;
        --no-prepare)        SKIP_PREPARE=true;                                    shift   ;;
        --compute-mean-ms)   _require_value "$1" "$2"; COMPUTE_MEAN_MS_OVERRIDE="$2"; shift 2 ;;
        -h|--help)    usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

if [[ -z "$DATA_ROOT" ]]; then
    echo "ERROR: --data-root is required." >&2
    usage
fi

# ── Activate virtualenv ───────────────────────────────────────────────────────
_activate_venv() {
    if [[ -n "$VENV_ACTIVATE" ]]; then
        # shellcheck source=/dev/null
        source "$VENV_ACTIVATE"
        return
    fi
    # Auto-detect: check if dlio_benchmark is already on PATH
    if command -v dlio_benchmark &>/dev/null; then
        return
    fi
    # Try common locations relative to REPO_ROOT
    for candidate in \
        "$REPO_ROOT/.venv/bin/activate" \
        "$REPO_ROOT/venv/bin/activate" \
        "$REPO_ROOT/../.venv/bin/activate"
    do
        if [[ -f "$candidate" ]]; then
            echo "[venv] Activating: $candidate"
            # shellcheck source=/dev/null
            source "$candidate"
            return
        fi
    done
    echo "WARNING: Could not find a virtualenv. Pass --venv /path/to/activate" >&2
    echo "         Continuing, but will fail if dlio_benchmark / python packages are missing." >&2
}

_activate_venv
DLIO_BIN="$(command -v dlio_benchmark 2>/dev/null || true)"

# ── Start logging ─────────────────────────────────────────────────────────────
exec > >(tee "$LOG") 2>&1
echo "=== $(date) : resnet50 validation start ==="
echo "    repo        : $REPO_ROOT"
echo "    data-root   : $DATA_ROOT"
echo "    workload    : $WORKLOAD"
echo "    log         : $LOG"
echo "    vmtouch     : ${VMTOUCH_BIN:-<not set, will try sudo drop_caches>}"

# ── Directories ───────────────────────────────────────────────────────────────
TRACES_DIR="$REPO_ROOT/traces/dlio"
mkdir -p "$DATA_ROOT" "$TRACES_DIR" "$RESULTS_DIR"

# ── Cache eviction ────────────────────────────────────────────────────────────
drop_cache() {
    local target_dir="${1:-}"
    if [[ -n "$VMTOUCH_BIN" ]] && [[ -x "$VMTOUCH_BIN" ]]; then
        if [[ -n "$target_dir" ]] && [[ -d "$target_dir" ]]; then
            "$VMTOUCH_BIN" -eq "$target_dir" 2>/dev/null || true
        fi
        sync
        echo "  [cache] evicted via vmtouch: ${target_dir:-all}"
    elif sudo -n true 2>/dev/null; then
        sync
        echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
        echo "  [cache] page cache dropped via /proc"
    else
        echo "  [cache] WARNING: cannot evict cache: provide --vmtouch or passwordless sudo" >&2
    fi
}

# ── Phase 1-2.5: prepare (generate, trace, sample) ───────────────────────────
if [[ "$SKIP_PREPARE" == false ]]; then
    echo ""
    echo "=== $(date) : Phase 1-2.5 prepare ==="
    prepare_args=(
        "--workloads" "$WORKLOAD"
        "--data-root"  "$DATA_ROOT"
        "--traces-dir" "$TRACES_DIR"
    )
    [[ -n "$DLIO_BIN" ]] && prepare_args+=("--dlio-bin" "$DLIO_BIN")
    python3 "$SCRIPT_DIR/prepare_dlio.py" "${prepare_args[@]}"
else
    echo ""
    echo "=== $(date) : Skipping prepare (--no-prepare) ==="
fi

# ── Resolve trace and manifest ────────────────────────────────────────────────
TRACE="$TRACES_DIR/${WORKLOAD}_sampled.bin"
[[ -f "$TRACE" ]] || TRACE="$TRACES_DIR/${WORKLOAD}_trace.bin"
if [[ ! -f "$TRACE" ]]; then
    echo "ERROR: No trace found at $TRACES_DIR/${WORKLOAD}_{sampled,trace}.bin" >&2
    echo "       Run without --no-prepare, or run prepare_dlio.py manually." >&2
    exit 1
fi
MANIFEST="$DATA_ROOT/$WORKLOAD/manifest.txt"
if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest not found: $MANIFEST" >&2
    exit 1
fi
if [[ ! -x "$BENCH" ]]; then
    echo "ERROR: bench binary not found or not executable: $BENCH" >&2
    echo "       Build it with: cd $REPO_ROOT/src && make" >&2
    exit 1
fi

# ── Read benchmark params from YAML (not hardcoded) ──────────────────────────
CFG_FILE="$CONFIG_DIR/${WORKLOAD}.yaml"
if [[ ! -f "$CFG_FILE" ]]; then
    echo "ERROR: config not found: $CFG_FILE" >&2
    exit 1
fi
if [[ -n "$COMPUTE_MEAN_MS_OVERRIDE" ]]; then
    MEAN_MS="$COMPUTE_MEAN_MS_OVERRIDE"
else
    MEAN_MS=$(python3 -c "import yaml; c=yaml.safe_load(open('$CFG_FILE')); print(c['train']['computation_time']*1000)")
fi
STD_MS=$(python3 -c "print($MEAN_MS * 0.1)")
QD=$(python3 -c "import yaml; c=yaml.safe_load(open('$CFG_FILE')); print(c['benchmark']['queue_depth'])")
NRUNS=$(python3 -c "import yaml; c=yaml.safe_load(open('$CFG_FILE')); print(c['benchmark']['n_runs'])")
# chunk_size = record size so SCISSOR sees 1 chunk per record and force_gap units
# match inter-record gap widths.  record_dims product gives bytes per sample.
# Clamped to a multiple of 4096 for O_DIRECT alignment.
CHUNK_SIZE=$(python3 -c "
import yaml, math, sys
c = yaml.safe_load(open('$CFG_FILE'))
dims = c['dataset'].get('record_dims', c['dataset'].get('record_length_bytes', 65536))
dims = dims if isinstance(dims, list) else [dims]
sz = math.prod(int(d) for d in dims)
# O_DIRECT requires file offsets AND transfer sizes to be multiples of 4096.
# For DLIO indexed_binary, offset(i) = i * record_size, so all offsets are
# aligned iff record_size % 4096 == 0.  Abort early with a clear message.
if sz % 4096 != 0:
    print(f'ERROR: record_size={sz} B is not a multiple of 4096. '
          'O_DIRECT requires 4 KB-aligned offsets and transfer sizes. '
          'Check dataset.record_dims in the workload YAML.', file=sys.stderr)
    sys.exit(1)
print(sz)
")

DATA_TRAIN_DIR="$DATA_ROOT/$WORKLOAD/train"

echo ""
echo "=== $(date) : Phase 3 benchmark ==="
echo "    trace       : $TRACE"
echo "    manifest    : $MANIFEST"
echo "    mean_ms     : $MEAN_MS  std_ms: $STD_MS  qd: $QD  runs: $NRUNS"
echo "    chunk_size  : $CHUNK_SIZE B (= record size, 1 chunk per record)"

# ── Engine runner ─────────────────────────────────────────────────────────────
run_engine() {
    local eng="$1" gap="$2" run="$3"
    local suffix="$eng"
    [[ -n "$gap" ]] && suffix="${eng}_g${gap}"
    local out="$RESULTS_DIR/${WORKLOAD}_${suffix}_r${run}.json"
    local extra_args=()
    [[ "$eng" != "mmap" ]] && extra_args+=("--o-direct")
    [[ -n "$gap" ]] && extra_args+=("--force-gap" "$gap")
    local engine_arg="$eng"

    drop_cache "$DATA_TRAIN_DIR"
    echo "  [$suffix] run $run ..."
    "$BENCH" \
        --trace           "$TRACE" \
        --manifest        "$MANIFEST" \
        --engine          "$engine_arg" \
        --queue-depth     "$QD" \
        --chunk-size      "$CHUNK_SIZE" \
        --compute-mean-ms "$MEAN_MS" \
        --compute-std-ms  "$STD_MS" \
        --compute-dist    normal \
        --report-json     "$out" \
        "${extra_args[@]}"
    python3 - "$out" << 'PYEOF'
import json, sys
with open(sys.argv[1]) as f: d = json.load(f)
print(f"    -> {d['throughput_gbs']:.3f} GB/s  "
      f"{d['iops']:,.0f} IOPS  "
      f"io_wait_p50={d['io_wait_p50_ms']:.2f} ms  "
      f"compute_stall={d['compute_stall_pct']:.1f}%")
PYEOF
}

# All engines × n_runs
for eng in mmap io_uring scissor scissor_sr; do
    echo ""
    echo "--- Engine: $eng ---"
    for run in $(seq 1 "$NRUNS"); do
        run_engine "$eng" "" "$run"
    done
done

# Gap sweep: force_gap values matching regressor candidates {0,1,2,4,8,16}
echo ""
echo "--- Gap sweep (scissor) ---"
for gap in 0 1 2 4 8 16; do
    run_engine "scissor" "$gap" 1
done

echo ""
echo "--- Gap sweep (scissor_sr) ---"
for gap in 0 1 2 4 8 16; do
    run_engine "scissor_sr" "$gap" 1
done

# ── Summary table ─────────────────────────────────────────────────────────────
echo ""
echo "=== $(date) : Summary ==="
python3 - "$RESULTS_DIR" "$WORKLOAD" << 'PYEOF'
import json, glob, sys

results_dir = sys.argv[1]
wl          = sys.argv[2]

def mean(vals): return sum(vals) / len(vals) if vals else float("nan")

print(f"\n{'Engine':<20} {'R1':>7} {'R2':>7} {'R3':>7} {'Mean':>7} {'IOPS':>10} {'p50 ms':>8}")
print("-" * 72)

for key in ["mmap", "io_uring",
            "scissor",
            "scissor_g0", "scissor_g1", "scissor_g2",
            "scissor_g4", "scissor_g8", "scissor_g16",
            "scissor_sr",
            "scissor_sr_g0", "scissor_sr_g1", "scissor_sr_g2",
            "scissor_sr_g4", "scissor_sr_g8", "scissor_sr_g16"]:
    vals, iops, p50 = [], [], []
    for f in sorted(glob.glob(f"{results_dir}/{wl}_{key}_r*.json")):
        with open(f) as fh:
            d = json.load(fh)
        vals.append(d["throughput_gbs"])
        iops.append(d["iops"])
        p50.append(d["io_wait_p50_ms"])
    if not vals:
        continue
    row = f"{key:<20}"
    for v in vals:
        row += f" {v:>7.3f}"
    for _ in range(3 - len(vals)):
        row += f" {'':>7}"
    row += f" {mean(vals):>7.3f} {mean(iops):>10,.0f} {mean(p50):>8.2f}"
    print(row)
PYEOF

echo ""
echo "=== $(date) : DONE. Full log: $LOG ==="
