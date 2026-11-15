#!/usr/bin/env bash
# run_all.sh: Run all engines × all workloads, N_RUNS times each.
#
# Each run is preceded by a full page-cache drop so IO hits storage,
# not the OS buffer cache.  Results are written as:
#   results/{workload}_{engine}_r{N}.json
#
# The scissor engine is run once with Regressor-chosen gap (gauto) and
# once per forced gap value in SCISSOR_GAPS.
#
# Environment overrides:
#   N_RUNS=5          number of repeated runs per configuration (default 3)
#   DATA_ROOT=/path   dataset root directory (default /tmp/ai_bench_data)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
SRC="$ROOT/src"
SCRIPTS="$ROOT/scripts"
CFG="$ROOT/config"
DATA_ROOT="${DATA_ROOT:-/tmp/ai_bench_data}"
RESULTS="$ROOT/results"
BENCH="$SRC/bench"

N_RUNS="${N_RUNS:-3}"

WORKLOADS=("resnet50" "bert" "unet3d")
BASE_ENGINES=("pread" "libaio" "io_uring")
SCISSOR_GAPS=(0 1 2 3)

mkdir -p "$RESULTS"

# ── Page-cache drop helper ──────────────────────────────────────────
drop_caches() {
    local label="${1:-run}"
    printf "    [cache] Dropping page cache for %s ... " "$label"
    if [ "$(id -u)" -eq 0 ]; then
        sync && echo 3 > /proc/sys/vm/drop_caches && echo "ok"
    elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
        sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches" && echo "ok"
    else
        echo "SKIPPED (need root or passwordless sudo, results may reflect cached IO)"
    fi
}

# ── YAML scalar reader ──────────────────────────────────────────────
get_yaml() {   # get_yaml <workload> <section> <key>
    python3 -c "
import yaml
with open('$CFG/${1}.yaml') as f:
    c = yaml.safe_load(f)
print(c['${2}']['${3}'])
"
}

# ── Phase 0: Clean old results ──────────────────────────────────────
echo "=== Phase 0: Cleaning old results ==="
rm -f "$RESULTS"/*.json
echo "  results/ cleared"

# ── Phase 1: Generate traces ────────────────────────────────────────
echo ""
echo "=== Phase 1: Generating traces ==="
for wl in "${WORKLOADS[@]}"; do
    echo "  -> $wl"
    python3 "$SCRIPTS/gen_trace.py" --config "$CFG/${wl}.yaml"
done

# ── Phase 2: Generate datasets ──────────────────────────────────────
echo ""
echo "=== Phase 2: Creating datasets ==="
for wl in "${WORKLOADS[@]}"; do
    echo "  -> $wl"
    python3 "$SCRIPTS/gen_dataset.py" \
        --config "$CFG/${wl}.yaml" \
        --outdir "$DATA_ROOT/$wl"
done

# ── Phase 3: Run benchmarks ─────────────────────────────────────────
echo ""
echo "=== Phase 3: Running benchmarks (N_RUNS=$N_RUNS) ==="

for wl in "${WORKLOADS[@]}"; do
    TRACE="$ROOT/traces/${wl}_trace.bin"
    MANIFEST="$DATA_ROOT/$wl/manifest.txt"
    MEAN=$(get_yaml "$wl" compute preprocess_mean_ms)
    STD=$(get_yaml  "$wl" compute preprocess_std_ms)
    DIST=$(get_yaml "$wl" compute distribution)
    QD=$(get_yaml   "$wl" benchmark queue_depth)

    echo ""
    echo "── Workload: $wl  (QD=$QD, compute=${MEAN}±${STD}ms $DIST) ──"

    # ── Non-scissor engines ────────────────────────────────────────
    for engine in "${BASE_ENGINES[@]}"; do
        for run in $(seq 1 "$N_RUNS"); do
            OUT="$RESULTS/${wl}_${engine}_r${run}.json"
            echo "  -> $wl / $engine  run $run/$N_RUNS"
            drop_caches "$wl/$engine/run$run"
            "$BENCH" \
                --trace           "$TRACE"    \
                --manifest        "$MANIFEST" \
                --engine          "$engine"   \
                --queue-depth     "$QD"       \
                --compute-mean-ms "$MEAN"     \
                --compute-std-ms  "$STD"      \
                --compute-dist    "$DIST"     \
                --o-direct                    \
                --report-json     "$OUT"      \
            || echo "  [WARN] $engine run $run failed (may not be available)"
        done
    done

    # ── Scissor: Regressor-chosen gap (gauto) ─────────────────────
    for run in $(seq 1 "$N_RUNS"); do
        OUT="$RESULTS/${wl}_scissor_gauto_r${run}.json"
        echo "  -> $wl / scissor (gap=auto)  run $run/$N_RUNS"
        drop_caches "$wl/scissor-auto/run$run"
        "$BENCH" \
            --trace           "$TRACE"    \
            --manifest        "$MANIFEST" \
            --engine          scissor     \
            --queue-depth     "$QD"       \
            --compute-mean-ms "$MEAN"     \
            --compute-std-ms  "$STD"      \
            --compute-dist    "$DIST"     \
            --o-direct                    \
            --report-json     "$OUT"      \
        || echo "  [WARN] scissor (auto) run $run failed"
    done

    # ── Scissor: forced gap sweep ──────────────────────────────────
    for gap in "${SCISSOR_GAPS[@]}"; do
        for run in $(seq 1 "$N_RUNS"); do
            OUT="$RESULTS/${wl}_scissor_g${gap}_r${run}.json"
            echo "  -> $wl / scissor (gap=$gap)  run $run/$N_RUNS"
            drop_caches "$wl/scissor-g${gap}/run$run"
            "$BENCH" \
                --trace           "$TRACE"    \
                --manifest        "$MANIFEST" \
                --engine          scissor     \
                --queue-depth     "$QD"       \
                --compute-mean-ms "$MEAN"     \
                --compute-std-ms  "$STD"      \
                --compute-dist    "$DIST"     \
                --force-gap       "$gap"      \
                --o-direct                    \
                --report-json     "$OUT"      \
            || echo "  [WARN] scissor (gap=$gap) run $run failed"
        done
    done
done

echo ""
echo "=== All benchmarks complete. Results in $RESULTS/ ==="
echo "  $(ls "$RESULTS"/*.json 2>/dev/null | wc -l) JSON files written"
ls -lh "$RESULTS/"
