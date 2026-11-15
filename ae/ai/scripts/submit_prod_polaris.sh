#!/bin/bash -l
# submit_prod_polaris.sh: PBS submission script for the full SCISSOR production
# benchmark on Polaris.
#
# Two benchmark modes (set PFS_BENCH):
#
#   PFS_BENCH=false: SSD mode
#     Data is rsynced from PFS → local NVMe SSD, then benchmarked from SSD.
#     Reflects compute-node IO throughput in isolation.
#     Expected result: io_uring wins, scissor_sr advantage not visible.
#
#   PFS_BENCH=true  (default): Lustre PFS mode
#     Data is read directly from the Lustre PFS (no rsync, no SSD).
#     Reflects real training IO under PFS latency (6 to 20 ms/RPC).
#     Expected result: scissor_sr outperforms io_uring 1.5 to 3× on ResNet
#     by coalescing adjacent records and cutting RPC count.
#     Note: vmtouch cannot evict Lustre page cache, so cache drop is best-effort.
#
# Three optional sweep modes (enable by setting the corresponding variable):
#
#   GAP_BATCH_SWEEP=true: fixed batch (32 MB), gaps {1,2,4,8} chunks
#     Isolates the effect of gap size with batch size held constant.
#
#   BATCH_SWEEP=true: fixed gap (1 chunk), batches {32,64,128,256} MB
#     Isolates the effect of batch size with gap held at 1 chunk.
#
#   GAP_SWEEP=true: original gap sweep: gaps {0,1,2,4,8,16}, used_chks=1
#     Kept for backwards compatibility with prior results.
#
# Cold-cache eviction (vmtouch or /proc/sys/vm/drop_caches) is applied
# before every individual run regardless of sweep mode.
#
# Phases 1 to 2.5 (data generation + trace extraction) are skipped when
# SKIP_PREPARE=true (data already on PFS from a previous job or login-node run).
#
# Submit from the ae/ai directory.  Account, queue and filesystems are passed
# on the qsub command line. Paths are forwarded with -v (PBS requires -v to
# forward variables, and prefix assignment alone is ignored):
#
#   PBS_ACCOUNT=<project>  PBS_QUEUE=<queue>  PBS_FS=<filesystems, e.g. home:<pfs>>
#   VARS="PFS_ROOT=/path/on/pfs/scissor_data,VENV=/path/to/venv"
#
#   qsub -A $PBS_ACCOUNT -q $PBS_QUEUE -l filesystems=$PBS_FS -v $VARS scripts/submit_prod_polaris.sh                       # baselines only (no sweeps)
#   qsub -A $PBS_ACCOUNT -q $PBS_QUEUE -l filesystems=$PBS_FS -v $VARS,GAP_SWEEP=true scripts/submit_prod_polaris.sh        # gap sweep {0,1,2,4,8,16}
#   qsub -A $PBS_ACCOUNT -q $PBS_QUEUE -l filesystems=$PBS_FS -v $VARS,GAP_BATCH_SWEEP=true scripts/submit_prod_polaris.sh  # fixed-batch gap sweep
#   qsub -A $PBS_ACCOUNT -q $PBS_QUEUE -l filesystems=$PBS_FS -v $VARS,BATCH_SWEEP=true scripts/submit_prod_polaris.sh      # fixed-gap batch sweep
#   qsub -A $PBS_ACCOUNT -q $PBS_QUEUE -l filesystems=$PBS_FS -v $VARS,GAP_BATCH_SWEEP=true,BATCH_SWEEP=true scripts/submit_prod_polaris.sh  # both controlled sweeps
#   qsub -A $PBS_ACCOUNT -q $PBS_QUEUE -l filesystems=$PBS_FS -v $VARS,PFS_BENCH=false,GAP_SWEEP=true scripts/submit_prod_polaris.sh       # SSD + gap sweep
#
# Variables forwarded with -v:
#   PFS_ROOT              (required) PFS directory for the dataset
#   TRACES_DIR            trace directory (default $PFS_ROOT/traces)
#   VENV                  Python virtualenv with requirements.txt + dlio_benchmark
#   MODULES_SCRIPT        optional site script that loads compiler/MPI modules
#   SSD_ROOT              node-local SSD directory (default /local/scratch/scissor_bench)
#   VMTOUCH_DIR           directory containing the vmtouch binary
#   SCISSOR_MODEL_CONFIG  optional fitted models.conf for the SCISSOR regressor
#   AI_DIR                ae/ai directory (default: the qsub submission directory)
#   N_RUNS, WORKLOADS, SKIP_PREPARE, PFS_BENCH, GAP_SWEEP, GAP_BATCH_SWEEP, BATCH_SWEEP
#
# Estimated walltime (SKIP_PREPARE=true, N_RUNS=2):
#   Base engines only              : ~1-2 h
#   + gap sweep                    : +~1 h
#   + gap-batch sweep + batch sweep: +~2 h
#   With SKIP_PREPARE=false        : add ~2-3 h for data generation
#
#PBS -N scissor-prod
#PBS -l select=1
#PBS -l walltime=1:00:00

# ── Top-level configuration (override with qsub -v) ───────────────────────────
if [[ -z "$PFS_ROOT" ]]; then
    echo "ERROR: PFS_ROOT is not set (qsub -v PFS_ROOT=...)" >&2
    exit 1
fi
TRACES_DIR=${TRACES_DIR:-${PFS_ROOT}/traces}               # trace files on the PFS
WORKLOADS=${WORKLOADS:-"bert_clustered resnet50_clustered"}
SKIP_PREPARE=${SKIP_PREPARE:-true}    # override with: qsub -v SKIP_PREPARE=false ...
PFS_BENCH=${PFS_BENCH:-true}         # override with: qsub -v PFS_BENCH=false ...
N_RUNS=${N_RUNS:-2}                   # repetitions per engine
VMTOUCH_DIR=${VMTOUCH_DIR:-}
SSD_ROOT=${SSD_ROOT:-/local/scratch/scissor_bench}
GAP_SWEEP=${GAP_SWEEP:-false}             # original gap sweep: gaps {0,1,2,4,8,16}, used_chks=1
GAP_BATCH_SWEEP=${GAP_BATCH_SWEEP:-false} # fixed batch (32 MB), gaps {1,2,4,8}
BATCH_SWEEP=${BATCH_SWEEP:-false}         # fixed gap (1 chunk), batches {32,64,128,256} MB
# ─────────────────────────────────────────────────────────────────────────────

NNODES=$(wc -l < "$PBS_NODEFILE")
NRANKS=4
NDEPTH=16
NTHREADS=16
NTOTRANKS=$(( NNODES * NRANKS ))

echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS} THREADS_PER_RANK= ${NTHREADS}"

# ── Environment ───────────────────────────────────────────────────────────────
[[ -n "$MODULES_SCRIPT" ]] && . "$MODULES_SCRIPT"
module load cray-python/3.11.7
[[ -n "$VENV" ]] && source "$VENV/bin/activate"

REPO=${AI_DIR:-$PBS_O_WORKDIR}
cd "$REPO"

[[ -n "$SCISSOR_MODEL_CONFIG" ]] && export SCISSOR_MODEL_CONFIG
export MPICH_GPU_SUPPORT_ENABLED=0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/lib

# ── Storage mode: SSD or PFS ─────────────────────────────────────────────────
if [[ "$PFS_BENCH" == true ]]; then
    # Read directly from Lustre, no rsync needed
    DATADIR="$PFS_ROOT"
    BENCH_COPY_ARG="--no-copy"
    echo "[mode] PFS benchmark: reading directly from Lustre ($PFS_ROOT)"
    echo "       Cache eviction on Lustre is best-effort (vmtouch targets local files only)."
else
    # Copy to local NVMe SSD first
    DATADIR="$SSD_ROOT"
    BENCH_COPY_ARG=""
    mkdir -p "$DATADIR"
    echo "[mode] SSD benchmark: data will be rsynced to $DATADIR"
fi

# ── Locate vmtouch ────────────────────────────────────────────────────────────
if [[ -n "$VMTOUCH_DIR" ]] && [[ -x "${VMTOUCH_DIR}/vmtouch" ]]; then
    VMTOUCH_ARG="--vmtouch-dir ${VMTOUCH_DIR}"
    echo "[cache] vmtouch found: ${VMTOUCH_DIR}/vmtouch"
else
    VMTOUCH_ARG=""
    echo "[cache] vmtouch not found at ${VMTOUCH_DIR}, will fall back to /proc/sys/vm/drop_caches"
fi

# ── Sanity checks ─────────────────────────────────────────────────────────────
BENCH_BIN="$REPO/src/bench"
if [[ ! -x "$BENCH_BIN" ]]; then
    echo "ERROR: bench binary not found or not executable: $BENCH_BIN" >&2
    echo "       Build it with: cd $REPO/src && make" >&2
    exit 1
fi

echo ""
echo "=== $(date) : scissor production benchmark ==="
echo "    repo        : $REPO"
echo "    pfs-root    : $PFS_ROOT"
echo "    data-root   : $DATADIR"
echo "    traces-dir  : $TRACES_DIR"
echo "    workloads   : $WORKLOADS"
echo "    pfs-bench   : $PFS_BENCH"
echo "    skip-prepare: $SKIP_PREPARE"
echo "    n_runs      : $N_RUNS"
echo "    gap-sweep   : $GAP_SWEEP"
echo "    gap-batch-sw: $GAP_BATCH_SWEEP"
echo "    batch-sweep : $BATCH_SWEEP"

GAP_SWEEP_ARG=""
[[ "$GAP_SWEEP"       == true ]] && GAP_SWEEP_ARG="--gap-sweep"
GAP_BATCH_SWEEP_ARG=""
[[ "$GAP_BATCH_SWEEP" == true ]] && GAP_BATCH_SWEEP_ARG="--gap-batch-sweep"
BATCH_SWEEP_ARG=""
[[ "$BATCH_SWEEP"     == true ]] && BATCH_SWEEP_ARG="--batch-sweep"

start=$(date +%s)

# ── Clean previous results, plots, and traces (dataset on PFS is preserved) ──
echo ""
echo "=== $(date) : Cleaning previous outputs ==="
rm -rf "$REPO/results/dlio" "$REPO/plots"
echo "    removed: results/dlio  plots"

# ── Phases 1-2.5: Generate dataset + extract + filter traces (on the PFS) ──
if [[ "$SKIP_PREPARE" == false ]]; then
    echo ""
    echo "=== $(date) : Phases 1-2.5: prepare (generate, trace, sample) ==="
    python3 scripts/prepare_dlio.py \
        --workloads $WORKLOADS \
        --data-root "$PFS_ROOT" \
        --traces-dir "$TRACES_DIR"
else
    echo ""
    echo "=== $(date) : Skipping Phases 1-2.5 (SKIP_PREPARE=true) ==="
fi

# ── Phases 3-4: Benchmark all engines, produce plots ─────────────────────────
echo ""
echo "=== $(date) : Phases 3-4: benchmark + plots ==="
python3 scripts/bench_dlio.py \
    --workloads   $WORKLOADS \
    --pfs-root    "$PFS_ROOT" \
    --ssd-root    "$DATADIR" \
    --traces-dir  "$TRACES_DIR" \
    --bench       "$BENCH_BIN" \
    --runs        "$N_RUNS" \
    ${GAP_SWEEP_ARG} \
    ${GAP_BATCH_SWEEP_ARG} \
    ${BATCH_SWEEP_ARG} \
    ${BENCH_COPY_ARG} \
    ${VMTOUCH_ARG}

end=$(date +%s)
echo ""
echo "=== $(date) : DONE. Elapsed: $(( end - start )) seconds ==="
