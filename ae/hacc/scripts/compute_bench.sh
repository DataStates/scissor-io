#!/usr/bin/env bash
#PBS -l select=1
#PBS -l place=scatter:exclhost
#PBS -l walltime=01:00:00
#PBS -o compute_bench.out
#PBS -e compute_bench.err

# ========================================================================
# compute_bench.sh -- compute-kernel timings for the compute model.
#
# Runs bench_compute (state-diff scripts/bench_compare_kernel.cpp): the
# element-wise comparison of two buffers over W chunks of C bytes with
# Kokkos, for W from the thread count to MAX_CHUNKS and C from 128 B to
# 128 KiB.  Each row logs the thread count, the work size W x C and the
# compute time: compute_benchmark.csv.
#
# THREADS="8 16"   thread counts
# NREPS=3          runs of the whole grid per thread count
# MAX_CHUNKS=65536 largest W (the two buffers take 2 x 65536 x 128 KiB =
#                  16 GiB of memory)
#
# qsub -A <account> -q <queue> -l filesystems=<filesystems> \
#      -v STATEDIFF_BUILD=<dir> compute_bench.sh
# ========================================================================

# PBS runs a copy of this script: common.sh is found in the submission
# directory (ae/hacc/scripts) or in SCRIPT_DIR
. "${SCRIPT_DIR:-${PBS_O_WORKDIR:-$(dirname "${BASH_SOURCE[0]}")}}/common.sh"

THREADS=${THREADS:-"8 16"}
NREPS=${NREPS:-3}
MAX_CHUNKS=${MAX_CHUNKS:-65536}

start_time=$SECONDS

bench_compute="${STATEDIFF_BUILD}/scripts/bench_compute"
outdir="$OUT_DIR/motivation/benchmark_data"
mkdir -p $outdir
for threads in $THREADS; do
    export OMP_NUM_THREADS=$threads
    for rep in $(seq 1 $NREPS); do
        run_cmd "$bench_compute $outdir/compute_benchmark.csv $MAX_CHUNKS --kokkos-num-threads=$threads"
    done
done

elapsed_time=$(( SECONDS - start_time ))
echo "Script executed in $elapsed_time seconds."
