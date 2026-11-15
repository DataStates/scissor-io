#!/usr/bin/env bash
#PBS -l select=1
#PBS -l place=scatter:exclhost
#PBS -l walltime=04:00:00
#PBS -o read_bench.out
#PBS -e read_bench.err

# ========================================================================
# read_bench.sh -- io_uring read throughput of two checkpoint files.
#
# Runs bench_load (state-diff scripts/bench_load_calls.cpp): the same
# offsets of the first checkpoint of run1 and run2 are read through one
# ring (--interleaved), N IOPs of size S per file.  Each row logs the IOP
# count, IOP size, bytes per file, load time and the throughput of both
# files.
#
# SWEEP=io     total size of 1 and 2 GiB per file split into 1 to 256K
#              IOPs, contiguous and random offsets:
#              {contiguous,random}_io_{1,2}_<TYPE>.csv
# SWEEP=calib  I/O model calibration points, random offsets, total sizes
#              of 16 MiB to 4 GiB per file split into IOPs of 4 KiB or more
#              (at most 2 GiB): random_chks_io__<TYPE>.csv
# EXP_ID=np1000-01bil
# TYPE=pfs | ssd | hyb (run1 on the PFS, run2 on the node-local SSD)
# NREPS=5      repetitions per point
# IO_SIZES, MAX_IOPS, CALIB_SIZES override the swept sizes (see below)
#
# qsub -A <account> -q <queue> -l filesystems=<filesystems> \
#      -v HACC_DIR=<dir>,STATEDIFF_BUILD=<dir>,SWEEP=io,EXP_ID=np1000-01bil,TYPE=pfs read_bench.sh
# ========================================================================

# PBS runs a copy of this script: common.sh is found in the submission
# directory (ae/hacc/scripts) or in SCRIPT_DIR
. "${SCRIPT_DIR:-${PBS_O_WORKDIR:-$(dirname "${BASH_SOURCE[0]}")}}/common.sh"

SWEEP=${SWEEP:-io}
EXP_ID=${EXP_ID:-np1000-01bil}
TYPE=${TYPE:-pfs}
NREPS=${NREPS:-5}
GiB=$(( 1024 * MB ))
# SWEEP=io: total sizes per file in GiB and largest IOP count
IO_SIZES=${IO_SIZES:-"1 2"}
MAX_IOPS=${MAX_IOPS:-262144}
# SWEEP=calib: total sizes per file in bytes
CALIB_SIZES=${CALIB_SIZES:-"$((16 * MB)) $((32 * MB)) $((64 * MB)) $((128 * MB)) $((256 * MB)) $((512 * MB)) $GiB $((2 * GiB)) $((3 * GiB)) $((4 * GiB))"}
MIN_IOP_SIZE=4096
MAX_IOP_SIZE=$(( 2 * GiB ))

start_time=$SECONDS

echo "------------------------------------------------------------------------"
echo " Environment Setup "
echo "------------------------------------------------------------------------"
case $TYPE in
    pfs) set_run_dirs $EXP_ID offline ;;
    ssd) set_run_dirs $EXP_ID ssd ;;
    hyb) set_run_dirs $EXP_ID ssd
         run0_dir="${HACC_DIR}/$EXP_ID/run1"
         DISABLE_CACHE="$vmtouch -eq ${run0_dir} && $vmtouch -eq ${run1_dir}" ;;
    *)   echo "TYPE must be pfs, ssd or hyb" >&2; exit 1 ;;
esac

bench_load="${STATEDIFF_BUILD}/scripts/bench_load"
file0=$(ls ${run0_dir}/m000p.mpirestart-combined-*.dat | head -n 1)
file1=$(ls ${run1_dir}/m000p.mpirestart-combined-*.dat | head -n 1)
echo "Files: $file0 $file1"

outdir="$OUT_DIR/motivation/benchmark_data"
mkdir -p $outdir

# read_point <IOP size> <IOP count> <log file> <extra flags>
read_point() {
    for rep in $(seq 1 $NREPS); do
        run_cmd "$bench_load $file0 $file1 $1 $2 $3 --interleaved $4"
    done
}

if [[ $SWEEP == "io" ]]; then
    for size in $IO_SIZES; do
        for offsets in contiguous random; do
            flags=""
            [[ $offsets == "random" ]] && flags="--rand"
            logfile="$outdir/${offsets}_io_${size}_${TYPE}.csv"
            for (( iops = 1; iops <= MAX_IOPS; iops *= 2 )); do
                read_point $(( size * GiB / iops )) $iops $logfile "$flags"
            done
        done
    done
elif [[ $SWEEP == "calib" ]]; then
    logfile="$outdir/random_chks_io__${TYPE}.csv"
    for size in $CALIB_SIZES; do
        for (( iops = 1; size / iops >= MIN_IOP_SIZE; iops *= 2 )); do
            (( size / iops > MAX_IOP_SIZE )) && continue
            read_point $(( size / iops )) $iops $logfile "--rand"
        done
    done
else
    echo "SWEEP must be io or calib" >&2; exit 1
fi

elapsed_time=$(( SECONDS - start_time ))
echo "Script executed in $elapsed_time seconds."
