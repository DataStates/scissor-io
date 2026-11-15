#!/usr/bin/env bash
# ========================================================================
# common.sh -- settings and helpers shared by the HACC job scripts.
#
# Sourced by e2e.sh, breakdown.sh, qdepth.sh and scalability.sh.  All
# site-specific values come from the environment:
#
#   HACC_DIR         (required) directory holding one sub-directory per
#                    simulation (e.g. np1000-01bil), each with run1/ and
#                    run2/ containing m000p.mpirestart-combined-*.dat
#   STATEDIFF_BUILD  (required) state-diff build directory holding
#                    scripts/eval_statediff
#   OUT_DIR          directory for the CSV logs (default: ./logs). The
#                    layout matches ae/hacc/data
#   VMTOUCH_DIR      directory holding the vmtouch binary used to evict
#                    the checkpoints from the page cache (default: PATH)
#   SCRATCH_DIR      node-local directory used when TYPE=ssd
#   MODULES_SCRIPT   optional site script that loads compiler/MPI modules
#   LAUNCH           MPI launch prefix (default: mpiexec with -n, --ppn,
#                    -d and --cpu-bind depth built from the values below)
#   NPROCS, PROCS_PER_NODE, CORES_PER_RANK, THREADS_PER_RANK, NITERS

if [[ -n "${MODULES_SCRIPT:-}" ]]; then
    . "$MODULES_SCRIPT"
fi

if [[ -n "${PBS_O_WORKDIR:-}" ]]; then
    cd ${PBS_O_WORKDIR}
fi

: "${HACC_DIR:?set HACC_DIR to the directory holding the HACC simulations}"
: "${STATEDIFF_BUILD:?set STATEDIFF_BUILD to the state-diff build directory}"
OUT_DIR=${OUT_DIR:-logs}

nprocs=${NPROCS:-8}
procs_per_node=${PROCS_PER_NODE:-4}
cores_per_rank=${CORES_PER_RANK:-8}
threads_per_rank=${THREADS_PER_RANK:-8}
niters=${NITERS:-3}

export OMP_NUM_THREADS=$threads_per_rank
export OMP_PROC_BIND=close
export OMP_PLACES=cores

eval_statediff="${STATEDIFF_BUILD}/scripts/eval_statediff"
vmtouch="${VMTOUCH_DIR:+$VMTOUCH_DIR/}vmtouch"

cmd_env=${LAUNCH:-"mpiexec -n $nprocs --ppn $procs_per_node -d $cores_per_rank --cpu-bind depth"}
kokkos_flags="--kokkos-num-threads=$threads_per_rank --kokkos-map-device-id-by=mpi_rank"

MB=$(( 1024 * 1024 ))
create_read_size=128
SYNC="sync"

stage_to_local_scratch() {
    local src="$1"
    local dst="$2"
    $cmd_env \
        bash -c '
            if [ "$PMI_LOCAL_RANK" -eq 0 ]; then
                mkdir -p '"$dst"'
                cp -a '"$src"'/. '"$dst"'
            fi
            # simple barrier to wait for local copy
            sleep 1
        '
}

# set_run_dirs <exp_id> <offline|ssd>: sets run0_dir, run1_dir and floc
set_run_dirs() {
    local exp_id="$1"
    local type="$2"
    if [[ $type == "ssd" ]]; then
        : "${SCRATCH_DIR:?set SCRATCH_DIR for TYPE=ssd}"
        run0_dir="${SCRATCH_DIR}/$exp_id/run1"
        run1_dir="${SCRATCH_DIR}/$exp_id/run2"
        stage_to_local_scratch "$HACC_DIR/$exp_id/run1" $run0_dir
        stage_to_local_scratch "$HACC_DIR/$exp_id/run2" $run1_dir
        floc=1
    else
        run0_dir="${HACC_DIR}/$exp_id/run1"
        run1_dir="${HACC_DIR}/$exp_id/run2"
        floc=0
    fi
    DISABLE_CACHE="$vmtouch -eq ${run0_dir} && $vmtouch -eq ${run1_dir}"
}

# run_cmd <command>: evicts the checkpoints from the page cache, runs the
# command and syncs
run_cmd() {
    local cmd="$1"
    set -x
    echo ${DISABLE_CACHE}
    echo ${cmd}
    echo ${SYNC}

    eval ${DISABLE_CACHE}
    eval ${cmd}
    eval ${SYNC}
    sleep 1
    set +x
}

# create_trees <chunk_size> <tol> <logname> <extra flags>: builds the
# state-diff trees of both runs (files *.compare-tree next to the data)
create_trees() {
    local chunk_size="$1"
    local tol="$2"
    local logname="$3"
    local flags="$4"
    echo "------------------------------------------------------------------------"
    echo " Method compare-tree : Prepare Checkpoints (Read size = $create_read_size)"
    echo "------------------------------------------------------------------------"
    rm -f ${run0_dir}/*.compare-tree ${run1_dir}/*.compare-tree
    run_cmd "$cmd_env $eval_statediff -c $chunk_size -b $(($create_read_size * $MB)) --error $tol --run0 ${run0_full_files[*]} ${run1_full_files[*]} -r $logname $flags"
    run0_files=( $(ls ${run0_dir}/*.compare-tree) )
    run1_files=( $(ls ${run1_dir}/*.compare-tree) )
    echo "Run 0 tree files: ${run0_files[@]}"
    echo "Run 1 tree files: ${run1_files[@]}"
    data_files="--run0-full ${run0_full_files[@]} --run1-full ${run1_full_files[@]}"
    tree_files="--run0 ${run0_files[@]} --run1 ${run1_files[@]}"
}

# engine_dir <scissor|iouring|mmap>: name of the log directory of an engine
engine_dir() {
    case "$1" in
        scissor) echo "DiffIO" ;;
        iouring) echo "IOUring" ;;
        mmap)    echo "MMap" ;;
    esac
}
