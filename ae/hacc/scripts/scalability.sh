#!/usr/bin/env bash
#PBS -l select=8
#PBS -l place=scatter:exclhost
#PBS -l walltime=00:45:00
#PBS -o scalability.out
#PBS -e scalability.err

# ========================================================================
# scalability.sh -- weak scaling of IOUring and SCISSOR, 2 ranks per node.
#
# NNODES        number of nodes. The checkpoints of simulation ranks 0 to
#               NNODES-1 (m000p.mpirestart-combined-<r>-*.dat) are spread
#               round-robin over the NNODES x PROCS_PER_NODE processes
# EXP_ID        simulation directory under HACC_DIR (default np2592-17bil)
#
# Request NNODES nodes with -l select, e.g.
# qsub -A <account> -q <queue> -l filesystems=<filesystems> -l select=8 \
#      -v HACC_DIR=<dir>,STATEDIFF_BUILD=<dir>,NNODES=8 scalability.sh
# ========================================================================

PROCS_PER_NODE=${PROCS_PER_NODE:-2}
NPROCS=$(( ${NNODES:?set NNODES} * PROCS_PER_NODE ))
# PBS runs a copy of this script: common.sh is found in the submission
# directory (ae/hacc/scripts) or in SCRIPT_DIR
. "${SCRIPT_DIR:-${PBS_O_WORKDIR:-$(dirname "${BASH_SOURCE[0]}")}}/common.sh"

EXP_ID=${EXP_ID:-np2592-17bil}

echo "------------------------------------------------------------------------"
echo " Environment Setup "
echo "------------------------------------------------------------------------"
echo "NUM_OF_NODES= ${NNODES} RANKS_PER_NODE= ${procs_per_node} THREADS_PER_RANK= ${threads_per_rank}"

start_time=$SECONDS

set_run_dirs $EXP_ID offline
tol=0.0000001
chunk_sizes=(32768 4096)
qdepth=32768

uring_outname="$OUT_DIR/scalability/IOUring/node-${NNODES}/scale-${NNODES}nodes"
diffio_outname="$OUT_DIR/scalability/DiffIO/node-${NNODES}/scale-${NNODES}nodes"
mkdir -p $(dirname $uring_outname) $(dirname $diffio_outname)

run0_full_files=()
run1_full_files=()

for ((n=0; n<NNODES; n++)); do
  run0_full_files+=( $(ls "$run0_dir"/m000p.mpirestart-combined-"$n"-*.dat 2>/dev/null | sort) )
  run1_full_files+=( $(ls "$run1_dir"/m000p.mpirestart-combined-"$n"-*.dat 2>/dev/null | sort) )
done

echo "Run 0 data full files: ${run0_full_files[@]}"
echo "Run 1 data full files: ${run1_full_files[@]}"

for chunk_size in "${chunk_sizes[@]}";
do
    echo "========================================================================"
    echo " Data type Float"
    echo " Chunk size $chunk_size"
    echo " Error tolerance $tol"
    echo "========================================================================"
    for iter in $(seq 1 $niters)
    do
        for engine in iouring scissor; do
            if [[ $engine == "iouring" ]]; then
                outname=$uring_outname
            else
                outname=$diffio_outname
            fi
            cmd_flags="--level 13 -t float -s 0 --engine $engine $kokkos_flags"
            echo "------------------------------------------------------------------------"
            echo " ($iter) Engine $engine"
            echo "------------------------------------------------------------------------"
            create_trees $chunk_size $tol $outname "$cmd_flags"
            run_cmd "$cmd_env $eval_statediff -c $chunk_size -q $qdepth --error $tol $data_files $tree_files -r $outname $cmd_flags"
        done
    done
done

rm -f ${run0_dir}/*.compare-tree ${run1_dir}/*.compare-tree
elapsed_time=$(( SECONDS - start_time ))
echo "Script executed in $elapsed_time seconds."
