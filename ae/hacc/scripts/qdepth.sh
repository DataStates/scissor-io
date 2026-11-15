#!/usr/bin/env bash
#PBS -l select=2
#PBS -l place=scatter:exclhost
#PBS -l walltime=07:00:00
#PBS -o qdepth.out
#PBS -e qdepth.err

# ========================================================================
# qdepth.sh -- submission and completion queue depth sweep.
#
# ENGINE=scissor : submission queue depth x completion queue depth
# ENGINE=iouring : submission queue depth
# at error tolerance 1e-7 for 32 KB and 4 KB chunks.
#
# EXP_ID=np1000-01bil
# TYPE=offline | ssd
#
# qsub -A <account> -q <queue> -l filesystems=<filesystems> \
#      -v HACC_DIR=<dir>,STATEDIFF_BUILD=<dir>,ENGINE=scissor,EXP_ID=np1000-01bil,TYPE=offline qdepth.sh
# ========================================================================

# PBS runs a copy of this script: common.sh is found in the submission
# directory (ae/hacc/scripts) or in SCRIPT_DIR
. "${SCRIPT_DIR:-${PBS_O_WORKDIR:-$(dirname "${BASH_SOURCE[0]}")}}/common.sh"

ENGINE=${ENGINE:-scissor}
EXP_ID=${EXP_ID:-np1000-01bil}
TYPE=${TYPE:-offline}

start_time=$SECONDS

echo "------------------------------------------------------------------------"
echo " Environment Setup "
echo "------------------------------------------------------------------------"
set_run_dirs $EXP_ID $TYPE

error_tols=(0.0000001)
chunk_sizes=(32768 4096)
queue_depths=(128 256 512 1024 2048 4096 8192 16384 32768)
if [[ $ENGINE == "scissor" ]]; then
    cqueue_depths=(16 32 64 128 256 512 1024 2048 4096 8192)
    outdir="$OUT_DIR/evaluation/qdvar/diffio"
else
    cqueue_depths=(1024)
    outdir="$OUT_DIR/evaluation/qdvar/liburing"
fi

outname="$outdir/qdvar"
mkdir -p $outdir
cmd_flags="--level 13 -t float -s $floc --engine $ENGINE $kokkos_flags"

var_name=combined
run0_full_files=( $(ls ${run0_dir}/m000p.mpirestart-${var_name}-*.dat) )
run1_full_files=( $(ls ${run1_dir}/m000p.mpirestart-${var_name}-*.dat) )
echo "Run 0 data full files: ${run0_full_files[@]}"
echo "Run 1 data full files: ${run1_full_files[@]}"

for chunk_size in "${chunk_sizes[@]}";
do
  for tol in "${error_tols[@]}";
  do
    echo "========================================================================"
    echo " Data type Float"
    echo " Engine $ENGINE"
    echo " Chunk size $chunk_size"
    echo " Error tolerance $tol"
    echo "========================================================================"
    create_trees $chunk_size $tol $outname "$cmd_flags"
    for iter in $(seq 1 $niters)
    do
        for qdepth in "${queue_depths[@]}"; do
            for cqdepth in "${cqueue_depths[@]}"; do
                echo "------------------------------------------------------------------------"
                echo " ($iter) Compare Checkpoints (Queue depth = $qdepth, CQueue depth = $cqdepth)"
                echo "------------------------------------------------------------------------"
                run_cmd "$cmd_env $eval_statediff -c $chunk_size -q $qdepth -z $cqdepth --error $tol $data_files $tree_files -r $outname $cmd_flags"
            done
        done
    done
  done
done
rm -f ${run0_dir}/*.compare-tree ${run1_dir}/*.compare-tree

elapsed_time=$(( SECONDS - start_time ))
echo "Script executed in $elapsed_time seconds."
