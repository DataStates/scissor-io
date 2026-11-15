#!/usr/bin/env bash
#PBS -l select=2
#PBS -l place=scatter:exclhost
#PBS -l walltime=07:00:00
#PBS -o e2e.out
#PBS -e e2e.err

# ========================================================================
# e2e.sh -- end-to-end throughput of SCISSOR, IOUring and MMap.
#
# ENGINE=scissor | iouring | mmap
# EXP_ID=np796-500mil | np1000-01bil
# TYPE=offline | ssd
# CSIZE=1 (4/8 KB) | 2 (16/32 KB) | 3 (4/8/16/32 KB)
#
# Account, queue and filesystems are given on the qsub command line, e.g.
# qsub -A <account> -q <queue> -l filesystems=<filesystems> \
#      -v HACC_DIR=<dir>,STATEDIFF_BUILD=<dir>,ENGINE=scissor,EXP_ID=np1000-01bil,TYPE=offline,CSIZE=3 e2e.sh
# ========================================================================

# PBS runs a copy of this script: common.sh is found in the submission
# directory (ae/hacc/scripts) or in SCRIPT_DIR
. "${SCRIPT_DIR:-${PBS_O_WORKDIR:-$(dirname "${BASH_SOURCE[0]}")}}/common.sh"

ENGINE=${ENGINE:-scissor}
EXP_ID=${EXP_ID:-np1000-01bil}
TYPE=${TYPE:-offline}
CSIZE=${CSIZE:-3}

start_time=$SECONDS

echo "------------------------------------------------------------------------"
echo " Environment Setup "
echo "------------------------------------------------------------------------"
set_run_dirs $EXP_ID $TYPE

error_tols=(0.0000001 0.00001 0.001)

if [[ $CSIZE == "1" ]]; then
    chunk_sizes=(4096 8192)
elif [[ $CSIZE == "2" ]]; then
    chunk_sizes=(16384 32768)
else
    chunk_sizes=(4096 8192 16384 32768)
fi

outdir="$OUT_DIR/evaluation/e2e/$(engine_dir $ENGINE)/$EXP_ID"
outname="$outdir/$TYPE"
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
        echo "------------------------------------------------------------------------"
        echo " ($iter) Bench compare-tree : Compare Checkpoints"
        echo "------------------------------------------------------------------------"
        run_cmd "$cmd_env $eval_statediff -c $chunk_size --error $tol $data_files $tree_files -r $outname $cmd_flags"
    done
  done
done
rm -f ${run0_dir}/*.compare-tree ${run1_dir}/*.compare-tree

elapsed_time=$(( SECONDS - start_time ))
echo "Script executed in $elapsed_time seconds."
