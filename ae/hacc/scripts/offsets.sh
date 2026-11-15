#!/usr/bin/env bash
#PBS -l select=2
#PBS -l place=scatter:exclhost
#PBS -l walltime=02:00:00
#PBS -o offsets.out
#PBS -e offsets.err

# ========================================================================
# offsets.sh -- chunk indices flagged by the state-diff tree comparison.
#
# For each chunk size and error tolerance, builds the trees of both runs
# and compares them once with --dump-offsets: one file per checkpoint
# pair, <run1 checkpoint file name>_<chunk size>_<error>.txt (e.g.
# m000p.mpirestart-combined-0-10.dat_4096_1e-7.txt), holding the flagged
# chunk indices in traversal order (the offsets the direct comparison
# reads).  The compare logs go to <TYPE>.<rank>.compare.csv next to them.
#
# EXP_ID=np1000-01bil
# TYPE=offline | ssd
#
# qsub -A <account> -q <queue> -l filesystems=<filesystems> \
#      -v HACC_DIR=<dir>,STATEDIFF_BUILD=<dir>,EXP_ID=np1000-01bil,TYPE=offline offsets.sh
# ========================================================================

# PBS runs a copy of this script: common.sh is found in the submission
# directory (ae/hacc/scripts) or in SCRIPT_DIR
. "${SCRIPT_DIR:-${PBS_O_WORKDIR:-$(dirname "${BASH_SOURCE[0]}")}}/common.sh"

EXP_ID=${EXP_ID:-np1000-01bil}
TYPE=${TYPE:-offline}

start_time=$SECONDS

echo "------------------------------------------------------------------------"
echo " Environment Setup "
echo "------------------------------------------------------------------------"
set_run_dirs $EXP_ID $TYPE

error_tols=(0.0000001 0.00001 0.001)
chunk_sizes=(4096 8192 16384 32768)

# offsets/<simulation name without the particle count>, e.g. np1000
outdir="$OUT_DIR/motivation/offsets/${EXP_ID%%-*}"
outname="$outdir/$TYPE"
mkdir -p $outdir
cmd_flags="--level 13 -t float -s $floc $kokkos_flags"

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
    echo " Chunk size $chunk_size"
    echo " Error tolerance $tol"
    echo "========================================================================"
    create_trees $chunk_size $tol $outname "$cmd_flags"
    run_cmd "$cmd_env $eval_statediff -c $chunk_size --error $tol $data_files $tree_files -r $outname --dump-offsets $outdir $cmd_flags"
  done
done
rm -f ${run0_dir}/*.compare-tree ${run1_dir}/*.compare-tree

elapsed_time=$(( SECONDS - start_time ))
echo "Script executed in $elapsed_time seconds."
