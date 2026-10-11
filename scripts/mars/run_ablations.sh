#!/bin/bash
# MARS ablations for the paper. Runs sequentially; each run uses all GPUs.
#   K sweep:      K in {1, 2, 4, 8}   (K=1, CRN off, beta=0 ~ NSR with soft/indicator advantage)
#   CRN on/off at K=4
#   progress hinge on/off (beta 0 vs 0.5)
#   baseline none vs group_mean
set -e
cd "$(dirname "$0")"

run() { echo "==== $* ===="; env "$@" bash train_mars.sh; }

for K in 1 2 4 8; do
    run MARS_K=$K MARS_CRN=true MARS_PROGRESS_COEF=0.5 EXP_NAME=mars_abl_K${K}
done
run MARS_K=4 MARS_CRN=false EXP_NAME=mars_abl_K4_nocrn
run MARS_K=4 MARS_PROGRESS_COEF=0.0 EXP_NAME=mars_abl_K4_beta0
run MARS_K=4 MARS_BASELINE=group_mean EXP_NAME=mars_abl_K4_groupmean
