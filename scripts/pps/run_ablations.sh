#!/bin/bash
# PPS ablations for the paper. Runs sequentially; each run uses all GPUs.
#   lambda = 0             -> MARS (outcome-only) baseline under identical rollouts
#   lambda sweep           {0.5, 1, 2}
#   negative_only off      (signed shaping: also reward risk-reducing segments)
#   probe all rollouts     (vs only failures)
#   segmentation           markers vs evenly spaced sentence ends; P in {2, 4, 6}
set -e
cd "$(dirname "$0")"

run() { echo "==== $* ===="; env "$@" bash train_pps.sh; }

run PPS_LAMBDA=0.0 EXP_NAME=pps_abl_l0_mars
for L in 0.5 1.0 2.0; do
    run PPS_LAMBDA=$L EXP_NAME=pps_abl_l${L}
done
run PPS_NEG_ONLY=false EXP_NAME=pps_abl_signed
run PPS_PROBE=all EXP_NAME=pps_abl_probe_all
run PPS_MARKERS=false EXP_NAME=pps_abl_nomarkers
for P in 2 6; do
    run PPS_SEGMENTS=$P EXP_NAME=pps_abl_P${P}
done
