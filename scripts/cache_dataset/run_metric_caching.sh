source ~/.bashrc
conda activate nd

TRAIN_TEST_SPLIT=navtest

export NAVSIM_DEVKIT_ROOT="/root/recogdrive/"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export OPENSCENE_DATA_ROOT="/root/navsim_workspace/dataset"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"

export NAVSIM_EXP_ROOT="/root/navsim_workspace/exps"
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
cache.cache_path=$CACHE_PATH