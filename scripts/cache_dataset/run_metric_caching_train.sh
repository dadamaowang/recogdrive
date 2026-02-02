TRAIN_TEST_SPLIT=navtrain

export NAVSIM_DEVKIT_ROOT="/root/recogdrive/"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export OPENSCENE_DATA_ROOT="/root/autodl-tmp/Dataset/navtrain_tiny"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"

export NAVSIM_EXP_ROOT="/root/autodl-tmp/exps/nv"
CACHE_PATH=$NAVSIM_EXP_ROOT/negdrive_mc_train

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
cache.cache_path=$CACHE_PATH