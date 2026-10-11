

export NAVSIM_EXP_ROOT="/root/navsim_workspace/exps"    

export NAVSIM_DEVKIT_ROOT="/root/recogdrive"
export OPENSCENE_DATA_ROOT="/root/navsim_workspace/dataset"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"

# python $NAVSIM_DEVKIT_ROOT/scripts/filter_high_risk_scenes/filter_high_risk_and_get_normal.py
python $NAVSIM_DEVKIT_ROOT/scripts/filter_high_risk_scenes/stats_high_risk_token_split.py