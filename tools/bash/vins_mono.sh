#!/usr/bin/env bash
VINS_OUTPUT_DIR=$OUTPUT_DIR/$MARS_DATA/$ALGO_NAME
mkdir -p $VINS_OUTPUT_DIR

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "$SCRIPT_DIR/sed_vins_config.sh"
VINS_PLOT_SCRIPT=$SCRIPT_DIR/../python/plotVinsMonoResult.py

cd $VINS_WS
source devel/setup.bash
cmd="roslaunch vins_estimator euroc.launch \
  config_path:=$VINS_TEMPLATE \
  bag_file:=$DATA_DIR/$MARS_DATA/movie.bag \
  bag_start:=$START_INTO_SESSION \
  bag_duration:=$SESSION_DURATION \
  rviz_file:=$VINS_RVIZ"
echo "Working in [$START_SEC $END_SEC] sec of $DATA_DIR/$MARS_DATA with vins mono"
echo "$cmd"
$cmd

python3 $VINS_PLOT_SCRIPT $VINS_OUTPUT_DIR