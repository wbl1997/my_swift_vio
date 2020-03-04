#!/usr/bin/env bash
MSCKF_OUTPUT_DIR=$OUTPUT_DIR/$MARS_DATA/$ALGO_NAME
mkdir -p $MSCKF_OUTPUT_DIR

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "$SCRIPT_DIR/sed_msckf_config.sh"

cd $MSCKF_WS
LAUNCH_FILE="$MSCKF_WS/src/msckf/launch/okvis_node_rosbag.launch"
source devel/setup.bash
cmd="roslaunch msckf $LAUNCH_FILE \
    config_filename:=$MSCKF_TEMPLATE \
    output_dir:=$MSCKF_OUTPUT_DIR/ \
    image_topic:="cam0/image_raw" \
    bag_file:=$BAG_FILE \
    image_noise_cov_multiplier:=$IMAGE_NOISE_FACTOR \
    start_into_bag:=$START_INTO_SESSION \
    play_rate:=$PLAY_RATE"
echo "$cmd"
$cmd

source "$SCRIPT_DIR/plot_msckf.sh"