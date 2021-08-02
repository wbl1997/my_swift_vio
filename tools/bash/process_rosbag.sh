#!/usr/bin/env bash
VIO_OUTPUT_DIR=$OUTPUT_DIR/$MARS_DATA/$ALGO_NAME
mkdir -p $VIO_OUTPUT_DIR

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "$SCRIPT_DIR/sed_config_benchmark.sh"
source "$SCRIPT_DIR/sed_config_cam_imu.sh"

cd $SWIFT_VIO_WS
LAUNCH_FILE="swift_vio_node_rosbag.launch"
source devel/setup.bash
cmd="roslaunch swift_vio $LAUNCH_FILE \
    config_filename:=$SWIFT_VIO_TEMPLATE \
    output_dir:=$VIO_OUTPUT_DIR/ \
    image_topic:="cam0/image_raw" \
    bag_file:=$BAG_FILE \
    image_noise_cov_multiplier:=$IMAGE_NOISE_FACTOR \
    start_into_bag:=$START_INTO_SESSION \
    play_rate:=$PLAY_RATE"
echo "$cmd"
$cmd

source "$SCRIPT_DIR/plot_vio_result.sh"