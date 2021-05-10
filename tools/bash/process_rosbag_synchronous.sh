#!/usr/bin/env bash
VIO_OUTPUT_DIR=$OUTPUT_DIR/$MARS_DATA/$ALGO_NAME
mkdir -p $VIO_OUTPUT_DIR

# https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "$SCRIPT_DIR/sed_config_benchmark.sh"
source "$SCRIPT_DIR/sed_config_cam_imu.sh"

cmd="$SWIFT_VIO_WS/devel/lib/swift_vio/swift_vio_node_synchronous \
 $SWIFT_VIO_TEMPLATE \
 --output_dir=$VIO_OUTPUT_DIR \
 --skip_first_seconds=$START_INTO_SESSION  \
 --max_inc_tol=10.0     --dump_output_option=$DUMP_OUTPUT_OPTION \
 --epipolar_sigma_keypoint_size=$SIGMA_KEYPOINT_SIZE   \
 --two_view_obs_seq_type=$TWO_VIEW_OBS_SEQ_TYPE \
 --load_input_option=1   \
 --bagname=$BAG_FILE

echo "$cmd"
$cmd

source "$SCRIPT_DIR/plot_vio_result.sh"