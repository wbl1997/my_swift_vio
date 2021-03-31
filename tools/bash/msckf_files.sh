#!/usr/bin/env bash

MSCKF_OUTPUT_DIR=$OUTPUT_DIR/$MARS_DATA/$ALGO_NAME
mkdir -p $MSCKF_OUTPUT_DIR
START_INDEX=$(( $START_INTO_SESSION*$CAMERA_RATE ))
FINISH_INDEX=$(( $SESSION_DURATION*$CAMERA_RATE ))

# https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "$SCRIPT_DIR/sed_msckf_config_benchmark.sh"
source "$SCRIPT_DIR/sed_msckf_config_cam_imu.sh"

cmd="$MSCKF_WS/devel/lib/msckf/okvis_node_synchronous \
 $MSCKF_TEMPLATE \
 --output_dir=$MSCKF_OUTPUT_DIR \
 --start_index=$START_INDEX     --finish_index=$FINISH_INDEX  \
 --max_inc_tol=10.0     --dump_output_option=$DUMP_OUTPUT_OPTION \
 --epipolar_sigma_keypoint_size=$SIGMA_KEYPOINT_SIZE   \
 --two_view_obs_seq_type=$TWO_VIEW_OBS_SEQ_TYPE \
 --load_input_option=1   \
 --video_file=$DATA_DIR/$MARS_DATA/${VIDEO_FILE[$j]}  \
 --time_file=$DATA_DIR/$MARS_DATA/${TIME_FILE[$j]}  \
 --imu_file=$DATA_DIR/$MARS_DATA/$IMU_FILE"

echo "$cmd"
$cmd

source "$SCRIPT_DIR/plot_msckf.sh"
