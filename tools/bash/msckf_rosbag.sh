#!/usr/bin/env bash
MSCKF_OUTPUT_DIR=$OUTPUT_DIR/$MARS_DATA/$ALGO_NAME
mkdir -p $MSCKF_OUTPUT_DIR

# https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "$SCRIPT_DIR/sed_msckf_config.sh"

cmd="$MSCKF_WS/devel/lib/msckf/okvis_node_synchronous \
 $MSCKF_TEMPLATE \
 --output_dir=$MSCKF_OUTPUT_DIR \
 --skip_first_seconds=$START_INTO_SESSION  \
 --max_inc_tol=10.0     --dump_output_option=$DUMP_OUTPUT_OPTION \
 --feature_tracking_method=0   \
 --epipolar_sigma_keypoint_size=$SIGMA_KEYPOINT_SIZE   \
 --two_view_obs_seq_type=$TWO_VIEW_OBS_SEQ_TYPE \
 --load_input_option=1   \
 --bagname=$BAG_FILE

echo "$cmd"
$cmd

source "$SCRIPT_DIR/plot_msckf.sh"