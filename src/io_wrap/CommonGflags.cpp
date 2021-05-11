#include <gflags/gflags.h>
#include <io_wrap/CommonGflags.hpp>

DEFINE_int32(
    dump_output_option, 3,
    "0, direct results to ROS publishers, other options "
    "ADDITIONALLY save results to a csv."
    "1, save states, 2, save states and camera extrinsics, "
    "3, save states, and all calibration parameters, 4, save states,"
    "all calibration parameters, feature tracks, and landmarks."
    "Warn: 0 or 1 does not pass T_BC so the visualized camera frustum"
    " may be in wrong orientation. Also 3 does not support the "
    "estimation method okvis");

DEFINE_int32(load_input_option, 1,
             "0, get input by subscribing to ros topics"
             "1, get input by reading files on a hard drive");

DEFINE_string(output_dir, "", "the directory to dump results");

DEFINE_string(image_folder, "", "folder of an input image sequence");

DEFINE_string(video_file, "", "full name of an input video file");

DEFINE_string(time_file, "",
              "full name of an input time file containing"
              " timestamps for each image seq. frame");

DEFINE_string(imu_file, "", "full name of an input IMU file");

DEFINE_int32(start_index, 0, "index of the first frame to be processed");

DEFINE_int32(finish_index, 0, "index of the last frame to be processed");

DEFINE_double(skip_first_seconds, 0, "Number of seconds to skip from the beginning!");

DEFINE_string(bagname, "", "Bag filename.");

DEFINE_string(camera_topics, "/cam0/image_raw,/cam1/image_raw",
              "Used image topics inside the bag. Should agree with the number "
              "of cameras in the config file");

DEFINE_string(imu_topic, "/imu0", "Imu topic inside the bag");

DEFINE_string(vocabulary_path,
              "vocabulary/ORBvoc.yml",
              "Path to BoW vocabulary file for LoopClosureDetector module.");

namespace swift_vio {
bool setInputParameters(InputData *input) {
  input->videoFile = FLAGS_video_file;
  input->imageFolder = FLAGS_image_folder;
  input->imuFile = FLAGS_imu_file;
  input->timeFile = FLAGS_time_file;
  input->startIndex = FLAGS_start_index;
  input->finishIndex = FLAGS_finish_index;
  return true;
}
} // namespace swift_vio
