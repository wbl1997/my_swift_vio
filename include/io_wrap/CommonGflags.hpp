#ifndef IO_WRAP_COMMON_GFLAGS_HPP_
#define IO_WRAP_COMMON_GFLAGS_HPP_
#include <okvis/Parameters.hpp>

DECLARE_int32(dump_output_option);

DECLARE_int32(load_input_option);

DECLARE_string(output_dir);

DECLARE_string(image_folder);

DECLARE_string(video_file);

DECLARE_string(time_file);

DECLARE_string(imu_file);

DECLARE_int32(start_index);

DECLARE_int32(finish_index);

DECLARE_double(skip_first_seconds);

DECLARE_string(bagname);

DECLARE_string(camera_topics);

DECLARE_string(imu_topic);

DECLARE_string(vocabulary_path);

DECLARE_string(lcd_params_yaml);

namespace swift_vio {

bool setInputParameters(InputData *input);

}  // namespace swift_vio
#endif  // IO_WRAP_COMMON_GFLAGS_HPP_
