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

namespace okvis {

bool setInputParameters(okvis::InputData *input);

} // namespace okvis
