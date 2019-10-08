#ifndef INCLUDE_IO_WRAP_STREAM_HELPER_HPP_
#define INCLUDE_IO_WRAP_STREAM_HELPER_HPP_
#include <string>
#include <sstream>

namespace okvis {
enum DUMP_RESULT_OPTION {
  FULL_STATE = 0,
  FULL_STATE_WITH_EXTRINSICS,
  FULL_STATE_WITH_ALL_CALIBRATION
};

class StreamHelper {
 public:
  static void composeHeaderLine(
      const std::string &imu_model, const std::string &cam0_proj_opt_rep,
      const std::string &cam0_extrinsic_opt_rep,
      const std::string &cam0_distortion_rep,
      DUMP_RESULT_OPTION result_option,
      std::string *header_line);
  static void composeHeaderLine(
      const std::string &imu_model, const int &cam0_proj_opt_mode,
      const int &cam0_extrinsic_opt_mode,
      const std::string &cam0_distortion_rep,
      DUMP_RESULT_OPTION result_option,
      std::string *header_line);
};
}  // namespace okvis
#endif  // INCLUDE_IO_WRAP_STREAM_HELPER_HPP_
