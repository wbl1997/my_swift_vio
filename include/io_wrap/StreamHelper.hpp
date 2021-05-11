#ifndef INCLUDE_IO_WRAP_STREAM_HELPER_HPP_
#define INCLUDE_IO_WRAP_STREAM_HELPER_HPP_
#include <string>
#include <sstream>
#include <vector>

namespace swift_vio {
enum DUMP_RESULT_OPTION {
  FULL_STATE = 0,
  FULL_STATE_WITH_EXTRINSICS,
  FULL_STATE_WITH_ALL_CALIBRATION
};

class StreamHelper {
 public:
  static void composeHeaderLine(
      const std::string &imu_model,
      const std::vector<std::string> &cam_extrinsic_opt_rep,
      const std::vector<std::string> &cam_proj_opt_rep,
      const std::vector<std::string> &cam_distortion_rep,
      DUMP_RESULT_OPTION result_option,
      std::string *header_line,
      bool include_frameid=true);

  static void composeHeaderLine(
      const std::string &imu_model,
      const std::vector<int> &cam_extrinsic_opt_mode,
      const std::vector<int> &cam_proj_opt_mode,
      const std::vector<std::string> &cam_distortion_rep,
      DUMP_RESULT_OPTION result_option, std::string *header_line,
      bool include_frameid = true);
};

std::vector<std::string> parseCommaSeparatedTopics(
    const std::string& topic_list);

std::string removeTrailingSlash(const std::string& path);
}  // namespace swift_vio
#endif  // INCLUDE_IO_WRAP_STREAM_HELPER_HPP_
