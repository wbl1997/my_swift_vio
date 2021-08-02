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

std::vector<std::string> parseCommaSeparatedTopics(
    const std::string& topic_list);

std::string removeTrailingSlash(const std::string& path);
}  // namespace swift_vio
#endif  // INCLUDE_IO_WRAP_STREAM_HELPER_HPP_
