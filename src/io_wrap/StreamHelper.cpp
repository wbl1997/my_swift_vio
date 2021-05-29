#include "io_wrap/StreamHelper.hpp"
#include "swift_vio/CameraRig.hpp"
#include "swift_vio/ExtrinsicModels.hpp"
#include "swift_vio/imu/ImuModels.hpp"
#include "swift_vio/ProjParamOptModels.hpp"

DEFINE_string(datafile_separator, ",",
              "the separator used for a ASCII output file");

namespace swift_vio {
std::vector<std::string> parseCommaSeparatedTopics(const std::string& topic_list) {
  std::vector<std::string> topics;
  std::stringstream ss(topic_list);
  std::string topic;
  while (getline(ss, topic, ',')) {
      topics.push_back(topic);
  }
  return topics;
}

// A better implementation is given here.
// https://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring
std::string removeTrailingSlash(const std::string &path) {
  std::string subpath(path);
  std::size_t slash_index = subpath.find_last_of("/\\");
  while (slash_index != std::string::npos &&
         slash_index == subpath.length() - 1) {
    subpath = subpath.substr(0, slash_index);
    slash_index = subpath.find_last_of("/\\");
  }
  return subpath;
}
}  // namespace swift_vio
