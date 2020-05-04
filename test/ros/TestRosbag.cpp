#include "rosbag/bag.h"
#include "gtest/gtest.h"
// Build this test with address sanitizer, add cxx_flags "-fsanitize=address -fno-omit-frame-pointer"
// "-O1 -g" are optional.
// For verbose output, before running, "export ASAN_OPTIONS=fast_unwind_on_malloc=0"
TEST(Rosbag, ClassLoader) {
  std::string bagname = "path/to/ros.bag";
  rosbag::Bag bag(bagname, rosbag::bagmode::Read);
}
