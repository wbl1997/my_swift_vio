#include <glog/logging.h>
#include <gtest/gtest.h>

/// Run all the tests that were declared with TEST()
int main(int argc, char **argv) {
  // init test before parse gflags so that args
  // like --gtest_filter can be stripped
  testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  // write LOG(INFO) to stderr
  FLAGS_stderrthreshold = 0; // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
  FLAGS_colorlogtostderr = 1;

  return RUN_ALL_TESTS();
}
