#include "gmock/gmock.h"

#include "TimeAndRotationCalibrator.h"

using namespace testing;

namespace sensor_calib::tests {
TEST(TimeAndRotationCalibrator, ImuCamera) {
  sensor_calib::TimeAndRotationCalibrator example{};
  ASSERT_THAT(example.getValue(), Eq(99));
}
TEST(TimeAndRotationCalibrator, ImuImu) {}
} // namespace sensor_calib::tests
