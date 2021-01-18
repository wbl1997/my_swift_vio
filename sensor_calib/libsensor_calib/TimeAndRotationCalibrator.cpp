#include "TimeAndRotationCalibrator.h"

namespace sensor_calib {
TimeAndRotationCalibrator::CalibrationStatus TimeAndRotationCalibrator::calibrate() {
  return CalibrationStatus::InsufficientData;
}
} // namespace sensor_calib
