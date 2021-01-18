#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "msckf/memory.h"
#include "okvis/ImuMeasurements.hpp"

namespace sensor_calib {

template <class Element>
int removeObsoleteData(Eigen::AlignedDeque<okvis::Measurement<Element>> *q,
                       double windowSize) {
  okvis::Time eraseUntil = q->back().timeStamp - okvis::Duration(windowSize);
  if (q->front().timeStamp > eraseUntil)
    return 0;

  auto eraseEnd = std::lower_bound(
      q->begin(), q->end(), okvis::Measurement<Element>(eraseUntil, Element()),
      [](okvis::Measurement<Element> lhs, okvis::Measurement<Element> rhs)
          -> bool { return lhs.timeStamp < rhs.timeStamp; });
  int removed = eraseEnd - q->begin();

  q->erase(q->begin(), eraseEnd);
  return removed;
}

/**
 * @brief The TimeAndRotationCalibrator class
 * Implementation of Algorithm 1 in
 * Qiu et al. 2020 T-RO. Real-time temporal and rotational calibration of
 * heterogeneous sensors using motion correlation analysis. Notation: W - world
 * frame, G - Generic sensor frame, I - reference IMU frame.
 */
class TimeAndRotationCalibrator {
public:
  enum CalibrationStatus {
    Unknown = 0,
    InsufficientData,
    FailedObservabilityCondition,
    Successful
  };

  TimeAndRotationCalibrator(double observationDuration = 8)
      : observationDuration_(observationDuration), status_(Unknown),
        calibratedTimeOffset_(0) {
    calibratedOrientation_.setIdentity();
  }

  void addImuAngularRate(okvis::Time time,
                         const Eigen::Vector3d &gyroMeasured) {
    angularRates_.emplace_back(time, gyroMeasured);
  }

  void addTargetAngularRate(okvis::Time time,
                            const Eigen::Vector3d &gyroMeasured) {
    targetAngularRates_.emplace_back(time, gyroMeasured);
  }

  void addTargetOrientation(okvis::Time time, const Eigen::Quaterniond &q_WG) {
    targetOrientations_.emplace_back(time, q_WG);
  }

  CalibrationStatus calibrate();

  int slideWindow() {
    double windowSize = 1.5 * observationDuration_;
    int a = removeObsoleteData(&angularRates_, windowSize);
    int b = removeObsoleteData(&targetAngularRates_, windowSize);
    int c = removeObsoleteData(&targetOrientations_, windowSize);
    return a + b + c;
  }

  CalibrationStatus status() { return status_; }

  Eigen::Quaterniond relativeOrientation() const {
    return calibratedOrientation_;
  }

  double relativeTimeOffset() const { return calibratedTimeOffset_; }

private:
  Eigen::AlignedDeque<okvis::Measurement<Eigen::Vector3d>> angularRates_;
  Eigen::AlignedDeque<okvis::Measurement<Eigen::Vector3d>> targetAngularRates_;
  Eigen::AlignedDeque<okvis::Measurement<Eigen::Quaterniond>>
      targetOrientations_;
  double observationDuration_; // d_o
  CalibrationStatus status_;
  Eigen::Quaterniond calibratedOrientation_; // R_IG
  double calibratedTimeOffset_; // timestamp by imu clock = timestamp by target
                                // sensor clock + time offset.
};

} // namespace sensor_calib
