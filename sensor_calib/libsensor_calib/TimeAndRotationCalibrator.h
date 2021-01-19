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
 * @brief cov compute covariance between two matrices.
 * @param[in] x rowwise observation matrix.
 * @param[in] y rowwise observation matrix.
 * @param[out] C_ output covariance matrix.
 * https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
 * An alternative is to use FFT as in cvMatchTemplate() with method=CV_TM_CCORR_NORMED.
 */
template <typename Derived, typename OtherDerived>
void cov(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& y, Eigen::MatrixBase<OtherDerived> const & C_)
{
  typedef typename Derived::Scalar Scalar;
  typedef typename Eigen::internal::plain_row_type<Derived>::type RowVectorType;

  const Scalar num_observations = static_cast<Scalar>(x.rows());

  const RowVectorType x_mean = x.colwise().sum() / num_observations;
  const RowVectorType y_mean = y.colwise().sum() / num_observations;

  Eigen::MatrixBase<OtherDerived>& C = const_cast< Eigen::MatrixBase<OtherDerived>& >(C_);

  C.derived().resize(x.cols(),x.cols()); // resize the derived object
  C = (x.rowwise() - x_mean).transpose() * (y.rowwise() - y_mean) / (num_observations - 1);
}

/**
 * @brief The TimeAndRotationCalibrator class
 * Implementation of Algorithm 1 in
 * Qiu et al. 2020 T-RO. Real-time temporal and rotational calibration of
 * heterogeneous sensors using motion correlation analysis.
 * Assume the central IMU sensor has a higher sampling rate than the target sensor.
 * Notation: W - world frame, G - Generic sensor frame, I - reference IMU frame.
 */
class TimeAndRotationCalibrator {
public:
  enum CalibrationStatus {
    Unknown = 0,
    InsufficientData,
    FailedObservabilityCondition,
    Successful
  };

  TimeAndRotationCalibrator(double observationDuration = 8,
                            double imuRate = 200.0,
                            double traceCorrelationLowerBound = 0.9,
                            double eigenvalueRatioUpperBound = 10,
                            double minimumEigenvalueLowerBound = 0.03,
                            double enumerationRangeMs = 1100);

  void addImuAngularRate(okvis::Time time,
                         const Eigen::Vector3d &gyroMeasured);

  void addTargetAngularRate(okvis::Time time,
                            const Eigen::Vector3d &gyroMeasured);

  void addTargetOrientation(okvis::Time time, const Eigen::Quaterniond &q_WG);

  CalibrationStatus calibrate();

  int slideWindow();

  CalibrationStatus status() const { return status_; }

  Eigen::Quaterniond relativeOrientation() const {
    return calibratedOrientation_;
  }

  double relativeTimeOffset() const { return calibratedTimeOffset_; }

private:
  void computeCovariance();



private:
  Eigen::AlignedDeque<okvis::Measurement<Eigen::Vector3d>> angularRates_; // IMU measured angular rates.
  Eigen::AlignedDeque<okvis::Measurement<Eigen::Vector3d>> targetAngularRates_; // angular rates measured by the target sensor.
  Eigen::AlignedDeque<okvis::Measurement<Eigen::Quaterniond>>
      targetOrientations_; // orientations measured by the target sensor.

  Eigen::AlignedDeque<okvis::Measurement<Eigen::Vector3d>> uniformAngularRates_; // uniform upsampled IMU angular rates.
  Eigen::AlignedVector<Eigen::AlignedDeque<okvis::Measurement<Eigen::Vector3d>>>
      averageImuAngularRateBuffer_;

  double observationDuration_; // d_o
  double imuRate_; // f_I, nominal rate of the central IMU sensor.
  double traceCorrelationLowerBound_; // \f$ \epsilon_b \f$
  double eigenvalueRatioUpperBound_; // \f$ \zeta_u \f$
  double minimumEigenvalueLowerBound_; // \f$ \zeta_b \f$
  double enumerationRangeMs_; // \f$ t_r \f$ in millisecs.
  double targetSensorRate_; // f_G, nominal rate of the generic sensor.
  int halfTotalSteps_;

  CalibrationStatus status_;
  Eigen::Quaterniond calibratedOrientation_; // R_IG
  double calibratedTimeOffset_; // timestamp by imu clock = timestamp by target
                                // sensor clock + time offset.
};

} // namespace sensor_calib
