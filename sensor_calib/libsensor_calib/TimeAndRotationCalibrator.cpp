#include "TimeAndRotationCalibrator.h"
#include "okvis/kinematics/sophus_operators.hpp"

namespace sensor_calib {

TimeAndRotationCalibrator::TimeAndRotationCalibrator(
    double observationDuration, double imuRate,
    double traceCorrelationLowerBound, double eigenvalueRatioUpperBound,
    double minimumEigenvalueLowerBound, double enumerationRangeMs)
    : observationDuration_(observationDuration), imuRate_(imuRate),
      traceCorrelationLowerBound_(traceCorrelationLowerBound),
      eigenvalueRatioUpperBound_(eigenvalueRatioUpperBound),
      minimumEigenvalueLowerBound_(minimumEigenvalueLowerBound),
      enumerationRangeMs_(enumerationRangeMs), status_(Unknown),
      calibratedTimeOffset_(0) {
  calibratedOrientation_.setIdentity();

  halfTotalSteps_ = (int)enumerationRangeMs_ * imuRate_ / 1000;
  averageImuAngularRateBuffer_.resize(2 * halfTotalSteps_ + 1);
}

void TimeAndRotationCalibrator::addImuAngularRate(
    okvis::Time time, const Eigen::Vector3d &gyroMeasured) {
  angularRates_.emplace_back(time, gyroMeasured);
  // TODO(jhuai): Also add entries to uniformAngularRates_ by uniform upsampling
  // based on linear interpolation.
  uniformAngularRates_.emplace_back(time, gyroMeasured);
}

void TimeAndRotationCalibrator::addTargetAngularRate(
    okvis::Time time, const Eigen::Vector3d &gyroMeasured) {
  // Enumerate time offsets td, and update buffer
  double stepSize = 1.0 / imuRate_;
  if (targetAngularRates_.size()) {
    okvis::Time t_k = targetAngularRates_.back().timeStamp;
    okvis::Time t_kp1 = time; // t_k+1
    for (int step = -halfTotalSteps_; step < halfTotalSteps_; ++step) {
      double td = step * stepSize;
      // TODO: Extract averaging angular velocity of IMU between td + t_k and td
      // + t_k+1 lower_bound on deque uniformAngularRates_, then integrate
      Eigen::Vector3d omega_Itd;
      averageImuAngularRateBuffer_.at(step + halfTotalSteps_)
          .emplace_back(t_k + okvis::Duration(td), omega_Itd);
    }
  }

  targetAngularRates_.emplace_back(time, gyroMeasured);
}

void TimeAndRotationCalibrator::addTargetOrientation(
    okvis::Time time, const Eigen::Quaterniond &q_WG) {
  if (targetOrientations_.size()) {
    // Extract averaging angular velocity of the target sensor.
    okvis::Time t_k = targetOrientations_.back().timeStamp;
    Eigen::Quaternion q_WGtk = targetOrientations_.back().measurement;
    double theta;
    Eigen::Vector3d omega_Gtk =
        okvis::kinematics::logAndTheta(q_WGtk.inverse() * q_WG, &theta);
    omega_Gtk /= (time - t_k).toSec();
    targetAngularRates_.emplace_back(t_k, omega_Gtk);
  }
  targetOrientations_.emplace_back(time, q_WG);
}

TimeAndRotationCalibrator::CalibrationStatus
TimeAndRotationCalibrator::calibrate() {
  if ((targetAngularRates_.back().timeStamp -
       targetAngularRates_.front().timeStamp)
          .toSec() < observationDuration_) {
    return CalibrationStatus::InsufficientData;
  }
  std::vector<double> traceCorrelations;
  traceCorrelations.resize(2 * halfTotalSteps_ + 1, 0);
  int tdIndex = 0;
  double stepSize = 1.0 / imuRate_;
  for (auto averageRateList : averageImuAngularRateBuffer_) {
    //   compose the observation matrices
    double td = (tdIndex - halfTotalSteps_) * stepSize;
    // compute correlation with averageImuAngularRateBuffer_[halfTotalSteps] and
    // targetAngularRates_ with a shift td which is obtained by linear interpolation.
    Eigen::Matrix<double, 3, 3> sigmaOmegaIOmegaG;
    // compute correlation with targetAngularRates_ and targetAngularRates_
    // with a shift td which is obtained by linear interpolation.
    Eigen::Matrix<double, 3, 3> sigmaOmegaGOmegaG;
    // compute correlation with targetAngularRates_ and averageRateList
    Eigen::Matrix<double, 3, 3> sigmaOmegaGOmegaI;
    // compute correlation with averageImuAngularRateBuffer_[halfTotalSteps] and averageRateList
    Eigen::Matrix<double, 3, 3> sigmaOmegaIOmegaI;


    //   compute trace correlation with cov()
    double rbar2 = (sigmaOmegaIOmegaI.inverse() * sigmaOmegaIOmegaG *
                    sigmaOmegaGOmegaG.inverse() * sigmaOmegaGOmegaI)
                       .diagonal()
                       .sum() /
                   3.0;
    traceCorrelations[tdIndex] = std::sqrt(rbar2);
    ++tdIndex;
  }
  // find the max correlation in traceCorrelations

  // quadratic fit to refine the time offset
  //  double refinedTraceCorrelation = quadraticFit();
  calibratedTimeOffset_ = 0;

  // estimate extrinsic orientation by SVD
  // first recompute Sigmas at the refined time offset.
  double refinedTraceCorrelation = 0;
  Eigen::Matrix<double, 3, 3> sigmaOmegaIOmegaG;
  Eigen::Matrix<double, 3, 3> sigmaOmegaGOmegaI;
  Eigen::Matrix<double, 3, 3> sigmaOmegaIOmegaI;
  Eigen::Matrix<double, 3, 3> sigmaOmegaGOmegaG;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sigmaOmegaIOmegaI);
  Eigen::Vector3d eigenvalues = saes.eigenvalues();
  double smallest = std::fabs(eigenvalues[0]);
  double largest = std::fabs(eigenvalues[2]);
  if (refinedTraceCorrelation > traceCorrelationLowerBound_ &&
      smallest > minimumEigenvalueLowerBound_ &&
      largest < smallest * eigenvalueRatioUpperBound_) {
    Eigen::Matrix3d sigmaProduct =
        sigmaOmegaGOmegaG.inverse() * sigmaOmegaGOmegaI;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sigmaProduct);
    Eigen::Vector3d diag{
        1, 1,
        (saes.eigenvectors() * saes.eigenvectors().transpose()).determinant()};
    Eigen::Matrix3d R_IG = saes.eigenvectors() * diag.asDiagonal() *
                           saes.eigenvectors().transpose();
    calibratedOrientation_ = Eigen::Quaterniond(R_IG);
  } else {
    return CalibrationStatus::FailedObservabilityCondition;
  }

  return CalibrationStatus::Successful;
}

int TimeAndRotationCalibrator::slideWindow() {
  double windowSize = 1.5 * observationDuration_;
  int a = removeObsoleteData(&angularRates_, windowSize);
  int b = removeObsoleteData(&targetAngularRates_, windowSize);
  int c = removeObsoleteData(&targetOrientations_, windowSize);
  // TODO: remove old uniformAngularRates_
  // remove old measurements in averageImuAngularRateBuffer_.
  return a + b + c;
}

} // namespace sensor_calib
