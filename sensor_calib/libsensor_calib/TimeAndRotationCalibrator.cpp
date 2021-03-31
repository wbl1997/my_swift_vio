#include "TimeAndRotationCalibrator.h"
#include "msckf/imu/SimpleImuOdometry.hpp"
#include "okvis/kinematics/sophus_operators.hpp"

namespace sensor_calib {

bool cmp(okvis::Measurement<Eigen::Vector3d> lhs,
         okvis::Measurement<Eigen::Vector3d> rhs) {
  return lhs.timeStamp < rhs.timeStamp;
}

int computeCovariance(
    okvis::Time startTimestamp, double observationDuration,
    const std::vector<
        const Eigen::AlignedDeque<okvis::Measurement<Eigen::Vector3d>> *> &xy,
    Eigen::Matrix3d *C) {
  Eigen::AlignedVector<okvis::Measurement<Eigen::Vector3d>> boundary;
  boundary.reserve(2);
  boundary.emplace_back(startTimestamp, Eigen::Vector3d());
  boundary.emplace_back(startTimestamp + okvis::Duration(observationDuration),
                        Eigen::Vector3d());

  std::vector<
      Eigen::AlignedDeque<okvis::Measurement<Eigen::Vector3d>>::const_iterator>
      rangeIters;
  rangeIters.reserve(4);
  for (std::vector<const Eigen::AlignedDeque<
           okvis::Measurement<Eigen::Vector3d>> *>::const_iterator xit =
           xy.begin();
       xit != xy.end(); ++xit) {
    for (auto val : boundary) {
      auto it = std::lower_bound((*xit)->begin(), (*xit)->end(), val, cmp);
      rangeIters.push_back(it);
    }
  }

  int numObservations = std::distance(rangeIters[0], rangeIters[1]);
  CHECK_EQ(numObservations, std::distance(rangeIters[2], rangeIters[3]))
      << "The two correlation arrays differs in timestamps!";

  Eigen::AlignedVector<Eigen::Matrix<double, -1, 3>> XY{
      Eigen::Matrix<double, -1, 3>(numObservations, 3),
      Eigen::Matrix<double, -1, 3>(numObservations, 3)};

  for (int i = 0; i < 2; ++i) {
    int j = 0;
    for (auto it = rangeIters[i * 2 + 0]; it != rangeIters[i * 2 + 1];
         ++it, ++j) {
      XY[i].row(j) = it->measurement;
    }
    CHECK_EQ(j, numObservations) << "Inconsistent indexing!";
  }
  cov(XY[0], XY[1], *C);
  return numObservations;
}

TimeAndRotationCalibrator::TimeAndRotationCalibrator(
    double observationDuration, double imuRate,
    double traceCorrelationLowerBound, double eigenvalueRatioUpperBound,
    double minimumEigenvalueLowerBound, double enumerationRangeSecs)
    : targetSamplingInterval_(100), observationDuration_(observationDuration),
      imuRate_(imuRate),
      traceCorrelationLowerBound_(traceCorrelationLowerBound),
      eigenvalueRatioUpperBound_(eigenvalueRatioUpperBound),
      minimumEigenvalueLowerBound_(minimumEigenvalueLowerBound),
      enumerationRangeSecs_(enumerationRangeSecs),
      samplingInterval_(1.0 / imuRate),
      samplingIntervalLambda_(0.9),
      status_(Unknown),
      calibratedTimeOffset_(0) {
  calibratedOrientation_.setIdentity();

  halfTotalSteps_ = (int)enumerationRangeSecs_ * imuRate_;
  averageImuAngularRateBuffer_.resize(2 * halfTotalSteps_ + 1);
}

void TimeAndRotationCalibrator::addImuAngularRate(
    okvis::Time time, const Eigen::Vector3d &gyroMeasured) {
  angularRates_.emplace_back(time, gyroMeasured);
  if (uniformAngularRates_.empty()) {
    uniformAngularRates_.emplace_back(time, gyroMeasured);
    return;
  }
  for (okvis::Time u =
           uniformAngularRates_.back().timeStamp + samplingInterval_;
       u <= time; u += samplingInterval_) {
    auto iter = std::lower_bound(
        angularRates_.begin(), angularRates_.end(),
        okvis::Measurement<Eigen::Vector3d>(u, Eigen::Vector3d()), cmp);
    if (iter == angularRates_.begin()) {
      break;
    } else {
      auto right = iter;
      auto left = iter;
      --left;
      double ratio = (u - left->timeStamp).toSec() /
                     (right->timeStamp - left->timeStamp).toSec();
      Eigen::Vector3d gyro =
          (right->measurement - left->measurement) * ratio + left->measurement;
      uniformAngularRates_.emplace_back(u, gyro);
    }
  }
}

void TimeAndRotationCalibrator::addTargetAngularRate(
    okvis::Time time, const Eigen::Vector3d &gyroMeasured) {
  if (targetAngularRates_.size()) {
    double observedSamplingInterval = (time - targetAngularRates_.back().timeStamp).toSec();
    targetSamplingInterval_ = (1 - samplingIntervalLambda_) * targetSamplingInterval_ +
        samplingIntervalLambda_ * observedSamplingInterval;
  }
  targetAngularRates_.emplace_back(time, gyroMeasured);

  using namespace std;
  clock_t begin = clock();

  for (int step = -halfTotalSteps_; step <= halfTotalSteps_; ++step) {
    okvis::Duration td(step * samplingInterval_.toSec());
    Eigen::AlignedDeque<okvis::Measurement<Eigen::Vector3d>> &tdBuffer =
        averageImuAngularRateBuffer_.at(step + halfTotalSteps_);
    auto iterEnd = targetAngularRates_.end();
    --iterEnd;
    auto iter = targetAngularRates_.begin();
    if (tdBuffer.size()) {
      iter = std::lower_bound(targetAngularRates_.begin(),
                              targetAngularRates_.end(),
                              okvis::Measurement<Eigen::Vector3d>(
                                  tdBuffer.back().timeStamp, Eigen::Vector3d()),
                              cmp);
      CHECK_EQ(iter->timeStamp, tdBuffer.back().timeStamp);
      ++iter; // skip the existing one.
      CHECK(iter != targetAngularRates_.end());
    }
    for (; iter != iterEnd; ++iter) {
      okvis::Time t_k = iter->timeStamp;
      okvis::Time t_kp1 = (iter + 1)->timeStamp;
      // Average angular velocity of IMU between td + t_k and td + t_k+1
      bool leftOk = uniformAngularRates_.front().timeStamp <= t_k + td;
      bool rightOk = uniformAngularRates_.back().timeStamp >= t_kp1 + td;

      Eigen::Vector3d rotVector;

      Eigen::Vector3d gyroBias = Eigen::Vector3d::Zero();
      if (leftOk && rightOk) {
        okvis::ceres::predictStates(uniformAngularRates_, &rotVector, gyroBias,
                                    t_k + td, t_kp1 + td);

        Eigen::Vector3d omega_I = rotVector / (t_kp1 - t_k).toSec();
        tdBuffer.emplace_back(t_k, omega_I);
      }
    }
  }
  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  //  LOG(INFO) << "Elapsed secs " << elapsed_secs;
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
    addTargetAngularRate(t_k + okvis::Duration((time - t_k).toSec() * 0.5), omega_Gtk);
  }
  targetOrientations_.emplace_back(time, q_WG);
}

TimeAndRotationCalibrator::CalibrationStatus
TimeAndRotationCalibrator::calibrate() {
  if (!hasSufficientData()) {
    return CalibrationStatus::InsufficientData;
  }
  std::vector<double> traceCorrelations;
  traceCorrelations.resize(2 * halfTotalSteps_ + 1, 0);
  int tdIndex = 0;
  okvis::Time startTime = correlationStartTime();

  for (const Eigen::AlignedDeque<okvis::Measurement<Eigen::Vector3d>>
           &averageRateList : averageImuAngularRateBuffer_) {
    Eigen::Matrix<double, 3, 3> sigmaOmegaIOmegaG;
    int a = computeCovariance(startTime, observationDuration_,
                              {&averageRateList, &targetAngularRates_},
                              &sigmaOmegaIOmegaG);
    Eigen::Matrix<double, 3, 3> sigmaOmegaGOmegaG;
    int b = computeCovariance(startTime, observationDuration_,
                              {&targetAngularRates_, &targetAngularRates_},
                              &sigmaOmegaGOmegaG);
    Eigen::Matrix<double, 3, 3> sigmaOmegaIOmegaI;
    int c = computeCovariance(startTime, observationDuration_,
                              {&averageRateList, &averageRateList},
                              &sigmaOmegaIOmegaI);
    CHECK_EQ(a, b);
    CHECK_EQ(a, c);
    Eigen::Matrix<double, 3, 3> sigmaOmegaGOmegaI =
        sigmaOmegaIOmegaG.transpose();

    Eigen::Matrix3d sigmaProduct =
        sigmaOmegaIOmegaI.inverse() * sigmaOmegaIOmegaG *
        sigmaOmegaGOmegaG.inverse() * sigmaOmegaGOmegaI;

    double rbar2 = sigmaProduct.diagonal().sum() / 3.0;
    traceCorrelations[tdIndex] = std::sqrt(rbar2);
    ++tdIndex;
  }

  // find the max correlation in traceCorrelations
  auto it = std::max_element(std::begin(traceCorrelations),
                             std::end(traceCorrelations));
  double maxCorr = *it;
  int maxPoint = std::distance(traceCorrelations.begin(), it);
  double refinedMaxPoint = maxPoint;
  double refinedMaxCorr = maxCorr;
  if (it != traceCorrelations.begin() && it != traceCorrelations.end()) {
    std::vector<double> xList{-1, 0, 1};
    Eigen::Vector3d yList{*(it - 1), *it, *(it + 1)};
    Eigen::Matrix3d coeff;
    for (int i = 0; i < 3; ++i) {
      coeff(i, 0) = xList[i] * xList[i];
      coeff(i, 1) = xList[i];
      coeff(i, 2) = 1;
    }
    Eigen::Vector3d abc = coeff.inverse() * yList;
    refinedMaxPoint = maxPoint - abc[1] / (2 * abc[0]);
    refinedMaxCorr = abc[2] - abc[1] * abc[1] / (4 * abc[0]);
  } else {
    LOG(WARNING) << "Max correlation happens at the boundary of index: "
                 << std::distance(traceCorrelations.begin(), it);
  }
  LOG(INFO) << "Max correlation index " << maxPoint << " is refined to "
            << refinedMaxPoint;

  okvis::Duration td((refinedMaxPoint - halfTotalSteps_) * samplingInterval_.toSec());
  calibratedTimeOffset_ = td.toSec() + 0.5 * targetSamplingInterval_;

  // first recompute Sigmas at the refined time offset.
  Eigen::AlignedDeque<okvis::Measurement<Eigen::Vector3d>> tdBuffer;
  auto iterEnd = targetAngularRates_.end();
  --iterEnd;
  auto iter = targetAngularRates_.begin();
  for (; iter != iterEnd; ++iter) {
    okvis::Time t_k = iter->timeStamp;
    okvis::Time t_kp1 = (iter + 1)->timeStamp;

    // Average angular velocity of IMU between td + t_k and td + t_k+1
    bool leftOk = uniformAngularRates_.front().timeStamp <= t_k + td;
    bool rightOk = uniformAngularRates_.back().timeStamp >= t_kp1 + td;
    Eigen::Vector3d rotVector;
    Eigen::Vector3d gyroBias = Eigen::Vector3d::Zero();
    if (leftOk && rightOk) {
      okvis::ceres::predictStates(uniformAngularRates_, &rotVector, gyroBias,
                                  t_k + td, t_kp1 + td);

      Eigen::Vector3d omega_I = rotVector / (t_kp1 - t_k).toSec();
      tdBuffer.emplace_back(t_k, omega_I);
    }
  }

  Eigen::Matrix<double, 3, 3> sigmaOmegaIOmegaG;
  int a =
      computeCovariance(startTime, observationDuration_,
                        {&tdBuffer, &targetAngularRates_}, &sigmaOmegaIOmegaG);
  Eigen::Matrix<double, 3, 3> sigmaOmegaGOmegaG;
  int b = computeCovariance(startTime, observationDuration_,
                            {&targetAngularRates_, &targetAngularRates_},
                            &sigmaOmegaGOmegaG);
  Eigen::Matrix<double, 3, 3> sigmaOmegaIOmegaI;
  int c = computeCovariance(startTime, observationDuration_,
                            {&tdBuffer, &tdBuffer}, &sigmaOmegaIOmegaI);
  CHECK_EQ(a, b);
  CHECK_EQ(a, c);
  Eigen::Matrix<double, 3, 3> sigmaOmegaGOmegaI = sigmaOmegaIOmegaG.transpose();

  Eigen::Matrix3d sigmaProduct =
      sigmaOmegaIOmegaI.inverse() * sigmaOmegaIOmegaG *
      sigmaOmegaGOmegaG.inverse() * sigmaOmegaGOmegaI;

  double rbar2 = sigmaProduct.diagonal().sum() / 3.0;
  double refinedTraceCorrelation = std::sqrt(rbar2);
  CHECK_LT(std::fabs(refinedTraceCorrelation - refinedMaxCorr), 1e-3);

  // estimate extrinsic orientation by SVD.
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sigmaOmegaIOmegaI);
  Eigen::Vector3d eigenvalues = saes.eigenvalues();
  double smallest = std::fabs(eigenvalues[0]);
  double largest = std::fabs(eigenvalues[2]);
  if (refinedTraceCorrelation > traceCorrelationLowerBound_ &&
      smallest > minimumEigenvalueLowerBound_ &&
      largest < smallest * eigenvalueRatioUpperBound_) {
    Eigen::Matrix3d sigmaProduct =
        sigmaOmegaGOmegaG.inverse() * sigmaOmegaGOmegaI;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        sigmaProduct, Eigen::ComputeThinU | Eigen::ComputeThinV);
    double det = (svd.matrixU() * svd.matrixV().transpose()).determinant();
    Eigen::Vector3d diag{1, 1, det};
    Eigen::Matrix3d R_GI =
        svd.matrixU() * diag.asDiagonal() * svd.matrixV().transpose();
    calibratedOrientation_ = Eigen::Quaterniond(R_GI.transpose());
  } else {
    return CalibrationStatus::FailedObservabilityCondition;
  }

  return CalibrationStatus::Successful;
}

int TimeAndRotationCalibrator::slideWindow() {
  double windowSize = 1.5 * observationDuration_;
  okvis::Time eraseUntil =
      angularRates_.back().timeStamp - okvis::Duration(windowSize);
  int a = removeObsoleteData(&angularRates_, eraseUntil);
  int b = removeObsoleteData(&targetAngularRates_, eraseUntil);
  int c = removeObsoleteData(&targetOrientations_, eraseUntil);
  int d = removeObsoleteData(&uniformAngularRates_, eraseUntil);
  int all = 0;
  for (Eigen::AlignedDeque<okvis::Measurement<Eigen::Vector3d>> &averageList :
       averageImuAngularRateBuffer_) {
    all += removeObsoleteData(&averageList, eraseUntil);
  }
  return a + b + c + d + all;
}

void TimeAndRotationCalibrator::check() const {
  int index = 0;
  for (auto array : averageImuAngularRateBuffer_) {
    LOG(INFO) << "Average array " << index << " size " << array.size();
    okvis::Time lastTime;
    int j = 0;
    for (auto item : array) {
      if (j == 0) {
        lastTime = item.timeStamp;
      } else {
        okvis::Duration gap = item.timeStamp - lastTime;
        CHECK_LT(std::fabs(gap.toSec() - targetSamplingInterval_), 1e-7);
      }
      lastTime = item.timeStamp;
      ++j;
    }
    ++index;
  }

  LOG(INFO) << "Target rates from " << targetAngularRates_.front().timeStamp
            << " to " << targetAngularRates_.back().timeStamp;
  LOG(INFO) << "IMU rates from " << angularRates_.front().timeStamp << " to "
            << angularRates_.back().timeStamp;
  LOG(INFO) << "Uniform rates from " << uniformAngularRates_.front().timeStamp
            << " to " << uniformAngularRates_.back().timeStamp;
}

bool TimeAndRotationCalibrator::hasSufficientData() const {
  return std::min(targetAngularRates_.back().timeStamp,
                  uniformAngularRates_.back().timeStamp) -
             std::max(targetAngularRates_.front().timeStamp,
                      uniformAngularRates_.front().timeStamp) >
         okvis::Duration(observationDuration_ + 2 * enumerationRangeSecs_);
}

okvis::Time TimeAndRotationCalibrator::correlationStartTime() const {
  return std::min(targetAngularRates_.back().timeStamp,
                  uniformAngularRates_.back().timeStamp) -
         okvis::Duration(observationDuration_ + enumerationRangeSecs_);
}

} // namespace sensor_calib
