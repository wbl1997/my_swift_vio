#include <swift_vio/CalibrationFilter.hpp>

#include <swift_vio/EkfUpdater.h>
#include <swift_vio/FilterHelper.hpp>

#include <glog/logging.h>

namespace swift_vio {
// Constructor if a ceres map is already available.
CalibrationFilter::CalibrationFilter(std::shared_ptr<okvis::ceres::Map> mapPtr)
    : HybridFilter(mapPtr) {
}

// The default constructor.
CalibrationFilter::CalibrationFilter()
    : HybridFilter() {
}

CalibrationFilter::~CalibrationFilter() {
  LOG(INFO) << "Destructing CalibrationFilter";
}

bool CalibrationFilter::addStates(okvis::MultiFramePtr multiFrame,
                             const okvis::ImuMeasurementDeque& imuMeasurements,
                             bool asKeyframe) {
  return HybridFilter::addStates(multiFrame, imuMeasurements, asKeyframe);
}

void CalibrationFilter::optimize(size_t /*numIter*/, size_t /*numThreads*/,
                            bool /*verbose*/) {
  // containers of Jacobians of measurements of points in the states
  Eigen::AlignedVector<Eigen::VectorXd> vr_i;
  Eigen::AlignedVector<Eigen::MatrixXd> vH_x;  // each entry has a size say 2j x(2 + 13 + 9m)
  Eigen::AlignedVector<Eigen::MatrixXd> vR_i;

  OKVIS_ASSERT_EQ_DBG(
      Exception, (size_t)covariance_.rows(),
      startIndexOfClonedStatesFast() +
          kClonedStateMinimalDimen * statesMap_.size(),
      "Inconsistent rows of covariance matrix and number of states");

  int numCamParamPoseVariables =
      (imuParametersVec_.at(0).estimateGravityDirection ? 2u : 0u) +
      minimalDimOfAllCameraParams() +
      kClonedStateMinimalDimen * statesMap_.size();

  size_t numSlamObservations = 0u;  // number of observations for landmarks in state and tracked now

  const uint64_t currFrameId = currentFrameId();
  size_t navAndImuParamsDim = navStateAndImuParamsMinimalDim();

  Eigen::MatrixXd variableCov = covariance_.block(
      navAndImuParamsDim, navAndImuParamsDim,
      numCamParamPoseVariables, numCamParamPoseVariables);  // covariance block for camera and pose state copies including the current pose state is used for SLAM features.

  std::vector<std::pair<uint64_t, int>> slamLandmarks;
  slamLandmarks.reserve(pointLandmarkOptions_.maxInStateLandmarks / 2);
  for (okvis::PointMap::iterator it = landmarksMap_.begin();
       it != landmarksMap_.end(); ++it) {
    if (it->second.trackedInCurrentFrame(currFrameId)) {
      slamLandmarks.emplace_back(it->first, it->second.observations.size());
    }
  }

  std::sort(slamLandmarks.begin(), slamLandmarks.end(),
            [](const std::pair<uint64_t, int> &a,
               const std::pair<uint64_t, int> &b) -> bool {
              return a.second > b.second;
            });

  for (auto idAndLength : slamLandmarks) {
    okvis::MapPoint &mapPoint = landmarksMap_.at(idAndLength.first);
    // compute residual and Jacobian for an observed checkerboard landmark.
    Eigen::VectorXd r_i;
    Eigen::MatrixXd H_x;
    Eigen::MatrixXd H_f;
    Eigen::MatrixXd R_i;

    Eigen::MatrixXd subH_f;

    Eigen::Vector4d homoPointRep = mapPoint.pointHomog;
    bool isValidJacobian = slamFeatureJacobian(mapPoint, homoPointRep, H_x, r_i, R_i, subH_f);
    if (!isValidJacobian) {
      mapPoint.setMeasurementFate(
          FeatureTrackStatus::kComputingJacobiansFailed);
      continue;
    }

    if (!FilterHelper::gatingTest(H_x, r_i, R_i, variableCov, optimizationOptions_.useMahalanobisGating)) {
      mapPoint.setMeasurementFate(FeatureTrackStatus::kPotentialOutlier);
      continue;
    }
    mapPoint.status.measurementFate = FeatureTrackStatus::kSuccessful;
    vr_i.push_back(r_i);
    vH_x.push_back(H_x);
    vR_i.push_back(R_i);
    numSlamObservations += r_i.size();
  }
  VLOG(0) << "Used #observation rows " << numSlamObservations
          << " out of #SLAM landmarks " << slamLandmarks.size();

  // update with SLAM features
  if (numSlamObservations) {
    Eigen::MatrixXd H_all(numSlamObservations, numCamParamPoseVariables);
    Eigen::VectorXd r_all(numSlamObservations);
    Eigen::MatrixXd R_all = Eigen::MatrixXd::Zero(numSlamObservations, numSlamObservations);
    size_t startRow = 0u;
    for (size_t jack = 0u; jack < vr_i.size(); ++jack) {
      int blockRows = vr_i[jack].rows();
      H_all.block(startRow, 0, blockRows, numCamParamPoseVariables) =
          vH_x[jack];
      r_all.segment(startRow, blockRows) = vr_i[jack];
      R_all.block(startRow, startRow, blockRows, blockRows) = vR_i[jack];
      startRow += blockRows;
    }

    DefaultEkfUpdater updater(covariance_, navAndImuParamsDim, numCamParamPoseVariables);
    computeKalmanGainTimer.start();
    Eigen::Matrix<double, Eigen::Dynamic, 1> deltaX =
        updater.computeCorrection(H_all, r_all, R_all);
    computeKalmanGainTimer.stop();
    updateStatesTimer.start();
    updateStates(deltaX);
    updateStatesTimer.stop();
    updateCovarianceTimer.start();
    updater.updateCovariance(&covariance_);
    updateCovarianceTimer.stop();
  }
}

bool CalibrationFilter::applyMarginalizationStrategy(
    okvis::MapPointVector &/*removedLandmarks*/) {
  std::vector<uint64_t> rm_cam_state_ids;
  rm_cam_state_ids.reserve(3);
  auto rit = statesMap_.rbegin();
  for (int j = 0; j < optimizationOptions_.numImuFrames; ++j) {
    ++rit;
  }
  int keyframeCount = 0;
  for (; rit != statesMap_.rend(); ++rit) {
    if (rit->second.isKeyframe) {
      ++keyframeCount;
    }
    if (keyframeCount >= optimizationOptions_.numKeyframes) {
      rm_cam_state_ids.push_back(rit->first);
    }
  }

  for (const auto &cam_id : rm_cam_state_ids) {
    auto statesIter = statesMap_.find(cam_id);
    int cam_sequence = std::distance(statesMap_.begin(), statesIter);
    int cam_state_start = startIndexOfClonedStatesFast() +
                          kClonedStateMinimalDimen * cam_sequence;
    int cam_state_end = cam_state_start + kClonedStateMinimalDimen;

    FilterHelper::pruneSquareMatrix(cam_state_start, cam_state_end,
                                    &covariance_);
    removeState(cam_id);
  }

  inertialMeasForStates_.pop_front(statesMap_.begin()->second.timestamp -
                                   half_window_);

  return true;
}

}  // namespace swift_vio
