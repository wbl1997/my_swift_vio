#include <msckf/TFVIO.hpp>

#include <glog/logging.h>

#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/CameraTimeParamBlock.hpp>
#include <okvis/IdProvider.hpp>
#include <okvis/MultiFrame.hpp>

#include <msckf/EpipolarJacobian.hpp>
#include <msckf/EuclideanParamBlock.hpp>
#include <msckf/FilterHelper.hpp>
#include <msckf/ImuOdometry.h>
#include <msckf/PreconditionedEkfUpdater.h>
#include <msckf/TwoViewPair.hpp>

DECLARE_bool(use_AIDP);

DECLARE_bool(use_mahalanobis);
DECLARE_bool(use_first_estimate);
DECLARE_bool(use_RK4);

DECLARE_bool(use_IEKF);

DECLARE_double(max_proj_tolerance);

DEFINE_int32(
    two_view_constraint_scheme, 0,
    "0 the entire feature track of a landmark is used to "
    "compose two-view constraints which are used in one filter update step "
    "as the landmark disappears; "
    "1, use the latest two observations of a landmark to "
    "form one two-view constraint in one filter update step; "
    "2, use the fixed head observation and "
    "the receding tail observation of a landmark to "
    "form one two-view constraint in one filter update step");
DEFINE_double(
    image_noise_cov_multiplier, 9.0,
    "Enlarge the image observation noise covariance by this multiplier.");

/// \brief okvis Main namespace of this package.
namespace okvis {

TFVIO::TFVIO(std::shared_ptr<okvis::ceres::Map> mapPtr,
             const double readoutTime)
    : HybridFilter(mapPtr, readoutTime) {}

// The default constructor.
TFVIO::TFVIO(const double readoutTime) : HybridFilter(readoutTime) {}

TFVIO::~TFVIO() {}

// Applies the dropping/marginalization strategy according to Michael A.
// Shelley's MS thesis
bool TFVIO::applyMarginalizationStrategy(
    size_t /*numKeyframes*/, size_t /*numImuFrames*/,
    okvis::MapPointVector& /*removedLandmarks*/) {
  std::vector<uint64_t> removeFrames;
  std::map<uint64_t, States>::reverse_iterator rit = statesMap_.rbegin();
  while (rit != statesMap_.rend()) {
    if (rit->first < minValidStateID) {
      removeFrames.push_back(rit->second.id);
    }
    ++rit;
  }

  // remove features tracked no more
  for (PointMap::iterator pit = landmarksMap_.begin();
       pit != landmarksMap_.end();) {
    if (pit->second.residualizeCase == NotInState_NotTrackedNow) {
      ceres::Map::ResidualBlockCollection residuals =
          mapPtr_->residuals(pit->first);
      ++mTrackLengthAccumulator[residuals.size()];
      for (size_t r = 0; r < residuals.size(); ++r) {
        std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
            std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
                residuals[r].errorInterfacePtr);
        OKVIS_ASSERT_TRUE(Exception, reprojectionError,
                          "Wrong index of reprojection error");
        removeObservation(residuals[r].residualBlockId);
      }

      mapPtr_->removeParameterBlock(pit->first);
      pit = landmarksMap_.erase(pit);
    } else {
      ++pit;
    }
  }

  for (size_t k = 0; k < removeFrames.size(); ++k) {
    okvis::Time removedStateTime = removeState(removeFrames[k]);
    mStateID2Imu.pop_front(removedStateTime - half_window_);
  }

  // update covariance matrix
  size_t numRemovedStates = removeFrames.size();
  if (numRemovedStates == 0) {
    return true;
  }

  int startIndex = startIndexOfClonedStates();
  int finishIndex = startIndex + numRemovedStates * 9;
  CHECK_NE(finishIndex, covariance_.rows())
      << "Never remove the covariance of the lastest state";
  FilterHelper::pruneSquareMatrix(startIndex, finishIndex, &covariance_);

  return true;
}


bool TFVIO::featureJacobian(
    const MapPoint& mp,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>* Hi,
    Eigen::Matrix<double, Eigen::Dynamic, 1>* ri,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>* Ri) const {
  const int camIdx = 0;
  std::shared_ptr<okvis::cameras::CameraBase> tempCameraGeometry =
      camera_rig_.getCameraGeometry(camIdx);
  // head and tail observations for this feature point
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      obsInPixels;
  // id of head and tail frames observing this feature point
  std::vector<uint64_t> frameIds;
  std::vector<double> imagePointNoiseStds;  // std noise in pixels

  // each entry is undistorted coordinates in image plane at
  // z=1 in the specific camera frame, [\bar{x},\bar{y},1]
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      obsDirections;
  std::vector<okvis::kinematics::Transformation,
              Eigen::aligned_allocator<okvis::kinematics::Transformation>>
      T_WSs;
  RetrieveObsSeqType seqType =
      static_cast<RetrieveObsSeqType>(FLAGS_two_view_constraint_scheme);
  size_t numFeatures = gatherPoseObservForTriang(mp, tempCameraGeometry, &frameIds, &T_WSs,
                            &obsDirections, &obsInPixels, &imagePointNoiseStds,
                            seqType);

  // compute obsDirection Jacobians and count the valid ones, and
  // meanwhile resize the relevant data structures
  std::vector<
      Eigen::Matrix<double, 3, Eigen::Dynamic>,
      Eigen::aligned_allocator<Eigen::Matrix<double, 3, Eigen::Dynamic>>>
      dfj_dXcam(numFeatures);
  std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>
      cov_fij(numFeatures);
  std::vector<bool> projectStatus(numFeatures);
  int projOptModelId = camera_rig_.getProjectionOptMode(camIdx);
  for (int j = 0; j < numFeatures; ++j) {
      double pixelNoiseStd = imagePointNoiseStds[2 * j];
      bool projectOk = obsDirectionJacobian(obsDirections[j], tempCameraGeometry, projOptModelId,
                             pixelNoiseStd, &dfj_dXcam[j], &cov_fij[j]);
      projectStatus[j]= projectOk;
  }
  removeUnsetElements<uint64_t>(&frameIds, projectStatus);
  removeUnsetMatrices<okvis::kinematics::Transformation>(&T_WSs, projectStatus);
  removeUnsetMatrices<Eigen::Vector3d>(&obsDirections, projectStatus);
  removeUnsetMatrices<Eigen::Vector2d>(&obsInPixels, projectStatus);
  removeUnsetMatrices<Eigen::Matrix<double, 3, -1>>(&dfj_dXcam, projectStatus);
  removeUnsetMatrices<Eigen::Matrix3d>(&cov_fij, projectStatus);
  size_t numValidDirectionJac = frameIds.size();
  if (numValidDirectionJac < 2u) { // A two view constraint requires at least two obs
      return false;
  }

  // enlarge cov of the head obs to counteract the noise reduction 
  // due to correlation in head_tail scheme
  size_t trackLength = mp.observations.size();
  double headObsCovModifier[2] = {1.0, 1.0};
  headObsCovModifier[0] =
      seqType == HEAD_TAIL
          ? (static_cast<double>(trackLength - minTrackLength_ + 2u))
          : 1.0;

  std::vector<std::pair<int, int>> featurePairs =
      TwoViewPair::getFramePairs(numValidDirectionJac, TwoViewPair::FIXED_MIDDLE);
  const int numConstraints = featurePairs.size();
  int featureVariableDimen = cameraParamsMinimalDimen() +
                             kClonedStateMinimalDimen * (statesMap_.size());
  Hi->resize(numConstraints, featureVariableDimen);
  ri->resize(numConstraints, 1);

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> H_fi(numConstraints,
                                                             3 * numValidDirectionJac);
  H_fi.setZero();
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cov_fi(3 * numValidDirectionJac,
                                                               3 * numValidDirectionJac);
  cov_fi.setZero();

  int extrinsicModelId = camera_rig_.getExtrinsicOptMode(camIdx);
  const int minExtrinsicDim = camera_rig_.getMinimalExtrinsicDimen(camIdx);
  const int minProjDim = camera_rig_.getMinimalProjectionDimen(camIdx);
  const int minDistortDim = camera_rig_.getDistortionDimen(camIdx);

  EpipolarMeasurement epiMeas(*this, tempCameraGeometry, camIdx, extrinsicModelId,
                              minExtrinsicDim, minProjDim, minDistortDim);
  for (int count = 0; count < numConstraints; ++count) {
    const std::pair<int, int>& feature_pair = featurePairs[count];
    std::vector<int> index_vec{feature_pair.first, feature_pair.second};
    Eigen::Matrix<double, 1, Eigen::Dynamic> H_xjk(1, featureVariableDimen);
    std::vector<Eigen::Matrix<double, 1, 3>,
                Eigen::aligned_allocator<Eigen::Matrix<double, 1, 3>>>
        H_fjk;
    double rjk;
    epiMeas.prepareTwoViewConstraint(frameIds, T_WSs, obsDirections,
                                     obsInPixels, dfj_dXcam, cov_fij, index_vec);
    epiMeas.measurementJacobian(&H_xjk, &H_fjk, &rjk);
    Hi->row(count) = H_xjk;
    (*ri)(count) = rjk;
    for (int j = 0; j < 2; ++j) {
      int index = index_vec[j];
      H_fi.block<1, 3>(count, index * 3) = H_fjk[j];
      // TODO(jhuai): account for the IMU noise
      cov_fi.block<3, 3>(index * 3, index * 3) =
          cov_fij[index] * FLAGS_image_noise_cov_multiplier * headObsCovModifier[j];
    }
  }

  Ri->resize(numConstraints, numConstraints);
  *Ri = H_fi * cov_fi * H_fi.transpose();
  return true;
}

int TFVIO::computeStackedJacobianAndResidual(
    Eigen::MatrixXd* T_H, Eigen::Matrix<double, Eigen::Dynamic, 1>* r_q,
    Eigen::MatrixXd* R_q) const {
  // compute and stack Jacobians and Residuals for landmarks observed in current
  // frame
  const int camParamStartIndex = startIndexOfCameraParams();
  int featureVariableDimen = covariance_.rows() - camParamStartIndex;
  int dimH[2] = {0, featureVariableDimen};
  const Eigen::MatrixXd variableCov = covariance_.block(
      camParamStartIndex, camParamStartIndex, dimH[1], dimH[1]);

  // containers of Jacobians of measurements
  std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> vr;
  std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> vH;
  std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> vR;
  RetrieveObsSeqType seqType =
      static_cast<RetrieveObsSeqType>(FLAGS_two_view_constraint_scheme);
  for (auto it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
    ResidualizeCase rc = it->second.residualizeCase;
    const size_t nNumObs = it->second.observations.size();
    if (seqType == ENTIRE_TRACK) {
      if (rc != NotInState_NotTrackedNow || nNumObs < minTrackLength_) {
        continue;
      }
    } else {
      if (rc != NotToAdd_TrackedNow || nNumObs < minTrackLength_) {
        continue;
      }
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Hi;
    Eigen::Matrix<double, Eigen::Dynamic, 1> ri;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Ri;
    bool isValidJacobian = featureJacobian(it->second, &Hi, &ri, &Ri);
    if (!isValidJacobian) {
      continue;
    }

    if (!FilterHelper::gatingTest(Hi, ri, Ri, variableCov)) {
      continue;
    }
    vr.push_back(ri);
    vR.push_back(Ri);
    vH.push_back(Hi);
    dimH[0] += Hi.rows();
  }
  if (dimH[0] == 0) {
    return 0;
  }
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dimH[0], featureVariableDimen);
  Eigen::MatrixXd r(dimH[0], 1);
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(dimH[0], dimH[0]);
  FilterHelper::stackJacobianAndResidual(vH, vr, vR, &H, &r, &R);
  FilterHelper::shrinkResidual(H, r, R, T_H, r_q, R_q);
  return dimH[0];
}

uint64_t TFVIO::getMinValidStateID() const {
  uint64_t min_state_id = statesMap_.rbegin()->first;
  for (auto it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
    if (it->second.residualizeCase == NotInState_NotTrackedNow) continue;

    auto itObs = it->second.observations.begin();
    if (itObs->first.frameId <
        min_state_id) {  // this assume that it->second.observations is an
                         // ordered map
      min_state_id = itObs->first.frameId;
    }
  }
  return min_state_id;
}

void TFVIO::optimize(size_t /*numIter*/, size_t /*numThreads*/, bool verbose) {
  uint64_t currFrameId = currentFrameId();
  OKVIS_ASSERT_EQ(Exception, covariance_.rows() - startIndexOfClonedStates(),
                  (int)(kClonedStateMinimalDimen * statesMap_.size()),
                  "Inconsistent covDim and number of states");
  retrieveEstimatesOfConstants();

  // mark tracks of features that are not tracked in current frame
  int numTracked = 0;
  int featureVariableDimen = cameraParamsMinimalDimen() +
                             kClonedStateMinimalDimen * statesMap_.size();

  for (okvis::PointMap::iterator it = landmarksMap_.begin();
       it != landmarksMap_.end(); ++it) {
    ResidualizeCase toResidualize = NotInState_NotTrackedNow;
    for (auto itObs = it->second.observations.rbegin(),
              iteObs = it->second.observations.rend();
         itObs != iteObs; ++itObs) {
      if (itObs->first.frameId == currFrameId) {
        toResidualize = NotToAdd_TrackedNow;
        ++numTracked;
        break;
      }
    }
    it->second.residualizeCase = toResidualize;
  }
  trackingRate_ = static_cast<double>(numTracked) /
                  static_cast<double>(landmarksMap_.size());

  if (FLAGS_use_IEKF) {
    // c.f., Faraz Mirzaei, a Kalman filter based algorithm for IMU-Camera
    // calibration
    Eigen::Matrix<double, Eigen::Dynamic, 1> deltaX,
        tempDeltaX;  // record the last update step, used to cancel last update
                     // in IEKF
    size_t numIteration = 0;
    const double epsilon = 1e-3;
    PreconditionedEkfUpdater pceu(covariance_, featureVariableDimen);
    while (numIteration < 5) {
      if (numIteration) {
        updateStates(-deltaX);  // effectively undo last update in IEKF
      }
      Eigen::MatrixXd T_H, R_q;
      Eigen::Matrix<double, Eigen::Dynamic, 1> r_q;
      int numResiduals = computeStackedJacobianAndResidual(&T_H, &r_q, &R_q);
      if (numResiduals == 0) {
        // update minValidStateID, so that these old
        // frames are removed later
        minValidStateID = getMinValidStateID();
        return;  // no need to optimize
      }

      if (numIteration) {
        computeKalmanGainTimer.start();
        tempDeltaX = pceu.computeCorrection(T_H, r_q, R_q, &deltaX);
        computeKalmanGainTimer.stop();
        updateStates(tempDeltaX);
        if ((deltaX - tempDeltaX).lpNorm<Eigen::Infinity>() < epsilon) break;

      } else {
        computeKalmanGainTimer.start();
        tempDeltaX = pceu.computeCorrection(T_H, r_q, R_q);
        computeKalmanGainTimer.stop();
        updateStates(tempDeltaX);
        if (tempDeltaX.lpNorm<Eigen::Infinity>() < epsilon) break;
      }

      deltaX = tempDeltaX;
      ++numIteration;
    }
    updateCovarianceTimer.start();
    pceu.updateCovariance(&covariance_);
    updateCovarianceTimer.stop();
  } else {
    Eigen::MatrixXd T_H, R_q;
    Eigen::Matrix<double, Eigen::Dynamic, 1> r_q;
    int numResiduals = computeStackedJacobianAndResidual(&T_H, &r_q, &R_q);
    if (numResiduals == 0) {
      // update minValidStateID, so that these old
      // frames are removed later
      minValidStateID = getMinValidStateID();
      return;  // no need to optimize
    }
    PreconditionedEkfUpdater pceu(covariance_, featureVariableDimen);
    computeKalmanGainTimer.start();
    Eigen::Matrix<double, Eigen::Dynamic, 1> deltaX =
        pceu.computeCorrection(T_H, r_q, R_q);
    computeKalmanGainTimer.stop();
    updateStates(deltaX);

    updateCovarianceTimer.start();
    pceu.updateCovariance(&covariance_);
    updateCovarianceTimer.stop();
  }

  // update landmarks that are tracked in the current frame(the newly inserted
  // state)
  {
    updateLandmarksTimer.start();
    retrieveEstimatesOfConstants();  // do this because states are just updated
    minValidStateID = statesMap_.rbegin()->first;
    for (auto it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
      if (it->second.residualizeCase == NotInState_NotTrackedNow) continue;
      // this happens with a just inserted landmark without triangulation.
      if (it->second.observations.size() < 2) continue;

      auto itObs = it->second.observations.begin();
      if (itObs->first.frameId < minValidStateID) {
        // this assume that it->second.observations is an ordered map
        minValidStateID = itObs->first.frameId;
      }

      // update coordinates of map points, this is only necessary when
      // (1) they are used to predict the points projection in new frames OR
      // (2) to visualize the point quality
      std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
          obsInPixel;
      std::vector<uint64_t> frameIds;
      std::vector<double> vRi;  // std noise in pixels
      Eigen::Vector4d v4Xhomog;
      const int camIdx = 0;
      std::shared_ptr<okvis::cameras::CameraBase> tempCameraGeometry =
          camera_rig_.getCameraGeometry(camIdx);

      bool bSucceeded =
          triangulateAMapPoint(it->second, obsInPixel, frameIds, v4Xhomog, vRi,
                               tempCameraGeometry, T_SC0_);
      if (bSucceeded) {
        it->second.quality = 1.0;
        it->second.pointHomog = v4Xhomog;
      } else {
        it->second.quality = 0.0;
      }
    }
    updateLandmarksTimer.stop();
  }

  if (verbose) {
    LOG(INFO) << mapPtr_->summary.FullReport();
  }
}

}  // namespace okvis
