#include <msckf/TFVIO.hpp>

#include <glog/logging.h>

#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/IdProvider.hpp>
#include <okvis/MultiFrame.hpp>

#include <msckf/CameraTimeParamBlock.hpp>
#include <msckf/EpipolarJacobian.hpp>
#include <msckf/EuclideanParamBlock.hpp>
#include <msckf/FilterHelper.hpp>
#include <msckf/ImuOdometry.h>
#include <msckf/PointLandmarkModels.hpp>
#include <msckf/PreconditionedEkfUpdater.h>

DECLARE_bool(use_IEKF);
DEFINE_int32(
    two_view_obs_seq_type, 2,
    "0 the entire feature track of a landmark is used to "
    "compose two-view constraints which are used in one filter update step "
    "as the landmark disappears; "
    "1, use the latest two observations of a landmark to "
    "form one two-view constraint in one filter update step; "
    "2, use the fixed head observation and "
    "the receding tail observation of a landmark to "
    "form one two-view constraint in one filter update step");

/// \brief okvis Main namespace of this package.
namespace okvis {

TFVIO::TFVIO(std::shared_ptr<okvis::ceres::Map> mapPtr)
    : HybridFilter(mapPtr) {}

// The default constructor.
TFVIO::TFVIO() {}

TFVIO::~TFVIO() {}

bool TFVIO::applyMarginalizationStrategy(
    size_t /*numKeyframes*/, size_t /*numImuFrames*/,
    okvis::MapPointVector& /*removedLandmarks*/) {
  std::vector<uint64_t> removeFrames;
  std::map<uint64_t, States>::reverse_iterator rit = statesMap_.rbegin();
  while (rit != statesMap_.rend()) {
    if (rit->first < minValidStateId_) {
      removeFrames.push_back(rit->second.id);
    }
    ++rit;
  }

  // remove features tracked no more
  for (PointMap::iterator pit = landmarksMap_.begin();
       pit != landmarksMap_.end();) {
    const MapPoint& mapPoint = pit->second;
    if (mapPoint.residualizeCase == NotInState_NotTrackedNow) {
      ++mTrackLengthAccumulator[mapPoint.observations.size()];
      for (std::map<okvis::KeypointIdentifier, uint64_t>::const_iterator it =
               mapPoint.observations.begin();
           it != mapPoint.observations.end(); ++it) {
        if (it->second) {
          mapPtr_->removeResidualBlock(
              reinterpret_cast<::ceres::ResidualBlockId>(it->second));
        }
        const KeypointIdentifier& kpi = it->first;
        auto mfp = multiFramePtrMap_.find(kpi.frameId);
        OKVIS_ASSERT_TRUE(Exception, mfp != multiFramePtrMap_.end(), "frame id not found in frame map!");
        mfp->second->setLandmarkId(kpi.cameraIndex, kpi.keypointIndex, 0);
      }
      mapPtr_->removeParameterBlock(pit->first);
      pit = landmarksMap_.erase(pit);
    } else {
      ++pit;
    }
  }

  for (size_t k = 0; k < removeFrames.size(); ++k) {
    okvis::Time removedStateTime = removeState(removeFrames[k]);
    inertialMeasForStates_.pop_front(removedStateTime - half_window_);
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
  updateCovarianceIndex();
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
      static_cast<RetrieveObsSeqType>(FLAGS_two_view_obs_seq_type);
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

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Hi(
        0, featureVariableDimen);
    Eigen::Matrix<double, Eigen::Dynamic, 1> ri;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Ri;

    bool isValidJacobian = featureJacobianEpipolar(it->second, &Hi, &ri, &Ri, seqType);
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

void TFVIO::optimize(size_t /*numIter*/, size_t /*numThreads*/, bool verbose) {
  uint64_t currFrameId = currentFrameId();
  OKVIS_ASSERT_EQ(Exception, covariance_.rows() - startIndexOfClonedStates(),
                  (int)(kClonedStateMinimalDimen * statesMap_.size()),
                  "Inconsistent covDim and number of states");

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
        minValidStateId_ = getMinValidStateId();
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
      minValidStateId_ = getMinValidStateId();
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
    const int camIdx = 0;
    const okvis::kinematics::Transformation T_SC0 = camera_rig_.getCameraExtrinsic(camIdx);
    minValidStateId_ = getMinValidStateId();
    for (auto it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
      if (it->second.residualizeCase == NotInState_NotTrackedNow) continue;
      // this happens with a just inserted landmark without triangulation.
      if (it->second.observations.size() < 2) continue;

      // update coordinates of map points, this is only necessary when
      // (1) they are used to predict the points projection in new frames OR
      // (2) to visualize the point quality
      std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
          obsInPixel;
      std::vector<double> vRi;  // std noise in pixels
      const int camIdx = 0;
      std::shared_ptr<const okvis::cameras::CameraBase> tempCameraGeometry =
          camera_rig_.getCameraGeometry(camIdx);

      msckf::PointLandmark pointLandmark(msckf::HomogeneousPointParameterization::kModelId);
      msckf::PointSharedData psd;
      msckf::TriangulationStatus status =
          triangulateAMapPoint(it->second, obsInPixel, pointLandmark, vRi,
                               tempCameraGeometry, T_SC0, &psd, nullptr, false);
      if (status.triangulationOk) {
        it->second.quality = 1.0;
        it->second.pointHomog = Eigen::Map<Eigen::Vector4d>(pointLandmark.data(), 4);
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
