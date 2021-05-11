#include <swift_vio/TFVIO.hpp>

#include <glog/logging.h>

#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/IdProvider.hpp>
#include <okvis/MultiFrame.hpp>

#include <swift_vio/ceres/CameraTimeParamBlock.hpp>
#include <swift_vio/EpipolarJacobian.hpp>
#include <swift_vio/ceres/EuclideanParamBlock.hpp>
#include <swift_vio/FilterHelper.hpp>

#include <swift_vio/PointLandmarkModels.hpp>
#include <swift_vio/EkfUpdater.h>

DECLARE_bool(use_IEKF);
DEFINE_int32(
    two_view_obs_seq_type, 0,
    "0 the entire feature track of a landmark is used to "
    "compose two-view constraints which are used in one filter update step "
    "as the landmark disappears; "
    "1, use the latest two observations of a landmark to "
    "form one two-view constraint in one filter update step; "
    "2, use the fixed head observation and "
    "the receding tail observation of a landmark to "
    "form one two-view constraint in one filter update step");

namespace swift_vio {
TFVIO::TFVIO(std::shared_ptr<okvis::ceres::Map> mapPtr)
    : HybridFilter(mapPtr),
      minValidStateId_(0u) {}

// The default constructor.
TFVIO::TFVIO() :
  minValidStateId_(0u) {}

TFVIO::~TFVIO() {}

bool TFVIO::applyMarginalizationStrategy(
    size_t /*numKeyframes*/, size_t /*numImuFrames*/,
    okvis::MapPointVector& removedLandmarks) {
  std::vector<uint64_t> removeFrames;
  std::map<uint64_t, States>::reverse_iterator rit = statesMap_.rbegin();
  while (rit != statesMap_.rend()) {
    if (rit->first < minValidStateId_) {
      removeFrames.push_back(rit->second.id);
    }
    ++rit;
  }

  // remove features tracked no more
  for (okvis::PointMap::iterator pit = landmarksMap_.begin();
       pit != landmarksMap_.end();) {
    const okvis::MapPoint& mapPoint = pit->second;
    if (mapPoint.shouldRemove(pointLandmarkOptions_.maxHibernationFrames)) {
      ++mTrackLengthAccumulator[mapPoint.observations.size()];
      for (std::map<okvis::KeypointIdentifier, uint64_t>::const_iterator it =
               mapPoint.observations.begin();
           it != mapPoint.observations.end(); ++it) {
        if (it->second) {
          mapPtr_->removeResidualBlock(
              reinterpret_cast<::ceres::ResidualBlockId>(it->second));
        }
        const okvis::KeypointIdentifier& kpi = it->first;
        auto mfp = multiFramePtrMap_.find(kpi.frameId);
        OKVIS_ASSERT_TRUE(Exception, mfp != multiFramePtrMap_.end(), "frame id not found in frame map!");
        mfp->second->setLandmarkId(kpi.cameraIndex, kpi.keypointIndex, 0);
      }
      mapPtr_->removeParameterBlock(pit->first);
      removedLandmarks.push_back(pit->second);
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

  size_t startIndex = startIndexOfClonedStatesFast();
  size_t finishIndex = startIndex + numRemovedStates * 9;
  CHECK_NE(finishIndex, covariance_.rows())
      << "Never remove the covariance of the lastest state";
  FilterHelper::pruneSquareMatrix(startIndex, finishIndex, &covariance_);
  updateCovarianceIndex();
  return true;
}

int TFVIO::computeStackedJacobianAndResidual(
    Eigen::MatrixXd* T_H, Eigen::Matrix<double, Eigen::Dynamic, 1>* r_q,
    Eigen::MatrixXd* R_q) {
  // compute and stack Jacobians and Residuals for landmarks observed in current
  // frame
  const int camParamStartIndex = startIndexOfCameraParamsFast(0u);
  int featureVariableDimen = covariance_.rows() - camParamStartIndex;
  int dimH[2] = {0, featureVariableDimen};
  const Eigen::MatrixXd variableCov = covariance_.block(
      camParamStartIndex, camParamStartIndex, dimH[1], dimH[1]);

  // containers of Jacobians of measurements
  Eigen::AlignedVector<Eigen::Matrix<double, -1, 1>> vr;
  Eigen::AlignedVector<Eigen::MatrixXd> vH;
  Eigen::AlignedVector<Eigen::MatrixXd> vR;
  RetrieveObsSeqType seqType =
      static_cast<RetrieveObsSeqType>(FLAGS_two_view_obs_seq_type);
  for (auto it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
    const size_t nNumObs = it->second.observations.size();
    if (seqType == ENTIRE_TRACK) {
      if (it->second.status.measurementType != FeatureTrackStatus::kMsckfTrack) {
        continue;
      }
    } else {
      if (!it->second.trackedInCurrentFrame(currentFrameId()) || nNumObs < pointLandmarkOptions_.minTrackLengthForMsckf) {
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
  Eigen::Matrix<double, -1, 1> r(dimH[0], 1);
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(dimH[0], dimH[0]);
  FilterHelper::stackJacobianAndResidual(vH, vr, vR, &H, &r, &R);
  FilterHelper::shrinkResidual(H, r, R, T_H, r_q, R_q);
  return dimH[0];
}

uint64_t TFVIO::getMinValidStateId() const {
  uint64_t currentFrameId = statesMap_.rbegin()->first;
  uint64_t minStateId = currentFrameId;
  for (auto it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
    if (it->second.status.measurementType == FeatureTrackStatus::kPremature) {
      auto itObs = it->second.observations.begin();
      if (itObs->first.frameId < minStateId) {
        minStateId = itObs->first.frameId;
      }
    }
  }
  // We keep at least one keyframe which is required for visualization.
  uint64_t lastKeyframeId = currentKeyframeId();
  // Also keep at least numImuFrames frames.
  uint64_t keepFrameId(0u);
  size_t i = 0u;
  size_t numFrameToKeep = optimizationOptions_.numImuFrames + optimizationOptions_.numKeyframes;
  for (std::map<uint64_t, States>::const_reverse_iterator rit = statesMap_.rbegin();
       rit != statesMap_.rend(); ++rit) {
    keepFrameId = rit->first;
    ++i;
    if (i == numFrameToKeep) {
      break;
    }
  }
  return std::min(minStateId, std::min(lastKeyframeId, keepFrameId));
}


void TFVIO::optimize(size_t /*numIter*/, size_t /*numThreads*/, bool verbose) {
  uint64_t currFrameId = currentFrameId();
  OKVIS_ASSERT_EQ(Exception, covariance_.rows() - startIndexOfClonedStatesFast(),
                  kClonedStateMinimalDimen * statesMap_.size(),
                  "Inconsistent covDim and number of states");

  // mark tracks of features that are not tracked in current frame
  int numTracked = 0;
  int featureVariableDimen = cameraParamsMinimalDimFast(0u) +
                             kClonedStateMinimalDimen * statesMap_.size();
  int navAndImuParamsDim = navStateAndImuParamsMinimalDim();
  for (okvis::PointMap::iterator it = landmarksMap_.begin();
       it != landmarksMap_.end(); ++it) {
    numTracked += (it->second.trackedInCurrentFrame(currFrameId) ? 1 : 0);
    it->second.updateStatus(currFrameId, pointLandmarkOptions_.minTrackLengthForMsckf,
                            std::numeric_limits<std::size_t>::max());
  }
  trackingRate_ = static_cast<double>(numTracked) /
                  static_cast<double>(landmarksMap_.size());

  if (FLAGS_use_IEKF) {
    updateIekf(navAndImuParamsDim, featureVariableDimen);
  } else {
    updateEkf(navAndImuParamsDim, featureVariableDimen);
  }
  if (numResiduals_ == 0) {
    minValidStateId_ = getMinValidStateId();
    return;
  }

  // update landmarks that are tracked in the current frame(the newly inserted
  // state)
  {
    updateLandmarksTimer.start();
    minValidStateId_ = getMinValidStateId();
    for (auto it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
      if (it->second.shouldRemove(pointLandmarkOptions_.maxHibernationFrames)) continue;
      // this happens with a just inserted landmark without triangulation.
      if (it->second.observations.size() < 2) continue;

      // update coordinates of map points, this is only necessary when
      // (1) they are used to predict the points projection in new frames OR
      // (2) to visualize the point quality
      std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
          obsInPixel;
      std::vector<double> vRi;  // std noise in pixels

      swift_vio::PointLandmark pointLandmark(swift_vio::HomogeneousPointParameterization::kModelId);
      swift_vio::PointSharedData psd;
      swift_vio::TriangulationStatus status =
          triangulateAMapPoint(it->second, obsInPixel, pointLandmark, vRi,
                               &psd, nullptr, false);
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
}  // namespace swift_vio
