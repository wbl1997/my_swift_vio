#include <msckf/SlidingWindowSmoother.hpp>

#include <glog/logging.h>

#include <okvis/assert_macros.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/RelativePoseError.hpp>
#include <okvis/ceres/SpeedAndBiasError.hpp>
#include <okvis/IdProvider.hpp>
#include <okvis/MultiFrame.hpp>

#include <msckf/EuclideanParamBlock.hpp>
#include <msckf/FeatureTriangulation.hpp>
#include <msckf/FilterHelper.hpp>

#include <msckf/MeasurementJacobianMacros.hpp>
#include <msckf/PointLandmark.hpp>
#include <msckf/PointLandmarkModels.hpp>
#include <msckf/PointSharedData.hpp>
#include <msckf/PreconditionedEkfUpdater.h>

#include <msckf/triangulate.h>
#include <msckf/triangulateFast.hpp>


DECLARE_bool(use_mahalanobis);
DECLARE_bool(use_first_estimate);
DECLARE_bool(use_RK4);

DECLARE_double(max_proj_tolerance);

DECLARE_bool(use_IEKF);

/// \brief okvis Main namespace of this package.
namespace okvis {

SlidingWindowSmoother::SlidingWindowSmoother(std::shared_ptr<okvis::ceres::Map> mapPtr)
    : HybridFilter(mapPtr) {}

// The default constructor.
SlidingWindowSmoother::SlidingWindowSmoother() {}

SlidingWindowSmoother::~SlidingWindowSmoother() {}

void SlidingWindowSmoother::findRedundantCamStates(
    std::vector<uint64_t>* rm_cam_state_ids) {
  int closeFrames(0), oldFrames(0);
  rm_cam_state_ids->clear();
  rm_cam_state_ids->reserve(minCulledFrames_);
  for (auto rit = ++statesMap_.rbegin(); rit != statesMap_.rend(); ++rit) {
    if (rm_cam_state_ids->size() >= minCulledFrames_) {
      break;
    }
    if (!rit->second.isKeyframe) {
      rm_cam_state_ids->push_back(rit->first);
      ++closeFrames;
    }
  }
  if (rm_cam_state_ids->size() < minCulledFrames_) {
    for (auto it = statesMap_.begin(); it != --statesMap_.end(); ++it) {
      if (it->second.isKeyframe) {
        rm_cam_state_ids->push_back(it->first);
        ++oldFrames;
      }
      if (rm_cam_state_ids->size() >= minCulledFrames_) {
        break;
      }
    }
  }

  sort(rm_cam_state_ids->begin(), rm_cam_state_ids->end());
  return;
}

int SlidingWindowSmoother::marginalizeRedundantFrames(size_t maxClonedStates) {
  if (statesMap_.size() < maxClonedStates) {
    return 0;
  }
  std::vector<uint64_t> rm_cam_state_ids;
  findRedundantCamStates(&rm_cam_state_ids);

  size_t nMarginalizedFeatures = 0u;
  int featureVariableDimen = cameraParamsMinimalDimen() +
      kClonedStateMinimalDimen * (statesMap_.size() - 1);
  int startIndexCamParams = startIndexOfCameraParams();
  const Eigen::MatrixXd featureVariableCov =
      covariance_.block(startIndexCamParams, startIndexCamParams,
                        featureVariableDimen, featureVariableDimen);
  int dimH_o[2] = {0, featureVariableDimen};
  // containers of Jacobians of measurements
  std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> vr_o;
  std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> vH_o;
  std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> vR_o;

  // for each map point in the landmarksMap_,
  // see if the landmark is observed in the redundant frames
  for (okvis::PointMap::iterator it = landmarksMap_.begin(); it != landmarksMap_.end();
       ++it) {
    const size_t nNumObs = it->second.observations.size();
    // this feature has been marginalized earlier in optimize()
    if (it->second.residualizeCase ==
            NotInState_NotTrackedNow ||
        nNumObs < minCulledFrames_) {
      continue;
    }

    std::vector<uint64_t> involved_cam_state_ids;
    auto obsMap = it->second.observations;
    auto obsSearchStart = obsMap.begin();
    for (auto camStateId : rm_cam_state_ids) {
      auto obsIter = std::find_if(obsSearchStart, obsMap.end(),
                                  msckf::IsObservedInFrame(camStateId));
      if (obsIter != obsMap.end()) {
        involved_cam_state_ids.emplace_back(camStateId);
        obsSearchStart = obsIter;
        ++obsSearchStart;
      }
    }
    if (involved_cam_state_ids.size() < minCulledFrames_) {
      continue;
    }

    Eigen::MatrixXd H_oi;                           //(nObsDim, dimH_o[1])
    Eigen::Matrix<double, Eigen::Dynamic, 1> r_oi;  //(nObsDim, 1)
    Eigen::MatrixXd R_oi;                           //(nObsDim, nObsDim)

    bool isValidJacobian =
        featureJacobian(it->second, H_oi, r_oi, R_oi, &involved_cam_state_ids);
    if (!isValidJacobian) {
      // Do we use epipolar constraints for the marginalized feature
      // observations when they do not exhibit enough disparity? It is probably
      // a overkill.
      continue;
    }

    if (!FilterHelper::gatingTest(H_oi, r_oi, R_oi, featureVariableCov)) {
      continue;
    }

    // TODO(jhuai): check if some observations of a map point
    // has been used for update in order to avoid triangulation again
    // which may harm the filter consistency
    it->second.usedForUpdate = true;
    vr_o.push_back(r_oi);
    vR_o.push_back(R_oi);
    vH_o.push_back(H_oi);
    dimH_o[0] += r_oi.rows();
    ++nMarginalizedFeatures;
  }

  if (nMarginalizedFeatures > 0u) {
    Eigen::MatrixXd H_o =
        Eigen::MatrixXd::Zero(dimH_o[0], featureVariableDimen);
    Eigen::MatrixXd r_o(dimH_o[0], 1);
    Eigen::MatrixXd R_o = Eigen::MatrixXd::Zero(dimH_o[0], dimH_o[0]);
    FilterHelper::stackJacobianAndResidual(vH_o, vr_o, vR_o, &H_o, &r_o, &R_o);
    Eigen::MatrixXd T_H, R_q;
    Eigen::Matrix<double, Eigen::Dynamic, 1> r_q;
    FilterHelper::shrinkResidual(H_o, r_o, R_o, &T_H, &r_q, &R_q);

    // perform filter update covariance and states (EKF)
    DefaultEkfUpdater pceu(covariance_, featureVariableDimen);
    computeKalmanGainTimer.start();
    Eigen::Matrix<double, Eigen::Dynamic, 1> deltaX =
        pceu.computeCorrection(T_H, r_q, R_q);
    computeKalmanGainTimer.stop();
    updateStates(deltaX);

    updateCovarianceTimer.start();
    pceu.updateCovariance(&covariance_);
    updateCovarianceTimer.stop();
  }

  // sanity check
  for (const auto &cam_id : rm_cam_state_ids) {
    int cam_sequence =
        std::distance(statesMap_.begin(), statesMap_.find(cam_id));
    OKVIS_ASSERT_EQ(Exception, cam_sequence, mStateID2CovID_[cam_id], "inconsistent mStateID2CovID_");
  }

  // remove observations in removed frames
  for (okvis::PointMap::iterator it = landmarksMap_.begin();
       it != landmarksMap_.end();) {
    okvis::MapPoint& mapPoint = it->second;
    bool removeAllEpipolarConstraints = false;
    std::map<okvis::KeypointIdentifier, uint64_t>::iterator obsIter =
        mapPoint.observations.begin();
    for (uint64_t camStateId : rm_cam_state_ids) {
      while (obsIter != mapPoint.observations.end() &&
             obsIter->first.frameId < camStateId) {
        ++obsIter;
      }
      while (obsIter != mapPoint.observations.end() &&
             obsIter->first.frameId == camStateId) {
        // in case there are dud observations for the
        // landmark in the same frame
        const KeypointIdentifier& kpi = obsIter->first;
        auto mfp = multiFramePtrMap_.find(kpi.frameId);
        mfp->second->setLandmarkId(kpi.cameraIndex, kpi.keypointIndex, 0);
        if (obsIter->second) {
          mapPtr_->removeResidualBlock(
              reinterpret_cast<::ceres::ResidualBlockId>(obsIter->second));
        } else {
          if (obsIter == mapPoint.observations.begin()) {
            // this is a head obs for epipolar constraints, remove all of them
            removeAllEpipolarConstraints = true;
          }  // else do nothing. This can happen if we removed an epipolar
             // constraint in a previous step and up to now the landmark has not
             // been initialized so its observations are not converted to
             // reprojection errors.
        }
        obsIter = mapPoint.observations.erase(obsIter);
      }
    }
    if (removeAllEpipolarConstraints) {
      for (std::map<okvis::KeypointIdentifier, uint64_t>::iterator obsIter =
               mapPoint.observations.begin();
           obsIter != mapPoint.observations.end(); ++obsIter) {
        if (obsIter->second) {
          ::ceres::ResidualBlockId rid =
              reinterpret_cast<::ceres::ResidualBlockId>(obsIter->second);
          std::shared_ptr<const okvis::ceres::ErrorInterface> err =
              mapPtr_->errorInterfacePtr(rid);
          OKVIS_ASSERT_EQ(Exception, err->residualDim(), 1,
                          "Head obs not associated to a residual means that "
                          "the following are all epipolar constraints");
          mapPtr_->removeResidualBlock(rid);
          obsIter->second = 0u;
        }
      }
    }
    if (mapPoint.observations.size() == 0u) {
      mapPtr_->removeParameterBlock(it->first);
      it = landmarksMap_.erase(it);
    } else {
      ++it;
    }
  }

  // check
//  int count = 0;
//  for (okvis::PointMap::iterator it = landmarksMap_.begin();
//       it != landmarksMap_.end(); ++it) {
//    okvis::MapPoint& mapPoint = it->second;
//    for (uint64_t camStateId : rm_cam_state_ids) {
//      auto obsIter = std::find_if(mapPoint.observations.begin(),
//                                  mapPoint.observations.end(),
//                                  msckf::IsObservedInFrame(camStateId));
//      if (obsIter != mapPoint.observations.end()) {
//        LOG(INFO) << "persist lmk " << mapPoint.id << " frm " << camStateId
//                  << " " << obsIter->first.cameraIndex << " "
//                  << obsIter->first.keypointIndex << " residual "
//                  << std::hex << obsIter->second;
//        ++count;
//      }
//    }
//  }
//  OKVIS_ASSERT_EQ(Exception, count, 0, "found residuals not removed!");

  for (const auto &cam_id : rm_cam_state_ids) {
    auto statesIter = statesMap_.find(cam_id);
    int cam_sequence =
        std::distance(statesMap_.begin(), statesIter);
    int cam_state_start =
        startIndexOfClonedStates() + kClonedStateMinimalDimen * cam_sequence;
    int cam_state_end = cam_state_start + kClonedStateMinimalDimen;

    FilterHelper::pruneSquareMatrix(cam_state_start, cam_state_end,
                                    &covariance_);
    removeState(cam_id);
  }
  updateCovarianceIndex();

  uint64_t firstStateId = statesMap_.begin()->first;
  minValidStateId_ = std::min(minValidStateId_, firstStateId);
  return rm_cam_state_ids.size();
}

bool SlidingWindowSmoother::applyMarginalizationStrategy(
    size_t numKeyframes, size_t numImuFrames,
    okvis::MapPointVector& removedLandmarks) {
  std::vector<uint64_t> removeFrames;
  std::map<uint64_t, States>::reverse_iterator rit = statesMap_.rbegin();
  while (rit != statesMap_.rend()) {
    if (rit->first < minValidStateId_) {
      removeFrames.push_back(rit->second.id);
    }
    ++rit;
  }
  if (removeFrames.size() == 0) {
    marginalizeRedundantFrames(numKeyframes + numImuFrames);
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

  int startIndex = startIndexOfClonedStates();
  int finishIndex = startIndex + numRemovedStates * 9;
  CHECK_NE(finishIndex, covariance_.rows())
      << "Never remove the covariance of the lastest state";
  FilterHelper::pruneSquareMatrix(startIndex, finishIndex, &covariance_);
  updateCovarianceIndex();
  return true;
}

bool SlidingWindowSmoother::measurementJacobianAIDP(
    const Eigen::Vector4d& ab1rho,
    std::shared_ptr<const okvis::cameras::CameraBase> tempCameraGeometry,
    const Eigen::Vector2d& obs,
    int observationIndex,
    std::shared_ptr<const msckf::PointSharedData> pointDataPtr,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* J_x,
    Eigen::Matrix<double, 2, 3>* J_pfi, Eigen::Vector2d* residual) const {
  // compute Jacobians for a measurement in image j of the current feature i
  // C_j is the current frame, Bj refers to the body frame associated with the
  // current frame, Ba refers to body frame associated with the anchor frame,
  // f_i is the feature in consideration

  Eigen::Vector2d imagePoint;  // projected pixel coordinates of the point
                               // ${z_u, z_v}$ in pixel units
  Eigen::Matrix2Xd
      intrinsicsJacobian;  //$\frac{\partial [z_u, z_v]^T}{\partial(intrinsics)}$
  Eigen::Matrix<double, 2, 3>
      pointJacobian3;  // $\frac{\partial [z_u, z_v]^T}{\partial
                       // p_{f_i}^{C_j}}$

  // $\frac{\partial [z_u, z_v]^T}{\partial(extrinsic, intrinsic, t_d, t_r)}$
  Eigen::Matrix<double, 2, Eigen::Dynamic> J_Xc(2, cameraParamsMinimalDimen());

  Eigen::Matrix<double, 2, kClonedStateMinimalDimen>
      J_XBj;  // $\frac{\partial [z_u, z_v]^T}{delta\p_{B_j}^G, \delta\alpha
              // (of q_{B_j}^G), \delta v_{B_j}^G$
  Eigen::Matrix<double, 3, kClonedStateMinimalDimen>
      factorJ_XBj;  // the second factor of J_XBj, see Michael Andrew Shelley
                    // Master thesis sec 6.5, p.55 eq 6.66
  Eigen::Matrix<double, 3, kClonedStateMinimalDimen> factorJ_XBa;
  Eigen::Vector2d J_td;
  Eigen::Vector2d J_tr;
  // $\frac{\partial [z_u, z_v]^T}{\partial (delta\p_{B_a}^G \alpha of q_{B_a}^G, \delta v_{B_a}^G)}$
  Eigen::Matrix<double, 2, kClonedStateMinimalDimen> J_XBa;

  uint64_t poseId = pointDataPtr->frameId(observationIndex);
  int camIdx = pointDataPtr->cameraIndex(observationIndex);

  int projOptModelId = camera_rig_.getProjectionOptMode(camIdx);
  int extrinsicModelId = camera_rig_.getExtrinsicOptMode(camIdx);
  const okvis::kinematics::Transformation T_SC0 = camera_rig_.getCameraExtrinsic(camIdx);

  double kpN = pointDataPtr->normalizedRow(observationIndex);
  const double featureTime = pointDataPtr->normalizedFeatureTime(observationIndex);

  kinematics::Transformation T_WB = pointDataPtr->T_WBtij(observationIndex);
  Eigen::Vector3d omega_Btij = pointDataPtr->omega_Btij(observationIndex);
  okvis::kinematics::Transformation T_WBa = pointDataPtr->T_WBa_list()[0];
  okvis::kinematics::Transformation T_GA = T_WBa * T_SC0;  // anchor frame to global frame
  okvis::kinematics::Transformation T_CA = (T_WB * T_SC0).inverse() * T_GA;  // anchor frame to current camera frame
  Eigen::Vector3d pfiinC = (T_CA * ab1rho).head<3>();

  cameras::CameraBase::ProjectionStatus status = tempCameraGeometry->project(
      pfiinC, &imagePoint, &pointJacobian3, &intrinsicsJacobian);
  *residual = obs - imagePoint;
  if (status != cameras::CameraBase::ProjectionStatus::Successful) {
    return false;
  } else if (!FLAGS_use_mahalanobis) {
    // some heuristics to defend outliers is used, e.g., ignore correspondences
    // of too large discrepancy between prediction and measurement
    if (std::fabs((*residual)[0]) > FLAGS_max_proj_tolerance ||
        std::fabs((*residual)[1]) > FLAGS_max_proj_tolerance) {
      return false;
    }
  }

  kinematics::Transformation lP_T_WB = pointDataPtr->T_WBtij_ForJacobian(observationIndex);
  Eigen::Vector3d lP_v_WB = pointDataPtr->v_WBtij_ForJacobian(observationIndex);

  double rho = ab1rho[3];
  okvis::kinematics::Transformation T_BcA = lP_T_WB.inverse() * T_GA;
  J_td = pointJacobian3 * T_SC0.C().transpose() *
         (okvis::kinematics::crossMx((T_BcA * ab1rho).head<3>()) *
              omega_Btij -
          T_WB.C().transpose() * lP_v_WB * rho);
  J_tr = J_td * kpN;

  if (fixCameraExtrinsicParams_[camIdx]) {
    if (fixCameraIntrinsicParams_[camIdx]) {
      J_Xc << J_td, J_tr;
    } else {
      ProjectionOptKneadIntrinsicJacobian(projOptModelId, &intrinsicsJacobian);
      J_Xc << intrinsicsJacobian, J_td, J_tr;
    }
  } else {
    Eigen::MatrixXd dpC_dExtrinsic;
    Eigen::Matrix3d R_CfCa = T_CA.C();
    ExtrinsicModel_dpC_dExtrinsic_AIDP(extrinsicModelId, pfiinC,
                                       T_SC0.C().transpose(), &dpC_dExtrinsic,
                                       &R_CfCa, &ab1rho);
    if (fixCameraIntrinsicParams_[camIdx]) {
      J_Xc << pointJacobian3 * dpC_dExtrinsic, J_td, J_tr;
    } else {
      ProjectionOptKneadIntrinsicJacobian(projOptModelId, &intrinsicsJacobian);
      J_Xc << pointJacobian3 * dpC_dExtrinsic, intrinsicsJacobian, J_td, J_tr;
    }
  }
  Eigen::Matrix3d tempM3d;
  tempM3d << T_CA.C().topLeftCorner<3, 2>(), T_CA.r();
  (*J_pfi) = pointJacobian3 * tempM3d;

  Eigen::Vector3d pfinG = (T_GA * ab1rho).head<3>();
  factorJ_XBj << -rho * Eigen::Matrix3d::Identity(),
      okvis::kinematics::crossMx(pfinG - lP_T_WB.r() * rho),
      -rho * Eigen::Matrix3d::Identity() * featureTime;
  J_XBj = pointJacobian3 * (T_WB.C() * T_SC0.C()).transpose() * factorJ_XBj;

  factorJ_XBa.topLeftCorner<3, 3>() = rho * Eigen::Matrix3d::Identity();
  factorJ_XBa.block<3, 3>(0, 3) =
      -okvis::kinematics::crossMx(T_WBa.C() * (T_SC0 * ab1rho).head<3>());
  factorJ_XBa.block<3, 3>(0, 6) = Eigen::Matrix3d::Zero();
  J_XBa = pointJacobian3 * (T_WB.C() * T_SC0.C()).transpose() * factorJ_XBa;

  J_x->setZero();
  const int minCamParamDim = cameraParamsMinimalDimen();
  J_x->topLeftCorner(2, minCamParamDim) = J_Xc;
  std::map<uint64_t, int>::const_iterator poseid_iter =
      mStateID2CovID_.find(poseId);
  int covid = poseid_iter->second;

  uint64_t anchorId = pointDataPtr->anchorIds()[0];
  if (poseId == anchorId) {
    J_x->block<2, 6>(0, minCamParamDim +
                           9 * covid + 3) =
        (J_XBj + J_XBa).block<2, 6>(0, 3);
  } else {
    J_x->block<2, 9>(0, minCamParamDim +
                           9 * covid) = J_XBj;
    std::map<uint64_t, int>::const_iterator anchorid_iter =
        mStateID2CovID_.find(anchorId);
    J_x->block<2, 9>(0, minCamParamDim +
                           9 * anchorid_iter->second) = J_XBa;
  }
  return true;
}

bool SlidingWindowSmoother::measurementJacobian(
    const Eigen::Vector4d& v4Xhomog,
    std::shared_ptr<const okvis::cameras::CameraBase> tempCameraGeometry,
    const Eigen::Vector2d& obs,
    int observationIndex,
    std::shared_ptr<const msckf::PointSharedData> pointDataPtr,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* J_Xc,
    Eigen::Matrix<double, 2, 9>* J_XBj, Eigen::Matrix<double, 2, 3>* J_pfi,
    Eigen::Vector2d* residual) const {
  const Eigen::Vector3d v3Point = v4Xhomog.head<3>();
  // compute Jacobians
  Eigen::Vector2d imagePoint;  // projected pixel coordinates of the point
                               // ${z_u, z_v}$ in pixel units
  Eigen::Matrix2Xd
      intrinsicsJacobian;  //$\frac{\partial [z_u, z_v]^T}{\partial( f_x, f_v,
                           // c_x, c_y, k_1, k_2, p_1, p_2, [k_3])}$
  Eigen::Matrix<double, 2, 3>
      pointJacobian3;  // $\frac{\partial [z_u, z_v]^T}{\partial
                       // p_{f_i}^{C_j}}$

  Eigen::Matrix<double, 3, 9>
      factorJ_XBj;  // the second factor of J_XBj, see Michael Andrew Shelley
                    // Master thesis sec 6.5, p.55 eq 6.66

  Eigen::Vector2d J_td;
  Eigen::Vector2d J_tr;

  int camIdx = pointDataPtr->cameraIndex(observationIndex);
  int projOptModelId = camera_rig_.getProjectionOptMode(camIdx);
  int extrinsicModelId = camera_rig_.getExtrinsicOptMode(camIdx);
  const okvis::kinematics::Transformation T_SC0 = camera_rig_.getCameraExtrinsic(camIdx);
  double featureTime = pointDataPtr->normalizedFeatureTime(observationIndex);
  double kpN = pointDataPtr->normalizedRow(observationIndex);

  kinematics::Transformation T_WB = pointDataPtr->T_WBtij(observationIndex);
  Eigen::Vector3d omega_Btij = pointDataPtr->omega_Btij(observationIndex);

  kinematics::Transformation T_CW = (T_WB * T_SC0).inverse();
  Eigen::Vector4d hp_C = T_CW * v4Xhomog;
  Eigen::Vector3d pfiinC = hp_C.head<3>();
  cameras::CameraBase::ProjectionStatus status = tempCameraGeometry->project(
      pfiinC, &imagePoint, &pointJacobian3, &intrinsicsJacobian);
  *residual = obs - imagePoint;
  if (status != cameras::CameraBase::ProjectionStatus::Successful) {
    return false;
  } else if (!FLAGS_use_mahalanobis) {
    if (std::fabs((*residual)[0]) > FLAGS_max_proj_tolerance ||
        std::fabs((*residual)[1]) > FLAGS_max_proj_tolerance) {
      return false;
    }
  }

  kinematics::Transformation lP_T_WB =
      pointDataPtr->T_WBtij_ForJacobian(observationIndex);
  Eigen::Vector3d lP_v_WBtij =
      pointDataPtr->v_WBtij_ForJacobian(observationIndex);

  J_td = pointJacobian3 * T_SC0.C().transpose() *
         (okvis::kinematics::crossMx((lP_T_WB.inverse() * v4Xhomog).head<3>()) *
              omega_Btij -
          T_WB.C().transpose() * lP_v_WBtij);
  J_tr = J_td * kpN;

  if (fixCameraExtrinsicParams_[camIdx]) {
    if (fixCameraIntrinsicParams_[camIdx]) {
      (*J_Xc) << J_td, J_tr;
    } else {
      ProjectionOptKneadIntrinsicJacobian(projOptModelId, &intrinsicsJacobian);
      (*J_Xc) << intrinsicsJacobian, J_td, J_tr;
    }
  } else {
    Eigen::MatrixXd dpC_dExtrinsic;
    ExtrinsicModel_dpC_dExtrinsic_HPP(extrinsicModelId, hp_C,
                                      T_SC0.C().transpose(), &dpC_dExtrinsic);
    if (fixCameraIntrinsicParams_[camIdx]) {
      (*J_Xc) << pointJacobian3 * dpC_dExtrinsic, J_td, J_tr;
    } else {
      ProjectionOptKneadIntrinsicJacobian(projOptModelId, &intrinsicsJacobian);
      (*J_Xc) << pointJacobian3 * dpC_dExtrinsic, intrinsicsJacobian, J_td,
          J_tr;
    }
  }
  (*J_pfi) = pointJacobian3 * T_CW.C();

  factorJ_XBj << -Eigen::Matrix3d::Identity(),
      okvis::kinematics::crossMx(v3Point - lP_T_WB.r()),
      -Eigen::Matrix3d::Identity() * featureTime;

  (*J_XBj) = (*J_pfi) * factorJ_XBj;
  return true;
}

bool SlidingWindowSmoother::featureJacobianGeneric(
    const MapPoint& mp, Eigen::MatrixXd& H_oi,
    Eigen::Matrix<double, Eigen::Dynamic, 1>& r_oi, Eigen::MatrixXd& R_oi,
    std::vector<uint64_t>* orderedCulledFrameIds) const {
  const int camIdx = 0;
  std::shared_ptr<const okvis::cameras::CameraBase> tempCameraGeometry =
      camera_rig_.getCameraGeometry(camIdx);
  const okvis::kinematics::Transformation T_SC0 =
      camera_rig_.getCameraExtrinsic(camIdx);
  // dimension of variables used in computing feature Jacobians, including
  // camera intrinsics and all cloned states except the most recent one
  // in which the marginalized observations should never occur.
  int featureVariableDimen = cameraParamsMinimalDimen() +
                             kClonedStateMinimalDimen * (statesMap_.size() - 1);
  int residualBlockDim = okvis::cameras::CameraObservationModelResidualDim(
        cameraObservationModelId_);
  // all observations for this feature point
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      obsInPixel;
  std::vector<double> vRi;         // std noise in pixels, 2Nx1
  computeHTimer.start();
  msckf::PointLandmark pointLandmark(landmarkModelId_);

  std::shared_ptr<msckf::PointSharedData> pointDataPtr(new msckf::PointSharedData());
  msckf::TriangulationStatus status = triangulateAMapPoint(
      mp, obsInPixel, pointLandmark, vRi, tempCameraGeometry, T_SC0,
      pointDataPtr.get(), orderedCulledFrameIds, useEpipolarConstraint_);
  if (!status.triangulationOk) {
    computeHTimer.stop();
    return false;
  }
  if (orderedCulledFrameIds) {
    pointDataPtr->removeExtraObservations(*orderedCulledFrameIds, &vRi);
  }
  pointDataPtr->computePoseAndVelocityForJacobians(FLAGS_use_first_estimate);
  pointDataPtr->computeSharedJacobians(cameraObservationModelId_);
  CHECK_NE(statesMap_.rbegin()->first, pointDataPtr->lastFrameId())
      << "The landmark should not be observed by the latest frame in MSCKF.";

  // containers of the above Jacobians for all observations of a mappoint
  std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> vJ_X;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>,
              Eigen::aligned_allocator<Eigen::Matrix<double, Eigen::Dynamic, 3>>>
      vJ_pfi;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>,
              Eigen::aligned_allocator<Eigen::Matrix<double, Eigen::Dynamic, 2>>>
      vJ_n;
  std::vector<Eigen::VectorXd,
              Eigen::aligned_allocator<Eigen::VectorXd>>
      vri;  // residuals for feature i

  size_t numPoses = pointDataPtr->numObservations();
  size_t numValidObs = 0;
  auto itFrameIds = pointDataPtr->begin();
  auto itRoi = vRi.begin();

  // compute Jacobians for a measurement in image j of the current feature i
  for (size_t observationIndex = 0; observationIndex < numPoses; ++observationIndex) {
    Eigen::MatrixXd J_x(residualBlockDim, featureVariableDimen);
    Eigen::Matrix<double, Eigen::Dynamic, 3> J_pfi(residualBlockDim, 3);
    Eigen::Matrix<double, Eigen::Dynamic, 2> J_n(residualBlockDim, 2);
    Eigen::VectorXd residual(residualBlockDim, 1);

    Eigen::Matrix2d obsCov = Eigen::Matrix2d::Identity();
    double imgNoiseStd = *itRoi;
    obsCov(0, 0) = imgNoiseStd * imgNoiseStd;
    imgNoiseStd = *(itRoi + 1);
    obsCov(1, 1) = imgNoiseStd * imgNoiseStd;
    msckf::MeasurementJacobianStatus status = measurementJacobianGeneric(
        pointLandmark, tempCameraGeometry, obsInPixel[observationIndex],
        obsCov, observationIndex, pointDataPtr, &J_x, &J_pfi, &J_n, &residual);
    if (status != msckf::MeasurementJacobianStatus::Successful) {
      if (status == msckf::MeasurementJacobianStatus::
                        AssociateAnchorProjectionFailed ||
          status == msckf::MeasurementJacobianStatus::
                        MainAnchorProjectionFailed) {
        computeHTimer.stop();
        return false;
      }
      ++itFrameIds;
      itRoi += 2;
      continue;
    }

    vri.push_back(residual);
    vJ_X.push_back(J_x);
    vJ_pfi.push_back(J_pfi);
    vJ_n.push_back(J_n);

    ++numValidObs;
    ++itFrameIds;
    itRoi += 2;
  }
  if (numValidObs < minTrackLength_) {
    computeHTimer.stop();
    return false;
  }

  // Now we stack the Jacobians and marginalize the point position related
  // dimensions. In other words, project $H_{x_i}$ onto the nullspace of
  // $H_{f^i}$
  Eigen::MatrixXd H_xi(residualBlockDim * numValidObs, featureVariableDimen);
  Eigen::MatrixXd H_fi(residualBlockDim * numValidObs, 3);
  Eigen::VectorXd ri(residualBlockDim * numValidObs, 1);
  Eigen::MatrixXd Ri =
      Eigen::MatrixXd::Identity(residualBlockDim * numValidObs, residualBlockDim * numValidObs);
  for (size_t saga = 0; saga < numValidObs; ++saga) {
    size_t sagax = saga * residualBlockDim;
    H_xi.block(sagax, 0, residualBlockDim, featureVariableDimen) = vJ_X[saga];
    H_fi.block(sagax, 0, residualBlockDim, 3) = vJ_pfi[saga];
    ri.segment(sagax, residualBlockDim) = vri[saga];
    Ri.block(sagax, sagax, residualBlockDim, residualBlockDim).setIdentity();
  }

  if (cameraObservationModelId_ != okvis::cameras::kEpipolarFactorId) {
    int columnRankHf = status.raysParallel ? 2 : 3;
    // (rDim * n) x ((rDim * n)-columnRankHf), n==numValidObs
    Eigen::MatrixXd nullQ =
        FilterHelper::leftNullspaceWithRankCheck(H_fi, columnRankHf);
    r_oi.noalias() = nullQ.transpose() * ri;
    H_oi.noalias() = nullQ.transpose() * H_xi;
    R_oi = nullQ.transpose() * (Ri * nullQ).eval();
  } else {
    r_oi = ri;
    H_oi = H_xi;
    R_oi = Ri;
  }

  vri.clear();
  vJ_n.clear();
  vJ_pfi.clear();
  vJ_X.clear();
  computeHTimer.stop();
  return true;
}

msckf::MeasurementJacobianStatus SlidingWindowSmoother::measurementJacobianGeneric(
    const msckf::PointLandmark& pointLandmark,
    std::shared_ptr<const okvis::cameras::CameraBase> baseCameraGeometry,
    const Eigen::Vector2d& obs,
    const Eigen::Matrix2d& obsCov, int observationIndex,
    std::shared_ptr<const msckf::PointSharedData> pointDataPtr,
    Eigen::MatrixXd* J_X, Eigen::Matrix<double, Eigen::Dynamic, 3>* J_pfi,
    Eigen::Matrix<double, Eigen::Dynamic, 2>* J_n,
    Eigen::VectorXd* residual) const {
  int camIdx = pointDataPtr->cameraIndex(observationIndex);
  okvis::cameras::NCameraSystem::DistortionType distortionType =
      camera_rig_.getDistortionType(camIdx);
  int projOptModelId = camera_rig_.getProjectionOptMode(camIdx);
  int extrinsicModelId = camera_rig_.getExtrinsicOptMode(camIdx);
  int imuModelId = imu_rig_.getModelId(0);
  msckf::MeasurementJacobianStatus status =
      msckf::MeasurementJacobianStatus::GeneralProjectionFailed;
  switch (imuModelId) {
    IMU_MODEL_CHAIN_CASE(Imu_BG_BA)
    IMU_MODEL_CHAIN_CASE(Imu_BG_BA_TG_TS_TA)
    IMU_MODEL_CHAIN_CASE(ScaledMisalignedImu)
    default:
      MODEL_DOES_NOT_EXIST_EXCEPTION
      break;
  }
#undef IMU_MODEL_CHAIN_CASE
#undef EXTRINSIC_MODEL_CHAIN_CASE
#undef DISTORTION_MODEL_CHAIN_CASE
#undef PROJECTION_INTRINSIC_MODEL_CHAIN_CASE
  return status;
}

// assume the rolling shutter camera reads data row by row, and rows are
//     aligned with the width of a frame
bool SlidingWindowSmoother::featureJacobian(const MapPoint &mp, Eigen::MatrixXd &H_oi,
                        Eigen::Matrix<double, Eigen::Dynamic, 1> &r_oi,
                        Eigen::MatrixXd &R_oi,
                        std::vector<uint64_t>* orderedCulledFrameIds) const {
  if (landmarkModelId_ == msckf::ParallaxAngleParameterization::kModelId ||
      cameraObservationModelId_ != okvis::cameras::kReprojectionErrorId) {
    return featureJacobianGeneric(mp, H_oi, r_oi, R_oi, orderedCulledFrameIds);
  }
  const int camIdx = 0;
  std::shared_ptr<const okvis::cameras::CameraBase> tempCameraGeometry =
      camera_rig_.getCameraGeometry(camIdx);
  const okvis::kinematics::Transformation T_SC0 = camera_rig_.getCameraExtrinsic(camIdx);
  // dimension of variables used in computing feature Jacobians, including
  // camera intrinsics and all cloned states except the most recent one
  // in which the marginalized observations should never occur.
  int featureVariableDimen = cameraParamsMinimalDimen() +
      kClonedStateMinimalDimen * (statesMap_.size() - 1);

  // all observations for this feature point
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      obsInPixel;
  std::vector<double> vRi; // std noise in pixels
  computeHTimer.start();
  if (landmarkModelId_ == msckf::InverseDepthParameterization::kModelId) {
    // The landmark is expressed with AIDP in the anchor frame    
    // if the feature is lost in current frame, the anchor frame is chosen
    // as the last frame observing the point.

    msckf::PointLandmark pointLandmark(landmarkModelId_);
    std::shared_ptr<msckf::PointSharedData> pointDataPtr(new msckf::PointSharedData());

    msckf::TriangulationStatus status = triangulateAMapPoint(
        mp, obsInPixel, pointLandmark, vRi, tempCameraGeometry, T_SC0,
        pointDataPtr.get(), orderedCulledFrameIds, useEpipolarConstraint_);

    Eigen::Map<Eigen::Vector4d> v4Xhomog(pointLandmark.data(), 4);
    if (!status.triangulationOk) {
      computeHTimer.stop();
      return false;
    }
    if (orderedCulledFrameIds) {
      pointDataPtr->removeExtraObservations(*orderedCulledFrameIds, &vRi);
    }
    pointDataPtr->computePoseAndVelocityForJacobians(FLAGS_use_first_estimate);
    pointDataPtr->computeSharedJacobians(cameraObservationModelId_);
    CHECK_NE(statesMap_.rbegin()->first, pointDataPtr->lastFrameId())
        << "The landmark should not be observed by the latest frame in MSCKF.";

    Eigen::Vector4d ab1rho = v4Xhomog;
    CHECK_GT(ab1rho[2], 0) << "Negative depth in anchor frame";
    ab1rho /= ab1rho[2];  //[\alpha = X/Z, \beta= Y/Z, 1, \rho=1/Z] in the
                          // anchor frame

    // containers of the above Jacobians for all observations of a mappoint
    std::vector<
        Eigen::Matrix<double, 2, Eigen::Dynamic>,
        Eigen::aligned_allocator<Eigen::Matrix<double, 2, Eigen::Dynamic>>>
        vJ_X;
    std::vector<Eigen::Matrix<double, 2, 3>,
                Eigen::aligned_allocator<Eigen::Matrix<double, 2, 3>>>
        vJ_pfi;
    std::vector<Eigen::Matrix<double, 2, 1>,
                Eigen::aligned_allocator<Eigen::Matrix<double, 2, 1>>>
        vri;  // residuals for feature i

    size_t numPoses = pointDataPtr->numObservations();
    size_t numValidObs = 0;
    auto itFrameIds = pointDataPtr->begin();
    auto itRoi = vRi.begin();
    // compute Jacobians for a measurement in image j of the current feature i
    for (size_t observationIndex = 0; observationIndex < numPoses; ++observationIndex) {
      Eigen::Matrix<double, 2, Eigen::Dynamic> J_x(2, featureVariableDimen);
      // $\frac{\partial [z_u, z_v]^T}{\partial [\alpha, \beta, \rho]}$
      Eigen::Matrix<double, 2, 3> J_pfi;
      Eigen::Vector2d residual;

      bool validJacobian = measurementJacobianAIDP(
          ab1rho, tempCameraGeometry, obsInPixel[observationIndex],
          observationIndex, pointDataPtr, &J_x, &J_pfi, &residual);
      if (!validJacobian) {
          ++itFrameIds;
          itRoi = vRi.erase(itRoi);
          itRoi = vRi.erase(itRoi);
          continue;
      }

      vri.push_back(residual);
      vJ_X.push_back(J_x);
      vJ_pfi.push_back(J_pfi);

      ++numValidObs;
      ++itFrameIds;
      itRoi += 2;
    }
    if (numValidObs < minTrackLength_) {
      computeHTimer.stop();
      return false;
    }

    // Now we stack the Jacobians and marginalize the point position related
    // dimensions. In other words, project $H_{x_i}$ onto the nullspace of
    // $H_{f^i}$

    Eigen::MatrixXd H_xi(2 * numValidObs, featureVariableDimen);
    Eigen::MatrixXd H_fi(2 * numValidObs, 3);
    Eigen::Matrix<double, Eigen::Dynamic, 1> ri(2 * numValidObs, 1);
    Eigen::MatrixXd Ri =
        Eigen::MatrixXd::Identity(2 * numValidObs, 2 * numValidObs);
    for (size_t saga = 0; saga < numValidObs; ++saga) {
      size_t saga2 = saga * 2;
      H_xi.block(saga2, 0, 2, featureVariableDimen) = vJ_X[saga];
      H_fi.block<2, 3>(saga2, 0) = vJ_pfi[saga];
      ri.segment<2>(saga2) = vri[saga];
      Ri(saga2, saga2) = vRi[saga2] * vRi[saga2];
      Ri(saga2 + 1, saga2 + 1) = vRi[saga2 + 1] * vRi[saga2 + 1];
    }

    int columnRankHf = status.raysParallel ? 2 : 3;
    // 2nx(2n-CR), n==numValidObs
    Eigen::MatrixXd nullQ = FilterHelper::leftNullspaceWithRankCheck(H_fi, columnRankHf);
    r_oi.noalias() = nullQ.transpose() * ri;
    H_oi.noalias() = nullQ.transpose() * H_xi;
    R_oi = nullQ.transpose() * (Ri * nullQ).eval();

    vri.clear();
    vJ_pfi.clear();
    vJ_X.clear();
    computeHTimer.stop();
    return true;
  } else {
    msckf::PointLandmark pointLandmark(landmarkModelId_);
    // The landmark is expressed with Euclidean coordinates in the global frame
    std::shared_ptr<msckf::PointSharedData> pointDataPtr(new msckf::PointSharedData());
    msckf::TriangulationStatus status = triangulateAMapPoint(
        mp, obsInPixel, pointLandmark, vRi, tempCameraGeometry, T_SC0,
        pointDataPtr.get(), orderedCulledFrameIds, useEpipolarConstraint_);
    Eigen::Map<Eigen::Vector4d> v4Xhomog(pointLandmark.data(), 4);
    if (!status.triangulationOk) {
      computeHTimer.stop();
      return false;
    }
    if (orderedCulledFrameIds) {
      pointDataPtr->removeExtraObservations(*orderedCulledFrameIds, &vRi);
    }
    pointDataPtr->computePoseAndVelocityForJacobians(FLAGS_use_first_estimate);
    pointDataPtr->computeSharedJacobians(cameraObservationModelId_);
    CHECK_NE(statesMap_.rbegin()->first, pointDataPtr->lastFrameId())
        << "The landmark should not be observed by the latest frame in MSCKF.";

    // containers of the above Jacobians for all observations of a mappoint
    const int cameraParamsDimen = cameraParamsMinimalDimen();
    std::vector<
        Eigen::Matrix<double, 2, Eigen::Dynamic>,
        Eigen::aligned_allocator<Eigen::Matrix<double, 2, Eigen::Dynamic>>>
        vJ_Xc;
    std::vector<Eigen::Matrix<double, 2, kClonedStateMinimalDimen>,
                Eigen::aligned_allocator<Eigen::Matrix<double, 2, kClonedStateMinimalDimen>>>
        vJ_XBj;
    std::vector<Eigen::Matrix<double, 2, 3>,
                Eigen::aligned_allocator<Eigen::Matrix<double, 2, 3>>>
        vJ_pfi;
    std::vector<Eigen::Matrix<double, 2, 1>,
                Eigen::aligned_allocator<Eigen::Matrix<double, 2, 1>>>
        vri;  // residuals for feature i

    std::vector<uint64_t> vFrameIds;

    size_t numPoses = pointDataPtr->numObservations();

    vJ_Xc.reserve(numPoses);
    vJ_XBj.reserve(numPoses);
    vJ_pfi.reserve(numPoses);
    vri.reserve(numPoses);
    vFrameIds.reserve(numPoses);


    auto itFrameIds = pointDataPtr->begin();
    auto itRoi = vRi.begin();
    // compute Jacobians for a measurement in image j of the current feature i
    for (size_t observationIndex = 0; observationIndex < numPoses; ++observationIndex) {
      // $\frac{\partial [z_u, z_v]^T}{\partial(extrinsic, intrinsic, td, tr)}$
      Eigen::Matrix<double, 2, Eigen::Dynamic> J_Xc(2, cameraParamsMinimalDimen());
      Eigen::Matrix<double, 2, kClonedStateMinimalDimen>
          J_XBj;  // $\frac{\partial [z_u, z_v]^T}{delta\p_{B_j}^G, \delta\alpha
                  // (of q_{B_j}^G), \delta v_{B_j}^G$
      Eigen::Matrix<double, 2, 3>
          J_pfi;  // $\frac{\partial [z_u, z_v]^T}{\partial p_{f_i}^G}$
      Eigen::Vector2d residual;

      bool validJacobian = measurementJacobian(
          v4Xhomog, tempCameraGeometry, obsInPixel[observationIndex], observationIndex, pointDataPtr,
          &J_Xc, &J_XBj, &J_pfi, &residual);
      if (!validJacobian) {
          ++itFrameIds;
          itRoi = vRi.erase(itRoi);
          itRoi = vRi.erase(itRoi);
          continue;
      }
      vJ_Xc.push_back(J_Xc);
      vJ_XBj.push_back(J_XBj);
      vJ_pfi.push_back(J_pfi);
      vri.push_back(residual);
      vFrameIds.push_back(itFrameIds->frameId);
      ++itFrameIds;
      itRoi += 2;
    }
    size_t numValidObs = vFrameIds.size();
    if (numValidObs < minTrackLength_) {
      computeHTimer.stop();
      return false;
    }

    // Now we stack the Jacobians and marginalize the point position related
    // dimensions. In other words, project $H_{x_i}$ onto the nullspace of
    // $H_{f^i}$
    Eigen::MatrixXd H_xi =
        Eigen::MatrixXd::Zero(2 * numValidObs, featureVariableDimen);
    Eigen::MatrixXd H_fi = Eigen::MatrixXd(2 * numValidObs, 3);
    Eigen::Matrix<double, Eigen::Dynamic, 1> ri =
        Eigen::Matrix<double, Eigen::Dynamic, 1>(2 * numValidObs, 1);
    Eigen::MatrixXd Ri =
        Eigen::MatrixXd::Identity(2 * numValidObs, 2 * numValidObs);
    for (size_t saga = 0; saga < numValidObs; ++saga) {
      size_t saga2 = saga * 2;
      H_xi.block(saga2, 0, 2, cameraParamsDimen) = vJ_Xc[saga];
      std::map<uint64_t, int>::const_iterator poseCovIndexIter =
          mStateID2CovID_.find(vFrameIds[saga]);
      H_xi.block<2, kClonedStateMinimalDimen>(
          saga2, cameraParamsDimen +
                     kClonedStateMinimalDimen * poseCovIndexIter->second) =
          vJ_XBj[saga];
      H_fi.block<2, 3>(saga2, 0) = vJ_pfi[saga];
      ri.segment<2>(saga2) = vri[saga];
      Ri(saga2, saga2) *= (vRi[saga2] * vRi[saga2]);
      Ri(saga2 + 1, saga2 + 1) *= (vRi[saga2 + 1] * vRi[saga2 + 1]);
    }

    int columnRankHf = status.raysParallel ? 2 : 3;
    // 2nx(2n-CR), n==numValidObs
    Eigen::MatrixXd nullQ = FilterHelper::leftNullspaceWithRankCheck(H_fi, columnRankHf);

    r_oi.noalias() = nullQ.transpose() * ri;
    H_oi.noalias() = nullQ.transpose() * H_xi;
    R_oi = nullQ.transpose() * (Ri * nullQ).eval();

    vri.clear();
    vJ_pfi.clear();
    vJ_XBj.clear();
    vJ_Xc.clear();
    computeHTimer.stop();
    return true;
  }
}

int SlidingWindowSmoother::computeStackedJacobianAndResidual(
    Eigen::MatrixXd *T_H, Eigen::Matrix<double, Eigen::Dynamic, 1> *r_q,
    Eigen::MatrixXd *R_q) const {
  // compute and stack Jacobians and Residuals for landmarks observed no more
  size_t nMarginalizedFeatures = 0;
  int culledPoints[2] = {0};
  int featureVariableDimen = cameraParamsMinimalDimen() +
      kClonedStateMinimalDimen * (statesMap_.size() - 1);
  int dimH_o[2] = {0, featureVariableDimen};
  const int camParamStartIndex = startIndexOfCameraParams();
  const Eigen::MatrixXd variableCov =
      covariance_.block(camParamStartIndex, camParamStartIndex, dimH_o[1], dimH_o[1]);
  // containers of Jacobians of measurements
  std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> vr_o;
  std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> vH_o;
  std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> vR_o;

  for (auto it = landmarksMap_.begin(); it != landmarksMap_.end();
       ++it) {
    const size_t nNumObs = it->second.observations.size();
    if (it->second.residualizeCase !=
            NotInState_NotTrackedNow ||
        nNumObs < minTrackLength_) {
      continue;
    }

    Eigen::MatrixXd H_oi;                           //(nObsDim, dimH_o[1])
    H_oi.resize(0, featureVariableDimen);
    Eigen::Matrix<double, Eigen::Dynamic, 1> r_oi;  //(nObsDim, 1)
    Eigen::MatrixXd R_oi;                           //(nObsDim, nObsDim)
    bool isValidJacobian = featureJacobian(it->second, H_oi, r_oi, R_oi);
    if (!isValidJacobian) {
      isValidJacobian = useEpipolarConstraint_
                            ? featureJacobianEpipolar(it->second, &H_oi, &r_oi,
                                                      &R_oi, ENTIRE_TRACK)
                            : isValidJacobian;
      if (!isValidJacobian) {
        ++culledPoints[0];
        continue;
      }
    }

    if (!FilterHelper::gatingTest(H_oi, r_oi, R_oi, variableCov)) {
        ++culledPoints[1];
        continue;
    }

    vr_o.push_back(r_oi);
    vR_o.push_back(R_oi);
    vH_o.push_back(H_oi);
    dimH_o[0] += r_oi.rows();
    ++nMarginalizedFeatures;
  }
  if (dimH_o[0] == 0) {
    return 0;
  }
  Eigen::MatrixXd H_o = Eigen::MatrixXd::Zero(dimH_o[0], featureVariableDimen);
  Eigen::MatrixXd r_o(dimH_o[0], 1);
  Eigen::MatrixXd R_o = Eigen::MatrixXd::Zero(dimH_o[0], dimH_o[0]);
  FilterHelper::stackJacobianAndResidual(vH_o, vr_o, vR_o, &H_o, &r_o, &R_o);
  FilterHelper::shrinkResidual(H_o, r_o, R_o, T_H, r_q, R_q);
  return dimH_o[0];
}

void SlidingWindowSmoother::optimize(size_t /*numIter*/, size_t /*numThreads*/, bool verbose) {
  uint64_t currFrameId = currentFrameId();
  OKVIS_ASSERT_EQ(
      Exception,
      covariance_.rows() - startIndexOfClonedStates(),
      (int)(kClonedStateMinimalDimen * statesMap_.size()), "Inconsistent covDim and number of states");

  // mark tracks of features that are not tracked in current frame
  int numTracked = 0;
  int featureVariableDimen = cameraParamsMinimalDimen() +
      kClonedStateMinimalDimen * (statesMap_.size() - 1);

  for (okvis::PointMap::iterator it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
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
    // (1) Iterated extended Kalman filter based visual-inertial odometry using direct photometric feedback
    // on: https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/263423/ROVIO.pdf?sequence=1&isAllowed=y
    // (2) Performance evaluation of iterated extended Kalman filter with variable step-length
    // on: https://iopscience.iop.org/article/10.1088/1742-6596/659/1/012022/pdf
    // (3) Faraz Mirzaei, a Kalman filter based algorithm for IMU-Camera calibration
    // on: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.6717&rep=rep1&type=pdf

    // Initial condition: $x_k^0 = x_{k|k-1}$
    // in each iteration,
    // $\Delta x_i = K_k^i(z - h(x_k^i) - H_k^i(x_{k|k-1}\boxminus x_k^i)) + x_{k|k-1}\boxminus x_k^i$
    // $x_k^{i+1} =  x_k^i\boxplus \Delta x_i$

    // We record the initial states, and update the estimator states in each
    // iteration which are used in computing Jacobians, and initializing landmarks.
    StatePointerAndEstimateList initialStates;
    cloneFilterStates(&initialStates);

    int numIteration = 0;
    DefaultEkfUpdater pceu(covariance_, featureVariableDimen);
    while (numIteration < maxNumIteration_) {
      Eigen::MatrixXd T_H, R_q;
      Eigen::Matrix<double, Eigen::Dynamic, 1> r_q;
      int numResiduals = computeStackedJacobianAndResidual(&T_H, &r_q, &R_q);
      if (numResiduals == 0) {
        minValidStateId_ = getMinValidStateId();
        return;  // no need to optimize
      }
      computeKalmanGainTimer.start();
      Eigen::VectorXd totalCorrection;
      boxminusFromInput(initialStates, &totalCorrection);
      Eigen::VectorXd deltax =
          pceu.computeCorrection(T_H, r_q, R_q, &totalCorrection);
      computeKalmanGainTimer.stop();
      updateStates(deltax);
      double lpNorm = deltax.lpNorm<Eigen::Infinity>();
//      LOG(INFO) << "num iteration " << numIteration << " deltax norm " << lpNorm;
      if (lpNorm < updateVecNormTermination_)
        break;
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
    DefaultEkfUpdater pceu(covariance_, featureVariableDimen);
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

    okvis::kinematics::Transformation T_WSc;
    get_T_WS(currentFrameId(), T_WSc);
    okvis::kinematics::Transformation T_CcW = (T_WSc * T_SC0).inverse();
    slamStats_.startUpdatingSceneDepth();

    for (auto it = landmarksMap_.begin(); it != landmarksMap_.end();
         ++it) {
      if (it->second.residualizeCase ==
          NotInState_NotTrackedNow)
        continue;
      // #Obs may be 1 for a new landmark by the KLT tracking frontend.
      // It ought to be >= 2 for descriptor matching frontend.
      if (it->second.observations.size() < 2)
        continue;

      // update coordinates of map points, this is only necessary when
      // (1) they are used to predict the points projection in new frames OR
      // (2) to visualize the point quality
      std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
          obsInPixel;
      std::vector<uint64_t> frameIds;
      std::vector<double> vRi;  // std noise in pixels
      const int camIdx = 0;
      std::shared_ptr<const okvis::cameras::CameraBase> tempCameraGeometry =
          camera_rig_.getCameraGeometry(camIdx);
      msckf::PointLandmark pointLandmark(msckf::HomogeneousPointParameterization::kModelId);
      msckf::PointSharedData psd;
      msckf::TriangulationStatus status =
          triangulateAMapPoint(it->second, obsInPixel, pointLandmark, vRi,
                               tempCameraGeometry, T_SC0, &psd, nullptr, false);
      Eigen::Map<Eigen::Vector4d> v4Xhomog(pointLandmark.data(), 4);
      if (status.triangulationOk) {
        it->second.quality = 1.0;
        it->second.pointHomog = v4Xhomog;
        if (!status.raysParallel && !status.flipped) {
          Eigen::Vector4d hpA = T_CcW * v4Xhomog;
          slamStats_.addLandmarkDepth(hpA[2]);
        }
      } else {
        it->second.quality = 0.0;
      }
    }
    slamStats_.finishUpdatingSceneDepth();
//    LOG(INFO) << "median scene depth " << slamStats_.medianSceneDepth();
    updateLandmarksTimer.stop();
  }

  if (verbose) {
    LOG(INFO) << mapPtr_->summary.FullReport();
  }
}
}  // namespace okvis
