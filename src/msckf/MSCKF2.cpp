#include <msckf/MSCKF2.hpp>

#include <glog/logging.h>

#include <okvis/assert_macros.hpp>
#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/RelativePoseError.hpp>
#include <okvis/ceres/SpeedAndBiasError.hpp>
#include <okvis/IdProvider.hpp>
#include <okvis/MultiFrame.hpp>

#include <msckf/EuclideanParamBlock.hpp>
#include <msckf/FilterHelper.hpp>
#include <msckf/ImuOdometry.h>
#include <msckf/PreconditionedEkfUpdater.h>
#include <msckf/triangulate.h>
#include <msckf/triangulateFast.hpp>

DEFINE_bool(use_AIDP, true,
            "use anchored inverse depth parameterization for a feature point?"
            " Preliminary result shows AIDP worsen the result slightly");

DECLARE_bool(use_mahalanobis);
DECLARE_bool(use_first_estimate);
DECLARE_bool(use_RK4);

DECLARE_double(max_proj_tolerance);

DEFINE_bool(use_IEKF, false,
            "use iterated EKF in optimization, empirically IEKF cost at"
            "least twice as much time as EKF");

DEFINE_bool(msckf_use_epipolar_constraint, true,
            "use epipolar constraints in case of low disparity or "
            "triangulation failure");

/// \brief okvis Main namespace of this package.
namespace okvis {

MSCKF2::MSCKF2(std::shared_ptr<okvis::ceres::Map> mapPtr)
    : HybridFilter(mapPtr) {}

// The default constructor.
MSCKF2::MSCKF2() {}

MSCKF2::~MSCKF2() {}

void MSCKF2::findRedundantCamStates(
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

int MSCKF2::marginalizeRedundantFrames(size_t maxClonedStates) {
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
    for (auto camStateId : rm_cam_state_ids) {
      auto obsIter = std::find_if(obsMap.begin(), obsMap.end(),
                                  IsObservedInFrame(camStateId));
      if (obsIter != obsMap.end()) {
        involved_cam_state_ids.emplace_back(camStateId);
      }
    }
    if (involved_cam_state_ids.size() < minCulledFrames_) {
      continue;
    }

    Eigen::MatrixXd H_oi;                           //(nObsDim, dimH_o[1])
    Eigen::Matrix<double, Eigen::Dynamic, 1> r_oi;  //(nObsDim, 1)
    Eigen::MatrixXd R_oi;                           //(nObsDim, nObsDim)

    // TODO(jhuai): do we consider the two view constraints here?
    bool isValidJacobian =
        featureJacobian(it->second, H_oi, r_oi, R_oi, &involved_cam_state_ids);
    if (!isValidJacobian) {
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

  for (const auto &cam_id : rm_cam_state_ids) {
    int cam_sequence =
        std::distance(statesMap_.begin(), statesMap_.find(cam_id));
    OKVIS_ASSERT_EQ(Exception, cam_sequence, mStateID2CovID_[cam_id], "inconsistent mStateID2CovID_");
  }

  // remove observations in removed frames
  for (okvis::PointMap::iterator it = landmarksMap_.begin();
       it != landmarksMap_.end();) {
    std::map<okvis::KeypointIdentifier, uint64_t>& obsMap =
        it->second.observations;
    bool removeAllEpipolarConstraints = false;
    for (const uint64_t camStateId : rm_cam_state_ids) {
      std::map<okvis::KeypointIdentifier, uint64_t>::iterator obsIter =
          std::find_if(obsMap.begin(), obsMap.end(), IsObservedInFrame(camStateId));
      if (obsIter != obsMap.end()) {
        const KeypointIdentifier& kpi = obsIter->first;
        auto mfp = multiFramePtrMap_.find(kpi.frameId);
        OKVIS_ASSERT_TRUE(Exception, mfp != multiFramePtrMap_.end(), "frame id not found in frame map!");
        mfp->second->setLandmarkId(kpi.cameraIndex, kpi.keypointIndex, 0);
        if (obsIter->second != 0u) {
          mapPtr_->removeResidualBlock(
              reinterpret_cast<::ceres::ResidualBlockId>(obsIter->second));
        } else {
          if (obsIter == obsMap.begin()) {
            // this is a head obs for epipolar constraints, remove all of them
            removeAllEpipolarConstraints = true;
          }  // else do nothing. This can happen if we removed an epipolar
             // constraint in a previous step and up to now the landmark has not
             // been initialized so its observations are not converted to
             // reprojection errors.
        }
        obsMap.erase(obsIter);
      }
    }
    if (removeAllEpipolarConstraints) {
      for (std::map<okvis::KeypointIdentifier, uint64_t>::iterator obsIter = obsMap.begin();
           obsIter != obsMap.end(); ++obsIter) {
        if (obsIter->second != 0u) {
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
    if (obsMap.size() == 0u) {
      mapPtr_->removeParameterBlock(it->first);
      it = landmarksMap_.erase(it);
    } else {
      ++it;
    }
  }

  for (const auto &cam_id : rm_cam_state_ids) {
    int cam_sequence =
        std::distance(statesMap_.begin(), statesMap_.find(cam_id));
    int cam_state_start =
        startIndexOfClonedStates() + kClonedStateMinimalDimen * cam_sequence;
    int cam_state_end = cam_state_start + kClonedStateMinimalDimen;

    FilterHelper::pruneSquareMatrix(cam_state_start, cam_state_end,
                                    &covariance_);
    removeState(cam_id);
  }

  mStateID2CovID_.clear();
  int nCovIndex = 0;
  for (auto iter = statesMap_.begin(); iter != statesMap_.end(); ++iter) {
    mStateID2CovID_[iter->first] = nCovIndex;
    ++nCovIndex;
  }
  uint64_t firstStateId = statesMap_.begin()->first;
  minValidStateID = std::min(minValidStateID, firstStateId);
  return rm_cam_state_ids.size();
}

bool MSCKF2::applyMarginalizationStrategy(
    size_t numKeyframes, size_t numImuFrames,
    okvis::MapPointVector& /*removedLandmarks*/) {
  std::vector<uint64_t> removeFrames;
  std::map<uint64_t, States>::reverse_iterator rit = statesMap_.rbegin();
  while (rit != statesMap_.rend()) {
    if (rit->first < minValidStateID) {
      removeFrames.push_back(rit->second.id);
    }
    ++rit;
  }
//  if (removeFrames.size() == 0) {
//    marginalizeRedundantFrames(numKeyframes + numImuFrames);
//  }

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

  return true;
}

bool MSCKF2::measurementJacobianAIDP(
    const Eigen::Vector4d& ab1rho,
    const std::shared_ptr<okvis::cameras::CameraBase> tempCameraGeometry,
    const Eigen::Vector2d& obs,
    uint64_t poseId, int camIdx,
    uint64_t anchorId, const okvis::kinematics::Transformation& T_WBa,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* H_x,
    Eigen::Matrix<double, 2, 3>* J_pfi, Eigen::Vector2d* residual) const {

  // compute Jacobians for a measurement in image j of the current feature i
  // C_j is the current frame, Bj refers to the body frame associated with the
  // current frame, Ba refers to body frame associated with the anchor frame,
  // f_i is the feature in consideration

  okvis::kinematics::Transformation T_GA =
      T_WBa * T_SC0_;  // anchor frame to global frame

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

  ImuMeasurement interpolatedInertialData;
  kinematics::Transformation T_WBj;
  get_T_WS(poseId, T_WBj);
  SpeedAndBiases sbj;
  getSpeedAndBias(poseId, 0, sbj);

  Time stateEpoch = statesMap_.at(poseId).timestamp;
  auto imuMeas = inertialMeasForStates_.findWindow(stateEpoch, half_window_);
  OKVIS_ASSERT_GT(Exception, imuMeas.size(), 0,
                  "the IMU measurement does not exist");

  uint32_t imageHeight = camera_rig_.getCameraGeometry(camIdx)->imageHeight();
  int projOptModelId = camera_rig_.getProjectionOptMode(camIdx);
  int extrinsicModelId = camera_rig_.getExtrinsicOptMode(camIdx);
  double kpN = obs[1] / imageHeight - 0.5;  // k per N
  const Duration featureTime =
      Duration(tdLatestEstimate + trLatestEstimate * kpN) -
      statesMap_.at(poseId).tdAtCreation;

  // for feature i, estimate $p_B^G(t_{f_i})$, $R_B^G(t_{f_i})$,
  // $v_B^G(t_{f_i})$, and $\omega_{GB}^B(t_{f_i})$ with the corresponding
  // states' LATEST ESTIMATES and imu measurements
  kinematics::Transformation T_WB = T_WBj;
  SpeedAndBiases sb = sbj;
  IMUErrorModel<double> iem(iem_);
  iem.resetBgBa(sb.tail<6>());
  if (FLAGS_use_RK4) {
    if (featureTime >= Duration()) {
      IMUOdometry::propagation_RungeKutta(imuMeas, imuParametersVec_.at(0),
                                          T_WB, sb, vTGTSTA_, stateEpoch,
                                          stateEpoch + featureTime);
    } else {
      IMUOdometry::propagationBackward_RungeKutta(
          imuMeas, imuParametersVec_.at(0), T_WB, sb, vTGTSTA_, stateEpoch,
          stateEpoch + featureTime);
    }
  } else {
    Eigen::Vector3d tempV_WS = sb.head<3>();
    if (featureTime >= Duration()) {
      IMUOdometry::propagation(
          imuMeas, imuParametersVec_.at(0), T_WB, tempV_WS, iem, stateEpoch,
          stateEpoch + featureTime);
    } else {
      IMUOdometry::propagationBackward(
          imuMeas, imuParametersVec_.at(0), T_WB, tempV_WS, iem, stateEpoch,
          stateEpoch + featureTime);
    }
    sb.head<3>() = tempV_WS;
  }

  IMUOdometry::interpolateInertialData(imuMeas, iem, stateEpoch + featureTime,
                                       interpolatedInertialData);

  okvis::kinematics::Transformation T_CA =
      (T_WB * T_SC0_).inverse() * T_GA;  // anchor frame to current camera frame
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

  kinematics::Transformation lP_T_WB = T_WB;
  SpeedAndBiases lP_sb = sb;

  if (FLAGS_use_first_estimate) {
    // compute p_WB, v_WB at (t_{f_i,j}) that use FIRST ESTIMATES of
    // position and velocity, i.e., their linearization point
    Eigen::Matrix<double, 6, 1> posVelFirstEstimate =
        statesMap_.at(poseId).linearizationPoint;
    lP_T_WB =
        kinematics::Transformation(posVelFirstEstimate.head<3>(), T_WBj.q());
    lP_sb = sbj;
    lP_sb.head<3>() = posVelFirstEstimate.tail<3>();
    Eigen::Vector3d tempV_WS = posVelFirstEstimate.tail<3>();

    if (featureTime >= Duration()) {
      IMUOdometry::propagation(
          imuMeas, imuParametersVec_.at(0), lP_T_WB, tempV_WS, iem, stateEpoch,
          stateEpoch + featureTime);
    } else {
      IMUOdometry::propagationBackward(
          imuMeas, imuParametersVec_.at(0), lP_T_WB, tempV_WS, iem, stateEpoch,
          stateEpoch + featureTime);
    }
    lP_sb.head<3>() = tempV_WS;
  }

  double rho = ab1rho[3];
  okvis::kinematics::Transformation T_BcA = lP_T_WB.inverse() * T_GA;
  J_td = pointJacobian3 * T_SC0_.C().transpose() *
         (okvis::kinematics::crossMx((T_BcA * ab1rho).head<3>()) *
              interpolatedInertialData.measurement.gyroscopes -
          T_WB.C().transpose() * lP_sb.head<3>() * rho);
  J_tr = J_td * kpN;
  Eigen::MatrixXd dpC_dExtrinsic;
  Eigen::Matrix3d R_CfCa = T_CA.C();
  ExtrinsicModel_dpC_dExtrinsic(extrinsicModelId, pfiinC, T_SC0_.C().transpose(),
                                &dpC_dExtrinsic, &R_CfCa, &ab1rho);
  ProjectionOptKneadIntrinsicJacobian(projOptModelId, &intrinsicsJacobian);
  if (dpC_dExtrinsic.size() == 0) {
    J_Xc << intrinsicsJacobian, J_td, J_tr;
  } else {
    J_Xc << pointJacobian3 * dpC_dExtrinsic, intrinsicsJacobian, J_td, J_tr;
  }
  Eigen::Matrix3d tempM3d;
  tempM3d << T_CA.C().topLeftCorner<3, 2>(), T_CA.r();
  (*J_pfi) = pointJacobian3 * tempM3d;

  Eigen::Vector3d pfinG = (T_GA * ab1rho).head<3>();
  factorJ_XBj << -rho * Eigen::Matrix3d::Identity(),
      okvis::kinematics::crossMx(pfinG - lP_T_WB.r() * rho),
      -rho * Eigen::Matrix3d::Identity() * featureTime.toSec();
  J_XBj = pointJacobian3 * (T_WB.C() * T_SC0_.C()).transpose() * factorJ_XBj;

  factorJ_XBa.topLeftCorner<3, 3>() = rho * Eigen::Matrix3d::Identity();
  factorJ_XBa.block<3, 3>(0, 3) =
      -okvis::kinematics::crossMx(T_WBa.C() * (T_SC0_ * ab1rho).head<3>());
  factorJ_XBa.block<3, 3>(0, 6) = Eigen::Matrix3d::Zero();
  J_XBa = pointJacobian3 * (T_WB.C() * T_SC0_.C()).transpose() * factorJ_XBa;

  H_x->setZero();
  const int minCamParamDim = cameraParamsMinimalDimen();
  H_x->topLeftCorner(2, minCamParamDim) = J_Xc;
  std::map<uint64_t, int>::const_iterator poseid_iter =
      mStateID2CovID_.find(poseId);
  int covid = poseid_iter->second;
  if (poseId == anchorId) {
    H_x->block<2, 6>(0, minCamParamDim +
                           9 * covid + 3) =
        (J_XBj + J_XBa).block<2, 6>(0, 3);
  } else {
    H_x->block<2, 9>(0, minCamParamDim +
                           9 * covid) = J_XBj;
    std::map<uint64_t, int>::const_iterator anchorid_iter =
        mStateID2CovID_.find(anchorId);
    H_x->block<2, 9>(0, minCamParamDim +
                           9 * anchorid_iter->second) = J_XBa;
  }
  return true;
}

bool MSCKF2::measurementJacobian(
    const Eigen::Vector4d& v4Xhomog,
    const std::shared_ptr<okvis::cameras::CameraBase> tempCameraGeometry,
    const Eigen::Vector2d& obs,
    uint64_t poseId,
    int camIdx, Eigen::Matrix<double, 2, Eigen::Dynamic>* J_Xc,
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
  ImuMeasurement interpolatedInertialData;

  kinematics::Transformation T_WBj;
  get_T_WS(poseId, T_WBj);
  SpeedAndBiases sbj;
  getSpeedAndBias(poseId, 0, sbj);

  Time stateEpoch = statesMap_.at(poseId).timestamp;
  auto imuMeas = inertialMeasForStates_.findWindow(stateEpoch, half_window_);
  OKVIS_ASSERT_GT(Exception, imuMeas.size(), 0,
                  "the IMU measurement does not exist");

  uint32_t imageHeight = camera_rig_.getCameraGeometry(camIdx)->imageHeight();
  int projOptModelId = camera_rig_.getProjectionOptMode(camIdx);
  int extrinsicModelId = camera_rig_.getExtrinsicOptMode(camIdx);
  double kpN = obs[1] / imageHeight - 0.5;  // k per N
  Duration featureTime = Duration(tdLatestEstimate + trLatestEstimate * kpN) -
                         statesMap_.at(poseId).tdAtCreation;

  // for feature i, estimate $p_B^G(t_{f_i})$, $R_B^G(t_{f_i})$,
  // $v_B^G(t_{f_i})$, and $\omega_{GB}^B(t_{f_i})$ with the corresponding
  // states' LATEST ESTIMATES and imu measurements

  kinematics::Transformation T_WB = T_WBj;
  SpeedAndBiases sb = sbj;
  IMUErrorModel<double> iem(iem_);
  iem.resetBgBa(sb.tail<6>());
  if (FLAGS_use_RK4) {
    if (featureTime >= Duration()) {
      IMUOdometry::propagation_RungeKutta(imuMeas, imuParametersVec_.at(0),
                                          T_WB, sb, vTGTSTA_, stateEpoch,
                                          stateEpoch + featureTime);
    } else {
      IMUOdometry::propagationBackward_RungeKutta(
          imuMeas, imuParametersVec_.at(0), T_WB, sb, vTGTSTA_, stateEpoch,
          stateEpoch + featureTime);
    }
  } else {
    Eigen::Vector3d tempV_WS = sb.head<3>();

    if (featureTime >= Duration()) {
      IMUOdometry::propagation(
          imuMeas, imuParametersVec_.at(0), T_WB, tempV_WS, iem, stateEpoch,
          stateEpoch + featureTime);
    } else {
      IMUOdometry::propagationBackward(
          imuMeas, imuParametersVec_.at(0), T_WB, tempV_WS, iem, stateEpoch,
          stateEpoch + featureTime);
    }
    sb.head<3>() = tempV_WS;
  }

  IMUOdometry::interpolateInertialData(imuMeas, iem, stateEpoch + featureTime,
                                       interpolatedInertialData);
  kinematics::Transformation T_CW = (T_WB * T_SC0_).inverse();
  Eigen::Vector3d pfiinC = (T_CW * v4Xhomog).head<3>();
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

  kinematics::Transformation lP_T_WB = T_WB;
  SpeedAndBiases lP_sb = sb;
  if (FLAGS_use_first_estimate) {
    lP_T_WB = T_WBj;
    lP_sb = sbj;
    Eigen::Matrix<double, 6, 1> posVelFirstEstimate =
        statesMap_.at(poseId).linearizationPoint;
    lP_T_WB =
        kinematics::Transformation(posVelFirstEstimate.head<3>(), lP_T_WB.q());
    lP_sb.head<3>() = posVelFirstEstimate.tail<3>();

    Eigen::Vector3d tempV_WS = lP_sb.head<3>();
    if (featureTime >= Duration()) {
      IMUOdometry::propagation(
          imuMeas, imuParametersVec_.at(0), lP_T_WB, tempV_WS, iem, stateEpoch,
          stateEpoch + featureTime);
    } else {
      IMUOdometry::propagationBackward(
          imuMeas, imuParametersVec_.at(0), lP_T_WB, tempV_WS, iem, stateEpoch,
          stateEpoch + featureTime);
    }
    lP_sb.head<3>() = tempV_WS;
  }

  J_td = pointJacobian3 * T_SC0_.C().transpose() *
         (okvis::kinematics::crossMx((lP_T_WB.inverse() * v4Xhomog).head<3>()) *
              interpolatedInertialData.measurement.gyroscopes -
          T_WB.C().transpose() * lP_sb.head<3>());
  J_tr = J_td * kpN;
  Eigen::MatrixXd dpC_dExtrinsic;
  ExtrinsicModel_dpC_dExtrinsic(extrinsicModelId, pfiinC, T_SC0_.C().transpose(),
                                &dpC_dExtrinsic, nullptr, nullptr);
  ProjectionOptKneadIntrinsicJacobian(projOptModelId, &intrinsicsJacobian);
  if (dpC_dExtrinsic.size() == 0) {
    (*J_Xc) << intrinsicsJacobian, J_td, J_tr;
  } else {
    (*J_Xc) << pointJacobian3 * dpC_dExtrinsic, intrinsicsJacobian, J_td, J_tr;
  }
  (*J_pfi) = pointJacobian3 * T_CW.C();

  factorJ_XBj << -Eigen::Matrix3d::Identity(),
      okvis::kinematics::crossMx(v3Point - lP_T_WB.r()),
      -Eigen::Matrix3d::Identity() * featureTime.toSec();

  (*J_XBj) = (*J_pfi) * factorJ_XBj;
  return true;
}

// assume the rolling shutter camera reads data row by row, and rows are
//     aligned with the width of a frame
bool MSCKF2::featureJacobian(const MapPoint &mp, Eigen::MatrixXd &H_oi,
                        Eigen::Matrix<double, Eigen::Dynamic, 1> &r_oi,
                        Eigen::MatrixXd &R_oi,
                        const std::vector<uint64_t>* involvedFrameIds) const {
  const int camIdx = 0;
  std::shared_ptr<okvis::cameras::CameraBase> tempCameraGeometry =
      camera_rig_.getCameraGeometry(camIdx);

  // dimension of variables used in computing feature Jacobians, including
  // camera intrinsics and all cloned states except the most recent one
  // in which the marginalized observations should never occur.
  int featureVariableDimen = cameraParamsMinimalDimen() +
      kClonedStateMinimalDimen * (statesMap_.size() - 1);

  // all observations for this feature point
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      obsInPixel;
  std::vector<uint64_t> frameIds; // id of frames observing this feature point
  std::vector<double> vRi; // std noise in pixels
  Eigen::Vector4d v4Xhomog; // triangulated point position in the global
  // frame expressed in [X,Y,Z,W],
  computeHTimer.start();
  if (FLAGS_use_AIDP) {
    // The landmark is expressed with AIDP in the anchor frame    
    // if the feature is lost in current frame, the anchor frame is chosen
    // as the last frame observing the point.

    uint64_t anchorId = mp.observations.rbegin()->first.frameId;
    int anchorSeqId = mp.observations.size() - 1;
    if (involvedFrameIds != nullptr) {
      anchorId = involvedFrameIds->back();
      const std::map<okvis::KeypointIdentifier, uint64_t>& obsMap = mp.observations;
      auto anchorIter = std::find_if(obsMap.begin(), obsMap.end(), IsObservedInFrame(anchorId));
      anchorSeqId = std::distance(obsMap.begin(), anchorIter);
    }
    bool bSucceeded = triangulateAMapPoint(
        mp, obsInPixel, frameIds, v4Xhomog, vRi, tempCameraGeometry, T_SC0_,
        anchorSeqId, FLAGS_msckf_use_epipolar_constraint);

    if (!bSucceeded) {
      computeHTimer.stop();
      return false;
    }

    Eigen::Vector4d ab1rho = v4Xhomog;
    if (ab1rho[2] <= 0) {  // negative depth
      computeHTimer.stop();
      return false;
    }
    ab1rho /= ab1rho[2];  //[\alpha = X/Z, \beta= Y/Z, 1, \rho=1/Z] in the
                          // anchor frame
    // transform from the body frame at the anchor frame epoch to the world frame
    kinematics::Transformation T_WBa;
    get_T_WS(anchorId, T_WBa);

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

    size_t numPoses = frameIds.size();
    size_t numValidObs = 0;
    auto itFrameIds = frameIds.begin();
    auto itRoi = vRi.begin();
    // compute Jacobians for a measurement in image j of the current feature i
    for (size_t kale = 0; kale < numPoses; ++kale) {
      Eigen::Matrix<double, 2, Eigen::Dynamic> H_x(2, featureVariableDimen);
      // $\frac{\partial [z_u, z_v]^T}{\partial [\alpha, \beta, \rho]}$
      Eigen::Matrix<double, 2, 3> J_pfi;
      Eigen::Vector2d residual;

      uint64_t poseId = *itFrameIds;
      const int camIdx = 0;

      if (involvedFrameIds != nullptr &&
          std::find(involvedFrameIds->begin(), involvedFrameIds->end(), poseId)
              == involvedFrameIds->end()) {
          itFrameIds = frameIds.erase(itFrameIds);
          itRoi = vRi.erase(itRoi);
          itRoi = vRi.erase(itRoi);
          continue;
      }

      bool validJacobian = measurementJacobianAIDP(
          ab1rho, tempCameraGeometry, obsInPixel[kale], poseId, camIdx, anchorId, T_WBa,
          &H_x, &J_pfi, &residual);
      if (!validJacobian) {
          itFrameIds = frameIds.erase(itFrameIds);
          itRoi = vRi.erase(itRoi);
          itRoi = vRi.erase(itRoi);
          continue;
      }

      vri.push_back(residual);
      vJ_X.push_back(H_x);
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

    Eigen::MatrixXd nullQ = vio::nullspace(H_fi);  // 2nx(2n-3), n==numValidObs

    r_oi.noalias() = nullQ.transpose() * ri;
    H_oi.noalias() = nullQ.transpose() * H_xi;
    R_oi = nullQ.transpose() * (Ri * nullQ).eval();

    vri.clear();
    vJ_pfi.clear();
    vJ_X.clear();
    frameIds.clear();
    computeHTimer.stop();
    return true;
  } else {
    // The landmark is expressed with Euclidean coordinates in the global frame
    bool bSucceeded = triangulateAMapPoint(mp, obsInPixel, frameIds, v4Xhomog,
                                           vRi, tempCameraGeometry, T_SC0_, -1,
                                           FLAGS_msckf_use_epipolar_constraint);
    if (!bSucceeded) {
      computeHTimer.stop();
      return false;
    }

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

    size_t numPoses = frameIds.size();
    size_t numValidObs = 0;
    auto itFrameIds = frameIds.begin();
    auto itRoi = vRi.begin();
    // compute Jacobians for a measurement in image j of the current feature i
    for (size_t kale = 0; kale < numPoses; ++kale) {
      uint64_t poseId = *itFrameIds;
      const int camIdx = 0;
      // $\frac{\partial [z_u, z_v]^T}{\partial(extrinsic, intrinsic, td, tr)}$
      Eigen::Matrix<double, 2, Eigen::Dynamic> J_Xc(2, cameraParamsMinimalDimen());
      Eigen::Matrix<double, 2, kClonedStateMinimalDimen>
          J_XBj;  // $\frac{\partial [z_u, z_v]^T}{delta\p_{B_j}^G, \delta\alpha
                  // (of q_{B_j}^G), \delta v_{B_j}^G$
      Eigen::Matrix<double, 2, 3>
          J_pfi;  // $\frac{\partial [z_u, z_v]^T}{\partial p_{f_i}^G}$
      Eigen::Vector2d residual;

      if (involvedFrameIds != nullptr &&
          std::find(involvedFrameIds->begin(), involvedFrameIds->end(), poseId)
              == involvedFrameIds->end()) {
          itFrameIds = frameIds.erase(itFrameIds);
          itRoi = vRi.erase(itRoi);
          itRoi = vRi.erase(itRoi);
          continue;
      }
      bool validJacobian = measurementJacobian(
          v4Xhomog, tempCameraGeometry, obsInPixel[kale], poseId, camIdx,
          &J_Xc, &J_XBj, &J_pfi, &residual);
      if (!validJacobian) {
          itFrameIds = frameIds.erase(itFrameIds);
          itRoi = vRi.erase(itRoi);
          itRoi = vRi.erase(itRoi);
          continue;
      }

      vJ_Xc.push_back(J_Xc);
      vJ_XBj.push_back(J_XBj);
      vJ_pfi.push_back(J_pfi);
      vri.push_back(residual);
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
      std::map<uint64_t, int>::const_iterator frmid_iter =
          mStateID2CovID_.find(frameIds[saga]);
      H_xi.block<2, kClonedStateMinimalDimen>(
          saga2, cameraParamsDimen + kClonedStateMinimalDimen *
                                         frmid_iter->second) = vJ_XBj[saga];
      H_fi.block<2, 3>(saga2, 0) = vJ_pfi[saga];
      ri.segment<2>(saga2) = vri[saga];
      Ri(saga2, saga2) *= (vRi[saga2] * vRi[saga2]);
      Ri(saga2 + 1, saga2 + 1) *= (vRi[saga2 + 1] * vRi[saga2 + 1]);
    }

    Eigen::MatrixXd nullQ = vio::nullspace(H_fi);  // 2nx(2n-3), n==numValidObs

    r_oi.noalias() = nullQ.transpose() * ri;
    H_oi.noalias() = nullQ.transpose() * H_xi;
    R_oi = nullQ.transpose() * (Ri * nullQ).eval();

    vri.clear();
    vJ_pfi.clear();
    vJ_XBj.clear();
    vJ_Xc.clear();
    frameIds.clear();
    computeHTimer.stop();
    return true;
  }
}

int MSCKF2::computeStackedJacobianAndResidual(
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
      isValidJacobian = FLAGS_msckf_use_epipolar_constraint
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

void MSCKF2::optimize(size_t /*numIter*/, size_t /*numThreads*/, bool verbose) {
  uint64_t currFrameId = currentFrameId();
  OKVIS_ASSERT_EQ(
      Exception,
      covariance_.rows() - startIndexOfClonedStates(),
      (int)(kClonedStateMinimalDimen * statesMap_.size()), "Inconsistent covDim and number of states");
  retrieveEstimatesOfConstants();

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

        //            double normInf = (deltaX-
        //            tempDeltaX).lpNorm<Eigen::Infinity>(); std::cout <<"iter
        //            "<< numIteration<<" normInf "<<normInf<<" normInf<eps?"<<
        //            (bool)(normInf<epsilon)<<std::endl<<
        //                            (deltaX-
        //                            tempDeltaX).transpose()<<std::endl;

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
    minValidStateID = getMinValidStateID();
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
