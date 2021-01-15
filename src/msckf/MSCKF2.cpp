#include <msckf/MSCKF2.hpp>

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
#include <msckf/EkfUpdater.h>

DECLARE_bool(use_mahalanobis);
DECLARE_bool(use_first_estimate);
DECLARE_bool(use_RK4);

DECLARE_double(max_proj_tolerance);

DEFINE_bool(use_IEKF, false,
            "use iterated EKF in optimization, empirically IEKF cost at"
            "least twice as much time as EKF");

/// \brief okvis Main namespace of this package.
namespace okvis {

MSCKF2::MSCKF2(std::shared_ptr<okvis::ceres::Map> mapPtr)
    : HybridFilter(mapPtr) {}

// The default constructor.
MSCKF2::MSCKF2() {}

MSCKF2::~MSCKF2() {}

bool MSCKF2::applyMarginalizationStrategy(
    size_t numKeyframes, size_t numImuFrames,
    okvis::MapPointVector& removedLandmarks) {

  marginalizeRedundantFrames(numKeyframes, numImuFrames);

  // remove features tracked no more.
  for (PointMap::iterator pit = landmarksMap_.begin();
       pit != landmarksMap_.end();) {
    const MapPoint& mapPoint = pit->second;
    if (mapPoint.shouldRemove(pointLandmarkOptions_.maxHibernationFrames)) {
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
  return true;
}

bool MSCKF2::measurementJacobianAIDPMono(
    const Eigen::Vector4d& ab1rho,
    const Eigen::Vector2d& obs,
    size_t observationIndex,
    std::shared_ptr<const msckf::PointSharedData> pointDataPtr,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* J_x,
    Eigen::Matrix<double, 2, 3>* J_pfi, Eigen::Vector2d* residual) const {
  // compute Jacobians for a measurement in current image j of feature i \f$f_i\f$.
  // C_{t(i,j)} is the camera frame at the observation epoch t(i,j).
  // B_{t(i,j)} is the body frame at the observation epoch t(i,j).
  // B_j is the body frame at the state epoch t_j associated with image j.
  // B_{t(i,a)} is the body frame at the epoch of observation in the anchor frame.

  Eigen::Vector2d imagePoint;  // projected pixel coordinates of the point
                               // \f${z_u, z_v}\f$ in pixel units
  Eigen::Matrix2Xd
      intrinsicsJacobian;  // \f$\frac{\partial [z_u, z_v]^T}{\partial(intrinsics)}\f$
  Eigen::Matrix<double, 2, 3>
      dz_drhoxpCtj;  // \f$\frac{\partial [z_u, z_v]^T}{\partial
                       // p_i^{C_{t(i,j)}}\f$

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
  size_t camIdx = pointDataPtr->cameraIndex(observationIndex);

  const int minCamParamDim = cameraParamsMinimalDimFast(camIdx);
  // $\frac{\partial [z_u, z_v]^T}{\partial(extrinsic, intrinsic, t_d, t_r)}$
  Eigen::Matrix<double, 2, Eigen::Dynamic> J_Xc(2, minCamParamDim);
  const okvis::kinematics::Transformation T_BCj = camera_rig_.getCameraExtrinsic(camIdx);

  size_t anchorCamIdx = pointDataPtr->anchorIds()[0].cameraIndex_;
  const int minAnchorCamParamDim = cameraParamsMinimalDimFast(anchorCamIdx);
  Eigen::Matrix<double, 2, Eigen::Dynamic> anchor_J_Xc(2, minAnchorCamParamDim);
  const okvis::kinematics::Transformation T_BCa = camera_rig_.getCameraExtrinsic(anchorCamIdx);
  okvis::kinematics::Transformation T_BC0 = camera_rig_.getCameraExtrinsic(kMainCameraIndex);

  int projOptModelId = camera_rig_.getProjectionOptMode(camIdx);
  int extrinsicModelId = camera_rig_.getExtrinsicOptMode(camIdx);

  double kpN = pointDataPtr->normalizedRow(observationIndex);
  const double featureTime = pointDataPtr->normalizedFeatureTime(observationIndex);

  kinematics::Transformation T_WBtj = pointDataPtr->T_WBtij(observationIndex);

  okvis::kinematics::Transformation T_WBta;
  if (pointLandmarkOptions_.anchorAtObservationTime) {
    T_WBta = pointDataPtr->T_WB_mainAnchor();
  } else {
    T_WBta = pointDataPtr->T_WB_mainAnchorStateEpoch();
  }

  okvis::kinematics::Transformation T_WCta = T_WBta * T_BCa;  // anchor frame to global frame
  okvis::kinematics::Transformation T_CtjCta = (T_WBtj * T_BCj).inverse() * T_WCta;  // anchor frame to current camera frame
  Eigen::Vector3d rhoxpCtj = (T_CtjCta * ab1rho).head<3>();

  std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry =
      camera_rig_.getCameraGeometry(camIdx);
  cameras::CameraBase::ProjectionStatus status = cameraGeometry->project(
      rhoxpCtj, &imagePoint, &dz_drhoxpCtj, &intrinsicsJacobian);
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

  double rho = ab1rho[3];
  kinematics::Transformation lP_T_WBtj = pointDataPtr->T_WBtij_ForJacobian(observationIndex);
  Eigen::Vector3d omega_Btj = pointDataPtr->omega_Btij(observationIndex);
  Eigen::Vector3d lP_v_WBtj = pointDataPtr->v_WBtij_ForJacobian(observationIndex);
  okvis::kinematics::Transformation T_BcA = lP_T_WBtj.inverse() * T_WCta;
  J_td = dz_drhoxpCtj * T_BCj.C().transpose() *
         (okvis::kinematics::crossMx((T_BcA * ab1rho).head<3>()) *
              omega_Btj -
          T_WBtj.C().transpose() * lP_v_WBtj * rho);
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
    Eigen::Matrix3d R_CfCa = T_CtjCta.C();
    ExtrinsicModel_dpC_dExtrinsic_AIDP(extrinsicModelId, rhoxpCtj,
                                       T_BCj.C().transpose(), &dpC_dExtrinsic,
                                       &R_CfCa, &ab1rho);
    if (fixCameraIntrinsicParams_[camIdx]) {
      J_Xc << dz_drhoxpCtj * dpC_dExtrinsic, J_td, J_tr;
    } else {
      ProjectionOptKneadIntrinsicJacobian(projOptModelId, &intrinsicsJacobian);
      J_Xc << dz_drhoxpCtj * dpC_dExtrinsic, intrinsicsJacobian, J_td, J_tr;
    }
  }

  // Jacobians relative to point landmark parameterization.
  Eigen::Matrix3d tempM3d;
  tempM3d << T_CtjCta.C().topLeftCorner<3, 2>(), T_CtjCta.r();
  (*J_pfi) = dz_drhoxpCtj * tempM3d;

  // Jacobians relative to nav states.
  Eigen::Vector3d pfinG = (T_WCta * ab1rho).head<3>();
  factorJ_XBj << -rho * Eigen::Matrix3d::Identity(),
      okvis::kinematics::crossMx(pfinG - lP_T_WBtj.r() * rho),
      -rho * Eigen::Matrix3d::Identity() * featureTime;
  J_XBj = dz_drhoxpCtj * (T_WBtj.C() * T_BCj.C()).transpose() * factorJ_XBj;

  factorJ_XBa.topLeftCorner<3, 3>() = rho * Eigen::Matrix3d::Identity();
  factorJ_XBa.block<3, 3>(0, 3) =
      -okvis::kinematics::crossMx(T_WBta.C() * (T_BCj * ab1rho).head<3>());
  factorJ_XBa.block<3, 3>(0, 6) = Eigen::Matrix3d::Zero();
  J_XBa = dz_drhoxpCtj * (T_WBtj.C() * T_BCj.C()).transpose() * factorJ_XBa;

  J_x->setZero();
  int camParamStartIndex = intraStartIndexOfCameraParams(camIdx);
  J_x->block(0, camParamStartIndex, 2, minCamParamDim) = J_Xc;

  size_t startIndexCameraParams = startIndexOfCameraParams(kMainCameraIndex);
  auto poseid_iter = statesMap_.find(poseId);
  int covid = poseid_iter->second.global.at(GlobalStates::T_WS).startIndexInCov;

  uint64_t anchorId = pointDataPtr->anchorIds()[0].frameId_;
  J_x->block<2, 9>(0, covid - startIndexCameraParams) = J_XBj;
  auto anchorid_iter = statesMap_.find(anchorId);
  J_x->block<2, 9>(
      0, anchorid_iter->second.global.at(GlobalStates::T_WS).startIndexInCov -
             startIndexCameraParams) += J_XBa;
  return true;
}


bool MSCKF2::measurementJacobianHPPMono(
    const Eigen::Vector4d& v4Xhomog,
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
      dz_dpCtij;  // $\frac{\partial [z_u, z_v]^T}{\partial
                       // p_{f_i}^{C_j}}$

  Eigen::Matrix<double, 3, 9>
      factorJ_XBj;  // the second factor of J_XBj, see Michael Andrew Shelley
                    // Master thesis sec 6.5, p.55 eq 6.66

  Eigen::Vector2d J_td;
  Eigen::Vector2d J_tr;

  size_t camIdx = pointDataPtr->cameraIndex(observationIndex);
  std::shared_ptr<const okvis::cameras::CameraBase> tempCameraGeometry =
      camera_rig_.getCameraGeometry(camIdx);
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
      pfiinC, &imagePoint, &dz_dpCtij, &intrinsicsJacobian);
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

  J_td = dz_dpCtij * T_SC0.C().transpose() *
         (okvis::kinematics::crossMx((lP_T_WB.inverse() * v4Xhomog).head<3>()) *
              omega_Btij -
          T_WB.C().transpose() * lP_v_WBtij);
  J_tr = J_td * kpN;

  J_Xc->resize(2, cameraParamsMinimalDimFast(camIdx));
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
      (*J_Xc) << dz_dpCtij * dpC_dExtrinsic, J_td, J_tr;
    } else {
      ProjectionOptKneadIntrinsicJacobian(projOptModelId, &intrinsicsJacobian);
      (*J_Xc) << dz_dpCtij * dpC_dExtrinsic, intrinsicsJacobian, J_td,
          J_tr;
    }
  }
  (*J_pfi) = dz_dpCtij * T_CW.C();
  Eigen::Matrix3d Phi_pq = pointDataPtr->Phi_pq_feature(observationIndex);
  factorJ_XBj << -Eigen::Matrix3d::Identity(),
      okvis::kinematics::crossMx(v3Point - lP_T_WB.r()) - Phi_pq,
      -Eigen::Matrix3d::Identity() * featureTime;

  (*J_XBj) = (*J_pfi) * factorJ_XBj;
  return true;
}

bool MSCKF2::featureJacobianGeneric(
    const MapPoint &mp, msckf::PointLandmark *pointLandmark, Eigen::MatrixXd &H_oi,
    Eigen::Matrix<double, Eigen::Dynamic, 1> &r_oi, Eigen::MatrixXd &R_oi,
    Eigen::Matrix<double, Eigen::Dynamic, 3> * /*pH_fi*/,
    std::vector<uint64_t> *orderedCulledFrameIds) const {
  const int camIdx = 0;
  std::shared_ptr<const okvis::cameras::CameraBase> tempCameraGeometry =
      camera_rig_.getCameraGeometry(camIdx);
  // dimension of variables used in computing feature Jacobians, including
  // camera intrinsics and all cloned states except the most recent one
  // in which the marginalized observations should never occur.
  int featureVariableDimen = minimalDimOfAllCameraParams() +
                             kClonedStateMinimalDimen * (statesMap_.size() - 1);
  int residualBlockDim = okvis::cameras::CameraObservationModelResidualDim(
        optimizationOptions_.cameraObservationModelId);
  // all observations for this feature point
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      obsInPixel;
  std::vector<double> vRi;         // std noise in pixels, 2Nx1
  computeHTimer.start();

  std::shared_ptr<msckf::PointSharedData> pointDataPtr(new msckf::PointSharedData());
  pointLandmark->setModelId(pointLandmarkOptions_.landmarkModelId);
  msckf::TriangulationStatus status = triangulateAMapPoint(
      mp, obsInPixel, *pointLandmark, vRi,
      pointDataPtr.get(), orderedCulledFrameIds, optimizationOptions_.useEpipolarConstraint);
  if (!status.triangulationOk) {
    computeHTimer.stop();
    return false;
  }
  if (orderedCulledFrameIds) {
    pointDataPtr->removeExtraObservations(*orderedCulledFrameIds, &vRi);
  }
  pointDataPtr->computePoseAndVelocityForJacobians(FLAGS_use_first_estimate);
  pointDataPtr->computeSharedJacobians(optimizationOptions_.cameraObservationModelId);
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
        *pointLandmark, tempCameraGeometry, obsInPixel[observationIndex],
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
  if (numValidObs < pointLandmarkOptions_.minTrackLengthForMsckf) {
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

  if (optimizationOptions_.cameraObservationModelId != okvis::cameras::kEpipolarFactorId) {
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

msckf::MeasurementJacobianStatus MSCKF2::measurementJacobianGeneric(
    const msckf::PointLandmark& pointLandmark,
    std::shared_ptr<const okvis::cameras::CameraBase> baseCameraGeometry,
    const Eigen::Vector2d& obs,
    const Eigen::Matrix2d& obsCov, int observationIndex,
    std::shared_ptr<const msckf::PointSharedData> pointDataPtr,
    Eigen::MatrixXd* J_X, Eigen::Matrix<double, Eigen::Dynamic, 3>* J_pfi,
    Eigen::Matrix<double, Eigen::Dynamic, 2>* J_n,
    Eigen::VectorXd* residual) const {
  size_t camIdx = pointDataPtr->cameraIndex(observationIndex);
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

bool MSCKF2::featureJacobian(
    const MapPoint &mp, msckf::PointLandmark *pointLandmark,
    Eigen::MatrixXd &H_oi, Eigen::Matrix<double, Eigen::Dynamic, 1> &r_oi,
    Eigen::MatrixXd &R_oi, Eigen::Matrix<double, Eigen::Dynamic, 3> *pH_fi,
    std::vector<uint64_t> *orderedCulledFrameIds) const {
  if (pointLandmarkOptions_.landmarkModelId ==
          msckf::ParallaxAngleParameterization::kModelId ||
      optimizationOptions_.cameraObservationModelId !=
          okvis::cameras::kReprojectionErrorId) {
    return featureJacobianGeneric(mp, pointLandmark, H_oi, r_oi, R_oi, pH_fi,
                                  orderedCulledFrameIds);
  } else {
    return HybridFilter::featureJacobian(mp, pointLandmark, H_oi, r_oi, R_oi,
                                         pH_fi, orderedCulledFrameIds);
  }
}

int MSCKF2::computeStackedJacobianAndResidual(
    Eigen::MatrixXd *T_H, Eigen::Matrix<double, Eigen::Dynamic, 1> *r_q,
    Eigen::MatrixXd *R_q) {
  // compute and stack Jacobians and Residuals for landmarks observed no more
  size_t nMarginalizedFeatures = 0;
  int culledPoints[2] = {0};
  int featureVariableDimen = minimalDimOfAllCameraParams() +
      kClonedStateMinimalDimen * (statesMap_.size() - 1);
  int dimH_o[2] = {0, featureVariableDimen};
  const int camParamStartIndex = startIndexOfCameraParamsFast(0u);
  const Eigen::MatrixXd variableCov =
      covariance_.block(camParamStartIndex, camParamStartIndex, dimH_o[1], dimH_o[1]);
  // containers of Jacobians of measurements
  Eigen::AlignedVector<Eigen::Matrix<double, -1, 1>> vr_o;
  Eigen::AlignedVector<Eigen::MatrixXd> vH_o;
  Eigen::AlignedVector<Eigen::MatrixXd> vR_o;

  for (auto it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
    if (it->second.status.measurementType != FeatureTrackStatus::kMsckfTrack) {
      continue;
    }

    msckf::PointLandmark pointLandmark;
    Eigen::MatrixXd H_oi;                           //(nObsDim, dimH_o[1])
    H_oi.resize(0, featureVariableDimen);
    Eigen::Matrix<double, Eigen::Dynamic, 1> r_oi;  //(nObsDim, 1)
    Eigen::MatrixXd R_oi;                           //(nObsDim, nObsDim)
    bool isValidJacobian = featureJacobian(it->second, &pointLandmark, H_oi, r_oi, R_oi);
    if (!isValidJacobian) {
      isValidJacobian = optimizationOptions_.useEpipolarConstraint
                            ? featureJacobianEpipolar(it->second, &H_oi, &r_oi,
                                                      &R_oi, ENTIRE_TRACK)
                            : isValidJacobian;
      if (!isValidJacobian) {
        it->second.status.measurementFate = FeatureTrackStatus::kComputingJacobiansFailed;
        ++culledPoints[0];
        continue;
      }
    }

    if (!FilterHelper::gatingTest(H_oi, r_oi, R_oi, variableCov)) {
      it->second.status.measurementFate = FeatureTrackStatus::kPotentialOutlier;
      ++culledPoints[1];
      continue;
    }
    it->second.status.measurementFate = FeatureTrackStatus::kSuccessful;
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
  Eigen::Matrix<double, -1, 1> r_o(dimH_o[0], 1);
  Eigen::MatrixXd R_o = Eigen::MatrixXd::Zero(dimH_o[0], dimH_o[0]);
  FilterHelper::stackJacobianAndResidual(vH_o, vr_o, vR_o, &H_o, &r_o, &R_o);
  FilterHelper::shrinkResidual(H_o, r_o, R_o, T_H, r_q, R_q);
  return dimH_o[0];
}

void MSCKF2::optimize(size_t /*numIter*/, size_t /*numThreads*/, bool verbose) {
  uint64_t currFrameId = currentFrameId();
  OKVIS_ASSERT_EQ(
      Exception,
      covariance_.rows() - startIndexOfClonedStatesFast(),
      kClonedStateMinimalDimen * statesMap_.size(), "Inconsistent covDim and number of states");
  if (loopFrameAndMatchesList_.size() > 0) {
    LOG(INFO) << "MSCKF receives #loop frames " << loopFrameAndMatchesList_.size()
              << " but has not implemented relocalization yet!";
    loopFrameAndMatchesList_.clear();
  }

  // mark tracks of features that are not tracked in current frame.
  int numTracked = 0;
  int featureVariableDimen = minimalDimOfAllCameraParams() +
      kClonedStateMinimalDimen * (statesMap_.size() - 1);
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
    return;
  }

  // update landmarks that are tracked in the current frame(the newly inserted
  // state)
  {
    updateLandmarksTimer.start();
    const int camIdx = 0;
    const okvis::kinematics::Transformation T_SC0 = camera_rig_.getCameraExtrinsic(camIdx);

    okvis::kinematics::Transformation T_WSc;
    get_T_WS(currentFrameId(), T_WSc);
    okvis::kinematics::Transformation T_CcW = (T_WSc * T_SC0).inverse();
    slamStats_.startUpdatingSceneDepth();

    for (auto it = landmarksMap_.begin(); it != landmarksMap_.end();
         ++it) {
      if (it->second.status.measurementType != FeatureTrackStatus::kPremature ||
          it->second.observations.size() < pointLandmarkOptions_.minTrackLengthForMsckf)
        continue;

      // update coordinates of map points, this is only necessary when
      // (1) they are used to predict the points projection in new frames OR
      // (2) to visualize the point quality
      Eigen::AlignedVector<Eigen::Vector2d> obsInPixel;
      std::vector<uint64_t> frameIds;
      std::vector<double> vRi;  // std noise in pixels

      msckf::PointLandmark pointLandmark(msckf::HomogeneousPointParameterization::kModelId);
      msckf::PointSharedData psd;
      msckf::TriangulationStatus status =
          triangulateAMapPoint(it->second, obsInPixel, pointLandmark, vRi,
                               &psd, nullptr, false);
      if (status.triangulationOk) {
        it->second.quality = 1.0;
        Eigen::Map<Eigen::Vector4d> v4Xhomog(pointLandmark.data(), 4);
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
