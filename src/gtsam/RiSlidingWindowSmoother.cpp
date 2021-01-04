/**
 * @file   RiSlidingWindowSmoother.cpp
 * @brief  Implementation of RiSlidingWindowSmoother which wraps
 * gtsam::FixedLagSmoother for VIO with right invariant error formulation.
 * @author Jianzhu Huai
 */

#include <gtsam/RiSlidingWindowSmoother.hpp>

#include <glog/logging.h>

#include <gtsam/nonlinear/Marginals.h>

#include <gtsam/ImuFactorTestHelpers.h>
#include <gtsam/RiExtendedPose3Prior.h>
#include <gtsam/RiImuFactor.h>
#include <gtsam/RiProjectionFactorIDP.h>
#include <gtsam/RiProjectionFactorIDPAnchor.h>
#include <gtsam/RiSmartProjectionFactor.h>

#include <loop_closure/GtsamWrap.hpp>

#include <msckf/FeatureTriangulation.hpp>

#include <okvis/IdProvider.hpp>
#include <okvis/ceres/ImuError.hpp>

DEFINE_bool(rifls_lock_jacobian, true,
            "Lock jacobians for RiExtendedPose3Prior and RiImuFactor?");

DEFINE_double(rifls_huber_threshold, -1,
              "if negative, Gaussian loss will be used, otherwise, robust "
              "projection factor with this huber threshold (say 2.447) will be "
              "used. However, it will be much slower.");

namespace okvis {
RiSlidingWindowSmoother::RiSlidingWindowSmoother(
    const okvis::BackendParams& vioParams,
    std::shared_ptr<okvis::ceres::Map> mapPtr)
    : SlidingWindowSmoother(vioParams, mapPtr) {}

RiSlidingWindowSmoother::RiSlidingWindowSmoother(
    const okvis::BackendParams& vioParams)
    : SlidingWindowSmoother(vioParams) {}

RiSlidingWindowSmoother::~RiSlidingWindowSmoother() {}

void RiSlidingWindowSmoother::addInitialPriorFactors() {
  uint64_t frameId = statesMap_.rbegin()->first;
  okvis::kinematics::Transformation T_WB;
  get_T_WS(frameId, T_WB);

  Eigen::Matrix<double, 9, 1> vel_bias;
  getSpeedAndBias(frameId, 0u, vel_bias);

  Eigen::Matrix<double, 9, 9> state_prior_covariance =
      Eigen::Matrix<double, 9, 9>::Zero();
  state_prior_covariance.diagonal().head<3>() = pvstd_.std_q_WS.cwiseAbs2();
  state_prior_covariance.diagonal().segment<3>(3) = pvstd_.std_v_WS.cwiseAbs2();
  state_prior_covariance.diagonal().tail(3) = pvstd_.std_p_WS.cwiseAbs2();

  // Add RVP prior.
  // Lock Jacobian for the prior factor is unneeded.
  new_imu_prior_and_other_factors_.push_back(
      boost::make_shared<gtsam::RiExtendedPose3Prior>(
          gtsam::Symbol('x', frameId),
          gtsam::RiExtendedPose3(gtsam::Rot3(T_WB.q()), vel_bias.head<3>(),
                                 T_WB.r()),
          state_prior_covariance, false));

  // Add initial bias priors.
  Eigen::Matrix<double, 6, 1> prior_biasSigmas;
  prior_biasSigmas.head<3>().setConstant(imuParametersVec_.at(0).sigma_ba);
  prior_biasSigmas.tail<3>().setConstant(imuParametersVec_.at(0).sigma_bg);
  gtsam::SharedNoiseModel imu_bias_prior_noise =
      gtsam::noiseModel::Diagonal::Sigmas(prior_biasSigmas);
  Eigen::Vector3d bg = vel_bias.segment<3>(3);
  Eigen::Vector3d ba = vel_bias.tail<3>();
  gtsam::imuBias::ConstantBias imuBias(ba, bg);
  new_imu_prior_and_other_factors_.push_back(
      boost::make_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
          gtsam::Symbol('b', frameId), imuBias, imu_bias_prior_noise));
}

void RiSlidingWindowSmoother::addImuValues() {
  uint64_t cur_id = statesMap_.rbegin()->first;
  okvis::kinematics::Transformation T_WB;
  get_T_WS(cur_id, T_WB);
  Eigen::Matrix<double, 9, 1> vel_bias;
  getSpeedAndBias(cur_id, 0u, vel_bias);

  new_values_.insert(gtsam::Symbol('x', cur_id),
                     gtsam::RiExtendedPose3(gtsam::Rot3(T_WB.q()),
                                            vel_bias.head<3>(), T_WB.r()));
  Eigen::Vector3d bg = vel_bias.segment<3>(3);
  Eigen::Vector3d ba = vel_bias.tail<3>();
  gtsam::imuBias::ConstantBias imuBias(ba, bg);
  new_values_.insert(gtsam::Symbol('b', cur_id), imuBias);
  navStateToLandmarks_.insert({cur_id, std::vector<uint64_t>()});
}

void RiSlidingWindowSmoother::addImuFactor() {
  auto lastElementIter = statesMap_.rbegin();
  auto penultimateElementIter = lastElementIter;
  ++penultimateElementIter;

  okvis::Time t_start = penultimateElementIter->second.timestamp;
  okvis::Time t_end = lastElementIter->second.timestamp;
  okvis::ImuMeasurementDeque imuMeasurements =
      inertialMeasForStates_.find(t_start, t_end);

  okvis::kinematics::Transformation T_WB;
  get_T_WS(penultimateElementIter->first, T_WB);
  Eigen::Matrix<double, 9, 1> speedAndBias;
  getSpeedAndBias(penultimateElementIter->first, 0u, speedAndBias);

  gtsam::RiExtendedPose3 extendedPosei = gtsam::RiExtendedPose3(
      gtsam::Rot3(T_WB.q()), speedAndBias.head<3>(), T_WB.r());

  gtsam::imuBias::ConstantBias biasi = gtsam::imuBias::ConstantBias(
      speedAndBias.tail<3>(), speedAndBias.segment<3>(3));

  gtsam::RiPreintegratedImuMeasurements ripim(
      imuMeasurements, imuParametersVec_.at(0u), t_start, t_end);
  ripim.redoPreintegration(extendedPosei, biasi);
  uint64_t from_id = penultimateElementIter->first;
  uint64_t to_id = lastElementIter->first;

  new_imu_prior_and_other_factors_.push_back(
      boost::make_shared<gtsam::RiImuFactor>(
          gtsam::Symbol('x', from_id), gtsam::Symbol('x', to_id),
          gtsam::Symbol('b', from_id), ripim, FLAGS_rifls_lock_jacobian));

  gtsam::imuBias::ConstantBias zero_bias;

  // 1/sqrt(nominalImuRate_) to discretize, then
  // sqrt(deltaTij() * nominalImuRate_) to count the nr of measurements.
  const double d = std::sqrt((t_end - t_start).toSec());
  Eigen::Matrix<double, 6, 1> biasSigmas;
  biasSigmas.head<3>().setConstant(d * imuParams_.acc_walk_);
  biasSigmas.tail<3>().setConstant(d * imuParams_.gyro_walk_);
  const gtsam::SharedNoiseModel& bias_noise_model =
      gtsam::noiseModel::Diagonal::Sigmas(biasSigmas);

  new_imu_prior_and_other_factors_.push_back(
      boost::make_shared<gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>>(
          gtsam::Symbol('b', from_id), gtsam::Symbol('b', to_id), zero_bias,
          bias_noise_model));
}

bool RiSlidingWindowSmoother::addLandmarkToGraph(
    uint64_t lmkId, const Eigen::Vector4d& hpW) {
  std::shared_ptr<okvis::ceres::HomogeneousPointParameterBlock>
      pointParameterBlock(
          new okvis::ceres::HomogeneousPointParameterBlock(hpW, lmkId));
  mapPtr_->addParameterBlock(pointParameterBlock,
                             okvis::ceres::Map::HomogeneousPoint);
  // addParameterBlock may fail when a landmark moves out of the smoother's
  // horizon and then reappears.

  uint64_t currentFrameId = statesMap_.rbegin()->first;
  okvis::kinematics::Transformation T_WB;
  get_T_WS(currentFrameId, T_WB);
  okvis::kinematics::Transformation T_CW = (T_WB * camera_rig_.getCameraExtrinsic(0)).inverse();
  Eigen::Vector4d hpC = T_CW * hpW;
  double rho = 1.0 / hpC[2];
  gtsam::Point3 abrho(hpC[0] * rho, hpC[1] * rho, rho);
  new_values_.insert(gtsam::symbol('l', lmkId), abrho);

  navStateToLandmarks_.at(currentFrameId).push_back(lmkId);

  uint64_t minValidStateId = statesMap_.begin()->first;
  okvis::MapPoint& mp = landmarksMap_.at(lmkId);
  IsObservedInFrame lastImageId(0u, 0u);
  for (auto obsIter = mp.observations.rbegin();
       obsIter != mp.observations.rend(); ++obsIter) {
    uint64_t observingFrameId = obsIter->first.frameId;
    if (observingFrameId < minValidStateId) {
      // Some observations may be outside the horizon.
      continue;
    }
    if (lastImageId(*obsIter)) {
      //      LOG(WARNING) << "okvis frontend may associate one landmark two
      //      observations in the same image!";
      continue;
    }
    // get the keypoint measurement
    okvis::MultiFramePtr multiFramePtr =
        multiFramePtrMap_.at(observingFrameId);
    Eigen::Vector2d measurement;
    multiFramePtr->getKeypoint(obsIter->first.cameraIndex,
                               obsIter->first.keypointIndex, measurement);

    double size = 1.0;
    multiFramePtr->getKeypointSize(obsIter->first.cameraIndex,
                                   obsIter->first.keypointIndex, size);
    Eigen::Matrix<double, 2, 1> variance;
    variance[0] = size * size / 64.0;
    variance[1] = variance[0];

    // TODO(jhuai): support more than one cameras by extending the gtsam
    // reprojection factor.
    if (obsIter == mp.observations.rbegin()) {
      // The anchor frame is set to the last observation's frame which is
      // essentially the last frame so the anchor frame will persist until the
      // landmark is marginalized.
      // This also means better "locality", see Ila 2017 3DV fast incremental...
      gtsam::RiProjectionFactorIDPAnchor::shared_ptr factor =
          boost::make_shared<gtsam::RiProjectionFactorIDPAnchor>(
              gtsam::Symbol('l', lmkId), variance, measurement,
              camera_rig_.getCameraGeometry(obsIter->first.cameraIndex),
              camera_rig_.getCameraExtrinsic(obsIter->first.cameraIndex),
              camera_rig_.getCameraExtrinsic(0));
      new_reprojection_factors_.add(factor);
      mp.anchorStateId = observingFrameId;
      OKVIS_ASSERT_EQ(Exception, mp.anchorStateId, currentFrameId,
                      "Landmark should have the last observation at the current frame.");
    } else {
      gtsam::RiProjectionFactorIDP::shared_ptr factor;
      if (FLAGS_rifls_huber_threshold > 0.01) {
        factor = boost::make_shared<gtsam::RiProjectionFactorIDP>(
            gtsam::Symbol('x', observingFrameId),
            gtsam::Symbol('x', mp.anchorStateId), gtsam::Symbol('l', lmkId),
            variance, measurement,
            camera_rig_.getCameraGeometry(obsIter->first.cameraIndex),
            camera_rig_.getCameraExtrinsic(obsIter->first.cameraIndex),
            camera_rig_.getCameraExtrinsic(0), FLAGS_rifls_huber_threshold);
      } else {
        factor = boost::make_shared<gtsam::RiProjectionFactorIDP>(
            gtsam::Symbol('x', observingFrameId),
            gtsam::Symbol('x', mp.anchorStateId), gtsam::Symbol('l', lmkId),
            variance, measurement,
            camera_rig_.getCameraGeometry(obsIter->first.cameraIndex),
            camera_rig_.getCameraExtrinsic(obsIter->first.cameraIndex),
            camera_rig_.getCameraExtrinsic(0));
      }
      new_reprojection_factors_.add(factor);
    }
    lastImageId =
        IsObservedInFrame(obsIter->first.frameId, obsIter->first.cameraIndex);
  }
  mp.residualizeCase = InState_TrackedNow;
  return true;
}

void RiSlidingWindowSmoother::updateLandmarkInGraph(uint64_t lmkId) {
  const okvis::MapPoint& mp = landmarksMap_.at(lmkId);
  auto obsIter = mp.observations.rbegin();
  OKVIS_ASSERT_EQ(
      Exception, obsIter->first.frameId, statesMap_.rbegin()->first,
      "Only update landmarks observed in the current frame.");

  // get the keypoint measurement.
  okvis::MultiFramePtr multiFramePtr =
      multiFramePtrMap_.at(obsIter->first.frameId);
  Eigen::Vector2d measurement;
  multiFramePtr->getKeypoint(obsIter->first.cameraIndex,
                             obsIter->first.keypointIndex, measurement);
  double size = 1.0;
  multiFramePtr->getKeypointSize(obsIter->first.cameraIndex,
                                 obsIter->first.keypointIndex, size);

  Eigen::Matrix<double, 2, 1> variance;
  variance[0] = size * size / 64.0;
  variance[1] = variance[0];

  gtsam::RiProjectionFactorIDP::shared_ptr factor;
  if (FLAGS_rifls_huber_threshold > 0.01) {
    factor = boost::make_shared<gtsam::RiProjectionFactorIDP>(
        gtsam::Symbol('x', obsIter->first.frameId),
        gtsam::Symbol('x', mp.anchorStateId), gtsam::Symbol('l', lmkId),
        variance, measurement,
        camera_rig_.getCameraGeometry(obsIter->first.cameraIndex),
        camera_rig_.getCameraExtrinsic(obsIter->first.cameraIndex),
        camera_rig_.getCameraExtrinsic(0), FLAGS_rifls_huber_threshold);
  } else {
    factor = boost::make_shared<gtsam::RiProjectionFactorIDP>(
        gtsam::Symbol('x', obsIter->first.frameId),
        gtsam::Symbol('x', mp.anchorStateId), gtsam::Symbol('l', lmkId),
        variance, measurement,
        camera_rig_.getCameraGeometry(obsIter->first.cameraIndex),
        camera_rig_.getCameraExtrinsic(obsIter->first.cameraIndex),
        camera_rig_.getCameraExtrinsic(0));
  }
  new_reprojection_factors_.add(factor);
}


void RiSlidingWindowSmoother::addLandmarkSmartFactorToGraph(const LandmarkId& lmkId) {
  uint64_t minValidStateId = statesMap_.begin()->first;
  okvis::MapPoint& mp = landmarksMap_.at(lmkId);
  auto obsIt = mp.observations.lower_bound(okvis::KeypointIdentifier(minValidStateId, 0u, 0u));
  size_t numValidObs = std::distance(obsIt, mp.observations.end());
  if (numValidObs < optimizationOptions_.minTrackLength) {
      return;
  }
  std::shared_ptr<okvis::ceres::HomogeneousPointParameterBlock>
      pointParameterBlock(
          new okvis::ceres::HomogeneousPointParameterBlock(mp.pointHomog, lmkId));
  mapPtr_->addParameterBlock(pointParameterBlock,
                             okvis::ceres::Map::HomogeneousPoint);
  // addParameterBlock may fail when a landmark moves out of the smoother's
  // horizon and then reappears.

  gtsam::RiSmartProjectionFactor<gtsam::Cal3DS2>::shared_ptr new_factor =
      boost::make_shared<gtsam::RiSmartProjectionFactor<gtsam::Cal3DS2>>(
          smart_noise_, body_P_cam0_, cal0_, camera_rig_.getCameraGeometry(0),
          smart_factors_params_);

  IsObservedInFrame lastImageId(0u, 0u);
  for (std::map<okvis::KeypointIdentifier, uint64_t>::const_iterator obsIter =
           mp.observations.begin();
       obsIter != mp.observations.end(); ++obsIter) {
    if (obsIter->first.frameId < minValidStateId) {
      // Some observations may be outside the horizon.
      continue;
    }
    if (lastImageId(*obsIter)) {
      //      LOG(WARNING) << "okvis frontend may associate one landmark two
      //      observations in the same image!";
      continue;
    }
    // get the keypoint measurement
    okvis::MultiFramePtr multiFramePtr =
        multiFramePtrMap_.at(obsIter->first.frameId);
    Eigen::Vector2d measurement;
    multiFramePtr->getKeypoint(obsIter->first.cameraIndex,
                               obsIter->first.keypointIndex, measurement);

    new_factor->add(measurement, gtsam::Symbol('x', obsIter->first.frameId));
    lastImageId =
        IsObservedInFrame(obsIter->first.frameId, obsIter->first.cameraIndex);
  }
  mp.residualizeCase = InState_TrackedNow;
  new_factor->setAnchorIndex(new_factor->keys().size() - 1);

  new_smart_factors_.emplace(lmkId, new_factor);
  old_smart_factors_.emplace(lmkId, std::make_pair(new_factor, -1));
}

void RiSlidingWindowSmoother::updateLandmarkSmartFactorInGraph(
    const LandmarkId& lmkId) {
  const okvis::MapPoint& mp = landmarksMap_.at(lmkId);
  auto obsIter = mp.observations.rbegin();
  OKVIS_ASSERT_EQ(Exception, obsIter->first.frameId, statesMap_.rbegin()->first,
                  "Only update landmarks observed in the current frame.");

  // get the keypoint measurement.
  okvis::MultiFramePtr multiFramePtr =
      multiFramePtrMap_.at(obsIter->first.frameId);
  Eigen::Vector2d measurement;
  multiFramePtr->getKeypoint(obsIter->first.cameraIndex,
                             obsIter->first.keypointIndex, measurement);

  // Update existing smart-factor
  auto old_smart_factors_it = old_smart_factors_.find(lmkId);
  CHECK(old_smart_factors_it != old_smart_factors_.end())
      << "Landmark not found in old_smart_factors_ with id: " << lmkId;
  const gtsam::RiSmartProjectionFactor<gtsam::Cal3DS2>::shared_ptr old_factor =
      boost::static_pointer_cast<
          gtsam::RiSmartProjectionFactor<gtsam::Cal3DS2>>(
          old_smart_factors_it->second.first);
  // Clone old factor to keep all previous measurements, now append one.
  gtsam::RiSmartProjectionFactor<gtsam::Cal3DS2>::shared_ptr new_factor =
      boost::make_shared<gtsam::RiSmartProjectionFactor<gtsam::Cal3DS2>>(
          *old_factor);
  new_factor->add(measurement, gtsam::Symbol('x', obsIter->first.frameId));

  // Update the factor
  Slot slot = old_smart_factors_it->second.second;
  if (slot != -1) {
    new_smart_factors_.emplace(lmkId, new_factor);
  } else {
    LOG(FATAL)
        << "If its slot in the graph is still -1, it means that the factor has "
           "not been inserted yet in the graph! Offensive lmk_id: "
        << lmkId;
  }
  old_smart_factors_it->second.first = new_factor;
}

void RiSlidingWindowSmoother::updateStates() {
  // TODO(jhuai): calculateEstimate often throws std::out_of_range exception
  // when the time_horizon is large say 4 with MH_04 EuRoC,
  // Requested variable 'b8736' is not in this VectorValues.
  gtsam::Values estimates = smoother_->calculateEstimate();

  state_ = estimates;

  // update poses, velocities, and biases from isam2 estimates.
  for (auto iter = statesMap_.begin(); iter != statesMap_.end(); ++iter) {
    uint64_t stateId = iter->first;
    auto xval = estimates.find(gtsam::Symbol('x', stateId));
    if (xval == estimates.end()) {
      if (stateId != statesMap_.begin()->first) {
        LOG(INFO) << "State of id " << stateId
                  << " not found in smoother estimates when the first nav "
                     "state id is "
                  << statesMap_.begin()->first;
      }
      continue;
    }

    gtsam::RiExtendedPose3 rvp =
        estimates.at<gtsam::RiExtendedPose3>(gtsam::Symbol('x', stateId));

    std::shared_ptr<ceres::PoseParameterBlock> poseParamBlockPtr =
        std::static_pointer_cast<ceres::PoseParameterBlock>(
            mapPtr_->parameterBlockPtr(stateId));
    kinematics::Transformation T_WB(rvp.position(),
                                    rvp.rotation().toQuaternion());
    poseParamBlockPtr->setEstimate(T_WB);

    // update imu sensor states
    const int imuIdx = 0;
    uint64_t SBId = iter->second.sensors.at(SensorStates::Imu)
                        .at(imuIdx)
                        .at(ImuSensorStates::SpeedAndBias)
                        .id;
    std::shared_ptr<ceres::SpeedAndBiasParameterBlock> sbParamBlockPtr =
        std::static_pointer_cast<ceres::SpeedAndBiasParameterBlock>(
            mapPtr_->parameterBlockPtr(SBId));
    SpeedAndBiases sb = sbParamBlockPtr->estimate();
    sb.head<3>() = rvp.velocity();

    gtsam::imuBias::ConstantBias imuBias =
        estimates.at<gtsam::imuBias::ConstantBias>(
            gtsam::Symbol('b', stateId));
    sb.segment<3>(3) = imuBias.gyroscope();
    sb.tail<3>(3) = imuBias.accelerometer();

    sbParamBlockPtr->setEstimate(sb);
  }

  // TODO(jhuai): update camera extrinsic parameters from isam2 estimates.

  // update landmark positions from isam2 estimates.
  {
    updateLandmarksTimer.start();
    for (auto it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
      if (it->second.residualizeCase == NotInState_NotTrackedNow) continue;
      uint64_t lmkId = it->first;
      auto estimatesIter = estimates.find(gtsam::Symbol('l', lmkId));
      if (estimatesIter == estimates.end()) {
        continue;
      }

      gtsam::Point3 abrho = estimates.at<gtsam::Point3>(gtsam::Symbol('l', lmkId));
      Eigen::Vector4d ab1rho(abrho[0], abrho[1], 1.0, abrho[2]);
      okvis::kinematics::Transformation T_WB;
      get_T_WS(it->second.anchorStateId, T_WB);
      okvis::kinematics::Transformation T_WC =
          T_WB * *camera_rig_.getCameraExtrinsicPtr(0u);
      it->second.quality = 1.0;
      it->second.pointHomog = T_WC * ab1rho;
    }
    updateLandmarksTimer.stop();
  }
}

bool RiSlidingWindowSmoother::gtsamMarginalCovariance(
    Eigen::MatrixXd* cov) const {
  gtsam::Marginals marginals(smoother_->getFactors(), state_,
                             gtsam::Marginals::Factorization::CHOLESKY);
  uint64_t stateId = statesMap_.rbegin()->first;
  okvis::kinematics::Transformation T_WB;
  get_T_WS(stateId, T_WB);
  Eigen::Matrix<double, 9, 1> speedAndBgBa;
  getSpeedAndBias(stateId, 0u, speedAndBgBa);

  *cov = Eigen::Matrix<double, 15, 15>::Identity();
  Eigen::Matrix<double, 15, 15> dokvis_drierror =
      gtsam::dokvis_drightinvariant(T_WB, speedAndBgBa.head<3>());
  Eigen::MatrixXd rvpMargCov =
      marginals.marginalCovariance(gtsam::Symbol('x', stateId));
  cov->topLeftCorner<9, 9>() =
      dokvis_drierror.topLeftCorner<9, 9>() * rvpMargCov *
      dokvis_drierror.topLeftCorner<9, 9>().transpose();
  cov->block<6, 6>(9, 9) =
      dokvis_drierror.bottomRightCorner<6, 6>() *
      marginals.marginalCovariance(gtsam::Symbol('b', stateId)) *
      dokvis_drierror.bottomRightCorner<6, 6>().transpose();
  return true;
}
}  // namespace okvis
