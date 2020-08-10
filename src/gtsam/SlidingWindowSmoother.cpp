#include <gtsam/SlidingWindowSmoother.hpp>

#include <glog/logging.h>

#include <okvis/ceres/ImuError.hpp>
#include <okvis/IdProvider.hpp>

#include <loop_closure/GtsamWrap.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/ImuFrontEnd.h>
#include <gtsam/navigation/NavState.h>

DEFINE_bool(use_combined_imu_factor, false,
            "CombinedImuFactor(PreintegratedCombinedMeasurement) or "
            "ImuFactor(PreintegratedImuMeasurement)");
/// \brief okvis Main namespace of this package.
namespace okvis {

/* -------------------------------------------------------------------------- */
// Set parameters for ISAM 2 incremental smoother.
void setIsam2Params(const okvis::BackendParams& vio_params,
                    gtsam::ISAM2Params* isam_param) {
  CHECK_NOTNULL(isam_param);
  // iSAM2 SETTINGS
  if (vio_params.useDogLeg_) {
    gtsam::ISAM2DoglegParams dogleg_params;
    dogleg_params.wildfireThreshold = vio_params.wildfire_threshold_;
    // dogleg_params.adaptationMode;
    // dogleg_params.initialDelta;
    // dogleg_params.setVerbose(false); // only for debugging.
    isam_param->optimizationParams = dogleg_params;
  } else {
    gtsam::ISAM2GaussNewtonParams gauss_newton_params;
    gauss_newton_params.wildfireThreshold = vio_params.wildfire_threshold_;
    isam_param->optimizationParams = gauss_newton_params;
  }

  // TODO Luca: Here there was commented code about setRelinearizeThreshold.
  // was it important?
  // gtsam::FastMap<char,gtsam::Vector> thresholds;
  // gtsam::Vector xThresh(6); // = {0.05, 0.05, 0.05, 0.1, 0.1, 0.1};
  // gtsam::Vector vThresh(3); //= {1.0, 1.0, 1.0};
  // gtsam::Vector bThresh(6); // = {1.0, 1.0, 1.0};
  // xThresh << relinearizeThresholdRot_, relinearizeThresholdRot_,
  // relinearizeThresholdRot_, relinearizeThresholdPos_,
  // relinearizeThresholdPos_, relinearizeThresholdPos_; vThresh <<
  // relinearizeThresholdVel_, relinearizeThresholdVel_,
  // relinearizeThresholdVel_; bThresh << relinearizeThresholdIMU_,
  // relinearizeThresholdIMU_, relinearizeThresholdIMU_,
  // relinearizeThresholdIMU_, relinearizeThresholdIMU_,
  // relinearizeThresholdIMU_; thresholds['x'] = xThresh; thresholds['v'] =
  // vThresh; thresholds['b'] = bThresh;
  // isam_param.setRelinearizeThreshold(thresholds);

  // TODO (Toni): remove hardcoded
  // Cache Linearized Factors seems to improve performance.
  isam_param->setCacheLinearizedFactors(true);
  isam_param->relinearizeThreshold = vio_params.relinearizeThreshold_;
  isam_param->relinearizeSkip = vio_params.relinearizeSkip_;
  isam_param->findUnusedFactorSlots = true;
  // isam_param->enablePartialRelinearizationCheck = true;
  isam_param->setEvaluateNonlinearError(false);  // only for debugging
  isam_param->enableDetailedResults = false;     // only for debugging.
  isam_param->factorization = gtsam::ISAM2Params::CHOLESKY;  // QR
}

void SlidingWindowSmoother::setupSmoother(
    const okvis::BackendParams& vioParams) {
#ifdef INCREMENTAL_SMOOTHER
  gtsam::ISAM2Params isam_param;
  setIsam2Params(vioParams, &isam_param);
  smoother_.reset(new Smoother(vioParams.horizon_, isam_param));
#else  // BATCH SMOOTHER
  gtsam::LevenbergMarquardtParams lmParams;
  lmParams.setlambdaInitial(0.0);     // same as GN
  lmParams.setlambdaLowerBound(0.0);  // same as GN
  lmParams.setlambdaUpperBound(0.0);  // same as GN)
  smoother_ =
      std::shared_ptr<Smoother>(new Smoother(vioParams.horizon_, lmParams));
#endif
}

SlidingWindowSmoother::SlidingWindowSmoother(
    const okvis::BackendParams& vioParams,
    std::shared_ptr<okvis::ceres::Map> mapPtr)
    : Estimator(mapPtr),
      backendParams_(vioParams),
      addLandmarkFactorsTimer("3.1 addLandmarkFactors", true),
      isam2UpdateTimer("3.2 isam2Update", true),
      computeCovarianceTimer("3.3 computeCovariance", true),
      marginalizeTimer("3.4 marginalize", true),
      updateLandmarksTimer("3.5 updateLandmarks", true) {
  setupSmoother(vioParams);
}

// The default constructor.
SlidingWindowSmoother::SlidingWindowSmoother(const okvis::BackendParams& vioParams)
    : Estimator(),
      addLandmarkFactorsTimer("3.1 addLandmarkFactors", true),
      isam2UpdateTimer("3.2 isam2Update", true),
      computeCovarianceTimer("3.3 computeCovariance", true),
      marginalizeTimer("3.4 marginalize", true),
      updateLandmarksTimer("3.5 updateLandmarks", true) {
  setupSmoother(vioParams);
}

SlidingWindowSmoother::~SlidingWindowSmoother() {}

void SlidingWindowSmoother::addInitialPriorFactors() {
  // Set initial covariance for inertial factors
  // W_Pose_Blkf_ set by motion capture to start with
  uint64_t frameId = statesMap_.rbegin()->first;
  okvis::kinematics::Transformation T_WB;
  get_T_WS(frameId, T_WB);
  Eigen::Matrix<double, 9, 1> vel_bias;
  getSpeedAndBias(frameId, 0u, vel_bias);
  Eigen::Matrix3d B_Rot_W = T_WB.C().transpose();

  // Set initial pose uncertainty: constrain mainly position and global yaw.
  // roll and pitch is observable, therefore low variance.
  Eigen::Matrix<double, 6, 6> pose_prior_covariance = Eigen::Matrix<double, 6, 6>::Zero();
  pose_prior_covariance.diagonal()[0] = backendParams_.initialRollPitchSigma_ *
                                        backendParams_.initialRollPitchSigma_;
  pose_prior_covariance.diagonal()[1] = backendParams_.initialRollPitchSigma_ *
                                        backendParams_.initialRollPitchSigma_;
  pose_prior_covariance.diagonal()[2] =
      backendParams_.initialYawSigma_ * backendParams_.initialYawSigma_;
  pose_prior_covariance.diagonal()[3] = backendParams_.initialPositionSigma_ *
                                        backendParams_.initialPositionSigma_;
  pose_prior_covariance.diagonal()[4] = backendParams_.initialPositionSigma_ *
                                        backendParams_.initialPositionSigma_;
  pose_prior_covariance.diagonal()[5] = backendParams_.initialPositionSigma_ *
                                        backendParams_.initialPositionSigma_;

  // Rotate initial uncertainty into local frame, where the uncertainty is
  // specified.
  pose_prior_covariance.topLeftCorner(3, 3) =
      B_Rot_W * pose_prior_covariance.topLeftCorner(3, 3) * B_Rot_W.transpose();

  // Add pose prior.
  // TODO(Toni): Make this noise model a member constant.
  gtsam::SharedNoiseModel noise_init_pose =
      gtsam::noiseModel::Gaussian::Covariance(pose_prior_covariance);
  new_imu_prior_and_other_factors_.push_back(
      boost::make_shared<gtsam::PriorFactor<gtsam::Pose3>>(
          gtsam::Symbol('x', frameId), VIO::GtsamWrap::toPose3(T_WB), noise_init_pose));

  // Add initial velocity priors.
  // TODO(Toni): Make this noise model a member constant.
  gtsam::SharedNoiseModel noise_init_vel_prior =
      gtsam::noiseModel::Isotropic::Sigma(
          3, backendParams_.initialVelocitySigma_);
  Eigen::Vector3d vel = vel_bias.head<3>();
  new_imu_prior_and_other_factors_.push_back(
      boost::make_shared<gtsam::PriorFactor<gtsam::Vector3>>(
          gtsam::Symbol('v', frameId), vel, noise_init_vel_prior));

  // Add initial bias priors:
  Eigen::Matrix<double, 6, 1> prior_biasSigmas;
  prior_biasSigmas.head<3>().setConstant(backendParams_.initialAccBiasSigma_);
  prior_biasSigmas.tail<3>().setConstant(backendParams_.initialGyroBiasSigma_);
  // TODO(Toni): Make this noise model a member constant.
  gtsam::SharedNoiseModel imu_bias_prior_noise =
      gtsam::noiseModel::Diagonal::Sigmas(prior_biasSigmas);
  Eigen::Vector3d bg = vel_bias.segment<3>(3);
  Eigen::Vector3d ba = vel_bias.tail<3>();
  gtsam::imuBias::ConstantBias imuBias(ba, bg);
  new_imu_prior_and_other_factors_.push_back(
      boost::make_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
          gtsam::Symbol('b', frameId), imuBias, imu_bias_prior_noise));

  VLOG(2) << "Added initial priors for frame " << frameId;
}

void SlidingWindowSmoother::addImuValues() {
  uint64_t cur_id = statesMap_.rbegin()->first;
  okvis::kinematics::Transformation T_WB;
  get_T_WS(cur_id, T_WB);
  Eigen::Matrix<double, 9, 1> vel_bias;
  getSpeedAndBias(cur_id, 0u, vel_bias);
  // Update state with initial guess
  new_values_.insert(gtsam::Symbol('x', cur_id), VIO::GtsamWrap::toPose3(T_WB));
  Eigen::Vector3d vel = vel_bias.head<3>();
  new_values_.insert(gtsam::Symbol('v', cur_id), vel);
  Eigen::Vector3d bg = vel_bias.segment<3>(3);
  Eigen::Vector3d ba = vel_bias.tail<3>();
  gtsam::imuBias::ConstantBias imuBias(ba, bg);
  new_values_.insert(gtsam::Symbol('b', cur_id), imuBias);
}

void SlidingWindowSmoother::addImuFactor() {
  auto lastElementIter = statesMap_.rbegin();
  auto penultimateElementIter = lastElementIter;
  ++penultimateElementIter;

  okvis::Time t_start = penultimateElementIter->second.timestamp;
  okvis::Time t_end = lastElementIter->second.timestamp;
  okvis::ImuMeasurementDeque imuMeasurements =
      inertialMeasForStates_.find(t_start, t_end);
  Eigen::Matrix<double, 9, 1> speedAndBias;
  getSpeedAndBias(penultimateElementIter->first, 0u, speedAndBias);
  ImuFrontEnd::PimPtr resultPim;
  imuFrontend_->preintegrateImuMeasurements(imuMeasurements, speedAndBias,
                                            t_start, t_end, resultPim);
  uint64_t from_id = penultimateElementIter->first;
  uint64_t to_id = lastElementIter->first;
  switch (imuParams_.imu_preintegration_type_) {
    case ImuPreintegrationType::kPreintegratedCombinedMeasurements:
      new_imu_prior_and_other_factors_.push_back(
          boost::make_shared<gtsam::CombinedImuFactor>(
              gtsam::Symbol('x', from_id), gtsam::Symbol('v', from_id),
              gtsam::Symbol('x', to_id), gtsam::Symbol('v', to_id),
              gtsam::Symbol('b', from_id), gtsam::Symbol('b', to_id),
              safeCastToPreintegratedCombinedImuMeasurements(*resultPim)));
      break;

    case ImuPreintegrationType::kPreintegratedImuMeasurements: {
      new_imu_prior_and_other_factors_.push_back(
          boost::make_shared<gtsam::ImuFactor>(
              gtsam::Symbol('x', from_id), gtsam::Symbol('v', from_id),
              gtsam::Symbol('x', to_id), gtsam::Symbol('v', to_id),
              gtsam::Symbol('b', from_id),
              safeCastToPreintegratedImuMeasurements(*resultPim)));

      gtsam::imuBias::ConstantBias zero_bias;
      // Factor to discretize and move normalize by the interval between
      // measurements:
      CHECK_NE(imuParams_.nominal_rate_, 0.0)
          << "Nominal IMU rate param cannot be 0.";
      // 1/sqrt(nominalImuRate_) to discretize, then
      // sqrt(pim_->deltaTij()/nominalImuRate_) to count the nr of measurements.
      const double d =
          std::sqrt(resultPim->deltaTij()) / imuParams_.nominal_rate_;
      Eigen::Matrix<double, 6, 1> biasSigmas;
      biasSigmas.head<3>().setConstant(d * imuParams_.acc_walk_);
      biasSigmas.tail<3>().setConstant(d * imuParams_.gyro_walk_);
      const gtsam::SharedNoiseModel& bias_noise_model =
          gtsam::noiseModel::Diagonal::Sigmas(biasSigmas);

      new_imu_prior_and_other_factors_.push_back(
          boost::make_shared<
              gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>>(
              gtsam::Symbol('b', from_id), gtsam::Symbol('b', to_id), zero_bias,
              bias_noise_model));
      break;
    }
    default:
      LOG(FATAL) << "Unknown IMU Preintegration Type.";
      break;
  }
}

int SlidingWindowSmoother::addImu(const okvis::ImuParameters& imuParameters) {
  int imuIndex = Estimator::addImu(imuParameters);

  imuParams_.set(imuParameters);

  imuFrontend_ = std::unique_ptr<ImuFrontEnd>(new ImuFrontEnd(imuParams_));
  return imuIndex;
}

void SlidingWindowSmoother::addCameraExtrinsicFactor() {
  // add relative sensor state errors
  auto lastElementIter = statesMap_.rbegin();
  auto penultimateElementIter = lastElementIter;
  ++penultimateElementIter;
//  uint64_t from_id = penultimateElementIter->first;
//  uint64_t to_id = lastElementIter->first;

  for (size_t i = 0; i < extrinsicsEstimationParametersVec_.size(); ++i) {
    if (penultimateElementIter->second.sensors.at(SensorStates::Camera)
            .at(i)
            .at(CameraSensorStates::T_SCi)
            .id != lastElementIter->second.sensors.at(SensorStates::Camera)
                       .at(i)
                       .at(CameraSensorStates::T_SCi)
                       .id) {
      // i.e. they are different estimated variables, so link them with a
      // temporal error term
      double dt = (lastElementIter->second.timestamp -
                   penultimateElementIter->second.timestamp)
                      .toSec();
      double translationSigmaC =
          extrinsicsEstimationParametersVec_.at(i).sigma_c_relative_translation;
      double translationVariance = translationSigmaC * translationSigmaC * dt;
      double rotationSigmaC =
          extrinsicsEstimationParametersVec_.at(i).sigma_c_relative_orientation;
      double rotationVariance = rotationSigmaC * rotationSigmaC * dt;

      Eigen::Matrix<double, 6, 1> variances;
      variances.head<3>().setConstant(rotationVariance);
      variances.tail<3>().setConstant(translationVariance);
      gtsam::SharedNoiseModel betweenNoise_ =
          gtsam::noiseModel::Diagonal::Variances(variances);

      gtsam::Pose3 from_id_Pose_to_id = gtsam::Pose3::identity();
      new_imu_prior_and_other_factors_.push_back(
          boost::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
              gtsam::Symbol('C', penultimateElementIter->second.sensors
                                     .at(SensorStates::Camera)
                                     .at(i)
                                     .at(CameraSensorStates::T_SCi)
                                     .id),
              gtsam::Symbol(
                  'C', lastElementIter->second.sensors.at(SensorStates::Camera)
                           .at(i)
                           .at(CameraSensorStates::T_SCi)
                           .id),
              from_id_Pose_to_id, betweenNoise_));
    }
  }
}

// add states to the factor graph, nav states, biases, record their ids in
// statesMap_
bool SlidingWindowSmoother::addStates(
    okvis::MultiFramePtr multiFrame,
    const okvis::ImuMeasurementDeque& imuMeasurements, bool asKeyframe) {
  // note: this is before matching...
  // TODO !!
  // record the imu measurements between two consecutive states
  inertialMeasForStates_.push_back(imuMeasurements);
  okvis::kinematics::Transformation T_WS;
  Eigen::Matrix<double, 9, 1> speedAndBias;
  if (statesMap_.empty()) {
    // in case this is the first frame ever, let's initialize the pose:
    if (pvstd_.initWithExternalSource)
      T_WS = okvis::kinematics::Transformation(pvstd_.p_WS, pvstd_.q_WS);
    else {
      bool success0 = initPoseFromImu(imuMeasurements, T_WS);
      OKVIS_ASSERT_TRUE_DBG(
          Exception, success0,
          "pose could not be initialized from imu measurements.");
      if (!success0) return false;
      pvstd_.updatePose(T_WS, multiFrame->timestamp());
    }
    speedAndBias.setZero();
    speedAndBias.head<3>() = pvstd_.v_WS;
    speedAndBias.segment<3>(6) = imuParametersVec_.at(0).a0;
  } else {
    // get the previous states
    uint64_t T_WS_id = statesMap_.rbegin()->second.id;
    uint64_t speedAndBias_id = statesMap_.rbegin()
                                   ->second.sensors.at(SensorStates::Imu)
                                   .at(0)
                                   .at(ImuSensorStates::SpeedAndBias)
                                   .id;
    OKVIS_ASSERT_TRUE_DBG(
        Exception, mapPtr_->parameterBlockExists(T_WS_id),
        "this is an okvis bug. previous pose does not exist.");
    T_WS = std::static_pointer_cast<ceres::PoseParameterBlock>(
               mapPtr_->parameterBlockPtr(T_WS_id))
               ->estimate();
    speedAndBias = std::static_pointer_cast<ceres::SpeedAndBiasParameterBlock>(
                       mapPtr_->parameterBlockPtr(speedAndBias_id))
                       ->estimate();

    // propagate pose and speedAndBias
    int numUsedImuMeasurements = ceres::ImuError::propagation(
        imuMeasurements, imuParametersVec_.at(0), T_WS, speedAndBias,
        statesMap_.rbegin()->second.timestamp, multiFrame->timestamp());
    OKVIS_ASSERT_TRUE_DBG(Exception, numUsedImuMeasurements > 1,
                          "propagation failed");
    if (numUsedImuMeasurements < 1) {
      LOG(INFO) << "numUsedImuMeasurements=" << numUsedImuMeasurements;
      return false;
    }
  }

  // create a states object:
  States states(asKeyframe, multiFrame->id(), multiFrame->timestamp());

  // check if id was used before
  OKVIS_ASSERT_TRUE_DBG(Exception,
                        statesMap_.find(states.id) == statesMap_.end(),
                        "pose ID" << states.id << " was used before!");

  // create global states
  std::shared_ptr<okvis::ceres::PoseParameterBlock> poseParameterBlock(
      new okvis::ceres::PoseParameterBlock(T_WS, states.id,
                                           multiFrame->timestamp()));
  states.global.at(GlobalStates::T_WS).exists = true;
  states.global.at(GlobalStates::T_WS).id = states.id;

  if (statesMap_.empty()) {
    referencePoseId_ = states.id;  // set this as reference pose
    if (!mapPtr_->addParameterBlock(poseParameterBlock, ceres::Map::Pose6d)) {
      return false;
    }
  } else {
    if (!mapPtr_->addParameterBlock(poseParameterBlock, ceres::Map::Pose6d)) {
      return false;
    }
  }

  // add to buffer
  statesMap_.insert(std::pair<uint64_t, States>(states.id, states));
  multiFramePtrMap_.insert(
      std::pair<uint64_t, okvis::MultiFramePtr>(states.id, multiFrame));

  // the following will point to the last states:
  std::map<uint64_t, States>::reverse_iterator lastElementIterator =
      statesMap_.rbegin();
  lastElementIterator++;

  // initialize new sensor states
  // cameras:
  for (size_t i = 0; i < extrinsicsEstimationParametersVec_.size(); ++i) {
    SpecificSensorStatesContainer cameraInfos(2);
    cameraInfos.at(CameraSensorStates::T_SCi).exists = true;
    cameraInfos.at(CameraSensorStates::Intrinsics).exists = false;
    if (((extrinsicsEstimationParametersVec_.at(i)
              .sigma_c_relative_translation < 1e-12) ||
         (extrinsicsEstimationParametersVec_.at(i)
              .sigma_c_relative_orientation < 1e-12)) &&
        (statesMap_.size() > 1)) {
      // use the same block...
      cameraInfos.at(CameraSensorStates::T_SCi).id =
          lastElementIterator->second.sensors.at(SensorStates::Camera)
              .at(i)
              .at(CameraSensorStates::T_SCi)
              .id;
    } else {
      const okvis::kinematics::Transformation T_SC = *multiFrame->T_SC(i);
      uint64_t id = IdProvider::instance().newId();
      std::shared_ptr<okvis::ceres::PoseParameterBlock>
          extrinsicsParameterBlockPtr(new okvis::ceres::PoseParameterBlock(
              T_SC, id, multiFrame->timestamp()));
      if (!mapPtr_->addParameterBlock(extrinsicsParameterBlockPtr,
                                      ceres::Map::Pose6d)) {
        return false;
      }
      cameraInfos.at(CameraSensorStates::T_SCi).id = id;
    }
    // update the states info
    statesMap_.rbegin()
        ->second.sensors.at(SensorStates::Camera)
        .push_back(cameraInfos);
    states.sensors.at(SensorStates::Camera).push_back(cameraInfos);
  }

  // IMU states are automatically propagated.
  for (size_t i = 0; i < imuParametersVec_.size(); ++i) {
    SpecificSensorStatesContainer imuInfo(2);
    imuInfo.at(ImuSensorStates::SpeedAndBias).exists = true;
    uint64_t id = IdProvider::instance().newId();
    std::shared_ptr<okvis::ceres::SpeedAndBiasParameterBlock>
        speedAndBiasParameterBlock(new okvis::ceres::SpeedAndBiasParameterBlock(
            speedAndBias, id, multiFrame->timestamp()));

    if (!mapPtr_->addParameterBlock(speedAndBiasParameterBlock)) {
      return false;
    }
    imuInfo.at(ImuSensorStates::SpeedAndBias).id = id;
    statesMap_.rbegin()
        ->second.sensors.at(SensorStates::Imu)
        .push_back(imuInfo);
    states.sensors.at(SensorStates::Imu).push_back(imuInfo);
  }

  addImuValues();

  // depending on whether or not this is the very beginning, we will add priors
  // or relative terms to the last state:
  if (statesMap_.size() == 1) {
    addInitialPriorFactors();
  } else {
    addImuFactor();
    addCameraExtrinsicFactor();
  }

  return true;
}


uint64_t SlidingWindowSmoother::getMinValidStateId() const {
  uint64_t min_state_id = statesMap_.rbegin()->first;
  okvis::Time currentFrameTime = statesMap_.rbegin()->second.timestamp;
  okvis::Time horizonTime = currentFrameTime - okvis::Duration(backendParams_.horizon_);
  std::map<uint64_t, States>::const_reverse_iterator rit = statesMap_.rbegin();
  while (rit != statesMap_.rend()) {
    if (rit->second.timestamp < horizonTime) {
      min_state_id = rit->first;
      break;
    }
    ++rit;
  }
  return min_state_id;
}

// the major job of marginalization is done in factorgraph optimization step
// here we only remove old landmarks and states from bookkeeping.
bool SlidingWindowSmoother::applyMarginalizationStrategy(
    size_t /*numKeyframes*/, size_t /*numImuFrames*/,
    okvis::MapPointVector& removedLandmarks) {
  uint64_t minValidStateId = getMinValidStateId();
  std::vector<uint64_t> removeFrames;
  std::map<uint64_t, States>::reverse_iterator rit = statesMap_.rbegin();
  while (rit != statesMap_.rend()) {
    if (rit->first < minValidStateId) {
      removeFrames.push_back(rit->second.id);
    }
    ++rit;
  }

  // remove feature tracks that are out of the sliding window. How?
  // Kimera-VIO simply removes old smart factors out of the time horizon, see
  // https://github.com/MIT-SPARK/Kimera-VIO/blob/master/src/backend/VioBackEnd.cpp#L926-L929.
  // Does Kimera-VIO use anchored inverse depth coordinates or world Euclidean coordinates?
  // World Euclidean coordinates because smart factors depends on camera poses which
  // are expressed in the world frame.
  okvis::Time currentFrameTime = statesMap_.rbegin()->second.timestamp;
  okvis::Time horizonTime = currentFrameTime - okvis::Duration(backendParams_.horizon_);
  for (PointMap::iterator pit = landmarksMap_.begin();
       pit != landmarksMap_.end();) {
    const MapPoint& mapPoint = pit->second;
    // Remove a landmark whose last observation is out of the horizon.
    uint64_t lastFrameId = mapPoint.observations.rbegin()->first.frameId;
    if (lastFrameId < minValidStateId) {
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

  return true;
}

double landmarkQuality(std::shared_ptr<okvis::ceres::Map> mapPtr,
                       uint64_t landmarkId) {
  Eigen::MatrixXd H(3, 3);
  mapPtr->getLhs(landmarkId, H);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(H);
  Eigen::Vector3d eigenvalues = saes.eigenvalues();
  const double smallest = (eigenvalues[0]);
  const double largest = (eigenvalues[2]);
  double quality = 0.0;
  if (smallest < 1.0e-12) {
    // this means, it has a non-observable depth
    quality = 0.0;
  } else {
    // OK, well constrained
    quality = sqrt(smallest) / sqrt(largest);
  }
  return quality;
}


/* -------------------------------------------------------------------------- */
// Adds a landmark to the graph for the first time.
void SlidingWindowSmoother::addLandmarkToGraph(uint64_t lmk_id) {
  // We use a unit pinhole projection camera for the smart factors to be
  // more efficient.
//  SmartStereoFactor::shared_ptr new_factor =
//      boost::make_shared<SmartStereoFactor>(
//          smart_noise_, smart_factors_params_, B_Pose_leftCam_);

//  VLOG(10) << "Adding landmark with: " << ft.obs_.size()
//           << " landmarks to graph, with keys: ";

//  // Add observations to smart factor
//  if (VLOG_IS_ON(10)) new_factor->print();
//  for (const std::pair<FrameId, StereoPoint2>& obs : ft.obs_) {
//    const FrameId& frame_id = obs.first;
//    const gtsam::Symbol& pose_symbol = gtsam::Symbol('x', frame_id);
//    if (smoother_->getFactors().exists(pose_symbol)) {
//      const StereoPoint2& measurement = obs.second;
//      new_factor->add(measurement, pose_symbol, stereo_cal_);
//    } else {
//      VLOG(10) << "Factor with lmk id " << lmk_id
//               << " is linking to a marginalized state!";
//    }

//    if (VLOG_IS_ON(10)) std::cout << " " << obs.first;
//  }
//  if (VLOG_IS_ON(10)) std::cout << std::endl;

//  // add new factor to suitable structures:
//  new_smart_factors_.insert(std::make_pair(lmk_id, new_factor));
//  old_smart_factors_.insert(
//      std::make_pair(lmk_id, std::make_pair(new_factor, -1)));
}

/* -------------------------------------------------------------------------- */
// Updates a landmark already in the graph.
void SlidingWindowSmoother::updateLandmarkInGraph(uint64_t lmk_id) {
  // Update existing smart-factor
//  auto old_smart_factors_it = old_smart_factors_.find(lmk_id);
//  CHECK(old_smart_factors_it != old_smart_factors_.end())
//      << "Landmark not found in old_smart_factors_ with id: " << lmk_id;

//  const SmartStereoFactor::shared_ptr& old_factor =
//      old_smart_factors_it->second.first;
//  // Clone old factor to keep all previous measurements, now append one.
//  SmartStereoFactor::shared_ptr new_factor =
//      boost::make_shared<SmartStereoFactor>(*old_factor);
//  gtsam::Symbol pose_symbol('x', new_measurement.first);
//  if (smoother_->getFactors().exists(pose_symbol)) {
//    const StereoPoint2& measurement = new_measurement.second;
//    new_factor->add(measurement, pose_symbol, stereo_cal_);
//  } else {
//    VLOG(10) << "Factor with lmk id " << lmk_id
//             << " is linking to a marginalized state!";
//  }

//  // Update the factor
//  Slot slot = old_smart_factors_it->second.second;
//  if (slot != -1) {
//    new_smart_factors_.insert(std::make_pair(lmk_id, new_factor));
//  } else {
//    // If it's slot in the graph is still -1, it means that the factor has not
//    // been inserted yet in the graph...
//    LOG(FATAL) << "When updating the smart factor, its slot should not be -1!"
//                  " Offensive lmk_id: "
//               << lmk_id;
//  }
//  old_smart_factors_it->second.first = new_factor;
//  VLOG(10) << "updateLandmarkInGraph: added observation to point: " << lmk_id;
}


// optimize the factor graph with new values, factors
void SlidingWindowSmoother::optimize(size_t /*numIter*/, size_t /*numThreads*/,
                                     bool /*verbose*/) {
  uint64_t currFrameId = currentFrameId();
  if (loopFrameAndMatchesList_.size() > 0) {
    LOG(INFO) << "Smoother receives #loop frames "
              << loopFrameAndMatchesList_.size()
              << " but has not implemented relocalization yet!";
    loopFrameAndMatchesList_.clear();
  }

  addLandmarkFactorsTimer.start();
  // Mark landmarks that are added to the graph solver.
  // A landmark has only two status:
  // 1. NotInState_NotTrackedNow means that it has not been added to the graph
  // solver.
  // 2. InState_TrackedNow means that it has been added to the graph solver.
  // To avoid confusion, do not interpret the second half of the status symbol.

  int numTracked = 0;
  for (okvis::PointMap::iterator it = landmarksMap_.begin();
       it != landmarksMap_.end(); ++it) {
    bool observedInCurrentFrame = false;
    for (auto itObs = it->second.observations.rbegin(),
              iteObs = it->second.observations.rend();
         itObs != iteObs; ++itObs) {
      if (itObs->first.frameId == currFrameId) {
        observedInCurrentFrame = true;
        ++numTracked;
        break;
      }
    }
    ResidualizeCase landmarkStatus = it->second.residualizeCase;
    if (landmarkStatus ==
        NotInState_NotTrackedNow) {  // the landmark has not been added to the
                                     // graph solver.
      if (observedInCurrentFrame) {
        double lmkQuality = landmarkQuality(mapPtr_, it->first);
        bool wellConstrained =
            lmkQuality > 1e-3;  // TODO(jhuai): learn this magic number.
        if (wellConstrained) {
          addLandmarkToGraph(it->first);
          it->second.residualizeCase = InState_TrackedNow;
        }  // else do nothing
      }    // else do nothing
    } else {
      if (observedInCurrentFrame) {
        updateLandmarkInGraph(it->first);
      }  // else do nothing
    }
  }
  addLandmarkFactorsTimer.stop();

  trackingRate_ = static_cast<double>(numTracked) /
                  static_cast<double>(landmarksMap_.size());

  isam2UpdateTimer.start();

  isam2UpdateTimer.stop();

  computeCovarianceTimer.start();

  computeCovarianceTimer.stop();

  // update poses, velocities, and biases, landmark positions, and camera
  // extrinsic parameters with isam2 estimates.
  gtsam::Values isamCurrentEstimate = smoother_->calculateEstimate();
  // some landmarks may have been marginalized and will not have estimates,
  // right?
  for (auto value : isamCurrentEstimate) {
  }

  // update landmarks that are tracked in the current frame(the newly inserted
  // state)
  {
    updateLandmarksTimer.start();
    for (auto it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
      if (it->second.residualizeCase == NotInState_NotTrackedNow) continue;
      // #Obs may be 1 for a new landmark by the KLT tracking frontend.
      // It ought to be >= 2 for descriptor matching frontend.
      if (it->second.observations.size() < 2) continue;
      double quality = landmarkQuality(mapPtr_, it->first);

      it->second.quality = quality;

      // update coordinates

      it->second.pointHomog = std::static_pointer_cast<
                                  okvis::ceres::HomogeneousPointParameterBlock>(
                                  mapPtr_->parameterBlockPtr(it->first))
                                  ->estimate();
    }
    updateLandmarksTimer.stop();
  }
}

bool SlidingWindowSmoother::computeCovariance(Eigen::MatrixXd* cov) const {
  *cov = Eigen::Matrix<double, 15, 15>::Identity();
  uint64_t T_WS_id = statesMap_.rbegin()->second.id;

  cov->topLeftCorner<6, 6>() =
      smoother_->marginalCovariance(gtsam::Symbol('x', T_WS_id));
  cov->block<3, 3>(6, 6) =
      smoother_->marginalCovariance(gtsam::Symbol('v', T_WS_id));
  cov->block<6, 6>(9, 9) =
      smoother_->marginalCovariance(gtsam::Symbol('b', T_WS_id));
  return true;
}
}  // namespace okvis

