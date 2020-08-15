#include <gtsam/SlidingWindowSmoother.hpp>

#include <glog/logging.h>

#include <okvis/ceres/ImuError.hpp>
#include <okvis/IdProvider.hpp>

#include <loop_closure/GtsamWrap.hpp>

#include <gtsam/base/ThreadsafeException.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/StereoCamera.h>
#include <gtsam/geometry/Cal3DS2.h>

#include <gtsam/inference/Factor.h>

#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/slam/ProjectionFactor.h>

#include <gtsam/ImuFrontEnd.h>

#include "loop_closure/GtsamWrap.hpp"

// TODO(jhuai): Theoretically, running iSAM2 for inertial odometry with priors on pose, velocity and biases should work.
// But after 70 secs, gtsam throws indeterminant linear system exception with about 200 m drift.

DEFINE_bool(process_cheirality,
            false,
            "Handle cheirality exception by removing problematic landmarks and "
            "re-running optimization.");
DEFINE_int32(max_number_of_cheirality_exceptions,
             5,
             "Sets the maximum number of times we process a cheirality "
             "exception for a given optimization problem. This is to avoid too "
             "many recursive calls to update the smoother");

/// \brief okvis Main namespace of this package.
namespace okvis {
void setIsam2Params(const okvis::BackendParams& vio_params,
                    gtsam::ISAM2Params* isam_param) {
  CHECK_NOTNULL(isam_param);

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
  mTrackLengthAccumulator = std::vector<size_t>(100, 0u);
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
  uint64_t frameId = statesMap_.rbegin()->first;
  okvis::kinematics::Transformation T_WB;
  get_T_WS(frameId, T_WB);
  Eigen::Matrix<double, 9, 1> vel_bias;
  getSpeedAndBias(frameId, 0u, vel_bias);
  Eigen::Matrix3d B_Rot_W = T_WB.C().transpose();

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
  gtsam::SharedNoiseModel noise_init_pose =
      gtsam::noiseModel::Gaussian::Covariance(pose_prior_covariance);
  new_imu_prior_and_other_factors_.push_back(
      boost::make_shared<gtsam::PriorFactor<gtsam::Pose3>>(
          gtsam::Symbol('x', frameId), VIO::GtsamWrap::toPose3(T_WB), noise_init_pose));

  // Add initial velocity priors.
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
  gtsam::SharedNoiseModel imu_bias_prior_noise =
      gtsam::noiseModel::Diagonal::Sigmas(prior_biasSigmas);
  Eigen::Vector3d bg = vel_bias.segment<3>(3);
  Eigen::Vector3d ba = vel_bias.tail<3>();
  gtsam::imuBias::ConstantBias imuBias(ba, bg);
  new_imu_prior_and_other_factors_.push_back(
      boost::make_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
          gtsam::Symbol('b', frameId), imuBias, imu_bias_prior_noise));
}

void SlidingWindowSmoother::addImuValues() {
  uint64_t cur_id = statesMap_.rbegin()->first;
  okvis::kinematics::Transformation T_WB;
  get_T_WS(cur_id, T_WB);
  Eigen::Matrix<double, 9, 1> vel_bias;
  getSpeedAndBias(cur_id, 0u, vel_bias);

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
  auto lastElementIter = statesMap_.rbegin(); // aka to node
  auto penultimateElementIter = lastElementIter; // aka from node
  ++penultimateElementIter;
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

void SlidingWindowSmoother::addCameraSystem(const okvis::cameras::NCameraSystem& cameras) {
  Estimator::addCameraSystem(cameras);
  Eigen::VectorXd intrinsics;
  camera_rig_.getCameraGeometry(0u)->getIntrinsics(intrinsics);
  OKVIS_ASSERT_EQ(Exception, intrinsics.size(), 8, "Sliding window smoother currently only work radial tangential distortion!");
  cal0_.reset(new gtsam::Cal3DS2(intrinsics[0], intrinsics[1], 0, intrinsics[2], intrinsics[3],
      intrinsics[4], intrinsics[5], intrinsics[6], intrinsics[7]));
  body_P_cam0_ = VIO::GtsamWrap::toPose3(camera_rig_.getCameraExtrinsic(0u));
}

bool SlidingWindowSmoother::addStates(
    okvis::MultiFramePtr multiFrame,
    const okvis::ImuMeasurementDeque& imuMeasurements, bool asKeyframe) {
  // note: this is before matching...
  // record the imu measurements between two consecutive states
  inertialMeasForStates_.push_back(imuMeasurements);
  okvis::kinematics::Transformation T_WS;
  Eigen::Matrix<double, 9, 1> speedAndBias;
  okvis:Time newStateTime = multiFrame->timestamp();
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
      pvstd_.updatePose(T_WS, newStateTime);
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
        statesMap_.rbegin()->second.timestamp, newStateTime);
    OKVIS_ASSERT_TRUE_DBG(Exception, numUsedImuMeasurements > 1,
                          "propagation failed");
    if (numUsedImuMeasurements < 1) {
      LOG(INFO) << "numUsedImuMeasurements=" << numUsedImuMeasurements;
      return false;
    }
  }

  // create a states object:
  States states(asKeyframe, multiFrame->id(), newStateTime);

  // check if id was used before
  OKVIS_ASSERT_TRUE_DBG(Exception,
                        statesMap_.find(states.id) == statesMap_.end(),
                        "pose ID" << states.id << " was used before!");

  // create global states
  std::shared_ptr<okvis::ceres::PoseParameterBlock> poseParameterBlock(
      new okvis::ceres::PoseParameterBlock(T_WS, states.id,
                                           newStateTime));
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
              T_SC, id, newStateTime));
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
            speedAndBias, id, newStateTime));

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
  std::map<uint64_t, States>::const_reverse_iterator rit = statesMap_.rbegin();
  while (rit != statesMap_.rend()) {
    if (state_.find(rit->first) == state_.end()) {
      break;
    }
    ++rit;
  }

  if (rit == statesMap_.rbegin()) {
    return rit->first;
  } else {
    --rit;
    return rit->first;
  }
}

// The major job of marginalization is done in the smoother optimization step.
// Here we only remove old landmarks and states.
bool SlidingWindowSmoother::applyMarginalizationStrategy(
    size_t /*numKeyframes*/, size_t /*numImuFrames*/,
    okvis::MapPointVector& removedLandmarks) {
  uint64_t minValidStateId = getMinValidStateId();
  std::vector<uint64_t> removeFrames;
  std::map<uint64_t, States>::iterator it = statesMap_.begin();
  while (it != statesMap_.end()) {
    if (it->first < minValidStateId) {
      removeFrames.push_back(it->second.id);
    } else {
      break;
    }
    ++it;
  }

  // remove feature tracks that do not overlap the sliding window.
  // Kimera-VIO simply removes old smart factors out of the time horizon, see
  // https://github.com/MIT-SPARK/Kimera-VIO/blob/master/src/backend/VioBackEnd.cpp#L926-L929.
  // Does Kimera-VIO use anchored inverse depth coordinates or world Euclidean coordinates?
  // World Euclidean coordinates because smart factors depends on camera poses which
  // are expressed in the world frame.

  for (PointMap::iterator pit = landmarksMap_.begin();
       pit != landmarksMap_.end();) {
    const MapPoint& mapPoint = pit->second;
    // Remove a landmark whose last observation is out of the horizon.
    uint64_t lastFrameId = mapPoint.observations.rbegin()->first.frameId;
    if (lastFrameId < minValidStateId) {
      ++mTrackLengthAccumulator[mapPoint.observations.size()];
// It is not necessary to remove residual blocks or uncheck keypoints in
// multiframes because these residual blocks and multiframes have been or
// will soon be removed as the associated nav state slides out of the optimization window.

//      for (std::map<okvis::KeypointIdentifier, uint64_t>::const_iterator it =
//               mapPoint.observations.begin();
//           it != mapPoint.observations.end(); ++it) {
//        if (it->second) {
//          mapPtr_->removeResidualBlock(
//              reinterpret_cast<::ceres::ResidualBlockId>(it->second));
//        }
//        const KeypointIdentifier& kpi = it->first;
//        auto mfp = multiFramePtrMap_.find(kpi.frameId);
//        OKVIS_ASSERT_TRUE(Exception, mfp != multiFramePtrMap_.end(), "frame id not found in frame map!");
//        mfp->second->setLandmarkId(kpi.cameraIndex, kpi.keypointIndex, 0);
//      }
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

  // remove unneeded extrinsic parameter blocks.
  for (size_t i = 0; i < extrinsicsEstimationParametersVec_.size(); ++i) {
    std::map<uint64_t, States>::iterator iter = statesMap_.begin();
    std::map<uint64_t, States>::iterator nextIter = ++iter;
    std::map<uint64_t, States>::iterator lastIter = statesMap_.end();
    while (nextIter != lastIter) {
      if (iter->first < minValidStateId) {
        if (iter->second.sensors.at(SensorStates::Camera)
                .at(i)
                .at(CameraSensorStates::T_SCi)
                .id != nextIter->second.sensors.at(SensorStates::Camera)
                           .at(i)
                           .at(CameraSensorStates::T_SCi)
                           .id) {
          mapPtr_->removeParameterBlock(
              it->second.sensors.at(SensorStates::Camera)
                  .at(i)
                  .at(CameraSensorStates::T_SCi)
                  .id);
          // The associated residual will be removed by ceres solver.
        }  // else do nothing
      } else {
        break;
      }
      iter = nextIter;
      ++nextIter;
    }
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

// Refer to InvUVFactor and process_feat_normal in CPI, closed-form preintegration repo of Eckenhoff.
void SlidingWindowSmoother::addLandmarkToGraph(uint64_t lmkId, const Eigen::Vector3d& externalPointW) {
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

  new_values_.insert(gtsam::symbol('l', lmkId), gtsam::Point3(externalPointW));

  uint64_t minValidStateId = statesMap_.begin()->first;
  const okvis::MapPoint& mp = landmarksMap_.at(lmkId);
  for (auto obsIter = mp.observations.begin(); obsIter != mp.observations.end();
       ++obsIter) {
    if (obsIter->first.frameId < minValidStateId) {
      // caution: some observations may be outside the horizon.
      continue;
    }

    // get the keypoint measurement
    okvis::MultiFramePtr multiFramePtr =
        multiFramePtrMap_.at(obsIter->first.frameId);
    Eigen::Vector2d measurement;
    multiFramePtr->getKeypoint(obsIter->first.cameraIndex,
                               obsIter->first.keypointIndex, measurement);

    double size = 1.0;
    multiFramePtr->getKeypointSize(obsIter->first.cameraIndex,
                                   obsIter->first.keypointIndex, size);

    gtsam::noiseModel::Isotropic::shared_ptr noise =
        gtsam::noiseModel::Isotropic::Sigma(2, size / 8.0);  // in u and v

    // TODO(jhuai): support more than one cameras by extending the gtsam
    // reprojection factor.
    gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3,
                                   gtsam::Cal3DS2>::shared_ptr factor =
        boost::make_shared<gtsam::GenericProjectionFactor<
            gtsam::Pose3, gtsam::Point3, gtsam::Cal3DS2>>(
            measurement, noise, gtsam::Symbol('x', obsIter->first.frameId),
            gtsam::Symbol('l', lmkId), cal0_, body_P_cam0_);

    new_reprojection_factors_.add(factor);
  }
}

void SlidingWindowSmoother::updateLandmarkInGraph(uint64_t lmkId) {
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


  const okvis::MapPoint& mp = landmarksMap_.at(lmkId);
  auto obsIter = mp.observations.rbegin();
  OKVIS_ASSERT_EQ(
      Exception, obsIter->first.frameId, statesMap_.rbegin()->first,
      "Only update landmark with observation in the current frame.");

  // get the keypoint measurement.
  okvis::MultiFramePtr multiFramePtr =
      multiFramePtrMap_.at(obsIter->first.frameId);
  Eigen::Vector2d measurement;
  multiFramePtr->getKeypoint(obsIter->first.cameraIndex,
                             obsIter->first.keypointIndex, measurement);
  double size = 1.0;
  multiFramePtr->getKeypointSize(obsIter->first.cameraIndex,
                                 obsIter->first.keypointIndex, size);

  gtsam::noiseModel::Isotropic::shared_ptr noise =
      gtsam::noiseModel::Isotropic::Sigma(2, size / 8.0);  // in u and v

  // TODO(jhuai): support more than one cameras by extending the gtsam
  // reprojection factor.
  gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3,
                                 gtsam::Cal3DS2>::shared_ptr factor =
      boost::make_shared<gtsam::GenericProjectionFactor<
          gtsam::Pose3, gtsam::Point3, gtsam::Cal3DS2>>(
          measurement, noise, gtsam::Symbol('x', obsIter->first.frameId),
          gtsam::Symbol('l', lmkId), cal0_, body_P_cam0_);
  new_reprojection_factors_.add(factor);
}

void SlidingWindowSmoother::updateStates() {
  gtsam::Values estimates = smoother_->calculateEstimate();
  state_ = estimates;

  // update poses, velocities, and biases from isam2 estimates.
  for (auto iter = statesMap_.begin(); iter != statesMap_.end(); ++iter) {
    uint64_t stateId = iter->first;
    auto xval = estimates.find(gtsam::Symbol('x', iter->first));
    if (xval == estimates.end()) {
      if (iter->first != statesMap_.begin()->first) {
        std::string msg =
            "The oldest nav state variables may just have been marginalized "
            "from "
            "iSAM2 in the preceding update step, but others should not.";
        LOG(WARNING) << "State of id " << iter->first
                     << " not found in smoother estimates when the first nav "
                        "state id is "
                     << statesMap_.begin()->first;
      }
      continue;
    }

    gtsam::Pose3 W_T_B =
        estimates.at<gtsam::Pose3>(gtsam::Symbol('x', iter->first));

    std::shared_ptr<ceres::PoseParameterBlock> poseParamBlockPtr =
        std::static_pointer_cast<ceres::PoseParameterBlock>(
            mapPtr_->parameterBlockPtr(stateId));
    kinematics::Transformation T_WB = VIO::GtsamWrap::toTransform(W_T_B);
    poseParamBlockPtr->setEstimate(T_WB);

    auto vval = estimates.find(gtsam::Symbol('v', iter->first));
    gtsam::Vector3 W_v_B =
        estimates.at<gtsam::Vector3>(gtsam::Symbol('v', iter->first));
    auto bval = estimates.find(gtsam::Symbol('b', iter->first));
    gtsam::imuBias::ConstantBias imuBias =
        estimates.at<gtsam::imuBias::ConstantBias>(
            gtsam::Symbol('b', iter->first));

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
    sb.head<3>() = W_v_B;
    sb.segment<3>(3) = imuBias.gyroscope();
    sb.tail<3>(3) = imuBias.accelerometer();
    sbParamBlockPtr->setEstimate(sb);
  }

  // update camera extrinsic parameters from isam2 estimates.

  // update landmark positions from isam2 estimates.
  {
    updateLandmarksTimer.start();
    for (auto it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
      if (it->second.residualizeCase == NotInState_NotTrackedNow) continue;
      // #Obs may be 1 for a new landmark by the KLT tracking frontend.
      // It ought to be >= 2 for descriptor matching frontend.
      if (it->second.observations.size() < 2) continue;
      uint64_t lmkId = it->first;
      auto estimatesIter = estimates.find(gtsam::Symbol('l', lmkId));
      if (estimatesIter == estimates.end()) {
          continue;
      }
      gtsam::Point3 pW = estimates.at<gtsam::Point3>(gtsam::Symbol('l', lmkId));
      Eigen::Vector4d hpW;
      hpW.head<3>() = pW;
      hpW[3] = 1.0;
      std::shared_ptr<okvis::ceres::HomogeneousPointParameterBlock> landmarkParamBlockPtr =
          std::static_pointer_cast<okvis::ceres::HomogeneousPointParameterBlock>(
                                        mapPtr_->parameterBlockPtr(it->first));
      landmarkParamBlockPtr->setEstimate(hpW);

      double quality = landmarkQuality(mapPtr_, it->first);
      it->second.quality = quality;
      it->second.pointHomog = hpW;
    }
    updateLandmarksTimer.stop();
  }
}


bool SlidingWindowSmoother::updateSmoother(gtsam::FixedLagSmoother::Result* result,
                                const gtsam::NonlinearFactorGraph& new_factors,
                                const gtsam::Values& new_values,
                                const std::map<gtsam::Key, double>& timestamps,
                                const gtsam::FactorIndices& delete_slots) {
  // This is not doing a full deep copy: it is keeping same shared_ptrs for
  // factors but copying the isam result.
  Smoother smoother_backup(*smoother_);

  bool got_cheirality_exception = false;
  gtsam::Symbol lmk_symbol_cheirality;
  try {
    LOG(INFO) << "Starting update of smoother_...";
    *result = smoother_->update(new_factors, new_values, timestamps, delete_slots);
    LOG(INFO) << "Finished update of smoother_.";
  } catch (const gtsam::IndeterminantLinearSystemException& e) {
    const gtsam::Key& var = e.nearbyVariable();
    gtsam::Symbol symb(var);

    LOG(ERROR) << "IndeterminantLinearSystemException: Nearby variable has type '" << symb.chr() << "' "
               << "and index " << symb.index() << std::endl;
    LOG(ERROR) << e.what();
    smoother_->getFactors().print("Smoother's factors:\n[\n\t");
    LOG(INFO) << " ]";
    state_.print("State values\n[\n\t");
    LOG(INFO) << " ]";
    printSmootherInfo(new_factors, delete_slots);
    return false;
  } catch (const gtsam::InvalidNoiseModel& e) {
    LOG(ERROR) << e.what();
    printSmootherInfo(new_factors, delete_slots);
    return false;
  } catch (const gtsam::InvalidMatrixBlock& e) {
    LOG(ERROR) << e.what();
    printSmootherInfo(new_factors, delete_slots);
    return false;
  } catch (const gtsam::InvalidDenseElimination& e) {
    LOG(ERROR) << e.what();
    printSmootherInfo(new_factors, delete_slots);
    return false;
  } catch (const gtsam::InvalidArgumentThreadsafe& e) {
    LOG(ERROR) << e.what();
    printSmootherInfo(new_factors, delete_slots);
    return false;
  } catch (const gtsam::ValuesKeyDoesNotExist& e) {
    LOG(ERROR) << e.what();
    printSmootherInfo(new_factors, delete_slots);
    return false;
  } catch (const gtsam::CholeskyFailed& e) {
    LOG(ERROR) << e.what();
    printSmootherInfo(new_factors, delete_slots);
    return false;
  } catch (const gtsam::CheiralityException& e) {
    LOG(ERROR) << e.what();
    const gtsam::Key& lmk_key = e.nearbyVariable();
    lmk_symbol_cheirality = gtsam::Symbol(lmk_key);
    LOG(ERROR) << "CheiralityException: Nearby variable has type '" << lmk_symbol_cheirality.chr()
               << "' "
               << "and index " << lmk_symbol_cheirality.index();
    printSmootherInfo(new_factors, delete_slots);
    got_cheirality_exception = true;
  } catch (const gtsam::StereoCheiralityException& e) {
    LOG(ERROR) << e.what();
    const gtsam::Key& lmk_key = e.nearbyVariable();
    lmk_symbol_cheirality = gtsam::Symbol(lmk_key);
    LOG(ERROR) << "StereoCheiralityException: Nearby variable has type '" << lmk_symbol_cheirality.chr()
               << "' "
               << "and index " << lmk_symbol_cheirality.index();
    printSmootherInfo(new_factors, delete_slots);
    got_cheirality_exception = true;
  } catch (const gtsam::RuntimeErrorThreadsafe& e) {
    LOG(ERROR) << e.what();
    printSmootherInfo(new_factors, delete_slots);
    return false;
  } catch (const gtsam::OutOfRangeThreadsafe& e) {
    LOG(ERROR) << e.what();
    printSmootherInfo(new_factors, delete_slots);
    return false;
  } catch (const std::out_of_range& e) {
    LOG(ERROR) << e.what();
    printSmootherInfo(new_factors, delete_slots);
    return false;
  } catch (const std::exception& e) {
    // Catch anything thrown within try block that derives from
    // std::exception.
    LOG(ERROR) << e.what();
    printSmootherInfo(new_factors, delete_slots);
    return false;
  } catch (...) {
    // Catch the rest of exceptions.
    LOG(ERROR) << "Unrecognized exception.";
    printSmootherInfo(new_factors, delete_slots);
    return false;
  }

  if (FLAGS_process_cheirality) {
    if (got_cheirality_exception) {
      LOG(WARNING) << "Starting processing cheirality exception # "
                   << counter_of_exceptions_;
      counter_of_exceptions_++;

      // Restore smoother as it was before failure.
      *smoother_ = smoother_backup;

      // Limit the number of cheirality exceptions per run.
      CHECK_LE(counter_of_exceptions_,
               FLAGS_max_number_of_cheirality_exceptions);

      // Check that we have a landmark.
      CHECK_EQ(lmk_symbol_cheirality.chr(), 'l');

      // Now that we know the lmk id, delete all factors attached to it!
      gtsam::NonlinearFactorGraph new_factors_tmp_cheirality;
      gtsam::Values new_values_cheirality;
      std::map<gtsam::Key, double> timestamps_cheirality;
      gtsam::FactorIndices delete_slots_cheirality;
      const gtsam::NonlinearFactorGraph& graph = smoother_->getFactors();
      VLOG(10) << "Starting cleanCheiralityLmk...";
      cleanCheiralityLmk(lmk_symbol_cheirality,
                         &new_factors_tmp_cheirality,
                         &new_values_cheirality,
                         &timestamps_cheirality,
                         &delete_slots_cheirality,
                         graph,
                         new_factors,
                         new_values,
                         timestamps,
                         delete_slots);
      VLOG(10) << "Finished cleanCheiralityLmk.";

      // Try again to optimize. This is a recursive call.
      LOG(WARNING) << "Starting updateSmoother after handling "
                      "cheirality exception.";
      bool status = updateSmoother(result,
                                   new_factors_tmp_cheirality,
                                   new_values_cheirality,
                                   timestamps_cheirality,
                                   delete_slots_cheirality);
      LOG(WARNING) << "Finished updateSmoother after handling "
                      "cheirality exception";
      return status;
    } else {
      counter_of_exceptions_ = 0;
    }
  }

  return true;
}

gtsam::TriangulationResult SlidingWindowSmoother::triangulateSafe(uint64_t lmkId) const {
  const MapPoint& mp = landmarksMap_.at(lmkId);
  uint64_t minValidStateId = statesMap_.begin()->first;

  gtsam::Cal3_S2 cal(1, 1, 0, 0, 0);
  double landmarkDistanceThreshold = 50;  // Ignore points farther than this.
  gtsam::TriangulationParameters params(
      1.0, false, landmarkDistanceThreshold);
  gtsam::CameraSet<gtsam::SimpleCamera> cameras;
  gtsam::Point2Vector measurements;

  for (auto itObs = mp.observations.begin(), iteObs = mp.observations.end();
       itObs != iteObs; ++itObs) {
    uint64_t poseId = itObs->first.frameId;
    if (poseId < minValidStateId) continue;

    Eigen::Vector2d measurement;
    auto multiFrameIter = multiFramePtrMap_.find(poseId);
    //    OKVIS_ASSERT_TRUE(Exception, multiFrameIter !=
    //    multiFramePtrMap_.end(), "multiframe not found");
    okvis::MultiFramePtr multiFramePtr = multiFrameIter->second;
    multiFramePtr->getKeypoint(itObs->first.cameraIndex,
                               itObs->first.keypointIndex, measurement);

    // use the latest estimates for camera intrinsic parameters
    Eigen::Vector3d backProjectionDirection;
    std::shared_ptr<const cameras::CameraBase> cameraGeometry =
        camera_rig_.getCameraGeometry(itObs->first.cameraIndex);
    bool validDirection =
        cameraGeometry->backProject(measurement, &backProjectionDirection);
    if (!validDirection) {
      continue;
    }
    okvis::kinematics::Transformation T_WB;
    get_T_WS(poseId, T_WB);
    gtsam::Pose3 W_T_B = VIO::GtsamWrap::toPose3(T_WB);
    gtsam::SimpleCamera camera(W_T_B.compose(body_P_cam0_), cal);

    cameras.push_back(camera);
    measurements.push_back(gtsam::Point2(backProjectionDirection.head<2>()));
  }

  return gtsam::triangulateSafe(cameras, measurements, params);
}

void SlidingWindowSmoother::optimize(size_t /*numIter*/, size_t /*numThreads*/,
                                     bool /*verbose*/) {
  uint64_t currFrameId = currentFrameId();
  LOG(INFO) << "Optimizing for current frame id " << currFrameId;
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
  // To avoid confusion, do not interpret the second half of the status value.

  int numTracked = 0;
  for (okvis::PointMap::iterator it = landmarksMap_.begin();
       it != landmarksMap_.end(); ++it) {
    bool observedInCurrentFrame = false;
    if (it->second.observations.size() < minTrackLength_)
      continue;
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
    if (landmarkStatus == NotInState_NotTrackedNow) {
      // The landmark has not been added to the graph.
      if (observedInCurrentFrame) {
        // TODO(jhuai): we need to tune the triangulation parameters.
        // Currently very few landmarks are triangulated successfully.
        gtsam::TriangulationResult result = triangulateSafe(it->first);
        if (result.valid()) {
          Eigen::Vector3d pW(*result);
//        double quality = landmarkQuality(mapPtr_, it->first);
//        if (quality > 1e-3) {
//          Eigen::Vector3d pW = it->second.pointHomog.head<3>() / it->second.pointHomog[3];
          addLandmarkToGraph(it->first, pW);
          it->second.residualizeCase = InState_TrackedNow;
        }  // else do nothing
      }  // else do nothing
    } else { // The landmark has been added to the graph.
      if (observedInCurrentFrame) {
        updateLandmarkInGraph(it->first);
      }  // else do nothing
    }
  }
  addLandmarkFactorsTimer.stop();

  trackingRate_ = static_cast<double>(numTracked) /
                  static_cast<double>(landmarksMap_.size());

  size_t new_reproj_factors_size = new_reprojection_factors_.size();
  gtsam::NonlinearFactorGraph new_factors_tmp;
  new_factors_tmp.reserve(new_reproj_factors_size +
                          new_imu_prior_and_other_factors_.size());
  new_factors_tmp.push_back(new_reprojection_factors_.begin(),
                            new_reprojection_factors_.end());
  new_factors_tmp.push_back(new_imu_prior_and_other_factors_.begin(),
                            new_imu_prior_and_other_factors_.end());

  // Use current timestamp for each new value. This timestamp will be used
  // to determine if the variable should be marginalized.
  // Needs to use DOUBLE in secs because gtsam works with that.
  std::map<gtsam::Key, double> timestamps;
  okvis::Time currentTime = statesMap_.rbegin()->second.timestamp;
  double currentTimeSecs = currentTime.toSec();
  for(const gtsam::Values::ConstKeyValuePair& key_value : new_values_) {
    timestamps[key_value.key] = currentTimeSecs;
  }

  gtsam::FactorIndices delete_slots;

  isam2UpdateTimer.start();
  LOG(INFO) << "iSAM2 update with " << new_factors_tmp.size() << " new factors "
           << ", " << new_values_.size() << " new values "
           << ", and " << delete_slots.size() << " deleted factors.";
  Smoother::Result result;
  bool is_smoother_ok = updateSmoother(&result, new_factors_tmp, new_values_,
                                       timestamps, delete_slots);

  if (is_smoother_ok) {
    // Reset everything for next round.
    new_reprojection_factors_.resize(0);

    // Reset list of new imu, prior and other factors to be added.
    new_imu_prior_and_other_factors_.resize(0);

    new_values_.clear();

    // Do some more optimization iterations.
    for (int n_iter = 1; n_iter < backendParams_.numOptimize_ && is_smoother_ok;
         ++n_iter) {
      LOG(INFO) << "Doing extra iteration nr: " << n_iter;
      is_smoother_ok = updateSmoother(&result);
    }

    // Update states we need for next iteration, if smoother is ok.
    if (is_smoother_ok) {
      updateStates();
    }
  }
  isam2UpdateTimer.stop();

  computeCovarianceTimer.start();
  computeCovariance(&covariance_);
  computeCovarianceTimer.stop();
}

bool SlidingWindowSmoother::computeCovariance(Eigen::MatrixXd* cov) const {
  uint64_t T_WS_id = statesMap_.rbegin()->second.id;
  *cov = Eigen::Matrix<double, 15, 15>::Identity();
  Eigen::Matrix<double, 6, 6> swapBaBg = Eigen::Matrix<double, 6, 6>::Zero();
  swapBaBg.topRightCorner<3, 3>().setIdentity();
  swapBaBg.bottomLeftCorner<3, 3>().setIdentity();

  okvis::kinematics::Transformation T_WB;
  get_T_WS(statesMap_.rbegin()->first, T_WB);
  // okvis pW = \hat pW + \delta pW, R_WB = \hat R_WB exp(\delta \theta_W);
  // gtsam pW = \hat pW + R_WB \delta pB, R_WB = exp(\delta \theta B) \hat R_WB.
  Eigen::Matrix<double, 6, 6> swapRT = Eigen::Matrix<double, 6, 6>::Zero();
  swapRT.topRightCorner<3, 3>() = T_WB.C();
  swapRT.bottomLeftCorner<3, 3>() = T_WB.C();

  cov->topLeftCorner<6, 6>() =
      swapRT * smoother_->marginalCovariance(gtsam::Symbol('x', T_WS_id)) * swapRT.transpose();
  cov->block<3, 3>(6, 6) =
      smoother_->marginalCovariance(gtsam::Symbol('v', T_WS_id));
  cov->block<6, 6>(9, 9) =
      swapBaBg * smoother_->marginalCovariance(gtsam::Symbol('b', T_WS_id)) * swapBaBg.transpose();
  return true;
}

void SlidingWindowSmoother::printSmootherInfo(
    const gtsam::NonlinearFactorGraph& new_factors_tmp,
    const gtsam::FactorIndices& delete_slots,
    const std::string& message,
    const bool& /*showDetails*/) const {
  LOG(INFO) << " =============== START:" << message << " =============== ";

//  const std::string* which_graph = nullptr;
//  const gtsam::NonlinearFactorGraph* graph = nullptr;
//  // Pick the graph that makes more sense:
//  // This is code is mostly run post update, when it throws exception,
//  // shouldn't we print the graph before optimization instead?
//  // Yes if available, but if not, then just ask the smoother.
//  static const std::string graph_before_opt = "(graph before optimization)";
//  static const std::string smoother_get_factors = "(smoother getFactors)";
//  if (debug_info_.graphBeforeOpt.size() != 0) {
//    which_graph = &graph_before_opt;
//    graph = &(debug_info_.graphBeforeOpt);
//  } else {
//    which_graph = &smoother_get_factors;
//    graph = &(smoother_->getFactors());
//  }
//  CHECK_NOTNULL(which_graph);
//  CHECK_NOTNULL(graph);

//  static constexpr bool print_smart_factors = true;  // There a lot of these!
//  static constexpr bool print_point_plane_factors = true;
//  static constexpr bool print_plane_priors = true;
//  static constexpr bool print_point_priors = true;
//  static constexpr bool print_linear_container_factors = true;
//  ////////////////////// Print all factors.
//  ///////////////////////////////////////
//  LOG(INFO) << "Nr of factors in graph " + *which_graph << ": " << graph->size()
//            << ", with factors:" << std::endl;
//  LOG(INFO) << "[\n";
//  printSelectedGraph(*graph,
//                     print_smart_factors,
//                     print_point_plane_factors,
//                     print_plane_priors,
//                     print_point_priors,
//                     print_linear_container_factors);
//  LOG(INFO) << " ]" << std::endl;

//  ///////////// Print factors that were newly added to the optimization.//////
  LOG(INFO) << "Nr of new factors to add: " << new_factors_tmp.size()
            << " with factors:" << std::endl;
//  LOG(INFO) << "[\n (slot # wrt to new_factors_tmp graph) \t";
//  printSelectedGraph(new_factors_tmp,
//                     print_smart_factors,
//                     print_point_plane_factors,
//                     print_plane_priors,
//                     print_point_priors,
//                     print_linear_container_factors);
//  LOG(INFO) << " ]" << std::endl;

//  ////////////////////////////// Print deleted /// slots.///////////////////////
  LOG(INFO) << "Nr deleted slots: " << delete_slots.size()
            << ", with slots:" << std::endl;
//  LOG(INFO) << "[\n\t";
//  if (debug_info_.graphToBeDeleted.size() != 0) {
//    // If we are storing the graph to be deleted, then print extended info
//    // besides the slot to be deleted.
//    CHECK_GE(debug_info_.graphToBeDeleted.size(), delete_slots.size());
//    for (size_t i = 0u; i < delete_slots.size(); ++i) {
//      CHECK(debug_info_.graphToBeDeleted.at(i));
//      if (print_point_plane_factors) {
//        printSelectedFactors(debug_info_.graphToBeDeleted.at(i),
//                             delete_slots.at(i),
//                             false,
//                             print_point_plane_factors,
//                             false,
//                             false,
//                             false);
//      } else {
//        std::cout << "\tSlot # " << delete_slots.at(i) << ":";
//        std::cout << "\t";
//        debug_info_.graphToBeDeleted.at(i)->printKeys();
//      }
//    }
//  } else {
//    for (size_t i = 0; i < delete_slots.size(); ++i) {
//      std::cout << delete_slots.at(i) << " ";
//    }
//  }
//  std::cout << std::endl;
//  LOG(INFO) << " ]" << std::endl;

//  //////////////////////// Print all values in state. ////////////////////////
//  LOG(INFO) << "Nr of values in state_ : " << state_.size() << ", with keys:";
//  std::cout << "[\n\t";
//  for (const gtsam::Values::ConstKeyValuePair& key_value : state_) {
//    std::cout << gtsam::DefaultKeyFormatter(key_value.key) << " ";
//  }
//  std::cout << std::endl;
//  LOG(INFO) << " ]";

//  // Print only new values.
  LOG(INFO) << "Nr values in new_values_ : " << new_values_.size()
            << ", with keys:";
  std::cout << "[\n\t";
  for (const gtsam::Values::ConstKeyValuePair& key_value : new_values_) {
    std::cout << " " << gtsam::DefaultKeyFormatter(key_value.key) << " ";
  }
  std::cout << std::endl;
  LOG(INFO) << " ]";

//  if (showDetails) {
//    graph->print("isam2 graph:\n");
//    new_factors_tmp.print("new_factors_tmp:\n");
//    new_values_.print("new values:\n");
//    // LOG(INFO) << "new_smart_factors_: "  << std::endl;
//    // for (auto& s : new_smart_factors_)
//    //	s.second->print();
//  }

  LOG(INFO) << " =============== END: " << message << " =============== ";
}

void SlidingWindowSmoother::cleanCheiralityLmk(
    const gtsam::Symbol& lmk_symbol,
    gtsam::NonlinearFactorGraph* new_factors_tmp_cheirality,
    gtsam::Values* new_values_cheirality,
    std::map<gtsam::Key, double>* timestamps_cheirality,
    gtsam::FactorIndices* delete_slots_cheirality,
    const gtsam::NonlinearFactorGraph& graph,
    const gtsam::NonlinearFactorGraph& new_factors_tmp,
    const gtsam::Values& new_values,
    const std::map<gtsam::Key, double>& timestamps,
    const gtsam::FactorIndices& delete_slots) {
  CHECK_NOTNULL(new_factors_tmp_cheirality);
  CHECK_NOTNULL(new_values_cheirality);
  CHECK_NOTNULL(timestamps_cheirality);
  CHECK_NOTNULL(delete_slots_cheirality);
  const gtsam::Key& lmk_key = lmk_symbol.key();

  // Delete from new factors.
  VLOG(10) << "Starting delete from new factors...";
  deleteAllFactorsWithKeyFromFactorGraph(
      lmk_key, new_factors_tmp, new_factors_tmp_cheirality);
  VLOG(10) << "Finished delete from new factors.";

  // Delete from new values.
  VLOG(10) << "Starting delete from new values...";
  bool is_deleted_from_values =
      deleteKeyFromValues(lmk_key, new_values, new_values_cheirality);
  VLOG(10) << "Finished delete from timestamps.";

  // Delete from new values.
  VLOG(10) << "Starting delete from timestamps...";
  bool is_deleted_from_timestamps =
      deleteKeyFromTimestamps(lmk_key, timestamps, timestamps_cheirality);
  VLOG(10) << "Finished delete from timestamps.";

  // Check that if we deleted from values, we should have deleted as well
  // from timestamps.
  CHECK_EQ(is_deleted_from_values, is_deleted_from_timestamps);

  // Delete slots in current graph.
  VLOG(10) << "Starting delete from current graph...";
  *delete_slots_cheirality = delete_slots;
  std::vector<size_t> slots_of_extra_factors_to_delete;
  // Achtung: This has the chance to make the plane underconstrained, if
  // we delete too many point_plane factors.
  findSlotsOfFactorsWithKey(lmk_key, graph, &slots_of_extra_factors_to_delete);
  delete_slots_cheirality->insert(delete_slots_cheirality->end(),
                                  slots_of_extra_factors_to_delete.begin(),
                                  slots_of_extra_factors_to_delete.end());
  VLOG(10) << "Finished delete from current graph.";

  //////////////////////////// BOOKKEEPING
  ////////////////////////////////////////
  const uint64_t& lmk_id = lmk_symbol.index();

  // Delete from feature tracks.
  VLOG(10) << "Starting delete from feature tracks...";
  CHECK(deleteLmkFromFeatureTracks(lmk_id));
  VLOG(10) << "Finished delete from feature tracks.";
}


// Returns if the key in feature tracks could be removed or not.
bool SlidingWindowSmoother::deleteLmkFromFeatureTracks(const uint64_t& /*lmk_id*/) {
//  if (feature_tracks_.find(lmk_id) != feature_tracks_.end()) {
//    VLOG(2) << "Deleting feature track for lmk with id: " << lmk_id;
//    feature_tracks_.erase(lmk_id);
//    return true;
//  }
  return false;
}

void SlidingWindowSmoother::deleteAllFactorsWithKeyFromFactorGraph(
    const gtsam::Key& key,
    const gtsam::NonlinearFactorGraph& factor_graph,
    gtsam::NonlinearFactorGraph* factor_graph_output) {
  CHECK_NOTNULL(factor_graph_output);
  size_t new_factors_slot = 0;
  *factor_graph_output = factor_graph;
  for (auto it = factor_graph_output->begin();
       it != factor_graph_output->end();) {
    if (*it) {
      if ((*it)->find(key) != (*it)->end()) {
        // We found our lmk in the list of keys of the factor.
        // Sanity check, this lmk has no priors right?
        CHECK(!boost::dynamic_pointer_cast<gtsam::PriorFactor<gtsam::Point3>>(
            *it));
        // We are not deleting a smart factor right?
        // Otherwise we need to update structure:
        // lmk_ids_of_new_smart_factors...
        CHECK(!boost::dynamic_pointer_cast<SmartStereoFactor>(*it));
        // Whatever factor this is, it has our lmk...
        // Delete it.
        LOG(WARNING) << "Delete factor in new_factors at slot # "
                     << new_factors_slot << " of new_factors graph.";
        it = factor_graph_output->erase(it);
      } else {
        it++;
      }
    } else {
      LOG(ERROR) << "*it, which is itself a pointer, is null.";
      it++;
    }
    new_factors_slot++;
  }
}

// Returns if the key in timestamps could be removed or not.
bool SlidingWindowSmoother::deleteKeyFromTimestamps(
    const gtsam::Key& key,
    const std::map<gtsam::Key, double>& timestamps,
    std::map<gtsam::Key, double>* timestamps_output) {
  CHECK_NOTNULL(timestamps_output);
  *timestamps_output = timestamps;
  if (timestamps_output->find(key) != timestamps_output->end()) {
    timestamps_output->erase(key);
    return true;
  }
  return false;
}

// Returns if the key in timestamps could be removed or not.
bool SlidingWindowSmoother::deleteKeyFromValues(const gtsam::Key& key,
                                     const gtsam::Values& values,
                                     gtsam::Values* values_output) {
  CHECK_NOTNULL(values_output);
  *values_output = values;
  if (values.find(key) != values.end()) {
    // We found the lmk in new values, delete it.
    LOG(WARNING) << "Delete value in new_values for key "
                 << gtsam::DefaultKeyFormatter(key);
    CHECK(values_output->find(key) != values_output->end());
    try {
      values_output->erase(key);
    } catch (const gtsam::ValuesKeyDoesNotExist& e) {
      LOG(FATAL) << e.what();
    } catch (...) {
      LOG(FATAL) << "Unhandled exception when erasing key"
                    " in new_values_cheirality";
    }
    return true;
  }
  return false;
}

// Returns if the key in timestamps could be removed or not.
void SlidingWindowSmoother::findSlotsOfFactorsWithKey(
    const gtsam::Key& key,
    const gtsam::NonlinearFactorGraph& graph,
    std::vector<size_t>* slots_of_factors_with_key) {
  CHECK_NOTNULL(slots_of_factors_with_key);
  slots_of_factors_with_key->resize(0);
  size_t slot = 0;
  for (const boost::shared_ptr<gtsam::NonlinearFactor>& g : graph) {
    if (g) {
      // Found a valid factor.
      if (g->find(key) != g->end()) {
        // Whatever factor this is, it has our lmk...
        // Sanity check, this lmk has no priors right?
        CHECK(!boost::dynamic_pointer_cast<gtsam::LinearContainerFactor>(g));
        CHECK(
            !boost::dynamic_pointer_cast<gtsam::PriorFactor<gtsam::Point3>>(g));
        // Sanity check that we are not deleting a smart factor.
        CHECK(!boost::dynamic_pointer_cast<SmartStereoFactor>(g));
        // Delete it.
        LOG(WARNING) << "Delete factor in graph at slot # " << slot
                     << " corresponding to lmk with id: "
                     << gtsam::Symbol(key).index();
        CHECK(graph.exists(slot));
        slots_of_factors_with_key->push_back(slot);
      }
    }
    slot++;
  }
}

bool SlidingWindowSmoother::print(std::ostream& stream) const {
  Estimator::print(stream);
  Eigen::IOFormat spaceInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols,
                               " ", " ", "", "", "", "");

  Eigen::Matrix<double, Eigen::Dynamic, 1> variances = covariance_.diagonal();
  stream << " " << variances.cwiseSqrt().transpose().format(spaceInitFmt);
  return true;
}

}  // namespace okvis

