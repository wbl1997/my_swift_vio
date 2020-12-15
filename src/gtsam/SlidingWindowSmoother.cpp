/**
 * @file   SlidingWindowSmoother.cpp
 * @brief  Implementation of SlidingWindowSmoother which wraps gtsam::FixedLagSmoother for VIO.
 * @author Jianzhu Huai
 */

#include <gtsam/SlidingWindowSmoother.hpp>

#include <glog/logging.h>

#include <gtsam/nonlinear/LinearContainerFactor.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/slam/ProjectionFactor.h>

#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/PinholeCamera.h>

#include <loop_closure/GtsamWrap.hpp>

#include <msckf/FeatureTriangulation.hpp>

#include <okvis/ceres/ImuError.hpp>
#include <okvis/IdProvider.hpp>

// In general, iSAM2 requires all variables are well constrained. For visual
// inertial SLAM/odometry, we need to add priors for a pose, a velocity, and the 6D bias.

// Warning: Running iSAM2 for inertial odometry with priors on pose, velocity
// and biases, throws indeterminant linear system exception after ~70 secs with
// about 200 m drift.

// Following Kimera-VIO, we use world Euclidean coordinates instead of anchored
// inverse depth coordinates. In Kimera-VIO, the landmarks are expressed in
// world Euclidean coordinates because smart factors depends on camera poses
// that are expressed in the world frame.

DEFINE_bool(process_cheirality,
            false,
            "Handle cheirality exception by removing problematic landmarks and "
            "re-running optimization.");

DEFINE_int32(max_number_of_cheirality_exceptions,
             5,
             "Sets the maximum number of times we process a cheirality "
             "exception for a given optimization problem. This is to avoid too "
             "many recursive calls to update the smoother");

DEFINE_double(time_horizon, 1.0, "Time horizon in secs");

DEFINE_double(ray_sigma_scalar, 6.0,
              "below how many sigmas do we consider rays have low disparity?");

namespace gtsam {
  using PinholeCameraCal3_S2 = gtsam::PinholeCamera<gtsam::Cal3_S2>;
}

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
  LOG(INFO) << "Backend modality " << (int)vio_params.backendModality_;
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

void setSmartFactorsParams(
    gtsam::SharedNoiseModel* smart_noise,
    gtsam::SmartProjectionParams* smart_factors_params,
    double smart_noise_sigma,
    double rank_tolerance,
    double landmark_distance_threshold,
    double retriangulation_threshold,
    double outlier_rejection) {
  // The first argument is irrelevant for Isotropic noise, i.e., can be 2 or 3.
  gtsam::SharedNoiseModel model =
      gtsam::noiseModel::Isotropic::Sigma(2, smart_noise_sigma);
  // smart_noise_ = gtsam::noiseModel::Robust::Create(
  //                  gtsam::noiseModel::mEstimator::Huber::Create(1.345),
  //                  model);
  *smart_noise = model;
  *smart_factors_params =
      gtsam::SmartProjectionParams(gtsam::HESSIAN,             // JACOBIAN_SVD
                        gtsam::ZERO_ON_DEGENERACY,  // IGNORE_DEGENERACY
                        false,                      // ThrowCherality = false
                        true);                      // verboseCherality = true
  smart_factors_params->setRankTolerance(rank_tolerance);
  smart_factors_params->setLandmarkDistanceThreshold(
      landmark_distance_threshold);
  smart_factors_params->setRetriangulationThreshold(retriangulation_threshold);
  smart_factors_params->setDynamicOutlierRejectionThreshold(outlier_rejection);
}

void setFactorsParams(
    const BackendParams& vio_params,
    gtsam::SharedNoiseModel* smart_noise,
    gtsam::SmartProjectionParams* smart_factors_params,
    gtsam::SharedNoiseModel* no_motion_prior_noise,
    gtsam::SharedNoiseModel* zero_velocity_prior_noise,
    gtsam::SharedNoiseModel* constant_velocity_prior_noise) {
  CHECK_NOTNULL(smart_noise);
  CHECK_NOTNULL(smart_factors_params);
  CHECK_NOTNULL(no_motion_prior_noise);
  CHECK_NOTNULL(zero_velocity_prior_noise);
  CHECK_NOTNULL(constant_velocity_prior_noise);

  //////////////////////// SMART PROJECTION FACTORS SETTINGS
  //////////////////////
  setSmartFactorsParams(smart_noise,
                        smart_factors_params,
                        vio_params.smartNoiseSigma_,
                        vio_params.rankTolerance_,
                        vio_params.landmarkDistanceThreshold_,
                        vio_params.retriangulationThreshold_,
                        vio_params.outlierRejection_);

  //////////////////////// NO MOTION FACTORS SETTINGS
  /////////////////////////////
  Eigen::Matrix<double, 6, 1> sigmas;
  sigmas.head<3>().setConstant(vio_params.noMotionRotationSigma_);
  sigmas.tail<3>().setConstant(vio_params.noMotionPositionSigma_);
  *no_motion_prior_noise = gtsam::noiseModel::Diagonal::Sigmas(sigmas);

  //////////////////////// ZERO VELOCITY FACTORS SETTINGS
  /////////////////////////
  *zero_velocity_prior_noise =
      gtsam::noiseModel::Isotropic::Sigma(3, vio_params.zeroVelocitySigma_);

  //////////////////////// CONSTANT VELOCITY FACTORS SETTINGS
  /////////////////////
  *constant_velocity_prior_noise =
      gtsam::noiseModel::Isotropic::Sigma(3, vio_params.constantVelSigma_);
}

void SlidingWindowSmoother::setupSmoother(
    const okvis::BackendParams& vioParams) {
#ifdef INCREMENTAL_SMOOTHER
  gtsam::ISAM2Params params;
  setIsam2Params(vioParams, &params);
#else  // BATCH SMOOTHER
  gtsam::LevenbergMarquardtParams params;
  params.setlambdaInitial(vioParams.initialLambda_);
  params.setlambdaLowerBound(vioParams.lowerBoundLambda_);
  params.setlambdaUpperBound(vioParams.upperBoundLambda_);
#endif
  smoother_.reset(new Smoother(FLAGS_time_horizon, params));

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
  // Set parameters for all factors.
  setFactorsParams(vioParams,
                   &smart_noise_,
                   &smart_factors_params_,
                   &no_motion_prior_noise_,
                   &zero_velocity_prior_noise_,
                   &constant_velocity_prior_noise_);
}

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
  Eigen::Matrix3d B_Rot_W = T_WB.C().transpose();

  Eigen::Matrix<double, 6, 6> pose_prior_covariance = Eigen::Matrix<double, 6, 6>::Zero();
  pose_prior_covariance.diagonal().head<3>() = pvstd_.std_q_WS.cwiseAbs2();
  pose_prior_covariance.diagonal().tail<3>() = pvstd_.std_p_WS.cwiseAbs2();

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

  Eigen::Matrix<double, 9, 1> vel_bias;
  getSpeedAndBias(frameId, 0u, vel_bias);
  // Add initial velocity priors.
  gtsam::SharedNoiseModel noise_init_vel_prior =
      gtsam::noiseModel::Diagonal::Sigmas(pvstd_.std_v_WS);
  Eigen::Vector3d vel = vel_bias.head<3>();
  new_imu_prior_and_other_factors_.push_back(
      boost::make_shared<gtsam::PriorFactor<gtsam::Vector3>>(
          gtsam::Symbol('v', frameId), vel, noise_init_vel_prior));

  // Add initial bias priors:
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
  navStateToLandmarks_.insert({cur_id, std::vector<uint64_t>()});
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
      // sqrt(pim_->deltaTij() * nominalImuRate_) to count the nr of measurements.
      const double d = std::sqrt(resultPim->deltaTij());
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
  std::string distortionName = camera_rig_.getCameraGeometry(0u)->distortionType();
  OKVIS_ASSERT_EQ(Exception, distortionName, "RadialTangentialDistortion",
                  "Sliding window smoother currently only work radial "
                  "tangential distortion!");
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
  Time newStateTime = multiFrame->timestamp();
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
    speedAndBias.segment<3>(3) = imuParametersVec_.at(0).g0;
    speedAndBias.tail<3>() = imuParametersVec_.at(0).a0;
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
  // TODO(jhuai): add camera extrinsic parameter blocks to gtsam graph?

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
    if (state_.find(gtsam::Symbol('x', rit->first)) == state_.end()) {
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

// Get a copy of all the landmarks as a PointMap.
size_t SlidingWindowSmoother::getLandmarks(MapPointVector& landmarks) const {
  if (backendParams_.backendModality_ == BackendModality::STRUCTURELESS) {
    std::lock_guard<std::mutex> l(Estimator::statesMutex_);
    LmkIdToLmkTypeMap lmk_id_to_lmk_type_map;
    size_t min_age = 2u;
    PointsWithIdMap points_with_id =
        getMapLmkIdsTo3dPointsInTimeHorizon(&lmk_id_to_lmk_type_map, min_age);

    landmarks.clear();
    landmarks.reserve(points_with_id.size());

    for (auto point : points_with_id) {
      landmarks.emplace_back(point.first,
                             Eigen::Vector4d(point.second[0], point.second[1],
                                             point.second[2], 1.0),
                             1.0, std::fabs(point.second[2]));
    }
//    LOG(INFO) << "Found " << points_with_id.size() << " valid landmarks out of "
//              << old_smart_factors_.size() << " smart factors!";
    return landmarks.size();
  } else {
    return Estimator::getLandmarks(landmarks);
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
    // TODO(jhuai): keep at least one keyframe for visualization.
    if (it->first < minValidStateId) {
      removeFrames.push_back(it->second.id);
    } else {
      break;
    }
    ++it;
  }

  // The smoother has also marginalized landmarks which were added to the graph
  // at the same time as these nav state variables. We reset the status of such
  // a landmark's observations in order to avoid adding further reprojection
  // factors for the landmark. This means new observations for a landmark moving
  // out the time horizon have to be associated to a new landmark.

  // Kimera-VIO simply removes old smart factors out of the time horizon, see
  // https://github.com/MIT-SPARK/Kimera-VIO/blob/master/src/backend/VioBackEnd.cpp#L926-L929.
  // It replaces the old smart factor with a new one when an observation for the
  // landmark arrives.
  for (auto frameId : removeFrames) {
    auto itemPtr = navStateToLandmarks_.find(frameId);
    if (itemPtr == navStateToLandmarks_.end()) continue;
    for (auto lmkId : itemPtr->second) {
      // This landmark is still in landmarksMap_ because its feature track still
      // overlaps the sliding window.
      landmarksMap_.at(lmkId).residualizeCase =
          NotInState_NotTrackedNow;  // set to not in state.
    }
    navStateToLandmarks_.erase(itemPtr);
  }

  // remove smart factors that are marginalized from the graph or whose pointer changes.
  const gtsam::NonlinearFactorGraph& graph = smoother_->getFactors();
  for (SmartFactorMap::const_iterator old_smart_factor_it =
           old_smart_factors_.begin();
       old_smart_factor_it != old_smart_factors_.end();) {
    // When a feature track is so long (longer than factor graph's time
    // horizon), then the factor is marginalization from the graph. Erase this
    // factor and feature track, as it has gone past the horizon.
    LandmarkId lmkId = old_smart_factor_it->first;
    Slot slot_id = old_smart_factor_it->second.second;
    if (!graph.exists(slot_id)) {
      old_smart_factor_it = old_smart_factors_.erase(old_smart_factor_it);
      landmarksMap_.at(lmkId).residualizeCase = NotInState_NotTrackedNow;
      continue;
    }

    // Check that the pointer smart_factor_ptr points to the right element
    // in the graph.
    auto smart_factor_ptr = old_smart_factor_it->second.first;
    if (smart_factor_ptr != graph.at(slot_id)) {
      // Pointer in the graph does not match
      // the one we stored in old_smart_factors_
      LOG(ERROR) << "The factor with slot id: " << slot_id
                 << " in the graph does not match the old_smart_factor of "
                 << "lmk with id: " << lmkId << "\n."
                 << "Deleting old_smart_factor of the lmk";
      old_smart_factor_it = old_smart_factors_.erase(old_smart_factor_it);
      landmarksMap_.at(lmkId).residualizeCase = NotInState_NotTrackedNow;
      continue;
    }
    ++old_smart_factor_it;
  }

  // remove feature tracks that do not overlap the sliding window.
  for (PointMap::iterator pit = landmarksMap_.begin();
       pit != landmarksMap_.end();) {
    const MapPoint& mapPoint = pit->second;
    // Remove a landmark whose last observation is out of the horizon.
    uint64_t lastFrameId = mapPoint.observations.rbegin()->first.frameId;
    if (lastFrameId < minValidStateId) {
      size_t numberObservations = mapPoint.observations.size();
      if (numberObservations + 1u > mTrackLengthAccumulator.size()) {
          numberObservations = mTrackLengthAccumulator.size() - 1u;
      }
      ++mTrackLengthAccumulator[numberObservations];
      // It is not necessary to remove residual blocks or uncheck keypoints in
      // multiframes because these residual blocks and multiframes have been or
      // will soon be removed as the associated nav state slides out of the
      // optimization window. see the removeState line.

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

// Refer to InvUVFactor and process_feat_normal in CPI, closed-form preintegration repo of Eckenhoff.
bool SlidingWindowSmoother::addLandmarkToGraph(uint64_t lmkId, const Eigen::Vector4d& hpW) {
  std::shared_ptr<okvis::ceres::HomogeneousPointParameterBlock>
      pointParameterBlock(
          new okvis::ceres::HomogeneousPointParameterBlock(hpW, lmkId));
  mapPtr_->addParameterBlock(pointParameterBlock,
                             okvis::ceres::Map::HomogeneousPoint);
  // addParameterBlock may fail when a landmark moves out of the smoother's
  // horizon and then reappears.

  new_values_.insert(gtsam::symbol('l', lmkId), gtsam::Point3(hpW.head<3>() / hpW[3]));

  navStateToLandmarks_.at(statesMap_.rbegin()->first).push_back(lmkId);

  uint64_t minValidStateId = statesMap_.begin()->first;
  okvis::MapPoint& mp = landmarksMap_.at(lmkId);
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
    lastImageId =
        IsObservedInFrame(obsIter->first.frameId, obsIter->first.cameraIndex);
  }
  mp.residualizeCase = InState_TrackedNow;
  return true;
}

void SlidingWindowSmoother::updateLandmarkInGraph(uint64_t lmkId) {
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

void SlidingWindowSmoother::addLandmarkSmartFactorToGraph(const LandmarkId& lmkId) {
  uint64_t minValidStateId = statesMap_.begin()->first;
  okvis::MapPoint& mp = landmarksMap_.at(lmkId);
  auto obsIt = mp.observations.lower_bound(okvis::KeypointIdentifier(minValidStateId, 0u, 0u));
  size_t numValidObs = std::distance(obsIt, mp.observations.end());
  if (numValidObs < minTrackLength_) {
      return;
  }
  std::shared_ptr<okvis::ceres::HomogeneousPointParameterBlock>
      pointParameterBlock(
          new okvis::ceres::HomogeneousPointParameterBlock(mp.pointHomog, lmkId));
  mapPtr_->addParameterBlock(pointParameterBlock,
                             okvis::ceres::Map::HomogeneousPoint);
  // addParameterBlock may fail when a landmark moves out of the smoother's
  // horizon and then reappears.

  gtsam::SmartProjectionPoseFactor<gtsam::Cal3DS2>::shared_ptr new_factor =
      boost::make_shared<gtsam::SmartProjectionPoseFactor<gtsam::Cal3DS2>>(
          smart_noise_, cal0_, body_P_cam0_, smart_factors_params_);
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

  new_smart_factors_.emplace(lmkId, new_factor);
  old_smart_factors_.emplace(lmkId, std::make_pair(new_factor, -1));
}

void SlidingWindowSmoother::updateLandmarkSmartFactorInGraph(
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
  const gtsam::SmartProjectionPoseFactor<gtsam::Cal3DS2>::shared_ptr
      old_factor = boost::static_pointer_cast<
          gtsam::SmartProjectionPoseFactor<gtsam::Cal3DS2>>(
          old_smart_factors_it->second.first);
  // Clone old factor to keep all previous measurements, now append one.
  gtsam::SmartProjectionPoseFactor<gtsam::Cal3DS2>::shared_ptr new_factor =
      boost::make_shared<gtsam::SmartProjectionPoseFactor<gtsam::Cal3DS2>>(
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

void SlidingWindowSmoother::assembleNewFactorsAndDeleteSlots(
    gtsam::NonlinearFactorGraph* new_factors_tmp,
    gtsam::FactorIndices* delete_slots,
    std::vector<LandmarkId>* lmk_ids_of_new_smart_factors_tmp) const {
  if (backendParams_.backendModality_ == BackendModality::STRUCTURELESS) {
    /////////////////////// BOOKKEEPING ////////////////////////////////////
    size_t new_smart_factors_size = new_smart_factors_.size();
    // We need to remove all previous smart factors in the factor graph
    // for which we have new observations.
    // The following is just to update the vector delete_slots with those
    // slots in the factor graph that correspond to smart factors for which
    // we've got new observations.
    // We initialize delete_slots with Extra factor slots to delete contains
    // potential factors that we want to delete, it is typically an empty
    // vector, and is only used to give flexibility to subclasses (regular
    // vio).

    // TODO we know the actual end size... but I am not sure how to use factor
    // graph API for appending factors without copying or re-allocation...

    lmk_ids_of_new_smart_factors_tmp->reserve(new_smart_factors_size);

    new_factors_tmp->reserve(new_smart_factors_size +
                            new_imu_prior_and_other_factors_.size());
    for (const auto& new_smart_factor : new_smart_factors_) {
      // Push back the smart factor to the list of new factors to add to the
      // graph. // Smart factor, so same address right?
      LandmarkId lmk_id = new_smart_factor.first;  // don't use &

      // Find smart factor and slot in old_smart_factors_ corresponding to
      // the lmk with id of the new smart factor.
      const auto& old_smart_factor_it = old_smart_factors_.find(lmk_id);
      CHECK(old_smart_factor_it != old_smart_factors_.end())
          << "Lmk with id: " << lmk_id
          << " could not be found in old_smart_factors_.";

      Slot slot = old_smart_factor_it->second.second;
      if (slot != -1) {
        // Smart factor Slot is different than -1, therefore the factor should
        // be already in the factor graph.
        DCHECK_GE(slot, 0);
        if (smoother_->getFactors().exists(slot)) {
          // Confirmed, the factor is in the graph.
          // We must delete the old smart factor from the graph.
          // TODO what happens if delete_slots has repeated elements?
          delete_slots->push_back(slot);
          // And we must add the new smart factor to the graph.
          new_factors_tmp->push_back(new_smart_factor.second);
          // Store lmk id of the smart factor to add to the graph.
          lmk_ids_of_new_smart_factors_tmp->push_back(lmk_id);
        } else {
          LOG(WARNING)
              << "Smart factor added to the graph is not found. It should not "
                 "happen because we purge smart factors from old_smart_factors "
                 "that are marginalized from the graph.";
        }
      } else {
        // We just add the new smart factor to the graph, as it has never been
        // there before.
        new_factors_tmp->push_back(new_smart_factor.second);
        // Store lmk id of the smart factor to add to the graph.
        lmk_ids_of_new_smart_factors_tmp->push_back(lmk_id);
      }
    }

    // Add also other factors (imu, priors).
    // SMART FACTORS MUST BE FIRST, otherwise when recovering the slots
    // for the smart factors we will mess up.
    // push back many factors with an iterator over shared_ptr
    // (factors are not copied)
    new_factors_tmp->push_back(new_imu_prior_and_other_factors_.begin(),
                              new_imu_prior_and_other_factors_.end());
  } else {
    size_t new_reproj_factors_size = new_reprojection_factors_.size();
    new_factors_tmp->reserve(new_reproj_factors_size +
                            new_imu_prior_and_other_factors_.size());
    new_factors_tmp->push_back(new_reprojection_factors_.begin(),
                              new_reprojection_factors_.end());
    new_factors_tmp->push_back(new_imu_prior_and_other_factors_.begin(),
                              new_imu_prior_and_other_factors_.end());
  }
}

void SlidingWindowSmoother::updateStates() {
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

    gtsam::Pose3 W_T_B =
        estimates.at<gtsam::Pose3>(gtsam::Symbol('x', stateId));

    std::shared_ptr<ceres::PoseParameterBlock> poseParamBlockPtr =
        std::static_pointer_cast<ceres::PoseParameterBlock>(
            mapPtr_->parameterBlockPtr(stateId));
    kinematics::Transformation T_WB = VIO::GtsamWrap::toTransform(W_T_B);
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

    auto vval = estimates.find(gtsam::Symbol('v', stateId));
    gtsam::Vector3 W_v_B =
        estimates.at<gtsam::Vector3>(gtsam::Symbol('v', stateId));
    sb.head<3>() = W_v_B;

    auto bval = estimates.find(gtsam::Symbol('b', stateId));
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
      gtsam::Point3 pW = estimates.at<gtsam::Point3>(gtsam::Symbol('l', lmkId));
      Eigen::Vector4d hpW;
      hpW.head<3>() = pW;
      hpW[3] = 1.0;

      it->second.quality = 1.0;
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
//    LOG(INFO) << "Starting update of smoother_...";
    *result = smoother_->update(new_factors, new_values, timestamps, delete_slots);
//    LOG(INFO) << "Finished update of smoother_.";
  } catch (const gtsam::IndeterminantLinearSystemException& e) {
    const gtsam::Key& var = e.nearbyVariable();
    gtsam::Symbol symb(var);

    LOG(ERROR) << "IndeterminantLinearSystemException: Nearby variable has type '" << symb.chr() << "' "
               << "and index " << symb.index() << std::endl;
    LOG(ERROR) << e.what();
//    smoother_->getFactors().print("Smoother's factors:\n[\n\t");
//    LOG(INFO) << " ]";
    state_.print("State values\n[\n\t");
    LOG(INFO) << " ]";
    printSmootherInfo(new_factors, delete_slots);
    if (symb.chr() == 'l') {
      got_cheirality_exception = true;
      lmk_symbol_cheirality = symb;
    } else {
      return false;
    }
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
    // jhuai: When a landmark in the NonlinearFactorGraph is indetermined, and
    // we delete all factors for the landmark from the graph, an instance of
    // 'std::out_of_range' map::at may be thrown here. I suspect
    // IncrementalFixedLagSmoother has problems erasing values not associated to
    // any factors.
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

bool SlidingWindowSmoother::triangulateSafe(uint64_t lmkId, Eigen::Vector4d* hpW) const {
  const MapPoint& mp = landmarksMap_.at(lmkId);
  uint64_t minValidStateId = statesMap_.begin()->first;

  gtsam::Cal3_S2 cal(1, 1, 0, 0, 0);
  double landmarkDistanceThreshold = 50;  // Ignore points farther than this.
  gtsam::TriangulationParameters params(
      1.0, false, landmarkDistanceThreshold);
  gtsam::CameraSet<gtsam::PinholeCameraCal3_S2> cameras;
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
    gtsam::PinholeCameraCal3_S2 camera(W_T_B.compose(body_P_cam0_), cal);

    cameras.push_back(camera);
    measurements.push_back(gtsam::Point2(backProjectionDirection.head<2>()));
  }

  gtsam::TriangulationResult result = gtsam::triangulateSafe(cameras, measurements, params);
  if (result.valid()) {
    hpW->head<3>() = Eigen::Vector3d(*result);
    hpW->w() = 1.0;
    return true;
  } else {
    return false;
  }
}

bool SlidingWindowSmoother::triangulateWithDisparityCheck(
    uint64_t lmkId, Eigen::Matrix<double, 4, 1>* hpW,
    double focalLength, double raySigmaScalar) const {
  const MapPoint& mp = landmarksMap_.at(lmkId);
  AlignedVector<Eigen::Vector3d> obsDirections;
  AlignedVector<okvis::kinematics::Transformation> T_CWs;
  std::vector<double> imageNoiseStd;
  size_t numObs = gatherMapPointObservations(mp, &obsDirections, &T_CWs, &imageNoiseStd);
  if (numObs < minTrackLength_) {
    return false;
  }
  if (msckf::hasLowDisparity(obsDirections, T_CWs, imageNoiseStd, focalLength, raySigmaScalar))
    return false;
  *hpW = msckf::triangulateHomogeneousDLT(obsDirections, T_CWs);
  *hpW /= hpW->w();
  return true;
}

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
  // To avoid confusion, do not interpret the second half of the status value.

  Eigen::VectorXd intrinsics;
  camera_rig_.getCameraGeometry(0)->getIntrinsics(intrinsics);
  double focalLength = intrinsics[0];
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
    if (landmarkStatus == NotInState_NotTrackedNow) {
      // The landmark has not been added to the graph.
      if (observedInCurrentFrame) {
        if (backendParams_.backendModality_ == BackendModality::PROJECTION) {
          Eigen::Vector4d hpW;
          // Preliminary test implied that triangulateSafe may be lead to worse
          // result than  triangulateWithDisparityCheck.
          //        bool triangulateOk = triangulateSafe(it->first, &pW);
          bool triangulateOk = triangulateWithDisparityCheck(
              it->first, &hpW, focalLength, FLAGS_ray_sigma_scalar);
          if (triangulateOk) {
            addLandmarkToGraph(it->first, hpW);
          }  // else do nothing
        } else {
          addLandmarkSmartFactorToGraph(it->first);
        }
      }  // else do nothing
    } else {  // The landmark has been added to the graph.
      if (observedInCurrentFrame) {
        if (backendParams_.backendModality_ == BackendModality::PROJECTION) {
          updateLandmarkInGraph(it->first);
        } else {
          updateLandmarkSmartFactorInGraph(it->first);
        }
      }  // else do nothing
    }
  }
  addLandmarkFactorsTimer.stop();

  trackingRate_ = static_cast<double>(numTracked) /
                  static_cast<double>(landmarksMap_.size());

  gtsam::NonlinearFactorGraph new_factors_tmp;
  gtsam::FactorIndices delete_slots;
  std::vector<LandmarkId> lmk_ids_of_new_smart_factors_tmp;
  assembleNewFactorsAndDeleteSlots(&new_factors_tmp, &delete_slots,
                                   &lmk_ids_of_new_smart_factors_tmp);

  // Use current timestamp for each new value. This timestamp will be used
  // to determine if the variable should be marginalized.
  // Needs to use DOUBLE in secs because gtsam works with that.
  std::map<gtsam::Key, double> timestamps;
  okvis::Time currentTime = statesMap_.rbegin()->second.timestamp;
  double currentTimeSecs = currentTime.toSec();
  for(const gtsam::Values::ConstKeyValuePair& key_value : new_values_) {
    timestamps[key_value.key] = currentTimeSecs;
  }

  isam2UpdateTimer.start();
//  LOG(INFO) << "iSAM2 update with " << new_factors_tmp.size() << " new factors"
//           << ", " << new_values_.size() << " new values"
//           << ", and " << delete_slots.size() << " deleted factors.";
  Smoother::Result result;
  bool is_smoother_ok = updateSmoother(&result, new_factors_tmp, new_values_,
                                       timestamps, delete_slots);

  if (is_smoother_ok) {
    // Reset everything for next round.
    new_smart_factors_.clear();
    new_reprojection_factors_.resize(0);

    // Reset list of new imu, prior and other factors to be added.
    new_imu_prior_and_other_factors_.resize(0);

    new_values_.clear();

#ifdef INCREMENTAL_SMOOTHER
    updateNewSmartFactorsSlots(lmk_ids_of_new_smart_factors_tmp,
                               &old_smart_factors_);
#else
    if (backendParams_.backendModality_ == BackendModality::STRUCTURELESS) {
      throw std::runtime_error("Updating smart factor indices has not been implemented!");
    }
#endif
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
  return gtsamMarginalCovariance(cov);
}

bool SlidingWindowSmoother::iSAM2MarginalCovariance(
    Eigen::MatrixXd* cov) const {
  uint64_t T_WS_id = statesMap_.rbegin()->second.id;
  *cov = Eigen::Matrix<double, 15, 15>::Identity();
  // OKVIS Bg Ba, GTSAM Ba Bg.
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
      swapBaBg * smoother_->marginalCovariance(gtsam::Symbol('b', T_WS_id)) *
      swapBaBg.transpose();
  return true;
}

bool SlidingWindowSmoother::gtsamMarginalCovariance(
    Eigen::MatrixXd* cov) const {
  gtsam::Marginals marginals(smoother_->getFactors(), state_,
                             gtsam::Marginals::Factorization::CHOLESKY);
  uint64_t T_WS_id = statesMap_.rbegin()->first;
  okvis::kinematics::Transformation T_WB;
  get_T_WS(T_WS_id, T_WB);

  // OKVIS Bg Ba, GTSAM Ba Bg.
  Eigen::Matrix<double, 6, 6> swapBaBg = Eigen::Matrix<double, 6, 6>::Zero();
  swapBaBg.topRightCorner<3, 3>().setIdentity();
  swapBaBg.bottomLeftCorner<3, 3>().setIdentity();

  // okvis pW = \hat pW + \delta pW, R_WB = \hat R_WB exp(\delta \theta_W);
  // gtsam pW = \hat pW + R_WB \delta pB, R_WB = exp(\delta \theta B) \hat R_WB.
  Eigen::Matrix<double, 6, 6> swapRT = Eigen::Matrix<double, 6, 6>::Zero();
  swapRT.topRightCorner<3, 3>() = T_WB.C();
  swapRT.bottomLeftCorner<3, 3>() = T_WB.C();
  *cov = Eigen::Matrix<double, 15, 15>::Identity();
  Eigen::MatrixXd poseMargCov =
      marginals.marginalCovariance(gtsam::Symbol('x', T_WS_id));
  cov->topLeftCorner<6, 6>() = swapRT * poseMargCov * swapRT.transpose();
  cov->block<3, 3>(6, 6) =
      marginals.marginalCovariance(gtsam::Symbol('v', T_WS_id));
  cov->block<6, 6>(9, 9) =
      swapBaBg * marginals.marginalCovariance(gtsam::Symbol('b', T_WS_id)) *
      swapBaBg.transpose();
  return true;
}

bool SlidingWindowSmoother::gtsamJointMarginalCovariance(
    Eigen::MatrixXd* cov) const {
  gtsam::Marginals marginals(smoother_->getFactors(), state_,
                             gtsam::Marginals::Factorization::CHOLESKY);
  uint64_t T_WS_id = statesMap_.rbegin()->first;
  // Current state includes pose, velocity and imu biases.
  gtsam::KeyVector keys;
  keys.push_back(gtsam::Symbol('x', T_WS_id));
  keys.push_back(gtsam::Symbol('v', T_WS_id));
  keys.push_back(gtsam::Symbol('b', T_WS_id));

  // The marginal covariance matrix [ba, bg, v_W, \delta p_B, \phi_B].
  Eigen::MatrixXd jointCov =
      marginals.jointMarginalCovariance(keys).fullMatrix();
  okvis::kinematics::Transformation T_WB;
  get_T_WS(T_WS_id, T_WB);
  Eigen::Matrix<double, 15, 15> dokvis_dgtsam =
      Eigen::Matrix<double, 15, 15>::Zero();
  dokvis_dgtsam.topRightCorner<3, 3>() = T_WB.C();
  dokvis_dgtsam.block<3, 3>(3, 9) = T_WB.C();
  dokvis_dgtsam.block<3, 3>(6, 6).setIdentity();
  dokvis_dgtsam.block<3, 3>(9, 3).setIdentity();
  dokvis_dgtsam.block<3, 3>(12, 0).setIdentity();
  // The marginal covariance matrix [\phi_W, \delta p_W, v_W, bg, ba].
  *cov = dokvis_dgtsam * jointCov * dokvis_dgtsam.transpose();
  return true;
}

bool SlidingWindowSmoother::getStateStd(
    Eigen::Matrix<double, Eigen::Dynamic, 1>* stateStd) const {
  Eigen::MatrixXd covariance;
  bool status = computeCovariance(&covariance);
  *stateStd = covariance.diagonal().cwiseSqrt();
  return status;
}

void printSmartFactor(
    boost::shared_ptr<
        gtsam::SmartProjectionFactor<gtsam::PinholePose<gtsam::Cal3DS2>>>
        gsf) {
  CHECK(gsf);
  std::cout << "Smart Factor (valid: " << (gsf->isValid() ? "yes" : "NO!")
            << ", deg: " << (gsf->isDegenerate() ? "YES!" : "no")
            << " isCheir: " << (gsf->isPointBehindCamera() ? "YES!" : "no")
            << "): \t";
  gsf->printKeys();
}

void printReprojectionFactor(boost::shared_ptr<gtsam::GenericProjectionFactor<
                             gtsam::Pose3, gtsam::Point3, gtsam::Cal3DS2>> gpf) {
  CHECK(gpf);
  std::cout << "Reprojection Factor: keys \t";
  gpf->printKeys();
}

void printPointPrior(
    boost::shared_ptr<gtsam::PriorFactor<gtsam::Point3>> ppp) {
  CHECK(ppp);
  std::cout << "Point Prior: point key \t";
  ppp->printKeys();
}

void printLinearContainerFactor(
    boost::shared_ptr<gtsam::LinearContainerFactor> lcf) {
  CHECK(lcf);
  std::cout << "Linear Container Factor: \t";
  lcf->printKeys();
}

void printSelectedFactors(
    const boost::shared_ptr<gtsam::NonlinearFactor> g,
    const size_t slot,
    const bool print_reprojection_factors,
    const bool print_smart_factors,
    const bool print_point_priors,
    const bool print_linear_container_factors) {
  if (print_reprojection_factors) {
    const auto& gpf =
        boost::dynamic_pointer_cast<gtsam::GenericProjectionFactor<
            gtsam::Pose3, gtsam::Point3, gtsam::Cal3DS2>>(g);
    if (gpf) {
      std::cout << "\tSlot # " << slot << ": ";
      printReprojectionFactor(gpf);
    }
  }

  if (print_smart_factors) {
    const auto& gsf = boost::dynamic_pointer_cast<
        gtsam::SmartProjectionFactor<gtsam::PinholePose<gtsam::Cal3DS2>>>(g);
    if (gsf) {
      std::cout << "\tSlot # " << slot << ": ";
      printSmartFactor(gsf);
    }
  }

  if (print_point_priors) {
    const auto& ppp =
        boost::dynamic_pointer_cast<gtsam::PriorFactor<gtsam::Point3>>(g);
    if (ppp) {
      std::cout << "\tSlot # " << slot << ": ";
      printPointPrior(ppp);
    }
  }

  if (print_linear_container_factors) {
    const auto& lcf =
        boost::dynamic_pointer_cast<gtsam::LinearContainerFactor>(g);
    if (lcf) {
      std::cout << "\tSlot # " << slot << ": ";
      printLinearContainerFactor(lcf);
    }
  }
}

void printSelectedGraph(
    const gtsam::NonlinearFactorGraph& graph,
    const bool print_reprojection_factors,
    const bool print_smart_factors,
    const bool print_point_priors,
    const bool print_linear_container_factors) {
  size_t slot = 0;
  for (const auto& g : graph) {
    printSelectedFactors(g,
                         slot,
                         print_reprojection_factors,
                         print_smart_factors,
                         print_point_priors,
                         print_linear_container_factors);
    slot++;
  }
  std::cout << std::endl;
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

  static constexpr bool print_reprojection_factors = true;
  static constexpr bool print_smart_factors = true;  // There a lot of these!
  static constexpr bool print_point_priors = true;
  static constexpr bool print_linear_container_factors = true;
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
  LOG(INFO) << "[\n (slot # wrt to new_factors_tmp graph) \t";
  printSelectedGraph(new_factors_tmp,
                     print_reprojection_factors,
                     print_smart_factors,
                     print_point_priors,
                     print_linear_container_factors);
  LOG(INFO) << " ]" << std::endl;

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
  std::stringstream ss("[\t");
  for (const gtsam::Values::ConstKeyValuePair& key_value : new_values_) {
    ss << " " << gtsam::DefaultKeyFormatter(key_value.key) << " ";
  }
  ss << "]";
  LOG(INFO) << ss.str();

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
bool SlidingWindowSmoother::deleteLmkFromFeatureTracks(uint64_t lmkId) {
  auto pit = landmarksMap_.find(lmkId);
  const MapPoint& mapPoint = pit->second;
  VLOG(2) << "Deleting feature track for lmk with id: " << lmkId;
  ++mTrackLengthAccumulator[mapPoint.observations.size() >=
                                    mTrackLengthAccumulator.size()
                                ? mTrackLengthAccumulator.size() - 1u
                                : mapPoint.observations.size()];
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
  return true;
}

void deleteAllFactorsWithKeyFromFactorGraph(
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
        CHECK(!boost::dynamic_pointer_cast<
              gtsam::SmartProjectionFactor<gtsam::PinholePose<gtsam::Cal3DS2>>>(
            *it));
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
bool deleteKeyFromTimestamps(
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
bool deleteKeyFromValues(const gtsam::Key& key,
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
void findSlotsOfFactorsWithKey(
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
        CHECK(!boost::dynamic_pointer_cast<
              gtsam::SmartProjectionFactor<gtsam::PinholePose<gtsam::Cal3DS2>>>(
            g));
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
  printNavStateAndBiases(stream, statesMap_.rbegin()->first);
  Eigen::Matrix<double, Eigen::Dynamic, 1> variances = covariance_.diagonal();
  stream << " " << variances.cwiseSqrt().transpose().format(kSpaceInitFmt);
  return true;
}

// BOOKKEEPING: updates the SlotIdx in the old_smart_factors such that
// this idx points to the updated slots in the graph after optimization.
// for next iteration to know which slots have to be deleted
// before adding the new smart factors.
void SlidingWindowSmoother::updateNewSmartFactorsSlots(
    const std::vector<LandmarkId>& lmk_ids_of_new_smart_factors,
    SmartFactorMap* old_smart_factors) {
  const gtsam::ISAM2Result& result = smoother_->getISAM2Result();

  // Simple version of find smart factors.
  for (size_t i = 0u; i < lmk_ids_of_new_smart_factors.size(); ++i) {
    DCHECK(i < result.newFactorsIndices.size())
        << "There are more new smart factors than new factors added to the "
           "graph.";
    // Get new slot in the graph for the newly added smart factor.
    const size_t& slot = result.newFactorsIndices.at(i);

    // TODO this will not work if there are non-smart factors!!!
    // Update slot using isam2 indices.
    // ORDER of inclusion of factors in the ISAM2::update() function
    // matters, as these indices have a 1-to-1 correspondence with the
    // factors.

    // BOOKKEEPING, for next iteration to know which slots have to be
    // deleted before adding the new smart factors. Find the entry in
    // old_smart_factors_.
    SmartFactorMap::iterator it =
        old_smart_factors->find(lmk_ids_of_new_smart_factors.at(i));

    DCHECK(it != old_smart_factors->end())
        << "Trying to access unavailable factor.";
    const gtsam::NonlinearFactorGraph& graph = smoother_->getFactors();
    auto factor = boost::dynamic_pointer_cast<
        gtsam::SmartProjectionFactor<gtsam::PinholePose<gtsam::Cal3DS2>>>(graph.at(slot));
    if (factor) {
      // CHECK that shared ptrs point to the same smart factor.
      // make sure no one is cloning SmartSteroFactors.
      DCHECK_EQ(it->second.first, factor)
          << "Non-matching addresses for same factors for lmk with id: "
          << lmk_ids_of_new_smart_factors.at(i) << " in old_smart_factors_ "
          << "VS factor in graph at slot: " << slot
          << ". Slot previous to update was: " << it->second.second;

      // Update slot number in old_smart_factors_.
      it->second.second = slot;
    } else {
      auto oldfactor = it->second.first;
      uint64_t frameId = gtsam::Symbol(oldfactor->keys().front()).index();
      OKVIS_ASSERT_LT(Exception, frameId, statesMap_.rbegin()->first,
                      "If a smart factor of landmark "
                          << lmk_ids_of_new_smart_factors.at(i)
                          << " with the first pose " << frameId
                          << " is no longer in the graph, it should have been "
                             "marginalized!");
    }
  }
}

PointsWithIdMap SlidingWindowSmoother::getMapLmkIdsTo3dPointsInTimeHorizon(
    LmkIdToLmkTypeMap* lmk_id_to_lmk_type_map,
    const size_t& min_age) const {
  PointsWithIdMap points_with_id;

  if (lmk_id_to_lmk_type_map) {
    lmk_id_to_lmk_type_map->clear();
  }

  // Step 1:
  /////////////// Add landmarks encoded in the smart factors. //////////////////
  const gtsam::NonlinearFactorGraph& graph = smoother_->getFactors();

  // old_smart_factors_ has all smart factors included so far.
  // Retrieve lmk ids from smart factors in state.
  size_t nr_valid_smart_lmks = 0, nr_smart_lmks = 0, nr_proj_lmks = 0;
  for (SmartFactorMap::const_iterator old_smart_factor_it =
           old_smart_factors_.begin();
       old_smart_factor_it != old_smart_factors_.end();) {  //!< landmarkId -> {SmartFactorPtr, SlotIndex}
    // Store number of smart lmks (one smart factor per landmark).
    nr_smart_lmks++;

    // Retrieve lmk_id of the smart factor.
    const LandmarkId& lmk_id = old_smart_factor_it->first;

    // Retrieve smart factor.
    const gtsam::SmartProjectionFactor<
        gtsam::PinholePose<gtsam::Cal3DS2>>::shared_ptr smart_factor_ptr =
        old_smart_factor_it->second.first;
    // Check that pointer is well definied.
    CHECK(smart_factor_ptr) << "Smart factor is not well defined.";

    // Retrieve smart factor slot in the graph.
    const Slot& slot_id = old_smart_factor_it->second.second;

    // Check that slot is admissible.
    // Slot should be positive.
    DCHECK(slot_id >= 0) << "Slot of smart factor is not admissible.";
    // Ensure the graph size is small enough to cast to int.
    DCHECK_LT(graph.size(), std::numeric_limits<Slot>::max())
        << "Invalid cast, that would cause an overflow!";
    // Slot should be inferior to the size of the graph.
    DCHECK_LT(slot_id, static_cast<Slot>(graph.size()));

    // Check that this slot_id exists in the graph, aka check that it is
    // in bounds and that the pointer is live (aka at(slot_id) works).
    if (!graph.exists(slot_id)) {
      // This slot does not exist in the current graph...
      LOG(WARNING)
          << "The slot with id: " << slot_id
          << " does not exist in the graph.\n"
          << "This should not happen because we purge old_smart_factors!";
      ++old_smart_factor_it;
      continue;
    }

    // Check that the pointer smart_factor_ptr points to the right element
    // in the graph.
    if (smart_factor_ptr != graph.at(slot_id)) {
      // Pointer in the graph does not match
      // the one we stored in old_smart_factors_
      // ERROR: if the pointers don't match, then the code that follows does
      // not make any sense, since we are using lmk_id which comes from
      // smart_factor and result which comes from graph[slot_id], we should
      // use smart_factor_ptr instead then...
      LOG(WARNING) << "The factor with slot id: " << slot_id
                 << " in the graph does not match the old_smart_factor of "
                 << "lmk with id: " << lmk_id << "\n."
                 << "Deleting old_smart_factor of lmk id: " << lmk_id
                 << "\nThis should not happen because we purge old_smart_factors!";
      ++old_smart_factor_it;
      continue;
    }

    // Why do we do this? all info is in smart_factor_ptr
    // such as the triangulated point, whether it is valid or not
    // and the number of observations...
    // Is graph more up to date?
    boost::shared_ptr<
        gtsam::SmartProjectionFactor<gtsam::PinholePose<gtsam::Cal3DS2>>>
        gsf = boost::dynamic_pointer_cast<
            gtsam::SmartProjectionFactor<gtsam::PinholePose<gtsam::Cal3DS2>>>(
            graph.at(slot_id));
    CHECK(gsf) << "Cannot cast factor in graph to a smart stereo factor.";

    // Get triangulation result from smart factor.
    const gtsam::TriangulationResult& result = gsf->point();
    // Check that the boost::optional result is initialized.
    // Otherwise we will be dereferencing a nullptr and we will head
    // directly to undefined behaviour wonderland.
    if (result.is_initialized()) {
      if (result.valid()) {
        if (gsf->measured().size() >= min_age) {
          // Triangulation result from smart factor is valid and
          // we have observed the lmk at least min_age times.
          VLOG(20) << "Adding lmk with id: " << lmk_id
                   << " to list of lmks in time horizon";
          // Check that we have not added this lmk already...
          CHECK(points_with_id.find(lmk_id) == points_with_id.end());
          points_with_id[lmk_id] = *result;
          if (lmk_id_to_lmk_type_map) {
            (*lmk_id_to_lmk_type_map)[lmk_id] = LandmarkType::SMART;
          }
          nr_valid_smart_lmks++;
        } else {
          VLOG(20) << "Rejecting lmk with id: " << lmk_id
                   << " from list of lmks in time horizon: "
                   << "not enough measurements, " << gsf->measured().size()
                   << ", vs min_age of " << min_age << ".";
        }  // gsf->measured().size() >= min_age ?
      } else {
        VLOG(20) << "Rejecting lmk with id: " << lmk_id
                 << " from list of lmks in time horizon:\n"
                 << "triangulation result is not valid (result= {" << result
                 << "}).";
      }  // result.valid()?
    } else {
      VLOG(20) << "Triangulation result for smart factor of lmk with id "
               << lmk_id << " is not initialized...";
    }  // result.is_initialized()?

    // Next iteration.
    old_smart_factor_it++;
  }

  // Step 2:
  ////////////// Add landmarks that now are in projection factors. /////////////
  for (const gtsam::Values::Filtered<gtsam::Value>::ConstKeyValuePair&
           key_value : state_.filter(gtsam::Symbol::ChrTest('l'))) {
    DCHECK_EQ(gtsam::Symbol(key_value.key).chr(), 'l');
    const LandmarkId& lmk_id = gtsam::Symbol(key_value.key).index();
    DCHECK(points_with_id.find(lmk_id) == points_with_id.end());
    points_with_id[lmk_id] = key_value.value.cast<gtsam::Point3>();
    if (lmk_id_to_lmk_type_map) {
      (*lmk_id_to_lmk_type_map)[lmk_id] = LandmarkType::PROJECTION;
    }
    nr_proj_lmks++;
  }

  // TODO aren't these points post-optimization? Shouldn't we instead add
  // the points before optimization? Then the regularities we enforce will
  // have the most impact, otherwise the points in the optimization horizon
  // do not move that much after optimizing... they are almost frozen and
  // are not visually changing much...
  // They might actually not be changing that much because we are not
  // enforcing the regularities on the points that are out of current frame
  // in the backend currently...

  VLOG(10) << "Landmark typology to be used for the mesh:\n"
           << "Number of valid smart factors " << nr_valid_smart_lmks
           << " out of " << nr_smart_lmks << "\n"
           << "Number of landmarks (not involved in a smart factor) "
           << nr_proj_lmks << ".\n Total number of landmarks: "
           << (nr_valid_smart_lmks + nr_proj_lmks);
  return points_with_id;
}

}  // namespace okvis

