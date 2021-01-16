#include <msckf/HybridFilter.hpp>

#include <glog/logging.h>

#include <io_wrap/StreamHelper.hpp>

#include <okvis/assert_macros.hpp>
#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/RelativePoseError.hpp>
#include <okvis/ceres/SpeedAndBiasError.hpp>
#include <okvis/IdProvider.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/timing/Timer.hpp>
#include <okvis/triangulation/stereo_triangulation.hpp>

#include <msckf/CameraTimeParamBlock.hpp>
#include <msckf/EpipolarJacobian.hpp>
#include <msckf/EuclideanParamBlock.hpp>
#include <msckf/ExtrinsicModels.hpp>

#include <msckf/EkfUpdater.h>
#include <msckf/FilterHelper.hpp>
#include <msckf/MultipleTransformPointJacobian.hpp>
#include <msckf/imu/ImuOdometry.h>
#include <msckf/PointLandmark.hpp>
#include <msckf/PointLandmarkModels.hpp>
#include <msckf/PointSharedData.hpp>
#include <msckf/RelativeMotionJacobian.hpp>
#include <msckf/TwoViewPair.hpp>
#include <msckf/memory.h>

#include <vio/eigen_utils.h>

DEFINE_bool(use_RK4, false,
            "use 4th order runge-kutta or the trapezoidal "
            "rule for integrating IMU data and computing"
            " Jacobians");
DEFINE_bool(use_first_estimate, true,
            "use first estimate jacobians to compute covariance?");

DECLARE_bool(use_mahalanobis);

DEFINE_double(max_proj_tolerance, 7,
              "maximum tolerable discrepancy between predicted and measured "
              "point coordinates in image in pixel units");

DEFINE_double(image_noise_cov_multiplier, 5.0,
              "Enlarge the image observation noise covariance by this "
              "multiplier for epipolar constraints to weaken their effect.");

DEFINE_double(
    epipolar_sigma_keypoint_size, 2.0,
    "The keypoint size for checking low disparity when epipolar constraints "
    "are enabled. The bigger this value, the more likely a feature track is "
    "deemed to have low disparity and to contribute epipolar constraints. "
    "Lower this value from reference 8.0 to achieve the effect of "
    "fiber-reinforced concrete for MSCKF. 1.0 - 2.0 is recommended ");

/// \brief okvis Main namespace of this package.
namespace okvis {

// Constructor if a ceres map is already available.
HybridFilter::HybridFilter(std::shared_ptr<okvis::ceres::Map> mapPtr)
    : Estimator(mapPtr),
      triangulateTimer("3.1.1.1 triangulateAMapPoint", true),
      computeHTimer("3.1.1 featureJacobian", true),
      updateLandmarksTimer("3.1.5 updateLandmarks", true),
      mTrackLengthAccumulator(100, 0u),
      minCulledFrames_(3u) {
}

// The default constructor.
HybridFilter::HybridFilter()
    : Estimator(),
      triangulateTimer("3.1.1.1 triangulateAMapPoint", true),
      computeHTimer("3.1.1 featureJacobian", true),
      updateLandmarksTimer("3.1.5 updateLandmarks", true),
      mTrackLengthAccumulator(100, 0u),
      minCulledFrames_(3u) {
}

HybridFilter::~HybridFilter() {
  LOG(INFO) << "Destructing HybridFilter";
}

void HybridFilter::addImuAugmentedStates(const okvis::Time stateTime,
                                         int imu_id,
                                         SpecificSensorStatesContainer* imuInfo) {
  imuInfo->at(ImuSensorStates::TG).exists = true;
  imuInfo->at(ImuSensorStates::TS).exists = true;
  imuInfo->at(ImuSensorStates::TA).exists = true;
  // In MSCKF, use the same block for those parameters that are assumed
  // constant and updated in the filter
  if (statesMap_.size() > 1) {
    std::map<uint64_t, States>::reverse_iterator lastElementIterator =
        statesMap_.rbegin();
    lastElementIterator++;
    const SpecificSensorStatesContainer& prevImuInfo =
        lastElementIterator->second.sensors.at(SensorStates::Imu).at(imu_id);
    imuInfo->at(ImuSensorStates::TG).id =
        prevImuInfo.at(ImuSensorStates::TG).id;
    imuInfo->at(ImuSensorStates::TS).id =
        prevImuInfo.at(ImuSensorStates::TS).id;
    imuInfo->at(ImuSensorStates::TA).id =
        prevImuInfo.at(ImuSensorStates::TA).id;
    imuInfo->at(ImuSensorStates::SpeedAndBias).startIndexInCov =
        prevImuInfo.at(ImuSensorStates::SpeedAndBias).startIndexInCov;
    imuInfo->at(ImuSensorStates::TG).startIndexInCov =
        prevImuInfo.at(ImuSensorStates::TG).startIndexInCov;
    imuInfo->at(ImuSensorStates::TS).startIndexInCov =
        prevImuInfo.at(ImuSensorStates::TS).startIndexInCov;
    imuInfo->at(ImuSensorStates::TA).startIndexInCov =
        prevImuInfo.at(ImuSensorStates::TA).startIndexInCov;
  } else {
    Eigen::VectorXd imuAugmentedParams =
        imu_rig_.getImuAugmentedEuclideanParams();
    int imuModelId = imu_rig_.getModelId(0);
    switch (imuModelId) {
      case Imu_BG_BA_TG_TS_TA::kModelId: {
        Eigen::Matrix<double, 9, 1> TG = imuAugmentedParams.head<9>();
        uint64_t id = IdProvider::instance().newId();
        std::shared_ptr<ceres::ShapeMatrixParamBlock> tgBlockPtr(
            new ceres::ShapeMatrixParamBlock(TG, id, stateTime));
        mapPtr_->addParameterBlock(tgBlockPtr, ceres::Map::Trivial);
        imuInfo->at(ImuSensorStates::TG).id = id;

        const Eigen::Matrix<double, 9, 1> TS = imuAugmentedParams.segment<9>(9);
        id = IdProvider::instance().newId();
        std::shared_ptr<okvis::ceres::ShapeMatrixParamBlock> tsBlockPtr(
            new okvis::ceres::ShapeMatrixParamBlock(TS, id, stateTime));
        mapPtr_->addParameterBlock(tsBlockPtr, ceres::Map::Trivial);
        imuInfo->at(ImuSensorStates::TS).id = id;

        Eigen::Matrix<double, 9, 1> TA = imuAugmentedParams.tail<9>();
        id = IdProvider::instance().newId();
        std::shared_ptr<okvis::ceres::ShapeMatrixParamBlock> taBlockPtr(
            new okvis::ceres::ShapeMatrixParamBlock(TA, id, stateTime));
        mapPtr_->addParameterBlock(taBlockPtr, ceres::Map::Trivial);
        imuInfo->at(ImuSensorStates::TA).id = id;
      } break;
      case Imu_BG_BA::kModelId:
        break;
      default:
        LOG(WARNING) << "Adding parameter block not implemented for IMU model "
                     << imuModelId;
        break;
    }
    // The startIndex in covariance will be initialized along with covariance.
  }
}

void HybridFilter::usePreviousCameraParamBlocks(
    std::map<uint64_t, States>::const_reverse_iterator prevStateRevIter,
    size_t cameraIndex, SpecificSensorStatesContainer* cameraInfos) const {
  // use the same block...
  // the following will point to the last states:
  const SpecificSensorStatesContainer& prevCameraInfo =
      prevStateRevIter->second.sensors.at(SensorStates::Camera).at(cameraIndex);
  cameraInfos->at(CameraSensorStates::T_SCi).id =
      prevCameraInfo.at(CameraSensorStates::T_SCi).id;
  cameraInfos->at(CameraSensorStates::Intrinsics).exists =
      prevCameraInfo.at(CameraSensorStates::Intrinsics).exists;
  cameraInfos->at(CameraSensorStates::Intrinsics).id =
      prevCameraInfo.at(CameraSensorStates::Intrinsics).id;
  cameraInfos->at(CameraSensorStates::Distortion).id =
      prevCameraInfo.at(CameraSensorStates::Distortion).id;
  cameraInfos->at(CameraSensorStates::TD).id =
      prevCameraInfo.at(CameraSensorStates::TD).id;
  cameraInfos->at(CameraSensorStates::TR).id =
      prevCameraInfo.at(CameraSensorStates::TR).id;

  cameraInfos->at(CameraSensorStates::T_SCi).startIndexInCov =
      prevCameraInfo.at(CameraSensorStates::T_SCi).startIndexInCov;
  cameraInfos->at(CameraSensorStates::Intrinsics).startIndexInCov =
      prevCameraInfo.at(CameraSensorStates::Intrinsics).startIndexInCov;
  cameraInfos->at(CameraSensorStates::Distortion).startIndexInCov =
      prevCameraInfo.at(CameraSensorStates::Distortion).startIndexInCov;
  cameraInfos->at(CameraSensorStates::TD).startIndexInCov =
      prevCameraInfo.at(CameraSensorStates::TD).startIndexInCov;
  cameraInfos->at(CameraSensorStates::TR).startIndexInCov =
      prevCameraInfo.at(CameraSensorStates::TR).startIndexInCov;
}

void HybridFilter::initializeCameraParamBlocks(
    okvis::Time stateEpoch, size_t cameraIndex,
    SpecificSensorStatesContainer* cameraInfos) {
  const okvis::kinematics::Transformation T_BC =
      camera_rig_.getCameraExtrinsic(cameraIndex);
  const okvis::kinematics::Transformation T_BC0 =
      camera_rig_.getCameraExtrinsic(kMainCameraIndex);
  uint64_t id = IdProvider::instance().newId();
  std::shared_ptr<okvis::ceres::PoseParameterBlock> extrinsicsParameterBlockPtr;
  switch (camera_rig_.getExtrinsicOptMode(cameraIndex)) {
    case Extrinsic_p_CB::kModelId:
    case Extrinsic_p_BC_q_BC::kModelId:
      extrinsicsParameterBlockPtr.reset(
          new okvis::ceres::PoseParameterBlock(T_BC, id, stateEpoch));
      break;
    case Extrinsic_p_C0C_q_C0C::kModelId:
      extrinsicsParameterBlockPtr.reset(new okvis::ceres::PoseParameterBlock(
          T_BC0.inverse() * T_BC, id, stateEpoch));
      break;
  }
  mapPtr_->addParameterBlock(extrinsicsParameterBlockPtr, ceres::Map::Pose6d);
  cameraInfos->at(CameraSensorStates::T_SCi).id = id;

  Eigen::VectorXd allIntrinsics;
  camera_rig_.getCameraGeometry(cameraIndex)->getIntrinsics(allIntrinsics);
  id = IdProvider::instance().newId();
  int projOptModelId = camera_rig_.getProjectionOptMode(cameraIndex);
  const int minProjectionDim =
      camera_rig_.getMinimalProjectionDimen(cameraIndex);
  if (!fixCameraIntrinsicParams_[cameraIndex]) {
    Eigen::VectorXd optProjIntrinsics;
    ProjectionOptGlobalToLocal(projOptModelId, allIntrinsics,
                               &optProjIntrinsics);
    std::shared_ptr<okvis::ceres::EuclideanParamBlock>
        projIntrinsicParamBlockPtr(new okvis::ceres::EuclideanParamBlock(
            optProjIntrinsics, id, stateEpoch, minProjectionDim));
    mapPtr_->addParameterBlock(projIntrinsicParamBlockPtr,
                               ceres::Map::Parameterization::Trivial);
    cameraInfos->at(CameraSensorStates::Intrinsics).id = id;
  } else {
    Eigen::VectorXd optProjIntrinsics = allIntrinsics.head<4>();
    std::shared_ptr<okvis::ceres::EuclideanParamBlock>
        projIntrinsicParamBlockPtr(new okvis::ceres::EuclideanParamBlock(
            optProjIntrinsics, id, stateEpoch, 4));
    mapPtr_->addParameterBlock(projIntrinsicParamBlockPtr,
                               ceres::Map::Parameterization::Trivial);
    cameraInfos->at(CameraSensorStates::Intrinsics).id = id;
    mapPtr_->setParameterBlockConstant(id);
  }
  id = IdProvider::instance().newId();
  const int distortionDim = camera_rig_.getDistortionDimen(cameraIndex);
  std::shared_ptr<okvis::ceres::EuclideanParamBlock> distortionParamBlockPtr(
      new okvis::ceres::EuclideanParamBlock(allIntrinsics.tail(distortionDim),
                                            id, stateEpoch,
                                            distortionDim));
  mapPtr_->addParameterBlock(distortionParamBlockPtr,
                             ceres::Map::Parameterization::Trivial);
  cameraInfos->at(CameraSensorStates::Distortion).id = id;

  id = IdProvider::instance().newId();
  std::shared_ptr<okvis::ceres::CameraTimeParamBlock> tdParamBlockPtr(
      new okvis::ceres::CameraTimeParamBlock(camera_rig_.getImageDelay(cameraIndex), id,
                                             stateEpoch));
  mapPtr_->addParameterBlock(tdParamBlockPtr,
                             ceres::Map::Parameterization::Trivial);
  cameraInfos->at(CameraSensorStates::TD).id = id;

  id = IdProvider::instance().newId();
  std::shared_ptr<okvis::ceres::CameraTimeParamBlock> trParamBlockPtr(
      new okvis::ceres::CameraTimeParamBlock(camera_rig_.getReadoutTime(cameraIndex), id,
                                             stateEpoch));
  mapPtr_->addParameterBlock(trParamBlockPtr,
                             ceres::Map::Parameterization::Trivial);
  cameraInfos->at(CameraSensorStates::TR).id = id;
  // The startIndex in covariance will be initialized along with covariance.
}

void HybridFilter::addCameraSystem(const okvis::cameras::NCameraSystem &cameras) {
  Estimator::addCameraSystem(cameras);
  minCulledFrames_ = 4u > camera_rig_.numberCameras() ? 4u - camera_rig_.numberCameras() : 1u;
}

bool HybridFilter::addStates(okvis::MultiFramePtr multiFrame,
                             const okvis::ImuMeasurementDeque& imuMeasurements,
                             bool asKeyframe) {
  // note: this is before matching...
  okvis::kinematics::Transformation T_WS;
  okvis::SpeedAndBiases speedAndBias;
  okvis::Duration tdEstimate;
  okvis::Time correctedStateTime;  // time of current multiFrame corrected with
                                   // current td estimate
  // record the imu measurements between two consecutive states
  inertialMeasForStates_.push_back(imuMeasurements);
  if (statesMap_.empty()) {
    // in case this is the first frame ever, let's initialize the pose:
    tdEstimate.fromSec(camera_rig_.getImageDelay(0));
    correctedStateTime = multiFrame->timestamp() + tdEstimate;

    if (pvstd_.initWithExternalSource) {
      T_WS = okvis::kinematics::Transformation(pvstd_.p_WS, pvstd_.q_WS);
    } else {
      bool success0 = initPoseFromImu(imuMeasurements, T_WS);
      OKVIS_ASSERT_TRUE_DBG(
          Exception, success0,
          "pose could not be initialized from imu measurements.");
      if (!success0) return false;
      pvstd_.updatePose(T_WS, correctedStateTime);
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

    uint64_t td_id = statesMap_.rbegin()
                         ->second.sensors.at(SensorStates::Camera)
                         .at(kMainCameraIndex)
                         .at(CameraSensorStates::TD)
                         .id;  // one camera assumption
    tdEstimate =
        okvis::Duration(std::static_pointer_cast<ceres::CameraTimeParamBlock>(
                            mapPtr_->parameterBlockPtr(td_id))
                            ->estimate());
    correctedStateTime = multiFrame->timestamp() + tdEstimate;

    Eigen::VectorXd imuAugmentedParams =
        imu_rig_.getImuAugmentedEuclideanParams(0);

    // propagate pose, speedAndBias, and covariance
    okvis::Time startTime = statesMap_.rbegin()->second.timestamp;
    size_t navAndImuParamsDim = navStateAndImuParamsMinimalDim();
    Eigen::MatrixXd Pkm1 =
        covariance_.topLeftCorner(navAndImuParamsDim, navAndImuParamsDim);
    Eigen::MatrixXd F_tot = Eigen::MatrixXd::Identity(navAndImuParamsDim, navAndImuParamsDim);
    int numUsedImuMeasurements = -1;
    okvis::Time latestImuEpoch = imuMeasurements.back().timeStamp;
    okvis::Time propagationTargetTime = correctedStateTime;
    if (latestImuEpoch < correctedStateTime) {
      propagationTargetTime = latestImuEpoch;
      LOG(WARNING) << "Latest IMU readings does not extend to corrected state "
                      "time. Is temporal_imu_data_overlap too small?";
    }
    ImuErrorModel<double> iem(speedAndBias.tail<6>(), imuAugmentedParams);
    if (FLAGS_use_first_estimate) {
      /// use latest estimate to propagate pose, speed and bias, and first
      /// estimate to propagate covariance and Jacobian
      std::shared_ptr<const Eigen::Matrix<double, 6, 1>> lP =
          statesMap_.rbegin()->second.linearizationPoint;
      Eigen::Vector3d v_WS = speedAndBias.head<3>();
      numUsedImuMeasurements = ImuOdometry::propagation(
          imuMeasurements, imuParametersVec_.at(0), T_WS, v_WS, iem, startTime,
          propagationTargetTime, &Pkm1, &F_tot, lP.get());
      speedAndBias.head<3>() = v_WS;
    } else {
      /// use latest estimate to propagate pose, speed and bias, and covariance
      if (FLAGS_use_RK4) {
        // method 1 RK4 a little bit more accurate but 4 times slower
        numUsedImuMeasurements = ImuOdometry::propagation_RungeKutta(
            imuMeasurements, imuParametersVec_.at(0), T_WS, speedAndBias, iem,
            startTime, propagationTargetTime, &Pkm1, &F_tot);
      } else {
        // method 2, i.e., adapt the imuError::propagation function of okvis by
        // the msckf derivation in Michael Andrew Shelley
        Eigen::Vector3d v_WS = speedAndBias.head<3>();
        numUsedImuMeasurements = ImuOdometry::propagation(
            imuMeasurements, imuParametersVec_.at(0), T_WS, v_WS, iem,
            startTime, propagationTargetTime, &Pkm1, &F_tot);
        speedAndBias.head<3>() = v_WS;
      }
    }
    if (numUsedImuMeasurements < 2) {
      LOG(WARNING) << "numUsedImuMeasurements=" << numUsedImuMeasurements
                   << " correctedStateTime " << correctedStateTime
                   << " lastFrameTimestamp " << startTime << " tdEstimate "
                   << tdEstimate << std::endl;
    }
    okvis::Time secondLatestStateTime = statesMap_.rbegin()->second.timestamp;
    auto imuMeasCoverSecond =
        inertialMeasForStates_.findWindow(secondLatestStateTime, half_window_);
    statesMap_.rbegin()->second.imuReadingWindow.reset(
        new okvis::ImuMeasurementDeque(imuMeasCoverSecond));

    int covDim = covariance_.rows();
    covariance_.topLeftCorner(navAndImuParamsDim, navAndImuParamsDim) = Pkm1;
    covariance_.block(0, navAndImuParamsDim, navAndImuParamsDim,
                      covDim - navAndImuParamsDim) =
        F_tot * covariance_.block(0, navAndImuParamsDim, navAndImuParamsDim,
                                  covDim - navAndImuParamsDim);
    covariance_
        .block(navAndImuParamsDim, 0, covDim - navAndImuParamsDim,
               navAndImuParamsDim)
        .noalias() = covariance_
                         .block(0, navAndImuParamsDim, navAndImuParamsDim,
                                covDim - navAndImuParamsDim)
                         .transpose();
  }

  // create a states object:
  States states(asKeyframe, multiFrame->id(), correctedStateTime, tdEstimate.toSec());

  // check if id was used before
  OKVIS_ASSERT_TRUE_DBG(Exception,
                        statesMap_.find(states.id) == statesMap_.end(),
                        "pose ID" << states.id << " was used before!");

  // create global states
  std::shared_ptr<okvis::ceres::PoseParameterBlock> poseParameterBlock(
      new okvis::ceres::PoseParameterBlock(T_WS, states.id,
                                           correctedStateTime));
  states.global.at(GlobalStates::T_WS).exists = true;
  states.global.at(GlobalStates::T_WS).id = states.id;
  // set first estimates
  states.linearizationPoint.reset(new Eigen::Matrix<double, 6, 1>());
  (*states.linearizationPoint) << T_WS.r(), speedAndBias.head<3>();
  auto imuMeasCover = inertialMeasForStates_.findWindow(correctedStateTime, half_window_);
  states.imuReadingWindow.reset(new okvis::ImuMeasurementDeque(imuMeasCover));
  if (statesMap_.empty()) {
    referencePoseId_ = states.id;  // set this as reference pose
  }
  mapPtr_->addParameterBlock(poseParameterBlock, ceres::Map::Pose6d);
  // add to buffer
  statesMap_.insert(std::pair<uint64_t, States>(states.id, states));
  multiFramePtrMap_.insert(
      std::pair<uint64_t, okvis::MultiFramePtr>(states.id, multiFrame));

  OKVIS_ASSERT_EQ_DBG(Exception, imuParametersVec_.size(), 1,
                      "Only one IMU is supported.");
  // initialize new sensor states
  // cameras:
  for (size_t i = 0; i < extrinsicsEstimationParametersVec_.size(); ++i) {
    SpecificSensorStatesContainer cameraInfos(5);
    cameraInfos.at(CameraSensorStates::T_SCi).exists = true;
    cameraInfos.at(CameraSensorStates::Intrinsics).exists = true;
    cameraInfos.at(CameraSensorStates::Distortion).exists = true;
    cameraInfos.at(CameraSensorStates::TD).exists = true;
    cameraInfos.at(CameraSensorStates::TR).exists = true;
    // In MSCKF, use the same block for those parameters that are assumed
    // constant and updated in the filter
    if (statesMap_.size() > 1) {
      std::map<uint64_t, States>::const_reverse_iterator lastElementIterator =
          statesMap_.rbegin();
      lastElementIterator++;
      usePreviousCameraParamBlocks(lastElementIterator, i, &cameraInfos);
    } else {
      initializeCameraParamBlocks(correctedStateTime, i, &cameraInfos);
    }
    // update the info in both copies of states
    statesMap_.rbegin()
        ->second.sensors.at(SensorStates::Camera)
        .push_back(cameraInfos);
    states.sensors.at(SensorStates::Camera).push_back(cameraInfos);
  }

  // IMU states are automatically propagated.
  for (size_t i = 0; i < imuParametersVec_.size(); ++i) {
    SpecificSensorStatesContainer imuInfo(4);
    imuInfo.at(ImuSensorStates::SpeedAndBias).exists = true;

    uint64_t id = IdProvider::instance().newId();
    std::shared_ptr<okvis::ceres::SpeedAndBiasParameterBlock>
        speedAndBiasParameterBlock(new okvis::ceres::SpeedAndBiasParameterBlock(
            speedAndBias, id, correctedStateTime));

    mapPtr_->addParameterBlock(speedAndBiasParameterBlock);
    imuInfo.at(ImuSensorStates::SpeedAndBias).id = id;

    addImuAugmentedStates(correctedStateTime, i, &imuInfo);

    statesMap_.rbegin()
        ->second.sensors.at(SensorStates::Imu)
        .push_back(imuInfo);
    states.sensors.at(SensorStates::Imu).push_back(imuInfo);
  }

  // depending on whether or not this is the very beginning, we will construct
  // covariance
  if (statesMap_.size() == 1) {
    initCovariance();
  }

  addCovForClonedStates();
  return true;
}

void HybridFilter::initCovariance() {
  int covDim = startIndexOfClonedStates();
  Eigen::Matrix<double, 6, 6> covPQ =
      Eigen::Matrix<double, 6, 6>::Zero();  // [\delta p_B^G, \delta \theta]

  covPQ.topLeftCorner<3, 3>() = pvstd_.std_p_WS.cwiseAbs2().asDiagonal();
  covPQ.bottomRightCorner<3, 3>() = pvstd_.std_q_WS.cwiseAbs2().asDiagonal();

  Eigen::Matrix<double, 9, 9> covSB =
      Eigen::Matrix<double, 9, 9>::Zero();  // $v_B^G, b_g, b_a$

  for (size_t i = 0; i < imuParametersVec_.size(); ++i) {
    // get these from parameter file
    const double sigma_bg = imuParametersVec_.at(0).sigma_bg;
    const double sigma_ba = imuParametersVec_.at(0).sigma_ba;
    const double gyrBiasVariance = sigma_bg * sigma_bg,
                 accBiasVariance = sigma_ba * sigma_ba;

    covSB.topLeftCorner<3, 3>() = pvstd_.std_v_WS.cwiseAbs2().asDiagonal();
    covSB.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * gyrBiasVariance;
    covSB.bottomRightCorner<3, 3>() =
        Eigen::Matrix3d::Identity() * accBiasVariance;
  }

  SpecificSensorStatesContainer& imuInfo =
      statesMap_.rbegin()->second.sensors.at(SensorStates::Imu).at(0u);

  covariance_ = Eigen::MatrixXd::Zero(covDim, covDim);
  covariance_.topLeftCorner<6, 6>() = covPQ;
  covariance_.block<9, 9>(6, 6) = covSB;
  imuInfo.at(ImuSensorStates::SpeedAndBias).startIndexInCov = 6u;
  int imuModelId = imu_rig_.getModelId(0);

  switch (imuModelId) {
    case Imu_BG_BA_TG_TS_TA::kModelId: {
      Eigen::Matrix<double, 27, 27> covTGTSTA =
          Eigen::Matrix<double, 27, 27>::Zero();
      const double sigmaTGElement = imuParametersVec_.at(0).sigma_TGElement;
      const double sigmaTSElement = imuParametersVec_.at(0).sigma_TSElement;
      const double sigmaTAElement = imuParametersVec_.at(0).sigma_TAElement;
      covTGTSTA.topLeftCorner<9, 9>() =
          Eigen::Matrix<double, 9, 9>::Identity() * std::pow(sigmaTGElement, 2);
      covTGTSTA.block<9, 9>(9, 9) =
          Eigen::Matrix<double, 9, 9>::Identity() * std::pow(sigmaTSElement, 2);
      covTGTSTA.block<9, 9>(18, 18) =
          Eigen::Matrix<double, 9, 9>::Identity() * std::pow(sigmaTAElement, 2);
      covariance_.block<27, 27>(15, 15) = covTGTSTA;
      imuInfo.at(ImuSensorStates::TG).startIndexInCov = 15u;
      imuInfo.at(ImuSensorStates::TS).startIndexInCov = 24u;
      imuInfo.at(ImuSensorStates::TA).startIndexInCov = 33u;
    } break;
    case Imu_BG_BA::kModelId:
      imuInfo.at(ImuSensorStates::TG).startIndexInCov = 15u;
      imuInfo.at(ImuSensorStates::TS).startIndexInCov = 15u;
      imuInfo.at(ImuSensorStates::TA).startIndexInCov = 15u;
      break;
    default:
      LOG(WARNING) << "Not implemented IMU model " << imuModelId;
      break;
  }

  for (size_t j = 0u; j < camera_rig_.numberCameras(); ++j) {
    initCameraParamCovariance(j);
  }
}

void HybridFilter::initCameraParamCovariance(int camIdx) {
  // camera sensor states
  int camParamIndex = startIndexOfCameraParams(camIdx);
  int minExtrinsicDim = camera_rig_.getMinimalExtrinsicDimen(camIdx);
  int minProjectionDim = camera_rig_.getMinimalProjectionDimen(camIdx);
  int distortionDim = camera_rig_.getDistortionDimen(camIdx);

  Eigen::MatrixXd covExtrinsic;
  Eigen::MatrixXd covProjIntrinsics;
  Eigen::MatrixXd covDistortion;
  Eigen::Matrix2d covTDTR;

  SpecificSensorStatesContainer& cameraInfos =
      statesMap_.rbegin()->second.sensors.at(SensorStates::Camera).at(camIdx);

  cameraInfos.at(CameraSensorStates::T_SCi).startIndexInCov = camParamIndex;
  if (!fixCameraExtrinsicParams_[camIdx]) {
    covExtrinsic =
        ExtrinsicModelInitCov(camera_rig_.getExtrinsicOptMode(camIdx),
                              extrinsicsEstimationParametersVec_.at(camIdx)
                                  .sigma_absolute_translation,
                              extrinsicsEstimationParametersVec_.at(camIdx)
                                  .sigma_absolute_orientation);
    covariance_.block(camParamIndex, camParamIndex, minExtrinsicDim,
                      minExtrinsicDim) = covExtrinsic;
    camParamIndex += minExtrinsicDim;
  }
  cameraInfos.at(CameraSensorStates::Intrinsics).startIndexInCov = camParamIndex;
  if (!fixCameraIntrinsicParams_[camIdx]) {
    covProjIntrinsics = ProjectionModelGetInitCov(
        camera_rig_.getProjectionOptMode(camIdx),
        extrinsicsEstimationParametersVec_.at(camIdx).sigma_focal_length,
        extrinsicsEstimationParametersVec_.at(camIdx).sigma_principal_point);

    covDistortion = Eigen::MatrixXd::Identity(distortionDim, distortionDim);
    for (int jack = 0; jack < distortionDim; ++jack)
      covDistortion(jack, jack) *= std::pow(
          extrinsicsEstimationParametersVec_.at(camIdx).sigma_distortion[jack],
          2);

    covariance_.block(camParamIndex, camParamIndex, minProjectionDim,
                      minProjectionDim) = covProjIntrinsics;
    camParamIndex += minProjectionDim;
    cameraInfos.at(CameraSensorStates::Distortion).startIndexInCov = camParamIndex;
    covariance_.block(camParamIndex, camParamIndex, distortionDim,
                      distortionDim) = covDistortion;
    camParamIndex += distortionDim;
  } else {
    cameraInfos.at(CameraSensorStates::Distortion).startIndexInCov = camParamIndex;
  }
  cameraInfos.at(CameraSensorStates::TD).startIndexInCov = camParamIndex;
  cameraInfos.at(CameraSensorStates::TR).startIndexInCov = camParamIndex + 1;
  covTDTR = Eigen::Matrix2d::Identity();
  covTDTR(0, 0) *=
      std::pow(extrinsicsEstimationParametersVec_.at(camIdx).sigma_td, 2);
  covTDTR(1, 1) *=
      std::pow(extrinsicsEstimationParametersVec_.at(camIdx).sigma_tr, 2);
  covariance_.block<2, 2>(camParamIndex, camParamIndex) = covTDTR;
}

void HybridFilter::addCovForClonedStates() {
  // augment states in the propagated covariance matrix
  int oldCovDim = covariance_.rows();
  const size_t numPointStates = 3 * mInCovLmIds.size();
  const size_t numOldNavImuCamPoseStates = oldCovDim - numPointStates;
  statesMap_.rbegin()->second.global.at(GlobalStates::T_WS).startIndexInCov =
      numOldNavImuCamPoseStates;

  size_t covDimAugmented = oldCovDim + 9;  //$\delta p,\delta \alpha,\delta v$
  Eigen::MatrixXd covarianceAugmented(covDimAugmented, covDimAugmented);
  covarianceAugmented.topLeftCorner(numOldNavImuCamPoseStates,
                                    numOldNavImuCamPoseStates) =
      covariance_.topLeftCorner(numOldNavImuCamPoseStates,
                                numOldNavImuCamPoseStates);

  covarianceAugmented.block(0, numOldNavImuCamPoseStates,
                            numOldNavImuCamPoseStates, 9) =
      covariance_.topLeftCorner(numOldNavImuCamPoseStates, 9);

  if (numPointStates > 0) {
    covarianceAugmented.topRightCorner(numOldNavImuCamPoseStates,
                                       numPointStates) =
        covariance_.topRightCorner(numOldNavImuCamPoseStates, numPointStates);

    covarianceAugmented.bottomLeftCorner(numPointStates,
                                         numOldNavImuCamPoseStates) =
        covariance_.bottomLeftCorner(numPointStates, numOldNavImuCamPoseStates);

    covarianceAugmented.block(numOldNavImuCamPoseStates + 9,
                              numOldNavImuCamPoseStates, numPointStates, 9) =
        covariance_.bottomLeftCorner(numPointStates, 9);

    covarianceAugmented.bottomRightCorner(numPointStates, numPointStates) =
        covariance_.bottomRightCorner(numPointStates, numPointStates);
  }

  covarianceAugmented.block(numOldNavImuCamPoseStates, 0, 9, covDimAugmented) =
      covarianceAugmented.topLeftCorner(9, covDimAugmented);

  covariance_ = covarianceAugmented;
}

void HybridFilter::findRedundantCamStates(
    std::vector<uint64_t>* rm_cam_state_ids,
    size_t numImuFrames) {
  int closeFrames(0), oldFrames(0);
  rm_cam_state_ids->clear();
  rm_cam_state_ids->reserve(minCulledFrames_);
  auto rit = statesMap_.rbegin();
  for (size_t j = 0; j < numImuFrames; ++j) {
    ++rit;
  }
  for (; rit != statesMap_.rend(); ++rit) {
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

int HybridFilter::marginalizeRedundantFrames(size_t numKeyframes, size_t numImuFrames) {
  if (statesMap_.size() < numKeyframes + numImuFrames) {
    return 0;
  }
  std::vector<uint64_t> rm_cam_state_ids;
  findRedundantCamStates(&rm_cam_state_ids, numImuFrames);

  size_t nMarginalizedFeatures = 0u;
  int featureVariableDimen = minimalDimOfAllCameraParams() +
      kClonedStateMinimalDimen * (statesMap_.size() - 1);
  int navAndImuParamsDim = navStateAndImuParamsMinimalDim();
  int startIndexCamParams = startIndexOfCameraParamsFast(0u);
  const Eigen::MatrixXd featureVariableCov =
      covariance_.block(startIndexCamParams, startIndexCamParams,
                        featureVariableDimen, featureVariableDimen);
  int dimH_o[2] = {0, featureVariableDimen};
  // containers of Jacobians of measurements
  Eigen::AlignedVector<Eigen::Matrix<double, -1, 1>> vr_o;
  Eigen::AlignedVector<Eigen::MatrixXd> vH_o;
  Eigen::AlignedVector<Eigen::MatrixXd> vR_o;

  // for each map point in the landmarksMap_,
  // see if the landmark is observed in the redundant frames
  for (okvis::PointMap::iterator it = landmarksMap_.begin();
       it != landmarksMap_.end(); ++it) {
    if (!it->second.goodForMarginalization(minCulledFrames_)) {
      continue;
    }

    std::vector<uint64_t> involved_cam_state_ids;
    auto obsMap = it->second.observations;
    auto obsSearchStart = obsMap.begin();
    for (auto camStateId : rm_cam_state_ids) {
      auto obsIter = std::find_if(obsSearchStart, obsMap.end(),
                                  okvis::IsObservedInNFrame(camStateId));
      if (obsIter != obsMap.end()) {
        involved_cam_state_ids.emplace_back(camStateId);
        obsSearchStart = obsIter;
        ++obsSearchStart;
      }
    }
    if (involved_cam_state_ids.size() < minCulledFrames_) {
      continue;
    }

    msckf::PointLandmark landmark;
    Eigen::MatrixXd H_oi;                           //(nObsDim, dimH_o[1])
    Eigen::Matrix<double, Eigen::Dynamic, 1> r_oi;  //(nObsDim, 1)
    Eigen::MatrixXd R_oi;                           //(nObsDim, nObsDim)

    bool isValidJacobian =
        featureJacobian(it->second, &landmark, H_oi, r_oi, R_oi, nullptr, &involved_cam_state_ids);
    if (!isValidJacobian) {
      // Do we use epipolar constraints for the marginalized feature
      // observations when they do not exhibit enough disparity? It is probably
      // a overkill.
      continue;
    }

    if (!FilterHelper::gatingTest(H_oi, r_oi, R_oi, featureVariableCov)) {
      continue;
    }

    vr_o.push_back(r_oi);
    vR_o.push_back(R_oi);
    vH_o.push_back(H_oi);
    dimH_o[0] += r_oi.rows();
    ++nMarginalizedFeatures;
  }

  if (nMarginalizedFeatures > 0u) {
    Eigen::MatrixXd H_o =
        Eigen::MatrixXd::Zero(dimH_o[0], featureVariableDimen);
    Eigen::Matrix<double, -1, 1> r_o(dimH_o[0], 1);
    Eigen::MatrixXd R_o = Eigen::MatrixXd::Zero(dimH_o[0], dimH_o[0]);
    FilterHelper::stackJacobianAndResidual(vH_o, vr_o, vR_o, &H_o, &r_o, &R_o);
    Eigen::MatrixXd T_H, R_q;
    Eigen::Matrix<double, Eigen::Dynamic, 1> r_q;
    FilterHelper::shrinkResidual(H_o, r_o, R_o, &T_H, &r_q, &R_q);

    DefaultEkfUpdater updater(covariance_, navAndImuParamsDim, featureVariableDimen);
    computeKalmanGainTimer.start();
    Eigen::Matrix<double, Eigen::Dynamic, 1> deltaX =
        updater.computeCorrection(T_H, r_q, R_q);
    computeKalmanGainTimer.stop();
    updateStates(deltaX);

    updateCovarianceTimer.start();
    updater.updateCovariance(&covariance_);
    updateCovarianceTimer.stop();
  }

  // sanity check
  for (const auto &cam_id : rm_cam_state_ids) {
    int cam_sequence =
        std::distance(statesMap_.begin(), statesMap_.find(cam_id));
    OKVIS_ASSERT_EQ(Exception,
                    cam_sequence * kClonedStateMinimalDimen +
                        startIndexOfClonedStatesFast(),
                    statesMap_[cam_id].global.at(GlobalStates::T_WS).startIndexInCov,
                    "Inconsistent state order in covariance");
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
        // loop in case there are dud observations for the
        // landmark in the same frame.
        const KeypointIdentifier& kpi = obsIter->first;
        auto mfp = multiFramePtrMap_.find(kpi.frameId);
        mfp->second->setLandmarkId(kpi.cameraIndex, kpi.keypointIndex, 0);
        if (obsIter->second) {
          mapPtr_->removeResidualBlock(
              reinterpret_cast<::ceres::ResidualBlockId>(obsIter->second));
        } else {
//          if (obsIter == mapPoint.observations.begin()) {
            // this is a head obs for epipolar constraints, remove all of them
//            removeAllEpipolarConstraints = true;
//          }  // else do nothing. This can happen if we removed an epipolar
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
//                                  okvis::IsObservedInNFrame(camStateId));
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

  // change anchor for affected landmarks.
  changeAnchors(rm_cam_state_ids);

  for (const auto &cam_id : rm_cam_state_ids) {
    auto statesIter = statesMap_.find(cam_id);
    int cam_sequence =
        std::distance(statesMap_.begin(), statesIter);
    int cam_state_start =
        startIndexOfClonedStatesFast() + kClonedStateMinimalDimen * cam_sequence;
    int cam_state_end = cam_state_start + kClonedStateMinimalDimen;

    FilterHelper::pruneSquareMatrix(cam_state_start, cam_state_end,
                                    &covariance_);
    removeState(cam_id);
  }
  updateCovarianceIndex();

  inertialMeasForStates_.pop_front(statesMap_.begin()->second.timestamp - half_window_);
  return rm_cam_state_ids.size();
}

void HybridFilter::removeAnchorlessLandmarks(
    const std::vector<uint64_t> &sortedRemovedStateIds) {
  // a lazy approach to deal with landmarks that have lost their anchors is to
  // remove them.
  std::vector<uint64_t> toRemoveLmIds;
  toRemoveLmIds.reserve(10);
  for (auto landmark : mInCovLmIds) {
    MapPoint &mapPoint = landmarksMap_.at(landmark.id());
    uint64_t newAnchorFrameId =
        mapPoint.shouldChangeAnchor(sortedRemovedStateIds);
    if (newAnchorFrameId) {
      // remove the landmark's observations.
      std::map<okvis::KeypointIdentifier, uint64_t> &observationList =
          mapPoint.observations;
      for (auto iter = observationList.begin(); iter != observationList.end();
           ++iter) {
        if (iter->second) {
          mapPtr_->removeResidualBlock(
              reinterpret_cast<::ceres::ResidualBlockId>(iter->second));
        }
        const KeypointIdentifier &kpi = iter->first;
        auto mfp = multiFramePtrMap_.find(kpi.frameId);
        mfp->second->setLandmarkId(kpi.cameraIndex, kpi.keypointIndex, 0);
      }
      mapPtr_->removeParameterBlock(landmark.id());
      landmarksMap_.erase(landmark.id());
      toRemoveLmIds.push_back(landmark.id());
    }
  }

  // remove the covariance entries and state variables.
  decimateCovarianceForLandmarks(toRemoveLmIds);
}

void HybridFilter::changeAnchors(const std::vector<uint64_t>& sortedRemovedStateIds) {
  int covDim = covariance_.rows();
  const size_t numNavImuCamStates = startIndexOfClonedStatesFast();
  const size_t numNavImuCamPoseStates =
      numNavImuCamStates + 9 * statesMap_.size();
  Eigen::Matrix<double, 3, -1> reparamJacobian(3, covDim); // Jacobians of feature reparameterization due to anchor change.
  Eigen::AlignedVector<Eigen::Matrix<double, 3, -1>> vJacobian;  // container of these reparameterizing Jacobians.
  vJacobian.reserve(10);
  std::vector<size_t> vCovPtId;  // id in covariance of point features to be reparameterized, 0 for the first landmark.
  vCovPtId.reserve(10);
  for (auto landmark : mInCovLmIds) {
    MapPoint& mapPoint = landmarksMap_.at(landmark.id());
    uint64_t newAnchorFrameId = mapPoint.shouldChangeAnchor(sortedRemovedStateIds);
    if (newAnchorFrameId) {
      // transform from the body frame at the anchor frame epoch to the world
      // frame.
      okvis::kinematics::Transformation T_WBa;
      get_T_WS(mapPoint.anchorStateId, T_WBa);
      okvis::kinematics::Transformation T_BCa;
      getCameraSensorStates(newAnchorFrameId, mapPoint.anchorCameraId, T_BCa);
      okvis::kinematics::Transformation T_WCa = T_WBa * T_BCa;

      // use the camera with the minimum index as the anchor camera.
      int newAnchorCameraId = -1;
      for (auto observationIter = mapPoint.observations.rbegin();
           observationIter != mapPoint.observations.rend(); ++observationIter) {
        if (observationIter->first.frameId == newAnchorFrameId) {
          newAnchorCameraId = observationIter->first.cameraIndex;
        } else {
          break;
        }
      }
      OKVIS_ASSERT_NE(Exception, newAnchorCameraId, -1,
                      "Anchor image not found!");
      okvis::kinematics::Transformation T_WBj;
      get_T_WS(newAnchorFrameId, T_WBj);
      okvis::kinematics::Transformation T_BCj;
      getCameraSensorStates(newAnchorFrameId, newAnchorCameraId, T_BCj);
      okvis::kinematics::Transformation T_WCj = T_WBj * T_BCj;

      uint64_t toFind = mapPoint.id;
      Eigen::AlignedDeque<
          okvis::ceres::HomogeneousPointParameterBlock>::iterator landmarkIter =
          std::find_if(
              mInCovLmIds.begin(), mInCovLmIds.end(),
              [toFind](const okvis::ceres::HomogeneousPointParameterBlock &x) {
                return x.id() == toFind;
              });
      OKVIS_ASSERT_TRUE(Exception, landmarkIter != mInCovLmIds.end(),
                        "The tracked landmark is not in mInCovLmIds ");

      // update covariance matrix
      OKVIS_ASSERT_EQ(Exception, pointLandmarkOptions_.landmarkModelId,
                      msckf::InverseDepthParameterization::kModelId,
                      "Only inverse depth parameterization is supported for "
                      "reparameterization!");
      Eigen::Vector4d ab1rho = landmarkIter->estimate();
      Eigen::Vector3d abrhoi(ab1rho[0], ab1rho[1], ab1rho[3]);
      Eigen::Vector3d abrhoj;
      Eigen::Matrix<double, 3, 9> jacobian;
      vio::reparameterize_AIDP(T_WCa.C(), T_WCj.C(), abrhoi, T_WCa.r(),
                               T_WCj.r(), abrhoj, &jacobian);

      reparamJacobian.setZero();
      size_t startRowC = statesMap_[newAnchorFrameId]
                             .global.at(GlobalStates::T_WS)
                             .startIndexInCov;
      size_t startRowA = statesMap_[mapPoint.anchorStateId]
                             .global.at(GlobalStates::T_WS)
                             .startIndexInCov;

      reparamJacobian.block<3, 3>(0, startRowA) = jacobian.block<3, 3>(0, 3);
      reparamJacobian.block<3, 3>(0, startRowC) = jacobian.block<3, 3>(0, 6);

      size_t covPtId = std::distance(mInCovLmIds.begin(), landmarkIter);
      vCovPtId.push_back(covPtId);
      reparamJacobian.block<3, 3>(0, numNavImuCamPoseStates + 3 * covPtId) =
          jacobian.topLeftCorner<3, 3>();
      vJacobian.push_back(reparamJacobian);

      ab1rho = T_WCj.inverse() * T_WCa * ab1rho;
      ab1rho /= ab1rho[2];
      landmarkIter->setEstimate(ab1rho);

      mapPoint.anchorStateId = newAnchorFrameId;
      mapPoint.anchorCameraId = newAnchorCameraId;
    }
  }

  // update covariance for reparameterized landmarks.
  if (vJacobian.size()) {
    int landmarkIndex = 0;
    Eigen::MatrixXd featureJacMat = Eigen::MatrixXd::Identity(
        covDim, covDim); // Jacobian of all the new states w.r.t the old states
    for (auto it = vJacobian.begin(); it != vJacobian.end();
         ++it, ++landmarkIndex) {
      featureJacMat.block(numNavImuCamPoseStates + vCovPtId[landmarkIndex] * 3,
                          0, 3, covDim) = vJacobian[landmarkIndex];
    }
    covariance_ =
        (featureJacMat * covariance_).eval() * featureJacMat.transpose();
  }
}

bool HybridFilter::applyMarginalizationStrategy(
    size_t numKeyframes, size_t numImuFrames,
    okvis::MapPointVector& removedLandmarks) {
  marginalizeRedundantFrames(numKeyframes, numImuFrames);

  // remove features no longer tracked which can be in or out of the state vector.
  std::vector<uint64_t> toRemoveLmIds;
  toRemoveLmIds.reserve(10);
  for (PointMap::iterator pit = landmarksMap_.begin();
       pit != landmarksMap_.end();) {
    FeatureTrackStatus status = pit->second.status;
    if (pit->second.shouldRemove(pointLandmarkOptions_.maxHibernationFrames)) {
      std::map<okvis::KeypointIdentifier, uint64_t> &observationList =
          pit->second.observations;
      for (auto iter = observationList.begin(); iter != observationList.end();
           ++iter) {
        if (iter->second) {
          mapPtr_->removeResidualBlock(
              reinterpret_cast<::ceres::ResidualBlockId>(iter->second));
        }
        const KeypointIdentifier &kpi = iter->first;
        auto mfp = multiFramePtrMap_.find(kpi.frameId);
        mfp->second->setLandmarkId(kpi.cameraIndex, kpi.keypointIndex, 0);
      }

      if (status.inState) {
        toRemoveLmIds.push_back(pit->first);
      }

      mapPtr_->removeParameterBlock(pit->first);
      removedLandmarks.push_back(pit->second);
      pit = landmarksMap_.erase(pit);
    } else {
      ++pit;
    }
  }

  decimateCovarianceForLandmarks(toRemoveLmIds);
  return true;
}

void HybridFilter::decimateCovarianceForLandmarks(const std::vector<uint64_t>& toRemoveLmIds) {
  // decimate covariance for landmarks to be removed from state.
  if (toRemoveLmIds.size() == 0u)
    return;
  const size_t numNavImuCamStates = startIndexOfClonedStatesFast();
  const size_t numNavImuCamPoseStates = numNavImuCamStates + 9 * statesMap_.size();

  std::vector<size_t> toRemoveIndices;  // start indices of removed columns,
                                        // each interval of size 3
  toRemoveIndices.reserve(toRemoveLmIds.size());
  int covDim = covariance_.rows();
  for (auto it = toRemoveLmIds.begin(), itEnd = toRemoveLmIds.end();
       it != itEnd; ++it) {
    uint64_t toFind = *it;
    auto idPos = std::find_if(
        mInCovLmIds.begin(), mInCovLmIds.end(),
        [toFind](const okvis::ceres::HomogeneousPointParameterBlock &x) {
          return x.id() == toFind;
        });
    OKVIS_ASSERT_TRUE(Exception, idPos != mInCovLmIds.end(),
                      "The landmark in state is not in mInCovLmIds!");

    int startIndex = numNavImuCamPoseStates + 3 * std::distance(mInCovLmIds.begin(), idPos);
    toRemoveIndices.push_back(startIndex);
  }
  std::sort(toRemoveIndices.begin(), toRemoveIndices.end());
  std::vector<std::pair<size_t, size_t>> vRowStartInterval;
  vRowStartInterval.reserve(toRemoveLmIds.size() + 1);
  size_t startKeptRow = 0;  // start index of the kept rows.
  for (auto it = toRemoveIndices.begin(), itEnd = toRemoveIndices.end();
       it != itEnd; ++it) {
    vRowStartInterval.push_back(
        std::make_pair(startKeptRow, *it - startKeptRow));
    startKeptRow = *it + 3;
  }
  if (startKeptRow != (size_t)covDim) {
    vRowStartInterval.push_back(
        std::make_pair(startKeptRow, (size_t)covDim - startKeptRow));
  }
  covariance_ =
      vio::extractBlocks(covariance_, vRowStartInterval, vRowStartInterval);

  for (auto it = toRemoveLmIds.begin(), itEnd = toRemoveLmIds.end();
       it != itEnd; ++it) {
    uint64_t toFind = *it;
    auto idPos = std::find_if(
        mInCovLmIds.begin(), mInCovLmIds.end(),
        [toFind](const okvis::ceres::HomogeneousPointParameterBlock &x) {
          return x.id() == toFind;
        });
    mInCovLmIds.erase(idPos);
  }
}

void HybridFilter::updateImuRig() {
  Eigen::VectorXd extraParams;
  getImuAugmentedStatesEstimate(&extraParams);
  imu_rig_.setImuAugmentedEuclideanParams(0, extraParams);
}

void HybridFilter::updateCovarianceIndex() {
  size_t nCovIndex = startIndexOfClonedStatesFast();
  for (std::map<uint64_t, States, std::less<uint64_t>,
                Eigen::aligned_allocator<std::pair<const uint64_t, States>>>::
           iterator iter = statesMap_.begin();
       iter != statesMap_.end(); ++iter) {
    iter->second.global.at(GlobalStates::T_WS).startIndexInCov = nCovIndex;
    nCovIndex += kClonedStateMinimalDimen;
  }
}

void HybridFilter::updateSensorRigs() {
  size_t numCameras = camera_rig_.numberCameras();
  const uint64_t currFrameId = currentFrameId();
  okvis::kinematics::Transformation T_BC0;
  getCameraSensorStates(currFrameId, kMainCameraIndex, T_BC0);

  for (size_t camIdx = 0u; camIdx < numCameras; ++camIdx) {
    int extrinsicModelId = camera_rig_.getExtrinsicOptMode(camIdx);
    okvis::kinematics::Transformation T_XCi;
    switch (extrinsicModelId) {
      case Extrinsic_p_CB::kModelId:
      case Extrinsic_p_BC_q_BC::kModelId:
        getCameraSensorStates(currFrameId, camIdx, T_XCi);
        camera_rig_.setCameraExtrinsic(camIdx, T_XCi);
        break;
      case Extrinsic_p_C0C_q_C0C::kModelId:
        getCameraSensorStates(currFrameId, camIdx, T_XCi);
        camera_rig_.setCameraExtrinsic(camIdx, T_BC0 * T_XCi);
        break;
    }

    Eigen::Matrix<double, Eigen::Dynamic, 1> projectionIntrinsic;
    getSensorStateEstimateAs<ceres::EuclideanParamBlock>(
        currFrameId, camIdx, SensorStates::Camera,
        CameraSensorStates::Intrinsics, projectionIntrinsic);

    Eigen::Matrix<double, Eigen::Dynamic, 1> distortionCoeffs;
    getSensorStateEstimateAs<ceres::EuclideanParamBlock>(
        currFrameId, camIdx, SensorStates::Camera,
        CameraSensorStates::Distortion, distortionCoeffs);
    camera_rig_.setCameraIntrinsics(camIdx, projectionIntrinsic,
                                    distortionCoeffs);

    double tdEstimate;
    getSensorStateEstimateAs<ceres::CameraTimeParamBlock>(
        currFrameId, camIdx, SensorStates::Camera, CameraSensorStates::TD,
        tdEstimate);
    camera_rig_.setImageDelay(camIdx, tdEstimate);

    double trEstimate;
    getSensorStateEstimateAs<ceres::CameraTimeParamBlock>(
        currFrameId, camIdx, SensorStates::Camera, CameraSensorStates::TR,
        trEstimate);
    camera_rig_.setReadoutTime(camIdx, trEstimate);
  } // every camera.

  updateImuRig();
}

bool HybridFilter::measurementJacobian(
    const Eigen::Vector4d& homogeneousPoint,
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
                       // \rho p_i^{C_{t(i,j)}}\f$

  size_t camIdx = pointDataPtr->cameraIndex(observationIndex);
  const okvis::kinematics::Transformation T_BCj = camera_rig_.getCameraExtrinsic(camIdx);
  kinematics::Transformation T_WBtj = pointDataPtr->T_WBtij(observationIndex);
  okvis::kinematics::Transformation T_BC0 =
      camera_rig_.getCameraExtrinsic(kMainCameraIndex);

  Eigen::AlignedVector<okvis::kinematics::Transformation> transformList;
  std::vector<int> exponentList;
  transformList.reserve(4);
  exponentList.reserve(4);
  // transformations from left to right.
  transformList.push_back(T_BCj);
  exponentList.push_back(-1);
  Eigen::AlignedVector<okvis::kinematics::Transformation> lP_transformList = transformList;
  lP_transformList.reserve(4);
  kinematics::Transformation lP_T_WBtj =
      pointDataPtr->T_WBtij_ForJacobian(observationIndex);
  lP_transformList.push_back(lP_T_WBtj);
  transformList.push_back(T_WBtj);
  exponentList.push_back(-1);

  std::vector<size_t> camIndices{camIdx};
  std::vector<size_t> mtpjExtrinsicIndices{0u};
  std::vector<size_t> mtpjPoseIndices{1u};
  Eigen::AlignedVector<okvis::kinematics::Transformation> T_BC_list{T_BCj};
  int extrinsicModelId = camera_rig_.getExtrinsicOptMode(camIdx);
  std::vector<int> extrinsicModelIdList{extrinsicModelId};

  std::vector<size_t> observationIndices{observationIndex};
  uint64_t poseId = pointDataPtr->frameId(observationIndex);
  std::vector<uint64_t> frameIndices{poseId};
  Eigen::AlignedVector<okvis::kinematics::Transformation> T_WBt_list{T_WBtj};

  Eigen::Matrix<double, 4, 3> dhomo_dparams; // dHomogeneousPoint_dParameters.
  dhomo_dparams.setZero();

  okvis::kinematics::Transformation T_CtjX; // X is W or \f$C_{t(i,a)}\f$ or \f$C_{t(a)}\f$.
  if (pointLandmarkOptions_.landmarkModelId ==
      msckf::InverseDepthParameterization::kModelId) {
    size_t anchorCamIdx = pointDataPtr->anchorIds()[0].cameraIndex_;
    const okvis::kinematics::Transformation T_BCa =
        camera_rig_.getCameraExtrinsic(anchorCamIdx);

    okvis::kinematics::Transformation T_WBta;
    size_t anchorObservationIndex = pointDataPtr->anchorIds()[0].observationIndex_;
    kinematics::Transformation lP_T_WBta;
    if (pointLandmarkOptions_.anchorAtObservationTime) {
      T_WBta = pointDataPtr->T_WB_mainAnchor();
      lP_T_WBta = pointDataPtr->T_WB_mainAnchorForJacobian(
            FLAGS_use_first_estimate);
    } else {
      T_WBta = pointDataPtr->T_WB_mainAnchorStateEpoch();
      lP_T_WBta = pointDataPtr->T_WB_mainAnchorStateEpochForJacobian(
            FLAGS_use_first_estimate);
    }
    okvis::kinematics::Transformation T_WCta = T_WBta * T_BCa;
    T_CtjX = (T_WBtj * T_BCj).inverse() * T_WCta;

    lP_transformList.push_back(lP_T_WBta);
    transformList.push_back(T_WBta);
    exponentList.push_back(1);

    lP_transformList.push_back(T_BCa);
    transformList.push_back(T_BCa);
    exponentList.push_back(1);

    camIndices.push_back(anchorCamIdx);
    mtpjExtrinsicIndices.push_back(3u);
    mtpjPoseIndices.push_back(2u);
    T_BC_list.push_back(T_BCa);

    int anchorExtrinsicModelId = camera_rig_.getExtrinsicOptMode(anchorCamIdx);
    extrinsicModelIdList.push_back(anchorExtrinsicModelId);
    observationIndices.push_back(anchorObservationIndex);
    frameIndices.push_back(pointDataPtr->anchorIds()[0].frameId_);
    T_WBt_list.push_back(T_WBta);

    dhomo_dparams(0, 0) = 1;
    dhomo_dparams(1, 1) = 1;
    dhomo_dparams(3, 2) = 1;
  } else {
    T_CtjX = (T_WBtj * T_BCj).inverse();

    dhomo_dparams(0, 0) = 1;
    dhomo_dparams(1, 1) = 1;
    dhomo_dparams(2, 2) = 1;
  }

  std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry =
      camera_rig_.getCameraGeometry(camIdx);
  Eigen::Vector3d rhoxpCtj = (T_CtjX * homogeneousPoint).head<3>();
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

  okvis::MultipleTransformPointJacobian lP_mtpj(lP_transformList, exponentList, homogeneousPoint);
  okvis::MultipleTransformPointJacobian mtpj(transformList, exponentList, homogeneousPoint);
  std::vector<std::pair<size_t, size_t>> startIndexToMinDim;
  Eigen::AlignedVector<Eigen::MatrixXd> dpoint_dX; // drhoxpCtj_dParameters
  // compute drhoxpCtj_dParameters
  size_t startIndexCameraParams = startIndexOfCameraParams(kMainCameraIndex);
  for (size_t ja = 0; ja < camIndices.size(); ++ja) { // observing camera and/or anchor camera.
    // Extrinsic Jacobians.
    int mainExtrinsicModelId =
        camera_rig_.getExtrinsicOptMode(kMainCameraIndex);
    if (!fixCameraExtrinsicParams_[camIndices[ja]]) {
      Eigen::Matrix<double, 4, 6> dpoint_dT_BC = mtpj.dp_dT(mtpjExtrinsicIndices[ja]);
      std::vector<size_t> involvedCameraIndices;
      involvedCameraIndices.reserve(2);
      involvedCameraIndices.push_back(camIndices[ja]);
      std::vector<std::pair<size_t, size_t>> startIndexToMinDimExtrinsics;
      Eigen::AlignedVector<Eigen::MatrixXd> dT_BC_dExtrinsics;
      computeExtrinsicJacobians(T_BC_list[ja], T_BC0, extrinsicModelIdList[ja],
                               mainExtrinsicModelId, &dT_BC_dExtrinsics,
                               &involvedCameraIndices, kMainCameraIndex);
      size_t camParamIdx = 0u;
      for (auto idx : involvedCameraIndices) {
        size_t extrinsicStartIndex = intraStartIndexOfCameraParams(idx);
        size_t extrinsicDim = camera_rig_.getMinimalExtrinsicDimen(idx);
        startIndexToMinDim.emplace_back(extrinsicStartIndex, extrinsicDim);
        dpoint_dX.emplace_back(dpoint_dT_BC * dT_BC_dExtrinsics[camParamIdx]);
        ++camParamIdx;
      }
    }

    // Jacobians relative to nav states
    Eigen::Matrix<double, 4, 6> lP_dpoint_dT_WBt = lP_mtpj.dp_dT(mtpjPoseIndices[ja]);
    Eigen::Matrix<double, 4, 6> dpoint_dT_WBt = mtpj.dp_dT(mtpjPoseIndices[ja]);
    auto stateIter = statesMap_.find(frameIndices[ja]);
    int orderInCov = stateIter->second.global.at(GlobalStates::T_WS).startIndexInCov;
    size_t navStateIndex = orderInCov - startIndexCameraParams;
    startIndexToMinDim.emplace_back(navStateIndex, 6u);

    // Jacobians relative to time parameters and velocity.
    if (ja == 1u && !pointLandmarkOptions_.anchorAtObservationTime) {
      // Because the anchor frame is at state epoch, then its pose to
      // time and velocity are zero.
      dpoint_dX.emplace_back(lP_dpoint_dT_WBt);
    } else {
      Eigen::Matrix3d Phi_pq_tij_tj = pointDataPtr->Phi_pq_feature(observationIndices[ja]);
      lP_dpoint_dT_WBt.rightCols(3) += lP_dpoint_dT_WBt.leftCols(3) * Phi_pq_tij_tj;
      dpoint_dX.emplace_back(lP_dpoint_dT_WBt);
      Eigen::Vector3d v_WBt =
          pointDataPtr->v_WBtij(observationIndices[ja]);
      Eigen::Matrix<double, 6, 1> dT_WBt_dt;
      dT_WBt_dt.head<3>() =
          msckf::SimpleImuPropagationJacobian::dp_dt(v_WBt);
      Eigen::Vector3d omega_Btij =
          pointDataPtr->omega_Btij(observationIndices[ja]);
      dT_WBt_dt.tail<3>() = msckf::SimpleImuPropagationJacobian::dtheta_dt(
          omega_Btij, T_WBt_list[ja].q());
      Eigen::Vector2d dt_dtdtr(1, 1);
      dt_dtdtr[1] = pointDataPtr->normalizedRow(observationIndices[ja]);

      size_t cameraDelayIntraIndex =
          intraStartIndexOfCameraParams(camIndices[ja], CameraSensorStates::TD);
      startIndexToMinDim.emplace_back(cameraDelayIntraIndex, 2u);
      dpoint_dX.emplace_back(dpoint_dT_WBt * dT_WBt_dt * dt_dtdtr.transpose());

      double featureDelay =
          pointDataPtr->normalizedFeatureTime(observationIndices[ja]);
      startIndexToMinDim.emplace_back(navStateIndex + 6u, 3u);
      dpoint_dX.emplace_back(lP_dpoint_dT_WBt.leftCols(3) * featureDelay);
    }
  }

  // According to Li 2013 IJRR high precision, eq 41 and 55, among all Jacobian
  // components, only the Jacobian of nav states need to use first estimates of
  // position and velocity. The Jacobians relative to intrinsic parameters, and
  // relative to \f$\rho p^{C(t_{i,j})}\f$ do not need to use first estimates.

  // Accumulate Jacobians relative to nav states.
  J_x->setZero();
  size_t iterIndex = 0u;
  for (auto& startAndLen : startIndexToMinDim) {
    J_x->block(0, startAndLen.first, 2, startAndLen.second) +=
        dz_drhoxpCtj * dpoint_dX[iterIndex].topRows<3>();
    ++iterIndex;
  }
  // Jacobian relative to camera parameters.
  if (!fixCameraIntrinsicParams_[camIdx]) {
    int projOptModelId = camera_rig_.getProjectionOptMode(camIdx);
    ProjectionOptKneadIntrinsicJacobian(projOptModelId, &intrinsicsJacobian);
    size_t startIndex = intraStartIndexOfCameraParams(camIdx, CameraSensorStates::Intrinsics);
    J_x->block(0, startIndex, 2, intrinsicsJacobian.cols()) = intrinsicsJacobian;
  }
  // Jacobian relative to landmark parameters.
  // According to Li 2013 IJRR high precision, eq 41 and 55, J_pfi does not need
  // to use first estimates. As a result, expression 2 should be used.
  // And tests show that (1) often cause divergence for mono MSCKF.
//  (*J_pfi) = dz_drhoxpCtj * lP_mtpj.dp_dpoint().topRows<3>() * dhomo_dparams; //  (1)
  (*J_pfi) = dz_drhoxpCtj * T_CtjX.T().topRows<3>() * dhomo_dparams; // (2)
  return true;
}

bool HybridFilter::featureJacobian(
    const MapPoint &mp, msckf::PointLandmark *pointLandmark,
    Eigen::MatrixXd &H_oi, Eigen::Matrix<double, Eigen::Dynamic, 1> &r_oi,
    Eigen::MatrixXd &R_oi, Eigen::Matrix<double, Eigen::Dynamic, 3> *pH_fi,
    std::vector<uint64_t> *orderedCulledFrameIds) const {
  // all observations for this feature point
  Eigen::AlignedVector<Eigen::Vector2d> obsInPixel;
  std::vector<double> imageNoiseStd; // std noise in pixels

  std::shared_ptr<msckf::PointSharedData> pointDataPtr(new msckf::PointSharedData());
  pointLandmark->setModelId(pointLandmarkOptions_.landmarkModelId);
  msckf::TriangulationStatus status = triangulateAMapPoint(
      mp, obsInPixel, *pointLandmark, imageNoiseStd,
      pointDataPtr.get(), orderedCulledFrameIds, optimizationOptions_.useEpipolarConstraint);
  if (!status.triangulationOk) {
    return false;
  }

  if (orderedCulledFrameIds) {
    pointDataPtr->removeExtraObservations(*orderedCulledFrameIds, &imageNoiseStd);
  }

  computeHTimer.start();
  // dimension of variables used in computing feature Jacobians, including
  // camera intrinsics and all cloned states except the most recent one
  // in which an observation should never occur for a MSCKF feature.
  int featureVariableDimen = minimalDimOfAllCameraParams() +
                             kClonedStateMinimalDimen * (statesMap_.size() - 1);
  if (pH_fi == NULL) {
    CHECK_NE(statesMap_.rbegin()->first, pointDataPtr->lastFrameId())
        << "The landmark should not be observed by the latest frame for an "
           "MSCKF feature.";
  } else {
    featureVariableDimen += 9;
    OKVIS_ASSERT_EQ_DBG(Exception, pointDataPtr->anchorIds()[0].frameId_,
                        (statesMap_.rbegin())->first,
                        "Anchor frame of a SLAM feature to be added to the "
                        "state vector should be the current frame.");
  }

  pointDataPtr->computePoseAndVelocityForJacobians(FLAGS_use_first_estimate);
  pointDataPtr->computeSharedJacobians(optimizationOptions_.cameraObservationModelId);

  size_t numObservations = pointDataPtr->numObservations();
  Eigen::AlignedVector<Eigen::Matrix<double, 2, 3>> vJ_pfi;
  Eigen::AlignedVector<Eigen::Matrix<double, 2, 1>> vri;  // residuals for feature i
  vJ_pfi.reserve(numObservations);
  vri.reserve(numObservations);

  Eigen::Vector4d homogeneousPoint =
      Eigen::Map<Eigen::Vector4d>(pointLandmark->data(), 4);

  // containers of the above Jacobians for all observations of a mappoint
  Eigen::AlignedVector<Eigen::Matrix<double, 2, Eigen::Dynamic>> vJ_X;
  vJ_X.reserve(numObservations);

  size_t numValidObs = 0u;
  auto observationIter = pointDataPtr->begin();
  auto imageNoiseIter = imageNoiseStd.begin();
  // compute Jacobians for a measurement in image j of the current feature i
  for (size_t observationIndex = 0; observationIndex < numObservations; ++observationIndex) {
    Eigen::Matrix<double, 2, Eigen::Dynamic> J_x(2, featureVariableDimen);
    Eigen::Matrix<double, 2, 3> J_pfi;
    Eigen::Vector2d residual;
    bool validJacobian = measurementJacobian(
        homogeneousPoint, obsInPixel[observationIndex],
        observationIndex, pointDataPtr, &J_x, &J_pfi, &residual);
    if (!validJacobian) {
        ++observationIter;
        imageNoiseIter = imageNoiseStd.erase(imageNoiseIter);
        imageNoiseIter = imageNoiseStd.erase(imageNoiseIter);
        continue;
    }

    vri.push_back(residual);
    vJ_X.push_back(J_x);
    vJ_pfi.push_back(J_pfi);

    ++numValidObs;
    ++observationIter;
    imageNoiseIter += 2;
  }
  if (numValidObs < pointLandmarkOptions_.minTrackLengthForMsckf) {
    computeHTimer.stop();
    return false;
  }

  // Now we stack the Jacobians and marginalize the point position related
  // dimensions by projecting \f$H_{x_i}\f$ onto the nullspace of $H_{f^i}$.
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
    Ri(saga2, saga2) = imageNoiseStd[saga2] * imageNoiseStd[saga2];
    Ri(saga2 + 1, saga2 + 1) = imageNoiseStd[saga2 + 1] * imageNoiseStd[saga2 + 1];
  }

  if (pH_fi) { // this point is to be included in the state vector.
    r_oi = ri;
    H_oi = H_xi;
    R_oi = Ri;
    *pH_fi = H_fi;
  } else {
    int columnRankHf = status.raysParallel ? 2 : 3;
    // 2nx(2n-ColumnRank), n==numValidObs
    Eigen::MatrixXd nullQ = FilterHelper::leftNullspaceWithRankCheck(H_fi, columnRankHf);

    r_oi.noalias() = nullQ.transpose() * ri;
    H_oi.noalias() = nullQ.transpose() * H_xi;
    R_oi = nullQ.transpose() * (Ri * nullQ).eval();
  }

  vri.clear();
  vJ_pfi.clear();
  vJ_X.clear();
  computeHTimer.stop();
  return true;
}

bool HybridFilter::slamFeatureJacobian(const MapPoint &mp, Eigen::MatrixXd &H_x,
                                       Eigen::Matrix<double, -1, 1> &r_i,
                                       Eigen::MatrixXd &R_i,
                                       Eigen::MatrixXd &H_f) const {
  std::shared_ptr<msckf::PointSharedData> pointDataPtr(new msckf::PointSharedData());

  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> obsInPixel;
  obsInPixel.reserve(2);
  std::vector<double> imageNoiseStd;
  imageNoiseStd.reserve(4);

  if (pointLandmarkOptions_.landmarkModelId ==
      msckf::InverseDepthParameterization::kModelId) {
    // add the observation for anchor camera frame which is only needed for
    // computing Jacobians.
    KeypointIdentifier anchorFrameId(mp.anchorStateId, mp.anchorCameraId, 0u);
    auto itObs = std::find_if(
        mp.observations.begin(), mp.observations.end(),
        [anchorFrameId](const std::pair<KeypointIdentifier, uint64_t> &v) {
          return v.first.frameId == anchorFrameId.frameId &&
                 v.first.cameraIndex == anchorFrameId.cameraIndex;
        });
    OKVIS_ASSERT_FALSE(Exception, itObs == mp.observations.end(),
                       "Anchor observation not found!");

    uint64_t poseId = mp.anchorStateId;
    Eigen::Vector2d measurement;
    auto multiFrameIter = multiFramePtrMap_.find(poseId);
    okvis::MultiFramePtr multiFramePtr = multiFrameIter->second;
    multiFramePtr->getKeypoint(itObs->first.cameraIndex,
                               itObs->first.keypointIndex, measurement);
    okvis::Time imageTimestamp =
        multiFramePtr->timestamp(itObs->first.cameraIndex);
    std::shared_ptr<const cameras::CameraBase> cameraGeometry =
        camera_rig_.getCameraGeometry(itObs->first.cameraIndex);
    uint32_t imageHeight = cameraGeometry->imageHeight();
    obsInPixel.push_back(measurement);

    double kpSize = 1.0;
    multiFramePtr->getKeypointSize(itObs->first.cameraIndex,
                                   itObs->first.keypointIndex, kpSize);
    double sigma = kpSize / 8;
    imageNoiseStd.push_back(sigma);
    imageNoiseStd.push_back(sigma);

    std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr;
    getGlobalStateParameterBlockPtr(poseId, GlobalStates::T_WS,
                                    parameterBlockPtr);

    double kpN = measurement[1] / imageHeight - 0.5;
    pointDataPtr->addKeypointObservation(itObs->first, parameterBlockPtr, kpN,
                                         imageTimestamp);

    okvis::AnchorFrameIdentifier anchorId{mp.anchorStateId, mp.anchorCameraId,
                                          0u};
    pointDataPtr->setAnchors({anchorId});
  }

  // add observations in images of the current frame.
  uint64_t currFrameId = currentFrameId();
  size_t numNewObservations = 0u;
  for (auto itObs = mp.observations.rbegin(), iteObs = mp.observations.rend();
       itObs != iteObs; ++itObs) {
    if (itObs->first.frameId == currFrameId) {
      uint64_t poseId = itObs->first.frameId;
      Eigen::Vector2d measurement;
      auto multiFrameIter = multiFramePtrMap_.find(poseId);
      okvis::MultiFramePtr multiFramePtr = multiFrameIter->second;
      multiFramePtr->getKeypoint(itObs->first.cameraIndex,
                                 itObs->first.keypointIndex, measurement);
      okvis::Time imageTimestamp = multiFramePtr->timestamp(itObs->first.cameraIndex);
      obsInPixel.push_back(measurement);

      double kpSize = 1.0;
      multiFramePtr->getKeypointSize(itObs->first.cameraIndex,
                                     itObs->first.keypointIndex, kpSize);

      double sigma = kpSize / 8;
      imageNoiseStd.push_back(sigma);
      imageNoiseStd.push_back(sigma);

      std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr;
      getGlobalStateParameterBlockPtr(poseId, GlobalStates::T_WS, parameterBlockPtr);

      std::shared_ptr<const cameras::CameraBase> cameraGeometry =
          camera_rig_.getCameraGeometry(itObs->first.cameraIndex);
      uint32_t imageHeight = cameraGeometry->imageHeight();
      double kpN = measurement[1] / imageHeight - 0.5;
      pointDataPtr->addKeypointObservation(
            itObs->first, parameterBlockPtr, kpN, imageTimestamp);
      ++numNewObservations;
    } else {
      break;
    }
  }

  OKVIS_ASSERT_GE(
      Exception, numNewObservations, 1u,
      "A point in slamFeatureJacobian should be observed in current frame!");

  propagatePoseAndVelocityForMapPoint(pointDataPtr.get());

  computeHTimer.start();
  // dimension of variables used in computing feature Jacobians, including
  // camera intrinsics and all cloned states.
  int featureVariableDimen = minimalDimOfAllCameraParams() +
                             kClonedStateMinimalDimen * statesMap_.size();

  pointDataPtr->computePoseAndVelocityForJacobians(FLAGS_use_first_estimate);
  pointDataPtr->computeSharedJacobians(optimizationOptions_.cameraObservationModelId);

  // get the landmark parameters.
  uint64_t hpbid = mp.id;
  auto idPos = std::find_if(
      mInCovLmIds.begin(), mInCovLmIds.end(),
      [hpbid](const okvis::ceres::HomogeneousPointParameterBlock &x) {
        return x.id() == hpbid;
      });
  OKVIS_ASSERT_TRUE(Exception, idPos != mInCovLmIds.end(),
                    "The tracked landmark is not in the state vector.");
  size_t covPtId = std::distance(mInCovLmIds.begin(), idPos);
  Eigen::Vector4d homogeneousPoint = idPos->estimate();
  if (pointLandmarkOptions_.landmarkModelId ==
      msckf::InverseDepthParameterization::kModelId) {
    if (homogeneousPoint[2] < 1e-6) {
      LOG(WARNING) << "Negative depth in anchor camera frame point: "
                   << homogeneousPoint.transpose();
      computeHTimer.stop();
      return false;
    }
    //[\alpha = X/Z, \beta= Y/Z, 1, \rho=1/Z] in anchor camera frame.
    homogeneousPoint /= homogeneousPoint[2];
  } else {
    if (homogeneousPoint[3] < 1e-6) {
      LOG(WARNING) << "Point at infinity in world frame: "
                   << homogeneousPoint.transpose();
      computeHTimer.stop();
      return false;
    }
    homogeneousPoint /= homogeneousPoint[3];  //[X, Y, Z, 1] in world frame.
  }

  size_t numObservations = pointDataPtr->numObservations();
  // compute Jacobians for all observations in the current frame.
  Eigen::AlignedVector<Eigen::Matrix<double, 2, Eigen::Dynamic>> vJ_X;
  vJ_X.reserve(numObservations);
  Eigen::AlignedVector<Eigen::Matrix<double, 2, 3>> vJ_pfi;
  Eigen::AlignedVector<Eigen::Matrix<double, 2, 1>> vri;
  vJ_pfi.reserve(numObservations);
  vri.reserve(numObservations);

  size_t numValidObs = 0u;

  auto observationIter = pointDataPtr->begin();
  ++observationIter;  // skip the anchor frame observation.
  auto imageNoiseIter = imageNoiseStd.begin() + 2;
  for (size_t observationIndex = 1u; observationIndex < numObservations; ++observationIndex) {
    Eigen::Matrix<double, 2, Eigen::Dynamic> J_x(2, featureVariableDimen);
    Eigen::Matrix<double, 2, 3> J_pfi;
    Eigen::Vector2d residual;
    bool validJacobian = measurementJacobian(
        homogeneousPoint, obsInPixel[observationIndex],
        observationIndex, pointDataPtr, &J_x, &J_pfi, &residual);
    if (!validJacobian) {
        ++observationIter;
        imageNoiseIter = imageNoiseStd.erase(imageNoiseIter);
        imageNoiseIter = imageNoiseStd.erase(imageNoiseIter);
        continue;
    }
    vri.push_back(residual);
    vJ_X.push_back(J_x);
    vJ_pfi.push_back(J_pfi);

    ++numValidObs;
    ++observationIter;
    imageNoiseIter += 2;
  }

  H_x.resize(2 * numValidObs, featureVariableDimen);
  H_f.resize(2 * numValidObs, 3 * mInCovLmIds.size());
  H_f.setZero();
  r_i.resize(2 * numValidObs, 1);
  R_i = Eigen::MatrixXd::Identity(2 * numValidObs, 2 * numValidObs);
  for (size_t saga = 0; saga < numValidObs; ++saga) {
    size_t saga2 = saga * 2;
    H_x.block(saga2, 0, 2, featureVariableDimen) = vJ_X[saga];
    H_f.block<2, 3>(saga2, covPtId * 3) = vJ_pfi[saga];
    r_i.segment<2>(saga2) = vri[saga];
    R_i(saga2, saga2) = imageNoiseStd[saga2] * imageNoiseStd[saga2];
    R_i(saga2 + 1, saga2 + 1) = imageNoiseStd[saga2 + 1] * imageNoiseStd[saga2 + 1];
  }

  vri.clear();
  vJ_pfi.clear();
  vJ_X.clear();
  computeHTimer.stop();
  return numValidObs > 0;
}

void HybridFilter::updateImuAugmentedStates(
    const States& stateInQuestion, const Eigen::VectorXd deltaAugmentedParams) {
  const int imuIdx = 0;
  int imuModelId = imu_rig_.getModelId(imuIdx);
  switch (imuModelId) {
    case Imu_BG_BA::kModelId:
      return;
    case Imu_BG_BA_TG_TS_TA::kModelId: {
      uint64_t TGId = stateInQuestion.sensors.at(SensorStates::Imu)
                          .at(imuIdx)
                          .at(ImuSensorStates::TG)
                          .id;
      std::shared_ptr<ceres::ShapeMatrixParamBlock> tgParamBlockPtr =
          std::static_pointer_cast<ceres::ShapeMatrixParamBlock>(
              mapPtr_->parameterBlockPtr(TGId));
      Eigen::Matrix<double, 9, 1> sm = tgParamBlockPtr->estimate();
      tgParamBlockPtr->setEstimate(sm + deltaAugmentedParams.head<9>());

      uint64_t TSId = stateInQuestion.sensors.at(SensorStates::Imu)
                          .at(imuIdx)
                          .at(ImuSensorStates::TS)
                          .id;
      std::shared_ptr<ceres::ShapeMatrixParamBlock> tsParamBlockPtr =
          std::static_pointer_cast<ceres::ShapeMatrixParamBlock>(
              mapPtr_->parameterBlockPtr(TSId));
      sm = tsParamBlockPtr->estimate();
      tsParamBlockPtr->setEstimate(sm + deltaAugmentedParams.segment<9>(9));

      uint64_t TAId = stateInQuestion.sensors.at(SensorStates::Imu)
                          .at(imuIdx)
                          .at(ImuSensorStates::TA)
                          .id;
      std::shared_ptr<ceres::ShapeMatrixParamBlock> taParamBlockPtr =
          std::static_pointer_cast<ceres::ShapeMatrixParamBlock>(
              mapPtr_->parameterBlockPtr(TAId));
      sm = taParamBlockPtr->estimate();
      taParamBlockPtr->setEstimate(sm + deltaAugmentedParams.segment<9>(18));
    } break;
    default:
      LOG(WARNING) << "UpdateState for IMU model " << imuModelId
                   << " not implemented!";
      break;
  }
}

void HybridFilter::cloneImuAugmentedStates(
    const States& stateInQuestion,
    StatePointerAndEstimateList*
        currentStates) const {
  const int imuIdx = 0;
  uint64_t TGId = stateInQuestion.sensors.at(SensorStates::Imu)
                      .at(imuIdx)
                      .at(ImuSensorStates::TG)
                      .id;
  std::shared_ptr<const ceres::ShapeMatrixParamBlock> tgParamBlockPtr =
      std::static_pointer_cast<const ceres::ShapeMatrixParamBlock>(
          mapPtr_->parameterBlockPtr(TGId));
  currentStates->emplace_back(tgParamBlockPtr, tgParamBlockPtr->estimate());

  uint64_t TSId = stateInQuestion.sensors.at(SensorStates::Imu)
                      .at(imuIdx)
                      .at(ImuSensorStates::TS)
                      .id;
  std::shared_ptr<const ceres::ShapeMatrixParamBlock> tsParamBlockPtr =
      std::static_pointer_cast<const ceres::ShapeMatrixParamBlock>(
          mapPtr_->parameterBlockPtr(TSId));
  currentStates->emplace_back(tsParamBlockPtr, tsParamBlockPtr->estimate());

  uint64_t TAId = stateInQuestion.sensors.at(SensorStates::Imu)
                      .at(imuIdx)
                      .at(ImuSensorStates::TA)
                      .id;
  std::shared_ptr<const ceres::ShapeMatrixParamBlock> taParamBlockPtr =
      std::static_pointer_cast<const ceres::ShapeMatrixParamBlock>(
          mapPtr_->parameterBlockPtr(TAId));
  currentStates->emplace_back(taParamBlockPtr, taParamBlockPtr->estimate());
}

void HybridFilter::updateCameraSensorStates(const States& stateInQuestion,
                                            const Eigen::VectorXd& deltaX) {
  size_t camParamIndex = startIndexOfCameraParamsFast(0u);
  const int numCameras = camera_rig_.numberCameras();
  for (int camIdx = 0; camIdx < numCameras; ++camIdx) {
    // It has been shown that estimating the extrinsic parameter T_ClCr performs
    // better than estimating T_SCr. see eq(5) in A comparative analysis of
    // tightly-coupled monocular, binocular, and stereo VINS.
    // Therefore, we may save T_C0Ci in place of T_SCi.
    if (!fixCameraExtrinsicParams_[camIdx]) {
      uint64_t extrinsicId = stateInQuestion.sensors.at(SensorStates::Camera)
                                 .at(camIdx)
                                 .at(CameraSensorStates::T_SCi)
                                 .id;
      std::shared_ptr<ceres::PoseParameterBlock> extrinsicParamBlockPtr =
          std::static_pointer_cast<ceres::PoseParameterBlock>(
              mapPtr_->parameterBlockPtr(extrinsicId));

      kinematics::Transformation T_XC = extrinsicParamBlockPtr->estimate();
      Eigen::Vector3d t_XC;
      Eigen::Quaterniond q_XC;
      int extrinsicOptModelId = camera_rig_.getExtrinsicOptMode(camIdx);
      int minExtrinsicDim = camera_rig_.getMinimalExtrinsicDimen(camIdx);
      ExtrinsicModelUpdateState(extrinsicOptModelId, T_XC.r(), T_XC.q(),
                                deltaX.segment(camParamIndex, minExtrinsicDim),
                                &t_XC, &q_XC);
      extrinsicParamBlockPtr->setEstimate(
          kinematics::Transformation(t_XC, q_XC));
      camParamIndex += minExtrinsicDim;
    }

    if (!fixCameraIntrinsicParams_[camIdx]) {
      const int minProjectionDim =
          camera_rig_.getMinimalProjectionDimen(camIdx);
      uint64_t intrinsicId = stateInQuestion.sensors.at(SensorStates::Camera)
                                 .at(camIdx)
                                 .at(CameraSensorStates::Intrinsics)
                                 .id;
      std::shared_ptr<ceres::EuclideanParamBlock> intrinsicParamBlockPtr =
          std::static_pointer_cast<ceres::EuclideanParamBlock>(
              mapPtr_->parameterBlockPtr(intrinsicId));
      Eigen::VectorXd cameraIntrinsics =
          intrinsicParamBlockPtr->estimate() +
          deltaX.segment(camParamIndex, minProjectionDim);
      intrinsicParamBlockPtr->setEstimate(cameraIntrinsics);
      camParamIndex += minProjectionDim;

      const int nDistortionCoeffDim = camera_rig_.getDistortionDimen(camIdx);
      uint64_t distortionId = stateInQuestion.sensors.at(SensorStates::Camera)
                                  .at(camIdx)
                                  .at(CameraSensorStates::Distortion)
                                  .id;
      std::shared_ptr<ceres::EuclideanParamBlock> distortionParamBlockPtr =
          std::static_pointer_cast<ceres::EuclideanParamBlock>(
              mapPtr_->parameterBlockPtr(distortionId));
      Eigen::VectorXd cameraDistortion =
          distortionParamBlockPtr->estimate() +
          deltaX.segment(camParamIndex, nDistortionCoeffDim);
      distortionParamBlockPtr->setEstimate(cameraDistortion);
      camParamIndex += nDistortionCoeffDim;
    }

    uint64_t tdId = stateInQuestion.sensors.at(SensorStates::Camera)
                        .at(camIdx)
                        .at(CameraSensorStates::TD)
                        .id;
    std::shared_ptr<ceres::ParameterBlock> tdParamBlockPtr =
        mapPtr_->parameterBlockPtr(tdId);
    tdParamBlockPtr->parameters()[0] += deltaX(camParamIndex);
    camParamIndex += 1;

    uint64_t trId = stateInQuestion.sensors.at(SensorStates::Camera)
                        .at(camIdx)
                        .at(CameraSensorStates::TR)
                        .id;
    std::shared_ptr<ceres::ParameterBlock> trParamBlockPtr =
        mapPtr_->parameterBlockPtr(trId);
    trParamBlockPtr->parameters()[0] += deltaX[camParamIndex];
    camParamIndex += 1;
  }
}

void HybridFilter::cloneCameraParameterStates(
    const States& stateInQuestion, StatePointerAndEstimateList* currentStates,
    size_t camIdx) const {
  if (!fixCameraExtrinsicParams_[camIdx]) {
    uint64_t extrinsicId = stateInQuestion.sensors.at(SensorStates::Camera)
                               .at(camIdx)
                               .at(CameraSensorStates::T_SCi)
                               .id;
    std::shared_ptr<const ceres::ParameterBlock> extrinsicBlockPtr =
            mapPtr_->parameterBlockPtr(extrinsicId);
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> T_BC(
        extrinsicBlockPtr->parameters());
    currentStates->emplace_back(extrinsicBlockPtr, T_BC);
  }

  if (!fixCameraIntrinsicParams_[camIdx]) {
    uint64_t intrinsicId = stateInQuestion.sensors.at(SensorStates::Camera)
                               .at(camIdx)
                               .at(CameraSensorStates::Intrinsics)
                               .id;
    std::shared_ptr<const ceres::EuclideanParamBlock> intrinsicParamBlockPtr =
        std::static_pointer_cast<const ceres::EuclideanParamBlock>(
            mapPtr_->parameterBlockPtr(intrinsicId));
    currentStates->emplace_back(intrinsicParamBlockPtr, intrinsicParamBlockPtr->estimate());

    uint64_t distortionId = stateInQuestion.sensors.at(SensorStates::Camera)
                                .at(camIdx)
                                .at(CameraSensorStates::Distortion)
                                .id;
    std::shared_ptr<const ceres::EuclideanParamBlock> distortionParamBlockPtr =
        std::static_pointer_cast<const ceres::EuclideanParamBlock>(
            mapPtr_->parameterBlockPtr(distortionId));
    currentStates->emplace_back(distortionParamBlockPtr, distortionParamBlockPtr->estimate());
  }

  uint64_t tdId = stateInQuestion.sensors.at(SensorStates::Camera)
                      .at(camIdx)
                      .at(CameraSensorStates::TD)
                      .id;
  std::shared_ptr<const ceres::ParameterBlock> tdParamBlockPtr =
          mapPtr_->parameterBlockPtr(tdId);
  currentStates->emplace_back(tdParamBlockPtr, Eigen::Matrix<double, 1, 1>(tdParamBlockPtr->parameters()));

  uint64_t trId = stateInQuestion.sensors.at(SensorStates::Camera)
                      .at(camIdx)
                      .at(CameraSensorStates::TR)
                      .id;
  std::shared_ptr<const ceres::ParameterBlock> trParamBlockPtr =
          mapPtr_->parameterBlockPtr(trId);
  currentStates->emplace_back(trParamBlockPtr, Eigen::Matrix<double, 1, 1>(trParamBlockPtr->parameters()));
}

void HybridFilter::cloneFilterStates(
    StatePointerAndEstimateList* currentStates) const {
  currentStates->reserve(statesMap_.size() - 1 + 2 + 3 + 5);
  // The order of state vectors follows updateStates().
  std::map<uint64_t, States>::const_reverse_iterator lastElementIterator =
      statesMap_.rbegin();
  const States& stateInQuestion = lastElementIterator->second;
  uint64_t stateId = stateInQuestion.id;

  // nav states
  std::shared_ptr<const ceres::ParameterBlock> poseParamBlockPtr =
          mapPtr_->parameterBlockPtr(stateId);
  Eigen::Map<const Eigen::Matrix<double, 7, 1>> T_WS(
      poseParamBlockPtr->parameters());
  currentStates->emplace_back(poseParamBlockPtr, T_WS);
  const int imuIdx = 0;
  uint64_t SBId = stateInQuestion.sensors.at(SensorStates::Imu)
                      .at(imuIdx)
                      .at(ImuSensorStates::SpeedAndBias)
                      .id;
  std::shared_ptr<const ceres::SpeedAndBiasParameterBlock> sbParamBlockPtr =
      std::static_pointer_cast<const ceres::SpeedAndBiasParameterBlock>(
          mapPtr_->parameterBlockPtr(SBId));
  currentStates->emplace_back(sbParamBlockPtr, sbParamBlockPtr->estimate());

  cloneImuAugmentedStates(stateInQuestion, currentStates);
  for (size_t j = 0u; j < camera_rig_.numberCameras(); ++j) {
    cloneCameraParameterStates(stateInQuestion, currentStates, j);
  }

  auto finalIter = statesMap_.end();
  --finalIter; // The last one has been added early on.
  for (auto iter = statesMap_.begin(); iter != finalIter; ++iter) {
    stateId = iter->first;
    std::shared_ptr<const ceres::ParameterBlock> poseParamBlockPtr =
        mapPtr_->parameterBlockPtr(stateId);
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> T_WS(
        poseParamBlockPtr->parameters());
    currentStates->emplace_back(poseParamBlockPtr, T_WS);

    SBId = iter->second.sensors.at(SensorStates::Imu)
               .at(imuIdx)
               .at(ImuSensorStates::SpeedAndBias)
               .id;
    sbParamBlockPtr =
        std::static_pointer_cast<const ceres::SpeedAndBiasParameterBlock>(
            mapPtr_->parameterBlockPtr(SBId));
    currentStates->emplace_back(sbParamBlockPtr, sbParamBlockPtr->estimate().head<3>());
  }
  // TODO(jhuai): consider point landmarks in the states.
}

void HybridFilter::boxminusFromInput(
    const StatePointerAndEstimateList& refStates,
    Eigen::Matrix<double, Eigen::Dynamic, 1>* deltaX) const {
  int covDim = covariance_.rows();
  deltaX->resize(covDim, 1);
  // nav states
  Eigen::Matrix<double, 6, 1> delta_T_WB;
  okvis::ceres::PoseLocalParameterization::minus(
      refStates.at(0).parameterBlockPtr->parameters(),
      refStates.at(0).parameterEstimate.data(), delta_T_WB.data());
  deltaX->head<6>() = delta_T_WB;

  Eigen::Map<const Eigen::Matrix<double, 9, 1>> currentSpeedAndBias(
      refStates.at(1).parameterBlockPtr->parameters());
  deltaX->segment<9>(6) =
      refStates.at(1).parameterEstimate - currentSpeedAndBias;

  // IMU augmented parameters
  int covStateIndex = 15;
  for (int j = 0; j < 3; ++j) {
    Eigen::Map<const Eigen::Matrix<double, 9, 1>> sm(
        refStates.at(j + 2).parameterBlockPtr->parameters());
    deltaX->segment<9>(covStateIndex + j * 9) =
        refStates.at(j + 2).parameterEstimate - sm;
  }
  covStateIndex += 3 * 9;
  int stateBlockIndex = 5;

  // camera related parameters
  for (size_t camIdx = 0u; camIdx < camera_rig_.numberCameras(); ++camIdx) {
    if (!fixCameraExtrinsicParams_[camIdx]) {
      int extrinsicOptModelId = camera_rig_.getExtrinsicOptMode(camIdx);
      int minExtrinsicDim = camera_rig_.getMinimalExtrinsicDimen(camIdx);
      Eigen::VectorXd delta(minExtrinsicDim);
      ExtrinsicModelOminus(
          extrinsicOptModelId,
          refStates.at(stateBlockIndex).parameterBlockPtr->parameters(),
          refStates.at(stateBlockIndex).parameterEstimate.data(), delta.data());
      deltaX->segment(covStateIndex, minExtrinsicDim) = delta;
      covStateIndex += minExtrinsicDim;
      ++stateBlockIndex;
    }

    if (!fixCameraIntrinsicParams_[camIdx]) {
      const int minProjectionDim =
          camera_rig_.getMinimalProjectionDimen(camIdx);
      Eigen::Map<const Eigen::VectorXd> cameraIntrinsics(
          refStates.at(stateBlockIndex).parameterBlockPtr->parameters(),
          minProjectionDim);
      deltaX->segment(covStateIndex, minProjectionDim) =
          refStates.at(stateBlockIndex).parameterEstimate - cameraIntrinsics;
      covStateIndex += minProjectionDim;
      ++stateBlockIndex;

      const int distortionCoeffDim = camera_rig_.getDistortionDimen(camIdx);
      Eigen::Map<const Eigen::VectorXd> cameraDistortion(
          refStates.at(stateBlockIndex).parameterBlockPtr->parameters(),
          distortionCoeffDim);
      deltaX->segment(covStateIndex, distortionCoeffDim) =
          refStates.at(stateBlockIndex).parameterEstimate - cameraDistortion;
      covStateIndex += distortionCoeffDim;
      ++stateBlockIndex;
    }

    (*deltaX)[covStateIndex] =
        refStates.at(stateBlockIndex).parameterEstimate[0] -
        refStates.at(stateBlockIndex).parameterBlockPtr->parameters()[0];
    ++covStateIndex;
    ++stateBlockIndex;

    (*deltaX)[covStateIndex] =
        refStates.at(stateBlockIndex).parameterEstimate[0] -
        refStates.at(stateBlockIndex).parameterBlockPtr->parameters()[0];
    ++covStateIndex;
    ++stateBlockIndex;
  }

  for (auto iter = refStates.begin() + stateBlockIndex; iter != refStates.end();
       ++iter) {
    Eigen::Matrix<double, 6, 1> delta_T_WB;
    okvis::ceres::PoseLocalParameterization::minus(
        iter->parameterBlockPtr->parameters(), iter->parameterEstimate.data(),
        delta_T_WB.data());
    deltaX->segment<6>(covStateIndex) = delta_T_WB;
    covStateIndex += 6;
    ++stateBlockIndex;
    ++iter;
    deltaX->segment<3>(covStateIndex) =
        iter->parameterEstimate - Eigen::Map<const Eigen::Vector3d>(
                                      iter->parameterBlockPtr->parameters());
    covStateIndex += 3;
    ++stateBlockIndex;
  }
  deltaX->segment<9>(covStateIndex) = deltaX->head<9>();
  covStateIndex += 9;
  deltaX->segment(covStateIndex, covDim - covStateIndex).setZero();
  // TODO(jhuai): consider point landmarks in the states.
}

void HybridFilter::updateStates(
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& deltaX) {
  const size_t numNavImuCamStates = startIndexOfClonedStatesFast();
  // number of navigation, imu, and camera states in the covariance.
  const size_t numNavImuCamPoseStates =
      numNavImuCamStates + 9 * statesMap_.size();
  CHECK_LT((deltaX.head<9>() - deltaX.segment<9>(numNavImuCamPoseStates - 9))
               .lpNorm<Eigen::Infinity>(),
           1e-8)
      << "Correction to the current states from head and tail should be "
         "identical!";

  std::map<uint64_t, States>::reverse_iterator lastElementIterator =
      statesMap_.rbegin();
  const States& stateInQuestion = lastElementIterator->second;
  uint64_t stateId = stateInQuestion.id;

  // update global states
  std::shared_ptr<ceres::PoseParameterBlock> poseParamBlockPtr =
      std::static_pointer_cast<ceres::PoseParameterBlock>(
          mapPtr_->parameterBlockPtr(stateId));
  kinematics::Transformation T_WS = poseParamBlockPtr->estimate();
  // In effect this amounts to PoseParameterBlock::plus().
  Eigen::Vector3d deltaAlpha = deltaX.segment<3>(3);
  Eigen::Quaterniond deltaq =
      okvis::kinematics::expAndTheta(deltaAlpha);
  T_WS = kinematics::Transformation(
      T_WS.r() + deltaX.head<3>(),
      deltaq * T_WS.q());
  poseParamBlockPtr->setEstimate(T_WS);

  // update imu sensor states
  const int imuIdx = 0;
  uint64_t SBId = stateInQuestion.sensors.at(SensorStates::Imu)
                      .at(imuIdx)
                      .at(ImuSensorStates::SpeedAndBias)
                      .id;
  std::shared_ptr<ceres::SpeedAndBiasParameterBlock> sbParamBlockPtr =
      std::static_pointer_cast<ceres::SpeedAndBiasParameterBlock>(
          mapPtr_->parameterBlockPtr(SBId));
  SpeedAndBiases sb = sbParamBlockPtr->estimate();
  sbParamBlockPtr->setEstimate(sb + deltaX.segment<9>(6));

  updateImuAugmentedStates(
      stateInQuestion, deltaX.segment(15, imu_rig_.getAugmentedMinimalDim(0)));

  updateCameraSensorStates(stateInQuestion, deltaX);

  // Update cloned states except for the last one, the current state,
  // which is already updated early on.
  size_t jack = 0;
  auto finalIter = statesMap_.end();
  --finalIter;

  for (auto iter = statesMap_.begin(); iter != finalIter; ++iter, ++jack) {
    stateId = iter->first;
    size_t qStart = startIndexOfClonedStatesFast() + 3 + kClonedStateMinimalDimen * jack;

    poseParamBlockPtr = std::static_pointer_cast<ceres::PoseParameterBlock>(
        mapPtr_->parameterBlockPtr(stateId));
    T_WS = poseParamBlockPtr->estimate();
    deltaAlpha = deltaX.segment<3>(qStart);
    deltaq = okvis::kinematics::expAndTheta(deltaAlpha);
    T_WS = kinematics::Transformation(
        T_WS.r() + deltaX.segment<3>(qStart - 3),
        deltaq * T_WS.q());
    poseParamBlockPtr->setEstimate(T_WS);

    SBId = iter->second.sensors.at(SensorStates::Imu)
               .at(imuIdx)
               .at(ImuSensorStates::SpeedAndBias)
               .id;
    sbParamBlockPtr =
        std::static_pointer_cast<ceres::SpeedAndBiasParameterBlock>(
            mapPtr_->parameterBlockPtr(SBId));
    sb = sbParamBlockPtr->estimate();
    sb.head<3>() += deltaX.segment<3>(qStart + 3);
    sbParamBlockPtr->setEstimate(sb);
  }

  // update feature states, correction is  \delta[\alpha, \beta, \rho], stored
  // states are [\alpha, \beta, 1, \rho]
  int numberLandmarks = 0;
  size_t lkStart = startIndexOfClonedStatesFast() + kClonedStateMinimalDimen * statesMap_.size();
  size_t aStart = lkStart - 3;  // a dummy initialization.
  for (auto iter = mInCovLmIds.begin(), iterEnd = mInCovLmIds.end();
       iter != iterEnd; ++iter, ++numberLandmarks) {
    Eigen::Vector4d homogeneousPoint = iter->estimate();
    aStart = lkStart + 3 * numberLandmarks;
    homogeneousPoint[0] += deltaX[aStart];
    homogeneousPoint[1] += deltaX[aStart + 1];
    switch (pointLandmarkOptions_.landmarkModelId) {
    case msckf::InverseDepthParameterization::kModelId:
      homogeneousPoint[3] += deltaX[aStart + 2];
      break;
    default:
      homogeneousPoint[2] += deltaX[aStart + 2];
      break;
    }
    iter->setEstimate(homogeneousPoint);
  }
  OKVIS_ASSERT_EQ_DBG(Exception, aStart + 3, (size_t)deltaX.rows(),
                      "deltaX size not equal to what's' expected.");

  updateSensorRigs();
}

int HybridFilter::computeStackedJacobianAndResidual(
    Eigen::MatrixXd */*T_H*/, Eigen::Matrix<double, Eigen::Dynamic, 1> */*r_q*/,
    Eigen::MatrixXd */*R_q*/) {
  // for each feature track
  //   if the feature track is to be initialized and marginalized
  //     compute Jacobians and residuals
  // stack the residuals and Jacobians
  // shrink if necessary
  return 0; // return residual dimension.
}

Eigen::Vector4d HybridFilter::anchoredInverseDepthToWorldCoordinates(
    const Eigen::Vector4d &ab1rho, uint64_t anchorStateId,
    size_t anchorCameraId) const {
  okvis::kinematics::Transformation T_WB =
      std::static_pointer_cast<ceres::PoseParameterBlock>(
          mapPtr_->parameterBlockPtr(anchorStateId))
          ->estimate();
  const okvis::kinematics::Transformation T_BCa =
      camera_rig_.getCameraExtrinsic(anchorCameraId);
  Eigen::Vector4d hpW = T_WB * T_BCa * ab1rho;
  double inverseW = 1.0 / hpW[3];
  return hpW * inverseW;
}

void HybridFilter::optimize(size_t /*numIter*/, size_t /*numThreads*/,
                            bool /*verbose*/) {
  // containers of Jacobians of measurements of marginalized features
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, 1>,
      Eigen::aligned_allocator<Eigen::Matrix<double, Eigen::Dynamic, 1>>>
      vr_o;
  std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>
      vH_o;  // each entry has a size say (2n-3)x(13+9m) where n is the number of observations,
  // and m is the number of cloned state variables in the sliding window.
  std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>
      vR_o;  // each entry has a size (2n-3)x(2n-3)

  // containers of Jacobians of measurements of points in the states
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> vr_i;
  std::vector<
      Eigen::Matrix<double, 2, Eigen::Dynamic>,
      Eigen::aligned_allocator<Eigen::Matrix<double, 2, Eigen::Dynamic>>>
      vH_x;  // each entry has a size say 2x(13 + 9m)
  std::vector<
      Eigen::Matrix<double, 2, Eigen::Dynamic>,
      Eigen::aligned_allocator<Eigen::Matrix<double, 2, Eigen::Dynamic>>>
      vH_f;  // each entry has a size 2 x 3k where k is the number of in state landmarks.
  std::vector<Eigen::Matrix2d,
              Eigen::aligned_allocator<Eigen::Matrix<double, 2, 2>>>
      vR_i;

  OKVIS_ASSERT_EQ_DBG(
      Exception, (size_t)covariance_.rows(),
      startIndexOfClonedStatesFast() +
          kClonedStateMinimalDimen * statesMap_.size() + 3 * mInCovLmIds.size(),
      "Inconsistent rows of covariance matrix and number of states");

  int numCamPosePointStates = cameraParamPoseAndLandmarkMinimalDimen();
  size_t dimH_o[2] = {0, numCamPosePointStates - 3 * mInCovLmIds.size() - kClonedStateMinimalDimen};
  size_t nMarginalizedFeatures =
      0;  // features not in state and not tracked in current frame
  size_t nInStateFeatures = 0;  // features in state and tracked now

  const uint64_t currFrameId = currentFrameId();
  size_t navAndImuParamsDim = navStateAndImuParamsMinimalDim();
  Eigen::MatrixXd variableCov = covariance_.block(
      navAndImuParamsDim, navAndImuParamsDim,
      dimH_o[1],
      dimH_o[1]); // covariance block for camera and pose state copies except for the current pose state is used for MSCKF features.
  Eigen::MatrixXd variableCov2 = covariance_.block(
      navAndImuParamsDim, navAndImuParamsDim,
      numCamPosePointStates, numCamPosePointStates);  // covariance block for camera and pose state copies including the current pose state is used for SLAM features.

  for (okvis::PointMap::iterator it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
    it->second.updateStatus(currFrameId, pointLandmarkOptions_.minTrackLengthForMsckf,
                            pointLandmarkOptions_.minTrackLengthForSlam);

    if (it->second.status.measurementType == FeatureTrackStatus::kMsckfTrack) {
      msckf::PointLandmark landmark;
      Eigen::MatrixXd H_oi;                           //(2n-3, dimH_o[1])
      Eigen::Matrix<double, Eigen::Dynamic, 1> r_oi;  //(2n-3, 1)
      Eigen::MatrixXd R_oi;                           //(2n-3, 2n-3)
      bool isValidJacobian =
          featureJacobian(it->second, &landmark, H_oi, r_oi, R_oi);
      if (!isValidJacobian) {
        it->second.setMeasurementFate(FeatureTrackStatus::kComputingJacobiansFailed);
        continue;
      }
      if (!FilterHelper::gatingTest(H_oi, r_oi, R_oi, variableCov)) {
        it->second.setMeasurementFate(FeatureTrackStatus::kPotentialOutlier);
        continue;
      }
      it->second.status.measurementFate = FeatureTrackStatus::kSuccessful;
      vr_o.push_back(r_oi);
      vR_o.push_back(R_oi);
      vH_o.push_back(H_oi);
      dimH_o[0] += r_oi.rows();
      ++nMarginalizedFeatures;
    } else if (it->second.status.measurementType == FeatureTrackStatus::kSlamObservation) {
      // compute residual and Jacobian for a observed point which is in the states
      Eigen::Matrix<double, -1, 1> r_i;
      Eigen::MatrixXd H_x;
      Eigen::MatrixXd H_f;
      Eigen::MatrixXd R_i;
      bool isValidJacobian =
          slamFeatureJacobian(it->second, H_x, r_i, R_i, H_f);
      if (!isValidJacobian) {
        it->second.setMeasurementFate(FeatureTrackStatus::kComputingJacobiansFailed);
        continue;
      }
      Eigen::MatrixXd H_xf(H_x.rows(), H_x.cols() + H_f.cols());
      H_xf.leftCols(H_x.cols()) = H_x;
      H_xf.rightCols(H_f.cols()) = H_f;
      if (!FilterHelper::gatingTest(H_xf, r_i, R_i, variableCov2)) {
        it->second.setMeasurementFate(FeatureTrackStatus::kPotentialOutlier);
        continue;
      }
      it->second.status.measurementFate = FeatureTrackStatus::kSuccessful;
      vr_i.push_back(r_i);
      vH_x.push_back(H_x);
      vH_f.push_back(H_f);
      vR_i.push_back(R_i);
      ++nInStateFeatures;
    }
  }  // every landmark


  // update with MSCKF features
  if (dimH_o[0] > 0) {
    Eigen::MatrixXd H_o(dimH_o[0], dimH_o[1]);
    Eigen::Matrix<double, Eigen::Dynamic, 1> r_o(dimH_o[0], 1);
    Eigen::MatrixXd R_o = Eigen::MatrixXd::Zero(dimH_o[0], dimH_o[0]);
    FilterHelper::stackJacobianAndResidual(vH_o, vr_o, vR_o, &H_o, &r_o, &R_o);
    Eigen::MatrixXd T_H, R_q;  // residual, Jacobian, and noise covariance after projecting to the column space of H_o.
    Eigen::Matrix<double, -1, 1> r_q;
    FilterHelper::shrinkResidual(H_o, r_o, R_o, &T_H, &r_q, &R_q);

    DefaultEkfUpdater updater(covariance_, navAndImuParamsDim, dimH_o[1]);
    computeKalmanGainTimer.start();
    Eigen::Matrix<double, Eigen::Dynamic, 1> deltaX =
        updater.computeCorrection(T_H, r_q, R_q);
    computeKalmanGainTimer.stop();
    updateStatesTimer.start();
    updateStates(deltaX);
    updateStatesTimer.stop();
    updateCovarianceTimer.start();
    updater.updateCovariance(&covariance_);
    updateCovarianceTimer.stop();
  }

  // update with SLAM features
  if (nInStateFeatures > 0) {
    const size_t obsRows = 2 * nInStateFeatures;
    const size_t numPointStates = 3 * mInCovLmIds.size();
    Eigen::MatrixXd H_all(obsRows, numCamPosePointStates);
    H_all.block(0, numCamPosePointStates - numPointStates, obsRows, numPointStates).setZero();
    Eigen::Matrix<double, Eigen::Dynamic, 1> r_all(obsRows, 1);
    Eigen::MatrixXd R_all = Eigen::MatrixXd::Zero(obsRows, obsRows);
    size_t startRow = 0u;
    for (size_t jack = 0; jack < nInStateFeatures; ++jack) {
      H_all.block(startRow, 0, 2, numCamPosePointStates - numPointStates) =
          vH_x[jack];
      H_all.block(startRow, numCamPosePointStates - numPointStates, 2,
                  numPointStates) = vH_f[jack];
      r_all.block<2, 1>(startRow, 0) = vr_i[jack];
      R_all.block<2, 2>(startRow, startRow) = vR_i[jack];
      startRow += 2;
    }

    DefaultEkfUpdater updater(covariance_, navAndImuParamsDim, numCamPosePointStates);
    computeKalmanGainTimer.start();
    Eigen::Matrix<double, Eigen::Dynamic, 1> deltaX =
        updater.computeCorrection(H_all, r_all, R_all);
    computeKalmanGainTimer.stop();
    updateStatesTimer.start();
    updateStates(deltaX);
    updateStatesTimer.stop();
    updateCovarianceTimer.start();
    updater.updateCovariance(&covariance_);
    updateCovarianceTimer.stop();
  }

  // initialize SLAM features and update state
  initializeLandmarksInFilter();

  // And update landmark positions which is only necessary when
  // (1) landmark coordinates are used to predict the points projection in
  // new frames OR (2) to visualize the points.
  for (auto it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
    if (it->second.status.inState) {
      // SLAM features
      it->second.quality = 1.0;
      uint64_t toFind = it->first;
      auto landmarkIter = std::find_if(
          mInCovLmIds.begin(), mInCovLmIds.end(),
          [toFind](const okvis::ceres::HomogeneousPointParameterBlock &x) {
            return x.id() == toFind;
          });
      if (it->second.anchorStateId == 0u) {
        it->second.pointHomog = landmarkIter->estimate();
      } else {
        it->second.pointHomog = anchoredInverseDepthToWorldCoordinates(
            landmarkIter->estimate(), it->second.anchorStateId,
            it->second.anchorCameraId);
      }
      double depth = it->second.pointHomog[2];
      if (depth < 1e-6) {
        it->second.quality = 0.0;
      }
    }
  }
}

void HybridFilter::initializeLandmarksInFilter() {
  updateLandmarksTimer.start();
  Eigen::Matrix<double, Eigen::Dynamic, 1> r_i;
  Eigen::MatrixXd H_i;
  Eigen::MatrixXd R_i;
  Eigen::Matrix<double, Eigen::Dynamic, 3> H_fi;
  Eigen::MatrixXd Q2;  // nullspace of H_fi
  Eigen::MatrixXd Q1;  // column space of H_fi

  Eigen::Matrix<double, Eigen::Dynamic, 1> z_o;
  Eigen::MatrixXd H_o;
  Eigen::MatrixXd R_o;

  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, 1>,
      Eigen::aligned_allocator<Eigen::Matrix<double, Eigen::Dynamic, 1>>>
      vz_1;
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, 1>,
      Eigen::aligned_allocator<Eigen::Matrix<double, Eigen::Dynamic, 1>>>
      vz_o;
  std::vector<
      Eigen::Matrix<double, 3, Eigen::Dynamic>,
      Eigen::aligned_allocator<Eigen::Matrix<double, 3, Eigen::Dynamic>>>
      vH_1;
  std::vector<Eigen::Matrix<double, 3, 3>,
              Eigen::aligned_allocator<Eigen::Matrix<double, 3, 3>>>
      vH_2;
  std::vector<Eigen::Matrix<double, 3, 3>,
              Eigen::aligned_allocator<Eigen::Matrix<double, 3, 3>>>
      vR_1;

  std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>
      vH_o;  // each entry has a size say (2n-3)x(13+9m) where n is the number of observations,
  // and m is the number of cloned state variables in the sliding window.
  std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>
      vR_o;  // each entry has a size (2n-3)x(2n-3)

  size_t totalObsDim = 0;  // total dimensions of all features' observations
  const uint64_t currFrameId = currentFrameId();
  size_t navAndImuParamsDim = navStateAndImuParamsMinimalDim();
  size_t numCamPosePointStates = cameraParamPoseAndLandmarkMinimalDimen();
  const size_t numCamPoseStates =
      numCamPosePointStates - 3 * mInCovLmIds.size();
  Eigen::MatrixXd variableCov = covariance_.block(
      navAndImuParamsDim, navAndImuParamsDim, numCamPoseStates,
      numCamPoseStates); // covariance of camera and pose copy states

  Eigen::AlignedVector<okvis::ceres::HomogeneousPointParameterBlock> landmarksToAdd;
  landmarksToAdd.reserve(20);
  for (okvis::PointMap::iterator pit = landmarksMap_.begin();
       pit != landmarksMap_.end(); ++pit) {
    if (pit->second.status.measurementType == FeatureTrackStatus::kSlamInitialization) {
      msckf::PointLandmark pointLandmark;
      bool isValidJacobian =
          featureJacobian(pit->second, &pointLandmark, H_i, r_i, R_i, &H_fi);
      if (!isValidJacobian) {  // This feature will be removed and marginalized as a MSCKF feature in the next optimization step.
        pit->second.setMeasurementFate(FeatureTrackStatus::kComputingJacobiansFailed);
        continue;
      }
      Eigen::Vector4d homogeneousPoint =
          Eigen::Map<Eigen::Vector4d>(pointLandmark.data(), 4);

      vio::leftNullspaceAndColumnSpace(H_fi, &Q2, &Q1);
      z_o = Q2.transpose() * r_i;
      H_o = Q2.transpose() * H_i;
      R_o = Q2.transpose() * R_i * Q2;

      if (!FilterHelper::gatingTest(H_o, z_o, R_o, variableCov)) {
        pit->second.setMeasurementFate(FeatureTrackStatus::kPotentialOutlier);
        continue;
      }

      if (pointLandmarkOptions_.landmarkModelId ==
          msckf::InverseDepthParameterization::kModelId) {
        // use the earliest camera frame as the anchor image.
        int anchorCameraId = -1;
        for (auto observationIter = pit->second.observations.rbegin();
             observationIter != pit->second.observations.rend();
             ++observationIter) {
          if (observationIter->first.frameId == currFrameId) {
            anchorCameraId = observationIter->first.cameraIndex;
          } else {
            break;
          }
        }
        OKVIS_ASSERT_NE(Exception, anchorCameraId, -1,
                        "Anchor image not found!");
        pit->second.anchorStateId = currFrameId;
        pit->second.anchorCameraId = anchorCameraId;
      }

      pit->second.status.inState = true;
      pit->second.setMeasurementFate(FeatureTrackStatus::kSuccessful);
      landmarksToAdd.emplace_back(homogeneousPoint, pit->first);

      vz_1.push_back(Q1.transpose() * r_i);
      vz_o.push_back(z_o);
      vH_1.push_back(Q1.transpose() * H_i);
      vH_2.push_back(Q1.transpose() * H_fi);
      vH_o.push_back(H_o);
      vR_o.push_back(R_o);
      vR_1.push_back(Q1.transpose() * R_i * Q1);
      totalObsDim += H_i.rows();
    }
  }

  // augment and update the covariance matrix.
  size_t nNewFeatures = landmarksToAdd.size();
//  LOG(INFO) << "Initializing " << nNewFeatures << " landmarks into the state vector of "
//            << mInCovLmIds.size() << " landmarks!";
  if (nNewFeatures) {
    Eigen::MatrixXd H_o(totalObsDim - 3 * nNewFeatures, numCamPoseStates);
    Eigen::MatrixXd H_1(3 * nNewFeatures, numCamPoseStates);
    Eigen::MatrixXd invH_2 =
        Eigen::MatrixXd::Zero(3 * nNewFeatures, 3 * nNewFeatures);
    Eigen::MatrixXd R_o = Eigen::MatrixXd::Zero(
        totalObsDim - 3 * nNewFeatures, totalObsDim - 3 * nNewFeatures);
    Eigen::MatrixXd R_1 =
        Eigen::MatrixXd::Zero(3 * nNewFeatures, 3 * nNewFeatures);
    Eigen::Matrix<double, Eigen::Dynamic, 1> z_1(nNewFeatures * 3, 1);
    Eigen::Matrix<double, Eigen::Dynamic, 1> z_o(
        totalObsDim - nNewFeatures * 3, 1);

    size_t startRow = 0u;
    for (size_t featureIndex = 0u; featureIndex < nNewFeatures; ++featureIndex) {
      H_o.block(startRow, 0, vH_o[featureIndex].rows(), numCamPoseStates) =
          vH_o[featureIndex];
      H_1.block(3 * featureIndex, 0, 3, numCamPoseStates) = vH_1[featureIndex];
      invH_2.block<3, 3>(3 * featureIndex, 3 * featureIndex) =
          vH_2[featureIndex].inverse();
      R_o.block(startRow, startRow, vH_o[featureIndex].rows(),
                vH_o[featureIndex].rows()) = vR_o[featureIndex];
      R_1.block<3, 3>(3 * featureIndex, 3 * featureIndex) = vR_1[featureIndex];
      z_1.segment<3>(3 * featureIndex) = vz_1[featureIndex];
      z_o.segment(startRow, vH_o[featureIndex].rows()) = vz_o[featureIndex];
      startRow += vH_o[featureIndex].rows();
    }

    // initialize features into the state vector with z_1, R_1, H_1, H_2.
    // also correct the landmark parameters
    updateCovarianceTimer.start();
    int covDim = covariance_.rows();
    Eigen::MatrixXd Paug(covDim + nNewFeatures * 3,
                         covDim + nNewFeatures * 3);
    Paug.topLeftCorner(covDim, covDim) = covariance_;
    Eigen::MatrixXd invH2H1 = invH_2 * H_1;
    Paug.block(covDim, 0, 3 * nNewFeatures, covDim) =
        -invH2H1 * Paug.block(navAndImuParamsDim, 0,
                              numCamPoseStates, covDim);
    Paug.block(0, covDim, covDim, 3 * nNewFeatures) =
        Paug.block(covDim, 0, 3 * nNewFeatures, covDim).transpose();
    Paug.bottomRightCorner(3 * nNewFeatures, 3 * nNewFeatures) =
        -Paug.block(covDim, navAndImuParamsDim,
                    3 * nNewFeatures, numCamPoseStates) *
            invH2H1.transpose() +
        invH_2 * R_1 * invH_2.transpose();
    covariance_ = Paug;
    updateCovarianceTimer.stop();

    Eigen::Matrix<double, Eigen::Dynamic, 1> deltaLandmarks = invH_2 * z_1;
    // TODO(jhuai): use polymorphism for variables.
    switch (pointLandmarkOptions_.landmarkModelId) {
      case msckf::InverseDepthParameterization::kModelId:
        for (size_t j = 0u; j < nNewFeatures; ++j) {
          Eigen::Vector4d ab1rho = landmarksToAdd[j].estimate();
          ab1rho[0] += deltaLandmarks[j * 3];
          ab1rho[1] += deltaLandmarks[j * 3 + 1];
          ab1rho[3] += deltaLandmarks[j * 3 + 2];
          landmarksToAdd[j].setEstimate(ab1rho);
        }
        break;
      case msckf::HomogeneousPointParameterization::kModelId:
        for (size_t j = 0u; j < nNewFeatures; ++j) {
            Eigen::Vector4d pointW = landmarksToAdd[j].estimate();
            pointW[0] += deltaLandmarks[j * 3];
            pointW[1] += deltaLandmarks[j * 3 + 1];
            pointW[2] += deltaLandmarks[j * 3 + 2];
            landmarksToAdd[j].setEstimate(pointW);
        }
        break;
      default:
        break;
    }

    mInCovLmIds.insert(mInCovLmIds.end(), landmarksToAdd.begin(),
                       landmarksToAdd.end());

    // TODO(jhuai): add homogeneous point parameter block to satisfy setLandmark
    // which potentially needs point blocks to be registered in the map.
    // This would be unnecessary if the frontend refers to the backendInterface
    // instead of the concrete Estimator class.
    for (auto landmark : landmarksToAdd) {
      uint64_t landmarkId = landmark.id();
      const MapPoint &mp = landmarksMap_.at(landmarkId);
      Eigen::Vector4d hpW;
      if (mp.anchorStateId == 0u) {
        hpW = landmark.estimate();
      } else {
        hpW = anchoredInverseDepthToWorldCoordinates(
            landmark.estimate(), mp.anchorStateId, mp.anchorCameraId);
      }
      std::shared_ptr<okvis::ceres::HomogeneousPointParameterBlock>
          pointParameterBlock(
              new okvis::ceres::HomogeneousPointParameterBlock(hpW, landmark.id()));
      if (!mapPtr_->addParameterBlock(pointParameterBlock,
                                      okvis::ceres::Map::HomogeneousPoint)) {
        LOG(WARNING) << "Failed to add block for landmark " << landmark.id();
        continue;
      }
    }

    // update the state vector with z_o, R_o, H_o.
    DefaultEkfUpdater updater(covariance_, navAndImuParamsDim, numCamPoseStates);
    computeKalmanGainTimer.start();
    Eigen::Matrix<double, Eigen::Dynamic, 1> deltaX =
        updater.computeCorrection(H_o, z_o, R_o);
    computeKalmanGainTimer.stop();
    updateStatesTimer.start();
    updateStates(deltaX);
    updateStatesTimer.stop();
    updateCovarianceTimer.start();
    updater.updateCovariance(&covariance_);
    updateCovarianceTimer.stop();
  }
  updateLandmarksTimer.stop();
}

size_t HybridFilter::gatherMapPointObservations(
    const MapPoint& mp,
    msckf::PointSharedData* pointDataPtr,
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>*
        obsDirections,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>*
        obsInPixel,
    std::vector<double>* imageNoiseStd,
    std::vector<std::pair<uint64_t, int>>* orderedBadFrameIds,
    RetrieveObsSeqType seqType) const {
  obsDirections->clear();
  obsInPixel->clear();
  imageNoiseStd->clear();
  const std::map<okvis::KeypointIdentifier, uint64_t>* observations =
      &mp.observations;
  std::map<okvis::KeypointIdentifier, uint64_t> obsPair;
  std::map<okvis::KeypointIdentifier, uint64_t>::const_reverse_iterator riter =
      mp.observations.rbegin();
  std::map<okvis::KeypointIdentifier, uint64_t>::const_reverse_iterator
      second_riter = ++mp.observations.rbegin();

  switch (seqType) {
  case HEAD_TAIL:
      obsPair[observations->begin()->first] = observations->begin()->second;
      obsPair[riter->first] = riter->second;
      observations = &obsPair;
      break;
  case LATEST_TWO:
      obsPair[riter->first] = riter->second;
      obsPair[second_riter->first] = second_riter->second;
      observations = &obsPair;
      break;
  case ENTIRE_TRACK:
  default:
      break;
  }

  for (auto itObs = observations->begin(), iteObs = observations->end();
       itObs != iteObs; ++itObs) {
    uint64_t poseId = itObs->first.frameId;
    Eigen::Vector2d measurement;
    auto multiFrameIter = multiFramePtrMap_.find(poseId);
//    OKVIS_ASSERT_TRUE(Exception, multiFrameIter != multiFramePtrMap_.end(), "multiframe not found");
    okvis::MultiFramePtr multiFramePtr = multiFrameIter->second;
    multiFramePtr->getKeypoint(itObs->first.cameraIndex,
                               itObs->first.keypointIndex, measurement);
    okvis::Time imageTimestamp = multiFramePtr->timestamp(itObs->first.cameraIndex);
    // use the latest estimates for camera intrinsic parameters
    Eigen::Vector3d backProjectionDirection;
    std::shared_ptr<const cameras::CameraBase> cameraGeometry =
        camera_rig_.getCameraGeometry(itObs->first.cameraIndex);
    bool validDirection = cameraGeometry->backProject(measurement, &backProjectionDirection);
    if (!validDirection) {
        orderedBadFrameIds->emplace_back(poseId, itObs->first.cameraIndex);
        continue;
    }
    obsDirections->push_back(backProjectionDirection);

    obsInPixel->push_back(measurement);

    double kpSize = 1.0;
    multiFramePtr->getKeypointSize(itObs->first.cameraIndex,
                                   itObs->first.keypointIndex, kpSize);

    imageNoiseStd->push_back(kpSize / 8);
    imageNoiseStd->push_back(kpSize / 8);

    std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr;
    getGlobalStateParameterBlockPtr(poseId, GlobalStates::T_WS, parameterBlockPtr);
    uint32_t imageHeight = cameraGeometry->imageHeight();
    double kpN = measurement[1] / imageHeight - 0.5;
    pointDataPtr->addKeypointObservation(
          itObs->first, parameterBlockPtr, kpN, imageTimestamp);
  }
  return pointDataPtr->numObservations();
}

bool HybridFilter::hasLowDisparity(
    const std::vector<Eigen::Vector3d,
                      Eigen::aligned_allocator<Eigen::Vector3d>>& obsDirections,
    const std::vector<
        okvis::kinematics::Transformation,
        Eigen::aligned_allocator<okvis::kinematics::Transformation>>& T_WSs,
    const Eigen::AlignedVector<okvis::kinematics::Transformation>& T_BCs,
    const std::vector<size_t>& camIndices) const {
  Eigen::VectorXd intrinsics;
  camera_rig_.getCameraGeometry(0)->getIntrinsics(intrinsics);
  double focalLength = intrinsics[0];
  double keypointAStdDev = 0.8 * FLAGS_epipolar_sigma_keypoint_size / 12.0;
  const double fourthRoot2 = 1.1892071150;
  double raySigma = fourthRoot2 * keypointAStdDev / focalLength;
  Eigen::Vector3d rayA_inA = obsDirections.front().normalized();
  Eigen::Vector3d rayB_inB = obsDirections.back().normalized();
  size_t frontCamIdx = camIndices.front();
  size_t backCamIdx = camIndices.back();
  Eigen::Vector3d rayB_inA =
      (T_WSs.front().C() * T_BCs.at(frontCamIdx).C()).transpose() *
      T_WSs.back().C() * T_BCs.at(backCamIdx).C() * rayB_inB;
  return okvis::triangulation::hasLowDisparity(rayA_inA, rayB_inB, rayB_inA,
                                               raySigma);
}

bool HybridFilter::isPureRotation(const MapPoint& mp) const {
  const std::map<okvis::KeypointIdentifier, uint64_t>& observations =
      mp.observations;
  const okvis::KeypointIdentifier& headKpi = observations.begin()->first;
  const okvis::KeypointIdentifier& tailKpi = observations.rbegin()->first;
  const int camIdx = 0;
  std::vector<uint64_t> relativeFrameIds(2);
  std::vector<RelativeMotionType> relativeMotionTypes(2);
  auto headIter = multiFramePtrMap_.find(headKpi.frameId);
  headIter->second->getRelativeMotion(
      camIdx, &relativeFrameIds[0], &relativeMotionTypes[0]);
  auto tailIter = multiFramePtrMap_.find(tailKpi.frameId);
  tailIter->second->getRelativeMotion(
      camIdx, &relativeFrameIds[1], &relativeMotionTypes[1]);
  // TODO(jhuai): do we need to make it more complex?
  return relativeMotionTypes[1] == ROTATION_ONLY;
}

void HybridFilter::propagatePoseAndVelocityForMapPoint(
    msckf::PointSharedData* pointDataPtr) const {
  std::vector<std::pair<uint64_t, size_t>> frameIds = pointDataPtr->frameIds();
  int observationIndex = 0;
  for (const std::pair<uint64_t, size_t>& frameAndCameraIndex : frameIds) {
    uint64_t frameId = frameAndCameraIndex.first;
    auto statesIter = statesMap_.find(frameId);
    pointDataPtr->setImuInfo(observationIndex, statesIter->second.timestamp,
                             statesIter->second.imuReadingWindow,
                             statesIter->second.linearizationPoint);

    std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr;
    const int imuIdx = 0;
    getSensorStateParameterBlockPtr(frameId, imuIdx, SensorStates::Imu,
                                    ImuSensorStates::SpeedAndBias,
                                    parameterBlockPtr);
    pointDataPtr->setVelocityParameterBlockPtr(observationIndex,
                                               parameterBlockPtr);
    ++observationIndex;
  }

  std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>
      cameraDelayParameterPtrs;
  std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>
      cameraReadoutTimeParameterPtrs;
  getCameraTimeParameterPtrs(&cameraDelayParameterPtrs,
                             &cameraReadoutTimeParameterPtrs);
  pointDataPtr->setCameraTimeParameterPtrs(cameraDelayParameterPtrs,
                                           cameraReadoutTimeParameterPtrs);

  std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>
      imuAugmentedParameterPtrs = getImuAugmentedParameterPtrs();
  pointDataPtr->setImuAugmentedParameterPtrs(imuAugmentedParameterPtrs,
                                             &imuParametersVec_.at(0));
  pointDataPtr->computePoseAndVelocityAtObservation();
}

msckf::TriangulationStatus HybridFilter::triangulateAMapPoint(
    const MapPoint& mp,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
        obsInPixel,
    msckf::PointLandmark& pointLandmark,
    std::vector<double>& imageNoiseStd,
    msckf::PointSharedData* pointDataPtr,
    std::vector<uint64_t>* orderedCulledFrameIds,
    bool checkDisparity) const {
  triangulateTimer.start();

  // each entry is undistorted coordinates in image plane at
  // z=1 in the specific camera frame, [\bar{x},\bar{y},1]
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      obsDirections;
  std::vector<std::pair<uint64_t, int>> badObservationIdentifiers;
  size_t numObs = gatherMapPointObservations(
      mp, pointDataPtr, &obsDirections, &obsInPixel,
      &imageNoiseStd, &badObservationIdentifiers);

  msckf::TriangulationStatus status;
  if (numObs < pointLandmarkOptions_.minTrackLengthForMsckf) {
      triangulateTimer.stop();
      status.lackObservations = true;
      return status;
  }

  propagatePoseAndVelocityForMapPoint(pointDataPtr);

  std::vector<okvis::kinematics::Transformation,
              Eigen::aligned_allocator<okvis::kinematics::Transformation>>
      T_WSs = pointDataPtr->poseAtObservationList();

  std::vector<size_t> camIndices = pointDataPtr->cameraIndexList();
  size_t numCameras = camera_rig_.numberCameras();
  Eigen::AlignedVector<okvis::kinematics::Transformation> T_BCs;
  T_BCs.reserve(numCameras);
  for (size_t j = 0u; j < numCameras; ++j) {
    T_BCs.push_back(camera_rig_.getCameraExtrinsic(j));
  }
  if (checkDisparity) {
//    if (isPureRotation(mp)) {
//      triangulateTimer.stop();
//      status.raysParallel = true;
//      return false;
//    }
    if (hasLowDisparity(obsDirections, T_WSs, T_BCs, camIndices)) {
      triangulateTimer.stop();
      status.raysParallel = true;
      return status;
    }
  }

  std::vector<std::pair<uint64_t, size_t>> frameIds = pointDataPtr->frameIds();
  std::vector<okvis::AnchorFrameIdentifier> anchorIds;

  if (orderedCulledFrameIds) {
    msckf::eraseBadObservations(badObservationIdentifiers, orderedCulledFrameIds);
    msckf::decideAnchors(frameIds, *orderedCulledFrameIds, pointLandmark.modelId(),
                         &anchorIds);
  } else {
    msckf::decideAnchors(frameIds, pointLandmark.modelId(), &anchorIds);
  }
  pointDataPtr->setAnchors(anchorIds);
  std::vector<size_t> anchorSeqIds;
  anchorSeqIds.reserve(anchorIds.size());

  for (auto anchorId : anchorIds) {
    size_t anchorObsIndex = anchorId.observationIndex_;
    anchorSeqIds.push_back(anchorObsIndex);
  }

  std::vector<
      okvis::kinematics::Transformation,
      Eigen::aligned_allocator<okvis::kinematics::Transformation>> T_WCa_list;
  T_WCa_list.reserve(anchorIds.size());
  if (pointLandmarkOptions_.anchorAtObservationTime) {
    // anchor body frame's pose is propagated to feature observation time with IMU readings.
    for (auto anchorId : anchorIds) {
      size_t anchorObsIndex = anchorId.observationIndex_;
      T_WCa_list.push_back(T_WSs.at(anchorObsIndex) *
                           T_BCs[camIndices[anchorObsIndex]]);
    }
  } else {
    // anchor body frame is at the state epoch, so its pose does not depend on td and tr any more.
    for (auto anchorId : anchorIds) {
      size_t anchorObsIndex = anchorId.observationIndex_;
      okvis::kinematics::Transformation T_WB =
          pointDataPtr->poseParameterBlockPtr(anchorObsIndex)->estimate();
      T_WCa_list.push_back(T_WB * T_BCs[camIndices[anchorObsIndex]]);
    }
  }
  status = pointLandmark.initialize(T_WSs, obsDirections, T_BCs, T_WCa_list,
                                    camIndices, anchorSeqIds);
  triangulateTimer.stop();
  return status;
}

bool HybridFilter::print(std::ostream& stream) const {
  printNavStateAndBiases(stream, statesMap_.rbegin()->first);
  const States stateInQuestion = statesMap_.rbegin()->second;
  Eigen::Matrix<double, Eigen::Dynamic, 1> extraParams;
  getImuAugmentedStatesEstimate(&extraParams);
  stream << " " << extraParams.transpose().format(kSpaceInitFmt);

  // camera extrinsic parameters.
  size_t numCameras = camera_rig_.numberCameras();
  for (int camIdx = 0; camIdx < (int)numCameras; ++camIdx) {
    uint64_t extrinsicId = stateInQuestion.sensors.at(SensorStates::Camera)
                               .at(camIdx)
                               .at(CameraSensorStates::T_SCi)
                               .id;
    std::shared_ptr<ceres::PoseParameterBlock> extrinsicParamBlockPtr =
        std::static_pointer_cast<ceres::PoseParameterBlock>(
            mapPtr_->parameterBlockPtr(extrinsicId));
    kinematics::Transformation T_XC = extrinsicParamBlockPtr->estimate();
    std::string extrinsicValues;
    ExtrinsicModelToParamsValueString(camera_rig_.getExtrinsicOptMode(camIdx),
                                      T_XC, " ", &extrinsicValues);
    stream << " " << extrinsicValues;
  }
  Eigen::VectorXd cameraParams;
  getCameraCalibrationEstimate(&cameraParams);
  stream << " " << cameraParams.transpose().format(kSpaceInitFmt);

  // stds
  const int stateDim = startIndexOfClonedStatesFast();
  Eigen::Matrix<double, Eigen::Dynamic, 1> variances =
      covariance_.topLeftCorner(stateDim, stateDim).diagonal();
  stream << " " << variances.cwiseSqrt().transpose().format(kSpaceInitFmt);
  return true;
}

void HybridFilter::printTrackLengthHistogram(std::ostream& stream) const {
  stream << "Track length histogram in one test with bins 0,1,2..."
         << std::endl;
  size_t bin = 0;
  for (auto it = mTrackLengthAccumulator.begin();
       it != mTrackLengthAccumulator.end(); ++it, ++bin)
    stream << bin << " " << *it << std::endl;
}

void HybridFilter::getCameraCalibrationEstimate(
    Eigen::Matrix<double, Eigen::Dynamic, 1>* cameraParams) const {
  const uint64_t poseId = statesMap_.rbegin()->first;
  size_t numCameras = camera_rig_.numberCameras();

  for (int camIdx = 0; camIdx < (int)numCameras; ++camIdx) {
    Eigen::VectorXd intrinsic;
    getSensorStateEstimateAs<ceres::EuclideanParamBlock>(
        poseId, camIdx, SensorStates::Camera, CameraSensorStates::Intrinsics,
        intrinsic);
    Eigen::VectorXd distortionCoeffs;
    getSensorStateEstimateAs<ceres::EuclideanParamBlock>(
        poseId, camIdx, SensorStates::Camera, CameraSensorStates::Distortion,
        distortionCoeffs);
    int oldSize = cameraParams->size();
    cameraParams->conservativeResize(
        oldSize + intrinsic.size() + distortionCoeffs.size() + 2, 1);
    cameraParams->segment(oldSize, intrinsic.size()) = intrinsic;
    cameraParams->segment(oldSize + intrinsic.size(), distortionCoeffs.size()) =
        distortionCoeffs;
    double tdEstimate(0), trEstimate(0);
    getSensorStateEstimateAs<ceres::CameraTimeParamBlock>(
        poseId, camIdx, SensorStates::Camera, CameraSensorStates::TD,
        tdEstimate);
    getSensorStateEstimateAs<ceres::CameraTimeParamBlock>(
        poseId, camIdx, SensorStates::Camera, CameraSensorStates::TR,
        trEstimate);
    cameraParams->tail<2>() = Eigen::Vector2d(tdEstimate, trEstimate);
  }
}

void HybridFilter::getCameraTimeParameterPtrs(
    std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>
        *cameraDelayParameterPtrs,
    std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>
        *cameraReadoutTimeParameterPtrs) const {
  size_t numCameras = camera_rig_.numberCameras();
  cameraDelayParameterPtrs->reserve(numCameras);
  cameraReadoutTimeParameterPtrs->reserve(numCameras);

  const States& oneState = statesMap_.rbegin()->second;
  for (size_t camIdx = 0u; camIdx < numCameras; ++camIdx) {
    uint64_t tdId = oneState.sensors.at(SensorStates::Camera)
                        .at(camIdx)
                        .at(CameraSensorStates::TD)
                        .id;
    std::shared_ptr<ceres::ParameterBlock> tdParamBlockPtr =
        mapPtr_->parameterBlockPtr(tdId);
    uint64_t trId = oneState.sensors.at(SensorStates::Camera)
                        .at(camIdx)
                        .at(CameraSensorStates::TR)
                        .id;
    std::shared_ptr<ceres::ParameterBlock> trParamBlockPtr =
        mapPtr_->parameterBlockPtr(trId);
    cameraDelayParameterPtrs->push_back(tdParamBlockPtr);
    cameraReadoutTimeParameterPtrs->push_back(trParamBlockPtr);
  }
}

std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>
HybridFilter::getImuAugmentedParameterPtrs() const {
  const int imuIdx = 0;
  int imuModelId = imu_rig_.getModelId(imuIdx);
  std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>
      imuParameterPtrs;
  const States stateInQuestion = statesMap_.rbegin()->second;
  switch (imuModelId) {
    case Imu_BG_BA::kModelId:
      break;
    case Imu_BG_BA_TG_TS_TA::kModelId: {
      imuParameterPtrs.reserve(3);

      uint64_t TGId = stateInQuestion.sensors.at(SensorStates::Imu)
                          .at(imuIdx)
                          .at(ImuSensorStates::TG)
                          .id;
      imuParameterPtrs.push_back(mapPtr_->parameterBlockPtr(TGId));
      uint64_t TSId = stateInQuestion.sensors.at(SensorStates::Imu)
                          .at(imuIdx)
                          .at(ImuSensorStates::TS)
                          .id;
      imuParameterPtrs.push_back(mapPtr_->parameterBlockPtr(TSId));

      uint64_t TAId = stateInQuestion.sensors.at(SensorStates::Imu)
                          .at(imuIdx)
                          .at(ImuSensorStates::TA)
                          .id;
      imuParameterPtrs.push_back(mapPtr_->parameterBlockPtr(TAId));
    } break;
    default:
      LOG(WARNING) << "Imu Augmented parameter ptrs not implemented for model "
                   << imuModelId;
      break;
  }
  return imuParameterPtrs;
}

void HybridFilter::getImuAugmentedStatesEstimate(
    Eigen::Matrix<double, Eigen::Dynamic, 1>* extraParams) const {
  std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>> TgTsTaPtr =
      getImuAugmentedParameterPtrs();
  okvis::getImuAugmentedStatesEstimate(TgTsTaPtr, extraParams, imu_rig_.getModelId(0));
}

bool HybridFilter::getStateStd(
    Eigen::Matrix<double, Eigen::Dynamic, 1>* stateStd) const {
  const int dim = startIndexOfClonedStatesFast();
  *stateStd = covariance_.topLeftCorner(dim, dim).diagonal().cwiseSqrt();
  return true;
}

HybridFilter::EpipolarMeasurement::EpipolarMeasurement(
    const HybridFilter& filter,
    const uint32_t imageHeight,
    int camIdx, int extrinsicModelId, int minExtrinsicDim, int minProjDim,
    int minDistortDim)
    : filter_(filter),
      camIdx_(camIdx),
      imageHeight_(imageHeight),
      extrinsicModelId_(extrinsicModelId),
      minExtrinsicDim_(minExtrinsicDim),
      minProjDim_(minProjDim),
      minDistortDim_(minDistortDim),
      obsDirection2(2),
      obsInPixel2(2),
      dfj_dXcam2(2) {

}

void HybridFilter::EpipolarMeasurement::prepareTwoViewConstraint(
    const std::vector<Eigen::Vector3d,
                      Eigen::aligned_allocator<Eigen::Vector3d>>& obsDirections,
    const std::vector<Eigen::Vector2d,
                      Eigen::aligned_allocator<Eigen::Vector2d>>& obsInPixels,
    const std::vector<
        Eigen::Matrix<double, 3, Eigen::Dynamic>,
        Eigen::aligned_allocator<Eigen::Matrix<double, 3, Eigen::Dynamic>>>&
        dfj_dXcam,
    const std::vector<int>& index_vec) {
  int j = 0;
  for (auto index : index_vec) {
    obsDirection2[j] = obsDirections[index];
    obsInPixel2[j] = obsInPixels[index];
    dfj_dXcam2[j] = dfj_dXcam[index];
    ++j;
  }
}

bool HybridFilter::EpipolarMeasurement::measurementJacobian(
    std::shared_ptr<const msckf::PointSharedData> pointDataPtr,
    const std::vector<int> observationIndexPair,
    Eigen::Matrix<double, 1, Eigen::Dynamic>* H_xjk,
    std::vector<Eigen::Matrix<double, 1, 3>,
                Eigen::aligned_allocator<Eigen::Matrix<double, 1, 3>>>* H_fjk,
    double* residual) const {
  // compute the head and tail pose, velocity, Jacobians, and covariance
  std::vector<okvis::kinematics::Transformation,
              Eigen::aligned_allocator<okvis::kinematics::Transformation>>
      T_WBtij, lP_T_WBtij;  // lp is short for linearization point
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      lP_v_WBtij;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      omega_WBtij;
  double dtij_dtr[2];
  double featureDelay[2];
  for (int j = 0; j < 2; ++j) {
    dtij_dtr[j] = pointDataPtr->normalizedRow(observationIndexPair[j]);
    featureDelay[j] = pointDataPtr->normalizedFeatureTime(observationIndexPair[j]);
    T_WBtij.emplace_back(pointDataPtr->T_WBtij(observationIndexPair[j]));
    omega_WBtij.emplace_back(pointDataPtr->omega_Btij(observationIndexPair[j]));
    lP_T_WBtij.emplace_back(pointDataPtr->T_WBtij_ForJacobian(observationIndexPair[j]));
    lP_v_WBtij.emplace_back(pointDataPtr->v_WBtij_ForJacobian(observationIndexPair[j]));
  }

  // compute residual
  okvis::kinematics::Transformation T_BC0 = filter_.camera_rig_.getCameraExtrinsic(camIdx_);
  okvis::kinematics::Transformation T_Ctij_Ctik =
      (T_WBtij[0] * T_BC0).inverse() * (T_WBtij[1] * T_BC0);
  okvis::kinematics::Transformation lP_T_Ctij_Ctik =
      (lP_T_WBtij[0] * T_BC0).inverse() * (lP_T_WBtij[1] * T_BC0);
  EpipolarJacobian epj(T_Ctij_Ctik.C(), T_Ctij_Ctik.r(), obsDirection2[0],
                       obsDirection2[1]);
  *residual = -epj.evaluate();  // observation is 0

  // compute Jacobians for camera parameters
  EpipolarJacobian epj_lp(lP_T_Ctij_Ctik.C(), lP_T_Ctij_Ctik.r(),
                          obsDirection2[0], obsDirection2[1]);
  Eigen::Matrix<double, 1, 3> de_dfj[2];
  epj_lp.de_dfj(&de_dfj[0]);
  epj_lp.de_dfk(&de_dfj[1]);
  Eigen::Matrix<double, 1, 3> de_dtheta_Ctij_Ctik, de_dt_Ctij_Ctik;
  epj_lp.de_dtheta_CjCk(&de_dtheta_Ctij_Ctik);
  epj_lp.de_dt_CjCk(&de_dt_Ctij_Ctik);
  RelativeMotionJacobian rmj_lp(T_BC0, lP_T_WBtij[0], lP_T_WBtij[1]);
  Eigen::Matrix<double, 3, 3> dtheta_dtheta_BC;
  Eigen::Matrix<double, 3, 3> dp_dtheta_BC;
  Eigen::Matrix<double, 3, 3> dp_dt_BC;
  Eigen::Matrix<double, 3, 3> dp_dt_CB;

  Eigen::Matrix<double, 1, Eigen::Dynamic> de_dExtrinsic;
  switch (extrinsicModelId_) {
    case Extrinsic_p_CB::kModelId:
      rmj_lp.dp_dt_CB(&dp_dt_CB);
      de_dExtrinsic = de_dt_Ctij_Ctik * dp_dt_CB;
      break;
    case Extrinsic_p_BC_q_BC::kModelId:
    default:
      rmj_lp.dtheta_dtheta_BC(&dtheta_dtheta_BC);
      rmj_lp.dp_dtheta_BC(&dp_dtheta_BC);
      rmj_lp.dp_dt_BC(&dp_dt_BC);
      de_dExtrinsic.resize(1, 6);
      de_dExtrinsic.head<3>() = de_dt_Ctij_Ctik * dp_dt_BC;
      de_dExtrinsic.tail<3>() = de_dt_Ctij_Ctik * dp_dtheta_BC +
                                de_dtheta_Ctij_Ctik * dtheta_dtheta_BC;
      break;
  }
  Eigen::Matrix<double, 1, Eigen::Dynamic> de_dxcam =
      de_dfj[0] * dfj_dXcam2[0] + de_dfj[1] * dfj_dXcam2[1];

  // compute Jacobians for time parameters
  Eigen::Matrix<double, 3, 3> dtheta_dtheta_GBtij[2];
  Eigen::Matrix<double, 3, 3> dp_dt_GBtij[2];
  Eigen::Matrix<double, 3, 3> dp_dtheta_GBtij[2];

  rmj_lp.dtheta_dtheta_GBj(&dtheta_dtheta_GBtij[0]);
  rmj_lp.dtheta_dtheta_GBk(&dtheta_dtheta_GBtij[1]);

  rmj_lp.dp_dtheta_GBj(&dp_dtheta_GBtij[0]);
  rmj_lp.dp_dtheta_GBk(&dp_dtheta_GBtij[1]);

  rmj_lp.dp_dt_GBj(&dp_dt_GBtij[0]);
  rmj_lp.dp_dt_GBk(&dp_dt_GBtij[1]);

  Eigen::Matrix<double, 3, 1> dtheta_GBtij_dtij[2];
  Eigen::Matrix<double, 3, 1> dt_GBtij_dtij[2];
  for (int j = 0; j < 2; ++j) {
    dtheta_GBtij_dtij[j] = lP_T_WBtij[j].C() * omega_WBtij[j];
    dt_GBtij_dtij[j] = lP_v_WBtij[j];
  }

  double de_dtj[2];
  for (int j = 0; j < 2; ++j) {
    Eigen::Matrix<double, 1, 1> de_dtj_eigen =
        de_dtheta_Ctij_Ctik * (dtheta_dtheta_GBtij[j] * dtheta_GBtij_dtij[j]) +
        de_dt_Ctij_Ctik * (dp_dt_GBtij[j] * dt_GBtij_dtij[j] +
                           dp_dtheta_GBtij[j] * dtheta_GBtij_dtij[j]);
    de_dtj[j] = de_dtj_eigen[0];
  }
  double de_dtd = de_dtj[0] + de_dtj[1];
  double de_dtr = de_dtj[0] * dtij_dtr[0] + de_dtj[1] * dtij_dtr[1];

  // Jacobians for motion
  Eigen::Matrix<double, 1, 3> de_dp_GBtj[2];
  Eigen::Matrix<double, 1, 3> de_dtheta_GBtj[2];
  Eigen::Matrix<double, 1, 3> de_dv_GBtj[2];

  // TODO(jhuai): let the IMU propagation function provides the Jacobians
  Eigen::Matrix3d dtheta_GBtij_dtheta_GBtj[2];
  Eigen::Matrix<double, 3, 3> dt_GBtij_dt_GBtj[2];
  Eigen::Matrix3d dt_GBtij_dv_GBtj[2];
  for (int j = 0; j < 2; ++j) {
    dtheta_GBtij_dtheta_GBtj[j].setIdentity();
    dt_GBtij_dt_GBtj[j].setIdentity();
    dt_GBtij_dv_GBtj[j] = Eigen::Matrix3d::Identity() * featureDelay[j];
  }

  for (int j = 0; j < 2; ++j) {
    de_dp_GBtj[j] = de_dt_Ctij_Ctik * dp_dt_GBtij[j] * dt_GBtij_dt_GBtj[j];
    de_dtheta_GBtj[j] = (de_dtheta_Ctij_Ctik * dtheta_dtheta_GBtij[j] +
                         de_dt_Ctij_Ctik * dp_dtheta_GBtij[j]) *
                        dtheta_GBtij_dtheta_GBtj[j];
    de_dv_GBtj[j] = de_dt_Ctij_Ctik * dp_dt_GBtij[j] * dt_GBtij_dv_GBtj[j];
  }

  // assemble the Jacobians
  H_xjk->setZero();

  if (minExtrinsicDim_ > 0) {
    H_xjk->topLeftCorner(1, minExtrinsicDim_) = de_dExtrinsic;
  }
  if (minProjDim_ +  minDistortDim_ > 0) {
    H_xjk->block(0, minExtrinsicDim_, 1, minProjDim_ + minDistortDim_) = de_dxcam;
  }
  int startIndex = minExtrinsicDim_ + minProjDim_ + minDistortDim_;
  (*H_xjk)(startIndex) = de_dtd;
  startIndex += 1;
  (*H_xjk)(startIndex) = de_dtr;

  size_t cameraParamStartIndex =
      filter_.startIndexOfCameraParamsFast(filter_.kMainCameraIndex);
  for (int j = 0; j < 2; ++j) {
    uint64_t poseId = pointDataPtr->frameId(observationIndexPair[j]);
    auto poseid_iter =
        filter_.statesMap_.find(poseId);
    int startIndex =
        poseid_iter->second.global.at(GlobalStates::T_WS).startIndexInCov -
        cameraParamStartIndex;
    H_xjk->block<1, 3>(0, startIndex) = de_dp_GBtj[j];
    H_xjk->block<1, 3>(0, startIndex + 3) = de_dtheta_GBtj[j];
    H_xjk->block<1, 3>(0, startIndex + 6) = de_dv_GBtj[j];
    H_fjk->emplace_back(de_dfj[j]);
  }
  return true;
}

bool HybridFilter::featureJacobianEpipolar(
    const MapPoint& mp,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>* Hi,
    Eigen::Matrix<double, Eigen::Dynamic, 1>* ri,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>* Ri,
    RetrieveObsSeqType seqType) const {
  const int camIdx = 0;
  std::shared_ptr<const okvis::cameras::CameraBase> tempCameraGeometry =
      camera_rig_.getCameraGeometry(camIdx);
  // head and tail observations for this feature point
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      obsInPixels;
  std::vector<double> imagePointNoiseStds;  // std noise in pixels

  // each entry is undistorted coordinates in image plane at
  // z=1 in the specific camera frame, [\bar{x},\bar{y},1]
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      obsDirections;
  std::shared_ptr<msckf::PointSharedData> pointDataPtr(new msckf::PointSharedData());
  std::vector<std::pair<uint64_t, int>> badObservationIdentifiers;
  size_t numFeatures = gatherMapPointObservations(
      mp, pointDataPtr.get(), &obsDirections, &obsInPixels,
      &imagePointNoiseStds, &badObservationIdentifiers, seqType);

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
  for (size_t j = 0; j < numFeatures; ++j) {
      Eigen::Matrix2d imageObservationCov = Eigen::Matrix2d::Identity();
      double pixelNoiseStd = imagePointNoiseStds[2 * j];
      imageObservationCov(0, 0) *= (pixelNoiseStd * pixelNoiseStd);
      pixelNoiseStd = imagePointNoiseStds[2 * j + 1];
      imageObservationCov(1, 1) *= (pixelNoiseStd * pixelNoiseStd);
      bool projectOk = obsDirectionJacobian(
          obsDirections[j], tempCameraGeometry, projOptModelId, imageObservationCov,
          &dfj_dXcam[j], &cov_fij[j]);
      projectStatus[j]= projectOk;
  }
  pointDataPtr->removeBadObservations(projectStatus);
  msckf::removeUnsetMatrices<Eigen::Vector3d>(&obsDirections, projectStatus);
  msckf::removeUnsetMatrices<Eigen::Vector2d>(&obsInPixels, projectStatus);
  msckf::removeUnsetMatrices<Eigen::Matrix<double, 3, -1>>(&dfj_dXcam, projectStatus);
  msckf::removeUnsetMatrices<Eigen::Matrix3d>(&cov_fij, projectStatus);
  size_t numValidDirectionJac = pointDataPtr->numObservations();
  if (numValidDirectionJac < 2u) { // A two view constraint requires at least two obs
      return false;
  }

  propagatePoseAndVelocityForMapPoint(pointDataPtr.get());
  pointDataPtr->computePoseAndVelocityForJacobians(FLAGS_use_first_estimate);
  pointDataPtr->computeSharedJacobians(optimizationOptions_.cameraObservationModelId);

  // enlarge cov of the head obs to counteract the noise reduction
  // due to correlation in head_tail scheme
  size_t trackLength = mp.observations.size();
  double headObsCovModifier[2] = {1.0, 1.0};
  headObsCovModifier[0] =
      seqType == HEAD_TAIL
          ? (static_cast<double>(trackLength - pointLandmarkOptions_.minTrackLengthForMsckf + 2u))
          : 1.0;

  std::vector<std::pair<int, int>> featurePairs =
      TwoViewPair::getFramePairs(numValidDirectionJac, TwoViewPair::SINGLE_HEAD_TAIL);
  const int numConstraints = featurePairs.size();
  int featureVariableDimen = Hi->cols();
  Hi->resize(numConstraints, Eigen::NoChange);
  ri->resize(numConstraints, 1);

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> H_fi(numConstraints,
                                                             3 * numValidDirectionJac);
  H_fi.setZero();
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cov_fi(3 * numValidDirectionJac,
                                                               3 * numValidDirectionJac);
  cov_fi.setZero();

  int extrinsicModelId = camera_rig_.getExtrinsicOptMode(camIdx);
  int minExtrinsicDim = camera_rig_.getMinimalExtrinsicDimen(camIdx);
  int minProjDim = camera_rig_.getMinimalProjectionDimen(camIdx);
  int minDistortDim = camera_rig_.getDistortionDimen(camIdx);
  if (fixCameraIntrinsicParams_[camIdx]) {
    minProjDim = 0;
    minDistortDim = 0;
  }
  if (fixCameraExtrinsicParams_[camIdx]) {
    minExtrinsicDim = 0;
  }
  EpipolarMeasurement epiMeas(*this, tempCameraGeometry->imageHeight(),
                              camIdx, extrinsicModelId,
                              minExtrinsicDim, minProjDim, minDistortDim);
  for (int count = 0; count < numConstraints; ++count) {
    const std::pair<int, int>& feature_pair = featurePairs[count];
    std::vector<int> index_vec{feature_pair.first, feature_pair.second};
    Eigen::Matrix<double, 1, Eigen::Dynamic> H_xjk(1, featureVariableDimen);
    std::vector<Eigen::Matrix<double, 1, 3>,
                Eigen::aligned_allocator<Eigen::Matrix<double, 1, 3>>>
        H_fjk;
    double rjk;
    epiMeas.prepareTwoViewConstraint(obsDirections, obsInPixels, dfj_dXcam, index_vec);
    epiMeas.measurementJacobian(pointDataPtr, index_vec, &H_xjk, &H_fjk, &rjk);
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

bool HybridFilter::getOdometryConstraintsForKeyframe(
    std::shared_ptr<okvis::LoopQueryKeyframeMessage> queryKeyframe) const {
  int j = 0;
  std::vector<std::shared_ptr<NeighborConstraintMessage>>&
      odometryConstraintList = queryKeyframe->odometryConstraintListMutable();
  odometryConstraintList.reserve(
      poseGraphOptions_.maxOdometryConstraintForAKeyframe);
  okvis::kinematics::Transformation T_WBr = queryKeyframe->T_WB_;
  auto kfCovIndexIter = statesMap_.find(queryKeyframe->id_);
  int cov_T_WBr_start = kfCovIndexIter->second.global.at(GlobalStates::T_WS).startIndexInCov;
  queryKeyframe->setCovariance(covariance_.block<6, 6>(cov_T_WBr_start, cov_T_WBr_start));
  auto riter = statesMap_.rbegin();
  for (++riter;  // skip the last frame which in this case should be a keyframe.
       riter != statesMap_.rend() && j < poseGraphOptions_.maxOdometryConstraintForAKeyframe;
       ++riter) {
    if (riter->second.isKeyframe) {
      okvis::kinematics::Transformation T_WBn;
      get_T_WS(riter->first, T_WBn);
      okvis::kinematics::Transformation T_BnBr = T_WBn.inverse() * T_WBr;
      std::shared_ptr<okvis::NeighborConstraintMessage> odometryConstraint(
          new okvis::NeighborConstraintMessage(
              riter->first, riter->second.timestamp, T_BnBr, T_WBn));
      odometryConstraint->core_.squareRootInfo_.setIdentity();

      auto poseCovIndexIter = statesMap_.find(riter->first);
      int cov_T_WBn_start =
          poseCovIndexIter->second.global.at(GlobalStates::T_WS)
              .startIndexInCov;

      odometryConstraint->cov_T_WB_ = covariance_.block<6, 6>(
            cov_T_WBn_start, cov_T_WBn_start);
      odometryConstraint->cov_T_WBr_T_WB_ = covariance_.block<6, 6>(
            cov_T_WBr_start, cov_T_WBn_start);
      odometryConstraintList.emplace_back(odometryConstraint);
      ++j;
    }
  }
  return true;
}

void computeExtrinsicJacobians(
    const okvis::kinematics::Transformation& T_BCi,
    const okvis::kinematics::Transformation& T_BC0,
    int cameraExtrinsicModelId,
    int mainCameraExtrinsicModelId,
    Eigen::AlignedVector<Eigen::MatrixXd>* dT_BCi_dExtrinsics,
    std::vector<size_t>* involvedCameraIndices,
    size_t mainCameraIndex) {
  dT_BCi_dExtrinsics->reserve(2);
  switch (cameraExtrinsicModelId) {
    case Extrinsic_p_CB::kModelId: {
      Eigen::Matrix<double, 6, Extrinsic_p_CB::kNumParams> dT_BC_dExtrinsic;
      Extrinsic_p_CB::dT_BC_dExtrinsic(T_BCi.C(), &dT_BC_dExtrinsic);
      dT_BCi_dExtrinsics->push_back(dT_BC_dExtrinsic);
    } break;
    case Extrinsic_p_BC_q_BC::kModelId: {
      Eigen::Matrix<double, 6, Extrinsic_p_BC_q_BC::kNumParams>
          dT_BC_dExtrinsic;
      Extrinsic_p_BC_q_BC::dT_BC_dExtrinsic(&dT_BC_dExtrinsic);
      dT_BCi_dExtrinsics->push_back(dT_BC_dExtrinsic);
    } break;
    case Extrinsic_p_C0C_q_C0C::kModelId: {
      involvedCameraIndices->push_back(mainCameraIndex);
      Eigen::Matrix<double, 6, Extrinsic_p_C0C_q_C0C::kNumParams> dT_BC_dT_C0Ci;
      Eigen::Matrix<double, 6, Extrinsic_p_C0C_q_C0C::kNumParams> dT_BC_dT_BC0;
      Extrinsic_p_C0C_q_C0C::dT_BC_dExtrinsic(T_BCi, T_BC0, &dT_BC_dT_C0Ci,
                                              &dT_BC_dT_BC0);
      dT_BCi_dExtrinsics->push_back(dT_BC_dT_C0Ci);

      switch (mainCameraExtrinsicModelId) {
        case Extrinsic_p_CB::kModelId: {
          Eigen::Matrix<double, 6, Extrinsic_p_CB::kNumParams>
              dT_BC0_dExtrinsic;
          Extrinsic_p_CB::dT_BC_dExtrinsic(T_BC0.C(), &dT_BC0_dExtrinsic);
          dT_BCi_dExtrinsics->push_back(dT_BC_dT_BC0 * dT_BC0_dExtrinsic);
        } break;
        case Extrinsic_p_BC_q_BC::kModelId: {
          Eigen::Matrix<double, 6, Extrinsic_p_BC_q_BC::kNumParams>
              dT_BC0_dExtrinsic;
          Extrinsic_p_BC_q_BC::dT_BC_dExtrinsic(&dT_BC0_dExtrinsic);
          dT_BCi_dExtrinsics->push_back(dT_BC_dT_BC0 * dT_BC0_dExtrinsic);
        } break;
        default:
          throw std::runtime_error(
              "Unknown extrinsic model type for main camera!");
      }
    } break;
    default:
      throw std::runtime_error("Unknown extrinsic model type for a camera!");
  }
}

}  // namespace okvis
