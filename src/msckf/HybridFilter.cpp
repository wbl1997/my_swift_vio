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

#include <msckf/FilterHelper.hpp>
#include <msckf/ImuOdometry.h>
#include <msckf/PointLandmark.hpp>
#include <msckf/PointLandmarkModels.hpp>
#include <msckf/PointSharedData.hpp>
#include <msckf/RelativeMotionJacobian.hpp>
#include <msckf/TwoViewPair.hpp>

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

DEFINE_double(image_noise_cov_multiplier, 4.0,
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
      minValidStateId_(0),
      triangulateTimer("3.1.1.1 triangulateAMapPoint", true),
      computeHTimer("3.1.1 featureJacobian", true),
      computeKalmanGainTimer("3.1.2 computeKalmanGain", true),
      updateStatesTimer("3.1.3 updateStates", true),
      updateCovarianceTimer("3.1.4 updateCovariance", true),
      updateLandmarksTimer("3.1.5 updateLandmarks", true),
      mTrackLengthAccumulator(100, 0),
      updateVecNormTermination_(1e-4),
      maxNumIteration_(6) {
  setLandmarkModel(msckf::InverseDepthParameterization::kModelId);
}

// The default constructor.
HybridFilter::HybridFilter()
    : Estimator(),
      minValidStateId_(0),
      triangulateTimer("3.1.1.1 triangulateAMapPoint", true),
      computeHTimer("3.1.1 featureJacobian", true),
      computeKalmanGainTimer("3.1.2 computeKalmanGain", true),
      updateStatesTimer("3.1.3 updateStates", true),
      updateCovarianceTimer("3.1.4 updateCovariance", true),
      updateLandmarksTimer("3.1.5 updateLandmarks", true),
      mTrackLengthAccumulator(100, 0),
      updateVecNormTermination_(1e-4),
      maxNumIteration_(6) {
  setLandmarkModel(msckf::InverseDepthParameterization::kModelId);
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
    imuInfo->at(ImuSensorStates::TG).id =
        lastElementIterator->second.sensors.at(SensorStates::Imu)
            .at(imu_id)
            .at(ImuSensorStates::TG)
            .id;
    imuInfo->at(ImuSensorStates::TS).id =
        lastElementIterator->second.sensors.at(SensorStates::Imu)
            .at(imu_id)
            .at(ImuSensorStates::TS)
            .id;
    imuInfo->at(ImuSensorStates::TA).id =
        lastElementIterator->second.sensors.at(SensorStates::Imu)
            .at(imu_id)
            .at(ImuSensorStates::TA)
            .id;
  } else {
    Eigen::Matrix<double, 27, 1> vTgTsTa = imu_rig_.getImuAugmentedEuclideanParams();
    Eigen::Matrix<double, 9, 1> TG = vTgTsTa.head<9>();
    uint64_t id = IdProvider::instance().newId();
    std::shared_ptr<ceres::ShapeMatrixParamBlock> tgBlockPtr(
        new ceres::ShapeMatrixParamBlock(TG, id, stateTime));
    mapPtr_->addParameterBlock(tgBlockPtr, ceres::Map::Trivial);
    imuInfo->at(ImuSensorStates::TG).id = id;

    const Eigen::Matrix<double, 9, 1> TS = vTgTsTa.segment<9>(9);
    id = IdProvider::instance().newId();
    std::shared_ptr<okvis::ceres::ShapeMatrixParamBlock> tsBlockPtr(
        new okvis::ceres::ShapeMatrixParamBlock(TS, id, stateTime));
    mapPtr_->addParameterBlock(tsBlockPtr, ceres::Map::Trivial);
    imuInfo->at(ImuSensorStates::TS).id = id;

    Eigen::Matrix<double, 9, 1> TA = vTgTsTa.tail<9>();
    id = IdProvider::instance().newId();
    std::shared_ptr<okvis::ceres::ShapeMatrixParamBlock> taBlockPtr(
        new okvis::ceres::ShapeMatrixParamBlock(TA, id, stateTime));
    mapPtr_->addParameterBlock(taBlockPtr, ceres::Map::Trivial);
    imuInfo->at(ImuSensorStates::TA).id = id;
  }
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

    if (pvstd_.initWithExternalSource_) {
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

    uint64_t td_id = statesMap_.rbegin()
                         ->second.sensors.at(SensorStates::Camera)
                         .at(0)
                         .at(CameraSensorStates::TD)
                         .id;  // one camera assumption
    tdEstimate =
        okvis::Duration(std::static_pointer_cast<ceres::CameraTimeParamBlock>(
                            mapPtr_->parameterBlockPtr(td_id))
                            ->estimate());
    correctedStateTime = multiFrame->timestamp() + tdEstimate;

    Eigen::Matrix<double, 27, 1> vTgTsTa = imu_rig_.getImuAugmentedEuclideanParams(0);

    // propagate pose, speedAndBias, and covariance
    okvis::Time startTime = statesMap_.rbegin()->second.timestamp;
    Eigen::Matrix<double, ceres::ode::OdoErrorStateDim,
                  ceres::ode::OdoErrorStateDim>
        Pkm1 = covariance_.topLeftCorner<ceres::ode::OdoErrorStateDim,
                                         ceres::ode::OdoErrorStateDim>();
    Eigen::Matrix<double, ceres::ode::OdoErrorStateDim,
                  ceres::ode::OdoErrorStateDim>
        F_tot;
    F_tot.setIdentity();
    int numUsedImuMeasurements = -1;
    okvis::Time latestImuEpoch = imuMeasurements.back().timeStamp;
    okvis::Time propagationTargetTime = correctedStateTime;
    if (latestImuEpoch < correctedStateTime) {
      propagationTargetTime = latestImuEpoch;
      LOG(WARNING) << "Latest IMU readings does not extend to corrected state "
                      "time. Is temporal_imu_data_overlap too small?";
    }
    if (FLAGS_use_first_estimate) {
      /// use latest estimate to propagate pose, speed and bias, and first
      /// estimate to propagate covariance and Jacobian
      std::shared_ptr<const Eigen::Matrix<double, 6, 1>> lP =
          statesMap_.rbegin()->second.linearizationPoint;
      Eigen::Vector3d tempV_WS = speedAndBias.head<3>();
      IMUErrorModel<double> tempIEM(speedAndBias.tail<6>(), vTgTsTa);
      numUsedImuMeasurements = IMUOdometry::propagation(
          imuMeasurements, imuParametersVec_.at(0), T_WS, tempV_WS, tempIEM,
          startTime, propagationTargetTime, &Pkm1, &F_tot, lP.get());
      speedAndBias.head<3>() = tempV_WS;
    } else {
      /// use latest estimate to propagate pose, speed and bias, and covariance
      if (FLAGS_use_RK4) {
        // method 1 RK4 a little bit more accurate but 4 times slower
        numUsedImuMeasurements = IMUOdometry::propagation_RungeKutta(
            imuMeasurements, imuParametersVec_.at(0), T_WS, speedAndBias,
            vTgTsTa, startTime, propagationTargetTime, &Pkm1, &F_tot);
      } else {
        // method 2, i.e., adapt the imuError::propagation function of okvis by
        // the msckf derivation in Michael Andrew Shelley
        Eigen::Vector3d tempV_WS = speedAndBias.head<3>();
        IMUErrorModel<double> tempIEM(speedAndBias.tail<6>(), vTgTsTa);
        numUsedImuMeasurements = IMUOdometry::propagation(
            imuMeasurements, imuParametersVec_.at(0), T_WS, tempV_WS, tempIEM,
            startTime, propagationTargetTime, &Pkm1, &F_tot);
        speedAndBias.head<3>() = tempV_WS;
      }
    }
    if (numUsedImuMeasurements < 2) {
      LOG(WARNING) << "numUsedImuMeasurements=" << numUsedImuMeasurements
                   << " correctedStateTime " << correctedStateTime
                   << " lastFrameTimestamp " << startTime << " tdEstimate "
                   << tdEstimate << std::endl;
    }
    okvis::Time secondLatestStateTime = statesMap_.rbegin()->second.timestamp;
    auto imuMeasCoverSecond = inertialMeasForStates_.findWindow(secondLatestStateTime, half_window_);
    statesMap_.rbegin()->second.imuReadingWindow.reset(new okvis::ImuMeasurementDeque(imuMeasCoverSecond));

    int covDim = covariance_.rows();
    covariance_.topLeftCorner(ceres::ode::OdoErrorStateDim,
                              ceres::ode::OdoErrorStateDim) = Pkm1;
    covariance_.block(0, ceres::ode::OdoErrorStateDim,
                      ceres::ode::OdoErrorStateDim,
                      covDim - ceres::ode::OdoErrorStateDim) =
        F_tot * covariance_.block(0, ceres::ode::OdoErrorStateDim,
                                  ceres::ode::OdoErrorStateDim,
                                  covDim - ceres::ode::OdoErrorStateDim);
    covariance_.block(ceres::ode::OdoErrorStateDim, 0,
                      covDim - ceres::ode::OdoErrorStateDim,
                      ceres::ode::OdoErrorStateDim) =
        covariance_.block(ceres::ode::OdoErrorStateDim, 0,
                          covDim - ceres::ode::OdoErrorStateDim,
                          ceres::ode::OdoErrorStateDim) *
        F_tot.transpose();
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

  OKVIS_ASSERT_EQ_DBG(Exception, extrinsicsEstimationParametersVec_.size(), 1,
                      "Only one camera is supported.");
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
      // use the same block...
      // the following will point to the last states:
      std::map<uint64_t, States>::reverse_iterator lastElementIterator =
          statesMap_.rbegin();
      lastElementIterator++;
      cameraInfos.at(CameraSensorStates::T_SCi).id =
          lastElementIterator->second.sensors.at(SensorStates::Camera)
              .at(i)
              .at(CameraSensorStates::T_SCi)
              .id;
      cameraInfos.at(CameraSensorStates::Intrinsics).exists =
          lastElementIterator->second.sensors.at(SensorStates::Camera)
              .at(i)
              .at(CameraSensorStates::Intrinsics)
              .exists;
      cameraInfos.at(CameraSensorStates::Intrinsics).id =
          lastElementIterator->second.sensors.at(SensorStates::Camera)
              .at(i)
              .at(CameraSensorStates::Intrinsics)
              .id;
      cameraInfos.at(CameraSensorStates::Distortion).id =
          lastElementIterator->second.sensors.at(SensorStates::Camera)
              .at(i)
              .at(CameraSensorStates::Distortion)
              .id;
      cameraInfos.at(CameraSensorStates::TD).id =
          lastElementIterator->second.sensors.at(SensorStates::Camera)
              .at(i)
              .at(CameraSensorStates::TD)
              .id;
      cameraInfos.at(CameraSensorStates::TR).id =
          lastElementIterator->second.sensors.at(SensorStates::Camera)
              .at(i)
              .at(CameraSensorStates::TR)
              .id;
    } else {
      const okvis::kinematics::Transformation T_BC =
          camera_rig_.getCameraExtrinsic(i);

      uint64_t id = IdProvider::instance().newId();
      std::shared_ptr<okvis::ceres::PoseParameterBlock>
          extrinsicsParameterBlockPtr(new okvis::ceres::PoseParameterBlock(
              T_BC, id, correctedStateTime));
      mapPtr_->addParameterBlock(extrinsicsParameterBlockPtr,
                                 ceres::Map::Pose6d);
      cameraInfos.at(CameraSensorStates::T_SCi).id = id;

      Eigen::VectorXd allIntrinsics;
      camera_rig_.getCameraGeometry(i)->getIntrinsics(allIntrinsics);
      id = IdProvider::instance().newId();
      int projOptModelId = camera_rig_.getProjectionOptMode(i);
      const int minProjectionDim = camera_rig_.getMinimalProjectionDimen(i);
      if (!fixCameraIntrinsicParams_[i]) {
        Eigen::VectorXd optProjIntrinsics;
        ProjectionOptGlobalToLocal(projOptModelId, allIntrinsics,
                                   &optProjIntrinsics);
        std::shared_ptr<okvis::ceres::EuclideanParamBlock>
            projIntrinsicParamBlockPtr(new okvis::ceres::EuclideanParamBlock(
                optProjIntrinsics, id, correctedStateTime, minProjectionDim));
        mapPtr_->addParameterBlock(projIntrinsicParamBlockPtr,
                                   ceres::Map::Parameterization::Trivial);
        cameraInfos.at(CameraSensorStates::Intrinsics).id = id;
      } else {
        Eigen::VectorXd optProjIntrinsics = allIntrinsics.head<4>();
        std::shared_ptr<okvis::ceres::EuclideanParamBlock>
            projIntrinsicParamBlockPtr(new okvis::ceres::EuclideanParamBlock(
                optProjIntrinsics, id, correctedStateTime, 4));
        mapPtr_->addParameterBlock(projIntrinsicParamBlockPtr,
                                   ceres::Map::Parameterization::Trivial);
        cameraInfos.at(CameraSensorStates::Intrinsics).id = id;
        mapPtr_->setParameterBlockConstant(id);
      }
      id = IdProvider::instance().newId();
      const int distortionDim = camera_rig_.getDistortionDimen(i);
      std::shared_ptr<okvis::ceres::EuclideanParamBlock>
          distortionParamBlockPtr(new okvis::ceres::EuclideanParamBlock(
              allIntrinsics.tail(distortionDim), id, correctedStateTime,
              distortionDim));
      mapPtr_->addParameterBlock(distortionParamBlockPtr,
                                 ceres::Map::Parameterization::Trivial);
      cameraInfos.at(CameraSensorStates::Distortion).id = id;

      id = IdProvider::instance().newId();
      std::shared_ptr<okvis::ceres::CameraTimeParamBlock> tdParamBlockPtr(
          new okvis::ceres::CameraTimeParamBlock(camera_rig_.getImageDelay(i),
                                                 id, correctedStateTime));
      mapPtr_->addParameterBlock(tdParamBlockPtr,
                                 ceres::Map::Parameterization::Trivial);
      cameraInfos.at(CameraSensorStates::TD).id = id;

      id = IdProvider::instance().newId();
      std::shared_ptr<okvis::ceres::CameraTimeParamBlock> trParamBlockPtr(
          new okvis::ceres::CameraTimeParamBlock(camera_rig_.getReadoutTime(i),
                                                 id, correctedStateTime));
      mapPtr_->addParameterBlock(trParamBlockPtr,
                                 ceres::Map::Parameterization::Trivial);
      cameraInfos.at(CameraSensorStates::TR).id = id;
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
    const int camIdx = 0;
    initCovariance(camIdx);
  }

  addCovForClonedStates();
  updateCovarianceIndex();
  return true;
}

void HybridFilter::initCovariance(int camIdx) {
  int covDim = startIndexOfClonedStates();
  Eigen::Matrix<double, 6, 6> covPQ =
      Eigen::Matrix<double, 6, 6>::Zero();  // [\delta p_B^G, \delta \theta]

  covPQ.topLeftCorner<3, 3>() = pvstd_.std_p_WS.cwiseAbs2().asDiagonal();
  covPQ.bottomRightCorner<3, 3>() = pvstd_.std_q_WS.cwiseAbs2().asDiagonal();

  Eigen::Matrix<double, 9, 9> covSB =
      Eigen::Matrix<double, 9, 9>::Zero();  // $v_B^G, b_g, b_a$
  Eigen::Matrix<double, 27, 27> covTGTSTA =
      Eigen::Matrix<double, 27, 27>::Zero();
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
    const double sigmaTGElement = imuParametersVec_.at(0).sigma_TGElement;
    const double sigmaTSElement = imuParametersVec_.at(0).sigma_TSElement;
    const double sigmaTAElement = imuParametersVec_.at(0).sigma_TAElement;
    covTGTSTA.topLeftCorner<9, 9>() =
        Eigen::Matrix<double, 9, 9>::Identity() * std::pow(sigmaTGElement, 2);
    covTGTSTA.block<9, 9>(9, 9) =
        Eigen::Matrix<double, 9, 9>::Identity() * std::pow(sigmaTSElement, 2);
    covTGTSTA.block<9, 9>(18, 18) =
        Eigen::Matrix<double, 9, 9>::Identity() * std::pow(sigmaTAElement, 2);
  }
  covariance_ = Eigen::MatrixXd::Zero(covDim, covDim);
  covariance_.topLeftCorner<6, 6>() = covPQ;
  covariance_.block<9, 9>(6, 6) = covSB;
  covariance_.block<27, 27>(15, 15) = covTGTSTA;

  initCameraParamCovariance(camIdx);
}

void HybridFilter::initCameraParamCovariance(int camIdx) {
  // camera sensor states
  int camParamStartIndex = startIndexOfCameraParams();
  int minExtrinsicDim = camera_rig_.getMinimalExtrinsicDimen(camIdx);
  int minProjectionDim = camera_rig_.getMinimalProjectionDimen(camIdx);
  int distortionDim = camera_rig_.getDistortionDimen(camIdx);

  Eigen::MatrixXd covExtrinsic;
  Eigen::MatrixXd covProjIntrinsics;
  Eigen::MatrixXd covDistortion;
  Eigen::Matrix2d covTDTR;

  int camParamIndex = camParamStartIndex;
  if (!fixCameraExtrinsicParams_[camIdx]) {
    covExtrinsic =
        ExtrinsicModelInitCov(camera_rig_.getExtrinsicOptMode(camIdx),
                              extrinsicsEstimationParametersVec_.at(camIdx)
                                  .sigma_absolute_translation,
                              extrinsicsEstimationParametersVec_.at(camIdx)
                                  .sigma_absolute_orientation);
    covariance_.block(camParamStartIndex, camParamStartIndex, minExtrinsicDim,
                      minExtrinsicDim) = covExtrinsic;
    camParamIndex += minExtrinsicDim;
  }

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
    covariance_.block(camParamIndex, camParamIndex, distortionDim,
                      distortionDim) = covDistortion;
    camParamIndex += distortionDim;
  }

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
  size_t covDimAugmented = oldCovDim + 9;  //$\delta p,\delta \alpha,\delta v$
  Eigen::MatrixXd covarianceAugmented(covDimAugmented, covDimAugmented);

  const size_t numPointStates = 3 * mInCovLmIds.size();
  const size_t numOldNavImuCamPoseStates = oldCovDim - numPointStates;

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

// Applies the dropping/marginalization strategy, i.e., state management,
// according to Li and Mourikis RSS 12 optimization based thesis
bool HybridFilter::applyMarginalizationStrategy(
    size_t /*numKeyframes*/, size_t /*numImuFrames*/,
    okvis::MapPointVector& /*removedLandmarks*/) {
  /// remove features tracked no more, the feature can be in state or not
  int covDim = covariance_.rows();
  Eigen::Matrix<double, 3, Eigen::Dynamic> reparamJacobian(
      3,
      covDim);  // Jacobians of feature reparameterization due to anchor
                 // change
  std::vector<
      Eigen::Matrix<double, 3, Eigen::Dynamic>,
      Eigen::aligned_allocator<Eigen::Matrix<double, 3, Eigen::Dynamic>>>
      vJacobian;  // container of these reparameterizing Jacobians
  std::vector<size_t> vCovPtId;  // id in covariance of point features to be
                                 // reparameterized, 0 based
  std::vector<uint64_t>
      toRemoveLmIds;  // id of landmarks to be removed that are in state
  const size_t numNavImuCamStates = startIndexOfClonedStates();
  // number of navigation, imu, and camera states in the covariance
  const size_t numNavImuCamPoseStates =
      numNavImuCamStates + 9 * statesMap_.size();
  // number of navigation, imu, camera, and pose copies states in the
  // covariance

  std::cout << "checking removed map points" << std::endl;
  for (PointMap::iterator pit = landmarksMap_.begin();
       pit != landmarksMap_.end();) {
    ResidualizeCase residualizeCase = pit->second.residualizeCase;
    if (residualizeCase == NotInState_NotTrackedNow ||
        residualizeCase == InState_NotTrackedNow) {
      ceres::Map::ResidualBlockCollection residuals =
          mapPtr_->residuals(pit->first);
      for (size_t r = 0; r < residuals.size(); ++r) {
        std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
            std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
                residuals[r].errorInterfacePtr);
        OKVIS_ASSERT_TRUE(Exception, reprojectionError,
                          "Wrong index of reprojection error");
        removeObservation(residuals[r].residualBlockId);
      }

      if (residualizeCase == InState_NotTrackedNow) {
        OKVIS_ASSERT_TRUE_DBG(Exception, pit->second.anchorStateId > 0,
                              "a tracked point in the states not recorded");
        toRemoveLmIds.push_back(pit->first);
      }

      mapPtr_->removeParameterBlock(pit->first);
      pit = landmarksMap_.erase(pit);
    } else {
      /// change anchor pose for features whose anchor is not in states
      /// anymore
      if (residualizeCase == InState_TrackedNow) {
        if (pit->second.anchorStateId < minValidStateId_) {
          uint64_t currFrameId = currentFrameId();
          okvis::kinematics::Transformation
              T_GBa;  // transform from the body frame at the anchor frame
                      // epoch to the global frame
          get_T_WS(pit->second.anchorStateId, T_GBa);
          okvis::kinematics::Transformation T_GBc;
          get_T_WS(currFrameId, T_GBc);
          okvis::kinematics::Transformation T_BC;
          const int camIdx = 0;
          getCameraSensorStates(currFrameId, camIdx, T_BC);

          okvis::kinematics::Transformation T_GA =
              T_GBa * T_BC;  // anchor camera frame to global frame
          okvis::kinematics::Transformation T_GC = T_GBc * T_BC;

          std::shared_ptr<okvis::ceres::HomogeneousPointParameterBlock> hppb =
              std::static_pointer_cast<
                  okvis::ceres::HomogeneousPointParameterBlock>(
                  mapPtr_->parameterBlockPtr(pit->first));

          // update covariance matrix
          Eigen::Vector4d ab1rho = hppb->estimate();
          Eigen::Vector3d abrhoi(ab1rho[0], ab1rho[1], ab1rho[3]);
          Eigen::Vector3d abrhoj;
          Eigen::Matrix<double, 3, 9> jacobian;
          vio::reparameterize_AIDP(T_GA.C(), T_GC.C(), abrhoi, T_GA.r(),
                                   T_GC.r(), abrhoj, &jacobian);

          reparamJacobian.setZero();
          size_t startRowC =
              numNavImuCamStates + 9 * mStateID2CovID_[currFrameId];
          size_t startRowA = numNavImuCamStates +
                             9 * mStateID2CovID_[pit->second.anchorStateId];

          reparamJacobian.block<3, 3>(0, startRowA) =
              jacobian.block<3, 3>(0, 3);
          reparamJacobian.block<3, 3>(0, startRowC) =
              jacobian.block<3, 3>(0, 6);

          std::deque<uint64_t>::iterator idPos =
              std::find(mInCovLmIds.begin(), mInCovLmIds.end(), pit->first);
          OKVIS_ASSERT_TRUE(Exception, idPos != mInCovLmIds.end(),
                            "The tracked landmark is not in mInCovLmIds ");
          size_t covPtId = idPos - mInCovLmIds.begin();
          vCovPtId.push_back(covPtId);
          reparamJacobian.block<3, 3>(0, numNavImuCamPoseStates + 3 * covPtId) =
              jacobian.topLeftCorner<3, 3>();
          vJacobian.push_back(reparamJacobian);

          ab1rho = T_GC.inverse() * T_GA * ab1rho;
          ab1rho /= ab1rho[2];
          hppb->setEstimate(ab1rho);

          pit->second.anchorStateId = currFrameId;
          pit->second.q_GA = T_GC.q();
          pit->second.p_BA_G = T_GC.r() - T_GBc.r();
        }
      }
      ++pit;
    }
  }

  // actual covariance update for reparameterized features
  int tempCounter = 0;
  Eigen::MatrixXd featureJacMat = Eigen::MatrixXd::Identity(
      covDim,
      covDim);  // Jacobian of all the new states w.r.t the old states
  for (auto it = vJacobian.begin(); it != vJacobian.end();
       ++it, ++tempCounter) {
    featureJacMat.block(numNavImuCamPoseStates + vCovPtId[tempCounter] * 3, 0,
                          3, covDim) = vJacobian[tempCounter];
  }
  if (vJacobian.size()) {
    covariance_ =
        (featureJacMat * covariance_).eval() * featureJacMat.transpose();
  }

  // actual covariance decimation for features in state and not tracked now
#if 0
    for(auto it= toRemoveLmIds.begin(), itEnd= toRemoveLmIds.end(); it!=itEnd; ++it){
        std::deque<uint64_t>::iterator idPos = std::find(mInCovLmIds.begin(), mInCovLmIds.end(), *it);
        OKVIS_ASSERT_TRUE(Exception, idPos != mInCovLmIds.end(), "The tracked landmark in state is not in mInCovLmIds ");

        // remove SLAM feature's dimension from the covariance matrix
        int startIndex = numNavImuCamPoseStates + 3*(idPos - mInCovLmIds.begin());
        int finishIndex = startIndex + 3;
        Eigen::MatrixXd slimCovariance(covDim - 3, covDim - 3);
        slimCovariance << covariance_.topLeftCorner(startIndex, startIndex),
                covariance_.block(0, finishIndex, startIndex, covDim - finishIndex),
                covariance_.block(finishIndex, 0, covDim - finishIndex, startIndex),
                covariance_.block(finishIndex, finishIndex, covDim - finishIndex, covDim - finishIndex);

        covariance_ = slimCovariance;
        covDim -= 3;
        mInCovLmIds.erase(idPos);
    }
#else

  std::vector<size_t> toRemoveIndices;  // start indices of removed columns,
                                        // each interval of size 3
  toRemoveIndices.reserve(toRemoveLmIds.size());

  for (auto it = toRemoveLmIds.begin(), itEnd = toRemoveLmIds.end();
       it != itEnd; ++it) {
    std::deque<uint64_t>::iterator idPos =
        std::find(mInCovLmIds.begin(), mInCovLmIds.end(), *it);
    OKVIS_ASSERT_TRUE(Exception, idPos != mInCovLmIds.end(),
                      "The tracked landmark in state is not in mInCovLmIds ");

    // to-be-removed SLAM feature's dimension from the covariance matrix
    int startIndex = numNavImuCamPoseStates + 3 * (idPos - mInCovLmIds.begin());
    toRemoveIndices.push_back(startIndex);
  }
  std::sort(toRemoveIndices.begin(), toRemoveIndices.end());
  std::vector<std::pair<size_t, size_t>> vRowStartInterval;
  vRowStartInterval.reserve(toRemoveLmIds.size() + 1);
  size_t startKeptRow =
      0;  // start id(based on the old matrix) of the kept rows
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
    std::deque<uint64_t>::iterator idPos =
        std::find(mInCovLmIds.begin(), mInCovLmIds.end(), *it);
    mInCovLmIds.erase(idPos);
  }
  covDim -= 3 * toRemoveLmIds.size();
#endif

  /// remove old frames
  // these will keep track of what we want to marginalize out.
  std::vector<uint64_t> paremeterBlocksToBeMarginalized;
  std::vector<uint64_t> removeFrames;
  std::cout << "selecting which states to remove " << std::endl;
  std::cout << "min valid state Id " << minValidStateId_
            << " oldest and latest stateid " << statesMap_.begin()->first << " "
            << statesMap_.rbegin()->first << std::endl;
  // std::map<uint64_t, States>::reverse_iterator
  for (auto rit = statesMap_.rbegin(); rit != statesMap_.rend();) {
    if (rit->first < minValidStateId_) {
      removeFrames.push_back(rit->first);
      rit->second.global[GlobalStates::T_WS].exists =
          false;  // remember we removed
      rit->second.sensors.at(SensorStates::Imu)
          .at(0)
          .at(ImuSensorStates::SpeedAndBias)
          .exists = false;  // remember we removed
      paremeterBlocksToBeMarginalized.push_back(
          rit->second.global[GlobalStates::T_WS].id);
      paremeterBlocksToBeMarginalized.push_back(
          rit->second.sensors.at(SensorStates::Imu)
              .at(0)
              .at(ImuSensorStates::SpeedAndBias)
              .id);
      mapPtr_->removeParameterBlock(rit->second.global[GlobalStates::T_WS].id);
      mapPtr_->removeParameterBlock(rit->second.sensors.at(SensorStates::Imu)
                                        .at(0)
                                        .at(ImuSensorStates::SpeedAndBias)
                                        .id);

      inertialMeasForStates_.pop_front(statesMap_.find(rit->first)->second.timestamp - half_window_);
      multiFramePtrMap_.erase(rit->first);

      //            std::advance(rit, 1);
      //            statesMap_.erase( rit.base() );//unfortunately this
      //            deletion does not work for statesMap_
    }
    //        else
    ++rit;
  }

  std::cout << "Marginalized covariance and states of Ids";
  for (auto iter = removeFrames.begin(); iter != removeFrames.end(); ++iter) {
    std::map<uint64_t, States>::iterator it = statesMap_.find(*iter);
    statesMap_.erase(it);
    std::cout << " " << *iter;
  }
  std::cout << std::endl;

  // update covariance matrix
  size_t numRemovedStates = removeFrames.size();
  if (numRemovedStates == 0) {
    return true;
  }

  int startIndex = startIndexOfClonedStates();
  int finishIndex = startIndex + numRemovedStates * 9;
  Eigen::MatrixXd slimCovariance(covDim - numRemovedStates * 9,
                                 covDim - numRemovedStates * 9);
  slimCovariance << covariance_.topLeftCorner(startIndex, startIndex),
      covariance_.block(0, finishIndex, startIndex, covDim - finishIndex),
      covariance_.block(finishIndex, 0, covDim - finishIndex, startIndex),
      covariance_.block(finishIndex, finishIndex, covDim - finishIndex,
                        covDim - finishIndex);

  covariance_ = slimCovariance;
  return true;
}

void HybridFilter::updateImuRig() {
  Eigen::VectorXd extraParams;
  getImuAugmentedStatesEstimate(&extraParams);
  imu_rig_.setImuAugmentedEuclideanParams(0, extraParams);
}

void HybridFilter::updateCovarianceIndex() {
  mStateID2CovID_.clear();
  int nCovIndex = 0;
  // note the statesMap_ is an ordered map!
  for (auto iter = statesMap_.begin(); iter != statesMap_.end(); ++iter) {
    mStateID2CovID_[iter->first] = nCovIndex;
    ++nCovIndex;
  }
}

void HybridFilter::updateSensorRigs() {
  const int camIdx = 0;
  const uint64_t currFrameId = currentFrameId();
  okvis::kinematics::Transformation T_BC0;
  getCameraSensorStates(currFrameId, camIdx, T_BC0);
  camera_rig_.setCameraExtrinsic(camIdx, T_BC0);

  Eigen::Matrix<double, Eigen::Dynamic, 1> projectionIntrinsic;
  getSensorStateEstimateAs<ceres::EuclideanParamBlock>(
      currFrameId, camIdx, SensorStates::Camera, CameraSensorStates::Intrinsics,
      projectionIntrinsic);

  Eigen::Matrix<double, Eigen::Dynamic, 1> distortionCoeffs;
  getSensorStateEstimateAs<ceres::EuclideanParamBlock>(
      currFrameId, camIdx, SensorStates::Camera, CameraSensorStates::Distortion,
      distortionCoeffs);
  camera_rig_.setCameraIntrinsics(camIdx, projectionIntrinsic, distortionCoeffs);

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

  updateImuRig();
}

// assume the rolling shutter camera reads data row by row, and rows are
// aligned with the width of a frame some heuristics to defend outliers is
// used, e.g., ignore correspondences of too large discrepancy between
// prediction and measurement

bool HybridFilter::slamFeatureJacobian(
    const uint64_t hpbid, const MapPoint& mp, Eigen::Matrix<double, 2, 1>& r_i,
    Eigen::Matrix<double, 2, Eigen::Dynamic>& H_x,
    Eigen::Matrix<double, 2, Eigen::Dynamic>& H_f, Eigen::Matrix2d& R_i) {
  computeHTimer.start();
  Eigen::Vector2d obsInPixel;
  Eigen::Vector4d ab1rho =
      std::static_pointer_cast<okvis::ceres::HomogeneousPointParameterBlock>(
          mapPtr_->parameterBlockPtr(hpbid))
          ->estimate();  // inverse depth parameterization in the anchor
                         // frame,
                         // [\alpha= X/Z, \beta= Y/Z, 1, \rho=1/Z]
  const uint64_t currFrameId = currentFrameId();
  const uint64_t anchorId = mp.anchorStateId;
  bool bObservedInCurrentFrame = false;
  for (auto itObs = mp.observations.rbegin(), iteObs = mp.observations.rend();
       itObs != iteObs; ++itObs) {
    if (itObs->first.frameId == currFrameId) {
      auto multiFrameIter = multiFramePtrMap_.find(currFrameId);
//      OKVIS_ASSERT_TRUE(Exception, multiFrameIter != multiFramePtrMap_.end(), "multiframe not found!");
      okvis::MultiFramePtr multiFramePtr = multiFrameIter->second;
      multiFramePtr->getKeypoint(itObs->first.cameraIndex,
                                 itObs->first.keypointIndex, obsInPixel);

      double kpSize = 1.0;
      multiFramePtr->getKeypointSize(itObs->first.cameraIndex,
                                     itObs->first.keypointIndex, kpSize);
      R_i(0, 0) = (kpSize / 8) * (kpSize / 8);
      R_i(1, 1) = R_i(
          0, 0);  // image pixel noise follows that in addObservation function
      R_i(0, 1) = 0;
      R_i(1, 0) = 0;
      bObservedInCurrentFrame = true;
      break;
    }
  }
  OKVIS_ASSERT_TRUE(
      Exception, bObservedInCurrentFrame,
      "a point in slamFeatureJacobian should be observed in current frame!");
  // compute Jacobians for a measurement in image j of the current feature i
  // C_j is the current frame, Bj refers to the body frame associated with the
  // current frame, Ba refers to body frame associated with the anchor frame,
  // f_i is the feature in consideration

  Eigen::Vector2d imagePoint;  // projected pixel coordinates of the point in
                               // current frame ${z_u, z_v}$ in pixel units
  Eigen::Matrix2Xd
      intrinsicsJacobian;  //$\frac{\partial [z_u, z_v]^T}{\partial( f_x, f_v,
                           // c_x, c_y, k_1, k_2, p_1, p_2, [k_3])}$
  Eigen::Matrix<double, 2, 3>
      pointJacobian3;  // $\frac{\partial [z_u, z_v]^T}{\partial
                       // p_{f_i}^{C_j}}$

  // $\frac{\partial [z_u, z_v]^T}{p_B^C, fx, fy, cx, cy, k1, k2, p1, p2,
  // [k3], t_d, t_r}$
  Eigen::Matrix<double, 2, Eigen::Dynamic> J_Xc(2, cameraParamsMinimalDimen());

  Eigen::Matrix<double, 2, 9>
      J_XBj;  // $\frac{\partial [z_u, z_v]^T}{\partial delta\p_{B_j}^G,
              // \delta\alpha (of q_{B_j}^G), \delta v_{B_j}^G$

  Eigen::Matrix<double, 3, 9>
      factorJ_XBj;  // the second factor of J_XBj, see Michael Andrew Shelley
                    // Master thesis sec 6.5, p.55 eq 6.66
  Eigen::Matrix<double, 3, 9> factorJ_XBa;
  Eigen::Matrix<double, 2, 3>
      J_pfi;  // $\frac{\partial [z_u, z_v]^T}{\partial [a, b, \rho]}$
  Eigen::Vector2d J_td;
  Eigen::Vector2d J_tr;
  Eigen::Matrix<double, 2, 9>
      J_XBa;  // $\frac{\partial [z_u, z_v]^T}{\partial delta\p_{B_a}^G)$


  kinematics::Transformation T_WBj;
  get_T_WS(currFrameId, T_WBj);

  SpeedAndBiases sbj;
  getSpeedAndBias(currFrameId, 0, sbj);
  auto statesIter = statesMap_.find(currFrameId);
  Time stateEpoch = statesIter->second.timestamp;
  auto imuMeas = *(statesIter->second.imuReadingWindow);
  OKVIS_ASSERT_GT(Exception, imuMeas.size(), 0,
                  "the IMU measurement does not exist");

  const int camIdx = 0;
  uint32_t imageHeight = camera_rig_.getCameraGeometry(camIdx)->imageHeight();
  int projOptModelId = camera_rig_.getProjectionOptMode(camIdx);
  int extrinsicModelId = camera_rig_.getExtrinsicOptMode(camIdx);
  const double tdEstimate = camera_rig_.getImageDelay(camIdx);
  const double trEstimate = camera_rig_.getReadoutTime(camIdx);
  const okvis::kinematics::Transformation T_BC0 = camera_rig_.getCameraExtrinsic(camIdx);

  double kpN = obsInPixel[1] / imageHeight - 0.5;  // k per N
  Duration featureTime = Duration(tdEstimate + trEstimate * kpN -
                         statesIter->second.tdAtCreation);

  // for feature i, estimate $p_B^G(t_{f_i})$, $R_B^G(t_{f_i})$,
  // $v_B^G(t_{f_i})$, and $\omega_{GB}^B(t_{f_i})$ with the corresponding
  // states' LATEST ESTIMATES and imu measurements

  kinematics::Transformation T_WB = T_WBj;
  SpeedAndBiases sb = sbj;
  ImuMeasurement
      interpolatedInertialData;  // inertial data at the feature capture epoch
  Eigen::Matrix<double, 27, 1> vTGTSTA = imu_rig_.getImuAugmentedEuclideanParams();
  poseAndVelocityAtObservation(
      imuMeas, vTGTSTA.data(), imuParametersVec_.at(0), stateEpoch, featureTime,
      &T_WB, &sb, &interpolatedInertialData, FLAGS_use_RK4);

  okvis::kinematics::Transformation T_WBa;
  get_T_WS(anchorId, T_WBa);
  okvis::kinematics::Transformation T_GA(
      mp.p_BA_G + T_WBa.r(), mp.q_GA);  // anchor frame to global frame
  okvis::kinematics::Transformation T_CA =
      (T_WB * T_BC0).inverse() * T_GA;  // anchor frame to current camera frame
  Eigen::Vector3d pfiinC = (T_CA * ab1rho).head<3>();
  std::shared_ptr<const okvis::cameras::CameraBase> tempCameraGeometry =
      camera_rig_.getCameraGeometry(camIdx);
  cameras::CameraBase::ProjectionStatus status = tempCameraGeometry->project(
      pfiinC, &imagePoint, &pointJacobian3, &intrinsicsJacobian);
  if (status != cameras::CameraBase::ProjectionStatus::Successful) {
    LOG(WARNING)
        << "Failed to compute Jacobian for distortion with anchored point : "
        << ab1rho.transpose() << " and [r,q]_CA" << T_CA.coeffs().transpose();
    computeHTimer.stop();
    return false;
  } else if (!FLAGS_use_mahalanobis) {
    Eigen::Vector2d discrep = obsInPixel - imagePoint;
    if (std::fabs(discrep[0]) > FLAGS_max_proj_tolerance ||
        std::fabs(discrep[1]) > FLAGS_max_proj_tolerance) {
      computeHTimer.stop();
      return false;
    }
  }

  r_i = obsInPixel - imagePoint;

  okvis::kinematics::Transformation lP_T_WB = T_WB;
  SpeedAndBiases lP_sb = sb;
  if (FLAGS_use_first_estimate) {
    // compute Jacobians with FIRST ESTIMATES of position and velocity
    lP_T_WB = T_WBj;
    lP_sb = sbj;
    std::shared_ptr<const Eigen::Matrix<double, 6, 1>> posVelFirstEstimatePtr =
        statesIter->second.linearizationPoint;
    lP_T_WB =
        kinematics::Transformation(posVelFirstEstimatePtr->head<3>(), lP_T_WB.q());
    lP_sb.head<3>() = posVelFirstEstimatePtr->tail<3>();
    poseAndLinearVelocityAtObservation(
        imuMeas, vTGTSTA.data(), imuParametersVec_.at(0), stateEpoch,
        featureTime, &lP_T_WB, &lP_sb);
  }
  double rho = ab1rho[3];
  okvis::kinematics::Transformation T_BcA =
      lP_T_WB.inverse() *
      T_GA;  // anchor frame to the body frame associated with current frame
  J_td = pointJacobian3 * T_BC0.C().transpose() *
         (okvis::kinematics::crossMx((T_BcA * ab1rho).head<3>()) *
              interpolatedInertialData.measurement.gyroscopes -
          T_WB.C().transpose() * lP_sb.head<3>() * rho);
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
                                       T_BC0.C().transpose(), &dpC_dExtrinsic,
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
  J_pfi = pointJacobian3 * tempM3d;

  Eigen::Vector3d pfinG = (T_GA * ab1rho).head<3>();

  factorJ_XBj << -rho * Eigen::Matrix3d::Identity(),
      okvis::kinematics::crossMx(pfinG - lP_T_WB.r() * rho),
      -rho * Eigen::Matrix3d::Identity() * featureTime.toSec();
  J_XBj = pointJacobian3 * (T_WB.C() * T_BC0.C()).transpose() * factorJ_XBj;

  factorJ_XBa.topLeftCorner<3, 3>() = rho * Eigen::Matrix3d::Identity();
  factorJ_XBa.block<3, 3>(0, 3) =
      -okvis::kinematics::crossMx(T_WBa.C() * (T_BC0 * ab1rho).head<3>());
  factorJ_XBa.block<3, 3>(0, 6) = Eigen::Matrix3d::Zero();
  J_XBa = pointJacobian3 * (T_WB.C() * T_BC0.C()).transpose() * factorJ_XBa;

  H_x.resize(2, cameraParamPoseAndLandmarkMinimalDimen() - 3 * mInCovLmIds.size());
  H_x.setZero();
  H_f.resize(2, 3 * mInCovLmIds.size());
  H_f.setZero();
  const int minCamParamDim = cameraParamsMinimalDimen();

  H_x.topLeftCorner(2, minCamParamDim) = J_Xc;

  H_x.block<2, 9>(
      0,
      minCamParamDim + 9 * mStateID2CovID_[currFrameId]) = J_XBj;
  H_x.block<2, 9>(
      0,
      minCamParamDim + 9 * mStateID2CovID_[anchorId]) = J_XBa;

  std::deque<uint64_t>::iterator idPos =
      std::find(mInCovLmIds.begin(), mInCovLmIds.end(), hpbid);
  OKVIS_ASSERT_TRUE(Exception, idPos != mInCovLmIds.end(),
                    "The tracked landmark is not in mInCovLmIds ");
  size_t covPtId = idPos - mInCovLmIds.begin();
  H_f.block<2, 3>(0, covPtId * 3) = J_pfi;

  computeHTimer.stop();
  return true;
}

// assume the rolling shutter camera reads data row by row, and rows are
// aligned with the width of a frame.
bool HybridFilter::featureJacobian(
    const MapPoint& mp,
    Eigen::MatrixXd& H_oi,
    Eigen::Matrix<double, Eigen::Dynamic, 1>& r_oi,
    Eigen::MatrixXd& R_oi, Eigen::Vector4d& ab1rho,
    Eigen::Matrix<double, Eigen::Dynamic, 3>* pH_fi) const {
  computeHTimer.start();

  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      obsInPixel;                  // all observations for this feature point
  std::vector<double> vSigmai;         // std noise in pixels

  const int camIdx = 0;
  std::shared_ptr<const okvis::cameras::CameraBase> tempCameraGeometry =
      camera_rig_.getCameraGeometry(camIdx);
  int projOptModelId = camera_rig_.getProjectionOptMode(camIdx);
  int extrinsicModelId = camera_rig_.getExtrinsicOptMode(camIdx);
  const okvis::kinematics::Transformation T_BC0 = camera_rig_.getCameraExtrinsic(camIdx);

  msckf::PointLandmark pointLandmark(msckf::InverseDepthParameterization::kModelId);
  std::shared_ptr<msckf::PointSharedData> pointDataPtr(new msckf::PointSharedData());
  msckf::TriangulationStatus status = triangulateAMapPoint(
      mp, obsInPixel, pointLandmark, vSigmai, tempCameraGeometry, T_BC0,
      pointDataPtr.get(), nullptr, false);
  Eigen::Map<Eigen::Vector4d> v4Xhomog(pointLandmark.data(), 4);
  if (!status.triangulationOk) {
    computeHTimer.stop();
    return false;
  }
  uint64_t anchorId = pointDataPtr->anchorIds()[0];

  size_t numCamPoseStates = cameraParamPoseAndLandmarkMinimalDimen() - 3 * mInCovLmIds.size();
  // camera states, pose states, excluding feature states, and the velocity
  // dimension for the anchor state
  if (pH_fi == NULL) {
    numCamPoseStates -= 9;  // anchor frame is the frame preceding current frame
    OKVIS_ASSERT_EQ_DBG(Exception, anchorId, (++statesMap_.rbegin())->first,
                        "anchor frame of marginalized point should be the "
                        "frame preceding current frame");
  } else {
    OKVIS_ASSERT_EQ_DBG(
        Exception, anchorId, (statesMap_.rbegin())->first,
        "anchor frame of to be included points should be the current frame");
  }
  pointDataPtr->computePoseAndVelocityForJacobians(FLAGS_use_first_estimate);
  pointDataPtr->computeSharedJacobians(cameraObservationModelId_);
  // compute Jacobians for a measurement in image j of the current feature i
  // C_j is the current frame, Bj refers to the body frame associated with the
  // current frame, Ba refers to body frame associated with the anchor frame,
  // f_i is the feature in consideration

  // transform from the body frame at the anchor frame epoch to the world frame
  okvis::kinematics::Transformation T_WBa = pointDataPtr->T_WBa_list()[0];
  okvis::kinematics::Transformation T_GA = T_WBa * T_BC0;  // anchor frame to global frame
  ab1rho = T_GA.inverse() * v4Xhomog;
  if (ab1rho[2] < 0) {
    std::cout << "negative depth in ab1rho " << ab1rho.transpose() << std::endl;
    std::cout << "original v4xhomog " << v4Xhomog.transpose() << std::endl;
    computeHTimer.stop();
    return false;
  }
  ab1rho /= ab1rho[2];  //[\alpha = X/Z, \beta= Y/Z, 1, \rho=1/Z] in the
                        // anchor frame

  Eigen::Vector2d imagePoint;  // projected pixel coordinates of the point
                               // ${z_u, z_v}$ in pixel units
  Eigen::Matrix2Xd
      intrinsicsJacobian;  //$\frac{\partial [z_u, z_v]^T}{\partial( f_x, f_v,
                           // c_x, c_y, k_1, k_2, p_1, p_2, [k_3])}$
  Eigen::Matrix<double, 2, 3>
      pointJacobian3;  // $\frac{\partial [z_u, z_v]^T}{\partial
                       // p_{f_i}^{C_j}}$

  // $\frac{\partial [z_u, z_v]^T}{p_B^C, fx, fy, cx, cy, k1, k2, p1, p2,
  // [k3], t_d, t_r}$
  const int minCamParamDim = cameraParamsMinimalDimen();
  Eigen::MatrixXd J_Xc(2, minCamParamDim);

  Eigen::Matrix<double, 2, 9>
      J_XBj;  // $\frac{\partial [z_u, z_v]^T}{delta\p_{B_j}^G, \delta\alpha
              // (of q_{B_j}^G), \delta v_{B_j}^G$
  Eigen::Matrix<double, 3, 9>
      factorJ_XBj;  // the second factor of J_XBj, see Michael Andrew Shelley
                    // Master thesis sec 6.5, p.55 eq 6.66
  Eigen::Matrix<double, 3, 9> factorJ_XBa;

  Eigen::Matrix<double, 2, 3> J_pfi;  // $\frac{\partial [z_u, z_v]^T}{\partial
                                      // [\alpha, \beta, \rho]}$
  Eigen::Vector2d J_td;
  Eigen::Vector2d J_tr;
  Eigen::Matrix<double, 2, 9>
      J_XBa;  // $\frac{\partial [z_u, z_v]^T}{\partial delta\p_{B_a}^G)$

  Eigen::Matrix<double, 2, Eigen::Dynamic> H_x(
      2, numCamPoseStates);  // Jacobians of a feature w.r.t these states


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
  auto itRoi = vSigmai.begin();
  // compute Jacobians for a measurement in image j of the current feature i
  for (size_t observationIndex = 0; observationIndex < numPoses; ++observationIndex) {
    double kpN = pointDataPtr->normalizedRow(observationIndex);
    double featureTime = pointDataPtr->normalizedFeatureTime(observationIndex);
    kinematics::Transformation T_WB = pointDataPtr->T_WBtij(observationIndex);
    Eigen::Vector3d omega_Btij = pointDataPtr->omega_Btij(observationIndex);

    okvis::kinematics::Transformation T_CA =
        (T_WB * T_BC0).inverse() * T_GA;  // anchor frame to current camera frame
    Eigen::Vector3d pfiinC = (T_CA * ab1rho).head<3>();

    cameras::CameraBase::ProjectionStatus status = tempCameraGeometry->project(
        pfiinC, &imagePoint, &pointJacobian3, &intrinsicsJacobian);
    if (status != cameras::CameraBase::ProjectionStatus::Successful) {
      LOG(WARNING) << "Failed to compute Jacobian for distortion with "
                      "anchored point : "
                   << ab1rho.transpose() << " and [r,q]_CA"
                   << T_CA.coeffs().transpose();

      ++itFrameIds;
      itRoi = vSigmai.erase(itRoi);
      itRoi = vSigmai.erase(itRoi);
      continue;
    } else if (!FLAGS_use_mahalanobis) {
      Eigen::Vector2d discrep = obsInPixel[observationIndex] - imagePoint;
      if (std::fabs(discrep[0]) > FLAGS_max_proj_tolerance ||
          std::fabs(discrep[1]) > FLAGS_max_proj_tolerance) {
        ++itFrameIds;
        itRoi = vSigmai.erase(itRoi);
        itRoi = vSigmai.erase(itRoi);
        continue;
      }
    }

    vri.push_back(obsInPixel[observationIndex] - imagePoint);

    okvis::kinematics::Transformation lP_T_WB = pointDataPtr->T_WBtij_ForJacobian(observationIndex);
    Eigen::Vector3d lP_v_WB = pointDataPtr->v_WBtij_ForJacobian(observationIndex);

    double rho = ab1rho[3];
    okvis::kinematics::Transformation T_BcA = lP_T_WB.inverse() * T_GA;
    J_td = pointJacobian3 * T_BC0.C().transpose() *
           (okvis::kinematics::crossMx((T_BcA * ab1rho).head<3>()) *
                omega_Btij -
            T_WB.C().transpose() * lP_v_WB * rho);
    J_tr = J_td * kpN;

    if (fixCameraExtrinsicParams_[camIdx]) {
      if (fixCameraIntrinsicParams_[camIdx]) {
        J_Xc << J_td, J_tr;
      } else {
        ProjectionOptKneadIntrinsicJacobian(projOptModelId,
                                            &intrinsicsJacobian);
        J_Xc << intrinsicsJacobian, J_td, J_tr;
      }
    } else {
      Eigen::MatrixXd dpC_dExtrinsic;
      Eigen::Matrix3d R_CfCa = T_CA.C();
      ExtrinsicModel_dpC_dExtrinsic_AIDP(extrinsicModelId, pfiinC,
                                         T_BC0.C().transpose(), &dpC_dExtrinsic,
                                         &R_CfCa, &ab1rho);
      if (fixCameraIntrinsicParams_[camIdx]) {
        J_Xc << pointJacobian3 * dpC_dExtrinsic, J_td, J_tr;
      } else {
        ProjectionOptKneadIntrinsicJacobian(projOptModelId,
                                            &intrinsicsJacobian);
        J_Xc << pointJacobian3 * dpC_dExtrinsic, intrinsicsJacobian, J_td, J_tr;
      }
    }

    Eigen::Matrix3d tempM3d;
    tempM3d << T_CA.C().topLeftCorner<3, 2>(), T_CA.r();
    J_pfi = pointJacobian3 * tempM3d;

    Eigen::Vector3d pfinG = (T_GA * ab1rho).head<3>();
    factorJ_XBj << -rho * Eigen::Matrix3d::Identity(),
        okvis::kinematics::crossMx(pfinG - lP_T_WB.r() * rho),
        -rho * Eigen::Matrix3d::Identity() * featureTime;
    J_XBj = pointJacobian3 * (T_WB.C() * T_BC0.C()).transpose() * factorJ_XBj;

    factorJ_XBa.topLeftCorner<3, 3>() = rho * Eigen::Matrix3d::Identity();
    factorJ_XBa.block<3, 3>(0, 3) =
        -okvis::kinematics::crossMx(T_WBa.C() * (T_BC0 * ab1rho).head<3>());
    factorJ_XBa.block<3, 3>(0, 6) = Eigen::Matrix3d::Zero();
    J_XBa = pointJacobian3 * (T_WB.C() * T_BC0.C()).transpose() * factorJ_XBa;

    H_x.setZero();
    H_x.topLeftCorner(2, minCamParamDim) = J_Xc;
    uint64_t poseId = itFrameIds->frameId;
    if (poseId == anchorId) {
      std::map<uint64_t, int>::const_iterator poseid_iter =
          mStateID2CovID_.find(poseId);
      H_x.block<2, 6>(0, minCamParamDim +
                             9 * poseid_iter->second + 3) =
          (J_XBj + J_XBa).block<2, 6>(0, 3);
    } else {
      std::map<uint64_t, int>::const_iterator poseid_iter =
          mStateID2CovID_.find(poseId);
      H_x.block<2, 9>(0, minCamParamDim +
                             9 * poseid_iter->second) = J_XBj;
      std::map<uint64_t, int>::const_iterator anchorid_iter =
          mStateID2CovID_.find(anchorId);
      H_x.block<2, 9>(0, minCamParamDim +
                             9 * anchorid_iter->second) = J_XBa;
    }

    vJ_X.push_back(H_x);
    vJ_pfi.push_back(J_pfi);

    ++numValidObs;
    ++itFrameIds;
    itRoi += 2;
  }
  // What if the Jacobians of the anchor frame is invalid? It should be safe to ignore it.
  if (numValidObs < minTrackLength_) {
    computeHTimer.stop();
    return false;
  }
  OKVIS_ASSERT_EQ_DBG(Exception, numValidObs * 2, vSigmai.size(),
                      "Inconsistent number of observations and frameIds");

  // Now we stack the Jacobians and marginalize the point position related
  // dimensions. In other words, project $H_{x_i}$ onto the nullspace of
  // $H_{f^i}$

  Eigen::MatrixXd H_xi(2 * numValidObs, numCamPoseStates);
  Eigen::MatrixXd H_fi(2 * numValidObs, 3);
  Eigen::Matrix<double, Eigen::Dynamic, 1> ri(2 * numValidObs, 1);
  Eigen::MatrixXd Ri =
      Eigen::MatrixXd::Identity(2 * numValidObs, 2 * numValidObs);
  for (size_t saga = 0; saga < numValidObs; ++saga) {
    size_t saga2 = saga * 2;
    H_xi.block(saga2, 0, 2, numCamPoseStates) = vJ_X[saga];
    H_fi.block<2, 3>(saga2, 0) = vJ_pfi[saga];
    ri.segment<2>(saga2) = vri[saga];
    Ri(saga2, saga2) = vSigmai[saga2] * vSigmai[saga2];
    Ri(saga2 + 1, saga2 + 1) = vSigmai[saga2 + 1] * vSigmai[saga2 + 1];
  }

  if (pH_fi)  // this point is to be included in the states
  {
    r_oi = ri;
    H_oi = H_xi;
    R_oi = Ri;
    *pH_fi = H_fi;
  } else {
    Eigen::MatrixXd nullQ = vio::nullspace(H_fi);  // 2nx(2n-3), n==numValidObs
    OKVIS_ASSERT_EQ_DBG(Exception, nullQ.cols(), (int)(2 * numValidObs - 3),
                        "Nullspace of Hfi should have 2n-3 columns");
    //    OKVIS_ASSERT_LT(Exception, (nullQ.transpose()* H_fi).norm(), 1e-6,
    //    "nullspace is not correct!");
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

void HybridFilter::updateImuAugmentedStates(const States& stateInQuestion, const Eigen::VectorXd deltaAugmentedParams) {
  const int imuIdx = 0;
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

void HybridFilter::cloneCameraParameterStates(
    const States& stateInQuestion,
    StatePointerAndEstimateList*
        currentStates) const {
  const int camIdx = 0;
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
  cloneCameraParameterStates(stateInQuestion, currentStates);

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
  const int camIdx = 0;
  if (!fixCameraExtrinsicParams_[camIdx]) {
    int extrinsicOptModelId = camera_rig_.getExtrinsicOptMode(camIdx);
    int minExtrinsicDim = camera_rig_.getMinimalExtrinsicDimen(camIdx);
    Eigen::VectorXd delta(minExtrinsicDim);
    ExtrinsicModelBoxminus(
        extrinsicOptModelId,
        refStates.at(stateBlockIndex).parameterBlockPtr->parameters(),
        refStates.at(stateBlockIndex).parameterEstimate.data(), delta.data());
    deltaX->segment(covStateIndex, minExtrinsicDim) = delta;
    covStateIndex += minExtrinsicDim;
    ++stateBlockIndex;
  }

  if (!fixCameraIntrinsicParams_[camIdx]) {
    const int minProjectionDim = camera_rig_.getMinimalProjectionDimen(camIdx);
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
  updateStatesTimer.start();
  const size_t numNavImuCamStates = startIndexOfClonedStates();
  // number of navigation, imu, and camera states in the covariance
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
      okvis::ceres::expAndTheta(deltaAlpha);
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

  // update camera sensor states
  int camParamIndex = startIndexOfCameraParams();
  const int camIdx = 0;

  if (!fixCameraExtrinsicParams_[camIdx]) {
    uint64_t extrinsicId = stateInQuestion.sensors.at(SensorStates::Camera)
                               .at(camIdx)
                               .at(CameraSensorStates::T_SCi)
                               .id;
    std::shared_ptr<ceres::PoseParameterBlock> extrinsicParamBlockPtr =
        std::static_pointer_cast<ceres::PoseParameterBlock>(
            mapPtr_->parameterBlockPtr(extrinsicId));

    kinematics::Transformation T_BC0 = extrinsicParamBlockPtr->estimate();
    Eigen::Vector3d t_BC;
    Eigen::Quaterniond q_BC;
    int extrinsicOptModelId = camera_rig_.getExtrinsicOptMode(camIdx);
    int minExtrinsicDim = camera_rig_.getMinimalExtrinsicDimen(camIdx);
    ExtrinsicModelUpdateState(extrinsicOptModelId, T_BC0.r(), T_BC0.q(),
                              deltaX.segment(camParamIndex, minExtrinsicDim),
                              &t_BC, &q_BC);
    extrinsicParamBlockPtr->setEstimate(kinematics::Transformation(t_BC, q_BC));
    camParamIndex += minExtrinsicDim;
  }

  if (!fixCameraIntrinsicParams_[camIdx]) {
    const int minProjectionDim = camera_rig_.getMinimalProjectionDimen(camIdx);
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

  // Update cloned states except for the last one, the current state,
  // which is already updated early on.
  size_t jack = 0;
  auto finalIter = statesMap_.end();
  --finalIter;

  for (auto iter = statesMap_.begin(); iter != finalIter; ++iter, ++jack) {
    stateId = iter->first;
    size_t qStart = startIndexOfClonedStates() + 3 + kClonedStateMinimalDimen * jack;

    poseParamBlockPtr = std::static_pointer_cast<ceres::PoseParameterBlock>(
        mapPtr_->parameterBlockPtr(stateId));
    T_WS = poseParamBlockPtr->estimate();
    deltaAlpha = deltaX.segment<3>(qStart);
    deltaq = okvis::ceres::expAndTheta(deltaAlpha);
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
  int kale = 0;
  size_t lkStart = startIndexOfClonedStates() + kClonedStateMinimalDimen * statesMap_.size();
  size_t aStart = lkStart - 3;  // a dummy initialization
  for (auto iter = mInCovLmIds.begin(), iterEnd = mInCovLmIds.end();
       iter != iterEnd; ++iter, ++kale) {
    std::shared_ptr<okvis::ceres::HomogeneousPointParameterBlock> hppb =
        std::static_pointer_cast<okvis::ceres::HomogeneousPointParameterBlock>(
            mapPtr_->parameterBlockPtr(*iter));
    Eigen::Vector4d ab1rho =
        hppb->estimate();  // inverse depth parameterization in the anchor
                           // frame, [\alpha= X/Z, \beta= Y/Z, 1, \rho=1/Z]
    aStart = lkStart + 3 * kale;
    ab1rho[0] += deltaX[aStart];
    ab1rho[1] += deltaX[aStart + 1];
    ab1rho[3] += deltaX[aStart + 2];
    hppb->setEstimate(ab1rho);
  }
  OKVIS_ASSERT_EQ_DBG(Exception, aStart + 3, (size_t)deltaX.rows(),
                      "deltaX size not equal to what's' expected.");
  updateStatesTimer.stop();

  updateSensorRigs();
}

void HybridFilter::optimize(size_t /*numIter*/, size_t /*numThreads*/,
                            bool verbose) {
  // containers of Jacobians of measurements of marginalized features
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, 1>,
      Eigen::aligned_allocator<Eigen::Matrix<double, Eigen::Dynamic, 1>>>
      vr_o;
  std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>
      vH_o;  // each entry (2n-3)x(13+9m), n, number of observations, m,
             // states in the sliding window
  std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>
      vR_o;  // each entry (2n-3)x(2n-3)
  // containers of Jacobians of measurements of points in the states
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> vr_i;
  std::vector<
      Eigen::Matrix<double, 2, Eigen::Dynamic>,
      Eigen::aligned_allocator<Eigen::Matrix<double, 2, Eigen::Dynamic>>>
      vH_x;  // each entry 2x(42+13+ 9m)
  std::vector<
      Eigen::Matrix<double, 2, Eigen::Dynamic>,
      Eigen::aligned_allocator<Eigen::Matrix<double, 2, Eigen::Dynamic>>>
      vH_f;  // each entry 2x(3s_k)
  std::vector<Eigen::Matrix2d,
              Eigen::aligned_allocator<Eigen::Matrix<double, 2, 2>>>
      vR_i;

  const uint64_t currFrameId = currentFrameId();
  int covDim = covariance_.rows();

  OKVIS_ASSERT_EQ_DBG(
      Exception, (size_t)covariance_.rows(),
      startIndexOfClonedStates() +
          kClonedStateMinimalDimen * statesMap_.size() + 3 * mInCovLmIds.size(),
      "Inconsistent covDim and number of states");

  int numCamPosePointStates = cameraParamPoseAndLandmarkMinimalDimen();
  size_t dimH_o[2] = {0, numCamPosePointStates - 3 * mInCovLmIds.size() - 9};
  size_t nMarginalizedFeatures =
      0;  // features not in state and not tracked in current frame
  size_t nInStateFeatures = 0;  // features in state and tracked now
  size_t nToAddFeatures =
      0;  // features tracked long enough and to be included in states

  Eigen::MatrixXd variableCov = covariance_.block(
      okvis::ceres::ode::OdoErrorStateDim, okvis::ceres::ode::OdoErrorStateDim,
      dimH_o[1],
      dimH_o[1]);  // covariance of camera and pose copy states
  Eigen::MatrixXd variableCov2 = covariance_.block(
      okvis::ceres::ode::OdoErrorStateDim, okvis::ceres::ode::OdoErrorStateDim,
      dimH_o[1] + 9,
      dimH_o[1] + 9);  // covariance of camera and pose copy states

  for (okvis::PointMap::iterator it = landmarksMap_.begin(); it != landmarksMap_.end();
       ++it) {
    ResidualizeCase toResidualize = NotInState_NotTrackedNow;
    const size_t nNumObs = it->second.observations.size();
    if (it->second.anchorStateId == 0) {  // this point is not in the states
      for (auto itObs = it->second.observations.rbegin(),
                iteObs = it->second.observations.rend();
           itObs != iteObs; ++itObs) {
        if (itObs->first.frameId == currFrameId) {
          if (nNumObs == maxTrackLength_) {
            // this point is to be included in the states
            toResidualize = ToAdd_TrackedNow;
            ++nToAddFeatures;
          } else {
            std::stringstream ss;
            ss << "A point not in state should not have consecutive"
                  " features more than " << maxTrackLength_;
            OKVIS_ASSERT_LT_DBG(Exception, nNumObs, maxTrackLength_,
                                ss.str());
            toResidualize = NotToAdd_TrackedNow;
          }
          break;
        }
      }
    } else {
      toResidualize = InState_NotTrackedNow;
      for (auto itObs = it->second.observations.rbegin(),
                iteObs = it->second.observations.rend();
           itObs != iteObs; ++itObs) {
        if (itObs->first.frameId ==
            currFrameId) {  // point in states are still tracked so far
          toResidualize = InState_TrackedNow;
          break;
        }
      }
    }
    it->second.residualizeCase = toResidualize;

    if (toResidualize == NotInState_NotTrackedNow &&
        nNumObs >= minTrackLength_) {
      Eigen::MatrixXd H_oi;                           //(2n-3, dimH_o[1])
      Eigen::Matrix<double, Eigen::Dynamic, 1> r_oi;  //(2n-3, 1)
      Eigen::MatrixXd R_oi;                           //(2n-3, 2n-3)
      Eigen::Vector4d ab1rho;
      bool isValidJacobian =
          featureJacobian(it->second, H_oi, r_oi, R_oi, ab1rho);
      if (!isValidJacobian) {
          continue;
      }
      if (!FilterHelper::gatingTest(H_oi, r_oi, R_oi, variableCov)) {
        continue;
      }

      vr_o.push_back(r_oi);
      vR_o.push_back(R_oi);
      vH_o.push_back(H_oi);
      dimH_o[0] += r_oi.rows();
      ++nMarginalizedFeatures;
    } else if (toResidualize == InState_TrackedNow) {
      // compute residual and Jacobian for a observed point which is in the states
      Eigen::Matrix<double, 2, 1> r_i;
      Eigen::Matrix<double, 2, Eigen::Dynamic> H_x;
      Eigen::Matrix<double, 2, Eigen::Dynamic> H_f;
      Eigen::Matrix2d R_i;
      bool isValidJacobian =
          slamFeatureJacobian(it->first, it->second, r_i, H_x, H_f, R_i);
      if (!isValidJacobian) continue;

      if (!FilterHelper::gatingTest(H_x, r_i, R_i, variableCov2)) {
        it->second.residualizeCase = InState_NotTrackedNow;
        continue;
      }

      vr_i.push_back(r_i);
      vH_x.push_back(H_x);
      vH_f.push_back(H_f);
      vR_i.push_back(R_i);
      ++nInStateFeatures;
    }
  }  // every landmark

  if (dimH_o[0] + 2 * nInStateFeatures > 0) {
    computeKalmanGainTimer.start();
    std::cout << "kalman observation dimH_o and 2*tracked instateFeatures "
              << dimH_o[0] << " " << 2 * nInStateFeatures << std::endl;
    // stack Jacobians and residuals for only marginalized features, prepare
    // for QR decomposition to reduce dimension
    Eigen::MatrixXd H_o(dimH_o[0], dimH_o[1]);
    Eigen::Matrix<double, Eigen::Dynamic, 1> r_o(dimH_o[0], 1);
    Eigen::MatrixXd R_o = Eigen::MatrixXd::Zero(dimH_o[0], dimH_o[0]);
    size_t startRow = 0;
    // marginalized features
    for (size_t jack = 0; jack < nMarginalizedFeatures; ++jack) {
      H_o.block(startRow, 0, vH_o[jack].rows(), dimH_o[1]) = vH_o[jack];
      r_o.block(startRow, 0, vH_o[jack].rows(), 1) = vr_o[jack];
      R_o.block(startRow, startRow, vH_o[jack].rows(), vH_o[jack].rows()) =
          vR_o[jack];
      startRow += vH_o[jack].rows();
    }

    Eigen::MatrixXd r_q, T_H,
        R_q;  // residual, Jacobian, and noise covariance after projecting to
              // the column space of H_o
    if (r_o.rows() <= static_cast<int>(dimH_o[1])) {
      // no need to reduce rows of H_o
      r_q = r_o;
      T_H = H_o;
      R_q = R_o;
    } else {  // project into the column space of H_o, reduce the residual
              // dimension
      Eigen::HouseholderQR<Eigen::MatrixXd> qr(H_o);
      Eigen::MatrixXd Q = qr.householderQ();
      Eigen::MatrixXd thinQ = Q.topLeftCorner(dimH_o[0], dimH_o[1]);

      r_q = thinQ.transpose() * r_o;
      R_q = thinQ.transpose() * R_o * thinQ;

      Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();
      for (size_t row = 0; row < dimH_o[1]; ++row) {
        for (size_t col = 0; col < dimH_o[1]; ++col) {
          if (std::fabs(R(row, col)) < 1e-10) R(row, col) = 0;
        }
      }
      T_H = R.topLeftCorner(dimH_o[1], dimH_o[1]);
    }

    // stack Jacobians and residuals for features in state, i.e., SLAM
    // features
    const size_t rqRows = r_q.rows();
    const size_t obsRows = rqRows + 2 * nInStateFeatures;
    const size_t numPointStates = 3 * mInCovLmIds.size();
    Eigen::MatrixXd H_all(obsRows, numCamPosePointStates);

    H_all.topLeftCorner(rqRows, dimH_o[1]) = T_H;
    H_all.block(0, dimH_o[1], rqRows, numPointStates + 9).setZero();

    Eigen::Matrix<double, Eigen::Dynamic, 1> r_all(obsRows, 1);
    r_all.head(rqRows) = r_q;
    Eigen::MatrixXd R_all = Eigen::MatrixXd::Zero(obsRows, obsRows);
    R_all.topLeftCorner(rqRows, rqRows) = R_q;

    startRow = rqRows;
    for (size_t jack = 0; jack < nInStateFeatures; ++jack) {
      H_all.block(startRow, 0, 2, numCamPosePointStates - numPointStates) =
          vH_x[jack];
      H_all.block(startRow, numCamPosePointStates - numPointStates, 2,
                  numPointStates) = vH_f[jack];
      r_all.block<2, 1>(startRow, 0) = vr_i[jack];
      R_all.block<2, 2>(startRow, startRow) = vR_i[jack];
      startRow += 2;
    }

    // Calculate Kalman gain
    Eigen::MatrixXd S =
        H_all *
            covariance_.bottomRightCorner(numCamPosePointStates,
                                          numCamPosePointStates) *
            H_all.transpose() +
        R_all;

    Eigen::MatrixXd K =
        (covariance_.bottomRightCorner(covDim, numCamPosePointStates) *
         H_all.transpose()) *
        S.inverse();

    // State correction
    Eigen::Matrix<double, Eigen::Dynamic, 1> deltaX = K * r_all;
    OKVIS_ASSERT_FALSE(Exception,
                       std::isnan(deltaX(0)) || std::isnan(deltaX(3)),
                       "nan in kalman filter's correction");

    computeKalmanGainTimer.stop();
    updateStates(deltaX);

    // Covariance correction
    updateCovarianceTimer.start();
#if 0  // Joseph form
        Eigen::MatrixXd tempMat = Eigen::MatrixXd::Identity(covDim, covDim);
        tempMat.block(0, okvis::ceres::ode::OdoErrorStateDim, covDim, numCamPosePointStates) -= K*H_all;
        covariance_ = tempMat * (covariance_ * tempMat.transpose()).eval() + K * R_all * K.transpose();
#else  // Li Mingyang RSS 12 optimization based..., positive semi-definiteness
       // not necessarily maintained
    covariance_ = covariance_ - K * S * K.transpose();
#endif
    if (covariance_.diagonal().minCoeff() < 0) {
      std::cout << "Warn: current diagonal in normal update " << std::endl
                << covariance_.diagonal().transpose() << std::endl;
      covariance_.diagonal() =
          covariance_.diagonal().cwiseAbs();  // TODO: hack is ugly!
      //        OKVIS_ASSERT_GT(Exception, covariance_.diagonal().minCoeff(),
      //        0, "negative covariance diagonal elements");
    }
    // another check the copied state should have the same covariance as its
    // source
    const size_t numNavImuCamStates = startIndexOfClonedStates();
    const size_t numNavImuCamPoseStates =
        numNavImuCamStates + 9 * statesMap_.size();
    if ((covariance_.topLeftCorner(covDim, 9) -
         covariance_.block(0, numNavImuCamPoseStates - 9, covDim, 9))
            .lpNorm<Eigen::Infinity>() > 1e-8) {
      std::cout << "Warn: Covariance of cloned state is not equal to source "
                << std::endl;
    }

    updateCovarianceTimer.stop();
  } else {
    LOG(WARNING) << "zero valid support from #landmarks:"
                 << landmarksMap_.size();
  }

  updateLandmarksTimer.start();
  if (nToAddFeatures) {
    /// initialize features tracked in all m images of the sliding window, see
    /// Li RSS 12 supplement material
    // intermediate variables
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
    vH_o.clear();
    vR_o.clear();

    const int camIdx = 0;
    const okvis::kinematics::Transformation T_BC0 = camera_rig_.getCameraExtrinsic(camIdx);
    size_t totalObsDim = 0;  // total dimensions of all features' observations
    const size_t numCamPoseStates =
        numCamPosePointStates - 3 * mInCovLmIds.size();
    Eigen::MatrixXd variableCov = covariance_.block(
        okvis::ceres::ode::OdoErrorStateDim,
        okvis::ceres::ode::OdoErrorStateDim, numCamPoseStates,
        numCamPoseStates);  // covariance of camera and pose copy states

    std::vector<uint64_t> toAddLmIds;  // id of landmarks to add to the states
    for (okvis::PointMap::iterator pit = landmarksMap_.begin();
         pit != landmarksMap_.end(); ++pit) {
      if (pit->second.residualizeCase == ToAdd_TrackedNow) {
        Eigen::Vector4d
            ab1rho;  //[\alpha, \beta, 1, \rho] of the point in the anchor
                     // frame, representing either an ordinary point or a ray
        bool isValidJacobian =
            featureJacobian(pit->second, H_i, r_i, R_i, ab1rho, &H_fi);

        if (!isValidJacobian) {  // remove this feature later
          pit->second.residualizeCase =
              NotInState_NotTrackedNow;
          continue;
        }

        vio::leftNullspaceAndColumnSpace(H_fi, &Q2, &Q1);
        z_o = Q2.transpose() * r_i;
        H_o = Q2.transpose() * H_i;
        R_o = Q2.transpose() * R_i * Q2;

        if (!FilterHelper::gatingTest(H_o, z_o, R_o, variableCov)) {
            pit->second.residualizeCase = NotInState_NotTrackedNow;
            continue;
        }

        // get homogeneous point parameter block ASSUMING it is created during
        // feature tracking, reset its estimate with inverse depth parameters
        // [\alpha, \beta, 1, \rho]
        std::shared_ptr<okvis::ceres::HomogeneousPointParameterBlock> hppb =
            std::static_pointer_cast<
                okvis::ceres::HomogeneousPointParameterBlock>(
                mapPtr_->parameterBlockPtr(pit->first));
        hppb->setEstimate(ab1rho);  // for debugging, we may compare the new and
                                    // old value of this triangulated point

        okvis::kinematics::Transformation
            T_GBa;  // transform from the body frame at the anchor frame epoch
                    // to the world frame
        get_T_WS(currFrameId, T_GBa);
        okvis::kinematics::Transformation T_GA = T_GBa * T_BC0;  // anchor frame to global frame

        // update members of the map point
        pit->second.anchorStateId = currFrameId;
        pit->second.p_BA_G = T_GA.r() - T_GBa.r();
        pit->second.q_GA = T_GA.q();

        toAddLmIds.push_back(pit->first);

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

    // augment and update the covariance matrix
    size_t nNewFeatures = toAddLmIds.size();
    if (nNewFeatures) {
      std::cout << "start initializing features into states " << nNewFeatures
                << std::endl;
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

      int tempCounter = 0;
      size_t startRow = 0;
      for (auto it = toAddLmIds.begin(), itEnd = toAddLmIds.end(); it != itEnd;
           ++it, ++tempCounter) {
        H_o.block(startRow, 0, vH_o[tempCounter].rows(), numCamPoseStates) =
            vH_o[tempCounter];
        H_1.block(3 * tempCounter, 0, 3, numCamPoseStates) = vH_1[tempCounter];
        invH_2.block<3, 3>(3 * tempCounter, 3 * tempCounter) =
            vH_2[tempCounter].inverse();
        R_o.block(startRow, startRow, vH_o[tempCounter].rows(),
                  vH_o[tempCounter].rows()) = vR_o[tempCounter];
        R_1.block<3, 3>(3 * tempCounter, 3 * tempCounter) = vR_1[tempCounter];
        z_1.segment<3>(3 * tempCounter) = vz_1[tempCounter];
        z_o.segment(startRow, vH_o[tempCounter].rows()) = vz_o[tempCounter];
        startRow += vH_o[tempCounter].rows();
      }
      // TODO: in case H_o has too many rows, we should update the state and
      // covariance for each feature, or use the qr decomposition approach as
      // in MSCKF in that case H_o is replaced by T_H, and z_o is replaced by
      // Q_1'*z_o

      Eigen::MatrixXd S =
          H_o *
              covariance_.block(okvis::ceres::ode::OdoErrorStateDim,
                                okvis::ceres::ode::OdoErrorStateDim,
                                numCamPoseStates, numCamPoseStates) *
              H_o.transpose() +
          R_o;

      Eigen::MatrixXd K =
          (covariance_.block(0, okvis::ceres::ode::OdoErrorStateDim, covDim,
                             numCamPoseStates) *
           H_o.transpose()) *
          S.inverse();

      updateCovarianceTimer.start();
      Eigen::MatrixXd Paug(covDim + nNewFeatures * 3,
                           covDim + nNewFeatures * 3);
#if 0  // Joseph form
            Eigen::MatrixXd tempMat = Eigen::MatrixXd::Identity(covDim, covDim);
            tempMat.block(0, okvis::ceres::ode::OdoErrorStateDim, covDim, numCamPoseStates) -= K*H_o;
            Paug.topLeftCorner(covDim, covDim) = tempMat * (covariance_ * tempMat.transpose()).eval() + K * R_o * K.transpose();
#else  // Li Mingyang RSS 12 optimization based
      Paug.topLeftCorner(covDim, covDim) =
          covariance_ - K * S * K.transpose();
#endif
      Eigen::MatrixXd invH2H1 = invH_2 * H_1;
      Paug.block(covDim, 0, 3 * nNewFeatures, covDim) =
          -invH2H1 * Paug.block(okvis::ceres::ode::OdoErrorStateDim, 0,
                                numCamPoseStates, covDim);
      Paug.block(0, covDim, covDim, 3 * nNewFeatures) =
          Paug.block(covDim, 0, 3 * nNewFeatures, covDim).transpose();
      Paug.bottomRightCorner(3 * nNewFeatures, 3 * nNewFeatures) =
          -Paug.block(covDim, okvis::ceres::ode::OdoErrorStateDim,
                      3 * nNewFeatures, numCamPoseStates) *
              invH2H1.transpose() +
          invH_2 * R_1 * invH_2.transpose();
      covariance_ = Paug;
      covDim = covariance_.rows();
      if (covariance_.diagonal().minCoeff() < 0) {
        std::cout << "Warn: current diagonal in adding points " << std::endl
                  << covariance_.diagonal().transpose() << std::endl;
        covariance_.diagonal() =
            covariance_.diagonal().cwiseAbs();  // TODO: hack is ugly!
        //        OKVIS_ASSERT_GT(Exception,
        //        covariance_.diagonal().minCoeff(), 0, "negative covariance
        //        diagonal elements");
      }
      // another check the copied state should have the same covariance as its
      // source
      const size_t numNavImuCamStates = startIndexOfClonedStates();
      const size_t numNavImuCamPoseStates =
          numNavImuCamStates + 9 * statesMap_.size();
      if ((covariance_.topLeftCorner(covDim, 9) -
           covariance_.block(0, numNavImuCamPoseStates - 9, covDim, 9))
              .lpNorm<Eigen::Infinity>() > 1e-8) {
        std::cout << "Warn: Covariance of cloned state is not equal to source "
                     "after inserting points "
                  << std::endl;
      }

      updateCovarianceTimer.stop();
      mInCovLmIds.insert(mInCovLmIds.end(), toAddLmIds.begin(),
                         toAddLmIds.end());

      // State correction
      Eigen::Matrix<double, Eigen::Dynamic, 1> deltaXo = K * z_o;
      Eigen::Matrix<double, Eigen::Dynamic, 1> deltaX(covDim, 1);
      deltaX.head(covDim - 3 * nNewFeatures) = deltaXo;
      deltaX.tail(3 * nNewFeatures) =
          -invH2H1 * deltaXo.segment(okvis::ceres::ode::OdoErrorStateDim,
                                     numCamPoseStates) +
          invH_2 * z_1;
      OKVIS_ASSERT_FALSE(Exception,
                         std::isnan(deltaX(0)) ||
                             std::isnan(deltaX(covDim - 3 * nNewFeatures)),
                         "nan in kalman filter's correction in adding points");
      updateStates(deltaX);
      std::cout << "finish initializing features into states " << std::endl;
    }
  }

  /// update minValidStateId_ for removing old states
  /// also update landmark positions which is only necessary when
  /// (1) landmark coordinates are used to predict the points projection in
  /// new frames OR (2) to visualize the points
  minValidStateId_ = currFrameId;
  for (auto it = landmarksMap_.begin(); it != landmarksMap_.end();
       ++it) {
    ResidualizeCase residualizeCase = it->second.residualizeCase;
    if (residualizeCase == NotInState_NotTrackedNow ||
        residualizeCase == InState_NotTrackedNow)
      continue;

    if (residualizeCase == NotToAdd_TrackedNow) {
      if (it->second.observations.size() <
          2)  // this happens with a just inserted landmark that has not been
              // triangulated.
        continue;

      auto itObs = it->second.observations.begin();
      if (itObs->first.frameId < minValidStateId_)
        minValidStateId_ = itObs->first.frameId;
    } else { // SLAM features
      it->second.quality = 1.0;
      // note this is position in the anchor frame
      it->second.pointHomog = std::static_pointer_cast<
                                  okvis::ceres::HomogeneousPointParameterBlock>(
                                  mapPtr_->parameterBlockPtr(it->first))
                                  ->estimate();

      double invDepth = it->second.pointHomog[3];
      if (invDepth < 1e-6) {
        it->second.quality = 0.0;
      }
    }
  }
  updateLandmarksTimer.stop();

  // summary output
  if (verbose) {
    LOG(INFO) << mapPtr_->summary.FullReport();
  }
}

// getters
bool HybridFilter::getImageDelay(uint64_t poseId, int camIdx,
                                okvis::Duration* td) const {
  double tdd;
  if (!getSensorStateEstimateAs<ceres::CameraTimeParamBlock>(
          poseId, camIdx, SensorStates::Camera, CameraSensorStates::TD, tdd)) {
    return false;
  }
  *td = okvis::Duration(tdd);
  return true;
}

int HybridFilter::getCameraExtrinsicOptType(size_t cameraIdx) const {
  return camera_rig_.getExtrinsicOptMode(cameraIdx);
}

// private stuff
size_t HybridFilter::gatherMapPointObservations(
    const MapPoint& mp,
    std::shared_ptr<const cameras::CameraBase> cameraGeometry,
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
  const int camIdx = 0;
  uint32_t imageHeight = camera_rig_.getCameraGeometry(camIdx)->imageHeight();
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
    // use the latest estimates for camera intrinsic parameters
    Eigen::Vector3d backProjectionDirection;
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
    // image pixel noise follows that in addObservation function
    imageNoiseStd->push_back(kpSize / 8);
    imageNoiseStd->push_back(kpSize / 8);

    std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr;
    getGlobalStateParameterBlockPtr(poseId, GlobalStates::T_WS, parameterBlockPtr);
    double kpN = measurement[1] / imageHeight - 0.5;
    pointDataPtr->addKeypointObservation(itObs->first, parameterBlockPtr, kpN);
  }
  return pointDataPtr->numObservations();
}

bool HybridFilter::hasLowDisparity(
    const std::vector<Eigen::Vector3d,
                      Eigen::aligned_allocator<Eigen::Vector3d>>& obsDirections,
    const std::vector<
        okvis::kinematics::Transformation,
        Eigen::aligned_allocator<okvis::kinematics::Transformation>>& T_WSs,
    const kinematics::Transformation& T_BC0) const {
  Eigen::VectorXd intrinsics;
  camera_rig_.getCameraGeometry(0)->getIntrinsics(intrinsics);
  double focalLength = intrinsics[0];
  double keypointAStdDev = 0.8 * FLAGS_epipolar_sigma_keypoint_size / 12.0;
  const double fourthRoot2 = 1.1892071150;
  double raySigma = fourthRoot2 * keypointAStdDev / focalLength;
  Eigen::Vector3d rayA_inA = obsDirections.front().normalized();
  Eigen::Vector3d rayB_inB = obsDirections.back().normalized();
  Eigen::Vector3d rayB_inA = (T_WSs.front().C() * T_BC0.C()).transpose() *
                             T_WSs.back().C() * T_BC0.C() * rayB_inB;
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
  std::vector<std::pair<uint64_t, int>> frameIds = pointDataPtr->frameIds();
  int observationIndex = 0;
  for (const std::pair<uint64_t, int>& frameAndCameraIndex : frameIds) {
    uint64_t frameId = frameAndCameraIndex.first;
    auto statesIter = statesMap_.find(frameId);
    pointDataPtr->setImuInfo(observationIndex, statesIter->second.timestamp,
                             statesIter->second.tdAtCreation,
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
      cameraTimeParameterPtrs = getCameraTimeParameterPtrs();
  pointDataPtr->setCameraTimeParameterPtrs(cameraTimeParameterPtrs[0],
                                           cameraTimeParameterPtrs[1]);

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
    std::shared_ptr<const cameras::CameraBase> cameraGeometry,
    const kinematics::Transformation& T_BC0,
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
      mp, cameraGeometry, pointDataPtr, &obsDirections, &obsInPixel,
      &imageNoiseStd, &badObservationIdentifiers);

  msckf::TriangulationStatus status;
  if (numObs < minTrackLength_) {
      triangulateTimer.stop();
      status.lackObservations = true;
      return status;
  }

  propagatePoseAndVelocityForMapPoint(pointDataPtr);

  std::vector<okvis::kinematics::Transformation,
              Eigen::aligned_allocator<okvis::kinematics::Transformation>>
      T_WSs = pointDataPtr->poseAtObservationList();

  if (checkDisparity) {
//    if (isPureRotation(mp)) {
//      triangulateTimer.stop();
//      status.raysParallel = true;
//      return false;
//    }
    if (hasLowDisparity(obsDirections, T_WSs, T_BC0)) {
      triangulateTimer.stop();
      status.raysParallel = true;
      return status;
    }
  }

  std::vector<std::pair<uint64_t, int>> frameIds = pointDataPtr->frameIds();
  std::vector<uint64_t> anchorIds;
  std::vector<int> anchorSeqIds;
  if (orderedCulledFrameIds) {
    msckf::eraseBadObservations(badObservationIdentifiers, orderedCulledFrameIds);
    msckf::decideAnchors(frameIds, *orderedCulledFrameIds, pointLandmark.modelId(),
                         &anchorIds, &anchorSeqIds);
  } else {
    msckf::decideAnchors(frameIds, pointLandmark.modelId(), &anchorIds, &anchorSeqIds);
  }
  pointDataPtr->setAnchors(anchorIds, anchorSeqIds);

  status = pointLandmark.initialize(T_WSs, obsDirections, T_BC0, anchorSeqIds);
  triangulateTimer.stop();
  return status;
}

bool HybridFilter::print(std::ostream& stream) const {
  Estimator::print(stream);
  Eigen::IOFormat spaceInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols,
                               " ", " ", "", "", "", "");
  const States stateInQuestion = statesMap_.rbegin()->second;
  Eigen::Matrix<double, Eigen::Dynamic, 1> extraParams;
  getImuAugmentedStatesEstimate(&extraParams);
  stream << " " << extraParams.transpose().format(spaceInitFmt);

  // camera sensor states
  const int camIdx = 0;
  uint64_t extrinsicId = stateInQuestion.sensors.at(SensorStates::Camera)
                             .at(camIdx)
                             .at(CameraSensorStates::T_SCi)
                             .id;
  std::shared_ptr<ceres::PoseParameterBlock> extrinsicParamBlockPtr =
      std::static_pointer_cast<ceres::PoseParameterBlock>(
          mapPtr_->parameterBlockPtr(extrinsicId));
  kinematics::Transformation T_BC0 = extrinsicParamBlockPtr->estimate();
  std::string extrinsicValues;
  ExtrinsicModelToParamsValueString(camera_rig_.getExtrinsicOptMode(camIdx),
                                    T_BC0, " ", &extrinsicValues);
  stream << " " << extrinsicValues;
  Eigen::VectorXd cameraParams;
  getCameraCalibrationEstimate(camIdx, &cameraParams);
  stream << " " << cameraParams.transpose().format(spaceInitFmt);

  // stds
  const int stateDim = startIndexOfClonedStates();
  Eigen::Matrix<double, Eigen::Dynamic, 1> variances =
      covariance_.topLeftCorner(stateDim, stateDim).diagonal();
  stream << " " << variances.cwiseSqrt().transpose().format(spaceInitFmt);
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
    int camIdx,
    Eigen::Matrix<double, Eigen::Dynamic, 1>* cameraParams) const {
  const uint64_t poseId = statesMap_.rbegin()->first;
  Eigen::VectorXd intrinsic;

  getSensorStateEstimateAs<ceres::EuclideanParamBlock>(
      poseId, camIdx, SensorStates::Camera, CameraSensorStates::Intrinsics,
      intrinsic);

  Eigen::VectorXd distortionCoeffs;
  getSensorStateEstimateAs<ceres::EuclideanParamBlock>(
      poseId, camIdx, SensorStates::Camera, CameraSensorStates::Distortion,
      distortionCoeffs);
  cameraParams->resize(intrinsic.size() + distortionCoeffs.size() + 2, 1);
  cameraParams->head(intrinsic.size()) = intrinsic;
  cameraParams->segment(intrinsic.size(), distortionCoeffs.size()) =
          distortionCoeffs;
  double tdEstimate(0), trEstimate(0);
  getSensorStateEstimateAs<ceres::CameraTimeParamBlock>(
      poseId, camIdx, SensorStates::Camera, CameraSensorStates::TD, tdEstimate);
  getSensorStateEstimateAs<ceres::CameraTimeParamBlock>(
      poseId, camIdx, SensorStates::Camera, CameraSensorStates::TR, trEstimate);
  cameraParams->tail<2>() = Eigen::Vector2d(tdEstimate, trEstimate);
}

std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>
HybridFilter::getCameraTimeParameterPtrs() const {
  std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>
      cameraTimeParameterPtrs;
  cameraTimeParameterPtrs.reserve(2);
  const States& oneState = statesMap_.rbegin()->second;
  const int camIdx = 0;
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
  cameraTimeParameterPtrs.push_back(tdParamBlockPtr);
  cameraTimeParameterPtrs.push_back(trParamBlockPtr);
  return cameraTimeParameterPtrs;
}

std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>
HybridFilter::getImuAugmentedParameterPtrs() const {
  std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>
      imuParameterPtrs;
  imuParameterPtrs.reserve(3);
  const int imuIdx = 0;
  const States stateInQuestion = statesMap_.rbegin()->second;
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
  return imuParameterPtrs;
}

void HybridFilter::getImuAugmentedStatesEstimate(
    Eigen::Matrix<double, Eigen::Dynamic, 1>* extraParams) const {
  std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>> TgTsTaPtr =
      getImuAugmentedParameterPtrs();
  okvis::getImuAugmentedStatesEstimate(TgTsTaPtr, extraParams, imu_rig_.getModelId(0));
}

void HybridFilter::getStateVariance(
    Eigen::Matrix<double, Eigen::Dynamic, 1>* variances) const {
  const int dim = startIndexOfClonedStates();
  *variances = covariance_.topLeftCorner(dim, dim).diagonal();
}

void HybridFilter::setKeyframeRedundancyThresholds(double dist, double angle,
                                                   double trackingRate,
                                                   size_t minTrackLength) {
  translationThreshold_ = dist;
  rotationThreshold_ = angle;
  trackingRateThreshold_ = trackingRate;
  minTrackLength_ = minTrackLength;
}

okvis::Time HybridFilter::removeState(uint64_t stateId) {
  std::map<uint64_t, States>::iterator it = statesMap_.find(stateId);
  okvis::Time removedStateTime = it->second.timestamp;
  it->second.global[GlobalStates::T_WS].exists = false;  // remember we removed
  it->second.sensors.at(SensorStates::Imu)
      .at(0)
      .at(ImuSensorStates::SpeedAndBias)
      .exists = false;  // remember we removed
  mapPtr_->removeParameterBlock(it->second.global[GlobalStates::T_WS].id);
  mapPtr_->removeParameterBlock(it->second.sensors.at(SensorStates::Imu)
                                    .at(0)
                                    .at(ImuSensorStates::SpeedAndBias)
                                    .id);


  multiFramePtrMap_.erase(stateId);
  statesMap_.erase(it);
  return removedStateTime;
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

  const int minCamParamDim = filter_.cameraParamsMinimalDimen();
  for (int j = 0; j < 2; ++j) {
    uint64_t poseId = pointDataPtr->frameId(observationIndexPair[j]);
    std::map<uint64_t, int>::const_iterator poseid_iter =
        filter_.mStateID2CovID_.find(poseId);
    int covid = poseid_iter->second;
    int startIndex = minCamParamDim + 9 * covid;
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
      mp, tempCameraGeometry, pointDataPtr.get(), &obsDirections, &obsInPixels,
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
  pointDataPtr->computeSharedJacobians(cameraObservationModelId_);

  // enlarge cov of the head obs to counteract the noise reduction
  // due to correlation in head_tail scheme
  size_t trackLength = mp.observations.size();
  double headObsCovModifier[2] = {1.0, 1.0};
  headObsCovModifier[0] =
      seqType == LATEST_TWO
          ? 1.0 : (static_cast<double>(trackLength - minTrackLength_ + 2u));

  std::vector<std::pair<int, int>> featurePairs =
      TwoViewPair::getFramePairs(numValidDirectionJac, TwoViewPair::FIXED_HEAD_RECEDING_TAIL);
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

uint64_t HybridFilter::getMinValidStateId() const {
  uint64_t min_state_id = statesMap_.rbegin()->first;
  for (auto it = landmarksMap_.begin(); it != landmarksMap_.end();
       ++it) {
    if (it->second.residualizeCase == NotInState_NotTrackedNow)
      continue;

    auto itObs = it->second.observations.begin();
    if (itObs->first.frameId <
        min_state_id) {  // this assume that it->second.observations is an
                         // ordered map
      min_state_id = itObs->first.frameId;
    }
  }
  OKVIS_ASSERT_LE(Exception, min_state_id, currentKeyframeId(),
                  "Removing the current keyframe!");
  return min_state_id;
}

}  // namespace okvis
