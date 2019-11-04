#include <msckf/TFVIO.hpp>

#include <glog/logging.h>

#include <okvis/IdProvider.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/RelativePoseError.hpp>
#include <okvis/ceres/SpeedAndBiasError.hpp>

#include <okvis/ceres/CameraTimeParamBlock.hpp>
#include <msckf/EuclideanParamBlock.hpp>

#include <msckf/ImuOdometry.h>
#include <msckf/PreconditionedEkfUpdater.h>

#include <msckf/FilterHelper.hpp>
#include <msckf/TwoViewPair.hpp>

// the following #include's are only for testing
#include <okvis/timing/Timer.hpp>
#include "vio/ImuErrorModel.h"
#include "vio/Sample.h"

DECLARE_bool(use_AIDP);

DECLARE_bool(use_mahalanobis);
DECLARE_bool(use_first_estimate);
DECLARE_bool(use_RK4);

DECLARE_bool(use_IEKF);

DECLARE_double(max_proj_tolerance);

DEFINE_int32(
    two_view_constraint_scheme, 0,
    "0 the entire feature track of a landmark is used to "
    "compose two-view constraints which are used in one filter update step "
    "as the landmark disappears; "
    "1, use the latest two observations of a landmark to "
    "form one two-view constraint in one filter update step; "
    "2, use the fixed head observation and "
    "the receding tail observation of a landmark to "
    "form one two-view constraint in one filter update step");
DEFINE_double(
    image_noise_cov_multiplier, 9.0,
    "Enlarge the image observation noise covariance by this multiplier.");

/// \brief okvis Main namespace of this package.
namespace okvis {

TFVIO::TFVIO(std::shared_ptr<okvis::ceres::Map> mapPtr,
             const double readoutTime)
    : HybridFilter(mapPtr, readoutTime) {}

// The default constructor.
TFVIO::TFVIO(const double readoutTime) : HybridFilter(readoutTime) {}

TFVIO::~TFVIO() {}

// TODO(jhuai): merge with the superclass implementation
bool TFVIO::addStates(okvis::MultiFramePtr multiFrame,
                      const okvis::ImuMeasurementDeque& imuMeasurements,
                      bool asKeyframe) {
  // note: this is before matching...
  okvis::kinematics::Transformation T_WS;
  okvis::SpeedAndBiases speedAndBias;
  okvis::Duration tdEstimate;
  okvis::Time correctedStateTime;  // time of current multiFrame corrected with
                                   // current td estimate

  Eigen::Matrix<double, 27, 1> vTgTsTa;
  int covDim = covariance_.rows();

  if (statesMap_.empty()) {
    // in case this is the first frame ever, let's initialize the pose:
    tdEstimate.fromSec(imuParametersVec_.at(0).td0);
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

    vTgTsTa.head<9>() = imuParametersVec_.at(0).Tg0;
    vTgTsTa.segment<9>(9) = imuParametersVec_.at(0).Ts0;
    vTgTsTa.tail<9>() = imuParametersVec_.at(0).Ta0;

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

    uint64_t shapeMatrix_id = statesMap_.rbegin()
                                  ->second.sensors.at(SensorStates::Imu)
                                  .at(0)
                                  .at(ImuSensorStates::TG)
                                  .id;
    vTgTsTa.head<9>() = std::static_pointer_cast<ceres::ShapeMatrixParamBlock>(
                            mapPtr_->parameterBlockPtr(shapeMatrix_id))
                            ->estimate();

    shapeMatrix_id = statesMap_.rbegin()
                         ->second.sensors.at(SensorStates::Imu)
                         .at(0)
                         .at(ImuSensorStates::TS)
                         .id;
    vTgTsTa.segment<9>(9) =
        std::static_pointer_cast<ceres::ShapeMatrixParamBlock>(
            mapPtr_->parameterBlockPtr(shapeMatrix_id))
            ->estimate();

    shapeMatrix_id = statesMap_.rbegin()
                         ->second.sensors.at(SensorStates::Imu)
                         .at(0)
                         .at(ImuSensorStates::TA)
                         .id;
    vTgTsTa.tail<9>() = std::static_pointer_cast<ceres::ShapeMatrixParamBlock>(
                            mapPtr_->parameterBlockPtr(shapeMatrix_id))
                            ->estimate();

    // propagate pose, speedAndBias, and covariance
    okvis::Time startTime = statesMap_.rbegin()->second.timestamp;
    Eigen::Matrix<double, ceres::ode::OdoErrorStateDim,
                  ceres::ode::OdoErrorStateDim>
        Pkm1 = covariance_.topLeftCorner<ceres::ode::OdoErrorStateDim,
                                         ceres::ode::OdoErrorStateDim>();
    Eigen::Matrix<double, ceres::ode::OdoErrorStateDim,
                  ceres::ode::OdoErrorStateDim>
        F_tot;

    int numUsedImuMeasurements = -1;
    if (FLAGS_use_first_estimate) {
      /// use latest estimate to propagate pose, speed and bias, and first
      /// estimate to propagate covariance and Jacobian
      Eigen::Matrix<double, 6, 1> lP =
          statesMap_.rbegin()->second.linearizationPoint;
      Eigen::Vector3d tempV_WS = speedAndBias.head<3>();
      IMUErrorModel<double> tempIEM(speedAndBias.tail<6>(), vTgTsTa);
      numUsedImuMeasurements = IMUOdometry::propagation(
          imuMeasurements, imuParametersVec_.at(0), T_WS, tempV_WS, tempIEM,
          startTime, correctedStateTime, &Pkm1, &F_tot, &lP);
      speedAndBias.head<3>() = tempV_WS;
    } else {
      /// use latest estimate to propagate pose, speed and bias, and covariance
      if (FLAGS_use_RK4) {
        // method 1 RK4 a little bit more accurate but 4 times slower
        F_tot.setIdentity();
        numUsedImuMeasurements = IMUOdometry::propagation_RungeKutta(
            imuMeasurements, imuParametersVec_.at(0), T_WS, speedAndBias,
            vTgTsTa, startTime, correctedStateTime, &Pkm1, &F_tot);
      } else {
        // method 2, i.e., adapt the imuError::propagation function of okvis by
        // the TFVIO derivation in Michael Andrew Shelley
        Eigen::Vector3d tempV_WS = speedAndBias.head<3>();
        IMUErrorModel<double> tempIEM(speedAndBias.tail<6>(), vTgTsTa);
        numUsedImuMeasurements = IMUOdometry::propagation(
            imuMeasurements, imuParametersVec_.at(0), T_WS, tempV_WS, tempIEM,
            startTime, correctedStateTime, &Pkm1, &F_tot);
        speedAndBias.head<3>() = tempV_WS;
      }
    }

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

    if (numUsedImuMeasurements < 2) {
      LOG(WARNING) << "numUsedImuMeasurements=" << numUsedImuMeasurements
                   << " correctedStateTime " << correctedStateTime
                   << " lastFrameTimestamp " << startTime << " tdEstimate "
                   << tdEstimate << std::endl;
    }
  }

  // create a states object:
  States states(asKeyframe, multiFrame->id(), correctedStateTime, tdEstimate);

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
  states.linearizationPoint << T_WS.r(), speedAndBias.head<3>();

  if (statesMap_.empty()) {
    referencePoseId_ = states.id;  // set this as reference pose
  }
  mapPtr_->addParameterBlock(poseParameterBlock, ceres::Map::Pose6d);

  // add to buffer
  statesMap_.insert(std::pair<uint64_t, States>(states.id, states));
  //    std::cout<<"Added STATE OF ID "<<states.id<< std::endl;
  multiFramePtrMap_.insert(
      std::pair<uint64_t, okvis::MultiFramePtr>(states.id, multiFrame));

  // the following will point to the last states:
  std::map<uint64_t, States>::reverse_iterator lastElementIterator =
      statesMap_.rbegin();
  lastElementIterator++;

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
      const cameras::NCameraSystem& camSystem = multiFrame->GetCameraSystem();
      camera_rig_.addCamera(
          multiFrame->T_SC(i), multiFrame->GetCameraSystem().cameraGeometry(i),
          imageReadoutTime, tdEstimate.toSec(), camSystem.projOptRep(i),
          camSystem.extrinsicOptRep(i));
      const okvis::kinematics::Transformation T_SC =
          camera_rig_.getCameraExtrinsic(i);

      uint64_t id = IdProvider::instance().newId();
      std::shared_ptr<okvis::ceres::PoseParameterBlock>
          extrinsicsParameterBlockPtr(new okvis::ceres::PoseParameterBlock(
              T_SC, id, correctedStateTime));
      mapPtr_->addParameterBlock(extrinsicsParameterBlockPtr,
                                 ceres::Map::Pose6d);
      cameraInfos.at(CameraSensorStates::T_SCi).id = id;

      Eigen::VectorXd allIntrinsics;
      camera_rig_.getCameraGeometry(i)->getIntrinsics(allIntrinsics);
      id = IdProvider::instance().newId();
      int projOptModelId = camera_rig_.getProjectionOptMode(i);
      const int minProjectionDim = camera_rig_.getMinimalProjectionDimen(i);
      if (minProjectionDim > 0) {
        Eigen::VectorXd optProjIntrinsics;
        ProjectionOptGlobalToLocal(projOptModelId, allIntrinsics,
                                   &optProjIntrinsics);
        std::shared_ptr<okvis::ceres::EuclideanParamBlock>
            intrinsicParamBlockPtr(new okvis::ceres::EuclideanParamBlock(
                optProjIntrinsics, id, correctedStateTime, minProjectionDim));
        mapPtr_->addParameterBlock(intrinsicParamBlockPtr,
                                   ceres::Map::Parameterization::Trivial);
        cameraInfos.at(CameraSensorStates::Intrinsics).id = id;
      } else {
        cameraInfos.at(CameraSensorStates::Intrinsics).exists = false;
        cameraInfos.at(CameraSensorStates::Intrinsics).id = 0u;
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
          new okvis::ceres::CameraTimeParamBlock(camera_rig_.getTimeDelay(i),
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

    imuInfo.at(ImuSensorStates::TG).exists = true;
    imuInfo.at(ImuSensorStates::TS).exists = true;
    imuInfo.at(ImuSensorStates::TA).exists = true;
    // In MSCKF, use the same block for those parameters that are assumed
    // constant and updated in the filter
    if (statesMap_.size() > 1) {
      // use the same block...
      imuInfo.at(ImuSensorStates::TG).id =
          lastElementIterator->second.sensors.at(SensorStates::Imu)
              .at(i)
              .at(ImuSensorStates::TG)
              .id;
      imuInfo.at(ImuSensorStates::TS).id =
          lastElementIterator->second.sensors.at(SensorStates::Imu)
              .at(i)
              .at(ImuSensorStates::TS)
              .id;
      imuInfo.at(ImuSensorStates::TA).id =
          lastElementIterator->second.sensors.at(SensorStates::Imu)
              .at(i)
              .at(ImuSensorStates::TA)
              .id;
    } else {
      Eigen::Matrix<double, 9, 1> TG = vTgTsTa.head<9>();
      uint64_t id = IdProvider::instance().newId();
      std::shared_ptr<ceres::ShapeMatrixParamBlock> tgBlockPtr(
          new ceres::ShapeMatrixParamBlock(TG, id, correctedStateTime));
      mapPtr_->addParameterBlock(tgBlockPtr, ceres::Map::Trivial);
      imuInfo.at(ImuSensorStates::TG).id = id;

      const Eigen::Matrix<double, 9, 1> TS = vTgTsTa.segment<9>(9);
      id = IdProvider::instance().newId();
      std::shared_ptr<okvis::ceres::ShapeMatrixParamBlock> tsBlockPtr(
          new okvis::ceres::ShapeMatrixParamBlock(TS, id, correctedStateTime));
      mapPtr_->addParameterBlock(tsBlockPtr, ceres::Map::Trivial);
      imuInfo.at(ImuSensorStates::TS).id = id;

      Eigen::Matrix<double, 9, 1> TA = vTgTsTa.tail<9>();
      id = IdProvider::instance().newId();
      std::shared_ptr<okvis::ceres::ShapeMatrixParamBlock> taBlockPtr(
          new okvis::ceres::ShapeMatrixParamBlock(TA, id, correctedStateTime));
      mapPtr_->addParameterBlock(taBlockPtr, ceres::Map::Trivial);
      imuInfo.at(ImuSensorStates::TA).id = id;
    }

    statesMap_.rbegin()
        ->second.sensors.at(SensorStates::Imu)
        .push_back(imuInfo);
    states.sensors.at(SensorStates::Imu).push_back(imuInfo);
  }

  // depending on whether or not this is the very beginning, we will add priors
  // or relative terms to the last state:
  if (statesMap_.size() == 1) {
    const int camIdx = 0;
    initCovariance(camIdx);
  }
  // record the imu measurements between two consecutive states
  mStateID2Imu.push_back(imuMeasurements);

  addCovForClonedStates();
  return true;
}

// Applies the dropping/marginalization strategy according to Michael A.
// Shelley's MS thesis
bool TFVIO::applyMarginalizationStrategy(
    size_t /*numKeyframes*/, size_t /*numImuFrames*/,
    okvis::MapPointVector& /*removedLandmarks*/) {
  std::vector<uint64_t> removeFrames;
  std::map<uint64_t, States>::reverse_iterator rit = statesMap_.rbegin();
  while (rit != statesMap_.rend()) {
    if (rit->first < minValidStateID) {
      removeFrames.push_back(rit->second.id);
    }
    ++rit;
  }

  // remove features tracked no more
  for (PointMap::iterator pit = landmarksMap_.begin();
       pit != landmarksMap_.end();) {
    if (pit->second.residualizeCase == NotInState_NotTrackedNow) {
      ceres::Map::ResidualBlockCollection residuals =
          mapPtr_->residuals(pit->first);
      ++mTrackLengthAccumulator[residuals.size()];
      for (size_t r = 0; r < residuals.size(); ++r) {
        std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
            std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
                residuals[r].errorInterfacePtr);
        OKVIS_ASSERT_TRUE(Exception, reprojectionError,
                          "Wrong index of reprojection error");
        removeObservation(residuals[r].residualBlockId);
      }

      mapPtr_->removeParameterBlock(pit->first);
      pit = landmarksMap_.erase(pit);
    } else {
      ++pit;
    }
  }

  for (size_t k = 0; k < removeFrames.size(); ++k) {
    okvis::Time removedStateTime = removeState(removeFrames[k]);
    mStateID2Imu.pop_front(removedStateTime - half_window_);
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


bool TFVIO::featureJacobian(
    const MapPoint& mp,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>* Hi,
    Eigen::Matrix<double, Eigen::Dynamic, 1>* ri,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>* Ri) const {
  const int camIdx = 0;
  std::shared_ptr<okvis::cameras::CameraBase> tempCameraGeometry =
      camera_rig_.getCameraGeometry(camIdx);
  // head and tail observations for this feature point
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      obsInPixels;
  // id of head and tail frames observing this feature point
  std::vector<uint64_t> frameIds;
  std::vector<double> imagePointNoiseStds;  // std noise in pixels

  // each entry is undistorted coordinates in image plane at
  // z=1 in the specific camera frame, [\bar{x},\bar{y},1]
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      obsDirections;
  std::vector<okvis::kinematics::Transformation,
              Eigen::aligned_allocator<okvis::kinematics::Transformation>>
      T_WSs;
  RetrieveObsSeqType seqType =
      static_cast<RetrieveObsSeqType>(FLAGS_two_view_constraint_scheme);
  size_t numFeatures = gatherPoseObservForTriang(mp, tempCameraGeometry, &frameIds, &T_WSs,
                            &obsDirections, &obsInPixels, &imagePointNoiseStds,
                            seqType);
  if (numFeatures < 2) { // A two view constraint requires at least two obs
      return false;
  }

  // enlarge cov of the head obs to counteract the noise reduction 
  // due to correlation in head_tail scheme
  size_t trackLength = mp.observations.size();
  double headObsCovModifier[2] = {1.0, 1.0};
  headObsCovModifier[0] =
      seqType == HEAD_TAIL
          ? (static_cast<double>(trackLength - minTrackLength_ + 2u))
          : 1.0;

  std::vector<std::pair<int, int>> featurePairs =
      TwoViewPair::getFramePairs(numFeatures, TwoViewPair::FIXED_MIDDLE);
  const int numConstraints = featurePairs.size();
  int featureVariableDimen = cameraParamsMinimalDimen() +
                             kClonedStateMinimalDimen * (statesMap_.size());
  Hi->resize(numConstraints, featureVariableDimen);
  ri->resize(numConstraints, 1);

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> H_fi(numConstraints,
                                                             3 * numFeatures);
  H_fi.setZero();
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cov_fi(3 * numFeatures,
                                                               3 * numFeatures);
  cov_fi.setZero();

  for (int count = 0; count < numConstraints; ++count) {
    const std::pair<int, int>& feature_pair = featurePairs[count];
    std::vector<uint64_t> frameId2;
    std::vector<okvis::kinematics::Transformation,
                Eigen::aligned_allocator<okvis::kinematics::Transformation>>
        T_WS2;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        obsDirection2;
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
        obsInPixel2;
    std::vector<double> imagePointNoiseStd2;
    std::vector<int> index_vec{feature_pair.first, feature_pair.second};
    for (auto index : index_vec) {
      frameId2.emplace_back(frameIds[index]);
      T_WS2.emplace_back(T_WSs[index]);
      obsDirection2.emplace_back(obsDirections[index]);
      obsInPixel2.emplace_back(obsInPixels[index]);
      imagePointNoiseStd2.emplace_back(imagePointNoiseStds[2 * index]);
      imagePointNoiseStd2.emplace_back(imagePointNoiseStds[2 * index + 1]);
    }
    Eigen::Matrix<double, 1, Eigen::Dynamic> H_xjk(1, featureVariableDimen);
    std::vector<Eigen::Matrix<double, 1, 3>,
                Eigen::aligned_allocator<Eigen::Matrix<double, 1, 3>>>
        H_fjk;
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>
        cov_fjk;
    double rjk;
    measurementJacobianEpipolar(tempCameraGeometry, frameId2, T_WS2, obsDirection2,
                        obsInPixel2, imagePointNoiseStd2, camIdx, &H_xjk, &H_fjk,
                        &cov_fjk, &rjk);
    Hi->row(count) = H_xjk;
    (*ri)(count) = rjk;
    for (int j = 0; j < 2; ++j) {
      int index = index_vec[j];
      H_fi.block<1, 3>(count, index * 3) = H_fjk[j];
      cov_fi.block<3, 3>(index * 3, index * 3) =
          cov_fjk[j] * FLAGS_image_noise_cov_multiplier * headObsCovModifier[j];
    }
  }
  Ri->resize(numConstraints, numConstraints);
  *Ri = H_fi * cov_fi * H_fi.transpose();
  return true;
}

int TFVIO::computeStackedJacobianAndResidual(
    Eigen::MatrixXd* T_H, Eigen::Matrix<double, Eigen::Dynamic, 1>* r_q,
    Eigen::MatrixXd* R_q) const {
  // compute and stack Jacobians and Residuals for landmarks observed in current
  // frame
  const int camParamStartIndex = startIndexOfCameraParams();
  int featureVariableDimen = covariance_.rows() - camParamStartIndex;
  int dimH[2] = {0, featureVariableDimen};
  const Eigen::MatrixXd variableCov = covariance_.block(
      camParamStartIndex, camParamStartIndex, dimH[1], dimH[1]);

  // containers of Jacobians of measurements
  std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> vr;
  std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> vH;
  std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> vR;
  RetrieveObsSeqType seqType =
      static_cast<RetrieveObsSeqType>(FLAGS_two_view_constraint_scheme);
  for (auto it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
    ResidualizeCase rc = it->second.residualizeCase;
    const size_t nNumObs = it->second.observations.size();
    if (seqType == ENTIRE_TRACK) {
      if (rc != NotInState_NotTrackedNow || nNumObs < minTrackLength_) {
        continue;
      }
    } else {
      if (rc != NotToAdd_TrackedNow || nNumObs < minTrackLength_) {
        continue;
      }
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Hi;
    Eigen::Matrix<double, Eigen::Dynamic, 1> ri;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Ri;
    bool isValidJacobian = featureJacobian(it->second, &Hi, &ri, &Ri);
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
  Eigen::MatrixXd r(dimH[0], 1);
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(dimH[0], dimH[0]);
  FilterHelper::stackJacobianAndResidual(vH, vr, vR, &H, &r, &R);
  FilterHelper::shrinkResidual(H, r, R, T_H, r_q, R_q);
  return dimH[0];
}

uint64_t TFVIO::getMinValidStateID() const {
  uint64_t min_state_id = statesMap_.rbegin()->first;
  for (auto it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
    if (it->second.residualizeCase == NotInState_NotTrackedNow) continue;

    auto itObs = it->second.observations.begin();
    if (itObs->first.frameId <
        min_state_id) {  // this assume that it->second.observations is an
                         // ordered map
      min_state_id = itObs->first.frameId;
    }
  }
  return min_state_id;
}

void TFVIO::optimize(size_t /*numIter*/, size_t /*numThreads*/, bool verbose) {
  uint64_t currFrameId = currentFrameId();
  OKVIS_ASSERT_EQ(Exception, covariance_.rows() - startIndexOfClonedStates(),
                  (int)(kClonedStateMinimalDimen * statesMap_.size()),
                  "Inconsistent covDim and number of states");
  retrieveEstimatesOfConstants();

  // mark tracks of features that are not tracked in current frame
  int numTracked = 0;
  int featureVariableDimen = cameraParamsMinimalDimen() +
                             kClonedStateMinimalDimen * statesMap_.size();

  for (okvis::PointMap::iterator it = landmarksMap_.begin();
       it != landmarksMap_.end(); ++it) {
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
    minValidStateID = statesMap_.rbegin()->first;
    for (auto it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
      if (it->second.residualizeCase == NotInState_NotTrackedNow) continue;
      // this happens with a just inserted landmark without triangulation.
      if (it->second.observations.size() < 2) continue;

      auto itObs = it->second.observations.begin();
      if (itObs->first.frameId < minValidStateID) {
        // this assume that it->second.observations is an ordered map
        minValidStateID = itObs->first.frameId;
      }

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
