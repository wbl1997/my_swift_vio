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
#include <okvis/ceres/EuclideanParamBlock.hpp>

#include <msckf/ImuOdometry.h>
#include <msckf/PreconditionedEkfUpdater.h>
#include <msckf/triangulate.h>
#include <msckf/EpipolarJacobian.hpp>
#include <msckf/FilterHelper.hpp>
#include <msckf/RelativeMotionJacobian.hpp>
#include <msckf/triangulateFast.hpp>

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
DEFINE_double(pixel_noise_scale_factor, 3.0,
              "Enlarge the original noise std by this scale factor");
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

    if (useExternalInitialPose_) {
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
    cameraInfos.at(CameraSensorStates::Intrinsic).exists = true;
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
      cameraInfos.at(CameraSensorStates::Intrinsic).id =
          lastElementIterator->second.sensors.at(SensorStates::Camera)
              .at(i)
              .at(CameraSensorStates::Intrinsic)
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
        cameraInfos.at(CameraSensorStates::Intrinsic).id = id;
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

bool TFVIO::measurementJacobian(
    const std::shared_ptr<okvis::cameras::CameraBase> tempCameraGeometry,
    const std::vector<uint64_t>& frameIds,
    const std::vector<
        okvis::kinematics::Transformation,
        Eigen::aligned_allocator<okvis::kinematics::Transformation>>& T_WSs,
    const std::vector<Eigen::Vector3d,
                      Eigen::aligned_allocator<Eigen::Vector3d>>& obsDirections,
    const std::vector<Eigen::Vector2d,
                      Eigen::aligned_allocator<Eigen::Vector2d>>& obsInPixel2,
    const std::vector<double>& imagePointNoiseStd2, int camIdx,
    Eigen::Matrix<double, 1, Eigen::Dynamic>* H_xjk,
    std::vector<Eigen::Matrix<double, 1, 3>,
                Eigen::aligned_allocator<Eigen::Matrix<double, 1, 3>>>* H_fjk,
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>*
        cov_fjk,
    double* residual) const {
  // camera related Jacobians and covariance
  std::vector<
      Eigen::Matrix<double, 3, Eigen::Dynamic>,
      Eigen::aligned_allocator<Eigen::Matrix<double, 3, Eigen::Dynamic>>>
      dfj_dXcam(2);
  std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>
      cov_fj(2);
  int projOptModelId = camera_rig_.getProjectionOptMode(camIdx);
  for (int j = 0; j < 2; ++j) {
    double pixelNoiseStd = imagePointNoiseStd2[j * 2];
    obsDirectionJacobian(obsDirections[j], tempCameraGeometry, projOptModelId,
                         pixelNoiseStd, &dfj_dXcam[j], &cov_fj[j]);
  }

  // compute the head and tail pose, velocity, Jacobians, and covariance
  uint32_t imageHeight = camera_rig_.getCameraGeometry(camIdx)->imageHeight();
  int extrinsicModelId = camera_rig_.getExtrinsicOptMode(camIdx);
  std::vector<okvis::kinematics::Transformation,
              Eigen::aligned_allocator<okvis::kinematics::Transformation>>
      T_WBtij, lP_T_WBtij;  // lp is short for linearization point
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      v_WBtij, lP_v_WBtij;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      omega_WBtij;
  double dtij_dtr[2];
  double featureDelay[2];

  for (int j = 0; j < 2; ++j) {
    ImuMeasurement interpolatedInertialData;
    uint64_t poseId = frameIds[j];
    kinematics::Transformation T_WBj = T_WSs[j];

    // TODO(jhuai): do we ue the latest estimate for bg ba or the saved ones for
    // each state?
    SpeedAndBiases sbj;
    getSpeedAndBias(poseId, 0, sbj);

    Time stateEpoch = statesMap_.at(poseId).timestamp;
    auto imuMeas = mStateID2Imu.findWindow(stateEpoch, half_window_);
    OKVIS_ASSERT_GT(Exception, imuMeas.size(), 0,
                    "the IMU measurement does not exist");

    double kpN = obsInPixel2[j][1] / imageHeight - 0.5;  // k per N
    dtij_dtr[j] = kpN;
    Duration featureTime = Duration(tdLatestEstimate + trLatestEstimate * kpN) -
                           statesMap_.at(poseId).tdAtCreation;
    featureDelay[j] = featureTime.toSec();
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
        IMUOdometry::propagation(imuMeas, imuParametersVec_.at(0), T_WB,
                                 tempV_WS, iem, stateEpoch,
                                 stateEpoch + featureTime);
      } else {
        IMUOdometry::propagationBackward(imuMeas, imuParametersVec_.at(0), T_WB,
                                         tempV_WS, iem, stateEpoch,
                                         stateEpoch + featureTime);
      }
      sb.head<3>() = tempV_WS;
    }
    IMUOdometry::interpolateInertialData(imuMeas, iem, stateEpoch + featureTime,
                                         interpolatedInertialData);
    T_WBtij.emplace_back(T_WB);
    v_WBtij.emplace_back(sb.head<3>());
    omega_WBtij.emplace_back(interpolatedInertialData.measurement.gyroscopes);

    kinematics::Transformation lP_T_WB = T_WB;
    Eigen::Vector3d lP_v = sb.head<3>();
    if (FLAGS_use_first_estimate) {
      Eigen::Matrix<double, 6, 1> posVelFirstEstimate =
          statesMap_.at(poseId).linearizationPoint;
      lP_T_WB =
          kinematics::Transformation(posVelFirstEstimate.head<3>(), T_WBj.q());
      lP_v = posVelFirstEstimate.tail<3>();
      if (featureTime >= Duration()) {
        IMUOdometry::propagation(imuMeas, imuParametersVec_.at(0), lP_T_WB,
                                 lP_v, iem, stateEpoch,
                                 stateEpoch + featureTime);
      } else {
        IMUOdometry::propagationBackward(imuMeas, imuParametersVec_.at(0),
                                         lP_T_WB, lP_v, iem, stateEpoch,
                                         stateEpoch + featureTime);
      }
    }

    lP_T_WBtij.emplace_back(lP_T_WB);
    lP_v_WBtij.emplace_back(lP_v);
  }

  // compute residual
  okvis::kinematics::Transformation T_Ctij_Ctik =
      (T_WBtij[0] * T_SC0_).inverse() * (T_WBtij[1] * T_SC0_);
  okvis::kinematics::Transformation lP_T_Ctij_Ctik =
      (lP_T_WBtij[0] * T_SC0_).inverse() * (lP_T_WBtij[1] * T_SC0_);
  EpipolarJacobian epj(T_Ctij_Ctik.q(), T_Ctij_Ctik.r(), obsDirections[0],
                       obsDirections[1]);
  *residual = -epj.evaluate();  // observation is 0

  // compute Jacobians for camera parameters
  EpipolarJacobian epj_lp(lP_T_Ctij_Ctik.q(), lP_T_Ctij_Ctik.r(),
                          obsDirections[0], obsDirections[1]);
  Eigen::Matrix<double, 1, 3> de_dfj[2];
  epj_lp.de_dfj(&de_dfj[0]);
  epj_lp.de_dfk(&de_dfj[1]);
  Eigen::Matrix<double, 1, 3> de_dtheta_Ctij_Ctik, de_dt_Ctij_Ctik;
  epj_lp.de_dtheta_CjCk(&de_dtheta_Ctij_Ctik);
  epj_lp.de_dt_CjCk(&de_dt_Ctij_Ctik);
  RelativeMotionJacobian rmj_lp(T_SC0_, lP_T_WBtij[0], lP_T_WBtij[1]);
  Eigen::Matrix<double, 3, 3> dtheta_dtheta_BC;
  Eigen::Matrix<double, 3, 3> dp_dtheta_BC;
  Eigen::Matrix<double, 3, 3> dp_dt_BC;
  Eigen::Matrix<double, 3, 3> dp_dt_CB;

  Eigen::Matrix<double, 1, Eigen::Dynamic> de_dExtrinsic;
  switch (extrinsicModelId) {
    case Extrinsic_p_SC_q_SC::kModelId:
      rmj_lp.dtheta_dtheta_BC(&dtheta_dtheta_BC);
      rmj_lp.dp_dtheta_BC(&dp_dtheta_BC);
      rmj_lp.dp_dt_BC(&dp_dt_BC);
      de_dExtrinsic.resize(1, 6);
      de_dExtrinsic.head<3>() = de_dt_Ctij_Ctik * dp_dt_BC;
      de_dExtrinsic.tail<3>() = de_dt_Ctij_Ctik * dp_dtheta_BC +
                                de_dtheta_Ctij_Ctik * dtheta_dtheta_BC;
      break;
    case Extrinsic_p_CS::kModelId:
      rmj_lp.dp_dt_CB(&dp_dt_CB);
      de_dExtrinsic = de_dt_Ctij_Ctik * dp_dt_CB;
      break;
    case ExtrinsicFixed::kModelId:
    default:
      break;
  }
  Eigen::Matrix<double, 1, Eigen::Dynamic> de_dxcam =
      de_dfj[0] * dfj_dXcam[0] + de_dfj[1] * dfj_dXcam[1];

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
  const int minExtrinsicDim = camera_rig_.getMinimalExtrinsicDimen(camIdx);
  const int minProjDim = camera_rig_.getMinimalProjectionDimen(camIdx);
  const int minDistortDim = camera_rig_.getDistortionDimen(camIdx);
  if (minExtrinsicDim > 0) {
    H_xjk->topLeftCorner(1, minExtrinsicDim) = de_dExtrinsic;
  }
  H_xjk->block(0, minExtrinsicDim, 1, minProjDim + minDistortDim) = de_dxcam;
  int startIndex = minExtrinsicDim + minProjDim + minDistortDim;
  (*H_xjk)(startIndex) = de_dtd;
  startIndex += 1;
  (*H_xjk)(startIndex) = de_dtr;

  const int minCamParamDim = cameraParamsMinimalDimen();
  double scale_factor2 =
      FLAGS_pixel_noise_scale_factor * FLAGS_pixel_noise_scale_factor;
  for (int j = 0; j < 2; ++j) {
    uint64_t poseId = frameIds[j];
    std::map<uint64_t, int>::const_iterator poseid_iter =
        mStateID2CovID_.find(poseId);
    int covid = poseid_iter->second;
    int startIndex = minCamParamDim + 9 * covid;
    H_xjk->block<1, 3>(0, startIndex) = de_dp_GBtj[j];
    H_xjk->block<1, 3>(0, startIndex + 3) = de_dtheta_GBtj[j];
    H_xjk->block<1, 3>(0, startIndex + 6) = de_dv_GBtj[j];
    H_fjk->emplace_back(de_dfj[j]);

    // TODO(jhuai): account for the IMU noise
    cov_fjk->emplace_back(cov_fj[j] * scale_factor2);
  }

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
  gatherPoseObservForTriang(mp, tempCameraGeometry, &frameIds, &T_WSs,
                            &obsDirections, &obsInPixels, &imagePointNoiseStds,
                            ENTIRE_TRACK);

  const int numFeatures = frameIds.size();
  std::vector<std::pair<int, int>> featurePairs = getFramePairs(numFeatures);
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
    measurementJacobian(tempCameraGeometry, frameId2, T_WS2, obsDirection2,
                        obsInPixel2, imagePointNoiseStd2, camIdx, &H_xjk, &H_fjk,
                        &cov_fjk, &rjk);
    Hi->row(count) = H_xjk;
    (*ri)(count) = rjk;
    for (int j = 0; j < 2; ++j) {
      int index = index_vec[j];
      H_fi.block<1, 3>(count, index * 3) = H_fjk[j];
      cov_fi.block<3, 3>(index * 3, index * 3) = cov_fjk[j];
    }
  }
  Ri->resize(numConstraints, numConstraints);
  *Ri = H_fi * cov_fi * H_fi.transpose();
  return true;
}

std::vector<std::pair<int, int>> getFramePairs(int numFeatures) {
  std::vector<std::pair<int, int>> framePairs;
  framePairs.reserve(2 * numFeatures - 3);
  // scheme 1 works
//  for (int j = 0; j < numFeatures - 1; ++j) {
//    framePairs.emplace_back(0, j + 1);
//  }
  // scheme 2 works
//  for (int j = 0; j < numFeatures - 1; ++j) {
//    framePairs.emplace_back(numFeatures - 1, j);
//  }
  // scheme 3 works best
  int halfFeatures = numFeatures / 2;
  for (int j = 0; j < halfFeatures; ++j) {
      framePairs.emplace_back(j, halfFeatures + j);
  }
  for (int j = 0; j < halfFeatures - 1; ++j) {
      framePairs.emplace_back(j, halfFeatures + j + 1);
  }
  // Surprisingly, more constraints degrades accuracy.
//  for (int j = 0; j < halfFeatures - 1; ++j) {
//      if (j != halfFeatures - j - 1) {
//          framePairs.emplace_back(j, halfFeatures + j - 1);
//      }
//  }
  return framePairs;
}

void obsDirectionJacobian(
    const Eigen::Vector3d& obsDirection,
    const std::shared_ptr<okvis::cameras::CameraBase> cameraGeometry,
    int projOptModelId, double pixelNoiseStd,
    Eigen::Matrix<double, 3, Eigen::Dynamic>* dfj_dXcam,
    Eigen::Matrix3d* cov_fj) {
  const Eigen::Vector3d& fj = obsDirection;
  Eigen::Vector2d imagePoint;
  Eigen::Matrix<double, 2, 3> pointJacobian;
  Eigen::Matrix2Xd intrinsicsJacobian;
  cameraGeometry->project(fj, &imagePoint, &pointJacobian, &intrinsicsJacobian);
  ProjectionOptKneadIntrinsicJacobian(projOptModelId, &intrinsicsJacobian);
  Eigen::Matrix2d dz_df12 = pointJacobian.topLeftCorner<2, 2>();
  Eigen::Matrix2d df12_dz = dz_df12.inverse();
  int cols = intrinsicsJacobian.cols();
  dfj_dXcam->resize(3, cols);
  dfj_dXcam->topLeftCorner(2, cols) = -df12_dz * intrinsicsJacobian;
  dfj_dXcam->row(2).setZero();
  cov_fj->setZero();
  cov_fj->topLeftCorner<2, 2>() = df12_dz * Eigen::Matrix2d::Identity() *
                                  df12_dz.transpose() * pixelNoiseStd *
                                  pixelNoiseStd;
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

  for (auto it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
    ResidualizeCase rc = it->second.residualizeCase;
    const size_t nNumObs = it->second.observations.size();
    if (rc != NotInState_NotTrackedNow ||
        nNumObs < minTrackLength_) {
      continue;
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Hi;
    Eigen::Matrix<double, Eigen::Dynamic, 1> ri;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Ri;
    bool isValidJacobian =
        featureJacobian(it->second, &Hi, &ri, &Ri);
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
