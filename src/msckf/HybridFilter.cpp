
#include <glog/logging.h>
#include <msckf/HybridFilter.hpp>
#include <okvis/IdProvider.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/RelativePoseError.hpp>
#include <okvis/ceres/SpeedAndBiasError.hpp>

#include <msckf/ImuOdometry.h>
#include <msckf/triangulate.h>
#include <msckf/triangulateFast.hpp>

#include <okvis/ceres/CameraDistortionParamBlock.hpp>
#include <okvis/ceres/CameraIntrinsicParamBlock.hpp>
#include <okvis/ceres/CameraTimeParamBlock.hpp>
#include <okvis/ceres/ShapeMatrixParamBlock.hpp>
#include <okvis/timing/Timer.hpp>

// the following 3 headers are only for testing
#include "vio/ImuErrorModel.h"
#include "vio/Sample.h"
#include "vio/eigen_utils.h"

DEFINE_bool(use_RK4, false,
            "use 4th order runge-kutta or the trapezoidal "
            "rule for integrating IMU data and computing"
            " Jacobians");

DECLARE_bool(use_mahalanobis);
DECLARE_bool(use_first_estimate);
/// \brief okvis Main namespace of this package.
namespace okvis {
const double maxProjTolerance =
    10;  // maximum tolerable discrepancy between predicted and measured point
         // coordinates in image in pixel units

// Constructor if a ceres map is already available.
HybridFilter::HybridFilter(std::shared_ptr<okvis::ceres::Map> mapPtr,
                           const double readoutTime)
    : mapPtr_(mapPtr),
      referencePoseId_(0),
      cauchyLossFunctionPtr_(new ::ceres::CauchyLoss(1)),
      huberLossFunctionPtr_(new ::ceres::HuberLoss(1)),
      marginalizationResidualId_(0),
      imageReadoutTime(readoutTime),
      minValidStateID(0),
      triangulateTimer("3.1.1.1 triangulateAMapPoint", true),
      computeHTimer("3.1.1 computeHoi", true),
      computeKalmanGainTimer("3.1.2 computeKalmanGain", true),
      updateStatesTimer("3.1.3 updateStates", true),
      updateCovarianceTimer("3.1.4 updateCovariance", true),
      updateLandmarksTimer("3.1.5 updateLandmarks", true),
      mM(0),
      mbUseExternalInitialPose(false),
      mTrackLengthAccumulator(100, 0) {}

// The default constructor.
HybridFilter::HybridFilter(const double readoutTime)
    : mapPtr_(new okvis::ceres::Map()),
      referencePoseId_(0),
      cauchyLossFunctionPtr_(new ::ceres::CauchyLoss(1)),
      huberLossFunctionPtr_(new ::ceres::HuberLoss(1)),
      marginalizationResidualId_(0),
      imageReadoutTime(readoutTime),
      minValidStateID(0),
      triangulateTimer("3.1.1.1 triangulateAMapPoint", true),
      computeHTimer("3.1.1 computeHoi", true),
      computeKalmanGainTimer("3.1.2 computeKalmanGain", true),
      updateStatesTimer("3.1.3 updateStates", true),
      updateCovarianceTimer("3.1.4 updateCovariance", true),
      updateLandmarksTimer("3.1.5 updateLandmarks", true),
      mM(0),
      mbUseExternalInitialPose(false),
      mTrackLengthAccumulator(100, 0) {}

HybridFilter::~HybridFilter() {}

// Add a camera to the configuration. Sensors can only be added and never
// removed.
int HybridFilter::addCamera(
    const ExtrinsicsEstimationParameters& extrinsicsEstimationParameters) {
  extrinsicsEstimationParametersVec_.push_back(extrinsicsEstimationParameters);
  return extrinsicsEstimationParametersVec_.size() - 1;
}

// Add an IMU to the configuration.
int HybridFilter::addImu(const ImuParameters& imuParameters) {
  if (imuParametersVec_.size() > 1) {
    LOG(ERROR) << "only one IMU currently supported";
    return -1;
  }
  imuParametersVec_.push_back(imuParameters);
  return imuParametersVec_.size() - 1;
}

// Remove all cameras from the configuration
void HybridFilter::clearCameras() {
  extrinsicsEstimationParametersVec_.clear();
}

// Remove all IMUs from the configuration.
void HybridFilter::clearImus() { imuParametersVec_.clear(); }

bool HybridFilter::addStates(okvis::MultiFramePtr multiFrame,
                             const okvis::ImuMeasurementDeque& imuMeasurements,
                             bool asKeyframe) {
  okvis::kinematics::Transformation T_WS;
  okvis::SpeedAndBiases speedAndBias;
  okvis::Duration tdEstimate;
  okvis::Time correctedStateTime;  // time of current multiFrame corrected with
                                   // current td estimate

  Eigen::Matrix<double, 27, 1> vTgTsTa;
  if (statesMap_.empty()) {
    correctedStateTime = multiFrame->timestamp() + tdEstimate;
    // in case this is the first frame ever, let's initialize the pose:
    if (mbUseExternalInitialPose)
      T_WS = okvis::kinematics::Transformation(pvstd_.p_WS, pvstd_.q_WS);
    else {
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

    vTgTsTa.setZero();
    for (int jack = 0; jack < 3; ++jack) {
      vTgTsTa[jack * 4] = 1;
      vTgTsTa[jack * 4 + 18] = 1;
    }

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
        // the msckf derivation in Michael Andrew Shelley
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
                      covDim_ - ceres::ode::OdoErrorStateDim) =
        F_tot * covariance_.block(0, ceres::ode::OdoErrorStateDim,
                                  ceres::ode::OdoErrorStateDim,
                                  covDim_ - ceres::ode::OdoErrorStateDim);
    covariance_.block(ceres::ode::OdoErrorStateDim, 0,
                      covDim_ - ceres::ode::OdoErrorStateDim,
                      ceres::ode::OdoErrorStateDim) =
        covariance_.block(ceres::ode::OdoErrorStateDim, 0,
                          covDim_ - ceres::ode::OdoErrorStateDim,
                          ceres::ode::OdoErrorStateDim) *
        F_tot.transpose();

    if (numUsedImuMeasurements < 2) {
      std::cout << "numUsedImuMeasurements=" << numUsedImuMeasurements
                << " correctedStateTime " << correctedStateTime
                << " lastFrameTimestamp " << startTime << " tdEstimate "
                << tdEstimate << std::endl;
      OKVIS_ASSERT_TRUE(Exception, numUsedImuMeasurements > 1,
                        "propagation failed");
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
  std::cout << "Added STATE OF ID " << states.id << std::endl;
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
      // initialize the evolving camera geometry
      camera_rig_.addCamera(multiFrame->T_SC(i),
                            multiFrame->GetCameraSystem().cameraGeometry(i),
                            imageReadoutTime, 0, std::vector<bool>());
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
      std::shared_ptr<okvis::ceres::CameraIntrinsicParamBlock>
          intrinsicParamBlockPtr(new okvis::ceres::CameraIntrinsicParamBlock(
              allIntrinsics.head<4>(), id, correctedStateTime));
      mapPtr_->addParameterBlock(intrinsicParamBlockPtr,
                                 ceres::Map::Parameterization::Trivial);
      cameraInfos.at(CameraSensorStates::Intrinsic).id = id;

      id = IdProvider::instance().newId();
      std::shared_ptr<okvis::ceres::CameraDistortionParamBlock>
          distortionParamBlockPtr(new okvis::ceres::CameraDistortionParamBlock(
              allIntrinsics.tail<okvis::ceres::nDistortionDim>(), id,
              correctedStateTime));
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
      Eigen::Matrix<double, 9, 1> TG;
      TG << 1, 0, 0, 0, 1, 0, 0, 0, 1;
      uint64_t id = IdProvider::instance().newId();
      std::shared_ptr<ceres::ShapeMatrixParamBlock> tgBlockPtr(
          new ceres::ShapeMatrixParamBlock(TG, id, correctedStateTime));
      mapPtr_->addParameterBlock(tgBlockPtr, ceres::Map::Trivial);
      imuInfo.at(ImuSensorStates::TG).id = id;

      const Eigen::Matrix<double, 9, 1> TS =
          Eigen::Matrix<double, 9, 1>::Zero();
      id = IdProvider::instance().newId();
      std::shared_ptr<okvis::ceres::ShapeMatrixParamBlock> tsBlockPtr(
          new okvis::ceres::ShapeMatrixParamBlock(TS, id, correctedStateTime));
      mapPtr_->addParameterBlock(tsBlockPtr, ceres::Map::Trivial);
      imuInfo.at(ImuSensorStates::TS).id = id;

      Eigen::Matrix<double, 9, 1> TA;
      TA << 1, 0, 0, 0, 1, 0, 0, 0, 1;
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

  // depending on whether or not this is the very beginning, we will construct
  // covariance:
  if (statesMap_.size() == 1) {
    // initialize the covariance
    covDim_ = 15 + 27 + 3 + 4 + okvis::ceres::nDistortionDim +
              2;  // one camera assumption
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
    covariance_ = Eigen::MatrixXd::Zero(covDim_, covDim_);
    covariance_.topLeftCorner<6, 6>() = covPQ;
    covariance_.block<9, 9>(6, 6) = covSB;
    covariance_.block<27, 27>(15, 15) = covTGTSTA;

    // camera sensor states
    Eigen::Matrix3d covPBinC = Eigen::Matrix3d::Zero();
    Eigen::Matrix<double, 4, 4> covProjIntrinsics;
    Eigen::Matrix<double, ceres::nDistortionDim, ceres::nDistortionDim>
        covDistortion;
    Eigen::Matrix2d covTDTR = Eigen::Matrix2d::Identity();

    for (size_t i = 0; i < extrinsicsEstimationParametersVec_.size(); ++i) {
      double translationStdev =
          extrinsicsEstimationParametersVec_.at(i).sigma_absolute_translation;
      double translationVariance = translationStdev * translationStdev;

      OKVIS_ASSERT_TRUE(Exception,
                        translationVariance > 1.0e-16 &&
                            extrinsicsEstimationParametersVec_.at(i)
                                    .sigma_absolute_orientation < 1e-16,
                        "sigma absolute translation should be positive and "
                        "sigma absolute rotation should be 0");

      covPBinC = Eigen::Matrix3d::Identity() *
                 translationVariance;  // note in covariance PBinC is
                                       // different from the state PCinB
      covProjIntrinsics = Eigen::Matrix<double, 4, 4>::Identity();
      covProjIntrinsics.topLeftCorner<2, 2>() *= std::pow(
          extrinsicsEstimationParametersVec_.at(i).sigma_focal_length, 2);
      covProjIntrinsics.bottomRightCorner<2, 2>() *= std::pow(
          extrinsicsEstimationParametersVec_.at(i).sigma_principal_point, 2);

      covDistortion = Eigen::Matrix<double, ceres::nDistortionDim,
                                    ceres::nDistortionDim>::Identity();
      for (size_t jack = 0; jack < ceres::nDistortionDim; ++jack)
        covDistortion(jack, jack) *= std::pow(
            extrinsicsEstimationParametersVec_.at(i).sigma_distortion[jack], 2);

      covTDTR = Eigen::Matrix2d::Identity();
      covTDTR(0, 0) *=
          std::pow(extrinsicsEstimationParametersVec_.at(i).sigma_td, 2);
      covTDTR(1, 1) *=
          std::pow(extrinsicsEstimationParametersVec_.at(i).sigma_tr, 2);
    }
    covariance_.block<3, 3>(42, 42) = covPBinC;
    covariance_.block<4, 4>(45, 45) = covProjIntrinsics;
    covariance_.block<ceres::nDistortionDim, ceres::nDistortionDim>(49, 49) =
        covDistortion;
    covariance_.block<2, 2>(49 + ceres::nDistortionDim,
                            49 + ceres::nDistortionDim) = covTDTR;
  }
  // record the imu measurements between two consecutive states
  mStateID2Imu[states.id] = imuMeasurements;

  // augment states in the propagated covariance matrix
  size_t covDimAugmented = covDim_ + 9;  //$\delta p,\delta \alpha,\delta v$
  ++mM;
  Eigen::MatrixXd covarianceAugmented(covDimAugmented, covDimAugmented);

  const size_t numPointStates = 3 * mInCovLmIds.size();
  const size_t numOldNavImuCamPoseStates = covDim_ - numPointStates;

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

  covDim_ = covDimAugmented;
  covariance_ = covarianceAugmented;
  return true;
}

// Add a landmark.
bool HybridFilter::addLandmark(uint64_t landmarkId,
                               const Eigen::Vector4d& landmark) {
  std::shared_ptr<okvis::ceres::HomogeneousPointParameterBlock>
      pointParameterBlock(new okvis::ceres::HomogeneousPointParameterBlock(
          landmark, landmarkId));
  if (!mapPtr_->addParameterBlock(pointParameterBlock,
                                  okvis::ceres::Map::HomogeneousPoint)) {
    return false;
  }

  // remember
  double dist = std::numeric_limits<double>::max();
  if (fabs(landmark[3]) > 1.0e-8) {
    dist = (landmark / landmark[3]).head<3>().norm();  // euclidean distance
  }
  landmarksMap_.insert(std::pair<uint64_t, MapPoint>(
      landmarkId, MapPoint(landmarkId, landmark, 0.0, dist)));
  OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId),
                        "bug: inconsistend landmarkdMap_ with mapPtr_.");
  return true;
}

// Remove an observation from a landmark.
bool HybridFilter::removeObservation(::ceres::ResidualBlockId residualBlockId) {
  const ceres::Map::ParameterBlockCollection parameters =
      mapPtr_->parameters(residualBlockId);
  const uint64_t landmarkId = parameters.at(1).first;
  // remove in landmarksMap
  MapPoint& mapPoint = landmarksMap_.at(landmarkId);
  for (std::map<okvis::KeypointIdentifier, uint64_t>::iterator it =
           mapPoint.observations.begin();
       it != mapPoint.observations.end();) {
    if (it->second == uint64_t(residualBlockId)) {
      it = mapPoint.observations.erase(it);
      break;  // added by Huai
    } else {
      it++;
    }
  }
  // remove residual block
  mapPtr_->removeResidualBlock(residualBlockId);
  return true;
}

// Remove an observation from a landmark, if available.
bool HybridFilter::removeObservation(uint64_t landmarkId, uint64_t poseId,
                                     size_t camIdx, size_t keypointIdx) {
  if (landmarksMap_.find(landmarkId) == landmarksMap_.end()) {
    for (PointMap::iterator it = landmarksMap_.begin();
         it != landmarksMap_.end(); ++it) {
      LOG(INFO) << it->first
                << ", no. obs = " << it->second.observations.size();
    }
    LOG(INFO) << landmarksMap_.at(landmarkId).id;
  }
  OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId),
                        "landmark not added");

  okvis::KeypointIdentifier kid(poseId, camIdx, keypointIdx);
  MapPoint& mapPoint = landmarksMap_.at(landmarkId);
  std::map<okvis::KeypointIdentifier, uint64_t>::iterator it =
      mapPoint.observations.find(kid);
  if (it == landmarksMap_.at(landmarkId).observations.end()) {
    return false;  // observation not present
  }

  // remove residual block
  mapPtr_->removeResidualBlock(
      reinterpret_cast<::ceres::ResidualBlockId>(it->second));

  // remove also in local map
  mapPoint.observations.erase(it);

  return true;
}

// Applies the dropping/marginalization strategy, i.e., state management,
// according to Li and Mourikis RSS 12 optimization based thesis
bool HybridFilter::applyMarginalizationStrategy() {
  /// remove features tracked no more, the feature can be in state or not
  size_t tempCounter = 0;

  Eigen::Matrix<double, 3, Eigen::Dynamic> reparamJacobian(
      3,
      covDim_);  // Jacobians of feature reparameterization due to anchor
                 // change
  std::vector<
      Eigen::Matrix<double, 3, Eigen::Dynamic>,
      Eigen::aligned_allocator<Eigen::Matrix<double, 3, Eigen::Dynamic>>>
      vJacobian;  // container of these reparameterizing Jacobians
  std::vector<size_t> vCovPtId;  // id in covariance of point features to be
                                 // reparameterized, 0 based
  std::vector<uint64_t>
      toRemoveLmIds;  // id of landmarks to be removed that are in state
  const size_t numNavImuCamStates =
      15 + 27 + 9 +
      cameras::RadialTangentialDistortion::NumDistortionIntrinsics;
  // number of navigation, imu, and camera states in the covariance
  const size_t numNavImuCamPoseStates =
      numNavImuCamStates + 9 * statesMap_.size();
  // number of navigation, imu, camera, and pose copies states in the
  // covariance

  std::cout << "checking removed map points" << std::endl;
  for (PointMap::iterator pit = landmarksMap_.begin();
       pit != landmarksMap_.end();) {
    ResidualizeCase residualizeCase =
        mLandmarkID2Residualize[tempCounter].second;
    if (residualizeCase == NotInState_NotTrackedNow ||
        residualizeCase == InState_NotTrackedNow) {
      OKVIS_ASSERT_EQ(Exception, mLandmarkID2Residualize[tempCounter].first,
                      pit->second.id,
                      "mLandmarkID2Residualize has inconsistent landmark ids "
                      "with landmarks map");
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
      ++tempCounter;

    } else {
      /// change anchor pose for features whose anchor is not in states
      /// anymore
      if (residualizeCase == InState_TrackedNow) {
        if (pit->second.anchorStateId < minValidStateID) {
          uint64_t currFrameId = currentFrameId();

          okvis::kinematics::Transformation
              T_GBa;  // transform from the body frame at the anchor frame
                      // epoch to the global frame
          get_T_WS(pit->second.anchorStateId, T_GBa);
          okvis::kinematics::Transformation T_GBc;
          get_T_WS(currFrameId, T_GBc);
          okvis::kinematics::Transformation T_SC;
          const int camIdx = 0;
          getCameraSensorStates(currFrameId, camIdx, T_SC);

          okvis::kinematics::Transformation T_GA =
              T_GBa * T_SC;  // anchor camera frame to global frame
          okvis::kinematics::Transformation T_GC = T_GBc * T_SC;

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
      ++tempCounter;
      ++pit;
    }
  }
  OKVIS_ASSERT_EQ(Exception, tempCounter, mLandmarkID2Residualize.size(),
                  "Inconsistent index in pruning landmarksMap");

  // actual covariance update for reparameterized features
  tempCounter = 0;
  Eigen::MatrixXd featureJacobian = Eigen::MatrixXd::Identity(
      covDim_,
      covDim_);  // Jacobian of all the new states w.r.t the old states
  for (auto it = vJacobian.begin(); it != vJacobian.end();
       ++it, ++tempCounter) {
    featureJacobian.block(numNavImuCamPoseStates + vCovPtId[tempCounter] * 3, 0,
                          3, covDim_) = vJacobian[tempCounter];
  }
  if (vJacobian.size()) {
    covariance_ =
        (featureJacobian * covariance_).eval() * featureJacobian.transpose();
  }

  // actual covariance decimation for features in state and not tracked now
#if 0
    for(auto it= toRemoveLmIds.begin(), itEnd= toRemoveLmIds.end(); it!=itEnd; ++it){
        std::deque<uint64_t>::iterator idPos = std::find(mInCovLmIds.begin(), mInCovLmIds.end(), *it);
        OKVIS_ASSERT_TRUE(Exception, idPos != mInCovLmIds.end(), "The tracked landmark in state is not in mInCovLmIds ");

        // remove SLAM feature's dimension from the covariance matrix
        int startIndex = numNavImuCamPoseStates + 3*(idPos - mInCovLmIds.begin());
        int finishIndex = startIndex + 3;
        Eigen::MatrixXd slimCovariance(covDim_ - 3, covDim_ - 3);
        slimCovariance << covariance_.topLeftCorner(startIndex, startIndex),
                covariance_.block(0, finishIndex, startIndex, covDim_ - finishIndex),
                covariance_.block(finishIndex, 0, covDim_ - finishIndex, startIndex),
                covariance_.block(finishIndex, finishIndex, covDim_ - finishIndex, covDim_ - finishIndex);

        covariance_ = slimCovariance;
        covDim_ -= 3;
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
  if (startKeptRow != covDim_) {
    vRowStartInterval.push_back(
        std::make_pair(startKeptRow, covDim_ - startKeptRow));
  }
  covariance_ =
      vio::extractBlocks(covariance_, vRowStartInterval, vRowStartInterval);

  for (auto it = toRemoveLmIds.begin(), itEnd = toRemoveLmIds.end();
       it != itEnd; ++it) {
    std::deque<uint64_t>::iterator idPos =
        std::find(mInCovLmIds.begin(), mInCovLmIds.end(), *it);
    mInCovLmIds.erase(idPos);
  }
  covDim_ -= 3 * toRemoveLmIds.size();
#endif

  /// remove old frames
  // these will keep track of what we want to marginalize out.
  std::vector<uint64_t> paremeterBlocksToBeMarginalized;
  std::vector<uint64_t> removeFrames;
  std::cout << "selecting which states to remove " << std::endl;
  std::cout << "min valid state Id " << minValidStateID
            << " oldest and latest stateid " << statesMap_.begin()->first << " "
            << statesMap_.rbegin()->first << std::endl;
  // std::map<uint64_t, States>::reverse_iterator
  for (auto rit = statesMap_.rbegin(); rit != statesMap_.rend();) {
    if (rit->first < minValidStateID) {
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

      mStateID2Imu.erase(rit->first);
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

  int startIndex =
      42 + 9 + cameras::RadialTangentialDistortion::NumDistortionIntrinsics;
  int finishIndex = startIndex + numRemovedStates * 9;

  Eigen::MatrixXd slimCovariance(covDim_ - numRemovedStates * 9,
                                 covDim_ - numRemovedStates * 9);
  slimCovariance << covariance_.topLeftCorner(startIndex, startIndex),
      covariance_.block(0, finishIndex, startIndex, covDim_ - finishIndex),
      covariance_.block(finishIndex, 0, covDim_ - finishIndex, startIndex),
      covariance_.block(finishIndex, finishIndex, covDim_ - finishIndex,
                        covDim_ - finishIndex);

  covariance_ = slimCovariance;
  covDim_ -= numRemovedStates * 9;
  mM -= numRemovedStates;
  return true;
}

// Prints state information to buffer.
void HybridFilter::printStates(uint64_t poseId, std::ostream& buffer) const {
  buffer << "GLOBAL: ";
  for (size_t i = 0; i < statesMap_.at(poseId).global.size(); ++i) {
    if (statesMap_.at(poseId).global.at(i).exists) {
      uint64_t id = statesMap_.at(poseId).global.at(i).id;
      if (mapPtr_->parameterBlockPtr(id)->fixed()) buffer << "(";
      buffer << "id=" << id << ":";
      buffer << mapPtr_->parameterBlockPtr(id)->typeInfo();
      if (mapPtr_->parameterBlockPtr(id)->fixed()) buffer << ")";
      buffer << ", ";
    }
  }
  buffer << "SENSOR: ";
  for (size_t i = 0; i < statesMap_.at(poseId).sensors.size(); ++i) {
    for (size_t j = 0; j < statesMap_.at(poseId).sensors.at(i).size(); ++j) {
      for (size_t k = 0; k < statesMap_.at(poseId).sensors.at(i).at(j).size();
           ++k) {
        if (statesMap_.at(poseId).sensors.at(i).at(j).at(k).exists) {
          uint64_t id = statesMap_.at(poseId).sensors.at(i).at(j).at(k).id;
          if (mapPtr_->parameterBlockPtr(id)->fixed()) buffer << "(";
          buffer << "id=" << id << ":";
          buffer << mapPtr_->parameterBlockPtr(id)->typeInfo();
          if (mapPtr_->parameterBlockPtr(id)->fixed()) buffer << ")";
          buffer << ", ";
        }
      }
    }
  }
  buffer << std::endl;
}

// Initialise pose from IMU measurements. For convenience as static.
// Huai: this also can be realized with Quaterniond::FromTwoVectors() c.f.
// https://github.com/dennisss/mvision
bool HybridFilter::initPoseFromImu(
    const okvis::ImuMeasurementDeque& imuMeasurements,
    okvis::kinematics::Transformation& T_WS) {
  // set translation to zero, unit rotation
  T_WS.setIdentity();

  if (imuMeasurements.size() == 0) return false;

  // acceleration vector
  Eigen::Vector3d acc_B = Eigen::Vector3d::Zero();
  for (okvis::ImuMeasurementDeque::const_iterator it = imuMeasurements.begin();
       it < imuMeasurements.end(); ++it) {
    acc_B += it->measurement.accelerometers;
  }
  acc_B /= double(imuMeasurements.size());
  Eigen::Vector3d e_acc = acc_B.normalized();

  // align with ez_W:  //huai:this is expected direction of applied force,
  // opposite to gravity
  Eigen::Vector3d ez_W(0.0, 0.0, 1.0);
  Eigen::Matrix<double, 6, 1> poseIncrement;
  poseIncrement.head<3>() = Eigen::Vector3d::Zero();
  poseIncrement.tail<3>() = ez_W.cross(e_acc).normalized();
  double angle = std::acos(ez_W.transpose() * e_acc);
  poseIncrement.tail<3>() *= angle;
  T_WS.oplus(-poseIncrement);

  return true;
}

// set latest estimates for the assumed constant states which are commonly
// used in computing Jacobians of all feature observations
void HybridFilter::retrieveEstimatesOfConstants() {
  // X_c and all the augmented states including the last inserted one in which
  // a marginalized point has no observation in it p_B^C, f_x, f_y, c_x, c_y,
  // k_1, k_2, p_1, p_2, [k_3], t_d, t_r, \pi_{B_i}(=[p_{B_i}^G, q_{B_i}^G,
  // v_{B_i}^G]),
  numCamPosePointStates_ =
      9 + cameras::RadialTangentialDistortion::NumDistortionIntrinsics +
      9 * (statesMap_.size()) + 3 * mInCovLmIds.size();

  mStateID2CovID_.clear();
  int nCovIndex = 0;

  // note the statesMap_ is an ordered map!
  for (auto iter = statesMap_.begin(); iter != statesMap_.end(); ++iter) {
    mStateID2CovID_[iter->first] = nCovIndex;
    ++nCovIndex;
  }

  const int camIdx = 0;
  const uint64_t currFrameId = currentFrameId();
  getCameraSensorStates(currFrameId, camIdx, T_SC0_);

  Eigen::Matrix<double, 4 /*cameras::PinholeCamera::NumProjectionIntrinsics*/,
                1>
      intrinsic;
  getSensorStateEstimateAs<ceres::CameraIntrinsicParamBlock>(
      currFrameId, camIdx, SensorStates::Camera, CameraSensorStates::Intrinsic,
      intrinsic);
  OKVIS_ASSERT_EQ_DBG(
      Exception, cameras::RadialTangentialDistortion::NumDistortionIntrinsics,
      ceres::nDistortionDim, "radial tangetial parameter size inconsistent");

  Eigen::Matrix<double,
                cameras::RadialTangentialDistortion::NumDistortionIntrinsics, 1>
      distortionCoeffs;
  getSensorStateEstimateAs<ceres::CameraDistortionParamBlock>(
      currFrameId, camIdx, SensorStates::Camera, CameraSensorStates::Distortion,
      distortionCoeffs);

  okvis::cameras::RadialTangentialDistortion distortion(
      distortionCoeffs[0], distortionCoeffs[1], distortionCoeffs[2],
      distortionCoeffs[3]);

  Eigen::Matrix<
      double, 4 + cameras::RadialTangentialDistortion::NumDistortionIntrinsics,
      1>
      intrinsicParameters;
  intrinsicParameters << intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3],
      distortionCoeffs[0], distortionCoeffs[1], distortionCoeffs[2],
      distortionCoeffs[3];

  camera_rig_.setCameraIntrinsics(camIdx, intrinsicParameters);

  getSensorStateEstimateAs<ceres::CameraTimeParamBlock>(
      currFrameId, camIdx, SensorStates::Camera, CameraSensorStates::TD,
      tdLatestEstimate);
  getSensorStateEstimateAs<ceres::CameraTimeParamBlock>(
      currFrameId, camIdx, SensorStates::Camera, CameraSensorStates::TR,
      trLatestEstimate);

  Eigen::Matrix<double, 9, 1> vSM;
  getSensorStateEstimateAs<ceres::ShapeMatrixParamBlock>(
      currFrameId, 0, SensorStates::Imu, ImuSensorStates::TG, vSM);
  vTGTSTA_.head<9>() = vSM;
  getSensorStateEstimateAs<ceres::ShapeMatrixParamBlock>(
      currFrameId, 0, SensorStates::Imu, ImuSensorStates::TS, vSM);
  vTGTSTA_.segment<9>(9) = vSM;
  getSensorStateEstimateAs<ceres::ShapeMatrixParamBlock>(
      currFrameId, 0, SensorStates::Imu, ImuSensorStates::TA, vSM);
  vTGTSTA_.tail<9>() = vSM;

  // we do not set bg and ba here because
  // every time iem_ is used, resetBgBa is called
  iem_ = IMUErrorModel<double>(Eigen::Matrix<double, 6, 1>::Zero(), vTGTSTA_);
}

// assume the rolling shutter camera reads data row by row, and rows are
// aligned with the width of a frame some heuristics to defend outliers is
// used, e.g., ignore correspondences of too large discrepancy between
// prediction and measurement

bool HybridFilter::computeHxf(const uint64_t hpbid, const MapPoint& mp,
                              Eigen::Matrix<double, 2, 1>& r_i,
                              Eigen::Matrix<double, 2, Eigen::Dynamic>& H_x,
                              Eigen::Matrix<double, 2, Eigen::Dynamic>& H_f,
                              Eigen::Matrix2d& R_i) {
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
      // get the keypoint measurement
      okvis::MultiFramePtr multiFramePtr = multiFramePtrMap_.at(currFrameId);
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
      "a point in computeHxf should be observed in current frame!");
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
  Eigen::Matrix<
      double, 2,
      9 + cameras::RadialTangentialDistortion::NumDistortionIntrinsics>
      J_Xc;

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
  ImuMeasurement
      interpolatedInertialData;  // inertial data at the feature capture epoch

  kinematics::Transformation T_WBj;
  get_T_WS(currFrameId, T_WBj);

  SpeedAndBiases sbj;
  getSpeedAndBias(currFrameId, 0, sbj);

  auto imuMeasPtr = mStateID2Imu.find(currFrameId);
  OKVIS_ASSERT_TRUE(Exception, imuMeasPtr != mStateID2Imu.end(),
                    "the IMU measurement does not exist");
  const ImuMeasurementDeque& imuMeas = imuMeasPtr->second;

  Time stateEpoch = statesMap_.at(currFrameId).timestamp;
  const int camIdx = 0;
  uint32_t imageHeight = camera_rig_.getCameraGeometry(camIdx)->imageHeight();
  double kpN = obsInPixel[1] / imageHeight - 0.5;  // k per N
  Duration featureTime = Duration(tdLatestEstimate + trLatestEstimate * kpN) -
                         statesMap_.at(currFrameId).tdAtCreation;

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
    int numUsedImuMeasurements = -1;
    if (featureTime >= Duration()) {
      numUsedImuMeasurements = IMUOdometry::propagation(
          imuMeas, imuParametersVec_.at(0), T_WB, tempV_WS, iem, stateEpoch,
          stateEpoch + featureTime);
    } else {
      numUsedImuMeasurements = IMUOdometry::propagationBackward(
          imuMeas, imuParametersVec_.at(0), T_WB, tempV_WS, iem, stateEpoch,
          stateEpoch + featureTime);
    }
    sb.head<3>() = tempV_WS;
  }

  IMUOdometry::interpolateInertialData(imuMeas, iem, stateEpoch + featureTime,
                                       interpolatedInertialData);
  okvis::kinematics::Transformation T_WBa;
  get_T_WS(anchorId, T_WBa);
  okvis::kinematics::Transformation T_GA(
      mp.p_BA_G + T_WBa.r(), mp.q_GA);  // anchor frame to global frame
  okvis::kinematics::Transformation T_CA =
      (T_WB * T_SC0_).inverse() * T_GA;  // anchor frame to current camera frame
  Eigen::Vector3d pfiinC = (T_CA * ab1rho).head<3>();
  std::shared_ptr<okvis::cameras::CameraBase> tempCameraGeometry =
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
    // either filter outliers with this simple heuristic in here or
    // the mahalanobis distance in optimize
    Eigen::Vector2d discrep = obsInPixel - imagePoint;
    if (std::fabs(discrep[0]) > maxProjTolerance ||
        std::fabs(discrep[1]) > maxProjTolerance) {
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
    Eigen::Matrix<double, 6, 1> posVelFirstEstimate =
        statesMap_.at(currFrameId).linearizationPoint;
    lP_T_WB =
        kinematics::Transformation(posVelFirstEstimate.head<3>(), lP_T_WB.q());
    lP_sb.head<3>() = posVelFirstEstimate.tail<3>();

    Eigen::Vector3d tempV_WS = lP_sb.head<3>();
    int numUsedImuMeasurements = -1;
    if (featureTime >= Duration()) {
      numUsedImuMeasurements = IMUOdometry::propagation(
          imuMeas, imuParametersVec_.at(0), lP_T_WB, tempV_WS, iem, stateEpoch,
          stateEpoch + featureTime);
    } else {
      numUsedImuMeasurements = IMUOdometry::propagationBackward(
          imuMeas, imuParametersVec_.at(0), lP_T_WB, tempV_WS, iem, stateEpoch,
          stateEpoch + featureTime);
    }
    lP_sb.head<3>() = tempV_WS;
  }
  double rho = ab1rho[3];
  okvis::kinematics::Transformation T_BcA =
      lP_T_WB.inverse() *
      T_GA;  // anchor frame to the body frame associated with current frame
  J_td = pointJacobian3 * T_SC0_.C().transpose() *
         (okvis::kinematics::crossMx((T_BcA * ab1rho).head<3>()) *
              interpolatedInertialData.measurement.gyroscopes -
          T_WB.C().transpose() * lP_sb.head<3>() * rho);
  J_tr = J_td * kpN;
  J_Xc << pointJacobian3 * rho * (Eigen::Matrix3d::Identity() - T_CA.C()),
      intrinsicsJacobian, J_td, J_tr;

  Eigen::Matrix3d tempM3d;
  tempM3d << T_CA.C().topLeftCorner<3, 2>(), T_CA.r();
  J_pfi = pointJacobian3 * tempM3d;

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

  H_x.resize(2, numCamPosePointStates_ - 3 * mInCovLmIds.size());
  H_x.setZero();
  H_f.resize(2, 3 * mInCovLmIds.size());
  H_f.setZero();

  H_x.topLeftCorner<2, 9 + okvis::cameras::RadialTangentialDistortion::
                               NumDistortionIntrinsics>() = J_Xc;

  H_x.block<2, 9>(
      0,
      9 + okvis::cameras::RadialTangentialDistortion::NumDistortionIntrinsics +
          9 * mStateID2CovID_[currFrameId]) = J_XBj;
  H_x.block<2, 9>(
      0,
      9 + okvis::cameras::RadialTangentialDistortion::NumDistortionIntrinsics +
          9 * mStateID2CovID_[anchorId]) = J_XBa;

  std::deque<uint64_t>::iterator idPos =
      std::find(mInCovLmIds.begin(), mInCovLmIds.end(), hpbid);
  OKVIS_ASSERT_TRUE(Exception, idPos != mInCovLmIds.end(),
                    "The tracked landmark is not in mInCovLmIds ");
  size_t covPtId = idPos - mInCovLmIds.begin();
  H_f.block<2, 3>(0, covPtId * 3) = J_pfi;

  computeHTimer.stop();
  return true;
}

// TODO: hpbid homogeneous point block id, used only for debug

// assume the rolling shutter camera reads data row by row, and rows are
// aligned with the width of a frame some heuristics to defend outliers is
// used, e.g., ignore correspondences of too large discrepancy between
// prediction and measurement
bool HybridFilter::computeHoi(
    const uint64_t hpbid, const MapPoint& mp,
    Eigen::Matrix<double, Eigen::Dynamic, 1>& r_oi, Eigen::MatrixXd& H_oi,
    Eigen::MatrixXd& R_oi, Eigen::Vector4d& ab1rho,
    Eigen::Matrix<double, Eigen::Dynamic, 3>* pH_fi) const {
  computeHTimer.start();

  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      obsInPixel;                  // all observations for this feature point
  std::vector<uint64_t> frameIds;  // id of frames observing this feature point
  std::vector<double> vRi;         // std noise in pixels
  Eigen::Vector4d v4Xhomog;        // triangulated point position in the global
                                   // frame expressed in [X,Y,Z,W],
  // representing either an ordinary point or a ray, e.g., a point at infinity
  const int camIdx = 0;
  std::shared_ptr<okvis::cameras::CameraBase> tempCameraGeometry =
      camera_rig_.getCameraGeometry(camIdx);
  bool bSucceeded =
      triangulateAMapPoint(mp, obsInPixel, frameIds, v4Xhomog, vRi,
                           *dynamic_cast<okvis::cameras::PinholeCamera<
                               okvis::cameras::RadialTangentialDistortion>*>(
                               tempCameraGeometry.get()),
                           T_SC0_, hpbid);

  if (!bSucceeded) {
    computeHTimer.stop();
    return false;
  }

  // the anchor frame is chosen as the last frame observing the point, i.e.,
  // the frame just before the current frame
  uint64_t anchorId = frameIds.back();

  size_t numCamPoseStates = numCamPosePointStates_ - 3 * mInCovLmIds.size();
  // camera states, pose states, excluding feature states, and the velocity
  // dimesnion for the anchor state
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

  // compute Jacobians for a measurement in image j of the current feature i
  // C_j is the current frame, Bj refers to the body frame associated with the
  // current frame, Ba refers to body frame associated with the anchor frame,
  // f_i is the feature in consideration
  okvis::kinematics::Transformation
      T_WBa;  // transform from the body frame at the anchor frame epoch to
              // the world frame
  get_T_WS(anchorId, T_WBa);
  okvis::kinematics::Transformation T_GA =
      T_WBa * T_SC0_;  // anchor frame to global frame

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
  Eigen::Matrix<
      double, 2,
      9 + cameras::RadialTangentialDistortion::NumDistortionIntrinsics>
      J_Xc;

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

  ImuMeasurement interpolatedInertialData;

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
    uint64_t poseId = *itFrameIds;
    kinematics::Transformation T_WBj;
    get_T_WS(poseId, T_WBj);
    SpeedAndBiases sbj;
    getSpeedAndBias(poseId, 0, sbj);

    auto imuMeasPtr = mStateID2Imu.find(poseId);
    OKVIS_ASSERT_TRUE(Exception, imuMeasPtr != mStateID2Imu.end(),
                      "the IMU measurement does not exist");
    const ImuMeasurementDeque& imuMeas = imuMeasPtr->second;

    Time stateEpoch = statesMap_.at(poseId).timestamp;
    const int camIdx = 0;
    uint32_t imageHeight = camera_rig_.getCameraGeometry(camIdx)->imageHeight();
    double kpN = obsInPixel[kale][1] / imageHeight - 0.5;  // k per N
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
      int numUsedImuMeasurements = -1;
      if (featureTime >= Duration()) {
        numUsedImuMeasurements = IMUOdometry::propagation(
            imuMeas, imuParametersVec_.at(0), T_WB, tempV_WS, iem, stateEpoch,
            stateEpoch + featureTime);
      } else {
        numUsedImuMeasurements = IMUOdometry::propagationBackward(
            imuMeas, imuParametersVec_.at(0), T_WB, tempV_WS, iem, stateEpoch,
            stateEpoch + featureTime);
      }
      sb.head<3>() = tempV_WS;
    }

    IMUOdometry::interpolateInertialData(imuMeas, iem, stateEpoch + featureTime,
                                         interpolatedInertialData);

    okvis::kinematics::Transformation T_CA =
        (T_WB * T_SC0_).inverse() *
        T_GA;  // anchor frame to current camera frame
    Eigen::Vector3d pfiinC = (T_CA * ab1rho).head<3>();

    cameras::CameraBase::ProjectionStatus status = tempCameraGeometry->project(
        pfiinC, &imagePoint, &pointJacobian3, &intrinsicsJacobian);
    if (status != cameras::CameraBase::ProjectionStatus::Successful) {
      LOG(WARNING) << "Failed to compute Jacobian for distortion with "
                      "anchored point : "
                   << ab1rho.transpose() << " and [r,q]_CA"
                   << T_CA.coeffs().transpose();

      itFrameIds = frameIds.erase(itFrameIds);
      itRoi = vRi.erase(itRoi);
      itRoi = vRi.erase(itRoi);
      continue;
    } else if (!FLAGS_use_mahalanobis) {
      // either filter outliers with this simple heuristic in here or
      // the mahalanobis distance in optimize
      Eigen::Vector2d discrep = obsInPixel[kale] - imagePoint;
      if (std::fabs(discrep[0]) > maxProjTolerance ||
          std::fabs(discrep[1]) > maxProjTolerance) {
        itFrameIds = frameIds.erase(itFrameIds);
        itRoi = vRi.erase(itRoi);
        itRoi = vRi.erase(itRoi);
        continue;
      }
    }

    vri.push_back(obsInPixel[kale] - imagePoint);

    okvis::kinematics::Transformation lP_T_WB = T_WB;
    SpeedAndBiases lP_sb = sb;
    if (FLAGS_use_first_estimate) {
      lP_T_WB = T_WBj;
      lP_sb = sbj;
      Eigen::Matrix<double, 6, 1> posVelFirstEstimate =
          statesMap_.at(poseId).linearizationPoint;
      lP_T_WB = kinematics::Transformation(posVelFirstEstimate.head<3>(),
                                           lP_T_WB.q());
      lP_sb.head<3>() = posVelFirstEstimate.tail<3>();

      Eigen::Vector3d tempV_WS = lP_sb.head<3>();
      int numUsedImuMeasurements = -1;
      if (featureTime >= Duration()) {
        numUsedImuMeasurements = IMUOdometry::propagation(
            imuMeas, imuParametersVec_.at(0), lP_T_WB, tempV_WS, iem,
            stateEpoch, stateEpoch + featureTime);
      } else {
        numUsedImuMeasurements = IMUOdometry::propagationBackward(
            imuMeas, imuParametersVec_.at(0), lP_T_WB, tempV_WS, iem,
            stateEpoch, stateEpoch + featureTime);
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
    J_Xc << pointJacobian3 * rho * (Eigen::Matrix3d::Identity() - T_CA.C()),
        intrinsicsJacobian, J_td, J_tr;
    Eigen::Matrix3d tempM3d;
    tempM3d << T_CA.C().topLeftCorner<3, 2>(), T_CA.r();
    J_pfi = pointJacobian3 * tempM3d;

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

    H_x.setZero();
    H_x.topLeftCorner<2, 9 + okvis::cameras::RadialTangentialDistortion::
                                 NumDistortionIntrinsics>() = J_Xc;

    if (poseId == anchorId) {
      std::map<uint64_t, int>::const_iterator poseid_iter =
          mStateID2CovID_.find(poseId);
      H_x.block<2, 6>(0, 9 +
                             okvis::cameras::RadialTangentialDistortion::
                                 NumDistortionIntrinsics +
                             9 * poseid_iter->second + 3) =
          (J_XBj + J_XBa).block<2, 6>(0, 3);
    } else {
      std::map<uint64_t, int>::const_iterator poseid_iter =
          mStateID2CovID_.find(poseId);
      H_x.block<2, 9>(0, 9 +
                             okvis::cameras::RadialTangentialDistortion::
                                 NumDistortionIntrinsics +
                             9 * poseid_iter->second) = J_XBj;
      std::map<uint64_t, int>::const_iterator anchorid_iter =
          mStateID2CovID_.find(anchorId);
      H_x.block<2, 9>(0, 9 +
                             okvis::cameras::RadialTangentialDistortion::
                                 NumDistortionIntrinsics +
                             9 * anchorid_iter->second) = J_XBa;
    }

    vJ_X.push_back(H_x);
    vJ_pfi.push_back(J_pfi);

    ++numValidObs;
    ++itFrameIds;
    itRoi += 2;
  }
  if (numValidObs < 2) {
    computeHTimer.stop();
    return false;
  }
  OKVIS_ASSERT_EQ_DBG(Exception, numValidObs, frameIds.size(),
                      "Inconsistent number of observations and frameIds");

  // Now we stack the Jacobians and marginalize the point position related
  // dimensions. In other words, project $H_{x_i}$ onto the nullspace of
  // $H_{f^i}$

  Eigen::MatrixXd H_xi(2 * numValidObs, numCamPoseStates);
  Eigen::MatrixXd H_fi(2 * numValidObs, 3);
  Eigen::Matrix<double, Eigen::Dynamic, 1> ri(2 * numValidObs, 1);
  Eigen::MatrixXd Ri =
      Eigen::MatrixXd::Identity(2 * numValidObs, 2 * numValidObs);
  for (size_t saga = 0; saga < frameIds.size(); ++saga) {
    size_t saga2 = saga * 2;
    H_xi.block(saga2, 0, 2, numCamPoseStates) = vJ_X[saga];
    H_fi.block<2, 3>(saga2, 0) = vJ_pfi[saga];
    ri.segment<2>(saga2) = vri[saga];
    Ri(saga2, saga2) = vRi[saga2] * vRi[saga2];
    Ri(saga2 + 1, saga2 + 1) = vRi[saga2 + 1] * vRi[saga2 + 1];
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
  frameIds.clear();
  computeHTimer.stop();
  return true;
}

void HybridFilter::updateStates(
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& deltaX) {
  updateStatesTimer.start();
  OKVIS_ASSERT_EQ_DBG(Exception, deltaX.rows(), (int)covDim_,
                      "Inconsistent size of update to the states");
  const size_t numNavImuCamStates =
      15 + 27 + 9 +
      cameras::RadialTangentialDistortion::NumDistortionIntrinsics;
  // number of navigation, imu, and camera states in the covariance
  const size_t numNavImuCamPoseStates =
      numNavImuCamStates + 9 * statesMap_.size();

  if ((deltaX.head<9>() - deltaX.segment<9>(numNavImuCamPoseStates - 9))
          .lpNorm<Eigen::Infinity>() > 1e-8) {
    std::cout << "Warn: Correction to the current states from head and tail "
              << deltaX.head<9>().transpose() << std::endl
              << deltaX.segment<9>(numNavImuCamPoseStates - 9).transpose()
              << std::endl;
  }
  //    OKVIS_ASSERT_NEAR_DBG(Exception, (deltaX.head<9>() -
  //    deltaX.segment<9>(numNavImuCamPoseStates-9)).lpNorm<Eigen::Infinity>(),
  //                          0, 1e-8, "Correction to the current states from
  //                          head and tail should be identical");

  std::map<uint64_t, States>::reverse_iterator lastElementIterator =
      statesMap_.rbegin();
  const States& stateInQuestion = lastElementIterator->second;
  uint64_t stateId = stateInQuestion.id;

  // update global states
  std::shared_ptr<ceres::PoseParameterBlock> poseParamBlockPtr =
      std::static_pointer_cast<ceres::PoseParameterBlock>(
          mapPtr_->parameterBlockPtr(stateId));
  kinematics::Transformation T_WS = poseParamBlockPtr->estimate();
  Eigen::Vector3d deltaAlpha = deltaX.segment<3>(3);
  Eigen::Quaterniond deltaq =
      vio::quaternionFromSmallAngle(deltaAlpha);  // rvec2quat(deltaAlpha);
  T_WS = kinematics::Transformation(
      T_WS.r() + deltaX.head<3>(),
      deltaq *
          T_WS.q());  // in effect this amounts to PoseParameterBlock::plus()
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

  uint64_t TGId = stateInQuestion.sensors.at(SensorStates::Imu)
                      .at(imuIdx)
                      .at(ImuSensorStates::TG)
                      .id;
  std::shared_ptr<ceres::ShapeMatrixParamBlock> tgParamBlockPtr =
      std::static_pointer_cast<ceres::ShapeMatrixParamBlock>(
          mapPtr_->parameterBlockPtr(TGId));
  Eigen::Matrix<double, 9, 1> sm = tgParamBlockPtr->estimate();
  tgParamBlockPtr->setEstimate(sm + deltaX.segment<9>(15));

  uint64_t TSId = stateInQuestion.sensors.at(SensorStates::Imu)
                      .at(imuIdx)
                      .at(ImuSensorStates::TS)
                      .id;
  std::shared_ptr<ceres::ShapeMatrixParamBlock> tsParamBlockPtr =
      std::static_pointer_cast<ceres::ShapeMatrixParamBlock>(
          mapPtr_->parameterBlockPtr(TSId));
  sm = tsParamBlockPtr->estimate();
  tsParamBlockPtr->setEstimate(sm + deltaX.segment<9>(24));

  uint64_t TAId = stateInQuestion.sensors.at(SensorStates::Imu)
                      .at(imuIdx)
                      .at(ImuSensorStates::TA)
                      .id;
  std::shared_ptr<ceres::ShapeMatrixParamBlock> taParamBlockPtr =
      std::static_pointer_cast<ceres::ShapeMatrixParamBlock>(
          mapPtr_->parameterBlockPtr(TAId));
  sm = taParamBlockPtr->estimate();
  taParamBlockPtr->setEstimate(sm + deltaX.segment<9>(33));

  // update camera sensor states
  const int camIdx = 0;
  uint64_t extrinsicId = stateInQuestion.sensors.at(SensorStates::Camera)
                             .at(camIdx)
                             .at(CameraSensorStates::T_SCi)
                             .id;
  std::shared_ptr<ceres::PoseParameterBlock> extrinsicParamBlockPtr =
      std::static_pointer_cast<ceres::PoseParameterBlock>(
          mapPtr_->parameterBlockPtr(extrinsicId));
  kinematics::Transformation T_SC0 = extrinsicParamBlockPtr->estimate();
  T_SC0 = kinematics::Transformation(
      T_SC0.r() - T_SC0.C() * deltaX.segment<3>(42),
      T_SC0.q());  // the error state is $\delta p_B^C$ or $\delta p_S^C$
  extrinsicParamBlockPtr->setEstimate(T_SC0);

  uint64_t intrinsicId = stateInQuestion.sensors.at(SensorStates::Camera)
                             .at(camIdx)
                             .at(CameraSensorStates::Intrinsic)
                             .id;
  std::shared_ptr<ceres::CameraIntrinsicParamBlock> intrinsicParamBlockPtr =
      std::static_pointer_cast<ceres::CameraIntrinsicParamBlock>(
          mapPtr_->parameterBlockPtr(intrinsicId));
  Eigen::Matrix<double, 4, 1> cameraIntrinsics =
      intrinsicParamBlockPtr->estimate();
  intrinsicParamBlockPtr->setEstimate(cameraIntrinsics + deltaX.segment<4>(45));

  const int nDistortionCoeffDim =
      okvis::cameras::RadialTangentialDistortion::NumDistortionIntrinsics;
  uint64_t distortionId = stateInQuestion.sensors.at(SensorStates::Camera)
                              .at(camIdx)
                              .at(CameraSensorStates::Distortion)
                              .id;
  std::shared_ptr<ceres::CameraDistortionParamBlock> distortionParamBlockPtr =
      std::static_pointer_cast<ceres::CameraDistortionParamBlock>(
          mapPtr_->parameterBlockPtr(distortionId));
  Eigen::Matrix<double, nDistortionCoeffDim, 1> cameraDistortion =
      distortionParamBlockPtr->estimate();
  distortionParamBlockPtr->setEstimate(cameraDistortion +
                                       deltaX.segment<nDistortionCoeffDim>(49));

  uint64_t tdId = stateInQuestion.sensors.at(SensorStates::Camera)
                      .at(camIdx)
                      .at(CameraSensorStates::TD)
                      .id;
  std::shared_ptr<ceres::CameraTimeParamBlock> tdParamBlockPtr =
      std::static_pointer_cast<ceres::CameraTimeParamBlock>(
          mapPtr_->parameterBlockPtr(tdId));
  double td = tdParamBlockPtr->estimate();
  tdParamBlockPtr->setEstimate(td + deltaX[49 + nDistortionCoeffDim]);

  uint64_t trId = stateInQuestion.sensors.at(SensorStates::Camera)
                      .at(camIdx)
                      .at(CameraSensorStates::TR)
                      .id;
  std::shared_ptr<ceres::CameraTimeParamBlock> trParamBlockPtr =
      std::static_pointer_cast<ceres::CameraTimeParamBlock>(
          mapPtr_->parameterBlockPtr(trId));
  double tr = trParamBlockPtr->estimate();
  trParamBlockPtr->setEstimate(tr + deltaX[50 + nDistortionCoeffDim]);

  // Update augmented states except for the last one which is the current
  // state already updated this section assumes that the statesMap is an
  // ordered map
  size_t jack = 0;
  auto finalIter = statesMap_.end();
  --finalIter;

  for (auto iter = statesMap_.begin(); iter != finalIter; ++iter, ++jack) {
    stateId = iter->first;
    size_t qStart = 51 + nDistortionCoeffDim + 3 + 9 * jack;

    poseParamBlockPtr = std::static_pointer_cast<ceres::PoseParameterBlock>(
        mapPtr_->parameterBlockPtr(stateId));
    T_WS = poseParamBlockPtr->estimate();
    deltaAlpha = deltaX.segment<3>(qStart);
    deltaq =
        vio::quaternionFromSmallAngle(deltaAlpha);  // rvec2quat(deltaAlpha);
    T_WS = kinematics::Transformation(
        T_WS.r() + deltaX.segment<3>(qStart - 3),
        deltaq * T_WS.q());  // in effect this amounts to
                             // PoseParameterBlock::plus()
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
  jack = 0;
  size_t lkStart = 51 + nDistortionCoeffDim + 9 * statesMap_.size();
  size_t aStart = lkStart - 3;  // a dummy initialization
  for (auto iter = mInCovLmIds.begin(), iterEnd = mInCovLmIds.end();
       iter != iterEnd; ++iter, ++jack) {
    std::shared_ptr<okvis::ceres::HomogeneousPointParameterBlock> hppb =
        std::static_pointer_cast<okvis::ceres::HomogeneousPointParameterBlock>(
            mapPtr_->parameterBlockPtr(*iter));
    Eigen::Vector4d ab1rho =
        hppb->estimate();  // inverse depth parameterization in the anchor
                           // frame, [\alpha= X/Z, \beta= Y/Z, 1, \rho=1/Z]
    aStart = lkStart + 3 * jack;
    ab1rho[0] += deltaX[aStart];
    ab1rho[1] += deltaX[aStart + 1];
    ab1rho[3] += deltaX[aStart + 2];
    hppb->setEstimate(ab1rho);
  }
  OKVIS_ASSERT_EQ_DBG(Exception, aStart + 3, (size_t)deltaX.rows(),
                      "deltaX size not equal to what's' expected.");
  updateStatesTimer.stop();
}

// TODO: theoretically the filtering update step can run several times for one
// set of observations, i.e., iterative EKF but the current implementation
// does not account for that.
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

  OKVIS_ASSERT_EQ_DBG(
      Exception, covDim_,
      51 + cameras::RadialTangentialDistortion::NumDistortionIntrinsics +
          9 * statesMap_.size() + 3 * mInCovLmIds.size(),
      "Inconsistent covDim and number of states");
  // prepare intermediate variables for computing Jacobians

  retrieveEstimatesOfConstants();

  size_t dimH_o[2] = {0, numCamPosePointStates_ - 3 * mInCovLmIds.size() - 9};
  size_t nMarginalizedFeatures =
      0;  // features not in state and not tracked in current frame
  size_t nInStateFeatures = 0;  // features in state and tracked now
  size_t nToAddFeatures =
      0;  // features tracked long enough and to be included in states
  mLandmarkID2Residualize.clear();

  Eigen::MatrixXd variableCov = covariance_.block(
      okvis::ceres::ode::OdoErrorStateDim, okvis::ceres::ode::OdoErrorStateDim,
      dimH_o[1],
      dimH_o[1]);  // covariance of camera and pose copy states
  Eigen::MatrixXd variableCov2 = covariance_.block(
      okvis::ceres::ode::OdoErrorStateDim, okvis::ceres::ode::OdoErrorStateDim,
      dimH_o[1] + 9,
      dimH_o[1] + 9);  // covariance of camera and pose copy states
  size_t tempCounter = 0;
  mLandmarkID2Residualize.resize(landmarksMap_.size());
  for (auto it = landmarksMap_.begin(); it != landmarksMap_.end();
       ++it, ++tempCounter) {
    ResidualizeCase toResidualize = NotInState_NotTrackedNow;
    const size_t nNumObs = it->second.observations.size();
    if (it->second.anchorStateId == 0) {  // this point is not in the states
      for (auto itObs = it->second.observations.rbegin(),
                iteObs = it->second.observations.rend();
           itObs != iteObs; ++itObs) {
        if (itObs->first.frameId == currFrameId) {
          if (nNumObs == mMaxM) {  // this point is to be included in the states
            toResidualize = ToAdd_TrackedNow;
            ++nToAddFeatures;
          } else {
            OKVIS_ASSERT_LT_DBG(Exception, nNumObs, mMaxM,
                                "A point not in state should not have "
                                "consecutive features more than mMaxM.");
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

    mLandmarkID2Residualize[tempCounter].first = it->second.id;
    mLandmarkID2Residualize[tempCounter].second = toResidualize;

    if (toResidualize == NotInState_NotTrackedNow &&
        nNumObs > 2)  // TODO: is >2 too harsh or lenient?
    {
      // H_oi , r_oi, and R_oi will resize in computeHoi
      Eigen::MatrixXd H_oi;                           //(2n-3, dimH_o[1])
      Eigen::Matrix<double, Eigen::Dynamic, 1> r_oi;  //(2n-3, 1)
      Eigen::MatrixXd R_oi;                           //(2n-3, 2n-3)
      Eigen::Vector4d ab1rho;
      bool isValidJacobian =
          computeHoi(it->first, it->second, r_oi, H_oi, R_oi, ab1rho);
      if (!isValidJacobian) continue;

      if (FLAGS_use_mahalanobis) {
        // the below test looks time consuming as it involves matrix
        // inversion. alternatively, some heuristics in computeHoi is used,
        // e.g., ignore correspondences of too large discrepancy
        /// remove outliders, cf. Li RSS12 optimization based ... eq 6
        double gamma =
            r_oi.transpose() *
            (H_oi * variableCov * H_oi.transpose() + R_oi).inverse() * r_oi;
        if (gamma > chi2_95percentile[r_oi.rows()]) continue;
      }

      vr_o.push_back(r_oi);
      vR_o.push_back(R_oi);
      vH_o.push_back(H_oi);
      dimH_o[0] += r_oi.rows();
      ++nMarginalizedFeatures;
    } else if (toResidualize ==
               InState_TrackedNow)  // compute residual and Jacobian for a
                                    // observed point which is in the states
    {
      Eigen::Matrix<double, 2, 1> r_i;
      Eigen::Matrix<double, 2, Eigen::Dynamic> H_x;
      Eigen::Matrix<double, 2, Eigen::Dynamic> H_f;
      Eigen::Matrix2d R_i;
      bool isValidJacobian =
          computeHxf(it->first, it->second, r_i, H_x, H_f, R_i);
      if (!isValidJacobian) continue;

      double gamma = r_i.transpose() *
                     (H_x * variableCov2 * H_x.transpose() + R_i).inverse() *
                     r_i;
      if (gamma > chi2_95percentile[2]) {
        mLandmarkID2Residualize[tempCounter].second = InState_NotTrackedNow;
        continue;
      }

      vr_i.push_back(r_i);
      vH_x.push_back(H_x);
      vH_f.push_back(H_f);
      vR_i.push_back(R_i);
      ++nInStateFeatures;
    }

  }  // every landmark

  tempCounter = 0;
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
    Eigen::MatrixXd H_all(obsRows, numCamPosePointStates_);

    H_all.topLeftCorner(rqRows, dimH_o[1]) = T_H;
    H_all.block(0, dimH_o[1], rqRows, numPointStates + 9).setZero();

    Eigen::Matrix<double, Eigen::Dynamic, 1> r_all(obsRows, 1);
    r_all.head(rqRows) = r_q;
    Eigen::MatrixXd R_all = Eigen::MatrixXd::Zero(obsRows, obsRows);
    R_all.topLeftCorner(rqRows, rqRows) = R_q;

    startRow = rqRows;
    for (size_t jack = 0; jack < nInStateFeatures; ++jack) {
      H_all.block(startRow, 0, 2, numCamPosePointStates_ - numPointStates) =
          vH_x[jack];
      H_all.block(startRow, numCamPosePointStates_ - numPointStates, 2,
                  numPointStates) = vH_f[jack];
      r_all.block<2, 1>(startRow, 0) = vr_i[jack];
      R_all.block<2, 2>(startRow, startRow) = vR_i[jack];
      startRow += 2;
    }

    // Calculate Kalman gain
    Eigen::MatrixXd S =
        H_all *
            covariance_.bottomRightCorner(numCamPosePointStates_,
                                          numCamPosePointStates_) *
            H_all.transpose() +
        R_all;

    Eigen::MatrixXd K =
        (covariance_.bottomRightCorner(covDim_, numCamPosePointStates_) *
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
        Eigen::MatrixXd tempMat = Eigen::MatrixXd::Identity(covDim_, covDim_);
        tempMat.block(0, okvis::ceres::ode::OdoErrorStateDim, covDim_, numCamPosePointStates_) -= K*H_all;
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
    const size_t numNavImuCamStates =
        15 + 27 + 9 +
        cameras::RadialTangentialDistortion::NumDistortionIntrinsics;
    const size_t numNavImuCamPoseStates =
        numNavImuCamStates + 9 * statesMap_.size();
    if ((covariance_.topLeftCorner(covDim_, 9) -
         covariance_.block(0, numNavImuCamPoseStates - 9, covDim_, 9))
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

    retrieveEstimatesOfConstants();  // do this because states are just
                                     // updated

    tempCounter = 0;
    size_t totalObsDim = 0;  // total dimensions of all features' observations
    const size_t numCamPoseStates =
        numCamPosePointStates_ - 3 * mInCovLmIds.size();
    Eigen::MatrixXd variableCov = covariance_.block(
        okvis::ceres::ode::OdoErrorStateDim,
        okvis::ceres::ode::OdoErrorStateDim, numCamPoseStates,
        numCamPoseStates);  // covariance of camera and pose copy states

    std::vector<uint64_t> toAddLmIds;  // id of landmarks to add to the states
    for (PointMap::iterator pit = landmarksMap_.begin();
         pit != landmarksMap_.end(); ++pit, ++tempCounter) {
      if (mLandmarkID2Residualize[tempCounter].second == ToAdd_TrackedNow) {
        Eigen::Vector4d
            ab1rho;  //[\alpha, \beta, 1, \rho] of the point in the anchor
                     // frame, representing either an ordinary point or a ray
        bool isValidJacobian =
            computeHoi(pit->first, pit->second, r_i, H_i, R_i, ab1rho, &H_fi);

        if (!isValidJacobian) {  // remove this feature later
          mLandmarkID2Residualize[tempCounter].second =
              NotInState_NotTrackedNow;
          continue;
        }

        vio::leftNullspaceAndColumnSpace(H_fi, &Q2, &Q1);
        z_o = Q2.transpose() * r_i;
        H_o = Q2.transpose() * H_i;
        R_o = Q2.transpose() * R_i * Q2;

        if (FLAGS_use_mahalanobis) {
          // the below test looks time consuming as it involves matrix
          // inversion. alternatively, some heuristics in computeHoi is used,
          // e.g., ignore correspondences of too large discrepancy
          /// remove outliders, cf. Li RSS12 optimization based ... eq 6
          double gamma = z_o.transpose() *
                         (H_o * variableCov * H_o.transpose() + R_o).inverse() *
                         z_o;
          if (gamma > chi2_95percentile[z_o.rows()]) {
            mLandmarkID2Residualize[tempCounter].second =
                NotInState_NotTrackedNow;
            continue;
          }
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
        okvis::kinematics::Transformation T_GA =
            T_GBa * T_SC0_;  // anchor frame to global frame

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

      tempCounter = 0;
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
          (covariance_.block(0, okvis::ceres::ode::OdoErrorStateDim, covDim_,
                             numCamPoseStates) *
           H_o.transpose()) *
          S.inverse();

      updateCovarianceTimer.start();
      Eigen::MatrixXd Paug(covDim_ + nNewFeatures * 3,
                           covDim_ + nNewFeatures * 3);
#if 0  // Joseph form
            Eigen::MatrixXd tempMat = Eigen::MatrixXd::Identity(covDim_, covDim_);
            tempMat.block(0, okvis::ceres::ode::OdoErrorStateDim, covDim_, numCamPoseStates) -= K*H_o;
            Paug.topLeftCorner(covDim_, covDim_) = tempMat * (covariance_ * tempMat.transpose()).eval() + K * R_o * K.transpose();
#else  // Li Mingyang RSS 12 optimization based
      Paug.topLeftCorner(covDim_, covDim_) =
          covariance_ - K * S * K.transpose();
#endif
      Eigen::MatrixXd invH2H1 = invH_2 * H_1;
      Paug.block(covDim_, 0, 3 * nNewFeatures, covDim_) =
          -invH2H1 * Paug.block(okvis::ceres::ode::OdoErrorStateDim, 0,
                                numCamPoseStates, covDim_);
      Paug.block(0, covDim_, covDim_, 3 * nNewFeatures) =
          Paug.block(covDim_, 0, 3 * nNewFeatures, covDim_).transpose();
      Paug.bottomRightCorner(3 * nNewFeatures, 3 * nNewFeatures) =
          -Paug.block(covDim_, okvis::ceres::ode::OdoErrorStateDim,
                      3 * nNewFeatures, numCamPoseStates) *
              invH2H1.transpose() +
          invH_2 * R_1 * invH_2.transpose();
      covariance_ = Paug;
      covDim_ = covariance_.rows();
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
      const size_t numNavImuCamStates =
          15 + 27 + 9 +
          cameras::RadialTangentialDistortion::NumDistortionIntrinsics;
      const size_t numNavImuCamPoseStates =
          numNavImuCamStates + 9 * statesMap_.size();
      if ((covariance_.topLeftCorner(covDim_, 9) -
           covariance_.block(0, numNavImuCamPoseStates - 9, covDim_, 9))
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
      Eigen::Matrix<double, Eigen::Dynamic, 1> deltaX(covDim_, 1);
      deltaX.head(covDim_ - 3 * nNewFeatures) = deltaXo;
      deltaX.tail(3 * nNewFeatures) =
          -invH2H1 * deltaXo.segment(okvis::ceres::ode::OdoErrorStateDim,
                                     numCamPoseStates) +
          invH_2 * z_1;
      OKVIS_ASSERT_FALSE(Exception,
                         std::isnan(deltaX(0)) ||
                             std::isnan(deltaX(covDim_ - 3 * nNewFeatures)),
                         "nan in kalman filter's correction in adding points");
      updateStates(deltaX);
      std::cout << "finish initializing features into states " << std::endl;
    }
  }

  /// update minValidStateID for removing old states
  /// also update landmark positions which is only necessary when
  /// (1) landmark coordinates are used to predict the points projection in
  /// new frames OR (2) to visualize the points
  tempCounter = 0;
  minValidStateID = currFrameId;
  for (auto it = landmarksMap_.begin(); it != landmarksMap_.end();
       ++it, ++tempCounter) {
    ResidualizeCase residualizeCase =
        mLandmarkID2Residualize[tempCounter].second;
    if (residualizeCase == NotInState_NotTrackedNow ||
        residualizeCase == InState_NotTrackedNow)
      continue;

    if (residualizeCase == NotToAdd_TrackedNow) {
      if (it->second.observations.size() <
          2)  // this happens with a just inserted landmark that has not been
              // triangulated.
        continue;

      auto itObs = it->second.observations.begin();
      if (itObs->first.frameId <
          minValidStateID)  // this assume that it->second.observations is an
                            // ordered map
        minValidStateID = itObs->first.frameId;

#if 0
            //only do the following when observation size is greater than 2
            //Superfluous: Update coordinates of landmarks tracked in the current frame but are not in the states
            std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > obsInPixel;
            std::vector<uint64_t> frameIds;
            std::vector<double> vRi; //std noise in pixels
            Eigen::Vector4d v4Xhomog;
            bool bSucceeded= triangulateAMapPoint(it->second, obsInPixel,
                                                  frameIds, v4Xhomog, vRi, *dynamic_cast<okvis::cameras::PinholeCamera<
                                                  okvis::cameras::RadialTangentialDistortion>*>(
                                                  tempCameraGeometry.get()), T_SC0_, it->first);
            if(bSucceeded){
                it->second.quality = 1.0;
                it->second.pointHomog = v4Xhomog; //this is position in the global frame
            }
            else
                it->second.quality = 0.0;
#endif
    } else  // SLAM features,
    {
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

// Set a time limit for the optimization process.
bool HybridFilter::setOptimizationTimeLimit(double timeLimit,
                                            int minIterations) {
  if (ceresCallback_ != nullptr) {
    if (timeLimit < 0.0) {
      // no time limit => set minimum iterations to maximum iterations
      ceresCallback_->setMinimumIterations(mapPtr_->options.max_num_iterations);
      return true;
    }
    ceresCallback_->setTimeLimit(timeLimit);
    ceresCallback_->setMinimumIterations(minIterations);
    return true;
  } else if (timeLimit >= 0.0) {
    ceresCallback_ = std::unique_ptr<okvis::ceres::CeresIterationCallback>(
        new okvis::ceres::CeresIterationCallback(timeLimit, minIterations));
    mapPtr_->options.callbacks.push_back(ceresCallback_.get());
    return true;
  }
  // no callback yet registered with ceres.
  // but given time limit is lower than 0, so no callback needed
  return true;
}

// getters
// Get a specific landmark.
bool HybridFilter::getLandmark(uint64_t landmarkId, MapPoint& mapPoint) const {
  std::lock_guard<std::mutex> l(statesMutex_);
  if (landmarksMap_.find(landmarkId) == landmarksMap_.end()) {
    OKVIS_THROW_DBG(Exception,
                    "landmark with id = " << landmarkId << " does not exist.")
    return false;
  }
  mapPoint = landmarksMap_.at(landmarkId);
  return true;
}

// Checks whether the landmark is initialized.
bool HybridFilter::isLandmarkInitialized(uint64_t landmarkId) const {
  OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId),
                        "landmark not added");
  return std::static_pointer_cast<okvis::ceres::HomogeneousPointParameterBlock>(
             mapPtr_->parameterBlockPtr(landmarkId))
      ->initialized();
}

// Get a copy of all the landmarks as a PointMap.
size_t HybridFilter::getLandmarks(PointMap& landmarks) const {
  std::lock_guard<std::mutex> l(statesMutex_);
  landmarks = landmarksMap_;
  return landmarksMap_.size();
}

// Get a copy of all the landmark in a MapPointVector. This is for legacy
// support. Use getLandmarks(okvis::PointMap&) if possible.
size_t HybridFilter::getLandmarks(MapPointVector& landmarks) const {
  std::lock_guard<std::mutex> l(statesMutex_);
  landmarks.clear();
  landmarks.reserve(landmarksMap_.size());
  for (PointMap::const_iterator it = landmarksMap_.begin();
       it != landmarksMap_.end(); ++it) {
    landmarks.push_back(it->second);
  }
  return landmarksMap_.size();
}

// Get pose for a given pose ID.
bool HybridFilter::get_T_WS(uint64_t poseId,
                            okvis::kinematics::Transformation& T_WS) const {
  if (!getGlobalStateEstimateAs<ceres::PoseParameterBlock>(
          poseId, GlobalStates::T_WS, T_WS)) {
    return false;
  }

  return true;
}

// Feel free to implement caching for them...
// Get speeds and IMU biases for a given pose ID.
bool HybridFilter::getSpeedAndBias(uint64_t poseId, uint64_t imuIdx,
                                   okvis::SpeedAndBiases& speedAndBias) const {
  if (!getSensorStateEstimateAs<ceres::SpeedAndBiasParameterBlock>(
          poseId, imuIdx, SensorStates::Imu, ImuSensorStates::SpeedAndBias,
          speedAndBias)) {
    return false;
  }
  return true;
}

// Get camera states for a given pose ID.
bool HybridFilter::getCameraSensorStates(
    uint64_t poseId, size_t cameraIdx,
    okvis::kinematics::Transformation& T_SCi) const {
  return getSensorStateEstimateAs<ceres::PoseParameterBlock>(
      poseId, cameraIdx, SensorStates::Camera, CameraSensorStates::T_SCi,
      T_SCi);
}

// Get the ID of the current keyframe.
uint64_t HybridFilter::currentKeyframeId() const {
  for (std::map<uint64_t, States>::const_reverse_iterator rit =
           statesMap_.rbegin();
       rit != statesMap_.rend(); ++rit) {
    if (rit->second.isKeyframe) {
      return rit->first;
    }
  }
  OKVIS_THROW_DBG(Exception, "no keyframes existing...");
  return 0;
}

// Get the ID of an older frame.
uint64_t HybridFilter::frameIdByAge(size_t age) const {
  std::map<uint64_t, States>::const_reverse_iterator rit = statesMap_.rbegin();
  for (size_t i = 0; i < age; ++i) {
    ++rit;
    OKVIS_ASSERT_TRUE_DBG(Exception, rit != statesMap_.rend(),
                          "requested age " << age << " out of range.");
  }
  return rit->first;
}

// Get the ID of the newest frame added to the state.
uint64_t HybridFilter::currentFrameId() const {
  OKVIS_ASSERT_TRUE_DBG(Exception, statesMap_.size() > 0,
                        "no frames added yet.")
  return statesMap_.rbegin()->first;
}

// Checks if a particular frame is still in the IMU window
bool HybridFilter::isInImuWindow(uint64_t frameId) const {
  if (statesMap_.at(frameId).sensors.at(SensorStates::Imu).size() == 0) {
    return false;  // no IMU added
  }
  return statesMap_.at(frameId)
      .sensors.at(SensorStates::Imu)
      .at(0)
      .at(ImuSensorStates::SpeedAndBias)
      .exists;
}

// Set pose for a given pose ID.
bool HybridFilter::set_T_WS(uint64_t poseId,
                            const okvis::kinematics::Transformation& T_WS) {
  if (!setGlobalStateEstimateAs<ceres::PoseParameterBlock>(
          poseId, GlobalStates::T_WS, T_WS)) {
    return false;
  }

  return true;
}

// Set the speeds and IMU biases for a given pose ID.
bool HybridFilter::setSpeedAndBias(uint64_t poseId, size_t imuIdx,
                                   const okvis::SpeedAndBiases& speedAndBias) {
  return setSensorStateEstimateAs<ceres::SpeedAndBiasParameterBlock>(
      poseId, imuIdx, SensorStates::Imu, ImuSensorStates::SpeedAndBias,
      speedAndBias);
}

// Set the transformation from sensor to camera frame for a given pose ID.
bool HybridFilter::setCameraSensorStates(
    uint64_t poseId, size_t cameraIdx,
    const okvis::kinematics::Transformation& T_SCi) {
  return setSensorStateEstimateAs<ceres::PoseParameterBlock>(
      poseId, cameraIdx, SensorStates::Camera, CameraSensorStates::T_SCi,
      T_SCi);
}

// Set the homogeneous coordinates for a landmark.
bool HybridFilter::setLandmark(uint64_t landmarkId,
                               const Eigen::Vector4d& landmark) {
  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr =
      mapPtr_->parameterBlockPtr(landmarkId);
#ifndef NDEBUG
  std::shared_ptr<ceres::HomogeneousPointParameterBlock>
      derivedParameterBlockPtr =
          std::dynamic_pointer_cast<ceres::HomogeneousPointParameterBlock>(
              parameterBlockPtr);
  if (!derivedParameterBlockPtr) {
    OKVIS_THROW_DBG(Exception, "wrong pointer type requested.")
    return false;
  }
  derivedParameterBlockPtr->setEstimate(landmark);
  ;
#else
  std::static_pointer_cast<ceres::HomogeneousPointParameterBlock>(
      parameterBlockPtr)
      ->setEstimate(landmark);
#endif

  // also update in map
  landmarksMap_.at(landmarkId).pointHomog = landmark;
  return true;
}

// Set the landmark initialization state.
void HybridFilter::setLandmarkInitialized(uint64_t landmarkId,
                                          bool initialized) {
  OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId),
                        "landmark not added");
  std::static_pointer_cast<okvis::ceres::HomogeneousPointParameterBlock>(
      mapPtr_->parameterBlockPtr(landmarkId))
      ->setInitialized(initialized);
}

// private stuff
// getters
bool HybridFilter::getGlobalStateParameterBlockPtr(
    uint64_t poseId, int stateType,
    std::shared_ptr<ceres::ParameterBlock>& stateParameterBlockPtr) const {
  // check existence in states set
  if (statesMap_.find(poseId) == statesMap_.end()) {
    OKVIS_THROW(Exception, "pose with id = " << poseId << " does not exist.")
    return false;
  }

  // obtain the parameter block ID
  uint64_t id = statesMap_.at(poseId).global.at(stateType).id;
  if (!mapPtr_->parameterBlockExists(id)) {
    OKVIS_THROW(Exception, "pose with id = " << id << " does not exist.")
    return false;
  }

  stateParameterBlockPtr = mapPtr_->parameterBlockPtr(id);
  return true;
}

bool HybridFilter::getSensorStateParameterBlockPtr(
    uint64_t poseId, int sensorIdx, int sensorType, int stateType,
    std::shared_ptr<ceres::ParameterBlock>& stateParameterBlockPtr) const {
  // check existence in states set
  if (statesMap_.find(poseId) == statesMap_.end()) {
    OKVIS_THROW_DBG(Exception,
                    "pose with id = " << poseId << " does not exist.")
    return false;
  }

  // obtain the parameter block ID
  uint64_t id = statesMap_.at(poseId)
                    .sensors.at(sensorType)
                    .at(sensorIdx)
                    .at(stateType)
                    .id;
  if (!mapPtr_->parameterBlockExists(id)) {
    OKVIS_THROW_DBG(Exception,
                    "pose with id = " << poseId << " does not exist.")
    return false;
  }
  stateParameterBlockPtr = mapPtr_->parameterBlockPtr(id);
  return true;
}

// TODO(jhuai): mapPtr_ and hpbid is only used for debug
void HybridFilter::gatherPoseObservForTriang(
    const MapPoint& mp,
    const cameras::PinholeCamera<cameras::RadialTangentialDistortion>&
        cameraGeometry,
    std::vector<uint64_t>* frameIds,
    std::vector<okvis::kinematics::Transformation>* T_WSs,
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>*
        obsDirections,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>*
        obsInPixel,
    std::vector<double>* vR_oi, const uint64_t& hpbid) const {
  frameIds->clear();
  T_WSs->clear();
  obsDirections->clear();
  obsInPixel->clear();
  vR_oi->clear();
  for (auto itObs = mp.observations.begin(), iteObs = mp.observations.end();
       itObs != iteObs; ++itObs) {
    uint64_t poseId = itObs->first.frameId;
    Eigen::Vector2d measurement;
    okvis::MultiFramePtr multiFramePtr =
        multiFramePtrMap_.at(itObs->first.frameId);
    multiFramePtr->getKeypoint(itObs->first.cameraIndex,
                               itObs->first.keypointIndex, measurement);

#if 0  // debugging
       std::shared_ptr <const ceres::ReprojectionError
               < okvis::cameras::PinholeCamera<
               okvis::cameras::RadialTangentialDistortion> > > reprojectionErrorPtr=
               std::dynamic_pointer_cast< const
               okvis::ceres::ReprojectionError< okvis::cameras::PinholeCamera<
               okvis::cameras::RadialTangentialDistortion> > >(
                   mapPtr_->errorInterfacePtr(reinterpret_cast< ::ceres::ResidualBlockId>(itObs->second)));

       Eigen::Vector2d measurement1 = reprojectionErrorPtr->measurement();
       OKVIS_ASSERT_LT(Exception, (measurement - measurement1).norm(), 1e-8,
                       "wrong keypoint correspondence in map and frame!");
#endif

    obsInPixel->push_back(measurement);

    double kpSize = 1.0;
    multiFramePtr->getKeypointSize(itObs->first.cameraIndex,
                                   itObs->first.keypointIndex, kpSize);
    // image pixel noise follows that in addObservation function
    vR_oi->push_back(kpSize / 8);
    vR_oi->push_back(kpSize / 8);

    // use the latest estimates for camera intrinsic parameters
    Eigen::Vector3d backProjectionDirection;
    cameraGeometry.backProject(measurement, &backProjectionDirection);
    // each observation is in image plane z=1, (\bar{x}, \bar{y}, 1)
    obsDirections->push_back(backProjectionDirection);

    okvis::kinematics::Transformation T_WS;
    get_T_WS(poseId, T_WS);
    T_WSs->push_back(T_WS);
    frameIds->push_back(poseId);
  }

#if 0  /// point's world position estimated during the feature tracking, for
     /// debugging
   std::shared_ptr<okvis::ceres::HomogeneousPointParameterBlock> pointPBPtr =
           std::static_pointer_cast<okvis::ceres::HomogeneousPointParameterBlock>( mapPtr_->parameterBlockPtr(hpbid));
   Eigen::Vector4d homoPoint = pointPBPtr->estimate();
   std::cout<< "okvis point estimate :"<< (homoPoint.head<3>().transpose()/homoPoint[3])<<std::endl;
#endif
}

bool HybridFilter::triangulateAMapPoint(
    const MapPoint& mp,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
        obsInPixel,
    std::vector<uint64_t>& frameIds, Eigen::Vector4d& v4Xhomog,
    std::vector<double>& vR_oi,
    const cameras::PinholeCamera<cameras::RadialTangentialDistortion>&
        cameraGeometry,
    const kinematics::Transformation& T_SC0, const uint64_t& hpbid,
    bool use_AIDP) const {
  triangulateTimer.start();

  // each entry is undistorted coordinates in image plane at
  // z=1 in the specific camera frame, [\bar{x},\bar{y},1]
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      obsDirections;
  // the SE3 transform from world to camera frame
  std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> T_CWs;

  std::vector<okvis::kinematics::Transformation> T_WSs;
  gatherPoseObservForTriang(mp, cameraGeometry, &frameIds, &T_WSs,
                            &obsDirections, &obsInPixel, &vR_oi, hpbid);
  if (use_AIDP) {  // Ca will play the role of W in T_CWs
    std::vector<okvis::kinematics::Transformation> reversed_T_CWs;
    okvis::kinematics::Transformation T_WCa;
    for (std::vector<okvis::kinematics::Transformation>::const_reverse_iterator
             riter = T_WSs.rbegin();
         riter != T_WSs.rend(); ++riter) {
      okvis::kinematics::Transformation T_WCi = (*riter) * T_SC0;
      if (riter == T_WSs.rbegin()) {
        reversed_T_CWs.emplace_back(okvis::kinematics::Transformation());
        T_WCa = T_WCi;
      } else {
        okvis::kinematics::Transformation T_CiW = T_WCi.inverse() * T_WCa;
        reversed_T_CWs.emplace_back(T_CiW);
      }
    }
    for (std::vector<okvis::kinematics::Transformation>::const_reverse_iterator
             riter = reversed_T_CWs.rbegin();
         riter != reversed_T_CWs.rend(); ++riter) {
      T_CWs.emplace_back(Sophus::SE3d(riter->q(), riter->r()));
    }
  } else {
    for (std::vector<okvis::kinematics::Transformation>::const_iterator iter =
             T_WSs.begin();
         iter != T_WSs.end(); ++iter) {
      okvis::kinematics::Transformation T_WCi = *iter * T_SC0;
      okvis::kinematics::Transformation T_CiW = T_WCi.inverse();
      T_CWs.emplace_back(Sophus::SE3d(T_CiW.q(), T_CiW.r()));
    }
  }

#if 1
  // Method 1 estimate point's world position with DLT + gauss newton,
  // for simulations with ideal conditions, this method may perform better
  //   than method 2
  bool isValid;
  bool isParallel;
  v4Xhomog = Get_X_from_xP_lin(obsDirections, T_CWs, isValid, isParallel);
  if (isValid && !isParallel) {
    Eigen::Vector3d res = v4Xhomog.head<3>() / v4Xhomog[3];
    //        std::cout<<"before Gauss Optimization:"<<
    //        res.transpose()<<std::endl;
    triangulate_refine_GN(obsDirections, T_CWs, res, 5);
    v4Xhomog[3] = 1;
    v4Xhomog.head<3>() = res;
    //        std::cout<<"after Gauss Optimization:"<<
    //        res.transpose()<<std::endl;
    triangulateTimer.stop();
    return true;
  } else {
    std::cout << " cannot triangulate pure rotation or infinity points "
              << v4Xhomog.transpose() << std::endl;
    triangulateTimer.stop();
    return false;
  }
#else   // method 2 Median point method and/or gauss newton
  bool isValid(false);     // is triangulation valid, i.e., not too large
                           // uncertainty
  bool isParallel(false);  // is a ray

  // test parallel following okvis::triangulateFast using only the first,
  // last, and middle observations
  okvis::kinematics::Transformation T_AW(T_CWs.front().translation(),
                                         T_CWs.front().unit_quaternion());
  okvis::kinematics::Transformation T_BW(T_CWs.back().translation(),
                                         T_CWs.back().unit_quaternion());
  double keypointAStdDev = vR_oi.front() * 8;
  keypointAStdDev = 0.8 * keypointAStdDev / 12.0;
  double raySigmasA =
      sqrt(sqrt(2)) * keypointAStdDev / cameraGeometry.focalLengthU();

  keypointAStdDev = vR_oi.back() * 8;
  keypointAStdDev = 0.8 * keypointAStdDev / 12.0;
  double raySigmasB =
      sqrt(sqrt(2)) * keypointAStdDev / cameraGeometry.focalLengthU();
  double sigmaR = std::max(raySigmasA, raySigmasB);

  Eigen::Vector3d e1 = (obsDirections.front()).normalized();
  Eigen::Vector3d e2 =
      (T_AW.C() * T_BW.C().transpose() * obsDirections.back()).normalized();
  Eigen::Vector3d e3 = e2;  // we also check the middle point for safety
  okvis::kinematics::Transformation T_MW = T_BW;
  if (obsDirections.size() > 2) {
    auto iter = T_CWs.begin() + obsDirections.size() / 2;
    T_MW = okvis::kinematics::Transformation(iter->translation(),
                                             iter->unit_quaternion());
    auto itObs = obsDirections.begin() + obsDirections.size() / 2;
    e3 = (T_AW.C() * T_MW.C().transpose() * (*itObs)).normalized();
  } else if (obsDirections.size() < 2) {
    // number of observations<=2, happens with triangulating points
    // for visualization
    return false;
  }  // else do nothing

  double e1e2 = e1.dot(e2);
  double e1e3 = e1.dot(e3);
  double e1e22 = e1e2 * e1e2;
  double e1e32 = e1e3 * e1e3;
  // ray parallax test
  const double parallaxThreshold =
      1.0 - 5e-5;  // the value is chosen to be comparable with
                   // okvis::triangulateFast, yet a little easier to declare
                   // parallel
  if (e1e22 > parallaxThreshold && e1e32 > parallaxThreshold) {
    isParallel = true;
    if ((e1.cross(e2)).norm() < 6 * sigmaR &&
        (e1.cross(e3)).norm() < 6 * sigmaR) {
      isValid = true;
    } else {
      OKVIS_ASSERT_TRUE(Exception, isValid, "Why isValid false?");
      triangulateTimer.stop();
      return false;
    }
    v4Xhomog.head<3>() = (e1 + e2 + e3) / 3.0;
    v4Xhomog[3] = 0;
    v4Xhomog = T_AW.inverse() * v4Xhomog;
    triangulateTimer.stop();
    // TODO(jhuai): return true for AIDP?
    return false;  // because Euclidean representation does not support rays
  }

  if (e1e22 < e1e32) {
    Eigen::Vector3d p2 =
        T_AW.r() - (T_AW.q() * T_BW.q().conjugate())._transformVector(T_BW.r());

    v4Xhomog = triangulateFastLocal(
        Eigen::Vector3d(0, 0, 0),  // center of A in A coordinates
        e1, p2,                    // center of B in A coordinates
        e2, sigmaR, isValid, isParallel);

  } else {
    Eigen::Vector3d p3 =
        T_AW.r() - (T_AW.q() * T_MW.q().conjugate())._transformVector(T_MW.r());

    v4Xhomog = triangulateFastLocal(
        Eigen::Vector3d(0, 0, 0),  // center of A in A coordinates
        e1, p3,                    // center of M in A coordinates
        e3, sigmaR, isValid, isParallel);
  }

  OKVIS_ASSERT_FALSE(Exception, isParallel, "a infinity point infiltrate ");

  if (isValid) {
    v4Xhomog /= v4Xhomog[3];
    v4Xhomog = T_AW.inverse() * v4Xhomog;
    Eigen::Vector3d res = v4Xhomog.head<3>();

    triangulate_refine_GN(obsDirections, T_CWs, res, 7);
    v4Xhomog[3] = 1;
    v4Xhomog.head<3>() = res;

    triangulateTimer.stop();
    return true;
  } else {
    //        std::cout<<" invalid triangulation with input to triangulateFast
    //        "<< obsDirections.front().transpose()<<std::endl<<
    //                   (T_AW*T_BW.inverse()).r().transpose()<<std::endl<<
    //                   (T_AW.C()*T_BW.C().transpose()*obsDirections.back()).transpose()
    //                   <<std::endl<< sigmaR<<std::endl;
    //        std::cout <<"my dlt and okvis dlt estiamtes are "<<
    //        v4Xhomog.transpose() <<" "<< v4Xhomog2.transpose()<< std::endl;
    triangulateTimer.stop();
    return false;
  }
#endif  // METHOD 2
}

okvis::Time HybridFilter::firstStateTimestamp() {
  return statesMap_.begin()->second.timestamp;
}
// for debugging with ground truth
void HybridFilter::checkStates() {
  const std::string truthFile = "okvis_estimator_output.csv";
  static vio::CsvReader gg(truthFile);  // ground truth file
  okvis::Time currentTime = statesMap_.rbegin()->second.timestamp;
  static Eigen::Matrix<double, 18, 1> gtCurrState, gtPrevState;
  while (1) {
    gg.getNextObservation();
    gtPrevState = gtCurrState;
    gtCurrState = gg.measurement;
    if (currentTime.sec > gtCurrState[0] ||
        (currentTime.sec == (uint32_t)gtCurrState[0] &&
         currentTime.nsec > (uint32_t)gtCurrState[1]))
      continue;
    else
      break;
  }
  okvis::Time gtCurrTime((uint32_t)gtCurrState[0], (uint32_t)gtCurrState[1]);
  okvis::Time gtPrevTime((uint32_t)gtPrevState[0], (uint32_t)gtPrevState[1]);
  Eigen::Matrix<double, 18, 1> currentState;
  if (gtCurrTime - currentTime > currentTime - gtPrevTime)
    currentState = gtPrevState;
  else
    currentState = gtCurrState;

  if (currentTime.sec != currentState[0] ||
      std::fabs(currentTime.nsec - (uint32_t)currentState[1]) > 500) {
    std::cout << currentTime.sec << " " << currentTime.nsec << " "
              << (uint32_t)currentState[0] << " " << (uint32_t)currentState[1]
              << std::endl;

    OKVIS_ASSERT_EQ(Exception, currentTime.sec, (uint32_t)currentState[0],
                    "Not aligned timestamps for ground truth");
    OKVIS_ASSERT_NEAR(Exception, currentTime.nsec, (uint32_t)currentState[1],
                      500, " Not aligned timestamps for ground truth");
  }
  uint64_t poseId = statesMap_.rbegin()->first;
  kinematics::Transformation T_WS0(
      currentState.segment<3>(2),
      Eigen::Quaterniond(currentState[8], currentState[5], currentState[6],
                         currentState[7]));

  // output deviation from ground truth
  std::shared_ptr<ceres::PoseParameterBlock> poseParamBlockPtr =
      std::static_pointer_cast<ceres::PoseParameterBlock>(
          mapPtr_->parameterBlockPtr(poseId));
  kinematics::Transformation T_WS = poseParamBlockPtr->estimate();
  std::cout << "time " << currentTime << " dev in T_WS " << std::setfill(' ')
            << (T_WS.parameters() - T_WS0.parameters()).transpose()
            << std::endl;
  // update imu sensor states
  const int imuIdx = 0;
  const States stateInQuestion = statesMap_.rbegin()->second;
  uint64_t SBId = stateInQuestion.sensors.at(SensorStates::Imu)
                      .at(imuIdx)
                      .at(ImuSensorStates::SpeedAndBias)
                      .id;
  std::shared_ptr<ceres::SpeedAndBiasParameterBlock> sbParamBlockPtr =
      std::static_pointer_cast<ceres::SpeedAndBiasParameterBlock>(
          mapPtr_->parameterBlockPtr(SBId));
  SpeedAndBiases sb = sbParamBlockPtr->estimate();
  std::cout << "sb dev " << (sb - currentState.segment<9>(9)).transpose()
            << std::endl;

  Eigen::Matrix<double, 27, 1> vTgTsTa = Eigen::Matrix<double, 27, 1>::Zero();
  for (int jack = 0; jack < 3; ++jack) {
    vTgTsTa[jack * 4] += 1;
    vTgTsTa[jack * 4 + 18] += 1;
  }
  Eigen::Matrix<double, 4, 4> matT_SC0;
  matT_SC0 << 0.0148655429818, -0.999880929698, 0.00414029679422,
      -0.0216401454975, 0.999557249008, 0.0149672133247, 0.025715529948,
      -0.064676986768, -0.0257744366974, 0.00375618835797, 0.999660727178,
      0.00981073058949, 0.0, 0.0, 0.0, 1.0;
  Eigen::Matrix<double, 4, 1> projIntrinsic;
  projIntrinsic << 458.654880721, 457.296696463, 367.215803962, 248.37534061;

  Eigen::Matrix<double, 4, 1> distIntrinsic;
  distIntrinsic << -0.28340811217, 0.0739590738929, 0.000193595028569,
      1.76187114545e-05;

  uint64_t TGId = stateInQuestion.sensors.at(SensorStates::Imu)
                      .at(imuIdx)
                      .at(ImuSensorStates::TG)
                      .id;
  std::shared_ptr<ceres::ShapeMatrixParamBlock> tgParamBlockPtr =
      std::static_pointer_cast<ceres::ShapeMatrixParamBlock>(
          mapPtr_->parameterBlockPtr(TGId));
  Eigen::Matrix<double, 9, 1> sm = tgParamBlockPtr->estimate();
  std::cout << "Tg dev " << (sm - vTgTsTa.head<9>()).transpose() << std::endl;

  uint64_t TSId = stateInQuestion.sensors.at(SensorStates::Imu)
                      .at(imuIdx)
                      .at(ImuSensorStates::TS)
                      .id;
  std::shared_ptr<ceres::ShapeMatrixParamBlock> tsParamBlockPtr =
      std::static_pointer_cast<ceres::ShapeMatrixParamBlock>(
          mapPtr_->parameterBlockPtr(TSId));
  sm = tsParamBlockPtr->estimate();
  std::cout << "Ts dev " << (sm - vTgTsTa.segment<9>(9)).transpose()
            << std::endl;

  uint64_t TAId = stateInQuestion.sensors.at(SensorStates::Imu)
                      .at(imuIdx)
                      .at(ImuSensorStates::TA)
                      .id;
  std::shared_ptr<ceres::ShapeMatrixParamBlock> taParamBlockPtr =
      std::static_pointer_cast<ceres::ShapeMatrixParamBlock>(
          mapPtr_->parameterBlockPtr(TAId));
  sm = taParamBlockPtr->estimate();
  std::cout << "Ta dev " << (sm - vTgTsTa.tail<9>()).transpose() << std::endl;

  // update camera sensor states
  const int camIdx = 0;
  uint64_t extrinsicId = stateInQuestion.sensors.at(SensorStates::Camera)
                             .at(camIdx)
                             .at(CameraSensorStates::T_SCi)
                             .id;
  std::shared_ptr<ceres::PoseParameterBlock> extrinsicParamBlockPtr =
      std::static_pointer_cast<ceres::PoseParameterBlock>(
          mapPtr_->parameterBlockPtr(extrinsicId));
  kinematics::Transformation T_SC0 = extrinsicParamBlockPtr->estimate();
  std::cout << "T_SC0 dev "
            << (kinematics::Transformation(matT_SC0).parameters() -
                T_SC0.parameters())
                   .transpose()
            << std::endl;

  uint64_t intrinsicId = stateInQuestion.sensors.at(SensorStates::Camera)
                             .at(camIdx)
                             .at(CameraSensorStates::Intrinsic)
                             .id;
  std::shared_ptr<ceres::CameraIntrinsicParamBlock> intrinsicParamBlockPtr =
      std::static_pointer_cast<ceres::CameraIntrinsicParamBlock>(
          mapPtr_->parameterBlockPtr(intrinsicId));
  Eigen::Matrix<double, 4, 1> cameraIntrinsics =
      intrinsicParamBlockPtr->estimate();
  std::cout << "intrinsic dev "
            << (cameraIntrinsics - projIntrinsic).transpose() << std::endl;

  const int nDistortionCoeffDim =
      okvis::cameras::RadialTangentialDistortion::NumDistortionIntrinsics;
  uint64_t distortionId = stateInQuestion.sensors.at(SensorStates::Camera)
                              .at(camIdx)
                              .at(CameraSensorStates::Distortion)
                              .id;
  std::shared_ptr<ceres::CameraDistortionParamBlock> distortionParamBlockPtr =
      std::static_pointer_cast<ceres::CameraDistortionParamBlock>(
          mapPtr_->parameterBlockPtr(distortionId));
  Eigen::Matrix<double, nDistortionCoeffDim, 1> cameraDistortion =
      distortionParamBlockPtr->estimate();
  std::cout << "distortion dev "
            << (cameraDistortion - distIntrinsic).transpose() << std::endl;

  uint64_t tdId = stateInQuestion.sensors.at(SensorStates::Camera)
                      .at(camIdx)
                      .at(CameraSensorStates::TD)
                      .id;
  std::shared_ptr<ceres::CameraTimeParamBlock> tdParamBlockPtr =
      std::static_pointer_cast<ceres::CameraTimeParamBlock>(
          mapPtr_->parameterBlockPtr(tdId));
  double td = tdParamBlockPtr->estimate();
  std::cout << "t_d dev " << td;

  uint64_t trId = stateInQuestion.sensors.at(SensorStates::Camera)
                      .at(camIdx)
                      .at(CameraSensorStates::TR)
                      .id;
  std::shared_ptr<ceres::CameraTimeParamBlock> trParamBlockPtr =
      std::static_pointer_cast<ceres::CameraTimeParamBlock>(
          mapPtr_->parameterBlockPtr(trId));
  double tr = trParamBlockPtr->estimate();
  std::cout << " t_r dev " << tr << std::endl;

  const bool bApplyGT = false;
  if (bApplyGT) {
    // substitute ground truth for the current estimate
    setGlobalStateEstimateAs<ceres::PoseParameterBlock>(
        poseId, GlobalStates::T_WS, T_WS0);
    setSensorStateEstimateAs<ceres::SpeedAndBiasParameterBlock>(
        poseId, 0, SensorStates::Imu, ImuSensorStates::SpeedAndBias,
        currentState.segment<9>(9));

    setSensorStateEstimateAs<ceres::ShapeMatrixParamBlock>(
        poseId, 0, SensorStates::Imu, ImuSensorStates::TG, vTgTsTa.head<9>());

    setSensorStateEstimateAs<ceres::ShapeMatrixParamBlock>(
        poseId, 0, SensorStates::Imu, ImuSensorStates::TS,
        vTgTsTa.segment<9>(9));

    setSensorStateEstimateAs<ceres::ShapeMatrixParamBlock>(
        poseId, 0, SensorStates::Imu, ImuSensorStates::TA, vTgTsTa.tail<9>());

    setSensorStateEstimateAs<ceres::PoseParameterBlock>(
        poseId, 0, SensorStates::Camera, CameraSensorStates::T_SCi,
        kinematics::Transformation(matT_SC0));
    setSensorStateEstimateAs<ceres::CameraIntrinsicParamBlock>(
        poseId, 0, SensorStates::Camera, CameraSensorStates::Intrinsic,
        projIntrinsic);
    setSensorStateEstimateAs<ceres::CameraDistortionParamBlock>(
        poseId, 0, SensorStates::Camera, CameraSensorStates::Distortion,
        distIntrinsic);
    setSensorStateEstimateAs<ceres::CameraTimeParamBlock>(
        poseId, 0, SensorStates::Camera, CameraSensorStates::TD, 0);
    setSensorStateEstimateAs<ceres::CameraTimeParamBlock>(
        poseId, 0, SensorStates::Camera, CameraSensorStates::TR, 0);
  }
}
// print states and their std
// std::string debugFile; // to store the states and stds
bool HybridFilter::print(const std::string debugFile) const {
  static std::ofstream mDebug;  // the stream corresponding to the debugFile
  if (!mDebug.is_open()) {
    mDebug.open(debugFile, std::ofstream::out);
    mDebug << "%state timestamp, frameIdInSource, T_WS(xyz, xyzw), v_WS, bg, "
              "ba, Tg, Ts, Ta, "
              "p_CB, fx, fy, cx, cy, k1, k2, p1, p2, td, tr, and their stds"
           << std::endl;
  }
  return print(mDebug);
}

bool HybridFilter::print(std::ostream& mDebug) const {
  uint64_t poseId = statesMap_.rbegin()->first;
  std::shared_ptr<ceres::PoseParameterBlock> poseParamBlockPtr =
      std::static_pointer_cast<ceres::PoseParameterBlock>(
          mapPtr_->parameterBlockPtr(poseId));
  kinematics::Transformation T_WS = poseParamBlockPtr->estimate();
  okvis::Time currentTime = statesMap_.rbegin()->second.timestamp;
  assert(multiFramePtrMap_.rbegin()->first == poseId);

  Eigen::IOFormat SpaceInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols,
                               " ", " ", "", "", "", "");
  mDebug << currentTime << " " << multiFramePtrMap_.rbegin()->second->idInSource
         << " " << std::setfill(' ')
         << T_WS.parameters().transpose().format(SpaceInitFmt);
  // update imu sensor states
  const int imuIdx = 0;
  const States stateInQuestion = statesMap_.rbegin()->second;
  uint64_t SBId = stateInQuestion.sensors.at(SensorStates::Imu)
                      .at(imuIdx)
                      .at(ImuSensorStates::SpeedAndBias)
                      .id;
  std::shared_ptr<ceres::SpeedAndBiasParameterBlock> sbParamBlockPtr =
      std::static_pointer_cast<ceres::SpeedAndBiasParameterBlock>(
          mapPtr_->parameterBlockPtr(SBId));
  SpeedAndBiases sb = sbParamBlockPtr->estimate();
  mDebug << " " << sb.transpose().format(SpaceInitFmt);

  uint64_t TGId = stateInQuestion.sensors.at(SensorStates::Imu)
                      .at(imuIdx)
                      .at(ImuSensorStates::TG)
                      .id;
  std::shared_ptr<ceres::ShapeMatrixParamBlock> tgParamBlockPtr =
      std::static_pointer_cast<ceres::ShapeMatrixParamBlock>(
          mapPtr_->parameterBlockPtr(TGId));
  Eigen::Matrix<double, 9, 1> sm = tgParamBlockPtr->estimate();
  mDebug << " " << sm.transpose().format(SpaceInitFmt);

  uint64_t TSId = stateInQuestion.sensors.at(SensorStates::Imu)
                      .at(imuIdx)
                      .at(ImuSensorStates::TS)
                      .id;
  std::shared_ptr<ceres::ShapeMatrixParamBlock> tsParamBlockPtr =
      std::static_pointer_cast<ceres::ShapeMatrixParamBlock>(
          mapPtr_->parameterBlockPtr(TSId));
  sm = tsParamBlockPtr->estimate();
  mDebug << " " << sm.transpose().format(SpaceInitFmt);

  uint64_t TAId = stateInQuestion.sensors.at(SensorStates::Imu)
                      .at(imuIdx)
                      .at(ImuSensorStates::TA)
                      .id;
  std::shared_ptr<ceres::ShapeMatrixParamBlock> taParamBlockPtr =
      std::static_pointer_cast<ceres::ShapeMatrixParamBlock>(
          mapPtr_->parameterBlockPtr(TAId));
  sm = taParamBlockPtr->estimate();
  mDebug << " " << sm.transpose().format(SpaceInitFmt);

  // update camera sensor states
  const int camIdx = 0;
  uint64_t extrinsicId = stateInQuestion.sensors.at(SensorStates::Camera)
                             .at(camIdx)
                             .at(CameraSensorStates::T_SCi)
                             .id;
  std::shared_ptr<ceres::PoseParameterBlock> extrinsicParamBlockPtr =
      std::static_pointer_cast<ceres::PoseParameterBlock>(
          mapPtr_->parameterBlockPtr(extrinsicId));
  kinematics::Transformation T_SC0 = extrinsicParamBlockPtr->estimate();
  mDebug << " " << T_SC0.inverse().r().transpose().format(SpaceInitFmt);

  uint64_t intrinsicId = stateInQuestion.sensors.at(SensorStates::Camera)
                             .at(camIdx)
                             .at(CameraSensorStates::Intrinsic)
                             .id;
  std::shared_ptr<ceres::CameraIntrinsicParamBlock> intrinsicParamBlockPtr =
      std::static_pointer_cast<ceres::CameraIntrinsicParamBlock>(
          mapPtr_->parameterBlockPtr(intrinsicId));
  Eigen::Matrix<double, 4, 1> cameraIntrinsics =
      intrinsicParamBlockPtr->estimate();
  mDebug << " " << cameraIntrinsics.transpose().format(SpaceInitFmt);

  const int nDistortionCoeffDim =
      okvis::cameras::RadialTangentialDistortion::NumDistortionIntrinsics;
  uint64_t distortionId = stateInQuestion.sensors.at(SensorStates::Camera)
                              .at(camIdx)
                              .at(CameraSensorStates::Distortion)
                              .id;
  std::shared_ptr<ceres::CameraDistortionParamBlock> distortionParamBlockPtr =
      std::static_pointer_cast<ceres::CameraDistortionParamBlock>(
          mapPtr_->parameterBlockPtr(distortionId));
  Eigen::Matrix<double, nDistortionCoeffDim, 1> cameraDistortion =
      distortionParamBlockPtr->estimate();
  mDebug << " " << cameraDistortion.transpose().format(SpaceInitFmt);

  uint64_t tdId = stateInQuestion.sensors.at(SensorStates::Camera)
                      .at(camIdx)
                      .at(CameraSensorStates::TD)
                      .id;
  std::shared_ptr<ceres::CameraTimeParamBlock> tdParamBlockPtr =
      std::static_pointer_cast<ceres::CameraTimeParamBlock>(
          mapPtr_->parameterBlockPtr(tdId));
  double td = tdParamBlockPtr->estimate();
  mDebug << " " << td;

  uint64_t trId = stateInQuestion.sensors.at(SensorStates::Camera)
                      .at(camIdx)
                      .at(CameraSensorStates::TR)
                      .id;
  std::shared_ptr<ceres::CameraTimeParamBlock> trParamBlockPtr =
      std::static_pointer_cast<ceres::CameraTimeParamBlock>(
          mapPtr_->parameterBlockPtr(trId));
  double tr = trParamBlockPtr->estimate();
  mDebug << " " << tr;

  // stds
  const int stateDim = 42 + 3 + 4 + 4 + 2;
  Eigen::Matrix<double, stateDim, 1> variances =
      covariance_.topLeftCorner<stateDim, stateDim>().diagonal();

  mDebug << " " << variances.cwiseSqrt().transpose().format(SpaceInitFmt)
         << std::endl;
  return true;
}

// Get frame id in source and whether the state is a keyframe
bool HybridFilter::getFrameId(uint64_t poseId, int& frameIdInSource,
                              bool& isKF) const {
  frameIdInSource = multiFramePtrMap_.at(poseId)->idInSource;
  isKF = statesMap_.at(poseId).isKeyframe;
  return true;
}

// return number of observations for a landmark in the landmark map
size_t HybridFilter::numObservations(uint64_t landmarkId) {
  PointMap::const_iterator it = landmarksMap_.find(landmarkId);
  if (it != landmarksMap_.end())
    return it->second.observations.size();
  else
    return 0;
}

void HybridFilter::printTrackLengthHistogram(std::ostream& mDebug) const {
  mDebug << std::endl
         << "track length histogram in one test with bins 0,1,2..."
         << std::endl;
  size_t bin = 0;
  for (auto it = mTrackLengthAccumulator.begin();
       it != mTrackLengthAccumulator.end(); ++it, ++bin)
    mDebug << bin << " " << *it << std::endl;
}

uint64_t HybridFilter::getCameraCalibrationEstimate(
    Eigen::Matrix<double, 10, 1>& vfckptdr) {
  const int camIdx = 0;
  const uint64_t poseId = statesMap_.rbegin()->first;
  Eigen::Matrix<double, 4 /*cameras::PinholeCamera::NumProjectionIntrinsics*/,
                1>
      intrinsic;
  getSensorStateEstimateAs<ceres::CameraIntrinsicParamBlock>(
      poseId, camIdx, SensorStates::Camera, CameraSensorStates::Intrinsic,
      intrinsic);
  Eigen::Matrix<double,
                cameras::RadialTangentialDistortion::NumDistortionIntrinsics, 1>
      distortionCoeffs;
  getSensorStateEstimateAs<ceres::CameraDistortionParamBlock>(
      poseId, camIdx, SensorStates::Camera, CameraSensorStates::Distortion,
      distortionCoeffs);

  vfckptdr.head<4>() = intrinsic;
  vfckptdr
      .segment<cameras::RadialTangentialDistortion::NumDistortionIntrinsics>(
          4) = distortionCoeffs;
  double tdEstimate(0), trEstimate(0);
  getSensorStateEstimateAs<ceres::CameraTimeParamBlock>(
      poseId, camIdx, SensorStates::Camera, CameraSensorStates::TD, tdEstimate);
  getSensorStateEstimateAs<ceres::CameraTimeParamBlock>(
      poseId, camIdx, SensorStates::Camera, CameraSensorStates::TR, trEstimate);
  vfckptdr.tail<2>() = Eigen::Vector2d(tdEstimate, trEstimate);
  return poseId;
}

uint64_t HybridFilter::getTgTsTaEstimate(
    Eigen::Matrix<double, 27, 1>& vTGTSTA) {
  const uint64_t poseId = statesMap_.rbegin()->first;
  Eigen::Matrix<double, 9, 1> vSM;
  getSensorStateEstimateAs<ceres::ShapeMatrixParamBlock>(
      poseId, 0, SensorStates::Imu, ImuSensorStates::TG, vSM);
  vTGTSTA.head<9>() = vSM;
  getSensorStateEstimateAs<ceres::ShapeMatrixParamBlock>(
      poseId, 0, SensorStates::Imu, ImuSensorStates::TS, vSM);
  vTGTSTA.segment<9>(9) = vSM;
  getSensorStateEstimateAs<ceres::ShapeMatrixParamBlock>(
      poseId, 0, SensorStates::Imu, ImuSensorStates::TA, vSM);
  vTGTSTA.tail<9>() = vSM;
  return poseId;
}

void HybridFilter::getVariance(Eigen::Matrix<double, 55, 1>& variances) {
  variances = covariance_.topLeftCorner<55, 55>().diagonal();
}

}  // namespace okvis
