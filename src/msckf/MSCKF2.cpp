#include <msckf/MSCKF2.hpp>

#include <glog/logging.h>

#include <okvis/IdProvider.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/RelativePoseError.hpp>
#include <okvis/ceres/SpeedAndBiasError.hpp>

#include <okvis/ceres/CameraDistortionParamBlock.hpp>
#include <okvis/ceres/CameraIntrinsicParamBlock.hpp>
#include <okvis/ceres/CameraTimeParamBlock.hpp>
#include <okvis/ceres/ShapeMatrixParamBlock.hpp>

#include <msckf/FilterHelper.hpp>
#include <msckf/ImuOdometry.h>
#include <msckf/PreconditionedEkfUpdater.h>
#include <msckf/triangulate.h>
#include <msckf/triangulateFast.hpp>

// the following #include's are only for testing
#include <okvis/timing/Timer.hpp>
#include "vio/ImuErrorModel.h"
#include "vio/Sample.h"

DEFINE_bool(use_AIDP, false,
            "use anchored inverse depth parameterization for a feature point?"
            " Preliminary result shows AIDP worsen the result slightly");

DECLARE_bool(use_mahalanobis);
DECLARE_bool(use_first_estimate);
DECLARE_bool(use_RK4);

/// \brief okvis Main namespace of this package.
namespace okvis {
const double maxProjTolerance =
    7;  // maximum tolerable discrepancy between predicted and measured point
        // coordinates in image in pixel

DEFINE_bool(use_IEKF, false,
            "use iterated EKF in optimization, empirically IEKF cost at"
            "least twice as much time as EKF");

MSCKF2::MSCKF2(std::shared_ptr<okvis::ceres::Map> mapPtr,
               const double readoutTime)
    : HybridFilter(mapPtr, readoutTime) {}

// The default constructor.
MSCKF2::MSCKF2(const double readoutTime) : HybridFilter(readoutTime) {}

MSCKF2::~MSCKF2() {}

// TODO(jhuai): merge with the superclass implementation
bool MSCKF2::addStates(okvis::MultiFramePtr multiFrame,
                       const okvis::ImuMeasurementDeque &imuMeasurements,
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
        // the msckf2 derivation in Michael Andrew Shelley
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
      std::cout << "numUsedImuMeasurements=" << numUsedImuMeasurements
                << " correctedStateTime " << correctedStateTime
                << " lastFrameTimestamp " << startTime << " tdEstimate "
                << tdEstimate << std::endl;
      //            OKVIS_ASSERT_TRUE(Exception, numUsedImuMeasurements > 1,
      //                              "propagation failed");
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
      camera_rig_.addCamera(
          multiFrame->T_SC(i), multiFrame->GetCameraSystem().cameraGeometry(i),
          imageReadoutTime, tdEstimate.toSec(), std::vector<bool>());
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
    // initialize the covariance
    covDim = startIndexOfClonedStates();
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
                 translationVariance;  // note in covariance PBinC is different
                                       // from the state PCinB
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
  mStateID2Imu.push_back(imuMeasurements);

  // augment states in the propagated covariance matrix
  size_t covDimAugmented = covDim + 9;  //$\delta p,\delta \alpha,\delta v$
  Eigen::MatrixXd covarianceAugmented(covDimAugmented, covDimAugmented);
  covarianceAugmented << covariance_, covariance_.topLeftCorner(covDim, 9),
      covariance_.topLeftCorner(9, covDim), covariance_.topLeftCorner<9, 9>();
  covariance_ = covarianceAugmented;
  return true;
}

void MSCKF2::findRedundantCamStates(
    std::vector<uint64_t>* rm_cam_state_ids) {
  // Move the iterator to the key position.
  auto key_cam_state_iter = statesMap_.end();
  for (int i = 0; i < minCulledFrames_ + 2; ++i)
    --key_cam_state_iter;
  auto cam_state_iter = key_cam_state_iter;
  ++cam_state_iter;
  auto first_cam_state_iter = statesMap_.begin();

  // Pose of the key camera state.
  okvis::kinematics::Transformation key_T_WS;
  get_T_WS(key_cam_state_iter->first, key_T_WS);

  const Eigen::Vector3d key_position = key_T_WS.r();
  const Eigen::Matrix3d key_rotation = key_T_WS.C();

  // Mark the camera states to be removed based on the
  // motion between states.
  int closeFrames(0), oldFrames(0);
  for (int i = 0; i < minCulledFrames_; ++i) {
    okvis::kinematics::Transformation T_WS;
    get_T_WS(cam_state_iter->first, T_WS);

    const Eigen::Vector3d position = T_WS.r();
    const Eigen::Matrix3d rotation = T_WS.C();

    double distance = (position-key_position).norm();
    double angle = Eigen::AngleAxisd(
        rotation*key_rotation.transpose()).angle();

    if (angle < rotationThreshold_ &&
        distance < translationThreshold_ &&
        trackingRate_ > trackingRateThreshold_) {
      rm_cam_state_ids->push_back(cam_state_iter->first);
      ++cam_state_iter;
      ++closeFrames;
    } else {
      rm_cam_state_ids->push_back(first_cam_state_iter->first);
      ++first_cam_state_iter;
      ++oldFrames;
    }
  }
  sort(rm_cam_state_ids->begin(), rm_cam_state_ids->end());
  return;
}

int MSCKF2::marginalizeRedundantFrames(size_t maxClonedStates) {
  if (statesMap_.size() < maxClonedStates) {
    return 0;
  }
  std::vector<uint64_t> rm_cam_state_ids;
  findRedundantCamStates(&rm_cam_state_ids);

  size_t nMarginalizedFeatures = 0u;
  int featureVariableDimen = cameraParamsMinimalDimen() +
      clonedStateMinimalDimen() * (statesMap_.size() - 1);
  int startIndexCamParams = startIndexOfCameraParams();
  const Eigen::MatrixXd featureVariableCov =
      covariance_.block(startIndexCamParams, startIndexCamParams,
                        featureVariableDimen, featureVariableDimen);
  int dimH_o[2] = {0, featureVariableDimen};
  // containers of Jacobians of measurements
  std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> vr_o;
  std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> vH_o;
  std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> vR_o;

  // for each map point in the landmarksMap_,
  // see if the landmark is observed in the redundant frames
  for (okvis::PointMap::iterator it = landmarksMap_.begin(); it != landmarksMap_.end();
       ++it) {
    const size_t nNumObs = it->second.observations.size();
    // this feature has been marginalized earlier in optimize()
    if (it->second.residualizeCase ==
            NotInState_NotTrackedNow ||
        nNumObs < minCulledFrames_) {
      continue;
    }

    std::vector<uint64_t> involved_cam_state_ids;
    auto obsMap = it->second.observations;
    for (auto camStateId : rm_cam_state_ids) {
      auto obsIter = std::find_if(obsMap.begin(), obsMap.end(),
                                  IsObservedInFrame(camStateId));
      if (obsIter != obsMap.end()) {
        involved_cam_state_ids.emplace_back(camStateId);
      }
    }
    if (involved_cam_state_ids.size() < minCulledFrames_) {
      continue;
    }

    Eigen::MatrixXd H_oi;                           //(nObsDim, dimH_o[1])
    Eigen::Matrix<double, Eigen::Dynamic, 1> r_oi;  //(nObsDim, 1)
    Eigen::MatrixXd R_oi;                           //(nObsDim, nObsDim)

    bool isValidJacobian =
        featureJacobian(it->second, H_oi, r_oi, R_oi, &involved_cam_state_ids);
    if (!isValidJacobian) {
      continue;
    }

    if (!FilterHelper::gatingTest(H_oi, r_oi, R_oi, featureVariableCov)) {
      continue;
    }

    // TODO(jhuai): check if some observations of a map point
    // has been used for update in order to avoid triangulation again
    // which may harm the filter consistency
    it->second.usedForUpdate = true;
    vr_o.push_back(r_oi);
    vR_o.push_back(R_oi);
    vH_o.push_back(H_oi);
    dimH_o[0] += r_oi.rows();
    ++nMarginalizedFeatures;
  }

  if (nMarginalizedFeatures > 0u) {
    Eigen::MatrixXd H_o =
        Eigen::MatrixXd::Zero(dimH_o[0], featureVariableDimen);
    Eigen::MatrixXd r_o(dimH_o[0], 1);
    Eigen::MatrixXd R_o = Eigen::MatrixXd::Zero(dimH_o[0], dimH_o[0]);
    FilterHelper::stackJacobianAndResidual(vH_o, vr_o, vR_o, &H_o, &r_o, &R_o);
    Eigen::MatrixXd T_H, R_q;
    Eigen::Matrix<double, Eigen::Dynamic, 1> r_q;
    FilterHelper::shrinkResidual(H_o, r_o, R_o, &T_H, &r_q, &R_q);

    // perform filter update covariance and states (EKF)
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

  for (const auto &cam_id : rm_cam_state_ids) {
    int cam_sequence =
        std::distance(statesMap_.begin(), statesMap_.find(cam_id));
    OKVIS_ASSERT_EQ(Exception, cam_sequence, mStateID2CovID_[cam_id], "inconsistent mStateID2CovID_");
  }

  // remove observations in removed frames
  for (okvis::PointMap::iterator it = landmarksMap_.begin(); it != landmarksMap_.end();
       ++it) {
    // this feature has been marginalized earlier in optimize(),
    // will be delete in applyMarginalizationStrategy
    if (it->second.residualizeCase ==
        NotInState_NotTrackedNow) {
      continue;
    }
    std::map<okvis::KeypointIdentifier, uint64_t>& obsMap = it->second.observations;
    for (auto camStateId : rm_cam_state_ids) {
      auto obsIter = std::find_if(obsMap.begin(), obsMap.end(),
                                  IsObservedInFrame(camStateId));
      if (obsIter != obsMap.end()) {
        mapPtr_->removeResidualBlock(
            reinterpret_cast<::ceres::ResidualBlockId>(obsIter->second));
        obsMap.erase(obsIter);
      }
    }
  }

  for (const auto &cam_id : rm_cam_state_ids) {
    int cam_sequence =
        std::distance(statesMap_.begin(), statesMap_.find(cam_id));
    int cam_state_start =
        startIndexOfClonedStates() + clonedStateMinimalDimen() * cam_sequence;
    int cam_state_end = cam_state_start + clonedStateMinimalDimen();

    FilterHelper::pruneSquareMatrix(cam_state_start, cam_state_end,
                                    &covariance_);
    removeState(cam_id);
  }

  mStateID2CovID_.clear();
  int nCovIndex = 0;
  for (auto iter = statesMap_.begin(); iter != statesMap_.end(); ++iter) {
    mStateID2CovID_[iter->first] = nCovIndex;
    ++nCovIndex;
  }
  uint64_t firstStateId = statesMap_.begin()->first;
  minValidStateID = std::min(minValidStateID, firstStateId);
  return rm_cam_state_ids.size();
}

// Applies the dropping/marginalization strategy according to Michael A.
// Shelley's MS thesis
bool MSCKF2::applyMarginalizationStrategy(
    size_t numKeyframes, size_t numImuFrames,
    okvis::MapPointVector& /*removedLandmarks*/) {
  std::vector<uint64_t> removeFrames;
  std::map<uint64_t, States>::reverse_iterator rit = statesMap_.rbegin();
  while (rit != statesMap_.rend()) {
    if (rit->first < minValidStateID) {
      removeFrames.push_back(rit->second.id);
    }
    ++rit;
  }
  if (removeFrames.size() == 0) {
    marginalizeRedundantFrames(numKeyframes + numImuFrames);
  }

  // remove features tracked no more
  for (PointMap::iterator pit = landmarksMap_.begin();
       pit != landmarksMap_.end();) {
    if (pit->second.residualizeCase ==
        NotInState_NotTrackedNow) {

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

// set latest estimates for the assumed constant states commonly used in
// computing Jacobians of all feature observations
// TODO(jhuai): merge with the super class implementation
void MSCKF2::retrieveEstimatesOfConstants() {
  mStateID2CovID_.clear();
  int nCovIndex = 0;

  // note the statesMap_ is an ordered map!
  for (auto iter = statesMap_.begin(); iter != statesMap_.end(); ++iter) {
    mStateID2CovID_[iter->first] = nCovIndex;
    ++nCovIndex;
  }

  const int camIdx = 0;
  getCameraSensorStates(statesMap_.rbegin()->first, camIdx, T_SC0_);

  Eigen::Matrix<double, 4 /*cameras::PinholeCamera::NumProjectionIntrinsics*/,
                1>
      intrinsic;
  getSensorStateEstimateAs<ceres::CameraIntrinsicParamBlock>(
      statesMap_.rbegin()->first, camIdx, SensorStates::Camera,
      CameraSensorStates::Intrinsic, intrinsic);
  OKVIS_ASSERT_EQ(
      Exception, cameras::RadialTangentialDistortion::NumDistortionIntrinsics,
      ceres::nDistortionDim, "radial tangetial parameter size inconsistent");
  Eigen::Matrix<double,
                cameras::RadialTangentialDistortion::NumDistortionIntrinsics, 1>
      distortionCoeffs;
  getSensorStateEstimateAs<ceres::CameraDistortionParamBlock>(
      statesMap_.rbegin()->first, camIdx, SensorStates::Camera,
      CameraSensorStates::Distortion, distortionCoeffs);

  // TODO(jhuai): create cameraGeometry from the intrinsicParameterBlock and
  // distortionParameterBlock, input specified model id, oldCameraGeometry
  // imageHeight Width
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
      statesMap_.rbegin()->first, camIdx, SensorStates::Camera,
      CameraSensorStates::TD, tdLatestEstimate);
  getSensorStateEstimateAs<ceres::CameraTimeParamBlock>(
      statesMap_.rbegin()->first, camIdx, SensorStates::Camera,
      CameraSensorStates::TR, trLatestEstimate);

  Eigen::Matrix<double, 9, 1> vSM;
  getSensorStateEstimateAs<ceres::ShapeMatrixParamBlock>(
      statesMap_.rbegin()->first, 0, SensorStates::Imu, ImuSensorStates::TG,
      vSM);
  vTGTSTA_.head<9>() = vSM;
  getSensorStateEstimateAs<ceres::ShapeMatrixParamBlock>(
      statesMap_.rbegin()->first, 0, SensorStates::Imu, ImuSensorStates::TS,
      vSM);
  vTGTSTA_.segment<9>(9) = vSM;
  getSensorStateEstimateAs<ceres::ShapeMatrixParamBlock>(
      statesMap_.rbegin()->first, 0, SensorStates::Imu, ImuSensorStates::TA,
      vSM);
  vTGTSTA_.tail<9>() = vSM;

  // we do not set bg and ba here because
  // every time iem_ is used, resetBgBa is called
  iem_ = IMUErrorModel<double>(Eigen::Matrix<double, 6, 1>::Zero(), vTGTSTA_);
}

bool MSCKF2::measurementJacobianAIDP(
    const Eigen::Vector4d& ab1rho,
    const std::shared_ptr<okvis::cameras::CameraBase> tempCameraGeometry,
    const Eigen::Vector2d& obs,
    uint64_t poseId, int camIdx,
    uint64_t anchorId, const okvis::kinematics::Transformation& T_WBa,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* H_x,
    Eigen::Matrix<double, 2, 3>* J_pfi, Eigen::Vector2d* residual) const {

  // compute Jacobians for a measurement in image j of the current feature i
  // C_j is the current frame, Bj refers to the body frame associated with the
  // current frame, Ba refers to body frame associated with the anchor frame,
  // f_i is the feature in consideration

  okvis::kinematics::Transformation T_GA =
      T_WBa * T_SC0_;  // anchor frame to global frame

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
  Eigen::Matrix<double, 2, Eigen::Dynamic> J_Xc(2, cameraParamsMinimalDimen());

  Eigen::Matrix<double, 2, 9>
      J_XBj;  // $\frac{\partial [z_u, z_v]^T}{delta\p_{B_j}^G, \delta\alpha
              // (of q_{B_j}^G), \delta v_{B_j}^G$
  Eigen::Matrix<double, 3, 9>
      factorJ_XBj;  // the second factor of J_XBj, see Michael Andrew Shelley
                    // Master thesis sec 6.5, p.55 eq 6.66
  Eigen::Matrix<double, 3, 9> factorJ_XBa;
  Eigen::Vector2d J_td;
  Eigen::Vector2d J_tr;
  Eigen::Matrix<double, 2, 9>
      J_XBa;  // $\frac{\partial [z_u, z_v]^T}{\partial delta\p_{B_a}^G)$

  ImuMeasurement interpolatedInertialData;
  kinematics::Transformation T_WBj;
  get_T_WS(poseId, T_WBj);
  SpeedAndBiases sbj;
  getSpeedAndBias(poseId, 0, sbj);

  Time stateEpoch = statesMap_.at(poseId).timestamp;
  auto imuMeas = mStateID2Imu.findWindow(stateEpoch, half_window_);
  OKVIS_ASSERT_GT(Exception, imuMeas.size(), 0,
                  "the IMU measurement does not exist");

  uint32_t imageHeight = camera_rig_.getCameraGeometry(camIdx)->imageHeight();
  double kpN = obs[1] / imageHeight - 0.5;  // k per N
  const Duration featureTime =
      Duration(tdLatestEstimate + trLatestEstimate * kpN) -
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

    if (featureTime >= Duration()) {
      IMUOdometry::propagation(
          imuMeas, imuParametersVec_.at(0), T_WB, tempV_WS, iem, stateEpoch,
          stateEpoch + featureTime);
    } else {
      IMUOdometry::propagationBackward(
          imuMeas, imuParametersVec_.at(0), T_WB, tempV_WS, iem, stateEpoch,
          stateEpoch + featureTime);
    }
    sb.head<3>() = tempV_WS;
  }

  IMUOdometry::interpolateInertialData(imuMeas, iem, stateEpoch + featureTime,
                                       interpolatedInertialData);

  okvis::kinematics::Transformation T_CA =
      (T_WB * T_SC0_).inverse() * T_GA;  // anchor frame to current camera frame
  Eigen::Vector3d pfiinC = (T_CA * ab1rho).head<3>();

  cameras::CameraBase::ProjectionStatus status = tempCameraGeometry->project(
      pfiinC, &imagePoint, &pointJacobian3, &intrinsicsJacobian);
  *residual = obs - imagePoint;
  if (status != cameras::CameraBase::ProjectionStatus::Successful) {
    return false;
  } else if (!FLAGS_use_mahalanobis) {
    // some heuristics to defend outliers is used, e.g., ignore correspondences
    // of too large discrepancy between prediction and measurement
    if (std::fabs((*residual)[0]) > maxProjTolerance ||
        std::fabs((*residual)[1]) > maxProjTolerance) {
      return false;
    }
  }

  kinematics::Transformation lP_T_WB = T_WB;
  SpeedAndBiases lP_sb = sb;

  if (FLAGS_use_first_estimate) {
    // compute p_WB, v_WB at (t_{f_i,j}) that use FIRST ESTIMATES of
    // position and velocity, i.e., their linearization point
    Eigen::Matrix<double, 6, 1> posVelFirstEstimate =
        statesMap_.at(poseId).linearizationPoint;
    lP_T_WB =
        kinematics::Transformation(posVelFirstEstimate.head<3>(), T_WBj.q());
    lP_sb = sbj;
    lP_sb.head<3>() = posVelFirstEstimate.tail<3>();
    Eigen::Vector3d tempV_WS = posVelFirstEstimate.tail<3>();

    if (featureTime >= Duration()) {
      IMUOdometry::propagation(
          imuMeas, imuParametersVec_.at(0), lP_T_WB, tempV_WS, iem, stateEpoch,
          stateEpoch + featureTime);
    } else {
      IMUOdometry::propagationBackward(
          imuMeas, imuParametersVec_.at(0), lP_T_WB, tempV_WS, iem, stateEpoch,
          stateEpoch + featureTime);
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
  (*J_pfi) = pointJacobian3 * tempM3d;

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

  H_x->setZero();
  H_x->topLeftCorner<2, 9 + okvis::cameras::RadialTangentialDistortion::
                               NumDistortionIntrinsics>() = J_Xc;
  std::map<uint64_t, int>::const_iterator poseid_iter =
      mStateID2CovID_.find(poseId);
  int covid = poseid_iter->second;
  if (poseId == anchorId) {
    H_x->block<2, 6>(0, 9 +
                           okvis::cameras::RadialTangentialDistortion::
                               NumDistortionIntrinsics +
                           9 * covid + 3) =
        (J_XBj + J_XBa).block<2, 6>(0, 3);
  } else {
    H_x->block<2, 9>(0, 9 +
                           okvis::cameras::RadialTangentialDistortion::
                               NumDistortionIntrinsics +
                           9 * covid) = J_XBj;
    std::map<uint64_t, int>::const_iterator anchorid_iter =
        mStateID2CovID_.find(anchorId);
    H_x->block<2, 9>(0, 9 +
                           okvis::cameras::RadialTangentialDistortion::
                               NumDistortionIntrinsics +
                           9 * anchorid_iter->second) = J_XBa;
  }
  return true;
}

bool MSCKF2::measurementJacobian(
    const Eigen::Vector4d& v4Xhomog,
    const std::shared_ptr<okvis::cameras::CameraBase> tempCameraGeometry,
    const Eigen::Vector2d& obs,
    uint64_t poseId,
    int camIdx, Eigen::Matrix<double, 2, Eigen::Dynamic>* J_Xc,
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
      pointJacobian3;  // $\frac{\partial [z_u, z_v]^T}{\partial
                       // p_{f_i}^{C_j}}$

  Eigen::Matrix<double, 3, 9>
      factorJ_XBj;  // the second factor of J_XBj, see Michael Andrew Shelley
                    // Master thesis sec 6.5, p.55 eq 6.66

  Eigen::Vector2d J_td;
  Eigen::Vector2d J_tr;
  ImuMeasurement interpolatedInertialData;

  kinematics::Transformation T_WBj;
  get_T_WS(poseId, T_WBj);
  SpeedAndBiases sbj;
  getSpeedAndBias(poseId, 0, sbj);

  Time stateEpoch = statesMap_.at(poseId).timestamp;
  auto imuMeas = mStateID2Imu.findWindow(stateEpoch, half_window_);
  OKVIS_ASSERT_GT(Exception, imuMeas.size(), 0,
                  "the IMU measurement does not exist");

  uint32_t imageHeight = camera_rig_.getCameraGeometry(camIdx)->imageHeight();
  double kpN = obs[1] / imageHeight - 0.5;  // k per N
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

    if (featureTime >= Duration()) {
      IMUOdometry::propagation(
          imuMeas, imuParametersVec_.at(0), T_WB, tempV_WS, iem, stateEpoch,
          stateEpoch + featureTime);
    } else {
      IMUOdometry::propagationBackward(
          imuMeas, imuParametersVec_.at(0), T_WB, tempV_WS, iem, stateEpoch,
          stateEpoch + featureTime);
    }
    sb.head<3>() = tempV_WS;
  }

  IMUOdometry::interpolateInertialData(imuMeas, iem, stateEpoch + featureTime,
                                       interpolatedInertialData);

  Eigen::Vector3d pfiinC = ((T_WB * T_SC0_).inverse() * v4Xhomog).head<3>();
  cameras::CameraBase::ProjectionStatus status = tempCameraGeometry->project(
      pfiinC, &imagePoint, &pointJacobian3, &intrinsicsJacobian);
  *residual = obs - imagePoint;
  if (status != cameras::CameraBase::ProjectionStatus::Successful) {
    return false;
  } else if (!FLAGS_use_mahalanobis) {
    if (std::fabs((*residual)[0]) > maxProjTolerance ||
        std::fabs((*residual)[1]) > maxProjTolerance) {
      return false;
    }
  }

  kinematics::Transformation lP_T_WB = T_WB;
  SpeedAndBiases lP_sb = sb;
  if (FLAGS_use_first_estimate) {
    lP_T_WB = T_WBj;
    lP_sb = sbj;
    Eigen::Matrix<double, 6, 1> posVelFirstEstimate =
        statesMap_.at(poseId).linearizationPoint;
    lP_T_WB =
        kinematics::Transformation(posVelFirstEstimate.head<3>(), lP_T_WB.q());
    lP_sb.head<3>() = posVelFirstEstimate.tail<3>();

    Eigen::Vector3d tempV_WS = lP_sb.head<3>();
    if (featureTime >= Duration()) {
      IMUOdometry::propagation(
          imuMeas, imuParametersVec_.at(0), lP_T_WB, tempV_WS, iem, stateEpoch,
          stateEpoch + featureTime);
    } else {
      IMUOdometry::propagationBackward(
          imuMeas, imuParametersVec_.at(0), lP_T_WB, tempV_WS, iem, stateEpoch,
          stateEpoch + featureTime);
    }
    lP_sb.head<3>() = tempV_WS;
  }

  J_td = pointJacobian3 * T_SC0_.C().transpose() *
         (okvis::kinematics::crossMx((lP_T_WB.inverse() * v4Xhomog).head<3>()) *
              interpolatedInertialData.measurement.gyroscopes -
          T_WB.C().transpose() * lP_sb.head<3>());
  J_tr = J_td * kpN;
  (*J_Xc) << pointJacobian3, intrinsicsJacobian, J_td, J_tr;

  (*J_pfi) = pointJacobian3 * ((T_WB.C() * T_SC0_.C()).transpose());

  factorJ_XBj << -Eigen::Matrix3d::Identity(),
      okvis::kinematics::crossMx(v3Point - lP_T_WB.r()),
      -Eigen::Matrix3d::Identity() * featureTime.toSec();

  (*J_XBj) = (*J_pfi) * factorJ_XBj;
  return true;
}

// assume the rolling shutter camera reads data row by row, and rows are
//     aligned with the width of a frame
bool MSCKF2::featureJacobian(const MapPoint &mp, Eigen::MatrixXd &H_oi,
                        Eigen::Matrix<double, Eigen::Dynamic, 1> &r_oi,
                        Eigen::MatrixXd &R_oi,
                        const std::vector<uint64_t>* involvedFrameIds) const {
  const int camIdx = 0;
  std::shared_ptr<okvis::cameras::CameraBase> tempCameraGeometry =
      camera_rig_.getCameraGeometry(camIdx);

  // dimension of variables used in computing feature Jacobians, including
  // camera intrinsics and all cloned states except the most recent one
  // in which the marginalized observations should never occur.
  int featureVariableDimen = cameraParamsMinimalDimen() +
      clonedStateMinimalDimen() * (statesMap_.size() - 1);

  // all observations for this feature point
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      obsInPixel;
  std::vector<uint64_t> frameIds; // id of frames observing this feature point
  std::vector<double> vRi; // std noise in pixels
  Eigen::Vector4d v4Xhomog; // triangulated point position in the global
  // frame expressed in [X,Y,Z,W],
  computeHTimer.start();
  if (FLAGS_use_AIDP) {
    // The landmark is expressed with AIDP in the anchor frame    
    // if the feature is lost in current frame, the anchor frame is chosen
    // as the last frame observing the point.

    uint64_t anchorId = mp.observations.rbegin()->first.frameId;
    int anchorSeqId = mp.observations.size() - 1;
    if (involvedFrameIds != nullptr) {
      anchorId = involvedFrameIds->back();
      const std::map<okvis::KeypointIdentifier, uint64_t>& obsMap = mp.observations;
      auto anchorIter = std::find_if(obsMap.begin(), obsMap.end(), IsObservedInFrame(anchorId));
      anchorSeqId = std::distance(obsMap.begin(), anchorIter);
    }
    bool bSucceeded =
        triangulateAMapPoint(mp, obsInPixel, frameIds, v4Xhomog, vRi,
                             tempCameraGeometry, T_SC0_, anchorSeqId);

    if (!bSucceeded) {
      computeHTimer.stop();
      return false;
    }

    Eigen::Vector4d ab1rho = v4Xhomog;
    if (ab1rho[2] <= 0) {  // negative depth
      computeHTimer.stop();
      return false;
    }
    ab1rho /= ab1rho[2];  //[\alpha = X/Z, \beta= Y/Z, 1, \rho=1/Z] in the
                          // anchor frame
    // transform from the body frame at the anchor frame epoch to the world frame
    kinematics::Transformation T_WBa;
    get_T_WS(anchorId, T_WBa);

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
      Eigen::Matrix<double, 2, Eigen::Dynamic> H_x(2, featureVariableDimen);
      // $\frac{\partial [z_u, z_v]^T}{\partial [\alpha, \beta, \rho]}$
      Eigen::Matrix<double, 2, 3> J_pfi;
      Eigen::Vector2d residual;

      uint64_t poseId = *itFrameIds;
      const int camIdx = 0;

      if (involvedFrameIds != nullptr &&
          std::find(involvedFrameIds->begin(), involvedFrameIds->end(), poseId)
              == involvedFrameIds->end()) {
          itFrameIds = frameIds.erase(itFrameIds);
          itRoi = vRi.erase(itRoi);
          itRoi = vRi.erase(itRoi);
          continue;
      }

      bool validJacobian = measurementJacobianAIDP(
          ab1rho, tempCameraGeometry, obsInPixel[kale], poseId, camIdx, anchorId, T_WBa,
          &H_x, &J_pfi, &residual);
      if (!validJacobian) {
          itFrameIds = frameIds.erase(itFrameIds);
          itRoi = vRi.erase(itRoi);
          itRoi = vRi.erase(itRoi);
          continue;
      }

      vri.push_back(residual);
      vJ_X.push_back(H_x);
      vJ_pfi.push_back(J_pfi);

      ++numValidObs;
      ++itFrameIds;
      itRoi += 2;
    }
    if (numValidObs < minTrackLength_) {
      computeHTimer.stop();
      return false;
    }

    // Now we stack the Jacobians and marginalize the point position related
    // dimensions. In other words, project $H_{x_i}$ onto the nullspace of
    // $H_{f^i}$

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
      Ri(saga2, saga2) = vRi[saga2] * vRi[saga2];
      Ri(saga2 + 1, saga2 + 1) = vRi[saga2 + 1] * vRi[saga2 + 1];
    }

    Eigen::MatrixXd nullQ = vio::nullspace(H_fi);  // 2nx(2n-3), n==numValidObs

    r_oi.noalias() = nullQ.transpose() * ri;
    H_oi.noalias() = nullQ.transpose() * H_xi;
    R_oi = nullQ.transpose() * (Ri * nullQ).eval();

    vri.clear();
    vJ_pfi.clear();
    vJ_X.clear();
    frameIds.clear();
    computeHTimer.stop();
    return true;
  } else {
    // The landmark is expressed with Euclidean coordinates in the global frame
    bool bSucceeded =
        triangulateAMapPoint(mp, obsInPixel, frameIds, v4Xhomog, vRi,
                             tempCameraGeometry, T_SC0_);
    if (!bSucceeded) {
      computeHTimer.stop();
      return false;
    }

    // containers of the above Jacobians for all observations of a mappoint
    const int cameraParamsDimen = cameraParamsMinimalDimen();
    std::vector<
        Eigen::Matrix<double, 2, Eigen::Dynamic>,
        Eigen::aligned_allocator<Eigen::Matrix<double, 2, Eigen::Dynamic>>>
        vJ_Xc;
    std::vector<Eigen::Matrix<double, 2, 9>,
                Eigen::aligned_allocator<Eigen::Matrix<double, 2, 9>>>
        vJ_XBj;
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
      const int camIdx = 0;
      // $\frac{\partial [z_u, z_v]^T}{p_B^C, fx, fy, cx, cy, k1, k2, p1, p2,
      // [k3], t_d, t_r}$
      Eigen::Matrix<double, 2, Eigen::Dynamic> J_Xc(2, cameraParamsMinimalDimen());
      Eigen::Matrix<double, 2, 9>
          J_XBj;  // $\frac{\partial [z_u, z_v]^T}{delta\p_{B_j}^G, \delta\alpha
                  // (of q_{B_j}^G), \delta v_{B_j}^G$
      Eigen::Matrix<double, 2, 3>
          J_pfi;  // $\frac{\partial [z_u, z_v]^T}{\partial p_{f_i}^G}$
      Eigen::Vector2d residual;

      if (involvedFrameIds != nullptr &&
          std::find(involvedFrameIds->begin(), involvedFrameIds->end(), poseId)
              == involvedFrameIds->end()) {
          itFrameIds = frameIds.erase(itFrameIds);
          itRoi = vRi.erase(itRoi);
          itRoi = vRi.erase(itRoi);
          continue;
      }
      bool validJacobian = measurementJacobian(
          v4Xhomog, tempCameraGeometry, obsInPixel[kale], poseId, camIdx,
          &J_Xc, &J_XBj, &J_pfi, &residual);
      if (!validJacobian) {
          itFrameIds = frameIds.erase(itFrameIds);
          itRoi = vRi.erase(itRoi);
          itRoi = vRi.erase(itRoi);
          continue;
      }

      vJ_Xc.push_back(J_Xc);
      vJ_XBj.push_back(J_XBj);
      vJ_pfi.push_back(J_pfi);
      vri.push_back(residual);
      ++numValidObs;
      ++itFrameIds;
      itRoi += 2;
    }
    if (numValidObs < minTrackLength_) {
      computeHTimer.stop();
      return false;
    }

    // Now we stack the Jacobians and marginalize the point position related
    // dimensions. In other words, project $H_{x_i}$ onto the nullspace of
    // $H_{f^i}$

    Eigen::MatrixXd H_xi =
        Eigen::MatrixXd::Zero(2 * numValidObs, featureVariableDimen);
    Eigen::MatrixXd H_fi = Eigen::MatrixXd(2 * numValidObs, 3);
    Eigen::Matrix<double, Eigen::Dynamic, 1> ri =
        Eigen::Matrix<double, Eigen::Dynamic, 1>(2 * numValidObs, 1);
    Eigen::MatrixXd Ri =
        Eigen::MatrixXd::Identity(2 * numValidObs, 2 * numValidObs);
    for (size_t saga = 0; saga < numValidObs; ++saga) {
      size_t saga2 = saga * 2;
      H_xi.block(saga2, 0, 2, cameraParamsDimen) = vJ_Xc[saga];
      std::map<uint64_t, int>::const_iterator frmid_iter =
          mStateID2CovID_.find(frameIds[saga]);
      H_xi.block<2, 9>(saga2, cameraParamsDimen + 9 * frmid_iter->second) = vJ_XBj[saga];
      H_fi.block<2, 3>(saga2, 0) = vJ_pfi[saga];
      ri.segment<2>(saga2) = vri[saga];
      Ri(saga2, saga2) *= (vRi[saga2] * vRi[saga2]);
      Ri(saga2 + 1, saga2 + 1) *= (vRi[saga2 + 1] * vRi[saga2 + 1]);
    }

    Eigen::MatrixXd nullQ = vio::nullspace(H_fi);  // 2nx(2n-3), n==numValidObs

    r_oi.noalias() = nullQ.transpose() * ri;
    H_oi.noalias() = nullQ.transpose() * H_xi;
    R_oi = nullQ.transpose() * (Ri * nullQ).eval();

    vri.clear();
    vJ_pfi.clear();
    vJ_XBj.clear();
    vJ_Xc.clear();
    frameIds.clear();
    computeHTimer.stop();
    return true;
  }
}

// TODO(jhuai): merge with the superclass implementation
void MSCKF2::updateStates(
    const Eigen::Matrix<double, Eigen::Dynamic, 1> &deltaX) {
  updateStatesTimer.start();

  OKVIS_ASSERT_NEAR(
      Exception,
      (deltaX.head<9>() - deltaX.tail<9>()).lpNorm<Eigen::Infinity>(), 0, 1e-8,
      "Correction to the current states from head and tail should be "
      "identical");

  std::map<uint64_t, States>::reverse_iterator lastElementIterator =
      statesMap_.rbegin();
  const States &stateInQuestion = lastElementIterator->second;
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

  // Update augmented states except for the last one which is the current state
  // already updated this section assumes that the statesMap is an ordered map
  size_t jack = 0;
  auto finalIter = statesMap_.end();
  --finalIter;

  for (auto iter = statesMap_.begin(); iter != finalIter; ++iter, ++jack) {
    stateId = iter->first;
    size_t qStart = startIndexOfClonedStates() + 3 + clonedStateMinimalDimen() * jack;

    poseParamBlockPtr = std::static_pointer_cast<ceres::PoseParameterBlock>(
        mapPtr_->parameterBlockPtr(stateId));
    T_WS = poseParamBlockPtr->estimate();
    deltaAlpha = deltaX.segment<3>(qStart);
    deltaq =
        vio::quaternionFromSmallAngle(deltaAlpha);  // rvec2quat(deltaAlpha);
    T_WS = kinematics::Transformation(
        T_WS.r() + deltaX.segment<3>(qStart - 3),
        deltaq *
            T_WS.q());  // in effect this amounts to PoseParameterBlock::plus()
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
  updateStatesTimer.stop();
}

int MSCKF2::computeStackedJacobianAndResidual(
    Eigen::MatrixXd *T_H, Eigen::Matrix<double, Eigen::Dynamic, 1> *r_q,
    Eigen::MatrixXd *R_q) const {
  // compute and stack Jacobians and Residuals for landmarks observed no more
  size_t nMarginalizedFeatures = 0;
  int culledPoints[2] = {0};
  int featureVariableDimen = cameraParamsMinimalDimen() +
      clonedStateMinimalDimen() * (statesMap_.size() - 1);
  int dimH_o[2] = {0, featureVariableDimen};
  const Eigen::MatrixXd variableCov =
      covariance_.block(42, 42, dimH_o[1], dimH_o[1]);
  // containers of Jacobians of measurements
  std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> vr_o;
  std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> vH_o;
  std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> vR_o;

  for (auto it = landmarksMap_.begin(); it != landmarksMap_.end();
       ++it) {
    const size_t nNumObs = it->second.observations.size();
    if (it->second.residualizeCase !=
            NotInState_NotTrackedNow ||
        nNumObs < minTrackLength_) {
      continue;
    }

    Eigen::MatrixXd H_oi;                           //(nObsDim, dimH_o[1])
    Eigen::Matrix<double, Eigen::Dynamic, 1> r_oi;  //(nObsDim, 1)
    Eigen::MatrixXd R_oi;                           //(nObsDim, nObsDim)
    bool isValidJacobian = featureJacobian(it->second, H_oi, r_oi, R_oi);
    if (!isValidJacobian) {
      ++culledPoints[0];
      continue;
    }

    if (!FilterHelper::gatingTest(H_oi, r_oi, R_oi, variableCov)) {
        ++culledPoints[1];
        continue;
    }

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
  Eigen::MatrixXd r_o(dimH_o[0], 1);
  Eigen::MatrixXd R_o = Eigen::MatrixXd::Zero(dimH_o[0], dimH_o[0]);
  FilterHelper::stackJacobianAndResidual(vH_o, vr_o, vR_o, &H_o, &r_o, &R_o);
  FilterHelper::shrinkResidual(H_o, r_o, R_o, T_H, r_q, R_q);
  return dimH_o[0];
}

uint64_t MSCKF2::getMinValidStateID() const {
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
  return min_state_id;
}

void MSCKF2::optimize(size_t /*numIter*/, size_t /*numThreads*/, bool verbose) {
  uint64_t currFrameId = currentFrameId();
  OKVIS_ASSERT_EQ(
      Exception,
      covariance_.rows() - startIndexOfClonedStates(),
      (int)(9 * statesMap_.size()), "Inconsistent covDim and number of states");
  retrieveEstimatesOfConstants();

  // mark tracks of features that are not tracked in current frame
  int numTracked = 0;
  int featureVariableDimen = cameraParamsMinimalDimen() +
      clonedStateMinimalDimen() * (statesMap_.size() - 1);

  for (okvis::PointMap::iterator it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
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

        //            double normInf = (deltaX-
        //            tempDeltaX).lpNorm<Eigen::Infinity>(); std::cout <<"iter
        //            "<< numIteration<<" normInf "<<normInf<<" normInf<eps?"<<
        //            (bool)(normInf<epsilon)<<std::endl<<
        //                            (deltaX-
        //                            tempDeltaX).transpose()<<std::endl;

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
    for (auto it = landmarksMap_.begin(); it != landmarksMap_.end();
         ++it) {
      if (it->second.residualizeCase ==
          NotInState_NotTrackedNow)
        continue;
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
      if (tempCameraGeometry == nullptr) {
        OKVIS_ASSERT_TRUE(
            Exception, tempCameraGeometry->distortionType() == "RadialTangentialDistortion",
            "Camera RadialTangentialDistortion is expected, actual is " +
                camera_rig_.getCameraGeometry(camIdx)->distortionType());
      }
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
