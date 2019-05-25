#include <okvis/msckf2.hpp>

#include <glog/logging.h>

#include <okvis/IdProvider.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/RelativePoseError.hpp>
#include <okvis/ceres/SpeedAndBiasError.hpp>

#include <okvis/triangulate.h>
#include <okvis/triangulateFast.hpp>

#include <okvis/ceres/CameraDistortionParamBlock.hpp>
#include <okvis/ceres/CameraIntrinsicParamBlock.hpp>
#include <okvis/ceres/CameraTimeParamBlock.hpp>
#include <okvis/ceres/ShapeMatrixParamBlock.hpp>

#include <okvis/IMUOdometry.h>

// the following 4 lines are only for testing
#include <okvis/timing/Timer.hpp>
#include "vio/IMUErrorModel.h"
#include "vio/eigen_utils.h"
#include "vio/rand_sampler.h"

DEFINE_bool(use_AIDP, false,
            "use anchored inverse depth parameterization for a feature point?"
            " Preliminary result shows AIDP worsen the result slightly");

DEFINE_double(max_inc_tol, 2,
              "the tolerance of the lpNorm of an EKF state update");

DECLARE_bool(use_mahalanobis);

/// \brief okvis Main namespace of this package.
namespace okvis {
const double maxProjTolerance =
    7;  // maximum tolerable discrepancy between predicted and measured point
        // coordinates in image in pixel

#undef USE_RK4  // use 4th order runge-kutta for integrating IMU data and
                // compute Jacobians,

#undef USE_IEKF  // use iterated EKF in optimization, empirically IEKF cost at
                 // least twice as much time as EKF,
// and its accuracy in case of inprecise tracked features with real data is
// worse than EKF Constructor if a ceres map is already available.
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
  addStatesTimer.start();
  if (statesMap_.empty()) {
    // in case this is the first frame ever, let's initialize the pose:
    tdEstimate.fromSec(imuParametersVec_.at(0).td0);
    correctedStateTime = multiFrame->timestamp() + tdEstimate;

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

#ifdef USE_FIRST_ESTIMATE
    /// use latest estimate to propagate pose, speed and bias, and first
    /// estimate to propagate covariance and Jacobian
    Eigen::Matrix<double, 6, 1> lP =
        statesMap_.rbegin()->second.linearizationPoint;
    Eigen::Vector3d tempV_WS = speedAndBias.head<3>();
    IMUErrorModel<double> tempIEM(speedAndBias.tail<6>(), vTgTsTa);
    int numUsedImuMeasurements = IMUOdometry::propagation(
        imuMeasurements, imuParametersVec_.at(0), T_WS, tempV_WS, tempIEM,
        startTime, correctedStateTime, &Pkm1, &F_tot, &lP);
    speedAndBias.head<3>() = tempV_WS;
#else
    /// use latest estimate to propagate pose, speed and bias, and covariance
#ifdef USE_RK4
    // method 1 RK4 a little bit more accurate but 4 times slower
    F_tot.setIdentity();
    int numUsedImuMeasurements = IMUOdometry::propagation_RungeKutta(
        imuMeasurements, imuParametersVec_.at(0), T_WS, speedAndBias, vTgTsTa,
        startTime, correctedStateTime, &Pkm1, &F_tot);
#else
    // method 2, i.e., adapt the imuError::propagation function of okvis by the
    // msckf2 derivation in Michael Andrew Shelley
    Eigen::Vector3d tempV_WS = speedAndBias.head<3>();
    IMUErrorModel<double> tempIEM(speedAndBias.tail<6>(), vTgTsTa);
    int numUsedImuMeasurements = IMUOdometry::propagation(
        imuMeasurements, imuParametersVec_.at(0), T_WS, tempV_WS, tempIEM,
        startTime, correctedStateTime, &Pkm1, &F_tot);
    speedAndBias.head<3>() = tempV_WS;
#endif

#endif

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
      const okvis::kinematics::Transformation T_SC = *multiFrame->T_SC(i);
      uint64_t id = IdProvider::instance().newId();
      std::shared_ptr<okvis::ceres::PoseParameterBlock>
          extrinsicsParameterBlockPtr(new okvis::ceres::PoseParameterBlock(
              T_SC, id, correctedStateTime));
      mapPtr_->addParameterBlock(extrinsicsParameterBlockPtr,
                                 ceres::Map::Pose6d);
      cameraInfos.at(CameraSensorStates::T_SCi).id = id;

      Eigen::VectorXd allIntrinsics;
      multiFrame->GetCameraSystem().cameraGeometry(i)->getIntrinsics(
          allIntrinsics);
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
          new okvis::ceres::CameraTimeParamBlock(tdEstimate.toSec(), id,
                                                 correctedStateTime));
      mapPtr_->addParameterBlock(tdParamBlockPtr,
                                 ceres::Map::Parameterization::Trivial);
      cameraInfos.at(CameraSensorStates::TD).id = id;

      id = IdProvider::instance().newId();
      std::shared_ptr<okvis::ceres::CameraTimeParamBlock> trParamBlockPtr(
          new okvis::ceres::CameraTimeParamBlock(imageReadoutTime, id,
                                                 correctedStateTime));
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
  mStateID2Imu[states.id] = imuMeasurements;

  // augment states in the propagated covariance matrix
  size_t covDimAugmented = covDim_ + 9;  //$\delta p,\delta \alpha,\delta v$
  Eigen::MatrixXd covarianceAugmented(covDimAugmented, covDimAugmented);
  covarianceAugmented << covariance_, covariance_.topLeftCorner(covDim_, 9),
      covariance_.topLeftCorner(9, covDim_), covariance_.topLeftCorner<9, 9>();
  covDim_ = covDimAugmented;
  covariance_ = covarianceAugmented;
  addStatesTimer.stop();
  return true;
}

// Applies the dropping/marginalization strategy according to Michael A.
// Shelley's MS thesis

bool MSCKF2::applyMarginalizationStrategy() {
  marginalizeTimer.start();

  // these will keep track of what we want to marginalize out.
  std::vector<uint64_t> paremeterBlocksToBeMarginalized;

  std::vector<uint64_t> removeFrames;
  std::map<uint64_t, States>::reverse_iterator rit = statesMap_.rbegin();
  while (rit != statesMap_.rend()) {
    if (rit->first < minValidStateID) removeFrames.push_back(rit->second.id);
    ++rit;  // check the next frame
  }
  // remove old frames
  for (size_t k = 0; k < removeFrames.size(); ++k) {
    std::map<uint64_t, States>::iterator it = statesMap_.find(removeFrames[k]);

    it->second.global[GlobalStates::T_WS].exists =
        false;  // remember we removed
    it->second.sensors.at(SensorStates::Imu)
        .at(0)
        .at(ImuSensorStates::SpeedAndBias)
        .exists = false;  // remember we removed
    paremeterBlocksToBeMarginalized.push_back(
        it->second.global[GlobalStates::T_WS].id);
    paremeterBlocksToBeMarginalized.push_back(
        it->second.sensors.at(SensorStates::Imu)
            .at(0)
            .at(ImuSensorStates::SpeedAndBias)
            .id);
    mapPtr_->removeParameterBlock(it->second.global[GlobalStates::T_WS].id);
    mapPtr_->removeParameterBlock(it->second.sensors.at(SensorStates::Imu)
                                      .at(0)
                                      .at(ImuSensorStates::SpeedAndBias)
                                      .id);

    mStateID2Imu.erase(it->second.id);
    multiFramePtrMap_.erase(it->second.id);
    statesMap_.erase(it->second.id);
  }
  // remove features tracked no more
  size_t tempCounter = 0;
  for (PointMap::iterator pit = landmarksMap_.begin();
       pit != landmarksMap_.end();) {
    if (mLandmarkID2Residualize[tempCounter].second ==
        NotInState_NotTrackedNow) {
      OKVIS_ASSERT_EQ(Exception, mLandmarkID2Residualize[tempCounter].first,
                      pit->second.id,
                      "mLandmarkID2Residualize has inconsistent landmark ids "
                      "with landmarks map");
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
      ++tempCounter;

    } else {
      ++tempCounter;
      ++pit;
    }
  }
  OKVIS_ASSERT_EQ(Exception, tempCounter, mLandmarkID2Residualize.size(),
                  "Inconsistent index in pruning landmarksMap");

  //    // marginalize everything but pose:
  //    for(size_t k = 0; k<removeAllButPose.size(); ++k){
  //        std::map<uint64_t, States>::iterator it =
  //        statesMap_.find(removeAllButPose[k]); for (size_t i = 0; i <
  //        it->second.global.size(); ++i) {
  //            if (i == GlobalStates::T_WS) {
  //                continue; // we do not remove the pose here.
  //            }
  //            if (!it->second.global[i].exists) {
  //                continue; // if it doesn't exist, we don't do anything.
  //            }
  //            if
  //            (mapPtr_->parameterBlockPtr(it->second.global[i].id)->fixed()) {
  //                continue;  // we never eliminate fixed blocks.
  //            }
  //            std::map<uint64_t, States>::iterator checkit = it;
  //            checkit++;
  //            // only get rid of it, if it's different
  //            if(checkit->second.global[i].exists &&
  //                    checkit->second.global[i].id ==
  //                    it->second.global[i].id){
  //                continue;
  //            }
  //            it->second.global[i].exists = false; // remember we removed
  //            paremeterBlocksToBeMarginalized.push_back(it->second.global[i].id);
  //            keepParameterBlocks.push_back(false);
  //            ceres::Map::ResidualBlockCollection residuals =
  //            mapPtr_->residuals(
  //                        it->second.global[i].id);
  //            for (size_t r = 0; r < residuals.size(); ++r) {
  //                std::shared_ptr<ceres::ReprojectionErrorBase>
  //                reprojectionError =
  //                        std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
  //                            residuals[r].errorInterfacePtr);
  //                if(!reprojectionError){   // we make sure no reprojection
  //                errors are yet included.
  //                    marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
  //                }
  //            }
  //        }
  // add all error terms of the sensor states.
  //        for (size_t i = 0; i < it->second.sensors.size(); ++i) {
  //            for (size_t j = 0; j < it->second.sensors[i].size(); ++j) {
  //                for (size_t k = 0; k < it->second.sensors[i][j].size(); ++k)
  //                {
  //                    if (i == SensorStates::Camera && k ==
  //                    CameraSensorStates::T_SCi) {
  //                        continue; // we do not remove the extrinsics pose
  //                        here.
  //                    }
  //                    if (!it->second.sensors[i][j][k].exists) {
  //                        continue;
  //                    }
  //                    if
  //                    (mapPtr_->parameterBlockPtr(it->second.sensors[i][j][k].id)
  //                            ->fixed()) {
  //                        continue;  // we never eliminate fixed blocks.
  //                    }
  //                    std::map<uint64_t, States>::iterator checkit = it;
  //                    checkit++;
  //                    // only get rid of it, if it's different
  //                    if(checkit->second.sensors[i][j][k].exists &&
  //                            checkit->second.sensors[i][j][k].id ==
  //                            it->second.sensors[i][j][k].id){
  //                        continue;
  //                    }
  //                    it->second.sensors[i][j][k].exists = false; // remember
  //                    we removed
  //                    paremeterBlocksToBeMarginalized.push_back(it->second.sensors[i][j][k].id);
  //                    keepParameterBlocks.push_back(false);
  //                    ceres::Map::ResidualBlockCollection residuals =
  //                    mapPtr_->residuals(
  //                                it->second.sensors[i][j][k].id);
  //                    for (size_t r = 0; r < residuals.size(); ++r) {
  //                        std::shared_ptr<ceres::ReprojectionErrorBase>
  //                        reprojectionError =
  //                                std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
  //                                    residuals[r].errorInterfacePtr);
  //                        if(!reprojectionError){   // we make sure no
  //                        reprojection errors are yet included.
  //                            marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
  //                        }
  //                    }
  //                }
  //            }
  //        }
  //    }
  // marginalize ONLY pose now:
  //    bool reDoFixation = false;
  //    for(size_t k = 0; k<removeFrames.size(); ++k){
  //        std::map<uint64_t, States>::iterator it =
  //        statesMap_.find(removeFrames[k]);

  //        // schedule removal - but always keep the very first frame.
  //        //if(it != statesMap_.begin()){
  //        if(true){ /////DEBUG
  //            it->second.global[GlobalStates::T_WS].exists = false; //
  //            remember we removed
  //            paremeterBlocksToBeMarginalized.push_back(it->second.global[GlobalStates::T_WS].id);
  //            keepParameterBlocks.push_back(false);
  //        }

  // add remaing error terms
  //        ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(
  //                    it->second.global[GlobalStates::T_WS].id);

  //        for (size_t r = 0; r < residuals.size(); ++r) {
  //            if(std::dynamic_pointer_cast<ceres::PoseError>(
  //                        residuals[r].errorInterfacePtr)){ // avoids
  //                        linearising initial pose error
  //                mapPtr_->removeResidualBlock(residuals[r].residualBlockId);
  //                reDoFixation = true;
  //                continue;
  //            }
  //            std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError
  //            =
  //                    std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
  //                        residuals[r].errorInterfacePtr);
  //            if(!reprojectionError){   // we make sure no reprojection errors
  //            are yet included.
  //                marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
  //            }
  //        }

  // add remaining error terms of the sensor states.
  //        size_t i = SensorStates::Camera;
  //        for (size_t j = 0; j < it->second.sensors[i].size(); ++j) {
  //            size_t k = CameraSensorStates::T_SCi;
  //            if (!it->second.sensors[i][j][k].exists) {
  //                continue;
  //            }
  //            if (mapPtr_->parameterBlockPtr(it->second.sensors[i][j][k].id)
  //                    ->fixed()) {
  //                continue;  // we never eliminate fixed blocks.
  //            }
  //            std::map<uint64_t, States>::iterator checkit = it;
  //            checkit++;
  //            // only get rid of it, if it's different
  //            if(checkit->second.sensors[i][j][k].exists &&
  //                    checkit->second.sensors[i][j][k].id ==
  //                    it->second.sensors[i][j][k].id){
  //                continue;
  //            }
  //            it->second.sensors[i][j][k].exists = false; // remember we
  //            removed
  //            paremeterBlocksToBeMarginalized.push_back(it->second.sensors[i][j][k].id);
  //            keepParameterBlocks.push_back(false);
  //            ceres::Map::ResidualBlockCollection residuals =
  //            mapPtr_->residuals(
  //                        it->second.sensors[i][j][k].id);
  //            for (size_t r = 0; r < residuals.size(); ++r) {
  //                std::shared_ptr<ceres::ReprojectionErrorBase>
  //                reprojectionError =
  //                        std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
  //                            residuals[r].errorInterfacePtr);
  //                if(!reprojectionError){   // we make sure no reprojection
  //                errors are yet included.
  //                    marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
  //                }
  //            }
  //        }

  // now finally we treat all the observations.
  //        OKVIS_ASSERT_TRUE_DBG(Exception, allLinearizedFrames.size()>0,
  //        "bug"); uint64_t currentKfId = allLinearizedFrames.at(0);

  //        {
  //            for(PointMap::iterator pit = landmarksMap_.begin();
  //                pit != landmarksMap_.end(); ){

  //                ceres::Map::ResidualBlockCollection residuals =
  //                mapPtr_->residuals(pit->first);

  // first check if we can skip
  //                bool skipLandmark = true;
  //                bool hasNewObservations = false;
  //                bool justDelete = false;
  //                bool marginalize = true;
  //                bool errorTermAdded = false;
  //                std::map<uint64_t,bool> visibleInFrame;
  //                size_t obsCount = 0;
  //                for (size_t r = 0; r < residuals.size(); ++r) {
  //                    std::shared_ptr<ceres::ReprojectionErrorBase>
  //                    reprojectionError =
  //                            std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
  //                                residuals[r].errorInterfacePtr);
  //                    if (reprojectionError) {
  //                        uint64_t poseId =
  //                        mapPtr_->parameters(residuals[r].residualBlockId).at(0).first;
  //                        // since we have implemented the linearisation to
  //                        account for robustification,
  //                        // we don't kick out bad measurements here any more
  //                        like
  //                        // if(vectorContains(allLinearizedFrames,poseId)){
  //                        ...
  //                        //   if (error.transpose() * error > 6.0) { ...
  //                        removeObservation ... }
  //                        // }
  //                        if(vectorContains(removeFrames,poseId)){
  //                            skipLandmark = false;
  //                        }
  //                        if(poseId>=currentKfId){
  //                            marginalize = false;
  //                            hasNewObservations = true;
  //                        }
  //                        if(vectorContains(allLinearizedFrames, poseId)){
  //                            visibleInFrame.insert(std::pair<uint64_t,bool>(poseId,true));
  //                            obsCount++;
  //                        }
  //                    }
  //                }

  //                if(residuals.size()==0){
  //                    mapPtr_->removeParameterBlock(pit->first);
  //                    removedLandmarks.push_back(pit->second);
  //                    pit = landmarksMap_.erase(pit);
  //                    continue;
  //                }

  //                if(skipLandmark) {
  //                    pit++;
  //                    continue;
  //                }

  // so, we need to consider it.
  //                for (size_t r = 0; r < residuals.size(); ++r) {
  //                    std::shared_ptr<ceres::ReprojectionErrorBase>
  //                    reprojectionError =
  //                            std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
  //                                residuals[r].errorInterfacePtr);
  //                    if (reprojectionError) {
  //                        uint64_t poseId =
  //                        mapPtr_->parameters(residuals[r].residualBlockId).at(0).first;
  //                        if((vectorContains(removeFrames,poseId) &&
  //                        hasNewObservations) ||
  //                                (!vectorContains(allLinearizedFrames,poseId)
  //                                && marginalize)){
  //                            // ok, let's ignore the observation.
  //                            removeObservation(residuals[r].residualBlockId);
  //                            residuals.erase(residuals.begin() + r);
  //                            r--;
  //                        } else if(marginalize &&
  //                        vectorContains(allLinearizedFrames,poseId)) {
  //                            // TODO: consider only the sensible ones for
  //                            marginalization if(obsCount<2){
  //                            //visibleInFrame.size()
  //                                removeObservation(residuals[r].residualBlockId);
  //                                residuals.erase(residuals.begin() + r);
  //                                r--;
  //                            } else {
  //                                // add information to be considered in
  //                                marginalization later. errorTermAdded =
  //                                true;
  //                                marginalizationErrorPtr_->addResidualBlock(
  //                                            residuals[r].residualBlockId,
  //                                            false);
  //                            }
  //                        }
  //                        // check anything left
  //                        if (residuals.size() == 0) {
  //                            justDelete = true;
  //                            marginalize = false;
  //                        }
  //                    }
  //                }

  //                if(justDelete){
  //                    mapPtr_->removeParameterBlock(pit->first);
  //                    removedLandmarks.push_back(pit->second);
  //                    pit = landmarksMap_.erase(pit);
  //                    continue;
  //                }
  //                if(marginalize&&errorTermAdded){
  //                    paremeterBlocksToBeMarginalized.push_back(pit->first);
  //                    keepParameterBlocks.push_back(false);
  //                    removedLandmarks.push_back(pit->second);
  //                    pit = landmarksMap_.erase(pit);
  //                    continue;
  //                }

  //                pit++;
  //            }
  //        }

  // update book-keeping and go to the next frame
  // if(it != statesMap_.begin()){ // let's remember that we kept the very first
  // pose
  //        if(true) { ///// DEBUG
  //            multiFramePtrMap_.erase(it->second.id);
  //            statesMap_.erase(it->second.id);
  //        }
  //    }

  // now apply the actual marginalization
  //    if(paremeterBlocksToBeMarginalized.size()>0){
  //        std::vector< ::ceres::ResidualBlockId> addedPriors;
  //        marginalizationErrorPtr_->marginalizeOut(paremeterBlocksToBeMarginalized,
  //        keepParameterBlocks);
  //    }

  // update error computation
  //    if(paremeterBlocksToBeMarginalized.size()>0){
  //        marginalizationErrorPtr_->updateErrorComputation();
  //    }

  // add the marginalization term again
  //    if(marginalizationErrorPtr_->num_residuals()==0){
  //        marginalizationErrorPtr_.reset();
  //    }
  //    if (marginalizationErrorPtr_) {
  //        std::vector<std::shared_ptr<okvis::ceres::ParameterBlock> >
  //        parameterBlockPtrs;
  //        marginalizationErrorPtr_->getParameterBlockPtrs(parameterBlockPtrs);
  //        marginalizationResidualId_ = mapPtr_->addResidualBlock(
  //                    marginalizationErrorPtr_, NULL, parameterBlockPtrs);
  //        OKVIS_ASSERT_TRUE_DBG(Exception, marginalizationResidualId_,
  //                              "could not add marginalization error");
  //        if (!marginalizationResidualId_)
  //            return false;
  //    }

  //    if(reDoFixation){
  //        // finally fix the first pose properly
  //        //mapPtr_->resetParameterization(statesMap_.begin()->first,
  //        ceres::Map::Pose3d); okvis::kinematics::Transformation T_WS_0;
  //        get_T_WS(statesMap_.begin()->first, T_WS_0);
  //        Eigen::Matrix<double,6,6> information =
  //        Eigen::Matrix<double,6,6>::Zero(); information(5,5) = 1.0e14;
  //        information(0,0) = 1.0e14; information(1,1) = 1.0e14;
  //        information(2,2) = 1.0e14; std::shared_ptr<ceres::PoseError >
  //        poseError(new ceres::PoseError(T_WS_0, information));
  //        mapPtr_->addResidualBlock(poseError,NULL,mapPtr_->parameterBlockPtr(statesMap_.begin()->first));
  //    }

  // update covariance matrix
  size_t numRemovedStates = removeFrames.size();
  if (numRemovedStates == 0) {
    marginalizeTimer.stop();
    return true;
  }

  //    std::cout <<"Marginalized covariance and states of Ids";
  //    for (auto iter = removeFrames.begin(); iter!= removeFrames.end();
  //    ++iter)
  //        std::cout <<" "<< *iter;
  //    std::cout << std::endl;

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
  covDim_ = covDim_ - numRemovedStates * 9;
  marginalizeTimer.stop();
  return true;
}

// set latest estimates for the assumed constant states commonly used in
// computing Jacobians of all feature observations
// TODO(jhuai): merge with the super class implementation
void MSCKF2::retrieveEstimatesOfConstants(
    const cameras::NCameraSystem &oldCameraSystem) {
  // X_c and all the augmented states except for the last one because the point
  // has no observation in that frame
  // p_B^C, f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, [k_3], t_d, t_r,
  // \pi_{B_i}(=[p_{B_i}^G, q_{B_i}^G, v_{B_i}^G])
  // also the states of the anchor frame(i.e., before the current frame)
  nVariableDim_ = 9 +
                  cameras::RadialTangentialDistortion::NumDistortionIntrinsics +
                  9 * (statesMap_.size() - 1);

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

  imageHeight_ = oldCameraSystem.cameraGeometry(camIdx)->imageHeight();
  imageWidth_ = oldCameraSystem.cameraGeometry(camIdx)->imageWidth();
  okvis::cameras::RadialTangentialDistortion distortion(
      distortionCoeffs[0], distortionCoeffs[1], distortionCoeffs[2],
      distortionCoeffs[3]);

  intrinsicParameters_ << intrinsic[0], intrinsic[1], intrinsic[2],
      intrinsic[3], distortionCoeffs[0], distortionCoeffs[1],
      distortionCoeffs[2], distortionCoeffs[3];
  tempCameraGeometry_ =
      okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion>(
          imageWidth_, imageHeight_, intrinsic[0], intrinsic[1], intrinsic[2],
          intrinsic[3], distortion);

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

  iem_ = IMUErrorModel<double>(Eigen::Matrix<double, 6, 1>::Zero(), vTGTSTA_);
}

// TODO(jhuai): hpbid homogeneous point block id, used only for debug
// assume the rolling shutter camera reads data row by row, and rows are
//     aligned with the width of a frame
// some heuristics to defend outliers is used, e.g., ignore correspondences of
//     too large discrepancy between prediction and measurement
bool MSCKF2::computeHoi(const uint64_t hpbid, const MapPoint &mp,
                        Eigen::Matrix<double, Eigen::Dynamic, 1> &r_oi,
                        Eigen::MatrixXd &H_oi, Eigen::MatrixXd &R_oi) {
  if (FLAGS_use_AIDP) {  // The landmark is expressed with AIDP in the anchor
                         // frame
    computeHTimer.start();

    // all observations for this feature point
    std::vector<Eigen::Vector2d> obsInPixel;
    // id of frames observing this feature point
    std::vector<uint64_t> frameIds;
    // std noise in pixels
    std::vector<double> vRi;

    Eigen::Vector4d v4Xhomog;  // triangulated point position in the global
                               // frame expressed in [X,Y,Z,W],
    // representing either an ordinary point or a ray, e.g., a point at infinity

    bool bSucceeded =
        triangulateAMapPoint(mp, obsInPixel, frameIds, v4Xhomog, vRi,
                             tempCameraGeometry_, T_SC0_, hpbid, true);

    if (!bSucceeded) {
      computeHTimer.stop();
      return false;
    }

    // the anchor frame is chosen as the last frame observing the point, i.e.,
    // the frame just before the current frame
    uint64_t anchorId = frameIds.back();

    OKVIS_ASSERT_EQ_DBG(
        Exception, anchorId, (++statesMap_.rbegin())->first,
        "anchor id should be the id of the frame preceding current frame");

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

    Eigen::Vector4d ab1rho = v4Xhomog;

    if (ab1rho[2] <= 0) {  // negative depth
      //        std::cout <<"negative depth in ab1rho "<<
      //        ab1rho.transpose()<<std::endl; std::cout << "original v4xhomog
      //        "<< v4Xhomog.transpose()<< std::endl;
      computeHTimer.stop();
      return false;
    }

    ab1rho /= ab1rho[2];  //[\alpha = X/Z, \beta= Y/Z, 1, \rho=1/Z] in the
                          //anchor frame

    Eigen::Vector2d imagePoint;  // projected pixel coordinates of the point
                                 // ${z_u, z_v}$ in pixel units
    Eigen::Matrix2Xd
        intrinsicsJacobian;  //$\frac{\partial [z_u, z_v]^T}{\partial( f_x, f_v,
                             //c_x, c_y, k_1, k_2, p_1, p_2, [k_3])}$
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

    Eigen::Matrix<double, 2, 3>
        J_pfi;  // $\frac{\partial [z_u, z_v]^T}{\partial [\alpha, \beta,
                // \rho]}$
    Eigen::Vector2d J_td;
    Eigen::Vector2d J_tr;
    Eigen::Matrix<double, 2, 9>
        J_XBa;  // $\frac{\partial [z_u, z_v]^T}{\partial delta\p_{B_a}^G)$
    const size_t numCamPoseStates = nVariableDim_;  // camera states, pose
                                                    // states
    Eigen::Matrix<double, 2, Eigen::Dynamic> H_x(
        2, numCamPoseStates);  // Jacobians of a feature w.r.t these states

    ImuMeasurement interpolatedInertialData;

    // containers of the above Jacobians for all observations of a mappoint
    std::vector<Eigen::Matrix<double, 2, Eigen::Dynamic> > vJ_X;
    std::vector<Eigen::Matrix<double, 2, 3> > vJ_pfi;
    std::vector<Eigen::Matrix<double, 2, 1> > vri;  // residuals for feature i

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
      const ImuMeasurementDeque &imuMeas = imuMeasPtr->second;

      Time stateEpoch = statesMap_.at(poseId).timestamp;
      double kpN = obsInPixel[kale][1] / imageHeight_ - 0.5;  // k per N
      const Duration featureTime =
          Duration(tdLatestEstimate + trLatestEstimate * kpN) -
          statesMap_.at(poseId).tdAtCreation;

      // for feature i, estimate $p_B^G(t_{f_i})$, $R_B^G(t_{f_i})$,
      // $v_B^G(t_{f_i})$, and $\omega_{GB}^B(t_{f_i})$ with the corresponding
      // states' LATEST ESTIMATES and imu measurements

      kinematics::Transformation T_WB = T_WBj;
      SpeedAndBiases sb = sbj;
      iem_.resetBgBa(sb.tail<6>());
#ifdef USE_RK4
      if (featureTime >= Duration()) {
        IMUOdometry::propagation_RungeKutta(imuMeas, imuParametersVec_.at(0),
                                            T_WB, sb, vTGTSTA_, stateEpoch,
                                            stateEpoch + featureTime);
      } else {
        IMUOdometry::propagationBackward_RungeKutta(
            imuMeas, imuParametersVec_.at(0), T_WB, sb, vTGTSTA_, stateEpoch,
            stateEpoch + featureTime);
      }
#else
      Eigen::Vector3d tempV_WS = sb.head<3>();
      int numUsedImuMeasurements = -1;
      if (featureTime >= Duration()) {
        numUsedImuMeasurements = IMUOdometry::propagation(
            imuMeas, imuParametersVec_.at(0), T_WB, tempV_WS, iem_, stateEpoch,
            stateEpoch + featureTime);
      } else {
        numUsedImuMeasurements = IMUOdometry::propagationBackward(
            imuMeas, imuParametersVec_.at(0), T_WB, tempV_WS, iem_, stateEpoch,
            stateEpoch + featureTime);
      }
      sb.head<3>() = tempV_WS;
#endif

      IMUOdometry::interpolateInertialData(
          imuMeas, iem_, stateEpoch + featureTime, interpolatedInertialData);

      okvis::kinematics::Transformation T_CA =
          (T_WB * T_SC0_).inverse() *
          T_GA;  // anchor frame to current camera frame
      Eigen::Vector3d pfiinC = (T_CA * ab1rho).head<3>();

      cameras::CameraBase::ProjectionStatus status =
          tempCameraGeometry_.projectWithExternalParameters(
              pfiinC, intrinsicParameters_, &imagePoint, &pointJacobian3,
              &intrinsicsJacobian);
      if (status != cameras::CameraBase::ProjectionStatus::Successful) {
        //            LOG(WARNING) << "Failed to compute Jacobian for distortion
        //            with anchored point : " << ab1rho.transpose() <<
        //                            " and [r,q]_CA" <<
        //                            T_CA.coeffs().transpose();

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

      kinematics::Transformation lP_T_WB = T_WB;
      SpeedAndBiases lP_sb = sb;

#ifdef USE_FIRST_ESTIMATE
      // compute p_WB, v_WB at (t_{f_i,j}) that use FIRST ESTIMATES of position
      // and velocity, i.e., their linearization point
      Eigen::Matrix<double, 6, 1> posVelFirstEstimate =
          statesMap_.at(poseId).linearizationPoint;
      lP_T_WB =
          kinematics::Transformation(posVelFirstEstimate.head<3>(), T_WBj.q());
      lP_sb = sbj;
      lP_sb.head<3>() = posVelFirstEstimate.tail<3>();
      tempV_WS = posVelFirstEstimate.tail<3>();
      numUsedImuMeasurements = -1;
      if (featureTime >= Duration()) {
        numUsedImuMeasurements = IMUOdometry::propagation(
            imuMeas, imuParametersVec_.at(0), lP_T_WB, tempV_WS, iem_,
            stateEpoch, stateEpoch + featureTime);
      } else {
        numUsedImuMeasurements = IMUOdometry::propagationBackward(
            imuMeas, imuParametersVec_.at(0), lP_T_WB, tempV_WS, iem_,
            stateEpoch, stateEpoch + featureTime);
      }
      lP_sb.head<3>() = tempV_WS;
#endif

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
      J_XBj =
          pointJacobian3 * (T_WB.C() * T_SC0_.C()).transpose() * factorJ_XBj;

      factorJ_XBa.topLeftCorner<3, 3>() = rho * Eigen::Matrix3d::Identity();
      factorJ_XBa.block<3, 3>(0, 3) =
          -okvis::kinematics::crossMx(T_WBa.C() * (T_SC0_ * ab1rho).head<3>());
      factorJ_XBa.block<3, 3>(0, 6) = Eigen::Matrix3d::Zero();
      J_XBa =
          pointJacobian3 * (T_WB.C() * T_SC0_.C()).transpose() * factorJ_XBa;

      H_x.setZero();
      H_x.topLeftCorner<2, 9 + okvis::cameras::RadialTangentialDistortion::
                                   NumDistortionIntrinsics>() = J_Xc;
      if (poseId == anchorId) {
        H_x.block<2, 6>(0, 9 +
                               okvis::cameras::RadialTangentialDistortion::
                                   NumDistortionIntrinsics +
                               9 * mStateID2CovID_[poseId] + 3) =
            (J_XBj + J_XBa).block<2, 6>(0, 3);
      } else {
        H_x.block<2, 9>(0, 9 +
                               okvis::cameras::RadialTangentialDistortion::
                                   NumDistortionIntrinsics +
                               9 * mStateID2CovID_[poseId]) = J_XBj;
        H_x.block<2, 9>(0, 9 +
                               okvis::cameras::RadialTangentialDistortion::
                                   NumDistortionIntrinsics +
                               9 * mStateID2CovID_[anchorId]) = J_XBa;
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
    OKVIS_ASSERT_EQ(Exception, numValidObs, frameIds.size(),
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

    Eigen::MatrixXd nullQ = vio::nullspace(H_fi);  // 2nx(2n-3), n==numValidObs
    OKVIS_ASSERT_EQ_DBG(Exception, nullQ.cols(), (int)(2 * numValidObs - 3),
                        "Nullspace of Hfi should have 2n-3 columns");
    //    OKVIS_ASSERT_LT(Exception, (nullQ.transpose()* H_fi).norm(), 1e-6,
    //    "nullspace is not correct!");
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
    computeHTimer.start();

    // gather all observations for this feature point
    std::vector<Eigen::Vector2d> obsInPixel;
    std::vector<uint64_t> frameIds;
    std::vector<double> vRi;  // std noise in pixels
    Eigen::Vector4d v4Xhomog;
    bool bSucceeded =
        triangulateAMapPoint(mp, obsInPixel, frameIds, v4Xhomog, vRi,
                             tempCameraGeometry_, T_SC0_, hpbid, false);
    if (!bSucceeded) {
      computeHTimer.stop();
      return false;
    }
    const Eigen::Vector3d v3Point = v4Xhomog.head<3>();

    // compute Jacobians
    Eigen::Vector2d imagePoint;  // projected pixel coordinates of the point
                                 // ${z_u, z_v}$ in pixel units
    Eigen::Matrix2Xd
        intrinsicsJacobian;  //$\frac{\partial [z_u, z_v]^T}{\partial( f_x, f_v,
                             //c_x, c_y, k_1, k_2, p_1, p_2, [k_3])}$
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
    Eigen::Matrix<double, 2, 3>
        J_pfi;  // $\frac{\partial [z_u, z_v]^T}{\partial p_{f_i}^G}$
    Eigen::Vector2d J_td;
    Eigen::Vector2d J_tr;
    ImuMeasurement interpolatedInertialData;

    // containers of the above Jacobians for all observations of a mappoint
    std::vector<Eigen::Matrix<
        double, 2,
        9 + cameras::RadialTangentialDistortion::NumDistortionIntrinsics> >
        vJ_Xc;
    std::vector<Eigen::Matrix<double, 2, 9> > vJ_XBj;
    std::vector<Eigen::Matrix<double, 2, 3> > vJ_pfi;
    std::vector<Eigen::Matrix<double, 2, 1> > vri;  // residuals for feature i

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
      const ImuMeasurementDeque &imuMeas = imuMeasPtr->second;

      Time stateEpoch = statesMap_.at(poseId).timestamp;
      double kpN = obsInPixel[kale][1] / imageHeight_ - 0.5;  // k per N
      Duration featureTime =
          Duration(tdLatestEstimate + trLatestEstimate * kpN) -
          statesMap_.at(poseId).tdAtCreation;

      // for feature i, estimate $p_B^G(t_{f_i})$, $R_B^G(t_{f_i})$,
      // $v_B^G(t_{f_i})$, and $\omega_{GB}^B(t_{f_i})$ with the corresponding
      // states' LATEST ESTIMATES and imu measurements

      kinematics::Transformation T_WB = T_WBj;
      SpeedAndBiases sb = sbj;
      iem_.resetBgBa(sb.tail<6>());
#ifdef USE_RK4
      if (featureTime >= Duration()) {
        IMUOdometry::propagation_RungeKutta(imuMeas, imuParametersVec_.at(0),
                                            T_WB, sb, vTGTSTA_, stateEpoch,
                                            stateEpoch + featureTime);
      } else {
        IMUOdometry::propagationBackward_RungeKutta(
            imuMeas, imuParametersVec_.at(0), T_WB, sb, vTGTSTA_, stateEpoch,
            stateEpoch + featureTime);
      }
#else
      Eigen::Vector3d tempV_WS = sb.head<3>();
      int numUsedImuMeasurements = -1;
      if (featureTime >= Duration()) {
        numUsedImuMeasurements = IMUOdometry::propagation(
            imuMeas, imuParametersVec_.at(0), T_WB, tempV_WS, iem_, stateEpoch,
            stateEpoch + featureTime);
      } else {
        numUsedImuMeasurements = IMUOdometry::propagationBackward(
            imuMeas, imuParametersVec_.at(0), T_WB, tempV_WS, iem_, stateEpoch,
            stateEpoch + featureTime);
      }
      sb.head<3>() = tempV_WS;
#endif

      IMUOdometry::interpolateInertialData(
          imuMeas, iem_, stateEpoch + featureTime, interpolatedInertialData);

      Eigen::Vector3d pfiinC = ((T_WB * T_SC0_).inverse() * v4Xhomog).head<3>();
      cameras::CameraBase::ProjectionStatus status =
          tempCameraGeometry_.projectWithExternalParameters(
              pfiinC, intrinsicParameters_, &imagePoint, &pointJacobian3,
              &intrinsicsJacobian);
      if (status != cameras::CameraBase::ProjectionStatus::Successful) {
        //            LOG(WARNING) << "Failed to project or to compute Jacobian
        //            for distortion with triangulated point in W: " <<
        //            v4Xhomog.head<3>().transpose() <<
        //                            " and [r,q]_WB" <<
        //                            T_WB.coeffs().transpose();

        itFrameIds = frameIds.erase(itFrameIds);
        itRoi = vRi.erase(itRoi);
        itRoi = vRi.erase(itRoi);
        continue;
      } else if (!FLAGS_use_mahalanobis) {
        // either filter outliers with this simple heuristic in computeHoi or
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

      kinematics::Transformation lP_T_WB = T_WB;
      SpeedAndBiases lP_sb = sb;
#ifdef USE_FIRST_ESTIMATE
      lP_T_WB = T_WBj;
      lP_sb = sbj;
      Eigen::Matrix<double, 6, 1> posVelFirstEstimate =
          statesMap_.at(poseId).linearizationPoint;
      lP_T_WB = kinematics::Transformation(posVelFirstEstimate.head<3>(),
                                           lP_T_WB.q());
      lP_sb.head<3>() = posVelFirstEstimate.tail<3>();

      tempV_WS = lP_sb.head<3>();
      numUsedImuMeasurements = -1;
      if (featureTime >= Duration()) {
        numUsedImuMeasurements = IMUOdometry::propagation(
            imuMeas, imuParametersVec_.at(0), lP_T_WB, tempV_WS, iem_,
            stateEpoch, stateEpoch + featureTime);
      } else {
        numUsedImuMeasurements = IMUOdometry::propagationBackward(
            imuMeas, imuParametersVec_.at(0), lP_T_WB, tempV_WS, iem_,
            stateEpoch, stateEpoch + featureTime);
      }
      lP_sb.head<3>() = tempV_WS;
#endif

      J_td = pointJacobian3 * T_SC0_.C().transpose() *
             (okvis::kinematics::crossMx(
                  (lP_T_WB.inverse() * v4Xhomog).head<3>()) *
                  interpolatedInertialData.measurement.gyroscopes -
              T_WB.C().transpose() * lP_sb.head<3>());
      J_tr = J_td * kpN;
      J_Xc << pointJacobian3, intrinsicsJacobian, J_td, J_tr;

      J_pfi = pointJacobian3 * ((T_WB.C() * T_SC0_.C()).transpose());

      factorJ_XBj << -Eigen::Matrix3d::Identity(),
          okvis::kinematics::crossMx(v3Point - lP_T_WB.r()),
          -Eigen::Matrix3d::Identity() * featureTime.toSec();

      J_XBj = J_pfi * factorJ_XBj;

      vJ_Xc.push_back(J_Xc);
      vJ_XBj.push_back(J_XBj);
      vJ_pfi.push_back(J_pfi);
      ++numValidObs;
      ++itFrameIds;
      itRoi += 2;
    }
    if (numValidObs < 2) {
      computeHTimer.stop();
      return false;
    }
    OKVIS_ASSERT_EQ(Exception, numValidObs, frameIds.size(),
                    "Inconsistent number of observations and frameIds");

    // Now we stack the Jacobians and marginalize the point position related
    // dimensions. In other words, project $H_{x_i}$ onto the nullspace of
    // $H_{f^i}$

    Eigen::MatrixXd H_xi =
        Eigen::MatrixXd::Zero(2 * numValidObs, nVariableDim_);
    Eigen::MatrixXd H_fi = Eigen::MatrixXd(2 * numValidObs, 3);
    Eigen::Matrix<double, Eigen::Dynamic, 1> ri =
        Eigen::Matrix<double, Eigen::Dynamic, 1>(2 * numValidObs, 1);
    Eigen::MatrixXd Ri =
        Eigen::MatrixXd::Identity(2 * numValidObs, 2 * numValidObs);
    for (size_t saga = 0; saga < frameIds.size(); ++saga) {
      size_t saga2 = saga * 2;
      H_xi.block<2, 9 + okvis::cameras::RadialTangentialDistortion::
                            NumDistortionIntrinsics>(saga2, 0) = vJ_Xc[saga];
      H_xi.block<2, 9>(saga2, 9 +
                                  okvis::cameras::RadialTangentialDistortion::
                                      NumDistortionIntrinsics +
                                  9 * mStateID2CovID_[frameIds[saga]]) =
          vJ_XBj[saga];
      H_fi.block<2, 3>(saga2, 0) = vJ_pfi[saga];
      ri.segment<2>(saga2) = vri[saga];
      Ri(saga2, saga2) *= (vRi[saga2] * vRi[saga2]);
      Ri(saga2 + 1, saga2 + 1) *= (vRi[saga2 + 1] * vRi[saga2 + 1]);
    }

    Eigen::MatrixXd nullQ = vio::nullspace(H_fi);  // 2nx(2n-3), n==numValidObs
    OKVIS_ASSERT_EQ(Exception, nullQ.cols(), (int)(2 * numValidObs - 3),
                    "Nullspace of Hfi should have 2n-3 columns");
    //    OKVIS_ASSERT_LT(Exception, (nullQ.transpose()* H_fi).norm(), 1e-6,
    //    "nullspace is not correct!");
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
  OKVIS_ASSERT_EQ_DBG(Exception, deltaX.rows(), (int)covDim_,
                      "Inconsistent size of update to the states");
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
    size_t qStart = 51 + nDistortionCoeffDim + 3 + 9 * jack;

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

#ifdef USE_IEKF
// iterated extended Kalman filter
void MSCKF2::optimize(bool verbose) {
  optimizeTimer.start();
  // containers of Jacobians of measurements
  std::vector<Eigen::MatrixXd> vr_o;
  std::vector<Eigen::MatrixXd> vH_o;
  std::vector<Eigen::MatrixXd> vR_o;
  // gather tracks of features that are not tracked in current frame
  uint64_t currFrameId = currentFrameId();
  const cameras::NCameraSystem oldCameraSystem =
      multiFramePtrMap_.at(currFrameId)
          ->GetCameraSystem();  // only used to get image width and height

  OKVIS_ASSERT_EQ(
      Exception,
      covDim_ - 51 -
          cameras::RadialTangentialDistortion::NumDistortionIntrinsics,
      9 * statesMap_.size(), "Inconsistent covDim and number of states");

  retrieveEstimatesOfConstants(oldCameraSystem);
  size_t dimH_o[2] = {0, nVariableDim_};

  mLandmarkID2Residualize.clear();
  size_t tempCounter = 0;
  Eigen::MatrixXd variableCov = covariance_.block(42, 42, dimH_o[1], dimH_o[1]);

  for (auto it = landmarksMap_.begin(); it != landmarksMap_.end();
       ++it, ++tempCounter) {
    ResidualizeCase toResidualize = NotInState_NotTrackedNow;

    for (auto itObs = it->second.observations.rbegin(),
              iteObs = it->second.observations.rend();
         itObs != iteObs; ++itObs) {
      if (itObs->first.frameId == currFrameId) {
        toResidualize = NotToAdd_TrackedNow;
        break;
      }
    }
    mLandmarkID2Residualize.push_back(
        std::make_pair(it->second.id, toResidualize));
  }

  /// iterations of EKF
  Eigen::Matrix<double, Eigen::Dynamic, 1> deltaX,
      tempDeltaX;  // record the last update step, used to cancel last update in
                   // IEKF
  size_t numIteration = 0;
  const double epsilon = 1e-3;
  Eigen::MatrixXd r_q, T_H, R_q, S, K;

  while (numIteration < 5) {
    if (numIteration) {
      updateStates(-deltaX);  // effectively undo last update in IEKF
      //            std::cout << "after undo update "<< print(std::cout)<<
      //            std::endl;
    }
    size_t nMarginalizedFeatures = 0;
    tempCounter = 0;
    dimH_o[0] = 0;
    vr_o.clear();
    vR_o.clear();
    vH_o.clear();

    for (auto it = landmarksMap_.begin(); it != landmarksMap_.end();
         ++it, ++tempCounter) {
      const size_t nNumObs = it->second.observations.size();
      if (mLandmarkID2Residualize[tempCounter].second !=
              NotInState_NotTrackedNow ||
          nNumObs < 3)  // TODO: is 3 too harsh?
        continue;

      // the rows of H_oi (denoted by nObsDim), and r_oi, and size of R_oi may
      // be resized in computeHoi
      Eigen::MatrixXd H_oi;                           //(nObsDim, dimH_o[1])
      Eigen::Matrix<double, Eigen::Dynamic, 1> r_oi;  //(nObsDim, 1)
      Eigen::MatrixXd R_oi;                           //(nObsDim, nObsDim)
      bool isValidJacobian =
          computeHoi(it->first, it->second, r_oi, H_oi, R_oi);
      if (!isValidJacobian) continue;

      if (FLAGS_use_mahalanobis) {
        // this test looks time consuming as it involves matrix inversion.
        // alternatively, some heuristics in computeHoi is used, e.g., ignore
        // correspondences of too large discrepancy remove outliders, cf. Li
        // ijrr2014 visual inertial navigation with rolling shutter cameras
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
    }

    if (dimH_o[0] == 0) {
      //            LOG(WARNING) << "zero valid support from landmarks#"<<
      //            landmarksMap_.size();

      // update minValidStateID, so that these old frames are removed later
      size_t tempCounter = 0;
      minValidStateID = statesMap_.rbegin()->first;
      for (auto it = landmarksMap_.begin(); it != landmarksMap_.end();
           ++it, ++tempCounter) {
        if (mLandmarkID2Residualize[tempCounter].second ==
            NotInState_NotTrackedNow)
          continue;

        auto itObs = it->second.observations.begin();
        if (itObs->first.frameId <
            minValidStateID)  // this assume that it->second.observations is an
                              // ordered map
          minValidStateID = itObs->first.frameId;
      }

      optimizeTimer.stop();
      return;
    }

    computeKalmanGainTimer.start();
    // stack the marginalized Jacobians and residuals
    Eigen::MatrixXd H_o = Eigen::MatrixXd::Zero(dimH_o[0], nVariableDim_);
    Eigen::MatrixXd r_o(dimH_o[0], 1);
    Eigen::MatrixXd R_o = Eigen::MatrixXd::Zero(dimH_o[0], dimH_o[0]);
    int startRow = 0;

    for (size_t jack = 0; jack < nMarginalizedFeatures; ++jack) {
      H_o.block(startRow, 0, vH_o[jack].rows(), dimH_o[1]) = vH_o[jack];
      r_o.block(startRow, 0, vH_o[jack].rows(), 1) = vr_o[jack];
      R_o.block(startRow, startRow, vH_o[jack].rows(), vH_o[jack].rows()) =
          vR_o[jack];
      startRow += vH_o[jack].rows();
    }

    if (r_o.rows() <= (int)nVariableDim_)  // no need to reduce rows of H_o
    {
      r_q = r_o;
      T_H = H_o;
      R_q = R_o;
    } else {  // project H_o into H_x, reduce the residual dimension
      Eigen::HouseholderQR<Eigen::MatrixXd> qr(H_o);
      Eigen::MatrixXd thinQ(Eigen::MatrixXd::Identity(H_o.rows(), H_o.cols()));
      thinQ = qr.householderQ() * thinQ;

      r_q = thinQ.transpose() * r_o;
      R_q = thinQ.transpose() * R_o * thinQ;
      Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();

      for (size_t row = 0; row < nVariableDim_ /*R.rows()*/; ++row) {
        for (size_t col = 0; col < nVariableDim_ /*R.cols()*/; ++col) {
          if (fabs(R(row, col)) < 1e-10) R(row, col) = 0;
        }
      }
      T_H = R.block(0, 0, nVariableDim_, nVariableDim_);
    }

    // Calculate Kalman gain
    S = T_H *
            covariance_.block(okvis::ceres::ode::OdoErrorStateDim,
                              okvis::ceres::ode::OdoErrorStateDim,
                              nVariableDim_, nVariableDim_) *
            T_H.transpose() +
        R_q;

    K = (covariance_.block(0, okvis::ceres::ode::OdoErrorStateDim, covDim_,
                           nVariableDim_) *
         T_H.transpose()) *
        S.inverse();

    // State correction
    //        std::cout << "before update "<< print(std::cout)<< std::endl;
    if (numIteration) {
      tempDeltaX =
          K * (r_q + T_H * deltaX.segment(okvis::ceres::ode::OdoErrorStateDim,
                                          nVariableDim_));
      if (std::isnan(tempDeltaX(0)) || std::isnan(tempDeltaX(1))) {
        OKVIS_ASSERT_TRUE(Exception, false, "nan in kalman filter");
      }
      computeKalmanGainTimer.stop();
      //            if((tempDeltaX.head<15>().cwiseQuotient(deltaX.head<15>())).lpNorm<Eigen::Infinity>()>2)//this
      //            causes worse result
      //                break;

      updateStates(tempDeltaX);
      if ((deltaX - tempDeltaX).lpNorm<Eigen::Infinity>() < epsilon) break;

      //            double normInf = (deltaX-
      //            tempDeltaX).lpNorm<Eigen::Infinity>(); std::cout <<"iter "<<
      //            numIteration<<" normInf "<<normInf<<" normInf<eps?"<<
      //            (bool)(normInf<epsilon)<<std::endl<<
      //                            (deltaX- tempDeltaX).transpose()<<std::endl;

    } else {
      tempDeltaX = K * r_q;
      if (std::isnan(tempDeltaX(0)) || std::isnan(tempDeltaX(1))) {
        OKVIS_ASSERT_TRUE(Exception, false, "nan in kalman filter");
      }
      computeKalmanGainTimer.stop();

      updateStates(tempDeltaX);
      if (tempDeltaX.lpNorm<Eigen::Infinity>() < epsilon) break;
    }
    // for debugging
    double tempNorm = tempDeltaX.head<15>().lpNorm<Eigen::Infinity>();
    if (tempNorm > FLAGS_max_inc_tol) {
      std::cout << tempDeltaX.transpose() << std::endl;
      OKVIS_ASSERT_LT(Exception, tempNorm, FLAGS_max_inc_tol,
                      "Warn too large increment>2 may imply wrong association");
    }
    // end debugging
    deltaX = tempDeltaX;
    ++numIteration;
  }

  //    std::cout << "IEKF iterations "<<numIteration<<std::endl;
  // Covariance correction
  updateCovarianceTimer.start();

#if 0
    Eigen::MatrixXd tempMat = Eigen::MatrixXd::Identity(covDim_, covDim_);
    tempMat.block(0, okvis::ceres::ode::OdoErrorStateDim, covDim_, nVariableDim_) -= K*T_H;
    covariance_ = tempMat * (covariance_ * tempMat.transpose()).eval() + K * R_q * K.transpose(); //joseph form
#else
  covariance_ = covariance_ - K * S * K.transpose();  // alternative
#endif
  double minDiagVal = covariance_.diagonal().minCoeff();
  if (minDiagVal < 0) {
    std::cout << "Warn: current diagonal has negative " << std::endl
              << covariance_.diagonal().transpose() << std::endl;
    OKVIS_ASSERT_GT(Exception, minDiagVal, 0,
                    "negative covariance diagonal elements less than 0.000");
    covariance_.diagonal() =
        covariance_.diagonal().cwiseAbs();  // TODO: hack is ugly!
  }
  updateCovarianceTimer.stop();
  // update landmarks that are tracked in the current frame(the newly inserted
  // state)
  {
    updateLandmarksTimer.start();
    retrieveEstimatesOfConstants(
        oldCameraSystem);  // do this because states are just updated
    size_t tempCounter = 0;
    minValidStateID = statesMap_.rbegin()->first;
    for (auto it = landmarksMap_.begin(); it != landmarksMap_.end();
         ++it, ++tempCounter) {
      if (mLandmarkID2Residualize[tempCounter].second ==
          NotInState_NotTrackedNow)
        continue;
      // this happens with a just inserted landmark without triangulation.
      if (it->second.observations.size() < 2) continue;

      auto itObs = it->second.observations.begin();
      if (itObs->first.frameId <
          minValidStateID)  // this assume that it->second.observations is an
                            // ordered map
        minValidStateID = itObs->first.frameId;

      // update coordinates of map points, this is only necessary when
      // (1) they are used to predict the points projection in new frames OR
      // (2) to visualize the point quality
      std::vector<Eigen::Vector2d> obsInPixel;
      std::vector<uint64_t> frameIds;
      std::vector<double> vRi;  // std noise in pixels
      Eigen::Vector4d v4Xhomog;
      bool bSucceeded =
          triangulateAMapPoint(it->second, obsInPixel, frameIds, v4Xhomog, vRi,
                               tempCameraGeometry_, T_SC0_, it->first, false);
      if (bSucceeded) {
        it->second.quality = 1.0;
        it->second.pointHomog = v4Xhomog;
      } else {
        it->second.quality = 0.0;
      }
    }
    updateLandmarksTimer.stop();
  }

  // summary output
  if (verbose) {
    LOG(INFO) << mapPtr_->summary.FullReport();
  }
  optimizeTimer.stop();
}
#else
// extended kalman filter
// Start msckf2 optimization.
void MSCKF2::optimize(bool verbose) {
  optimizeTimer.start();
  // containers of Jacobians of measurements
  std::vector<Eigen::MatrixXd> vr_o;
  std::vector<Eigen::MatrixXd> vH_o;
  std::vector<Eigen::MatrixXd> vR_o;
  // gather tracks of features that are not tracked in current frame
  uint64_t currFrameId = currentFrameId();
  const cameras::NCameraSystem oldCameraSystem =
      multiFramePtrMap_.at(currFrameId)
          ->GetCameraSystem();  // only used to get image width and height

  OKVIS_ASSERT_EQ(
      Exception,
      covDim_ - 51 -
          cameras::RadialTangentialDistortion::NumDistortionIntrinsics,
      9 * statesMap_.size(), "Inconsistent covDim and number of states");

  retrieveEstimatesOfConstants(oldCameraSystem);
  size_t dimH_o[2] = {0, nVariableDim_};
  size_t nMarginalizedFeatures = 0;
  mLandmarkID2Residualize.clear();
  size_t tempCounter = 0;
  Eigen::MatrixXd variableCov = covariance_.block(42, 42, dimH_o[1], dimH_o[1]);
  int culledPoints[2] = {0};
  for (auto it = landmarksMap_.begin(); it != landmarksMap_.end();
       ++it, ++tempCounter) {
    ResidualizeCase toResidualize = NotInState_NotTrackedNow;
    const size_t nNumObs = it->second.observations.size();
    for (auto itObs = it->second.observations.rbegin(),
              iteObs = it->second.observations.rend();
         itObs != iteObs; ++itObs) {
      if (itObs->first.frameId == currFrameId) {
        toResidualize = NotToAdd_TrackedNow;
        break;
      }
    }
    mLandmarkID2Residualize.push_back(
        std::make_pair(it->second.id, toResidualize));
    if (toResidualize != NotInState_NotTrackedNow ||
        nNumObs < 3) {  // TODO: is 3 too harsh?
      continue;
    }

    // the rows of H_oi (denoted by nObsDim), and r_oi, and size of R_oi may
    // be resized in computeHoi
    Eigen::MatrixXd H_oi;                           //(nObsDim, dimH_o[1])
    Eigen::Matrix<double, Eigen::Dynamic, 1> r_oi;  //(nObsDim, 1)
    Eigen::MatrixXd R_oi;                           //(nObsDim, nObsDim)
    bool isValidJacobian = computeHoi(it->first, it->second, r_oi, H_oi, R_oi);
    if (!isValidJacobian) {
      ++culledPoints[0];
      continue;
    }

    if (FLAGS_use_mahalanobis) {
      // this test looks time consuming as it involves matrix inversion.
      // alternatively, some heuristics in computeHoi is used, e.g., ignore
      // correspondences of too large discrepancy remove outliders, cf. Li
      // ijrr2014 visual inertial navigation with rolling shutter cameras
      double gamma = r_oi.transpose() *
                     (H_oi * variableCov * H_oi.transpose() + R_oi).inverse() *
                     r_oi;
      if (gamma > chi2_95percentile[r_oi.rows()]) {
        ++culledPoints[1];
        continue;
      }
    }

    vr_o.push_back(r_oi);
    vR_o.push_back(R_oi);
    vH_o.push_back(H_oi);
    dimH_o[0] += r_oi.rows();
    ++nMarginalizedFeatures;
  }
  //    std::cout <<"number of marginalized features for msckf2 "<<
  //    nMarginalizedFeatures<<" cull points "<< culledPoints[0] << " "<<
  //    culledPoints[1]<<std::endl;
  if (dimH_o[0] == 0) {
    //        LOG(WARNING) << "zero valid support from landmarks#"<<
    //        landmarksMap_.size();

    // update minValidStateID, so that these old frames are removed later
    size_t tempCounter = 0;
    minValidStateID = statesMap_.rbegin()->first;
    for (auto it = landmarksMap_.begin(); it != landmarksMap_.end();
         ++it, ++tempCounter) {
      if (mLandmarkID2Residualize[tempCounter].second ==
          NotInState_NotTrackedNow)
        continue;

      auto itObs = it->second.observations.begin();
      if (itObs->first.frameId <
          minValidStateID)  // this assume that it->second.observations is an
                            // ordered map
        minValidStateID = itObs->first.frameId;
    }

    optimizeTimer.stop();
    return;
  }

  computeKalmanGainTimer.start();
  // stack the marginalized Jacobians and residuals
  Eigen::MatrixXd H_o = Eigen::MatrixXd::Zero(dimH_o[0], nVariableDim_);
  Eigen::MatrixXd r_o(dimH_o[0], 1);
  Eigen::MatrixXd R_o = Eigen::MatrixXd::Zero(dimH_o[0], dimH_o[0]);
  int startRow = 0;

  for (size_t jack = 0; jack < nMarginalizedFeatures; ++jack) {
    H_o.block(startRow, 0, vH_o[jack].rows(), dimH_o[1]) = vH_o[jack];
    r_o.block(startRow, 0, vH_o[jack].rows(), 1) = vr_o[jack];
    R_o.block(startRow, startRow, vH_o[jack].rows(), vH_o[jack].rows()) =
        vR_o[jack];
    startRow += vH_o[jack].rows();
  }

  Eigen::MatrixXd r_q, T_H, R_q;

  if (r_o.rows() <= (int)nVariableDim_)  // no need to reduce rows of H_o
  {
    r_q = r_o;
    T_H = H_o;
    R_q = R_o;

    // make T_H compatible with covariance dimension by expanding T_H with zeros
    // corresponding to the rest states besides nVariableDim
    //        Eigen::MatrixXd expandedT_H(dimH_o[0], covDim_);
    //        expandedT_H<< Eigen::MatrixXd::Zero(dimH_o[0], 42), T_H,
    //        Eigen::MatrixXd::Zero(dimH_o[0], 9); T_H= expandedT_H;
  } else {  // project H_o into H_x, reduce the residual dimension
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(H_o);
    Eigen::MatrixXd thinQ(Eigen::MatrixXd::Identity(H_o.rows(), H_o.cols()));
    thinQ = qr.householderQ() * thinQ;

    r_q = thinQ.transpose() * r_o;
    R_q = thinQ.transpose() * R_o * thinQ;
    Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();

    for (size_t row = 0; row < nVariableDim_ /*R.rows()*/; ++row) {
      for (size_t col = 0; col < nVariableDim_ /*R.cols()*/; ++col) {
        if (fabs(R(row, col)) < 1e-10) R(row, col) = 0;
      }
    }
    T_H = R.block(0, 0, nVariableDim_, nVariableDim_);
  }

  // Calculate Kalman gain
  Eigen::MatrixXd S = T_H *
                          covariance_.block(okvis::ceres::ode::OdoErrorStateDim,
                                            okvis::ceres::ode::OdoErrorStateDim,
                                            nVariableDim_, nVariableDim_) *
                          T_H.transpose() +
                      R_q;

  Eigen::MatrixXd K = (covariance_.block(0, okvis::ceres::ode::OdoErrorStateDim,
                                         covDim_, nVariableDim_) *
                       T_H.transpose()) *
                      S.inverse();

  // State correction
  Eigen::Matrix<double, Eigen::Dynamic, 1> deltaX = K * r_q;
  if (std::isnan(deltaX(0)) || std::isnan(deltaX(1))) {
    OKVIS_ASSERT_TRUE(Exception, false, "nan in kalman filter");
  }
  // for debugging
  double tempNorm = deltaX.head<15>().lpNorm<Eigen::Infinity>();
  if (tempNorm > FLAGS_max_inc_tol) {
    std::cout << deltaX.transpose() << std::endl;
    OKVIS_ASSERT_LT(Exception, tempNorm, FLAGS_max_inc_tol,
                    "Warn too large increment>2 may imply wrong association ");
  }
  // end debugging
  computeKalmanGainTimer.stop();

  updateStates(deltaX);

  // Covariance correction
  updateCovarianceTimer.start();

#if 0
    Eigen::MatrixXd tempMat = Eigen::MatrixXd::Identity(covDim_, covDim_);
    tempMat.block(0, okvis::ceres::ode::OdoErrorStateDim, covDim_, nVariableDim_) -= K*T_H;
    covariance_ = tempMat * (covariance_ * tempMat.transpose()).eval() + K * R_q * K.transpose(); //joseph form
#else
  covariance_ = covariance_ - K * S * K.transpose();  // alternative
#endif
  if (covariance_.diagonal().minCoeff() < 0) {
    std::cout << "Warn: current diagonal" << std::endl
              << covariance_.diagonal().transpose() << std::endl;
    covariance_.diagonal() =
        covariance_.diagonal().cwiseAbs();  // TODO: hack is ugly!
    //        OKVIS_ASSERT_GT(Exception, covariance_.diagonal().minCoeff(), 0,
    //        "negative covariance diagonal elements");
  }
  updateCovarianceTimer.stop();
  // update landmarks that are tracked in the current frame(the newly inserted
  // state)
  {
    updateLandmarksTimer.start();
    retrieveEstimatesOfConstants(
        oldCameraSystem);  // do this because states are just updated
    size_t tempCounter = 0;
    minValidStateID = statesMap_.rbegin()->first;
    for (auto it = landmarksMap_.begin(); it != landmarksMap_.end();
         ++it, ++tempCounter) {
      if (mLandmarkID2Residualize[tempCounter].second ==
          NotInState_NotTrackedNow)
        continue;
      // this happens with a just inserted landmark without triangulation.
      if (it->second.observations.size() < 2) continue;

      auto itObs = it->second.observations.begin();
      if (itObs->first.frameId <
          minValidStateID)  // this assume that it->second.observations is an
                            // ordered map
        minValidStateID = itObs->first.frameId;

      // update coordinates of map points, this is only necessary when
      // (1) they are used to predict the points projection in new frames OR
      // (2) to visualize the point quality
      std::vector<Eigen::Vector2d> obsInPixel;
      std::vector<uint64_t> frameIds;
      std::vector<double> vRi;  // std noise in pixels
      Eigen::Vector4d v4Xhomog;
      bool bSucceeded =
          triangulateAMapPoint(it->second, obsInPixel, frameIds, v4Xhomog, vRi,
                               tempCameraGeometry_, T_SC0_, it->first, false);
      if (bSucceeded) {
        it->second.quality = 1.0;
        it->second.pointHomog = v4Xhomog;
      } else {
        it->second.quality = 0.0;
      }
    }
    updateLandmarksTimer.stop();
  }

  // summary output
  if (verbose) {
    LOG(INFO) << mapPtr_->summary.FullReport();
  }
  optimizeTimer.stop();
}
#endif

}  // namespace okvis
