#include <iostream>
#include "gtest/gtest.h"

#include <msckf/MSCKF2.hpp>
#include <okvis/IdProvider.hpp>

std::shared_ptr<okvis::MultiFrame> createMultiFrame(
    okvis::Time timestamp,
    std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem) {
  // assemble a multi-frame
  std::shared_ptr<okvis::MultiFrame> mf(new okvis::MultiFrame);
  mf->setId(okvis::IdProvider::instance().newId());
  mf->setTimestamp(timestamp);
  mf->resetCameraSystemAndFrames(*cameraSystem);
  return mf;
}
double angular_rate = 0.3;  // rad/sec
const double radius = 1.5;
okvis::ImuMeasurementDeque createImuMeasurements(okvis::Time t0) {
  int DURATION = 10;
  int IMU_RATE = 200;
  double DT = 1.0 / IMU_RATE;

  double g = 9.8;
  okvis::ImuMeasurementDeque imuMeasurements;
  okvis::ImuSensorReadings nominalImuSensorReadings(
      Eigen::Vector3d(0, 0, angular_rate),
      Eigen::Vector3d(-radius * angular_rate * angular_rate, 0, g));
  for (int i = -2; i <= DURATION * IMU_RATE + 2; ++i) {
    Eigen::Vector3d gyr = nominalImuSensorReadings.gyroscopes;
    Eigen::Vector3d acc = nominalImuSensorReadings.accelerometers;
    imuMeasurements.push_back(okvis::ImuMeasurement(
        t0 + okvis::Duration(DT * i), okvis::ImuSensorReadings(gyr, acc)));
  }
  return imuMeasurements;
}

TEST(MSCKF2, MeasurementJacobian) {
  double trNoisy = 0.033;
  std::shared_ptr<okvis::ceres::Map> mapPtr(new okvis::ceres::Map);
  okvis::MSCKF2 estimator(mapPtr, trNoisy);

  okvis::Time t0(5, 0);
  okvis::ImuMeasurementDeque imuMeasurements = createImuMeasurements(t0);

  std::shared_ptr<okvis::cameras::CameraBase> tempCameraGeometry(
      new okvis::cameras::PinholeCamera<
          okvis::cameras::RadialTangentialDistortion>(
          752, 480, 350, 360, 378, 238,
          okvis::cameras::RadialTangentialDistortion(0.00, 0.00, 0.000,
                                                     0.000)));

  std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem(
      new okvis::cameras::NCameraSystem);
  Eigen::Matrix<double, 4, 4> matT_SC0;
  matT_SC0 << 1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1;
  std::shared_ptr<const okvis::kinematics::Transformation> T_SC_0(
      new okvis::kinematics::Transformation(matT_SC0));
  cameraSystem->addCamera(
      T_SC_0, tempCameraGeometry,
      okvis::cameras::NCameraSystem::DistortionType::RadialTangential);

  // add sensors
  // some parameters on how to do the online estimation:
  okvis::ExtrinsicsEstimationParameters extrinsicsEstimationParameters;
  extrinsicsEstimationParameters.sigma_absolute_translation = 2e-3;
  // not used
  extrinsicsEstimationParameters.sigma_absolute_orientation = 0;
  // not used
  extrinsicsEstimationParameters.sigma_c_relative_translation = 0;
  // not used in msckf
  extrinsicsEstimationParameters.sigma_c_relative_orientation = 0;

  extrinsicsEstimationParameters.sigma_focal_length = 0.01;
  extrinsicsEstimationParameters.sigma_principal_point = 0.01;
  // k1, k2, p1, p2, [k3]
  extrinsicsEstimationParameters.sigma_distortion << 1E-3, 1E-4, 1E-4, 1E-4,
      1E-5;
  extrinsicsEstimationParameters.sigma_td = 1E-4;
  extrinsicsEstimationParameters.sigma_tr = 1E-4;

  estimator.addCamera(extrinsicsEstimationParameters);
  // set the imu parameters
  okvis::ImuParameters imuParameters;
  imuParameters.a0.setZero();
  imuParameters.g = 9.81;
  imuParameters.a_max = 1000.0;
  imuParameters.g_max = 1000.0;
  imuParameters.rate = 100;
  imuParameters.sigma_g_c = 6.0e-4;
  imuParameters.sigma_a_c = 2.0e-3;
  imuParameters.sigma_gw_c = 3.0e-6;
  imuParameters.sigma_aw_c = 2.0e-5;
  imuParameters.tau = 3600.0;

  imuParameters.sigma_bg = 1e-2;  ///< Initial gyroscope bias.
  imuParameters.sigma_ba = 5e-2;  ///< Initial accelerometer bias

  // std for every element in shape matrix T_g
  imuParameters.sigma_TGElement = 1e-5;
  imuParameters.sigma_TSElement = 1e-5;
  imuParameters.sigma_TAElement = 1e-5;
  Eigen::Matrix<double, 9, 1> eye;
  eye << 1, 0, 0, 0, 1, 0, 0, 0, 1;
  imuParameters.Tg0 = eye;
  imuParameters.Ts0.setZero();
  imuParameters.Ta0 = eye;

  estimator.addImu(imuParameters);

  Eigen::Vector3d p_WS = Eigen::Vector3d(radius, 0, 0);
  Eigen::Vector3d v_WS = Eigen::Vector3d(0, angular_rate * radius, 0);
  Eigen::Matrix3d R_WS;
  // the RungeKutta method assumes that the z direction of
  // the world frame is negative gravity direction
  R_WS << 1, 0, 0, 0, 1, 0, 0, 0, 1;
  Eigen::Quaterniond q_WS = Eigen::Quaterniond(R_WS);


  okvis::InitialPVandStd pvstd;
  pvstd.p_WS = p_WS;
  pvstd.q_WS = Eigen::Quaterniond(q_WS);
  pvstd.v_WS = v_WS;
  pvstd.std_p_WS = Eigen::Vector3d(1e-2, 1e-2, 1e-2);
  pvstd.std_v_WS = Eigen::Vector3d(1e-1, 1e-1, 1e-1);
  pvstd.std_q_WS = Eigen::Vector3d(5e-2, 5e-2, 5e-2);
  estimator.resetInitialPVandStd(pvstd, true);

  for (int k = 0; k < 3; ++k) {
    okvis::Time currentKFTime = t0 + okvis::Duration(0.5 * k + 0.5);
    okvis::Time lastKFTime = t0;

    if (k != 0) {
        lastKFTime = estimator.statesMap_.rbegin()->second.timestamp;
    }
    okvis::ImuMeasurementDeque imuSegment =
        okvis::getImuMeasurments(lastKFTime, currentKFTime, imuMeasurements, nullptr);

    std::shared_ptr<okvis::MultiFrame> mf =
        createMultiFrame(currentKFTime, cameraSystem);
    estimator.addStates(mf, imuSegment, true);
  }
  int stateMapSize = estimator.statesMap_.size();
  int featureVariableDimen = estimator.cameraParamsMinimalDimen() +
                             estimator.clonedStateMinimalDimen() * (stateMapSize - 1);
  Eigen::Matrix<double, 2, Eigen::Dynamic> H_x(2, featureVariableDimen);
  // $\frac{\partial [z_u, z_v]^T}{\partial [\alpha, \beta, \rho]}$
  Eigen::Matrix<double, 2, 3> J_pfi;
  Eigen::Vector2d residual;

  Eigen::Vector4d ab1rho;
  ab1rho << 0.2, -0.3, 1, 1.0;

  Eigen::Vector2d obs(100, -150);

  uint64_t poseId = estimator.statesMap_.begin()->first;
  const int camIdx = 0;
  uint64_t anchorId = estimator.statesMap_.rbegin()->first;
  std::cout << "poseId " << poseId << " anchor " << anchorId << std::endl;
  okvis::kinematics::Transformation T_WBa(Eigen::Vector3d(0, 0, 1),
                                          Eigen::Quaterniond(1, 0, 0, 0));

  bool result = estimator.measurementJacobianAIDP(
      ab1rho, tempCameraGeometry, obs, poseId, camIdx, anchorId, T_WBa, &H_x,
      &J_pfi, &residual);
  std::cout << "H_x\n" << H_x << std::endl;
  std::cout << "J_pfi\n" << J_pfi << std::endl;
  std::cout << "residual\n" << residual << std::endl;
}
