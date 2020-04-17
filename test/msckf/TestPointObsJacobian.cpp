#include <iostream>
#include "gtest/gtest.h"

#include <msckf/MSCKF2.hpp>
#include <msckf/PointLandmarkModels.hpp>

#include <okvis/IdProvider.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/FovDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>

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

void computeFeatureMeasJacobian(
    okvis::cameras::NCameraSystem::DistortionType distortionId,
    const Eigen::Vector2d& expectedObservation,
    const Eigen::Matrix<double, 2, 3>& expectedJpoint,
    const Eigen::MatrixXd& expectedJstates) {
  double imageDelay = 0.0;
  double trNoisy = 0.033;
  std::shared_ptr<okvis::ceres::Map> mapPtr(new okvis::ceres::Map);
  okvis::MSCKF2 estimator(mapPtr);
  estimator.setLandmarkModel(msckf::InverseDepthParameterization::kModelId);

  okvis::Time t0(5, 0);
  okvis::ImuMeasurementDeque imuMeasurements = createImuMeasurements(t0);

  std::shared_ptr<const okvis::cameras::CameraBase> tempCameraGeometry;
  if (distortionId == okvis::cameras::NCameraSystem::DistortionType::FOV) {
    tempCameraGeometry.reset(
        new okvis::cameras::PinholeCamera<okvis::cameras::FovDistortion>(
            752, 480, 350, 360, 378, 238, okvis::cameras::FovDistortion(0.9),
            imageDelay, trNoisy));
  } else if (distortionId ==
             okvis::cameras::NCameraSystem::DistortionType::RadialTangential) {
    tempCameraGeometry.reset(new okvis::cameras::PinholeCamera<
                             okvis::cameras::RadialTangentialDistortion>(
                             752, 480, 350, 360, 378, 238,
                             okvis::cameras::RadialTangentialDistortion(0.10, 0.00, 0.000, 0.000),
                             imageDelay, trNoisy));
  }

  std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem(
      new okvis::cameras::NCameraSystem);
  Eigen::Matrix<double, 4, 4> matT_SC0;
  matT_SC0 << 1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1;
  std::shared_ptr<const okvis::kinematics::Transformation> T_SC_0(
      new okvis::kinematics::Transformation(matT_SC0));

  cameraSystem->addCamera(
      T_SC_0, tempCameraGeometry, distortionId, "FXY_CXY", "P_CB");

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
  extrinsicsEstimationParameters.sigma_distortion =
      std::vector<double>{1e-3, 1e-4, 1e-4, 1e-4, 1e-5};
  extrinsicsEstimationParameters.sigma_td = 1e-4;
  extrinsicsEstimationParameters.sigma_tr = 1e-4;

  estimator.addCameraParameterStds(extrinsicsEstimationParameters);
  estimator.addCameraSystem(*cameraSystem);

  // set the imu parameters
  okvis::ImuParameters imuParameters;
  imuParameters.g = 9.81;
  imuParameters.sigma_g_c = 6.0e-4;
  imuParameters.sigma_a_c = 2.0e-3;
  imuParameters.sigma_gw_c = 3.0e-6;
  imuParameters.sigma_aw_c = 2.0e-5;

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

  okvis::InitialNavState pvstd;
  pvstd.initWithExternalSource = true;
  pvstd.p_WS = p_WS;
  pvstd.q_WS = Eigen::Quaterniond(q_WS);
  pvstd.v_WS = v_WS;
  pvstd.std_p_WS = Eigen::Vector3d(1e-2, 1e-2, 1e-2);
  pvstd.std_v_WS = Eigen::Vector3d(1e-1, 1e-1, 1e-1);
  pvstd.std_q_WS = Eigen::Vector3d(5e-2, 5e-2, 5e-2);
  estimator.setInitialNavState(pvstd);

  uint64_t landmarkId = 1000;
  Eigen::Vector4d ab1rho;
  ab1rho << 0.4, 0.3, 1.0, 0.3;
  okvis::kinematics::Transformation T_WCa;
  estimator.addLandmark(landmarkId, T_WCa * ab1rho/ab1rho[3]);
  estimator.setLandmarkInitialized(landmarkId, true);

  // We have 3 observations in 3 frames.
  int numObservations = 3; // Warn: This should be at least minTrackLength_.
  int numStates = numObservations + 1;
  for (int k = 0; k < numStates; ++k) {
    okvis::Time currentKFTime = t0 + okvis::Duration(0.5 * k + 0.5);
    okvis::Time lastKFTime = t0;

    if (k != 0) {
        lastKFTime = estimator.currentFrameTimestamp();
    }
    okvis::ImuMeasurementDeque imuSegment =
        okvis::getImuMeasurements(lastKFTime, currentKFTime, imuMeasurements, nullptr);

    std::shared_ptr<okvis::MultiFrame> mf =
        createMultiFrame(currentKFTime, cameraSystem);

    // add landmark observations
    std::vector<cv::KeyPoint> keypoints;
    Eigen::Vector2d measurement = expectedObservation;
    keypoints.emplace_back(measurement[0], measurement[1], 8.0);
    mf->resetKeypoints(0, keypoints);
    mf->setLandmarkId(0, 0, landmarkId);
    int cameraIndex = 0;
    int keypointIndex = 0;
    estimator.addStates(mf, imuSegment, true);
    if (k < numObservations) {
      // We skip adding observation for the latest frame because
      // triangulateAMapPoint will set the anchor id as the last frame observing
      // the point.
      if (distortionId == okvis::cameras::NCameraSystem::DistortionType::FOV) {
        estimator.addObservation<
            okvis::cameras::PinholeCamera<okvis::cameras::FovDistortion>>(
            landmarkId, mf->id(), cameraIndex, keypointIndex);
      } else if (distortionId == okvis::cameras::NCameraSystem::DistortionType::
                                     RadialTangential) {
        estimator.addObservation<okvis::cameras::PinholeCamera<
            okvis::cameras::RadialTangentialDistortion>>(
            landmarkId, mf->id(), cameraIndex, keypointIndex);
      }
    }
  }

  int stateMapSize = estimator.statesMapSize();
  int featureVariableDimen = estimator.cameraParamsMinimalDimen() +
                             estimator.kClonedStateMinimalDimen * (stateMapSize - 1);
  Eigen::Matrix<double, 2, Eigen::Dynamic> J_x(2, featureVariableDimen);
  // $\frac{\partial [z_u, z_v]^T}{\partial [\alpha, \beta, \rho]}$
  Eigen::Matrix<double, 2, 3> J_pfi;
  Eigen::Vector2d residual;

  uint64_t anchorId = estimator.frameIdByAge(1);

  okvis::kinematics::Transformation T_WBa(Eigen::Vector3d(0, 0, 1),
                                          Eigen::Quaterniond(1, 0, 0, 0));
  msckf::PointLandmark pointLandmark(msckf::InverseDepthParameterization::kModelId);
  std::shared_ptr<msckf::PointSharedData> pointDataPtr(new msckf::PointSharedData());

  // all observations for this feature point
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      obsInPixel;
  std::vector<double> vRi; // std noise in pixels
  okvis::PointMap landmarkMap;
  size_t numLandmarks = estimator.getLandmarks(landmarkMap);
  EXPECT_EQ(numLandmarks, 1);
  const okvis::MapPoint& mp = landmarkMap.begin()->second;
  estimator.triangulateAMapPoint(
      mp, obsInPixel, pointLandmark, vRi, tempCameraGeometry, *T_SC_0,
      pointDataPtr.get(), nullptr, false);
  pointDataPtr->computePoseAndVelocityForJacobians(true);
  EXPECT_EQ(anchorId, pointDataPtr->anchorIds()[0]);
  int observationIndex = 0;
  bool result = estimator.measurementJacobianAIDP(
      ab1rho, tempCameraGeometry, expectedObservation, observationIndex,
      pointDataPtr, &J_x, &J_pfi, &residual);
  Eigen::IOFormat spaceInitFmt(Eigen::FullPrecision, Eigen::DontAlignCols,
                               ",", " ", "", "", "", "");
  std::cout << "J_x\n" << J_x.rows() << " " << J_x.cols() << "\n" << J_x.format(spaceInitFmt) << std::endl;
  EXPECT_LT((J_x - expectedJstates).lpNorm<Eigen::Infinity>(), 1e-6);
  EXPECT_LT((J_pfi - expectedJpoint).lpNorm<Eigen::Infinity>(), 1e-6);
  EXPECT_LT(residual.lpNorm<Eigen::Infinity>(), 1e-4);
  EXPECT_TRUE(result);
}

TEST(MSCKF2, MeasurementJacobian) {
  Eigen::Vector2d expectedObservation(360.4815, 238.0);
  Eigen::Matrix<double, 2, 3> expectedJpoint;
  expectedJpoint << 350.258, 9.93345e-11, -525.386, 1.02593e-10, 360.084, -360.084;
  Eigen::MatrixXd expectedJstates(2, 40);
  expectedJstates << -0.000216805851973353, -1.98955441749053e-16,
      0.00433440512740632, -0.0500529547647059, 0, 1, 0, -0.0438561999225605,
      -0.000109817857626036, 9.93354729270363e-10, 2.62924628013702,
      102.975852337977, -0.429066051408235, -105.077360523391,
      -5.25377046353618, 2.98001502045618e-11, 5.95977831679945e-10,
      -9.92806083732939e-09, 351.139998429583, 0.0144480319946058,
      0.000722388184965761, -4.09749085297704e-15, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      105.077360523391, 5.25377046353618, -2.98001502045618e-11,
      5.25377046343684, -105.077360523351, -343.252841126589, 0, 0,
      0, -1.26430388652045e-13, -1.16516105242636e-25, 5.37359365296899e-15, 0,
      -2.83657371915451e-11, 0, 1, -2.55640588735688e-11, -6.40135301890805e-14,
      0.901455867475549, 1.0217362929638e-09, 0.000148532458908287,
      -6.18885245451193e-07, -3.07780126661939e-11, -3.0649816240008e-09,
      108.02526136408, 360.090888561366, 18.0042100894899, 6.13425546685075e-10,
      4.23194596358899e-15, 4.21431908318486e-13, -0.0148533654122997, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 3.07780126661939e-11, 3.0649816240008e-09,
      -108.02526136408, -360.084204543869, 144.033681818743,
      3.98404878978042e-09, 0, 0, 0;
  computeFeatureMeasJacobian(
      okvis::cameras::NCameraSystem::DistortionType::RadialTangential,
      expectedObservation, expectedJpoint, expectedJstates);
  try {
    expectedObservation = Eigen::Vector2d(359.214, 238.0);
    expectedJpoint << 374.828, -3.30823e-10, -562.241,
                      -3.39825e-10, 386.137, -386.137;
    expectedJstates.resize(2, 37);
    expectedJstates << -0.00023201451463455, -2.12911920186466e-16,
        0.0046384582921142, -0.053674389058464, 0, 1, 0, -3.07170483111539,
        110.199481157046, -0.459164504821023, -112.44840754534,
        -5.62231597073543, -9.92471170912074e-11, 2.00651122793431e-10,
        -1.06463583243582e-08, 375.77203550036, 0.0154615435890767,
        0.000773062823660151, 1.36463793529239e-14, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        112.44840754534, 5.62231597073543, 9.92471170912074e-11,
        5.62231597106625, -112.448407545472, -367.33160385682, 0, 0,
        0, -1.35299440712587e-13, -1.2469076402355e-25, 1.9564742098705e-16, 0,
        -3.04180566583197e-11, 0, 1, -1.79051635307018e-09,
        0.000159278955759189, -6.63662315663286e-07, 1.01947450414553e-10,
        -3.27999186221071e-09, 115.841111356055, 386.144205470152,
        19.3068517448449, 2.06834163514887e-10, -1.40176724845506e-14,
        4.5099560106211e-13, -0.0159280369703463, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        -1.01947450414553e-10, 3.27999186221071e-09, -115.841111356055,
        -386.137037850238, 154.454815141509, 4.71314731766279e-09, 0, 0, 0;
    computeFeatureMeasJacobian(
        okvis::cameras::NCameraSystem::DistortionType::FOV, expectedObservation,
        expectedJpoint, expectedJstates);
  } catch (...) {
    std::cout << "Error occurred!\n";
  }
}
