#include <iostream>
#include "gtest/gtest.h"

#include <msckf/MSCKF2.hpp>
#include <msckf/PointLandmarkModels.hpp>

#include <okvis/IdProvider.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/FovDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>

std::shared_ptr<okvis::MultiFrame> createMultiFrame(
    okvis::Time timestamp,
    std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem) {
  // assemble a multi-frame
  std::shared_ptr<okvis::MultiFrame> mf(new okvis::MultiFrame());
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
    const Eigen::Vector2d& expectedObservation) {
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
  EXPECT_EQ(numLandmarks, 1u);
  const okvis::MapPoint& mp = landmarkMap.begin()->second;
  estimator.triangulateAMapPoint(
      mp, obsInPixel, pointLandmark, vRi,
      pointDataPtr.get(), nullptr, false);
  pointDataPtr->computePoseAndVelocityForJacobians(true);

  uint64_t anchorId = estimator.frameIdByAge(1);
  EXPECT_EQ(anchorId, pointDataPtr->anchorIds()[0].frameId_);
  size_t observationIndex = 0u;
  bool result = estimator.measurementJacobianAIDP(
      ab1rho, expectedObservation, observationIndex,
      pointDataPtr, &J_x, &J_pfi, &residual);

  Eigen::Matrix<double, 2, Eigen::Dynamic> J_x_legacy(2, featureVariableDimen);
  // $\frac{\partial [z_u, z_v]^T}{\partial [\alpha, \beta, \rho]}$
  Eigen::Matrix<double, 2, 3> J_pfi_legacy;
  Eigen::Vector2d residual_legacy;
  bool result_legacy = estimator.measurementJacobianAIDPMono(
              ab1rho, expectedObservation, observationIndex,
              pointDataPtr, &J_x_legacy, &J_pfi_legacy, &residual_legacy);
  EXPECT_EQ(result, result_legacy);
  EXPECT_TRUE(result);
  Eigen::IOFormat commaInitFmt(Eigen::StreamPrecision, 0, ", ", "\n", "", "", "", "");
  size_t tdIndex = estimator.intraStartIndexOfCameraParams(
      0u, okvis::Estimator::CameraSensorStates::TD);
  Eigen::MatrixXd J_x_diff = J_x - J_x_legacy;
  EXPECT_LT(J_x_diff.leftCols(tdIndex).lpNorm<Eigen::Infinity>(), 1e-6);
  EXPECT_LT(J_x_diff.rightCols(J_x.cols() - tdIndex - 2).lpNorm<Eigen::Infinity>(), 0.02)
          << "J_x\n" << J_x.format(commaInitFmt)
          << "\nJ_x_legacy\n" << J_x_legacy.format(commaInitFmt);
  // We do not check Jacobians relative to time as the new AIDP implementation
  // considers the effect of time error on anchor frame.
  EXPECT_LT((J_pfi - J_pfi_legacy).lpNorm<Eigen::Infinity>(), 1e-6);
  EXPECT_LT((residual - residual_legacy).lpNorm<Eigen::Infinity>(), 1e-4);
}

TEST(MSCKF2, MeasurementJacobianAIDP) {
  // TODO(jhuai): This test is fragile because of these hard coded numbers.
  // Rewrite the test with an independent method for computing Jacobians.
  Eigen::Vector2d expectedObservation(360.4815, 238.0);
  computeFeatureMeasJacobian(
      okvis::cameras::NCameraSystem::DistortionType::RadialTangential,
      expectedObservation);
  try {
    expectedObservation = Eigen::Vector2d(359.214, 238.0);
    computeFeatureMeasJacobian(
        okvis::cameras::NCameraSystem::DistortionType::FOV, expectedObservation);
  } catch (...) {
    std::cout << "Error occurred!\n";
  }
}
