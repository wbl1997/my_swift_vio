#include <iostream>
#include "gtest/gtest.h"

#include <swift_vio/MSCKF.hpp>
#include <swift_vio/PointLandmarkModels.hpp>

#include <okvis/IdProvider.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/cameras/FovDistortion.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>


class CameraMeasurementJacobianTest {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CameraMeasurementJacobianTest(
      okvis::cameras::NCameraSystem::DistortionType distortionId,
      const swift_vio::PointLandmarkOptions plOptions)
      : distortionId_(distortionId),
        pointLandmarkOptions_(plOptions),
        estimator_(std::shared_ptr<okvis::ceres::Map>(new okvis::ceres::Map)),
        t0_(5, 0)
  {
    initialize();
  }

  void initialize();

  void computeAndCheckJacobians() const;

 private:

  std::shared_ptr<okvis::MultiFrame> createMultiFrame(
      okvis::Time timestamp,
      std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem) {
    // assemble a multi-frame
    std::shared_ptr<okvis::MultiFrame> mf(new okvis::MultiFrame());
    mf->setId(okvis::IdProvider::instance().newId());
    mf->setTimestamp(timestamp);

    mf->resetCameraSystemAndFrames(*cameraSystem);
    mf->setTimestamp(0u, timestamp);
    return mf;
  }

  void initializeImuParameters() {
    imuParameters_.g = 9.81;
    imuParameters_.sigma_g_c = 6.0e-4;
    imuParameters_.sigma_a_c = 2.0e-3;
    imuParameters_.sigma_gw_c = 3.0e-6;
    imuParameters_.sigma_aw_c = 2.0e-5;

    imuParameters_.sigma_bg = 1e-2;  ///< Initial gyroscope bias.
    imuParameters_.sigma_ba = 5e-2;  ///< Initial accelerometer bias

    // std for every element in shape matrix T_g
    imuParameters_.sigma_TGElement = 1e-5;
    imuParameters_.sigma_TSElement = 1e-5;
    imuParameters_.sigma_TAElement = 1e-5;
    Eigen::Matrix<double, 9, 1> eye;
    eye << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    imuParameters_.Tg0 = eye;
    imuParameters_.Ts0.setZero();
    imuParameters_.Ta0 = eye;
  }

  /**
   * @brief createImuMeasurements for a rotational motion.
   * Body frame rotates about the z of the world frame which is negative gravity.
   * @param t0 start epoch
   * @return imu measurements.
   */
  okvis::ImuMeasurementDeque createImuMeasurements(okvis::Time t0, double gravity) {
    int DURATION = 10;
    int IMU_RATE = 200;
    double DT = 1.0 / IMU_RATE;

    okvis::ImuMeasurementDeque imuMeasurements;
    okvis::ImuSensorReadings nominalImuSensorReadings(
        Eigen::Vector3d(0, 0, angular_rate),
        Eigen::Vector3d(-radius * angular_rate * angular_rate, 0, gravity));
    for (int i = -2; i <= DURATION * IMU_RATE + 2; ++i) {
      Eigen::Vector3d gyr = nominalImuSensorReadings.gyroscopes;
      Eigen::Vector3d acc = nominalImuSensorReadings.accelerometers;
      imuMeasurements.push_back(okvis::ImuMeasurement(
          t0 + okvis::Duration(DT * i), okvis::ImuSensorReadings(gyr, acc)));
    }
    return imuMeasurements;
  }

  swift_vio::InitialNavState initializeNavState() {
    Eigen::Vector3d p_WS = Eigen::Vector3d(radius, 0, 0);
    Eigen::Vector3d v_WS = Eigen::Vector3d(0, angular_rate * radius, 0);
    Eigen::Matrix3d R_WS;
    // the RungeKutta method assumes that the z direction of
    // the world frame is negative gravity direction
    R_WS << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    Eigen::Quaterniond q_WS = Eigen::Quaterniond(R_WS);

    swift_vio::InitialNavState pvstd;
    pvstd.initWithExternalSource = true;
    pvstd.p_WS = p_WS;
    pvstd.q_WS = Eigen::Quaterniond(q_WS);
    pvstd.v_WS = v_WS;
    pvstd.std_p_WS = Eigen::Vector3d(1e-2, 1e-2, 1e-2);
    pvstd.std_v_WS = Eigen::Vector3d(1e-1, 1e-1, 1e-1);
    pvstd.std_q_WS = Eigen::Vector3d(5e-2, 5e-2, 5e-2);
    return pvstd;
  }

  okvis::kinematics::Transformation getPoseAt(okvis::Time t) {
    okvis::Duration deltat = t - t0_;
    double theta = deltat.toSec() * angular_rate;
    double ct = cos(theta);
    double st = sin(theta);
    Eigen::Vector3d pW(ct * radius, st * radius, 0);
    Eigen::Matrix3d R_WB;
    R_WB << ct, -st, 0,
        st, ct, 0,
        0, 0, 1;
    return okvis::kinematics::Transformation(pW, Eigen::Quaterniond(R_WB));
  }

  okvis::cameras::NCameraSystem::DistortionType distortionId_;
  swift_vio::PointLandmarkOptions pointLandmarkOptions_;

  swift_vio::MSCKF estimator_;
  okvis::Time t0_; // start time.

  okvis::ImuParameters imuParameters_;

  std::shared_ptr<const okvis::kinematics::Transformation> T_SC0_;
  std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry_;
  Eigen::Vector4d hpW_;


  const uint64_t landmarkId_ = 1000; // a unique numnber
  const double angular_rate = 0.3;  // rad/sec
  const double radius = 1.5;

};  // class CameraMeasurementJacobianTest

void CameraMeasurementJacobianTest::initialize() {
  double imageDelay = 0.0;
  double trNoisy = 0.033;
  estimator_.setPointLandmarkOptions(pointLandmarkOptions_);

  if (distortionId_ == okvis::cameras::NCameraSystem::DistortionType::FOV) {
    cameraGeometry_.reset(
        new okvis::cameras::PinholeCamera<okvis::cameras::FovDistortion>(
            752, 480, 350, 360, 378, 238, okvis::cameras::FovDistortion(0.9),
            imageDelay, trNoisy));
  } else if (distortionId_ ==
             okvis::cameras::NCameraSystem::DistortionType::RadialTangential) {
    cameraGeometry_.reset(new okvis::cameras::PinholeCamera<
                             okvis::cameras::RadialTangentialDistortion>(
        752, 480, 350, 360, 378, 238,
        okvis::cameras::RadialTangentialDistortion(0.10, 0.00, 0.000, 0.000),
        imageDelay, trNoisy));
  }

  std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem(
      new okvis::cameras::NCameraSystem);
  Eigen::Matrix<double, 4, 4> matT_SC0;
  matT_SC0 << 1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1;
  T_SC0_.reset(new okvis::kinematics::Transformation(matT_SC0));

  cameraSystem->addCamera(T_SC0_, cameraGeometry_, distortionId_, "FXY_CXY",
                          "P_CB");

  // add sensors
  // some parameters on how to do the online estimation:
  okvis::ExtrinsicsEstimationParameters extrinsicsEstimationParameters;
  extrinsicsEstimationParameters.sigma_absolute_translation = 2e-3;
  // not used
  extrinsicsEstimationParameters.sigma_absolute_orientation = 0;
  // not used
  extrinsicsEstimationParameters.sigma_c_relative_translation = 0;
  // not used in HybridFilter
  extrinsicsEstimationParameters.sigma_c_relative_orientation = 0;

  extrinsicsEstimationParameters.sigma_focal_length = 0.01;
  extrinsicsEstimationParameters.sigma_principal_point = 0.01;
  // k1, k2, p1, p2, [k3]
  extrinsicsEstimationParameters.sigma_distortion =
      std::vector<double>{1e-3, 1e-4, 1e-4, 1e-4, 1e-5};
  extrinsicsEstimationParameters.sigma_td = 1e-4;
  extrinsicsEstimationParameters.sigma_tr = 1e-4;

  estimator_.addCameraParameterStds(extrinsicsEstimationParameters);
  estimator_.addCameraSystem(*cameraSystem);

  // set the imu parameters
  initializeImuParameters();
  estimator_.addImu(imuParameters_);

  okvis::ImuMeasurementDeque imuMeasurements = createImuMeasurements(t0_, imuParameters_.g);

  estimator_.setInitialNavState(initializeNavState());

  Eigen::Vector4d ab1rho;
  ab1rho << -0.8, -0.2, 1.0, 0.2;
  okvis::kinematics::Transformation T_WB0 = getPoseAt(t0_);
  okvis::kinematics::Transformation T_WC0 = T_WB0 * (*T_SC0_);
  hpW_ = T_WC0 * ab1rho / ab1rho[3];
  estimator_.addLandmark(landmarkId_, hpW_);
  estimator_.setLandmarkInitialized(landmarkId_, true);

  // We have 3 observations in 3 frames.
  int numObservations = 3;
  int numStates = numObservations + 1;
  for (int k = 0; k < numStates; ++k) {
    okvis::Time currentKFTime = t0_ + okvis::Duration(0.5 * k + 0.5);
    okvis::Time lastKFTime = t0_;

    if (k != 0) {
      lastKFTime = estimator_.currentFrameTimestamp();
    }
    okvis::ImuMeasurementDeque imuSegment = swift_vio::getImuMeasurements(
        lastKFTime, currentKFTime, imuMeasurements, nullptr);

    std::shared_ptr<okvis::MultiFrame> mf =
        createMultiFrame(currentKFTime, cameraSystem);

    // add landmark observations
    std::vector<cv::KeyPoint> keypoints;
    okvis::kinematics::Transformation T_WB = getPoseAt(currentKFTime);
    okvis::kinematics::Transformation T_WC = T_WB * (*T_SC0_);

    Eigen::Vector2d measurement;
    okvis::cameras::CameraBase::ProjectionStatus projectOk =
        cameraGeometry_->projectHomogeneous(T_WC.inverse() * hpW_, &measurement);
    EXPECT_TRUE(projectOk == okvis::cameras::CameraBase::ProjectionStatus::Successful);
    measurement += Eigen::Vector2d::Random();

    keypoints.emplace_back(measurement[0], measurement[1], 8.0);
    mf->resetKeypoints(0, keypoints);
    mf->setLandmarkId(0, 0, landmarkId_);
    int cameraIndex = 0;
    int keypointIndex = 0;
    estimator_.addStates(mf, imuSegment, true);
    if (k < numObservations) {
      // triangulateAMapPoint will set anchor id as the last frame observing the point.
      if (distortionId_ == okvis::cameras::NCameraSystem::DistortionType::FOV) {
        estimator_.addObservation<
            okvis::cameras::PinholeCamera<okvis::cameras::FovDistortion>>(
            landmarkId_, mf->id(), cameraIndex, keypointIndex);
      } else if (distortionId_ == okvis::cameras::NCameraSystem::
                                      DistortionType::RadialTangential) {
        estimator_.addObservation<okvis::cameras::PinholeCamera<
            okvis::cameras::RadialTangentialDistortion>>(
            landmarkId_, mf->id(), cameraIndex, keypointIndex);
      }
    }
  }
}

void CameraMeasurementJacobianTest::computeAndCheckJacobians() const {
  int stateMapSize = estimator_.statesMapSize();
  int featureVariableDimen =
      estimator_.cameraParamsMinimalDimen() +
      estimator_.kClonedStateMinimalDimen * (stateMapSize - 1);
  Eigen::Matrix<double, 2, Eigen::Dynamic> J_x(2, featureVariableDimen);
  // $\frac{\partial [z_u, z_v]^T}{\partial [\alpha, \beta, \rho]}$
  Eigen::Matrix<double, 2, 3> J_pfi;
  Eigen::Vector2d residual;

  swift_vio::PointLandmark pointLandmark(pointLandmarkOptions_.landmarkModelId);
  std::shared_ptr<swift_vio::PointSharedData> pointDataPtr(
      new swift_vio::PointSharedData());

  // all observations for this feature point
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      obsInPixel;
  std::vector<double> vRi;  // std noise in pixels
  okvis::PointMap landmarkMap;
  size_t numLandmarks = estimator_.getLandmarks(landmarkMap);
  EXPECT_EQ(numLandmarks, 1u);
  const okvis::MapPoint& mp = landmarkMap.begin()->second;
  estimator_.triangulateAMapPoint(mp, obsInPixel, pointLandmark, vRi,
                                  pointDataPtr.get(), nullptr, false);

  Eigen::Vector4d homogeneousPoint = Eigen::Map<Eigen::Vector4d>(pointLandmark.data(), 4);
  pointDataPtr->computePoseAndVelocityForJacobians(true);
  size_t observationIndex = 0u;
  Eigen::IOFormat commaInitFmt(Eigen::StreamPrecision, 0, ", ", "\n", "", "",
                               "", "");
  if (pointLandmarkOptions_.landmarkModelId == swift_vio::InverseDepthParameterization::kModelId) {
    uint64_t anchorId = estimator_.frameIdByAge(1);
    EXPECT_EQ(anchorId, pointDataPtr->anchorIds()[0].frameId_);

    Eigen::Vector4d ab1rho_estimate = homogeneousPoint / homogeneousPoint[2];
    bool result = estimator_.measurementJacobian(
        ab1rho_estimate, obsInPixel[observationIndex], observationIndex,
        pointDataPtr, &J_x, &J_pfi, &residual);

    Eigen::Matrix<double, 2, Eigen::Dynamic> J_x_legacy(2,
                                                        featureVariableDimen);
    // $\frac{\partial [z_u, z_v]^T}{\partial [\alpha, \beta, \rho]}$
    Eigen::Matrix<double, 2, 3> J_pfi_legacy;
    Eigen::Vector2d residual_legacy;
    bool result_legacy = estimator_.measurementJacobianAIDPMono(
        ab1rho_estimate, obsInPixel[observationIndex], observationIndex,
        pointDataPtr, &J_x_legacy, &J_pfi_legacy, &residual_legacy);
    EXPECT_EQ(result, result_legacy);
    EXPECT_TRUE(result);

    size_t tdIndex = estimator_.intraStartIndexOfCameraParams(
        0u, okvis::Estimator::CameraSensorStates::TD);
    Eigen::MatrixXd J_x_diff = J_x - J_x_legacy;
    EXPECT_LT(J_x_diff.leftCols(tdIndex).lpNorm<Eigen::Infinity>(), 1e-6);

    if (pointLandmarkOptions_.anchorAtObservationTime) {
      // We do not check Jacobians relative to time (td, tr) as the new AIDP
      // implementation considers the effect of time error on anchor frame. and
      // we do not check Jacobian relative to anchor frame velocity as the new
      // AIDP implementation considers the velocity of anchor frame.
      EXPECT_LT(J_x_diff.block(0, tdIndex + 2, 2, J_x.cols() - tdIndex - 2 - 3)
                    .lpNorm<Eigen::Infinity>(),
                0.02)
          << "J_x\n"
          << J_x.format(commaInitFmt) << "\nJ_x_legacy\n"
          << J_x_legacy.format(commaInitFmt);
    } else {
      EXPECT_LT(J_x_diff.block(0, tdIndex, 2, J_x.cols() - tdIndex)
                    .lpNorm<Eigen::Infinity>(),
                0.02)
          << "J_x\n"
          << J_x.format(commaInitFmt) << "\nJ_x_legacy\n"
          << J_x_legacy.format(commaInitFmt);
    }

    EXPECT_LT((J_pfi - J_pfi_legacy).lpNorm<Eigen::Infinity>(), 1e-6);
    EXPECT_LT((residual - residual_legacy).lpNorm<Eigen::Infinity>(), 1e-4);
  } else {
    // $\frac{\partial [z_u, z_v]^T}{\partial(extrinsic, intrinsic, td, tr)}$
    Eigen::Matrix<double, 2, Eigen::Dynamic> J_Xc;
    Eigen::Matrix<double, 2, 9> J_XBj;  // $\frac{\partial [z_u, z_v]^T}{delta\p_{B_j}^G, \delta\alpha
                // (of q_{B_j}^G), \delta v_{B_j}^G$
    Eigen::Matrix<double, 2, 3> J_pfi_legacy;  // $\frac{\partial [z_u, z_v]^T}{\partial p_{f_i}^G}$
    Eigen::Vector2d residual_legacy;
    bool result_legacy = estimator_.measurementJacobianHPPMono(
        homogeneousPoint, obsInPixel[observationIndex], observationIndex,
        pointDataPtr, &J_Xc, &J_XBj, &J_pfi_legacy, &residual_legacy);
    EXPECT_TRUE(result_legacy);

    Eigen::Matrix<double, 2, 3> J_pfi;
    bool result = estimator_.measurementJacobian(
        homogeneousPoint, obsInPixel[observationIndex], observationIndex,
        pointDataPtr, &J_x, &J_pfi, &residual);
    EXPECT_TRUE(result);
    Eigen::MatrixXd J_x_diff = J_x;
    int numCameraIntrinsics = J_Xc.cols();
    J_x_diff.leftCols(numCameraIntrinsics) -= J_Xc;
    J_x_diff.block<2, 9>(0, numCameraIntrinsics) -= J_XBj;
    EXPECT_LT(J_x_diff.lpNorm<Eigen::Infinity>(), 1e-6)
        << "numCameraIntrinsics " << numCameraIntrinsics << " J_x_diff\n"
        << J_x_diff.format(commaInitFmt) << "\nlegacy J_Xc\n"
        << J_Xc.format(commaInitFmt) << "\nlegacy J_XBj\n" << J_XBj.format(commaInitFmt);
    EXPECT_LT((J_pfi - J_pfi_legacy).lpNorm<Eigen::Infinity>(), 1e-6);
    EXPECT_LT((residual - residual_legacy).lpNorm<Eigen::Infinity>(), 1e-6);
  }
}

TEST(CameraMeasurementJacobianTest, RadialTangential_AIDP) {
  swift_vio::PointLandmarkOptions plOptions;
  plOptions.landmarkModelId = swift_vio::InverseDepthParameterization::kModelId;
  plOptions.anchorAtObservationTime = true;
  CameraMeasurementJacobianTest cmjt(
      okvis::cameras::NCameraSystem::DistortionType::RadialTangential,
      plOptions);
  cmjt.computeAndCheckJacobians();
}

TEST(CameraMeasurementJacobianTest, FOV_AIDP) {
  swift_vio::PointLandmarkOptions plOptions;
  plOptions.landmarkModelId = swift_vio::InverseDepthParameterization::kModelId;
  plOptions.anchorAtObservationTime = true;
  CameraMeasurementJacobianTest cmjt(
      okvis::cameras::NCameraSystem::DistortionType::FOV,
      plOptions);
  cmjt.computeAndCheckJacobians();
}

TEST(CameraMeasurementJacobianTest, RadialTangential_AIDP_onesided) {
  swift_vio::PointLandmarkOptions plOptions;
  plOptions.landmarkModelId = swift_vio::InverseDepthParameterization::kModelId;
  plOptions.anchorAtObservationTime = false;
  CameraMeasurementJacobianTest cmjt(
      okvis::cameras::NCameraSystem::DistortionType::RadialTangential,
      plOptions);
  cmjt.computeAndCheckJacobians();
}

TEST(CameraMeasurementJacobianTest, FOV_AIDP_onesided) {
  swift_vio::PointLandmarkOptions plOptions;
  plOptions.landmarkModelId = swift_vio::InverseDepthParameterization::kModelId;
  plOptions.anchorAtObservationTime = false;
  CameraMeasurementJacobianTest cmjt(
      okvis::cameras::NCameraSystem::DistortionType::FOV,
      plOptions);
  cmjt.computeAndCheckJacobians();
}

TEST(CameraMeasurementJacobianTest, RadialTangential_HPP) {
  swift_vio::PointLandmarkOptions plOptions;
  plOptions.landmarkModelId = swift_vio::HomogeneousPointParameterization::kModelId;
  plOptions.anchorAtObservationTime = false;
  CameraMeasurementJacobianTest cmjt(
      okvis::cameras::NCameraSystem::DistortionType::RadialTangential,
      plOptions);
  cmjt.computeAndCheckJacobians();
}

TEST(CameraMeasurementJacobianTest, FOV_HPP) {
  swift_vio::PointLandmarkOptions plOptions;
  plOptions.landmarkModelId = swift_vio::HomogeneousPointParameterization::kModelId;
  plOptions.anchorAtObservationTime = false;
  CameraMeasurementJacobianTest cmjt(
      okvis::cameras::NCameraSystem::DistortionType::FOV,
      plOptions);
  cmjt.computeAndCheckJacobians();
}
