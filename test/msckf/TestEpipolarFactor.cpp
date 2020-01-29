#include "gtest/gtest.h"

#include <Eigen/StdVector>

#include <msckf/CameraSystemCreator.hpp>
#include <msckf/CameraTimeParamBlock.hpp>
#include <msckf/EpipolarFactor.hpp>
#include <msckf/EuclideanParamBlock.hpp>
#include <msckf/ExtrinsicModels.hpp>
#include <msckf/ImuSimulator.h>
#include <msckf/numeric_ceres_residual_Jacobian.hpp>

#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>

TEST(CeresErrorTerms, EpipolarFactor) {
  typedef okvis::cameras::PinholeCamera<
      okvis::cameras::RadialTangentialDistortion>
      DistortedPinholeCameraGeometry;
  srand((unsigned int)time(0));
  bool rollingShutter = true;
  // create two view geometry poses, imu meas, from a simulation trajectory
  bool addPriorNoise = false;
  bool addSystemError = false;
  double bg_std = 5e-3;
  double ba_std = 2e-2;
  double Ta_std = 5e-3;
  double sigma_td = 5e-3;
  bool zeroImuIntrinsicParamNoise = !addSystemError;
  okvis::ImuParameters imuParameters;
  imu::initImuNoiseParams(&imuParameters, addPriorNoise, addSystemError,
                          bg_std, ba_std, Ta_std, zeroImuIntrinsicParamNoise);

  std::shared_ptr<imu::CircularSinusoidalTrajectory> cst;
  cst.reset(new imu::RoundedSquare(imuParameters.rate,
                                   Eigen::Vector3d(0, 0, -imuParameters.g),
                                   okvis::Time(0, 0), 1.0, 0, 0.8));
  const std::vector<okvis::Time> twoTimes{okvis::Time{20},
                                          okvis::Time{20 + 0.4}};

  // simulate the imu measurements
  okvis::ImuMeasurementDeque imuMeasurements;
  cst->getTrueInertialMeasurements(twoTimes[0] - okvis::Duration(1),
                                   twoTimes[1] + okvis::Duration(1),
                                   imuMeasurements);
  std::vector<okvis::kinematics::Transformation> two_T_WS = {
      cst->computeGlobalPose(twoTimes[0]), cst->computeGlobalPose(twoTimes[1])};
  std::vector<Eigen::Vector3d> two_vW = {
      cst->computeGlobalLinearVelocity(twoTimes[0]),
      cst->computeGlobalLinearVelocity(twoTimes[1])};

  // create camera geometry
  std::string projOptModelName = "FXY_CXY";
  std::string extrinsicModelName = "P_CB";
  int cameraOrientation = 0;
  simul::CameraSystemCreator csc(cameraOrientation, projOptModelName,
                                 extrinsicModelName, 0.0, 0.0);

  // reference camera system
  std::shared_ptr<okvis::cameras::CameraBase> cameraGeometry0;
  std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem0;
  csc.createNominalCameraSystem(&cameraGeometry0, &cameraSystem0);
  cameraGeometry0->setImageDelay(vio::gauss_rand(0, sigma_td));

  // create the camera visible landmarks, compute their observations in two
  // views
  okvis::kinematics::Transformation T_SC = *(cameraSystem0->T_SC(0));
  int numValidPoint = 0;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      allPointW;
  // every two observations is for one point
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      imagePointPairs;
  const int attempts = 10;
  for (int i = 0; i < attempts; ++i) {
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> pC(
        2);
    pC[0] = cameraGeometry0->createRandomVisiblePoint();
    okvis::kinematics::Transformation T_C1C0 = (two_T_WS[1] * T_SC).inverse() * (two_T_WS[0] * T_SC);
    pC[1] = T_C1C0.C() * pC[0] + T_C1C0.r();
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> ipC(
        2);
    bool projectionOk = true;
    for (int j = 0; j < 2; ++j) {
      okvis::cameras::CameraBase::ProjectionStatus ps =
          cameraGeometry0->project(pC[j], &ipC[j]);

      if (ps != okvis::cameras::CameraBase::ProjectionStatus::Successful) {
        projectionOk = false;
        break;
      }
    }
    if (projectionOk) {
      okvis::kinematics::Transformation T_WC0 = two_T_WS[0] * T_SC;
      allPointW.push_back(T_WC0.C() * pC[0] + T_WC0.r());
      imagePointPairs.push_back(ipC[0]);
      imagePointPairs.push_back(ipC[1]);
      ++numValidPoint;
    }
  }
  LOG(INFO) << "Found " << numValidPoint << " visible in both views out of "
            << attempts << " tries.";
  std::shared_ptr<DistortedPinholeCameraGeometry> cameraGeometryArg =
      std::static_pointer_cast<DistortedPinholeCameraGeometry>(
          cameraGeometry0);
  // create epipolar constraints from two view measurements and imu meas
  for (int j = 0; j < numValidPoint; ++j) {
    const int distortionDim =
        DistortedPinholeCameraGeometry::distortion_t::NumDistortionIntrinsics;
    const int projIntrinsicDim = okvis::ProjectionOptFXY_CXY::kNumParams;
    const Eigen::Matrix2d covariance = Eigen::Matrix2d::Identity() / 0.36;
    const double tdAtCreation = 0.0;

    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
        measurement12;
    measurement12.push_back(imagePointPairs[2 * j]);
    measurement12.push_back(imagePointPairs[2 * j + 1]);

    std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>>
        covariance12;
    covariance12.push_back(covariance);
    covariance12.push_back(covariance);

    std::vector<std::shared_ptr<const okvis::ImuMeasurementDeque>> imuMeasCanopy;
    imuMeasCanopy.emplace_back(new okvis::ImuMeasurementDeque(imuMeasurements));
    imuMeasCanopy.emplace_back(new okvis::ImuMeasurementDeque(imuMeasurements));

    std::vector<double> tdAtCreation2{tdAtCreation, tdAtCreation};

    std::vector<okvis::ceres::SpeedAndBias,
                Eigen::aligned_allocator<okvis::ceres::SpeedAndBias>>
        sb2(2);
    for (int j = 0; j < 2; ++j) {
      okvis::ceres::SpeedAndBias sb;
      sb.setZero();
      sb.head<3>() = two_vW[j] + Eigen::Vector3d::Random();
      sb2[j] = sb;
    }

    ::ceres::CostFunction* epiFactor =
        new okvis::ceres::EpipolarFactor<DistortedPinholeCameraGeometry,
                                         okvis::Extrinsic_p_BC_q_BC,
                                         okvis::ProjectionOptFXY_CXY>(
            cameraGeometryArg, j + 100, measurement12, covariance12,
            imuMeasCanopy, twoTimes, tdAtCreation2, sb2, imuParameters.g);

    // add parameter blocks
    uint64_t id = 1u;
    okvis::ceres::PoseParameterBlock poseParameterBlock0(two_T_WS[0], id++,
                                                         twoTimes[0]);
    okvis::ceres::PoseParameterBlock poseParameterBlock1(two_T_WS[1], id++,
                                                         twoTimes[1]);

    okvis::ceres::PoseParameterBlock extrinsicsParameterBlock(T_SC, id++,
                                                              twoTimes[0]);

    // camera parameters
    Eigen::VectorXd intrinsicParams;
    cameraGeometry0->getIntrinsics(intrinsicParams);
    double tr = 0;
    if (rollingShutter) {
      tr = 0.033;
    }

    int projOptModelId = okvis::ProjectionOptNameToId(projOptModelName);
    Eigen::VectorXd projIntrinsics;
    okvis::ProjectionOptGlobalToLocal(projOptModelId, intrinsicParams,
                                      &projIntrinsics);

    Eigen::VectorXd distortion = intrinsicParams.tail(distortionDim);
    okvis::ceres::EuclideanParamBlock projectionParamBlock(
        projIntrinsics, id++, twoTimes[0], projIntrinsicDim);
    okvis::ceres::EuclideanParamBlock distortionParamBlock(
        distortion, id++, twoTimes[0], distortionDim);
    okvis::ceres::CameraTimeParamBlock trParamBlock(tr, id++, twoTimes[0]);

    double timeOffset(0.0);
    okvis::ceres::CameraTimeParamBlock tdParamBlock(timeOffset, id++,
                                                    twoTimes[0]);

    double const* const parameters[] = {poseParameterBlock0.parameters(),
                                        poseParameterBlock1.parameters(),
                                        extrinsicsParameterBlock.parameters(),
                                        projectionParamBlock.parameters(),
                                        distortionParamBlock.parameters(),
                                        trParamBlock.parameters(),
                                        tdParamBlock.parameters()};
    double residual;
    Eigen::Matrix<double, 1, 7, Eigen::RowMajor> de_dT_WS[2];
    Eigen::Matrix<double, 1, 7, Eigen::RowMajor> de_dT_SC;
    Eigen::Matrix<double, 1, 6, Eigen::RowMajor> de_dT_WS_minimal[2];
    Eigen::Matrix<double, 1, 6, Eigen::RowMajor> de_dT_SC_minimal;
    Eigen::Matrix<double, 1, projIntrinsicDim, Eigen::RowMajor>
        de_dproj_intrinsic;
    Eigen::Matrix<double, 1, distortionDim, Eigen::RowMajor> de_ddistortion;
    Eigen::Matrix<double, 1, 1> de_dtr;
    Eigen::Matrix<double, 1, 1> de_dtd;

    double* jacobians[] = {de_dT_WS[0].data(),    de_dT_WS[1].data(),
                           de_dT_SC.data(),       de_dproj_intrinsic.data(),
                           de_ddistortion.data(), de_dtr.data(),
                           de_dtd.data()};
    double* jacobiansMinimal[] = {de_dT_WS_minimal[0].data(),
                                  de_dT_WS_minimal[1].data(),
                                  de_dT_SC_minimal.data(),
                                  de_dproj_intrinsic.data(),
                                  de_ddistortion.data(),
                                  de_dtr.data(),
                                  de_dtd.data()};

    okvis::ceres::EpipolarFactor<DistortedPinholeCameraGeometry,
                                 okvis::Extrinsic_p_BC_q_BC,
                                 okvis::ProjectionOptFXY_CXY>* costFuncPtr =
        static_cast<okvis::ceres::EpipolarFactor<DistortedPinholeCameraGeometry,
                                                 okvis::Extrinsic_p_BC_q_BC,
                                                 okvis::ProjectionOptFXY_CXY>*>(
            epiFactor);
    costFuncPtr->EvaluateWithMinimalJacobians(parameters, &residual, jacobians,
                                              jacobiansMinimal);

    const Eigen::Matrix<double, 1, 1> residualMat(residual);
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> de_dT_WS_numeric[2] = {
      Eigen::Matrix<double, -1, -1, Eigen::RowMajor>(1, 7),
      Eigen::Matrix<double, -1, -1, Eigen::RowMajor>(1, 7)
    };
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> de_dT_SC_numeric(1, 7);
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> de_dT_WS_minimal_numeric[2] =
    {
      Eigen::Matrix<double, -1, -1, Eigen::RowMajor>(1, 6),
      Eigen::Matrix<double, -1, -1, Eigen::RowMajor>(1, 6)
    };
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> de_dT_SC_minimal_numeric(1, 6);
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor>
        de_dproj_intrinsic_numeric(1, projIntrinsicDim);
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> de_ddistortion_numeric(1, distortionDim);
    Eigen::Matrix<double, 1, 1> de_dtr_numeric;
    Eigen::Matrix<double, 1, 1> de_dtd_numeric;
    simul::computeNumericJacPose(poseParameterBlock0, costFuncPtr, parameters,
                                 residualMat, &de_dT_WS_numeric[0], false);
    simul::computeNumericJacPose(poseParameterBlock0, costFuncPtr, parameters,
                                 residualMat, &de_dT_WS_minimal_numeric[0], true);

    simul::computeNumericJacPose(poseParameterBlock1, costFuncPtr, parameters,
                                 residualMat, &de_dT_WS_numeric[1], false);
    simul::computeNumericJacPose(poseParameterBlock1, costFuncPtr, parameters,
                                 residualMat, &de_dT_WS_minimal_numeric[1], true);

    simul::computeNumericJacPose(extrinsicsParameterBlock, costFuncPtr,
                                 parameters, residualMat,
                                 &de_dT_SC_numeric, false);
    simul::computeNumericJacPose(extrinsicsParameterBlock, costFuncPtr,
                                 parameters, residualMat, &de_dT_SC_minimal_numeric,
                                 true);

    simul::computeNumericJac(projectionParamBlock, costFuncPtr, parameters,
                             residualMat, &de_dproj_intrinsic_numeric);
    simul::computeNumericJac(distortionParamBlock, costFuncPtr, parameters,
                             residualMat, &de_ddistortion_numeric);
    simul::computeNumericJac<Eigen::Matrix<double, 1, 1>>(trParamBlock, costFuncPtr, parameters,
                             residualMat, &de_dtr_numeric);
    simul::computeNumericJac<Eigen::Matrix<double, 1, 1>>(tdParamBlock, costFuncPtr, parameters,
                             residualMat, &de_dtd_numeric);

    double tol = 1e-5;
    ARE_MATRICES_CLOSE(de_dT_WS_numeric[0], de_dT_WS[0], tol);
    ARE_MATRICES_CLOSE(de_dT_WS_numeric[1], de_dT_WS[1], tol);
    ARE_MATRICES_CLOSE(de_dT_SC_numeric, de_dT_SC, tol);
    ARE_MATRICES_CLOSE(de_dT_WS_minimal_numeric[0], de_dT_WS_minimal[0], tol);
    ARE_MATRICES_CLOSE(de_dT_WS_minimal_numeric[1], de_dT_WS_minimal[1], tol);
    ARE_MATRICES_CLOSE(de_dT_SC_minimal_numeric, de_dT_SC_minimal, tol);
    ARE_MATRICES_CLOSE(de_dproj_intrinsic_numeric, de_dproj_intrinsic, tol);
    ARE_MATRICES_CLOSE(de_ddistortion_numeric, de_ddistortion, tol);
    // Surprisingly, the numeric Jacobians for tr td speed and biases are zeros.
    tol = 1e-6;
    ARE_MATRICES_CLOSE(de_dtr_numeric, de_dtr, tol);
    ARE_MATRICES_CLOSE(de_dtd_numeric, de_dtd, tol);
    LOG(INFO) << "de_dtr " << de_dtr << " numeric " << de_dtr_numeric;
    LOG(INFO) << "de_dtd " << de_dtd << " numeric " << de_dtd_numeric;

    // TODO(jhuai): add to the ceres solver for optimization
  }
}
