#include <glog/logging.h>
#include <gtest/gtest.h>

#include <ceres/ceres.h>

#include <msckf/EuclideanParamBlock.hpp>
#include <msckf/EuclideanParamBlockSized.hpp>
#include <msckf/ExtrinsicModels.hpp>
#include <msckf/ProjParamOptModels.hpp>
#include <msckf/RsReprojectionError.hpp>
#include <msckf/numeric_ceres_residual_Jacobian.hpp>

#include <okvis/FrameTypedefs.hpp>
#include <okvis/Time.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/ceres/CameraTimeParamBlock.hpp>
#include <okvis/ceres/HomogeneousPointError.hpp>
#include <okvis/ceres/HomogeneousPointLocalParameterization.hpp>
#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/ReprojectionError.hpp>
#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>
#include <okvis/kinematics/Transformation.hpp>

// When readout time, tr is non zero, analytic, numeric and automatic Jacobians
// of the rolling shutter reprojection factor are roughly the same.
// Surprisingly, if tr is zero, automatic Jacobians of the rolling shutter
// reprojection factor relative to the time offset and readout time are zeros
// and disagree with the values supported by both numeric and analytic
// approaches. Other than that, the rest Jacobians by the three method when tr
// is zero are roughly the same.

void setupPoseOptProblem(bool perturbPose, bool rollingShutter,
                         bool noisyKeypoint) {
  // srand((unsigned int) time(0));
  bool expectedZeroResidual = !(perturbPose || rollingShutter || noisyKeypoint);
  ::ceres::Problem problem;
  std::cout << "set up a random geometry... " << std::flush;
  okvis::kinematics::Transformation T_WS;
  T_WS.setRandom(10.0, M_PI);
  okvis::kinematics::Transformation T_disturb;
  T_disturb.setRandom(1, 0.01);
  okvis::kinematics::Transformation T_WS_init = T_WS;
  if (perturbPose) {
    T_WS_init = T_WS * T_disturb;
  }
  okvis::kinematics::Transformation private_T_SC;
  private_T_SC.setRandom(0.2, M_PI);
  const okvis::kinematics::Transformation T_SC = private_T_SC;

  const okvis::Time stateEpoch(2.0);
  uint64_t id = 1u;
  ::ceres::LocalParameterization* poseLocalParameterization(
      new okvis::ceres::PoseLocalParameterization);
  okvis::ceres::PoseParameterBlock poseParameterBlock(T_WS_init, id++,
                                                      stateEpoch);
  okvis::ceres::PoseParameterBlock extrinsicsParameterBlock(T_SC, id++,
                                                            stateEpoch);
  problem.AddParameterBlock(poseParameterBlock.parameters(),
                            poseParameterBlock.dimension(),
                            poseLocalParameterization);
  problem.AddParameterBlock(extrinsicsParameterBlock.parameters(),
                            extrinsicsParameterBlock.dimension(),
                            poseLocalParameterization);

  problem.SetParameterBlockVariable(poseParameterBlock.parameters());
  problem.SetParameterBlockConstant(extrinsicsParameterBlock.parameters());
  std::cout << " [ OK ] " << std::endl;

  // let's use our own local quaternion perturbation
  // TODO(jhuai): do we still need to setParameterization?
  //  std::cout << "setting local parameterization for pose... " << std::flush;
  //  problem.SetParameterization(poseParameterBlock.parameters(),
  //                              poseLocalParameterization);
  //  problem.SetParameterization(extrinsicsParameterBlock.parameters(),
  //                              poseLocalParameterization);
  //  std::cout << " [ OK ] " << std::endl;

  // set up a random camera geometry
  std::cout << "set up a random camera geometry... " << std::flush;
  typedef okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>
      DistortedPinholeCameraGeometry;
  const int distortionDim =
      DistortedPinholeCameraGeometry::distortion_t::NumDistortionIntrinsics;
  const int projIntrinsicDim = okvis::ProjectionOptFXY_CXY::kNumParams;
  std::shared_ptr<DistortedPinholeCameraGeometry> cameraGeometry =
      std::static_pointer_cast<DistortedPinholeCameraGeometry>(
          DistortedPinholeCameraGeometry::createTestObject());
  std::cout << " [ OK ] " << std::endl;

  Eigen::VectorXd intrinsicParams;
  cameraGeometry->getIntrinsics(intrinsicParams);
  double tr = 0;
  if (rollingShutter) {
    tr = 0.033;
  }

  std::string projOptModelName = "FXY_CXY";
  std::string extrinsicOptModelName = "P_SC_Q_SC";

  int projOptModelId = okvis::ProjectionOptNameToId(projOptModelName);
  int extrinsicOptModelId =
      okvis::ExtrinsicModelNameToId(extrinsicOptModelName);

  Eigen::VectorXd projIntrinsics;
  okvis::ProjectionOptGlobalToLocal(projOptModelId, intrinsicParams,
                                    &projIntrinsics);

  Eigen::VectorXd distortion = intrinsicParams.tail(distortionDim);
  okvis::ceres::EuclideanParamBlock projectionParamBlock(
      projIntrinsics, id++, stateEpoch, projIntrinsicDim);
  okvis::ceres::EuclideanParamBlock distortionParamBlock(
      distortion, id++, stateEpoch, distortionDim);

  okvis::ceres::CameraTimeParamBlock trParamBlock(tr, id++, stateEpoch);

  problem.AddParameterBlock(projectionParamBlock.parameters(),
                            projIntrinsicDim);
  problem.SetParameterBlockConstant(projectionParamBlock.parameters());
  problem.AddParameterBlock(distortionParamBlock.parameters(), distortionDim);
  problem.SetParameterBlockConstant(distortionParamBlock.parameters());
  problem.AddParameterBlock(trParamBlock.parameters(), 1);
  problem.SetParameterBlockConstant(trParamBlock.parameters());

  double timeOffset(0.0);
  okvis::ceres::CameraTimeParamBlock tdBlock(timeOffset, id++, stateEpoch);
  problem.AddParameterBlock(tdBlock.parameters(),
                            okvis::ceres::CameraTimeParamBlock::Dimension);
  problem.SetParameterBlockConstant(tdBlock.parameters());

  okvis::ceres::SpeedAndBias sb = okvis::ceres::SpeedAndBias::Random();
  okvis::ceres::SpeedAndBiasParameterBlock sbBlock(sb, id++, stateEpoch);
  problem.AddParameterBlock(
      sbBlock.parameters(),
      okvis::ceres::SpeedAndBiasParameterBlock::Dimension);
  problem.SetParameterBlockConstant(sbBlock.parameters());
  EXPECT_EQ(id, 8u);

  // and the parameterization for points:
  ::ceres::LocalParameterization* homogeneousPointLocalParameterization(
      new okvis::ceres::HomogeneousPointLocalParameterization);
  std::shared_ptr<okvis::ImuMeasurementDeque> imuMeasPtr(
      new okvis::ImuMeasurementDeque());

  // time * 100 Hz imu data centering at stateEpoch
  double gravity = 9.80665;
  int frequency = 100;
  double duration = 2;
  for (size_t jack = 0; jack < duration * frequency; ++jack) {
    okvis::Time time =
        stateEpoch + okvis::Duration(static_cast<double>(jack) /
                                         static_cast<double>(frequency) -
                                     duration * 0.5);
    Eigen::Vector3d gyrMeas = Eigen::Vector3d::Random();
    Eigen::Vector3d accMeas =
        Eigen::Vector3d::Random() +
        T_WS.C().transpose() * Eigen::Vector3d(0, 0, gravity);
    imuMeasPtr->push_back(okvis::ImuMeasurement(
        time, okvis::ImuSensorReadings(gyrMeas, accMeas)));
  }

  // get some random points and build error terms
  const size_t N = 100;
  const int frameId = 1;
  std::cout << "create N=" << N
            << " visible points and add respective reprojection error terms... "
            << std::endl;
  std::vector<::ceres::CostFunction*> allCostFunctions;
  std::vector<okvis::ceres::HomogeneousPointParameterBlock*> allPointBlocks;
  for (size_t i = 1; i < 100; ++i) {
    Eigen::Vector4d point = cameraGeometry->createRandomVisibleHomogeneousPoint(
        double(i % 10) * 3 + 2.0);
    okvis::ceres::HomogeneousPointParameterBlock*
        homogeneousPointParameterBlock_ptr(
            new okvis::ceres::HomogeneousPointParameterBlock(
                T_WS * T_SC * point, i - 1 + id));
    allPointBlocks.emplace_back(homogeneousPointParameterBlock_ptr);
    problem.AddParameterBlock(
        homogeneousPointParameterBlock_ptr->parameters(),
        okvis::ceres::HomogeneousPointParameterBlock::Dimension);
    problem.SetParameterBlockConstant(
        homogeneousPointParameterBlock_ptr->parameters());

    // get a randomized projection
    Eigen::Vector2d kp;
    cameraGeometry->projectHomogeneous(point, &kp);
    if (noisyKeypoint) {
      kp += Eigen::Vector2d::Random();
    }

    // Set up the only cost function (also known as residual).
    Eigen::Matrix2d information = Eigen::Matrix2d::Identity() * 0.36;
    double tdAtCreation = 0.0;

    ::ceres::CostFunction* cost_function(
        new okvis::ceres::RsReprojectionError<DistortedPinholeCameraGeometry,
                                              okvis::ProjectionOptFXY_CXY,
                                              okvis::Extrinsic_p_SC_q_SC>(
            cameraGeometry, frameId, kp, information, imuMeasPtr, T_SC,
            stateEpoch, tdAtCreation, gravity));
    allCostFunctions.emplace_back(cost_function);

    problem.AddResidualBlock(
        cost_function, NULL, poseParameterBlock.parameters(),
        homogeneousPointParameterBlock_ptr->parameters(),
        extrinsicsParameterBlock.parameters(),
        projectionParamBlock.parameters(), distortionParamBlock.parameters(),
        trParamBlock.parameters(), tdBlock.parameters(), sbBlock.parameters());

    // set the parameterization
    problem.SetParameterization(
        homogeneousPointParameterBlock_ptr->parameters(),
        homogeneousPointLocalParameterization);

    // compare Jacobians obtained by analytic diff and auto diff
    double const* const parameters[] = {
        poseParameterBlock.parameters(),
        homogeneousPointParameterBlock_ptr->parameters(),
        extrinsicsParameterBlock.parameters(),
        projectionParamBlock.parameters(),
        distortionParamBlock.parameters(),
        trParamBlock.parameters(),
        tdBlock.parameters(),
        sbBlock.parameters()};
    Eigen::Vector2d residuals;
    Eigen::Vector2d residuals_auto;

    Eigen::Matrix<double, 2, 7, Eigen::RowMajor> duv_deltaTWS_auto;
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> duv_deltaTWS_minimal_auto;
    Eigen::Matrix<double, 2, 4, Eigen::RowMajor> duv_deltahpW_auto;
    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> duv_deltahpW_minimal_auto;
    Eigen::Matrix<double, 2, 7, Eigen::RowMajor> duv_deltaTSC_auto;
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> duv_deltaTSC_minimal_auto;

    Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::RowMajor>
        duv_proj_intrinsic_auto(2, projIntrinsicDim);
    Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::RowMajor>
        duv_distortion_auto(2, distortionDim);
    Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::RowMajor> duv_tr_auto(2, 1);
    Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::RowMajor>
        duv_proj_intrinsic_minimal_auto(2, projIntrinsicDim);
    Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::RowMajor>
        duv_distortion_minimal_auto(2, distortionDim);
    Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::RowMajor>
        duv_tr_minimal_auto(2, 1);

    Eigen::Matrix<double, 2, 1> duv_td_auto;
    Eigen::Matrix<double, 2, 1> duv_td_minimal_auto;
    Eigen::Matrix<double, 2, 9, Eigen::RowMajor> duv_sb_auto;
    Eigen::Matrix<double, 2, 9, Eigen::RowMajor> duv_sb_minimal_auto;
    double* jacobiansAD[] = {
        duv_deltaTWS_auto.data(),   duv_deltahpW_auto.data(),
        duv_deltaTSC_auto.data(),   duv_proj_intrinsic_auto.data(),
        duv_distortion_auto.data(), duv_tr_auto.data(),
        duv_td_auto.data(),         duv_sb_auto.data()};
    double* jacobiansMinimalAD[] = {duv_deltaTWS_minimal_auto.data(),
                                    duv_deltahpW_minimal_auto.data(),
                                    duv_deltaTSC_minimal_auto.data(),
                                    duv_proj_intrinsic_minimal_auto.data(),
                                    duv_distortion_minimal_auto.data(),
                                    duv_tr_minimal_auto.data(),
                                    duv_td_minimal_auto.data(),
                                    duv_sb_minimal_auto.data()};

    Eigen::Matrix<double, 2, 7, Eigen::RowMajor> duv_deltaTWS;
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> duv_deltaTWS_minimal;
    Eigen::Matrix<double, 2, 4, Eigen::RowMajor> duv_deltahpW;
    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> duv_deltahpW_minimal;
    Eigen::Matrix<double, 2, 7, Eigen::RowMajor> duv_deltaTSC;
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> duv_deltaTSC_minimal;

    Eigen::Matrix<double, 2, projIntrinsicDim, Eigen::RowMajor>
        duv_proj_intrinsic;
    Eigen::Matrix<double, 2, distortionDim, Eigen::RowMajor> duv_distortion;
    Eigen::Matrix<double, 2, 1> duv_tr;

    Eigen::Matrix<double, 2, projIntrinsicDim, Eigen::RowMajor>
        duv_proj_intrinsic_minimal;
    Eigen::Matrix<double, 2, distortionDim, Eigen::RowMajor>
        duv_distortion_minimal;
    Eigen::Matrix<double, 2, 1> duv_tr_minimal;

    Eigen::Matrix<double, 2, 1> duv_td;
    Eigen::Matrix<double, 2, 1> duv_td_minimal;
    Eigen::Matrix<double, 2, 9, Eigen::RowMajor> duv_sb;
    Eigen::Matrix<double, 2, 9, Eigen::RowMajor> duv_sb_minimal;

    double* jacobians[] = {duv_deltaTWS.data(),   duv_deltahpW.data(),
                           duv_deltaTSC.data(),   duv_proj_intrinsic.data(),
                           duv_distortion.data(), duv_tr.data(),
                           duv_td.data(),         duv_sb.data()};
    double* jacobiansMinimal[] = {
        duv_deltaTWS_minimal.data(),   duv_deltahpW_minimal.data(),
        duv_deltaTSC_minimal.data(),   duv_proj_intrinsic_minimal.data(),
        duv_distortion_minimal.data(), duv_tr_minimal.data(),
        duv_td_minimal.data(),         duv_sb_minimal.data()};

    okvis::ceres::RsReprojectionError<DistortedPinholeCameraGeometry,
                                      okvis::ProjectionOptFXY_CXY,
                                      okvis::Extrinsic_p_SC_q_SC>* costFuncPtr =
        static_cast<okvis::ceres::RsReprojectionError<
            DistortedPinholeCameraGeometry, okvis::ProjectionOptFXY_CXY,
            okvis::Extrinsic_p_SC_q_SC>*>(cost_function);

    costFuncPtr->EvaluateWithMinimalJacobians(parameters, residuals.data(),
                                              jacobians, jacobiansMinimal);
    costFuncPtr->EvaluateWithMinimalJacobiansAutoDiff(
        parameters, residuals_auto.data(), jacobiansAD, jacobiansMinimalAD);

    if (i % 20 == 0) {
      if (expectedZeroResidual) {
        EXPECT_TRUE(residuals.isMuchSmallerThan(1, 1e-8))
            << "Without noise, residual should be fairly close to zero!";
        EXPECT_TRUE(residuals_auto.isMuchSmallerThan(1, 1e-8))
            << "Without noise, residual should be fairly close to zero!";
      }
      double tol = 1e-8;
      // analytic full vs minimal
      ARE_MATRICES_CLOSE(duv_proj_intrinsic, duv_proj_intrinsic_minimal, tol);
      ARE_MATRICES_CLOSE(duv_distortion, duv_distortion_minimal, tol);
      ARE_MATRICES_CLOSE(duv_tr, duv_tr_minimal, tol);
      ARE_MATRICES_CLOSE(duv_td, duv_td_minimal, tol);
      ARE_MATRICES_CLOSE(duv_sb, duv_sb_minimal, tol);

      // automatic vs analytic
      tol = 1e-3;
      ARE_MATRICES_CLOSE(residuals_auto, residuals, tol);
      ARE_MATRICES_CLOSE(duv_deltaTWS_auto, duv_deltaTWS, tol);
      ARE_MATRICES_CLOSE(duv_deltahpW_auto, duv_deltahpW, tol);
      ARE_MATRICES_CLOSE(duv_deltaTSC_auto, duv_deltaTSC, tol);
      ARE_MATRICES_CLOSE(duv_proj_intrinsic_auto, duv_proj_intrinsic, 4e-3);
      ARE_MATRICES_CLOSE(duv_distortion_auto, duv_distortion, 4e-3);

      if (rollingShutter) {
        ARE_MATRICES_CLOSE(duv_tr_auto, duv_tr, 1e-1);
        ARE_MATRICES_CLOSE(duv_td_auto, duv_td, 1e-1);
      }

      Eigen::Matrix<double, 2, 3> duv_ds_auto =
          duv_sb_auto.topLeftCorner<2, 3>();
      Eigen::Matrix<double, 2, 3> duv_ds = duv_sb.topLeftCorner<2, 3>();
      ARE_MATRICES_CLOSE(duv_ds_auto, duv_ds, tol);
      Eigen::Matrix<double, 2, 3> duv_dbg_auto = duv_sb_auto.block<2, 3>(0, 3);
      Eigen::Matrix<double, 2, 3> duv_dbg = duv_sb.block<2, 3>(0, 3);
      EXPECT_LT((duv_dbg_auto - duv_dbg).lpNorm<Eigen::Infinity>(), 5e-2)
          << "duv_dbg_auto:\n"
          << duv_dbg_auto << "\nduv_dbg\n"
          << duv_dbg;
      Eigen::Matrix<double, 2, 3> duv_dba_auto =
          duv_sb_auto.topRightCorner<2, 3>();
      Eigen::Matrix<double, 2, 3> duv_dba = duv_sb.topRightCorner<2, 3>();
      EXPECT_LT((duv_dba_auto - duv_dba).lpNorm<Eigen::Infinity>(), 5e-3)
          << "duv_dba_auto\n"
          << duv_dba_auto << "\nduv_dba\n"
          << duv_dba;

      ARE_MATRICES_CLOSE(duv_deltaTWS_minimal_auto, duv_deltaTWS_minimal, 5e-3);
      ARE_MATRICES_CLOSE(duv_deltahpW_minimal_auto, duv_deltahpW_minimal, tol);
      ARE_MATRICES_CLOSE(duv_deltaTSC_minimal_auto, duv_deltaTSC_minimal, tol);

      ARE_MATRICES_CLOSE(duv_proj_intrinsic_minimal_auto,
                         duv_proj_intrinsic_minimal, 4e-3);
      ARE_MATRICES_CLOSE(duv_distortion_minimal_auto, duv_distortion_minimal,
                         4e-3);
      if (rollingShutter) {
        ARE_MATRICES_CLOSE(duv_tr_minimal_auto, duv_tr_minimal, 1e-1);
        ARE_MATRICES_CLOSE(duv_td_minimal_auto, duv_td_minimal, 1e-1);
      }
      ARE_MATRICES_CLOSE(duv_sb_minimal_auto, duv_sb_minimal_auto, tol);

      // compute the numeric diff and check
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          duv_deltaTWS_numeric(2, 7);
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          duv_deltaTWS_minimal_numeric(2, 6);
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          duv_deltahpW_numeric(2, 4);
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          duv_deltahpW_minimal_numeric(2, 3);
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          duv_deltaTSC_numeric(2, 7);
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          duv_deltaTSC_minimal_numeric(2, 6);

      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          duv_proj_intrinsic_numeric(2, projIntrinsicDim);
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          duv_distortion_numeric(2, distortionDim);
      Eigen::Matrix<double, 2, 1> duv_tr_numeric;

      Eigen::Matrix<double, 2, 1> duv_td_numeric;
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          duv_sb_numeric(2, 9);

      simul::computeNumericJacPose(poseParameterBlock, costFuncPtr, parameters,
                                   residuals, &duv_deltaTWS_minimal_numeric,
                                   true);
      simul::computeNumericJacPose(poseParameterBlock, costFuncPtr, parameters,
                                   residuals, &duv_deltaTWS_numeric, false);

      simul::computeNumericJacPoint(*homogeneousPointParameterBlock_ptr,
                                    costFuncPtr, parameters, residuals,
                                    &duv_deltahpW_minimal_numeric, true);
      simul::computeNumericJacPoint(*homogeneousPointParameterBlock_ptr,
                                    costFuncPtr, parameters, residuals,
                                    &duv_deltahpW_numeric, false);

      simul::computeNumericJacPose(extrinsicsParameterBlock, costFuncPtr,
                                   parameters, residuals,
                                   &duv_deltaTSC_minimal_numeric, true);
      simul::computeNumericJacPose(extrinsicsParameterBlock, costFuncPtr,
                                   parameters, residuals, &duv_deltaTSC_numeric,
                                   false);

      simul::computeNumericJac(projectionParamBlock, costFuncPtr, parameters,
                               residuals, &duv_proj_intrinsic_numeric);

      simul::computeNumericJac(distortionParamBlock, costFuncPtr, parameters,
                               residuals, &duv_distortion_numeric);

      simul::computeNumericJac<Eigen::Matrix<double, 2, 1>>(
          trParamBlock, costFuncPtr, parameters, residuals, &duv_tr_numeric);

      simul::computeNumericJac<Eigen::Matrix<double, 2, 1>>(
          tdBlock, costFuncPtr, parameters, residuals, &duv_td_numeric);

      simul::computeNumericJac(sbBlock, costFuncPtr, parameters, residuals,
                               &duv_sb_numeric);

      ARE_MATRICES_CLOSE(duv_deltaTWS_numeric, duv_deltaTWS, tol);
      ARE_MATRICES_CLOSE(duv_deltaTWS_minimal_numeric, duv_deltaTWS_minimal,
                         5e-3);
      ARE_MATRICES_CLOSE(duv_deltahpW_numeric, duv_deltahpW, tol);
      ARE_MATRICES_CLOSE(duv_deltahpW_minimal_numeric, duv_deltahpW_minimal,
                         tol);
      ARE_MATRICES_CLOSE(duv_deltaTSC_numeric, duv_deltaTSC, tol);
      ARE_MATRICES_CLOSE(duv_deltaTSC_minimal_numeric, duv_deltaTSC_minimal,
                         5e-3);

      ARE_MATRICES_CLOSE(duv_proj_intrinsic_numeric, duv_proj_intrinsic, 4e-3);
      ARE_MATRICES_CLOSE(duv_distortion_numeric, duv_distortion, 4e-3);
      ARE_MATRICES_CLOSE(duv_tr_numeric, duv_tr, 1e-1);
      ARE_MATRICES_CLOSE(duv_td_numeric, duv_td, 1e-1);

      Eigen::Matrix<double, 2, 3> duv_ds_numeric =
          duv_sb_numeric.topLeftCorner<2, 3>();
      ARE_MATRICES_CLOSE(duv_ds_numeric, duv_ds, tol);
      Eigen::Matrix<double, 2, 3> duv_dbg_numeric =
          duv_sb_numeric.block<2, 3>(0, 3);
      EXPECT_LT((duv_dbg_numeric - duv_dbg).lpNorm<Eigen::Infinity>(), 5e-2)
          << "duv_dbg_numeric\n"
          << duv_dbg_numeric << "\nduv_dbg\n"
          << duv_dbg;
      Eigen::Matrix<double, 2, 3> duv_dba_numeric =
          duv_sb_numeric.topRightCorner<2, 3>();
      EXPECT_LT((duv_dba_numeric - duv_dba).lpNorm<Eigen::Infinity>(), 5e-3)
          << "duv_dba_numeric\n"
          << duv_dba_numeric << "\nduv_dba\n"
          << duv_dba;
    }
  }
  std::cout << " [ OK ] " << std::endl;

  // Run the solver!
  std::cout << "run the solver... " << std::endl;
  ::ceres::Solver::Options options;
  // options.check_gradients=true;
  // options.numeric_derivative_relative_step_size = 1e-6;
  // options.gradient_check_relative_precision=1e-2;
  options.minimizer_progress_to_stdout = false;
  ::FLAGS_stderrthreshold =
      google::WARNING;  // enable console warnings (Jacobian verification)
  ::ceres::Solver::Summary summary;
  Solve(options, &problem, &summary);

  // delete
  // Do not delete the below things which are likely managed by ceres solver
  // {
  //  for (size_t j = 0; j < allCostFunctions.size(); ++j) {
  //    delete allCostFunctions[j];
  //  }
  //  delete homogeneousPointLocalParameterization;
  //  delete poseLocalParameterization;
  // }
  for (size_t j = 0; j < allPointBlocks.size(); ++j) {
    delete allPointBlocks[j];
  }

  // print some infos about the optimization
  // std::cout << summary.BriefReport() << "\n";

  std::cout << "initial T_WS : " << T_WS_init.T() << "\n"
            << "optimized T_WS : " << poseParameterBlock.estimate().T() << "\n"
            << "correct T_WS : " << T_WS.T() << "\n";

  // make sure it converged
  EXPECT_LT(
      2 * (T_WS.q() * poseParameterBlock.estimate().q().inverse()).vec().norm(),
      1e-2)
      << "quaternions not close enough";
  EXPECT_LT((T_WS.r() - poseParameterBlock.estimate().r()).norm(), 1e-1)
      << "translation not close enough";
}

TEST(CeresErrorTerms, ReprojectionErrorNoiseFree) {
  setupPoseOptProblem(false, false, false);
}

TEST(CeresErrorTerms, ReprojectionErrorNoisy) {
  setupPoseOptProblem(true, true, true);
}
