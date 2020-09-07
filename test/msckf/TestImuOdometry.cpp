
#include <gtest/gtest.h>

#include "gtsam/ImuFactorTestHelpers.h"

#include "msckf/imu/ImuErrorModel.h"
#include <msckf/imu/ImuOdometry.h>
#include <msckf/ImuOdometryLegacy.hpp>

#include <okvis/ceres/ImuError.hpp>
#include <okvis/timing/Timer.hpp>

#include "vio/Sample.h"
#include "vio/eigen_utils.h"

#include "sophus/se3.hpp"

#include "CovPropConfig.hpp"


// compare RungeKutta and Euler forward and backward integration
class BackwardIntegrationTest : public ::testing::Test {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  BackwardIntegrationTest() :
      cpc(true, false),
      p_WS_W(cpc.get_p_WS_W0()),
      q_WS(cpc.get_q_WS0()),
      sb(cpc.get_sb0()),
      T_WB(cpc.get_p_WS_W0(), cpc.get_q_WS0()),
      imuParams(cpc.get_imu_params()) {
    std::cout << "States before forward integration:" << std::endl;
    print_p_q_sb(p_WS_W, q_WS, sb);
  }

  void SetUp() override {};

  CovPropConfig cpc;
  Eigen::Vector3d p_WS_W;
  Eigen::Quaterniond q_WS;
  okvis::SpeedAndBiases sb;
  okvis::kinematics::Transformation T_WB;
  okvis::ImuParameters imuParams;
private:
};

TEST_F(BackwardIntegrationTest, BackwardRK) {
  ImuErrorModel<double> iem(sb.tail<6>(), cpc.get_vTgTsTa());
  okvis::ImuOdometry::propagation_RungeKutta(
      cpc.get_imu_measurements(), cpc.get_imu_params(), T_WB, sb,
      iem, cpc.get_meas_begin_time(), cpc.get_meas_end_time());
  okvis::ImuOdometry::propagationBackward_RungeKutta(
      cpc.get_imu_measurements(), imuParams, T_WB, sb,
      iem, cpc.get_meas_end_time(), cpc.get_meas_begin_time());
  p_WS_W = T_WB.r();
  q_WS = T_WB.q();

  Eigen::Vector3d p_WS_RoundTrip = p_WS_W;
  Eigen::Quaterniond q_WS_RoundTrip = q_WS;
  okvis::SpeedAndBiases speedAndBiasRoundTrip = sb;
  std::cout << "States after forward-backward RK integration:" << std::endl;
  print_p_q_sb(p_WS_RoundTrip, q_WS_RoundTrip, speedAndBiasRoundTrip);

  // Runge Kutta return to the starting position ?
  cpc.check_q_near(q_WS_RoundTrip, 1e-8);
  cpc.check_v_near(speedAndBiasRoundTrip, 2e-2);
  cpc.check_p_near(p_WS_RoundTrip, 1.5);
}

TEST_F(BackwardIntegrationTest, StepwiseBackwardRK) {
  // DETAILED INTEGRATION
  okvis::ImuMeasurementDeque imuMeasurements = cpc.get_imu_measurements();
  auto iterLast = imuMeasurements.begin();
  for (auto iter = imuMeasurements.begin(); iter != imuMeasurements.end();
       ++iter) {
    if (iter == imuMeasurements.begin()) continue;
    ImuErrorModel<double> iem(sb.tail<6>(), cpc.get_vTgTsTa());
    okvis::ceres::ode::integrateOneStep_RungeKutta(
        iterLast->measurement.gyroscopes, iterLast->measurement.accelerometers,
        iter->measurement.gyroscopes, iter->measurement.accelerometers,
        cpc.get_g(), cpc.get_sigma_g_c(), cpc.get_sigma_a_c(),
        cpc.get_sigma_gw_c(), cpc.get_sigma_aw_c(), cpc.get_dt(), p_WS_W, q_WS,
        sb, iem);
    iterLast = iter;
  }

  std::cout << "States after forward RK integration:" << std::endl;
  print_p_q_sb(p_WS_W, q_WS, sb);

  // backward
  auto iterRLast = imuMeasurements.rbegin();
  for (auto iterR = imuMeasurements.rbegin(); iterR != imuMeasurements.rend();
       ++iterR) {
    if (iterR == imuMeasurements.rbegin()) continue;
    ImuErrorModel<double> iem(sb.tail<6>(), cpc.get_vTgTsTa());
    okvis::ceres::ode::integrateOneStepBackward_RungeKutta(
        iterR->measurement.gyroscopes, iterR->measurement.accelerometers,
        iterRLast->measurement.gyroscopes,
        iterRLast->measurement.accelerometers, cpc.get_g(), cpc.get_sigma_g_c(),
        cpc.get_sigma_a_c(), cpc.get_sigma_gw_c(), cpc.get_sigma_aw_c(),
        cpc.get_dt(), p_WS_W, q_WS, sb, iem);
    iterRLast = iterR;
  }

  std::cout << "States after backward RK integration:" << std::endl;
  print_p_q_sb(p_WS_W, q_WS, sb);

  Eigen::Vector3d p_WS_RoundTrip = p_WS_W;
  Eigen::Quaterniond q_WS_RoundTrip = q_WS;
  okvis::SpeedAndBiases speedAndBiasRoundTrip = sb;

  cpc.check_q_near(q_WS_RoundTrip, 1e-8);
  cpc.check_v_near(speedAndBiasRoundTrip, 2e-2);
  cpc.check_p_near(p_WS_RoundTrip, 1.5);
}

TEST_F(BackwardIntegrationTest, BackwardEuler) {
  Eigen::Vector3d tempV_WS = sb.head<3>();
  ImuErrorModel<double> iem(sb.tail<6>(), cpc.get_vTgTsTa());
  okvis::ImuOdometry::propagation(cpc.get_imu_measurements(), cpc.get_imu_params(),
                           T_WB, tempV_WS, iem, cpc.get_meas_begin_time(),
                           cpc.get_meas_end_time());

  std::cout << "States after forward Euler integration:" << std::endl;
  sb.head<3>() = tempV_WS;
  print_p_q_sb(T_WB.r(), T_WB.q(), sb);

  okvis::ImuOdometry::propagationBackward(
      cpc.get_imu_measurements(), imuParams, T_WB, tempV_WS, iem,
      cpc.get_meas_end_time(), cpc.get_meas_begin_time());
  p_WS_W = T_WB.r();
  q_WS = T_WB.q();
  sb.head<3>() = tempV_WS;
  std::cout << "States after backward Euler integration:" << std::endl;
  print_p_q_sb(T_WB.r(), T_WB.q(), sb);

  Eigen::Vector3d p_WS_RoundTrip = p_WS_W;
  Eigen::Quaterniond q_WS_RoundTrip = q_WS;
  okvis::SpeedAndBiases speedAndBiasRoundTrip = sb;

  // Euler return to the starting position ?
  cpc.check_q_near(q_WS_RoundTrip, 1e-6);
  cpc.check_v_near(speedAndBiasRoundTrip, 0.1);
  cpc.check_p_near(p_WS_RoundTrip, 5.0);
}

void IMUOdometryTrapezoidRule(
    const CovPropConfig& cpc,
    Eigen::Vector3d* p_WS_W1,
    Eigen::Quaterniond* q_WS1, okvis::SpeedAndBiases* sb1,
    Eigen::MatrixXd* covariance,
    Eigen::MatrixXd* jacobian,
    bool zeroVarianceForImuExtraParams = true,
    bool usePositionVelocityLin = true,
    bool verbose = false) {
  const Eigen::Vector3d p_WS_W0 = cpc.get_p_WS_W0();
  const Eigen::Quaterniond q_WS0 = cpc.get_q_WS0();
  const okvis::SpeedAndBiases sb0 = cpc.get_sb0();
  const Eigen::Matrix<double, kImuAugmentedParamsDim, 1>& vTgTsTa = cpc.get_vTgTsTa();
  const okvis::ImuMeasurementDeque& imuMeasurements = cpc.get_imu_measurements();
  const okvis::ImuParameters& imuParams = cpc.get_imu_params();

  okvis::kinematics::Transformation T_WS(p_WS_W0, q_WS0);
  okvis::SpeedAndBiases sb = sb0;
  okvis::timing::Timer okvisTimer("okvis", false);

  *covariance = Eigen::Matrix<double, kNavAndImuParamsDim, kNavAndImuParamsDim>::Identity();
  if (zeroVarianceForImuExtraParams) {
    covariance->bottomRightCorner<kImuAugmentedParamsDim, kImuAugmentedParamsDim>().setZero();
  }
  *jacobian = Eigen::Matrix<double, kNavAndImuParamsDim, kNavAndImuParamsDim>::Identity();

  int numUsedImuMeasurements = 0;

  Eigen::Vector3d v_WS = sb.head<3>();
  ImuErrorModel<double> imuModel(sb.tail<6>(), vTgTsTa);
  Eigen::Matrix<double, 6, 1> linearizationPoint;
  linearizationPoint << T_WS.r(), v_WS;
  numUsedImuMeasurements = okvis::ImuOdometry::propagation(
      imuMeasurements, imuParams, T_WS, v_WS, imuModel,
      imuMeasurements.begin()->timeStamp, imuMeasurements.rbegin()->timeStamp,
      covariance, jacobian,
      usePositionVelocityLin ? &linearizationPoint : nullptr);
  sb.head<3>() = v_WS;
  double timeElapsed = okvisTimer.stop();

  *p_WS_W1 = T_WS.r();
  *q_WS1 = T_WS.q();
  *sb1 = sb;
  Eigen::Matrix<double, 42, 1> covDiagonal = covariance->diagonal();
  Eigen::Matrix<double, 42, 1> sqrtCovDiagonal = covDiagonal.cwiseSqrt();
  Eigen::Matrix<double, 42, 1> jacobianFirstRow = jacobian->row(0);

  std::cout
      << "time used by huai trapezoid rule forward propagtion with covariance "
      << timeElapsed << std::endl;
  if (verbose) {
    std::cout << "numUsedMeas " << numUsedImuMeasurements << " totalMeas "
              << (int)imuMeasurements.size() << std::endl;
    std::cout << "q_WS " << q_WS1->w() << " " << q_WS1->x() << " " << q_WS1->y()
              << " " << q_WS1->z() << std::endl;
    std::cout << "p_WS_W " << p_WS_W1->transpose() << std::endl;
    std::cout << "speed and bias " << sb1->transpose() << std::endl;
    std::cout << "cov diagonal sqrt " << std::endl;
    std::cout << sqrtCovDiagonal.transpose() << std::endl;
    std::cout << "Jacobian diagonal " << std::endl;
    std::cout << vio::superdiagonal(*jacobian).transpose() << std::endl;
    std::cout << "Jacobian first row " << std::endl;
    std::cout << jacobianFirstRow.transpose() << std::endl;
  }
}

TEST(ImuOdometry, linearizationPointForPV) {
  srand((unsigned int)time(0));
  CovPropConfig cpc(false, true);
  bool zeroVarForImuExtraParams = true;
  bool verbose = false;

  bool usePositionVelocityLin = false;
  Eigen::Vector3d p_WS;
  Eigen::Quaterniond q_WS;
  okvis::SpeedAndBiases speedAndBias;

  Eigen::MatrixXd cov;
  Eigen::MatrixXd jacobian;
  IMUOdometryTrapezoidRule(cpc, &p_WS, &q_WS, &speedAndBias, &cov, &jacobian,
                           zeroVarForImuExtraParams, usePositionVelocityLin,
                           verbose);

  usePositionVelocityLin = true;
  Eigen::Vector3d p_WS_lin;
  Eigen::Quaterniond q_WS_lin;
  okvis::SpeedAndBiases speedAndBiasLin;
  Eigen::MatrixXd covLin;
  Eigen::MatrixXd jacobianLin;
  IMUOdometryTrapezoidRule(cpc, &p_WS_lin, &q_WS_lin, &speedAndBiasLin, &covLin,
                           &jacobianLin, zeroVarForImuExtraParams,
                           usePositionVelocityLin, verbose);

  checkSelectiveRatio(p_WS, p_WS_lin, 1e-4, 1e-4, 1e-3);
  check_q_near(q_WS, q_WS_lin, 1e-6);
  checkSelectiveRatio(speedAndBias, speedAndBiasLin, 1e-4, 1e-4, 1e-3);
  checkSelectiveRatio(cov, covLin, 1e-4, 1e-4, 1e-3);
  checkSelectiveRatio(jacobian, jacobianLin, 1e-4, 1e-4, 1e-3);
}

/// test and compare the propagation for both states and covariance by both the
/// classic RK4 and okvis's state transition method
TEST(ImuOdometry, IMUCovariancePropagation) {
  // use positive variance for elements in Tg Ts Ta?
  bool zeroVarForImuExtraParams = true;
  bool verbose = true;  // print the covariance and jacobian results

  srand((unsigned int)time(0));
  CovPropConfig cpc(false, true);

  /// method 1: RK4
  Eigen::Vector3d p_WS_W = cpc.get_p_WS_W0();
  Eigen::Quaterniond q_WS = cpc.get_q_WS0();
  okvis::SpeedAndBiases sb = cpc.get_sb0();

  Eigen::MatrixXd covRK4 = Eigen::MatrixXd::Identity(kNavAndImuParamsDim, kNavAndImuParamsDim);
  if (zeroVarForImuExtraParams) {
    covRK4.bottomRightCorner<kImuAugmentedParamsDim, kImuAugmentedParamsDim>().setZero();
  }
  Eigen::MatrixXd jacobianRK4 = Eigen::MatrixXd::Identity(kNavAndImuParamsDim, kNavAndImuParamsDim);

  okvis::timing::Timer RK4Timer("RK4", false);
  const okvis::ImuMeasurementDeque& imuMeasurements = cpc.get_imu_measurements();
  auto iterLast = imuMeasurements.begin();
  for (auto iter = imuMeasurements.begin(); iter != imuMeasurements.end();
       ++iter) {
    if (iter == imuMeasurements.begin()) continue;
    ImuErrorModel<double> iem(sb.tail<6>(), cpc.get_vTgTsTa());
    okvis::ceres::ode::integrateOneStep_RungeKutta(
        iterLast->measurement.gyroscopes, iterLast->measurement.accelerometers,
        iter->measurement.gyroscopes, iter->measurement.accelerometers,
        cpc.get_g(), cpc.get_sigma_g_c(), cpc.get_sigma_a_c(),
        cpc.get_sigma_gw_c(), cpc.get_sigma_aw_c(), cpc.get_dt(), p_WS_W, q_WS,
        sb, iem, &covRK4, &jacobianRK4);
    iterLast = iter;
  }
  double timeElapsed = RK4Timer.stop();

  Eigen::Vector3d p_WS_W_RK4 = p_WS_W;
  Eigen::Quaterniond q_WS_RK4 = q_WS;
  okvis::SpeedAndBiases speedAndBiasRK4 = sb;
  Eigen::Matrix<double, 42, 1> covDiagonal = covRK4.diagonal();
  Eigen::Matrix<double, 42, 1> sqrtCovDiagonalRK4 = covDiagonal.cwiseSqrt();
  Eigen::Matrix<double, 42, 1> jacobianFirstRowRK4 = jacobianRK4.row(0).transpose();
  std::cout << "time used by RK4 forward propagtion with covariance "
            << timeElapsed << std::endl;

  if (verbose) {
    std::cout << "q_WS " << q_WS_RK4.w() << " " << q_WS_RK4.x() << " " << q_WS_RK4.y()
              << " " << q_WS_RK4.z() << std::endl;
    std::cout << "p_WS_W " << p_WS_W_RK4.transpose() << std::endl;
    std::cout << "speed and bias " << speedAndBiasRK4.transpose() << std::endl;

    std::cout << "cov(r_s^w, \\phi^w, v_s^w, b_g, b_a, T_g, T_s, T_a), its "
                 "diagonal sqrt "
              << std::endl;
    std::cout << sqrtCovDiagonalRK4.transpose() << std::endl;
    std::cout << "Jacobian superdiagonal " << std::endl;
    std::cout << vio::superdiagonal(jacobianRK4).transpose() << std::endl;
    std::cout << "Jacobian first row " << std::endl;
    std::cout << jacobianFirstRowRK4.transpose() << std::endl;
  }

  /// method 2 : propagation by using trapezoid rules implemented with okvis error convention.
  Eigen::Vector3d p_WS_Trapezoid;
  Eigen::Quaterniond q_WS_Trapezoid;
  okvis::SpeedAndBiases speedAndBiasTrapezoid;
  bool usePositionVelocityLin = true;
  Eigen::MatrixXd covTrapezoid;
  Eigen::MatrixXd jacobianTrapezoid;
  IMUOdometryTrapezoidRule(cpc, &p_WS_Trapezoid, &q_WS_Trapezoid,
                           &speedAndBiasTrapezoid, &covTrapezoid,
                           &jacobianTrapezoid, zeroVarForImuExtraParams,
                           usePositionVelocityLin, verbose);
  Eigen::Matrix<double, 42, 1> sqrtCovDiagonalTrapezoid = covTrapezoid.diagonal().cwiseSqrt();
  Eigen::Matrix<double, 42, 1> jacobianFirstRowTrapezoid = jacobianTrapezoid.row(0);

  /// method 3 : okvis propagation leutenegger's implementation
  okvis::kinematics::Transformation T_WS =
      okvis::kinematics::Transformation(cpc.get_p_WS_W0(), cpc.get_q_WS0());
  sb = cpc.get_sb0();
  okvis::timing::Timer leutenTimer("leutenegger", false);

  Eigen::Matrix<double, 15, 15> covOkvis;
  covOkvis.setIdentity();
  Eigen::Matrix<double, 15, 15> jacobianOkvis;
  jacobianOkvis.setIdentity();

  // The Leutenegger's ImuError propagation function starts propagation with an
  // zero covariance. The original implementation has some issue in covariance
  // propagation, its Jacobian is correct though.
  int numUsedImuMeasurements = okvis::ceres::ImuError::propagation(
      cpc.get_imu_measurements(), cpc.get_imu_params(), T_WS, sb,
      cpc.get_meas_begin_time(), cpc.get_meas_end_time(), &covOkvis, &jacobianOkvis);
  timeElapsed = leutenTimer.stop();

  Eigen::Vector3d p_WS_Okvis = T_WS.r();
  Eigen::Quaterniond q_WS_Okvis = T_WS.q();
  okvis::SpeedAndBiases speedAndBiasOkvis = sb;
  Eigen::Matrix<double, 15, 1> covDiagonalOkvis = covOkvis.diagonal();
  Eigen::Matrix<double, 15, 1> sqrtCovDiagonalOkvis = covDiagonalOkvis.cwiseSqrt();
  Eigen::Matrix<double, 15, 1> jacobianFirstRowOkvis = jacobianOkvis.row(0).transpose();

  std::cout << "time used by OKVIS Leutenegger forward propagtion with covariance "
            << timeElapsed << std::endl;
  if (verbose) {
    std::cout << "numUsedMeas " << numUsedImuMeasurements << " totalMeas "
              << (int)cpc.get_meas_size() << std::endl;
    std::cout << "q_WS " << q_WS_Okvis.w() << " " << q_WS_Okvis.x() << " "
              << q_WS_Okvis.y() << " " << q_WS_Okvis.z() << std::endl;
    std::cout << "p_WS_W " << p_WS_Okvis.transpose() << std::endl;
    std::cout << "speed and bias " << speedAndBiasOkvis.transpose() << std::endl;
    std::cout << "starting with 0s, cov diagonal sqrt " << std::endl;
    std::cout << sqrtCovDiagonalOkvis.transpose() << std::endl;
    std::cout << "Jacobian diagonal " << std::endl;
    std::cout << vio::superdiagonal(jacobianOkvis).transpose() << std::endl;
    std::cout << "Jacobian first row " << std::endl;
    std::cout << jacobianFirstRowOkvis.transpose() << std::endl;
  }

  /// method 4: simple integration
  Eigen::Matrix<double, 15, 15> covEuler;
  covEuler.setIdentity();

  Sophus::SE3d T_WS0_se3(cpc.get_q_WS0(), cpc.get_p_WS_W0());
  sb = cpc.get_sb0();
  double time_pair[2] = {cpc.get_meas_begin_time().toSec(),
                         cpc.get_meas_end_time().toSec()};
  std::vector<Eigen::Matrix<double, 7, 1>,
              Eigen::aligned_allocator<Eigen::Matrix<double, 7, 1>>>
      measurements = cpc.get_imu_measurement_vector();

  Eigen::Matrix<double, 6, 1> gwomegaw;
  gwomegaw.setZero();
  gwomegaw.head<3>() = Eigen::Vector3d(0, 0, -cpc.get_g());

  Eigen::Matrix<double, 12, 1> q_n_aw_babw = cpc.get_q_n_aw_babw();

  Sophus::SE3d T_WS1_se3;
  Eigen::Matrix<double, 3, 1> v_WS1_Euler;
  okvis::timing::Timer simpleTimer("simple", false);
  okvis::ceres::ode::predictStates(T_WS0_se3, sb, time_pair, measurements,
                                   gwomegaw, q_n_aw_babw, &T_WS1_se3,
                                   &v_WS1_Euler, &covEuler, cpc.get_vTgTsTa());
  timeElapsed = simpleTimer.stop();

  std::cout << "time used by 1st order propagtion with covariance "
            << timeElapsed << std::endl;

  Eigen::Vector3d p_WS_Euler = T_WS1_se3.translation();
  Eigen::Quaterniond q_WS_Euler = T_WS1_se3.unit_quaternion();
  okvis::SpeedAndBiases speedAndBiasEuler = sb;
  speedAndBiasEuler.head<3>() = v_WS1_Euler;
  Eigen::Matrix<double, 15, 1> covDiagonalEuler = covEuler.diagonal();
  Eigen::Matrix<double, 15, 1> sqrtCovDiagonalEuler = covDiagonalEuler.cwiseSqrt();
  Eigen::Matrix<double, 15, 1> sqrtCovDiagonalPermutated = sqrtCovDiagonalEuler;
  sqrtCovDiagonalPermutated.segment<3>(3) = sqrtCovDiagonalEuler.segment<3>(6);
  sqrtCovDiagonalPermutated.segment<3>(6) = sqrtCovDiagonalEuler.segment<3>(3);
  sqrtCovDiagonalPermutated.segment<3>(12) = sqrtCovDiagonalEuler.segment<3>(9);
  sqrtCovDiagonalPermutated.segment<3>(9) = sqrtCovDiagonalEuler.segment<3>(12);
  sqrtCovDiagonalEuler = sqrtCovDiagonalPermutated;
  if (verbose) {
    std::cout << "q_WS " << q_WS_Euler.w() << " " << q_WS_Euler.x() << " " << q_WS_Euler.y()
              << " " << q_WS_Euler.z() << std::endl;
    std::cout << "p_WS_W " << p_WS_Euler.transpose() << std::endl;
    std::cout << "speed and bias " << speedAndBiasEuler.transpose() << std::endl;
    std::cout << "cov diagonal sqrt " << std::endl;
    std::cout << sqrtCovDiagonalEuler.transpose() << std::endl;
  }

  // RK4 vs Trapezoid state
  EXPECT_TRUE(std::fabs(q_WS_Trapezoid.w() - q_WS_RK4.w()) < 1e-5 &&
              std::fabs(q_WS_Trapezoid.x() - q_WS_RK4.x()) < 1e-5 &&
              std::fabs(q_WS_Trapezoid.y() - q_WS_RK4.y()) < 1e-5 &&
              std::fabs(q_WS_Trapezoid.z() - q_WS_RK4.z()) < 1e-5);
  EXPECT_LT((p_WS_W_RK4 - p_WS_Trapezoid).norm(), 50);
  EXPECT_LT((speedAndBiasRK4 - speedAndBiasTrapezoid).norm(), 2);

  // RK4 vs Euler state
  EXPECT_TRUE(std::fabs(q_WS_Euler.w() - q_WS_RK4.w()) < 5e-2 &&
              std::fabs(q_WS_Euler.x() - q_WS_RK4.x()) < 5e-2 &&
              std::fabs(q_WS_Euler.y() - q_WS_RK4.y()) < 5e-2 &&
              std::fabs(q_WS_Euler.z() - q_WS_RK4.z()) < 5e-2);
  EXPECT_LT((p_WS_W_RK4 - p_WS_Euler).norm(), 2000);
  EXPECT_LT((speedAndBiasRK4 - speedAndBiasEuler).norm(), 50);

  // RK4 vs Trapezoid covariance
  EXPECT_LT((sqrtCovDiagonalTrapezoid.head<3>() - sqrtCovDiagonalRK4.head<3>()).norm(), 300);
  EXPECT_LT((sqrtCovDiagonalTrapezoid.segment<3>(3) - sqrtCovDiagonalRK4.segment<3>(3)).norm(),
            2e-2);
  EXPECT_LT((sqrtCovDiagonalTrapezoid.segment<3>(6) - sqrtCovDiagonalRK4.segment<3>(6)).norm(),
            10);

  EXPECT_LT((sqrtCovDiagonalTrapezoid.segment<3>(9) - sqrtCovDiagonalRK4.segment<3>(9)).norm(),
            1e-6);
  EXPECT_LT((sqrtCovDiagonalTrapezoid.segment<3>(12) - sqrtCovDiagonalRK4.segment<3>(12)).norm(),
            1e-6);

  EXPECT_LT((sqrtCovDiagonalTrapezoid.segment<9>(15) - sqrtCovDiagonalRK4.segment<9>(15)).norm(),
            1e-6);
  EXPECT_LT((sqrtCovDiagonalTrapezoid.segment<9>(24) - sqrtCovDiagonalRK4.segment<9>(24)).norm(),
            1e-6);
  EXPECT_LT((sqrtCovDiagonalTrapezoid.segment<9>(33) - sqrtCovDiagonalRK4.segment<9>(33)).norm(),
            1e-6);

  // RK4 vs Trapezoid jacobian
  EXPECT_LT((jacobianFirstRowRK4.head<3>() - jacobianFirstRowTrapezoid.head<3>()).norm() /
                jacobianFirstRowRK4.head<3>().norm(),
            1e-3);
  EXPECT_LT((jacobianFirstRowRK4.segment<6>(3) - jacobianFirstRowTrapezoid.segment<6>(3)).norm() /
                jacobianFirstRowRK4.segment<6>(3).norm(),
            1e-3);
  EXPECT_LT((jacobianFirstRowRK4.segment<6>(9) - jacobianFirstRowTrapezoid.segment<6>(9)).norm() /
                jacobianFirstRowRK4.segment<6>(9).norm(),
            1e-3);
  EXPECT_LT((jacobianFirstRowRK4.segment<9>(15) - jacobianFirstRowTrapezoid.segment<9>(15)).norm() /
                jacobianFirstRowRK4.segment<9>(15).norm(),
            5e-3);
  EXPECT_LT((jacobianFirstRowRK4.segment<9>(24) - jacobianFirstRowTrapezoid.segment<9>(24)).norm() /
                jacobianFirstRowRK4.segment<9>(24).norm(),
            1e-3);
  EXPECT_LT((jacobianFirstRowRK4.segment<9>(33) - jacobianFirstRowTrapezoid.segment<9>(33)).norm() /
                jacobianFirstRowRK4.segment<9>(33).norm(),
            1e-3);

  // RK4 vs Euler covariance
  EXPECT_LT((sqrtCovDiagonalEuler.head<3>() - sqrtCovDiagonalRK4.head<3>()).norm() /
                sqrtCovDiagonalEuler.head<3>().norm(),
            8e-2);
  EXPECT_LT((sqrtCovDiagonalEuler.segment<3>(3) - sqrtCovDiagonalRK4.segment<3>(3)).norm() /
                sqrtCovDiagonalEuler.segment<3>(3).norm(),
            5e-2);
  EXPECT_LT((sqrtCovDiagonalEuler.segment<3>(6) - sqrtCovDiagonalRK4.segment<3>(6)).norm() /
                sqrtCovDiagonalEuler.segment<3>(6).norm(),
            0.1);

  EXPECT_LT((sqrtCovDiagonalEuler.segment<3>(9) - sqrtCovDiagonalRK4.segment<3>(9)).norm(),
            1e-6);
  EXPECT_LT((sqrtCovDiagonalEuler.segment<3>(12) - sqrtCovDiagonalRK4.segment<3>(12)).norm(),
            1e-6);
}

TEST(ImuFactor, dokvis_dri) {
  okvis::kinematics::Transformation T_WB;
  T_WB.setRandom();
  Eigen::Vector3d v_WB;
  v_WB.setRandom();
  Eigen::Matrix<double, 15, 15> permutator = gtsam::dokvis_drightinvariant(T_WB, v_WB);
  Eigen::Matrix<double, 15, 15> permutator_inv =
      gtsam::drightinvariant_dokvis(T_WB, v_WB);
  EXPECT_LT(((permutator * permutator_inv) -
             Eigen::Matrix<double, 15, 15>::Identity())
                .lpNorm<Eigen::Infinity>(),
            1e-8);
}

TEST(ImuFactor, dokvis_dforster) {
  okvis::kinematics::Transformation T_WB;
  T_WB.setRandom();
  Eigen::Matrix<double, 15, 15> permutator = gtsam::dokvis_dforster(T_WB.C());
  Eigen::Matrix<double, 15, 15> permutator_inv =
      gtsam::dforster_dokvis(T_WB.C());
  EXPECT_LT(((permutator * permutator_inv) -
             Eigen::Matrix<double, 15, 15>::Identity())
                .lpNorm<Eigen::Infinity>(),
            1e-8);
}

void propagateRiCovarianceFromZero(
    std::shared_ptr<const Eigen::Vector3d> p_WS0_ptr,
    std::shared_ptr<const Eigen::Vector3d> v_WS0_ptr, bool verbose) {
  srand((unsigned int)time(0));
  CovPropConfig cpc(false, true);
  Eigen::Vector3d initialPosition = cpc.get_p_WS_W0();
  if (p_WS0_ptr) {
    initialPosition = *p_WS0_ptr;
  }
  Eigen::Vector3d initialVelocity = cpc.get_sb0().head<3>();
  if (v_WS0_ptr) {
    initialVelocity = *v_WS0_ptr;
  }
  if (verbose) {
    std::cout << "Initial position " << initialPosition.transpose()
              << " velocity " << initialVelocity.transpose() << "\n";
  }
  okvis::kinematics::Transformation T_WS =
      okvis::kinematics::Transformation(initialPosition, cpc.get_q_WS0());
  okvis::SpeedAndBiases sb = cpc.get_sb0();
  sb.head<3>() = initialVelocity;
  okvis::timing::Timer okvisTimer("OKVIS Corrected", false);
  Eigen::Matrix<double, 15, 15> covOkvis, jacobianOkvis;
  covOkvis.setZero();
  jacobianOkvis.setIdentity();

  int numUsedImuMeasurements =
      okvis::ceres::ImuError::propagation(
          cpc.get_imu_measurements(), cpc.get_imu_params(), T_WS, sb,
          cpc.get_meas_begin_time(), cpc.get_meas_end_time(), &covOkvis,
          &jacobianOkvis);
  double timeElapsed = okvisTimer.stop();

  Eigen::Vector3d p_WS_Okvis = T_WS.r();
  Eigen::Quaterniond q_WS_Okvis = T_WS.q();
  okvis::SpeedAndBiases speedAndBiasOkvis = sb;
  Eigen::Matrix<double, 15, 1> covDiagonalOkvis = covOkvis.diagonal();
  Eigen::Matrix<double, 15, 1> sqrtCovDiagonalOkvis = covDiagonalOkvis.cwiseSqrt();

  if (verbose) {
    std::cout
        << "time used by OKVIS Leutenegger corrected forward propagtion with "
           "0 initial covariance: "
        << timeElapsed << "\n";
    std::cout << "numUsedMeas " << numUsedImuMeasurements << " totalMeas "
              << (int)cpc.get_meas_size() << std::endl;
    std::cout << "q_WS " << q_WS_Okvis.w() << " " << q_WS_Okvis.x() << " "
              << q_WS_Okvis.y() << " " << q_WS_Okvis.z() << std::endl;
    std::cout << "p_WS_W " << p_WS_Okvis.transpose() << std::endl;
    std::cout << "speed and bias " << speedAndBiasOkvis.transpose() << std::endl;
    std::cout << "cov diagonal sqrt " << std::endl;
    std::cout << sqrtCovDiagonalOkvis.transpose() << std::endl;
    std::cout << "Jacobian super diagonal " << std::endl;
    std::cout << vio::superdiagonal(jacobianOkvis).transpose() << std::endl;
    std::cout << "cov\n" << covOkvis << "\njac\n" << jacobianOkvis << "\n";
  }

  // Right invariant error method.
  Eigen::Matrix<double, 15, 15> covariance_ri = Eigen::Matrix<double, 15, 15>::Zero();
  Eigen::Matrix<double, 15, 15> jacobian_ri;

  okvis::kinematics::Transformation T_WS_ri(initialPosition, cpc.get_q_WS0());
  Eigen::Matrix<double, 9, 1> speedAndBias_ri = cpc.get_sb0();
  speedAndBias_ri.head<3>() = initialVelocity;
  Eigen::Vector3d v_WS_ri = initialVelocity;
  ImuErrorModel<double> iem(speedAndBias_ri.tail<6>());

  okvis::timing::Timer riTimer("right invariant error", false);
  numUsedImuMeasurements = okvis::ImuOdometry::propagationRightInvariantError(
      cpc.get_imu_measurements(), cpc.get_imu_params(), T_WS_ri, v_WS_ri, iem,
      cpc.get_meas_begin_time(), cpc.get_meas_end_time(), &covariance_ri,
      &jacobian_ri);
  double riElapsed = riTimer.stop();

  speedAndBias_ri.head<3>() = v_WS_ri;

  EXPECT_LT((p_WS_Okvis - T_WS_ri.r()).norm() / p_WS_Okvis.norm(), 5e-3);
  check_q_near(q_WS_Okvis, T_WS_ri.q(), 1e-5);
  EXPECT_LT((speedAndBiasOkvis - speedAndBias_ri).head<3>().norm() /
                speedAndBiasOkvis.norm(),
            1e-3);
  EXPECT_LT((speedAndBiasOkvis - speedAndBias_ri).tail<6>().norm(), 1e-6);

  Eigen::Matrix<double, 15, 15> permutator_j =
      gtsam::dokvis_drightinvariant(T_WS_ri, v_WS_ri);

  Eigen::Matrix<double, 15, 15> permutator_i_inv = gtsam::drightinvariant_dokvis(
      okvis::kinematics::Transformation(initialPosition, cpc.get_q_WS0()),
      initialVelocity);

  jacobian_ri = permutator_j * jacobian_ri * permutator_i_inv;
  covariance_ri = permutator_j * covariance_ri * permutator_j.transpose();

  if (verbose) {
    std::cout << "time used by right invariant error forward propagtion with "
                 "0 initial covariance: "
              << riElapsed << "\n";
    std::cout << "cov diag sqrt\n"
              << covariance_ri.diagonal().cwiseSqrt().transpose() << "\n";
    std::cout << "Jacobian super diagonal\n"
              << vio::superdiagonal(jacobian_ri).transpose() << "\n";
    std::cout << "cov\n" << covariance_ri << "\njac\n" << jacobian_ri << "\n";
  }

  std::cout << "Check jacobians of okvis and right invariant error\n";
  checkSelectiveRatio(jacobianOkvis.topLeftCorner<15, 9>(),
                      jacobian_ri.topLeftCorner<15, 9>(), 0.09, 1e-3);
  checkSelectiveRatio(jacobianOkvis.topRightCorner<15, 6>(),
                      jacobian_ri.topRightCorner<15, 6>(), 0.1, 1e-3);

  std::cout << "Check P, Q, V covariance of okvis and right invariant error\n";
  checkSelectiveRatio(covOkvis.topLeftCorner<15, 9>(),
                      covariance_ri.topLeftCorner<15, 9>(), 0.35, 3, 1);
  std::cout << "Check Bg Ba covariance of okvis and right invariant error\n";
  checkSelectiveRatio(covOkvis.topRightCorner<15, 6>(),
                      covariance_ri.topRightCorner<15, 6>(), 2.5e-1, 10, 5);
}

TEST(ImuOdometry, RiCovariancePropagationFromZero) {
  propagateRiCovarianceFromZero(nullptr, nullptr, true);
}

TEST(ImuOdometry, RiCovariancePropagationWithLargeP) {
  // Propagation from a large position does not seem to worsen covariance or
  // jacobian.
  std::shared_ptr<const Eigen::Vector3d> positionPtr(
      new Eigen::Vector3d(Eigen::Vector3d::Random() * 1000));
  propagateRiCovarianceFromZero(positionPtr, nullptr, true);
}

TEST(ImuOdometry, RiCovariancePropagationWithLargeV) {
  // Propagation from a large velocity does not seem to worsen covariance or
  // jacobian.
  std::shared_ptr<const Eigen::Vector3d> velocityPtr(
      new Eigen::Vector3d(Eigen::Vector3d::Random() * 1000));
  propagateRiCovarianceFromZero(nullptr, velocityPtr, true);
}

TEST(ImuOdometry, InitPoseFromImu) {
  using namespace okvis;
  Eigen::Vector3d acc_B(2, -1, 3);
  Eigen::Vector3d e_acc = acc_B.normalized();

  // align with ez_W:
  Eigen::Vector3d ez_W(0.0, 0.0, 1.0);  // Negative gravity direction, i.e., Z
                                        // direction, in the local frame
  Eigen::Matrix<double, 6, 1> poseIncrement;
  poseIncrement.head<3>() = Eigen::Vector3d::Zero();
  poseIncrement.tail<3>() = ez_W.cross(e_acc).normalized();
  double angle = std::acos(ez_W.transpose() * e_acc);
  poseIncrement.tail<3>() *= angle;
  okvis::kinematics::Transformation T_WS;
  T_WS.setIdentity();
  T_WS.oplus(-poseIncrement);
  Eigen::Vector3d transVec = T_WS.q()._transformVector(acc_B);
  transVec.normalize();  // predicted negative gravity

  ASSERT_LT((transVec - ez_W).norm(), 1e-8);
}
