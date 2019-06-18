
#include <okvis/ceres/ImuError.hpp>
#include <okvis/timing/Timer.hpp>
#include "okvis/IMUOdometry.h"
#include "vio/IMUErrorModel.h"
#include "vio/rand_sampler.h"

#include <gtest/gtest.h>

// compare RungeKutta and Euler forward and backward integration
// It is observed that RungeKutta may perform worse than Euler's implementation
// for forward + backward propagation
TEST(IMUOdometry, BackwardIntegration) {
  using namespace Eigen;
  using namespace okvis;
  srand((unsigned int)time(0));

  double g = 9.81;

  double sigma_g_c = 5e-2;
  double sigma_a_c = 3e-2;
  double sigma_gw_c = 7e-3;
  double sigma_aw_c = 2e-3;
  double dt = 0.1;
  Eigen::Vector3d p_WS_W0(0, 0, 0);
  Eigen::Quaterniond q_WS0(1, 0, 0, 0);
  okvis::SpeedAndBiases sb0;
  sb0 << 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;
  Eigen::Matrix<double, 27, 1> vTgTsTa =
      1e-2 * (Eigen::Matrix<double, 27, 1>::Random());
  for (int jack = 0; jack < 3; ++jack) {
    vTgTsTa[jack * 4] += 1;
    vTgTsTa[jack * 4 + 18] += 1;
  }
  Eigen::Matrix<double, okvis::ceres::ode::OdoErrorStateDim,
                okvis::ceres::ode::OdoErrorStateDim>* P_ptr = 0;
  Eigen::Matrix<double, okvis::ceres::ode::OdoErrorStateDim,
                okvis::ceres::ode::OdoErrorStateDim>* F_tot_ptr = 0;

  okvis::ImuMeasurementDeque imuMeasurements;
  for (int jack = 0; jack < 1000; ++jack) {
    Eigen::Vector3d gyr = Vector3d::Random();  // range from [-1, 1]
    Eigen::Vector3d acc = Vector3d::Random();
    acc[2] = 10 + vio::gauss_rand(0.0, 0.1);
    imuMeasurements.push_back(
        ImuMeasurement(okvis::Time(jack * dt), ImuSensorReadings(gyr, acc)));
  }

  // BUNDLE_INTEGRATION
  okvis::ImuParameters imuParams;
  imuParams.g_max = 7.8;
  imuParams.a_max = 176;
  imuParams.sigma_g_c = sigma_g_c;
  imuParams.sigma_a_c = sigma_a_c;
  imuParams.sigma_gw_c = sigma_gw_c;
  imuParams.sigma_aw_c = sigma_aw_c;
  imuParams.g = g;

  Eigen::Vector3d p_WS_W = p_WS_W0;
  Eigen::Quaterniond q_WS = q_WS0;
  okvis::SpeedAndBiases sb = sb0;
  okvis::kinematics::Transformation T_WB(p_WS_W0, q_WS0);

  IMUOdometry::propagation_RungeKutta(
      imuMeasurements, imuParams, T_WB, sb, vTgTsTa,
      imuMeasurements.begin()->timeStamp, imuMeasurements.rbegin()->timeStamp);

  IMUOdometry::propagationBackward_RungeKutta(
      imuMeasurements, imuParams, T_WB, sb, vTgTsTa,
      imuMeasurements.rbegin()->timeStamp, imuMeasurements.begin()->timeStamp);
  p_WS_W = T_WB.r();
  q_WS = T_WB.q();

  Eigen::Vector3d p_WS_W1 = p_WS_W;
  Eigen::Quaterniond q_WS1 = q_WS;
  okvis::SpeedAndBiases sb1 = sb;

  // DETAILED INTEGRATION
  p_WS_W = p_WS_W0;
  q_WS = q_WS0;
  sb = sb0;

  auto iterLast = imuMeasurements.begin();
  for (auto iter = imuMeasurements.begin(); iter != imuMeasurements.end();
       ++iter) {
    if (iter == imuMeasurements.begin()) continue;
    okvis::ceres::ode::integrateOneStep_RungeKutta(
        iterLast->measurement.gyroscopes, iterLast->measurement.accelerometers,
        iter->measurement.gyroscopes, iter->measurement.accelerometers, g,
        sigma_g_c, sigma_a_c, sigma_gw_c, sigma_aw_c, dt, p_WS_W, q_WS, sb,
        vTgTsTa);
    iterLast = iter;
  }

  //    std::cout <<"states after forward integration: q_WS, p_WS_W, speed and
  //    bias"<<std::endl; std::cout<< q_WS.w()<<" "<<q_WS.x()<<" "<<q_WS.y()<<"
  //    "<<q_WS.z()<<std::endl; std::cout<< p_WS_W.transpose() << std::endl;
  //    std::cout<< sb.transpose()<< std::endl;

  // backward
  auto iterRLast = imuMeasurements.rbegin();
  for (auto iterR = imuMeasurements.rbegin(); iterR != imuMeasurements.rend();
       ++iterR) {
    if (iterR == imuMeasurements.rbegin()) continue;
    okvis::ceres::ode::integrateOneStepBackward_RungeKutta(
        iterR->measurement.gyroscopes, iterR->measurement.accelerometers,
        iterRLast->measurement.gyroscopes,
        iterRLast->measurement.accelerometers, g, sigma_g_c, sigma_a_c,
        sigma_gw_c, sigma_aw_c, dt, p_WS_W, q_WS, sb, vTgTsTa);
    iterRLast = iterR;
  }

  Eigen::Vector3d p_WS_W2 = p_WS_W;
  Eigen::Quaterniond q_WS2 = q_WS;
  okvis::SpeedAndBiases sb2 = sb;

  // NUMERICAL_INTEGRATION EULER

  sb = sb0;
  T_WB = kinematics::Transformation(p_WS_W0, q_WS0);
  Eigen::Vector3d tempV_WS = sb.head<3>();
  IMUErrorModel<double> iem(sb.tail<6>(), vTgTsTa);
  IMUOdometry::propagation(imuMeasurements, imuParams, T_WB, tempV_WS, iem,
                           imuMeasurements.begin()->timeStamp,
                           imuMeasurements.rbegin()->timeStamp);

  IMUOdometry::propagationBackward(imuMeasurements, imuParams, T_WB, tempV_WS,
                                   iem, imuMeasurements.rbegin()->timeStamp,
                                   imuMeasurements.begin()->timeStamp);
  p_WS_W = T_WB.r();
  q_WS = T_WB.q();
  sb.head<3>() = tempV_WS;

  Eigen::Vector3d p_WS_W3 = p_WS_W;
  Eigen::Quaterniond q_WS3 = q_WS;
  okvis::SpeedAndBiases sb3 = sb;

  // two methods results the same?
  ASSERT_TRUE(fabs(q_WS2.w() - q_WS1.w()) < 1e-8 &&
              fabs(q_WS2.x() - q_WS1.x()) < 1e-8 &&
              fabs(q_WS2.y() - q_WS1.y()) < 1e-8 &&
              fabs(q_WS2.z() - q_WS1.z()) < 1e-8);
  ASSERT_LT((p_WS_W1 - p_WS_W2).norm(), 1e-8);
  ASSERT_LT((sb1 - sb2).norm(), 1e-8);
  // Runge Kutta return to the starting position ?
  ASSERT_TRUE(fabs(q_WS0.w() - q_WS1.w()) < 1e-8 &&
              fabs(q_WS0.x() - q_WS1.x()) < 1e-8 &&
              fabs(q_WS0.y() - q_WS1.y()) < 1e-8 &&
              fabs(q_WS0.z() - q_WS1.z()) < 1e-8);

  EXPECT_LT(((sb1 - sb0).head<3>().norm()), 2e-2);
  EXPECT_LT((p_WS_W1 - p_WS_W0).norm(), 0.8);

  // Euler return to the starting position ?
  ASSERT_TRUE(fabs(q_WS0.w() - q_WS3.w()) < 1e-6 &&
              fabs(q_WS0.x() - q_WS3.x()) < 1e-6 &&
              fabs(q_WS0.y() - q_WS3.y()) < 1e-6 &&
              fabs(q_WS0.z() - q_WS3.z()) < 1e-6);

  EXPECT_LT(((sb3 - sb0).head<3>().norm()), 1e-2);
  EXPECT_LT((p_WS_W3 - p_WS_W0).norm(), 1e-2);
}

TEST(Eigen, SuperDiagonal) {
  Eigen::MatrixXd M = Eigen::MatrixXd::Random(3, 5);
  ASSERT_LT((vio::superdiagonal(M) - vio::subdiagonal(M.transpose())).norm(),
            1e-9);

  M = Eigen::MatrixXd::Random(4, 4);
  ASSERT_LT((vio::superdiagonal(M) - vio::subdiagonal(M.transpose())).norm(),
            1e-9);

  M = Eigen::MatrixXd::Random(5, 3);
  ASSERT_LT((vio::superdiagonal(M) - vio::subdiagonal(M.transpose())).norm(),
            1e-9);
}
/// test and compare the propagation for both states and covariance by both the
/// classic RK4 and okvis's state transition method
TEST(IMUOdometry, IMUCovariancePropagation) {
  using namespace Eigen;
  using namespace okvis;
  bool bVarianceForShapeMatrices =
      false;             // use positive variance for elements in Tg Ts Ta?
  bool bVerbose = true;  // print the covariance and jacobian results
  bool bNominalTgTsTa =
      true;  // use nominal values for Tg Ts Ta, i.e., identity, zero, identity?
  // if false, add noise to them, therefore,
  // set true if testing a method that does not support Tg Ts Ta model
  srand((unsigned int)time(0));
  double g = 9.81;
  double sigma_g_c = 5e-2;
  double sigma_a_c = 3e-2;
  double sigma_gw_c = 7e-3;
  double sigma_aw_c = 2e-3;
  const double dt = 0.1;
  Eigen::Vector3d p_WS_W0(vio::gauss_rand(0, 1), vio::gauss_rand(0, 1),
                          vio::gauss_rand(0, 1));
  Eigen::Quaterniond q_WS0(vio::gauss_rand(0, 1), vio::gauss_rand(0, 1),
                           vio::gauss_rand(0, 1), vio::gauss_rand(0, 1));
  q_WS0.normalize();

  okvis::SpeedAndBiases sb0;
  sb0 << 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;

  okvis::ImuMeasurementDeque imuMeasurements;
  for (int jack = 0; jack < 1000; ++jack) {
    Eigen::Vector3d gyr = Vector3d::Random();  // range from [-1, 1]
    Eigen::Vector3d acc = Vector3d::Random();
    acc[2] = 10 + vio::gauss_rand(0.0, 0.1);
    imuMeasurements.push_back(
        ImuMeasurement(okvis::Time(jack * dt), ImuSensorReadings(gyr, acc)));
  }
  // if to test Leutenegger's propagation method, set vTgTsTa as nominal values
  Eigen::Matrix<double, 27, 1> vTgTsTa = Eigen::Matrix<double, 27, 1>::Zero();
  // otherwise, random initialization is OK
  if (!bNominalTgTsTa)
    vTgTsTa = 1e-2 * (Eigen::Matrix<double, 27, 1>::Random());

  for (int jack = 0; jack < 3; ++jack) {
    vTgTsTa[jack * 4] += 1;
    vTgTsTa[jack * 4 + 18] += 1;
  }

  /// method 1: RK4
  Eigen::Vector3d p_WS_W = p_WS_W0;
  Eigen::Quaterniond q_WS = q_WS0;
  okvis::SpeedAndBiases sb = sb0;

  Eigen::Matrix<double, okvis::ceres::ode::OdoErrorStateDim,
                okvis::ceres::ode::OdoErrorStateDim>
      P1;
  P1.setIdentity();
  if (!bVarianceForShapeMatrices) P1.bottomRightCorner<27, 27>().setZero();
  Eigen::Matrix<double, okvis::ceres::ode::OdoErrorStateDim,
                okvis::ceres::ode::OdoErrorStateDim>
      F_tot;
  F_tot.setIdentity();

  timing::Timer RK4Timer("RK4", false);
  auto iterLast = imuMeasurements.begin();
  for (auto iter = imuMeasurements.begin(); iter != imuMeasurements.end();
       ++iter) {
    if (iter == imuMeasurements.begin()) continue;
    okvis::ceres::ode::integrateOneStep_RungeKutta(
        iterLast->measurement.gyroscopes, iterLast->measurement.accelerometers,
        iter->measurement.gyroscopes, iter->measurement.accelerometers, g,
        sigma_g_c, sigma_a_c, sigma_gw_c, sigma_aw_c, dt, p_WS_W, q_WS, sb,
        vTgTsTa, &P1, &F_tot);
    iterLast = iter;
  }
  double timeElapsed = RK4Timer.stop();

  Eigen::Vector3d p_WS_W1 = p_WS_W;
  Eigen::Quaterniond q_WS1 = q_WS;
  okvis::SpeedAndBiases sb1 = sb;
  Eigen::Matrix<double, 42, 1> P1Diag = P1.diagonal();
  Eigen::Matrix<double, 42, 1> sqrtDiagCov1 = P1Diag.cwiseSqrt();
  Eigen::Matrix<double, 42, 1> firstRow1 = F_tot.row(0).transpose();
  std::cout << "time used by RK4 forward propagtion with covariance "
            << timeElapsed << std::endl;

  if (bVerbose) {
    std::cout << "q_WS " << q_WS1.w() << " " << q_WS1.x() << " " << q_WS1.y()
              << " " << q_WS1.z() << std::endl;
    std::cout << "p_WS_W " << p_WS_W1.transpose() << std::endl;
    std::cout << "speed and bias " << sb1.transpose() << std::endl;

    std::cout << "cov(r_s^w, \\phi^w, v_s^w, b_g, b_a, T_g, T_s, T_a), its "
                 "diagonal sqrt "
              << std::endl;
    std::cout << sqrtDiagCov1.transpose() << std::endl;
    std::cout << "Jacobian superdiagonal " << std::endl;
    std::cout << vio::superdiagonal(F_tot).transpose() << std::endl;
    std::cout << "Jacobian first row " << std::endl;
    std::cout << firstRow1.transpose() << std::endl;
  }
  /// method 2 : huai implementation of trapezoid rules
  okvis::ImuParameters imuParams;
  imuParams.g_max = 7.8;
  imuParams.a_max = 176;
  imuParams.sigma_g_c = sigma_g_c;
  imuParams.sigma_a_c = sigma_a_c;
  imuParams.sigma_gw_c = sigma_gw_c;
  imuParams.sigma_aw_c = sigma_aw_c;
  imuParams.g = g;

  okvis::kinematics::Transformation T_WS(p_WS_W0, q_WS0);
  sb = sb0;
  timing::Timer okvisTimer("okvis", false);

  Eigen::Matrix<double, 42, 42> P2;
  P2.setIdentity();
  if (!bVarianceForShapeMatrices) P2.bottomRightCorner<27, 27>().setZero();
  F_tot.setIdentity();

#if 0  // this is the same as the first case in effect
    int numUsedImuMeasurements = IMUOdometry::propagation_RungeKutta(imuMeasurements,
                                                                     imuParams,
                                                                     T_WS, sb, vTgTsTa, imuMeasurements.begin()->timeStamp,
                                                                     imuMeasurements.rbegin()->timeStamp, &P2, &F_tot);
#else
  Eigen::Vector3d tempV_WS = sb.head<3>();
  IMUErrorModel<double> tempIEM(sb.tail<6>(), vTgTsTa);
  int numUsedImuMeasurements = IMUOdometry::propagation(
      imuMeasurements, imuParams, T_WS, tempV_WS, tempIEM,
      imuMeasurements.begin()->timeStamp, imuMeasurements.rbegin()->timeStamp,
      &P2, &F_tot);
  sb.head<3>() = tempV_WS;

#endif
  timeElapsed = okvisTimer.stop();

  Eigen::Vector3d p_WS_W2 = T_WS.r();
  Eigen::Quaterniond q_WS2 = T_WS.q();
  okvis::SpeedAndBiases sb2 = sb;
  Eigen::Matrix<double, 42, 1> P2Diag = P2.diagonal();
  Eigen::Matrix<double, 42, 1> sqrtDiagCov2 = P2Diag.cwiseSqrt();
  Eigen::Matrix<double, 42, 1> firstRow2 = F_tot.row(0).transpose();

  std::cout
      << "time used by huai trapezoid rule forward propagtion with covariance "
      << timeElapsed << std::endl;
  if (bVerbose) {
    std::cout << "numUsedMeas " << numUsedImuMeasurements << " totalMeas "
              << (int)imuMeasurements.size() << std::endl;
    std::cout << "q_WS " << q_WS2.w() << " " << q_WS2.x() << " " << q_WS2.y()
              << " " << q_WS2.z() << std::endl;
    std::cout << "p_WS_W " << p_WS_W2.transpose() << std::endl;
    std::cout << "speed and bias " << sb2.transpose() << std::endl;
    std::cout << "cov diagonal sqrt " << std::endl;
    std::cout << sqrtDiagCov2.transpose() << std::endl;
    std::cout << "Jacobian diagonal " << std::endl;
    std::cout << vio::superdiagonal(F_tot).transpose() << std::endl;
    std::cout << "Jacobian first row " << std::endl;
    std::cout << firstRow2.transpose() << std::endl;
  }

  /// method 3 : okvis propagation leutenegger's implementation
  T_WS = okvis::kinematics::Transformation(p_WS_W0, q_WS0);
  sb = sb0;
  timing::Timer leutenTimer("leutenegger", false);

  Eigen::Matrix<double, 15, 15> leutenP;
  leutenP.setIdentity();
  Eigen::Matrix<double, 15, 15> leutenF;
  leutenF.setIdentity();

  numUsedImuMeasurements = okvis::ceres::ImuError::propagation(
      imuMeasurements, imuParams, T_WS, sb, imuMeasurements.begin()->timeStamp,
      imuMeasurements.rbegin()->timeStamp, &leutenP, &leutenF);
  timeElapsed = leutenTimer.stop();

  Eigen::Vector3d leuten_p_WS_W = T_WS.r();
  Eigen::Quaterniond leuten_q_WS = T_WS.q();
  okvis::SpeedAndBiases leuten_sb = sb;
  Eigen::Matrix<double, 15, 1> leuten_PDiag = leutenP.diagonal();
  Eigen::Matrix<double, 15, 1> leuten_sqrtDiagCov = leuten_PDiag.cwiseSqrt();
  Eigen::Matrix<double, 15, 1> leuten_firstRow = leutenF.row(0).transpose();

  std::cout << "time used by leutenegger forward propagtion with covariance "
            << timeElapsed << std::endl;
  if (bVerbose) {
    std::cout << "numUsedMeas " << numUsedImuMeasurements << " totalMeas "
              << (int)imuMeasurements.size() << std::endl;
    std::cout << "q_WS " << leuten_q_WS.w() << " " << leuten_q_WS.x() << " "
              << leuten_q_WS.y() << " " << leuten_q_WS.z() << std::endl;
    std::cout << "p_WS_W " << leuten_p_WS_W.transpose() << std::endl;
    std::cout << "speed and bias " << leuten_sb.transpose() << std::endl;
    std::cout << "cov diagonal sqrt " << std::endl;
    std::cout << leuten_sqrtDiagCov.transpose() << std::endl;
    std::cout << "Jacobian diagonal " << std::endl;
    std::cout << vio::superdiagonal(leutenF).transpose() << std::endl;
    std::cout << "Jacobian first row " << std::endl;
    std::cout << leuten_firstRow.transpose() << std::endl;
  }

  /// method 4 : okvis propagation leutenegger's implementation corrected by
  /// huai
  T_WS = okvis::kinematics::Transformation(p_WS_W0, q_WS0);
  sb = sb0;
  timing::Timer leutenTimer2("leuteneggerCorrected", false);
  Eigen::Matrix<double, 15, 15> leutenP2, leutenF2;
  leutenP2.setIdentity();
  leutenF2.setIdentity();

  numUsedImuMeasurements =
      okvis::IMUOdometry::propagation_leutenegger_corrected(
          imuMeasurements, imuParams, T_WS, sb,
          imuMeasurements.begin()->timeStamp,
          imuMeasurements.rbegin()->timeStamp, &leutenP2, &leutenF2);
  timeElapsed = leutenTimer2.stop();

  Eigen::Vector3d leuten_p_WS_W2 = T_WS.r();
  Eigen::Quaterniond leuten_q_WS2 = T_WS.q();
  okvis::SpeedAndBiases leuten_sb2 = sb;
  Eigen::Matrix<double, 15, 1> leuten_PDiag2 = leutenP2.diagonal();
  Eigen::Matrix<double, 15, 1> leuten_sqrtDiagCov2 = leuten_PDiag2.cwiseSqrt();
  Eigen::Matrix<double, 15, 1> leuten_firstRow2 = leutenF2.row(0).transpose();

  std::cout << "time used by leutenegger corrected forward propagtion with "
               "covariance "
            << timeElapsed << std::endl;
  if (bVerbose) {
    std::cout << "numUsedMeas " << numUsedImuMeasurements << " totalMeas "
              << (int)imuMeasurements.size() << std::endl;
    std::cout << "q_WS " << leuten_q_WS2.w() << " " << leuten_q_WS2.x() << " "
              << leuten_q_WS2.y() << " " << leuten_q_WS2.z() << std::endl;
    std::cout << "p_WS_W " << leuten_p_WS_W2.transpose() << std::endl;
    std::cout << "speed and bias " << leuten_sb2.transpose() << std::endl;
    std::cout << "cov diagonal sqrt " << std::endl;
    std::cout << leuten_sqrtDiagCov2.transpose() << std::endl;
    std::cout << "Jacobian diagonal " << std::endl;
    std::cout << vio::superdiagonal(leutenF2).transpose() << std::endl;
    std::cout << "Jacobian first row " << std::endl;
    std::cout << leuten_firstRow2.transpose() << std::endl;
  }

  /// method 5: simple integration
  Eigen::Matrix<double, 15, 15> P3;
  P3.setIdentity();

  Sophus::SE3d T_s1_to_w(q_WS0, p_WS_W0);
  sb = sb0;
  double time_pair[2] = {imuMeasurements.begin()->timeStamp.toSec(),
                         imuMeasurements.rbegin()->timeStamp.toSec()};
  std::vector<Eigen::Matrix<double, 7, 1> > measurements;
  Eigen::Matrix<double, 7, 1> meas;
  for (auto iter = imuMeasurements.begin(); iter != imuMeasurements.end();
       ++iter) {
    meas << iter->timeStamp.toSec(), iter->measurement.gyroscopes,
        iter->measurement.accelerometers;
    measurements.push_back(meas);
  }

  Eigen::Matrix<double, 6, 1> gwomegaw;
  gwomegaw.setZero();
  Eigen::Vector3d g_W =
      -imuParams.g * Eigen::Vector3d(0, 0, 6371009).normalized();
  gwomegaw.head<3>() = g_W;

  Eigen::Matrix<double, 12, 1> q_n_aw_babw;
  q_n_aw_babw << pow(imuParams.sigma_a_c, 2), pow(imuParams.sigma_a_c, 2),
      pow(imuParams.sigma_a_c, 2), pow(imuParams.sigma_g_c, 2),
      pow(imuParams.sigma_g_c, 2), pow(imuParams.sigma_g_c, 2),
      pow(imuParams.sigma_aw_c, 2), pow(imuParams.sigma_aw_c, 2),
      pow(imuParams.sigma_aw_c, 2), pow(imuParams.sigma_gw_c, 2),
      pow(imuParams.sigma_gw_c, 2), pow(imuParams.sigma_gw_c, 2);

  Sophus::SE3d pred_T_s2_to_w;
  Eigen::Matrix<double, 3, 1> pred_speed_2;
  timing::Timer simpleTimer("simple", false);
  okvis::ceres::ode::predictStates(T_s1_to_w, sb, time_pair, measurements,
                                   gwomegaw, q_n_aw_babw, &pred_T_s2_to_w,
                                   &pred_speed_2, &P3, vTgTsTa);
  timeElapsed = simpleTimer.stop();

  std::cout << "time used by 1st order propagtion with covariance "
            << timeElapsed << std::endl;

  Eigen::Vector3d p_WS_W3 = pred_T_s2_to_w.translation();
  Eigen::Quaterniond q_WS3 = pred_T_s2_to_w.unit_quaternion();
  okvis::SpeedAndBiases sb3 = sb;
  sb3.head<3>() = pred_speed_2;
  Eigen::Matrix<double, 15, 1> P3Diag = P3.diagonal();
  Eigen::Matrix<double, 15, 1> sqrtDiagCov3 = P3Diag.cwiseSqrt();
  Eigen::Matrix<double, 15, 1> tempSqrtDiagCov3 = sqrtDiagCov3;
  tempSqrtDiagCov3.segment<3>(3) = sqrtDiagCov3.segment<3>(6);
  tempSqrtDiagCov3.segment<3>(6) = sqrtDiagCov3.segment<3>(3);
  tempSqrtDiagCov3.segment<3>(12) = sqrtDiagCov3.segment<3>(9);
  tempSqrtDiagCov3.segment<3>(9) = sqrtDiagCov3.segment<3>(12);
  sqrtDiagCov3 = tempSqrtDiagCov3;
  if (bVerbose) {
    std::cout << "q_WS " << q_WS3.w() << " " << q_WS3.x() << " " << q_WS3.y()
              << " " << q_WS3.z() << std::endl;
    std::cout << "p_WS_W " << p_WS_W3.transpose() << std::endl;
    std::cout << "speed and bias " << sb3.transpose() << std::endl;
    std::cout << "cov diagonal sqrt " << std::endl;
    std::cout << sqrtDiagCov3.transpose() << std::endl;
  }
  // 1,2 states
  EXPECT_TRUE(fabs(q_WS2.w() - q_WS1.w()) < 1e-5 &&
              fabs(q_WS2.x() - q_WS1.x()) < 1e-5 &&
              fabs(q_WS2.y() - q_WS1.y()) < 1e-5 &&
              fabs(q_WS2.z() - q_WS1.z()) < 1e-5);
  EXPECT_LT((p_WS_W1 - p_WS_W2).norm(), 50);
  EXPECT_LT((sb1 - sb2).norm(), 2);
  // 1,3 states

  EXPECT_TRUE(fabs(q_WS3.w() - q_WS1.w()) < 5e-2 &&
              fabs(q_WS3.x() - q_WS1.x()) < 5e-2 &&
              fabs(q_WS3.y() - q_WS1.y()) < 5e-2 &&
              fabs(q_WS3.z() - q_WS1.z()) < 5e-2);
  EXPECT_LT((p_WS_W1 - p_WS_W3).norm(), 1500);
  EXPECT_LT((sb1 - sb3).norm(), 30);

  // 1,2 covariance
  EXPECT_LT((sqrtDiagCov2.head<3>() - sqrtDiagCov1.head<3>()).norm(), 300);
  EXPECT_LT((sqrtDiagCov2.segment<3>(3) - sqrtDiagCov1.segment<3>(3)).norm(),
            2e-2);
  EXPECT_LT((sqrtDiagCov2.segment<3>(6) - sqrtDiagCov1.segment<3>(6)).norm(),
            10);

  EXPECT_LT((sqrtDiagCov2.segment<3>(9) - sqrtDiagCov1.segment<3>(9)).norm(),
            1e-6);
  EXPECT_LT((sqrtDiagCov2.segment<3>(12) - sqrtDiagCov1.segment<3>(12)).norm(),
            1e-6);

  EXPECT_LT((sqrtDiagCov2.segment<9>(15) - sqrtDiagCov1.segment<9>(15)).norm(),
            1e-6);
  EXPECT_LT((sqrtDiagCov2.segment<9>(24) - sqrtDiagCov1.segment<9>(24)).norm(),
            1e-6);
  EXPECT_LT((sqrtDiagCov2.segment<9>(33) - sqrtDiagCov1.segment<9>(33)).norm(),
            1e-6);

  // 1,2 jacobian
  EXPECT_LT((firstRow1.head<3>() - firstRow2.head<3>()).norm() /
                firstRow1.head<3>().norm(),
            1e-3);
  EXPECT_LT((firstRow1.segment<6>(3) - firstRow2.segment<6>(3)).norm() /
                firstRow1.segment<6>(3).norm(),
            1e-3);
  EXPECT_LT((firstRow1.segment<6>(9) - firstRow2.segment<6>(9)).norm() /
                firstRow1.segment<6>(9).norm(),
            1e-3);
  EXPECT_LT((firstRow1.segment<9>(15) - firstRow2.segment<9>(15)).norm() /
                firstRow1.segment<9>(15).norm(),
            5e-3);
  EXPECT_LT((firstRow1.segment<9>(24) - firstRow2.segment<9>(24)).norm() /
                firstRow1.segment<9>(24).norm(),
            1e-3);
  EXPECT_LT((firstRow1.segment<9>(33) - firstRow2.segment<9>(33)).norm() /
                firstRow1.segment<9>(33).norm(),
            1e-3);

  // 1,3 covariance
  EXPECT_LT((sqrtDiagCov3.head<3>() - sqrtDiagCov1.head<3>()).norm() /
                sqrtDiagCov3.head<3>().norm(),
            5e-2);
  EXPECT_LT((sqrtDiagCov3.segment<3>(3) - sqrtDiagCov1.segment<3>(3)).norm() /
                sqrtDiagCov3.segment<3>(3).norm(),
            5e-2);
  EXPECT_LT((sqrtDiagCov3.segment<3>(6) - sqrtDiagCov1.segment<3>(6)).norm() /
                sqrtDiagCov3.segment<3>(6).norm(),
            5e-2);

  EXPECT_LT((sqrtDiagCov3.segment<3>(9) - sqrtDiagCov1.segment<3>(9)).norm(),
            1e-6);
  EXPECT_LT((sqrtDiagCov3.segment<3>(12) - sqrtDiagCov1.segment<3>(12)).norm(),
            1e-6);
}

TEST(IMUOdometry, InitPoseFromImu) {
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
