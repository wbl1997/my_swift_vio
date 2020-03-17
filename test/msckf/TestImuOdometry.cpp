
#include <gtest/gtest.h>

#include "msckf/imu/ImuErrorModel.h"
#include <msckf/imu/ImuOdometry.h>
#include <msckf/ImuOdometryLegacy.hpp>

#include <okvis/ceres/ImuError.hpp>
#include <okvis/timing/Timer.hpp>

#include "vio/Sample.h"
#include "vio/eigen_utils.h"

#include "sophus/se3.hpp"

static void check_q_near(const Eigen::Quaterniond& q_WS0,
                         const Eigen::Quaterniond& q_WS1, const double tol) {
  ASSERT_TRUE(
      std::fabs(q_WS0.w() - q_WS1.w()) < tol && std::fabs(q_WS0.x() - q_WS1.x()) < tol &&
      std::fabs(q_WS0.y() - q_WS1.y()) < tol && std::fabs(q_WS0.z() - q_WS1.z()) < tol);
}

static void check_sb_near(const Eigen::Matrix<double, 9, 1>& sb0,
                          const Eigen::Matrix<double, 9, 1>& sb1,
                          const double tol) {
  EXPECT_LT(((sb1 - sb0).norm()), tol);
}

static void check_v_near(const Eigen::Matrix<double, 9, 1>& sb0,
                         const Eigen::Matrix<double, 9, 1>& sb1,
                         const double tol) {
  EXPECT_LT(((sb1 - sb0).head<3>().norm()), tol);
}

static void check_p_near(const Eigen::Matrix<double, 3, 1>& p_WS_W0,
                         const Eigen::Matrix<double, 3, 1>& p_WS_W1,
                         const double tol) {
  EXPECT_LT((p_WS_W1 - p_WS_W0).norm(), tol);
}

struct CovPropConfig {
 private:
  const double g;
  const double sigma_g_c;
  const double sigma_a_c;
  const double sigma_gw_c;
  const double sigma_aw_c;
  const double dt;

  Eigen::Vector3d p_WS_W0;
  Eigen::Quaterniond q_WS0;
  okvis::SpeedAndBiases sb0;
  okvis::ImuMeasurementDeque imuMeasurements;
  Eigen::Matrix<double, 27, 1> vTgTsTa;
  okvis::ImuParameters imuParams;

  const bool bNominalTgTsTa;  // use nominal values for Tg Ts Ta, i.e.,
                              // identity, zero, identity?
                              // if false, add noise to them, therefore,
  // set true if testing a method that does not support Tg Ts Ta model
 public:
  CovPropConfig(bool nominalPose, bool nominalTgTsTa)
      : g(9.81),
        sigma_g_c(5e-2),
        sigma_a_c(3e-2),
        sigma_gw_c(7e-3),
        sigma_aw_c(2e-3),
        dt(0.1),
        bNominalTgTsTa(nominalTgTsTa) {
    imuParams.g_max = 7.8;
    imuParams.a_max = 176;
    imuParams.sigma_g_c = sigma_g_c;
    imuParams.sigma_a_c = sigma_a_c;
    imuParams.sigma_gw_c = sigma_gw_c;
    imuParams.sigma_aw_c = sigma_aw_c;
    imuParams.g = g;

    if (nominalPose) {
      p_WS_W0 = Eigen::Vector3d(0, 0, 0);
      q_WS0 = Eigen::Quaterniond(1, 0, 0, 0);
    } else {
      p_WS_W0 = Eigen::Vector3d(vio::gauss_rand(0, 1), vio::gauss_rand(0, 1),
                                vio::gauss_rand(0, 1));
      q_WS0 = Eigen::Quaterniond(vio::gauss_rand(0, 1), vio::gauss_rand(0, 1),
                                 vio::gauss_rand(0, 1), vio::gauss_rand(0, 1));
      q_WS0.normalize();
    }
    sb0 << 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;
    for (int jack = 0; jack < 1000; ++jack) {
      Eigen::Vector3d gyr = Eigen::Vector3d::Random();  // range from [-1, 1]
      Eigen::Vector3d acc = Eigen::Vector3d::Random();
      acc[2] = 10 + vio::gauss_rand(0.0, 0.1);
      imuMeasurements.push_back(okvis::ImuMeasurement(
          okvis::Time(jack * dt), okvis::ImuSensorReadings(gyr, acc)));
    }
    // if to test Leutenegger's propagation method, set vTgTsTa as nominal
    // values
    vTgTsTa = Eigen::Matrix<double, 27, 1>::Zero();
    // otherwise, random initialization is OK
    if (!bNominalTgTsTa)
      vTgTsTa = 1e-2 * (Eigen::Matrix<double, 27, 1>::Random());

    for (int jack = 0; jack < 3; ++jack) {
      vTgTsTa[jack * 4] += 1;
      vTgTsTa[jack * 4 + 18] += 1;
    }
  }

  double get_g() const { return g; }
  double get_sigma_g_c() const { return sigma_g_c; }
  double get_sigma_a_c() const { return sigma_a_c; }
  double get_sigma_gw_c() const { return sigma_gw_c; }
  double get_sigma_aw_c() const { return sigma_aw_c; }
  double get_dt() const { return dt; }
  Eigen::Vector3d get_p_WS_W0() const { return p_WS_W0; }
  Eigen::Quaterniond get_q_WS0() const { return q_WS0; }
  okvis::SpeedAndBiases get_sb0() const { return sb0; }
  okvis::ImuMeasurementDeque get_imu_measurements() const {
    return imuMeasurements;
  }
  Eigen::Matrix<double, 27, 1> get_vTgTsTa() const { return vTgTsTa; }
  okvis::ImuParameters get_imu_params() const { return imuParams; }
  okvis::Time get_meas_begin_time() const {
    return imuMeasurements.begin()->timeStamp;
  }
  okvis::Time get_meas_end_time() const {
    return imuMeasurements.rbegin()->timeStamp;
  }
  size_t get_meas_size() const { return imuMeasurements.size(); }

  Eigen::Matrix<double, 12, 1> get_q_n_aw_babw() const {
    Eigen::Matrix<double, 12, 1> q_n_aw_babw;
    q_n_aw_babw << pow(imuParams.sigma_a_c, 2), pow(imuParams.sigma_a_c, 2),
        pow(imuParams.sigma_a_c, 2), pow(imuParams.sigma_g_c, 2),
        pow(imuParams.sigma_g_c, 2), pow(imuParams.sigma_g_c, 2),
        pow(imuParams.sigma_aw_c, 2), pow(imuParams.sigma_aw_c, 2),
        pow(imuParams.sigma_aw_c, 2), pow(imuParams.sigma_gw_c, 2),
        pow(imuParams.sigma_gw_c, 2), pow(imuParams.sigma_gw_c, 2);
    return q_n_aw_babw;
  }
  std::vector<Eigen::Matrix<double, 7, 1>,
              Eigen::aligned_allocator<Eigen::Matrix<double, 7, 1>>>
  get_imu_measurement_vector() const {
    std::vector<Eigen::Matrix<double, 7, 1>,
                Eigen::aligned_allocator<Eigen::Matrix<double, 7, 1>>>
        measurements;

    Eigen::Matrix<double, 7, 1> meas;
    for (auto iter = imuMeasurements.begin(); iter != imuMeasurements.end();
         ++iter) {
      meas << iter->timeStamp.toSec(), iter->measurement.gyroscopes,
          iter->measurement.accelerometers;
      measurements.push_back(meas);
    }
    return measurements;
  }

  void check_q_near(const Eigen::Quaterniond& q_WS1, const double tol) const {
    ::check_q_near(q_WS0, q_WS1, tol);
  }
  void check_v_near(const Eigen::Matrix<double, 9, 1>& sb1,
                    const double tol) const {
    ::check_v_near(sb0, sb1, tol);
  }

  void check_p_near(const Eigen::Matrix<double, 3, 1>& p_WS_W1,
                    const double tol) {
    ::check_p_near(p_WS_W0, p_WS_W1, tol);
  }
};

static void print_p_q_sb(const Eigen::Vector3d& p_WS_W,
                         const Eigen::Quaterniond& q_WS,
                         const Eigen::Matrix<double, 9, 1>& sb) {
  std::cout << "p:" << p_WS_W.transpose() << std::endl;
  std::cout << "q:" << q_WS.x() << " " << q_WS.y() << " " << q_WS.z() << " "
            << q_WS.w() << std::endl;
  std::cout << "v:" << sb.head<3>().transpose() << std::endl;
  std::cout << "bg ba:" << sb.tail<6>().transpose() << std::endl;
}

// compare RungeKutta and Euler forward and backward integration
TEST(IMUOdometry, BackwardIntegration) {
  srand((unsigned int)time(0));

  CovPropConfig cpc(true, false);

  Eigen::Vector3d p_WS_W = cpc.get_p_WS_W0();
  Eigen::Quaterniond q_WS = cpc.get_q_WS0();
  okvis::SpeedAndBiases sb = cpc.get_sb0();
  okvis::kinematics::Transformation T_WB(cpc.get_p_WS_W0(), cpc.get_q_WS0());

  std::cout << "States before forward-backward RK integration:" << std::endl;
  print_p_q_sb(p_WS_W, q_WS, sb);
  okvis::IMUOdometry::propagation_RungeKutta(
      cpc.get_imu_measurements(), cpc.get_imu_params(), T_WB, sb,
      cpc.get_vTgTsTa(), cpc.get_meas_begin_time(), cpc.get_meas_end_time());

  okvis::IMUOdometry::propagationBackward_RungeKutta(
      cpc.get_imu_measurements(), cpc.get_imu_params(), T_WB, sb,
      cpc.get_vTgTsTa(), cpc.get_meas_end_time(), cpc.get_meas_begin_time());
  p_WS_W = T_WB.r();
  q_WS = T_WB.q();

  Eigen::Vector3d p_WS_W1 = p_WS_W;
  Eigen::Quaterniond q_WS1 = q_WS;
  okvis::SpeedAndBiases sb1 = sb;
  std::cout << "States after forward-backward RK integration:" << std::endl;
  print_p_q_sb(p_WS_W, q_WS, sb);

  // DETAILED INTEGRATION
  p_WS_W = cpc.get_p_WS_W0();
  q_WS = cpc.get_q_WS0();
  sb = cpc.get_sb0();
  std::cout << "States before forward RK integration:" << std::endl;
  print_p_q_sb(p_WS_W, q_WS, sb);

  okvis::ImuMeasurementDeque imuMeasurements = cpc.get_imu_measurements();
  auto iterLast = imuMeasurements.begin();
  for (auto iter = imuMeasurements.begin(); iter != imuMeasurements.end();
       ++iter) {
    if (iter == imuMeasurements.begin()) continue;
    okvis::ceres::ode::integrateOneStep_RungeKutta(
        iterLast->measurement.gyroscopes, iterLast->measurement.accelerometers,
        iter->measurement.gyroscopes, iter->measurement.accelerometers,
        cpc.get_g(), cpc.get_sigma_g_c(), cpc.get_sigma_a_c(),
        cpc.get_sigma_gw_c(), cpc.get_sigma_aw_c(), cpc.get_dt(), p_WS_W, q_WS,
        sb, cpc.get_vTgTsTa());
    iterLast = iter;
  }

  std::cout << "States after forward RK integration:" << std::endl;
  print_p_q_sb(p_WS_W, q_WS, sb);

  // backward
  auto iterRLast = imuMeasurements.rbegin();
  for (auto iterR = imuMeasurements.rbegin(); iterR != imuMeasurements.rend();
       ++iterR) {
    if (iterR == imuMeasurements.rbegin()) continue;
    okvis::ceres::ode::integrateOneStepBackward_RungeKutta(
        iterR->measurement.gyroscopes, iterR->measurement.accelerometers,
        iterRLast->measurement.gyroscopes,
        iterRLast->measurement.accelerometers, cpc.get_g(), cpc.get_sigma_g_c(),
        cpc.get_sigma_a_c(), cpc.get_sigma_gw_c(), cpc.get_sigma_aw_c(),
        cpc.get_dt(), p_WS_W, q_WS, sb, cpc.get_vTgTsTa());
    iterRLast = iterR;
  }

  std::cout << "States after backward RK integration:" << std::endl;
  print_p_q_sb(p_WS_W, q_WS, sb);

  Eigen::Vector3d p_WS_W2 = p_WS_W;
  Eigen::Quaterniond q_WS2 = q_WS;
  okvis::SpeedAndBiases sb2 = sb;

  // NUMERICAL_INTEGRATION EULER
  std::cout << "States before forward Euler integration:" << std::endl;
  print_p_q_sb(cpc.get_p_WS_W0(), cpc.get_q_WS0(), cpc.get_sb0());

  sb = cpc.get_sb0();
  T_WB = okvis::kinematics::Transformation(cpc.get_p_WS_W0(), cpc.get_q_WS0());
  Eigen::Vector3d tempV_WS = sb.head<3>();
  IMUErrorModel<double> iem(sb.tail<6>(), cpc.get_vTgTsTa());
  okvis::IMUOdometry::propagation(cpc.get_imu_measurements(), cpc.get_imu_params(),
                           T_WB, tempV_WS, iem, cpc.get_meas_begin_time(),
                           cpc.get_meas_end_time());

  std::cout << "States after forward Euler integration:" << std::endl;
  sb.head<3>() = tempV_WS;
  print_p_q_sb(T_WB.r(), T_WB.q(), sb);

  okvis::IMUOdometry::propagationBackward(
      cpc.get_imu_measurements(), cpc.get_imu_params(), T_WB, tempV_WS, iem,
      cpc.get_meas_end_time(), cpc.get_meas_begin_time());
  p_WS_W = T_WB.r();
  q_WS = T_WB.q();
  sb.head<3>() = tempV_WS;
  std::cout << "States after backward Euler integration:" << std::endl;
  print_p_q_sb(T_WB.r(), T_WB.q(), sb);

  Eigen::Vector3d p_WS_W3 = p_WS_W;
  Eigen::Quaterniond q_WS3 = q_WS;
  okvis::SpeedAndBiases sb3 = sb;

  // two methods results the same?
  check_q_near(q_WS1, q_WS2, 1e-8);
  check_p_near(p_WS_W1, p_WS_W2, 1e-8);
  check_sb_near(sb1, sb2, 1e-8);

  // Runge Kutta return to the starting position ?
  cpc.check_q_near(q_WS1, 1e-8);
  cpc.check_v_near(sb1, 2e-2);
  cpc.check_p_near(p_WS_W1, 1.5);

  // Euler return to the starting position ?
  cpc.check_q_near(q_WS3, 1e-6);
  cpc.check_v_near(sb3, 0.1);
  cpc.check_p_near(p_WS_W3, 5.0);
}

void IMUOdometryTrapezoidRule(
    const Eigen::Vector3d& p_WS_W0, const Eigen::Quaterniond& q_WS0,
    const okvis::SpeedAndBiases& sb0,
    const Eigen::Matrix<double, 27, 1>& vTgTsTa,
    const okvis::ImuMeasurementDeque& imuMeasurements,
    const okvis::ImuParameters& imuParams, Eigen::Vector3d* p_WS_W2,
    Eigen::Quaterniond* q_WS2, okvis::SpeedAndBiases* sb2,
    Eigen::Matrix<double, 42, 1>* sqrtDiagCov2,
    Eigen::Matrix<double, 42, 1>* firstRow2,
    bool bVarianceForShapeMatrices = false, bool bVerbose = false) {
  okvis::kinematics::Transformation T_WS(p_WS_W0, q_WS0);
  okvis::SpeedAndBiases sb = sb0;
  okvis::timing::Timer okvisTimer("okvis", false);

  Eigen::Matrix<double, 42, 42> P2;
  P2.setIdentity();
  if (!bVarianceForShapeMatrices) P2.bottomRightCorner<27, 27>().setZero();
  Eigen::Matrix<double, okvis::ceres::ode::OdoErrorStateDim,
                okvis::ceres::ode::OdoErrorStateDim>
      F_tot;
  F_tot.setIdentity();

  int numUsedImuMeasurements = 0;
  if (0) {  // this is the same as the first case in effect which uses
            // integrateOneStep_RungeKutta
    numUsedImuMeasurements = okvis::IMUOdometry::propagation_RungeKutta(
        imuMeasurements, imuParams, T_WS, sb, vTgTsTa,
        imuMeasurements.begin()->timeStamp, imuMeasurements.rbegin()->timeStamp,
        &P2, &F_tot);
  } else {
    Eigen::Vector3d tempV_WS = sb.head<3>();
    IMUErrorModel<double> tempIEM(sb.tail<6>(), vTgTsTa);
    Eigen::Matrix<double, 6, 1> lP;
    lP << T_WS.r(), tempV_WS;
    numUsedImuMeasurements = okvis::IMUOdometry::propagation(
        imuMeasurements, imuParams, T_WS, tempV_WS, tempIEM,
        imuMeasurements.begin()->timeStamp, imuMeasurements.rbegin()->timeStamp,
        &P2, &F_tot, &lP);
    sb.head<3>() = tempV_WS;
  }

  double timeElapsed = okvisTimer.stop();

  *p_WS_W2 = T_WS.r();
  *q_WS2 = T_WS.q();
  *sb2 = sb;
  Eigen::Matrix<double, 42, 1> P2Diag = P2.diagonal();
  *sqrtDiagCov2 = P2Diag.cwiseSqrt();
  *firstRow2 = F_tot.row(0).transpose();

  std::cout
      << "time used by huai trapezoid rule forward propagtion with covariance "
      << timeElapsed << std::endl;
  if (bVerbose) {
    std::cout << "numUsedMeas " << numUsedImuMeasurements << " totalMeas "
              << (int)imuMeasurements.size() << std::endl;
    std::cout << "q_WS " << q_WS2->w() << " " << q_WS2->x() << " " << q_WS2->y()
              << " " << q_WS2->z() << std::endl;
    std::cout << "p_WS_W " << p_WS_W2->transpose() << std::endl;
    std::cout << "speed and bias " << sb2->transpose() << std::endl;
    std::cout << "cov diagonal sqrt " << std::endl;
    std::cout << sqrtDiagCov2->transpose() << std::endl;
    std::cout << "Jacobian diagonal " << std::endl;
    std::cout << vio::superdiagonal(F_tot).transpose() << std::endl;
    std::cout << "Jacobian first row " << std::endl;
    std::cout << firstRow2->transpose() << std::endl;
  }
}

/// test and compare the propagation for both states and covariance by both the
/// classic RK4 and okvis's state transition method
TEST(IMUOdometry, IMUCovariancePropagation) {
  bool bVarianceForShapeMatrices =
      false;             // use positive variance for elements in Tg Ts Ta?
  bool bVerbose = true;  // print the covariance and jacobian results

  srand((unsigned int)time(0));
  CovPropConfig cpc(false, true);

  /// method 1: RK4
  Eigen::Vector3d p_WS_W = cpc.get_p_WS_W0();
  Eigen::Quaterniond q_WS = cpc.get_q_WS0();
  okvis::SpeedAndBiases sb = cpc.get_sb0();

  Eigen::Matrix<double, okvis::ceres::ode::OdoErrorStateDim,
                okvis::ceres::ode::OdoErrorStateDim>
      P1;
  P1.setIdentity();
  if (!bVarianceForShapeMatrices) P1.bottomRightCorner<27, 27>().setZero();
  Eigen::Matrix<double, okvis::ceres::ode::OdoErrorStateDim,
                okvis::ceres::ode::OdoErrorStateDim>
      F_tot;
  F_tot.setIdentity();

  okvis::timing::Timer RK4Timer("RK4", false);
  okvis::ImuMeasurementDeque imuMeasurements = cpc.get_imu_measurements();
  auto iterLast = imuMeasurements.begin();
  for (auto iter = imuMeasurements.begin(); iter != imuMeasurements.end();
       ++iter) {
    if (iter == imuMeasurements.begin()) continue;
    okvis::ceres::ode::integrateOneStep_RungeKutta(
        iterLast->measurement.gyroscopes, iterLast->measurement.accelerometers,
        iter->measurement.gyroscopes, iter->measurement.accelerometers,
        cpc.get_g(), cpc.get_sigma_g_c(), cpc.get_sigma_a_c(),
        cpc.get_sigma_gw_c(), cpc.get_sigma_aw_c(), cpc.get_dt(), p_WS_W, q_WS,
        sb, cpc.get_vTgTsTa(), &P1, &F_tot);
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
  Eigen::Vector3d p_WS_W2;
  Eigen::Quaterniond q_WS2;
  okvis::SpeedAndBiases sb2;
  Eigen::Matrix<double, 42, 1> sqrtDiagCov2;
  Eigen::Matrix<double, 42, 1> firstRow2;
  IMUOdometryTrapezoidRule(
      cpc.get_p_WS_W0(), cpc.get_q_WS0(), cpc.get_sb0(), cpc.get_vTgTsTa(),
      cpc.get_imu_measurements(), cpc.get_imu_params(), &p_WS_W2, &q_WS2, &sb2,
      &sqrtDiagCov2, &firstRow2, bVarianceForShapeMatrices, bVerbose);

  /// method 3 : okvis propagation leutenegger's implementation
  okvis::kinematics::Transformation T_WS =
      okvis::kinematics::Transformation(cpc.get_p_WS_W0(), cpc.get_q_WS0());
  sb = cpc.get_sb0();
  okvis::timing::Timer leutenTimer("leutenegger", false);

  Eigen::Matrix<double, 15, 15> leutenP;
  leutenP.setIdentity();
  Eigen::Matrix<double, 15, 15> leutenF;
  leutenF.setIdentity();

  // The Leutenegger's ImuError propagation function starts propagation with an
  // zero covariance. The original implementation has some issue in covariance
  // propagation, its Jacobian is correct though.
  int numUsedImuMeasurements = okvis::ceres::ImuError::propagation(
      cpc.get_imu_measurements(), cpc.get_imu_params(), T_WS, sb,
      cpc.get_meas_begin_time(), cpc.get_meas_end_time(), &leutenP, &leutenF);
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
              << (int)cpc.get_meas_size() << std::endl;
    std::cout << "q_WS " << leuten_q_WS.w() << " " << leuten_q_WS.x() << " "
              << leuten_q_WS.y() << " " << leuten_q_WS.z() << std::endl;
    std::cout << "p_WS_W " << leuten_p_WS_W.transpose() << std::endl;
    std::cout << "speed and bias " << leuten_sb.transpose() << std::endl;
    std::cout << "starting with 0s, cov diagonal sqrt " << std::endl;
    std::cout << leuten_sqrtDiagCov.transpose() << std::endl;
    std::cout << "Jacobian diagonal " << std::endl;
    std::cout << vio::superdiagonal(leutenF).transpose() << std::endl;
    std::cout << "Jacobian first row " << std::endl;
    std::cout << leuten_firstRow.transpose() << std::endl;
  }

  /// method 4 : okvis propagation leutenegger's implementation corrected by
  /// huai
  T_WS = okvis::kinematics::Transformation(cpc.get_p_WS_W0(), cpc.get_q_WS0());
  sb = cpc.get_sb0();
  okvis::timing::Timer leutenTimer2("leuteneggerCorrected", false);
  Eigen::Matrix<double, 15, 15> leutenP2, leutenF2;
  leutenP2.setIdentity();
  leutenF2.setIdentity();

  numUsedImuMeasurements =
      okvis::IMUOdometry::propagation_leutenegger_corrected(
          cpc.get_imu_measurements(), cpc.get_imu_params(), T_WS, sb,
          cpc.get_meas_begin_time(), cpc.get_meas_end_time(), &leutenP2,
          &leutenF2);
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
              << (int)cpc.get_meas_size() << std::endl;
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

  Sophus::SE3d T_s1_to_w(cpc.get_q_WS0(), cpc.get_p_WS_W0());
  sb = cpc.get_sb0();
  double time_pair[2] = {cpc.get_meas_begin_time().toSec(),
                         cpc.get_meas_end_time().toSec()};
  std::vector<Eigen::Matrix<double, 7, 1>,
              Eigen::aligned_allocator<Eigen::Matrix<double, 7, 1>>>
      measurements = cpc.get_imu_measurement_vector();

  Eigen::Matrix<double, 6, 1> gwomegaw;
  gwomegaw.setZero();
  Eigen::Vector3d g_W =
      -cpc.get_g() * Eigen::Vector3d(0, 0, 6371009).normalized();
  gwomegaw.head<3>() = g_W;

  Eigen::Matrix<double, 12, 1> q_n_aw_babw = cpc.get_q_n_aw_babw();

  Sophus::SE3d pred_T_s2_to_w;
  Eigen::Matrix<double, 3, 1> pred_speed_2;
  okvis::timing::Timer simpleTimer("simple", false);
  okvis::ceres::ode::predictStates(T_s1_to_w, sb, time_pair, measurements,
                                   gwomegaw, q_n_aw_babw, &pred_T_s2_to_w,
                                   &pred_speed_2, &P3, cpc.get_vTgTsTa());
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
  EXPECT_TRUE(std::fabs(q_WS2.w() - q_WS1.w()) < 1e-5 &&
              std::fabs(q_WS2.x() - q_WS1.x()) < 1e-5 &&
              std::fabs(q_WS2.y() - q_WS1.y()) < 1e-5 &&
              std::fabs(q_WS2.z() - q_WS1.z()) < 1e-5);
  EXPECT_LT((p_WS_W1 - p_WS_W2).norm(), 50);
  EXPECT_LT((sb1 - sb2).norm(), 2);
  // 1,5 states
  EXPECT_TRUE(std::fabs(q_WS3.w() - q_WS1.w()) < 5e-2 &&
              std::fabs(q_WS3.x() - q_WS1.x()) < 5e-2 &&
              std::fabs(q_WS3.y() - q_WS1.y()) < 5e-2 &&
              std::fabs(q_WS3.z() - q_WS1.z()) < 5e-2);
  EXPECT_LT((p_WS_W1 - p_WS_W3).norm(), 2000);
  EXPECT_LT((sb1 - sb3).norm(), 50);

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

  // 1,5 covariance
  EXPECT_LT((sqrtDiagCov3.head<3>() - sqrtDiagCov1.head<3>()).norm() /
                sqrtDiagCov3.head<3>().norm(),
            8e-2);
  EXPECT_LT((sqrtDiagCov3.segment<3>(3) - sqrtDiagCov1.segment<3>(3)).norm() /
                sqrtDiagCov3.segment<3>(3).norm(),
            5e-2);
  EXPECT_LT((sqrtDiagCov3.segment<3>(6) - sqrtDiagCov1.segment<3>(6)).norm() /
                sqrtDiagCov3.segment<3>(6).norm(),
            0.1);

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
