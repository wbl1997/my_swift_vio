#ifndef COVPROPCONFIG_HPP
#define COVPROPCONFIG_HPP

#include <gtest/gtest.h>
#include <iostream>
#include <Eigen/Geometry>

#include <vio/Sample.h>

#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>

static void check_q_near(const Eigen::Quaterniond& q_WS0,
                         const Eigen::Quaterniond& q_WS1, const double tol) {
  Eigen::Quaterniond dq = q_WS0.inverse() * q_WS1;
  EXPECT_LT(std::fabs(std::fabs(dq.w()) - 1), tol);
  EXPECT_LT(std::fabs(dq.x()), tol);
  EXPECT_LT(std::fabs(dq.y()), tol);
  EXPECT_LT(std::fabs(dq.z()), tol);
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


static void print_p_q_sb(const Eigen::Vector3d& p_WS_W,
                         const Eigen::Quaterniond& q_WS,
                         const Eigen::Matrix<double, 9, 1>& sb) {
  std::cout << "p:" << p_WS_W.transpose() << std::endl;
  std::cout << "q:" << q_WS.x() << " " << q_WS.y() << " " << q_WS.z() << " "
            << q_WS.w() << std::endl;
  std::cout << "v:" << sb.head<3>().transpose() << std::endl;
  std::cout << "bg ba:" << sb.tail<6>().transpose() << std::endl;
}

static void checkSelectiveRatio(Eigen::MatrixXd ref, Eigen::MatrixXd est,
                                double ratioTolForLargeValue,
                                double tolForTinyValue, double valueThreshold = 1e-3) {
  for (int i = 0; i < ref.rows(); ++i) {
    for (int j = 0; j < ref.cols(); ++j) {
      double diff = std::fabs(ref(i, j) - est(i, j));
      double refValue = std::fabs(ref(i, j));
      if (refValue < valueThreshold) {
        EXPECT_LT(std::fabs(est(i, j)), tolForTinyValue)
            << "(" << i << ", " << j << "): ref " << ref(i, j) << " est "
            << est(i, j);
        EXPECT_LT(std::fabs(ref(i, j)), tolForTinyValue)
            << "(" << i << ", " << j << "): ref " << ref(i, j) << " est "
            << est(i, j);
      } else {
        EXPECT_LT(diff / refValue, ratioTolForLargeValue)
            << "(" << i << ", " << j << "): ref " << ref(i, j) << " est "
            << est(i, j);
      }
    }
  }
}

const int kNavAndImuParamsDim = 42;
const int kImuAugmentedParamsDim = 27;

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
  Eigen::Matrix<double, 9, 1> sb0;
  okvis::ImuMeasurementDeque imuMeasurements;
  Eigen::Matrix<double, kImuAugmentedParamsDim, 1> vTgTsTa;
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
    srand((unsigned int)time(0));

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
    vTgTsTa = Eigen::Matrix<double, kImuAugmentedParamsDim, 1>::Zero();
    // otherwise, random initialization is OK
    if (!bNominalTgTsTa)
      vTgTsTa = 1e-2 * (Eigen::Matrix<double, kImuAugmentedParamsDim, 1>::Random());

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
  Eigen::Matrix<double, 9, 1> get_sb0() const { return sb0; }
  okvis::ImuMeasurementDeque get_imu_measurements() const {
    return imuMeasurements;
  }
  Eigen::Matrix<double, kImuAugmentedParamsDim, 1> get_vTgTsTa() const { return vTgTsTa; }
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

#endif // COVPROPCONFIG_HPP
