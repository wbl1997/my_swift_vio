#include <glog/logging.h>
#include <gtest/gtest.h>
#include "okvis/kinematics/sophus_operators.hpp"
#include "msckf/ImuSimulator.h"

void testSquircle(double radius, double sideLength, double velocity) {
  okvis::Time startEpoch(0, 0);
  simul::RoundedSquare rs(100, Eigen::Vector3d{0, 0, -9.8}, startEpoch, radius,
                        sideLength, velocity);

  double half_d = sideLength * 0.5;
  double half_pi = M_PI * 0.5;
  std::vector<double> keyEpochs = rs.getEndEpochs();
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >
      ref_xytheta_list{
          {half_d + radius, 0, half_pi},
          {half_d + radius, half_d, half_pi},
          {half_d, half_d + radius, M_PI},
          {-half_d, half_d + radius, M_PI},
          {-half_d - radius, half_d, half_pi + M_PI},
          {-half_d - radius, -half_d, half_pi + M_PI},
          {-half_d, -half_d - radius, 2 * M_PI},
          {half_d, -half_d - radius, 2 * M_PI},
          {half_d + radius, -half_d, half_pi},
          {half_d + radius, 0, half_pi},
      };

  std::vector<okvis::kinematics::Transformation,
              Eigen::aligned_allocator<okvis::kinematics::Transformation> >
      ref_T_WS_list;
  for (const Eigen::Vector3d xytheta : ref_xytheta_list) {
    Eigen::Quaterniond q_WB(
        Eigen::AngleAxisd(xytheta[2], Eigen::Vector3d::UnitZ()));
    if (q_WB.w() < 0) {
      q_WB.coeffs() *= -1;
    }
    ref_T_WS_list.emplace_back(Eigen::Vector3d(xytheta[0], xytheta[1], 0),
                               q_WB);
  }

  keyEpochs.insert(keyEpochs.begin(), startEpoch.toSec());
  int jack = 0;
  okvis::kinematics::Transformation eye_AB;
  for (const double& time : keyEpochs) {
    okvis::Time wrap;
    okvis::kinematics::Transformation T_WB =
        rs.computeGlobalPose(wrap.fromSec(time));
    okvis::kinematics::Transformation deltaT =
        T_WB.inverse() * ref_T_WS_list[jack];
    if (deltaT.q().w() < 0) {
      deltaT = okvis::kinematics::Transformation(
          deltaT.r(), Eigen::Quaterniond(-deltaT.q().coeffs()));
    }
    EXPECT_LT((deltaT.coeffs() - eye_AB.coeffs()).lpNorm<Eigen::Infinity>(),
              1e-6)
        << "Large discrepancy at " << jack
        << "\nEst:" << T_WB.parameters().transpose()
        << "\nRef:" << ref_T_WS_list[jack].parameters().transpose();
    ++jack;
  }
}

TEST(Squircle, Normal) {
  double radius = 1.0;
  double velocity = 1.2;
  double sideLength = 2.0;
  testSquircle(radius, sideLength, velocity);
}

TEST(Squircle, Circle) {
  double radius = 1.0;
  double velocity = 0.8;
  double sideLength = 0.0;
  testSquircle(radius, sideLength, velocity);
}

TEST(Squircle, Dot) {
  double radius = 0.01;
  double velocity = 0.008;
  double sideLength = 0.0;
  testSquircle(radius, sideLength, velocity);
}

TEST(WavyCircle, AngularVelocity) {
  simul::WavyCircle wavyCircle(100, Eigen::Vector3d(0, 0, -9.80665), 5.0, 4.0, 3,
                             10, 1.2);
  EXPECT_NEAR(wavyCircle.waveHeight(), 0.18, 1e-8);
  okvis::Time time(0.8 / 1.2);
  okvis::kinematics::Transformation T_WB = wavyCircle.computeGlobalPose(time);
  Eigen::Vector3d omegaW = wavyCircle.computeGlobalAngularRate(time);
  double h = 1e-6;
  okvis::kinematics::Transformation T_WBdelta = wavyCircle.computeGlobalPose(time + okvis::Duration(h));
  Eigen::Matrix3d Rdelta = (T_WBdelta.C() - T_WB.C()) / h;
  Eigen::Matrix3d OmegaW = Rdelta * T_WB.C().transpose();
  EXPECT_LT((okvis::kinematics::vee(OmegaW) - omegaW).lpNorm<Eigen::Infinity>(), 5e-6);
}

TEST(WavyCircle, LinearVelocity) {
  simul::WavyCircle wavyCircle(100, Eigen::Vector3d(0, 0, -9.80665), 5.0, 4.0, 3,
                             10, 1.2);
  okvis::Time time(0.8 / 1.2);
  okvis::kinematics::Transformation T_WB = wavyCircle.computeGlobalPose(time);
  Eigen::Vector3d v_WB = wavyCircle.computeGlobalLinearVelocity(time);
  double h = 1e-6;
  okvis::kinematics::Transformation T_WBdelta = wavyCircle.computeGlobalPose(time + okvis::Duration(h));
  Eigen::Vector3d v_numeric = (T_WBdelta.r() - T_WB.r())/ h;

  EXPECT_LT((v_numeric - v_WB).lpNorm<Eigen::Infinity>(), 5e-6);
}

TEST(WavyCircle, LinearAcceleration) {
  Eigen::Vector3d gw(0, 0, -9.80665);
  simul::WavyCircle wavyCircle(100, gw, 5.0, 4.0, 3,
                             10, 1.2);
  okvis::Time time(0.8 / 1.2);
  Eigen::Vector3d a_WB = wavyCircle.computeGlobalLinearAcceleration(time);
  Eigen::Vector3d v_WB = wavyCircle.computeGlobalLinearVelocity(time);
  double h = 1e-6;
  Eigen::Vector3d v_WBdelta = wavyCircle.computeGlobalLinearVelocity(time + okvis::Duration(h));
  Eigen::Vector3d a_numeric = (v_WBdelta - v_WB) / h;

  EXPECT_LT((a_numeric - gw - a_WB).lpNorm<Eigen::Infinity>(), 5e-6);
}
