#include "swift_vio/RelativeMotionJacobian.hpp"

#include <iostream>
#include "gtest/gtest.h"
#include <okvis/kinematics/sophus_operators.hpp>

TEST(RelativeMotionJacobian, de_ddelta_BC) {
  srand((unsigned int)time(0));  // comment this for deterministic behavior
  const double eps = 1e-6;
  okvis::kinematics::Transformation T_BC;
  T_BC.setRandom();
  okvis::kinematics::Transformation T_GBj;
  T_GBj.setRandom();
  okvis::kinematics::Transformation T_GBk;
  T_GBk.setRandom();
  swift_vio::RelativeMotionJacobian rmj(T_BC, T_GBj, T_GBk);
  okvis::kinematics::Transformation T_CjCk = rmj.relativeMotionT();
  Eigen::Matrix3d dtheta_dtheta_BC;
  rmj.dtheta_dtheta_BC(&dtheta_dtheta_BC);
  Eigen::Matrix3d dtheta_dt_BC;
  dtheta_dt_BC.setZero();

  Eigen::Matrix3d dp_dtheta_BC;
  rmj.dp_dtheta_BC(&dp_dtheta_BC);
  Eigen::Matrix3d dp_dt_BC;
  rmj.dp_dt_BC(&dp_dt_BC);
  Eigen::Matrix<double, 3, 6> dp_ddelta_BC_anal, dtheta_ddelta_BC_anal;
  dp_ddelta_BC_anal << dp_dt_BC, dp_dtheta_BC;
  dtheta_ddelta_BC_anal << dtheta_dt_BC, dtheta_dtheta_BC;

  Eigen::Matrix<double, 6, 1> delta;
  Eigen::Matrix<double, 3, 6> dtheta_ddelta_BC;
  Eigen::Matrix<double, 3, 6> dp_ddelta_BC;
  for (int i = 0; i < 6; ++i) {
    delta.setZero();
    delta(i) = eps;
    okvis::kinematics::Transformation T_BC_bar = T_BC;
    T_BC_bar.oplus(delta);
    swift_vio::RelativeMotionJacobian rmj_bar(T_BC_bar, T_GBj, T_GBk);
    okvis::kinematics::Transformation T_CjCk_bar = rmj_bar.relativeMotionT();
    Eigen::Matrix<double, 6, 1> delta = okvis::kinematics::ominus(T_CjCk_bar, T_CjCk) / eps;
    dp_ddelta_BC.col(i) = delta.head<3>();
    dtheta_ddelta_BC.col(i) = delta.tail<3>();
  }

  EXPECT_LT((dp_ddelta_BC - dp_ddelta_BC_anal).lpNorm<Eigen::Infinity>(), eps);
  EXPECT_LT((dtheta_ddelta_BC - dtheta_ddelta_BC_anal).lpNorm<Eigen::Infinity>(), eps);

  std::cout << "dp_ddelta_BC numeric:\n" << dp_ddelta_BC << "\n";
  std::cout << "dp_ddelta_BC analytic:\n"
            << dp_ddelta_BC_anal << "\n";
  std::cout << "dtheta_ddelta_BC numeric:\n" << dtheta_ddelta_BC << "\n";
  std::cout << "dtheta_ddelta_BC analytic:\n"
            << dtheta_ddelta_BC_anal << "\n";

  Eigen::Matrix3d dp_dt_CB_anal, dp_dt_CB;
  rmj.dp_dt_CB(&dp_dt_CB_anal);
  for (int i = 0; i < 3; ++i) {
      delta.setZero();
      delta(i) = eps;
      okvis::kinematics::Transformation T_CB_bar = T_BC.inverse();
      T_CB_bar.oplus(delta);
      swift_vio::RelativeMotionJacobian rmj_bar(T_CB_bar.inverse(), T_GBj, T_GBk);
      okvis::kinematics::Transformation T_CjCk_bar = rmj_bar.relativeMotionT();
      Eigen::Matrix<double, 6, 1> delta = okvis::kinematics::ominus(T_CjCk_bar, T_CjCk) / eps;
      dp_dt_CB.col(i) = delta.head<3>();
  }

  EXPECT_LT((dp_dt_CB - dp_dt_CB_anal).lpNorm<Eigen::Infinity>(), eps);
  std::cout << "dp_dt_CB numeric:\n" << dp_dt_CB << "\n";
  std::cout << "dp_dt_CB analytic:\n"
            << dp_dt_CB_anal << "\n";
}

TEST(RelativeMotionJacobian, de_ddelta_GBj) {
  srand((unsigned int)time(0));  // comment this for deterministic behavior
  const double eps = 1e-6;
  okvis::kinematics::Transformation T_BC;
  T_BC.setRandom();
  okvis::kinematics::Transformation T_GBj;
  T_GBj.setRandom();
  okvis::kinematics::Transformation T_GBk;
  T_GBk.setRandom();
  swift_vio::RelativeMotionJacobian rmj(T_BC, T_GBj, T_GBk);
  okvis::kinematics::Transformation T_CjCk = rmj.relativeMotionT();
  Eigen::Matrix3d dtheta_dtheta_GBj;
  rmj.dtheta_dtheta_GBj(&dtheta_dtheta_GBj);
  Eigen::Matrix3d dtheta_dt_GBj;
  dtheta_dt_GBj.setZero();

  Eigen::Matrix3d dp_dtheta_GBj;
  rmj.dp_dtheta_GBj(&dp_dtheta_GBj);
  Eigen::Matrix3d dp_dt_GBj;
  rmj.dp_dt_GBj(&dp_dt_GBj);
  Eigen::Matrix<double, 3, 6> dp_ddelta_GBj_anal, dtheta_ddelta_GBj_anal;
  dp_ddelta_GBj_anal << dp_dt_GBj, dp_dtheta_GBj;
  dtheta_ddelta_GBj_anal << dtheta_dt_GBj, dtheta_dtheta_GBj;

  Eigen::Matrix<double, 6, 1> delta;
  Eigen::Matrix<double, 3, 6> dtheta_ddelta_GBj;
  Eigen::Matrix<double, 3, 6> dp_ddelta_GBj;
  for (int i = 0; i < 6; ++i) {
    delta.setZero();
    delta(i) = eps;
    okvis::kinematics::Transformation T_GBj_bar = T_GBj;
    T_GBj_bar.oplus(delta);
    swift_vio::RelativeMotionJacobian rmj_bar(T_BC, T_GBj_bar, T_GBk);
    okvis::kinematics::Transformation T_CjCk_bar = rmj_bar.relativeMotionT();
    Eigen::Matrix<double, 6, 1> delta = okvis::kinematics::ominus(T_CjCk_bar, T_CjCk) / eps;
    dp_ddelta_GBj.col(i) = delta.head<3>();
    dtheta_ddelta_GBj.col(i) = delta.tail<3>();
  }

  EXPECT_LT((dp_ddelta_GBj - dp_ddelta_GBj_anal).lpNorm<Eigen::Infinity>(), eps);
  EXPECT_LT((dtheta_ddelta_GBj - dtheta_ddelta_GBj_anal).lpNorm<Eigen::Infinity>(), eps);

  std::cout << "dp_ddelta_GBj numeric:\n" << dp_ddelta_GBj << "\n";
  std::cout << "dp_ddelta_GBj analytic:\n"
            << dp_ddelta_GBj_anal << "\n";
  std::cout << "dtheta_ddelta_GBj numeric:\n" << dtheta_ddelta_GBj << "\n";
  std::cout << "dtheta_ddelta_GBj analytic:\n"
            << dtheta_ddelta_GBj_anal << "\n";


}

TEST(RelativeMotionJacobian, de_ddelta_GBk) {
  srand((unsigned int)time(0));  // comment this for deterministic behavior
  const double eps = 1e-6;
  okvis::kinematics::Transformation T_BC;
  T_BC.setRandom();
  okvis::kinematics::Transformation T_GBj;
  T_GBj.setRandom();
  okvis::kinematics::Transformation T_GBk;
  T_GBk.setRandom();
  swift_vio::RelativeMotionJacobian rmj(T_BC, T_GBj, T_GBk);
  okvis::kinematics::Transformation T_CjCk = rmj.relativeMotionT();
  Eigen::Matrix3d dtheta_dtheta_GBk;
  rmj.dtheta_dtheta_GBk(&dtheta_dtheta_GBk);
  Eigen::Matrix3d dtheta_dt_GBk;
  dtheta_dt_GBk.setZero();

  Eigen::Matrix3d dp_dtheta_GBk;
  rmj.dp_dtheta_GBk(&dp_dtheta_GBk);
  Eigen::Matrix3d dp_dt_GBk;
  rmj.dp_dt_GBk(&dp_dt_GBk);
  Eigen::Matrix<double, 3, 6> dp_ddelta_GBk_anal, dtheta_ddelta_GBk_anal;
  dp_ddelta_GBk_anal << dp_dt_GBk, dp_dtheta_GBk;
  dtheta_ddelta_GBk_anal << dtheta_dt_GBk, dtheta_dtheta_GBk;

  Eigen::Matrix<double, 6, 1> delta;
  Eigen::Matrix<double, 3, 6> dtheta_ddelta_GBk;
  Eigen::Matrix<double, 3, 6> dp_ddelta_GBk;
  for (int i = 0; i < 6; ++i) {
    delta.setZero();
    delta(i) = eps;
    okvis::kinematics::Transformation T_GBk_bar = T_GBk;
    T_GBk_bar.oplus(delta);
    swift_vio::RelativeMotionJacobian rmj_bar(T_BC, T_GBj, T_GBk_bar);
    okvis::kinematics::Transformation T_CjCk_bar = rmj_bar.relativeMotionT();
    Eigen::Matrix<double, 6, 1> delta = okvis::kinematics::ominus(T_CjCk_bar, T_CjCk) / eps;
    dp_ddelta_GBk.col(i) = delta.head<3>();
    dtheta_ddelta_GBk.col(i) = delta.tail<3>();
  }

  EXPECT_LT((dp_ddelta_GBk - dp_ddelta_GBk_anal).lpNorm<Eigen::Infinity>(), eps);
  EXPECT_LT((dtheta_ddelta_GBk - dtheta_ddelta_GBk_anal).lpNorm<Eigen::Infinity>(), eps);

  std::cout << "dp_ddelta_GBk numeric:\n" << dp_ddelta_GBk << "\n";
  std::cout << "dp_ddelta_GBk analytic:\n"
            << dp_ddelta_GBk_anal << "\n";
  std::cout << "dtheta_ddelta_GBk numeric:\n" << dtheta_ddelta_GBk << "\n";
  std::cout << "dtheta_ddelta_GBk analytic:\n"
            << dtheta_ddelta_GBk_anal << "\n";


}
