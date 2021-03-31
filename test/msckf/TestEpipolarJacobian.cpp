#include <iostream>
#include "gtest/gtest.h"

#include "msckf/EpipolarJacobian.hpp"
#include "msckf/RelativeMotionJacobian.hpp"
#include "okvis/kinematics/sophus_operators.hpp"

TEST(EpipolarJacobian, de_dfjk) {
  srand((unsigned int)time(0));  // comment this for deterministic behavior
  const double eps = 1e-6;
  okvis::kinematics::Transformation T_BC;
  T_BC.setRandom();
  okvis::kinematics::Transformation T_GBj;
  T_GBj.setRandom();
  okvis::kinematics::Transformation T_GBk;
  T_GBk.setRandom();
  okvis::RelativeMotionJacobian rmj(T_BC, T_GBj, T_GBk);
  okvis::kinematics::Transformation T_CjCk = rmj.relativeMotionT();
  Eigen::Vector2d fj12 = Eigen::Vector2d::Random();
  Eigen::Vector2d fk12 = Eigen::Vector2d::Random();
  Eigen::Vector3d fj;
  fj << fj12, 1;
  Eigen::Vector3d fk;
  fk << fk12, 1;
  okvis::EpipolarJacobian epj(T_CjCk.C(), T_CjCk.r(), fj, fk);
  Eigen::Matrix<double, 1, 3> de_dtheta_CjCk_anal;
  epj.de_dtheta_CjCk(&de_dtheta_CjCk_anal);
  Eigen::Matrix<double, 1, 3> de_dt_CjCk_anal;
  epj.de_dt_CjCk(&de_dt_CjCk_anal);
  Eigen::Matrix<double, 1, 3> de_dfj_anal;
  epj.de_dfj(&de_dfj_anal);
  Eigen::Matrix<double, 1, 3> de_dfk_anal;
  epj.de_dfk(&de_dfk_anal);

  Eigen::Matrix<double, 1, 3> de_dtheta_CjCk, de_dt_CjCk, de_dfj, de_dfk;
  Eigen::Vector3d delta;
  for (int i = 0; i < 3; ++i) {
    delta.setZero();
    delta[i] = eps;
    Eigen::Quaterniond deltaq = okvis::kinematics::expAndTheta(delta);
    okvis::EpipolarJacobian epj_bar((deltaq * T_CjCk.q()).toRotationMatrix(), T_CjCk.r(), fj, fk);
    de_dtheta_CjCk[i] = (epj_bar.evaluate() - epj.evaluate()) / eps;
  }
  EXPECT_LT(
      (de_dtheta_CjCk - de_dtheta_CjCk_anal).lpNorm<Eigen::Infinity>(),
      5e-6);

  for (int i = 0; i < 3; ++i) {
    delta.setZero();
    delta[i] = eps;
    okvis::EpipolarJacobian epj_bar(T_CjCk.C(), delta + T_CjCk.r(), fj, fk);
    de_dt_CjCk[i] = (epj_bar.evaluate() - epj.evaluate()) / eps;
  }
  EXPECT_LT(
      (de_dt_CjCk - de_dt_CjCk_anal).lpNorm<Eigen::Infinity>(),
      eps);
  for (int i = 0; i < 3; ++i) {
    delta.setZero();
    delta[i] = eps;
    okvis::EpipolarJacobian epj_bar(T_CjCk.C(), T_CjCk.r(), fj + delta, fk);
    de_dfj[i] = (epj_bar.evaluate() - epj.evaluate()) / eps;
  }
  EXPECT_LT(
      (de_dfj - de_dfj_anal).lpNorm<Eigen::Infinity>(),
      eps);
  for (int i = 0; i < 3; ++i) {
    delta.setZero();
    delta[i] = eps;
    okvis::EpipolarJacobian epj_bar(T_CjCk.C(), T_CjCk.r(), fj,
                                    fk + delta);
    de_dfk[i] = (epj_bar.evaluate() - epj.evaluate()) / eps;
  }
  EXPECT_LT(
      (de_dfk - de_dfk_anal).lpNorm<Eigen::Infinity>(),
      eps);
}

TEST(EpipolarJacobian, de_ddelta_BC) {
  srand((unsigned int)time(0));  // comment this for deterministic behavior
  const double eps = 1e-6;
  okvis::kinematics::Transformation T_BC;
  T_BC.setRandom();
  okvis::kinematics::Transformation T_GBj;
  T_GBj.setRandom();
  okvis::kinematics::Transformation T_GBk;
  T_GBk.setRandom();
  okvis::RelativeMotionJacobian rmj(T_BC, T_GBj, T_GBk);
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

  Eigen::Vector2d fj12 = Eigen::Vector2d::Random();
  Eigen::Vector2d fk12 = Eigen::Vector2d::Random();
  Eigen::Vector3d fj;
  fj << fj12, 1;
  Eigen::Vector3d fk;
  fk << fk12, 1;
  okvis::EpipolarJacobian epj(T_CjCk.C(), T_CjCk.r(), fj, fk);
  Eigen::Matrix<double, 1, 3> de_dtheta_CjCk_anal;
  epj.de_dtheta_CjCk(&de_dtheta_CjCk_anal);
  Eigen::Matrix<double, 1, 3> de_dt_CjCk_anal;
  epj.de_dt_CjCk(&de_dt_CjCk_anal);
  Eigen::Matrix<double, 1, 3> de_dfj_anal;
  epj.de_dfj(&de_dfj_anal);
  Eigen::Matrix<double, 1, 3> de_dfk_anal;
  epj.de_dfk(&de_dfk_anal);

  Eigen::Matrix<double, 1, 6> de_ddelta_BC_anal =
          de_dtheta_CjCk_anal * dtheta_ddelta_BC_anal +
          de_dt_CjCk_anal * dp_ddelta_BC_anal;

  Eigen::Matrix<double, 6, 1> delta;
  Eigen::Matrix<double, 1, 6> de_ddelta_BC;
  for (int i = 0; i < 6; ++i) {
    delta.setZero();
    delta(i) = eps;
    okvis::kinematics::Transformation T_BC_bar = T_BC;
    T_BC_bar.oplus(delta);
    okvis::RelativeMotionJacobian rmj_bar(T_BC_bar, T_GBj, T_GBk);
    okvis::kinematics::Transformation T_CjCk_bar = rmj_bar.relativeMotionT();
    okvis::EpipolarJacobian epj_bar(T_CjCk_bar.C(), T_CjCk_bar.r(), fj, fk);
    de_ddelta_BC[i] = (epj_bar.evaluate() - epj.evaluate()) / eps;
  }

  EXPECT_LT((de_ddelta_BC - de_ddelta_BC_anal).lpNorm<Eigen::Infinity>(), 5*eps);

  std::cout << "de_ddelta_BC numeric:\n" << de_ddelta_BC << "\n";
  std::cout << "de_ddelta_BC analytic:\n"
            << de_ddelta_BC_anal << "\n";

}


TEST(EpipolarJacobian, de_ddelta_GBj) {
  srand((unsigned int)time(0));  // comment this for deterministic behavior
  const double eps = 1e-6;
  okvis::kinematics::Transformation T_BC;
  T_BC.setRandom();
  okvis::kinematics::Transformation T_GBj;
  T_GBj.setRandom();
  okvis::kinematics::Transformation T_GBk;
  T_GBk.setRandom();
  okvis::RelativeMotionJacobian rmj(T_BC, T_GBj, T_GBk);
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

  Eigen::Vector2d fj12 = Eigen::Vector2d::Random();
  Eigen::Vector2d fk12 = Eigen::Vector2d::Random();
  Eigen::Vector3d fj;
  fj << fj12, 1;
  Eigen::Vector3d fk;
  fk << fk12, 1;
  okvis::EpipolarJacobian epj(T_CjCk.C(), T_CjCk.r(), fj, fk);
  Eigen::Matrix<double, 1, 3> de_dtheta_CjCk_anal;
  epj.de_dtheta_CjCk(&de_dtheta_CjCk_anal);
  Eigen::Matrix<double, 1, 3> de_dt_CjCk_anal;
  epj.de_dt_CjCk(&de_dt_CjCk_anal);
  Eigen::Matrix<double, 1, 3> de_dfj_anal;
  epj.de_dfj(&de_dfj_anal);
  Eigen::Matrix<double, 1, 3> de_dfk_anal;
  epj.de_dfk(&de_dfk_anal);

  Eigen::Matrix<double, 1, 6> de_ddelta_GBj_anal =
          de_dtheta_CjCk_anal * dtheta_ddelta_GBj_anal +
          de_dt_CjCk_anal * dp_ddelta_GBj_anal;

  Eigen::Matrix<double, 6, 1> delta;
  Eigen::Matrix<double, 1, 6> de_ddelta_GBj;
  for (int i = 0; i < 6; ++i) {
    delta.setZero();
    delta(i) = eps;
    okvis::kinematics::Transformation T_GBj_bar = T_GBj;
    T_GBj_bar.oplus(delta);
    okvis::RelativeMotionJacobian rmj_bar(T_BC, T_GBj_bar, T_GBk);
    okvis::kinematics::Transformation T_CjCk_bar = rmj_bar.relativeMotionT();
    okvis::EpipolarJacobian epj_bar(T_CjCk_bar.C(), T_CjCk_bar.r(), fj, fk);
    de_ddelta_GBj[i] = (epj_bar.evaluate() - epj.evaluate()) / eps;
  }

  EXPECT_LT((de_ddelta_GBj - de_ddelta_GBj_anal).lpNorm<Eigen::Infinity>(), 5*eps);

  std::cout << "de_ddelta_GBj numeric:\n" << de_ddelta_GBj << "\n";
  std::cout << "de_ddelta_GBj analytic:\n"
            << de_ddelta_GBj_anal << "\n";

}


TEST(EpipolarJacobian, de_ddelta_GBk) {
  srand((unsigned int)time(0));  // comment this for deterministic behavior
  const double eps = 1e-6;
  okvis::kinematics::Transformation T_BC;
  T_BC.setRandom();
  okvis::kinematics::Transformation T_GBj;
  T_GBj.setRandom();
  okvis::kinematics::Transformation T_GBk;
  T_GBk.setRandom();
  okvis::RelativeMotionJacobian rmj(T_BC, T_GBj, T_GBk);
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

  Eigen::Vector2d fj12 = Eigen::Vector2d::Random();
  Eigen::Vector2d fk12 = Eigen::Vector2d::Random();
  Eigen::Vector3d fj;
  fj << fj12, 1;
  Eigen::Vector3d fk;
  fk << fk12, 1;
  okvis::EpipolarJacobian epj(T_CjCk.C(), T_CjCk.r(), fj, fk);
  Eigen::Matrix<double, 1, 3> de_dtheta_CjCk_anal;
  epj.de_dtheta_CjCk(&de_dtheta_CjCk_anal);
  Eigen::Matrix<double, 1, 3> de_dt_CjCk_anal;
  epj.de_dt_CjCk(&de_dt_CjCk_anal);
  Eigen::Matrix<double, 1, 3> de_dfj_anal;
  epj.de_dfj(&de_dfj_anal);
  Eigen::Matrix<double, 1, 3> de_dfk_anal;
  epj.de_dfk(&de_dfk_anal);

  Eigen::Matrix<double, 1, 6> de_ddelta_GBk_anal =
          de_dtheta_CjCk_anal * dtheta_ddelta_GBk_anal +
          de_dt_CjCk_anal * dp_ddelta_GBk_anal;

  Eigen::Matrix<double, 6, 1> delta;
  Eigen::Matrix<double, 1, 6> de_ddelta_GBk;
  for (int i = 0; i < 6; ++i) {
    delta.setZero();
    delta(i) = eps;
    okvis::kinematics::Transformation T_GBk_bar = T_GBk;
    T_GBk_bar.oplus(delta);
    okvis::RelativeMotionJacobian rmj_bar(T_BC, T_GBj, T_GBk_bar);
    okvis::kinematics::Transformation T_CjCk_bar = rmj_bar.relativeMotionT();
    okvis::EpipolarJacobian epj_bar(T_CjCk_bar.C(), T_CjCk_bar.r(), fj, fk);
    de_ddelta_GBk[i] = (epj_bar.evaluate() - epj.evaluate()) / eps;
  }

  EXPECT_LT((de_ddelta_GBk - de_ddelta_GBk_anal).lpNorm<Eigen::Infinity>(), 5*eps);

  std::cout << "de_ddelta_GBk numeric:\n" << de_ddelta_GBk << "\n";
  std::cout << "de_ddelta_GBk analytic:\n"
            << de_ddelta_GBk_anal << "\n";

}
