#include <gtest/gtest.h>

#include <swift_vio/TransformMultiplyJacobian.hpp>
#include <okvis/kinematics/sophus_operators.hpp>

class TransformMultiplyTest : public ::testing::Test {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 protected:
  void SetUp() override {
    srand((unsigned int)time(0));  // comment this for deterministic behavior
    T_AB_.setRandom();
    T_BC_.setRandom();
    tmj_.initialize(T_AB_, T_BC_);
    T_AC_ = tmj_.multiplyT();
    computeNumericJacobians();
  }

  void computeNumericJacobians() {
    Eigen::Matrix<double, 6, 1> delta;

    for (int i = 0; i < 6; ++i) {
      delta.setZero();
      delta(i) = eps;
      okvis::kinematics::Transformation T_AB_bar = T_AB_;
      T_AB_bar.oplus(delta);
      swift_vio::TransformMultiplyJacobian tmj_bar(T_AB_bar, T_BC_);
      okvis::kinematics::Transformation T_AC_bar = tmj_bar.multiplyT();
      Eigen::Matrix<double, 6, 1> ratio =
          okvis::kinematics::ominus(T_AC_bar, T_AC_) / eps;
      dp_ddelta_AB_.col(i) = ratio.head<3>();
      dtheta_ddelta_AB_.col(i) = ratio.tail<3>();
    }

    for (int i = 0; i < 6; ++i) {
      delta.setZero();
      delta(i) = eps;
      okvis::kinematics::Transformation T_BC_bar = T_BC_;
      T_BC_bar.oplus(delta);
      swift_vio::TransformMultiplyJacobian tmj_bar(T_AB_, T_BC_bar);
      okvis::kinematics::Transformation T_AC_bar = tmj_bar.multiplyT();
      Eigen::Matrix<double, 6, 1> ratio =
          okvis::kinematics::ominus(T_AC_bar, T_AC_) / eps;
      dp_ddelta_BC_.col(i) = ratio.head<3>();
      dtheta_ddelta_BC_.col(i) = ratio.tail<3>();
    }
  }

  // void TearDown() override {}
  okvis::kinematics::Transformation T_AB_;
  okvis::kinematics::Transformation T_BC_;
  swift_vio::TransformMultiplyJacobian tmj_;
  okvis::kinematics::Transformation T_AC_;

  Eigen::Matrix<double, 3, 6> dtheta_ddelta_AB_;
  Eigen::Matrix<double, 3, 6> dp_ddelta_AB_;
  Eigen::Matrix<double, 3, 6> dtheta_ddelta_BC_;
  Eigen::Matrix<double, 3, 6> dp_ddelta_BC_;

  const double eps = 1e-6;
};

TEST_F(TransformMultiplyTest, translationJacobians) {
  Eigen::Matrix3d dp_dtheta_AB = tmj_.dp_dtheta_AB();
  Eigen::Matrix3d dp_dt_AB = tmj_.dp_dp_AB();
  Eigen::Matrix3d dp_dtheta_BC = tmj_.dp_dtheta_BC();
  Eigen::Matrix3d dp_dt_BC = tmj_.dp_dp_BC();
  EXPECT_LT((dp_dt_AB - dp_ddelta_AB_.topLeftCorner<3, 3>())
                .lpNorm<Eigen::Infinity>(),
            eps) << "dp_dt_AB\n" << dp_dt_AB << "\ndp_ddelta_AB_\n" << dp_ddelta_AB_.topLeftCorner<3, 3>();
  EXPECT_LT((dp_dtheta_AB - dp_ddelta_AB_.topRightCorner<3, 3>())
                .lpNorm<Eigen::Infinity>(),
            eps) << "dp_dtheta_AB\n" << dp_dtheta_AB <<
                    "\ndp_ddelta_AB_.topRightCorner<3, 3>()\n" << dp_ddelta_AB_.topRightCorner<3, 3>();
  EXPECT_LT((dp_dt_BC - dp_ddelta_BC_.topLeftCorner<3, 3>())
                .lpNorm<Eigen::Infinity>(),
            eps) << "dp_dt_BC\n" << dp_dt_BC << "\ndp_ddelta_BC_.topLeftCorner<3, 3>()\n"
                 << dp_ddelta_BC_.topLeftCorner<3, 3>();
  EXPECT_LT((dp_dtheta_BC - dp_ddelta_BC_.topRightCorner<3, 3>())
                .lpNorm<Eigen::Infinity>(),
            eps);
}

TEST_F(TransformMultiplyTest, rotationJacobians) {
  Eigen::Matrix3d dtheta_dtheta_AB = tmj_.dtheta_dtheta_AB();
  Eigen::Matrix3d dtheta_dp_AB = tmj_.dtheta_dp_AB();
  Eigen::Matrix3d dtheta_dtheta_BC = tmj_.dtheta_dtheta_BC();
  Eigen::Matrix3d dtheta_dp_BC = tmj_.dtheta_dp_BC();
  EXPECT_LT((dtheta_dp_AB - dtheta_ddelta_AB_.topLeftCorner<3, 3>())
                .lpNorm<Eigen::Infinity>(),
            eps);
  EXPECT_LT((dtheta_dtheta_AB - dtheta_ddelta_AB_.topRightCorner<3, 3>())
                .lpNorm<Eigen::Infinity>(),
            eps) << "dtheta_dtheta_AB\n" << dtheta_dtheta_AB << "\ndtheta_ddelta_AB_.topRightCorner<3, 3>()\n"
                 << dtheta_ddelta_AB_.topRightCorner<3, 3>();
  EXPECT_LT((dtheta_dp_BC - dtheta_ddelta_BC_.topLeftCorner<3, 3>())
                .lpNorm<Eigen::Infinity>(),
            eps);
  EXPECT_LT((dtheta_dtheta_BC - dtheta_ddelta_BC_.topRightCorner<3, 3>())
                .lpNorm<Eigen::Infinity>(),
            eps) << "dtheta_dtheta_BC\n" << dtheta_dtheta_BC << "\n dtheta_ddelta_BC_.topRightCorner<3, 3>()\n"
                 << dtheta_ddelta_BC_.topRightCorner<3, 3>();
}

TEST(TransformMultiply, multiply) {
  okvis::kinematics::Transformation T_AB, T_BC, T_AC;
  T_AB.setRandom();
  T_BC.setRandom();
  T_AC = T_AB * T_BC;
  swift_vio::TransformMultiplyJacobian tmj(T_AB, T_BC);
  okvis::kinematics::Transformation T_AC_computed = tmj.multiplyT();
  std::pair<Eigen::Vector3d, Eigen::Quaterniond> pair_T_AC = tmj.multiply();
  EXPECT_LT((T_AC.T() - T_AC_computed.T()).lpNorm<Eigen::Infinity>(), 1e-8);
  EXPECT_LT((T_AC.r() - pair_T_AC.first).lpNorm<Eigen::Infinity>(), 1e-8);
  EXPECT_LT((T_AC.q().coeffs() - pair_T_AC.second.coeffs()).lpNorm<Eigen::Infinity>(), 1e-8);
}
