#include "gtest/gtest.h"

#include "loop_closure/GtsamWrap.hpp"

TEST(BetweenFactor, JacobianToCustomRetract) {
  const double eps = 1e-6;
  const double h = 1e-6;
  gtsam::SharedNoiseModel noiseOdom =
      gtsam::noiseModel::Isotropic::Variance(6, 0.1);
  gtsam::Pose3 Tx(gtsam::Rot3(Eigen::Quaterniond::UnitRandom()),
                  Eigen::Vector3d::Random());
  gtsam::Pose3 Ty(gtsam::Rot3(Eigen::Quaterniond::UnitRandom()),
                  Eigen::Vector3d::Random());
  gtsam::Pose3 Tz = Tx.inverse() * Ty;
  gtsam::BetweenFactor<gtsam::Pose3> bf(0, 1, Tz, noiseOdom);
  Eigen::VectorXd residual = bf.evaluateError(Tx, Ty);
  EXPECT_LT(residual.lpNorm<Eigen::Infinity>(), eps);

  Eigen::Matrix3d analJToCustomRetract = -Tz.rotation().matrix().transpose();
  Eigen::Matrix<double, 6, 6> numericJ;
  for (int i = 0; i < 6; ++i) {
    Eigen::Matrix<double, 6, 1> delta = Eigen::Matrix<double, 6, 1>::Zero();
    delta[i] = h;
    gtsam::Pose3 Tzp = VIO::GtsamWrap::retract(Tz, delta);
    gtsam::BetweenFactor<gtsam::Pose3> bfp(0, 1, Tzp, noiseOdom);
    Eigen::VectorXd residualp = bfp.evaluateError(Tx, Ty);
    numericJ.col(i) = (residualp - residual) / h;
  }
  EXPECT_LT((numericJ.topRightCorner<3, 3>() - analJToCustomRetract).lpNorm<Eigen::Infinity>(),
            eps) << "numericJ.topRightCorner<3, 3>()\n" << numericJ.topRightCorner<3, 3>()
      << "\nanalJToCustomRetract\n" << analJToCustomRetract;
  EXPECT_LT((numericJ.topLeftCorner<3, 3>()).lpNorm<Eigen::Infinity>(), eps)
      << "numericJ.topLeftCorner<3, 3>()\n" << numericJ.topLeftCorner<3, 3>();
  EXPECT_LT((numericJ.bottomLeftCorner<3, 3>() - analJToCustomRetract)
                .lpNorm<Eigen::Infinity>(),
            eps);
  EXPECT_LT((numericJ.bottomRightCorner<3, 3>()).lpNorm<Eigen::Infinity>(),
            eps);

  // Test the Jacobians in a bundle.
  Eigen::Matrix<double, 6, 6> J6;
  VIO::GtsamWrap::toMeasurementJacobianBetweenFactor(Tz, &J6);
  EXPECT_LT((numericJ - J6).lpNorm<Eigen::Infinity>(), eps);
}
