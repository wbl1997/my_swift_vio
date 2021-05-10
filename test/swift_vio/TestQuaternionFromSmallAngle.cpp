#include <gtest/gtest.h>

#include <Eigen/Core>
#include <vio/eigen_utils.h>
#include <okvis/kinematics/sophus_operators.hpp>

TEST(QuaternionFromSmallAngle, FourMethods) {
  for (int j = 0; j < 5; ++j) {
    Eigen::Vector3d rvec = Eigen::Vector3d::Random();
    Eigen::Quaterniond q1 = vio::quaternionFromSmallAngle(rvec);
    Eigen::Quaterniond q2 = okvis::kinematics::expAndTheta(rvec);
    Eigen::Quaterniond q3 = vio::rvec2quat(rvec);
    Eigen::Quaterniond q4(Eigen::AngleAxisd(rvec.norm(), rvec.normalized()));
    EXPECT_LT((q1.coeffs() - q4.coeffs()).lpNorm<Eigen::Infinity>(), 5e-2);
    EXPECT_LT((q2.coeffs() - q4.coeffs()).lpNorm<Eigen::Infinity>(), 1e-8);
    EXPECT_LT((q3.coeffs() - q4.coeffs()).lpNorm<Eigen::Infinity>(), 1e-8);
  }
}
