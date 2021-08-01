/**
 * @file    TestPoseError.cpp
 * @brief   Unit test for PoseError
 * @author  Jianzhu Huai
 */

#include <gtest/gtest.h>
#include <okvis/ceres/PoseError.hpp>
#include <okvis/kinematics/MatrixPseudoInverse.hpp>

TEST(okvisTestSuite, PoseError) {
  Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Zero();
  information(5, 5) = 1.0e8;
  information(0, 0) = 1.0e8;
  information(1, 1) = 1.0e8;
  information(2, 2) = 1.0e8;
  Eigen::Matrix<double, 6, 6> expectedCov;
  okvis::MatrixPseudoInverse::pseudoInverseSymm(information, expectedCov);
  Eigen::Matrix<double, 6, 6> expectedRootInfo, L;
  okvis::MatrixPseudoInverse::pseudoSymmSqrt(information, L);
  expectedRootInfo = L.transpose();
  okvis::ceres::PoseError poseError(okvis::kinematics::Transformation(),
                                    information);
  // However, the square root info matrices computed by ldlt and
  // selfAdjointEigenSolver can be very different.
  EXPECT_LT(
      (poseError.rootInformation().transpose() * poseError.rootInformation() -
       information)
          .lpNorm<Eigen::Infinity>(),
      1e-8);
  EXPECT_LT((expectedRootInfo.transpose() * expectedRootInfo - information)
                .lpNorm<Eigen::Infinity>(),
            1e-8);

  EXPECT_LT((poseError.covariance() - expectedCov).lpNorm<Eigen::Infinity>(),
            1e-8)
      << "actual cov\n"
      << poseError.covariance() << "\nexpected cov\n"
      << expectedCov;
}
