/**
 * @file   TestRiExtendedPose3Prior.cpp
 * @brief  Unit tests for RiExtendedPose3Prior class
 */
#include <gtest/gtest.h>

#include "gtsam/RiExtendedPose3Prior.h"

#include <gtsam/base/numericalDerivative.h>
#include <gtsam/inference/Symbol.h>

class RiExtendedPose3PriorTest : public ::testing::Test {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  void SetUp() override {
    prior_.setRandom();
    xi_.setRandom();
    state_ = prior_;
    state_ = state_.expmap(xi_);
    factor_ = gtsam::RiExtendedPose3Prior(
        gtsam::Symbol('x', 1u), prior_,
        Eigen::Matrix<double, 9, 9>::Identity() * 100);
  }

  gtsam::RiExtendedPose3 prior_;
  Eigen::Matrix<double, 9, 1> xi_;
  gtsam::RiExtendedPose3 state_;

  gtsam::RiExtendedPose3Prior factor_;
};

TEST_F(RiExtendedPose3PriorTest, evaluateError) {
  Eigen::Matrix<double, 9, 1> error = factor_.evaluateError(state_);
  EXPECT_LT((error - xi_).lpNorm<Eigen::Infinity>(), 1e-7)
      << "Expected error " << xi_.transpose() << "\nActual error "
      << error.transpose();
}

TEST_F(RiExtendedPose3PriorTest, Jacobian) {
  Eigen::MatrixXd actualH;
  factor_.evaluateError(state_, actualH);

  boost::function<gtsam::Vector9(const gtsam::RiExtendedPose3&)> f =
      boost::bind(&gtsam::RiExtendedPose3Prior::evaluateError, factor_, _1,
                  boost::none);

  Eigen::MatrixXd expectedH = gtsam::numericalDerivative11(f, state_);
  EXPECT_TRUE(gtsam::assert_equal(expectedH, actualH));
}
