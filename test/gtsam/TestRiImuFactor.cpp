
/**
 * @file  TestRiImuFactor.cpp
 * @brief Test right invariant IMU factor.
 */

#include "gtest/gtest.h"
#include "glog/logging.h"

#include <gtsam/base/numericalDerivative.h>
#include <gtsam/inference/Symbol.h>

#include "gtsam/ImuFrontEnd.h"
#include "gtsam/RiImuFactor.h"
#include "../msckf/CovPropConfig.hpp"

class RiImuFactorTest : public ::testing::Test {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  void SetUp() override {
    srand((unsigned int)time(0));
    CovPropConfig cpc(false, true);
    const okvis::ImuMeasurementDeque& imuMeas = cpc.get_imu_measurements();
    Eigen::Matrix<double, 9, 1> speedAndBias_i = cpc.get_sb0();
    extendedPosei_ =
        gtsam::RiExtendedPose3(gtsam::Rot3(cpc.get_q_WS0()),
                               speedAndBias_i.head<3>(), cpc.get_p_WS_W0());
    biasi_ = gtsam::imuBias::ConstantBias(speedAndBias_i.tail<3>(),
                                          speedAndBias_i.segment<3>(3));
    gtsam::RiPreintegratedImuMeasurements ripim(imuMeas, cpc.get_imu_params(),
                                                imuMeas.front().timeStamp,
                                                imuMeas.back().timeStamp);
    ripim.redoPreintegration(extendedPosei_, biasi_);
    riFactor_ =
        gtsam::RiImuFactor(gtsam::Symbol('x', 1u), gtsam::Symbol('x', 2u),
                           gtsam::Symbol('b', 1u), ripim);
  }

  void  checkJacobians(const gtsam::RiExtendedPose3& extendedPosej, bool forceRedo) {
    Eigen::MatrixXd aH1, aH2, aH3; // analytical Jacobians
    const Eigen::VectorXd error =
        riFactor_.evaluateError(extendedPosei_, extendedPosej, biasi_, aH1, aH2, aH3);

    boost::function<gtsam::Vector9(const gtsam::RiExtendedPose3&,
                                   const gtsam::RiExtendedPose3&,
                                   const gtsam::imuBias::ConstantBias&)>
        f = boost::bind(&gtsam::RiImuFactor::evaluateError, riFactor_, _1, _2, _3,
                        boost::none, boost::none, boost::none);

    LOG(INFO) << "gtsam computing numerical Jacobians for state i";
    EXPECT_TRUE(gtsam::assert_equal(
        gtsam::numericalDerivative31(f, extendedPosei_, extendedPosej, biasi_),
        aH1, 1e-4));

    LOG(INFO) << "gtsam computing numerical Jacobians for state j";
    EXPECT_TRUE(gtsam::assert_equal(
        gtsam::numericalDerivative32(f, extendedPosei_, extendedPosej, biasi_),
        aH2, 1e-4));

    // Ensure that the redoIntegration is invoked, otherwise, the Jacobian is
    // simply dx_{j|i}/dx_i instead of de / dx_i.
    double thresholdOrder = std::log10(
        std::max(
            riFactor_.pim().kRelinThresholdGyro,
            riFactor_.pim()
                .RiPreintegratedImuMeasurements::kRelinThresholdAccelerometer) /
        riFactor_.pim().dt());
    double delta = 0;
    if (forceRedo) {
      delta = std::pow(10, std::ceil(thresholdOrder));
    } else {
      delta = std::pow(10, std::floor(thresholdOrder));
    }

    LOG(INFO) << "gtsam computing numerical Jacobians for Ba Bg at i with h " << delta;
    Eigen::Matrix<double, 9, 6> nH3 =
        gtsam::numericalDerivative33(f, extendedPosei_, extendedPosej, biasi_, delta);
    Eigen::Matrix<double, 9, 3> nHa = nH3.leftCols(3);
    Eigen::Matrix<double, 9, 3> nHg = nH3.rightCols(3);
    Eigen::Matrix<double, 9, 3> aHa = aH3.leftCols(3);
    Eigen::Matrix<double, 9, 3> aHg = aH3.rightCols(3);

    EXPECT_TRUE(gtsam::assert_equal(nHa, aHa, 1e-4));
    checkSelectiveRatio(nHg, aHg, 5e-3, 1e-4, 1e-3);
  }

  gtsam::RiExtendedPose3 extendedPosei_;
  gtsam::imuBias::ConstantBias biasi_;
  gtsam::RiImuFactor riFactor_;
};

TEST_F(RiImuFactorTest, JacobiansNoisyStateJ) {
  Eigen::Matrix<double, 9, 1> stateNoise;
  stateNoise.setRandom();
  stateNoise *= 1e-2;
  gtsam::RiPreintegratedImuMeasurements pimCopy = riFactor_.pim();
  const gtsam::RiExtendedPose3 extendedPosej =
      pimCopy.predict(extendedPosei_);
  const gtsam::RiExtendedPose3 extendedPosejNoisy =
      extendedPosej.retract(stateNoise);
  const Eigen::VectorXd errorOrig =
      riFactor_.evaluateError(extendedPosei_, extendedPosej, biasi_);
  EXPECT_LT(errorOrig.lpNorm<Eigen::Infinity>(), 1e-7);
  LOG(INFO) << "Check Jacobians for noisy state j";
  checkJacobians(extendedPosejNoisy, true);
}

TEST_F(RiImuFactorTest, JacobiansRandomStateJ) {
  gtsam::RiExtendedPose3 extendedPosej;
  extendedPosej.setRandom();
  LOG(INFO) << "Check Jacobians for random state j";
  checkJacobians(extendedPosej, true);
}

TEST_F(RiImuFactorTest, JacobiansRandomStateJLin) {
  gtsam::RiExtendedPose3 extendedPosej;
  extendedPosej.setRandom();
  LOG(INFO) << "Check Jacobians for random state j without redo propagation";
  checkJacobians(extendedPosej, false);
}

TEST_F(RiImuFactorTest, evaluateError) {
  gtsam::RiExtendedPose3 extendedPosej;
  extendedPosej.setRandom();
  Eigen::VectorXd errorRef =
      riFactor_.evaluateErrorCheck(extendedPosei_, extendedPosej, biasi_);
  Eigen::VectorXd error =
      riFactor_.evaluateError(extendedPosei_, extendedPosej, biasi_);
  Eigen::VectorXd diff = errorRef - error;

  EXPECT_LT(diff.head<3>().lpNorm<Eigen::Infinity>(), 1e-5)
      << "Rotation error: Ri " << error.head<3>().transpose() << "\nGtsam "
      << errorRef.head<3>().transpose();

  checkSelectiveRatio(errorRef.segment<3>(3), error.segment<3>(3), 5e-2, 0.1);

  EXPECT_LT(diff.tail<3>()
                .cwiseQuotient(errorRef.tail<3>())
                .lpNorm<Eigen::Infinity>(),
            5e-2)
      << "Position error: Ri " << error.tail<3>().transpose() << "\nGtsam "
      << errorRef.tail<3>().transpose();
}

TEST_F(RiImuFactorTest, predict) {
  // compare with prediction results from another method based on gtsam PIM.
  gtsam::RiPreintegratedImuMeasurements ripim = riFactor_.pim();
  gtsam::RiExtendedPose3 extendedPosej = ripim.predict(extendedPosei_);

  okvis::ImuFrontEnd::PimPtr combinedPim;
  okvis::ImuParams imuParamsKimera;
  imuParamsKimera.set(riFactor_.pim().imuParameters());
  imuParamsKimera.imu_preintegration_type_ =
      okvis::ImuPreintegrationType::kPreintegratedCombinedMeasurements;

  okvis::ImuFrontEnd imuIntegrator(imuParamsKimera);
  Eigen::Matrix<double, 9, 1> sb0;
  sb0.head<3>() = extendedPosei_.velocity();
  sb0.segment<3>(3) = biasi_.gyroscope();
  sb0.tail<3>() = biasi_.accelerometer();
  imuIntegrator.preintegrateImuMeasurements(riFactor_.pim().imuMeasurements(),
                                            sb0, riFactor_.pim().ti(),
                                            riFactor_.pim().tj(), combinedPim);

  gtsam::NavState state_i(extendedPosei_.rotation(), extendedPosei_.position(),
                          extendedPosei_.velocity());
  gtsam::NavState predictedState_j = combinedPim->predict(state_i, biasi_);
  EXPECT_TRUE(gtsam::assert_equal(predictedState_j.pose().rotation(),
                                  extendedPosej.rotation()));
  checkSelectiveRatio(predictedState_j.velocity(),
                      extendedPosej.velocity().vector(), 8e-2, 1e-4, 1e-3);
  Eigen::Vector3d predictedPosition_j = predictedState_j.position();
  checkSelectiveRatio(predictedPosition_j, extendedPosej.position().vector(),
                      8e-2, 1e-3, 1e-3);
}

TEST_F(RiImuFactorTest, predict2) {
  gtsam::RiPreintegratedImuMeasurements ripim(riFactor_.pim());
  gtsam::RiExtendedPose3 extendedPosej = ripim.predict(extendedPosei_);

  Eigen::VectorXd error =
      riFactor_.evaluateError(extendedPosei_, extendedPosej, biasi_);
  EXPECT_LT(error.lpNorm<Eigen::Infinity>(), 1e-8)
      << "Error " << error.transpose();
}
