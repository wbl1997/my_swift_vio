/**
 * @file    TestRiProjectionFactorIDP.cpp
 * @brief   Unit tests for RiProjectionFactorIDP and RiProjectionFactorIDPAnchor.
 * @author  Jianzhu Huai
 */

#include "gtsam/RiProjectionFactorIDP.h"
#include "gtsam/RiProjectionFactorIDPAnchor.h"

#include <gtsam/base/numericalDerivative.h>
#include <gtsam/inference/Symbol.h>

#include "gtest/gtest.h"

#include <msckf/memory.h>
#include "msckf/CameraSystemCreator.hpp"
#include "msckf/SimulationNView.h"

class RiProjectionFactorIDPTest : public ::testing::Test {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  void SetUp() override {
    // camera model
    simul::SimCameraModelType cameraModelId = simul::SimCameraModelType::EUROC;
    simul::CameraOrientation cameraOrientationId =
        simul::CameraOrientation::Forward;
    std::string projOptModelName = "FIXED";
    std::string extrinsicModelName = "FIXED";
    double td = 0.0;
    double tr = 0.0;
    simul::CameraSystemCreator csc(cameraModelId, cameraOrientationId,
                                   projOptModelName, extrinsicModelName, td,
                                   tr);
    // reference camera system
    std::shared_ptr<okvis::cameras::CameraBase> cameraGeometryRef;
    std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystemRef;
    csc.createNominalCameraSystem(&cameraGeometryRef, &cameraSystemRef);

    // camera extrinsics
    okvis::kinematics::Transformation T_BC = *cameraSystemRef->T_SC(0u);

    const Eigen::Matrix<double, 2, 1> variance =
        Eigen::Matrix<double, 2, 1>::Constant(1.0);
    const Eigen::Vector2d uv;
    simul::SimulationTwoView snv(0, 1.0, 0.0);
//    simul::SimulationNViewSphere snv(0.0);

    AlignedVector<Eigen::Vector3d> rayList = snv.obsDirections();
    AlignedVector<Eigen::Vector2d> imagePointList;
    for (auto ray : rayList) {
      Eigen::Vector2d imagePoint;
      okvis::cameras::CameraBase::ProjectionStatus status =
          cameraGeometryRef->project(ray, &imagePoint);
      EXPECT_EQ(status,
                okvis::cameras::CameraBase::ProjectionStatus::Successful);
      imagePointList.emplace_back(imagePoint);
    }

    AlignedVector<okvis::kinematics::Transformation> T_WC_list =
        snv.camStates();
    okvis::kinematics::Transformation T_CB = T_BC.inverse();
    for (auto T_WC : T_WC_list) {
      okvis::kinematics::Transformation T_WB = T_WC * T_CB;
      state_list_.emplace_back(gtsam::Rot3(T_WB.q()), Eigen::Vector3d::Random(), T_WB.r());
    }

    Eigen::Vector4d hpW = snv.truePoint();
    Eigen::Vector4d ab1rho = T_WC_list[kAnchorIndex].inverse() * hpW;
    ab1rho /= ab1rho[2];
    abrho_ = gtsam::Point3(ab1rho[0], ab1rho[1], ab1rho[3]);

    factor_ = gtsam::RiProjectionFactorIDP(
        gtsam::Symbol('x', kObservingIndex), gtsam::Symbol('x', kAnchorIndex),
        gtsam::Symbol('l', 1u), variance, imagePointList[kObservingIndex],
        cameraGeometryRef, T_BC, T_BC);
    factorAnchor_ = gtsam::RiProjectionFactorIDPAnchor(
        gtsam::Symbol('l', 1u), variance, imagePointList[kAnchorIndex],
        cameraGeometryRef, T_BC, T_BC);
  }

  gtsam::RiProjectionFactorIDP factor_;
  gtsam::RiProjectionFactorIDPAnchor factorAnchor_;
  AlignedVector<gtsam::RiExtendedPose3> state_list_;
  gtsam::Point3 abrho_;
  const size_t kAnchorIndex = 0u;
  const size_t kObservingIndex = 1u;
};

TEST_F(RiProjectionFactorIDPTest, evaluateError) {
  Eigen::Vector2d error = factor_.evaluateError(
      state_list_[kObservingIndex], state_list_[kAnchorIndex], abrho_);
  EXPECT_LT(error.lpNorm<Eigen::Infinity>(), 1e-8);
}

TEST_F(RiProjectionFactorIDPTest, evaluateErrorAnchor) {
  Eigen::Vector2d error = factorAnchor_.evaluateError(abrho_);
  EXPECT_LT(error.lpNorm<Eigen::Infinity>(), 1e-8);
}

TEST_F(RiProjectionFactorIDPTest, jacobians) {
  Eigen::MatrixXd aHj, aHa, aHp;
  factor_.evaluateError(state_list_[kObservingIndex], state_list_[kAnchorIndex],
                        abrho_, aHj, aHa, aHp);
  boost::function<Eigen::Vector2d(const gtsam::RiExtendedPose3&,
                                  const gtsam::RiExtendedPose3&,
                                  const gtsam::Point3&)>
      evaluate =
          boost::bind(&gtsam::RiProjectionFactorIDP::evaluateError, factor_, _1,
                      _2, _3, boost::none, boost::none, boost::none);
  EXPECT_TRUE(gtsam::assert_equal(
      gtsam::numericalDerivative31(evaluate, state_list_[kObservingIndex],
                                   state_list_[kAnchorIndex], abrho_),
      aHj, 1e-7));
  EXPECT_TRUE(gtsam::assert_equal(
      gtsam::numericalDerivative32(evaluate, state_list_[kObservingIndex],
                                   state_list_[kAnchorIndex], abrho_),
      aHa, 1e-7));
  EXPECT_TRUE(gtsam::assert_equal(
      gtsam::numericalDerivative33(evaluate, state_list_[kObservingIndex],
                                   state_list_[kAnchorIndex], abrho_),
      aHp, 1e-8));
}

TEST_F(RiProjectionFactorIDPTest, jacobiansAnchor) {
  Eigen::MatrixXd aHp;
  factorAnchor_.evaluateError(abrho_, aHp);

  boost::function<Eigen::Vector2d(const gtsam::Point3&)>
      evaluate =
          boost::bind(&gtsam::RiProjectionFactorIDPAnchor::evaluateError, factorAnchor_, _1,
                      boost::none);
  EXPECT_TRUE(gtsam::assert_equal(
      gtsam::numericalDerivative11(evaluate, abrho_),
      aHp, 1e-8));
}
