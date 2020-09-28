/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 *  @file  TestRiSmartProjectionFactor.cpp
 *  @brief Unit tests for RiSmartProjectionFactor Class
 *  @author Chris Beall
 *  @author Luca Carlone
 *  @author Zsolt Kira
 *  @author Frank Dellaert
 *  @date   Sept 2013
 */
#include <gtsam/RiSmartProjectionFactor.h>

#include <gtsam/slam/dataset.h>
#include "smartFactorScenarios.h"

#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <gtsam/base/serializationTestHelpers.h>
// #include <CppUnitLite/TestHarness.h>
#include <boost/assign/std/map.hpp>
#include <iostream>

#include <gtest/gtest.h>

#include <msckf/memory.h>

#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>

using namespace boost::assign;
#define EXPECT_DOUBLES_EQUAL(expected, actual, threshold) \
  EXPECT_LT(std::fabs(expected - actual), threshold);

#define DOUBLES_EQUAL(expected, actual, threshold) \
  EXPECT_LT(std::fabs(expected - actual), threshold);

static const Symbol l1('l', 1), l2('l', 2), l3('l', 3);
static const Key c1 = 1, c2 = 2;
static const Point2 measurement1(323.0, 240.0);
static const double rankTol = 1.0;

template <class CALIBRATION>
PinholeCamera<CALIBRATION> perturbCameraPoseAndCalibration(
    const PinholeCamera<CALIBRATION>& camera) {
  Pose3 noise_pose =
      Pose3(Rot3::Ypr(-M_PI / 10, 0., -M_PI / 10), Point3(0.5, 0.1, 0.3));
  Pose3 cameraPose = camera.pose();
  Pose3 perturbedCameraPose = cameraPose.compose(noise_pose);
  typename gtsam::traits<CALIBRATION>::TangentVector d;
  d.setRandom();
  d *= 0.1;
  CALIBRATION perturbedCalibration = camera.calibration().retract(d);
  return PinholeCamera<CALIBRATION>(perturbedCameraPose, perturbedCalibration);
}

namespace vanilla {
boost::shared_ptr<Cal3_S2> Kptr(new Cal3_S2(K2));
std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry(
    new okvis::cameras::PinholeCamera<
        okvis::cameras::RadialTangentialDistortion>(
        w, h, Kptr->fx(), Kptr->fy(), Kptr->px(), Kptr->py(),
        okvis::cameras::RadialTangentialDistortion(0, 0, 0, 0), 0,
        0 /*, id ?*/));

gtsam::Pose3 body_P_sensor = gtsam::Pose3();
}  // namespace vanilla

namespace vanillaInf {
typedef PinholeCamera<Cal3_S2> Camera;
typedef SmartProjectionFactor<Camera> SmartFactor;
static Cal3_S2 K(fov, w, h);
static Cal3_S2 K2(1500, 1200, 0, w, h);
Camera level_camera(level_pose, K2);
Camera level_camera_right(pose_right, K2);
gtsam::Point3 landmarkInf(50000, 0.5, 1.2);
Point2 level_uv = level_camera.project(landmarkInf);
Point2 level_uv_right = level_camera_right.project(landmarkInf);
Camera cam1(level_pose, K2);
Camera cam2(pose_right, K2);
Camera cam3(pose_above, K2);
typedef GeneralSFMFactor<Camera, Point3> SFMFactor;
SmartProjectionParams params;

boost::shared_ptr<Cal3_S2> Kptr(new Cal3_S2(K2));
std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry(
    new okvis::cameras::PinholeCamera<
        okvis::cameras::RadialTangentialDistortion>(
        w, h, Kptr->fx(), Kptr->fy(), Kptr->px(), Kptr->py(),
        okvis::cameras::RadialTangentialDistortion(0, 0, 0, 0), 0,
        0 /*, id ?*/));

gtsam::Pose3 body_P_sensor = gtsam::Pose3();
}  // namespace vanillaInf

/* ************************************************************************* */
TEST(RiSmartProjectionFactor, perturbCameraPose) {
  using namespace vanilla;
  Pose3 noise_pose =
      Pose3(Rot3::Ypr(-M_PI / 10, 0., -M_PI / 10), Point3(0.5, 0.1, 0.3));
  Pose3 perturbed_level_pose = level_pose.compose(noise_pose);
  Camera actualCamera(perturbed_level_pose, K2);

  Camera expectedCamera = perturbCameraPose(level_camera);
  ASSERT_TRUE(assert_equal(expectedCamera, actualCamera));
}

/* ************************************************************************* */
TEST(RiSmartProjectionFactor, Constructor) {
  using namespace vanilla;
  RiSmartProjectionFactor<Cal3_S2>::shared_ptr factor1(
      new RiSmartProjectionFactor<Cal3_S2>(unit2, body_P_sensor, Kptr, cameraGeometry));
}

/* ************************************************************************* */
TEST(RiSmartProjectionFactor, Constructor2) {
  using namespace vanilla;
  params.setRankTolerance(rankTol);

  RiSmartProjectionFactor<Cal3_S2> factor1(unit2, body_P_sensor, Kptr, cameraGeometry, params);
}

/* ************************************************************************* */
TEST(RiSmartProjectionFactor, Constructor3) {
  using namespace vanilla;
  RiSmartProjectionFactor<Cal3_S2>::shared_ptr factor1(
      new RiSmartProjectionFactor<Cal3_S2>(unit2, body_P_sensor, Kptr, cameraGeometry));
  factor1->add(measurement1, c1);
}

/* ************************************************************************* */
TEST(RiSmartProjectionFactor, Constructor4) {
  using namespace vanilla;
  params.setRankTolerance(rankTol);
  RiSmartProjectionFactor<Cal3_S2> factor1(unit2, body_P_sensor, Kptr, cameraGeometry, params);
  factor1.add(measurement1, c1);
}

/* ************************************************************************* */
TEST(RiSmartProjectionFactor, Equals) {
  using namespace vanilla;
  RiSmartProjectionFactor<Cal3_S2>::shared_ptr factor1(
      new RiSmartProjectionFactor<Cal3_S2>(unit2, body_P_sensor, Kptr, cameraGeometry));
  factor1->add(measurement1, c1);

  RiSmartProjectionFactor<Cal3_S2>::shared_ptr factor2(
      new RiSmartProjectionFactor<Cal3_S2>(unit2, body_P_sensor, Kptr, cameraGeometry));
  factor2->add(measurement1, c1);
}

/* *************************************************************************/
TEST(RiSmartProjectionFactor, noiseless) {
  using namespace vanilla;

  Values values;
  gtsam::Pose3 T_WB1 = level_pose * body_P_sensor.inverse();
  gtsam::Pose3 T_WB2 = pose_right * body_P_sensor.inverse();
  values.insert(c1, RiExtendedPose3(T_WB1.rotation(), gtsam::Point3(), T_WB1.translation()));
  values.insert(c2, RiExtendedPose3(T_WB2.rotation(), gtsam::Point3(), T_WB2.translation()));

  RiSmartProjectionFactor<Cal3_S2>::shared_ptr factor1(
      new RiSmartProjectionFactor<Cal3_S2>(unit2, body_P_sensor, Kptr, cameraGeometry));
  factor1->add(level_uv, c1);
  factor1->add(level_uv_right, c2);

  double expectedError = 0.0;
  DOUBLES_EQUAL(expectedError, factor1->error(values), 1e-7);
  ASSERT_TRUE(assert_equal(
      Z_4x1, factor1->reprojectionErrorAfterTriangulation(values), 1e-7));

  factor1->setAnchorIndex(0u);
  AlignedVector<RiExtendedPose3> stateList{RiExtendedPose3(T_WB1.rotation(), gtsam::Point3(), T_WB1.translation()),
        RiExtendedPose3(T_WB2.rotation(), gtsam::Point3(), T_WB2.translation())};
  Eigen::Vector3d pC = level_pose.transformTo(landmark1);
  double zinv = 1.0 / pC[2];
  Eigen::Vector3d abrho(pC[0] * zinv, pC[1] * zinv, zinv);
  ASSERT_TRUE(assert_equal(
      Z_4x1, factor1->unwhitenedError(stateList, abrho), 1e-7));
}

/* *************************************************************************/
TEST(RiSmartProjectionFactor, noisy) {
  using namespace vanilla;

  // Project one landmark into two cameras and add noise on first
  Point2 level_uv = level_camera.project(landmark1) + Point2(0.2, 0.2);
  Point2 level_uv_right = level_camera_right.project(landmark1);

  gtsam::Pose3 T_WB1 = level_pose * body_P_sensor.inverse();
  gtsam::Pose3 T_WB2 = pose_right * body_P_sensor.inverse();
  Values values;
  values.insert(c1, RiExtendedPose3(T_WB1.rotation(), gtsam::Point3(), T_WB1.translation()));
  Camera perturbed_level_camera_right = perturbCameraPose(level_camera_right);
  gtsam::Pose3 perturbed_pose_right(perturbed_level_camera_right.pose() * body_P_sensor.inverse());
  values.insert(c2, RiExtendedPose3(perturbed_pose_right.rotation(), gtsam::Point3(), perturbed_pose_right.translation()));

  RiSmartProjectionFactor<Cal3_S2>::shared_ptr factor1(
      new RiSmartProjectionFactor<Cal3_S2>(unit2, body_P_sensor, Kptr, cameraGeometry));
  factor1->add(level_uv, c1);
  factor1->add(level_uv_right, c2);

  // Point is now uninitialized before a triangulation event
  EXPECT_TRUE(!factor1->point());

  double expectedError = 58640;
  double actualError1 = factor1->error(values);
  EXPECT_DOUBLES_EQUAL(expectedError, actualError1, 1);

  // Check triangulated point
  ASSERT_TRUE(factor1->point());
  EXPECT_TRUE(assert_equal(Point3(13.7587, 1.43851, -1.14274),
                           *factor1->point(), 1e-4));

  // Check whitened errors
  Vector expected(4);
  expected << -7, 235, 58, -242;
  RiSmartProjectionFactor<Cal3_S2>::Cameras cameras1 = factor1->cameras(values);
  Point3 point1 = *factor1->point();
  Vector actual = factor1->whitenedError(cameras1, point1);
  EXPECT_TRUE(assert_equal(expected, actual, 1));

  RiSmartProjectionFactor<Cal3_S2>::shared_ptr factor2(
      new RiSmartProjectionFactor<Cal3_S2>(unit2, body_P_sensor, Kptr, cameraGeometry));
  Point2Vector measurements;
  measurements.push_back(level_uv);
  measurements.push_back(level_uv_right);

  KeyVector views{c1, c2};

  factor2->add(measurements, views);

  double actualError2 = factor2->error(values);
  EXPECT_DOUBLES_EQUAL(expectedError, actualError2, 1);
}

/* *************************************************************************/
TEST(RiSmartProjectionFactor, noiselessBundler) {
  using namespace vanilla;
  Values values;
  gtsam::Pose3 T_WB1 = level_pose * body_P_sensor.inverse();
  gtsam::Pose3 T_WB2 = pose_right * body_P_sensor.inverse();
  values.insert(c1, RiExtendedPose3(T_WB1.rotation(), gtsam::Point3(), T_WB1.translation()));
  values.insert(c2, RiExtendedPose3(T_WB2.rotation(), gtsam::Point3(), T_WB2.translation()));

  RiSmartProjectionFactor<Cal3_S2>::shared_ptr factor1(
      new RiSmartProjectionFactor<Cal3_S2>(unit2, body_P_sensor, Kptr, cameraGeometry));
  factor1->add(level_uv, c1);
  factor1->add(level_uv_right, c2);

  double actualError = factor1->error(values);

  double expectedError = 0.0;
  DOUBLES_EQUAL(expectedError, actualError, 1e-3);

  Point3 oldPoint(0, 0, 0);  // this takes the point stored in the factor (we
                             // are not interested in this)
  if (factor1->point()) oldPoint = *(factor1->point());

  Point3 expectedPoint(0, 0, 0);
  if (factor1->point(values)) expectedPoint = *(factor1->point(values));

  EXPECT_TRUE(assert_equal(expectedPoint, landmark1, 1e-3));
}

/* *************************************************************************/
TEST(RiSmartProjectionFactor, comparisonGeneralSfMFactor) {
  using namespace vanilla;
  Values values;
  gtsam::Pose3 T_WB1 = level_pose * body_P_sensor.inverse();
  gtsam::Pose3 T_WB2 = pose_right * body_P_sensor.inverse();
  values.insert(c1, RiExtendedPose3(T_WB1.rotation(), gtsam::Point3(), T_WB1.translation()));
  values.insert(c2, RiExtendedPose3(T_WB2.rotation(), gtsam::Point3(), T_WB2.translation()));

  NonlinearFactorGraph smartGraph;
  RiSmartProjectionFactor<Cal3_S2>::shared_ptr factor1(
      new RiSmartProjectionFactor<Cal3_S2>(unit2, body_P_sensor, Kptr, cameraGeometry));
  factor1->add(level_uv, c1);
  factor1->add(level_uv_right, c2);
  smartGraph.push_back(factor1);
  double expectedError = factor1->error(values);
  double expectedErrorGraph = smartGraph.error(values);
  Point3 expectedPoint(0, 0, 0);
  if (factor1->point()) expectedPoint = *(factor1->point());

  // COMMENTS:
  // 1) triangulation introduces small errors, then for a fair comparison we use
  // expectedPoint as value in the generalGrap
  NonlinearFactorGraph generalGraph;
  SFMFactor sfm1(level_uv, unit2, c1, l1);
  SFMFactor sfm2(level_uv_right, unit2, c2, l1);
  generalGraph.push_back(sfm1);
  generalGraph.push_back(sfm2);
  Values gtsamValues;
  gtsamValues.insert(c1, level_camera);
  gtsamValues.insert(c2, level_camera_right);
  gtsamValues.insert(l1, expectedPoint);  // note: we get rid of possible errors in
                                     // the triangulation
  Vector e1 = sfm1.evaluateError(level_camera, expectedPoint);
  Vector e2 = sfm2.evaluateError(level_camera_right, expectedPoint);
  double actualError = 0.5 * (e1.squaredNorm() + e2.squaredNorm());
  double actualErrorGraph = generalGraph.error(gtsamValues);

  DOUBLES_EQUAL(expectedErrorGraph, actualErrorGraph, 1e-7);
  DOUBLES_EQUAL(expectedErrorGraph, expectedError, 1e-7);
  DOUBLES_EQUAL(actualErrorGraph, actualError, 1e-7);
  DOUBLES_EQUAL(expectedError, actualError, 1e-7);
}

/* *************************************************************************/
TEST(RiSmartProjectionFactor, compareHessianAndInfoVector) {
  // values 2 RiExtendedPose3 of camera frames and 1 landmark in inverse depth
  // factors 1 RiProjectionFactorIDPAnchor, 1 RiProjectionFactorIDP
  // VS. factors 1 RiSmartProjectionFactor with Hessian

  using namespace vanilla;
  Values values;
  gtsam::Pose3 T_WB1 = level_pose * body_P_sensor.inverse();
  gtsam::Pose3 T_WB2 = pose_right * body_P_sensor.inverse();
  values.insert(c1, RiExtendedPose3(T_WB1.rotation(), gtsam::Point3(), T_WB1.translation()));
  values.insert(c2, RiExtendedPose3(T_WB2.rotation(), gtsam::Point3(), T_WB2.translation()));

  NonlinearFactorGraph smartGraph;
  RiSmartProjectionFactor<Cal3_S2>::shared_ptr factor1(
      new RiSmartProjectionFactor<Cal3_S2>(unit2, body_P_sensor, Kptr, cameraGeometry));
  factor1->add(level_uv, c1);
  factor1->add(level_uv_right, c2);
  factor1->setAnchorIndex(0u);
  smartGraph.push_back(factor1);
  Matrix expectedHessian = smartGraph.linearize(values)->hessian().first;
  Vector expectedInfoVector = smartGraph.linearize(values)->hessian().second;
  Point3 expectedPoint(0, 0, 0);
  if (factor1->point()) expectedPoint = *(factor1->point());

  // COMMENTS:
  // 1) triangulation introduces small errors, then for a fair comparison we use
  // expectedPoint as value in the generalGrap
  NonlinearFactorGraph generalGraph;

  okvis::kinematics::Transformation T_BC(body_P_sensor.translation(), body_P_sensor.rotation().toQuaternion());
  RiProjectionFactorIDPAnchor sfm1(l1, Eigen::Vector2d(1.0, 1.0), level_uv, cameraGeometry, T_BC, T_BC);
  RiProjectionFactorIDP sfm2(c2, c1, l1, Eigen::Vector2d(1.0, 1.0), level_uv_right, cameraGeometry, T_BC, T_BC);
  generalGraph.push_back(sfm1);
  generalGraph.push_back(sfm2);
  Eigen::Vector3d pC = level_pose.transformTo(expectedPoint);
  double zinv = 1.0 / pC[2];
  gtsam::Point3 abrho(pC[0] * zinv, pC[1] * zinv, zinv);
  values.insert(l1, abrho);  // note: we get rid of possible errors in
                                     // the triangulation
  Eigen::Vector2d error1 = sfm1.evaluateError(abrho);
  Eigen::Vector2d error2 = sfm2.evaluateError(
      RiExtendedPose3(T_WB2.rotation(), gtsam::Point3(), T_WB2.translation()),
      RiExtendedPose3(T_WB1.rotation(), gtsam::Point3(), T_WB1.translation()),
      abrho);
  EXPECT_LT(error1.lpNorm<Eigen::Infinity>(), 1e-7);
  EXPECT_LT(error2.lpNorm<Eigen::Infinity>(), 1e-7);
  Matrix actualFullHessian = generalGraph.linearize(values)->hessian().first;
  Matrix actualFullInfoVector = generalGraph.linearize(values)->hessian().second;
  Matrix actualHessian = actualFullHessian.block(0, 0, 18, 18) -
                         actualFullHessian.block(0, 18, 18, 3) *
                             (actualFullHessian.block(18, 18, 3, 3)).inverse() *
                             actualFullHessian.block(18, 0, 3, 18);
  Vector actualInfoVector =
      actualFullInfoVector.block(0, 0, 18, 1) -
      actualFullHessian.block(0, 18, 18, 3) *
          (actualFullHessian.block(18, 18, 3, 3)).inverse() *
          actualFullInfoVector.block(18, 0, 3, 1);

  EXPECT_TRUE(assert_equal(expectedHessian, actualHessian, 1e-7));
  EXPECT_TRUE(assert_equal(expectedInfoVector, actualInfoVector, 1e-7));
}

TEST(RiSmartProjectionFactor, approximateErrorByHessianAndInfoVector) {
  // values 2 RiExtendedPose3 of camera frames and 1 landmark in inverse depth
  // factors 1 RiProjectionFactorIDPAnchor, 1 RiProjectionFactorIDP
  // VS. factors 1 RiSmartProjectionFactor with Hessian

  using namespace vanilla;
  Values values;
  gtsam::Pose3 T_WB1 = level_pose * body_P_sensor.inverse();
  gtsam::Pose3 T_WB2 = pose_right * body_P_sensor.inverse();
  RiExtendedPose3 v1(T_WB1.rotation(), gtsam::Point3(), T_WB1.translation());
  RiExtendedPose3 v2(T_WB2.rotation(), gtsam::Point3(), T_WB2.translation());
  values.insert(c1, v1);
  values.insert(c2, v2);

  NonlinearFactorGraph smartGraph;
  RiSmartProjectionFactor<Cal3_S2>::shared_ptr factor1(
      new RiSmartProjectionFactor<Cal3_S2>(unit2, body_P_sensor, Kptr,
                                           cameraGeometry));
  factor1->add(level_uv, c1);
  factor1->add(level_uv_right, c2);
  factor1->setAnchorIndex(0u);
  smartGraph.push_back(factor1);
  Matrix expectedHessian = smartGraph.linearize(values)->hessian().first;
  Vector expectedInfoVector = smartGraph.linearize(values)->hessian().second;
  Point3 expectedPoint(0, 0, 0);
  if (factor1->point()) expectedPoint = *(factor1->point());
  Eigen::Vector3d pC = level_pose.transformTo(expectedPoint);
  double zinv = 1.0 / pC[2];
  gtsam::Point3 v3(pC[0] * zinv, pC[1] * zinv, zinv);

  double dh = 1e-3;
  Eigen::Matrix<double, 9, 1> dx1 = Eigen::Matrix<double, 9, 1>::Random() * dh;


  RiExtendedPose3 perturbedv1 = v1.retract(dx1);
  RiExtendedPose3 perturbedv2 = v2.retract(-dx1);

  Values perturbedValues;
  perturbedValues.insert(c1, perturbedv1);
  perturbedValues.insert(c2, perturbedv2);
  perturbedValues.insert(l1, v3);

  double error = factor1->error(values);
  double perturbedError = factor1->error(perturbedValues);

  Eigen::VectorXd delta(9 + 9);
  delta << dx1, -dx1;
  Eigen::MatrixXd expectedDeltaEigen =
      0.5 * delta.transpose() * expectedHessian * delta -
      delta.transpose() * expectedInfoVector;
  double expectedDelta = expectedDeltaEigen(0, 0);
  double actualDelta = perturbedError - error;
  EXPECT_LT(std::fabs(expectedDelta - actualDelta), 5e-4);
}


/* *************************************************************************/
TEST(RiSmartProjectionFactor, approximateErrorByHessianAndInfoVectorInf) {
  // values 2 RiExtendedPose3 of camera frames and 1 landmark in inverse depth
  // factors 1 RiProjectionFactorIDPAnchor, 1 RiProjectionFactorIDP
  // VS. factors 1 RiSmartProjectionFactor with Hessian

  using namespace vanillaInf;
  Values values;
  gtsam::Pose3 T_WB1 = level_pose * body_P_sensor.inverse();
  gtsam::Pose3 T_WB2 = pose_right * body_P_sensor.inverse();
  RiExtendedPose3 v1(T_WB1.rotation(), gtsam::Point3(), T_WB1.translation());
  RiExtendedPose3 v2(T_WB2.rotation(), gtsam::Point3(), T_WB2.translation());
  values.insert(c1, v1);
  values.insert(c2, v2);

  NonlinearFactorGraph smartGraph;
  RiSmartProjectionFactor<Cal3_S2>::shared_ptr factor1(
      new RiSmartProjectionFactor<Cal3_S2>(unit2, body_P_sensor, Kptr,
                                           cameraGeometry));
  factor1->add(level_uv, c1);
  factor1->add(level_uv_right, c2);
  factor1->setAnchorIndex(0u);
  smartGraph.push_back(factor1);
  Matrix expectedHessian = smartGraph.linearize(values)->hessian().first;
  Vector expectedInfoVector = smartGraph.linearize(values)->hessian().second;
  Point3 expectedPoint(0, 0, 0);
  if (factor1->point()) expectedPoint = *(factor1->point());
  Eigen::Vector3d pC = level_pose.transformTo(expectedPoint);
  double zinv = 1.0 / pC[2];
  gtsam::Point3 v3(pC[0] * zinv, pC[1] * zinv, zinv);

  double dh = 1e-3;
  Eigen::Matrix<double, 9, 1> dx1 = Eigen::Matrix<double, 9, 1>::Random() * dh;


  RiExtendedPose3 perturbedv1 = v1.retract(dx1);
  RiExtendedPose3 perturbedv2 = v2.retract(-dx1);

  Values perturbedValues;
  perturbedValues.insert(c1, perturbedv1);
  perturbedValues.insert(c2, perturbedv2);
  perturbedValues.insert(l1, v3);

  double error = factor1->error(values);
  double perturbedError = factor1->error(perturbedValues);

  Eigen::VectorXd delta(9 + 9);
  delta << dx1, -dx1;
  Eigen::MatrixXd expectedDeltaEigen =
      0.5 * delta.transpose() * expectedHessian * delta -
      delta.transpose() * expectedInfoVector;
  double expectedDelta = expectedDeltaEigen(0, 0);
  double actualDelta = perturbedError - error;
  EXPECT_LT(std::fabs(expectedDelta - actualDelta), 5e-4)
      << "Expected from Hessian " << expectedDelta << " Actual by projection "
      << actualDelta;
}

/* *************************************************************************/
// Have to think about how to compare multiplyHessianAdd in generalSfMFactor and
// smartFactors
// TEST(RiSmartProjectionFactor, comparisonGeneralSfMFactor2 ){
//
//  Values values;
//  gtsam::Pose3 T_WB1 = level_pose * body_P_sensor.inverse();
//  gtsam::Pose3 T_WB2 = pose_right * body_P_sensor.inverse();
//  values.insert(c1, RiExtendedPose3(T_WB1.rotation(), gtsam::Point3(), T_WB1.translation()));
//  values.insert(c2, RiExtendedPose3(T_WB2.rotation(), gtsam::Point3(), T_WB2.translation()));
//
//  NonlinearFactorGraph smartGraph;
//  RiSmartProjectionFactor<Cal3Bundler>::shared_ptr factor1(new
//  RiSmartProjectionFactor<Cal3Bundler>()); factor1->add(level_uv, c1, unit2);
//  factor1->add(level_uv_right, c2, unit2);
//  factor1->setAnchorIndex(0u);
//  smartGraph.push_back(factor1);
//  GaussianFactorGraph::shared_ptr gfgSmart = smartGraph.linearize(values);
//
//  Point3 expectedPoint(0,0,0);
//  if(factor1->point())
//    expectedPoint = *(factor1->point());
//
//  // COMMENTS:
//  // 1) triangulation introduces small errors, then for a fair comparison we
//  use expectedPoint as
//  // value in the generalGrap
//  NonlinearFactorGraph generalGraph;
//  SFMFactor sfm1(level_uv, unit2, c1, l1);
//  SFMFactor sfm2(level_uv_right, unit2, c2, l1);
//  generalGraph.push_back(sfm1);
//  generalGraph.push_back(sfm2);
//  values.insert(l1, expectedPoint); // note: we get rid of possible errors in
//  the triangulation GaussianFactorGraph::shared_ptr gfg =
//  generalGraph.linearize(values);
//
//  double alpha = 1.0;
//
//  VectorValues yExpected, yActual, ytmp;
//  VectorValues xtmp = map_list_of
//      (c1, (Vec(9) << 0,0,0,0,0,0,0,0,0))
//      (c2, (Vec(9) << 0,0,0,0,0,0,0,0,0))
//      (l1, (Vec(3) << 5.5, 0.5, 1.2));
//  gfg ->multiplyHessianAdd(alpha, xtmp, ytmp);
//
//  VectorValues x = map_list_of
//      (c1, (Vec(9) << 1,2,3,4,5,6,7,8,9))
//      (c2, (Vec(9) << 11,12,13,14,15,16,17,18,19))
//      (l1, (Vec(3) << 5.5, 0.5, 1.2));
//
//  gfgSmart->multiplyHessianAdd(alpha, ytmp + x, yActual);
//  gfg ->multiplyHessianAdd(alpha, x, yExpected);
//
//  EXPECT_TRUE(assert_equal(yActual,yExpected, 1e-7));
//}
/* *************************************************************************/
TEST(RiSmartProjectionFactor, computeImplicitJacobian) {
  using namespace vanilla;
  Values values;
  gtsam::Pose3 T_WB1 = level_pose * body_P_sensor.inverse();
  gtsam::Pose3 T_WB2 = pose_right * body_P_sensor.inverse();
  values.insert(c1, RiExtendedPose3(T_WB1.rotation(), gtsam::Point3(), T_WB1.translation()));
  values.insert(c2, RiExtendedPose3(T_WB2.rotation(), gtsam::Point3(), T_WB2.translation()));

  RiSmartProjectionFactor<Cal3_S2>::shared_ptr factor1(
      new RiSmartProjectionFactor<Cal3_S2>(unit2, body_P_sensor, Kptr, cameraGeometry));
  factor1->add(level_uv, c1);
  factor1->add(level_uv_right, c2);
  factor1->setAnchorIndex(0u);
  Matrix expectedE;
  Vector expectedb;

  AlignedVector<RiExtendedPose3> stateList;
  stateList.emplace_back(T_WB1.rotation(), gtsam::Point3(), T_WB1.translation());
  stateList.emplace_back(T_WB2.rotation(), gtsam::Point3(), T_WB2.translation());

  factor1->error(
      values);  // this is important to have a triangulation of the point
  Point3 point(0, 0, 0);
  if (factor1->point()) point = *(factor1->point());
  Eigen::Vector3d pC = level_pose.transformTo(point);
  double zinv = 1.0 / pC[2];
  Eigen::Vector3d abrho(pC[0] * zinv, pC[1] * zinv, zinv);

  Matrix Fs;
  factor1->computeJacobians(Fs, expectedE, expectedb, stateList, abrho);
  Matrix3 expectedVinv = factor1->PointCov(expectedE);

  NonlinearFactorGraph generalGraph;
  okvis::kinematics::Transformation T_BC(body_P_sensor.translation(), body_P_sensor.rotation().toQuaternion());
  RiProjectionFactorIDPAnchor sfm1(l1, Eigen::Vector2d(1.0, 1.0), level_uv, cameraGeometry, T_BC, T_BC);
  RiProjectionFactorIDP sfm2(c2, c1, l1, Eigen::Vector2d(1.0, 1.0), level_uv_right, cameraGeometry, T_BC, T_BC);
  generalGraph.push_back(sfm1);
  generalGraph.push_back(sfm2);
  gtsam::Point3 abrhoValue(abrho);
  values.insert(l1, abrhoValue);  // note: we get rid of possible errors in
                                     // the triangulation
  Eigen::Vector2d error1 = sfm1.evaluateError(abrho);
  Eigen::Vector2d error2 = sfm2.evaluateError(
      RiExtendedPose3(T_WB2.rotation(), gtsam::Point3(), T_WB2.translation()),
      RiExtendedPose3(T_WB1.rotation(), gtsam::Point3(), T_WB1.translation()),
      abrho);
  EXPECT_LT(error1.lpNorm<Eigen::Infinity>(), 1e-7);
  EXPECT_LT(error2.lpNorm<Eigen::Infinity>(), 1e-7);
  Matrix actualFullHessian = generalGraph.linearize(values)->hessian().first;
  Matrix actualVinv = (actualFullHessian.block(18, 18, 3, 3)).inverse();

  EXPECT_TRUE(assert_equal(expectedVinv, actualVinv, 1e-7));
}

/* *************************************************************************/
//TEST(RiSmartProjectionFactor, implicitJacobianFactor) {
//  using namespace vanilla;

//  Values values;
//  gtsam::Pose3 T_WB1 = level_pose * body_P_sensor.inverse();
//  gtsam::Pose3 T_WB2 = pose_right * body_P_sensor.inverse();
//  values.insert(c1, RiExtendedPose3(T_WB1.rotation(), gtsam::Point3(), T_WB1.translation()));
//  values.insert(c2, RiExtendedPose3(T_WB2.rotation(), gtsam::Point3(), T_WB2.translation()));
//  double rankTol = 1;
//  bool useEPI = false;

//  // Hessian version
//  SmartProjectionParams params;
//  params.setRankTolerance(rankTol);
//  params.setLinearizationMode(gtsam::HESSIAN);
//  params.setDegeneracyMode(gtsam::IGNORE_DEGENERACY);
//  params.setEnableEPI(useEPI);

//  RiSmartProjectionFactor<Cal3_S2>::shared_ptr explicitFactor(
//      new RiSmartProjectionFactor<Cal3_S2>(unit2, body_P_sensor, Kptr, cameraGeometry, params));
//  explicitFactor->add(level_uv, c1);
//  explicitFactor->add(level_uv_right, c2);
//  explicitFactor->setAnchorIndex(0u);
//  GaussianFactor::shared_ptr gaussianHessianFactor =
//      explicitFactor->linearize(values);
//  HessianFactor& hessianFactor =
//      dynamic_cast<HessianFactor&>(*gaussianHessianFactor);

//  // Implicit Schur version
//  params.setLinearizationMode(gtsam::IMPLICIT_SCHUR);
//  RiSmartProjectionFactor<Cal3_S2>::shared_ptr implicitFactor(
//      new RiSmartProjectionFactor<Cal3_S2>(unit2, body_P_sensor, Kptr, cameraGeometry, params));
//  implicitFactor->add(level_uv, c1);
//  implicitFactor->add(level_uv_right, c2);
//  GaussianFactor::shared_ptr gaussianImplicitSchurFactor =
//      implicitFactor->linearize(values);
//  ASSERT_TRUE(gaussianImplicitSchurFactor);
//  typedef RegularImplicitSchurFactor<Camera> Implicit9;
//  Implicit9& implicitSchurFactor =
//      dynamic_cast<Implicit9&>(*gaussianImplicitSchurFactor);

//  VectorValues x =
//      map_list_of(c1, (Vector(9) << 1, 2, 3, 4, 5, 6, 7, 8, 9).finished())(
//          c2, (Vector(9) << 11, 12, 13, 14, 15, 16, 17, 18, 19).finished());

//  VectorValues yExpected, yActual;
//  double alpha = 1.0;
//  hessianFactor.multiplyHessianAdd(alpha, x, yActual);
//  implicitSchurFactor.multiplyHessianAdd(alpha, x, yExpected);
//  EXPECT_TRUE(assert_equal(yActual, yExpected, 1e-7));
//}

/* ************************************************************************* */
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Constrained,
                        "gtsam_noiseModel_Constrained");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Diagonal,
                        "gtsam_noiseModel_Diagonal");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Gaussian,
                        "gtsam_noiseModel_Gaussian");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Unit, "gtsam_noiseModel_Unit");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Isotropic,
                        "gtsam_noiseModel_Isotropic");
BOOST_CLASS_EXPORT_GUID(gtsam::SharedNoiseModel, "gtsam_SharedNoiseModel");
BOOST_CLASS_EXPORT_GUID(gtsam::SharedDiagonal, "gtsam_SharedDiagonal");

// This test fails because RiSmartProjectionFactor::cameraGeometry_ has
// protected constructor which prevents boost serialization.
// TEST(RiSmartProjectionFactor, serialize) {
//  using namespace vanilla;
//  using namespace gtsam::serializationTestHelpers;
//  RiSmartProjectionFactor<Cal3_S2> factor(unit2, body_P_sensor, Kptr,
//  cameraGeometry);

//  EXPECT_TRUE(equalsObj(factor));
//  EXPECT_TRUE(equalsXML(factor));
//  EXPECT_TRUE(equalsBinary(factor));
//}
