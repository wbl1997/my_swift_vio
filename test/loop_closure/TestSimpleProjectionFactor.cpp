
#include <gtest/gtest.h>
#include <loop_closure/GtsamWrap.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include "glog/logging.h"
#include "swift_vio/ceres/tiny_solver.h"

TEST(GtsamWrap, StackedProjectionFactorDynamic) {
  // initialize random number generator
  // srand((unsigned int) time(0)); // disabled: make unit tests deterministic

  LOG(INFO) << "set up a random geometry... ";
  okvis::kinematics::Transformation T_WS;  // world to sensor
  T_WS.setRandom(10.0, M_PI);
  okvis::kinematics::Transformation T_disturb;
  T_disturb.setRandom(1, 0.01);
  okvis::kinematics::Transformation T_WS_init =
      T_WS * T_disturb;                    // world to sensor
  okvis::kinematics::Transformation T_SC;  // sensor to camera
  T_SC.setRandom(0.2, M_PI);

  LOG(INFO) << "set up a random camera geometry... ";
  typedef okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>
      DistortedPinholeCameraGeometry;
  std::shared_ptr<const DistortedPinholeCameraGeometry> cameraGeometry =
      std::static_pointer_cast<const DistortedPinholeCameraGeometry>(
          DistortedPinholeCameraGeometry::createTestObject());
  LOG(INFO) << " [ OK ] ";

  // get some random points and build error terms
  const size_t N = 100;
  LOG(INFO) << "create N=" << N
            << " visible points and add respective reprojection error terms... ";
  Eigen::AlignedVector<Eigen::Vector3d> pointList, rayList;
  pointList.reserve(N);
  rayList.reserve(N);

  double invFx = 1.0 / cameraGeometry->focalLengthU();
  for (size_t i = 1; i < N; ++i) {
    Eigen::Vector4d point = cameraGeometry->createRandomVisibleHomogeneousPoint(
        double(i % 10) * 3 + 2.0);
    Eigen::Vector4d pointW = T_WS * T_SC * point;
    pointList.push_back(pointW.head<3>());
    Eigen::Vector3d pointAtZ1 = point.head<3>() / point[2];
    pointAtZ1.head<2>() += Eigen::Vector2d::Random() * invFx;
    rayList.emplace_back(pointAtZ1);
  }
  LOG(INFO) << " [ OK ] ";

  VIO::StackedProjectionFactorDynamic relativePoseProblem(pointList, rayList,
                                                          T_SC);
  LOG(INFO) << "computing residual and jacobian";
  Eigen::Matrix<double, -1, 1> residuals(pointList.size() * 2, 1);
  Eigen::Matrix<double, -1, 6> jacobian(pointList.size() * 2, 6);
  relativePoseProblem(T_WS_init.coeffs().data(), residuals.data(),
                      jacobian.data());
  LOG(INFO) << "run the tiny solver";
  EXPECT_GT(residuals.squaredNorm() / 2.0, 1e-1);
  Eigen::Matrix<double, 7, 1> estimated_T_WS_coeffs = T_WS_init.coeffs();
  VIO::GtsamPose3Parameterization localParameterization;
  swift_vio::ceres::TinySolver<VIO::StackedProjectionFactorDynamic> solver(
      &localParameterization);
  solver.Solve(relativePoseProblem, &estimated_T_WS_coeffs);
  okvis::kinematics::Transformation estimated_T_WS;
  estimated_T_WS.setCoeffs(estimated_T_WS_coeffs);
  EXPECT_NEAR(0.0, solver.summary.final_cost, 5e-4);

  // print some infos about the optimization
  LOG(INFO) << (int)solver.summary.status << "\n";
  LOG(INFO) << "initial T_WS : " << T_WS_init.T() << "\n"
            << "optimized T_WS : " << estimated_T_WS.T() << "\n"
            << "correct T_WS : " << T_WS.T();

  // make sure it converged
  EXPECT_LT(2 * (T_WS.q() * estimated_T_WS.q().inverse()).vec().norm(), 1e-2)
      << "quaternions not close enough";
  EXPECT_LT((T_WS.r() - estimated_T_WS.r()).norm(), 1e-1)
      << "translation not close enough";
}
