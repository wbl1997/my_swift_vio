#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include "gtsam/SlidingWindowSmoother.hpp"

#include "msckf/FeatureTriangulation.hpp"
#include "msckf/memory.h"

TEST(gtsam, TriangulateHomogeneousDLT) {
  Eigen::Vector3d point = Eigen::Vector3d(1.5, 3, 25);
  AlignedVector<okvis::kinematics::Transformation> vse3CFromW(3);
  AlignedVector<Eigen::Vector3d> vV3ImPlane(3);
  Eigen::Quaterniond so3M = Eigen::Quaterniond::FromTwoVectors(
      Eigen::Vector3d(0.0, 0.0, 1.0), Eigen::Vector3d(0.3, 0.05, 2.5));
  vse3CFromW[0] =
      okvis::kinematics::Transformation(Eigen::Vector3d(0.0, 0.0, 0.7), so3M);

  Eigen::Quaterniond so3N = Eigen::Quaterniond::FromTwoVectors(
      Eigen::Vector3d(0.0, 0.0, 1.0), Eigen::Vector3d(0, 0.1, 1));
  vse3CFromW[1] =
      okvis::kinematics::Transformation(Eigen::Vector3d(1.0, -0.3, 3.0), so3N);

  Eigen::Quaterniond so3P = Eigen::Quaterniond::FromTwoVectors(
      Eigen::Vector3d(0.0, 0.0, 1.0), Eigen::Vector3d(-0.5, 0.3, 5));
  vse3CFromW[2] =
      okvis::kinematics::Transformation(Eigen::Vector3d(1.0, -0.2, -5.0), so3P);

  for (int i = 0; i < 3; ++i) {
    Eigen::Vector3d v3Cam = vse3CFromW[i].q() * point + vse3CFromW[i].r();
    if (v3Cam[2] < 0.001) {
      LOG(WARNING) << "Point " << i << " is behind the image";
    }
    vV3ImPlane[i] = v3Cam / v3Cam[2];
  }

  Eigen::Vector4d hpW =
      msckf::triangulateHomogeneousDLT(vV3ImPlane, vse3CFromW);
  Eigen::Vector3d pW = hpW.head<3>() / hpW[3];
  double rms = (pW - point).norm();
  EXPECT_LT(rms, 1e-3);

  // Show that normalized image coordinates do not work.
  AlignedVector<Eigen::Vector3d> normalRays = vV3ImPlane;
  for (size_t j = 0u; j < normalRays.size(); ++j) {
    normalRays.at(j).normalize();
  }
  Eigen::Vector4d hpW1 =
      msckf::triangulateHomogeneousDLT(normalRays, vse3CFromW);
  Eigen::Vector3d pW1 = hpW1.head<3>() / hpW1[3];
  double diff = (pW1 - pW).norm();
  EXPECT_GT(diff, 0.2);
}
