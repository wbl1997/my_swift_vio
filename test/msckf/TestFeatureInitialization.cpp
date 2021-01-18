/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#include <iostream>
#include <vector>
#include <map>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <gtest/gtest.h>


#include <msckf/FeatureTriangulation.hpp>
#include <simul/SimulationNView.h>

using namespace std;
using namespace Eigen;


TEST(TriangulateRobustLM, sphereDistribution) {
  simul::SimulationNViewSphere snvs;

  // Initialize a feature object.
  msckf::Feature feature_object(snvs.obsDirectionsZ1(), snvs.camStates());
  // Compute the 3d position of the feature.
  feature_object.initializePosition();

  // Check the difference between the computed 3d
  // feature position and the groud truth.
  Eigen::Vector3d error = feature_object.position - snvs.truePoint().head<3>();
  EXPECT_NEAR(error.norm(), 0, 0.05);
}

void testInitializeWithStationaryCamera(bool addSidewaysView,
                                        bool addObsNoise) {
  simul::SimulationNViewStatic snvs(addSidewaysView, addObsNoise);

  // Initialize a feature object.
  msckf::Feature feature_object(snvs.obsDirectionsZ1(), snvs.camStates());
  // Compute the 3d position of the feature.
  feature_object.initializePosition();

  // Check the difference between the computed 3d
  // feature position and the groud truth.
  if (addSidewaysView) {
    Eigen::Vector3d error =
        feature_object.position - snvs.truePoint().head<3>();
    EXPECT_NEAR(error.norm(), 0, 0.05);
  } else {
    Eigen::Vector3d error =
        feature_object.position / feature_object.position[2] -
        snvs.truePoint().head<3>() / snvs.truePoint()[2];
    EXPECT_NEAR(error.norm(), 0, 0.08);
  }
}

TEST(TriangulateRobustLM, StationaryCamera) {
  testInitializeWithStationaryCamera(false, true);
  testInitializeWithStationaryCamera(true, true);
  testInitializeWithStationaryCamera(false, false);
  testInitializeWithStationaryCamera(true, false);
}

TEST(TriangulateRobustLM, TwoView) {
  {
    simul::SimulationTwoView s2v(0);
    msckf::Feature feature_object(s2v.obsDirectionsZ1(), s2v.camStates());
    feature_object.initializePosition();
    Eigen::Vector4d truePoint = s2v.truePoint();

    Eigen::Vector4d pointEstimate;
    pointEstimate.head<3>() = feature_object.position;
    pointEstimate[3] = 1.0;
    EXPECT_TRUE((feature_object.position / feature_object.position[2]).isApprox(
          truePoint.head<3>() / truePoint[2], 5e-3));
  }
  {
    simul::SimulationThreeView snv;
    int numObs = snv.numObservations();
    std::vector<okvis::kinematics::Transformation,
                Eigen::aligned_allocator<okvis::kinematics::Transformation>>
        cam_states_all = snv.camStates();
    std::vector<okvis::kinematics::Transformation,
                Eigen::aligned_allocator<okvis::kinematics::Transformation>>
        cam_states = {cam_states_all[0], cam_states_all[numObs - 1]};
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
        measurements{snv.obsDirectionZ1(0), snv.obsDirectionZ1(numObs - 1)};
    msckf::Feature feature_object(measurements, cam_states);
    feature_object.initializePosition();
    EXPECT_LT((feature_object.position - snv.truePoint().head<3>()).norm(),
              0.2);
  }
}

TEST(TriangulateRobustLM, Initialization) {
  simul::SimulationThreeView snv;
  const int numObs = snv.numObservations();
  okvis::kinematics::Transformation T_AW(snv.T_CW(0));
  okvis::kinematics::Transformation T_BW(snv.T_CW(numObs - 1));

  msckf::Feature feature_dummy;
  Eigen::Vector3d initial_position(0.0, 0.0, 0.0);

  okvis::kinematics::Transformation T_AB = T_AW * T_BW.inverse();
  okvis::kinematics::Transformation T_C1C0 = T_AB.inverse();
  feature_dummy.generateInitialGuess(T_C1C0, snv.obsDirectionZ1(0),
                                     snv.obsDirectionZ1(numObs - 1),
                                     initial_position);
  EXPECT_LT((initial_position - (T_AW * snv.truePoint()).head<3>()).norm(), 0.2);
}

TEST(TriangulateRobustLM, RotationOnly) {
  {
  simul::SimulationTwoView s2v(3);
  msckf::Feature feature_object(s2v.obsDirectionsZ1(), s2v.camStates());
  feature_object.initializePosition();
  EXPECT_LT((feature_object.position.normalized() - s2v.truePoint().head<3>().normalized()).norm(),
            1e-5);
  }
  {
  simul::SimulationTwoView s2v(4);
  msckf::Feature feature_object(s2v.obsDirectionsZ1(), s2v.camStates());
  feature_object.initializePosition();
  EXPECT_LT((feature_object.position.normalized() - s2v.truePoint().head<3>().normalized()).norm(),
            1e-5);
  }
}

TEST(TriangulateRobustLM, FarPoints) {
  // See what happens with increasingly far points
  double distances[] = {3, 3e2, 3e4, 3e8};
  for (size_t jack = 0; jack < sizeof(distances) / sizeof(distances[0]);
       ++jack) {
    double dist = distances[jack];
    simul::SimulationTwoView stv(5, dist);

    msckf::Feature feature_object(stv.obsDirectionsZ1(), stv.camStates());
    feature_object.initializePosition();
    if (dist < 1000) {
      EXPECT_LT((feature_object.position - stv.truePoint().head<3>()).norm() / dist,
                1e-4);
    } else {
      EXPECT_LT((feature_object.position / feature_object.position[2] -
                stv.truePoint().head<3>() / stv.truePoint()[2]).norm(),
                1e-4);
    }
  }
}
