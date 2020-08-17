#include <gtest/gtest.h>

#include <iostream>


#include <msckf/SimulationNView.h>
#include <msckf/triangulate.h>

#include <okvis/triangulation/stereo_triangulation.hpp>

// verify that given points' estimated coordinates are correctly
// flipped in triangulateFast
TEST(TriangulateFast, Flip) {
  {
    simul::SimulationTwoView s2v(0);
    double sigmaR = simul::SimulationNView::raySigma(270);
    bool isValid, isParallel, flipped;
    Eigen::Vector4d v4Xhomog = okvis::triangulation::triangulateFast(
        s2v.T_WC(0).r(),  // center of A in A coordinates
        s2v.obsUnitDirection(0),
        s2v.T_WC(1).r(),  // center of B in A coordinates
        s2v.obsUnitDirection(1), sigmaR, isValid, isParallel, flipped);
    EXPECT_TRUE(isValid);
    EXPECT_FALSE(isParallel);

    Eigen::Matrix<double, Eigen::Dynamic, 1> sv;
    Eigen::Vector4d v4Xhomog2 =
        triangulateHomogeneousDLT(s2v.obsDirections(), s2v.se3_T_CWs(), &sv);

    v4Xhomog2.normalize();
    EXPECT_LT((v4Xhomog - v4Xhomog2).head<3>().norm(), 0.06);
    EXPECT_LT((v4Xhomog/v4Xhomog[3] - s2v.truePoint()).norm(), 5e-4)
        << "Triangulation result with flip on should be the same ";
  }
  {
    simul::SimulationTwoView s2v(1);
    double sigmaR = simul::SimulationNView::raySigma(270);
    bool isValid, isParallel, flipped;
    Eigen::Vector4d v4Xhomog = okvis::triangulation::triangulateFast(
        s2v.T_WC(0).r(),  // center of A in A coordinates
        s2v.obsUnitDirection(0),
        s2v.T_WC(1).r(),  // center of B in A coordinates
        s2v.obsUnitDirection(1), sigmaR, isValid, isParallel, flipped);
    EXPECT_TRUE(isValid);
    EXPECT_FALSE(isParallel);

    Eigen::Matrix<double, Eigen::Dynamic, 1> sv;
    Eigen::Vector4d v4Xhomog2 =
        triangulateHomogeneousDLT(s2v.obsDirections(), s2v.se3_T_CWs(), &sv);
    v4Xhomog2.normalize();

    EXPECT_LT((v4Xhomog - v4Xhomog2).head<3>().norm(), 0.03);
    EXPECT_LT((v4Xhomog/v4Xhomog[3] - s2v.truePoint()).norm(), 5e-5)
        << "Triangulation result with flip on should be the same";
  }
  {
    simul::SimulationTwoView s2v(2);
    double sigmaR = simul::SimulationNView::raySigma(270);
    bool isValid, isParallel, flipped;
    Eigen::Vector4d v4Xhomog = okvis::triangulation::triangulateFast(
        s2v.T_WC(0).r(),  // center of A in A coordinates
        s2v.obsUnitDirection(0),
        s2v.T_WC(1).r(),  // center of B in A coordinates
        s2v.obsUnitDirection(1), sigmaR, isValid, isParallel, flipped);
    EXPECT_TRUE(isValid);
    EXPECT_FALSE(isParallel);
    Eigen::Matrix<double, Eigen::Dynamic, 1> sv;
    Eigen::Vector4d v4Xhomog2 =
        triangulateHomogeneousDLT(s2v.obsDirections(), s2v.se3_T_CWs(), &sv);
    v4Xhomog2.normalize();
    EXPECT_LT((v4Xhomog - v4Xhomog2).norm(), 0.1);
  }
}

// see how okvis::triangulation::triangulateFast and triangulateHomogeneousDLT deal with
// A, an ordinary point, B, identical observations of an ordinary point, C, an
// infinity point
TEST(TriangulateFastVsDlt, RealPoint) {
  simul::SimulationThreeView snv;

  bool isValid;
  bool isParallel;
  bool flipped;
  size_t numObs = snv.numObservations();
  okvis::kinematics::Transformation T_AW(snv.T_CW(0));
  okvis::kinematics::Transformation T_BW(snv.T_CW(numObs - 1));

  double raySigmasA = simul::SimulationNView::raySigma(960);
  double raySigmasB = simul::SimulationNView::raySigma(960);

  double sigmaR = std::max(raySigmasA, raySigmasB);

  // Method 1: triangulate the point w.r.t the world frame using okvis
  Eigen::Vector4d v4Xhomog1 = okvis::triangulation::triangulateFast(
      snv.T_WC(0).r(),  // center of A in W coordinates
      (snv.T_WC(0).C() * snv.obsDirection(0)).normalized(),
      snv.T_WC(numObs - 1).r(),  // center of B in W coordinates
      (snv.T_WC(numObs - 1).C() * snv.obsDirection(numObs - 1)).normalized(),
      sigmaR, isValid, isParallel, flipped);
  v4Xhomog1 /= v4Xhomog1[3];

  // Method 2: triangulate the point w.r.t the A frame using okvis
  Eigen::Vector4d v4Xhomog2 = okvis::triangulation::triangulateFast(
      Eigen::Vector3d(0, 0, 0),  // center of A in A coordinates
      (snv.obsDirection(0)).normalized(),
      (T_AW * T_BW.inverse()).r(),  // center of B in A coordinates
      (T_AW.C() * T_BW.C().transpose() * snv.obsDirection(numObs - 1))
          .normalized(),
      sigmaR, isValid, isParallel, flipped);
  v4Xhomog2 /= v4Xhomog2[3];

  EXPECT_LT((snv.truePoint() - v4Xhomog1).head<3>().norm(), 0.2);
  EXPECT_LT((v4Xhomog1 - T_AW.inverse() * v4Xhomog2).norm(), 1e-7)
      << "results from the two triangulateFast should be the same";

  // Method 3: custom DLT implementation
  std::vector<Eigen::Vector3d,
              Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>
      allObs = snv.obsDirections();
  std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> se3_CWs =
      snv.se3_T_CWs();
  Eigen::Matrix<double, Eigen::Dynamic, 1> sv;
  Eigen::Vector4d v4Xhomog = triangulateHomogeneousDLT(allObs, se3_CWs, &sv);
  EXPECT_LT((snv.truePoint().head<3>() - v4Xhomog.head<3>() / v4Xhomog[3]).norm(), 0.15);

  // triangulate with 3 observations (but two are the same) by SVD DLT
  allObs[2] = allObs[0];
  se3_CWs[2] = se3_CWs[0];
  v4Xhomog = triangulateHomogeneousDLT(allObs, se3_CWs, &sv);
  EXPECT_LT((snv.truePoint().head<3>() - v4Xhomog.head<3>() / v4Xhomog[3]).norm(), 0.15);

  // triangulate with 3 observations (but all are the same) by SVD DLT
  allObs[1] = allObs[0];
  for (size_t i = 0; i < numObs; ++i) {
    se3_CWs[i] = Sophus::SE3d();
  }
  v4Xhomog = triangulateHomogeneousDLT(allObs, se3_CWs, &sv);
  EXPECT_LT((v4Xhomog.head<3>() - snv.obsDirection(0).normalized()).norm(), 1e-5);
  // triangulate with 3 observations (but all are the same) by OKVIS DLT
  v4Xhomog2 = okvis::triangulation::triangulateFast(
      Eigen::Vector3d(0, 0, 0),  // center of A in A coordinates
      (snv.obsDirection(0)).normalized(),
      Eigen::Vector3d(0, 0, 0),  // center of B in A coordinates
      (snv.obsDirection(0)).normalized(), sigmaR, isValid, isParallel, flipped);

  EXPECT_TRUE(isParallel && isValid);
  EXPECT_LT((v4Xhomog2.head<3>() - snv.obsDirection(0).normalized()).norm(), 1e-5);
}

TEST(TriangulateFastVsDlt, FarPoints) {
  // See what happens with increasingly far points
  double distances[] = {3, 3e2, 3e4, 3e8};
  for (size_t jack = 0; jack < sizeof(distances) / sizeof(distances[0]);
       ++jack) {
    double dist = distances[jack];
    simul::SimulationTwoView stv(5, dist);
    bool isValid = false;
    bool isParallel = false;
    bool flipped = false;
    Eigen::Vector4d v4Xhomog2 = okvis::triangulation::triangulateFast(
        stv.T_WC(0).r(),  // center of A in A coordinates
        stv.obsDirection(0).normalized(), stv.T_WC(1).r(),   // center of B in A coordinates
        (stv.T_WC(1).C() * stv.obsDirection(1)).normalized(),
          simul::SimulationNView::raySigma(960), isValid, isParallel, flipped);

    Eigen::Matrix<double, Eigen::Dynamic, 1> sv;
    Eigen::Vector4d v4Xhomog = triangulateHomogeneousDLT(stv.obsDirections(), stv.se3_T_CWs(), &sv);

    EXPECT_LT((v4Xhomog2 - v4Xhomog).head<3>().norm(), 2e-4);
    EXPECT_TRUE(isValid);
    if (jack >= 2) {
      EXPECT_TRUE(isParallel);
    } else {
      EXPECT_FALSE(isParallel);
    }
  }
}

void testRotationOnly(int caseId) {
  // see what happens with pure rotation
  simul::SimulationTwoView stv(caseId);

  bool isValid = false;
  bool isParallel = false;
  bool flipped = false;
  Eigen::Vector4d v4Xhomog2 = okvis::triangulation::triangulateFast(
      stv.T_WC(0).r(),  // center of A in A coordinates
      stv.obsDirection(0).normalized(),
      stv.T_WC(1).r(),   // center of B in A coordinates
      (stv.T_WC(1).C() * stv.obsDirection(1)).normalized(),
      simul::SimulationNView::raySigma(960), isValid, isParallel, flipped);

  Eigen::Matrix<double, Eigen::Dynamic, 1> sv;
  Eigen::Vector4d v4Xhomog = triangulateHomogeneousDLT(stv.obsDirections(), stv.se3_T_CWs(), &sv);
  EXPECT_TRUE(isParallel && isValid);
  EXPECT_LT((v4Xhomog - v4Xhomog2).head<3>().norm(), 1e-6);
}

TEST(TriangulateFastVsDlt, RotationOnly) {
  testRotationOnly(3);
  testRotationOnly(4);
}
