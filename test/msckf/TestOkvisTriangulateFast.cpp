#include <gtest/gtest.h>

#include <iostream>


#include <msckf/SimulationNView.h>
#include <msckf/triangulate.h>

#include <okvis/triangulation/stereo_triangulation.hpp>

#include <msckf/SimulationNView.h>

// verify that given points' estimated coordinates are correctly
// flipped in triangulateFast
TEST(TriangulateFast, Flip) {
  {
    simul::SimulationTwoView s2v(0);
    double sigmaR = simul::SimulationNView::raySigma(270);
    bool isValid, isParallel;
    Eigen::Vector4d v4Xhomog = okvis::triangulation::triangulateFast(
        s2v.T_WC(0).r(),  // center of A in A coordinates
        s2v.obsUnitDirection(0),
        s2v.T_WC(1).r(),  // center of B in A coordinates
        s2v.obsUnitDirection(1), sigmaR, isValid, isParallel);
    EXPECT_TRUE(isValid);
    EXPECT_FALSE(isParallel);

    Eigen::Matrix<double, Eigen::Dynamic, 1> sv;
    Eigen::Vector4d v4Xhomog2 =
        Get_X_from_xP_lin(s2v.obsDirections(), s2v.se3_T_CWs(), &sv);

    v4Xhomog2.normalize();
    EXPECT_LT((v4Xhomog - v4Xhomog2).head<3>().norm(), 0.06);
    EXPECT_LT((v4Xhomog - s2v.truePoint()).norm(), 2e-5)
        << "Triangulation result with flip on should be the same ";

    msckf_vio::Feature feature_object(s2v.obsDirectionsZ1(), s2v.camStates());
    feature_object.initializePosition();
    EXPECT_LT((feature_object.position - v4Xhomog2.head<3>() / v4Xhomog2[3]).norm(), 0.02);
  }
  {
    simul::SimulationTwoView s2v(1);
    double sigmaR = simul::SimulationNView::raySigma(270);
    bool isValid, isParallel;
    Eigen::Vector4d v4Xhomog = okvis::triangulation::triangulateFast(
        s2v.T_WC(0).r(),  // center of A in A coordinates
        s2v.obsUnitDirection(0),
        s2v.T_WC(1).r(),  // center of B in A coordinates
        s2v.obsUnitDirection(1), sigmaR, isValid, isParallel);
    EXPECT_TRUE(isValid);
    EXPECT_FALSE(isParallel);

    Eigen::Matrix<double, Eigen::Dynamic, 1> sv;
    Eigen::Vector4d v4Xhomog2 =
        Get_X_from_xP_lin(s2v.obsDirections(), s2v.se3_T_CWs(), &sv);
    v4Xhomog2.normalize();

    EXPECT_LT((v4Xhomog - v4Xhomog2).head<3>().norm(), 0.03);
    EXPECT_LT((v4Xhomog - s2v.truePoint()).norm(), 2e-5)
        << "Triangulation result with flip on should be the same";
  }
  {
    simul::SimulationTwoView s2v(2);
    double sigmaR = simul::SimulationNView::raySigma(270);
    bool isValid, isParallel;
    Eigen::Vector4d v4Xhomog = okvis::triangulation::triangulateFast(
        s2v.T_WC(0).r(),  // center of A in A coordinates
        s2v.obsUnitDirection(0),
        s2v.T_WC(1).r(),  // center of B in A coordinates
        s2v.obsUnitDirection(1), sigmaR, isValid, isParallel);
    EXPECT_TRUE(isValid);
    EXPECT_FALSE(isParallel);
    Eigen::Matrix<double, Eigen::Dynamic, 1> sv;
    Eigen::Vector4d v4Xhomog2 =
        Get_X_from_xP_lin(s2v.obsDirections(), s2v.se3_T_CWs(), &sv);
    v4Xhomog2.normalize();
    EXPECT_LT((v4Xhomog - v4Xhomog2).norm(), 0.1);
  }
}

// see how okvis::triangulation::triangulateFast and Get_X_from_xP_lin deal with
// A, an ordinary point, B, identical observations of an ordinary point, C, an
// infinity point
TEST(TriangulateFastVsDlt, RealPoint) {
  simul::SimulationNView snv;

  bool isValid;
  bool isParallel;
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
      sigmaR, isValid, isParallel);
  v4Xhomog1 /= v4Xhomog1[3];

  // Method 2: triangulate the point w.r.t the A frame using okvis
  Eigen::Vector4d v4Xhomog2 = okvis::triangulation::triangulateFast(
      Eigen::Vector3d(0, 0, 0),  // center of A in A coordinates
      (snv.obsDirection(0)).normalized(),
      (T_AW * T_BW.inverse()).r(),  // center of B in A coordinates
      (T_AW.C() * T_BW.C().transpose() * snv.obsDirection(numObs - 1))
          .normalized(),
      sigmaR, isValid, isParallel);
  v4Xhomog2 /= v4Xhomog2[3];

  EXPECT_LT((snv.truePoint() - v4Xhomog1).head<3>().norm(), 0.2);
  EXPECT_LT((v4Xhomog1 - T_AW.inverse() * v4Xhomog2).norm(), 1e-7)
      << "results from the two triangulateFast should be the same";

  // Method 3: stereo initialization with msckf_vio
  msckf_vio::Feature feature_dummy;
  Eigen::Vector3d initial_position(0.0, 0.0, 0.0);
  Eigen::Isometry3d T_C1C0;
  okvis::kinematics::Transformation T_AB = T_AW * T_BW.inverse();
  T_C1C0.setIdentity();
  T_C1C0.linear() = T_AB.C().transpose();
  T_C1C0.translation() = - T_AB.C().transpose() * T_AB.r();
  feature_dummy.generateInitialGuess(T_C1C0, snv.obsDirectionZ1(0),
                                     snv.obsDirectionZ1(numObs - 1),
                                     initial_position);
  EXPECT_LT((initial_position - v4Xhomog2.head<3>()).norm(), 3e-2);

  // Method 4: custom DLT implementation
  std::vector<Eigen::Vector3d,
              Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>
      allObs = snv.obsDirections();
  std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> se3_CWs =
      snv.se3_T_CWs();
  Eigen::Matrix<double, Eigen::Dynamic, 1> sv;
  Eigen::Vector4d v4Xhomog = Get_X_from_xP_lin(allObs, se3_CWs, &sv);
  EXPECT_LT((snv.truePoint().head<3>() - v4Xhomog.head<3>() / v4Xhomog[3]).norm(), 0.15);

  // Method 5: LM method in msckf_vio
  msckf_vio::CamStateServer cam_states_all = snv.camStates();
  msckf_vio::CamStateServer cam_states = {cam_states_all[0], cam_states_all[numObs - 1]};
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      measurements{snv.obsDirectionZ1(0), snv.obsDirectionZ1(numObs - 1)};
  msckf_vio::Feature feature_object(measurements, cam_states);
  feature_object.initializePosition();
  EXPECT_LT((feature_object.position - snv.truePoint().head<3>()).norm(), 0.2);

  // triangulate with 3 observations (but two are the same) by SVD DLT
  allObs[2] = allObs[0];
  se3_CWs[2] = se3_CWs[0];
  v4Xhomog = Get_X_from_xP_lin(allObs, se3_CWs, &sv);
  EXPECT_LT((snv.truePoint().head<3>() - v4Xhomog.head<3>() / v4Xhomog[3]).norm(), 0.15);

  // triangulate with 3 observations (but all are the same) by SVD DLT
  allObs[1] = allObs[0];
  for (size_t i = 0; i < numObs; ++i) {
    se3_CWs[i] = Sophus::SE3d();
  }
  v4Xhomog = Get_X_from_xP_lin(allObs, se3_CWs, &sv);
  EXPECT_LT((v4Xhomog.head<3>() - snv.obsDirection(0).normalized()).norm(), 1e-5);
  // triangulate with 3 observations (but all are the same) by OKVIS DLT
  v4Xhomog2 = okvis::triangulation::triangulateFast(
      Eigen::Vector3d(0, 0, 0),  // center of A in A coordinates
      (snv.obsDirection(0)).normalized(),
      Eigen::Vector3d(0, 0, 0),  // center of B in A coordinates
      (snv.obsDirection(0)).normalized(), sigmaR, isValid, isParallel);

  EXPECT_TRUE(isParallel && isValid);
  EXPECT_LT((v4Xhomog2.head<3>() - snv.obsDirection(0).normalized()).norm(), 1e-5);
}

TEST(TriangulateFastVsDlt, FarPoints) {
  // See what happens with increasingly far points
  double distances[] = {3, 3e2, 3e4, 3e8};
  for (size_t jack = 0; jack < sizeof(distances) / sizeof(distances[0]);
       ++jack) {
    double dist = distances[jack];

    Eigen::Matrix3d Ri = Eigen::Matrix3d::Identity();  // i to global
    Eigen::Vector3d ptini;                             // point in i frame
    ptini << dist * cos(15 * M_PI / 180) * cos(45 * M_PI / 180),
        -dist * sin(15 * M_PI / 180),
        dist * cos(15 * M_PI / 180) * sin(45 * M_PI / 180);
    Eigen::Matrix3d Rj =
        Eigen::AngleAxisd(30 * M_PI / 180, Eigen::Vector3d::UnitY())
            .toRotationMatrix();  // j to global

    Eigen::Vector3d pi = Eigen::Vector3d::Zero();    // i in global
    Eigen::Vector3d pj = Eigen::Vector3d::Random();  // j in global

    Eigen::Vector3d ptinj = Rj.transpose() * (ptini - pj);

    Eigen::Vector3d abrhoi = ptini / ptini[2];
    Eigen::Vector3d abrhoj = ptinj / ptinj[2];
    bool isValid = false;
    bool isParallel = false;
    Eigen::Vector4d v4Xhomog2 = okvis::triangulation::triangulateFast(
        Eigen::Vector3d(0, 0, 0),  // center of A in A coordinates
        abrhoi.normalized(), pj,   // center of B in A coordinates
        (Rj * abrhoj).normalized(), simul::SimulationNView::raySigma(960), isValid, isParallel);

    std::vector<Eigen::Vector3d,
                Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>
        allObs(2);
    allObs[0] = abrhoi;
    allObs[1] = abrhoj;
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> se3_CWs(
        2);
    se3_CWs[1] = Sophus::SE3d(Rj, pj).inverse();
    Eigen::Matrix<double, Eigen::Dynamic, 1> sv;
    Eigen::Vector4d v4Xhomog = Get_X_from_xP_lin(allObs, se3_CWs, &sv);

    EXPECT_LT((v4Xhomog2 - v4Xhomog).head<3>().norm(), 1e-5);
    EXPECT_TRUE(isValid);
    if (jack > 1) {
      EXPECT_TRUE(isParallel);
    } else {
      EXPECT_FALSE(isParallel);
    }
  }
}

TEST(TriangulateFastVsDlt, PointsAtInfinity) {
  // see what happens with infinity points
  Eigen::Matrix3d Ri = Eigen::Matrix3d::Identity();  // i to global
  Eigen::Vector3d pi = Eigen::Vector3d::Zero();      // i in global
  // point direction in i frame
  Eigen::Vector3d ptini(cos(15 * M_PI / 180) * cos(45 * M_PI / 180),
                        -sin(15 * M_PI / 180),
                        cos(15 * M_PI / 180) * sin(45 * M_PI / 180));

  Eigen::Matrix3d Rj =
      Eigen::AngleAxisd(30 * M_PI / 180, Eigen::Vector3d::UnitY())
          .toRotationMatrix();                     // j to global
  Eigen::Vector3d pj = Eigen::Vector3d::Random();  // j in global

  Eigen::Vector3d ptinj = Rj.transpose() * ptini;  // point direction in j frame

  Eigen::Vector3d abrhoi = ptini / ptini[2];
  Eigen::Vector3d abrhoj = ptinj / ptinj[2];
  bool isValid = false;
  bool isParallel = false;
  Eigen::Vector4d v4Xhomog2 = okvis::triangulation::triangulateFast(
      Eigen::Vector3d(0, 0, 0),  // center of A in A coordinates
      abrhoi.normalized(), pj,   // center of B in A coordinates
      (Rj * abrhoj).normalized(), simul::SimulationNView::raySigma(960), isValid, isParallel);

  std::vector<Eigen::Vector3d,
              Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>
      allObs(2);
  allObs[0] = abrhoi;
  allObs[1] = abrhoj;
  std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> se3_CWs(2);
  se3_CWs[1] = Sophus::SE3d(Rj, pj).inverse();
  Eigen::Matrix<double, Eigen::Dynamic, 1> sv;
  Eigen::Vector4d v4Xhomog = Get_X_from_xP_lin(allObs, se3_CWs, &sv);

  EXPECT_TRUE(isParallel && isValid);
  EXPECT_LT((v4Xhomog - v4Xhomog2).head<3>().norm(), 1e-6);
}

TEST(TriangulateFastVsDlt, RotationOnly) {
  // see what happens with pure rotation
  Eigen::Matrix3d Ri = Eigen::Matrix3d::Identity();  // i to global
  Eigen::Vector3d pi = Eigen::Vector3d::Zero();      // i in global
  // point in i frame
  Eigen::Vector3d ptini(cos(15 * M_PI / 180) * cos(45 * M_PI / 180),
                        -sin(15 * M_PI / 180),
                        cos(15 * M_PI / 180) * sin(45 * M_PI / 180));

  Eigen::Matrix3d Rj =
      Eigen::AngleAxisd(30 * M_PI / 180, Eigen::Vector3d::UnitY())
          .toRotationMatrix();                   // j to global
  Eigen::Vector3d pj = Eigen::Vector3d::Zero();  // j in global

  Eigen::Vector3d ptinj = Rj.transpose() * (ptini - pj);  // point in j frame

  Eigen::Vector3d abrhoi = ptini / ptini[2];
  Eigen::Vector3d abrhoj = ptinj / ptinj[2];
  bool isValid = false;
  bool isParallel = false;
  Eigen::Vector4d v4Xhomog2 = okvis::triangulation::triangulateFast(
      Eigen::Vector3d(0, 0, 0),  // center of A in A coordinates
      abrhoi.normalized(), pj,   // center of B in A coordinates
      (Rj * abrhoj).normalized(), simul::SimulationNView::raySigma(960), isValid, isParallel);

  std::vector<Eigen::Vector3d,
              Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>
      allObs(2);
  allObs[0] = abrhoi;
  allObs[1] = abrhoj;
  std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> se3_CWs(2);
  se3_CWs[1] = Sophus::SE3d(Rj, pj).inverse();
  Eigen::Matrix<double, Eigen::Dynamic, 1> sv;
  Eigen::Vector4d v4Xhomog = Get_X_from_xP_lin(allObs, se3_CWs, &sv);

  EXPECT_TRUE(isParallel && isValid);
  EXPECT_LT((v4Xhomog - v4Xhomog2).head<3>().norm(), 1e-6);
}

TEST(TriangulateFastVsDlt, RotationOnly2) {
  // see what happens with fake observations computed by pure rotation
  Eigen::Matrix3d Ri = Eigen::Matrix3d::Identity();  // i to global
  Eigen::Vector3d pi = Eigen::Vector3d::Zero();      // i in global
  // point in i frame
  Eigen::Vector3d ptini(cos(15 * M_PI / 180) * cos(45 * M_PI / 180),
                        -sin(15 * M_PI / 180),
                        cos(15 * M_PI / 180) * sin(45 * M_PI / 180));

  Eigen::Matrix3d Rj = Eigen::Matrix3d::Identity();  // j to global
  Eigen::Vector3d pj = Eigen::Vector3d::Zero();      // j in global

  Eigen::Vector3d ptinj = Rj.transpose() * (ptini - pj);  // point in j frame

  Eigen::Vector3d abrhoi = ptini / ptini[2];
  Eigen::Vector3d abrhoj = ptinj / ptinj[2];
  bool isValid = false;
  bool isParallel = false;
  Eigen::Vector4d v4Xhomog2 = okvis::triangulation::triangulateFast(
      Eigen::Vector3d(0, 0, 0),  // center of A in A coordinates
      abrhoi.normalized(), pj,   // center of B in A coordinates
      (Rj * abrhoj).normalized(), simul::SimulationNView::raySigma(960), isValid, isParallel);

  std::vector<Eigen::Vector3d,
              Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>
      allObs(2);
  allObs[0] = abrhoi;
  allObs[1] = abrhoj;
  std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> se3_CWs(2);
  se3_CWs[1] = Sophus::SE3d(Rj, pj).inverse();
  Eigen::Matrix<double, Eigen::Dynamic, 1> sv;
  Eigen::Vector4d v4Xhomog = Get_X_from_xP_lin(allObs, se3_CWs, &sv);

  EXPECT_TRUE(isParallel && isValid);
  EXPECT_LT((v4Xhomog - v4Xhomog2).head<3>().norm(), 1e-6);
}
