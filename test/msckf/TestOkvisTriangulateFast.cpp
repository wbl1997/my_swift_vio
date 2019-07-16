#include <msckf/triangulate.h>
#include <iostream>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/triangulation/stereo_triangulation.hpp>
#include <sophus/se3.hpp>

OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error);
// copied out from okvis, to check what's going on inside this function
// Triangulate the intersection of two rays.
Eigen::Vector4d triangulateFastDebug(const Eigen::Vector3d& p1,
                                     const Eigen::Vector3d& e1,
                                     const Eigen::Vector3d& p2,
                                     const Eigen::Vector3d& e2, double sigma,
                                     bool& isValid, bool& isParallel) {
  isParallel = false;  // This should be the default.
  // But parallel and invalid is not the same. Points at infinity are valid and
  // parallel.
  isValid = false;  // hopefully this will be reset to true.

  // stolen and adapted from the Kneip toolchain
  // geometric_vision/include/geometric_vision/triangulation/impl/triangulation.hpp
  Eigen::Vector3d t12 = p2 - p1;

  // check parallel
  /*if (t12.dot(e1) - t12.dot(e2) < 1.0e-12) {
  if ((e1.cross(e2)).norm() < 6 * sigma) {
    isValid = true;  // check parallel
    isParallel = true;
    return (Eigen::Vector4d((e1[0] + e2[0]) / 2.0, (e1[1] + e2[1]) / 2.0,
                            (e1[2] + e2[2]) / 2.0, 1e-2).normalized());
  }
}*/

  Eigen::Vector2d b;
  b[0] = t12.dot(e1);
  b[1] = t12.dot(e2);
  Eigen::Matrix2d A;
  A(0, 0) = e1.dot(e1);
  A(1, 0) = e1.dot(e2);
  A(0, 1) = -A(1, 0);
  A(1, 1) = -e2.dot(e2);

  if (A(1, 0) < 0.0) {
    A(1, 0) = -A(1, 0);
    A(0, 1) = -A(0, 1);
    // wrong viewing direction
  };

  bool invertible;
  Eigen::Matrix2d A_inverse;
  A.computeInverseWithCheck(A_inverse, invertible, 1.0e-6);
  Eigen::Vector2d lambda = A_inverse * b;
  if (!invertible) {
    isParallel = true;  // let's note this.
    // parallel. that's fine. but A is not invertible. so handle it separately.
    if ((e1.cross(e2)).norm() < 6 * sigma) {
      isValid = true;  // check parallel
    }
    return (Eigen::Vector4d((e1[0] + e2[0]) / 2.0, (e1[1] + e2[1]) / 2.0,
                            (e1[2] + e2[2]) / 2.0, 1e-3)
                .normalized());
  }

  Eigen::Vector3d xm = lambda[0] * e1 + p1;
  Eigen::Vector3d xn = lambda[1] * e2 + p2;
  Eigen::Vector3d midpoint = (xm + xn) / 2.0;

  // check it
  Eigen::Vector3d error = midpoint - xm;
  Eigen::Vector3d diff = midpoint - (p1 + 0.5 * t12);
  const double diff_sq = diff.dot(diff);
  const double chi2 = error.dot(error) * (1.0 / (diff_sq * sigma * sigma));
  std::cout << "chi2 " << chi2 << std::endl;
  isValid = true;
  if (chi2 > 9) {
    isValid = false;  // reject large chi2-errors
  }

  // flip if necessary
  if (diff.dot(e1) < 0) {
    std::cout << "flipped " << std::endl;
    midpoint = (p1 + 0.5 * t12) - diff;
  }

  return Eigen::Vector4d(midpoint[0], midpoint[1], midpoint[2], 1.0)
      .normalized();
}
// verify that given points' estimated coordinates are indeed unreasonably
// flipped in triangulateFast
void testTriangulateFast0() {
  Eigen::Vector3d obsA(-0.51809, 0.149922, 1);
  Eigen::Vector3d p_AB(-0.00161695, 0.00450899, 0.0435501);
  Eigen::Vector3d obsB(-0.512516, 0.150439, 0.997178);
  double sigmaR = 0.00263914;

  Eigen::Vector4d precomputed;
  precomputed << -0.444625, 0.129972, 0.865497, 0.190606;
  bool isValid, isParallel;
  Eigen::Vector4d v4Xhomog = triangulateFastDebug(
      Eigen::Vector3d(0, 0, 0),  // center of A in A coordinates
      obsA.normalized(), p_AB,   // center of B in A coordinates
      obsB.normalized(), sigmaR, isValid, isParallel);
  OKVIS_ASSERT_LT(Exception, (v4Xhomog - precomputed).norm(), 2e-5,
                  "Triangulation result with flip on should be the same ");

  obsA << -0.130586, 0.0692999, 1;
  p_AB << -0.00398223, 0.0133035, 0.112868;
  obsB << -0.129893, 0.0710077, 0.998044;
  sigmaR = 0.00263914;
  precomputed << -0.124202, 0.0682118, 0.962293, 0.23219;

  v4Xhomog = triangulateFastDebug(
      Eigen::Vector3d(0, 0, 0),  // center of A in A coordinates
      obsA.normalized(), p_AB,   // center of B in A coordinates
      obsB.normalized(), sigmaR, isValid, isParallel);
  OKVIS_ASSERT_LT(Exception, (v4Xhomog - precomputed).norm(), 2e-5,
                  "Triangulation result with flip on should be the same ");

  obsA << 0.0604648, 0.0407913, 1;
  p_AB << 0.0132888, 0.0840325, 0.148991;
  obsB << 0.0681188, 0.0312081, 1.00403;
  sigmaR = 0.00131957;
  v4Xhomog << 0, 0, 0, 0;
  v4Xhomog = triangulateFastDebug(
      Eigen::Vector3d(0, 0, 0),  // center of A in A coordinates
      obsA.normalized(), p_AB,   // center of B in A coordinates
      obsB.normalized(), sigmaR, isValid, isParallel);
  std::cout << "invalid triangulation with result valid? parallel? " << isValid
            << " " << isParallel << " " << v4Xhomog.transpose() << std::endl;
}

// see how okvis::triangulation::triangulateFast and Get_X_from_xP_lin deal with
// A, an ordinary point, B, identical observations of an ordinary point, C, an
// infinity point
void testTriangulateFast() {
  Eigen::Vector3d truePoint(2.31054434, -1.58786347, 9.79390227);
  std::vector<Eigen::Vector3d,
              Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>
      obsDirections(3);

  obsDirections[0] = Eigen::Vector3d(0.1444439, -0.0433997, 1);
  obsDirections[1] = Eigen::Vector3d(0.21640816, -0.34998059, 1);
  obsDirections[2] = Eigen::Vector3d(0.23628522, -0.31005748, 1);
  Eigen::Matrix3d rot;
  rot << 0.99469755, -0.02749299, -0.09910054, 0.02924625, 0.99943961,
      0.0162823, 0.09859735, -0.01909428, 0.9949442;
  Eigen::Vector3d tcinw(0.37937094, -1.06289834, 1.93156378);
  std::vector<okvis::kinematics::Transformation> T_CWs(3);

  T_CWs[0] =
      okvis::kinematics::Transformation(-rot * tcinw, Eigen::Quaterniond(rot));

  rot << 0.99722659, -0.0628095, 0.03992603, 0.0671776, 0.99054536, -0.11961209,
      -0.03203577, 0.1219625, 0.99201757;
  tcinw << 0.78442247, 0.69195074, -0.10575422;

  T_CWs[1] =
      okvis::kinematics::Transformation(-rot * tcinw, Eigen::Quaterniond(rot));

  rot << 0.99958901, 0.01425856, 0.02486967, -0.01057666, 0.98975966,
      -0.14235147, -0.02664472, 0.14202993, 0.98950369;
  tcinw << 0.35451434, -0.09596801, 0.30737151;

  T_CWs[2] =
      okvis::kinematics::Transformation(-rot * tcinw, Eigen::Quaterniond(rot));

  bool isValid;
  bool isParallel;
  okvis::kinematics::Transformation T_AW(T_CWs.front());
  okvis::kinematics::Transformation T_BW(T_CWs.back());
  int kpSize = 9;
  int fx = 960;
  double keypointAStdDev = kpSize;
  keypointAStdDev = 0.8 * keypointAStdDev / 12.0;
  double raySigmasA = sqrt(sqrt(2)) * keypointAStdDev / fx;

  keypointAStdDev = kpSize;
  keypointAStdDev = 0.8 * keypointAStdDev / 12.0;
  double raySigmasB = sqrt(sqrt(2)) * keypointAStdDev / fx;

  double sigmaR = std::max(raySigmasA, raySigmasB);
  /// triangulate the point w.r.t the world frame
  Eigen::Vector4d v4Xhomog1 = okvis::triangulation::triangulateFast(
      T_AW.inverse().r(),  // center of A in W coordinates
      (T_AW.C().transpose() * obsDirections.front()).normalized(),
      T_BW.inverse().r(),  // center of B in W coordinates
      (T_BW.C().transpose() * obsDirections.back()).normalized(), sigmaR,
      isValid, isParallel);
  v4Xhomog1 /= v4Xhomog1[3];
  /// triangulate the point w.r.t the A frame
  Eigen::Vector4d v4Xhomog2 = okvis::triangulation::triangulateFast(
      Eigen::Vector3d(0, 0, 0),  // center of A in A coordinates
      (obsDirections.front()).normalized(),
      (T_AW * T_BW.inverse()).r(),  // center of B in A coordinates
      (T_AW.C() * T_BW.C().transpose() * obsDirections.back()).normalized(),
      sigmaR, isValid, isParallel);
  v4Xhomog2 = T_AW.inverse() * v4Xhomog2;
  v4Xhomog2 /= v4Xhomog2[3];

  std::cout << "True point " << truePoint.transpose() << std::endl;
  std::cout << "2 homog points solutions "
            << " " << v4Xhomog1.transpose() << " " << v4Xhomog2.transpose()
            << std::endl;
  OKVIS_ASSERT_LT(Exception, (v4Xhomog1 - v4Xhomog2).norm(), 1e-7,
                  "results from the two triangulateFast should be the same");

  /// triangulate the point with SVD DLT
  std::vector<Eigen::Vector3d,
              Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>
      allObs = obsDirections;
  std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> se3_CWs(3);
  for (size_t jack = 0; jack < se3_CWs.size(); ++jack) {
    se3_CWs[jack] = Sophus::SE3d(T_CWs[jack].q(), T_CWs[jack].r());
  }
  Eigen::Matrix<double, Eigen::Dynamic, 1> sv;
  Eigen::Vector4d v4Xhomog = Get_X_from_xP_lin(allObs, se3_CWs, &sv);
  std::cout << "singular values " << sv.transpose() << std::endl;
  std::cout << "eigen svd DLT result of a normal point "
            << (v4Xhomog / v4Xhomog[3]).transpose() << std::endl;

  /// triangulate with 3 observations (but two are the same) by SVD DLT
  allObs[2] = allObs[0];
  se3_CWs[2] = se3_CWs[0];
  v4Xhomog = Get_X_from_xP_lin(allObs, se3_CWs, &sv);

  std::cout
      << "eigen svd DLT result after copying the first observation to third "
      << std::endl
      << (v4Xhomog / v4Xhomog[3]).transpose() << std::endl;
  std::cout << "singular values " << sv.transpose() << std::endl;

  /// triangulate with 3 observations (but all are the same) by SVD DLT
  allObs[1] = allObs[0];
  se3_CWs[1] = se3_CWs[0];
  v4Xhomog = Get_X_from_xP_lin(allObs, se3_CWs, &sv);

  std::cout << "eigen svd DLT result after copying the first observation to "
               "second and third "
            << std::endl
            << (v4Xhomog / v4Xhomog[3]).transpose() << std::endl;
  std::cout << "singular values " << sv.transpose() << std::endl;

  /// triangulate with 3 observations (but all are the same) by OKVIS DLT
  v4Xhomog2 = okvis::triangulation::triangulateFast(
      Eigen::Vector3d(0, 0, 0),  // center of A in A coordinates
      (obsDirections.front()).normalized(),
      Eigen::Vector3d(0, 0, 0),  // center of B in A coordinates
      (obsDirections.front()).normalized(), sigmaR, isValid, isParallel);

  std::cout << "DLT with identical observations valid? parallel? " << isValid
            << " " << isParallel << " " << std::endl
            << v4Xhomog2.transpose() << " comaprable with "
            << obsDirections.front().normalized().transpose() << std::endl
            << std::endl;

  /// See what happens with increasingly far points
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

    v4Xhomog2 = okvis::triangulation::triangulateFast(
        Eigen::Vector3d(0, 0, 0),  // center of A in A coordinates
        abrhoi.normalized(), pj,   // center of B in A coordinates
        (Rj * abrhoj).normalized(), sigmaR, isValid, isParallel);

    std::cout << "DLT with observations of point at " << ptini.transpose()
              << " valid? parallel? " << isValid << " " << isParallel
              << std::endl
              << v4Xhomog2.transpose() << std::endl;

    std::vector<Eigen::Vector3d,
                Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>
        allObs(2);
    allObs[0] = abrhoi;
    allObs[1] = abrhoj;
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> se3_CWs(
        2);
    se3_CWs[1] = Sophus::SE3d(Rj, pj).inverse();
    v4Xhomog = Get_X_from_xP_lin(allObs, se3_CWs, &sv);

    std::cout << "eigen svd DLT result " << std::endl
              << (v4Xhomog).transpose() << std::endl;
    std::cout << "singular values " << sv.transpose() << std::endl << std::endl;
  }

  {
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

    Eigen::Vector3d ptinj =
        Rj.transpose() * ptini;  // point direction in j frame

    Eigen::Vector3d abrhoi = ptini / ptini[2];
    Eigen::Vector3d abrhoj = ptinj / ptinj[2];

    v4Xhomog2 = okvis::triangulation::triangulateFast(
        Eigen::Vector3d(0, 0, 0),  // center of A in A coordinates
        abrhoi.normalized(), pj,   // center of B in A coordinates
        (Rj * abrhoj).normalized(), sigmaR, isValid, isParallel);

    std::cout << "DLT with observations of point at infinity valid? parallel? "
              << isValid << " " << isParallel << std::endl
              << v4Xhomog2.transpose() << std::endl;

    std::vector<Eigen::Vector3d,
                Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>
        allObs(2);
    allObs[0] = abrhoi;
    allObs[1] = abrhoj;
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> se3_CWs(
        2);
    se3_CWs[1] = Sophus::SE3d(Rj, pj).inverse();
    v4Xhomog = Get_X_from_xP_lin(allObs, se3_CWs, &sv);

    std::cout << "eigen svd DLT result " << std::endl
              << (v4Xhomog).transpose() << std::endl;
    std::cout << "singular values " << sv.transpose() << std::endl << std::endl;
  }

  {
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

    v4Xhomog2 = okvis::triangulation::triangulateFast(
        Eigen::Vector3d(0, 0, 0),  // center of A in A coordinates
        abrhoi.normalized(), pj,   // center of B in A coordinates
        (Rj * abrhoj).normalized(), sigmaR, isValid, isParallel);

    std::cout
        << "DLT with observations of point with pure rotation valid? parallel? "
        << isValid << " " << isParallel << std::endl
        << v4Xhomog2.transpose() << std::endl;

    std::vector<Eigen::Vector3d,
                Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>
        allObs(2);
    allObs[0] = abrhoi;
    allObs[1] = abrhoj;
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> se3_CWs(
        2);
    se3_CWs[1] = Sophus::SE3d(Rj, pj).inverse();
    v4Xhomog = Get_X_from_xP_lin(allObs, se3_CWs, &sv);

    std::cout << "eigen svd DLT result " << std::endl
              << (v4Xhomog).transpose() << std::endl;
    std::cout << "singular values " << sv.transpose() << std::endl << std::endl;
  }

  {
    // see what happens with static observations
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

    v4Xhomog2 = okvis::triangulation::triangulateFast(
        Eigen::Vector3d(0, 0, 0),  // center of A in A coordinates
        abrhoi.normalized(), pj,   // center of B in A coordinates
        (Rj * abrhoj).normalized(), sigmaR, isValid, isParallel);

    std::cout << "DLT with observations of point with no motion rotation "
                 "valid? parallel? "
              << isValid << " " << isParallel << std::endl
              << v4Xhomog2.transpose() << std::endl;

    std::vector<Eigen::Vector3d,
                Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>
        allObs(2);
    allObs[0] = abrhoi;
    allObs[1] = abrhoj;
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> se3_CWs(
        2);
    se3_CWs[1] = Sophus::SE3d(Rj, pj).inverse();
    v4Xhomog = Get_X_from_xP_lin(allObs, se3_CWs, &sv);

    std::cout << "eigen svd DLT result " << std::endl
              << (v4Xhomog).transpose() << std::endl;
    std::cout << "singular values " << sv.transpose() << std::endl;
  }
}
