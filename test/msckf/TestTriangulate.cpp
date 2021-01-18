#include <iostream>
#include <ceres/ceres.h>
#include <gtest/gtest.h>

#include <msckf/triangulate.h>
#include <simul/SimulationNView.h>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

// test AutoDiffCostFunction and AddResidualBlock dimensions with Powell's
// minimization problem compare with examples/powell.cc in ceres
struct Powell {
  template <typename T>
  bool operator()(const T* const x, T* residual) const {
    // f1 = x1 + 10 * x2;
    residual[0] = x[0] + T(10.0) * x[1];

    // f2 = sqrt(5) (x3 - x4)
    residual[1] = T(sqrt(5.0)) * (x[2] - x[3]);

    // f3 = (x2 - 2 x3)^2
    residual[2] = (x[1] - T(2.0) * x[2]) * (x[1] - T(2.0) * x[2]);

    // f4 = sqrt(10) (x1 - x4)^2
    residual[3] = T(sqrt(10.0)) * (x[0] - x[3]) * (x[0] - x[3]);
    return true;
  }
};

// this test verifies that CostFunction operator() should only take variables
// and residuals as arguments, excluding fixed parameters and observations.
TEST(CeresOptimize, Powell) {
  double x[4] = {3.0, -1.0, 0.0, 1.0};

  Problem problem;
  // Add residual terms to the problem using the using the autodiff
  // wrapper to get the derivatives automatically. The parameters, x1 through
  // x4, are modified in place.
  problem.AddResidualBlock(
        new AutoDiffCostFunction<Powell, 4, 4>(new Powell), NULL, x);

  Solver::Options options;

  options.minimizer_type = ceres::TRUST_REGION;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;

  Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  ASSERT_NEAR(x[0], 0, 3e-4);
  ASSERT_NEAR(x[1], 0, 3e-4);
  ASSERT_NEAR(x[2], 0, 3e-4);
  ASSERT_NEAR(x[3], 0, 3e-4);
}

// to refine triangulated results, use LM or Dogleg solver or GN as provided by
// Ceres solver this test verified that (1) the result using autodiff and
// analytical diff is identical for both LM and Dogleg; (2) Gauss-Newton method
// is very competitive for structure only point optimization compared to LM and
// Dogleg, given good initial estimate such as obtained by two view mid point.
TEST(Triangulate, AllMethods) {
  const int num_methods = 8;
  int trials = 1e3;
  std::vector<double> deviation[num_methods];
  for (int jack = 0; jack < num_methods; ++jack)
    deviation[jack] = std::vector<double>(trials, 0);
  for (int i = 0; i < trials; ++i) {
    simul::SimulationThreeView snv;
    bool isValid = snv.randomThreeView(100);
    if (!isValid)
      continue;

    std::vector<Eigen::Vector2d,
                Eigen::aligned_allocator<Eigen::Matrix<double, 2, 1>>>
        vV2ImPlane = snv.obsDirectionsZ1();
    std::vector<Eigen::Vector3d,
                Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>
        obs = snv.obsDirections();
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>
        vse3CFromW =
        snv.se3_T_CWs();
    std::vector<Eigen::Vector3d,
           Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>
        res(num_methods);
    Eigen::Vector4d v4Xhomog = triangulateHomogeneousDLT(obs, vse3CFromW);
    if (fabs(v4Xhomog[3]) < 1e-9)
      res[0] = Eigen::Vector3d(0, 0, -1000);
    else
      res[0] = v4Xhomog.head<3>() / v4Xhomog[3];
    res[1] = triangulate2View_midpoint(vV2ImPlane, vse3CFromW);
    res[2] = triangulate_midpoint(vV2ImPlane, vse3CFromW);

    for (int zinc = 3; zinc < num_methods; ++zinc) res[zinc] = res[1];

    triangulate_refine_GN(obs, vse3CFromW, res[3], 5);

    // The following methods use ceres solver.
    triangulate_refine(obs, vse3CFromW, res[4], 5);
    triangulate_refine(obs, vse3CFromW, res[5], 5, 1);
    triangulate_refineJ(obs, vse3CFromW, res[6], 5);
    triangulate_refineJ(obs, vse3CFromW, res[7], 5, 1);

    for (int zinc = 0; zinc < num_methods; ++zinc) {
      Eigen::Vector3d point = snv.truePoint().head<3>();
      deviation[zinc][i] =
          (res[zinc] - point).transpose() * (res[zinc] - point);
    }
  }
  std::cout << "SquaredResidual of DLT, 2 view MP, multiview MP, GN refined "
               "2view MP, LM AD, Dogleg AD, LM anal, Dogleg anal"
            << std::endl;
  std::cout << "residual medians:";
  for (int zinc = 0; zinc < num_methods; ++zinc) {
    double medianDev = CalcMHWScore(deviation[zinc]);
    std::cout << medianDev << " ";
    ASSERT_LT(medianDev, 0.5);
  }
  std::cout << std::endl;
}
