#include <iostream>
#include "gtest/gtest.h"
#include "msckf/triangulate.h"

#define HAVE_CERES
#ifdef HAVE_CERES
#include "ceres/ceres.h"  //LM and Dogleg optimization
#include "ceres/rotation.h"
#include "glog/logging.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
#endif

// test AutoDiffCostFunction and AddResidualBlock dimensions with Powell's
// minimization problem compare with examples/powell.cc in ceres

struct Fall {
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

// DEFINE_string(minimizer, "trust_region",
//              "Minimizer type to use, choices are: line_search &
//              trust_region");

TEST(Triangulate, Powell) {
  double x[4] = {3.0, -1.0, 0.0, 1.0};

  Problem problem;
  // Add residual terms to the problem using the using the autodiff
  // wrapper to get the derivatives automatically. The parameters, x1 through
  // x4, are modified in place.
  problem.AddResidualBlock(new AutoDiffCostFunction<Fall, 4, 4>(new Fall), NULL,
                           x);

  Solver::Options options;
  //  LOG_IF(FATAL, !ceres::StringToMinimizerType(FLAGS_minimizer,
  //                                              &options.minimizer_type))
  //      << "Invalid minimizer: " << FLAGS_minimizer
  //      << ", valid options are: trust_region and line_search.";

  options.minimizer_type = ceres::TRUST_REGION;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;

  //  std::cout << "Initial x1 = " << x[0]
  //            << ", x2 = " << x[1]
  //            << ", x3 = " << x[2]
  //            << ", x4 = " << x[3]
  //            << "\n";

  // Run the solver!
  Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  //  std::cout << summary.FullReport() << "\n";
  //  std::cout << "Final x1 = " << x[0]
  //            << ", x2 = " << x[1]
  //            << ", x3 = " << x[2]
  //            << ", x4 = " << x[3]
  //            << "\n";
  ASSERT_NEAR(x[0], 0, 3e-4);
  ASSERT_NEAR(x[1], 0, 3e-4);
  ASSERT_NEAR(x[2], 0, 3e-4);
  ASSERT_NEAR(x[3], 0, 3e-4);
}

TEST(Triangulate, AllMethods) {
#ifdef HAVE_CERES
  const int num_methods = 8;
#else
  const int num_methods = 4;
#endif

  std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> vse3CFromW(
      3);
  std::vector<Eigen::Vector2d,
              Eigen::aligned_allocator<Eigen::Matrix<double, 2, 1>>>
      vV2ImPlane(3);
  int trials = 1e3;

  vector<double> deviation[num_methods];
  for (int jack = 0; jack < num_methods; ++jack)
    deviation[jack] = vector<double>(trials, 0);
  for (int i = 0; i < trials; ++i) {
#if 1
    int N = 100;
    Eigen::Vector3d point(1.5 * (rand() % N) / N, 3 * (rand() % N) / N,
                          25 * (rand() % N) / N);

    Eigen::Quaterniond so3M(0.8, 0.3 * (rand() % N) / N, 0.1, 0.2);
    so3M.normalize();
    vse3CFromW[0] = Sophus::SE3d(so3M, Eigen::Vector3d(0.0, 0.0, 0.7));
    Eigen::Quaterniond so3N(5.8, 0.1, 1.0 * (rand() % N) / N, 0.2);
    so3N.normalize();
    vse3CFromW[1] =
        Sophus::SE3d(so3N, Eigen::Vector3d(1.0, -0.3 * (rand() % N) / N, 3.0));
    Eigen::Quaterniond so3P(4.5, 0.8, 1.0, 1.4 * (rand() % N) / N);
    so3P.normalize();
    vse3CFromW[2] =
        Sophus::SE3d(so3P, Eigen::Vector3d(1.0, -0.2 * (rand() % N) / N, -5.0));
    for (int i = 0; i < 3; ++i) {
      Eigen::Vector3d v3Cam = vse3CFromW[i] * point;
      if (v3Cam[2] < 0) {
        //                std::cerr<<v3Cam[2]<<std::endl;
        //                std::cerr<<"Point "<<i<<" is behind the
        //                image"<<std::endl;
        continue;
      }
      vV2ImPlane[i] =
          v3Cam.head<2>() / v3Cam[2] + Vector2d(1, 1) * (rand() % 10) / 1000;
    }
#else
    Vector3d true_point(2.31054434, -1.58786347, 9.79390227);
    vV2ImPlane[0] = Vector2d(0.1444439, -0.0433997);
    vV2ImPlane[1] = Vector2d(0.21640816, -0.34998059);
    vV2ImPlane[2] = Vector2d(0.23628522, -0.31005748);
    Matrix3d rot;
    rot << 0.99469755, -0.02749299, -0.09910054, 0.02924625, 0.99943961,
        0.0162823, 0.09859735, -0.01909428, 0.9949442;
    Vector3d tcinw(0.37937094, -1.06289834, 1.93156378);
    vse3CFromW[0] = SE3d(rot, -rot * tcinw);

    rot << 0.99722659, -0.0628095, 0.03992603, 0.0671776, 0.99054536,
        -0.11961209, -0.03203577, 0.1219625, 0.99201757;
    tcinw << 0.78442247, 0.69195074, -0.10575422;
    vse3CFromW[1] = SE3d(rot, -rot * tcinw);

    rot << 0.99958901, 0.01425856, 0.02486967, -0.01057666, 0.98975966,
        -0.14235147, -0.02664472, 0.14202993, 0.98950369;
    tcinw << 0.35451434, -0.09596801, 0.30737151;
    vse3CFromW[2] = SE3d(rot, -rot * tcinw);
#endif

    vector<Eigen::Vector3d,
           Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>
        obs(3);
    for (int zinc = 0; zinc < 3; ++zinc)
      obs[zinc] = unproject2d(vV2ImPlane[zinc]);

    vector<Eigen::Vector3d,
           Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>
        res(num_methods);
    Eigen::Vector4d v4Xhomog = Get_X_from_xP_lin(obs, vse3CFromW);
    if (fabs(v4Xhomog[3]) < 1e-9)
      res[0] = Vector3d(0, 0, -1000);
    else
      res[0] = v4Xhomog.head<3>() / v4Xhomog[3];
    res[1] = triangulate2View_midpoint(vV2ImPlane, vse3CFromW);
    res[2] = triangulate_midpoint(vV2ImPlane, vse3CFromW);

    for (int zinc = 3; zinc < num_methods; ++zinc) res[zinc] = res[1];

    triangulate_refine_GN(obs, vse3CFromW, res[3], 5);
#ifdef HAVE_CERES
    triangulate_refine(obs, vse3CFromW, res[4], 5);
    triangulate_refine(obs, vse3CFromW, res[5], 5, 1);
    triangulate_refineJ(obs, vse3CFromW, res[6], 5);
    triangulate_refineJ(obs, vse3CFromW, res[7], 5, 1);
#endif
    // cout<<"estimated point by 3 methods"<<endl<<
    // res<<endl<<res2<<endl<<res3<<endl;
    for (int zinc = 0; zinc < num_methods; ++zinc) {
      deviation[zinc][i] =
          (res[zinc] - point).transpose() * (res[zinc] - point);
      //            cout<<deviation[zinc][i]<<" ";
    }
    //        cout<<std::endl;
  }
  std::cout << "SquaredResidual of DLT, 2 view MP, multiview MP, GN refined "
               "2view MP, LM AD, Dogleg AD, LM anal, Dogleg anal"
            << endl;
  cout << "residual medians:";
  for (int zinc = 0; zinc < num_methods; ++zinc) {
    double medianDev = CalcMHWScore(deviation[zinc]);
    cout << medianDev << " ";
    ASSERT_LT(medianDev, 0.5);
  }
  cout << endl;
}
