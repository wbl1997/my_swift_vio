// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)
//
// An example program that minimizes Powell's singular function.
//
//   F = 1/2 (f1^2 + f2^2 + f3^2 + f4^2)
//
//   f1 = x1 + 10*x2;
//   f2 = sqrt(5) * (x3 - x4)
//   f3 = (x2 - 2*x3)^2
//   f4 = sqrt(10) * (x1 - x4)^2
//
// The starting values are x1 = 3, x2 = -1, x3 = 0, x4 = 1.
// The minimum is 0 at (x1, x2, x3, x4) = 0.
//
// From: Testing Unconstrained Optimization Software by Jorge J. More, Burton S.
// Garbow and Kenneth E. Hillstrom in ACM Transactions on Mathematical Software,
// Vol 7(1), March 1981.

#include <vector>
#include <Eigen/Core>
#include "ceres/ceres.h"
#include "gtest/gtest.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

struct F1 {
  template <typename T> bool operator()(const T* const x1,
                                        const T* const x2,
                                        T* residual) const {
    // f1 = x1 + 10 * x2;
    residual[0] = x1[0] + T(10.0) * x2[0];
    return true;
  }
};

struct F2 {
  template <typename T> bool operator()(const T* const x3,
                                        const T* const x4,
                                        T* residual) const {
    // f2 = sqrt(5) (x3 - x4)
    residual[0] = T(sqrt(5.0)) * (x3[0] - x4[0]);
    return true;
  }
};

struct F3 {
  template <typename T> bool operator()(const T* const x2,
                                        const T* const x4,
                                        T* residual) const {
    // f3 = (x2 - 2 x3)^2
    residual[0] = (x2[0] - T(2.0) * x4[0]) * (x2[0] - T(2.0) * x4[0]);
    return true;
  }
};

struct F4 {
  template <typename T> bool operator()(const T* const x1,
                                        const T* const x4,
                                        T* residual) const {
    // f4 = sqrt(10) (x1 - x4)^2
    residual[0] = T(sqrt(10.0)) * (x1[0] - x4[0]) * (x1[0] - x4[0]);
    return true;
  }
};

struct Prior {
  Prior(const double p, const double _std = 1.0):prior(p), sqrt_info(1.0/_std){}

  template <typename T> bool operator()(const T* const x,
                                        T* residual) const {
    // f = (x - meas)^2
    residual[0] = (T)sqrt_info*(x[0] - (T)prior);
    return true;
  }
  double jacobian() const {
    return sqrt_info;
  }
  
  const double prior;
  const double sqrt_info;
};

struct RelativePose {
  RelativePose(double _meas, const double _std = 1.0):meas(_meas), sqrt_info(1.0/_std){}
  template <typename T> bool operator()(const T* const x1,
                                        const T* const x2,
                                        T* residual) const {
    residual[0] = (T)sqrt_info*((x2[0] - x1[0]) - (T) meas);
    return true;
  }
  Eigen::Vector2d jacobian() const {
    return Eigen::Vector2d(-sqrt_info, sqrt_info);
  }
  
  const double meas;
  const double sqrt_info;
};

TEST(CovarianceEstimation, Powell){
    double x1 =  3.0;
    double x2 = -1.0;
    double x3 =  0.0;
    double x4 =  1.0;

    Problem problem;
    // Add residual terms to the problem using the using the autodiff
    // wrapper to get the derivatives automatically. The parameters, x1 through
    // x4, are modified in place.
    problem.AddResidualBlock(new AutoDiffCostFunction<F1, 1, 1, 1>(new F1),
                             NULL,
                             &x1, &x2);
    problem.AddResidualBlock(new AutoDiffCostFunction<F2, 1, 1, 1>(new F2),
                             NULL,
                             &x3, &x4);
    problem.AddResidualBlock(new AutoDiffCostFunction<F3, 1, 1, 1>(new F3),
                             NULL,
                             &x2, &x3);
    problem.AddResidualBlock(new AutoDiffCostFunction<F4, 1, 1, 1>(new F4),
                             NULL,
                             &x1, &x4);

    x1 = 2.56;
    problem.AddParameterBlock(&x1, 1, NULL);
    Solver::Options options;
    ceres::StringToMinimizerType("trust_region", &options.minimizer_type);

    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    std::cout << "Initial x1 = " << x1
              << ", x2 = " << x2
              << ", x3 = " << x3
              << ", x4 = " << x4
              << "\n";

    // Run the solver!
    Solver::Summary summary;
    Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << "\n";
    std::cout << "Final x1 = " << x1
              << ", x2 = " << x2
              << ", x3 = " << x3
              << ", x4 = " << x4
              << "\n";

    ceres::Covariance::Options covOptions;
    ceres::Covariance covariance(covOptions);
    
    std::vector<std::pair<const double*, const double*> > covariance_blocks;
    covariance_blocks.push_back(std::make_pair(&x1, &x1));
    covariance_blocks.push_back(std::make_pair(&x2, &x2));
    covariance_blocks.push_back(std::make_pair(&x1, &x2));
    
    EXPECT_TRUE(covariance.Compute(covariance_blocks, &problem));
    
    double covariance_xx[1];
    double covariance_yy[1];
    double covariance_xy[1];
    covariance.GetCovarianceBlock(&x1, &x1, covariance_xx);
    covariance.GetCovarianceBlock(&x2, &x2, covariance_yy);
    covariance.GetCovarianceBlock(&x1, &x2, covariance_xy);

    std::cout << "covariance x1 " << covariance_xx[0] << std::endl;
    std::cout << "covariance x2 " << covariance_yy[0] << std::endl;
    std::cout << "covariance x1_x2 " << covariance_xy[0] << std::endl;

}

inline Eigen::Matrix<double, 4, 1> computeResidual(const double * const x, const double * const y, 
const std::vector<Prior*>& pxy, const std::vector<RelativePose*>& rxy) {
  Eigen::Matrix<double, 4, 1> b;
  (*pxy[0])(x, b.data());
  (*pxy[1])(y, b.data() + 1);
  (*rxy[0])(x, y, b.data() + 2);
  (*rxy[1])(x, y, b.data() + 3);
  return b;
}

TEST(CovarianceEstimation, Simple)
{
    double initx = 3;
    double inity = 5;
//    double truex = 2.0;
//    double truey = 4.0;
    double x = initx;
    double y = inity;
    Problem problem;
    
    Eigen::Vector4d obsstd(0.1, 0.3, 0.2, 0.25);
    Eigen::Matrix4d mycov = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d myinfo = Eigen::Matrix4d::Identity();
    for(size_t jack=0; jack<4; ++jack){
        mycov(jack,jack) *= obsstd[jack]*obsstd[jack];
        myinfo(jack,jack) /= (obsstd[jack]*obsstd[jack]);
    }
    Prior* px = new Prior(1.9, obsstd[0]);
    Prior* py = new Prior(4.3, obsstd[1]);
    RelativePose * r1 = new RelativePose(2.15, obsstd[2]);
    RelativePose * r2 = new RelativePose(1.8, obsstd[3]);

    problem.AddResidualBlock(new AutoDiffCostFunction<Prior, 1, 1>(px),
                             NULL,
                             &x);
    problem.AddResidualBlock(new AutoDiffCostFunction<Prior, 1, 1>(py),
                             NULL,
                             &y);
    problem.AddResidualBlock(new AutoDiffCostFunction<RelativePose, 1, 1, 1>(r1),
                             NULL,
                             &x, &y);
    problem.AddResidualBlock(new AutoDiffCostFunction<RelativePose, 1, 1, 1>(r2),
                             NULL,
                             &x, &y);

    Eigen::Matrix<double, 4, 2> A;
    A(0, 0) = px->jacobian();
    A(0, 1) = 0;
    A(1, 0) = 0;
    A(1, 1) = py->jacobian();
    A.row(2) = r1->jacobian();
    A.row(3) = r2->jacobian();
    
    Solver::Options options;
    ceres::StringToMinimizerType("trust_region", &options.minimizer_type);

    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    // Fix both
    Solver::Summary summary;
    problem.SetParameterBlockConstant(&x);
    problem.SetParameterBlockConstant(&y);
    Solve(options, &problem, &summary);

    EXPECT_NEAR(x, initx, 1e-8);
    EXPECT_NEAR(y, inity, 1e-8);
    // Fix x
    problem.SetParameterBlockVariable(&y);
    Solve(options, &problem, &summary);
    double expectedy = y;

    EXPECT_NEAR(x, initx, 1e-8);
    y = inity;
    for (int j = 0; j < 3; ++j) {
      Eigen::Matrix<double, 4, 1> b = computeResidual(&x, &y, {px, py}, {r1, r2});
      Eigen::Matrix<double, 1, 1> deltay = (A.col(1).transpose() * A.col(1)).inverse() * (A.col(1).transpose() * b);
      y -= deltay[0];
    }
    EXPECT_NEAR(y, expectedy, 2e-5);

    // Fix y
    y = inity;
    problem.SetParameterBlockVariable(&x);
    problem.SetParameterBlockConstant(&y);
    Solve(options, &problem, &summary);
    double expectedx = x;
    x = initx;
    for (int j = 0; j < 3; ++j) {
      Eigen::Matrix<double, 4, 1> b = computeResidual(&x, &y, {px, py}, {r1, r2});
      Eigen::Matrix<double, 1, 1> deltax = (A.col(0).transpose() * A.col(0)).inverse() * (A.col(0).transpose() * b);
      x -= deltax[0];
    }
    EXPECT_NEAR(x, expectedx, 1e-4);
    EXPECT_NEAR(y, inity, 1e-8);

    // Loosen both
    x = initx;
    y = inity;
    problem.SetParameterBlockVariable(&x);
    problem.SetParameterBlockVariable(&y);
    
    Solve(options, &problem, &summary);
    expectedx = x;
    expectedy = y;
    x = initx;
    y = inity;
    Eigen::Matrix<double, 2, 2> expectedcov;
    for (int j = 0; j < 3; ++j) {
      Eigen::Matrix<double, 4, 1> b = computeResidual(&x, &y, {px, py}, {r1, r2});
      expectedcov = (A.transpose() * A).inverse();
      Eigen::Matrix<double, 2, 1> deltaxy = expectedcov * A.transpose() * b;
      x -= deltaxy[0];
      y -= deltaxy[1];
    }

    EXPECT_NEAR(x, expectedx, 5e-8);
    EXPECT_NEAR(y, expectedy, 5e-8);

    ceres::Covariance::Options covOptions;
    ceres::Covariance covariance(covOptions);

    std::vector<std::pair<const double*, const double*> > covariance_blocks;
    covariance_blocks.push_back(std::make_pair(&x, &x));
    covariance_blocks.push_back(std::make_pair(&y, &y));
    covariance_blocks.push_back(std::make_pair(&x, &y));

    EXPECT_TRUE(covariance.Compute(covariance_blocks, &problem));

    double covariance_xx[1];
    double covariance_yy[1];
    double covariance_xy[1];
    covariance.GetCovarianceBlock(&x, &x, covariance_xx);
    covariance.GetCovarianceBlock(&y, &y, covariance_yy);
    covariance.GetCovarianceBlock(&x, &y, covariance_xy);

    Eigen::Matrix2d cerescov;
    cerescov << covariance_xx[0], covariance_xy[0], covariance_xy[0], covariance_yy[0];

    EXPECT_LT((expectedcov - cerescov).lpNorm<Eigen::Infinity>(), 1e-8)
        << "Ceres gives a covariance different from the expected covariance\n"
        << expectedcov << "\nCeres covariance\n" << cerescov;
}

// Test ceres covariance computation with a relative large scale visual inertial problem.
TEST(CovarianceEstimation, MonoVisualInertial) {
  // TODO(jhuai): Currently, OKVIS estimator constructs a ceres Problem, add
  // parameter blocks and residual blocks incrementally, then optimize the
  // solution, and marginalize redundant variables. But regardless of the
  // marginalization step, the estimator always fails to compute the covariance
  // because of rank deficient Jacobian with SPARSE_QR approach.

  // refer to ceres-solver/internal/ceres/covariance_test.cc, and TestEstimator
  // to construct a test fixture.
}
