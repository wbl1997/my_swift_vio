#include <iostream>
#include "gtest/gtest.h"

#include <swift_vio/EuclideanParamError.hpp>

TEST(EuclideanParamError, Jacobian) {
  const int kVarDim = 4;
  const double eps = 1e-8;
  Eigen::Matrix<double, kVarDim, 1> meas =
      Eigen::Matrix<double, kVarDim, 1>::Random();
  Eigen::Matrix<double, kVarDim, 1> var =
      Eigen::Matrix<double, kVarDim, 1>::Ones();
  var = var.cwiseAbs();
  okvis::ceres::EuclideanParamError<kVarDim> epe(meas, var);
  Eigen::Matrix<double, kVarDim, 1> state =
      Eigen::Matrix<double, kVarDim, 1>::Random();
  Eigen::Matrix<double, kVarDim, kVarDim, Eigen::RowMajor> jac, minimalJac;
  Eigen::Matrix<double, kVarDim, 1> residuals;
  double const* const params[] = {state.data()};
  double* jacData[] = {jac.data()};
  double* minimalJacData[] = {minimalJac.data()};
  epe.EvaluateWithMinimalJacobians(params, residuals.data(), jacData,
                                   minimalJacData);
  EXPECT_TRUE(jac.isApprox(
      Eigen::Matrix<double, kVarDim, kVarDim, Eigen::RowMajor>::Identity() *
          (-1),
      eps));
  EXPECT_TRUE(minimalJac.isApprox(
      Eigen::Matrix<double, kVarDim, kVarDim, Eigen::RowMajor>::Identity() *
          (-1),
      eps));
}
