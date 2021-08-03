/**
 * @file   ImuFactorTestHelpers.cpp
 * @brief  Helper functions for testing gtsam ImuFactor and RiImuFactor.
 * @author Jianzhu Huai
 */
#include <gtsam/ImuFactorTestHelpers.h>
#include <okvis/kinematics/operators.hpp>

namespace swift_vio {
Eigen::Matrix<double, 15, 15> dokvis_drightinvariant(
    const okvis::kinematics::Transformation& T_WB,
    const Eigen::Vector3d& v_WB) {
  Eigen::Matrix<double, 15, 15> jacobian = Eigen::Matrix<double, 15, 15>::Zero();
  jacobian.block<3, 3>(0, 0) = okvis::kinematics::crossMx(-T_WB.r());
  jacobian.block<3, 3>(0, 6).setIdentity();
  jacobian.block<3, 3>(3, 0).setIdentity();
  jacobian.block<3, 3>(6, 0) = okvis::kinematics::crossMx(-v_WB);
  jacobian.block<3, 3>(6, 3).setIdentity();
  jacobian.block<3, 3>(9, 12).setIdentity();
  jacobian.block<3, 3>(12, 9).setIdentity();
  return jacobian;
}

Eigen::Matrix<double, 15, 15> drightinvariant_dokvis(
    const okvis::kinematics::Transformation& T_WB,
    const Eigen::Vector3d& v_WB) {
  Eigen::Matrix<double, 15, 15> jacobian = Eigen::Matrix<double, 15, 15>::Zero();
  jacobian.block<3, 3>(0, 3).setIdentity();
  jacobian.block<3, 3>(3, 3) = okvis::kinematics::crossMx(v_WB);
  jacobian.block<3, 3>(3, 6).setIdentity();
  jacobian.block<3, 3>(6, 0).setIdentity();
  jacobian.block<3, 3>(6, 3) = okvis::kinematics::crossMx(T_WB.r());
  jacobian.block<3, 3>(9, 12).setIdentity();
  jacobian.block<3, 3>(12, 9).setIdentity();
  return jacobian;
}

Eigen::Matrix<double, 15, 15> dokvis_dforster(const Eigen::Matrix3d& R_WB) {
  Eigen::Matrix<double, 15, 15> jacobian = Eigen::Matrix<double, 15, 15>::Zero();
  jacobian.block<3, 3>(0, 3) = R_WB;
  jacobian.block<3, 3>(3, 0) = R_WB;
  jacobian.block<3, 3>(6, 6).setIdentity();
  jacobian.block<3, 3>(9, 12).setIdentity();
  jacobian.block<3, 3>(12, 9).setIdentity();
  return jacobian;
}

Eigen::Matrix<double, 15, 15> dforster_dokvis(const Eigen::Matrix3d& R_WB) {
  Eigen::Matrix<double, 15, 15> jacobian = Eigen::Matrix<double, 15, 15>::Zero();
  jacobian.block<3, 3>(0, 3) = R_WB.transpose();
  jacobian.block<3, 3>(3, 0) = R_WB.transpose();
  jacobian.block<3, 3>(6, 6).setIdentity();
  jacobian.block<3, 3>(9, 12).setIdentity();
  jacobian.block<3, 3>(12, 9).setIdentity();
  return jacobian;
}
}  // namespace swift_vio
