/**
 * @file   ImuFactorTestHelpers.h
 * @brief  Helper functions for testing gtsam ImuFactor and RiImuFactor.
 * @author Jianzhu Huai
 */

#ifndef IMUFACTORTESTHELPERS_H
#define IMUFACTORTESTHELPERS_H

#include <okvis/kinematics/Transformation.hpp>

namespace swift_vio {
/**
 * @brief dokvis_drightinvariant Jacobian of the okvis error vector relative to
 * the right invariant error vector for a nav state. The okvis error vector is
 * used in okvis::ceres::ImuError. The right invariant error vector is used in
 * RiImuFactor.
 * @param T_WB
 * @param v_WB
 * @return d(p, \alpha, v, bg, ba) / d(\phi, v, p, ba, bg)
 */
Eigen::Matrix<double, 15, 15> dokvis_drightinvariant(
    const okvis::kinematics::Transformation& T_WB, const Eigen::Vector3d& v_WB);

/**
 * @brief drightinvariant_dokvis
 * @param T_WB
 * @param v_WB
 * @return d(\phi, v, p, ba, bg) / d(p, \alpha, v, bg, ba)
 */
Eigen::Matrix<double, 15, 15> drightinvariant_dokvis(
    const okvis::kinematics::Transformation& T_WB, const Eigen::Vector3d& v_WB);

/**
 * @brief dokvis_dforster Jacobian of the okvis error vector relative to
 * the error vector of Forster et al. On Manifold Preintegration for a nav
 * state. The okvis error vector is used in okvis::ceres::ImuError. The Forster
 * error vector is used in gtsam::ImuFactor.
 * @param T_WB
 * @param v_WB
 * @return d(p, \alpha, v, bg, ba) / d(\phi, p, v, ba, bg)
 */
Eigen::Matrix<double, 15, 15> dokvis_dforster(
    const Eigen::Matrix3d& R_WB);

inline Eigen::Matrix<double, 15, 15> dokvis_dforster(
    const Eigen::Quaterniond& q_WB) {
  return dokvis_dforster(q_WB.toRotationMatrix());
}

/**
 * @brief dforster_dokvis
 * @param T_WB
 * @param v_WB
 * @return  d(\phi, p, v, ba, bg) / d(p, \alpha, v, bg, ba)
 */
Eigen::Matrix<double, 15, 15> dforster_dokvis(
    const Eigen::Matrix3d& R_WB);

inline Eigen::Matrix<double, 15, 15> dforster_dokvis(
    const Eigen::Quaterniond& q_WB) {
  return dforster_dokvis(q_WB.toRotationMatrix());
}
}; // namespace swift_vio
#endif // IMUFACTORTESTHELPERS_H
