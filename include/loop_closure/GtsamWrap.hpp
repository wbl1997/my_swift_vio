#ifndef INCLUDE_LOOP_CLOSURE_GTSAM_WRAP_HPP_
#define INCLUDE_LOOP_CLOSURE_GTSAM_WRAP_HPP_

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>

#include "ceres/internal/autodiff.h"
#include "sophus/se3.hpp"

#include <okvis/kinematics/Transformation.hpp>

namespace VIO {
/**
 * @brief compute Jacobian of the between factor unwhitened error relative
 * to the measurement error by autoDifferentiate.
 * @warning use this class sparingly as it is likely expensive.
 */
class BetweenFactorPose3Wrap {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BetweenFactorPose3Wrap(const gtsam::Pose3& Tz, const gtsam::Pose3& Tx,
                         const gtsam::Pose3& Ty);

  explicit BetweenFactorPose3Wrap(const gtsam::Pose3& Tz);

  template <typename T>
  bool operator()(const T* deltaz, T* residual) const {
    Eigen::Map<const Eigen::Matrix<T, 6, 1>> deltavec(deltaz);
    Eigen::Matrix<T, 3, 1> omega = deltavec.template tail<3>();
    Sophus::SO3Group<T> so3Delta = Sophus::SO3Group<T>::exp(omega);
    Eigen::Quaternion<T> qDelta = so3Delta.unit_quaternion();
    Eigen::Quaternion<T> q = Tz_.rotation().toQuaternion().cast<T>();
    Eigen::Matrix<T, 3, 1> t = Tz_.translation().cast<T>();
    Eigen::Quaternion<T> qNew = qDelta * q;
    Eigen::Matrix<T, 3, 1> tNew = t + deltavec.template head<3>();
    Eigen::Quaternion<T> qxy = Txy_.rotation().toQuaternion().cast<T>();
    Eigen::Matrix<T, 3, 1> txy = Txy_.translation().cast<T>();
    Sophus::SE3Group<T> deltaT(qNew.conjugate() * qxy,
                               qNew.conjugate() * (txy - tNew));
    Eigen::Map<Eigen::Matrix<T, 6, 1>> residualp(residual);
    residualp = deltaT.log();
    Eigen::Matrix<T, 3, 1> temp = residualp.template head<3>();
    residualp.template head<3>() = residualp.template tail<3>();
    residualp.template tail<3>() = temp;
    return true;
  }

  void toMeasurmentJacobian(Eigen::Matrix<double, 6, 6, Eigen::RowMajor>* autoJ,
                            Eigen::Matrix<double, 6, 1>* residual);

  const gtsam::Pose3 Tz_;
  const gtsam::Pose3 Txy_;
};

class GtsamWrap {
 public:
  /**
   * @brief retract
   * @param Tz
   * @param delta [\nu, \omega]
   * @return
   */
  static gtsam::Pose3 retract(gtsam::Pose3 Tz,
                              Eigen::Matrix<double, 6, 1>& delta);

  /**
   * @brief compute Jacobian of the between factor unwhitened error relative
   * to the measurement error.
   * The unwhitened error of the between factor is defined by retraction of
   * Pose3 in gtsam. Its error order is [\omega, t]. The measurement error is
   * defined with oplus in okvis::Transformation. Its error order is [t, \omega].
   * @warning This function is inaccurate. Use BetweenFactorPose3Wrap instead.
   * @param Tz the measurement used to construct the BetweenFactor<Pose3>.
   * @param JtoCustomRetract
   */
  static void toMeasurementJacobianBetweenFactor(
      gtsam::Pose3 Tz, gtsam::Pose3 Tx, gtsam::Pose3 Ty,
      Eigen::Matrix<double, 6, 6>* JtoCustomRetract);

  /**
   * @brief compute Jacobian of the between factor unwhitened error relative
   * to the measurement error, assuming Tz is close to Tx^{-1}Ty
   * @param Tz
   * @param JtoCustomRetract
   */
  static void toMeasurementJacobianBetweenFactor(
      gtsam::Pose3 Tz,
      Eigen::Matrix<double, 6, 6>* JtoCustomRetract);

  inline static gtsam::Pose3 toPose3(const okvis::kinematics::Transformation& T) {
    return gtsam::Pose3(gtsam::Rot3(T.q()), T.r());
  }

  inline static okvis::kinematics::Transformation toTransform(const gtsam::Pose3& P) {
    return okvis::kinematics::Transformation(
          P.translation(), P.rotation().toQuaternion());
  }
};
} // namespace VIO

#endif // INCLUDE_LOOP_CLOSURE_GTSAM_WRAP_HPP_
