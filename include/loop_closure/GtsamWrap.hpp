#ifndef INCLUDE_LOOP_CLOSURE_GTSAM_WRAP_HPP_
#define INCLUDE_LOOP_CLOSURE_GTSAM_WRAP_HPP_

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>

#include <okvis/kinematics/Transformation.hpp>

namespace VIO {
class GtsamWrap {
 public:
  static gtsam::Pose3 retract(gtsam::Pose3 Tz,
                              Eigen::Matrix<double, 6, 1>& delta);

  /**
   * @brief toMeasurementJacobianBetweenFactor compute Jacobian of the between
   * factor unwhitened error relative to the measurement error.
   * The unwhitened error of the between factor is defined by retraction of
   * Pose3 in gtsam. Its error order is [\omega, t] The measurement error is
   * defined with retraction in OKVIS, i.e., oplus in Transformation. Its error
   * order is [t, \omega].
   * @param Tz the measurement used to construct the BetweenFactor<Pose3>.
   * @param JtoCustomRetract
   */
  static void toMeasurementJacobianBetweenFactor(
      gtsam::Pose3 Tz, Eigen::Matrix<double, 6, 6>* JtoCustomRetract);

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
