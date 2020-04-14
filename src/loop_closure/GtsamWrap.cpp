#include "loop_closure/GtsamWrap.hpp"

namespace VIO {
gtsam::Pose3 GtsamWrap::retract(gtsam::Pose3 Tz,
                                Eigen::Matrix<double, 6, 1>& delta) {
  okvis::kinematics::Transformation Tza = toTransform(Tz);
  Tza.oplus(delta);
  return toPose3(Tza);
}

void GtsamWrap::toMeasurementJacobianBetweenFactor(
    gtsam::Pose3 Tz, Eigen::Matrix<double, 6, 6>* JtoCustomRetract) {
  JtoCustomRetract->topLeftCorner<3, 3>().setZero();
  Eigen::Matrix3d mRzp = -Tz.rotation().matrix().transpose();
  JtoCustomRetract->topRightCorner<3, 3>() = mRzp;
  JtoCustomRetract->bottomLeftCorner<3, 3>() = mRzp;
  JtoCustomRetract->bottomRightCorner<3, 3>().setZero();
}

} // namespace VIO
