/**
 * @file   RiExtendedPose3Prior.cpp
 * @brief  RiExtendedPose3Prior class.
 */

#include "gtsam/RiExtendedPose3Prior.h"

namespace gtsam {
/**
 * Called on when optimizing to get the error of this measurement
 */
gtsam::Vector RiExtendedPose3Prior::evaluateError(
    const RiExtendedPose3& state_i, boost::optional<Matrix&> H1) const {
  Eigen::Matrix<double, 9, 1> error = measured_.localCoordinates(state_i);
  if (H1) {
    if (lockJacobian_) {
      *H1 = Eigen::Matrix<double, 9, 9>::Identity();
    } else {
      *H1 = gtsam::geometry::SEK3Jl_inv(error);
    }
  }
  return error;
}
}  // namespace gtsam
