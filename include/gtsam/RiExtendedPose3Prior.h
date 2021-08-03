/**
 * @file   RiExtendedPose3Prior.h
 * @brief  RiExtendedPose3Prior class header.
 */

#ifndef RI_EXTENDED_POSE3_PRIOR_H
#define RI_EXTENDED_POSE3_PRIOR_H

#include <gtsam/base/Manifold.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <Eigen/Dense>
#include <Eigen/Eigen>

#include "gtsam/RiExtendedPose3.h"

namespace gtsam {
/**
 * \brief RiExtendedPose3 Prior
 */
class RiExtendedPose3Prior : public NoiseModelFactor1<RiExtendedPose3> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  RiExtendedPose3 measured_;  ///< R_WB, V_WB, P_WB
  bool lockJacobian_;

 public:
  RiExtendedPose3Prior() {}

  RiExtendedPose3Prior(Key state_i, const RiExtendedPose3 &z,
                       const Eigen::Matrix<double, 9, 9> &covariance,
                       bool lockJacobian)
      : NoiseModelFactor1<RiExtendedPose3>(
            noiseModel::Gaussian::Covariance(covariance), state_i),
        measured_(z),
        lockJacobian_(lockJacobian) {}

  RiExtendedPose3Prior(Key state_i, const Eigen::Quaterniond &q, const Eigen::Vector3d &v,
                       const Eigen::Vector3d &p, const Eigen::Matrix<double, 9, 9>& covariance)
      : NoiseModelFactor1<RiExtendedPose3>(
            noiseModel::Gaussian::Covariance(covariance), state_i),
        measured_(gtsam::Rot3(q), v, p) {}

  /// Return rotation quaternion.
  Eigen::Quaterniond q() const { return measured_.rotation().toQuaternion(); }

  /// Return position as Vector3
  Vector3 p() const { return measured_.position(); }

  /// Return velocity as Vector3
  Vector3 v() const { return measured_.velocity(); }

  const RiExtendedPose3& measurement() const {
    return measured_;
  }
  /**
   * @brief evaluateError Given the current states, calculate the measurement
   * error/residual
   * @param state_i
   * @param H1
   * @return measurement.localCoordinates(state_i)
   */
  gtsam::Vector evaluateError(const RiExtendedPose3 &state_i,
                              boost::optional<Matrix &> H1 = boost::none) const;

  /// How this node gets printed in the ostream
  GTSAM_EXPORT
  friend std::ostream &operator<<(std::ostream &os,
                                  const RiExtendedPose3Prior &state) {
    os << "prior:" << state.measurement();
    return os;
  }

  /// Print function for this factor
  void print(const std::string &s,
             const KeyFormatter &keyFormatter = DefaultKeyFormatter) const {
    std::cout << s << "RiExtendedPose3Prior(" << keyFormatter(this->key())
              << ")" << std::endl;
    std::cout << "  measured: " << std::endl << *this << std::endl;
    this->noiseModel_->print("  noise model: ");
  }

  /// Define how two factors can be equal to each other
  bool equals(const NonlinearFactor &expected, double tol = 1e-9) const {
    // Cast the object
    const RiExtendedPose3Prior *e =
        dynamic_cast<const RiExtendedPose3Prior *>(&expected);
    if (e == NULL) return false;
    // Success, compare base noise values and the measurement values
    return NoiseModelFactor1<RiExtendedPose3>::equals(*e, tol) &&
           gtsam::equal(measured_, e->measurement(), tol);
  }
};

}  // namespace gtsam

#endif /* RI_EXTENDED_POSE3_PRIOR_H */
