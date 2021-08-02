/**
 * @file    RiProjectionFactorIDP.h
 * @brief   Reprojection factor with inverse depth parameterization at a frame
 * other than the anchor.
 * @author  Jianzhu Huai
 */

#ifndef GTSAM_RI_PROJECTION_FACTOR_IDP_H
#define GTSAM_RI_PROJECTION_FACTOR_IDP_H

#include <Eigen/Core>

#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/linear/NoiseModel.h>

#include "gtsam/RiExtendedPose3.h"

#include <okvis/cameras/CameraBase.hpp>
#include <okvis/kinematics/Transformation.hpp>

namespace gtsam {

/**
 * \brief Reprojection factor with inverse depth parameterization at a general
 * frame other than the anchor. Links a anchor RiExtendedPose3, observing frame
 * RiExtendedPose3 and a 3D feature in the environment of an inverse depth
 * representation.
 */
class RiProjectionFactorIDP
    : public NoiseModelFactor3<RiExtendedPose3, RiExtendedPose3, Point3> {
    // body pose at observing frame j, body pose at anchor frame a, landmark parameters.
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  Eigen::Vector2d uv_;  ///< uv coordinates in pixels.

  std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry_;

  okvis::kinematics::Transformation T_BCj_; ///< transform from this camera frame to the body frame.

  okvis::kinematics::Transformation T_BCa_; ///< transform from the ANCHOR camera frame to the body frame.

  std::vector<int> exponentList_;

 public:
  RiProjectionFactorIDP() {}

  RiProjectionFactorIDP(Key state_j, Key state_a, Key state_p,
                        const Eigen::Vector2d& variance,
                        const Eigen::Vector2d& uv,
                        std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry,
                        const okvis::kinematics::Transformation& T_BCj,
                        const okvis::kinematics::Transformation& T_BCa);

  //  * @param
  /**
   * @brief RiProjectionFactorIDP
   * @param state_j
   * @param state_a
   * @param state_p
   * @param variance
   * @param uv
   * @param cameraGeometry
   * @param T_BCj
   * @param T_BCa
   * @param huber_threshold in units of sigmas. sqrt(5.99) ~ 2.447.
   * A sound value is obtained by checking the Chi2 distribution with 2DOF at alpha=5%.
   */
  RiProjectionFactorIDP(
      Key state_j, Key state_a, Key state_p,
      const Eigen::Vector2d& variance, const Eigen::Vector2d& uv,
      std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry,
      const okvis::kinematics::Transformation& T_BCj,
      const okvis::kinematics::Transformation& T_BCa,
      double huber_threshold);

  const Eigen::Vector2d& measurement() const { return uv_; }

  std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry() const {
      return cameraGeometry_;
  }

  const okvis::kinematics::Transformation& T_BCj() const {
    return T_BCj_;
  }

  const okvis::kinematics::Transformation& T_BCa() const {
    return T_BCa_;
  }

  gtsam::Vector evaluateError(const RiExtendedPose3& state_j,
                              const RiExtendedPose3& state_a, const Point3& state_p,
                              boost::optional<Matrix&> H1 = boost::none,
                              boost::optional<Matrix&> H2 = boost::none,
                              boost::optional<Matrix&> H3 = boost::none) const;

  GTSAM_EXPORT
  friend std::ostream& operator<<(std::ostream& os,
                                  const RiProjectionFactorIDP& factor) {
    os << "uv:[" << factor.measurement()(0) << ", " << factor.measurement()(1) << "]'\n";
    return os;
  }

  void print(const std::string& s,
             const KeyFormatter& keyFormatter = DefaultKeyFormatter) const {
    std::cout << s << "RiProjectionFactorIDP(" << keyFormatter(this->key1())
              << "," << keyFormatter(this->key2()) << ","
              << keyFormatter(this->key3()) << ")" << std::endl;
    std::cout << "  measured: " << std::endl << *this << std::endl;
    this->noiseModel_->print("  noise model: ");
  }

  bool equals(const NonlinearFactor& expected, double tol = 1e-9) const {
    const RiProjectionFactorIDP* e =
        dynamic_cast<const RiProjectionFactorIDP*>(&expected);
    if (e == NULL) return false;
    return NoiseModelFactor3<RiExtendedPose3, RiExtendedPose3, Point3>::equals(*e,
                                                                       tol) &&
           gtsam::equal(uv_, e->measurement(), tol) &&
           cameraGeometry_.get() == e->cameraGeometry().get() &&
           gtsam::equal(T_BCj_.T3x4(), e->T_BCj().T3x4(), tol) &&
           gtsam::equal(T_BCa_.T3x4(), e->T_BCa().T3x4(), tol);
  }
};
}  // namespace gtsam

#endif /* GTSAM_RI_PROJECTION_FACTOR_IDP_H */
