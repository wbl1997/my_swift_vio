/**
 * @file    RiProjectionFactorIDPAnchor.h
 * @brief   Reprojection factor with inverse depth parameterization at the
 * anchor frame.
 * @author  Jianzhu Huai
 */

#ifndef GTSAM_RI_PROJECTION_FACTOR_IDP_ANCHOR_H
#define GTSAM_RI_PROJECTION_FACTOR_IDP_ANCHOR_H

#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <okvis/cameras/CameraBase.hpp>
#include <okvis/kinematics/Transformation.hpp>

namespace gtsam {

/**
 * \brief Projection factor with inverse depth parameterization at the anchor frame.
 * NOTE: anchor only needs to link to the feature node
 */
class RiProjectionFactorIDPAnchor : public NoiseModelFactor1<Point3> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  Eigen::Vector2d uv_;

  std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry_;
  okvis::kinematics::Transformation T_BCj_;
  okvis::kinematics::Transformation T_BCa_;

 public:
  RiProjectionFactorIDPAnchor() {}

  RiProjectionFactorIDPAnchor(Key state_p,
                        const Eigen::Vector2d& variance,
                        const Eigen::Vector2d& uv,
                        std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry,
                        const okvis::kinematics::Transformation& T_BCj,
                        const okvis::kinematics::Transformation& T_BCa)
      : NoiseModelFactor1<Point3>(
            noiseModel::Isotropic::Variances(variance), state_p),
        uv_(uv),
        cameraGeometry_(cameraGeometry),
        T_BCj_(T_BCj),
        T_BCa_(T_BCa) {}

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

  gtsam::Vector evaluateError(const Point3& state_p,
                              boost::optional<Matrix&> H1 = boost::none) const;

  GTSAM_EXPORT
  friend std::ostream& operator<<(std::ostream& os,
                                  const RiProjectionFactorIDPAnchor& factor) {
    os << "uv:[" << factor.measurement()(0) << ", " << factor.measurement()(1) << "]'"
       << std::endl;
    return os;
  }

  void print(const std::string& s,
             const KeyFormatter& keyFormatter = DefaultKeyFormatter) const {
    std::cout << s << "RiProjectionFactorIDPAnchor("
              << keyFormatter(this->key()) << ")" << std::endl;
    std::cout << "  measured: " << std::endl << *this << std::endl;
    this->noiseModel_->print("  noise model: ");
  }

  bool equals(const NonlinearFactor& expected, double tol = 1e-9) const {
    const RiProjectionFactorIDPAnchor* e =
        dynamic_cast<const RiProjectionFactorIDPAnchor*>(&expected);
    if (e == NULL) return false;
    return NoiseModelFactor1<Point3>::equals(*e, tol) &&
            gtsam::equal(uv_, e->measurement(), tol) &&
            cameraGeometry_.get() == e->cameraGeometry().get() &&
            gtsam::equal(T_BCj_.T3x4(), e->T_BCj().T3x4(), tol) &&
            gtsam::equal(T_BCa_.T3x4(), e->T_BCa().T3x4(), tol);
  }
};
}  // namespace gtsam

#endif /* GTSAM_REPROJECTION_FACTOR_IDP_ANCHOR_H */
