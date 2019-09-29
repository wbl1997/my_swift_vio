#ifndef INCLUDE_MSCKF_RELATIVE_MOTION_JACOBIAN_HPP_
#define INCLUDE_MSCKF_RELATIVE_MOTION_JACOBIAN_HPP_

#include "okvis/kinematics/Transformation.hpp"
#include "vio/eigen_utils.h"

namespace okvis {
inline okvis::kinematics::Transformation randomTransform() {
  Eigen::Quaterniond q(Eigen::Vector4d::Random());
  q.normalize();
  Eigen::Vector3d r(Eigen::Vector3d::Random());
  return okvis::kinematics::Transformation(r, q);
}

inline Eigen::Matrix<double, 6, 1> ominus(
    const okvis::kinematics::Transformation& Tbar,
    const okvis::kinematics::Transformation& T) {
  Eigen::Matrix<double, 3, 4> dT = Tbar.T3x4() - T.T3x4();
  Eigen::Matrix<double, 6, 1> delta;
  delta.head<3>() = dT.col(3);
  delta.tail<3>() = vio::unskew3d(dT.topLeftCorner<3, 3>() * T.C().transpose());
  return delta;
}

class RelativeMotionJacobian {
 public:
  RelativeMotionJacobian(const okvis::kinematics::Transformation& T_BC,
                         const okvis::kinematics::Transformation& T_GBj,
                         const okvis::kinematics::Transformation& T_GBk);
  okvis::kinematics::Transformation evaluate() const;
  void dtheta_dtheta_BC(Eigen::Matrix3d* jac) const;
  void dtheta_dtheta_GBj(Eigen::Matrix3d* jac) const;
  void dtheta_dtheta_GBk(Eigen::Matrix3d* jac) const;
  void dp_dtheta_BC(Eigen::Matrix3d* jac) const;
  void dp_dtheta_GBj(Eigen::Matrix3d* jac) const;
  void dp_dtheta_GBk(Eigen::Matrix3d* jac) const;
  void dp_dt_BC(Eigen::Matrix3d* jac) const;
  void dp_dt_GBj(Eigen::Matrix3d* jac) const;
  void dp_dt_GBk(Eigen::Matrix3d* jac) const;

 private:
  const okvis::kinematics::Transformation T_BC_;
  const okvis::kinematics::Transformation T_GBj_;
  const okvis::kinematics::Transformation T_GBk_;
};
} // namespace okvis
#endif  // INCLUDE_MSCKF_RELATIVE_MOTION_JACOBIAN_HPP_
