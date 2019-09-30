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
  inline RelativeMotionJacobian(const okvis::kinematics::Transformation& T_BC,
                         const okvis::kinematics::Transformation& T_GBj,
                         const okvis::kinematics::Transformation& T_GBk);
  inline okvis::kinematics::Transformation evaluate() const;
  inline void dtheta_dtheta_BC(Eigen::Matrix3d* jac) const;
  inline void dtheta_dtheta_GBj(Eigen::Matrix3d* jac) const;
  inline void dtheta_dtheta_GBk(Eigen::Matrix3d* jac) const;
  inline void dp_dtheta_BC(Eigen::Matrix3d* jac) const;
  inline void dp_dtheta_GBj(Eigen::Matrix3d* jac) const;
  inline void dp_dtheta_GBk(Eigen::Matrix3d* jac) const;
  inline void dp_dt_BC(Eigen::Matrix3d* jac) const;
  inline void dp_dt_GBj(Eigen::Matrix3d* jac) const;
  inline void dp_dt_GBk(Eigen::Matrix3d* jac) const;
  inline void dp_dt_CB(Eigen::Matrix3d* jac) const;

 private:
  const okvis::kinematics::Transformation T_BC_;
  const okvis::kinematics::Transformation T_GBj_;
  const okvis::kinematics::Transformation T_GBk_;
};

inline RelativeMotionJacobian::RelativeMotionJacobian(
    const okvis::kinematics::Transformation& T_BC,
    const okvis::kinematics::Transformation& T_GBj,
    const okvis::kinematics::Transformation& T_GBk)
    : T_BC_(T_BC), T_GBj_(T_GBj), T_GBk_(T_GBk) {}
inline okvis::kinematics::Transformation RelativeMotionJacobian::evaluate() const {
  return (T_GBj_ * T_BC_).inverse() * (T_GBk_ * T_BC_);
}

inline void RelativeMotionJacobian::dtheta_dtheta_BC(Eigen::Matrix3d* jac) const {
  Eigen::Matrix3d R_BjBk = T_GBj_.C().transpose() * T_GBk_.C();
  *jac = T_BC_.C().transpose() *
         (R_BjBk - Eigen::Matrix3d::Identity());
}
inline void RelativeMotionJacobian::dtheta_dtheta_GBj(Eigen::Matrix3d* jac) const {
  Eigen::Matrix3d R_GCj = T_GBj_.C() * T_BC_.C();
  *jac = -R_GCj.transpose();
}
inline void RelativeMotionJacobian::dtheta_dtheta_GBk(Eigen::Matrix3d* jac) const {
  dtheta_dtheta_GBj(jac);
  *jac = -(*jac);
}

inline void RelativeMotionJacobian::dp_dtheta_BC(Eigen::Matrix3d* jac) const {
  okvis::kinematics::Transformation T_CjCk = evaluate();
  *jac = vio::skew3d(T_CjCk.r()) * T_BC_.C().transpose();
}
inline void RelativeMotionJacobian::dp_dtheta_GBj(Eigen::Matrix3d* jac) const {
  okvis::kinematics::Transformation T_GCk = T_GBk_ * T_BC_;
  Eigen::Matrix3d R_GCj = T_GBj_.C() * T_BC_.C();
  *jac = R_GCj.transpose() * vio::skew3d(T_GCk.r() - T_GBj_.r());
}
inline void RelativeMotionJacobian::dp_dtheta_GBk(Eigen::Matrix3d* jac) const {
  Eigen::Matrix3d R_GCj = T_GBj_.C() * T_BC_.C();
  *jac = -R_GCj.transpose() * vio::skew3d(T_GBk_.C() * T_BC_.r());
}
inline void RelativeMotionJacobian::dp_dt_BC(Eigen::Matrix3d* jac) const {
  okvis::kinematics::Transformation T_CjCk = evaluate();
  *jac = (T_CjCk.C() - Eigen::Matrix3d::Identity()) * T_BC_.C().transpose();
}
inline void RelativeMotionJacobian::dp_dt_GBj(Eigen::Matrix3d* jac) const {
  *jac = -(T_GBj_.C() * T_BC_.C()).transpose();
}
inline void RelativeMotionJacobian::dp_dt_GBk(Eigen::Matrix3d* jac) const {
  *jac = (T_GBj_.C() * T_BC_.C()).transpose();
}
inline void RelativeMotionJacobian::dp_dt_CB(Eigen::Matrix3d* jac) const {
  okvis::kinematics::Transformation T_CjCk = evaluate();
  *jac = Eigen::Matrix3d::Identity() - T_CjCk.C();
}

} // namespace okvis
#endif  // INCLUDE_MSCKF_RELATIVE_MOTION_JACOBIAN_HPP_
