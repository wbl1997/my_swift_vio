#include "msckf/RelativeMotionJacobian.hpp"

#include "vio/eigen_utils.h"

namespace okvis {
RelativeMotionJacobian::RelativeMotionJacobian(
    const okvis::kinematics::Transformation& T_BC,
    const okvis::kinematics::Transformation& T_GBj,
    const okvis::kinematics::Transformation& T_GBk)
    : T_BC_(T_BC), T_GBj_(T_GBj), T_GBk_(T_GBk) {}
okvis::kinematics::Transformation RelativeMotionJacobian::evaluate() const {
  return (T_GBj_ * T_BC_).inverse() * (T_GBk_ * T_BC_);
}

void RelativeMotionJacobian::dtheta_dtheta_BC(Eigen::Matrix3d* jac) const {
  Eigen::Matrix3d R_BjBk = T_GBj_.C().transpose() * T_GBk_.C();
  *jac = T_BC_.C().transpose() *
         (R_BjBk - Eigen::Matrix3d::Identity());
}
void RelativeMotionJacobian::dtheta_dtheta_GBj(Eigen::Matrix3d* jac) const {
  Eigen::Matrix3d R_GCj = T_GBj_.C() * T_BC_.C();
  *jac = -R_GCj.transpose();
}
void RelativeMotionJacobian::dtheta_dtheta_GBk(Eigen::Matrix3d* jac) const {
  dtheta_dtheta_GBj(jac);
  *jac = -(*jac);
}

void RelativeMotionJacobian::dp_dtheta_BC(Eigen::Matrix3d* jac) const {
  okvis::kinematics::Transformation T_CjCk = evaluate();
  *jac = vio::skew3d(T_CjCk.r()) * T_BC_.C().transpose();
}
void RelativeMotionJacobian::dp_dtheta_GBj(Eigen::Matrix3d* jac) const {
  okvis::kinematics::Transformation T_GCk = T_GBk_ * T_BC_;
  Eigen::Matrix3d R_GCj = T_GBj_.C() * T_BC_.C();
  *jac = R_GCj.transpose() * vio::skew3d(T_GCk.r() - T_GBj_.r());
}
void RelativeMotionJacobian::dp_dtheta_GBk(Eigen::Matrix3d* jac) const {
  Eigen::Matrix3d R_GCj = T_GBj_.C() * T_BC_.C();
  *jac = -R_GCj.transpose() * vio::skew3d(T_GBk_.C() * T_BC_.r());
}
void RelativeMotionJacobian::dp_dt_BC(Eigen::Matrix3d* jac) const {
  okvis::kinematics::Transformation T_CjCk = evaluate();
  *jac = (T_CjCk.C() - Eigen::Matrix3d::Identity()) * T_BC_.C().transpose();
}
void RelativeMotionJacobian::dp_dt_GBj(Eigen::Matrix3d* jac) const {
  *jac = -(T_GBj_.C() * T_BC_.C()).transpose();
}
void RelativeMotionJacobian::dp_dt_GBk(Eigen::Matrix3d* jac) const {
  *jac = (T_GBj_.C() * T_BC_.C()).transpose();
}

}  // namespace okvis
