/**
 * @file   InverseTransformPointJacobian.h
 * @brief  Jacobians for T^{-1} * p.
 * where p is a 4D homogeneous point,
 * T = [Expmap(\alpha) * R   t + \delta t;
 *       0^T                 1].
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_SWIFT_VIO_INVERSE_TRANSFORM_POINT_JACOBIAN_HPP
#define INCLUDE_SWIFT_VIO_INVERSE_TRANSFORM_POINT_JACOBIAN_HPP

#include <okvis/kinematics/Transformation.hpp>
namespace swift_vio {
class InverseTransformPointJacobian {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  inline InverseTransformPointJacobian();

  inline virtual ~InverseTransformPointJacobian();

  inline void initialize(const okvis::kinematics::Transformation& T_AB,
                         const Eigen::Vector4d& hpA);

  inline InverseTransformPointJacobian(
      const okvis::kinematics::Transformation& T_AB,
      const Eigen::Vector4d& hpA);

  inline virtual void dhpB_dT_AB(Eigen::Matrix<double, 4, 6>* j) const;

  inline void dhpB_dhpA(Eigen::Matrix<double, 4, 4>* j) const;

  inline Eigen::Vector4d evaluate() const;

 protected:
  okvis::kinematics::Transformation T_AB_;
  Eigen::Vector4d hpA_;
};

InverseTransformPointJacobian::InverseTransformPointJacobian() {}

InverseTransformPointJacobian::~InverseTransformPointJacobian() {}

void InverseTransformPointJacobian::initialize(
    const okvis::kinematics::Transformation& T_AB, const Eigen::Vector4d& hpA) {
  T_AB_ = T_AB;
  hpA_ = hpA;
}

InverseTransformPointJacobian::InverseTransformPointJacobian(
    const okvis::kinematics::Transformation& T_AB, const Eigen::Vector4d& hpA)
    : T_AB_(T_AB), hpA_(hpA) {}

void InverseTransformPointJacobian::dhpB_dT_AB(
    Eigen::Matrix<double, 4, 6>* j) const {
  j->topLeftCorner<3, 3>() = T_AB_.C().transpose() * (-hpA_[3]);
  j->topRightCorner<3, 3>() =
      T_AB_.C().transpose() *
      okvis::kinematics::crossMx(hpA_.head<3>() - T_AB_.r() * hpA_[3]);
  j->row(3).setZero();
}

void InverseTransformPointJacobian::dhpB_dhpA(
    Eigen::Matrix<double, 4, 4>* j) const {
  *j = T_AB_.inverse().T();
}

Eigen::Vector4d InverseTransformPointJacobian::evaluate() const {
  return T_AB_.inverse() * hpA_;
}

}  // namespace swift_vio
#endif  // INCLUDE_SWIFT_VIO_INVERSE_TRANSFORM_POINT_JACOBIAN_HPP
