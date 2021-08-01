/**
 * @file   TransformPointJacobian.h
 * @brief  Jacobians for T * p.
 * where p is a 4D homogeneous point,
 * T = [Expmap(\alpha) * R   t + \delta t;
 *       0^T                 1].
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_SWIFT_VIO_TRANSFORM_POINT_JACOBIAN_HPP
#define INCLUDE_SWIFT_VIO_TRANSFORM_POINT_JACOBIAN_HPP

#include <okvis/kinematics/Transformation.hpp>

namespace swift_vio {
class TransformPointJacobian {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  inline TransformPointJacobian();

  inline virtual ~TransformPointJacobian();

  inline void initialize(const okvis::kinematics::Transformation& T_AB,
                         const Eigen::Vector4d& hpB);

  inline TransformPointJacobian(const okvis::kinematics::Transformation& T_AB,
                                const Eigen::Vector4d& hpB);

  inline virtual void dhpA_dT_AB(Eigen::Matrix<double, 4, 6>* j) const;

  inline void dhpA_dhpB(Eigen::Matrix<double, 4, 4>* j) const;

  inline Eigen::Vector4d evaluate() const;

 protected:
  okvis::kinematics::Transformation T_AB_;
  Eigen::Vector4d hpB_;
};

TransformPointJacobian::TransformPointJacobian() {}

TransformPointJacobian::~TransformPointJacobian() {}

void TransformPointJacobian::initialize(
    const okvis::kinematics::Transformation& T_AB, const Eigen::Vector4d& hpB) {
  T_AB_ = T_AB;
  hpB_ = hpB;
}

TransformPointJacobian::TransformPointJacobian(
    const okvis::kinematics::Transformation& T_AB, const Eigen::Vector4d& hpB)
    : T_AB_(T_AB), hpB_(hpB) {}

void TransformPointJacobian::dhpA_dT_AB(Eigen::Matrix<double, 4, 6>* j) const {
  Eigen::Vector3d w3;
  w3.setConstant(hpB_[3]);
  j->topLeftCorner<3, 3>() = w3.asDiagonal();
  j->topRightCorner<3, 3>() =
      okvis::kinematics::crossMx(T_AB_.C() * (-hpB_.head<3>()));
  j->row(3).setZero();
}

void TransformPointJacobian::dhpA_dhpB(Eigen::Matrix<double, 4, 4>* j) const {
  *j = T_AB_.T();
}

Eigen::Vector4d TransformPointJacobian::evaluate() const {
  return T_AB_ * hpB_;
}

}  // namespace swift_vio
#endif  // INCLUDE_SWIFT_VIO_TRANSFORM_POINT_JACOBIAN_HPP
