/**
 * @file   RiTransformPointJacobian.h
 * @brief  Jacobians for T * p, T^{-1} * p, and T_1^{a_1} * ... * T_n^{a_2} * p.
 * where p is a 4D homogeneous point, and T_i = Expmap(xi) \hat{T}_i.
 * @author Jianzhu Huai
 */

#ifndef RITRANSFORMPOINTJACOBIAN_H
#define RITRANSFORMPOINTJACOBIAN_H

#include <swift_vio/InverseTransformPointJacobian.hpp>
#include <swift_vio/TransformPointJacobian.hpp>

namespace gtsam {
class RiTransformPointJacobian : public swift_vio::TransformPointJacobian {
 public:
  RiTransformPointJacobian() : swift_vio::TransformPointJacobian() {}

  virtual ~RiTransformPointJacobian() {}

  RiTransformPointJacobian(const okvis::kinematics::Transformation& T_AB,
                           const Eigen::Vector4d& hpB)
      : swift_vio::TransformPointJacobian(T_AB, hpB) {}

  /**
   * @brief dhpA_dT_AB \xi = [\phi, \delta t]
   * @param j
   */
  void dhpA_dT_AB(Eigen::Matrix<double, 4, 6>* j) const override {
    Eigen::Vector3d wpA = T_AB_.C() * hpB_.head<3>() + T_AB_.r() * hpB_[3];
    j->topLeftCorner<3, 3>() = okvis::kinematics::crossMx(-wpA);

    Eigen::Vector3d w3 = Eigen::Vector3d::Constant(hpB_[3]);
    j->topRightCorner<3, 3>() = w3.asDiagonal();

    j->row(3).setZero();
  }
};

class RiInverseTransformPointJacobian
    : public swift_vio::InverseTransformPointJacobian {
 public:
  RiInverseTransformPointJacobian() : swift_vio::InverseTransformPointJacobian() {}

  virtual ~RiInverseTransformPointJacobian() {}

  RiInverseTransformPointJacobian(const okvis::kinematics::Transformation& T_AB,
                                  const Eigen::Vector4d& hpA)
      : swift_vio::InverseTransformPointJacobian(T_AB, hpA) {}

  void dhpB_dT_AB(Eigen::Matrix<double, 4, 6>* j) const override {
    j->topLeftCorner<3, 3>().noalias() =
        T_AB_.C().transpose() * okvis::kinematics::crossMx(hpA_.head<3>());
    j->topRightCorner<3, 3>().noalias() = T_AB_.C().transpose() * (-hpA_[3]);
    j->row(3).setZero();
  }
};
}  // namespace gtsam

#endif // RITRANSFORMPOINTJACOBIAN_H
