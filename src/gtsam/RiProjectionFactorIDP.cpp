/**
 * @file    RiProjectionFactorIDP.cpp
 * @brief   Reprojection factor with inverse depth parameterization at a frame
 * other than the anchor.
 * @author  Jianzhu Huai
 */

#include "gtsam/RiProjectionFactorIDP.h"
#include "gtsam/RiTransformPointJacobian.h"

#include "msckf/MultipleTransformPointJacobian.hpp"

namespace gtsam {
RiProjectionFactorIDP::RiProjectionFactorIDP(
    Key state_j, Key state_a, Key state_p,
    const Eigen::Matrix<double, 2, 2>& covariance, const Eigen::Vector2d& uv,
    std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry,
    const okvis::kinematics::Transformation& T_BCj,
    const okvis::kinematics::Transformation& T_BCa)
    : NoiseModelFactor3<RiExtendedPose3, RiExtendedPose3, Point3>(
          noiseModel::Gaussian::Covariance(covariance), state_j, state_a,
          state_p),
      uv_(uv),
      cameraGeometry_(cameraGeometry),
      T_BCj_(T_BCj),
      T_BCa_(T_BCa),
      exponentList_{-1, -1, 1, 1} {}

gtsam::Vector RiProjectionFactorIDP::evaluateError(
    const RiExtendedPose3& state_j, const RiExtendedPose3& state_a,
    const Point3& abrho, boost::optional<Matrix&> H1,
    boost::optional<Matrix&> H2, boost::optional<Matrix&> H3) const {
  Eigen::Vector2d error;
  Eigen::Matrix<double, 4, 1> ab1rho(abrho[0], abrho[1], 1, abrho[2]);

  AlignedVector<okvis::kinematics::Transformation> transformList(4);
  transformList[0] = T_BCj_;
  transformList[3] = T_BCa_;

  transformList[1] = okvis::kinematics::Transformation(
      state_j.position(), state_j.rotation().toQuaternion());
  transformList[2] = okvis::kinematics::Transformation(
      state_a.position(), state_a.rotation().toQuaternion());

  std::shared_ptr<okvis::TransformPointJacobian> transformPointObject(
      new gtsam::RiTransformPointJacobian());
  std::shared_ptr<okvis::InverseTransformPointJacobian>
      inverseTransformPointObject(new gtsam::RiInverseTransformPointJacobian());
  okvis::MultipleTransformPointJacobian mtpj(transformList, exponentList_,
                                             ab1rho, transformPointObject,
                                             inverseTransformPointObject);

  Eigen::Matrix<double, 4, 1> hpCj = mtpj.evaluate();
  Eigen::Vector2d imagePoint;
  Eigen::Matrix<double, 2, 4> Jp;
  okvis::cameras::CameraBase::ProjectionStatus status =
      cameraGeometry_->projectHomogeneous(hpCj, &imagePoint, &Jp);

  if (status == okvis::cameras::CameraBase::ProjectionStatus::Successful) {
    if (H1) {
      Eigen::Matrix<double, 2, 6> Hqp = Jp * mtpj.dp_dT(1);
      H1->resize(2, 9);
      H1->leftCols(3) = Hqp.leftCols(3);
      H1->middleCols(3, 3).setZero();
      H1->rightCols(3) = Hqp.rightCols(3);
    }
    if (H2) {
      Eigen::Matrix<double, 2, 6> Hqp = Jp * mtpj.dp_dT(2);
      H2->resize(2, 9);
      H2->leftCols(3) = Hqp.leftCols(3);
      H2->middleCols(3, 3).setZero();
      H2->rightCols(3) = Hqp.rightCols(3);
    }
    if (H3) {
      Eigen::Matrix<double, 2, 4> de_dhp = Jp * mtpj.dp_dpoint();
      H3->resize(2, 3);
      H3->leftCols(2) = de_dhp.leftCols(2);
      H3->col(2) = de_dhp.col(3);
    }
  } else {
    if (H1) *H1 = Eigen::Matrix<double, 2, 9>::Zero();
    if (H2) *H2 = Eigen::Matrix<double, 2, 9>::Zero();
    if (H3) *H3 = Eigen::Matrix<double, 2, 3>::Zero();
  }
  return imagePoint - uv_;
}
}  // namespace gtsam
