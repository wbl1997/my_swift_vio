/**
 * @file    RiProjectionFactorIDPAnchor.cpp
 * @brief   Reprojection factor with inverse depth parameterization at the
 * anchor frame.
 * @author  Jianzhu Huai
 */

#include "gtsam/RiProjectionFactorIDPAnchor.h"
#include "okvis/cameras/PinholeCamera.hpp"

namespace gtsam {
gtsam::Vector RiProjectionFactorIDPAnchor::evaluateError(
    const Point3& state_p, boost::optional<Matrix&> H1) const {
  const Eigen::Vector3d& abrho = state_p;
  Eigen::Matrix<double, 4, 1> ab1rho(abrho[0], abrho[1], 1, abrho[2]);

  okvis::kinematics::Transformation T_CjCa = T_BCj_.inverse() * T_BCa_;
  Eigen::Matrix<double, 4, 1> hpCj = T_CjCa * ab1rho;

  Eigen::Vector2d uvProjected;

  Eigen::Matrix<double, 2, 4> Jp;
  okvis::cameras::CameraBase::ProjectionStatus status;
  if (H1) {
    status = cameraGeometry_->projectHomogeneous(hpCj, &uvProjected, &Jp);
    if (status == okvis::cameras::CameraBase::ProjectionStatus::Successful) {
      Eigen::Matrix<double, 2, 4> Jfull = Jp * T_CjCa.T();
      H1->resize(2, 3);
      H1->leftCols(2) = Jfull.leftCols(2);
      H1->col(2) = Jfull.col(3);
    } else {
      *H1 = Eigen::Matrix<double, 2, 3>::Zero();
    }
  } else {
    status = cameraGeometry_->projectHomogeneous(hpCj, &uvProjected);
  }
  return uvProjected - uv_;
}
}  // namespace gtsam
