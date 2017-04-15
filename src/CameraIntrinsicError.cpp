/**
 * @file CameraIntrinsicError.cpp
 * @brief Source file for the CameraIntrinsicError class.
 * @author Jianzhu Huai
 */

#include <okvis/ceres/CameraIntrinsicError.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

// Construct with measurement and information matrix
CameraIntrinsicError::CameraIntrinsicError(const okvis::ceres::IntrinsicParams & measurement,
                                     const information_t & information) {
  setMeasurement(measurement);
  setInformation(information);
}

// Construct with measurement and variance.
CameraIntrinsicError::CameraIntrinsicError(const okvis::ceres::IntrinsicParams& measurement,
                                     double focalLengthVariance,
                                     double ppVariance) {
  setMeasurement(measurement);

  information_t information;
  information.setZero();
  information.topLeftCorner<2, 2>() = Eigen::Matrix2d::Identity() * 1.0
      / focalLengthVariance;
  information.block<2, 2>(2, 2) = Eigen::Matrix2d::Identity() * 1.0
      / ppVariance;

  setInformation(information);
}

// Set the information.
void CameraIntrinsicError::setInformation(const information_t & information) {
  information_ = information;
  covariance_ = information.inverse();
  // perform the Cholesky decomposition on order to obtain the correct error weighting
  Eigen::LLT<information_t> lltOfInformation(information_);
  squareRootInformation_ = lltOfInformation.matrixL().transpose();
}

// This evaluates the error term and additionally computes the Jacobians.
bool CameraIntrinsicError::Evaluate(double const* const * parameters,
                                 double* residuals, double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation.
bool CameraIntrinsicError::EvaluateWithMinimalJacobians(
    double const* const * parameters, double* residuals, double** jacobians,
    double** jacobiansMinimal) const {

  // compute error
  Eigen::Map<const okvis::ceres::IntrinsicParams> estimate(parameters[0]);
  okvis::ceres::IntrinsicParams error = measurement_ - estimate;

  // weigh it
  Eigen::Map<Eigen::Matrix<double, 4, 1> > weighted_error(residuals);
  weighted_error = squareRootInformation_ * error;

  // compute Jacobian - this is rather trivial in this case...
  if (jacobians != NULL) {
    if (jacobians[0] != NULL) {
      Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor> > J0(
          jacobians[0]);
      J0 = -squareRootInformation_ * Eigen::Matrix<double, 4, 4>::Identity();
    }
  }
  if (jacobiansMinimal != NULL) {
    if (jacobiansMinimal[0] != NULL) {
      Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor> > J0min(
          jacobiansMinimal[0]);
      J0min = -squareRootInformation_ * Eigen::Matrix<double, 4, 4>::Identity();
    }
  }

  return true;
}

}  // namespace ceres
}  // namespace okvis
