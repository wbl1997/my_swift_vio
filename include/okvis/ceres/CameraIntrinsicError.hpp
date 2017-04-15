/**
 * @file CameraIntrinsicError.hpp
 * @brief Header file for the CameraIntrinsicError class.
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_OKVIS_CERES_CAMERAINTRINSICERROR_HPP_
#define INCLUDE_OKVIS_CERES_CAMERAINTRINSICERROR_HPP_

#include <vector>
#include <Eigen/Core>
#include "ceres/ceres.h"
#include <okvis/assert_macros.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/ceres/ErrorInterface.hpp>
#include <okvis/ceres/CameraIntrinsicParamBlock.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

class CameraIntrinsicError : public ::ceres::SizedCostFunction<
    4 /* number of residuals */,
    4 /* size of first parameter */>,
    public ErrorInterface {
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief The base class type.
  typedef ::ceres::SizedCostFunction<4, 4> base_t;

  /// \brief Number of residuals (4)
  static const int kNumResiduals = 4;

  /// \brief The information matrix type (9x9).
  typedef Eigen::Matrix<double, 4, 4> information_t;

  /// \brief The covariance matrix type (same as information).
  typedef Eigen::Matrix<double, 4, 4> covariance_t;

  /// \brief Default constructor.
  CameraIntrinsicError();

  /// \brief Construct with measurement and information matrix
  /// @param[in] measurement The measurement.
  /// @param[in] information The information (weight) matrix.
  CameraIntrinsicError(const okvis::ceres::IntrinsicParams & measurement,
                    const information_t & information);

  /// \brief Construct with measurement and variance.
  /// @param[in] measurement The measurement.
  /// @param[in] focalLengthVariance The variance of the focalLength measurement, i.e. information_ has variance in its diagonal.
  /// @param[in] ppVariance The variance of the principal point measurement, i.e. information_ has variance in its diagonal.
  CameraIntrinsicError(const okvis::ceres::IntrinsicParams& measurement,
                    double focalLengthVariance, double ppVariance);

  /// \brief Trivial destructor.
  virtual ~CameraIntrinsicError() {
  }

  // setters
  /// \brief Set the measurement.
  /// @param[in] measurement The measurement.
  void setMeasurement(const okvis::ceres::IntrinsicParams & measurement) {
    measurement_ = measurement;
  }

  /// \brief Set the information.
  /// @param[in] information The information (weight) matrix.
  void setInformation(const information_t & information);

  // getters
  /// \brief Get the measurement.
  /// \return The measurement vector.
  const okvis::ceres::IntrinsicParams& measurement() const {
    return measurement_;
  }

  /// \brief Get the information matrix.
  /// \return The information (weight) matrix.
  const information_t& information() const {
    return information_;
  }

  /// \brief Get the covariance matrix.
  /// \return The inverse information (covariance) matrix.
  const covariance_t& covariance() const {
    return covariance_;
  }

  // error term and Jacobian implementation
  /**
   * @brief This evaluates the error term and additionally computes the Jacobians.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @return success of th evaluation.
   */
  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const;

  /**
   * @brief This evaluates the error term and additionally computes
   *        the Jacobians in the minimal internal representation.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @param jacobiansMinimal Pointer to the minimal Jacobians (equivalent to jacobians).
   * @return Success of the evaluation.
   */
  virtual bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                            double* residuals,
                                            double** jacobians,
                                            double** jacobiansMinimal) const;

  // sizes
  /// \brief Residual dimension.
  size_t residualDim() const {
    return kNumResiduals;
  }

  /// \brief Number of parameter blocks.
  size_t parameterBlocks() const {
    return parameter_block_sizes().size();
  }

  /// \brief Dimension of an individual parameter block.
  /// @param[in] parameterBlockId ID of the parameter block of interest.
  /// \return The dimension.
  size_t parameterBlockDim(size_t parameterBlockId) const {
    return base_t::parameter_block_sizes().at(parameterBlockId);
  }

  /// @brief Residual block type as string
  virtual std::string typeInfo() const {
    return "CameraIntrinsicError";
  }

 protected:

  // the measurement
  okvis::ceres::IntrinsicParams measurement_; ///< The (9D) measurement.

  // weighting related
  information_t information_; ///< The 9x9 information matrix.
  information_t squareRootInformation_; ///< The 9x9 square root information matrix.
  covariance_t covariance_; ///< The 9x9 covariance matrix.

};

}  // namespace ceres
}  // namespace okvis

#endif /* INCLUDE_OKVIS_CERES_CAMERAINTRINSICERROR_HPP_ */
