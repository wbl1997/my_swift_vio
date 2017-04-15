/**
 * @file CameraIntrinsicParamBlock.cpp
 * @brief Source file for the CameraIntrinsicParamBlock class.
 * @author Jianzhu Huai
 */

#include <okvis/ceres/CameraIntrinsicParamBlock.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

// Default constructor (assumes not fixed).
CameraIntrinsicParamBlock::CameraIntrinsicParamBlock()
    : base_t::ParameterBlockSized() {
  setFixed(false);
}

// Trivial destructor.
CameraIntrinsicParamBlock::~CameraIntrinsicParamBlock() {
}

// Constructor with estimate and time.
CameraIntrinsicParamBlock::CameraIntrinsicParamBlock(
    const IntrinsicParams& intrinsicParams, uint64_t id,
    const okvis::Time& timestamp) {
  setEstimate(intrinsicParams);
  setId(id);
  setTimestamp(timestamp);
  setFixed(false);
}

// setters
// Set estimate of this parameter block.
void CameraIntrinsicParamBlock::setEstimate(const IntrinsicParams& intrinsicParams) {
  for (int i = 0; i < base_t::Dimension; ++i)
    parameters_[i] = intrinsicParams[i];
}

// getters
// Get estimate.
IntrinsicParams CameraIntrinsicParamBlock::estimate() const {
  IntrinsicParams intrinsicParams;
  for (int i = 0; i < base_t::Dimension; ++i)
    intrinsicParams[i] = parameters_[i];
  return intrinsicParams;
}

}  // namespace ceres
}  // namespace okvis
