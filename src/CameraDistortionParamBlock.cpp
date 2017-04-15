/**
 * @file CameraDistortionParamBlock.cpp
 * @brief Source file for the CameraDistortionParamBlock class.
 * @author Jianzhu Huai
 */

#include <okvis/ceres/CameraDistortionParamBlock.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

// Default constructor (assumes not fixed).
CameraDistortionParamBlock::CameraDistortionParamBlock()
    : base_t::ParameterBlockSized() {
  setFixed(false);
}

// Trivial destructor.
CameraDistortionParamBlock::~CameraDistortionParamBlock() {
}

// Constructor with estimate and time.
CameraDistortionParamBlock::CameraDistortionParamBlock(
    const DistortionCoeffs& distortionCoeffs, uint64_t id,
    const okvis::Time& timestamp) {
  setEstimate(distortionCoeffs);
  setId(id);
  setTimestamp(timestamp);
  setFixed(false);
}

// setters
// Set estimate of this parameter block.
void CameraDistortionParamBlock::setEstimate(const DistortionCoeffs& distortionCoeffs) {
  for (int i = 0; i < base_t::Dimension; ++i)
    parameters_[i] = distortionCoeffs[i];
}

// getters
// Get estimate.
DistortionCoeffs CameraDistortionParamBlock::estimate() const {
  DistortionCoeffs distortionCoeffs;
  for (int i = 0; i < base_t::Dimension; ++i)
    distortionCoeffs[i] = parameters_[i];
  return distortionCoeffs;
}

}  // namespace ceres
}  // namespace okvis
