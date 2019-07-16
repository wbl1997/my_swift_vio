/**
 * @file ShapeMatrixParamBlock.cpp
 * @brief Source file for the ShapeMatrixParamBlock class.
 * @author Jianzhu Huai
 */

#include <okvis/ceres/ShapeMatrixParamBlock.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

// Default constructor (assumes not fixed).
ShapeMatrixParamBlock::ShapeMatrixParamBlock()
    : base_t::ParameterBlockSized() {
  setFixed(false);
}

// Trivial destructor.
ShapeMatrixParamBlock::~ShapeMatrixParamBlock() {
}

// Constructor with estimate and time.
ShapeMatrixParamBlock::ShapeMatrixParamBlock(
    const ShapeMatrixVector& shapeMatrixVector, uint64_t id,
    const okvis::Time& timestamp) {
  setEstimate(shapeMatrixVector);
  setId(id);
  setTimestamp(timestamp);
  setFixed(false);
}

// setters
// Set estimate of this parameter block.
void ShapeMatrixParamBlock::setEstimate(const ShapeMatrixVector& shapeMatrixVector) {
  for (int i = 0; i < base_t::Dimension; ++i)
    parameters_[i] = shapeMatrixVector[i];
}

// getters
// Get estimate.
ShapeMatrixVector ShapeMatrixParamBlock::estimate() const {
  ShapeMatrixVector shapeMatrixVector;
  for (int i = 0; i < base_t::Dimension; ++i)
    shapeMatrixVector[i] = parameters_[i];
  return shapeMatrixVector;
}

}  // namespace ceres
}  // namespace okvis
