#ifndef INCLUDE_MSCKF_VIOFACTORY_METHODS_HPP_
#define INCLUDE_MSCKF_VIOFACTORY_METHODS_HPP_

#include <okvis/Frontend.hpp>

#include <okvis/Estimator.hpp>

#include <okvis/LoopClosureMethod.hpp>
#include <okvis/LoopClosureParameters.hpp>

namespace msckf {
std::shared_ptr<okvis::Frontend> createFrontend(
    int numCameras, bool initializeWithoutEnoughParallax,
    okvis::EstimatorAlgorithm algorithm);

std::shared_ptr<okvis::Estimator> createBackend(okvis::EstimatorAlgorithm algorithm);

std::shared_ptr<okvis::LoopClosureMethod> createLoopClosureMethod(
    const okvis::LoopClosureParameters& lcParams);

}  // namespace msckf
#endif // INCLUDE_MSCKF_VIOFACTORY_METHODS_HPP_
