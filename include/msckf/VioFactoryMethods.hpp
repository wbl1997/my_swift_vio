#ifndef INCLUDE_MSCKF_VIOFACTORY_METHODS_HPP_
#define INCLUDE_MSCKF_VIOFACTORY_METHODS_HPP_

#include <okvis/Frontend.hpp>

#include <okvis/Estimator.hpp>

namespace msckf {
std::shared_ptr<okvis::Frontend> createFrontend(
    int numCameras, bool initializeWithoutEnoughParallax,
    okvis::EstimatorAlgorithm algorithm);

std::shared_ptr<okvis::Estimator> createBackend(okvis::EstimatorAlgorithm algorithm);

}  // namespace msckf
#endif // INCLUDE_MSCKF_VIOFACTORY_METHODS_HPP_
