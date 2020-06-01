#ifndef INCLUDE_MSCKF_VIOFACTORY_METHODS_HPP_
#define INCLUDE_MSCKF_VIOFACTORY_METHODS_HPP_

#include <okvis/Frontend.hpp>

#include <okvis/Estimator.hpp>

#include <okvis/LoopClosureMethod.hpp>
#include <okvis/LoopClosureParameters.hpp>

#include <loop_closure/LoopClosureDetector.h>

namespace msckf {
std::shared_ptr<okvis::Frontend> createFrontend(
    int numCameras, const okvis::FrontendOptions& frontendOptions,
    okvis::EstimatorAlgorithm algorithm);

std::shared_ptr<okvis::Estimator> createBackend(okvis::EstimatorAlgorithm algorithm);

/**
 * @brief createLoopClosureMethod
 * @param lcParams
 * @return
 */
std::shared_ptr<okvis::LoopClosureMethod> createLoopClosureMethod(
    std::shared_ptr<VIO::LoopClosureDetectorParams> lcParams);

}  // namespace msckf
#endif // INCLUDE_MSCKF_VIOFACTORY_METHODS_HPP_
