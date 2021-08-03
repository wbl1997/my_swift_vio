#ifndef INCLUDE_SWIFT_VIO_VIOFACTORY_METHODS_HPP_
#define INCLUDE_SWIFT_VIO_VIOFACTORY_METHODS_HPP_

#include <okvis/Frontend.hpp>

#include <okvis/Estimator.hpp>

#include <loop_closure/LoopClosureMethod.hpp>
#include <loop_closure/LoopClosureParameters.hpp>
#include <gtsam/VioBackEndParams.h>

#include <loop_closure/LoopClosureDetectorParams.h>

namespace swift_vio {
std::shared_ptr<okvis::Frontend> createFrontend(
    int numCameras, const FrontendOptions& frontendOptions,
    EstimatorAlgorithm algorithm);

std::shared_ptr<okvis::Estimator> createBackend(
    EstimatorAlgorithm algorithm,
    const BackendParams& backendParams,
    std::shared_ptr<okvis::ceres::Map> mapPtr);

/**
 * @brief createLoopClosureMethod
 * @param lcParams
 * @return
 */
std::shared_ptr<LoopClosureMethod> createLoopClosureMethod(
    std::shared_ptr<LoopClosureDetectorParams> lcParams);

}  // namespace swift_vio
#endif // INCLUDE_SWIFT_VIO_VIOFACTORY_METHODS_HPP_
