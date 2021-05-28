#include <swift_vio/VioFactoryMethods.hpp>

#include <swift_vio/HybridFrontend.hpp>
#include <swift_vio/MSCKF.hpp>
#include <gtsam/RiSlidingWindowSmoother.hpp>
#include <gtsam/SlidingWindowSmoother.hpp>
#include <swift_vio/TFVIO.hpp>

#include <loop_closure/LoopClosureDetector.h>

namespace swift_vio {
std::shared_ptr<okvis::Frontend> createFrontend(
    int numCameras, const FrontendOptions& frontendOptions,
    EstimatorAlgorithm algorithm) {
  switch (algorithm) {
    case EstimatorAlgorithm::OKVIS:
      return std::shared_ptr<okvis::Frontend>(
          new okvis::Frontend(numCameras, frontendOptions));
    default:
      break;
  }
  return std::shared_ptr<okvis::Frontend>(
      new HybridFrontend(numCameras, frontendOptions));
}

std::shared_ptr<okvis::Estimator> createBackend(
    EstimatorAlgorithm algorithm,
    const BackendParams& backendParams,
    std::shared_ptr<okvis::ceres::Map> mapPtr) {
  switch (algorithm) {
    // we do not use make_shared because it may interfere with alignment, see
    // http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1049
    case EstimatorAlgorithm::OKVIS:
      return std::shared_ptr<okvis::Estimator>(new okvis::Estimator(mapPtr));

    case EstimatorAlgorithm::SlidingWindowSmoother:
      return std::shared_ptr<okvis::Estimator>(
          new SlidingWindowSmoother(backendParams, mapPtr));

    case EstimatorAlgorithm::RiSlidingWindowSmoother:
      return std::shared_ptr<okvis::Estimator>(
          new RiSlidingWindowSmoother(backendParams, mapPtr));

    case EstimatorAlgorithm::HybridFilter:
      return std::shared_ptr<okvis::Estimator>(new HybridFilter(mapPtr));

    case EstimatorAlgorithm::MSCKF:
      return std::shared_ptr<okvis::Estimator>(new MSCKF(mapPtr));

    case EstimatorAlgorithm::TFVIO:
      return std::shared_ptr<okvis::Estimator>(new TFVIO(mapPtr));

    case EstimatorAlgorithm::InvariantEKF:
      return std::shared_ptr<okvis::Estimator>(new MSCKF(mapPtr));

    default:
      LOG(ERROR) << "Unknown Estimator type!";
      break;
  }
  return std::shared_ptr<okvis::Estimator>();
}

std::shared_ptr<LoopClosureMethod> createLoopClosureMethod(
    std::shared_ptr<LoopClosureDetectorParams> lcParams) {
  switch (lcParams->loop_closure_method_) {
    case LoopClosureMethodType::OrbBoW:
      return std::shared_ptr<LoopClosureMethod>(
          new LoopClosureDetector(lcParams));
    default:
      return std::shared_ptr<LoopClosureMethod>(
          new LoopClosureMethod());
  }
}

}  // namespace swift_vio
