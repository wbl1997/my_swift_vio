#include <msckf/VioFactoryMethods.hpp>

#include <msckf/ConsistentEstimator.hpp>
#include <msckf/GeneralEstimator.hpp>
#include <msckf/HybridFrontend.hpp>
#include <msckf/MSCKF2.hpp>
#include <msckf/InvariantEKF.hpp>
#include <gtsam/RiSlidingWindowSmoother.hpp>
#include <gtsam/SlidingWindowSmoother.hpp>
#include <msckf/TFVIO.hpp>

#include <loop_closure/LoopClosureDetector.h>

namespace msckf {
std::shared_ptr<okvis::Frontend> createFrontend(
    int numCameras, const okvis::FrontendOptions& frontendOptions,
    okvis::EstimatorAlgorithm algorithm) {
  switch (algorithm) {
    case okvis::EstimatorAlgorithm::General:
    case okvis::EstimatorAlgorithm::Consistent:
    case okvis::EstimatorAlgorithm::OKVIS:
      return std::shared_ptr<okvis::Frontend>(
          new okvis::Frontend(numCameras, frontendOptions));
    default:
      break;
  }
  return std::shared_ptr<okvis::Frontend>(
      new okvis::HybridFrontend(numCameras, frontendOptions));
}

std::shared_ptr<okvis::Estimator> createBackend(
    okvis::EstimatorAlgorithm algorithm,
    const okvis::BackendParams& backendParams,
    std::shared_ptr<okvis::ceres::Map> mapPtr) {
  switch (algorithm) {
    // we do not use make_shared because it may interfere with alignment, see
    // http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1049
    case okvis::EstimatorAlgorithm::OKVIS:
      return std::shared_ptr<okvis::Estimator>(new okvis::Estimator(mapPtr));

    case okvis::EstimatorAlgorithm::General:
      return std::shared_ptr<okvis::Estimator>(
          new okvis::GeneralEstimator(mapPtr));

    case okvis::EstimatorAlgorithm::Consistent:
      return std::shared_ptr<okvis::Estimator>(
          new okvis::ConsistentEstimator(mapPtr));

    case okvis::EstimatorAlgorithm::SlidingWindowSmoother:
      return std::shared_ptr<okvis::Estimator>(
          new okvis::SlidingWindowSmoother(backendParams, mapPtr));

    case okvis::EstimatorAlgorithm::RiSlidingWindowSmoother:
      return std::shared_ptr<okvis::Estimator>(
          new okvis::RiSlidingWindowSmoother(backendParams, mapPtr));

    case okvis::EstimatorAlgorithm::HybridFilter:
      return std::shared_ptr<okvis::Estimator>(new okvis::HybridFilter(mapPtr));

    case okvis::EstimatorAlgorithm::MSCKF:
      return std::shared_ptr<okvis::Estimator>(new okvis::MSCKF2(mapPtr));

    case okvis::EstimatorAlgorithm::TFVIO:
      return std::shared_ptr<okvis::Estimator>(new okvis::TFVIO(mapPtr));

    case okvis::EstimatorAlgorithm::InvariantEKF:
      return std::shared_ptr<okvis::Estimator>(new okvis::InvariantEKF(mapPtr));

    default:
      LOG(ERROR) << "Unknown Estimator type!";
      break;
  }
  return std::shared_ptr<okvis::Estimator>();
}

std::shared_ptr<okvis::LoopClosureMethod> createLoopClosureMethod(
    std::shared_ptr<VIO::LoopClosureDetectorParams> lcParams) {
  switch (lcParams->loop_closure_method_) {
    case VIO::LoopClosureMethodType::OrbBoW:
      return std::shared_ptr<okvis::LoopClosureMethod>(
          new VIO::LoopClosureDetector(lcParams));
    default:
      return std::shared_ptr<okvis::LoopClosureMethod>(
          new okvis::LoopClosureMethod());
  }
}

}  // namespace msckf
