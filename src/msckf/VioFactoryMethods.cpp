#include <msckf/VioFactoryMethods.hpp>

#include <msckf/HybridFrontend.hpp>
#include <msckf/MSCKF2.hpp>
#include <msckf/InvariantEKF.hpp>
#include <gtsam/SlidingWindowSmoother.hpp>
#include <msckf/TFVIO.hpp>
#include <msckf/GeneralEstimator.hpp>
#include <msckf/ConsistentEstimator.hpp>

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

std::shared_ptr<okvis::Estimator> createBackend(okvis::EstimatorAlgorithm algorithm,
                                                const okvis::BackendParams& backendParams) {
  switch (algorithm) {
    // http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1049
    case okvis::EstimatorAlgorithm::General:
      return std::shared_ptr<okvis::Estimator>(new okvis::GeneralEstimator());

    case okvis::EstimatorAlgorithm::Consistent:
      return std::shared_ptr<okvis::Estimator>(new okvis::ConsistentEstimator());

    case okvis::EstimatorAlgorithm::OKVIS:
      return std::shared_ptr<okvis::Estimator>(new okvis::Estimator());

    case okvis::EstimatorAlgorithm::MSCKF:
      return std::shared_ptr<okvis::Estimator>(new okvis::MSCKF2());

    case okvis::EstimatorAlgorithm::TFVIO:
      return std::shared_ptr<okvis::Estimator>(new okvis::TFVIO());

    case okvis::EstimatorAlgorithm::InvariantEKF:
      return std::shared_ptr<okvis::Estimator>(new okvis::InvariantEKF());

    case okvis::EstimatorAlgorithm::SlidingWindowSmoother:
      return std::shared_ptr<okvis::Estimator>(new okvis::SlidingWindowSmoother(backendParams));

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
