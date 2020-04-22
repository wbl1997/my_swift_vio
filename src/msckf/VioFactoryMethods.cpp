#include <msckf/VioFactoryMethods.hpp>

#include <msckf/HybridFrontend.hpp>
#include <msckf/MSCKF2.hpp>
#include <msckf/InvariantEKF.hpp>
#include <msckf/SlidingWindowSmoother.hpp>
#include <msckf/TFVIO.hpp>
#include <msckf/GeneralEstimator.hpp>
#include <msckf/PriorlessEstimator.hpp>

#include <okvis/LoopClosureParameters.hpp>
#include <loop_closure/LoopClosureDetector.h>

namespace msckf {
std::shared_ptr<okvis::Frontend> createFrontend(
    int numCameras, bool initializeWithoutEnoughParallax,
    okvis::EstimatorAlgorithm algorithm) {
  switch (algorithm) {
    case okvis::EstimatorAlgorithm::General:
    case okvis::EstimatorAlgorithm::Priorless:
    case okvis::EstimatorAlgorithm::OKVIS:
      return std::shared_ptr<okvis::Frontend>(new okvis::Frontend(numCameras));
    default:
      break;
  }
  return std::shared_ptr<okvis::Frontend>(
      new okvis::HybridFrontend(numCameras, initializeWithoutEnoughParallax));
}

std::shared_ptr<okvis::Estimator> createBackend(okvis::EstimatorAlgorithm algorithm) {
  switch (algorithm) {
    // http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1049
    case okvis::EstimatorAlgorithm::General:
      return std::shared_ptr<okvis::Estimator>(new okvis::GeneralEstimator());

    case okvis::EstimatorAlgorithm::Priorless:
      return std::shared_ptr<okvis::Estimator>(new okvis::PriorlessEstimator());

    case okvis::EstimatorAlgorithm::OKVIS:
      return std::shared_ptr<okvis::Estimator>(new okvis::Estimator());

    case okvis::EstimatorAlgorithm::MSCKF:
      return std::shared_ptr<okvis::Estimator>(new okvis::MSCKF2());

    case okvis::EstimatorAlgorithm::TFVIO:
      return std::shared_ptr<okvis::Estimator>(new okvis::TFVIO());

    case okvis::EstimatorAlgorithm::InvariantEKF:
      return std::shared_ptr<okvis::Estimator>(new okvis::InvariantEKF());

    case okvis::EstimatorAlgorithm::SlidingWindowSmoother:
      return std::shared_ptr<okvis::Estimator>(new okvis::SlidingWindowSmoother());

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
