#include <msckf/SlidingWindowSmoother.hpp>

#include <glog/logging.h>

#include <okvis/assert_macros.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/RelativePoseError.hpp>
#include <okvis/ceres/SpeedAndBiasError.hpp>
#include <okvis/IdProvider.hpp>
#include <okvis/MultiFrame.hpp>

#include <msckf/EuclideanParamBlock.hpp>
#include <msckf/FeatureTriangulation.hpp>
#include <msckf/FilterHelper.hpp>

#include <msckf/MeasurementJacobianMacros.hpp>
#include <msckf/PointLandmark.hpp>
#include <msckf/PointLandmarkModels.hpp>
#include <msckf/PointSharedData.hpp>
#include <msckf/PreconditionedEkfUpdater.h>

#include <msckf/triangulate.h>
#include <msckf/triangulateFast.hpp>

DEFINE_bool(use_combined_imu_factor, false,
            "CombinedImuFactor(PreintegratedCombinedMeasurement) or "
            "ImuFactor(PreintegratedImuMeasurement)");
DECLARE_bool(use_mahalanobis);
DECLARE_bool(use_first_estimate);
DECLARE_bool(use_RK4);

DECLARE_double(max_proj_tolerance);

DECLARE_bool(use_IEKF);

/// \brief okvis Main namespace of this package.
namespace okvis {

SlidingWindowSmoother::SlidingWindowSmoother(
    std::shared_ptr<okvis::ceres::Map> mapPtr)
    : Estimator(mapPtr) {}

// The default constructor.
SlidingWindowSmoother::SlidingWindowSmoother() {}

SlidingWindowSmoother::~SlidingWindowSmoother() {}

// add states to the factor graph, nav states, biases, record their ids in
// statesMap_
bool SlidingWindowSmoother::addStates(
    okvis::MultiFramePtr /*multiFrame*/,
    const okvis::ImuMeasurementDeque& /*imuMeasurements*/,
    bool /*asKeyframe*/) {
  return false;
}

// the major job of marginalization is done in factorgraph optimization step
// here we only remove old landmarks and states from bookkeeping.
bool SlidingWindowSmoother::applyMarginalizationStrategy(
    size_t /*numKeyframes*/, size_t /*numImuFrames*/,
    okvis::MapPointVector& /*removedLandmarks*/) {
  return false;
}

// optimize the factor graph with new values, factors
void SlidingWindowSmoother::optimize(size_t /*numIter*/, size_t /*numThreads*/,
                                     bool /*verbose*/) {}

}  // namespace okvis
