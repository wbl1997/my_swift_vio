#ifndef INCLUDE_OKVIS_SLIDING_WINDOW_SMOOTHER_HPP_
#define INCLUDE_OKVIS_SLIDING_WINDOW_SMOOTHER_HPP_

#include <array>
#include <memory>
//#include <mutex>

#include <gflags/gflags.h>

//#include <ceres/ceres.h>
//#include <okvis/kinematics/Transformation.hpp>

#include <okvis/Estimator.hpp>
//#include <okvis/FrameTypedefs.hpp>
//#include <okvis/Measurements.hpp>
//#include <okvis/MultiFrame.hpp>
//#include <okvis/Variables.hpp>
//#include <okvis/assert_macros.hpp>
//#include <okvis/ceres/CeresIterationCallback.hpp>
//#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>
//#include <okvis/ceres/Map.hpp>
//#include <okvis/ceres/MarginalizationError.hpp>
//#include <okvis/ceres/PoseParameterBlock.hpp>
//#include <okvis/ceres/ReprojectionError.hpp>
//#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>

#define INCREMENTAL_SMOOTHER

#ifdef HAVE_GTSAM
#include <gtsam_unstable/nonlinear/BatchFixedLagSmoother.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtsam_unstable/slam/SmartStereoProjectionPoseFactor.h>
#else
#include <msckf/MockSmoother.hpp>
#endif

/// \brief okvis Main namespace of this package.
namespace okvis {
#ifdef HAVE_GTSAM
  #ifdef INCREMENTAL_SMOOTHER
    typedef gtsam::IncrementalFixedLagSmoother Smoother;
  #else
    typedef gtsam::BatchFixedLagSmoother Smoother;
  #endif
#else
  typedef msckf::MockSmoother Smoother;
#endif 

/**
 * SlidingWindowSmoother builds upon gtsam FixedLagSmoother.
 */
class SlidingWindowSmoother : public Estimator {
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief The default constructor.
   */
  SlidingWindowSmoother();

  /**
   * @brief Constructor if a ceres map is already available.
   * @param mapPtr Shared pointer to ceres map.
   */
  SlidingWindowSmoother(std::shared_ptr<okvis::ceres::Map> mapPtr);

  virtual ~SlidingWindowSmoother();

  /**
   * @brief add a state to the state map
   * @param multiFrame Matched multiFrame.
   * @param imuMeasurements IMU measurements from last state to new one.
   * imuMeasurements covers at least the current state and the last state in
   * time, with an extension on both sides.
   * @param asKeyframe Is this new frame a keyframe?
   * @return True if successful.
   * If it is the first state, initialize it and the covariance matrix. In
   * initialization, please make sure the world frame has z axis in negative
   * gravity direction which is assumed in the IMU propagation Only one IMU is
   * supported for now
   */
  virtual bool addStates(okvis::MultiFramePtr multiFrame,
                         const okvis::ImuMeasurementDeque &imuMeasurements,
                         bool asKeyframe) final;

  virtual void optimize(size_t numIter, size_t numThreads = 1,
                        bool verbose = false) final;

  /**
   * @brief Applies the dropping/marginalization strategy according to the
   * RSS'13/IJRR'14 paper. The new number of frames in the window will be
   * numKeyframes+numImuFrames.
   * @return True if successful.
   */
  virtual bool applyMarginalizationStrategy(
      size_t numKeyframes, size_t numImuFrames,
      okvis::MapPointVector& removedLandmarks) final;

private:
  std::shared_ptr<Smoother> smoother_;
};

}  // namespace okvis

#endif /* INCLUDE_OKVIS_SLIDING_WINDOW_SMOOTHER_HPP_ */
