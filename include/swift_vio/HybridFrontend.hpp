
#ifndef INCLUDE_OKVIS_HYBRID_FRONTEND_HPP_
#define INCLUDE_OKVIS_HYBRID_FRONTEND_HPP_

#include <mutex>
#include <okvis/DenseMatcher.hpp>
#include <okvis/Frontend.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/timing/Timer.hpp>

#include <feature_tracker/FeatureTracker.h>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/density.hpp>
#include <boost/accumulators/statistics/stats.hpp>

namespace swift_vio {
/**
 * @brief A frontend that uses BRISK descriptor based matching or
 * KLT feature tracking.
 */
class HybridFrontend : public okvis::Frontend {
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

#ifdef DEACTIVATE_TIMERS
  typedef okvis::timing::DummyTimer TimerSwitchable;
#else
  typedef okvis::timing::Timer TimerSwitchable;
#endif

  /**
   * @brief Constructor.
   * @param numCameras Number of cameras in the sensor configuration.
   */
  HybridFrontend(size_t numCameras, const FrontendOptions& frontendOptions);
  virtual ~HybridFrontend() {}

  ///@{
 

  /**
   * @brief Matching as well as initialization of landmarks and state.
   * @warning This method is not threadsafe.
   * @warning This method uses the estimator. Make sure to not access it in
   * another thread.
   * @param estimator       Estimator.
   * @param T_WS_propagated Pose of sensor at image capture time.
   * @param params          Configuration parameters.
   * @param map             Unused.
   * @param framesInOut     Multiframe including the descriptors of all the
   * keypoints.
   * @param[out] asKeyframe Should the frame be a keyframe?
   * @return True if successful.
   */
  virtual bool dataAssociationAndInitialization(
      okvis::Estimator& estimator,
      okvis::kinematics::Transformation& T_WS_propagated,
      const okvis::VioParameters& params,
      const std::shared_ptr<okvis::MapPointVector> map,
      std::shared_ptr<okvis::MultiFrame> framesInOut, bool* asKeyframe);

 
  ///@}
 
  virtual bool isDescriptorBasedMatching() const;

  virtual void setLandmarkTriangulationParameters(
      double triangulationTranslationThreshold,
      double triangulationMaxDepth) final;

 private:
  feature_tracker::FeatureTracker featureTracker_;

  /**
   * @brief Decision whether a new frame should be keyframe or not after the
   * currentFrame has matched to its previous frame, by examining common
   * landmarks between currentFrame and latest keyframe.
   * @param estimator     const reference to the estimator.
   * @param currentFrame  Keyframe candidate.
   * @return True if it should be a new keyframe.
   */
  bool doWeNeedANewKeyframePosterior(
      const okvis::Estimator& estimator,
      std::shared_ptr<okvis::MultiFrame> currentFrame);



  template <class CAMERA_GEOMETRY_T>
  int matchToLastFrameKLT(okvis::Estimator& estimator,
                          const okvis::VioParameters& params,
                          std::shared_ptr<okvis::MultiFrame> framesInOut,
                          bool& rotationOnly, bool usePoseUncertainty = true,
                          bool removeOutliers = true);

  /**
   * @brief Match a new multiframe to the last frame.
   * @tparam MATCHING_ALGORITHM Algorithm to match new keypoints to existing
   * landmarks
   * @warning As this function uses the estimator it is not threadsafe.
   * @param estimator           HybridFilter.
   * @param params              Parameter struct.
   * @param currentFrameId      ID of the current frame that should be matched
   * against the last one.
   * @param usePoseUncertainty  Use the pose uncertainty for the matching.
   * @param removeOutliers      Remove outliers during RANSAC.
   * @return The number of matches in total.
   */
  template <class MATCHING_ALGORITHM>
  int matchToLastFrame(
      okvis::Estimator& estimator,
      const okvis::VioParameters& params, const uint64_t currentFrameId,
      bool& rotationOnly,
      bool usePoseUncertainty = true, bool removeOutliers = true);



  /**
   * @brief Check the relative motion between current frame and an older frame,
   * and record the relative motion in the estimator. If overlap area or
   * matching ratio in the overlap area between the two frames does not meet
   * thresholds, the current frame will be chosen as a keyframe, i.e.,
   * asKeyframe is set true.
   * @param estimator
   * @param params
   * @param currentFrameId
   * @param olderFrameId
   * @param removeOutliers
   * @param asKeyframe
   * @return
   */
  int checkMotionByRansac2d2d(okvis::Estimator& estimator,
                              const okvis::VioParameters& params,
                              uint64_t currentFrameId, uint64_t olderFrameId,
                              bool removeOutliers, bool* asKeyframe);
};

}  // namespace swift_vio
#endif  // INCLUDE_OKVIS_HYBRID_FRONTEND_HPP_
