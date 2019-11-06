
#ifndef INCLUDE_OKVIS_SIMULATION_FRONTEND_HPP_
#define INCLUDE_OKVIS_SIMULATION_FRONTEND_HPP_

#include <mutex>
#include <okvis/DenseMatcher.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/timing/Timer.hpp>

#include <okvis/Estimator.hpp>

#include <feature_tracker/feature_tracker.h>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/density.hpp>
#include <boost/accumulators/statistics/stats.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

/**
 * @brief A frontend using BRISK features
 */
class SimulationFrontend {
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)


  /**
   * @brief Constructor.
   * @param numCameras Number of cameras in the sensor configuration.
   */
  SimulationFrontend(size_t numCameras, bool addImageNoise, int maxTrackLength,
                     std::string pointFile);
  virtual ~SimulationFrontend() {}

  ///@{
  
  /**
   * @brief Matching as well as initialization of landmarks and state.
   * @warning This method is not threadsafe.
   * @warning This method uses the estimator. Make sure to not access it in
   * another thread.
   * @param estimator
   * @param T_WS_propagated Pose of sensor at image capture time.
   * @param params          Configuration parameters.
   * @param map             Unused.
   * @param framesInOut     Multiframe including the descriptors of all the
   * keypoints.
   * @param[out] asKeyframe Should the frame be a keyframe?
   * @return True if successful.
   */
  int dataAssociationAndInitialization(
      okvis::Estimator& estimator, okvis::kinematics::Transformation& T_WS_ref,
      std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystemRef,
      std::shared_ptr<okvis::MultiFrame> framesInOut, bool* asKeyframe);

  ///@}


  /// @name Other getters
  /// @{

  /// @brief Returns true if the initialization has been completed (RANSAC with actual translation)
  bool isInitialized() {
    return isInitialized_;
  }

  /// @}
  

  // output the distribution of number of features in images
  void printNumFeatureDistribution(std::ofstream& stream) const;


 private:

  bool isInitialized_;       ///< Is the pose initialised?
  const size_t numCameras_;  ///< Number of cameras in the configuration.
  bool addImageNoise_; ///< Add noise to image observations
  int maxTrackLength_; ///< Cap feature track length
  // used to check if a keyframe is needed
  std::shared_ptr<okvis::MultiFrame> previousKeyFrame;
  okvis::kinematics::Transformation previousKeyFramePose;

  // scene landmarks
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
      homogeneousPoints_;
  std::vector<uint64_t> lmIds_;
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
      noisyHomogeneousPoints_;
  static const double imageNoiseMag_; // pixel unit

  /**
   * @brief Decision whether a new frame should be keyframe or not.
   * @param estimator     const reference to the estimator.
   * @param currentFrame  Keyframe candidate.
   * @return True if it should be a new keyframe.
   */
  bool doWeNeedANewKeyframe(
      const okvis::Estimator& estimator,
      std::shared_ptr<okvis::MultiFrame>
          currentFrame);  // based on some overlap area heuristics

};

inline bool isFilteringMethod(int algorithmId) {
  return algorithmId >= 4;
}

}  // namespace okvis

#endif  // INCLUDE_OKVIS_SIMULATION_FRONTEND_HPP_
