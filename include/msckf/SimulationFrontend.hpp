
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
  std::shared_ptr<okvis::MultiFrame> previousKeyframe_;
  okvis::kinematics::Transformation previousKeyframePose_;
  // the keypoint index corresponding to each scene landmark in the previous keyframe
  std::vector<std::vector<int>> previousKeyframeKeypointIndices_;
  // feature tracking
  std::shared_ptr<okvis::MultiFrame> previousFrame_;
  okvis::kinematics::Transformation previousFramePose_;
  // the keypoint index corresponding to each scene landmark in the previous frame
  std::vector<std::vector<int>> previousFrameKeypointIndices_;

  // scene landmarks
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
      homogeneousPoints_;
  std::vector<uint64_t> lmIds_;
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
      noisyHomogeneousPoints_;
  static const double imageNoiseMag_; // pixel unit
  static const double fourthRoot2_; // sqrt(sqrt(2))

  struct LandmarkKeypointMatch {
    KeypointIdentifier currentKeypoint;
    KeypointIdentifier previousKeypoint;
    uint64_t landmarkId; // unique identifier
    size_t landmarkIdInVector; // index in the scene grid
  };
  /**
   * @brief Decision whether a new frame should be keyframe or not.
   * @param estimator     const reference to the estimator.
   * @param currentFrame  Keyframe candidate.
   * @param T_WS reference pose of currentFrame
   * @return True if it should be a new keyframe.
   */
  bool doWeNeedANewKeyframe(
      const okvis::Estimator& estimator,
      std::shared_ptr<okvis::MultiFrame> currentFrame,
      const okvis::kinematics::Transformation& T_WS) const;

  /**
   * @brief matchToFrame find the keypoint matches between two multiframes
   * @param previousKeypointIndices the keypoint index for each landmark in each previous frame, -1 if not exist.
   * @param currentKeypointIndices the keypoint index for each landmark in each current frame, -1 if not exist.
   * @param prevFrameId
   * @param currFrameId
   * @param landmarkMatches the list of keypoint match between the two frames of one landmark
   * @return
   */
  int matchToFrame(const std::vector<std::vector<int>>& previousKeypointIndices,
                   const std::vector<std::vector<int>>& currentKeypointIndices,
                   const uint64_t prevFrameId, const uint64_t currFrameId,
                   std::vector<LandmarkKeypointMatch>* landmarkMatches) const;

  /**
   * @brief given landmark matches between two frames, add proper constraints to the estimator
   * @param estimator
   * @param prevFrames
   * @param currFrames
   * @param T_WSp_ref reference pose for the previous frame
   * @param T_WSc_ref reference pose for the current frame
   * @param landmarkMatches the list of keypoint match between the two frames of one landmark
   */
  template <class CAMERA_GEOMETRY_T>
  int addMatchToEstimator(
      okvis::Estimator& estimator,
      std::shared_ptr<okvis::MultiFrame> prevFrames,
      std::shared_ptr<okvis::MultiFrame> currFrames,
      const okvis::kinematics::Transformation& T_WSp_ref,
      const okvis::kinematics::Transformation& T_WSc_ref,
      const std::vector<LandmarkKeypointMatch>& landmarkMatches) const;
};

inline bool isFilteringMethod(int algorithmId) {
  return algorithmId >= 4;
}

}  // namespace okvis

#endif  // INCLUDE_OKVIS_SIMULATION_FRONTEND_HPP_
