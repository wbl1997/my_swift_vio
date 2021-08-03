
#ifndef INCLUDE_OKVIS_SIMULATION_FRONTEND_HPP_
#define INCLUDE_OKVIS_SIMULATION_FRONTEND_HPP_

#include <mutex>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/density.hpp>
#include <boost/accumulators/statistics/stats.hpp>

#include <okvis/DenseMatcher.hpp>
#include <okvis/Estimator.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/timing/Timer.hpp>
#include <okvis/triangulation/ProbabilisticStereoTriangulator.hpp>

#include <feature_tracker/FeatureTracker.h>

#include <swift_vio/memory.h>

/// \brief okvis Main namespace of this package.
namespace simul {

struct AssociatedFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::shared_ptr<okvis::MultiFrame> nframe_;
  okvis::kinematics::Transformation pose_; // pose of the body frame at the nframe epoch.
  // keypoint indices for m scene landmarks in nframes. Each component has m elements,
  // -1 indicating no associted keypoint for a landmark.
  std::vector<std::vector<int>> keypointIndices_;
  bool isKeyframe_;

  AssociatedFrame(std::shared_ptr<okvis::MultiFrame> nframe, const okvis::kinematics::Transformation& pose,
                  const std::vector<std::vector<int>>& keypointIndices, bool isKeyframe) :
    nframe_(nframe), pose_(pose), keypointIndices_(keypointIndices), isKeyframe_(isKeyframe) {

  }
};

struct SimFrontendOptions {
  int maxTrackLength_; ///< Cap feature track length
  int maxMatchKeyframes_;
  double minKeyframeDistance_;
  double minKeyframeAngle_;
  bool useTrueLandmarkPosition_;

  SimFrontendOptions(int maxTrackLength = 10, int maxMatchKeyframes = 3,
                     double minKeyframeDistance = 0.4,
                     double minKeyframeAngle = 10 * M_PI / 180,
                     bool useTrueLandmarkPosition = false)
      : maxTrackLength_(maxTrackLength), maxMatchKeyframes_(maxMatchKeyframes),
        minKeyframeDistance_(minKeyframeDistance),
        minKeyframeAngle_(minKeyframeAngle),
        useTrueLandmarkPosition_(useTrueLandmarkPosition) {}
};

/**
 * @brief A frontend for simulation with predefined landmarks.
 */
class SimulationFrontend {
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)


  /**
   * @brief Constructor.
   * @param numCameras Number of cameras in the sensor configuration.
   */
  SimulationFrontend(
      const std::vector<Eigen::Vector4d,
                        Eigen::aligned_allocator<Eigen::Vector4d>>
          &homogeneousPoints,
      const std::vector<uint64_t> &lmIds, size_t numCameras,
      const SimFrontendOptions &options);

  virtual ~SimulationFrontend() {}

  ///@{

  /**
   * @brief given keypoints associated with landmarks, add features to the estimator.
   * @param estimator
   * @param[in] keypointIndices indices of keypoints for landmarks observed by frames.
   * Each subvector is as long as the number of landmarks.
   * @param[in, out] nframes
   * @param[out] asKeyframe
   * @return number of tracked features.
   */
  int dataAssociationAndInitialization(
      okvis::Estimator& estimator, const std::vector<std::vector<int>>& keypointIndices,
      std::shared_ptr<okvis::MultiFrame> nframes, bool* asKeyframe);

  ///@}


  /// @name Other getters
  /// @{

  /// @brief Returns true if the initialization has been completed (RANSAC with actual translation)
  bool isInitialized() const {
    return isInitialized_;
  }

  int numKeyframes() const {
    return numKeyframes_;
  }
  /// @}
  

  // output the distribution of number of features in images
  void printNumFeatureDistribution(std::ofstream& stream) const;

  static const double fourthRoot2_; // sqrt(sqrt(2))

 private:
  bool isInitialized_;       ///< Is the pose initialised?
  const size_t numCameras_;  ///< Number of cameras in the configuration.

  SimFrontendOptions options_;
  Eigen::AlignedDeque<AssociatedFrame> nframeList_;

  okvis::kinematics::Transformation previousKeyframePose_;

  // scene landmarks
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
      homogeneousPoints_;
  std::vector<uint64_t> lmIds_;

  int numNFrames_;
  int numKeyframes_;

  struct LandmarkKeypointMatch {
    okvis::KeypointIdentifier currentKeypoint;
    okvis::KeypointIdentifier previousKeypoint;
    uint64_t landmarkId; // unique identifier
    size_t landmarkIdInVector; // index in the scene grid
  };
  /**
   * @brief Decision whether a new frame should be keyframe or not.
   * @param estimator     const reference to the estimator.
   * @param currentFrame  Keyframe candidate.
   * @return True if it should be a new keyframe.
   */
  bool doWeNeedANewKeyframe(
      const okvis::Estimator& estimator,
      std::shared_ptr<okvis::MultiFrame> currentFrame) const;

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
   * @param landmarkMatches the list of keypoint match between the two frames of one landmark
   */
  template <class CAMERA_GEOMETRY_T>
  int addMatchToEstimator(
      okvis::Estimator& estimator,
      std::shared_ptr<okvis::MultiFrame> prevFrames,
      std::shared_ptr<okvis::MultiFrame> currFrames,
      const std::vector<LandmarkKeypointMatch>& landmarkMatches) const;
};

}  // namespace simul

#endif  // INCLUDE_OKVIS_SIMULATION_FRONTEND_HPP_
