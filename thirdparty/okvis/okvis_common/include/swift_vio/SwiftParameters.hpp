
/**
 * @file Parameters.hpp
 * @brief This file contains struct definitions that encapsulate parameters and settings for swift_vio.
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_SWIFT_VIO_PARAMETERS_HPP_
#define INCLUDE_SWIFT_VIO_PARAMETERS_HPP_

#include <string>

namespace swift_vio {
enum class EstimatorAlgorithm {
  OKVIS = 0,  ///< Okvis original keyframe-based estimator.
  SlidingWindowSmoother, ///< Gtsam::FixedLagSmoother.
  RiSlidingWindowSmoother, ///< Gtsam::FixedLagSmoother with right invariant errors.
  HybridFilter, ///< MSCKF + EKF-SLAM with keyframe-based marginalization.
  CalibrationFilter, ///< EKF for RS camera-IMU calibration.
  MSCKF,  ///< MSCKF with keyframe-based marginalization.
  TFVIO  ///< Triangulate-free VIO with only epipolar constraints.
};

EstimatorAlgorithm EstimatorAlgorithmNameToId(std::string description);

std::string EstimatorAlgorithmIdToName(EstimatorAlgorithm id);

struct FrontendOptions {
  ///< stereo matching with epipolar check and landmark fusion or
  /// the okvis stereo matching 2d-2d + 3d-2d + 3d-2d?
  bool stereoMatchWithEpipolarCheck;

  double epipolarDistanceThreshold;

  ///< 0 default okvis brisk keyframe and back-to-back frame matching
  ///< 1 KLT back-to-back frame matching,
  ///< 2 brisk back-to-back frame matching
  int featureTrackingMethod;

  FrontendOptions(bool initWithoutEnoughParallax = true,
                  bool stereoWithEpipolarCheck = true,
                  double epipolarDistanceThreshold = 2.5,
                  int featureTrackingMethod = 0);
};

struct PointLandmarkOptions {
  int landmarkModelId;
  size_t minTrackLengthForMsckf;
  size_t maxHibernationFrames;   ///< max number of miss frames, each frame has potentially many images.
  size_t minTrackLengthForSlam;  ///< min track length of a landmark to be included in state.
  int maxInStateLandmarks;       ///< max number of landmarks in the state vector.
  int maxMarginalizedLandmarks;  ///< max number of marginalized landmarks in one update step.
  PointLandmarkOptions();
  PointLandmarkOptions(int lmkModelId, size_t minMsckfTrackLength,
                       size_t hibernationFrames, size_t minSlamTrackLength,
                       int maxInStateLandmarks, int maxMarginalizedLandmarks);
  std::string toString(std::string lead) const;
};

struct PoseGraphOptions {
  int maxOdometryConstraintForAKeyframe;
  double minDistance;
  double minAngle;
  PoseGraphOptions();
};

struct InputData {
  std::string imageFolder;
  std::string timeFile;
  std::string videoFile;
  std::string imuFile;
  int startIndex;
  int finishIndex;
};
}  // namespace swift_vio

#endif // INCLUDE_SWIFT_VIO_PARAMETERS_HPP_
