#ifndef INCLUDE_SWIFT_VIO_POINT_LANDMARK_SIMULATION_HPP
#define INCLUDE_SWIFT_VIO_POINT_LANDMARK_SIMULATION_HPP

#include <okvis/MultiFrame.hpp>
#include <okvis/cameras/NCameraSystem.hpp>
#include <okvis/kinematics/Transformation.hpp>

#include <simul/curves.h>

namespace simul {
class PointLandmarkSimulationRS
{
 public:
  /**
   * @brief projectLandmarksToNFrame
   * @param[in] homogeneousPoints
   * @param[in] simulatedTrajectory
   * @param[in] trueCentralRowEpoch
   * @param[in] cameraSystemRef
   * @param[out] nframes will be assigned keypoints for individual frames.
   * Only keypoints for successfully projected landmarks are kept.
   * @param[out] frameLandmarkIndices landmark indices of keypoints in frames.
   * @param[out] keypointIndices keypoint indices for landmarks in frames. -1 means projection failure.
   * @param[in] imageNoiseMag
   */
  static void projectLandmarksToNFrame(
      const std::vector<Eigen::Vector4d,
                        Eigen::aligned_allocator<Eigen::Vector4d>>&
          homogeneousPoints,
      std::shared_ptr<const simul::CircularSinusoidalTrajectory> simulatedTrajectory,
      okvis::Time trueCentralRowEpoch,
      std::shared_ptr<const okvis::cameras::NCameraSystem> cameraSystemRef,
      std::shared_ptr<okvis::MultiFrame> nframes,
      std::vector<std::vector<size_t>>* frameLandmarkIndices,
      std::vector<std::vector<int>>* keypointIndices,
      const double* imageNoiseMag);
};
}  // namespace simul
#endif // INCLUDE_SWIFT_VIO_POINT_LANDMARK_SIMULATION_HPP
