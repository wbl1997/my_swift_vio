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
   * @param homogeneousPoints
   * @param T_WS_ref
   * @param cameraSystemRef
   * @param framesInOut the keypoints for every frame are created from
   * observations of successfully projected landmarks.
   * @param frameLandmarkIndices {{landmark index of every keypoint} in every
   * frame}, every entry >= 0, length = number of valid projections = number of keypoints.
   * @param keypointIndices {{every landmark's keypoint index} in every frame},
   * valid entry >= 0, void entry -1, length = number of landmarks.
   * @param imageNoiseMag
   */
  static void projectLandmarksToNFrame(
      const std::vector<Eigen::Vector4d,
                        Eigen::aligned_allocator<Eigen::Vector4d>>&
          homogeneousPoints,
      std::shared_ptr<const simul::CircularSinusoidalTrajectory> simulatedTrajectory,
      okvis::Time trueCentralRowEpoch,
      std::shared_ptr<const okvis::cameras::NCameraSystem> cameraSystemRef,
      std::shared_ptr<okvis::MultiFrame> framesInOut,
      std::vector<std::vector<size_t>>* frameLandmarkIndices,
      std::vector<std::vector<int>>* keypointIndices,
      const double* imageNoiseMag);
};
}  // namespace simul
#endif // INCLUDE_SWIFT_VIO_POINT_LANDMARK_SIMULATION_HPP
