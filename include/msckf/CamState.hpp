#ifndef INCLUDE_MSCKF_CAM_STATE_HPP_
#define INCLUDE_MSCKF_CAM_STATE_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace msckf_vio {

typedef uint64_t StateIDType;
struct CAMState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // An unique identifier for the CAM state.
  StateIDType id;

  // Orientation
  // Take a vector from the world frame to the camera frame.
  Eigen::Quaterniond orientation;

  // Position of the camera frame in the world frame.
  Eigen::Vector3d position;

  CAMState(): id(0u),
    orientation(1, 0, 0, 0),
    position(Eigen::Vector3d::Zero()) {}

  CAMState(
      const StateIDType& new_id,
      const Eigen::Quaterniond& R_C_W, const Eigen::Vector3d& t_W_C):
      id(new_id),
    orientation(R_C_W),
    position(t_W_C) {}

  CAMState(const CAMState& rhs)
      : id(rhs.id), orientation(rhs.orientation), position(rhs.position) {}

  CAMState(CAMState&& rhs)
      : id{std::move(rhs.id)},
        orientation{std::move(rhs.orientation)},
        position{std::move(rhs.position)} {}

  CAMState& operator= (const CAMState& rhs) {
      if (this != &rhs) {
          id = rhs.id;
          orientation = rhs.orientation;
          position = rhs.position;
      }
      return *this;
  }
  CAMState& operator= (CAMState&& rhs) {
      std::swap(id, rhs.id);
      std::swap(orientation, rhs.orientation);
      std::swap(position, rhs.position);
      return *this;
  }
};

typedef std::vector<CAMState, Eigen::aligned_allocator<
        CAMState > > CamStateServer;
} // namespace msckf_vio

#endif // INCLUDE_MSCKF_CAM_STATE_HPP_
