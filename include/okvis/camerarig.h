#ifndef CAMERA_RIG_H
#define CAMERA_RIG_H
#include <okvis/cameras/CameraBase.hpp>
#include <okvis/kinematics/Transformation.hpp>

namespace okvis {
namespace cameras {

std::shared_ptr<cameras::CameraBase> cloneCameraGeometry(
    std::shared_ptr<const cameras::CameraBase> cameraGeometry);

class CameraRig {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  ///< Mounting transformations from IMU
  std::vector<std::shared_ptr<okvis::kinematics::Transformation>> T_SC_;
  ///< Camera geometries
  std::vector<std::shared_ptr<cameras::CameraBase>> cameraGeometries_;
  ///< time in secs to read out a frame, applies to rolling shutter cameras
  std::vector<double> frame_readout_time_;
  ///< at the same epoch, timestamp by camera_clock + time_delay =
  ///  timestamp by IMU clock
  std::vector<double> time_delay_;

 public:
  inline double getTimeDelay(int camera_id) { return time_delay_[camera_id]; }
  inline double getReadoutTime(int camera_id) {
    return frame_readout_time_[camera_id];
  }
  inline uint32_t getImageWidth(int camera_id) {
    return cameraGeometries_[camera_id]->imageWidth();
  }
  inline uint32_t getImageHeight(int camera_id) {
    return cameraGeometries_[camera_id]->imageHeight();
  }
  inline okvis::kinematics::Transformation getCameraExtrinsic(int camera_id) {
    return *(T_SC_[camera_id]);
  }

  inline void setTimeDelay(int camera_id, double td) {
    time_delay_[camera_id] = td;
  }
  inline void setReadoutTime(int camera_id, double tr) {
    frame_readout_time_[camera_id] = tr;
  }
  inline void setCameraExtrinsic(
      int camera_id, const okvis::kinematics::Transformation& T_SC) {
    *(T_SC_[camera_id]) = T_SC;
  }

  inline int addCamera(
      std::shared_ptr<const okvis::kinematics::Transformation> T_SC,
      std::shared_ptr<const cameras::CameraBase> cameraGeometry, double tr,
      double td) {
    T_SC_.emplace_back(
        std::make_shared<okvis::kinematics::Transformation>(*T_SC));
    cameraGeometries_.emplace_back(cloneCameraGeometry(cameraGeometry));
    frame_readout_time_.push_back(tr);
    time_delay_.push_back(td);
    return static_cast<int>(T_SC_.size()) - 1;
  }
};
}  // namespace cameras
}  // namespace okvis
#endif  // CAMERA_RIG_H
