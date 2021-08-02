/**
 * @file Player.hpp
 * @brief Header file for the Player class.
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_SWIFT_VIO_PLAYER_HPP_
#define INCLUDE_SWIFT_VIO_PLAYER_HPP_

#include <atomic>
#include <deque>
#include <memory>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <dynamic_reconfigure/server.h>
#include <image_geometry/pinhole_camera_model.h>
#include <ros/ros.h>
#pragma GCC diagnostic pop
#include <image_transport/image_transport.h>
#include "sensor_msgs/Imu.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/opencv.hpp>
#pragma GCC diagnostic pop
#include <Eigen/Core>

#include <okvis/Time.hpp>
#include <okvis/VioInterface.hpp>
#include <okvis/cameras/NCameraSystem.hpp>

#include <io_wrap/Publisher.hpp>
#include <okvis/VioParametersReader.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/kinematics/Transformation.hpp>

#include "vio/FrameGrabber.h"  // for reading images
#include "vio/ImuGrabber.h"    // for reading IMU data

namespace swift_vio {
/**
 * @brief This class handles all the buffering of incoming data.
 */
class Player {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  ~Player();

  Player(okvis::VioInterface* vioInterfacePtr,
         const okvis::VioParameters& param_reader);

  void Run();
  // Block if previous processing has not finished.
  void RunBlocking();

  std::atomic<bool> mbFinished;

 protected:
  /// @name ROS callbacks
  /// @{

  /// @}
  /// @name Direct (no ROS) callbacks and other sensor related methods.
  /// @{

  /// @}
  /// @name Node and subscriber related
  /// @{

  /// @}

  okvis::VioInterface* vioInterface_;  ///< The VioInterface.
  okvis::VioParameters
      vioParameters_;  ///< The parameters and settings. //huai: although
                       ///< cameraGeometry info is included but not used through
                       ///< this member

  std::string mVideoFile;
  std::string mImageFolder;
  std::string mTimeFile;
  std::string mImuFile;

  std::shared_ptr<vio::IMUGrabber> mIG;
  std::shared_ptr<vio::FrameGrabber> mFG;
  std::shared_ptr<vio::StatesGrabber> mSG;

 private:
  Player(const Player& rhs);
  Player& operator=(const Player&) = delete;
};
}  // namespace swift_vio

#endif /* INCLUDE_SWIFT_VIO_PLAYER_HPP_ */
