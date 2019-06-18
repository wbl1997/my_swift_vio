/**
 * @file Player.hpp
 * @brief Header file for the Player class.
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_MSCKF2_PLAYER_HPP_
#define INCLUDE_MSCKF2_PLAYER_HPP_

#include <atomic>
#include <deque>
#include <memory>

//#include <boost/shared_ptr.hpp>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <dynamic_reconfigure/server.h>
#include <image_geometry/pinhole_camera_model.h>
#include <ros/ros.h>
// #include <okvis_ros/CameraConfig.h> // generated
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
//#include <okvis/HybridVio.hpp>
#include <okvis/Publisher.hpp>
#include <okvis/VioParametersReader.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/kinematics/Transformation.hpp>

#include "vio/FrameGrabber.h"  //for reading images
#include "vio/ImuGrabber.h"    //for reading IMU data
/// \brief okvis Main namespace of this package.
namespace okvis {

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
  void RunWithSavedTracks();
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

  okvis::VioInterface* vioInterface_;  ///< The VioInterface. (E.g. HybridVio)
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

}  // namespace okvis

#endif /* INCLUDE_MSCKF2_PLAYER_HPP_ */
