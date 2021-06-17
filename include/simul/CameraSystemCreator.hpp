#ifndef INCLUDE_SWIFT_VIO_CAMERA_SYSTEM_CREATOR_HPP_
#define INCLUDE_SWIFT_VIO_CAMERA_SYSTEM_CREATOR_HPP_

#include <memory>
#include <string>

#include <vio/Sample.h>

#include <okvis/cameras/CameraBase.hpp>
#include <okvis/cameras/NCameraSystem.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>

#include <okvis/Parameters.hpp>

namespace simul {
// Sideways orientation is geometrically favorable for motion estimation.
enum class CameraOrientation {
  Forward = 0,
  Backward,
  Left,
  Right,
};

enum class SimCameraModelType {
  EUROC = 0,
  VGA,
};

/**
 * @brief create_T_BC
 * The body frame is forward-left-up. Relative to the camera,
 * the camera frame is right-down-forward.
 * @param orientationId: forward, backward, left, right
 * @return T_BC
 */
Eigen::Matrix<double, 4, 4> create_T_BC(CameraOrientation orientationId,
                                        int /*camIdx*/);

/**
 * @brief createNoisyCameraSystem add noise to a reference camera system.
 * @warning Currently no noise is added to camera relative orientation.
 * @param cameraSystem reference system
 * @param cameraNoiseParams
 * @return noisy camera system.
 */
std::shared_ptr<okvis::cameras::NCameraSystem> createNoisyCameraSystem(
    std::shared_ptr<const okvis::cameras::NCameraSystem> cameraSystem,
    const okvis::ExtrinsicsEstimationParameters &cameraNoiseParams, const std::string extrinsicModel);

/**
 * @brief loadCameraSystemYaml
 * @param camImuChainYaml in format of the output of kalibr camera imu calibration.
 * @return
 */
std::shared_ptr<okvis::cameras::NCameraSystem>
loadCameraSystemYaml(const std::string &camImuChainYaml);

class CameraSystemCreator {
 public:
  CameraSystemCreator(SimCameraModelType cameraModelId,
                      CameraOrientation cameraOrientationId,
                      const std::string projIntrinsicRep,
                      const std::string extrinsicRep, double td, double tr);

  std::shared_ptr<okvis::cameras::CameraBase> createNominalCameraSystem(
      std::shared_ptr<okvis::cameras::NCameraSystem> *cameraSystem);

 private:
  std::shared_ptr<okvis::cameras::CameraBase> createCameraGeometry(
      SimCameraModelType cameraModelId);

  static const okvis::cameras::NCameraSystem::DistortionType distortType_ =
      okvis::cameras::NCameraSystem::DistortionType::RadialTangential;
  static const std::string distortName_;
  static const int camIdx_ = 0;

  const SimCameraModelType cameraModelId_;
  const CameraOrientation cameraOrientationId_;
  const std::string projIntrinsicRep_;
  const std::string extrinsicRep_;
  const double timeOffset_;
  const double readoutTime_;
};
} // namespace simul

#endif // INCLUDE_SWIFT_VIO_CAMERA_SYSTEM_CREATOR_HPP_
