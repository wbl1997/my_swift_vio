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

class CameraSystemCreator {
 public:
  CameraSystemCreator(SimCameraModelType cameraModelId,
                      CameraOrientation cameraOrientationId,
                      const std::string projIntrinsicRep,
                      const std::string extrinsicRep, double td, double tr)
      : cameraModelId_(cameraModelId),
        cameraOrientationId_(cameraOrientationId),
        projIntrinsicRep_(projIntrinsicRep),
        extrinsicRep_(extrinsicRep),
        timeOffset_(td),
        readoutTime_(tr) {}

  void createDummyCameraSystem(
      std::shared_ptr<okvis::cameras::NCameraSystem> *cameraSystem) {
    Eigen::Matrix<double, 4, 4> matT_SC0 =
        create_T_BC(cameraOrientationId_, camIdx_);
    std::shared_ptr<okvis::kinematics::Transformation> T_SC_0(
        new okvis::kinematics::Transformation(matT_SC0));
    std::shared_ptr<okvis::cameras::CameraBase> cameraGeometry(
        new okvis::cameras::PinholeCamera<
            okvis::cameras::RadialTangentialDistortion>(
            0, 0, 0, 0, 0, 0,
            okvis::cameras::RadialTangentialDistortion(0.00, 0.00, 0.000,
                                                       0.000),
            0.0, 0.0));
    cameraSystem->reset(new okvis::cameras::NCameraSystem);
    (*cameraSystem)
        ->addCamera(
            T_SC_0, cameraGeometry,
            okvis::cameras::NCameraSystem::DistortionType::RadialTangential,
            projIntrinsicRep_, extrinsicRep_);
  }

  std::shared_ptr<okvis::cameras::CameraBase> createNominalCameraSystem(
      std::shared_ptr<okvis::cameras::NCameraSystem> *cameraSystem) {
    Eigen::Matrix<double, 4, 4> matT_SC0 =
        create_T_BC(cameraOrientationId_, camIdx_);
    std::shared_ptr<okvis::kinematics::Transformation> T_SC_0(
        new okvis::kinematics::Transformation(matT_SC0));

    std::shared_ptr<okvis::cameras::CameraBase> cameraGeometry0 =
        createCameraGeometry(cameraModelId_);
    cameraSystem->reset(new okvis::cameras::NCameraSystem);
    (*cameraSystem)
        ->addCamera(
            T_SC_0, cameraGeometry0,
            okvis::cameras::NCameraSystem::DistortionType::RadialTangential,
            projIntrinsicRep_, extrinsicRep_);
    return cameraGeometry0;
  }

  void createNoisyCameraSystem(
      std::shared_ptr<okvis::cameras::NCameraSystem> *cameraSystem,
      const okvis::ExtrinsicsEstimationParameters &cameraNoiseParams) {
    Eigen::Matrix<double, 4, 1> fcNoise = vio::Sample::gaussian(1, 4);
    fcNoise.head<2>() *= cameraNoiseParams.sigma_focal_length;
    fcNoise.tail<2>() *= cameraNoiseParams.sigma_principal_point;
    Eigen::Matrix<double, 4, 1> kpNoise = vio::Sample::gaussian(1, 4);
    for (int jack = 0; jack < 4; ++jack) {
      kpNoise[jack] =
          std::fabs(kpNoise[jack]) * cameraNoiseParams.sigma_distortion[jack];
    }
    Eigen::Vector3d p_CBNoise;
    for (int jack = 0; jack < 3; ++jack) {
      p_CBNoise[jack] =
          vio::gauss_rand(0, cameraNoiseParams.sigma_absolute_translation);
    }

    okvis::kinematics::Transformation ref_T_SC(
        create_T_BC(cameraOrientationId_, 0));
    std::shared_ptr<okvis::kinematics::Transformation> T_SC_noisy(
        new okvis::kinematics::Transformation(
            ref_T_SC.r() - ref_T_SC.C() * p_CBNoise, ref_T_SC.q()));
    std::shared_ptr<okvis::cameras::CameraBase> refCameraGeometry =
        createCameraGeometry(cameraModelId_);
    Eigen::VectorXd projDistortIntrinsics;
    refCameraGeometry->getIntrinsics(projDistortIntrinsics);
    std::shared_ptr<okvis::cameras::CameraBase> cameraGeometry(
        new okvis::cameras::PinholeCamera<
            okvis::cameras::RadialTangentialDistortion>(
            refCameraGeometry->imageWidth(), refCameraGeometry->imageHeight(),
            projDistortIntrinsics[0] + fcNoise[0],
            projDistortIntrinsics[1] + fcNoise[1],
            projDistortIntrinsics[2] + fcNoise[2],
            projDistortIntrinsics[3] + fcNoise[3],
            okvis::cameras::RadialTangentialDistortion(kpNoise[0], kpNoise[1],
                                                       kpNoise[2], kpNoise[3]),
            vio::gauss_rand(refCameraGeometry->imageDelay(),
                            cameraNoiseParams.sigma_td),
            std::fabs(vio::gauss_rand(refCameraGeometry->readoutTime(),
                                      cameraNoiseParams.sigma_tr))));
    cameraSystem->reset(new okvis::cameras::NCameraSystem);
    (*cameraSystem)
        ->addCamera(
            T_SC_noisy, cameraGeometry,
            okvis::cameras::NCameraSystem::DistortionType::RadialTangential,
            projIntrinsicRep_, extrinsicRep_);
  }

 private:

  std::shared_ptr<okvis::cameras::CameraBase> createCameraGeometry(
      SimCameraModelType cameraModelId) {
    std::shared_ptr<okvis::cameras::CameraBase> cameraGeometry;
    switch (cameraModelId) {
      case SimCameraModelType::VGA:
        cameraGeometry.reset(new okvis::cameras::PinholeCamera<
                             okvis::cameras::RadialTangentialDistortion>(
            640, 480, 350, 350, 322, 238,
            okvis::cameras::RadialTangentialDistortion(0, 0, 0, 0),
                               timeOffset_, readoutTime_));
        break;
      case SimCameraModelType::EUROC:
      default:
        cameraGeometry.reset(new okvis::cameras::PinholeCamera<
                             okvis::cameras::RadialTangentialDistortion>(
            752, 480, 350, 360, 378, 238,
            okvis::cameras::RadialTangentialDistortion(0.00, 0.00, 0.000,
                                                       0.000), timeOffset_, readoutTime_));
        break;
    }
    return cameraGeometry;
  }

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
