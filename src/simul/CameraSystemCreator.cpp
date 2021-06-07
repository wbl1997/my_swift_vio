#include <simul/CameraSystemCreator.hpp>

#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/FovDistortion.hpp>
#include <okvis/cameras/NoDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>

#include <glog/logging.h>
#include <opencv2/core/eigen.hpp>

namespace simul {

const std::string CameraSystemCreator::distortName_ = "RadialTangentialDistortion";

Eigen::Matrix<double, 4, 4> create_T_BC(CameraOrientation orientationId,
                                        int /*camIdx*/) {
  Eigen::Matrix<double, 4, 4> matT_SC0;
  switch (orientationId) {
    case CameraOrientation::Backward: // Backward motion: The camera faces backward when the device goes straight forward.
      matT_SC0 << 0, 0, -1, 0,
                  1, 0, 0, 0,
                  0, -1, 0, 0,
                  0, 0, 0, 1;
      break;
    case CameraOrientation::Left: // Sideways motion: The camera faces left if the device goes straight forward.
      matT_SC0 << 1, 0, 0, 0,
                  0, 0, 1, 0,
                  0, -1, 0, 0,
                  0, 0, 0, 1;
      break;
    case CameraOrientation::Right: // Sideways motion: The camera faces right if the device goes straight forward.
      matT_SC0 << -1, 0, 0, 0,
                  0, 0, -1, 0,
                  0, -1, 0, 0,
                  0, 0, 0, 1;
      break;
    case CameraOrientation::Forward: // Forward motion: The camera faces forward when the device goes straight forward.
    default:
      matT_SC0 << 0, 0, 1, 0,
                 -1, 0, 0, 0,
                 0, -1, 0, 0,
                 0, 0, 0, 1;
      break;
  }
  return matT_SC0;
}

std::shared_ptr<okvis::cameras::NCameraSystem> createNoisyCameraSystem(
    std::shared_ptr<const okvis::cameras::NCameraSystem> cameraSystem,
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

  const size_t camIdx = 0u;
  std::shared_ptr<const okvis::kinematics::Transformation> ref_T_SC = cameraSystem->T_SC(camIdx);
  std::shared_ptr<okvis::kinematics::Transformation> T_SC_noisy(
      new okvis::kinematics::Transformation(
          ref_T_SC->r() - ref_T_SC->C() * p_CBNoise, ref_T_SC->q()));
  std::shared_ptr<const okvis::cameras::CameraBase> refCameraGeometry =
      cameraSystem->cameraGeometry(camIdx);
  Eigen::VectorXd projDistortIntrinsics;
  refCameraGeometry->getIntrinsics(projDistortIntrinsics);
  std::shared_ptr<okvis::cameras::CameraBase> cameraGeometry;
  switch (cameraSystem->distortionType(camIdx)) {
  case okvis::cameras::NCameraSystem::FOV:
    cameraGeometry.reset(
        new okvis::cameras::PinholeCamera<okvis::cameras::FovDistortion>(
            refCameraGeometry->imageWidth(), refCameraGeometry->imageHeight(),
            projDistortIntrinsics[0] + fcNoise[0],
            projDistortIntrinsics[1] + fcNoise[1],
            projDistortIntrinsics[2] + fcNoise[2],
            projDistortIntrinsics[3] + fcNoise[3],
            okvis::cameras::FovDistortion(projDistortIntrinsics[4] +
                                          kpNoise[0]),
            vio::gauss_rand(refCameraGeometry->imageDelay(),
                            cameraNoiseParams.sigma_td),
            std::fabs(vio::gauss_rand(refCameraGeometry->readoutTime(),
                                      cameraNoiseParams.sigma_tr))));
    break;

  case okvis::cameras::NCameraSystem::Equidistant:
    cameraGeometry.reset(new okvis::cameras::PinholeCamera<
                         okvis::cameras::EquidistantDistortion>(
        refCameraGeometry->imageWidth(), refCameraGeometry->imageHeight(),
        projDistortIntrinsics[0] + fcNoise[0],
        projDistortIntrinsics[1] + fcNoise[1],
        projDistortIntrinsics[2] + fcNoise[2],
        projDistortIntrinsics[3] + fcNoise[3],
        okvis::cameras::EquidistantDistortion(
            projDistortIntrinsics[4] + kpNoise[0],
            projDistortIntrinsics[5] + kpNoise[1],
            projDistortIntrinsics[6] + kpNoise[2],
            projDistortIntrinsics[7] + kpNoise[3]),
        vio::gauss_rand(refCameraGeometry->imageDelay(),
                        cameraNoiseParams.sigma_td),
        std::fabs(vio::gauss_rand(refCameraGeometry->readoutTime(),
                                  cameraNoiseParams.sigma_tr))));
    break;
  case okvis::cameras::NCameraSystem::NoDistortion:
    cameraGeometry.reset(
        new okvis::cameras::PinholeCamera<okvis::cameras::NoDistortion>(
            refCameraGeometry->imageWidth(), refCameraGeometry->imageHeight(),
            projDistortIntrinsics[0] + fcNoise[0],
            projDistortIntrinsics[1] + fcNoise[1],
            projDistortIntrinsics[2] + fcNoise[2],
            projDistortIntrinsics[3] + fcNoise[3],
            okvis::cameras::NoDistortion(),
            vio::gauss_rand(refCameraGeometry->imageDelay(),
                            cameraNoiseParams.sigma_td),
            std::fabs(vio::gauss_rand(refCameraGeometry->readoutTime(),
                                      cameraNoiseParams.sigma_tr))));
    break;
  case okvis::cameras::NCameraSystem::RadialTangential:
  default:
    cameraGeometry.reset(new okvis::cameras::PinholeCamera<
                         okvis::cameras::RadialTangentialDistortion>(
        refCameraGeometry->imageWidth(), refCameraGeometry->imageHeight(),
        projDistortIntrinsics[0] + fcNoise[0],
        projDistortIntrinsics[1] + fcNoise[1],
        projDistortIntrinsics[2] + fcNoise[2],
        projDistortIntrinsics[3] + fcNoise[3],
        okvis::cameras::RadialTangentialDistortion(
            projDistortIntrinsics[4] + kpNoise[0],
            projDistortIntrinsics[5] + kpNoise[1],
            projDistortIntrinsics[6] + kpNoise[2],
            projDistortIntrinsics[7] + kpNoise[3]),
        vio::gauss_rand(refCameraGeometry->imageDelay(),
                        cameraNoiseParams.sigma_td),
        std::fabs(vio::gauss_rand(refCameraGeometry->readoutTime(),
                                  cameraNoiseParams.sigma_tr))));
    break;
  }
  std::shared_ptr<okvis::cameras::NCameraSystem> noisyCameraSystem(
      new okvis::cameras::NCameraSystem);
  noisyCameraSystem->addCamera(
      T_SC_noisy, cameraGeometry, cameraSystem->distortionType(camIdx),
      cameraSystem->projOptRep(camIdx), cameraSystem->extrinsicOptRep(camIdx));
  return noisyCameraSystem;
}

std::shared_ptr<okvis::cameras::NCameraSystem>
loadCameraSystemYaml(const std::string &camImuChainYaml) {
  cv::FileStorage file(camImuChainYaml, cv::FileStorage::READ);
  OKVIS_ASSERT_TRUE(std::runtime_error, file.isOpened(),
                    "Could not open config file: " << camImuChainYaml);
  int camIdx = 0;
  for (; camIdx < 10; ++camIdx) {
    const cv::FileNode &camNode = file["cam" + std::to_string(camIdx)];
    if (camNode.isMap() &&
        camNode["resolution"].isSeq() && camNode["resolution"].size() == 2 &&
        camNode["distortion_coeffs"].isSeq() &&
        camNode["distortion_coeffs"].size() >= 1 &&
        camNode["distortion_model"].isString() &&
        camNode["intrinsics"].isSeq() && camNode["intrinsics"].size() == 4) {
    } else {
      break;
    }
  }
  int numCameras = camIdx;
  LOG(INFO) << "Found calibration in configuration file for #camera "
            << numCameras;

  std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem(
      new okvis::cameras::NCameraSystem);
  for (camIdx = 0; camIdx < numCameras; ++camIdx) {
    const cv::FileNode &camNode = file["cam" + std::to_string(camIdx)];

    cv::Mat T_mat;
    camNode["T_cam_imu"] >> T_mat;
    Eigen::Matrix4d T_eigen;
    cv::cv2eigen(T_mat, T_eigen);

    const cv::FileNode &resolutionNode = camNode["resolution"];
    std::vector<int> imageDimension{static_cast<int>(resolutionNode[0]),
                                    static_cast<int>(resolutionNode[1])};

    const cv::FileNode &intrinsicsNode = camNode["intrinsics"];
    std::vector<double> projectionIntrinsics(intrinsicsNode.size());
    for (size_t i = 0u; i < intrinsicsNode.size(); ++i) {
      projectionIntrinsics[i] = intrinsicsNode[i];
    }
    const cv::FileNode &distortionCoeffsNode = camNode["distortion_coeffs"];
    std::vector<double> distortionCoeffs(distortionCoeffsNode.size());
    for (size_t i = 0u; i < distortionCoeffsNode.size(); ++i) {
      distortionCoeffs[i] = distortionCoeffsNode[i];
    }
    std::string distortionType = camNode["distortion_model"];
    double imageDelaySecs;
    camNode["timeshift_cam_imu"] >> imageDelaySecs;
    double readoutTimeSecs;
    camNode["line_delay_nanoseconds"] >> readoutTimeSecs;
    readoutTimeSecs *= imageDimension[1];
    readoutTimeSecs /= 1000000000;
    std::string extrinsicOptMode;
    if (camNode["extrinsic_opt_mode"].isString()) {
      extrinsicOptMode =
          static_cast<std::string>(camNode["extrinsic_opt_mode"]);
    }

    std::string projectionOptMode;
    if (camNode["projection_opt_mode"].isString()) {
      projectionOptMode =
          static_cast<std::string>(camNode["projection_opt_mode"]);
    }

    okvis::kinematics::Transformation T_CS(T_eigen);
    okvis::kinematics::Transformation T_SC = T_CS.inverse();
    std::shared_ptr<okvis::kinematics::Transformation> T_SC_ptr(
        new okvis::kinematics::Transformation(T_SC.r(), T_SC.q().normalized()));

    std::transform(distortionType.begin(), distortionType.end(),
                   distortionType.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (strcmp(distortionType.c_str(), "equidistant") == 0) {
      cameraSystem->addCamera(
          T_SC_ptr,
          std::shared_ptr<okvis::cameras::CameraBase>(
              new okvis::cameras::PinholeCamera<
                  okvis::cameras::EquidistantDistortion>(
                  imageDimension[0], imageDimension[1], projectionIntrinsics[0],
                  projectionIntrinsics[1], projectionIntrinsics[2],
                  projectionIntrinsics[3],
                  okvis::cameras::EquidistantDistortion(
                      distortionCoeffs[0], distortionCoeffs[1],
                      distortionCoeffs[2], distortionCoeffs[3]),
                  imageDelaySecs, readoutTimeSecs
                  /*, id ?*/)),
          okvis::cameras::NCameraSystem::Equidistant, projectionOptMode,
          extrinsicOptMode
          /*, computeOverlaps ?*/);
      std::stringstream s;
      s << T_SC.T();
      LOG(INFO) << "Equidistant pinhole camera " << camIdx << " with T_SC=\n"
                << s.str();
    } else if (strcmp(distortionType.c_str(), "radialtangential") == 0 ||
               strcmp(distortionType.c_str(), "plumb_bob") == 0) {
      cameraSystem->addCamera(
          T_SC_ptr,
          std::shared_ptr<okvis::cameras::CameraBase>(
              new okvis::cameras::PinholeCamera<
                  okvis::cameras::RadialTangentialDistortion>(
                  imageDimension[0], imageDimension[1], projectionIntrinsics[0],
                  projectionIntrinsics[1], projectionIntrinsics[2],
                  projectionIntrinsics[3],
                  okvis::cameras::RadialTangentialDistortion(
                      distortionCoeffs[0], distortionCoeffs[1],
                      distortionCoeffs[2], distortionCoeffs[3]),
                  imageDelaySecs, readoutTimeSecs
                  /*, id ?*/)),
          okvis::cameras::NCameraSystem::RadialTangential, projectionOptMode,
          extrinsicOptMode
          /*, computeOverlaps ?*/);
      std::stringstream s;
      s << T_SC.T();
      LOG(INFO) << "Radial tangential pinhole camera " << camIdx
                << " with T_SC=\n"
                << s.str();
    } else if (strcmp(distortionType.c_str(), "radialtangential8") == 0 ||
               strcmp(distortionType.c_str(), "plumb_bob8") == 0) {
      cameraSystem->addCamera(
          T_SC_ptr,
          std::shared_ptr<okvis::cameras::CameraBase>(
              new okvis::cameras::PinholeCamera<
                  okvis::cameras::RadialTangentialDistortion8>(
                  imageDimension[0], imageDimension[1], projectionIntrinsics[0],
                  projectionIntrinsics[1], projectionIntrinsics[2],
                  projectionIntrinsics[3],
                  okvis::cameras::RadialTangentialDistortion8(
                      distortionCoeffs[0], distortionCoeffs[1],
                      distortionCoeffs[2], distortionCoeffs[3],
                      distortionCoeffs[4], distortionCoeffs[5],
                      distortionCoeffs[6], distortionCoeffs[7]),
                  imageDelaySecs, readoutTimeSecs
                  /*, id ?*/)),
          okvis::cameras::NCameraSystem::RadialTangential8, projectionOptMode,
          extrinsicOptMode
          /*, computeOverlaps ?*/);
      std::stringstream s;
      s << T_SC.T();
      LOG(INFO) << "Radial tangential 8 pinhole camera " << camIdx
                << " with T_SC=\n"
                << s.str();
    } else if (strcmp(distortionType.c_str(), "fov") == 0) {
      std::shared_ptr<okvis::cameras::CameraBase> camPtr(
          new okvis::cameras::PinholeCamera<okvis::cameras::FovDistortion>(
              imageDimension[0], imageDimension[1], projectionIntrinsics[0],
              projectionIntrinsics[1], projectionIntrinsics[2],
              projectionIntrinsics[3],
              okvis::cameras::FovDistortion(distortionCoeffs[0]),
              imageDelaySecs, readoutTimeSecs
              /*, id ?*/));
      Eigen::VectorXd intrin(5);
      intrin[0] = projectionIntrinsics[0];
      intrin[1] = projectionIntrinsics[1];
      intrin[2] = projectionIntrinsics[2];
      intrin[3] = projectionIntrinsics[3];
      intrin[4] = distortionCoeffs[0];
      camPtr->setIntrinsics(intrin);
      cameraSystem->addCamera(T_SC_ptr, camPtr,
                              okvis::cameras::NCameraSystem::FOV,
                              projectionOptMode, extrinsicOptMode
                              /*, computeOverlaps ?*/);
      std::stringstream s;
      s << T_SC.T();
      LOG(INFO) << "FOV pinhole camera " << camIdx << " with Omega "
                << distortionCoeffs[0] << " with T_SC=\n"
                << s.str();
    } else {
      LOG(ERROR) << "unrecognized distortion type " << distortionType;
    }
  }
  file.release();
  return cameraSystem;
}

CameraSystemCreator::CameraSystemCreator(SimCameraModelType cameraModelId,
                    CameraOrientation cameraOrientationId,
                    const std::string projIntrinsicRep,
                    const std::string extrinsicRep, double td, double tr)
    : cameraModelId_(cameraModelId),
      cameraOrientationId_(cameraOrientationId),
      projIntrinsicRep_(projIntrinsicRep),
      extrinsicRep_(extrinsicRep),
      timeOffset_(td),
      readoutTime_(tr) {}

std::shared_ptr<okvis::cameras::CameraBase> CameraSystemCreator::createNominalCameraSystem(
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

std::shared_ptr<okvis::cameras::CameraBase> CameraSystemCreator::createCameraGeometry(
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

} // namespace simul
