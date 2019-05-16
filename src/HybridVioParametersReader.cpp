/**
 * @file HybridVioParametersReader.cpp
 * @brief Source file for the HybridVioParametersReader class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */
#include <okvis/HybridVioParametersReader.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <glog/logging.h>

#include <okvis/cameras/NCameraSystem.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>

#include <opencv2/core/core.hpp>

#ifdef HAVE_LIBVISENSOR
  #include <visensor/visensor_api.hpp>
#endif

/// \brief okvis Main namespace of this package.
namespace okvis {
// The default constructor.
HybridVioParametersReader::HybridVioParametersReader()
    : useDriver(false),
      readConfigFile_(false) {
  vioParameters_.publishing.publishRate = 0;
}

// The constructor. This calls readConfigFile().
HybridVioParametersReader::HybridVioParametersReader(
    const std::string& filename) {
  readConfigFile(filename);
}

static void parseExpandedCameraParamSigmas(
    cv::FileNode cameraParamNode,
    ExtrinsicsEstimationParameters *camera_extrinsics) {
  if (cameraParamNode["sigma_focal_length"].isReal()) {
    cameraParamNode["sigma_focal_length"] >>
        camera_extrinsics->sigma_focal_length;
  } else {
    camera_extrinsics->sigma_focal_length = 0.0;
    LOG(WARNING) << "camera_params: sigma_focal_length parameter not provided. "
                 << "Setting to default 0.0";
  }
  if (cameraParamNode["sigma_principal_point"].isReal()) {
    cameraParamNode["sigma_principal_point"] >>
        camera_extrinsics->sigma_principal_point;
  } else {
    camera_extrinsics->sigma_principal_point = 0.0;
    LOG(WARNING)
        << "camera_params: sigma_principal_point parameter not provided. "
        << "Setting to default 0.0";
  }
  cv::FileNode a0Node = cameraParamNode["sigma_distortion"];
  camera_extrinsics->sigma_distortion.setZero();
  if (a0Node.isSeq()) {
    for (size_t jack = 0; jack < a0Node.size(); ++jack)
      camera_extrinsics->sigma_distortion[jack] =
          static_cast<double>(a0Node[jack]);
  } else {
    LOG(WARNING) << "camera_params: sigma_distortion parameter not provided. "
                 << "Setting to default 0.0";
  }
  if (cameraParamNode["sigma_td"].isReal()) {
    cameraParamNode["sigma_td"] >> camera_extrinsics->sigma_td;
  } else {
    camera_extrinsics->sigma_td = 0.0;
    LOG(WARNING) << "camera_params: sigma_td parameter not provided. "
                 << "Setting to default 0.0";
  }
  if (cameraParamNode["sigma_tr"].isReal()) {
    cameraParamNode["sigma_tr"] >> camera_extrinsics->sigma_tr;
  } else {
    camera_extrinsics->sigma_tr = 0.0;
    LOG(WARNING) << "camera_params: sigma_tr parameter not provided. "
                 << "Setting to default 0.0";
  }
}

void parseInputData(cv::FileNode inputDataNode, InputData *input) {
  if (inputDataNode["video_file"].isString()) {
    std::string path = (std::string)inputDataNode["video_file"];
    // cut out first word. str currently contains everything including comments
    input->videoFile = path.substr(0, path.find(" "));
  } else {
    input->videoFile = "";
  }
  if (inputDataNode["image_folder"].isString()) {
    std::string path = (std::string)inputDataNode["image_folder"];
    // cut out first word. str currently contains everything including comments
    input->imageFolder = path.substr(0, path.find(" "));
  } else {
    input->imageFolder = "";
  }
  if (inputDataNode["imu_file"].isString()) {
    std::string path = (std::string)inputDataNode["imu_file"];
    // cut out first word. str currently contains everything including comments
    input->imuFile = path.substr(0, path.find(" "));
  } else {
    input->imuFile = "";
  }
  if (inputDataNode["time_file"].isString()) {
    std::string path = (std::string)inputDataNode["time_file"];
    // cut out first word. str currently contains everything including comments
    input->timeFile = path.substr(0, path.find(" "));
  } else {
    input->timeFile = "";
  }
  if (inputDataNode["startIndex"].isInt()) {
    input->startIndex = inputDataNode["startIndex"];
  } else {
    input->startIndex = 0;
  }
  if (inputDataNode["finishIndex"].isInt()) {
    input->finishIndex = inputDataNode["finishIndex"];
  } else {
    input->finishIndex = 0;
  }

  if (inputDataNode["vo_poses_file"].isString()) {
    std::string path = (std::string)inputDataNode["vo_poses_file"];
    // cut out first word. str currently contains everything including comments
    input->voPosesFile = path.substr(0, path.find(" "));
  } else {
    input->voPosesFile = "";
  }
  if (inputDataNode["vo_feature_tracks_file"].isString()) {
    std::string path =
        static_cast<std::string>(inputDataNode["vo_feature_tracks_file"]);
    // cut out first word. str currently contains everything including comments
    input->voFeatureTracksFile = path.substr(0, path.find(" "));
  } else {
    input->voFeatureTracksFile = "";
  }
}

void parseInitialState(cv::FileNode initialStateNode,
                       InitialState *initialState) {
  bool bUseExternalState = true;
  cv::FileNode timeNode = initialStateNode["state_time"];
  if (timeNode.isReal()) {
    double time;
    timeNode >> time;
    initialState->stateTime = okvis::Time(time);
  } else {
    bUseExternalState = false;
  }

  cv::FileNode vsNode = initialStateNode["v_WS"];
  if (vsNode.isSeq()) {
    Eigen::Vector3d vs;
    vs << vsNode[0], vsNode[1], vsNode[2];
    initialState->v_WS = vs;
  } else {
    bUseExternalState = false;
    initialState->v_WS.setZero();
  }

  cv::FileNode stdvsNode = initialStateNode["std_v_WS"];
  if (stdvsNode.isSeq()) {
    Eigen::Vector3d stdvs;
    stdvs << stdvsNode[0], stdvsNode[1], stdvsNode[2];
    initialState->std_v_WS = stdvs;
  } else {
    initialState->std_v_WS = Eigen::Vector3d(1, 1, 1) * 1e-1;
  }

  cv::FileNode stdpsNode = initialStateNode["std_p_WS"];
  if (stdpsNode.isSeq()) {
    Eigen::Vector3d stdps;
    stdps << stdpsNode[0], stdpsNode[1], stdpsNode[2];
    initialState->std_p_WS = stdps;
  } else {
    initialState->std_p_WS = Eigen::Vector3d(1, 1, 1) * 1e-2;
  }

  cv::FileNode qsNode = initialStateNode["q_WS"];
  if (qsNode.isSeq()) {
    Eigen::Vector4d qs;
    qs << qsNode[0], qsNode[1], qsNode[2], qsNode[3];
    initialState->q_WS = Eigen::Quaterniond(qs[3], qs[0], qs[1], qs[2]);
  } else {
    bUseExternalState = false;
    initialState->q_WS = Eigen::Quaterniond(1, 0, 0, 0);
  }

  cv::FileNode stdqsNode = initialStateNode["std_q_WS"];
  if (stdqsNode.isSeq()) {
    Eigen::Vector3d stdqs;
    stdqs << stdqsNode[0], stdqsNode[1], stdqsNode[2];
    initialState->std_q_WS = stdqs;
  } else {
    initialState->std_q_WS = Eigen::Vector3d(1, 1, 3) * M_PI / 180;
  }

  initialState->bUseExternalInitState = bUseExternalState;
  LOG(INFO) << "initial velocity in the global frame z pointing neg gravity "
            << initialState->v_WS.transpose() << std::endl
            << " and its std " << initialState->std_v_WS.transpose()
            << std::endl
            << "the std of the initial position in that frame "
            << initialState->std_p_WS.transpose();
  LOG(INFO) << "initial quaternion from body/IMU frame to the global frame "
            << initialState->q_WS.coeffs().transpose() << std::endl
            << " and its std " << initialState->std_q_WS.transpose()
            << std::endl;
}

// Read and parse a config file.
void HybridVioParametersReader::readConfigFile(const std::string& filename) {
  vioParameters_.optimization.useMedianFilter = false;
  vioParameters_.optimization.timeReserve.fromSec(0.005);

  cv::FileStorage file(filename, cv::FileStorage::READ);
  OKVIS_ASSERT_TRUE(Exception, file.isOpened(),
                    "Could not open config file: " << filename);
  LOG(INFO) << "Opened configuration file: " << filename;

  // number of keyframes
  if (file["numKeyframes"].isInt()) {
    file["numKeyframes"] >> vioParameters_.optimization.numKeyframes;
  } else {
    LOG(WARNING) << "numKeyframes parameter not provided. "
                 << "Setting to default numKeyframes=5.";
    vioParameters_.optimization.numKeyframes = 5;
  }
  // number of IMU frames
  if (file["numImuFrames"].isInt()) {
    file["numImuFrames"] >> vioParameters_.optimization.numImuFrames;
  } else {
    LOG(WARNING)
        << "numImuFrames parameter not provided. "
        << "Setting to default numImuFrames=2.";
    vioParameters_.optimization.numImuFrames = 2;
  }
  // minimum ceres iterations
  if (file["ceres_options"]["minIterations"].isInt()) {
    file["ceres_options"]["minIterations"]
        >> vioParameters_.optimization.min_iterations;
  } else {
    LOG(WARNING)
        << "ceres_options: minIterations parameter not provided. "
        << "Setting to default minIterations=1";
    vioParameters_.optimization.min_iterations = 1;
  }
  // maximum ceres iterations
  if (file["ceres_options"]["maxIterations"].isInt()) {
    file["ceres_options"]["maxIterations"]
        >> vioParameters_.optimization.max_iterations;
  } else {
    LOG(WARNING)
        << "ceres_options: maxIterations parameter not provided. "
        << "Setting to default maxIterations=10.";
    vioParameters_.optimization.max_iterations = 10;
  }
  // ceres time limit
  if (file["ceres_options"]["timeLimit"].isReal()) {
    file["ceres_options"]["timeLimit"] >>
        vioParameters_.optimization.timeLimitForMatchingAndOptimization;
  } else {
    LOG(WARNING)
        << "ceres_options: timeLimit parameter not provided. "
        << "Setting no time limit.";
    vioParameters_.optimization.timeLimitForMatchingAndOptimization = -1.0;
  }

  // do we use the direct driver?
  bool success = parseBoolean(file["useDriver"], useDriver);
  OKVIS_ASSERT_TRUE(Exception, success,
                    "'useDriver' parameter missing in configuration file.");

  // display images?
  success = parseBoolean(file["displayImages"],
                         vioParameters_.visualization.displayImages);
  OKVIS_ASSERT_TRUE(Exception, success,
                    "'displayImages' parameter missing in configuration file.");

  // detection threshold
  success = file["detection_options"]["threshold"].isReal();
  OKVIS_ASSERT_TRUE(
      Exception, success,
      "'detection threshold' parameter missing in configuration file.");
  file["detection_options"]["threshold"] >>
      vioParameters_.optimization.detectionThreshold;

  // detection octaves
  success = file["detection_options"]["octaves"].isInt();
  OKVIS_ASSERT_TRUE(
      Exception, success,
      "'detection octaves' parameter missing in configuration file.");
  file["detection_options"]["octaves"] >>
      vioParameters_.optimization.detectionOctaves;
  OKVIS_ASSERT_TRUE(Exception,
                    vioParameters_.optimization.detectionOctaves >= 0,
                    "Invalid parameter value.");

  // maximum detections
  success = file["detection_options"]["maxNoKeypoints"].isInt();
  OKVIS_ASSERT_TRUE(
      Exception, success,
      "'detection maxNoKeypoints' parameter missing in configuration file.");
  file["detection_options"]["maxNoKeypoints"] >>
      vioParameters_.optimization.maxNoKeypoints;
  OKVIS_ASSERT_TRUE(Exception,
                    vioParameters_.optimization.maxNoKeypoints >= 0,
                    "Invalid parameter value.");

  // image delay
  success = file["imageDelay"].isReal();
  OKVIS_ASSERT_TRUE(Exception, success,
                    "'imageDelay' parameter missing in configuration file.");
  file["imageDelay"] >> vioParameters_.sensors_information.imageDelay;
  LOG(INFO) << "imageDelay=" << vioParameters_.sensors_information.imageDelay;

  // camera rate
  success = file["camera_params"]["camera_rate"].isInt();
  OKVIS_ASSERT_TRUE(
      Exception, success,
      "'camera_params: camera_rate' parameter missing in configuration file.");
  file["camera_params"]["camera_rate"]
      >> vioParameters_.sensors_information.cameraRate;

  success = file["camera_params"]["image_readout_time"].isReal();
  OKVIS_ASSERT_TRUE(
      Exception, success,
      "'camera_params: image_readout_time' parameter "
      "missing in configuration file.");
  file["camera_params"]["image_readout_time"]
      >> vioParameters_.sensors_information.imageReadoutTime;

  // timestamp tolerance
  if (file["camera_params"]["timestamp_tolerance"].isReal()) {
    file["camera_params"]["timestamp_tolerance"]
        >> vioParameters_.sensors_information.frameTimestampTolerance;
    OKVIS_ASSERT_TRUE(
        Exception,
        vioParameters_.sensors_information.frameTimestampTolerance
            < 0.5 / vioParameters_.sensors_information.cameraRate,
        "Timestamp tolerance for stereo frames is larger than "
        "half the time between frames.");
    OKVIS_ASSERT_TRUE(
        Exception,
        vioParameters_.sensors_information.frameTimestampTolerance >= 0.0,
        "Timestamp tolerance is smaller than 0");
  } else {
    vioParameters_.sensors_information.frameTimestampTolerance = 0.2
        / vioParameters_.sensors_information.cameraRate;
    LOG(WARNING)
        << "No timestamp tolerance for stereo frames specified. Setting to "
        << vioParameters_.sensors_information.frameTimestampTolerance;
  }

  // camera params
  if (file["camera_params"]["sigma_absolute_translation"].isReal()) {
    file["camera_params"]["sigma_absolute_translation"]
        >> vioParameters_.camera_extrinsics.sigma_absolute_translation;
  } else {
    vioParameters_.camera_extrinsics.sigma_absolute_translation = 0.0;
    LOG(WARNING)
        << "camera_params: sigma_absolute_translation parameter not provided. "
        << "Setting to default 0.0";
  }
  if (file["camera_params"]["sigma_absolute_orientation"].isReal()) {
    file["camera_params"]["sigma_absolute_orientation"]
        >> vioParameters_.camera_extrinsics.sigma_absolute_orientation;
  } else {
    vioParameters_.camera_extrinsics.sigma_absolute_orientation = 0.0;
    LOG(WARNING)
        << "camera_params: sigma_absolute_orientation parameter not provided. "
        << "Setting to default 0.0";
  }
  if (file["camera_params"]["sigma_c_relative_translation"].isReal()) {
    file["camera_params"]["sigma_c_relative_translation"]
        >> vioParameters_.camera_extrinsics.sigma_c_relative_translation;
  } else {
    vioParameters_.camera_extrinsics.sigma_c_relative_translation = 0.0;
    LOG(WARNING)
        << "camera_params: sigma_c_relative_translation parameter not provided."
        <<" Setting to default 0.0";
  }
  if (file["camera_params"]["sigma_c_relative_orientation"].isReal()) {
    file["camera_params"]["sigma_c_relative_orientation"]
        >> vioParameters_.camera_extrinsics.sigma_c_relative_orientation;
  } else {
    vioParameters_.camera_extrinsics.sigma_c_relative_orientation = 0.0;
    LOG(WARNING)
        << "camera_params: sigma_c_relative_orientation parameter not provided."
        << " Setting to default 0.0";
  }

  parseExpandedCameraParamSigmas(file["camera_params"],
                                 &vioParameters_.camera_extrinsics);

  if (file["publishing_options"]["publish_rate"].isInt()) {
    file["publishing_options"]["publish_rate"] >>
        vioParameters_.publishing.publishRate;
  }

  if (file["publishing_options"]["landmarkQualityThreshold"].isReal()) {
    file["publishing_options"]["landmarkQualityThreshold"]
        >> vioParameters_.publishing.landmarkQualityThreshold;
  }

  if (file["publishing_options"]["maximumLandmarkQuality"].isReal()) {
    file["publishing_options"]["maximumLandmarkQuality"]
        >> vioParameters_.publishing.maxLandmarkQuality;
  }

  if (file["publishing_options"]["maxPathLength"].isInt()) {
    vioParameters_.publishing.maxPathLength =
        static_cast<int>(file["publishing_options"]["maxPathLength"]);
  }

  parseBoolean(file["publishing_options"]["publishImuPropagatedState"],
                   vioParameters_.publishing.publishImuPropagatedState);

  parseBoolean(file["publishing_options"]["publishLandmarks"],
                   vioParameters_.publishing.publishLandmarks);

  cv::FileNode T_Wc_W_ = file["publishing_options"]["T_Wc_W"];
  if (T_Wc_W_.isSeq()) {
    Eigen::Matrix4d T_Wc_W_e;
    T_Wc_W_e << T_Wc_W_[0], T_Wc_W_[1], T_Wc_W_[2], T_Wc_W_[3], T_Wc_W_[4],
        T_Wc_W_[5], T_Wc_W_[6], T_Wc_W_[7], T_Wc_W_[8], T_Wc_W_[9], T_Wc_W_[10],
        T_Wc_W_[11], T_Wc_W_[12], T_Wc_W_[13], T_Wc_W_[14], T_Wc_W_[15];

    vioParameters_.publishing.T_Wc_W =
        okvis::kinematics::Transformation(T_Wc_W_e);
    std::stringstream s;
    s << vioParameters_.publishing.T_Wc_W.T();
    LOG(INFO) << "Custom World frame provided T_Wc_W=\n" << s.str();
  }

  if (file["publishing_options"]["trackedBodyFrame"].isString()) {
    std::string frame =
        static_cast<std::string>(
            file["publishing_options"]["trackedBodyFrame"]);
    // cut out first word. str currently contains everything including comments
    frame = frame.substr(0, frame.find(" "));
    if (frame.compare("B") == 0) {
      vioParameters_.publishing.trackedBodyFrame = FrameName::B;
    } else if (frame.compare("S") == 0) {
      vioParameters_.publishing.trackedBodyFrame = FrameName::S;
    } else {
      LOG(WARNING) << frame << " unknown/invalid frame for trackedBodyFrame, "
                   <<"setting to B";
      vioParameters_.publishing.trackedBodyFrame = FrameName::B;
    }
  }

  if (file["publishing_options"]["velocitiesFrame"].isString()) {
    std::string frame = static_cast<std::string>(
        file["publishing_options"]["velocitiesFrame"]);
    // cut out first word. str currently contains everything including comments
    frame = frame.substr(0, frame.find(" "));
    if (frame.compare("B") == 0) {
      vioParameters_.publishing.velocitiesFrame = FrameName::B;
    } else if (frame.compare("S") == 0) {
      vioParameters_.publishing.velocitiesFrame = FrameName::S;
    } else if (frame.compare("Wc") == 0) {
      vioParameters_.publishing.velocitiesFrame = FrameName::Wc;
    } else {
      LOG(WARNING) << frame << " unknown/invalid frame for velocitiesFrame,"
                   << " setting to Wc";
      vioParameters_.publishing.velocitiesFrame = FrameName::Wc;
    }
  }

  parseInputData(file["input_data"], &vioParameters_.input);
  parseInitialState(file["initial_state"], &vioParameters_.initialState);

  // camera calibration
  std::vector<CameraCalibration,
      Eigen::aligned_allocator<CameraCalibration>> calibrations;
  if (!getCameraCalibration(calibrations, file))
    LOG(FATAL) << "Did not find any calibration!";

  size_t camIdx = 0;
  for (size_t i = 0; i < calibrations.size(); ++i) {
    std::shared_ptr<const okvis::kinematics::Transformation> T_SC_okvis_ptr(
        new okvis::kinematics::Transformation(
            calibrations[i].T_SC.r(), calibrations[i].T_SC.q().normalized()));

    if (strcmp(calibrations[i].distortionType.c_str(), "equidistant") == 0) {
      vioParameters_.nCameraSystem.addCamera(
          T_SC_okvis_ptr,
          std::shared_ptr<const okvis::cameras::CameraBase>(
              new okvis::cameras::PinholeCamera<
                  okvis::cameras::EquidistantDistortion>(
                  calibrations[i].imageDimension[0],
                  calibrations[i].imageDimension[1],
                  calibrations[i].focalLength[0],
                  calibrations[i].focalLength[1],
                  calibrations[i].principalPoint[0],
                  calibrations[i].principalPoint[1],
                  okvis::cameras::EquidistantDistortion(
                    calibrations[i].distortionCoefficients[0],
                    calibrations[i].distortionCoefficients[1],
                    calibrations[i].distortionCoefficients[2],
                    calibrations[i].distortionCoefficients[3])/*, id ?*/)),
          okvis::cameras::NCameraSystem::Equidistant/*, computeOverlaps ?*/);
      std::stringstream s;
      s << calibrations[i].T_SC.T();
      LOG(INFO) << "Equidistant pinhole camera " << camIdx
                << " with T_SC=\n" << s.str();
    } else if (
        strcmp(calibrations[i].distortionType.c_str(), "radialtangential") == 0
        || strcmp(calibrations[i].distortionType.c_str(), "plumb_bob") == 0) {
      vioParameters_.nCameraSystem.addCamera(
          T_SC_okvis_ptr,
          std::shared_ptr<const okvis::cameras::CameraBase>(
              new okvis::cameras::PinholeCamera<
                  okvis::cameras::RadialTangentialDistortion>(
                  calibrations[i].imageDimension[0],
                  calibrations[i].imageDimension[1],
                  calibrations[i].focalLength[0],
                  calibrations[i].focalLength[1],
                  calibrations[i].principalPoint[0],
                  calibrations[i].principalPoint[1],
                  okvis::cameras::RadialTangentialDistortion(
                      calibrations[i].distortionCoefficients[0],
                      calibrations[i].distortionCoefficients[1],
                      calibrations[i].distortionCoefficients[2],
                      calibrations[i].distortionCoefficients[3])/*, id ?*/)),
          okvis::cameras::NCameraSystem::RadialTangential
          /*, computeOverlaps ?*/);
      std::stringstream s;
      s << calibrations[i].T_SC.T();
      LOG(INFO) << "Radial tangential pinhole camera " << camIdx
                << " with T_SC=\n" << s.str();
    } else if (
        strcmp(calibrations[i].distortionType.c_str(), "radialtangential8") == 0
        || strcmp(calibrations[i].distortionType.c_str(), "plumb_bob8") == 0) {
      vioParameters_.nCameraSystem.addCamera(
          T_SC_okvis_ptr,
          std::shared_ptr<const okvis::cameras::CameraBase>(
              new okvis::cameras::PinholeCamera<
                  okvis::cameras::RadialTangentialDistortion8>(
                  calibrations[i].imageDimension[0],
                  calibrations[i].imageDimension[1],
                  calibrations[i].focalLength[0],
                  calibrations[i].focalLength[1],
                  calibrations[i].principalPoint[0],
                  calibrations[i].principalPoint[1],
                  okvis::cameras::RadialTangentialDistortion8(
                      calibrations[i].distortionCoefficients[0],
                      calibrations[i].distortionCoefficients[1],
                      calibrations[i].distortionCoefficients[2],
                      calibrations[i].distortionCoefficients[3],
                      calibrations[i].distortionCoefficients[4],
                      calibrations[i].distortionCoefficients[5],
                      calibrations[i].distortionCoefficients[6],
                      calibrations[i].distortionCoefficients[7])/*, id ?*/)),
          okvis::cameras::NCameraSystem::RadialTangential8
          /*, computeOverlaps ?*/);
      std::stringstream s;
      s << calibrations[i].T_SC.T();
      LOG(INFO) << "Radial tangential 8 pinhole camera " << camIdx
                << " with T_SC=\n" << s.str();
    } else {
      LOG(ERROR) << "unrecognized distortion type "
                 << calibrations[i].distortionType;
    }
    ++camIdx;
  }

  vioParameters_.sensors_information.imuIdx = 0;

  cv::FileNode T_BS_ = file["imu_params"]["T_BS"];
  OKVIS_ASSERT_TRUE(
      Exception,
      T_BS_.isSeq(),
      "'T_BS' parameter missing in the configuration file or "
      "in the wrong format.")

  Eigen::Matrix4d T_BS_e;
  T_BS_e << T_BS_[0], T_BS_[1], T_BS_[2], T_BS_[3],
            T_BS_[4], T_BS_[5], T_BS_[6], T_BS_[7],
            T_BS_[8], T_BS_[9], T_BS_[10], T_BS_[11],
            T_BS_[12], T_BS_[13], T_BS_[14], T_BS_[15];

  vioParameters_.imu.T_BS = okvis::kinematics::Transformation(T_BS_e);
  std::stringstream s;
  s << vioParameters_.imu.T_BS.T();
  LOG(INFO) << "IMU with transformation T_BS=\n" << s.str();

  // the IMU parameters
  cv::FileNode imu_params = file["imu_params"];
  OKVIS_ASSERT_TRUE(
      Exception, imu_params["a_max"].isReal(),
      "'imu_params: a_max' parameter missing in configuration file.");
  OKVIS_ASSERT_TRUE(
      Exception, imu_params["g_max"].isReal(),
      "'imu_params: g_max' parameter missing in configuration file.");
  OKVIS_ASSERT_TRUE(
      Exception, imu_params["sigma_g_c"].isReal(),
      "'imu_params: sigma_g_c' parameter missing in configuration file.");
  OKVIS_ASSERT_TRUE(
      Exception, imu_params["sigma_a_c"].isReal(),
      "'imu_params: sigma_a_c' parameter missing in configuration file.");
  OKVIS_ASSERT_TRUE(
       Exception, imu_params["sigma_bg"].isReal(),
       "'imu_params: sigma_bg' parameter missing in configuration file.");
  OKVIS_ASSERT_TRUE(
       Exception, imu_params["sigma_ba"].isReal(),
       "'imu_params: sigma_ba' parameter missing in configuration file.");
  OKVIS_ASSERT_TRUE(
      Exception, imu_params["sigma_gw_c"].isReal(),
      "'imu_params: sigma_gw_c' parameter missing in configuration file.");
  OKVIS_ASSERT_TRUE(
      Exception, imu_params["sigma_g_c"].isReal(),
      "'imu_params: sigma_g_c' parameter missing in configuration file.");
  OKVIS_ASSERT_TRUE(
      Exception, imu_params["tau"].isReal(),
      "'imu_params: tau' parameter missing in configuration file.");
  OKVIS_ASSERT_TRUE(Exception, imu_params["g"].isReal(),
                    "'imu_params: g' parameter missing in configuration file.");
  OKVIS_ASSERT_TRUE(Exception, imu_params["a0"].isSeq(),
                    "'imu_params: a0' parameter missing in "
                    "configuration file.");
  OKVIS_ASSERT_TRUE(
      Exception, imu_params["imu_rate"].isInt(),
      "'imu_params: imu_rate' parameter missing in configuration file.");

  OKVIS_ASSERT_TRUE(
       Exception, imu_params["sigma_TGElement"].isReal(),
       "'imu_params: sigma_TGElement' parameter missing in "
       "configuration file.");
  OKVIS_ASSERT_TRUE(
       Exception, imu_params["sigma_TSElement"].isReal(),
       "'imu_params: sigma_TSElement' parameter missing in "
       "configuration file.");
  OKVIS_ASSERT_TRUE(
       Exception, imu_params["sigma_TAElement"].isReal(),
       "'imu_params: sigma_TAElement' parameter missing in "
       "configuration file.");

  imu_params["a_max"] >> vioParameters_.imu.a_max;
  imu_params["g_max"] >> vioParameters_.imu.g_max;
  imu_params["sigma_g_c"] >> vioParameters_.imu.sigma_g_c;
  imu_params["sigma_a_c"] >> vioParameters_.imu.sigma_a_c;
  imu_params["sigma_bg"] >> vioParameters_.imu.sigma_bg;
  imu_params["sigma_ba"] >> vioParameters_.imu.sigma_ba;
  imu_params["sigma_gw_c"] >> vioParameters_.imu.sigma_gw_c;
  imu_params["sigma_aw_c"] >> vioParameters_.imu.sigma_aw_c;
  imu_params["imu_rate"] >> vioParameters_.imu.rate;
  imu_params["tau"] >> vioParameters_.imu.tau;
  imu_params["g"] >> vioParameters_.imu.g;

  vioParameters_.imu.a0 =
      Eigen::Vector3d(static_cast<double>(imu_params["a0"][0]),
                      static_cast<double>(imu_params["a0"][1]),
                      static_cast<double>(imu_params["a0"][2]));

  cv::FileNode initGyroBias = file["imu_params"]["g0"];
  if (initGyroBias.isSeq()) {
      Eigen::Vector3d g0;
      g0 << initGyroBias[0], initGyroBias[1], initGyroBias[2];
      vioParameters_.imu.g0 = g0;
  } else {
    vioParameters_.imu.g0 = Eigen::Vector3d::Zero();
  }

  imu_params["sigma_TGElement"] >> vioParameters_.imu.sigma_TGElement;
  imu_params["sigma_TSElement"] >> vioParameters_.imu.sigma_TSElement;
  imu_params["sigma_TAElement"] >> vioParameters_.imu.sigma_TAElement;

  cv::FileNode initTg = file["imu_params"]["Tg0"];
  if (initTg.isSeq()) {
    Eigen::Matrix<double, 9, 1> Tg;
    Tg << initTg[0], initTg[1], initTg[2], initTg[3], initTg[4], initTg[5],
        initTg[6], initTg[7], initTg[8];
    vioParameters_.imu.Tg0 = Tg;
  } else {
    vioParameters_.imu.Tg0 << 1, 0, 0, 0, 1, 0, 0, 0, 1;
  }

  cv::FileNode initTs = file["imu_params"]["Ts0"];
  if (initTs.isSeq()) {
    Eigen::Matrix<double, 9, 1> Ts;
    Ts << initTs[0], initTs[1], initTs[2], initTs[3], initTs[4], initTs[5],
        initTs[6], initTs[7], initTs[8];
    vioParameters_.imu.Ts0 = Ts;
  } else {
    vioParameters_.imu.Ts0.setZero();
  }

  cv::FileNode initTa = file["imu_params"]["Ta0"];
  if (initTa.isSeq()) {
    Eigen::Matrix<double, 9, 1> Ta;
    Ta << initTa[0], initTa[1], initTa[2], initTa[3], initTa[4], initTa[5],
        initTa[6], initTa[7], initTa[8];
    vioParameters_.imu.Ta0 = Ta;
  } else {
    vioParameters_.imu.Ta0 << 1, 0, 0, 0, 1, 0, 0, 0, 1;
  }
  vioParameters_.imu.td0 = 0;
  readConfigFile_ = true;
}

// Parses booleans from a cv::FileNode. OpenCV sadly has no implementation
// like this.
bool HybridVioParametersReader::parseBoolean(
    cv::FileNode node, bool& val) const {
  if (node.isInt()) {
    val = static_cast<int>(node) != 0;
    return true;
  }
  if (node.isString()) {
    std::string str = (std::string) (node);
    // cut out first word. str currently contains everything including comments
    str = str.substr(0, str.find(" "));
    // transform it to all lowercase
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    /* from yaml.org/type/bool.html:
     * Booleans are formatted as English words
     * (“true”/“false”, “yes”/“no” or “on”/“off”)
     * for readability and may be abbreviated as
     * a single character “y”/“n” or “Y”/“N”. */
    if (str.compare("false")  == 0
        || str.compare("no")  == 0
        || str.compare("n")   == 0
        || str.compare("off") == 0) {
      val = false;
      return true;
    }
    if (str.compare("true")   == 0
        || str.compare("yes") == 0
        || str.compare("y")   == 0
        || str.compare("on")  == 0) {
      val = true;
      return true;
    }
  }
  return false;
}

bool HybridVioParametersReader::getCameraCalibration(
    std::vector<CameraCalibration,
        Eigen::aligned_allocator<CameraCalibration>> & calibrations,
    cv::FileStorage& configurationFile) {
  bool success = getCalibrationViaConfig(
      calibrations, configurationFile["cameras"]);

#ifdef HAVE_LIBVISENSOR
  if (useDriver && !success) {
    // start up sensor
    viSensor = std::shared_ptr<void>(
          new visensor::ViSensorDriver());
    try {
      // use autodiscovery to find sensor. TODO(jhuai): specify IP in config?
      std::static_pointer_cast<visensor::ViSensorDriver>(viSensor)->init();
    } catch (Exception const &ex) {
      LOG(ERROR) << ex.what();
      exit(1);
    }

    success = getCalibrationViaVisensorAPI(calibrations);
  }
#endif

  return success;
}

void HybridVioParametersReader::print(
    const HybridVioParametersReader::CameraCalibration & cc) const {
     LOG(INFO) << "T_SC " << std::endl << cc.T_SC.T3x4() << std::endl
               << "image dimension after downscale "
               << cc.imageDimension.transpose() << " distortion type "
               << cc.distortionType.c_str() << std::endl << " distortion "
               << cc.distortionCoefficients.transpose() << std::endl
               << " focal " << cc.focalLength.transpose()
               << " principalpoint " << cc.principalPoint.transpose();
}

// Get the camera calibration via the configuration file.
bool HybridVioParametersReader::getCalibrationViaConfig(
    std::vector<CameraCalibration,
        Eigen::aligned_allocator<CameraCalibration>> & calibrations,
    cv::FileNode cameraNode) const {
  calibrations.clear();
  bool gotCalibration = false;
  // first check if calibration is available in config file
  if (cameraNode.isSeq()
     && cameraNode.size() > 0) {
    size_t camIdx = 0;
    for (cv::FileNodeIterator it = cameraNode.begin();
        it != cameraNode.end(); ++it) {
      if ((*it).isMap()
          && (*it)["T_SC"].isSeq()
          && (*it)["image_dimension"].isSeq()
          && (*it)["image_dimension"].size() == 2
          && (*it)["distortion_coefficients"].isSeq()
          && (*it)["distortion_coefficients"].size() >= 4
          && (*it)["distortion_type"].isString()
          && (*it)["focal_length"].isSeq()
          && (*it)["focal_length"].size() == 2
          && (*it)["principal_point"].isSeq()
          && (*it)["principal_point"].size() == 2) {
        LOG(INFO) << "Found calibration in configuration file"
                  << " for camera " << camIdx;
        gotCalibration = true;
      } else {
        LOG(WARNING) << "Found incomplete calibration in configuration "
                     << "file for camera " << camIdx
                     << ". Will not use the calibration from the "
                     << "configuration file.";
        return false;
      }
      ++camIdx;
    }
  } else {
    LOG(INFO) << "Did not find a calibration in the configuration file.";
  }

  if (gotCalibration) {
    for (cv::FileNodeIterator it = cameraNode.begin();
        it != cameraNode.end(); ++it) {
      CameraCalibration calib;
      cv::FileNode T_SC_node = (*it)["T_SC"];
      int downScale = 1;
      if ((*it)["down_scale"].isInt())
        downScale = (*it)["down_scale"];
      cv::FileNode imageDimensionNode = (*it)["image_dimension"];
      cv::FileNode distortionCoefficientNode = (*it)["distortion_coefficients"];
      cv::FileNode focalLengthNode = (*it)["focal_length"];
      cv::FileNode principalPointNode = (*it)["principal_point"];

      // extrinsics
      Eigen::Matrix4d T_SC;
      T_SC << T_SC_node[0], T_SC_node[1], T_SC_node[2], T_SC_node[3],
              T_SC_node[4], T_SC_node[5], T_SC_node[6], T_SC_node[7],
              T_SC_node[8], T_SC_node[9], T_SC_node[10], T_SC_node[11],
              T_SC_node[12], T_SC_node[13], T_SC_node[14], T_SC_node[15];
      calib.T_SC = okvis::kinematics::Transformation(T_SC);

      calib.imageDimension << static_cast<double>(imageDimensionNode[0]) /
                                  downScale,
          static_cast<double>(imageDimensionNode[1]) / downScale;
      calib.distortionCoefficients.resize(distortionCoefficientNode.size());
      for (size_t i = 0; i < distortionCoefficientNode.size(); ++i) {
        calib.distortionCoefficients[i] = distortionCoefficientNode[i];
      }
      calib.focalLength << static_cast<double>(focalLengthNode[0]) / downScale,
          static_cast<double>(focalLengthNode[1]) / downScale;
      calib.principalPoint << static_cast<double>(principalPointNode[0]) /
                                  downScale,
          static_cast<double>(principalPointNode[1]) / downScale;
      calib.distortionType = static_cast<std::string>((*it)["distortion_type"]);

      calibrations.push_back(calib);
      print(calib);
    }
  }
  return gotCalibration;
}

// Get the camera calibrations via the visensor API.
bool HybridVioParametersReader::getCalibrationViaVisensorAPI(
    std::vector<CameraCalibration, Eigen::aligned_allocator<CameraCalibration>>
        &calibrations) const {
#ifdef HAVE_LIBVISENSOR
  if (viSensor == nullptr) {
    LOG(ERROR) << "Tried to get calibration from the sensor. "
               << "But the sensor is not set up.";
    return false;
  }

  calibrations.clear();

  std::vector<visensor::SensorId::SensorId> listOfCameraIds =
      std::static_pointer_cast<visensor::ViSensorDriver>(viSensor)->
          getListOfCameraIDs();

  for (auto it = listOfCameraIds.begin(); it != listOfCameraIds.end(); ++it) {
    visensor::ViCameraCalibration calibrationFromAPI;
    okvis::HybridVioParametersReader::CameraCalibration calibration;
    if (!std::static_pointer_cast<visensor::ViSensorDriver>(viSensor)
             ->getCameraCalibration(*it, calibrationFromAPI)) {
      LOG(ERROR) << "Reading the calibration via the sensor API failed.";
      calibrations.clear();
      return false;
    }
    LOG(INFO) << "Reading the calbration for camera " << size_t(*it)
              << " via API successful";
    double* R = calibrationFromAPI.R;
    double* t = calibrationFromAPI.t;
    // getCameraCalibration apparently gives T_CI back.
    // (Confirmed by comparing it to output of service)
    Eigen::Matrix4d T_CI;
    T_CI << R[0], R[1], R[2], t[0],
            R[3], R[4], R[5], t[1],
            R[6], R[7], R[8], t[2],
            0,    0,    0,    1;
    okvis::kinematics::Transformation T_CI_okvis(T_CI);
    calibration.T_SC = T_CI_okvis.inverse();

    calibration.focalLength << calibrationFromAPI.focal_point[0],
                               calibrationFromAPI.focal_point[1];
    calibration.principalPoint << calibrationFromAPI.principal_point[0],
                                  calibrationFromAPI.principal_point[1];
    calibration.distortionCoefficients.resize(4); // FIXME: 8 coeff support?
    calibration.distortionCoefficients << calibrationFromAPI.dist_coeff[0],
                                          calibrationFromAPI.dist_coeff[1],
                                          calibrationFromAPI.dist_coeff[2],
                                          calibrationFromAPI.dist_coeff[3];
    calibration.imageDimension << 752, 480;
    calibration.distortionType = "plumb_bob";
    calibrations.push_back(calibration);
  }

  return calibrations.empty() == false;
#else
  static_cast<void>(calibrations); // unused
  LOG(ERROR) << "Tried to get calibration directly from the sensor. "
             << "However libvisensor was not found.";
  return false;
#endif
}
}  // namespace okvis
