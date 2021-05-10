#include "VioSystemWrap.hpp"

#include <io_wrap/CommonGflags.hpp>
#include <io_wrap/StreamHelper.hpp>

namespace okvis {
void VioSystemWrap::registerCallbacks(
    const std::string& output_dir, const okvis::VioParameters& parameters,
    okvis::ThreadedKFVio* vioSystem, okvis::StreamPublisher* publisher,
    okvis::PgoPublisher* pgoPublisher) {
  std::string path = okvis::removeTrailingSlash(output_dir);

  vioSystem->setFullStateCallback(
      std::bind(&okvis::StreamPublisher::publishFullStateAsCallback, publisher,
                std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3, std::placeholders::_4));
  vioSystem->setLandmarksCallback(std::bind(
      &okvis::StreamPublisher::publishLandmarksAsCallback, publisher,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

  std::string stateFilename = path + "/swift_vio.csv";
  std::string headerLine;
  size_t numCameras = parameters.nCameraSystem.numCameras();
  std::vector<std::string> extrinsicParamRepList(numCameras);
  std::vector<std::string> projectionParamRepList(numCameras);
  std::vector<std::string> distortionParamRepList(numCameras);
  for (size_t camIdx = 0; camIdx < numCameras; ++camIdx) {
    extrinsicParamRepList[camIdx] =
        parameters.nCameraSystem.extrinsicOptRep(camIdx);
    projectionParamRepList[camIdx] = parameters.nCameraSystem.projOptRep(camIdx);
    distortionParamRepList[camIdx] =
        parameters.nCameraSystem.cameraGeometry(camIdx)->distortionType();
  }

  okvis::StreamHelper::composeHeaderLine(
      parameters.imu.model_type, extrinsicParamRepList, projectionParamRepList,
      distortionParamRepList, okvis::FULL_STATE_WITH_ALL_CALIBRATION,
      &headerLine);
  publisher->setCsvFile(stateFilename, headerLine);
  if (FLAGS_dump_output_option == 2) {
    // save estimates of evolving states, and camera extrinsics
    vioSystem->setFullStateCallbackWithExtrinsics(std::bind(
        &okvis::StreamPublisher::csvSaveFullStateWithExtrinsicsAsCallback, publisher,
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
        std::placeholders::_4, std::placeholders::_5, std::placeholders::_6));
  } else if (FLAGS_dump_output_option == 3 || FLAGS_dump_output_option == 4) {
    // save estimates of evolving states, camera extrinsics,
    // and all other calibration parameters
    vioSystem->setFullStateCallbackWithAllCalibration(std::bind(
        &okvis::StreamPublisher::csvSaveFullStateWithAllCalibrationAsCallback,
        publisher, std::placeholders::_1, std::placeholders::_2,
        std::placeholders::_3, std::placeholders::_4, std::placeholders::_5,
        std::placeholders::_6, std::placeholders::_7, std::placeholders::_8,
        std::placeholders::_9, std::placeholders::_10));
    if (FLAGS_dump_output_option == 4) {
      vioSystem->setImuCsvFile(path + "/imu0_data.csv");
      const unsigned int numCameras = parameters.nCameraSystem.numCameras();
      for (size_t i = 0; i < numCameras; ++i) {
        std::stringstream num;
        num << i;
        vioSystem->setTracksCsvFile(i,
                                    path + "/cam" + num.str() + "_tracks.csv");
      }
      publisher->setLandmarksCsvFile(path + "/vioSystem_landmarks.csv");
    }
  }
  vioSystem->setStateCallback(
      std::bind(&okvis::StreamPublisher::publishStateAsCallback, publisher,
                std::placeholders::_1, std::placeholders::_2));

  pgoPublisher->setCsvFile(path + "/online_pgo.csv");
  vioSystem->appendPgoStateCallback(
      std::bind(&okvis::PgoPublisher::csvSaveStateAsCallback, pgoPublisher,
                std::placeholders::_1, std::placeholders::_2));
}
}  // namespace okvis
