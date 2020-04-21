#include "VioSystemWrap.hpp"

#include <io_wrap/CommonGflags.hpp>
#include <io_wrap/StreamHelper.hpp>

namespace okvis {
void VioSystemWrap::registerCallbacks(
    const std::string& output_dir, const okvis::VioParameters& parameters,
    okvis::ThreadedKFVio* vioSystem, okvis::Publisher* publisher,
    okvis::PgoPublisher* pgoPublisher) {
  std::string path = okvis::removeTrailingSlash(output_dir);
  int camIdx = 0;
  vioSystem->setFullStateCallback(
      std::bind(&okvis::Publisher::publishFullStateAsCallback, publisher,
                std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3, std::placeholders::_4));
  vioSystem->setLandmarksCallback(std::bind(
      &okvis::Publisher::publishLandmarksAsCallback, publisher,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

  std::string stateFilename = path + "/msckf_estimates.csv";
  std::string headerLine;
  okvis::StreamHelper::composeHeaderLine(
      parameters.imu.model_type, parameters.nCameraSystem.projOptRep(camIdx),
      parameters.nCameraSystem.extrinsicOptRep(camIdx),
      parameters.nCameraSystem.cameraGeometry(camIdx)->distortionType(),
      okvis::FULL_STATE_WITH_ALL_CALIBRATION, &headerLine);
  publisher->setCsvFile(stateFilename, headerLine);
  if (FLAGS_dump_output_option == 2) {
    // save estimates of evolving states, and camera extrinsics
    vioSystem->setFullStateCallbackWithExtrinsics(std::bind(
        &okvis::Publisher::csvSaveFullStateWithExtrinsicsAsCallback, publisher,
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
        std::placeholders::_4, std::placeholders::_5, std::placeholders::_6));
  } else if (FLAGS_dump_output_option == 3 || FLAGS_dump_output_option == 4) {
    // save estimates of evolving states, camera extrinsics,
    // and all other calibration parameters
    vioSystem->setFullStateCallbackWithAllCalibration(std::bind(
        &okvis::Publisher::csvSaveFullStateWithAllCalibrationAsCallback,
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
      std::bind(&okvis::Publisher::publishStateAsCallback, publisher,
                std::placeholders::_1, std::placeholders::_2));

  pgoPublisher->setCsvFile(path + "/online_pgo.csv");
  vioSystem->appendPgoStateCallback(
      std::bind(&okvis::PgoPublisher::csvSaveStateAsCallback, pgoPublisher,
                std::placeholders::_1, std::placeholders::_2));
}
}  // namespace okvis
