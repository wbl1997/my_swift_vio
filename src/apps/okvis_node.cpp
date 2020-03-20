/**
 * @file okvis_node.cpp
 * @brief This file includes the ROS node implementation.
 * @author Jianzhu Huai
 */

#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <ros/ros.h>
#pragma GCC diagnostic pop
#include <image_transport/image_transport.h>
#include "sensor_msgs/Imu.h"

#include <glog/logging.h>

#include <okvis/Parameters.hpp>
#include <okvis/ThreadedKFVio.hpp>

#include <io_wrap/CommonGflags.hpp>
#include <io_wrap/StreamHelper.hpp>
#include <io_wrap/Player.hpp>
#include <io_wrap/Publisher.hpp>
#include <io_wrap/RosParametersReader.hpp>
#include <io_wrap/Subscriber.hpp>
#include <msckf/HybridVio.hpp>

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);  // true to strip gflags
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_stderrthreshold = 0;  // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
  FLAGS_colorlogtostderr = 1;

  ros::init(argc, argv, "okvis_node");
  ros::NodeHandle nh("okvis_node");
  okvis::Publisher publisher(nh);

  std::string configFilename;
  if (argc >= 2) {
    configFilename = argv[1];
  } else {
    std::cout << "You can either invoke okvis_node through a ros launch file,"
                 " or through Qt debug. "
              << "In the latter case, you either need to provide the"
                 " config_filename in the command line,"
              << " or use rosparam e.g.," << std::endl
              << "rosparam set /okvis_node/config_filename "
              << "/path/to/config/config_fpga_p2_euroc.yaml" << std::endl;
    std::cout << "To run msckf on image sequences or a video and their "
                 "associated inertial data, "
              << std::endl
              << "set load_input_option properly as an input argument, then"
                 " in a terminal, input "
              << std::endl
              << argv[0] << " /path/to/config/file.yaml" << std::endl;
    std::cout << "Set publishing_options.publishImuPropagatedState to false "
                 "in the settings.yaml to only save optimized states"
              << std::endl;

    if (!nh.getParam("config_filename", configFilename)) {
      LOG(ERROR) << "Please specify filename of configuration!";
      return 1;
    }
  }
  okvis::RosParametersReader vio_parameters_reader(configFilename);
  okvis::VioParameters parameters;
  vio_parameters_reader.getParameters(parameters);
  okvis::setInputParameters(&parameters.input);

  std::shared_ptr<okvis::VioInterface> okvis_estimator;
  switch (parameters.optimization.algorithm) {
    case okvis::EstimatorAlgorithm::OKVIS:
    case okvis::EstimatorAlgorithm::General:
    case okvis::EstimatorAlgorithm::Priorless:
      // http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1049
      okvis_estimator.reset(new okvis::ThreadedKFVio(parameters));
      break;
    case okvis::EstimatorAlgorithm::MSCKF:
    case okvis::EstimatorAlgorithm::TFVIO:
      okvis_estimator.reset(new okvis::HybridVio(parameters));
      break;
    default:
      LOG(ERROR) << "Estimator not implemented!";
      return 1;
  }

  std::string path = FLAGS_output_dir;
  path = okvis::removeTrailingSlash(path);
  int camIdx = 0;

  okvis_estimator->setFullStateCallback(
      std::bind(&okvis::Publisher::publishFullStateAsCallback, &publisher,
                std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3, std::placeholders::_4));
  okvis_estimator->setLandmarksCallback(std::bind(
      &okvis::Publisher::publishLandmarksAsCallback, &publisher,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

  std::string stateFilename = path + "/msckf_estimates.csv";
  std::string headerLine;
  okvis::StreamHelper::composeHeaderLine(
      parameters.imu.model_type, parameters.nCameraSystem.projOptRep(camIdx),
      parameters.nCameraSystem.extrinsicOptRep(camIdx),
      parameters.nCameraSystem.cameraGeometry(camIdx)->distortionType(),
      okvis::FULL_STATE_WITH_ALL_CALIBRATION, &headerLine);
  publisher.setCsvFile(stateFilename, headerLine);
  if (FLAGS_dump_output_option == 2) {
    // save estimates of evolving states, and camera extrinsics
    okvis_estimator->setFullStateCallbackWithExtrinsics(std::bind(
        &okvis::Publisher::csvSaveFullStateWithExtrinsicsAsCallback, &publisher,
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
        std::placeholders::_4, std::placeholders::_5, std::placeholders::_6));
  } else if (FLAGS_dump_output_option == 3 || FLAGS_dump_output_option == 4) {
    // save estimates of evolving states, camera extrinsics,
    // and all other calibration parameters
    okvis_estimator->setFullStateCallbackWithAllCalibration(std::bind(
        &okvis::Publisher::csvSaveFullStateWithAllCalibrationAsCallback,
        &publisher, std::placeholders::_1, std::placeholders::_2,
        std::placeholders::_3, std::placeholders::_4, std::placeholders::_5,
        std::placeholders::_6, std::placeholders::_7, std::placeholders::_8,
        std::placeholders::_9, std::placeholders::_10));
    if (FLAGS_dump_output_option == 4) {
      okvis_estimator->setImuCsvFile(path + "/imu0_data.csv");
      const unsigned int numCameras = parameters.nCameraSystem.numCameras();
      for (size_t i = 0; i < numCameras; ++i) {
        std::stringstream num;
        num << i;
        okvis_estimator->setTracksCsvFile(
            i, path + "/cam" + num.str() + "_tracks.csv");
      }
      publisher.setLandmarksCsvFile(path + "/okvis_estimator_landmarks.csv");
    }
  }

  okvis_estimator->setStateCallback(
      std::bind(&okvis::Publisher::publishStateAsCallback, &publisher,
                std::placeholders::_1, std::placeholders::_2));
  publisher.setParameters(parameters);  // pass the specified publishing stuff

  // player to grab messages directly from files on a hard drive
  std::shared_ptr<okvis::Player> pPlayer;
  std::shared_ptr<std::thread> ptPlayer;
  if (FLAGS_load_input_option == 0) {
    okvis::Subscriber subscriber(nh, okvis_estimator.get(),
                                 vio_parameters_reader);
    ros::Rate rate(20);
    while (ros::ok()) {
      ros::spinOnce();
      okvis_estimator->display();
      rate.sleep();
    }
  } else {
    pPlayer.reset(new okvis::Player(okvis_estimator.get(), parameters));
    ptPlayer.reset(new std::thread(&okvis::Player::Run, std::ref(*pPlayer)));

    ros::Rate rate(20);
    while (!pPlayer->mbFinished) {
      okvis_estimator->display();
      rate.sleep();
    }
    ptPlayer->join();
    std::this_thread::sleep_for(
        std::chrono::seconds(5));  // in case the optimizer lags
  }

  std::string filename = path + "/feature_statistics.txt";
  okvis_estimator->saveStatistics(filename);
  return 0;
}
