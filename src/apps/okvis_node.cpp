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

#include <okvis/ThreadedKFVio.hpp>

#include <io_wrap/CommonGflags.hpp>
#include <io_wrap/Player.hpp>
#include <io_wrap/RosParametersReader.hpp>
#include <io_wrap/Subscriber.hpp>
#include <loop_closure/LoopClosureDetectorParams.h>
#include <msckf/VioFactoryMethods.hpp>
#include "VioSystemWrap.hpp"

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);  // true to strip gflags
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_stderrthreshold = 0;  // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
  FLAGS_colorlogtostderr = 1;

  const std::string nodeName = "okvis_node";
  ros::init(argc, argv, nodeName);
  ros::NodeHandle nh(nodeName);


  std::string configFilename;
  if (argc >= 2) {
    configFilename = argv[1];
  } else {
    if (!nh.getParam("config_filename", configFilename)) {
      LOG(ERROR) << "Please specify filename of configuration!";
      return 1;
    }
  }

  okvis::RosParametersReader vio_parameters_reader(configFilename);
  okvis::VioParameters parameters;
  vio_parameters_reader.getParameters(parameters);
  okvis::setInputParameters(&parameters.input);

  std::shared_ptr<okvis::Estimator> estimator =
      msckf::createBackend(parameters.optimization.algorithm);
  std::shared_ptr<okvis::Frontend> frontend = msckf::createFrontend(
      parameters.nCameraSystem.numCameras(),
      parameters.optimization.initializeWithoutEnoughParallax,
      parameters.optimization.algorithm);
  std::shared_ptr<VIO::LoopClosureDetectorParams> lcParams(
        new VIO::LoopClosureDetectorParams());
  std::shared_ptr<okvis::LoopClosureMethod> loopClosureMethod =
      msckf::createLoopClosureMethod(VIO::LoopClosureMethodType::OrbBoW, lcParams);
  okvis::ThreadedKFVio okvis_estimator(parameters, estimator, frontend,
                                       loopClosureMethod);

  okvis::Publisher publisher(nh);
  publisher.setParameters(parameters);
  okvis::PgoPublisher pgoPublisher;
  okvis::VioSystemWrap::registerCallbacks(
      FLAGS_output_dir, parameters, &okvis_estimator, &publisher,
      &pgoPublisher);

  // player to grab messages directly from files on a hard drive.
  std::shared_ptr<okvis::Player> pPlayer;
  std::shared_ptr<std::thread> ptPlayer;
  if (FLAGS_load_input_option == 0) {
    okvis::Subscriber subscriber(nh, &okvis_estimator,
                                 vio_parameters_reader);
    ros::Rate rate(20);
    while (ros::ok()) {
      ros::spinOnce();
      okvis_estimator.display();
      rate.sleep();
    }
  } else {
    pPlayer.reset(new okvis::Player(&okvis_estimator, parameters));
    ptPlayer.reset(new std::thread(&okvis::Player::Run, std::ref(*pPlayer)));

    ros::Rate rate(20);
    while (!pPlayer->mbFinished) {
      okvis_estimator.display();
      rate.sleep();
    }
    ptPlayer->join();
    std::this_thread::sleep_for(
        std::chrono::seconds(5));  // in case the optimizer lags
  }

  std::string filename =
      okvis::removeTrailingSlash(FLAGS_output_dir) + "/feature_statistics.txt";
  okvis_estimator.saveStatistics(filename);
  return 0;
}
