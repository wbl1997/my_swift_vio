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
#include <gflags/gflags.h>

#include <okvis/Subscriber.hpp>
#include <okvis/Publisher.hpp>
#include <okvis/RosParametersReader.hpp>
#include <okvis/HybridVio.hpp>

#include <okvis/Player.hpp>

DEFINE_int32(dump_output_option, 3,
    "0, direct results to ROS publishers, other options save results to csvs"
    "1, save states, 2, save states and camera extrinsics, "
    "3, save states, and all calibration parameters, 4, save states,"
    "all calibration parameters, feature tracks, and landmarks");

DEFINE_int32(load_input_option, 1,
    "0, get input by subscribing to ros topics"
    "1, get input by reading files on a hard drive");

DEFINE_string(output_dir, "", "the directory to dump results");

DEFINE_string(image_folder, "", "folder of an input image sequence");

DEFINE_string(video_file, "", "full name of an input video file");

DEFINE_string(time_file, "",
              "full name of an input time file containing"
              " timestamps for each image seq. frame");

DEFINE_string(imu_file, "", "full name of an input IMU file");

DEFINE_string(vo_poses_file, "", "full name of external VO poses");

DEFINE_string(vo_feature_tracks_file, "",
              "full name of external VO output feature tracks");

DEFINE_int32(start_index, 0, "index of the first frame to be processed");

DEFINE_int32(finish_index, 0, "index of the last frame to be processed");

static void displayArgs(const int argc, char **argv) {
  std::cout << "args\n";
  for (int i = 0; i < argc; ++i) {
    std::cout << i << " " << argv[i] << std::endl;
  }
  std::cout << std::endl;
}

bool setInputParameters(okvis::InputData *input) {
  input->videoFile = FLAGS_video_file;
  input->imageFolder = FLAGS_image_folder;
  input->imuFile = FLAGS_imu_file;
  input->timeFile = FLAGS_time_file;
  input->voPosesFile = FLAGS_vo_poses_file;
  input->voFeatureTracksFile = FLAGS_vo_feature_tracks_file;
  input->startIndex = FLAGS_start_index;
  input->finishIndex = FLAGS_finish_index;
  return true;
}

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true); // true to strip gflags
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_stderrthreshold = 0; // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
  FLAGS_colorlogtostderr = 1;

  ros::init(argc, argv, "okvis_node");
  ros::NodeHandle nh("okvis_node");
  okvis::Publisher publisher(
      nh,
      okvis::Publisher::DUMP_RESULT_OPTION::FULL_STATE_WITH_ALL_CALIBRATION);

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
              << "/path/to/config/config_fpga_p2_euroc.yaml" << std::endl
              << "To remap the topics in terminal during rosbag play, "
                 "in our case, "
              << std::endl
              << "rosbag play --pause --start=0.0 --rate=1.0 MH_01_easy.bag"
                 " /cam0/image_raw:=/camera0 /imu0:=/imu"
              << std::endl;
    std::cout << "To run the tests, go to terminal," << std::endl
              << " catkin_make tests msckf2" << std::endl
              << "then to run them " << std::endl
              << "rosrun run_tests msckf2" << std::endl;
    std::cout << "To visualize it in RVIZ "
              << "rosrun rviz rviz -d config/rviz.rviz" << std::endl;
    std::cout << "To run msckf2 on image sequences or a video and their "
                 "associated inertial data, "
              << std::endl
              << "set load_input_option properly as an input argument, then"
                 " in a terminal, input "
              << std::endl
              << "msckf2 /path/to/config/file.yaml" << std::endl;
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
  setInputParameters(&parameters.input);
  okvis::HybridVio okvis_estimator(parameters);
  std::string path = FLAGS_output_dir;

  if (FLAGS_dump_output_option == 0) {
    okvis_estimator.setFullStateCallback(
        std::bind(&okvis::Publisher::publishFullStateAsCallback, &publisher,
                  std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3, std::placeholders::_4));
    okvis_estimator.setLandmarksCallback(std::bind(
        &okvis::Publisher::publishLandmarksAsCallback, &publisher,
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
  } else {
    publisher.setCsvFile(path + "/msckf2_estimator_output.csv");
    if (FLAGS_dump_output_option == 1) {
      // save estimates of evolving states: position, velocity, attitude, bg, ba
      okvis_estimator.setFullStateCallback(std::bind(
          &okvis::Publisher::csvSaveFullStateAsCallback, &publisher,
          std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
          std::placeholders::_4, std::placeholders::_5));
    } else if (FLAGS_dump_output_option == 2) {
      // save estimates of evolving states, and camera extrinsics
      okvis_estimator.setFullStateCallbackWithExtrinsics(
          std::bind(&okvis::Publisher::csvSaveFullStateWithExtrinsicsAsCallback,
                    &publisher, std::placeholders::_1, std::placeholders::_2,
                    std::placeholders::_3, std::placeholders::_4,
                    std::placeholders::_5, std::placeholders::_6));
    } else if (FLAGS_dump_output_option == 3 || FLAGS_dump_output_option == 4) {
      // save estimates of evolving states, camera extrinsics,
      // and all other calibration parameters
      okvis_estimator.setFullStateCallbackWithAllCalibration(std::bind(
          &okvis::Publisher::csvSaveFullStateWithAllCalibrationAsCallback,
          &publisher, std::placeholders::_1, std::placeholders::_2,
          std::placeholders::_3, std::placeholders::_4, std::placeholders::_5,
          std::placeholders::_6, std::placeholders::_7, std::placeholders::_8,
          std::placeholders::_9));
      if (FLAGS_dump_output_option == 4) {
        okvis_estimator.setImuCsvFile(path + "/imu0_data.csv");
        const unsigned int numCameras = parameters.nCameraSystem.numCameras();
        for (size_t i = 0; i < numCameras; ++i) {
          std::stringstream num;
          num << i;
          okvis_estimator.setTracksCsvFile(i, path + "/cam" + num.str() +
                                                  "_tracks.csv");
        }
        publisher.setLandmarksCsvFile(path + "/okvis_estimator_landmarks.csv");
        okvis_estimator.setLandmarksCallback(
            std::bind(&okvis::Publisher::csvSaveLandmarksAsCallback, &publisher,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3));
      }
    }
  }

  okvis_estimator.setStateCallback(
      std::bind(&okvis::Publisher::publishStateAsCallback, &publisher,
                std::placeholders::_1, std::placeholders::_2));
  publisher.setParameters(parameters); // pass the specified publishing stuff

  // player to grab messages directly from files on a hard drive
  std::shared_ptr<okvis::Player> pPlayer;
  std::shared_ptr<std::thread> ptPlayer;
  if (FLAGS_load_input_option == 0) {
    okvis::Subscriber subscriber(nh, &okvis_estimator, vio_parameters_reader);
    ros::Rate rate(20);
    while (ros::ok()) {
      ros::spinOnce();
      okvis_estimator.display();
      rate.sleep();
    }
  } else {
    if (parameters.input.videoFile.empty()) { // image sequence input
      pPlayer.reset(
          new okvis::Player(&okvis_estimator, parameters, std::string()));
      ptPlayer.reset(new std::thread(&okvis::Player::Run, std::ref(*pPlayer)));
    } else { // video input
      std::cout << "working with video file " << parameters.input.videoFile
                << std::endl;
      pPlayer.reset(new okvis::Player(&okvis_estimator, parameters));
      ptPlayer.reset(new std::thread(&okvis::Player::Run, std::ref(*pPlayer)));
    }
    ros::Rate rate(20);
    while (!pPlayer->mbFinished) {
      okvis_estimator.display();
      rate.sleep();
    }
    ptPlayer->join();
    std::this_thread::sleep_for(
        std::chrono::seconds(5)); // in case the optimizer lags
  }

  std::string filename = path + "/feature_statistics.txt";
  okvis_estimator.saveStatistics(filename);
  return 0;
}
