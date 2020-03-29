/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Jun 26, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file okvis_node_synchronous.cpp
 * @brief This file includes the synchronous ROS node implementation.

          This node goes through a rosbag in order and waits until all
 processing is done before adding a new message to algorithm

 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <stdlib.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>

#include <gflags/gflags.h>

#include "sensor_msgs/Imu.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/opencv.hpp>
#pragma GCC diagnostic pop

#include "rosbag/bag.h"
#include "rosbag/chunked_file.h"
#include "rosbag/view.h"

#include <io_wrap/CommonGflags.hpp>
#include <io_wrap/Player.hpp>
#include <io_wrap/Publisher.hpp>
#include <io_wrap/RosParametersReader.hpp>
#include <io_wrap/Subscriber.hpp>
#include <msckf/VioFactoryMethods.hpp>
#include <okvis/ThreadedKFVio.hpp>

DEFINE_double(skip_first_seconds, 0, "Number of seconds to skip from the beginning!");

DEFINE_string(bagname, "", "Bag filename.");

// this is just a workbench. most of the stuff here will go into the Frontend
// class.
int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);  // true to strip gflags
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_stderrthreshold = 0;  // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
  FLAGS_colorlogtostderr = 1;

  ros::init(argc, argv, "okvis_node_synchronous");
  // set up the node
  ros::NodeHandle nh("okvis_node");

  // publisher
  okvis::Publisher publisher(nh);

  std::string configFilename;
  if (argc >= 2) {
    configFilename = argv[1];
  } else {
    std::cout << "You can either invoke okvis_node_synchronous through a ros launch file,"
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
              << argv[0] <<" /path/to/config/file.yaml" << std::endl;
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

  std::shared_ptr<okvis::Estimator> estimator =
      msckf::createBackend(parameters.optimization.algorithm);
  std::shared_ptr<okvis::Frontend> frontend = msckf::createFrontend(
      parameters.nCameraSystem.numCameras(),
      parameters.optimization.initializeWithoutEnoughParallax,
      parameters.optimization.algorithm);
  okvis::ThreadedKFVio okvis_estimator(parameters, estimator, frontend);

  std::string path = FLAGS_output_dir;
  path = okvis::removeTrailingSlash(path);
  int camIdx = 0;

  okvis_estimator.setFullStateCallback(
      std::bind(&okvis::Publisher::publishFullStateAsCallback, &publisher,
                std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3, std::placeholders::_4));
  okvis_estimator.setLandmarksCallback(std::bind(
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
    okvis_estimator.setFullStateCallbackWithExtrinsics(std::bind(
        &okvis::Publisher::csvSaveFullStateWithExtrinsicsAsCallback, &publisher,
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
        std::placeholders::_4, std::placeholders::_5, std::placeholders::_6));
  } else if (FLAGS_dump_output_option == 3 || FLAGS_dump_output_option == 4) {
    // save estimates of evolving states, camera extrinsics,
    // and all other calibration parameters
    okvis_estimator.setFullStateCallbackWithAllCalibration(std::bind(
        &okvis::Publisher::csvSaveFullStateWithAllCalibrationAsCallback,
        &publisher, std::placeholders::_1, std::placeholders::_2,
        std::placeholders::_3, std::placeholders::_4, std::placeholders::_5,
        std::placeholders::_6, std::placeholders::_7, std::placeholders::_8,
        std::placeholders::_9, std::placeholders::_10));
    if (FLAGS_dump_output_option == 4) {
      okvis_estimator.setImuCsvFile(path + "/imu0_data.csv");
      const unsigned int numCameras = parameters.nCameraSystem.numCameras();
      for (size_t i = 0; i < numCameras; ++i) {
        std::stringstream num;
        num << i;
        okvis_estimator.setTracksCsvFile(
            i, path + "/cam" + num.str() + "_tracks.csv");
      }
      publisher.setLandmarksCsvFile(path + "/okvis_estimator_landmarks.csv");
    }
  }

  okvis_estimator.setStateCallback(
      std::bind(&okvis::Publisher::publishStateAsCallback, &publisher,
                std::placeholders::_1, std::placeholders::_2));
  okvis_estimator.setBlocking(true);
  publisher.setParameters(parameters);  // pass the specified publishing stuff

  if (FLAGS_bagname.empty()) {
    // player to grab messages directly from files on a hard drive
    okvis::Player player(&okvis_estimator, parameters);
    player.RunBlocking();
    std::string filename = path + "/feature_statistics.txt";
    okvis_estimator.saveStatistics(filename);
    return 0;
  }

  okvis::Duration deltaT(FLAGS_skip_first_seconds);
  const unsigned int numCameras = parameters.nCameraSystem.numCameras();

  // open the bag
  rosbag::Bag bag(FLAGS_bagname, rosbag::bagmode::Read);
  // views on topics. the slash is needs to be correct, it's ridiculous...
  std::string imu_topic("/imu0");
  rosbag::View view_imu(bag, rosbag::TopicQuery(imu_topic));
  if (view_imu.size() == 0) {
    LOG(ERROR) << "no imu topic";
    return -1;
  }
  rosbag::View::iterator view_imu_iterator = view_imu.begin();
  LOG(INFO) << "No. IMU messages: " << view_imu.size();

  std::vector<std::shared_ptr<rosbag::View> > view_cams_ptr;
  std::vector<rosbag::View::iterator> view_cam_iterators;
  std::vector<okvis::Time> times;
  okvis::Time latest(0);
  for (size_t i = 0; i < numCameras; ++i) {
    std::string camera_topic("/cam" + std::to_string(i) + "/image_raw");
    std::shared_ptr<rosbag::View> view_ptr(
        new rosbag::View(bag, rosbag::TopicQuery(camera_topic)));
    if (view_ptr->size() == 0) {
      LOG(ERROR) << "no camera topic";
      return 1;
    }
    view_cams_ptr.push_back(view_ptr);
    view_cam_iterators.push_back(view_ptr->begin());
    sensor_msgs::ImageConstPtr msg1 =
        view_cam_iterators[i]->instantiate<sensor_msgs::Image>();
    times.push_back(
        okvis::Time(msg1->header.stamp.sec, msg1->header.stamp.nsec));
    if (times.back() > latest) latest = times.back();
    LOG(INFO) << "No. cam " << i
              << " messages: " << view_cams_ptr.back()->size();
  }

  for (size_t i = 0; i < numCameras; ++i) {
    if ((latest - times[i]).toSec() > 0.01) view_cam_iterators[i]++;
  }

  int counter = 0;
  okvis::Time start(0.0);
  while (ros::ok()) {
    ros::spinOnce();
    okvis_estimator.display();

    // check if at the end
    if (view_imu_iterator == view_imu.end()) {
      std::cout << std::endl
                << "Finished. Press any key to exit." << std::endl
                << std::flush;
      char k = 0;
      while (k == 0 && ros::ok()) {
        k = cv::waitKey(1);
        ros::spinOnce();
      }
      return 0;
    }
    for (size_t i = 0; i < numCameras; ++i) {
      if (view_cam_iterators[i] == view_cams_ptr[i]->end()) {
        std::cout << std::endl
                  << "Finished. Press any key to exit." << std::endl
                  << std::flush;
        char k = 0;
        while (k == 0 && ros::ok()) {
          k = cv::waitKey(1);
          ros::spinOnce();
        }
        return 0;
      }
    }

    // add images
    okvis::Time t;
    okvis::Time lastImuMsgTime;
    for (size_t i = 0; i < numCameras; ++i) {
      sensor_msgs::ImageConstPtr msg1 =
          view_cam_iterators[i]->instantiate<sensor_msgs::Image>();
      cv::Mat filtered = okvis::convertImageMsgToMat(msg1);
//      cv::Mat filtered(msg1->height, msg1->width, CV_8UC1,
//                       const_cast<uint8_t*>(&msg1->data[0]), msg1->step);
//      memcpy(filtered.data, &msg1->data[0], msg1->height * msg1->width);
      t = okvis::Time(msg1->header.stamp.sec, msg1->header.stamp.nsec);
      if (start == okvis::Time(0.0)) {
        start = t;
      }

      // get all IMU measurements till then
      okvis::Time t_imu = start;
      do {
        sensor_msgs::ImuConstPtr msg =
            view_imu_iterator->instantiate<sensor_msgs::Imu>();
        Eigen::Vector3d gyr(msg->angular_velocity.x, msg->angular_velocity.y,
                            msg->angular_velocity.z);
        Eigen::Vector3d acc(msg->linear_acceleration.x,
                            msg->linear_acceleration.y,
                            msg->linear_acceleration.z);

        t_imu = okvis::Time(msg->header.stamp.sec, msg->header.stamp.nsec);
        if (lastImuMsgTime >= t_imu) {
          continue;
        } else {
          lastImuMsgTime = t_imu;
        }
        // add the IMU measurement for (blocking) processing
        if (t_imu - start > deltaT)
          okvis_estimator.addImuMeasurement(t_imu, acc, gyr);

        view_imu_iterator++;
      } while (view_imu_iterator != view_imu.end() && t_imu <= t);

      // add the image to the frontend for (blocking) processing
      if (t - start > deltaT) okvis_estimator.addImage(t, i, filtered);

      view_cam_iterators[i]++;
    }
    ++counter;

    // display progress
    if (counter % 20 == 0) {
      std::cout << "\rProgress: "
                << int(double(counter) / double(view_cams_ptr.back()->size()) *
                       100)
                << "%  ";
    }
  }

  std::cout << std::endl;
  return 0;
}
