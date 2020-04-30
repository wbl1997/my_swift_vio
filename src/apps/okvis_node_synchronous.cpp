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
#include <io_wrap/RosParametersReader.hpp>
#include <io_wrap/Subscriber.hpp>
#include <loop_closure/LoopClosureDetectorParams.h>
#include <msckf/VioFactoryMethods.hpp>
#include "VioSystemWrap.hpp"

DEFINE_double(skip_first_seconds, 0, "Number of seconds to skip from the beginning!");

DEFINE_string(bagname, "", "Bag filename.");

DEFINE_string(camera_topics, "/cam0/image_raw,/cam1/image_raw",
              "Used image topics inside the bag. Should agree with the number "
              "of cameras in the config file");

DEFINE_string(imu_topic, "/imu0", "Imu topic inside the bag");

// this is just a workbench. most of the stuff here will go into the Frontend
// class.
int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);  // true to strip gflags
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_stderrthreshold = 0;  // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
  FLAGS_colorlogtostderr = 1;

  const std::string nodeName = "okvis_node_synchronous";
  ros::init(argc, argv, nodeName);
  ros::NodeHandle nh("okvis_node"); // use okvis_node because it is the prefix for topics shown in rviz.

  std::string configFilename;
  std::string lcdConfigFilename;
  if (argc >= 2) {
    configFilename = argv[1];
    if (argc >= 3) {
      lcdConfigFilename = argv[2];
    }
  } else {
    if (!nh.getParam("config_filename", configFilename)) {
      LOG(ERROR) << "Usage:" << argv[0] << " <config yml> <lcd config yml> [extra gflags]";
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
  if (lcdConfigFilename.empty()) {
    LOG(WARNING) << "Default parameters for loop closure will be used as no "
                    "configuration filename is provided!";
  } else {
    lcParams->parseYAML(lcdConfigFilename);
  }
  if (!frontend->isDescriptorBasedMatching()) {
    lcParams->loop_closure_method_ = VIO::LoopClosureMethodType::Mock;
    LOG(WARNING)
        << "Loop closure module requires descriptors for keypoints to perform "
           "matching. But the KLT frontend does not extract descriptors. "
           "Descriptors can be extracted for KLT points in creating loop query "
           "keyframes but this is not done yet.";
  }
  std::shared_ptr<okvis::LoopClosureMethod> loopClosureMethod =
      msckf::createLoopClosureMethod(lcParams);
  okvis::ThreadedKFVio okvis_estimator(parameters, estimator, frontend,
                                       loopClosureMethod);

  okvis::Publisher publisher(nh);
  publisher.setParameters(parameters);
  okvis::PgoPublisher pgoPublisher;
  okvis::VioSystemWrap::registerCallbacks(
      FLAGS_output_dir, parameters, &okvis_estimator, &publisher,
      &pgoPublisher);

  okvis_estimator.setBlocking(true);

  if (FLAGS_bagname.empty()) {
    // player to grab messages directly from files on a hard drive
    okvis::Player player(&okvis_estimator, parameters);
    player.RunBlocking();
    std::string filename = okvis::removeTrailingSlash(FLAGS_output_dir) +
                           "/feature_statistics.txt";
    okvis_estimator.saveStatistics(filename);
    return 0;
  }

  okvis::Duration deltaT(FLAGS_skip_first_seconds);
  const unsigned int numCameras = parameters.nCameraSystem.numCameras();

  // open the bag
  rosbag::Bag bag(FLAGS_bagname, rosbag::bagmode::Read);
  std::vector<std::string> camera_topics =
      okvis::parseCommaSeparatedTopics(FLAGS_camera_topics);
  // views on topics. the slash is needs to be correct, it's ridiculous...
  std::string imu_topic = FLAGS_imu_topic;
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
    std::string camera_topic = camera_topics[i];
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
                << "Finished IMU data. Press any key to exit." << std::endl
                << std::flush;
      char k = 0;
      while (k == 0 && ros::ok()) {
        k = cv::waitKey(1);
        ros::spinOnce();
      }
      std::cout << "Returning from okvis_node_sync IMU branch!";
      return 0;
    }
    for (size_t i = 0; i < numCameras; ++i) {
      if (view_cam_iterators[i] == view_cams_ptr[i]->end()) {
        std::cout << std::endl
                  << "Finished images. Press any key to exit." << std::endl
                  << std::flush;
        char k = 0;
        while (k == 0 && ros::ok()) {
          k = cv::waitKey(1);
          ros::spinOnce();
        }
        // TODO(jhuai): There is a erratic segmentation fault at the end of execution.
        // It happens before destructors for ThreadedKFVio or any Estimator are called.
        std::cout << "Returning from okvis_node_sync image branch!";
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
