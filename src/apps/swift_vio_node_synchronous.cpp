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
 * @file swift_vio_node_synchronous.cpp
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
#include <swift_vio/VioFactoryMethods.hpp>
#include "VioSystemWrap.hpp"

DEFINE_bool(
    publish_via_ros, true,
    "If to publish via ros, rosnode will be created, and rostopics registers.");

class RosbagIteratorChecker {
 public:
  RosbagIteratorChecker(
      rosbag::View& view_imu,
      const std::vector<std::shared_ptr<rosbag::View> >& view_cams_ptr)
      : view_imu_(view_imu), view_cams_ptr_(view_cams_ptr), numCameras_(view_cams_ptr.size()) {}

  bool atImuEnd(rosbag::View::iterator view_imu_iterator) const {
    if (view_imu_iterator == view_imu_.end()) {
      LOG(INFO) << "Finished IMU data. Press any key to exit.";
      char k = 0;
      while (k == 0 && ros::ok()) {
        k = cv::waitKey(1);
        ros::spinOnce();
      }
      LOG(INFO) << "Returning from swift_vio_node_sync IMU branch!";
      return true;
    } else {
      return false;
    }
  }

  bool atImageEnd(const std::vector<rosbag::View::iterator>& view_cam_iterators) const {
    for (size_t i = 0; i < numCameras_; ++i) {
      if (view_cam_iterators[i] == view_cams_ptr_[i]->end()) {
        LOG(INFO) << "Finished images. Press any key to exit.";
        char k = 0;
        while (k == 0 && ros::ok()) {
          k = cv::waitKey(1);
          ros::spinOnce();
        }
        LOG(INFO) << "Returning from swift_vio_node_sync image branch!";
        return true;
      }
    }
    return false;
  }

 private:
  rosbag::View& view_imu_;
  std::vector<std::shared_ptr<rosbag::View> > view_cams_ptr_;
  size_t numCameras_;
};

bool messageCenterOk() {
  if (FLAGS_publish_via_ros) {
    return ros::ok();
  } else {
    return true;
  }
}

void messageCenterSpinOnce() {
  if (FLAGS_publish_via_ros) {
    return ros::spinOnce();
  }
}

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);  // true to strip gflags
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_stderrthreshold = 0;  // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
  FLAGS_colorlogtostderr = 1;

  std::shared_ptr<ros::NodeHandle> nh;
  if (FLAGS_publish_via_ros) {
    const std::string nodeName = "swift_vio_node_synchronous";
    ros::init(argc, argv, nodeName);
    // use swift_vio_node because it is the prefix for topics shown in rviz.
    nh.reset(new ros::NodeHandle("swift_vio_node"));
  }

  std::string configFilename;
  if (argc >= 2) {
    configFilename = argv[1];
  } else {
    LOG(ERROR) << "Usage:" << argv[0]
               << " <config yml> <lcd config yml> [extra gflags]";
    return 1;
  }

  swift_vio::RosParametersReader vio_parameters_reader(configFilename);
  okvis::VioParameters parameters;
  vio_parameters_reader.getParameters(parameters);
  swift_vio::setInputParameters(&parameters.input);

  // Caution: Objects including shared_ptrs are destroyed in reverse order of construction,
  // and an object shared by shared_ptr will be destroyed when no shared_ptr shares its ownership.
  // Publisher is created before ThreadedKFVio as it depends on publisher in publisherLoop().
  std::shared_ptr<swift_vio::StreamPublisher> publisher;
  if (FLAGS_publish_via_ros) {
    publisher.reset(new swift_vio::Publisher(*nh));
  } else {
    publisher.reset(new swift_vio::StreamPublisher());
  }
  publisher->setParameters(parameters);
  swift_vio::PgoPublisher pgoPublisher;

  swift_vio::BackendParams backendParams;
  backendParams.parseYAML(configFilename);
  std::shared_ptr<okvis::ceres::Map> mapPtr(new okvis::ceres::Map());
  std::shared_ptr<okvis::Estimator> estimator = swift_vio::createBackend(
      parameters.optimization.algorithm, backendParams, mapPtr);
  std::shared_ptr<okvis::Frontend> frontend = swift_vio::createFrontend(
      parameters.nCameraSystem.numCameras(),
      parameters.frontendOptions,
      parameters.optimization.algorithm);
  std::shared_ptr<swift_vio::LoopClosureDetectorParams> lcParams(
        new swift_vio::LoopClosureDetectorParams());
  std::string lcdConfigFilename = FLAGS_lcd_params_yaml;
  if (lcdConfigFilename.empty()) {
    LOG(WARNING) << "Default parameters for loop closure will be used as no "
                    "configuration filename is provided!";
  } else {
    LOG(WARNING) << "The corporeal loop closure module for now is costly and "
                    "harms accuracy esp. in a stereo setup.";
    lcParams->parseYAML(lcdConfigFilename);
  }
  if (!frontend->isDescriptorBasedMatching()) {
    lcParams->loop_closure_method_ = swift_vio::LoopClosureMethodType::Mock;
    LOG(WARNING)
        << "Loop closure module requires descriptors for keypoints to perform "
           "matching. But the KLT frontend does not extract descriptors. "
           "Descriptors can be extracted for KLT points in creating loop query "
           "keyframes but this is not done yet.";
  }
  std::shared_ptr<swift_vio::LoopClosureMethod> loopClosureMethod =
      swift_vio::createLoopClosureMethod(lcParams);
  okvis::ThreadedKFVio vioSystem(parameters, estimator, frontend,
                                       loopClosureMethod);
  swift_vio::VioSystemWrap::registerCallbacks(
      FLAGS_output_dir, parameters, &vioSystem, publisher.get(),
      &pgoPublisher);

  vioSystem.setBlocking(true);

  if (FLAGS_bagname.empty()) {
    // player to grab messages directly from files on a hard drive
    swift_vio::Player player(&vioSystem, parameters);
    player.RunBlocking();
    std::string filename = swift_vio::removeTrailingSlash(FLAGS_output_dir) +
                           "/feature_statistics.txt";
    vioSystem.saveStatistics(filename);
    return 0;
  }

  okvis::Duration deltaT(FLAGS_skip_first_seconds);
  const unsigned int numCameras = parameters.nCameraSystem.numCameras();

  // open the bag
  rosbag::Bag bag(FLAGS_bagname, rosbag::bagmode::Read);
  std::vector<std::string> camera_topics =
      swift_vio::parseCommaSeparatedTopics(FLAGS_camera_topics);
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
  RosbagIteratorChecker endGuard(view_imu, view_cams_ptr);
  while (messageCenterOk()) {
    messageCenterSpinOnce();
    vioSystem.display();

    // check if at the end
    if (endGuard.atImuEnd(view_imu_iterator)) {
      return 0;
    }
    if (endGuard.atImageEnd(view_cam_iterators)) {
      return 0;
    }

    // add images
    okvis::Time t;
    for (size_t i = 0; i < numCameras; ++i) {
      sensor_msgs::ImageConstPtr msg1 =
          view_cam_iterators[i]->instantiate<sensor_msgs::Image>();
      cv::Mat filtered = swift_vio::convertImageMsgToMat(msg1);
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

        // add the IMU measurement for (blocking) processing
        if (t_imu - start > deltaT)
          vioSystem.addImuMeasurement(t_imu, acc, gyr);

        view_imu_iterator++;
      } while (view_imu_iterator != view_imu.end() && t_imu <= t);

      // add the image to the frontend for (blocking) processing
      if (t - start > deltaT) vioSystem.addImage(t, i, filtered);
      // The imu messages may end for the next image message in the NFrame.
      if (endGuard.atImuEnd(view_imu_iterator)) {
        return 0;
      }
      view_cam_iterators[i]++;
    }
    ++counter;

    // display progress
    if (counter % 20 == 0) {
      LOG(INFO) << "Progress: "
                << int(double(counter) / double(view_cams_ptr.back()->size()) *
                       100)
                << "%  ";
    }
  }

  return 0;
}
