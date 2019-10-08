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
 *  Created on: Apr 27, 2012
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file Publisher.cpp
 * @brief Source file for the Publisher class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <glog/logging.h>

#include <algorithm>
#include <string>
#include <vector>

#include <io_wrap/Publisher.hpp>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <ros/package.h>
#pragma GCC diagnostic pop
#include <sensor_msgs/fill_image.h>
#include <sensor_msgs/image_encodings.h>

#include <okvis/FrameTypedefs.hpp>
#include <msckf/CameraRig.hpp>
#include <msckf/ExtrinsicModels.hpp>
#include <msckf/ImuRig.hpp>
#include <msckf/ProjParamOptModels.hpp>

DECLARE_string(datafile_separator);

/// \brief okvis Main namespace of this package.
namespace okvis {

// Default constructor.
Publisher::Publisher(const DUMP_RESULT_OPTION dro)
    : nh_(nullptr), ctr2_(0), result_option_(dro) {}

Publisher::~Publisher() {
  // close file
  if (csvLandmarksFile_) {
    // write down also the current landmarks
    if (csvLandmarksFile_->good()) {
      for (size_t l = 0; l < pointsMatched2_.size(); ++l) {
        Eigen::Vector4d landmark = pointsMatched2_.at(l).pointHomog;
        *csvLandmarksFile_ << std::setprecision(19) << pointsMatched2_.at(l).id
                           << FLAGS_datafile_separator << std::scientific
                           << std::setprecision(18) << landmark[0]
                           << FLAGS_datafile_separator << landmark[1]
                           << FLAGS_datafile_separator << landmark[2]
                           << FLAGS_datafile_separator << landmark[3]
                           << FLAGS_datafile_separator
                           << pointsMatched2_.at(l).quality << std::endl;
      }
    }
    csvLandmarksFile_->close();
  }
  if (csvFile_) csvFile_->close();
}

// Constructor. Calls setNodeHandle().
Publisher::Publisher(ros::NodeHandle &nh, const DUMP_RESULT_OPTION dro)
    : Publisher(dro) {
  setNodeHandle(nh);
}

// Set the node handle and advertise topics.
void Publisher::setNodeHandle(ros::NodeHandle &nh) {
  nh_ = &nh;

  // advertise
  //  pubPointsMatched_ = nh_->advertise<sensor_msgs::PointCloud2>(
  //      "okvis_points_matched", 1);
  //  pubPointsUnmatched_ = nh_->advertise<sensor_msgs::PointCloud2>(
  //      "okvis_points_unmatched", 1);
  //  pubPointsTransferred_ = nh_->advertise<sensor_msgs::PointCloud2>(
  //      "okvis_points_transferred", 1);
  // Configure Reference MapPoints
  float fPointSize = 0.01;
  pointsMatched_.header.frame_id = "world";
  pointsMatched_.ns = "okvis_points_matched";
  pointsMatched_.id = 0;
  pointsMatched_.type = visualization_msgs::Marker::POINTS;
  pointsMatched_.scale.x = fPointSize;
  pointsMatched_.scale.y = fPointSize;
  pointsMatched_.pose.orientation.w = 1.0;
  pointsMatched_.action = visualization_msgs::Marker::ADD;
  pointsMatched_.color.r = 1.0f;
  pointsMatched_.color.a = 1.0;

  pointsUnmatched_.header.frame_id = "world";
  pointsUnmatched_.ns = "okvis_points_unmatched";
  pointsUnmatched_.id = 1;
  pointsUnmatched_.type = visualization_msgs::Marker::POINTS;
  pointsUnmatched_.scale.x = fPointSize;
  pointsUnmatched_.scale.y = fPointSize;
  pointsUnmatched_.pose.orientation.w = 1.0;
  pointsUnmatched_.action = visualization_msgs::Marker::ADD;
  pointsUnmatched_.color.g = 1.0f;
  pointsUnmatched_.color.a = 1.0;

  pointsTransferred_.header.frame_id = "world";
  pointsTransferred_.ns = "okvis_points_transferred";
  pointsTransferred_.id = 2;
  pointsTransferred_.type = visualization_msgs::Marker::POINTS;
  pointsTransferred_.scale.x = fPointSize;
  pointsTransferred_.scale.y = fPointSize;
  pointsTransferred_.pose.orientation.w = 1.0;
  pointsTransferred_.action = visualization_msgs::Marker::ADD;
  pointsTransferred_.color.b = 1.0f;
  pointsTransferred_.color.a = 1.0;

  pubPointsMatched_ =
      nh_->advertise<visualization_msgs::Marker>("okvis_points_matched", 1);
  pubPointsUnmatched_ =
      nh_->advertise<visualization_msgs::Marker>("okvis_points_unmatched", 1);
  pubPointsTransferred_ =
      nh_->advertise<visualization_msgs::Marker>("okvis_points_transferred", 1);

  pubObometry_ = nh_->advertise<nav_msgs::Odometry>("okvis_odometry", 1);
  pubPath_ = nh_->advertise<nav_msgs::Path>("okvis_path", 1);
  pubTransform_ =
      nh_->advertise<geometry_msgs::TransformStamped>("okvis_transform", 1);
}

// Write CSV header.
bool Publisher::writeCsvDescription(const std::string& headerLine) {
  if (!csvFile_) return false;
  if (!csvFile_->good()) return false;
  *csvFile_ << headerLine << std::endl;
  return true;
}

// Write CSV header for landmarks file.
bool Publisher::writeLandmarksCsvDescription() {
  if (!csvLandmarksFile_) return false;
  if (!csvLandmarksFile_->good()) return false;
  *csvLandmarksFile_ << FLAGS_datafile_separator << "id"
                     << FLAGS_datafile_separator << "l_x"
                     << FLAGS_datafile_separator << "l_y"
                     << FLAGS_datafile_separator << "l_z"
                     << FLAGS_datafile_separator << "l_w"
                     << FLAGS_datafile_separator << "quality"
                     << FLAGS_datafile_separator << "distance" << std::endl;
  return true;
}

// Set an odometry output CSV file.
bool Publisher::setCsvFile(std::fstream &csvFile, const std::string& headerLine) {
  if (csvFile_) {
    csvFile_->close();
  }
  csvFile_.reset(&csvFile);
  writeCsvDescription(headerLine);
  return csvFile_->good();
}
// Set an odometry output CSV file.
bool Publisher::setCsvFile(const std::string &csvFileName, const std::string& headerLine) {
  csvFile_.reset(new std::fstream(csvFileName.c_str(), std::ios_base::out));
  writeCsvDescription(headerLine);
  return csvFile_->good();
}

// Set a CVS file where the landmarks will be saved to.
bool Publisher::setLandmarksCsvFile(std::fstream &csvFile) {
  if (csvLandmarksFile_) {
    csvLandmarksFile_->close();
  }
  csvLandmarksFile_.reset(&csvFile);
  writeLandmarksCsvDescription();
  return csvLandmarksFile_->good();
}
// Set a CVS file where the landmarks will be saved to.
bool Publisher::setLandmarksCsvFile(std::string &csvFileName) {
  csvLandmarksFile_.reset(
      new std::fstream(csvFileName.c_str(), std::ios_base::out));
  writeLandmarksCsvDescription();
  return csvLandmarksFile_->good();
}
// Set a CVS file where the landmarks will be saved to.
bool Publisher::setLandmarksCsvFile(std::string csvFileName) {
  csvLandmarksFile_.reset(
      new std::fstream(csvFileName.c_str(), std::ios_base::out));
  writeLandmarksCsvDescription();
  return csvLandmarksFile_->good();
}

// Set the pose message that is published next.
void Publisher::setPose(const okvis::kinematics::Transformation &T_WS) {
  okvis::kinematics::Transformation T;
  if (parameters_.publishing.trackedBodyFrame == FrameName::S) {
    poseMsg_.child_frame_id = "sensor";
    T = parameters_.publishing.T_Wc_W * T_WS;
  } else if (parameters_.publishing.trackedBodyFrame == FrameName::B) {
    poseMsg_.child_frame_id = "body";
    T = parameters_.publishing.T_Wc_W * T_WS * parameters_.imu.T_BS.inverse();
  } else {
    LOG(ERROR) << "Pose frame does not exist for publishing."
                  " Choose 'S' or 'B'.";
    poseMsg_.child_frame_id = "body";
    T = parameters_.publishing.T_Wc_W * T_WS * parameters_.imu.T_BS.inverse();
  }

  poseMsg_.header.frame_id = "world";
  poseMsg_.header.stamp = _t;
  if ((ros::Time::now() - _t).toSec() > 10.0)
    poseMsg_.header.stamp = ros::Time::now();

  // fill orientation
  Eigen::Quaterniond q = T.q();
  poseMsg_.transform.rotation.x = q.x();
  poseMsg_.transform.rotation.y = q.y();
  poseMsg_.transform.rotation.z = q.z();
  poseMsg_.transform.rotation.w = q.w();

  // fill position
  Eigen::Vector3d r = T.r();
  poseMsg_.transform.translation.x = r[0];
  poseMsg_.transform.translation.y = r[1];
  poseMsg_.transform.translation.z = r[2];
}

// Set the odometry message that is published next.
void Publisher::setOdometry(const okvis::kinematics::Transformation &T_WS,
                            const okvis::SpeedAndBiases &speedAndBiases,
                            const Eigen::Vector3d &omega_S) {
  // header.frame_id is the frame in which the pose is given.
  //   I.e. world frame in our case
  // child_frame_id is the frame in which the twist part of the
  //   odometry message is given.
  // see also nav_msgs/Odometry Message documentation

  odometryMsg_.header.stamp = _t;
  // the pose to be published. T_WS or T_WB depending on 'trackedBodyFrame'
  okvis::kinematics::Transformation T;
  Eigen::Vector3d omega_W =
      parameters_.publishing.T_Wc_W.C() * T_WS.C() * omega_S;
  Eigen::Vector3d t_W_ofFrame;  // lever arm in W-system
  Eigen::Vector3d v_W_ofFrame;  // velocity in W-system. v_S_in_W or v_B_in_W

  if (parameters_.publishing.trackedBodyFrame == FrameName::S) {
    odometryMsg_.header.frame_id = "world";
    T = parameters_.publishing.T_Wc_W * T_WS;
    t_W_ofFrame.setZero();  // r_SS_in_W
    v_W_ofFrame = parameters_.publishing.T_Wc_W.C() *
                  speedAndBiases.head<3>();  // world-centric
  } else if (parameters_.publishing.trackedBodyFrame == FrameName::B) {
    odometryMsg_.header.frame_id = "world";
    T = parameters_.publishing.T_Wc_W * T_WS * parameters_.imu.T_BS.inverse();
    t_W_ofFrame =
        (parameters_.publishing.T_Wc_W * T_WS * parameters_.imu.T_BS.inverse())
            .r() -
        (parameters_.publishing.T_Wc_W * T_WS).r();  // r_BS_in_W
    v_W_ofFrame =
        speedAndBiases.head<3>() + omega_W.cross(t_W_ofFrame);  // world-centric
  } else {
    LOG(ERROR) << "Pose frame does not exist for publishing. "
                  "Choose 'S' or 'B'.";
    odometryMsg_.header.frame_id = "world";
    T = parameters_.publishing.T_Wc_W * T_WS;
    t_W_ofFrame.setZero();  // r_SS_in_W
    v_W_ofFrame = parameters_.publishing.T_Wc_W.C() *
                  speedAndBiases.head<3>();  // world-centric
  }

  // fill orientation
  Eigen::Quaterniond q = T.q();
  odometryMsg_.pose.pose.orientation.x = q.x();
  odometryMsg_.pose.pose.orientation.y = q.y();
  odometryMsg_.pose.pose.orientation.z = q.z();
  odometryMsg_.pose.pose.orientation.w = q.w();

  // fill position
  Eigen::Vector3d r = T.r();
  odometryMsg_.pose.pose.position.x = r[0];
  odometryMsg_.pose.pose.position.y = r[1];
  odometryMsg_.pose.pose.position.z = r[2];

  Eigen::Matrix3d C_v;
  Eigen::Matrix3d C_omega;
  if (parameters_.publishing.velocitiesFrame == FrameName::S) {
    C_v = (parameters_.publishing.T_Wc_W * T_WS).inverse().C();
    C_omega.setIdentity();
    odometryMsg_.child_frame_id = "sensor";
  } else if (parameters_.publishing.velocitiesFrame == FrameName::B) {
    C_v = (parameters_.imu.T_BS * T_WS.inverse()).C() *
          parameters_.publishing.T_Wc_W.inverse().C();
    C_omega = parameters_.imu.T_BS.C();
    odometryMsg_.child_frame_id = "body";
  } else if (parameters_.publishing.velocitiesFrame == FrameName::Wc) {
    C_v.setIdentity();
    C_omega = T_WS.C();
    odometryMsg_.child_frame_id = "world";
  } else {
    LOG(ERROR) << "Speeds frame does not exist for publishing. "
                  "Choose 'S', 'B', or 'Wc'.";
    C_v = (parameters_.imu.T_BS * T_WS.inverse()).C() *
          parameters_.publishing.T_Wc_W.inverse().C();
    C_omega = parameters_.imu.T_BS.C();
    odometryMsg_.child_frame_id = "body";
  }

  // fill velocity
  // v_S_in_'speedsInThisFrame' or v_B_in_'speedsInThisFrame'
  Eigen::Vector3d v = C_v * v_W_ofFrame;
  odometryMsg_.twist.twist.linear.x = v[0];
  odometryMsg_.twist.twist.linear.y = v[1];
  odometryMsg_.twist.twist.linear.z = v[2];

  // fill angular velocity
  Eigen::Vector3d omega = C_omega * omega_S;  // omega_in_'speedsInThisFrame'
  odometryMsg_.twist.twist.angular.x = omega[0];
  odometryMsg_.twist.twist.angular.y = omega[1];
  odometryMsg_.twist.twist.angular.z = omega[2];

  // linear acceleration ?? - would also need point of percussion mapping!!
}

// Set the points that are published next.
void Publisher::setPoints(const okvis::MapPointVector &pointsMatched,
                          const okvis::MapPointVector &pointsUnmatched,
                          const okvis::MapPointVector &pointsTransferred) {
  // Huai{
  pointsMatched2_.clear();
  pointsMatched2_ = pointsMatched;
  pointsMatched_.points.clear();
  pointsMatched_.colors.clear();
  pointsUnmatched_.points.clear();
  pointsUnmatched_.colors.clear();
  pointsTransferred_.points.clear();
  pointsTransferred_.colors.clear();
  for (size_t i = 0; i < pointsMatched.size(); ++i) {
    // check infinity
    if (fabs(static_cast<double>(pointsMatched[i].pointHomog[3])) < 1.0e-8)
      continue;

    // check quality
    if (static_cast<float>(pointsMatched[i].quality) <
        parameters_.publishing.landmarkQualityThreshold)
      continue;

    pointsMatched_.points.push_back(geometry_msgs::Point());
    const Eigen::Vector4d point = pointsMatched[i].pointHomog;
    pointsMatched_.points.back().x = point[0] / point[3];
    pointsMatched_.points.back().y = point[1] / point[3];
    pointsMatched_.points.back().z = point[2] / point[3];
    pointsMatched_.colors.push_back(std_msgs::ColorRGBA());
    pointsMatched_.colors.back().g =
        1.0f * (std::min(parameters_.publishing.maxLandmarkQuality,
                         static_cast<float>(pointsMatched[i].quality)) /
                parameters_.publishing.maxLandmarkQuality);
    pointsMatched_.colors.back().a = 1.0f;
  }
  pointsMatched_.header.frame_id = "world";

  pointsMatched_.header.stamp = _t;

  for (size_t i = 0; i < pointsUnmatched.size(); ++i) {
    // check infinity
    if (fabs(static_cast<double>(pointsUnmatched[i].pointHomog[3])) < 1.0e-8)
      continue;

    // check quality
    if (pointsUnmatched[i].quality <
        parameters_.publishing.landmarkQualityThreshold)
      continue;

    pointsUnmatched_.points.push_back(geometry_msgs::Point());
    const Eigen::Vector4d point = pointsUnmatched[i].pointHomog;
    pointsUnmatched_.points.back().x = point[0] / point[3];
    pointsUnmatched_.points.back().y = point[1] / point[3];
    pointsUnmatched_.points.back().z = point[2] / point[3];
    pointsUnmatched_.colors.push_back(std_msgs::ColorRGBA());
    pointsUnmatched_.colors.back().b =
        1.0f * (std::min(parameters_.publishing.maxLandmarkQuality,
                         static_cast<float>(pointsUnmatched[i].quality)) /
                parameters_.publishing.maxLandmarkQuality);
    pointsUnmatched_.colors.back().a = 1.0f;
  }
  pointsUnmatched_.header.frame_id = "world";

  pointsUnmatched_.header.stamp = _t;

  for (size_t i = 0; i < pointsTransferred.size(); ++i) {
    // check infinity

    if (fabs(static_cast<double>(pointsTransferred[i].pointHomog[3])) < 1.0e-10)
      continue;

    // check quality
    if (pointsTransferred[i].quality <
        parameters_.publishing.landmarkQualityThreshold)
      continue;

    pointsTransferred_.points.push_back(geometry_msgs::Point());
    const Eigen::Vector4d point = pointsTransferred[i].pointHomog;
    pointsTransferred_.points.back().x = point[0] / point[3];
    pointsTransferred_.points.back().y = point[1] / point[3];
    pointsTransferred_.points.back().z = point[2] / point[3];
    /*float intensity =*/
        std::min(parameters_.publishing.maxLandmarkQuality,
                 static_cast<float>(pointsTransferred[i].quality)) /
        parameters_.publishing.maxLandmarkQuality;
    pointsTransferred_.colors.push_back(std_msgs::ColorRGBA());
    pointsTransferred_.colors.back().r = 1.0;  // intensity;
    pointsTransferred_.colors.back().g = 0;    // intensity;
    pointsTransferred_.colors.back().b = 0;    // intensity;
    pointsTransferred_.colors.back().a = 1.0f;
    // _omfile << point[0] << " " << point[1] << " "
    // << point[2] << ";" <<std::endl;
  }
  pointsTransferred_.header.frame_id = "world";
  pointsTransferred_.header.seq = ctr2_++;

  pointsTransferred_.header.stamp = _t;
  // }Huai
}

// Publish the pose.
void Publisher::publishPose() {
  if ((_t - lastOdometryTime2_).toSec() <
      1.0 / parameters_.publishing.publishRate)
    return;  // control the publish rate
  pubTf_.sendTransform(poseMsg_);
  lastOdometryTime2_ = _t;  // remember
}

// Publish the last set odometry.
void Publisher::publishOdometry() {
  if ((_t - lastOdometryTime_).toSec() <
      1.0 / parameters_.publishing.publishRate)
    return;  // control the publish rate
  pubObometry_.publish(odometryMsg_);
  lastOdometryTime_ = _t;  // remember
}

// Publish the T_WS transform.
void Publisher::publishTransform() {
  if ((_t - lastTransfromTime_).toSec() <
      1.0 / parameters_.publishing.publishRate)
    return;                         // control the publish rate
  pubTransform_.publish(poseMsg_);  // publish stamped transform for MSF
  lastTransfromTime_ = _t;          // remember
}

// Set and publish pose.
void Publisher::publishStateAsCallback(
    const okvis::Time &t, const okvis::kinematics::Transformation &T_WS) {
  setTime(t);
  setPose(T_WS);  // TODO(sleuten): provide setters for this hack
  publishPose();
}
// Set and publish full state.
void Publisher::publishFullStateAsCallback(
    const okvis::Time &t, const okvis::kinematics::Transformation &T_WS,
    const Eigen::Matrix<double, 9, 1> &speedAndBiases,
    const Eigen::Matrix<double, 3, 1> &omega_S) {
  setTime(t);
  setOdometry(T_WS, speedAndBiases,
              omega_S);  // TODO(sleuten): provide setters for this hack
  setPath(T_WS);
  publishOdometry();
  publishTransform();
  publishPath();
}

// Set and write full state to CSV file.
void Publisher::csvSaveFullStateAsCallback(
    const okvis::Time &t, const okvis::kinematics::Transformation &T_WS,
    const Eigen::Matrix<double, 9, 1> &speedAndBiases,
    const Eigen::Matrix<double, 3, 1> &omega_S, const int frameIdInSource) {
  setTime(t);
  setOdometry(T_WS, speedAndBiases,
              omega_S);  // TODO(sleuten): provide setters for this hack
  if (csvFile_) {
    // LOG(INFO)<<"filePtr: ok; ";
    if (csvFile_->good()) {
      // LOG(INFO)<<"file: good.";
      Eigen::Vector3d p_WS_W = T_WS.r();
      Eigen::Quaterniond q_WS = T_WS.q();
      std::stringstream time;
      time << t.sec << std::setw(9) << std::setfill('0') << t.nsec;
      *csvFile_ << time.str() << FLAGS_datafile_separator << frameIdInSource
                << FLAGS_datafile_separator << std::scientific
                << std::setprecision(8) << p_WS_W[0] << FLAGS_datafile_separator
                << p_WS_W[1] << FLAGS_datafile_separator << p_WS_W[2]
                << FLAGS_datafile_separator << q_WS.x()
                << FLAGS_datafile_separator << q_WS.y()
                << FLAGS_datafile_separator << q_WS.z()
                << FLAGS_datafile_separator << q_WS.w()
                << FLAGS_datafile_separator << speedAndBiases[0]
                << FLAGS_datafile_separator << speedAndBiases[1]
                << FLAGS_datafile_separator << speedAndBiases[2]
                << FLAGS_datafile_separator << speedAndBiases[3]
                << FLAGS_datafile_separator << speedAndBiases[4]
                << FLAGS_datafile_separator << speedAndBiases[5]
                << FLAGS_datafile_separator << speedAndBiases[6]
                << FLAGS_datafile_separator << speedAndBiases[7]
                << FLAGS_datafile_separator << speedAndBiases[8] << std::endl;
    }
  }
}

// Set and write full state including camera extrinsics to file.
void Publisher::csvSaveFullStateWithExtrinsicsAsCallback(
    const okvis::Time &t, const okvis::kinematics::Transformation &T_WS,
    const Eigen::Matrix<double, 9, 1> &speedAndBiases,
    const Eigen::Matrix<double, 3, 1> &omega_S, const int frameIdInSource,
    const std::vector<
        okvis::kinematics::Transformation,
        Eigen::aligned_allocator<okvis::kinematics::Transformation>>
        &extrinsics) {
  setTime(t);
  setOdometry(T_WS, speedAndBiases,
              omega_S);  // TODO(sleuten): provide setters for this hack
  if (csvFile_) {
    if (csvFile_->good()) {
      Eigen::Vector3d p_WS_W = T_WS.r();
      Eigen::Quaterniond q_WS = T_WS.q();
      std::stringstream time;
      time << t.sec << std::setw(9) << std::setfill('0') << t.nsec;
      *csvFile_ << time.str() << FLAGS_datafile_separator << frameIdInSource
                << FLAGS_datafile_separator << std::setprecision(6) << p_WS_W[0]
                << FLAGS_datafile_separator << p_WS_W[1]
                << FLAGS_datafile_separator << p_WS_W[2]
                << FLAGS_datafile_separator << q_WS.x()
                << FLAGS_datafile_separator << q_WS.y()
                << FLAGS_datafile_separator << q_WS.z()
                << FLAGS_datafile_separator << q_WS.w()
                << FLAGS_datafile_separator << speedAndBiases[0]
                << FLAGS_datafile_separator << speedAndBiases[1]
                << FLAGS_datafile_separator << speedAndBiases[2]
                << FLAGS_datafile_separator << speedAndBiases[3]
                << FLAGS_datafile_separator << speedAndBiases[4]
                << FLAGS_datafile_separator << speedAndBiases[5]
                << FLAGS_datafile_separator << speedAndBiases[6]
                << FLAGS_datafile_separator << speedAndBiases[7]
                << FLAGS_datafile_separator << speedAndBiases[8];
      for (size_t i = 0; i < extrinsics.size(); ++i) {
        Eigen::Vector3d p_SCi = extrinsics[i].r();
        Eigen::Quaterniond q_SCi = extrinsics[i].q();
        *csvFile_ << FLAGS_datafile_separator << p_SCi[0]
                  << FLAGS_datafile_separator << p_SCi[1]
                  << FLAGS_datafile_separator << p_SCi[2]
                  << FLAGS_datafile_separator << q_SCi.x()
                  << FLAGS_datafile_separator << q_SCi.y()
                  << FLAGS_datafile_separator << q_SCi.z()
                  << FLAGS_datafile_separator << q_SCi.w();
      }

      *csvFile_ << std::endl;
    }
  }
}

// Set and write full state including camera extrinsics to file.
void Publisher::csvSaveFullStateWithAllCalibrationAsCallback(
    const okvis::Time &t, const okvis::kinematics::Transformation &T_WS,
    const Eigen::Matrix<double, 9, 1> &speedAndBiases,
    const Eigen::Matrix<double, 3, 1> &omega_S, const int frameIdInSource,
    const std::vector<
        okvis::kinematics::Transformation,
        Eigen::aligned_allocator<okvis::kinematics::Transformation>>
        &extrinsics,
    const Eigen::Matrix<double, Eigen::Dynamic, 1> &vTgsa,
    const Eigen::Matrix<double, Eigen::Dynamic, 1> &vfckptdr,
    const Eigen::Matrix<double, Eigen::Dynamic, 1> &vVariance) {
  setTime(t);
  setOdometry(T_WS, speedAndBiases,
              omega_S);  // TODO(sleuten): provide setters for this hack
  if (csvFile_) {
    if (csvFile_->good()) {
      Eigen::Vector3d p_WS_W = T_WS.r();
      Eigen::Quaterniond q_WS = T_WS.q();
      std::stringstream time;
      time << t.sec << std::setw(9) << std::setfill('0') << t.nsec;
      *csvFile_ << time.str() << FLAGS_datafile_separator << frameIdInSource
                << FLAGS_datafile_separator << std::setprecision(6) << p_WS_W[0]
                << FLAGS_datafile_separator << p_WS_W[1]
                << FLAGS_datafile_separator << p_WS_W[2]
                << FLAGS_datafile_separator << q_WS.x()
                << FLAGS_datafile_separator << q_WS.y()
                << FLAGS_datafile_separator << q_WS.z()
                << FLAGS_datafile_separator << q_WS.w()
                << FLAGS_datafile_separator << speedAndBiases[0]
                << FLAGS_datafile_separator << speedAndBiases[1]
                << FLAGS_datafile_separator << speedAndBiases[2]
                << FLAGS_datafile_separator << speedAndBiases[3]
                << FLAGS_datafile_separator << speedAndBiases[4]
                << FLAGS_datafile_separator << speedAndBiases[5]
                << FLAGS_datafile_separator << speedAndBiases[6]
                << FLAGS_datafile_separator << speedAndBiases[7]
                << FLAGS_datafile_separator << speedAndBiases[8];

      for (int jack = 0; jack < vTgsa.size(); ++jack)
        *csvFile_ << FLAGS_datafile_separator << vTgsa[jack];

      for (size_t i = 0; i < extrinsics.size(); ++i) {
        Eigen::Vector3d p_SCi = extrinsics[i].r();
        *csvFile_ << FLAGS_datafile_separator << p_SCi[0]
                  << FLAGS_datafile_separator << p_SCi[1]
                  << FLAGS_datafile_separator << p_SCi[2];
      }

      for (int jack = 0; jack < vfckptdr.size(); ++jack)
        *csvFile_ << FLAGS_datafile_separator << vfckptdr[jack];

      for (int jack = 0; jack < vVariance.size(); ++jack)
        *csvFile_ << FLAGS_datafile_separator << std::sqrt(vVariance[jack]);

      *csvFile_ << std::endl;
    }
  }
}

// Set and publish landmarks.
void Publisher::publishLandmarksAsCallback(
    const okvis::Time & /*t*/, const okvis::MapPointVector &actualLandmarks,
    const okvis::MapPointVector &transferredLandmarks) {
  if (parameters_.publishing.publishLandmarks) {
    okvis::MapPointVector empty;
    setPoints(actualLandmarks, empty, transferredLandmarks);
    publishPoints();
  }
}

// Set and write landmarks to file.
void Publisher::csvSaveLandmarksAsCallback(
    const okvis::Time & /*t*/, const okvis::MapPointVector &actualLandmarks,
    const okvis::MapPointVector &transferredLandmarks) {
  okvis::MapPointVector empty;
  setPoints(actualLandmarks, empty, transferredLandmarks);
  if (csvLandmarksFile_) {
    if (csvLandmarksFile_->good()) {
      for (size_t l = 0; l < actualLandmarks.size(); ++l) {
        Eigen::Vector4d landmark = actualLandmarks.at(l).pointHomog;
        *csvLandmarksFile_
            << std::setprecision(19) << actualLandmarks.at(l).id
            << FLAGS_datafile_separator << std::scientific
            << std::setprecision(18) << landmark[0] << FLAGS_datafile_separator
            << landmark[1] << FLAGS_datafile_separator << landmark[2]
            << FLAGS_datafile_separator << landmark[3]
            << FLAGS_datafile_separator
            << actualLandmarks.at(l).quality
            // << FLAGS_datafile_separator << actualLandmarks.at(l).distance
            << std::endl;
      }
    }
  }
}

// Publish the last set points.
void Publisher::publishPoints() {
  pubPointsMatched_.publish(pointsMatched_);
  pubPointsUnmatched_.publish(pointsUnmatched_);
  pubPointsTransferred_.publish(pointsTransferred_);
}

// Set the images to be published next.
void Publisher::setImages(const std::vector<cv::Mat> &images) {
  // copy over
  images_.resize(images.size());
  for (size_t i = 0; i < images.size(); ++i) images_[i] = images[i];
}

// Add a pose to the path that is published next. The path contains a maximum
// of maxPathLength poses that are published. Once the maximum is reached,
// the last pose is copied in a new path message. The rest are deleted.
void Publisher::setPath(const okvis::kinematics::Transformation &T_WS) {
  if (path_.poses.size() >= parameters_.publishing.maxPathLength) {
    geometry_msgs::PoseStamped lastPose = path_.poses.back();
    path_.poses.clear();
    path_.poses.reserve(parameters_.publishing.maxPathLength);
    path_.poses.push_back(lastPose);
  }
  geometry_msgs::PoseStamped pose;
  pose.header.stamp = _t;
  pose.header.frame_id = "world";
  const Eigen::Vector3d &r = T_WS.r();
  pose.pose.position.x = r[0];
  pose.pose.position.y = r[1];
  pose.pose.position.z = r[2];
  const Eigen::Quaterniond &q = T_WS.q();
  pose.pose.orientation.x = q.x();
  pose.pose.orientation.y = q.y();
  pose.pose.orientation.z = q.z();
  pose.pose.orientation.w = q.w();

  path_.header.stamp = _t;
  path_.header.frame_id = "world";
  path_.poses.push_back(pose);
}

// Publish the last set images.
void Publisher::publishImages() {
  // advertise what's been missing:
  if (images_.size() != pubImagesVector_.size()) {
    pubImagesVector_.clear();
    for (size_t i = 0; i < images_.size(); ++i) {
      std::stringstream drawingNameStream;
      drawingNameStream << "okvis_drawing_" << i;
      imageTransportVector_.push_back(image_transport::ImageTransport(*nh_));
      pubImagesVector_.push_back(
          imageTransportVector_[i].advertise(drawingNameStream.str(), 10));
    }
  }

  // publish:
  for (size_t i = 0; i < images_.size(); ++i) {
    sensor_msgs::Image msg;
    std::stringstream cameraNameStream;
    cameraNameStream << "camera_" << i;
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = cameraNameStream.str();
    sensor_msgs::fillImage(msg, sensor_msgs::image_encodings::MONO8,
                           images_[i].rows, images_[i].cols,
                           images_[i].step.buf[0], images_[i].data);
    pubImagesVector_[i].publish(msg);
  }
}

// Publish the last set path.
void Publisher::publishPath() { pubPath_.publish(path_); }

}  // namespace okvis
