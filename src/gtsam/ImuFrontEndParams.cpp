/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   ImuFrontEndParams.cpp
 * @brief  Params for ImuFrontEnd.
 * @author Antoni Rosinol
 */

#include "gtsam/ImuFrontEndParams.h"

#include <glog/logging.h>

#include <gtsam/geometry/Pose3.h>

//#include "kimera-vio/utils/UtilsOpenCV.h"
#include "io_wrap/YamlParser.h"

namespace swift_vio {
ImuParams::ImuParams() : PipelineParams("IMU params") {
}

void ImuParams::set(const okvis::ImuParameters& imuParameters) {
  gyro_noise_ = imuParameters.sigma_g_c;
  gyro_walk_ = imuParameters.sigma_gw_c;
  acc_noise_ = imuParameters.sigma_a_c;
  acc_walk_ = imuParameters.sigma_aw_c;
  imu_shift_ = 0.0;
  nominal_rate_ = imuParameters.rate;
  imu_integration_sigma_ = 1e-8;
  biasAccOmegaInit_ = Eigen::Matrix<double, 6, 1>::Zero();
  n_gravity_ = Eigen::Vector3d(0, 0, -imuParameters.g);
}

bool ImuParams::parseYAML(const std::string& filepath) {
  YamlParser yaml_parser(filepath);

  int imu_preintegration_type;
  yaml_parser.getYamlParam("imu_preintegration_type", &imu_preintegration_type);
  imu_preintegration_type_ =
      static_cast<ImuPreintegrationType>(imu_preintegration_type);

  // Rows and cols are redundant info, since the pose 4x4, but we parse just
  // to check we are all on the same page.
  // int n_rows = 0;
  // yaml_parser.getNestedYamlParam("T_BS", "rows", &n_rows);
  // CHECK_EQ(n_rows, 4u);
  // int n_cols = 0;
  // yaml_parser.getNestedYamlParam("T_BS", "cols", &n_cols);
  // CHECK_EQ(n_cols, 4u);
  std::vector<double> vector_pose;
  yaml_parser.getNestedYamlParam("T_BS", "data", &vector_pose);
//  const gtsam::Pose3& body_Pose_cam =
//      UtilsOpenCV::poseVectorToGtsamPose3(vector_pose);

  // Sanity check: IMU is usually chosen as the body frame.
//  LOG_IF(FATAL, !body_Pose_cam.equals(gtsam::Pose3()))
//      << "parseImuData: we expected identity body_Pose_cam_: is everything ok?";

  int rate = 0;
  yaml_parser.getYamlParam("rate_hz", &rate);
  CHECK_GT(rate, 0u);
  nominal_rate_ = 1 / static_cast<double>(rate);

  // IMU PARAMS
  yaml_parser.getYamlParam("gyroscope_noise_density", &gyro_noise_);
  yaml_parser.getYamlParam("accelerometer_noise_density", &acc_noise_);
  yaml_parser.getYamlParam("gyroscope_random_walk", &gyro_walk_);
  yaml_parser.getYamlParam("accelerometer_random_walk", &acc_walk_);
  yaml_parser.getYamlParam("imu_integration_sigma", &imu_integration_sigma_);
  yaml_parser.getYamlParam("imu_time_shift", &imu_shift_);
  std::vector<double> n_gravity;
  yaml_parser.getYamlParam("n_gravity", &n_gravity);
  CHECK_EQ(n_gravity.size(), 3);
  for (int k = 0; k < 3; k++) n_gravity_(k) = n_gravity[k];

  return true;
}

void ImuParams::print() const {
  std::stringstream out;
  PipelineParams::print(out,
                        "gyroscope_noise_density: ",
                        gyro_noise_,
                        "gyroscope_random_walk: ",
                        gyro_walk_,
                        "accelerometer_noise_density: ",
                        acc_noise_,
                        "accelerometer_random_walk: ",
                        acc_walk_,
                        "imu_integration_sigma: ",
                        imu_integration_sigma_,
                        "imu_time_shift: ",
                        imu_shift_,
                        "n_gravity: ",
                        n_gravity_);
  LOG(INFO) << out.str();
}

bool ImuParams::equals(const PipelineParams& obj) const {
  const auto& rhs = static_cast<const ImuParams&>(obj);
  // clang-format off
  return imu_preintegration_type_ == rhs.imu_preintegration_type_ &&
      gyro_noise_ == rhs.gyro_noise_ &&
      gyro_walk_ == rhs.gyro_walk_ &&
      acc_noise_ == rhs.acc_noise_ &&
      acc_walk_ == rhs.acc_walk_ &&
      imu_shift_ == rhs.imu_shift_ &&
      nominal_rate_ == rhs.nominal_rate_ &&
      imu_integration_sigma_ == rhs.imu_integration_sigma_ &&
      n_gravity_ == rhs.n_gravity_;
  // clang-format on
}
}  // namespace swift_vio
