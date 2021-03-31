/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   testImuFrontEnd.cpp
 * @brief  Unit tests ImuFrontEnd class' functionality.
 * @author Antoni Rosinol
 */

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "gtsam/ImuFrontEnd.h"
#include "gtsam/ImuFrontEndParams.h"

namespace okvis {

/* -------------------------------------------------------------------------- */
TEST(ImuFrontEnd, ImuFrontEndInitialization) {
  // Check that IMU frontend instantiation works.
  ImuParams imu_params;
  imu_params.acc_walk_ = 1.0;
  imu_params.acc_noise_ = 1.0;
  imu_params.gyro_walk_ = 1.0;
  imu_params.gyro_noise_ = 1.0;
  imu_params.n_gravity_ << 1.0, 1.0, 1.0;
  imu_params.imu_integration_sigma_ = 1.0;
  imu_params.imu_preintegration_type_ =
      ImuPreintegrationType::kPreintegratedImuMeasurements;
  Eigen::Vector3d bias_acc(1.0, 1.0, 1.0);
  Eigen::Vector3d bias_gyr(1.0, 1.0, 1.0);
  gtsam::imuBias::ConstantBias imu_bias(bias_acc, bias_gyr);
  ImuFrontEnd imu_frontend(imu_params);
  EXPECT_EQ(imu_frontend.getImuPreintegrationType(),
            ImuPreintegrationType::kPreintegratedImuMeasurements);
}

/* -------------------------------------------------------------------------- */
TEST(ImuFrontEnd, UpdateBias) {
  // Check that IMU frontend bias update works.
  ImuParams imu_params;
  imu_params.acc_walk_ = 1.0;
  imu_params.acc_noise_ = 1.0;
  imu_params.gyro_walk_ = 1.0;
  imu_params.gyro_noise_ = 1.0;
  imu_params.n_gravity_ << 1.0, 1.0, 1.0;
  imu_params.imu_integration_sigma_ = 1.0;
  Eigen::Vector3d bias_acc(1.0, 1.0, 1.0);
  Eigen::Vector3d bias_gyr(1.0, 1.0, 1.0);
  gtsam::imuBias::ConstantBias imu_bias(bias_acc, bias_gyr);
  ImuFrontEnd imu_frontend(imu_params);

  std::vector<okvis::Time> imu_timestamps{okvis::Time(1.0), okvis::Time(2.0)};
  ImuAccGyrS imu_measurements = Eigen::MatrixXd::Random(6, 2);

  ImuFrontEnd::PimPtr resultPim;
  imu_frontend.preintegrateImuMeasurements(imu_timestamps, imu_measurements,
                                           imu_bias, okvis::Time(1.0),
                                           okvis::Time(2.0), resultPim);
  EXPECT_TRUE(resultPim->biasHat().equals(imu_bias));
}

}  // namespace okvis
