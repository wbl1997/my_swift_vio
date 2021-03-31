/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   ImuFrontEndParams.h
 * @brief  Params for ImuFrontEnd
 * @author Antoni Rosinol
 */

#pragma once

#include <gtsam/base/Vector.h>
#include "okvis/PipelineParams.h"
#include "okvis/Parameters.hpp"

#include "gtsam/ImuFrontEnd-definitions.h"
#include "okvis/class_macros.hpp"

namespace okvis {

struct ImuParams : public PipelineParams {
 public:
  POINTER_TYPEDEFS(ImuParams);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuParams();

  void set(const okvis::ImuParameters& okvisImuParams);

  virtual ~ImuParams() = default;

 public:
  bool parseYAML(const std::string& filepath);
  void print() const;

protected:
  bool equals(const PipelineParams& obj) const;

 public:
  ImuPreintegrationType imu_preintegration_type_ =
      ImuPreintegrationType::kPreintegratedImuMeasurements;

  double gyro_noise_ = 0.0;
  double gyro_walk_ = 0.0;
  double acc_noise_ = 0.0;
  double acc_walk_ = 0.0;
  double imu_shift_ = 0.0;  // Defined as t_imu = t_cam + imu_shift

  double nominal_rate_ = 100.0;
  double imu_integration_sigma_ = 1e-8;
  Eigen::Matrix<double, 6, 1> biasAccOmegaInit_ = Eigen::Matrix<double, 6, 1>::Zero();
  gtsam::Vector3 n_gravity_ = gtsam::Vector3(0, 0, -9.81);
};

}  // namespace okvis
