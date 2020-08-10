/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   ImuFrontEnd.h
 * @brief  Class to preintegrate IMU measurements.
 * @author Antoni Rosinol, Luca Carlone
 */

#pragma once

#include <map>
#include <string>
#include <tuple>
#include <thread>
#include <utility>
#include <mutex>

#include <Eigen/Dense>

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/navigation/AHRSFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>  // Used if IMU combined is off.
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ImuFactor.h>

#include "gtsam/ImuFrontEnd-definitions.h"
#include "gtsam/ImuFrontEndParams.h"

#include <okvis/Measurements.hpp>
#include "okvis/class_macros.hpp"

namespace okvis {
/*
 * Class implementing the Imu Front End preintegration.
 * Construct using imu_params, and optionally an initial ImuBias.
 * Call preintegrateImuMeasurements to progressively integrate a bunch of Imu
 * measurements.
 * Eventually, call resetIntegrationAndSetBias when you would like to update
 * the Imu bias.
 */
class ImuFrontEnd {
public:
 using PimPtr = std::shared_ptr<gtsam::PreintegrationType>;
 using PimUniquePtr = std::unique_ptr<gtsam::PreintegrationType>;

public:
 POINTER_TYPEDEFS(ImuFrontEnd);
 DELETE_COPY_CONSTRUCTORS(ImuFrontEnd);
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 /* ------------------------------------------------------------------------
  * Class to do IMU preintegration.
  * [in] imu_params: IMU parameters used for the preintegration.
  * [in] imu_bias: IMU bias used to initialize PreintegratedImuMeasurements
  * !! The user of this class must update the bias and reset the integration
  * manually in order to preintegrate the IMU with the latest IMU bias coming
  * from the backend optimization.
  */
 ImuFrontEnd(const ImuParams& imu_params);
 ~ImuFrontEnd() = default;

 /* ------------------------------------------------------------------------ */
 int preintegrateImuMeasurements(
     const okvis::ImuMeasurementDeque& imuMeasurements,
     const okvis::SpeedAndBias& speedAndBias, const okvis::Time t_start,
     const okvis::Time t_end, ImuFrontEnd::PimPtr& resultPim) const;

 int preintegrateImuMeasurements(
     const std::vector<okvis::Time>& imu_timestamps,
     const ImuAccGyrS&  accGyrSeq,
     const gtsam::imuBias::ConstantBias accGyrBias, const okvis::Time t_start,
     const okvis::Time t_end, ImuFrontEnd::PimPtr& resultPim) const;

  /* ------------------------------------------------------------------------ */
  // Reset gravity value in pre-integration.
  // This is needed for the online initialization.
  // THREAD-SAFE.
  inline void resetPreintegrationGravity(const gtsam::Vector3& reset_value) {
    LOG(WARNING) << "Resetting value of gravity in ImuFrontEnd to: "
                 << reset_value;
    pim_->params()->n_gravity = reset_value;
    CHECK(gtsam::assert_equal(pim_->params()->getGravity(), reset_value));
    // TODO(Toni): should we update imu_params n_gravity for consistency?
    // imu_params_.n_gravity_ = reset_value;
  }

  /* ------------------------------------------------------------------------ */
  // THREAD-SAFE.
  inline ImuPreintegrationType getImuPreintegrationType() const {
    return imu_params_.imu_preintegration_type_;
  }

  inline gtsam::Vector3 getPreintegrationGravity() const {
    return imu_params_.n_gravity_;
  }

  /* ------------------------------------------------------------------------ */
  inline gtsam::PreintegrationType::Params getGtsamImuParams() const {
    return *(pim_->params());
  }

  /* ------------------------------------------------------------------------ */
  // Convert parameters for imu preintegration from the given ImuParams.
  static gtsam::PreintegrationType::Params convertVioImuParamsToGtsam(
      const ImuParams& imu_params);

  static boost::shared_ptr<gtsam::PreintegratedCombinedMeasurements::Params>
  generateCombinedImuParams(const ImuParams& imu_params);

  static boost::shared_ptr<gtsam::PreintegratedImuMeasurements::Params>
  generateRegularImuParams(const ImuParams& imu_params);

 private:
  void initializeImuFrontEnd();

 private:
  ImuParams imu_params_;
  PimUniquePtr pim_ = nullptr;
};

} // End of okvis namespace.
