/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   ImuFrontEnd.cpp
 * @brief  Class to preintegrate IMU measurements.
 * @author Antoni Rosinol
 * @author Luca Carlone
 */

#include "gtsam/ImuFrontEnd.h"

#include <glog/logging.h>

//#include "kimera-vio/common/vio_types.h"
#include "gtsam/ImuFrontEnd-definitions.h"
//#include "kimera-vio/utils/UtilsNumerical.h"

namespace swift_vio {
ImuFrontEnd::ImuFrontEnd(const ImuParams& imu_params)
    : imu_params_(imu_params) {
  CHECK_GT(imu_params.acc_noise_, 0.0);
  CHECK_GT(imu_params.acc_walk_, 0.0);
  CHECK_GT(imu_params.gyro_noise_, 0.0);
  CHECK_GT(imu_params.gyro_walk_, 0.0);
  CHECK_GT(imu_params.imu_integration_sigma_, 0.0);
  LOG_IF(WARNING, imu_params.imu_shift_ != 0.0)
      << "Applying IMU timestamp shift of: " << imu_params.imu_shift_ << "ns.";
  initializeImuFrontEnd();
}

void ImuFrontEnd::initializeImuFrontEnd() {
  gtsam::imuBias::ConstantBias imu_bias;
  switch (imu_params_.imu_preintegration_type_) {
    case ImuPreintegrationType::kPreintegratedCombinedMeasurements: {
      pim_ = std::unique_ptr<gtsam::PreintegratedCombinedMeasurements>(
          new gtsam::PreintegratedCombinedMeasurements(
              generateCombinedImuParams(imu_params_), imu_bias));
      break;
    }
    case ImuPreintegrationType::kPreintegratedImuMeasurements: {
      pim_ = std::unique_ptr<gtsam::PreintegratedImuMeasurements>(
          new gtsam::PreintegratedImuMeasurements(
              generateRegularImuParams(imu_params_), imu_bias));
      break;
    }
    default: {
      LOG(ERROR) << "Unknown Imu frontend type.";
      break;
    }
  }
  CHECK(pim_);

  if (VLOG_IS_ON(10)) {
    LOG(ERROR) << "IMU PREINTEGRATION PARAMS GIVEN TO IMU FRONTEND.";
    imu_params_.print();
    LOG(ERROR) << "IMU PREINTEGRATION COVARIANCE: ";
    pim_->print("PIM type: " + std::to_string(to_underlying(
                                   imu_params_.imu_preintegration_type_)));
  }
}

int ImuFrontEnd::preintegrateImuMeasurements(
    const std::vector<okvis::Time>& imu_timestamps, const ImuAccGyrS& accGyrSeq,
    const gtsam::imuBias::ConstantBias accGyrBias, const okvis::Time t_start,
    const okvis::Time t_end, ImuFrontEnd::PimPtr& resultPim) const {
  okvis::ImuMeasurementDeque imuMeasurements;
  imuMeasurements.resize(imu_timestamps.size());
  for (size_t i = 0u; i < imu_timestamps.size(); ++i) {
    imuMeasurements.emplace_back(okvis::ImuMeasurement(
        imu_timestamps[i], okvis::ImuSensorReadings(accGyrSeq.block<3, 1>(3, i),
                                             accGyrSeq.block<3, 1>(0, i))));
  }
  okvis::SpeedAndBias speedAndBias = okvis::SpeedAndBias::Zero();
  speedAndBias.segment<3>(3) = accGyrBias.gyroscope();
  speedAndBias.tail<3>() = accGyrBias.accelerometer();
  return preintegrateImuMeasurements(imuMeasurements, speedAndBias, t_start, t_end,
                              resultPim);
}

/* -------------------------------------------------------------------------- */
int ImuFrontEnd::preintegrateImuMeasurements(
    const okvis::ImuMeasurementDeque& imuMeasurements,
    const okvis::SpeedAndBias& speedAndBias, const okvis::Time t_start,
    const okvis::Time t_end, ImuFrontEnd::PimPtr& resultPim) const {
  Eigen::Vector3d bg = speedAndBias.segment<3>(3);
  Eigen::Vector3d ba = speedAndBias.tail<3>();
  gtsam::imuBias::ConstantBias bias(
      ba, bg);  // Biases (acc, rot)
  pim_->resetIntegrationAndSetBias(bias);

  // now the propagation
  okvis::Time time = t_start;
  okvis::Time end = t_end;

  // sanity check:
  assert(imuMeasurements.front().timeStamp <= time);

  if (!(imuMeasurements.back().timeStamp >= end)) {
    throw std::runtime_error(
        "in CreatePreintegratedMeasurement of a keyframe imuMeas back "
        "timeStamp less than end time!");
    resultPim.reset();
    return -1;  // nothing to do...
  }

  double Delta_t = 0;
  bool hasStarted = false;
  int i = 0;
  for (okvis::ImuMeasurementDeque::const_iterator it = imuMeasurements.begin();
       it != imuMeasurements.end(); ++it) {
    Eigen::Vector3d omega_S_0 = it->measurement.gyroscopes;
    Eigen::Vector3d acc_S_0 = it->measurement.accelerometers;
    Eigen::Vector3d omega_S_1 = (it + 1)->measurement.gyroscopes;
    Eigen::Vector3d acc_S_1 = (it + 1)->measurement.accelerometers;

    // time delta
    okvis::Time nexttime;
    if ((it + 1) == imuMeasurements.end()) {
      nexttime = t_end;
    } else
      nexttime = (it + 1)->timeStamp;
    double dt = (nexttime - time).toSec();

    if (end < nexttime) {
      double interval = (nexttime - it->timeStamp).toSec();
      nexttime = t_end;
      dt = (nexttime - time).toSec();
      const double r = dt / interval;
      omega_S_1 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
      acc_S_1 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();
    }

    if (dt <= 0.0) {
      continue;
    }
    Delta_t += dt;

    if (!hasStarted) {
      hasStarted = true;
      const double r = dt / (nexttime - it->timeStamp).toSec();
      omega_S_0 = (r * omega_S_0 + (1.0 - r) * omega_S_1).eval();
      acc_S_0 = (r * acc_S_0 + (1.0 - r) * acc_S_1).eval();
    }

    /*
    // ensure integrity
    if (fabs(omega_S_0[0]) > imuParams.g_max
            || fabs(omega_S_0[1]) > imuParams.g_max
            || fabs(omega_S_0[2]) > imuParams.g_max
            || fabs(omega_S_1[0]) > imuParams.g_max
            || fabs(omega_S_1[1]) > imuParams.g_max
            || fabs(omega_S_1[2]) > imuParams.g_max) {

        SLAM_ERROR_STREAM("gyr saturation");
    }

    if (fabs(acc_S_0[0]) > imuParams.a_max || fabs(acc_S_0[1]) > imuParams.a_max
            || fabs(acc_S_0[2]) > imuParams.a_max
            || fabs(acc_S_1[0]) > imuParams.a_max
            || fabs(acc_S_1[1]) > imuParams.a_max
            || fabs(acc_S_1[2]) > imuParams.a_max) {
        SLAM_ERROR_STREAM("acc saturation");
    }*/

    // actual propagation
    const Eigen::Vector3d omega_S = 0.5 * (omega_S_0 + omega_S_1);
    const Eigen::Vector3d acc_S = 0.5 * (acc_S_0 + acc_S_1);

    assert(fabs(dt) > 1e-9);  // check the first dt, becuase if its zero, nan
                              // comes into preintegrated covariance
    pim_->integrateMeasurement(acc_S, omega_S, dt);

    // memory shift
    time = nexttime;
    ++i;

    if (nexttime == t_end) break;
  }
  assert(fabs(Delta_t - (t_end - t_start).toSec()) < 1e-9);
  // Create a copy of the current pim, because the ImuFrontEnd pim will be
  // reused over and over. Avoid object slicing by using the derived type of
  // pim. All of this is to deal with gtsam's idiosyncracies with base classes.
  switch (imu_params_.imu_preintegration_type_) {
    case ImuPreintegrationType::kPreintegratedCombinedMeasurements:
      resultPim.reset(new gtsam::PreintegratedCombinedMeasurements(
          safeCastToPreintegratedCombinedImuMeasurements(*pim_)));
      break;
    case ImuPreintegrationType::kPreintegratedImuMeasurements:
      resultPim.reset(new gtsam::PreintegratedImuMeasurements(
          safeCastToPreintegratedImuMeasurements(*pim_)));
      break;
    default:
      resultPim.reset();
      LOG(FATAL) << "Unknown IMU Preintegration Type.";
      break;
  }
  return i;
}

/* -------------------------------------------------------------------------- */
gtsam::PreintegrationBase::Params ImuFrontEnd::convertVioImuParamsToGtsam(
    const ImuParams& imu_params) {
  CHECK_GT(imu_params.acc_noise_, 0.0);
  CHECK_GT(imu_params.gyro_noise_, 0.0);
  CHECK_GT(imu_params.imu_integration_sigma_, 0.0);
  gtsam::PreintegrationBase::Params preint_imu_params(imu_params.n_gravity_);
  preint_imu_params.gyroscopeCovariance =
      std::pow(imu_params.gyro_noise_, 2.0) * Eigen::Matrix3d::Identity();
  preint_imu_params.accelerometerCovariance =
      std::pow(imu_params.acc_noise_, 2.0) * Eigen::Matrix3d::Identity();
  preint_imu_params.integrationCovariance =
      std::pow(imu_params.imu_integration_sigma_, 2.0) *
      Eigen::Matrix3d::Identity();
  // TODO(Toni): REMOVE HARDCODED
  preint_imu_params.use2ndOrderCoriolis = false;

  return preint_imu_params;
}


boost::shared_ptr<gtsam::PreintegratedCombinedMeasurements::Params>
ImuFrontEnd::generateCombinedImuParams(const ImuParams& imu_params) {
  boost::shared_ptr<gtsam::PreintegratedCombinedMeasurements::Params>
      combined_imu_params =
          boost::make_shared<gtsam::PreintegratedCombinedMeasurements::Params>(
              imu_params.n_gravity_);
  gtsam::PreintegrationParams gtsam_imu_params =
      ImuFrontEnd::convertVioImuParamsToGtsam(imu_params);
  if (gtsam_imu_params.body_P_sensor) {
    combined_imu_params->setBodyPSensor(*gtsam_imu_params.getBodyPSensor());
  }
  if (gtsam_imu_params.omegaCoriolis) {
    combined_imu_params->setOmegaCoriolis(*gtsam_imu_params.getOmegaCoriolis());
  }
  combined_imu_params->setGyroscopeCovariance(
      gtsam_imu_params.getGyroscopeCovariance());
  combined_imu_params->setUse2ndOrderCoriolis(
      gtsam_imu_params.getUse2ndOrderCoriolis());
  combined_imu_params->setIntegrationCovariance(
      gtsam_imu_params.getIntegrationCovariance());
  combined_imu_params->setAccelerometerCovariance(
      gtsam_imu_params.getAccelerometerCovariance());
  ///< covariance of bias used for pre-integration
  // TODO(Toni): how come we are initializing like this?
  // We should parametrize perhaps this as well.
  combined_imu_params->biasAccOmegaInt = imu_params.biasAccOmegaInit_.cwiseAbs2().asDiagonal();
  ///< continuous-time "Covariance" describing
  ///< accelerometer bias random walk
  combined_imu_params->biasAccCovariance =
      std::pow(imu_params.acc_walk_, 2.0) * Eigen::Matrix3d::Identity();
  ///< continuous-time "Covariance" describing gyroscope bias random walk
  combined_imu_params->biasOmegaCovariance =
      std::pow(imu_params.gyro_walk_, 2.0) * Eigen::Matrix3d::Identity();
  return combined_imu_params;
}

boost::shared_ptr<gtsam::PreintegratedImuMeasurements::Params>
ImuFrontEnd::generateRegularImuParams(const ImuParams& imu_params) {
  boost::shared_ptr<gtsam::PreintegratedImuMeasurements::Params>
      regular_imu_params =
          boost::make_shared<gtsam::PreintegratedImuMeasurements::Params>(
              imu_params.n_gravity_);
  gtsam::PreintegrationParams gtsam_imu_params =
      ImuFrontEnd::convertVioImuParamsToGtsam(imu_params);
  if (gtsam_imu_params.body_P_sensor) {
    regular_imu_params->setBodyPSensor(*gtsam_imu_params.getBodyPSensor());
  }
  if (gtsam_imu_params.omegaCoriolis) {
    regular_imu_params->setOmegaCoriolis(*gtsam_imu_params.getOmegaCoriolis());
  }
  regular_imu_params->setGyroscopeCovariance(
      gtsam_imu_params.getGyroscopeCovariance());
  regular_imu_params->setUse2ndOrderCoriolis(
      gtsam_imu_params.getUse2ndOrderCoriolis());
  regular_imu_params->setIntegrationCovariance(
      gtsam_imu_params.getIntegrationCovariance());
  regular_imu_params->setAccelerometerCovariance(
      gtsam_imu_params.getAccelerometerCovariance());
  return regular_imu_params;
}

}  // namespace swift_vio
