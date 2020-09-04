/**
 * @file    RiImuFactor.cpp
 * @brief   3-way IMU factor with right invariant errors.
 * @author  Jianzhu Huai
 */

#include <gtsam/RiImuFactor.h>

#include <gtsam/ImuFrontEnd.h>

#include <msckf/imu/ImuOdometry.h>

namespace gtsam {

const double RiPreintegratedImuMeasurements::kRelinThresholdGyro = 0.0001;
const double RiPreintegratedImuMeasurements::kRelinThresholdAccelerometer =
    0.0009;

void RiPreintegratedImuMeasurements::redoPreintegration(
    const RiExtendedPose3& x_i, const imuBias::ConstantBias& bias_i) {
  okvis::kinematics::Transformation T_WB(x_i.position(),
                                         x_i.rotation().toQuaternion());
  Eigen::Vector3d v_WB(x_i.velocity());
  Eigen::Matrix<double, 6, 1> bgba;
  bgba.head<3>() = bias_i.gyroscope();
  bgba.tail<3>() = bias_i.accelerometer();
  ImuErrorModel<double> iem(bgba);
  covariance_.setZero();
  jacobian_.setIdentity();

  okvis::ImuOdometry::propagationRightInvariantError(
      imuMeasurements_, imuParams_, T_WB, v_WB, iem, ti_, tj_, &covariance_,
      &jacobian_);

  // compute deltaXij_
  Eigen::Vector3d gW(0, 0, -imuParams_.g);
  double tij = (tj_ - ti_).toSec();
  Eigen::Quaterniond qij = x_i.rotation().toQuaternion().inverse() * T_WB.q();
  deltaXij_ = RiExtendedPose3(
      Rot3(qij), x_i.rotation().inverse() * (v_WB - x_i.velocity() - gW * tij),
      x_i.rotation().inverse() * (T_WB.r() - x_i.position() -
                                  x_i.velocity() * tij - gW * tij * tij * 0.5));

  // set bias linearization point.
  imuParams_.g0 = bias_i.gyroscope();
  imuParams_.a0 = bias_i.accelerometer();
  RiLin_ = x_i.rotation();
}

gtsam::Vector RiImuFactor::evaluateError(
    const RiExtendedPose3& x_i, const RiExtendedPose3& x_j,
    const imuBias::ConstantBias& bias_i,
    boost::optional<Matrix&> H1,
    boost::optional<Matrix&> H2,
    boost::optional<Matrix&> H3) const {
  Vector9 error;
  redo_ = redo_ || pim_.needToRedoPreintegration(bias_i);
  if (redo_) {
    pim_.redoPreintegration(x_i, bias_i);
    redoCounter_++;
    redo_ = false;
    //    if (redoCounter_ > 1) {
    //      LOG(INFO) << "pre-integration no. " << redoCounter_;
    //    }

    RiExtendedPose3 x_j_from_i = pim_.predict(x_i);
    error = RiExtendedPose3::Logmap(x_j * x_j_from_i.inverse());

    Eigen::Matrix<double, 9, 9> Jr_inv = geometry::SEK3Jr_inv(error);
    de_db_i_ = -Jr_inv * pim_.jacobian_.block<9, 6>(0, 9);
    biasJacobianReady_= true;
  } else {
    RiExtendedPose3 x_j_from_i = pim_.predict(x_i);
    // also account for the effect of bias.
    error = RiExtendedPose3::Logmap(x_j * x_j_from_i.inverse());
    if (biasJacobianReady_) {
      error += de_db_i_ * (bias_i - pim_.biasLin()).vector();
    } else {
      Eigen::Matrix<double, 9, 9> Jr_inv = geometry::SEK3Jr_inv(error);
      de_db_i_ = -Jr_inv * pim_.jacobian_.block<9, 6>(0, 9);
      biasJacobianReady_ = true;
    }
  }

  if (H1) {  // de / dx_i = de / dx_{j|i} * dx_{j|i} / dx_i
    *H1 = - geometry::SEK3Jr_inv(error) * pim_.jacobian_.topLeftCorner<9, 9>();
  }
  if (H2) {  // de / dx_j
    *H2 = geometry::SEK3Jl_inv(error);
  }
  if (H3) {  // de / db_i = de / dx_{j|i} * dx_{j|i} / db_i
    // The last term partly alleviates the inaccuracy caused by changing x_i.
    Eigen::Matrix3d incrementalRotation = (pim_.RotiLin().inverse() * x_i.rotation()).matrix();
    H3->resize(9, 6);
    Eigen::Matrix<double, 9, 9> Jr_inv = geometry::SEK3Jr_inv(error);
    H3->topLeftCorner<9, 3>() =
        -Jr_inv * pim_.jacobian_.block<9, 3>(0, 9) * incrementalRotation;
    H3->topRightCorner<9, 3>() =
        -Jr_inv * pim_.jacobian_.block<9, 3>(0, 12) * incrementalRotation;
  }
  return error;
}

gtsam::Vector RiImuFactor::evaluateErrorCheck(
    const RiExtendedPose3& state_i, const RiExtendedPose3& state_j,
    const imuBias::ConstantBias& bias_i) const {
  okvis::ImuFrontEnd::PimPtr combinedPim;
  okvis::ImuParams imuParamsKimera;
  imuParamsKimera.set(pim_.imuParameters());
  imuParamsKimera.imu_preintegration_type_ =
      okvis::ImuPreintegrationType::kPreintegratedCombinedMeasurements;

  okvis::ImuFrontEnd imuIntegrator(imuParamsKimera);
  Eigen::Matrix<double, 9, 1> speedAndBiasi;
  speedAndBiasi.segment<3>(3) = bias_i.gyroscope();
  speedAndBiasi.tail<3>() = bias_i.accelerometer();

  imuIntegrator.preintegrateImuMeasurements(
      pim_.imuMeasurements(), speedAndBiasi, pim_.ti(), pim_.tj(), combinedPim);

  gtsam::NavState navState_i(state_i.rotation(), state_i.position(),
                             state_i.velocity());

  gtsam::NavState predictedState_j = combinedPim->predict(navState_i, bias_i);
  RiExtendedPose3 predictedRiState_j(predictedState_j.pose().rotation(),
                                     predictedState_j.v(),
                                     predictedState_j.position());
  return predictedRiState_j.localCoordinates(state_j);
}

} // namespace gtsam
