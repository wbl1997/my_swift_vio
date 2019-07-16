#ifndef IMU_ODOMETRY_H__
#define IMU_ODOMETRY_H__
#include <vector>

#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/Time.hpp>
#include <okvis/Variables.hpp>
#include <okvis/assert_macros.hpp>

#include "msckf/odeHybrid.hpp"

namespace okvis {

class IMUOdometry {
  /// \brief The type of the covariance.
  typedef Eigen::Matrix<double, 15, 15> covariance_t;

  /// \brief The type of the information (same matrix dimension as covariance).
  typedef covariance_t information_t;

  /// \brief The type of hte overall Jacobian.
  typedef Eigen::Matrix<double, 15, 15> jacobian_t;

 public:
  /**
   * @brief Propagates pose, speeds and biases with given IMU measurements.
   * @remark This can be used externally to perform propagation
   * @param[in] imuMeasurements All the IMU measurements.
   * @param[in] imuParams The parameters to be used.
   * @param[inout] T_WS Start pose.
   * @param[inout] speedAndBiases Start speed and biases.
   * @param[in] t_start Start time.
   * @param[in] t_end End time.
   * @param[out] covariance Covariance for GIVEN start states.
   * @param[out] jacobian Jacobian w.r.t. start states.
   * @param[in] linearizationPointAtTStart is the first estimates of position
   * p_WS and velocity v_WS at t_start
   * @return Number of integration steps.
   * assume W frame has z axis pointing up
   * Euler approximation is used to incrementally compute the integrals, and
   * the length of integral interval only adversely affect the covariance and
   * jacobian a little.
   */
  static int propagation(
      const okvis::ImuMeasurementDeque& imuMeasurements,
      const okvis::ImuParameters& imuParams,
      okvis::kinematics::Transformation& T_WS, Eigen::Vector3d& v_WS,
      IMUErrorModel<double>& iem, const okvis::Time& t_start,
      const okvis::Time& t_end,
      Eigen::Matrix<double, ceres::ode::OdoErrorStateDim,
                    ceres::ode::OdoErrorStateDim>* covariance_t = 0,
      Eigen::Matrix<double, ceres::ode::OdoErrorStateDim,
                    ceres::ode::OdoErrorStateDim>* jacobian = 0,
      const Eigen::Matrix<double, 6, 1>* linearizationPointAtTStart = 0);
  // a copy of the original implementation forreference
  static int propagation_original(
      const okvis::ImuMeasurementDeque& imuMeasurements,
      const okvis::ImuParameters& imuParams,
      okvis::kinematics::Transformation& T_WS, Eigen::Vector3d& v_WS,
      IMUErrorModel<double>& iem, const okvis::Time& t_start,
      const okvis::Time& t_end,
      Eigen::Matrix<double, ceres::ode::OdoErrorStateDim,
                    ceres::ode::OdoErrorStateDim>* covariance_t = 0,
      Eigen::Matrix<double, ceres::ode::OdoErrorStateDim,
                    ceres::ode::OdoErrorStateDim>* jacobian = 0);

  // t_start is greater than t_end
  // this function does not support backward covariance propagation
  static int propagationBackward(
      const okvis::ImuMeasurementDeque& imuMeasurements,
      const okvis::ImuParameters& imuParams,
      okvis::kinematics::Transformation& T_WS, Eigen::Vector3d& v_WS,
      IMUErrorModel<double>& iem, const okvis::Time& t_start,
      const okvis::Time& t_end);

  static int propagation_RungeKutta(
      const okvis::ImuMeasurementDeque& imuMeasurements,
      const okvis::ImuParameters& imuParams,
      okvis::kinematics::Transformation& T_WS,
      okvis::SpeedAndBiases& speedAndBias,
      const Eigen::Matrix<double, 27, 1>& vTgTsTa, const okvis::Time& t_start,
      const okvis::Time& t_end,
      Eigen::Matrix<double, okvis::ceres::ode::OdoErrorStateDim,
                    okvis::ceres::ode::OdoErrorStateDim>* P_ptr = 0,
      Eigen::Matrix<double, okvis::ceres::ode::OdoErrorStateDim,
                    okvis::ceres::ode::OdoErrorStateDim>* F_tot_ptr = 0);

  // propagate pose, speedAndBias
  // startTime is greater than finishTime
  // note this method assumes that the z direction of the world frame is
  // negative gravity direction
  static int propagationBackward_RungeKutta(
      const okvis::ImuMeasurementDeque& imuMeasurements,
      const okvis::ImuParameters& imuParams,
      okvis::kinematics::Transformation& T_WS,
      okvis::SpeedAndBiases& speedAndBias,
      const Eigen::Matrix<double, 27, 1>& vTgTsTa, const okvis::Time& t_start,
      const okvis::Time& t_end);

  // this function only changed intermediate members of IMUErrorModel
  static void interpolateInertialData(const okvis::ImuMeasurementDeque& imuMeas,
                                      IMUErrorModel<double>& iem,
                                      const okvis::Time& queryTime,
                                      okvis::ImuMeasurement& queryValue);

  static int propagation_leutenegger_corrected(
      const okvis::ImuMeasurementDeque& imuMeasurements,
      const okvis::ImuParameters& imuParams,
      okvis::kinematics::Transformation& T_WS,
      okvis::SpeedAndBias& speedAndBiases, const okvis::Time& t_start,
      const okvis::Time& t_end, covariance_t* covariance = 0,
      jacobian_t* jacobian = 0);
};
}  // namespace okvis
#endif
