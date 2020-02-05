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
 *  Created on: Jan 7, 2014
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file ode.hpp
 * @brief File for ODE integration functionality.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_MSCKF_IMU_ODOMETRY_LEGACY_HPP_
#define INCLUDE_MSCKF_IMU_ODOMETRY_LEGACY_HPP_

#include <msckf/ImuErrorModel.h>

#include <Eigen/Core>
//#include <okvis/FrameTypedefs.hpp>
//#include <okvis/Measurements.hpp>
//#include <okvis/Variables.hpp>
//#include <okvis/assert_macros.hpp>
//#include <okvis/kinematics/Transformation.hpp>
//#include <okvis/kinematics/operators.hpp>

#include <vio/eigen_utils.h>  //only for rvec2quat
#include <sophus/se3.hpp>     //only for first order propagation

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {
/// \brief ode Namespace for functionality related to ODE integration
/// implemented in okvis.
namespace ode {

// s0 can be any local world frame
// acc, m/s^2, estimated acc from imu in s frame with bias removed, gyro, rad/s,
// estimated angular rate by imu with bias removed for better accuracy within
// this function, (1) use a gravity model as simple as Heiskanen and Moritz
// 1967, (2) higher order integration method
template <typename Scalar>
void strapdown_local_quat_bias(const Eigen::Matrix<Scalar, 3, 1>& rs0,
                               const Eigen::Matrix<Scalar, 3, 1>& vs0,
                               const Eigen::Quaternion<Scalar>& qs0_2_s,
                               const Eigen::Matrix<Scalar, 3, 1>& a,
                               const Eigen::Matrix<Scalar, 3, 1>& w, Scalar dt,
                               const Eigen::Matrix<Scalar, 6, 1>& gomegas0,
                               Eigen::Matrix<Scalar, 3, 1>* rs0_new,
                               Eigen::Matrix<Scalar, 3, 1>* vs0_new,
                               Eigen::Quaternion<Scalar>* qs0_2_s_new) {
  Eigen::Matrix<Scalar, 3, 1> wie2s0 = gomegas0.template tail<3>();
  // Update attitude
  // method (1) second order integration
  Eigen::Quaternion<Scalar> qe = vio::rvec2quat(wie2s0 * dt);
  (*qs0_2_s_new) = qs0_2_s * qe;
  Eigen::Quaternion<Scalar> qb = vio::rvec2quat(-w * dt);
  (*qs0_2_s_new) = qb * (*qs0_2_s_new);

  //// method (2) Runge-Kutta 4th order integration, empirically, this sometimes
  // gives worse result than second order integration
  // wie2s=quatrot_v000(rvqs0(7:10),wie2s0,0);
  // omega=zeros(4,2);
  // omega(1,2)=dt;
  // omega(2:4,1)=w-wie2s;
  // omega(2:4,2)=lastw-wie2s;
  // qs2s0=rotationRK4( omega, [rvqs0(7); -rvqs0(8:10)]);
  // rvqs0_new(7:10)=[qs2s0(1); -qs2s0(2:4)];

  //// better velocity and position integration than first order rectanglar rule
  // Update Velocity
  //    Vector3d
  //    vel_inc1=(quatrot(qs0_2_s,a*dt,1)+quatrot(*qs0_2_s_new,a*dt,1))/2;
  Eigen::Matrix<Scalar, 3, 1> vel_inc1 =
      (qs0_2_s.conjugate()._transformVector(a * dt) +
       (*qs0_2_s_new).conjugate()._transformVector(a * dt)) *
      Scalar(0.5);

  Eigen::Matrix<Scalar, 3, 1> vel_inc2 =
      (gomegas0.template head<3>() - Scalar(2) * wie2s0.cross(vs0)) * dt;

  (*vs0_new) = vs0 + vel_inc1 + vel_inc2;
  // Update_pos
  (*rs0_new) = rs0 + ((*vs0_new) + vs0) * dt * Scalar(0.5);
}

/// returns the 3D cross product skew symmetric matrix of a given 3D vector
template <class Derived>
inline Eigen::Matrix<typename Derived::Scalar, 3, 3> skew3d(
    const Eigen::MatrixBase<Derived>& vec) {
  typedef typename Derived::Scalar Scalar;
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 3);
  return (Eigen::Matrix<typename Derived::Scalar, 3, 3>() << Scalar(0.0),
          -vec[2], vec[1], vec[2], Scalar(0.0), -vec[0], -vec[1], vec[0],
          Scalar(0.0))
      .finished();
}

// system process model in local world frame denoted by s0 ( which has no other
// meanings) with dcm formulation rs in s0, vs in s0, qs0 2s, covariance P
// corresponds to error states, \delta rs in w, \delta v s in w, \psi w, ba, bg.
// where \tilde{R}_s^w=(I-[\psi^w]_\times)R_s^w and \delta v_s^w=
// \tilde{v}_s^w-v_s^w acc bias is assumed random walk, gyro bias also random
// walk acc, m/s^2, estimated acc from imu in s frame with bias removed, gyro,
// rad/s, estimated angular rate by imu with bias removed w.r.t i frame
// coordinated in s frame, dt, time interval for covariance update q_n_a:
// accelerometer VRW noise density squared, q_n_ba, accelerometer bias noise
// density squared input P stores previous covariance, in output P sotres
// predicted covariance for better accuracy and possibly slower computation, use
// the covariance propagation model in ethz asl msf on github
template <typename Scalar>
void sys_local_dcm_bias(const Eigen::Matrix<Scalar, 3, 1>& /*rs0*/,
                        const Eigen::Matrix<Scalar, 3, 1>& /*vs0*/,
                        const Eigen::Quaternion<Scalar>& qs0_2_s,
                        const Eigen::Matrix<Scalar, 3, 1>& acc,
                        const Eigen::Matrix<Scalar, 3, 1>& /*gyro*/, Scalar dt,
                        const Eigen::Matrix<Scalar, 3, 1>& q_n_a,
                        const Eigen::Matrix<Scalar, 3, 1>& q_n_w,
                        const Eigen::Matrix<Scalar, 3, 1>& q_n_ba,
                        const Eigen::Matrix<Scalar, 3, 1>& q_n_bw,
                        Eigen::Matrix<Scalar, 15, 15>* P) {
  // system disturbance coefs
  const int navStates = 9;
  Eigen::Matrix<Scalar, navStates, 6> Nnav =
      Eigen::Matrix<Scalar, navStates, 6>::Zero();
  Eigen::Matrix<Scalar, 3, 3> Cs02s = qs0_2_s.toRotationMatrix();

  Nnav.template block<3, 3>(3, 0) = Cs02s.transpose();   // velocity
  Nnav.template block<3, 3>(6, 3) = -Cs02s.transpose();  // attitude
  // system matrix
  Eigen::Matrix<Scalar, navStates, navStates> Anav =
      Eigen::Matrix<Scalar, navStates, navStates>::Zero();
  Anav.template block<3, 3>(0, 3).template setIdentity();  // rs in s0
  // Velocity  // for low grade IMU, w_ie^e is buried
  Anav.template block<3, 3>(3, 6) = skew3d(Cs02s.transpose() * acc);
  // Imu error model parameters
  Eigen::Matrix<Scalar, 6, 6> Rimu = Eigen::Matrix<Scalar, 6, 6>::Zero();

  Rimu.template topLeftCorner<3, 3>() = q_n_a.asDiagonal();
  Rimu.template bottomRightCorner<3, 3>() = q_n_w.asDiagonal();
  Eigen::Matrix<Scalar, 6, 6> Qimu_d = Eigen::Matrix<Scalar, 6, 6>::Zero();
  Qimu_d.template topLeftCorner<3, 3>() = q_n_ba.asDiagonal() * dt;
  Qimu_d.template bottomRightCorner<3, 3>() = q_n_bw.asDiagonal() * dt;

  // Combine and discretize nav and imu models
  // this discretization can also be accomplished by Loan's matrix exponential
  // method, see sys_metric_phipsi_v000.m

  Eigen::Matrix<Scalar, navStates, navStates> Anav_d =
      Eigen::Matrix<Scalar, navStates, navStates>::Identity() +
      dt * Anav;  // Use 1st order taylor series to discretize Anav
  Eigen::Matrix<Scalar, navStates, navStates> Qnav =
      Nnav * Rimu * Nnav.transpose();
  Eigen::Matrix<Scalar, navStates, navStates> Qnav_d =
      dt * Scalar(0.5) *
      (Anav_d * Qnav +
       Qnav * Anav_d.transpose());  // Use trapezoidal rule to discretize Rimu

  Eigen::Matrix<Scalar, 15, 15> STM = Eigen::Matrix<Scalar, 15, 15>::Zero();

  STM.template topLeftCorner<navStates, navStates>() = Anav_d;
  STM.template block<navStates, 6>(0, navStates) = Nnav * dt;
  STM.template block<6, 6>(navStates, navStates).setIdentity();

  Eigen::Matrix<Scalar, 15, 15> Qd = Eigen::Matrix<Scalar, 15, 15>::Zero();

  Qd.template topLeftCorner<navStates, navStates>() = Qnav_d;
  Qd.template block<6, 6>(navStates, navStates) = Qimu_d;
  Qd.template block<navStates, 6>(0, navStates) =
      Nnav * Qimu_d * dt * Scalar(0.5);
  Qd.template block<6, navStates>(navStates, 0) =
      Qd.template block<navStates, 6>(0, navStates).template transpose();

  (*P) = STM * (*P) * STM.transpose() +
         Qd;  // covariance of the navigation states and imu error terms
}

// given pose/velocity of IMU sensor, i.e., T_sensor_to_world,
// v_sensor_in_world, and IMU biases at epoch t(k), i.e., time_pair[0],
// measurements from t(p^k-1) to t(p^{k+1}-1), where t(p^k-1) is the closest
// epoch to t(k) less or equal to t(k), and gravity in world frame in m/s^2
// which can be roughly computed using some EGM model or assume constant,
// e.g., 9.81 m/s^2 and earth rotation rate in world frame in rad/sec which can
// often be set to 0 predict states in terms of IMU sensor frame at epoch
// t(k+1), i.e., time_pair[1] optionally, propagate covariance in the local
// world frame, the covariance corresponds to states, \delta rs in w, \delta v s
// in w, \psi w, ba, bg. where \tilde{R}_s^w=(I-[\psi^w]_\times)R_s^w covariance
// of states can be treated more rigorously as in ethz asl sensor_fusion on
// github by Stephan Weiss P stores covariance at t(k), update it to t(k+1)
// optionally, shape_matrices which includes random constants T_g, T_s, T_a, can
// be used to correct IMU measurements. each measurement has timestamp in sec,
// gyro measurements in m/s^2, accelerometer measurements in m/s^2

template <typename Scalar>
void predictStates(
    const Sophus::SE3Group<Scalar>& T_sk_to_w,
    const Eigen::Matrix<Scalar, 9, 1>& speed_bias_k, const Scalar* time_pair,
    const std::vector<Eigen::Matrix<Scalar, 7, 1>,
                      Eigen::aligned_allocator<Eigen::Matrix<Scalar, 7, 1>>>&
        measurements,
    const Eigen::Matrix<Scalar, 6, 1>& gwomegaw,
    const Eigen::Matrix<Scalar, 12, 1>& q_n_aw_babw,
    Sophus::SE3Group<Scalar>* pred_T_skp1_to_w,
    Eigen::Matrix<Scalar, 3, 1>* pred_speed_kp1,
    Eigen::Matrix<Scalar, 15, 15>* P,
    const Eigen::Matrix<Scalar, 27, 1> shape_matrices =
        Eigen::Matrix<Scalar, 27, 1>::Zero()) {
  bool predict_cov = (P != NULL);
  int every_n_reading = 2;  // update covariance every n IMU readings,
  // the eventual covariance has little to do with this param as long as it
  // remains small
  Eigen::Matrix<Scalar, 3, 1> r_new, r_old(T_sk_to_w.translation()), v_new,
      v_old(speed_bias_k.template head<3>());
  Eigen::Quaternion<Scalar> q_new,
      q_old(T_sk_to_w.unit_quaternion().conjugate());
  Scalar dt = measurements[1][0] - time_pair[0];
  Scalar covupt_time(
      time_pair[0]);  // the time to which the covariance is updated. N.B. the
                      // initial covariance is updated to $t_k$

  assert(dt >= Scalar(0) && dt <= Scalar(1 + 1e-8));
  IMUErrorModel<Scalar> iem(speed_bias_k.template block<6, 1>(3, 0),
                            shape_matrices);
  iem.estimate(measurements[0].template block<3, 1>(1, 0),
               measurements[0].template block<3, 1>(4, 0));

  const Eigen::Matrix<Scalar, 3, 1> qna = q_n_aw_babw.template head<3>(),
                                    qnw = q_n_aw_babw.template segment<3>(3),
                                    qnba = q_n_aw_babw.template segment<3>(6),
                                    qnbw = q_n_aw_babw.template tail<3>();
  strapdown_local_quat_bias(r_old, v_old, q_old, iem.a_est, iem.w_est, dt,
                            gwomegaw, &r_new, &v_new, &q_new);

  if (predict_cov) {
    sys_local_dcm_bias(r_old, v_old, q_old, iem.a_est, iem.w_est,
                       measurements[1][0] - covupt_time, qna, qnw, qnba, qnbw,
                       P);
    // for more precise covariance update, we can use average estimated accel
    // and angular rate over n(every_n_reading) IMU readings
    covupt_time = measurements[1][0];
  }
  r_old = r_new;
  v_old = v_new;
  q_old = q_new;
  int unsigned i = 1;
  for (; i < measurements.size() - 1; ++i) {
    dt = measurements[i + 1][0] - measurements[i][0];
    iem.estimate(measurements[i].template block<3, 1>(1, 0),
                 measurements[i].template block<3, 1>(4, 0));
    strapdown_local_quat_bias(r_old, v_old, q_old, iem.a_est, iem.w_est, dt,
                              gwomegaw, &r_new, &v_new, &q_new);
    if (predict_cov && (i % every_n_reading == 0)) {
      sys_local_dcm_bias(r_old, v_old, q_old, iem.a_est, iem.w_est,
                         measurements[i + 1][0] - covupt_time, qna, qnw, qnba,
                         qnbw, P);
      covupt_time = measurements[i + 1][0];
    }
    r_old = r_new;
    v_old = v_new;
    q_old = q_new;
  }
  // assert(i==measurements.size()-1);
  dt = time_pair[1] - measurements[i][0];  // the last measurement
  assert(dt >= Scalar(0) && dt < Scalar(0.01));
  iem.estimate(measurements[i].template block<3, 1>(1, 0),
               measurements[i].template block<3, 1>(4, 0));
  strapdown_local_quat_bias(r_old, v_old, q_old, iem.a_est, iem.w_est, dt,
                            gwomegaw, &r_new, &v_new, &q_new);
  if (predict_cov) {
    sys_local_dcm_bias(r_old, v_old, q_old, iem.a_est, iem.w_est,
                       time_pair[1] - covupt_time, qna, qnw, qnba, qnbw, P);
    covupt_time = time_pair[1];
  }
  pred_T_skp1_to_w->setQuaternion(q_new.conjugate());
  pred_T_skp1_to_w->translation() = r_new;
  (*pred_speed_kp1) = v_new;
}

}  // namespace ode

}  // namespace ceres
}  // namespace okvis

#endif // INCLUDE_MSCKF_IMU_ODOMETRY_LEGACY_HPP_
