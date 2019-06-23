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

#ifndef INCLUDE_OKVIS_CERES_ODE_ODE_HPP_
#define INCLUDE_OKVIS_CERES_ODE_ODE_HPP_

#include "vio/IMUErrorModel.h"

#include <Eigen/Core>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Variables.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/kinematics/operators.hpp>

#include <vio/eigen_utils.h>  //only for rvec2quat
#include <sophus/se3.hpp>     //only for first order propagation

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {
/// \brief ode Namespace for functionality related to ODE integration
/// implemented in okvis.
namespace ode {

// to make things a bit faster than using angle-axis conversion:
__inline__ double sinc(double x) {
  if (fabs(x) > 1e-6) {
    return sin(x) / x;
  } else {
    static const double c_2 = 1.0 / 6.0;
    static const double c_4 = 1.0 / 120.0;
    static const double c_6 = 1.0 / 5040.0;
    const double x_2 = x * x;
    const double x_4 = x_2 * x_2;
    const double x_6 = x_2 * x_2 * x_2;
    return 1.0 - c_2 * x_2 + c_4 * x_4 - c_6 * x_6;
  }
}
const int OdoErrorStateDim = 15 + 27;
// Note this function assume that the W frame is with z up, negative gravity
// direction, because in computing sb_dot and G world-centric velocities
__inline__ void evaluateContinuousTimeOde(
    const Eigen::Vector3d& gyr, const Eigen::Vector3d& acc, double g,
    const Eigen::Vector3d& p_WS_W, const Eigen::Quaterniond& q_WS,
    const okvis::SpeedAndBiases& sb,
    const Eigen::Matrix<double, 27, 1>& vTgTsTa, Eigen::Vector3d& p_WS_W_dot,
    Eigen::Vector4d& q_WS_dot, okvis::SpeedAndBiases& sb_dot,
    Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim>* F_c_ptr = 0) {
  // "true" rates and accelerations
  IMUErrorModel<double> iem(sb.tail<6>(), vTgTsTa);
  iem.estimate(gyr, acc);
  const Eigen::Vector3d omega_S = iem.w_est;  // gyr - sb.segment<3>(3);
  const Eigen::Vector3d acc_S = iem.a_est;    // acc - sb.tail<3>();

  // nonlinear states
  // start with the pose
  p_WS_W_dot = sb.head<3>();

  // now the quaternion
  Eigen::Vector4d dq;
  q_WS_dot.head<3>() = 0.5 * omega_S;
  q_WS_dot[3] = 0.0;
  Eigen::Matrix3d C_WS = q_WS.toRotationMatrix();

  // the rest is straightforward
  // consider Earth's radius. Model the Earth as a sphere, since we neither
  // know the position nor yaw (except if coupled with GPS and magnetometer).
  Eigen::Vector3d G =
      -p_WS_W - Eigen::Vector3d(0, 0, 6371009);  // vector to Earth center
  sb_dot.head<3>() = (C_WS * acc_S + g * G.normalized());  // s
  // biases
  sb_dot.tail<6>().setZero();

  // linearized system:
  if (F_c_ptr) {
    F_c_ptr->setZero();
    F_c_ptr->block<3, 3>(0, 6) = Eigen::Matrix3d::Identity();
    //    F_c_ptr->block<3, 3>(3, 9) -= C_WS;
    F_c_ptr->block<3, 3>(6, 3) -= okvis::kinematics::crossMx(C_WS * acc_S);
    //    F_c_ptr->block<3, 3>(6, 12) -= C_WS;
    Eigen::Matrix<double, 9, 6> N = Eigen::Matrix<double, 9, 6>::Zero();
    N.block<3, 3>(3, 0) = C_WS;
    N.block<3, 3>(6, 3) = C_WS;
    Eigen::Matrix<double, 6, 6 + 27> dwaB_dbgbaSTS;
    iem.dwa_B_dbgbaSTS(dwaB_dbgbaSTS);
    F_c_ptr->block<9, 6 + 27>(0, 9) = N * dwaB_dbgbaSTS;
  }
}

// p_WS_W, q_WS, sb, vTgTsTa are states at k
/* * covariance error states $\Delta y = \Delta[\mathbf{p}_{WS}^W,
\Delta\mathbf{\alpha}, \Delta\mathbf{v}_S^W,
 * \Delta\mathbf{b_g}, \Delta\mathbf{b_a}, \Delta\mathbf{Tg}, \Delta\mathbf{Ts},
\Delta\mathbf{Ta}]$
 * $\tilde{R}_s^w=(I-[\delta\alpha^w]_\times)R_s^w$
 * this DEFINITION OF ROTATION ERROR is the same in Mingyang Li's later
publications, okvis, and huai ION GNSS+ 2015 $y = [\mathbf{p}_{WS}^W,
\mathbf{q}_S^W, \mathbf{v}_S^W, \mathbf{b_g}, \mathbf{b_a}]$ $u =
[\mathbf{\omega}_{WS}^S,\mathbf{a}^S]$ $h = t_{n+1}-t_n$
$\mathbf{p}_{WS}^W \oplus = \mathbf{p}_{WS}^W +$
$\mathbf{v}_{WS}^W \oplus = \mathbf{v}_{WS}^W +$
$\mathbf{q}_{S}^W \oplus \mathbf{\omega}_{WS}^S h/2 =
\mathbf{q}_{S}^W\begin{bmatrix}
cos(\mathbf{\omega}_{WS}^S h/2) \\
sin(\mathbf{\omega}_{WS}^S
h/2)\frac{\mathbf{\omega}_{WS}^S/2}{\vert\mathbf{\omega}_{WS}^S/2\vert}
\end{bmatrix}$

$k_1 = f(t_n,y_n,u_n)$
$k_2 = f(t_n+h/2,y_n\oplus k_1 h/2 ,(u_n +u_{n+1})/2)$
$k_3 = f(t_n+h/2,y_n\oplus k_2 h/2 ,(u_n +u_{n+1})/2)$
$k_4 = f(t_n+h,y_n\oplus k_3 h , u_{n+1})$
$y_{n+1}=y_n\oplus\left(h(k_1 +2k_2 +2k_3 +k_4)/6 \right )$
Caution: provide both F_tot_ptr(e.g., identity) and P_ptr(e.g., zero) if
covariance is to be computed
*/
__inline__ void integrateOneStep_RungeKutta(
    const Eigen::Vector3d& gyr_0, const Eigen::Vector3d& acc_0,
    const Eigen::Vector3d& gyr_1, const Eigen::Vector3d& acc_1, double g,
    double sigma_g_c, double sigma_a_c, double sigma_gw_c, double sigma_aw_c,
    double dt, Eigen::Vector3d& p_WS_W, Eigen::Quaterniond& q_WS,
    okvis::SpeedAndBiases& sb, const Eigen::Matrix<double, 27, 1>& vTgTsTa,
    Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim>* P_ptr = 0,
    Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim>* F_tot_ptr = 0) {
  Eigen::Vector3d k1_p_WS_W_dot;
  Eigen::Vector4d k1_q_WS_dot;
  okvis::SpeedAndBiases k1_sb_dot;
  Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim> k1_F_c;
  evaluateContinuousTimeOde(gyr_0, acc_0, g, p_WS_W, q_WS, sb, vTgTsTa,
                            k1_p_WS_W_dot, k1_q_WS_dot, k1_sb_dot, &k1_F_c);

  Eigen::Vector3d p_WS_W1 = p_WS_W;
  Eigen::Quaterniond q_WS1 = q_WS;
  okvis::SpeedAndBiases sb1 = sb;
  // state propagation:
  p_WS_W1 += k1_p_WS_W_dot * 0.5 * dt;
  Eigen::Quaterniond dq;
  double theta_half = k1_q_WS_dot.head<3>().norm() * 0.5 * dt;
  double sinc_theta_half = sinc(theta_half);
  double cos_theta_half = cos(theta_half);
  dq.vec() = sinc_theta_half * k1_q_WS_dot.head<3>() * 0.5 * dt;
  dq.w() = cos_theta_half;
  q_WS1 = q_WS * dq;
  sb1 += k1_sb_dot * 0.5 * dt;

  Eigen::Vector3d k2_p_WS_W_dot;
  Eigen::Vector4d k2_q_WS_dot;
  okvis::SpeedAndBiases k2_sb_dot;
  Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim> k2_F_c;
  evaluateContinuousTimeOde(0.5 * (gyr_0 + gyr_1), 0.5 * (acc_0 + acc_1), g,
                            p_WS_W1, q_WS1, sb1, vTgTsTa, k2_p_WS_W_dot,
                            k2_q_WS_dot, k2_sb_dot, &k2_F_c);

  Eigen::Vector3d p_WS_W2 = p_WS_W;
  Eigen::Quaterniond q_WS2 = q_WS;
  okvis::SpeedAndBiases sb2 = sb;
  // state propagation:
  p_WS_W2 += k2_p_WS_W_dot * 0.5 * dt;
  theta_half = k2_q_WS_dot.head<3>().norm() * 0.5 * dt;
  sinc_theta_half = sinc(theta_half);
  cos_theta_half = cos(theta_half);
  dq.vec() = sinc_theta_half * k2_q_WS_dot.head<3>() * 0.5 * dt;
  dq.w() = cos_theta_half;
  // std::cout<<dq.transpose()<<std::endl;
  q_WS2 = q_WS2 * dq;
  sb2 += k1_sb_dot * 0.5 * dt;

  Eigen::Vector3d k3_p_WS_W_dot;
  Eigen::Vector4d k3_q_WS_dot;
  okvis::SpeedAndBiases k3_sb_dot;
  Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim> k3_F_c;
  evaluateContinuousTimeOde(0.5 * (gyr_0 + gyr_1), 0.5 * (acc_0 + acc_1), g,
                            p_WS_W2, q_WS2, sb2, vTgTsTa, k3_p_WS_W_dot,
                            k3_q_WS_dot, k3_sb_dot, &k3_F_c);

  Eigen::Vector3d p_WS_W3 = p_WS_W;
  Eigen::Quaterniond q_WS3 = q_WS;
  okvis::SpeedAndBiases sb3 = sb;
  // state propagation:
  p_WS_W3 += k3_p_WS_W_dot * dt;
  theta_half = k3_q_WS_dot.head<3>().norm() * dt;
  sinc_theta_half = sinc(theta_half);
  cos_theta_half = cos(theta_half);
  dq.vec() = sinc_theta_half * k3_q_WS_dot.head<3>() * dt;
  dq.w() = cos_theta_half;
  // std::cout<<dq.transpose()<<std::endl;
  q_WS3 = q_WS3 * dq;
  sb3 += k3_sb_dot * dt;

  Eigen::Vector3d k4_p_WS_W_dot;
  Eigen::Vector4d k4_q_WS_dot;
  okvis::SpeedAndBiases k4_sb_dot;
  Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim> k4_F_c;
  evaluateContinuousTimeOde(gyr_1, acc_1, g, p_WS_W3, q_WS3, sb3, vTgTsTa,
                            k4_p_WS_W_dot, k4_q_WS_dot, k4_sb_dot, &k4_F_c);

  // now assemble
  p_WS_W +=
      (k1_p_WS_W_dot + 2 * (k2_p_WS_W_dot + k3_p_WS_W_dot) + k4_p_WS_W_dot) *
      dt / 6.0;
  Eigen::Vector3d theta_half_vec =
      (k1_q_WS_dot.head<3>() +
       2 * (k2_q_WS_dot.head<3>() + k3_q_WS_dot.head<3>()) +
       k4_q_WS_dot.head<3>()) *
      dt / 6.0;
  theta_half = theta_half_vec.norm();
  sinc_theta_half = sinc(theta_half);
  cos_theta_half = cos(theta_half);
  dq.vec() = sinc_theta_half * theta_half_vec;
  dq.w() = cos_theta_half;
  q_WS = q_WS * dq;
  sb += (k1_sb_dot + 2 * (k2_sb_dot + k3_sb_dot) + k4_sb_dot) * dt / 6.0;

  q_WS.normalize();  // do not accumulate errors!

  if (F_tot_ptr) {
    // compute state transition matrix, note $\frac{d\Phi(t, t_0)}{dt}=
    // F(t)\Phi(t, t_0)$
    Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim>& F_tot =
        *F_tot_ptr;
    const Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim>& J1 =
        k1_F_c;
    const Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim> J2 =
        k2_F_c *
        (Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim>::Identity() +
         0.5 * dt * J1);
    const Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim> J3 =
        k3_F_c *
        (Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim>::Identity() +
         0.5 * dt * J2);
    const Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim> J4 =
        k4_F_c *
        (Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim>::Identity() +
         dt * J3);
    Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim> F =
        Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim>::Identity() +
        dt * (J1 + 2 * (J2 + J3) + J4) / 6.0;
    //        + dt * J1;
    //    std::cout<<J1<<std::endl << std::endl << J2 <<std::endl;
    F_tot =
        (F * F_tot)
            .eval();  // F is $\Phi(t_k, t_{k-1})$, F_tot is $\Phi(t_k, t_{0})$

    if (P_ptr) {
      Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim>& cov = *P_ptr;
      cov = F * (cov * F.transpose()).eval();

      // add process noise
      const double Q_g = sigma_g_c * sigma_g_c * dt;
      const double Q_a = sigma_a_c * sigma_a_c * dt;
      const double Q_gw = sigma_gw_c * sigma_gw_c * dt;
      const double Q_aw = sigma_aw_c * sigma_aw_c * dt;
      cov(3, 3) += Q_g;
      cov(4, 4) += Q_g;
      cov(5, 5) += Q_g;
      cov(6, 6) += Q_a;
      cov(7, 7) += Q_a;
      cov(8, 8) += Q_a;
      cov(9, 9) += Q_gw;
      cov(10, 10) += Q_gw;
      cov(11, 11) += Q_gw;
      cov(12, 12) += Q_aw;
      cov(13, 13) += Q_aw;
      cov(14, 14) += Q_aw;

      // force symmetric
      // huai: this may help keep cov positive semi-definite after propagation
      cov = 0.5 * cov + 0.5 * cov.transpose().eval();
    }
  }
}

/* p_WS_W, q_WS, sb, vTgTsTa are states at k+1, dt= t(k+1) -t(k)
$y = [\mathbf{p}_{WS}^W, \mathbf{q}_S^W, \mathbf{v}_S^W, \mathbf{b_g},
\mathbf{b_a}]$ $u = [\mathbf{\omega}_{WS}^S,\mathbf{a}^S]$ $h = t_{n}-t_{n+1}$
$\mathbf{p}_{WS}^W \oplus = \mathbf{p}_{WS}^W +$
$\mathbf{v}_{WS}^W \oplus = \mathbf{v}_{WS}^W +$
$\mathbf{q}_{S}^W \oplus \mathbf{\omega}_{WS}^S h/2 =
\mathbf{q}_{S}^W\begin{bmatrix}
cos(\mathbf{\omega}_{WS}^S h/2) \\
sin(\mathbf{\omega}_{WS}^S
h/2)\frac{\mathbf{\omega}_{WS}^S/2}{\vert\mathbf{\omega}_{WS}^S/2\vert}
\end{bmatrix}$

$k_1 = f(t_{n+1},y_{n+1},u_{n+1})$
$k_2 = f(t_{n+1}+h/2,y_{n+1}\oplus k_1 h/2 ,(u_n +u_{n+1})/2)$
$k_3 = f(t_{n+1}+h/2,y_{n+1}\oplus k_2 h/2 ,(u_n +u_{n+1})/2)$
$k_4 = f(t_n,y_{n+1}\oplus k_3 h , u_{n})$
$y_{n}=y_{n+1}\oplus\left(h(k_1 +2k_2 +2k_3 +k_4)/6 \right )$
*/
__inline__ void integrateOneStepBackward_RungeKutta(
    const Eigen::Vector3d& gyr_0, const Eigen::Vector3d& acc_0,
    const Eigen::Vector3d& gyr_1, const Eigen::Vector3d& acc_1, double g,
    double sigma_g_c, double sigma_a_c, double sigma_gw_c, double sigma_aw_c,
    double dt, Eigen::Vector3d& p_WS_W, Eigen::Quaterniond& q_WS,
    okvis::SpeedAndBiases& sb, const Eigen::Matrix<double, 27, 1>& vTgTsTa,
    Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim>* P_ptr = 0,
    Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim>* F_tot_ptr = 0) {
  Eigen::Vector3d k1_p_WS_W_dot;
  Eigen::Vector4d k1_q_WS_dot;
  okvis::SpeedAndBiases k1_sb_dot;
  Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim> k1_F_c;
  evaluateContinuousTimeOde(gyr_1, acc_1, g, p_WS_W, q_WS, sb, vTgTsTa,
                            k1_p_WS_W_dot, k1_q_WS_dot, k1_sb_dot, &k1_F_c);

  Eigen::Vector3d p_WS_W1 = p_WS_W;
  Eigen::Quaterniond q_WS1 = q_WS;
  okvis::SpeedAndBiases sb1 = sb;
  // state propagation:
  p_WS_W1 -= k1_p_WS_W_dot * 0.5 * dt;
  Eigen::Quaterniond dq;
  double theta_half = -k1_q_WS_dot.head<3>().norm() * 0.5 * dt;
  double sinc_theta_half = sinc(theta_half);
  double cos_theta_half = cos(theta_half);
  dq.vec() = -sinc_theta_half * k1_q_WS_dot.head<3>() * 0.5 * dt;
  dq.w() = cos_theta_half;
  q_WS1 = q_WS * dq;
  sb1 -= k1_sb_dot * 0.5 * dt;

  Eigen::Vector3d k2_p_WS_W_dot;
  Eigen::Vector4d k2_q_WS_dot;
  okvis::SpeedAndBiases k2_sb_dot;
  Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim> k2_F_c;
  evaluateContinuousTimeOde(0.5 * (gyr_0 + gyr_1), 0.5 * (acc_0 + acc_1), g,
                            p_WS_W1, q_WS1, sb1, vTgTsTa, k2_p_WS_W_dot,
                            k2_q_WS_dot, k2_sb_dot, &k2_F_c);

  Eigen::Vector3d p_WS_W2 = p_WS_W;
  Eigen::Quaterniond q_WS2 = q_WS;
  okvis::SpeedAndBiases sb2 = sb;
  // state propagation:
  p_WS_W2 -= k2_p_WS_W_dot * 0.5 * dt;
  theta_half = -k2_q_WS_dot.head<3>().norm() * 0.5 * dt;
  sinc_theta_half = sinc(theta_half);
  cos_theta_half = cos(theta_half);
  dq.vec() = -sinc_theta_half * k2_q_WS_dot.head<3>() * 0.5 * dt;
  dq.w() = cos_theta_half;
  // std::cout<<dq.transpose()<<std::endl;
  q_WS2 = q_WS2 * dq;
  sb2 -= k1_sb_dot * 0.5 * dt;

  Eigen::Vector3d k3_p_WS_W_dot;
  Eigen::Vector4d k3_q_WS_dot;
  okvis::SpeedAndBiases k3_sb_dot;
  Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim> k3_F_c;
  evaluateContinuousTimeOde(0.5 * (gyr_0 + gyr_1), 0.5 * (acc_0 + acc_1), g,
                            p_WS_W2, q_WS2, sb2, vTgTsTa, k3_p_WS_W_dot,
                            k3_q_WS_dot, k3_sb_dot, &k3_F_c);

  Eigen::Vector3d p_WS_W3 = p_WS_W;
  Eigen::Quaterniond q_WS3 = q_WS;
  okvis::SpeedAndBiases sb3 = sb;
  // state propagation:
  p_WS_W3 -= k3_p_WS_W_dot * dt;
  theta_half = -k3_q_WS_dot.head<3>().norm() * dt;
  sinc_theta_half = sinc(theta_half);
  cos_theta_half = cos(theta_half);
  dq.vec() = -sinc_theta_half * k3_q_WS_dot.head<3>() * dt;
  dq.w() = cos_theta_half;
  // std::cout<<dq.transpose()<<std::endl;
  q_WS3 = q_WS3 * dq;
  sb3 -= k3_sb_dot * dt;

  Eigen::Vector3d k4_p_WS_W_dot;
  Eigen::Vector4d k4_q_WS_dot;
  okvis::SpeedAndBiases k4_sb_dot;
  Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim> k4_F_c;
  evaluateContinuousTimeOde(gyr_0, acc_0, g, p_WS_W3, q_WS3, sb3, vTgTsTa,
                            k4_p_WS_W_dot, k4_q_WS_dot, k4_sb_dot, &k4_F_c);

  // now assemble
  p_WS_W -=
      (k1_p_WS_W_dot + 2 * (k2_p_WS_W_dot + k3_p_WS_W_dot) + k4_p_WS_W_dot) *
      dt / 6.0;
  Eigen::Vector3d theta_half_vec =
      -(k1_q_WS_dot.head<3>() +
        2 * (k2_q_WS_dot.head<3>() + k3_q_WS_dot.head<3>()) +
        k4_q_WS_dot.head<3>()) *
      dt / 6.0;
  theta_half = theta_half_vec.norm();
  sinc_theta_half = sinc(theta_half);
  cos_theta_half = cos(theta_half);
  dq.vec() = sinc_theta_half * theta_half_vec;
  dq.w() = cos_theta_half;
  q_WS = q_WS * dq;
  sb -= (k1_sb_dot + 2 * (k2_sb_dot + k3_sb_dot) + k4_sb_dot) * dt / 6.0;

  q_WS.normalize();  // do not accumulate errors!

  if (F_tot_ptr) {
    assert(false);  // the following section is not well perused and tested
    // compute state transition matrix, note $\frac{d\Phi(t, t_0)}{dt}=
    // F(t)\Phi(t, t_0)$
    Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim>& F_tot =
        *F_tot_ptr;
    const Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim>& J1 =
        k1_F_c;
    const Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim> J2 =
        k2_F_c *
        (Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim>::Identity() -
         0.5 * dt * J1);
    const Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim> J3 =
        k3_F_c *
        (Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim>::Identity() -
         0.5 * dt * J2);
    const Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim> J4 =
        k4_F_c *
        (Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim>::Identity() -
         dt * J3);
    Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim> F =
        Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim>::Identity() -
        dt * (J1 + 2 * (J2 + J3) + J4) / 6.0;
    //+ dt * J1;
    // std::cout<<F<<std::endl;
    F_tot = (F * F_tot).eval();

    if (P_ptr) {
      Eigen::Matrix<double, OdoErrorStateDim, OdoErrorStateDim>& cov = *P_ptr;
      cov = F * (cov * F.transpose()).eval();

      // add process noise
      const double Q_g = sigma_g_c * sigma_g_c * dt;
      const double Q_a = sigma_a_c * sigma_a_c * dt;
      const double Q_gw = sigma_gw_c * sigma_gw_c * dt;
      const double Q_aw = sigma_aw_c * sigma_aw_c * dt;
      cov(3, 3) += Q_g;
      cov(4, 4) += Q_g;
      cov(5, 5) += Q_g;
      cov(6, 6) += Q_a;
      cov(7, 7) += Q_a;
      cov(8, 8) += Q_a;
      cov(9, 9) += Q_gw;
      cov(10, 10) += Q_gw;
      cov(11, 11) += Q_gw;
      cov(12, 12) += Q_aw;
      cov(13, 13) += Q_aw;
      cov(14, 14) += Q_aw;

      // force symmetric - TODO: is this really needed here?
      // cov = 0.5 * cov + 0.5 * cov.transpose().eval();
    }
  }
}

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
void sys_local_dcm_bias(const Eigen::Matrix<Scalar, 3, 1>& rs0,
                        const Eigen::Matrix<Scalar, 3, 1>& vs0,
                        const Eigen::Quaternion<Scalar>& qs0_2_s,
                        const Eigen::Matrix<Scalar, 3, 1>& acc,
                        const Eigen::Matrix<Scalar, 3, 1>& gyro, Scalar dt,
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

#endif /* INCLUDE_OKVIS_CERES_ODE_ODE_HPP_ */
