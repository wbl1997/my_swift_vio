/**
 * @file    TestGtsamImuFactor.cpp
 * @brief   Unit test for Gtsam Imu propagation with the preintegration
 * measurement against OKVIS corrected propagation
 * @author  J. Huai
 */
#include <gtest/gtest.h>

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include "gtsam/ImuFrontEnd.h"
#include "gtsam/ImuFactorTestHelpers.h"

#include "vio/Sample.h"
#include "vio/eigen_utils.h"

#include <okvis/ceres/ImuError.hpp>
#include <okvis/timing/Timer.hpp>

#include "../msckf/CovPropConfig.hpp"

Eigen::IOFormat commaInitFmt(Eigen::StreamPrecision, 0, ", ", "\n", "", "",
                             "", "");

/**
 * @brief ImuOdometryGtsam
 * @param p_WS_W0
 * @param q_WS0
 * @param sb0
 * @param imuMeasurements
 * @param imuParams
 * @param p_WS_W1
 * @param q_WS1
 * @param sb1
 * @param[out] covariance cov(\xi_j)
 * @param jacobian
 * @param verbose
 */
void ImuOdometryGtsam(const Eigen::Vector3d& p_WS_W0,
                      const Eigen::Quaterniond& q_WS0,
                      const okvis::SpeedAndBiases& sb0,
                      const okvis::ImuMeasurementDeque& imuMeasurements,
                      const okvis::ImuParameters& imuParams,
                      Eigen::Vector3d* p_WS_W1, Eigen::Quaterniond* q_WS1,
                      okvis::SpeedAndBiases* sb1,
                      Eigen::Matrix<double, 15, 15>* covariance,
                      Eigen::Matrix<double, 15, 15>* jacobian,
                      bool verbose = false) {
  /// gtsam error definition:
  /// R_{wb} \approx \hat{R}_{wb} (I+\theta_i\times),
  /// p_{wb} = \hat{p}_{wb} + R_{wb}\delta p_b,
  /// v_{wb} = \hat{v}_{wb} + \delta v_b^w
  /// and definitions of \delta b_g and \delta b_a follows \delta v_b^w.
  /// gtsam error vector [\theta, \delta p \delta v_s^w, \delta b_a, \delta b_g]
  /// gtsam imu factor error definition
  /// e_{gtsam-imu}= \begin{bmatrix} log((\hat{R}_j^i(z_{imu}))^{-1}R_j^i)^{V}
  /// \\ R^j_g(\hat{p}_j^g - p_j^g) \\ R^j_g(\hat{v}_j^g - v_j^g) \\ {b}_{a,i} -
  /// b_{a,j}\\ {b}_{g,i} - b_{g,j}\end{bmatrix}

  okvis::Time startEpoch = imuMeasurements.front().timeStamp;
  okvis::Time finishEpoch = imuMeasurements.back().timeStamp;
  gtsam::Vector3 n_gravity = Eigen::Vector3d(
      0, 0, -imuParams.g);  ///< Gravity vector in nav frame (namely, the global
                            ///< frame, as used in the client's VINS)

  okvis::ImuFrontEnd::PimPtr combinedPim;

  okvis::timing::Timer gtsamTimer("gtsam", false);
  okvis::ImuParams imuParamsKimera;
  imuParamsKimera.set(imuParams);
  imuParamsKimera.imu_preintegration_type_ =
      okvis::ImuPreintegrationType::kPreintegratedCombinedMeasurements;

  okvis::ImuFrontEnd imuIntegrator(imuParamsKimera);
  imuIntegrator.preintegrateImuMeasurements(imuMeasurements, sb0, startEpoch,
                                            finishEpoch, combinedPim);

  double timeElapsed = gtsamTimer.stop();
  std::cout << "Time used by gtsam preintegration " << timeElapsed << std::endl;

  Eigen::Matrix<double, 9, 6> D_r_pose_i, D_r_pose_j, D_r_bias_i;
  Eigen::Matrix<double, 9, 3> D_r_vel_i, D_r_vel_j;

  Eigen::Vector3d p_ij = combinedPim->deltaPij();
  Eigen::Vector3d v_ij = combinedPim->deltaVij();
  gtsam::Rot3 gtR_ij = combinedPim->deltaRij();
  *q_WS1 = q_WS0 * gtR_ij.toQuaternion();

  *p_WS_W1 = p_WS_W0 + sb0.head<3>() * combinedPim->deltaTij() +
      0.5 * n_gravity * combinedPim->deltaTij() * combinedPim->deltaTij() +
      q_WS0 * p_ij;
  Eigen::Vector3d v_WS1 =
      sb0.head<3>() + n_gravity * combinedPim->deltaTij() + q_WS0 * v_ij;

  Eigen::Matrix<double, 15, 15> cov_r =
      dynamic_cast<const gtsam::PreintegratedCombinedMeasurements&>(*combinedPim)
          .preintMeasCov();

  // jacobian of the OKVIS error \xi relative to gtsam error state at t_j.
  Eigen::Matrix<double, 15, 15> dxi_deta_j = gtsam::dokvis_dforster(*q_WS1);

  gtsam::Pose3 pose_i(gtsam::Rot3(q_WS0), p_WS_W0);
  Eigen::Vector3d vel_i = sb0.head<3>();
  gtsam::Pose3 pose_j(gtsam::Rot3(*q_WS1), *p_WS_W1);
  Eigen::Vector3d vel_j = v_WS1;
  gtsam::imuBias::ConstantBias bias_i(sb0.tail<3>(), sb0.segment<3>(3));

  /*Eigen::Matrix<double, 9, 1> r_Rpv =*/combinedPim->computeErrorAndJacobians(
      pose_i, vel_i, pose_j, vel_j, bias_i, &D_r_pose_i, &D_r_vel_i,
      &D_r_pose_j, &D_r_vel_j, &D_r_bias_i);
  Eigen::Matrix<double, 15, 15> dr_deta_i;
  // Jacobian of imu factor error relative to the error in the state at t_i,
  // $ \frac{\partial e(X_{j}, X_{i}, z_{imu})}{\partial (\theta_i,\delta
  // p_i,\delta v_i, \delta b_{a,i},\delta b_{g,i})} $
  // see Forster et al. eq (70) and (39) for their definitions.
  dr_deta_i << D_r_pose_i, D_r_vel_i, D_r_bias_i, Eigen::Matrix<double, 6, 9>::Zero(),
      Eigen::Matrix<double, 6, 6>::Identity();


  // Jacobian of imu factor error relative to the error in the state at t_j.
  Eigen::Matrix<double, 15, 15> dr_deta_j;
  dr_deta_j << D_r_pose_j, D_r_vel_j, Eigen::Matrix<double, 9, 6>::Zero(),
      Eigen::Matrix<double, 6, 9>::Zero(),
      -Eigen::Matrix<double, 6, 6>::Identity(); // -1 because CombinedImuFactor computeError in bias as e_b = b_i - b_j.

  Eigen::Matrix<double, 15, 15> dxi_deta_i = gtsam::dokvis_dforster(q_WS0);

  sb1->head<3>() = v_WS1;
  sb1->tail<6>() = sb0.tail<6>();

  // Jacobian of the okvis error \xi_{j|i} at t_j relative to the reordered gtsam error \eta at t_i.
  // d\xi_{j|i} / d\xi_i = d\xi_{j|i} / d\eta_{j|i} * d\eta_{j|i} / d\eta_{i} * (d\xi_i / d\eta_i)^{-1}.
  // where d\eta_{j|i} / d\eta_{i}  = de / d\eta_{j|i} * de / d\eta_i.
  // we use transpose because it has the same effect as inverse.
  *jacobian = dxi_deta_j * (-dr_deta_j.transpose()) * dr_deta_i * dxi_deta_i.transpose();

  Eigen::Matrix<double, 15, 15> dr_deta_j_inv = dr_deta_j.transpose();
  *covariance =
      dxi_deta_j * dr_deta_j_inv * cov_r * dr_deta_j * dxi_deta_j.transpose();
  Eigen::Matrix<double, 15, 1> sqrtDiagCov1 = covariance->diagonal().cwiseSqrt();

  if (verbose) {
    std::cout << "q_WS " << q_WS1->w() << " " << q_WS1->x() << " " << q_WS1->y()
              << " " << q_WS1->z() << std::endl;
    std::cout << "p_WS_W " << p_WS_W1->transpose() << std::endl;
    std::cout << "speed and bias " << v_WS1.transpose() << " "
              << combinedPim->biasHat().gyroscope().transpose() << " "
              << combinedPim->biasHat().accelerometer().transpose() << std::endl;
    std::cout << "cov diagonal sqrt\n" << sqrtDiagCov1.transpose() << "\n";
    std::cout << "Jacobian super diagonal " << std::endl;
    std::cout << vio::superdiagonal(*jacobian).transpose() << std::endl;
//    std::cout << "Covariance\n" << covariance->format(commaInitFmt) << "\n";
//    std::cout << "Jacobian\n" << jacobian->format(commaInitFmt) << "\n";
  }
}

TEST(ImuOdometry, CovariancePropagationFromZero) {
  bool verbose = true;
  srand((unsigned int)time(0));
  CovPropConfig cpc(false, true);
  okvis::kinematics::Transformation T_WS =
      okvis::kinematics::Transformation(cpc.get_p_WS_W0(), cpc.get_q_WS0());
  okvis::SpeedAndBiases sb = cpc.get_sb0();
  okvis::timing::Timer okvisTimer("OKVIS Corrected", false);
  Eigen::Matrix<double, 15, 15> covOkvis, jacobianOkvis;
  covOkvis.setZero();
  jacobianOkvis.setIdentity();

  int numUsedImuMeasurements =
      okvis::ceres::ImuError::propagation(
          cpc.get_imu_measurements(), cpc.get_imu_params(), T_WS, sb,
          cpc.get_meas_begin_time(), cpc.get_meas_end_time(), &covOkvis,
          &jacobianOkvis);
  double timeElapsed = okvisTimer.stop();

  Eigen::Vector3d p_WS_Okvis = T_WS.r();
  Eigen::Quaterniond q_WS_Okvis = T_WS.q();
  okvis::SpeedAndBiases speedAndBiasOkvis = sb;
  Eigen::Matrix<double, 15, 1> covDiagonalOkvis = covOkvis.diagonal();
  Eigen::Matrix<double, 15, 1> sqrtCovDiagonalOkvis = covDiagonalOkvis.cwiseSqrt();

  std::cout << "time used by OKVIS Leutenegger corrected forward propagtion with "
               "0 initial covariance " << timeElapsed << std::endl;
  if (verbose) {
    std::cout << "numUsedMeas " << numUsedImuMeasurements << " totalMeas "
              << (int)cpc.get_meas_size() << std::endl;
    std::cout << "q_WS " << q_WS_Okvis.w() << " " << q_WS_Okvis.x() << " "
              << q_WS_Okvis.y() << " " << q_WS_Okvis.z() << std::endl;
    std::cout << "p_WS_W " << p_WS_Okvis.transpose() << std::endl;
    std::cout << "speed and bias " << speedAndBiasOkvis.transpose() << std::endl;
    std::cout << "cov diagonal sqrt " << std::endl;
    std::cout << sqrtCovDiagonalOkvis.transpose() << std::endl;
    std::cout << "Jacobian super diagonal " << std::endl;
    std::cout << vio::superdiagonal(jacobianOkvis).transpose() << std::endl;
//    std::cout << "Covariance\n" << covOkvis.format(commaInitFmt) << "\n";
//    std::cout << "Jacobian\n" << jacobianOkvis.format(commaInitFmt) << std::endl;
  }

  Eigen::Vector3d p_WS_gtsam;
  Eigen::Quaterniond q_WS_gtsam;
  okvis::SpeedAndBiases speedAndBiasGtsam;
  Eigen::Matrix<double, 15, 15> covGtsam;
  Eigen::Matrix<double, 15, 15> jacobianGtsam;
  ImuOdometryGtsam(cpc.get_p_WS_W0(), cpc.get_q_WS0(), cpc.get_sb0(),
                   cpc.get_imu_measurements(), cpc.get_imu_params(),
                   &p_WS_gtsam, &q_WS_gtsam, &speedAndBiasGtsam, &covGtsam,
                   &jacobianGtsam, verbose);

  EXPECT_LT((p_WS_Okvis - p_WS_gtsam).norm() / p_WS_Okvis.norm(),  5e-3);
  check_q_near(q_WS_Okvis, q_WS_gtsam, 1e-5);
  EXPECT_LT((speedAndBiasOkvis - speedAndBiasGtsam).head<3>().norm() / speedAndBiasOkvis.norm(), 1e-3);
  EXPECT_LT((speedAndBiasOkvis - speedAndBiasGtsam).tail<6>().norm(), 1e-6);

  std::cout << "Check jacobians of okvis and gtsam\n";
  checkSelectiveRatio(jacobianOkvis.topLeftCorner<15, 9>(),
                      jacobianGtsam.topLeftCorner<15, 9>(), 0.09, 1e-3);
  checkSelectiveRatio(jacobianOkvis.topRightCorner<15, 6>(),
                      jacobianGtsam.topRightCorner<15, 6>(), 0.2, 1e-3);

  std::cout << "Check P, V, Q covariance of okvis and gtsam\n";
  checkSelectiveRatio(covOkvis.topLeftCorner<15, 9>(),
                      covGtsam.topLeftCorner<15, 9>(), 0.35, 2, 1);
  std::cout << "Check Bg, Ba covariance of okvis and gtsam\n";
  checkSelectiveRatio(covOkvis.topRightCorner<15, 6>(),
                      covGtsam.topRightCorner<15, 6>(), 2.5e-1, 10, 5);

}
