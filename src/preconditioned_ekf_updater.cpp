#include "okvis/preconditioned_ekf_updater.h"

#include <Eigen/Cholesky>
#include <Eigen/QR>

#include <gflags/gflags.h>

#include <vio/eigen_utils.h>

#include <okvis/odeHybrid.hpp>

DEFINE_double(max_inc_tol, 2,
              "the tolerance of the lpNorm of an EKF state update");

namespace okvis {
PreconditionedEkfUpdater::PreconditionedEkfUpdater(const Eigen::MatrixXd &cov,
                                                   int variable_dim)
    : cov_ref_(cov), cov_dim_(cov_ref_.rows()), variable_dim_(variable_dim) {}

Eigen::Matrix<double, Eigen::Dynamic, 1>
PreconditionedEkfUpdater::computeCorrection(
    const Eigen::MatrixXd &T_H,
    const Eigen::Matrix<double, Eigen::Dynamic, 1> &r_q,
    const Eigen::MatrixXd &R_q,
    const Eigen::Matrix<double, Eigen::Dynamic, 1> *prev_deltaX) {
  // Calculate Kalman gain
  Eigen::MatrixXd Py = T_H *
                           cov_ref_.block(okvis::ceres::ode::OdoErrorStateDim,
                                          okvis::ceres::ode::OdoErrorStateDim,
                                          variable_dim_, variable_dim_) *
                           T_H.transpose() +
                       R_q;

  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VecX;
  VecX SVec =
      (Py.diagonal().cwiseAbs() + VecX::Constant(Py.cols(), 1)).cwiseSqrt();
  VecX SVecI = SVec.cwiseInverse();

  PyScaled_ = SVecI.asDiagonal() * Py * SVecI.asDiagonal();
  VecX rqScaled = SVecI.asDiagonal() * r_q;

  Eigen::LLT<Eigen::MatrixXd> llt_py(PyScaled_);
  if (llt_py.info() != Eigen::Success) {
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_decomp(PyScaled_);
    std::cout << "PyScaled diagonal\n"
              << PyScaled_.diagonal().transpose() << "\n";
    std::cout << "PyScaled " << PyScaled_.rows() << "x" << PyScaled_.cols()
              << " condition number "
              << vio::getConditionNumberOfMatrix(PyScaled_) << " and rank "
              << qr_decomp.rank() << std::endl;
    OKVIS_ASSERT_TRUE(Exception, false, "LLT failed for PyScaled!");
  }

  //  Eigen::MatrixXd PyScaledInv(PyScaled_.rows(), PyScaled_.cols());
  //  PyScaledInv.setIdentity();
  //  llt_py.solveInPlace(PyScaledInv);
  //  Eigen::MatrixXd KScaled =
  //      (cov_ref_.block(0, okvis::ceres::ode::OdoErrorStateDim, cov_dim_,
  //                         variable_dim_) *
  //       T_H.transpose()) *
  //      SVecI.asDiagonal() * PyScaledInv;

  // There is not much difference between the above and the below approach
  Eigen::MatrixXd KScaled_transpose =
      llt_py.solve(SVecI.asDiagonal() *
                   (T_H * cov_ref_
                              .block(0, okvis::ceres::ode::OdoErrorStateDim,
                                     cov_dim_, variable_dim_)
                              .transpose()));
  KScaled_ = KScaled_transpose.transpose();

  // State correction
  Eigen::Matrix<double, Eigen::Dynamic, 1> deltaX;
  if (prev_deltaX == nullptr) {
    deltaX = KScaled_ * rqScaled;
  } else {
    deltaX = KScaled_ * (rqScaled + SVecI.asDiagonal() * T_H *
                                        prev_deltaX->segment(
                                            okvis::ceres::ode::OdoErrorStateDim,
                                            variable_dim_));
  }
  if (std::isnan(deltaX(0)) || std::isnan(deltaX(1))) {
    OKVIS_ASSERT_TRUE(Exception, false, "nan in kalman filter");
  }

  // for debugging
  double tempNorm = deltaX.head<15>().lpNorm<Eigen::Infinity>();
  if (tempNorm > FLAGS_max_inc_tol) {
    std::cout << "PyScaled of condition number:"
              << vio::getConditionNumberOfMatrix(PyScaled_) << "\n"
              << "\nKScaled\n"
              << KScaled_ << std::endl;
    std::cout << "rqScaled\n"
              << rqScaled.transpose() << "\ndeltaX\n"
              << deltaX.transpose() << std::endl;
    OKVIS_ASSERT_LT(Exception, tempNorm, FLAGS_max_inc_tol,
                    "Warn too large increment may imply wrong association ");
  }
  // end debugging
  return deltaX;
}

void PreconditionedEkfUpdater::updateCovariance(
    Eigen::MatrixXd *cov_ptr) const {
  (*cov_ptr) = (*cov_ptr) - KScaled_ * PyScaled_ * KScaled_.transpose();
  Eigen::MatrixXd cov_symm = (cov_ptr->transpose() + (*cov_ptr)) * 0.5;
  (*cov_ptr) = cov_symm;
  if (cov_ptr->diagonal().minCoeff() < 0) {
    std::cout << "Warn: negative entry in cov diagonal\n"
              << cov_ptr->diagonal().transpose() << std::endl;
    cov_ptr->diagonal() = cov_ptr->diagonal().cwiseAbs();
  }
}

}  // namespace okvis
