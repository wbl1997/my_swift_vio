#ifndef OKVIS_PRECONDITIONED_EKF_UPDATER_H
#define OKVIS_PRECONDITIONED_EKF_UPDATER_H

#include <Eigen/Core>
#include <okvis/assert_macros.hpp>

namespace okvis {
class PreconditionedEkfUpdater {
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  const Eigen::MatrixXd &cov_ref_;
  const int cov_dim_;
  const int variable_dim_;
  Eigen::MatrixXd KScaled_;
  Eigen::MatrixXd PyScaled_;

 public:
  PreconditionedEkfUpdater(const Eigen::MatrixXd &cov, int variable_dim);

  Eigen::Matrix<double, Eigen::Dynamic, 1> computeCorrection(
      const Eigen::MatrixXd &T_H,
      const Eigen::Matrix<double, Eigen::Dynamic, 1> &r_q,
      const Eigen::MatrixXd &R_q,
      const Eigen::Matrix<double, Eigen::Dynamic, 1> *prev_deltaX = nullptr);

  void updateCovariance(Eigen::MatrixXd *cov_ptr) const;
};

class DefaultEkfUpdater {
public:
 OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
 const Eigen::MatrixXd &cov_ref_;
 const int cov_dim_;
 const int variable_dim_;
 Eigen::MatrixXd KScaled_;
 Eigen::MatrixXd PyScaled_;
public:
 DefaultEkfUpdater(const Eigen::MatrixXd &cov, int variable_dim);

 Eigen::Matrix<double, Eigen::Dynamic, 1> computeCorrection(
     const Eigen::MatrixXd &T_H,
     const Eigen::Matrix<double, Eigen::Dynamic, 1> &r_q,
     const Eigen::MatrixXd &R_q,
     const Eigen::Matrix<double, Eigen::Dynamic, 1> *prev_deltaX = nullptr);

 void updateCovariance(Eigen::MatrixXd *cov_ptr) const;
};
}  // namespace okvis
#endif  // OKVIS_PRECONDITIONED_EKF_UPDATER_H
