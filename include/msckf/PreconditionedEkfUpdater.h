#ifndef OKVIS_PRECONDITIONED_EKF_UPDATER_H
#define OKVIS_PRECONDITIONED_EKF_UPDATER_H

#include <Eigen/Core>
#include <okvis/assert_macros.hpp>

namespace okvis {
class DefaultEkfUpdater {
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 protected:
  const Eigen::MatrixXd &cov_ref_;
  const int cov_dim_; /// rows or cols of covariance matrix.
  const int observationVariableStartIndex_; /// start index in covariance matrix of variables involved in camera observations.
  const int variable_dim_; /// dim of variables involved in camera observations.
  Eigen::MatrixXd KScaled_;
  Eigen::MatrixXd PyScaled_;

 public:
  DefaultEkfUpdater(const Eigen::MatrixXd &cov, int obsVarStartIndex,
                    int variable_dim);

  /**
   * @brief computeCorrection
   * @param T_H
   * @param r_q
   * @param R_q
   * @param totalCorrection $x_{k|k-1} \boxminus x_k^i$ where $x_{k|k-1} :=
   * x_k^0$
   * @return
   */
  virtual Eigen::Matrix<double, Eigen::Dynamic, 1> computeCorrection(
      const Eigen::MatrixXd &T_H,
      const Eigen::Matrix<double, Eigen::Dynamic, 1> &r_q,
      const Eigen::MatrixXd &R_q,
      const Eigen::Matrix<double, Eigen::Dynamic, 1> *totalCorrection =
          nullptr);

  virtual void updateCovariance(Eigen::MatrixXd *cov_ptr) const;
};

class PreconditionedEkfUpdater : public DefaultEkfUpdater {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PreconditionedEkfUpdater(const Eigen::MatrixXd &cov, int obsVarStartIndex,
                           int variable_dim);

  Eigen::Matrix<double, Eigen::Dynamic, 1> computeCorrection(
      const Eigen::MatrixXd &T_H,
      const Eigen::Matrix<double, Eigen::Dynamic, 1> &r_q,
      const Eigen::MatrixXd &R_q,
      const Eigen::Matrix<double, Eigen::Dynamic, 1> *totalCorrection =
          nullptr) final;

  void updateCovariance(Eigen::MatrixXd *cov_ptr) const final;
};
}  // namespace okvis
#endif  // OKVIS_PRECONDITIONED_EKF_UPDATER_H
