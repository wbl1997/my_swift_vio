#ifndef OKVIS_PRECONDITIONED_EKF_UPDATER_H
#define OKVIS_PRECONDITIONED_EKF_UPDATER_H

#include <Eigen/Core>
#include <okvis/assert_macros.hpp>

namespace swift_vio {
class DefaultEkfUpdater {
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 protected:
  const Eigen::MatrixXd &covRef_;
  const int covDim_; /// rows or cols of covariance matrix.
  const int observationVariableStartIndex_; /// start index in covariance matrix of variables involved in observations.
  const int variableDim_; /// dim of variables involved in observations.
  Eigen::MatrixXd KScaled_;
  Eigen::MatrixXd PyScaled_;

 public:
  DefaultEkfUpdater(const Eigen::MatrixXd &cov, int obsVarStartIndex,
                    int variableDim);

  virtual ~DefaultEkfUpdater();

  /**
   * @brief computeCorrection
   * @param T_H has variableDim_ columns.
   * @param r_q
   * @param R_q
   * @param totalCorrection used for iterative EKF.
   * \f $x_{k|k-1} \boxminus x_k^i$ where $x_{k|k-1} := x_k^0 \f $
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
                           int variableDim);

  virtual ~PreconditionedEkfUpdater();

  Eigen::Matrix<double, Eigen::Dynamic, 1> computeCorrection(
      const Eigen::MatrixXd &T_H,
      const Eigen::Matrix<double, Eigen::Dynamic, 1> &r_q,
      const Eigen::MatrixXd &R_q,
      const Eigen::Matrix<double, Eigen::Dynamic, 1> *totalCorrection =
          nullptr) final;

  void updateCovariance(Eigen::MatrixXd *cov_ptr) const final;
};
}  // namespace swift_vio
#endif  // OKVIS_PRECONDITIONED_EKF_UPDATER_H
