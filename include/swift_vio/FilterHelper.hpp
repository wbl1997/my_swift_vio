#ifndef INCLUDE_SWIFT_VIO_FILTER_HELPER_HPP_
#define INCLUDE_SWIFT_VIO_FILTER_HELPER_HPP_

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <vector>

namespace swift_vio {
class FilterHelper {
public:
  /**
   * @brief stackJacobianAndResidual fill every entry of H_o, r_o, and R_o by
   * components in vH_o, vr_o, vR_o.
   * @param vH_o
   * @param vr_o
   * @param vR_o
   * @param H_o allocated in advance.
   * @param r_o allocated in advance.
   * @param R_o allocated and initialized to zero in advance.
   */
  static void stackJacobianAndResidual(
      const std::vector<Eigen::MatrixXd,
                        Eigen::aligned_allocator<Eigen::MatrixXd>> &vH_o,
      const std::vector<Eigen::Matrix<double, -1, 1>,
                        Eigen::aligned_allocator<Eigen::Matrix<double, -1, 1>>>
          &vr_o,
      const std::vector<Eigen::MatrixXd,
                        Eigen::aligned_allocator<Eigen::MatrixXd>> &vR_o,
      Eigen::MatrixXd *H_o, Eigen::Matrix<double, -1, 1> *r_o,
      Eigen::MatrixXd *R_o);
  static void shrinkResidual(const Eigen::MatrixXd &H_o,
                             const Eigen::MatrixXd &r_o,
                             const Eigen::MatrixXd &R_o, Eigen::MatrixXd *T_H,
                             Eigen::Matrix<double, Eigen::Dynamic, 1> *r_q,
                             Eigen::MatrixXd *R_q);
  static int pruneSquareMatrix(int rm_state_start, int rm_state_end,
                               Eigen::MatrixXd *state_cov);

  static bool gatingTest(const Eigen::MatrixXd &H, const Eigen::VectorXd &r,
                         const Eigen::MatrixXd &R, const Eigen::MatrixXd &cov);

  static Eigen::MatrixXd leftNullspaceWithRankCheck(const Eigen::MatrixXd &A,
                                                    int columnRankHint);

  static bool multiplyLeftNullspaceWithGivens(
      Eigen::MatrixXd *Hf, Eigen::MatrixXd *Hx,
      Eigen::Matrix<double, Eigen::Dynamic, 1> *residual, Eigen::MatrixXd *R,
      int columnRankHint);

  /**
   * Chi-square thresholds based on the DOF of state (chi2(0.95,DOF))
   * degrees from 0, 1, 2, ...
   */
  static const double chi2_95percentile[];
};
}  // namespace swift_vio
#endif // INCLUDE_SWIFT_VIO_FILTER_HELPER_HPP_
