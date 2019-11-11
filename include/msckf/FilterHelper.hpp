#ifndef MSCKF_FILTER_HELPER_HPP_
#define MSCKF_FILTER_HELPER_HPP_

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <vector>

class FilterHelper {
 public:
  // fill every entry of H_o, r_o, and R_o by components in vH_o, vr_o, vR_o
  static void stackJacobianAndResidual(
      const std::vector<Eigen::MatrixXd,
                        Eigen::aligned_allocator<Eigen::MatrixXd>>& vH_o,
      const std::vector<Eigen::MatrixXd,
                        Eigen::aligned_allocator<Eigen::MatrixXd>>& vr_o,
      const std::vector<Eigen::MatrixXd,
                        Eigen::aligned_allocator<Eigen::MatrixXd>>& vR_o,
      Eigen::MatrixXd* H_o, Eigen::MatrixXd* r_o, Eigen::MatrixXd* R_o);
  static void shrinkResidual(const Eigen::MatrixXd& H_o,
                             const Eigen::MatrixXd& r_o,
                             const Eigen::MatrixXd& R_o, Eigen::MatrixXd* T_H,
                             Eigen::Matrix<double, Eigen::Dynamic, 1>* r_q,
                             Eigen::MatrixXd* R_q);
  static int pruneSquareMatrix(int rm_state_start, int rm_state_end,
                                Eigen::MatrixXd* state_cov);

  static bool gatingTest(const Eigen::MatrixXd& H, const Eigen::VectorXd& r,
                         const Eigen::MatrixXd& R, const Eigen::MatrixXd& cov);


  /**
   * Chi-square thresholds based on the DOF of state (chi2(0.95,DOF))
   * degrees from 0, 1, 2, ...
   */
  static const double chi2_95percentile[];
};

template <typename Derived>
void removeUnsetMatrices(
    std::vector<Derived, Eigen::aligned_allocator<Derived>>* matrices,
    const std::vector<bool>& markers) {
  //  if (matrices->size() != markers.size()) {
  //    std::cerr << "The input size of matrices(" << matrices->size()
  //              << ") and markers(" << markers.size() << ") does not
  //              match.\n";
  //  }
  auto iter = matrices->begin();
  auto keepIter = matrices->begin();
  for (size_t i = 0; i < markers.size(); ++i) {
    if (markers[i] == 0) {
      ++iter;
    } else {
      if (keepIter != iter) *keepIter = *iter;
      ++iter;
      ++keepIter;
    }
  }
  matrices->resize(keepIter - matrices->begin());
}

template <typename T>
void removeUnsetElements(std::vector<T>* elements,
                         const std::vector<bool>& markers) {
  //  if (elements->size() != markers.size()) {
  //    std::cerr << "The input size of elements(" << elements->size()
  //              << ") and markers(" << markers.size() << ") does not
  //              match.\n";
  //  }
  auto iter = elements->begin();
  auto keepIter = elements->begin();
  for (size_t i = 0; i < markers.size(); ++i) {
    if (markers[i] == 0) {
      ++iter;
    } else {
      *keepIter = *iter;
      ++iter;
      ++keepIter;
    }
  }
  elements->resize(keepIter - elements->begin());
}

template <typename T>
void removeUnsetElements(std::vector<T>* elements,
                         const std::vector<bool>& markers, const int step) {
  //  if (elements->size() != markers.size()) {
  //    std::cerr << "The input size of elements(" << elements->size()
  //              << ") and markers(" << markers.size() << ") does not
  //              match.\n";
  //  }
  auto iter = elements->begin();
  auto keepIter = elements->begin();
  for (size_t i = 0; i < markers.size(); ++i) {
    if (markers[i] == 0) {
      iter += step;
    } else {
      for (int j = 0; j < step; ++j) {
        *keepIter = *iter;
        ++iter;
        ++keepIter;
      }
    }
  }
  elements->resize(keepIter - elements->begin());
}

#endif  // MSCKF_FILTER_HELPER_HPP_
