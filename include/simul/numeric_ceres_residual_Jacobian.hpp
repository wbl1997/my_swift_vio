#ifndef INCLUDE_SWIFT_VIO_NUMERIC_CERES_RESIDUAL_JACOBIAN_HPP_
#define INCLUDE_SWIFT_VIO_NUMERIC_CERES_RESIDUAL_JACOBIAN_HPP_
#include <Eigen/Core>
#include <okvis/ceres/ErrorInterface.hpp>
#include <okvis/ceres/ParameterBlock.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>

namespace simul {
#define ARE_MATRICES_CLOSE(ref, computed, tol)                             \
  do {                                                                     \
    double devi2 = (ref - computed)                                        \
                       .cwiseAbs()                                         \
                       .cwiseQuotient(Eigen::MatrixXd(                     \
                           (ref.cwiseAbs().array() > tol)                  \
                               .select(ref.cwiseAbs().array(), 1)))        \
                       .maxCoeff();                                        \
    if (devi2 > tol) {                                                     \
      std::cout << "Two matrices " << #ref << " and " << #computed         \
                << " largest elementwise difference is " +                 \
                       std::to_string(devi2)                               \
                << std::endl;                                              \
      std::cout << #ref << std::endl;                                      \
      if (ref.cols() == 1)                                                 \
        std::cout << ref.transpose() << std::endl;                         \
      else                                                                 \
        std::cout << ref << std::endl;                                     \
      std::cout << #computed << std::endl;                                 \
      if (computed.cols() == 1)                                            \
        std::cout << computed.transpose() << std::endl;                    \
      else                                                                 \
        std::cout << computed << std::endl;                                \
      std::cout << '(' << #ref << '-' << #computed << ").cwiseAbs"         \
                << std::endl;                                              \
      if (ref.cols() == 1)                                                 \
        std::cout << (ref - computed).cwiseAbs().transpose() << std::endl; \
      else                                                                 \
        std::cout << (ref - computed).cwiseAbs() << std::endl;             \
      std::cout << "cwiseQuotient" << std::endl;                           \
      if (ref.cols() == 1)                                                 \
        std::cout << (ref - computed)                                      \
                         .cwiseAbs()                                       \
                         .cwiseQuotient(Eigen::MatrixXd(                   \
                             (ref.cwiseAbs().array() > tol)                \
                                 .select(ref.cwiseAbs().array(), 1)))      \
                         .transpose();                                     \
      else                                                                 \
        std::cout << (ref - computed)                                      \
                         .cwiseAbs()                                       \
                         .cwiseQuotient(Eigen::MatrixXd(                   \
                             (ref.cwiseAbs().array() > tol)                \
                                 .select(ref.cwiseAbs().array(), 1)));     \
      std::cout << std::endl;                                              \
    }                                                                      \
  } while (0)

template <class JacEigenType = Eigen::Matrix<double, Eigen::Dynamic,
                                             Eigen::Dynamic, Eigen::RowMajor>>
void computeNumericJac(okvis::ceres::ParameterBlock& paramBlock,
                       okvis::ceres::ErrorInterface* costFuncPtr,
                       double const* const* parameters,
                       const Eigen::VectorXd& residuals,
                       JacEigenType* jacNumeric) {
  const double epsilon = 1e-6;
  size_t paramDim = paramBlock.dimension();
  Eigen::VectorXd purturbedResiduals = residuals;
  for (size_t jack = 0; jack < paramDim; ++jack) {
    Eigen::VectorXd deltaVec = Eigen::VectorXd::Zero(paramDim);
    deltaVec[jack] = epsilon;
    Eigen::VectorXd currEst(paramDim);
    Eigen::VectorXd currAndDelta(paramDim);
    double* paramPtr = paramBlock.parameters();
    for (size_t k = 0; k < paramDim; ++k) {
      currEst[k] = paramPtr[k];
    }
    currAndDelta = currEst + deltaVec;
    paramBlock.setParameters(currAndDelta.data());
    costFuncPtr->EvaluateWithMinimalJacobians(
        parameters, purturbedResiduals.data(), NULL, NULL);
    jacNumeric->col(jack) = (purturbedResiduals - residuals) / epsilon;
    paramBlock.setParameters(currEst.data());
  }
}

void computeNumericJacPose(okvis::ceres::PoseParameterBlock& paramBlock,
                           okvis::ceres::ErrorInterface* costFuncPtr,
                           double const* const* parameters,
                           const Eigen::VectorXd& residuals,
                           Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                         Eigen::RowMajor>* jacNumeric,
                           bool minimal);

void computeNumericJacPoint(
    okvis::ceres::HomogeneousPointParameterBlock& paramBlock,
    okvis::ceres::ErrorInterface* costFuncPtr, double const* const* parameters,
    const Eigen::VectorXd& residuals,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>*
        jacNumeric,
    bool minimal);
} // namespace simul
#endif // INCLUDE_SWIFT_VIO_NUMERIC_CERES_RESIDUAL_JACOBIAN_HPP_
