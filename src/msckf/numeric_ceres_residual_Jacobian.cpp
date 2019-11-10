
#include <msckf/numeric_ceres_residual_Jacobian.hpp>

namespace simul {

void computeNumericJacPose(okvis::ceres::PoseParameterBlock& paramBlock,
                           okvis::ceres::ErrorInterface* costFuncPtr,
                           double const* const* parameters,
                           const Eigen::VectorXd& residuals,
                           Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                         Eigen::RowMajor>* jacNumeric,
                           bool minimal) {
  const double epsilon = 1e-6;
  Eigen::VectorXd purturbedResiduals = residuals;
  okvis::kinematics::Transformation T_init = paramBlock.estimate();
  if (minimal) {
    for (size_t jack = 0; jack < 6; ++jack) {
      Eigen::Matrix<double, 6, 1> deltaVec =
          Eigen::Matrix<double, 6, 1>::Zero();
      deltaVec[jack] = epsilon;
      okvis::kinematics::Transformation T_purturb = T_init;
      T_purturb.oplus(deltaVec);
      paramBlock.setEstimate(T_purturb);
      costFuncPtr->EvaluateWithMinimalJacobians(
          parameters, purturbedResiduals.data(), NULL, NULL);
      jacNumeric->col(jack) = (purturbedResiduals - residuals) / epsilon;
      paramBlock.setEstimate(T_init);
    }
    return;
  }
  for (size_t jack = 0; jack < 7; ++jack) {
    okvis::kinematics::Transformation T_purturb = T_init;
    Eigen::Matrix<double, 7, 1> originalParams = T_init.parameters();
    Eigen::Matrix<double, 7, 1> deltaParams =
        Eigen::Matrix<double, 7, 1>::Zero();
    deltaParams[jack] = epsilon;
    Eigen::Matrix<double, 7, 1> newParams = originalParams + deltaParams;
    Eigen::Matrix<double, 4, 1> quat = newParams.tail<4>();
    newParams.tail<4>() = quat.normalized();
    T_purturb.setCoeffs(newParams);
    paramBlock.setEstimate(T_purturb);

    costFuncPtr->EvaluateWithMinimalJacobians(
        parameters, purturbedResiduals.data(), NULL, NULL);
    jacNumeric->col(jack) = (purturbedResiduals - residuals) / epsilon;
    paramBlock.setEstimate(T_init);
  }
}

void computeNumericJacPoint(
    okvis::ceres::HomogeneousPointParameterBlock& paramBlock,
    okvis::ceres::ErrorInterface* costFuncPtr, double const* const* parameters,
    const Eigen::VectorXd& residuals,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>*
        jacNumeric,
    bool minimal) {
  const double epsilon = 1e-6;
  Eigen::VectorXd purturbedResiduals = residuals;
  Eigen::Vector4d originalEstimate = paramBlock.estimate();
  if (minimal) {
    for (size_t jack = 0; jack < 3; ++jack) {
      Eigen::Matrix<double, 3, 1> deltaVec =
          Eigen::Matrix<double, 3, 1>::Zero();
      deltaVec[jack] = epsilon;
      Eigen::Vector4d perturbedEstimate = originalEstimate;
      perturbedEstimate.head<3>() += deltaVec;
      paramBlock.setEstimate(perturbedEstimate);
      costFuncPtr->EvaluateWithMinimalJacobians(
          parameters, purturbedResiduals.data(), NULL, NULL);
      jacNumeric->col(jack) = (purturbedResiduals - residuals) / epsilon;
      paramBlock.setEstimate(originalEstimate);
    }
    return;
  }
  for (size_t jack = 0; jack < 4; ++jack) {
    Eigen::Matrix<double, 4, 1> deltaVec = Eigen::Matrix<double, 4, 1>::Zero();
    deltaVec[jack] = epsilon;
    paramBlock.setEstimate(originalEstimate + deltaVec);
    costFuncPtr->EvaluateWithMinimalJacobians(
        parameters, purturbedResiduals.data(), NULL, NULL);
    jacNumeric->col(jack) = (purturbedResiduals - residuals) / epsilon;
    paramBlock.setEstimate(originalEstimate);
  }
}
} // namespace simul
