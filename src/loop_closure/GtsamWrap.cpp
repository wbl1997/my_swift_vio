#include "loop_closure/GtsamWrap.hpp"
#include "okvis/kinematics/sophus_operators.hpp"

namespace VIO {
std::pair<double, double> computeRotationAndTranslationErrors(
    const gtsam::Pose3& expectedPose, const gtsam::Pose3& actualPose,
    const bool upToScale) {
  // compute errors
  gtsam::Rot3 rotErrorMat =
      (expectedPose.rotation()).between(actualPose.rotation());
  gtsam::Vector3 rotErrorVector = gtsam::Rot3::Logmap(rotErrorMat);
  double rotError = rotErrorVector.norm();

  gtsam::Vector3 actualTranslation = actualPose.translation().vector();
  gtsam::Vector3 expectedTranslation = expectedPose.translation().vector();
  if (upToScale) {
    double normExpected = expectedTranslation.norm();
    double normActual = actualTranslation.norm();
    if (normActual > 1e-5)
      actualTranslation = normExpected * actualTranslation /
                          normActual;  // we manually add the scale here
  }
  gtsam::Vector3 tranErrorVector = expectedTranslation - actualTranslation;
  double tranError = tranErrorVector.norm();
  return std::make_pair(rotError, tranError);
}

gtsam::Pose3 GtsamWrap::retract(gtsam::Pose3 Tz,
                                Eigen::Matrix<double, 6, 1>& delta) {
  okvis::kinematics::Transformation Tza = toTransform(Tz);
  Tza.oplus(delta);
  return toPose3(Tza);
}

BetweenFactorPose3Wrap::BetweenFactorPose3Wrap(const gtsam::Pose3& Tz,
                                               const gtsam::Pose3& Tx,
                                               const gtsam::Pose3& Ty)
    : Tz_(Tz), Txy_(Tx.inverse() * Ty) {}

BetweenFactorPose3Wrap::BetweenFactorPose3Wrap(const gtsam::Pose3& Tz,
                                               const gtsam::Pose3& Txy)
    : Tz_(Tz), Txy_(Txy) {}

BetweenFactorPose3Wrap::BetweenFactorPose3Wrap(const gtsam::Pose3& Tz)
    : Tz_(Tz), Txy_(Tz) {}

void BetweenFactorPose3Wrap::toMeasurmentJacobian(
    Eigen::Matrix<double, 6, 6, Eigen::RowMajor>* autoJ,
    Eigen::Matrix<double, 6, 1>* residual) {
  Eigen::Matrix<double, 6, 1> deltaz = Eigen::Matrix<double, 6, 1>::Zero();
  const double* const parameters[] = {deltaz.data()};
  double* jacobians[] = {autoJ->data()};
  ::ceres::internal::AutoDifferentiate<
      ::ceres::internal::StaticParameterDims<6>>(*this, parameters, 6,
                                                 residual->data(), jacobians);
}

void GtsamWrap::toMeasurementJacobianBetweenFactor(
    gtsam::Pose3 Tz, gtsam::Pose3 Tx, gtsam::Pose3 Ty,
    Eigen::Matrix<double, 6, 6>* JtoCustomRetract) {
  JtoCustomRetract->topLeftCorner<3, 3>().setZero();
  Eigen::Matrix3d Rzp = Tz.rotation().matrix().transpose();
  JtoCustomRetract->topRightCorner<3, 3>() = -Rzp;
  JtoCustomRetract->bottomLeftCorner<3, 3>() = -Rzp;

  // This exact derivation uses the definition of SE(3) exp([\nu, \omega]),
  // and the definition of error in Tz.
  // A similar derivation is given in A.40 of Hauke Strasdat Thesis though
  // with different perturbations.

  gtsam::Rot3 rotDelta = Tz.rotation().inverse() * (Tx.rotation().inverse() * Ty.rotation());
  Eigen::Quaterniond qDelta = rotDelta.toQuaternion();
  double theta;
  Eigen::Vector3d omega = okvis::kinematics::logAndTheta(qDelta, &theta);
  Eigen::Matrix3d omegaSkew = okvis::kinematics::crossMx(omega);
  Eigen::Matrix3d V = Eigen::Matrix3d::Identity() + 0.5 * omegaSkew + 1/6 * omegaSkew * omegaSkew;
  Eigen::Matrix3d invV = V.inverse();

//  Eigen::Matrix3d invV = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_xy = Tx.rotation().inverse() * (Ty.translation() - Tx.translation());
  Eigen::Vector3d t_z = Tz.translation();
  JtoCustomRetract->bottomRightCorner<3, 3>() = invV * Rzp * okvis::kinematics::crossMx(t_xy - t_z);
}

void GtsamWrap::toMeasurementJacobianBetweenFactor(
    gtsam::Pose3 Tz,
    Eigen::Matrix<double, 6, 6>* JtoCustomRetract) {
  JtoCustomRetract->topLeftCorner<3, 3>().setZero();
  Eigen::Matrix3d Rzp = Tz.rotation().matrix().transpose();
  JtoCustomRetract->topRightCorner<3, 3>() = -Rzp;
  JtoCustomRetract->bottomLeftCorner<3, 3>() = -Rzp;
  JtoCustomRetract->bottomRightCorner<3, 3>().setZero();
}

} // namespace VIO
