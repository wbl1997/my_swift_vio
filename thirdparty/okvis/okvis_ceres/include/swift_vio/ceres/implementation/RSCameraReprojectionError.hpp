
/**
 * @file implementation/RSCameraReprojectionError.hpp
 * @brief Header implementation file for the RSCameraReprojectionError class.
 * @author Jianzhu Huai
 */
#include "ceres/internal/autodiff.h"

#include <okvis/kinematics/Transformation.hpp>
#include <okvis/kinematics/operators.hpp>

#include <swift_vio/ceres/JacobianHelpers.hpp>
#include <swift_vio/Measurements.hpp>
#include <swift_vio/imu/SimpleImuOdometry.hpp>

namespace okvis {
namespace ceres {
template <class GEOMETRY_TYPE>
RSCameraReprojectionError<GEOMETRY_TYPE>::RSCameraReprojectionError() {}

template <class GEOMETRY_TYPE>
RSCameraReprojectionError<GEOMETRY_TYPE>::RSCameraReprojectionError(
    const measurement_t& measurement,
    const covariance_t& covariance,
    std::shared_ptr<const okvis::cameras::CameraGeometryBase> targetCamera,
    std::shared_ptr<const okvis::ImuMeasurementDeque> imuMeasurementCanopy,
    std::shared_ptr<const okvis::ImuParameters> imuParameters,
    okvis::Time targetStateTime, okvis::Time targetImageTime)
    : imuMeasCanopy_(imuMeasCanopy),
      imuParameters_(imuParameters),
      targetCamera_(targetCamera),
      targetStateTime_(targetStateTime),
      targetImageTime_(targetImageTime) {
  measurement_ = measurement;
  setCovariance(covariance);
}

template <class GEOMETRY_TYPE>
void RSCameraReprojectionError<GEOMETRY_TYPE>::
    setCovariance(const covariance_t& covariance) {
  information_ = covariance.inverse();
  covariance_ = covariance;
  // perform the Cholesky decomposition on order to obtain the correct error
  // weighting
  Eigen::LLT<Eigen::Matrix2d> lltOfInformation(information_);
  squareRootInformation_ = lltOfInformation.matrixL().transpose();
}

template <class GEOMETRY_TYPE>
bool RSCameraReprojectionError<GEOMETRY_TYPE>::
    Evaluate(double const* const* parameters, double* residuals,
             double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

template <class GEOMETRY_TYPE>
bool RSCameraReprojectionError<GEOMETRY_TYPE>::
    EvaluateWithMinimalJacobians(double const* const* parameters,
                                 double* residuals, double** jacobians,
                                 double** jacobiansMinimal) const {
  return EvaluateWithMinimalJacobiansAnalytic(parameters, residuals, jacobians,
                                              jacobiansMinimal);
}

template <class GEOMETRY_TYPE>
bool RSCameraReprojectionError<GEOMETRY_TYPE>::
    EvaluateWithMinimalJacobiansAnalytic(double const* const* parameters,
                                 double* residuals, double** jacobians,
                                 double** jacobiansMinimal) const {
  Eigen::Map<const Eigen::Vector3d> p_WBt(parameters[Index::T_WBt]);
  Eigen::Map<const Eigen::Quaterniond> q_WBt(parameters[Index::T_WBt] + 3);

  Eigen::Map<const Eigen::Vector4d> scaled_p_Ch(parameters[Index::hp_Ch]);
  Eigen::Vector4d hp_Ch = scaled_p_Ch / scaled_p_Ch[3];  // homogeneous representation.

  Eigen::Map<const Eigen::Vector3d> p_BCt(parameters[Index::T_BCt]);
  Eigen::Map<const Eigen::Quaterniond> q_BCt(parameters[Index::T_BCt] + 3);
  okvis::kinematics::Transformation T_BCt(p_BCt, q_BCt);

  Eigen::Map<const Eigen::Vector3d> p_BCh(parameters[Index::T_BCh]);
  Eigen::Map<const Eigen::Quaterniond> q_BCh(parameters[Index::T_BCh] + 3);
  okvis::kinematics::Transformation T_BCh(p_BCh, q_BCh);

  Eigen::Map<const Eigen::Vector3d> p_WBh(parameters[Index::T_WBh]);
  Eigen::Map<const Eigen::Quaterniond> q_WBh(parameters[Index::T_WBh] + 3);
  okvis::kinematics::Transformation T_WBh(p_WBh, q_WBh);

  double readoutTime = parameters[Index::ReadoutTime][0];
  double cameraTd = parameters[Index::CameraTd][0];

  Eigen::Matrix<double, -1, 1> intrinsics = Eigen::Map<const Eigen::Matrix<double, kIntrinsicsDim, 1>>(parameters[Index::Intrinsics]);

  Eigen::Matrix<double, 9, 1> speedBgBa = Eigen::Map<const Eigen::Matrix<double, 9, 1>>(parameters[Index::SpeedAndBiases]);

  Eigen::Map<const Eigen::Matrix<double, 9, 1>> Tg(parameters[Index::Intrinsics]); // not used for now.
  Eigen::Map<const Eigen::Matrix<double, 9, 1>> Ts(parameters[Index::Intrinsics]); // not used for now.
  Eigen::Map<const Eigen::Matrix<double, 6, 1>> Ta(parameters[Index::Intrinsics]); // not used for now.

  double ypixel(measurement_[1]);
  uint32_t height = cameraGeometryBase_->imageHeight();
  double kpN = ypixel / height - 0.5;
  double relativeFeatureTime = targetImageTime_ + cameraTd + readoutTime * kpN - targetStateTime_;
  std::pair<Eigen::Matrix<double, 3, 1>, Eigen::Quaternion<double>> pair_T_WBt(p_WBt, q_WBt);

  const okvis::Time t_start = targetStateTime_;
  const okvis::Time t_end = targetStateTime_ + okvis::Duration(relativeFeatureTime);
  const double wedge = 5e-8;
  if (relativeFeatureTime >= wedge) {
    swift_vio::ode::predictStates(*imuMeasCanopy_, imuParameters_->g, pair_T_WBt,
                                speedBgBa, t_start, t_end);
  } else if (relativeFeatureTime <= -wedge) {
    swift_vio::ode::predictStatesBackward(*imuMeasCanopy_, imuParameters_->g, pair_T_WBt,
                                        speedBgBa, t_start, t_end);
  }
  okvis::kinematics::Transformation T_WBt(pair_T_WBt.first, pair_T_WBt.second);

  Eigen::Vector4d hp_Ct = (T_WBt * T_BCt).inverse() * (T_WBh * T_BCh) * hp_Ch;

  // calculate the reprojection error
  measurement_t kp;
  Eigen::Matrix<double, 2, 3> Jh;
  Eigen::Matrix<double, 2, 3> Jh_weighted;
  Eigen::Matrix<double, 2, Eigen::Dynamic> Jpi;
  Eigen::Matrix<double, 2, Eigen::Dynamic> Jpi_weighted;
  if (jacobians != NULL) {
    cameraGeometryBase_->projectWithExternalParameters(hp_Ct.head<3>(), intrinsics, &kp, &Jh, &Jpi);
    Jh_weighted = squareRootInformation_ * Jh;
    Jpi_weighted = squareRootInformation_ * Jpi;
  } else {
    cameraGeometryBase_->projectWithExternalParameters(hp_Ct.head<3>(), intrinsics, &kp);
  }
  measurement_t error = kp - measurement_;

  // weight:
  measurement_t weighted_error = squareRootInformation_ * error;

  residuals[0] = weighted_error[0];
  residuals[1] = weighted_error[1];

  bool valid = true;
  if (fabs(hp_Ct[3]) > 1.0e-8) {
    Eigen::Vector3d p_C = hp_Ct.template head<3>() / hp_Ct[3];
    if (p_C[2] < 0.2) {  // 20 cm - not very generic... but reasonable
      valid = false;
    }
  }

  if (jacobians != NULL) {
    if (!valid) {
      setJacobiansZero(jacobians, jacobiansMinimal);
      return true;
    }

    // TODO(jhuai): binliang: correct the below to compute the Jacobians analytically.

    Eigen::Matrix<double, 4, 6> dhC_deltaTWS;
    Eigen::Matrix<double, 4, 4> dhC_deltahpW;
    Eigen::Matrix<double, 4, EXTRINSIC_MODEL::kNumParams> dhC_dExtrinsic;
    Eigen::Vector4d dhC_td;
    Eigen::Matrix<double, 4, 9> dhC_sb;

    Eigen::Vector3d p_BP_W = hp_W.head<3>() - t_WB_W * hp_W[3];
    Eigen::Matrix<double, 4, 6> dhS_deltaTWS;
    dhS_deltaTWS.topLeftCorner<3, 3>() = -C_BW * hp_W[3];
    dhS_deltaTWS.topRightCorner<3, 3>() =
        C_BW * okvis::kinematics::crossMx(p_BP_W);
    dhS_deltaTWS.row(3).setZero();
    dhC_deltaTWS = T_CB * dhS_deltaTWS;
    dhC_deltahpW = T_CB * T_BW;

    EXTRINSIC_MODEL::dhC_dExtrinsic_HPP(hp_C, C_CB, &dhC_dExtrinsic);

    okvis::ImuMeasurement queryValue;
    swift_vio::ode::interpolateInertialData(*imuMeasCanopy_, t_end, queryValue);
    queryValue.measurement.gyroscopes -= speedAndBiases.segment<3>(3);
    Eigen::Vector3d p =
        okvis::kinematics::crossMx(queryValue.measurement.gyroscopes) *
            hp_B.head<3>() +
        C_BW * speedAndBiases.head<3>() * hp_W[3];
    dhC_td.head<3>() = -C_CB * p;
    dhC_td[3] = 0;

    Eigen::Matrix3d dhC_vW = -C_CB * C_BW * relativeFeatureTime * hp_W[3];
    Eigen::Matrix3d dhC_bg =
        -C_CB * C_BW *
        okvis::kinematics::crossMx(hp_W.head<3>() - hp_W[3] * t_WB_W) *
        relativeFeatureTime * q_WB0.toRotationMatrix();

    dhC_sb.row(3).setZero();
    dhC_sb.topRightCorner<3, 3>().setZero();
    dhC_sb.topLeftCorner<3, 3>() = dhC_vW;
    dhC_sb.block<3, 3>(0, 3) = dhC_bg;

  }
  return true;
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation via autodiff
template <class GEOMETRY_TYPE>
bool RSCameraReprojectionError<GEOMETRY_TYPE>::
    EvaluateWithMinimalJacobiansAutoDiff(double const* const* parameters,
                                         double* residuals, double** jacobians,
                                         double** jacobiansMinimal) const {
  // TODO(jhuai): binliang: either implement auto diff here or use autodiff in the test.
  return true;
}

template <class GEOMETRY_TYPE>
void RSCameraReprojectionError<GEOMETRY_TYPE>::
    setJacobiansZero(double** jacobians, double** jacobiansMinimal) const {
  zeroJacobian<7, 6, 2>(Index::T_WBt, jacobians, jacobiansMinimal);
  zeroJacobian<4, 3, 2>(Index::hp_Ch, jacobians, jacobiansMinimal);

  // TODO(jhuai): binliang: correct the below for other parameters.
  zeroJacobian<7, EXTRINSIC_MODEL::kNumParams, 2>(2, jacobians, jacobiansMinimal);
  zeroJacobian<PROJ_INTRINSIC_MODEL::kNumParams,
               PROJ_INTRINSIC_MODEL::kNumParams, 2>(3, jacobians,
                                                    jacobiansMinimal);
  zeroJacobian<kDistortionDim, kDistortionDim, 2>(4, jacobians,
                                                  jacobiansMinimal);
  zeroJacobian<1, 1, 2>(5, jacobians, jacobiansMinimal);
  zeroJacobian<1, 1, 2>(6, jacobians, jacobiansMinimal);
  zeroJacobian<9, 9, 2>(7, jacobians, jacobiansMinimal);
}
}  // namespace ceres
}  // namespace okvis
