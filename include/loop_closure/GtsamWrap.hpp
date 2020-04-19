#ifndef INCLUDE_LOOP_CLOSURE_GTSAM_WRAP_HPP_
#define INCLUDE_LOOP_CLOSURE_GTSAM_WRAP_HPP_

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>

#include "ceres/internal/autodiff.h"
#include "ceres/local_parameterization.h"
#include "sophus/se3.hpp"

#include <gtsam/linear/NoiseModel.h>

#include "msckf/memory.h"
#include <okvis/kinematics/Transformation.hpp>

namespace VIO {

// given two gtsam::Pose3 computes the relative rotation and translation errors:
// rotError,tranError
std::pair<double, double> computeRotationAndTranslationErrors(
    const gtsam::Pose3& expectedPose, const gtsam::Pose3& actualPose,
    const bool upToScale);

/**
 * @brief createNoiseModel for a BetweenFactor.
 * We create an individual noise model for each BetweenFactor because the
 * noise for each factor is different. This practice is found in
 * gtsam/gtsam/slam/dataset.cpp and gtsam/tests/testNonlinearOptimizer.cpp
 * @param cov_e covariance of the obsrevation factor.
 * @param huber_threshold in units of sigmas.
 * A sound value is obtained by checking the Chi2 distribution with 6DOF at alpha=5%.
 */
inline gtsam::SharedNoiseModel createRobustNoiseModel(
    const Eigen::Matrix<double, 6, 6>& cov_e, double huber_threshold=std::sqrt(12.59)) {
  bool tryToSimplify = false;
  const gtsam::SharedNoiseModel noise_model_input =
      gtsam::noiseModel::Gaussian::Covariance(cov_e, tryToSimplify);
  gtsam::SharedNoiseModel noise_model_output =
      gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::Huber::Create(
              huber_threshold, gtsam::noiseModel::mEstimator::Huber::Block),
          noise_model_input);
  return noise_model_output;
}

/**
 * @brief createRobustNoiseModelSqrtR
 * @param sqrtR: R such that R'R = \Lambda = \Sigma^{-1}.
 * @param huber_threshold
 * @return
 */
inline gtsam::SharedNoiseModel createRobustNoiseModelSqrtR(
    const Eigen::Matrix<double, 6, 6>& sqrtR, double huber_threshold=std::sqrt(12.59)) {
  bool tryToSimplify = false;
  const gtsam::SharedNoiseModel noise_model_input =
      gtsam::noiseModel::Gaussian::SqrtInformation(sqrtR, tryToSimplify);
  gtsam::SharedNoiseModel noise_model_output =
      gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::Huber::Create(
              huber_threshold, gtsam::noiseModel::mEstimator::Huber::Block),
          noise_model_input);
  return noise_model_output;
}

/**
 * @brief compute Jacobian of the gtsam between factor unwhitened error relative
 * to the measurement error by autoDifferentiate.
 * The between factor in gtsam is defined as $e = log_{SE3}(T_z^{-1} T_x^{-1} T_y)$.
 * @warning use this class sparingly as it is likely expensive.
 */
class BetweenFactorPose3Wrap {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BetweenFactorPose3Wrap(const gtsam::Pose3& Tz, const gtsam::Pose3& Tx,
                         const gtsam::Pose3& Ty);

  BetweenFactorPose3Wrap(const gtsam::Pose3& Tz,
                         const gtsam::Pose3& Txy);

  explicit BetweenFactorPose3Wrap(const gtsam::Pose3& Tz);

  template <typename T>
  bool operator()(const T* deltaz, T* residual) const {
    Eigen::Map<const Eigen::Matrix<T, 6, 1>> deltavec(deltaz);
    Eigen::Matrix<T, 3, 1> omega = deltavec.template tail<3>();
    Sophus::SO3Group<T> so3Delta = Sophus::SO3Group<T>::exp(omega);
    Eigen::Quaternion<T> qDelta = so3Delta.unit_quaternion();
    Eigen::Quaternion<T> q = Tz_.rotation().toQuaternion().cast<T>();
    Eigen::Matrix<T, 3, 1> t = Tz_.translation().cast<T>();
    Eigen::Quaternion<T> qNew = qDelta * q;
    Eigen::Matrix<T, 3, 1> tNew = t + deltavec.template head<3>();
    Eigen::Quaternion<T> qxy = Txy_.rotation().toQuaternion().cast<T>();
    Eigen::Matrix<T, 3, 1> txy = Txy_.translation().cast<T>();
    Sophus::SE3Group<T> deltaT(qNew.conjugate() * qxy,
                               qNew.conjugate() * (txy - tNew));
    Eigen::Map<Eigen::Matrix<T, 6, 1>> residualp(residual);
    residualp = deltaT.log();
    Eigen::Matrix<T, 3, 1> temp = residualp.template head<3>();
    residualp.template head<3>() = residualp.template tail<3>();
    residualp.template tail<3>() = temp;
    return true;
  }
  /**
   * @brief toMeasurmentJacobian
   * @param autoJ de_dTz
   * @param residual
   */
  void toMeasurmentJacobian(Eigen::Matrix<double, 6, 6, Eigen::RowMajor>* autoJ,
                            Eigen::Matrix<double, 6, 1>* residual);

  const gtsam::Pose3 Tz_;
  const gtsam::Pose3 Txy_;
};


/**
 * @brief compute Jacobian of the gtsam prior factor unwhitened error relative
 * to the measurement error by autoDifferentiate.
 * In GTSAM, the PriorFactor<Pose3> is defined as $e= log_{SE3}(T_z^{-1} T_x)$
 */
class PriorFactorPose3Wrap {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PriorFactorPose3Wrap(const gtsam::Pose3& Tz) : Tz_(Tz), Tx_(Tz) {}

  PriorFactorPose3Wrap(const gtsam::Pose3& Tz, const gtsam::Pose3& Tx)
      : Tz_(Tz), Tx_(Tx) {}

  inline void toMeasurementJacobian(
      Eigen::Matrix<double, 6, 6, Eigen::RowMajor>* autoJ,
      Eigen::Matrix<double, 6, 1>* residual) {
    BetweenFactorPose3Wrap bfw(Tz_, Tx_);
    bfw.toMeasurmentJacobian(autoJ, residual);
  }

  const gtsam::Pose3 Tz_;
  const gtsam::Pose3 Tx_;
};

class GtsamWrap {
 public:
  /**
   * @brief retract
   * @param Tz
   * @param delta [\nu, \omega]
   * @return
   */
  static gtsam::Pose3 retract(gtsam::Pose3 Tz,
                              Eigen::Matrix<double, 6, 1>& delta);

  /**
   * @brief compute Jacobian of the between factor unwhitened error relative
   * to the measurement error.
   * The unwhitened error of the between factor is defined by retraction of
   * Pose3 in gtsam. Its error order is [\omega, t]. The measurement error is
   * defined with oplus in okvis::Transformation. Its error order is [t, \omega].
   * @warning This function is inaccurate. Use BetweenFactorPose3Wrap instead.
   * @param Tz the measurement used to construct the BetweenFactor<Pose3>.
   * @param JtoCustomRetract
   */
  static void toMeasurementJacobianBetweenFactor(
      gtsam::Pose3 Tz, gtsam::Pose3 Tx, gtsam::Pose3 Ty,
      Eigen::Matrix<double, 6, 6>* JtoCustomRetract);

  /**
   * @brief compute Jacobian of the between factor unwhitened error relative
   * to the measurement error, assuming Tz is close to Tx^{-1}Ty
   * @param Tz
   * @param JtoCustomRetract
   */
  static void toMeasurementJacobianBetweenFactor(
      gtsam::Pose3 Tz,
      Eigen::Matrix<double, 6, 6>* JtoCustomRetract);

  inline static gtsam::Pose3 toPose3(const okvis::kinematics::Transformation& T) {
    return gtsam::Pose3(gtsam::Rot3(T.q()), T.r());
  }

  inline static okvis::kinematics::Transformation toTransform(const gtsam::Pose3& P) {
    return okvis::kinematics::Transformation(
          P.translation(), P.rotation().toQuaternion());
  }
};

/**
 * @brief compute residuals and jacobians for a simple projection factor
 * The error is e = h(T_BC^{-1} T_WB^{-1} p^W) - z
 * h([X,Y,Z]) = [X/Z, Y/Z]
 * z = [\hat{x}, \hat{y}]
 * The only variable is T_WB with perturbation defined as gtsam::Pose3
 * T_WB = \hat{T}_{WB} exp(\epsilon)
 * \epsilon = [\omega, \nu].
 */
class StackedProjectionFactorDynamic {
 public:
  typedef double Scalar;
  enum {
    NUM_RESIDUALS = Eigen::Dynamic,
    NUM_PARAMETERS = 7,
    NUM_LOCAL_PARAMETERS = 6,
  };

  /**
   * @brief StackedProjectionFactorDynamic
   * @param pointList
   * @param rayList each entry [X/Z, Y/Z, 1]
   * @param inlierList
   * @param T_BC
   * @param T_WB
   */
  StackedProjectionFactorDynamic(const AlignedVector<Eigen::Vector3d>& pointList,
                                const AlignedVector<Eigen::Vector3d>& rayList,
                                const okvis::kinematics::Transformation& T_BC) :
    pointList_(pointList), rayList_(rayList),
    T_BC_(T_BC), kNumResiduals((int)pointList_.size() * 2),
    kEpsilon(1e-4) {
  }

  int NumResiduals() const {
    return kNumResiduals;
  }

  bool operator()(const double* parameters, double* residuals,
                  double* jacobian) const {
    Eigen::Map<const Eigen::Matrix<double, 3, 1>> t_WB(parameters);
    Eigen::Quaterniond q_WB(parameters[6], parameters[3], parameters[4],
                            parameters[5]);
    Eigen::Map<Eigen::Matrix<double, -1, 1>> resVec(residuals, kNumResiduals,
                                                    1);
    Eigen::Quaterniond q_CW = (q_WB * T_BC_.q()).conjugate();
    Eigen::Vector3d t_CW =
        T_BC_.q().conjugate() * (-(q_WB.conjugate() * t_WB) - T_BC_.r());
    std::vector<bool> status(pointList_.size(), true);
    AlignedVector<Eigen::Vector3d> predictedKeypointList;
    predictedKeypointList.resize(pointList_.size());
    for (size_t index = 0u; index < pointList_.size(); ++index) {
      Eigen::Vector3d pC = q_CW * pointList_[index] + t_CW;
      if (pC[2] < kEpsilon) {
        status[index] = false;
        resVec.segment<2>(index * 2).setZero();
        predictedKeypointList[index].setZero();
      } else {
        Eigen::Vector3d xy1 = pC / pC[2];
        resVec.segment<2>(index * 2) = (xy1 - rayList_[index]).head<2>();
        predictedKeypointList[index] = pC;
      }
    }
    if (jacobian) {
      Eigen::Map<Eigen::Matrix<double, -1, 6>> jacColMajor(jacobian,
                                                           kNumResiduals, 6);
      Eigen::Matrix3d R_CB = T_BC_.C().transpose();
      for (size_t index = 0u; index < pointList_.size(); ++index) {
        if (status[index]) {
          Eigen::Matrix<double, 2, 3> dxy_dxyz;
          double invz = 1.0 / predictedKeypointList[index][2];
          dxy_dxyz(0, 0) = invz;
          dxy_dxyz(0, 1) = 0;
          dxy_dxyz(1, 0) = 0;
          dxy_dxyz(1, 1) = invz;
          dxy_dxyz.col(2) = - predictedKeypointList[index].head<2>() * invz * invz;

          Eigen::Vector3d pB = q_WB.conjugate() * (pointList_[index] - t_WB);
          Eigen::Matrix<double, 2, 3> factor = dxy_dxyz * R_CB;
          jacColMajor.block<2, 3>(index * 2, 0) =
              factor * okvis::kinematics::crossMx(pB);      // rotation
          jacColMajor.block<2, 3>(index * 2, 3) = -factor;  // translation
        } else {
          jacColMajor.block<2, 6>(index * 2, 0).setZero();
        }
      }
    }
    return true;
  }

  const AlignedVector<Eigen::Vector3d>& pointList_;
  const AlignedVector<Eigen::Vector3d>& rayList_;
  const okvis::kinematics::Transformation T_BC_;
  int kNumResiduals;
  const double kEpsilon; ///< minimum depth of point in the camera frame to consider a correspondence to be valid.
};

/**
 * @brief The GtsamPose3Parameterization class
 * Parameter space: T = [x, y, z, qx, qy, qz, qw]
 * Tangent space: $\epsilon = [\omega, \nu]$
 * Plus: $T = hat(T) Exp(\epsilon)$
 */
class GtsamPose3Parameterization final: public ::ceres::LocalParameterization {
public:
  static const int kGlobalDim = 7;
  static const int kLocalDim = 6;

  // Generalization of the addition operation,
  //
  //   x_plus_delta = Plus(x, delta)
  //
  // with the condition that Plus(x, 0) = x.
  bool Plus(const double* x, const double* delta, double* x_plus_delta) const {
    return plus(x, delta, x_plus_delta);
  }

  // The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
  //
  // jacobian is a row-major GlobalSize() x LocalSize() matrix.
  bool ComputeJacobian(const double* /*x*/, double* jacobian) const {
    Eigen::Map<Eigen::Matrix<double, kGlobalDim, kLocalDim, Eigen::RowMajor>> j(jacobian);
    j.setIdentity();
    return true;
  }

  // Size of x.
  int GlobalSize() const final {
    return kGlobalDim;
  }

  // Size of delta.
  int LocalSize() const final {
    return kLocalDim;
  }

  /// \brief Generalization of the addition operation,
  ///        x_plus_delta = Plus(x, delta)
  ///        with the condition that Plus(x, 0) = x.
  /// @param[in] x Variable.
  /// @param[in] delta Perturbation.
  /// @param[out] x_plus_delta Perturbed x.
  static bool plus(const double* x, const double* delta, double* x_plus_delta) {
    Eigen::Map<const Eigen::Vector3d> t(x);
    Eigen::Map<const Eigen::Quaterniond> q(x + 3);
    gtsam::Pose3 p(gtsam::Rot3(q), t);
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> omega_v(delta);
    gtsam::Pose3 p_plus_delta = p.retract(omega_v);
    Eigen::Map<Eigen::Vector3d> t_plus_delta(x_plus_delta);
    t_plus_delta = p_plus_delta.translation();
    Eigen::Map<Eigen::Quaterniond> q_plus_delta(x_plus_delta + 3);
    q_plus_delta = p_plus_delta.rotation().toQuaternion();
    return true;
  }

  /// \brief Computes the Jacobian from minimal space to naively overparameterised space as used by ceres.
  /// @param[in] x Variable.
  /// @param[out] jacobian the Jacobian (dimension minDim x dim).
  /// \return True on success.
  static bool liftJacobian(const double* /*x*/, double* jacobian) {
    Eigen::Map<Eigen::Matrix<double, kLocalDim, kGlobalDim, Eigen::RowMajor>> j(jacobian);
    j.setIdentity();
    return true;
  }
};

} // namespace VIO

#endif // INCLUDE_LOOP_CLOSURE_GTSAM_WRAP_HPP_
