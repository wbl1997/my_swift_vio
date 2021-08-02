#ifndef INCLUDE_SWIFT_VIO_EXTRINSIC_MODELS_HPP_
#define INCLUDE_SWIFT_VIO_EXTRINSIC_MODELS_HPP_

#include <Eigen/Core>

#include <ceres/local_parameterization.h>

#include <swift_vio/ceres/JacobianHelpers.hpp>
#include <okvis/ModelSwitch.hpp>

#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/kinematics/sophus_operators.hpp>

namespace swift_vio {
class Extrinsic_p_CB final : public ::ceres::LocalParameterization {
  // of T_BC only p_CB is variable.
 public:
  static const int kModelId = 1;
  static const size_t kNumParams = 3;
  static const size_t kGlobalDim = 7;
  static inline int getMinimalDim() { return kNumParams; }
  static inline Eigen::MatrixXd initCov(double sigma_translation,
                                        double /*sigma_orientation*/) {
    return Eigen::MatrixXd::Identity(3, 3) *
           (sigma_translation * sigma_translation);
  }

  static void dT_BC_dExtrinsic(const Eigen::Matrix3d& R_BC,
                               Eigen::Matrix<double, 6, kNumParams>* j) {
    j->topRows<3>() = -R_BC;
    j->bottomRows<3>().setZero();
  }

  static void dpC_dExtrinsic_AIDP(const Eigen::Vector3d& /*pC*/,
                             const Eigen::Matrix3d& /*R_CB*/,
                             Eigen::MatrixXd* dpC_dT,
                             const Eigen::Matrix3d* R_CfCa,
                             const Eigen::Vector4d* ab1rho) {
    *dpC_dT = (*ab1rho)[3] * (Eigen::Matrix3d::Identity() - (*R_CfCa));
  }

  static void dpC_dExtrinsic_HPP(const Eigen::Vector4d& hpC,
                             const Eigen::Matrix3d& /*R_CB*/,
                             Eigen::MatrixXd* dpC_dT) {
      *dpC_dT = Eigen::Matrix3d::Identity() * hpC[3];
  }

  static void dhC_dExtrinsic_HPP(const Eigen::Matrix<double, 4, 1>& hpC,
                 const Eigen::Matrix<double, 3, 3>& /*R_CB*/,
                 Eigen::Matrix<double, 4, kNumParams>* dhC_deltaTBC) {
    dhC_deltaTBC->topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity() * hpC[3];
    dhC_deltaTBC->row(3).setZero();
  }

  // the error state is $\delta p_B^C$ or $\delta p_S^C$
  static void updateState(const Eigen::Vector3d& r, const Eigen::Quaterniond& q,
                          const Eigen::VectorXd& delta,
                          Eigen::Vector3d* r_delta,
                          Eigen::Quaterniond* q_delta) {
    *r_delta = r - q * delta;
    *q_delta = q;
  }

  template <typename Scalar>
  static void oplus(
      const Scalar* const deltaT_BC,
      std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>* T_BC) {
    Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>> delta_t(deltaT_BC);
    T_BC->first -= T_BC->second * delta_t;
  }

  template <typename Scalar>
  static void ominus(const Scalar* T_BC,
                     const Scalar* T_BC_plus,
                     Scalar* delta) {
    Eigen::Map<const Eigen::Quaternion<Scalar>> q(T_BC + 3);
    Eigen::Map<Eigen::Vector3d> deltaVec(delta);
    deltaVec = q.conjugate() * (Eigen::Map<const Eigen::Vector3d>(T_BC) -
                                Eigen::Map<const Eigen::Vector3d>(T_BC_plus));
  }

  static void toDimensionLabels(std::vector<std::string>* extrinsicLabels) {
    *extrinsicLabels = {"p_CS_C_x[m]", "p_CS_C_y", "p_CS_C_z"};
  }

  static void toMinDimensionLabels(std::vector<std::string>* extrinsicLabels) {
    *extrinsicLabels = {"p_CS_C_x[m]", "p_CS_C_y", "p_CS_C_z"};
  }

  static void toDesiredStdevs(Eigen::VectorXd* desiredStdevs) {
    desiredStdevs->resize(3);
    desiredStdevs->setConstant(0.04);
  }

  static void toParamValues(
      const okvis::kinematics::Transformation& T_BC,
      Eigen::VectorXd* extrinsic_opt_coeffs) {
    *extrinsic_opt_coeffs = T_BC.q().conjugate() * (-T_BC.r());
  }

  template <class Scalar>
  static std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>> get_T_BC(
      const okvis::kinematics::Transformation& T_BC_base,
      const Scalar* parameters) {
    Eigen::Matrix<Scalar, 3, 1> t_CB_C(parameters[0], parameters[1], parameters[2]);
    Eigen::Quaternion<Scalar> q_BC = T_BC_base.q().cast<Scalar>();
    return std::make_pair(q_BC * (-t_CB_C), q_BC);
  }

  /// \brief Computes the Jacobian from minimal space to naively overparameterised space as used by ceres.
  /// @param[in] x Variable.
  /// @param[out] jacobian the Jacobian (dimension minDim x dim).
  /// \return True on success.
  static bool liftJacobian(const double* /*x*/, double* jacobian) {
    Eigen::Map<Eigen::Matrix<double, kNumParams, kGlobalDim, Eigen::RowMajor> > J_lift(jacobian);
    J_lift.setIdentity();
    return true;
  }

  bool Plus(const double *x, const double *delta, double *x_plus_delta) const final {
    // transform to okvis::kinematics framework
    std::pair<Eigen::Matrix<double, 3, 1>, Eigen::Quaternion<double>> T(
          Eigen::Vector3d(x[0], x[1], x[2]),
          Eigen::Quaterniond(x[6], x[3], x[4], x[5]));
    // call oplus operator in okvis::kinematis
    oplus(delta, &T);

    // copy back
    const Eigen::Vector3d& r = T.first;
    x_plus_delta[0] = r[0];
    x_plus_delta[1] = r[1];
    x_plus_delta[2] = r[2];
    return true;
  }

  bool ComputeJacobian(const double */*x*/, double *jacobian) const final {
    Eigen::Map<Eigen::Matrix<double, kGlobalDim, kNumParams, Eigen::RowMajor>> j(jacobian);
    j.setIdentity();
    return true;
  }

  int GlobalSize() const final { return kGlobalDim; }
  int LocalSize() const final { return kNumParams; }
};

// This is different from PoseLocalParameterization in liftJacobian() and
// ComputeJacobian() in that both are set to identity here.
class Extrinsic_p_BC_q_BC_base : public ::ceres::LocalParameterization {
  // T_BC is represented by p_BC and R_BC in the states.
 public:
  static const size_t kNumParams = 6;
  static const size_t kGlobalDim = 7;

  static inline int getMinimalDim() { return kNumParams; }

  static inline Eigen::MatrixXd initCov(double sigma_translation,
                                        double sigma_orientation) {
    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity();
    cov.topLeftCorner<3, 3>() *= (sigma_translation * sigma_translation);
    cov.bottomRightCorner<3, 3>() *= (sigma_orientation * sigma_orientation);
    return cov;
  }

  static void updateState(const Eigen::Vector3d& r, const Eigen::Quaterniond& q,
                          const Eigen::VectorXd& delta,
                          Eigen::Vector3d* r_delta,
                          Eigen::Quaterniond* q_delta) {
    Eigen::Vector3d deltaAlpha = delta.segment<3>(3);   
    *r_delta = r + delta.head<3>();
    *q_delta = okvis::kinematics::expAndTheta(deltaAlpha) * q;
    q_delta->normalize();
  }

  /**
   * X can be B or C0.
   */
  template <typename Scalar>
  static void oplus(
      const Scalar* const deltaT_XC,
      std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>* T_XC) {
    Eigen::Map<const Eigen::Matrix<Scalar, 6, 1>> delta(deltaT_XC);
    T_XC->first += delta.template head<3>();
    Eigen::Matrix<Scalar, 3, 1> omega = delta.template tail<3>();
    Eigen::Quaternion<Scalar> dq = okvis::kinematics::expAndTheta(omega);
    T_XC->second = dq * T_XC->second;
  }

  /**
   * This implementation is preferred over okvis::ceres::
   * PoseLocalParameterization::minus(x, x_plus_delta, delta);
   */
  template <typename Scalar>
  static void ominus(const Scalar* T_XC,
                     const Scalar* T_XC_plus,
                     Scalar* delta) {
    delta[0] = T_XC_plus[0] - T_XC[0];
    delta[1] = T_XC_plus[1] - T_XC[1];
    delta[2] = T_XC_plus[2] - T_XC[2];
    const Eigen::Quaterniond q_plus(T_XC_plus[6], T_XC_plus[3],
                                           T_XC_plus[4], T_XC_plus[5]);
    const Eigen::Quaterniond q(T_XC[6], T_XC[3], T_XC[4], T_XC[5]);
    Eigen::Map<Eigen::Vector3d> delta_q(&delta[3]);
    double theta;
    delta_q = okvis::kinematics::logAndTheta(q_plus * q.inverse(), &theta);
  }

  /**
   * @brief toParamValues
   * @param T_XC
   * @param extrinsic_opt_coeffs
   */
  static void toParamValues(
      const okvis::kinematics::Transformation& T_XC,
      Eigen::VectorXd* extrinsic_opt_coeffs) {
    *extrinsic_opt_coeffs = T_XC.coeffs();
  }

  /// \brief Computes the Jacobian from minimal space to naively overparameterised space as used by ceres.
  ///     It must be the pseudo inverse of the local parametrization Jacobian as obtained by COmputeJacobian.
  /// @param[in] x Variable.
  /// @param[out] jacobian the Jacobian (dimension minDim x dim).
  /// \return True on success.
  static bool liftJacobian(const double* /*x*/, double* jacobian) {
    Eigen::Map<Eigen::Matrix<double, kNumParams, kGlobalDim, Eigen::RowMajor> > J_lift(jacobian);
    J_lift.setIdentity();
    return true;
  }

  bool Plus(const double *x, const double *delta, double *x_plus_delta) const final {
    // transform to okvis::kinematics framework
    std::pair<Eigen::Matrix<double, 3, 1>, Eigen::Quaternion<double>> T(
          Eigen::Vector3d(x[0], x[1], x[2]),
          Eigen::Quaterniond(x[6], x[3], x[4], x[5]));
    // call oplus operator in okvis::kinematis
    oplus(delta, &T);

    // copy back
    const Eigen::Vector3d& r = T.first;
    x_plus_delta[0] = r[0];
    x_plus_delta[1] = r[1];
    x_plus_delta[2] = r[2];
    const Eigen::Vector4d& q = T.second.coeffs();
    x_plus_delta[3] = q[0];
    x_plus_delta[4] = q[1];
    x_plus_delta[5] = q[2];
    x_plus_delta[6] = q[3];
    return true;
  }

  bool ComputeJacobian(const double */*x*/, double *jacobian) const final {
    Eigen::Map<Eigen::Matrix<double, kGlobalDim, kNumParams, Eigen::RowMajor>> j(jacobian);
    j.setIdentity();
    return true;
  }

  int GlobalSize() const final { return kGlobalDim; }
  int LocalSize() const final { return kNumParams; }
};

class Extrinsic_p_BC_q_BC final : public Extrinsic_p_BC_q_BC_base {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static const int kModelId = 2;

  static void toDimensionLabels(std::vector<std::string>* extrinsicLabels) {
    *extrinsicLabels = {"p_SC_S_x[m]", "p_SC_S_y", "p_SC_S_z", "q_SC_x", "q_SC_y", "q_SC_z", "q_SC_w"};
  }

  static void toMinDimensionLabels(std::vector<std::string>* extrinsicLabels) {
    *extrinsicLabels = {"p_SC_S_x[m]", "p_SC_S_y", "p_SC_S_z", "theta_SC_x", "theta_SC_y", "theta_SC_z"};
  }

  static void toDesiredStdevs(Eigen::VectorXd* desiredStdevs) {
    desiredStdevs->resize(6);
    desiredStdevs->head<3>().setConstant(0.04);
    desiredStdevs->tail<3>().setConstant(0.01);
  }

  template <class Scalar>
  static std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>> get_T_BC(
      const okvis::kinematics::Transformation& /*T_BC_base*/,
      const Scalar* parameters) {
    Eigen::Matrix<Scalar, 3, 1> t_BC_B(parameters[0], parameters[1], parameters[2]);
    Eigen::Quaternion<Scalar> q_BC(parameters[6], parameters[3], parameters[4],
                                   parameters[5]);
    return std::make_pair(t_BC_B, q_BC);
  }

  static void dT_BC_dExtrinsic(
      Eigen::Matrix<double, 6, Extrinsic_p_BC_q_BC::kNumParams>* j) {
    j->setIdentity();
  }

  /**
   * @brief dpC_dExtrinsic_AIDP anchored inverse depth
   * @param pC pC = (T_BC^{-1} * T_WB_{f_i}^{-1} * T_WB_a * T_BC * [a, b, 1, \rho]^T)_{1:3}
   *     R_BC = exp(\delta\theta) \hat{R}_BC
   *     t_BC = \delta t + \hat{t}_BC
   * @param R_CB
   * @param dpC_dT
   * @param R_CfCa
   * @param ab1rho
   */
  static void dpC_dExtrinsic_AIDP(const Eigen::Vector3d& pC,
                                  const Eigen::Matrix3d& R_CB,
                                  Eigen::MatrixXd* dpC_dT,
                                  const Eigen::Matrix3d* R_CfCa,
                                  const Eigen::Vector4d* ab1rho) {
    dpC_dT->resize(3, 6);
    dpC_dT->block<3, 3>(0, 0) =
        ((*R_CfCa) - Eigen::Matrix3d::Identity()) * R_CB * (*ab1rho)[3];
    dpC_dT->block<3, 3>(0, 3) =
        (okvis::kinematics::crossMx(pC) -
         (*R_CfCa) * okvis::kinematics::crossMx(ab1rho->head<3>())) *
        R_CB;
  }

  /**
   * @brief dpC_dExtrinsic_HPP homogeneous point
   * @param hpC hpC = T_BC^{-1} * T_WB_{f_i}^{-1} * [x,y,z,w]_W^T
   *     hpC = [pC, w]
   *     R_BC = exp(\delta\theta) \hat{R}_BC
   *     t_BC = \delta t + \hat{t}_BC
   * @param R_CB
   * @param dpC_dT 3x6
   */
  static void dpC_dExtrinsic_HPP(const Eigen::Vector4d& hpC,
                                 const Eigen::Matrix3d& R_CB,
                                 Eigen::MatrixXd* dpC_dT) {
    dpC_dT->resize(3, 6);
    dpC_dT->block<3, 3>(0, 0) = -R_CB * hpC[3];
    dpC_dT->block<3, 3>(0, 3) = okvis::kinematics::crossMx(hpC.head<3>()) * R_CB;
  }

  // see dpC_dExtrinsic_HPP
  static void dhC_dExtrinsic_HPP(const Eigen::Matrix<double, 4, 1>& hpC,
                 const Eigen::Matrix<double, 3, 3>& R_CB,
                 Eigen::Matrix<double, 4, kNumParams>* dhC_deltaTBC) {
    dhC_deltaTBC->block<3, 3>(0, 0) = -R_CB * hpC[3];
    dhC_deltaTBC->block<3, 3>(0, 3) = okvis::kinematics::crossMx(hpC.head<3>()) * R_CB;
    dhC_deltaTBC->row(3).setZero();
  }
};

class Extrinsic_p_C0C_q_C0C final : public Extrinsic_p_BC_q_BC_base {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static const int kModelId = 3;

  static void toDimensionLabels(std::vector<std::string>* extrinsicLabels) {
    *extrinsicLabels = {"p_C0C_C0_x[m]", "p_C0C_C0_y", "p_C0C_C0_z", "q_C0C_x", "q_C0C_y", "q_C0C_z", "q_C0C_w"};
  }

  static void toMinDimensionLabels(std::vector<std::string>* extrinsicLabels) {
    *extrinsicLabels = {"p_C0C_C0_x[m]", "p_C0C_C0_y", "p_C0C_C0_z", "theta_C0C_x", "theta_C0C_y", "theta_C0C_z"};
  }

  static void toDesiredStdevs(Eigen::VectorXd* desiredStdevs) {
    desiredStdevs->resize(6);
    desiredStdevs->head<3>().setConstant(0.04);
    desiredStdevs->tail<3>().setConstant(0.01);
  }

  static void dT_BC_dExtrinsic(const okvis::kinematics::Transformation& T_BCi,
                               const okvis::kinematics::Transformation& T_BC0,
                               Eigen::Matrix<double, 6, kNumParams>* j_C0Ci,
                               Eigen::Matrix<double, 6, kNumParams>* j_BC0);

  static void dpC_dExtrinsic_AIDP(const Eigen::Vector3d& /*pC*/,
                                  const Eigen::Matrix3d& /*R_CB*/,
                                  Eigen::MatrixXd* /*dpC_dT*/,
                                  const Eigen::Matrix3d* /*R_CfCa*/,
                                  const Eigen::Vector4d* /*ab1rho*/) {
    throw std::runtime_error("Method not implemented!");
  }

  static void dpC_dExtrinsic_HPP(const Eigen::Vector4d& /*hpC*/,
                                 const Eigen::Matrix3d& /*R_CB*/,
                                 Eigen::MatrixXd* /*dpC_dT*/) {
    throw std::runtime_error("Method not implemented!");
  }
};

#ifndef EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASES               \
  EXTRINSIC_MODEL_CASE(Extrinsic_p_CB)      \
  EXTRINSIC_MODEL_CASE(Extrinsic_p_BC_q_BC) \
  EXTRINSIC_MODEL_CASE(Extrinsic_p_C0C_q_C0C)
#endif

inline int ExtrinsicModelGetMinimalDim(int model_id) {
  switch (model_id) {
#define MODEL_CASES EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASE(ExtrinsicModel) \
  case ExtrinsicModel::kModelId:             \
    return ExtrinsicModel::getMinimalDim();

    MODEL_SWITCH_CASES

#undef EXTRINSIC_MODEL_CASE
#undef MODEL_CASES
  }
  return 0;
}

inline Eigen::MatrixXd ExtrinsicModelInitCov(int model_id,
                                             double sigma_translation,
                                             double sigma_orientation) {
  switch (model_id) {
#define MODEL_CASES EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASE(ExtrinsicModel) \
  case ExtrinsicModel::kModelId:             \
    return ExtrinsicModel::initCov(sigma_translation, sigma_orientation);

    MODEL_SWITCH_CASES

#undef EXTRINSIC_MODEL_CASE
#undef MODEL_CASES
  }
}

inline int ExtrinsicModelNameToId(std::string extrinsic_opt_rep, bool* isFixed = nullptr) {
  std::transform(extrinsic_opt_rep.begin(), extrinsic_opt_rep.end(),
                 extrinsic_opt_rep.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  if (extrinsic_opt_rep.compare("P_BC_Q_BC") == 0) {
    if (isFixed) {
      *isFixed = false;
    }
    return Extrinsic_p_BC_q_BC::kModelId;
  } else if (extrinsic_opt_rep.compare("P_C0C_Q_C0C") == 0) {
    if (isFixed) {
      *isFixed = false;
    }
    return Extrinsic_p_C0C_q_C0C::kModelId;
  } else if (extrinsic_opt_rep.compare("P_CB") == 0) {
    if (isFixed) {
      *isFixed = false;
    }
    return Extrinsic_p_CB::kModelId;
  } else {
    if (isFixed) {
      *isFixed = true;
    }
    return Extrinsic_p_BC_q_BC::kModelId;
  }
}

inline void ExtrinsicModelUpdateState(int model_id, const Eigen::Vector3d& r,
                                      const Eigen::Quaterniond& q,
                                      const Eigen::VectorXd& delta,
                                      Eigen::Vector3d* r_delta,
                                      Eigen::Quaterniond* q_delta) {
  switch (model_id) {
#define MODEL_CASES EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASE(ExtrinsicModel) \
  case ExtrinsicModel::kModelId:             \
    return ExtrinsicModel::updateState(r, q, delta, r_delta, q_delta);

    MODEL_SWITCH_CASES

#undef EXTRINSIC_MODEL_CASE
#undef MODEL_CASES
  }
}

inline void ExtrinsicModelOplus(int model_id, const double* const deltaT_XC,
                                okvis::kinematics::Transformation* T_XC) {
  std::pair<Eigen::Matrix<double, 3, 1>, Eigen::Quaternion<double>>
      pair_T_XC = std::make_pair(T_XC->r(), T_XC->q());
  switch (model_id) {
#define MODEL_CASES EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASE(ExtrinsicModel)                          \
  case ExtrinsicModel::kModelId:                                      \
    ExtrinsicModel::oplus(deltaT_XC, &pair_T_XC);                     \
    T_XC->set(pair_T_XC.first, pair_T_XC.second);                     \
    break;

    MODEL_SWITCH_CASES

#undef EXTRINSIC_MODEL_CASE
#undef MODEL_CASES
  }
}

inline void ExtrinsicModelOminus(int model_id, const double* rq,
                                   const double* rq_delta,
                                   double* delta) {
  switch (model_id) {
#define MODEL_CASES EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASE(ExtrinsicModel) \
  case ExtrinsicModel::kModelId:             \
    return ExtrinsicModel::ominus(rq, rq_delta, delta);

    MODEL_SWITCH_CASES

#undef EXTRINSIC_MODEL_CASE
#undef MODEL_CASES
  }
}

inline void ExtrinsicModel_dpC_dExtrinsic_AIDP(int model_id, const Eigen::Vector3d& pC,
                                               const Eigen::Matrix3d& R_CB,
                                               Eigen::MatrixXd* dpC_dT,
                                               const Eigen::Matrix3d* R_CfCa,
                                               const Eigen::Vector4d* ab1rho) {
  switch (model_id) {
#define MODEL_CASES EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASE(ExtrinsicModel) \
  case ExtrinsicModel::kModelId:             \
    return ExtrinsicModel::dpC_dExtrinsic_AIDP(pC, R_CB, dpC_dT, R_CfCa, ab1rho);

    MODEL_SWITCH_CASES

#undef EXTRINSIC_MODEL_CASE
#undef MODEL_CASES
  }
}

inline void ExtrinsicModel_dpC_dExtrinsic_HPP(int model_id, const Eigen::Vector4d& hpC,
                                              const Eigen::Matrix3d& R_CB,
                                              Eigen::MatrixXd* dpC_dT) {
  switch (model_id) {
#define MODEL_CASES EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASE(ExtrinsicModel) \
  case ExtrinsicModel::kModelId:             \
    return ExtrinsicModel::dpC_dExtrinsic_HPP(hpC, R_CB, dpC_dT);

    MODEL_SWITCH_CASES

#undef EXTRINSIC_MODEL_CASE
#undef MODEL_CASES
  }
}

inline void ExtrinsicModelToDimensionLabels(
    int model_id, std::vector<std::string>* dimensionLabels) {
    switch (model_id) {
  #define MODEL_CASES EXTRINSIC_MODEL_CASES
  #define EXTRINSIC_MODEL_CASE(ExtrinsicModel) \
    case ExtrinsicModel::kModelId:             \
      return ExtrinsicModel::toDimensionLabels(dimensionLabels);

      MODEL_SWITCH_CASES

  #undef EXTRINSIC_MODEL_CASE
  #undef MODEL_CASES
    }
}

inline void ExtrinsicModelToMinDimensionLabels(
    int model_id, std::vector<std::string>* dimensionLabels) {
    switch (model_id) {
  #define MODEL_CASES EXTRINSIC_MODEL_CASES
  #define EXTRINSIC_MODEL_CASE(ExtrinsicModel) \
    case ExtrinsicModel::kModelId:             \
      return ExtrinsicModel::toMinDimensionLabels(dimensionLabels);

      MODEL_SWITCH_CASES

  #undef EXTRINSIC_MODEL_CASE
  #undef MODEL_CASES
    }
}

inline void ExtrinsicModelToParamValues(
    int model_id, const okvis::kinematics::Transformation& T_XC,
    Eigen::VectorXd* extrinsic_opt_coeffs) {
  switch (model_id) {
#define MODEL_CASES EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASE(ExtrinsicModel)                    \
  case ExtrinsicModel::kModelId:                                \
    return ExtrinsicModel::toParamValues(                       \
        T_XC, extrinsic_opt_coeffs);

    MODEL_SWITCH_CASES

#undef EXTRINSIC_MODEL_CASE
#undef MODEL_CASES
  }
}


inline void ExtrinsicModelToDesiredStdevs(
    int model_id, Eigen::VectorXd* desiredStdevs) {
  switch (model_id) {
#define MODEL_CASES EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASE(ExtrinsicModel)                    \
  case ExtrinsicModel::kModelId:                                \
    return ExtrinsicModel::toDesiredStdevs(desiredStdevs);

    MODEL_SWITCH_CASES

#undef EXTRINSIC_MODEL_CASE
#undef MODEL_CASES
  }
}
}  // namespace swift_vio
#endif  // INCLUDE_SWIFT_VIO_EXTRINSIC_MODELS_HPP_
