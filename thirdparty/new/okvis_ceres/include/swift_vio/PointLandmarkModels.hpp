#ifndef INCLUDE_SWIFT_VIO_POINT_LANDMARK_MODELS_HPP_
#define INCLUDE_SWIFT_VIO_POINT_LANDMARK_MODELS_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
namespace swift_vio {
// Expressed in an anchor camera frame [\alpha, \beta, 1, \rho] = [x, y, z, w]/z.
class InverseDepthParameterization final: public ::ceres::LocalParameterization
{
public:
  static const int kModelId = 1;
  static const int kGlobalDim = 4;
  static const int kLocalDim = 3;

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
    j.setZero();
    j(0, 0) = 1.0;
    j(1, 1) = 1.0;
    j(3, 2) = 1.0;
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

  static bool plus(const double* x, const double* delta, double* x_plus_delta);

  static bool minus(const double* x, const double* x_plus_delta, double* delta);

  static bool liftJacobian(const double* /*x*/, double* jacobian) {
    Eigen::Map<Eigen::Matrix<double, kLocalDim, kGlobalDim, Eigen::RowMajor>> j(jacobian);
    j.setZero();
    j(0, 0) = 1.0;
    j(1, 1) = 1.0;
    j(2, 3) = 1.0;
    return true;
  }
};

// [x, y, z, w, c, s]
// [x, y, z, w] is the quaternion underlying unit bearing vector n such that
// n = q(w, x, y, z) * [0, 0, 1]'.
// c, s are cos(theta) and sin(theta) where \theta is the parallax angle.
class ParallaxAngleParameterization final: public ::ceres::LocalParameterization {
public:
  static const int kModelId = 2;
  static const int kGlobalDim = 6;
  static const int kLocalDim = 3;

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
  static bool plus(const double* x, const double* delta, double* x_plus_delta);

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

// add the model switch functions
} // namespace swift_vio

#endif // INCLUDE_SWIFT_VIO_POINT_LANDMARK_MODELS_HPP_
