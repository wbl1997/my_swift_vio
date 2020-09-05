
/**
 *@file  RiExtendedPose3.h
 *@brief right invariant Extended Pose
 */

#ifndef INCLUDE_GTSAM_RI_EXTENDED_POSE3_H
#define INCLUDE_GTSAM_RI_EXTENDED_POSE3_H

#include <Eigen/Dense>

#include <gtsam/geometry/BearingRange.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/base/Manifold.h>

namespace gtsam {
namespace geometry {
inline Eigen::Matrix3d SO3Jr(const Eigen::Vector3d& omega) {
  return Rot3::ExpmapDerivative(omega);
}

Eigen::Matrix3d SO3Jr_eigen(const Eigen::Vector3d& omega);

inline Eigen::Matrix3d SO3Jr_inv(const Eigen::Vector3d& omega) {
  return Rot3::LogmapDerivative(omega);
}

inline Eigen::Matrix3d SO3Jl(const Eigen::Vector3d& omega) { return SO3Jr(-omega); }

inline Eigen::Matrix3d SO3Jl_inv(const Eigen::Vector3d& omega) {
  return Rot3::LogmapDerivative(-omega);
}

inline Eigen::Matrix<double, 6, 6> SE3Jr(const Eigen::Matrix<double, 6, 1>& xi) {
  return gtsam::Pose3::ExpmapDerivative(xi);
}

inline Eigen::Matrix<double, 6, 6> SE3Jr_inv(const Eigen::Matrix<double, 6, 1>& xi) {
  return gtsam::Pose3::LogmapDerivative(gtsam::Pose3::Expmap(xi));
}

inline Eigen::Matrix<double, 6, 6> SE3Jl(const Eigen::Matrix<double, 6, 1>& xi) {
  return gtsam::Pose3::ExpmapDerivative(-xi);
}

inline Eigen::Matrix<double, 6, 6> SE3Jl_inv(const Eigen::Matrix<double, 6, 1>& xi) {
  return gtsam::Pose3::LogmapDerivative(gtsam::Pose3::Expmap(-xi));
}

/** The closed-form formula in Barfoot14tro eq. (102).
 */
Eigen::Matrix3d SE3Ql(const Eigen::Vector3d& omega, const Eigen::Vector3d& rho);

/**
 * @brief SE3Qr
 * This in effect should be identical to
 * gtsam::Pose3::computeQforExpmapDerivative.
 * @param omega
 * @param rho
 * @return
 */
inline Eigen::Matrix3d SE3Qr(const Eigen::Vector3d& omega,
                             const Eigen::Vector3d& rho) {
  return SE3Ql(-omega, -rho);
}

/**
 * Compute the 6x3 bottom-left block Qs of the SE_2(3) Expmap derivative
 * matrix with the left invariant error formulation.
 */
Matrix63 computeQrforExpmapDerivative(const Vector9& xi);

Matrix9 SEK3Jr(const Vector9& xi);

Matrix9 SEK3Jr_inv(const Vector9& xi);

inline Matrix9 SEK3Jl(const Vector9& xi) { return SEK3Jr(-xi); }

inline Matrix9 SEK3Jl_inv(const Vector9& xi) { return SEK3Jr_inv(-xi); }

};  // namespace geometry

/**
 * A 3D extended pose (R,v,p) : (Rot3,Point3,Point3)
 * @addtogroup geometry
 * \nosubgrouping
 */
class GTSAM_EXPORT RiExtendedPose3 {
 public:
  /** Pose Concept requirements */
  typedef Rot3 Rotation;
  typedef Point3 Velocity;
  typedef Point3 Position;
  typedef Position Translation;
  typedef OptionalJacobian<9, 9> ChartJacobian;

 private:
  Rot3 R_;    ///< Rotation gRp, between global and pose frame
  Point3 v_;  ///< Velocity gVp, from global origin to pose frame origin
  Point3 p_;  ///< Position gPp, from global origin to pose frame origin

 public:
  enum { dimension = 9 };
  /// @name Standard Constructors
  /// @{

  /** Default constructor is origin */
  RiExtendedPose3()
      : R_(traits<Rot3>::Identity()),
        v_(traits<Point3>::Identity()),
        p_(traits<Point3>::Identity()) {}

  /** Copy constructor */
  RiExtendedPose3(const RiExtendedPose3& pose)
      : R_(pose.R_), v_(pose.v_), p_(pose.p_) {}

  /** Construct from R,v,p */
  RiExtendedPose3(const Rot3& R, const Point3& v, const Point3& p)
      : R_(R), v_(v), p_(p) {}

  RiExtendedPose3& operator=(RiExtendedPose3 other) {
    R_ = other.R_;
    v_ = other.v_;
    p_ = other.p_;
    return *this;
  }

  // explicit Pose3(const Pose2& pose2);

  /** Constructor from 5*5 matrix */
  RiExtendedPose3(const Matrix& T)
      : R_(T(0, 0), T(0, 1), T(0, 2), T(1, 0), T(1, 1), T(1, 2), T(2, 0),
           T(2, 1), T(2, 2)),
        v_(T(0, 3), T(1, 3), T(2, 3)),
        p_(T(0, 4), T(1, 4), T(2, 4)) {}

  /// Named constructor with derivatives
  //    static RiExtendedPose3 Create(const Rot3& R, const Point3& v, const
  //    Point3& p,
  //                        OptionalJacobian<9, 3> HR = boost::none,
  //                        OptionalJacobian<9, 3> Hv = boost::none,
  //                        OptionalJacobian<9, 3> Hp = boost::none);

  /// @}
  /// @name Testable
  /// @{

  /// print with optional string
  void print(const std::string& s = "") const;

  /// assert equality up to a tolerance
  bool equals(const RiExtendedPose3& pose, double tol = 1e-9) const;

  /// @}
  /// @name Group
  /// @{

  /// identity for group operation
  static RiExtendedPose3 identity() { return RiExtendedPose3(); }

  /// inverse transformation with derivatives
  RiExtendedPose3 inverse() const;

  /// compose syntactic sugar
  RiExtendedPose3 operator*(const RiExtendedPose3& T) const {
    return RiExtendedPose3(R_ * T.R_, v_ + R_ * T.v_, p_ + R_ * T.p_);
  }

  /// @}
  /// @name Manifold, with left invariant error formulation.
  /// @{
  /**
   * @brief retract Retract with optional derivatives (given correction, change this navstate).
   * @param xi
   * @return exp(xi) * X.
   */
  RiExtendedPose3 retract(const Vector9& xi) const;

  /**
   * @brief localCoordinates Converting function from our overparameterization to the local representation.
   * a.localCoordinates(a.retract(xi)) = xi.
   * @param state
   * @return Logmap(state * X^{-1})
   */
  Vector9 localCoordinates(const RiExtendedPose3& state) const;

  /**
   * @brief expmap Retract with optional derivatives (given correction, change this navstate).
   * @param xi
   * @return exp(xi) * X.
   */
  RiExtendedPose3 expmap(const Vector9& xi) const;

  /**
   * @brief logmap Converting function from our overparameterization to the local representation.
   * @param state
   * @return Logmap(state * X^{-1})
   */
  Vector9 logmap(const RiExtendedPose3& state) const;
  /// @}

  /// @name Lie Group
  /// @{

  /// Exponential map at identity - create a rotation from canonical coordinates
  /// \f$ [R_x,R_y,R_z,V_x,V_y,V_z,P_x,P_y,P_z] \f$
  static RiExtendedPose3 Expmap(const Vector9& xi,
                                OptionalJacobian<9, 9> Hxi = boost::none);

  /// Log map at identity - return the canonical coordinates \f$
  /// [R_x,R_y,R_z,V_x,V_y,V_z,P_x,P_y,P_z] \f$ of this rotation
  static Vector9 Logmap(const RiExtendedPose3& pose,
                        OptionalJacobian<9, 9> Hpose = boost::none);

  /// @name gtsam conventional Lie functions.
  /// @brief The below 4 compose and between functions are copied from gtsam/gtsam/base/Lie.h.
  /// @{
  RiExtendedPose3 compose(const RiExtendedPose3& g) const { return *this * g; }

  RiExtendedPose3 between(const RiExtendedPose3& g) const {
    return inverse() * g;
  }

  RiExtendedPose3 compose(const RiExtendedPose3& g, ChartJacobian H1,
                          ChartJacobian H2 = boost::none) const {
    if (H1) *H1 = g.inverse().AdjointMap();
    if (H2) *H2 = Eigen::Matrix<double, 9, 9>::Identity();
    return *this * g;
  }

  RiExtendedPose3 between(const RiExtendedPose3& g, ChartJacobian H1,
                          ChartJacobian H2 = boost::none) const {
    RiExtendedPose3 result = inverse() * g;
    if (H1) *H1 = -result.inverse().AdjointMap();
    if (H2) *H2 = Eigen::Matrix<double, 9, 9>::Identity();
    return result;
  }
  /// @}

  inline void setRandom() { setRandom(1.0, M_PI, 2.0); }

  // Set this to a random ExtendedPose3 with bounded rotation, translation,
  // and velocity.
  inline void setRandom(double translationMaxMeters, double rotationMaxRadians,
                        double maxSpeed) {
    Eigen::Vector3d axis = rotationMaxRadians * Eigen::Vector3d::Random();
    Eigen::Vector3d r = translationMaxMeters * Eigen::Vector3d::Random();
    p_ = r;
    R_ = gtsam::Rot3(
        Eigen::AngleAxisd(axis.norm(), axis.normalized()).toRotationMatrix());
    v_ = maxSpeed * Eigen::Vector3d::Random();
  }

  /**
   * Calculate Adjoint map, transforming a twist in the this pose's (i.e, body)
   * frame to the world spatial frame Ad_pose is 9*9 matrix that when applied to
   * twist xi \f$ [R_x,R_y,R_z,V_x,V_y,V_z,P_x,P_y,P_z] \f$, returns Ad_pose(xi)
   */
  Matrix9 AdjointMap() const;  /// FIXME Not tested - marked as incorrect

  /**
   * Apply this pose's AdjointMap Ad_g to a twist \f$ \xi_b \f$, i.e. a
   * body-fixed velocity, transforming it to the spatial frame \f$ \xi^s =
   * g*\xi^b*g^{-1} = Ad_g * \xi^b \f$
   */
  Vector9 Adjoint(const Vector9& xi_b) const {
    return AdjointMap() * xi_b;
  }  /// FIXME Not tested - marked as incorrect

  /**
   * Compute the ad operator
   */
  static Matrix9 adjointMap(const Vector9& xi);

  /**
   * Action of the adjointMap on a Lie-algebra vector y, with optional
   * derivatives
   */
  static Vector9 adjoint(const Vector9& xi, const Vector9& y,
                         OptionalJacobian<9, 9> Hxi = boost::none);

  // temporary fix for wrappers until case issue is resolved
  static Matrix9 adjointMap_(const Vector9& xi) { return adjointMap(xi); }
  static Vector9 adjoint_(const Vector9& xi, const Vector9& y) {
    return adjoint(xi, y);
  }

  /**
   * The dual version of adjoint action, acting on the dual space of the
   * Lie-algebra vector space.
   */
  static Vector9 adjointTranspose(const Vector9& xi, const Vector9& y,
                                  OptionalJacobian<9, 9> Hxi = boost::none);

  /**
   * @brief ExpmapDerivative Derivative of Expmap
   * Expmap(xi).local(Expmap(xi + \delta)) \approx ExpmapDerivative(x) * \delta.
   * @param xi
   * @return J_l(xi)
   */
  static Matrix9 ExpmapDerivative(const Vector9& xi);

  /**
   * @brief LogmapDerivative Derivative of Logmap
   * logmap(x.retract(\xi)) - logmap(x) \approx LogmapDerivative(x) * \xi
   * This maps a perturbation on the manifold (x) to a perturbation in the
   * tangent space.
   * @param x = expmap(\phi)
   * @return J_l^{-1}(Logmap(x))
   */
  static Matrix9 LogmapDerivative(const RiExtendedPose3& x);

  // Chart at origin, depends on compile-time flag GTSAM_POSE3_EXPMAP
  struct ChartAtOrigin {
    static RiExtendedPose3 Retract(const Vector9& xi,
                                   ChartJacobian Hxi = boost::none);
    static Vector9 Local(const RiExtendedPose3& pose,
                         ChartJacobian Hpose = boost::none);
  };

  Vector9 boxminus(const RiExtendedPose3& g) const;

  /**
   * wedge for RiExtendedPose3:
   * @param xi 9-dim twist (omega,nu,rho)
   * @return 5*5 element of Lie algebra
   */
  static Matrix wedge(double phix, double phiy, double phiz, double nux,
                      double nuy, double nuz, double rhox, double rhoy,
                      double rhoz) {
    return (Matrix(5, 5) << 0., -phiz, phiy, nux, rhox, phiz, 0., -phix, nuy,
            rhoy, -phiy, phix, 0., nuz, rhoz, 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0.)
        .finished();
  }

  static Matrix wedge(const Vector& xi) {
    return wedge(xi[0], xi[1], xi[2], xi[3], xi[4], xi[5], xi[6], xi[7], xi[8]);
  }

  /// @}
  /// @name Group Action on Point3
  /// @{

  /**
   * @brief takes point in Pose coordinates and transforms it to world
   * coordinates
   * @param point point in Pose coordinates
   * @param Hself optional 3*6 Jacobian wrpt this pose
   * @param Hpoint optional 3*3 Jacobian wrpt point
   * @return point in world coordinates
   */
  Point3 transformFrom(const Point3& point,
                       OptionalJacobian<3, 9> Hself = boost::none,
                       OptionalJacobian<3, 3> Hpoint = boost::none) const;

  /** syntactic sugar for transformFrom */
  inline Point3 operator*(const Point3& point) const {
    return transformFrom(point);
  }

  /**
   * @brief takes point in world coordinates and transforms it to Pose
   * coordinates
   * @param point point in world coordinates
   * @param Hself optional 3*6 Jacobian wrpt this pose
   * @param Hpoint optional 3*3 Jacobian wrpt point
   * @return point in Pose coordinates
   */
  Point3 transformTo(const Point3& point,
                     OptionalJacobian<3, 9> Hself = boost::none,
                     OptionalJacobian<3, 3> Hpoint = boost::none) const;

  /// @}
  /// @name Standard Interface
  /// @{

  /// get rotation
  const Rot3& rotation(OptionalJacobian<3, 9> Hself = boost::none) const;

  /// get translation
  const Point3& velocity(OptionalJacobian<3, 9> Hself = boost::none) const;

  /// get position
  const Point3& position(OptionalJacobian<3, 9> Hself = boost::none) const;

  const Point3& translation(OptionalJacobian<3, 9> Hself = boost::none) const;

  /// get x
  double x() const { return p_.x(); }

  /// get y
  double y() const { return p_.y(); }

  /// get z
  double z() const { return p_.z(); }

  /** convert to 5*5 matrix */
  Matrix5 matrix() const;

  /**
   *  Assuming self == wTa, takes a pose wTb in world coordinates
   * and transforms it to local coordinates aTb = inv(wTa) * wTb
   */
  RiExtendedPose3 transformPoseTo(
      const RiExtendedPose3& wTb, OptionalJacobian<9, 9> Hself = boost::none,
      OptionalJacobian<9, 9> HwTb = boost::none) const;

  /**
   * Calculate range to a landmark
   * @param point 3D location of landmark
   * @return range (double)
   */
  double range(const Point3& point, OptionalJacobian<1, 9> Hself = boost::none,
               OptionalJacobian<1, 3> Hpoint = boost::none) const;

  /**
   * Calculate range to another pose
   * @param pose Other SO(3) pose
   * @return range (double)
   */
  double range(const RiExtendedPose3& pose,
               OptionalJacobian<1, 9> Hself = boost::none,
               OptionalJacobian<1, 9> Hpose = boost::none) const;

  /**
   * Calculate bearing to a landmark
   * @param point 3D location of landmark
   * @return bearing (Unit3)
   */
  Unit3 bearing(const Point3& point, OptionalJacobian<2, 9> Hself = boost::none,
                OptionalJacobian<2, 3> Hpoint = boost::none) const;

  /**
   * Calculate bearing to another pose
   * @param other 3D location and orientation of other body. The orientation
   * information is ignored.
   * @return bearing (Unit3)
   */
  Unit3 bearing(const RiExtendedPose3& pose,
                OptionalJacobian<2, 9> Hself = boost::none,
                OptionalJacobian<2, 9> Hpose = boost::none) const;

  /// @}
  /// @name Advanced Interface
  /// @{

  /**
   * Return the start and end indices (inclusive) of the translation component
   * of the exponential map parameterization
   * @return a pair of [start, end] indices into the tangent space vector
   */
  inline static std::pair<size_t, size_t> velocityInterval() {
    return std::make_pair(3, 5);
  }

  /**
   * Return the start and end indices (inclusive) of the translation component
   * of the exponential map parameterization
   * @return a pair of [start, end] indices into the tangent space vector
   */
  inline static std::pair<size_t, size_t> positionInterval() {
    return std::make_pair(6, 8);
  }
  inline static std::pair<size_t, size_t> translationInterval() {
    return std::make_pair(6, 8);
  }

  /**
   * Return the start and end indices (inclusive) of the rotation component of
   * the exponential map parameterization
   * @return a pair of [start, end] indices into the tangent space vector
   */
  static std::pair<size_t, size_t> rotationInterval() {
    return std::make_pair(0, 2);
  }

  /// Output stream operator
  GTSAM_EXPORT
  friend std::ostream& operator<<(std::ostream& os, const RiExtendedPose3& p);

 private:
  /** Serialization function */
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int /*version*/) {
    ar& BOOST_SERIALIZATION_NVP(R_);
    ar& BOOST_SERIALIZATION_NVP(v_);
    ar& BOOST_SERIALIZATION_NVP(p_);
  }
  /// @}

#ifdef GTSAM_USE_QUATERNIONS
  // Align if we are using Quaternions
 public:
  GTSAM_MAKE_ALIGNED_OPERATOR_NEW
#endif
  };
  // RiExtendedPose3 class

  template <>
  struct traits<RiExtendedPose3> : internal::Manifold<RiExtendedPose3> {};

} // namespace gtsam

#endif // INCLUDE_GTSAM_RI_EXTENDED_POSE3_H
