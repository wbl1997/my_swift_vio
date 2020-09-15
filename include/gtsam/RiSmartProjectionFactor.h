
/**
 * @file   RiSmartProjectionFactor.h
 * @brief  Smart factor on RiExtendedPose3
 * @author Jianzhu Huai
 */

#pragma once

#include <gtsam/slam/SmartFactorBase.h>
#include <gtsam/slam/SmartFactorParams.h>

#include <gtsam/geometry/triangulation.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/dataset.h>

#include "gtsam/RiExtendedPose3.h"
#include "gtsam/RiProjectionFactorIDP.h"
#include "gtsam/RiProjectionFactorIDPAnchor.h"

#include <boost/optional.hpp>
#include <boost/make_shared.hpp>
#include <vector>

#include <okvis/cameras/CameraBase.hpp>
#include <okvis/kinematics/Transformation.hpp>

namespace gtsam {

/**
 * RiSmartProjectionFactor: triangulates point and keeps an estimate of it around.
 * This factor operates with RiExtendedPose3 and local inverse depth parameters.
 * The calibration is fixed and assumed to be the same for all observations.
 */
template<class CALIBRATION>
class RiSmartProjectionFactor: public SmartFactorBase<PinholePose<CALIBRATION> > {

public:
  enum {
    kDim = 9, // RiExtendedPose3
    kZDim = 2, // image measurement
  };
private:
  typedef PinholePose<CALIBRATION> CAMERA;
  typedef SmartFactorBase<CAMERA> Base;
  typedef RiSmartProjectionFactor<CALIBRATION> This;
  typedef RiSmartProjectionFactor<CALIBRATION> SmartProjectionCameraFactor;

protected:

  /// @name Parameters
  /// @{
  boost::shared_ptr<CALIBRATION> K_;

  std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry_;
  SmartProjectionParams params_;
  /// @}

  /// @name Caching triangulation
  /// @{
  mutable TriangulationResult result_; ///< result from triangulateSafe
  mutable std::vector<Pose3, Eigen::aligned_allocator<Pose3> > cameraPosesTriangulation_; ///< current triangulation poses
  /// @}

  int anchorIndex_; ///< index of the anchor camera frame in the observation array.

public:

  /// shorthand for a smart pointer to a factor
  typedef boost::shared_ptr<This> shared_ptr;

  /// shorthand for a set of cameras
  typedef CameraSet<CAMERA> Cameras;

  /**
   * Default constructor, only for serialization
   */
  RiSmartProjectionFactor() {}

  /**
   * Constructor
   * @param sharedNoiseModel isotropic noise model for the 2D feature measurements
   * @param K (fixed) calibration, assumed to be the same for all cameras
   * @param body_P_sensor pose of the camera in the body frame (optional)
   * @param params parameters for the smart projection factors
   */
  RiSmartProjectionFactor(
      const SharedNoiseModel& sharedNoiseModel,      
      const boost::optional<Pose3> body_P_sensor,
      const boost::shared_ptr<CALIBRATION> K,
      std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry,
      const SmartProjectionParams& params = SmartProjectionParams())
      : Base(sharedNoiseModel, body_P_sensor),
        K_(K),
        cameraGeometry_(cameraGeometry),
        params_(params),
        result_(TriangulationResult::Degenerate()), anchorIndex_(-1) {
  }

  /** Virtual destructor */
  virtual ~RiSmartProjectionFactor() {
  }

  /**
   * print
   * @param s optional string naming the factor
   * @param keyFormatter optional formatter useful for printing Symbols
   */
  void print(const std::string& s = "", const KeyFormatter& keyFormatter =
      DefaultKeyFormatter) const override {
    std::cout << s << "RiSmartProjectionFactor\n";
    std::cout << "linearizationMode:\n" << (int)params_.linearizationMode
        << std::endl;
    std::cout << "triangulationParameters:\n" << params_.triangulation
        << std::endl;
    std::cout << "result:\n" << result_ << std::endl;
    Base::print("", keyFormatter);
  }

  /// equals
  bool equals(const NonlinearFactor& p, double tol = 1e-9) const override {
    const This *e = dynamic_cast<const This*>(&p);
    return e && params_.linearizationMode == e->params_.linearizationMode
        && Base::equals(p, tol);
  }

  void setAnchorIndex(int anchorIndex) {
    anchorIndex_ = anchorIndex;
  }

  /// Check if the new linearization point is the same as the one used for previous triangulation
  bool decideIfTriangulate(const Cameras& cameras) const {
    // several calls to linearize will be done from the same linearization point, hence it is not needed to re-triangulate
    // Note that this is not yet "selecting linearization", that will come later, and we only check if the
    // current linearization is the "same" (up to tolerance) w.r.t. the last time we triangulated the point

    size_t m = cameras.size();

    bool retriangulate = false;

    // if we do not have a previous linearization point or the new linearization point includes more poses
    if (cameraPosesTriangulation_.empty()
        || cameras.size() != cameraPosesTriangulation_.size())
      retriangulate = true;

    if (!retriangulate) {
      for (size_t i = 0; i < cameras.size(); i++) {
        if (!cameras[i].pose().equals(cameraPosesTriangulation_[i],
            params_.retriangulationThreshold)) {
          retriangulate = true; // at least two poses are different, hence we retriangulate
          break;
        }
      }
    }

    if (retriangulate) { // we store the current poses used for triangulation
      cameraPosesTriangulation_.clear();
      cameraPosesTriangulation_.reserve(m);
      for (size_t i = 0; i < m; i++)
        cameraPosesTriangulation_.push_back(cameras[i].pose());
    }

    return retriangulate; // if we arrive to this point all poses are the same and we don't need re-triangulation
  }

  /// triangulateSafe
  TriangulationResult triangulateSafe(const Cameras& cameras) const {

    size_t m = cameras.size();
    if (m < 2) // if we have a single pose the corresponding factor is uninformative
      return TriangulationResult::Degenerate();

    bool retriangulate = decideIfTriangulate(cameras);
    if (retriangulate)
      result_ = gtsam::triangulateSafe(cameras, this->measured_,
          params_.triangulation);
    return result_;
  }

  /// triangulate
  bool triangulateForLinearize(const Cameras& cameras) const {
    triangulateSafe(cameras); // imperative, might reset result_
    return bool(result_);
  }

  std::vector<RiExtendedPose3, Eigen::aligned_allocator<RiExtendedPose3>>
  toStateList(const Cameras& cameras) const {
    std::vector<RiExtendedPose3, Eigen::aligned_allocator<RiExtendedPose3>>
        stateList;
    stateList.reserve(cameras.size());
    for (auto camera : cameras) {
      gtsam::Pose3 T_WC = camera.pose();
      gtsam::Pose3 T_WB = T_WC * (*Base::body_P_sensor_).inverse();
      stateList.emplace_back(T_WB.rotation(), gtsam::Point3(), T_WB.translation());
    }
    return stateList;
  }

  /**
   * Do Schur complement, given Jacobian as Fs,E,P, return SymmetricBlockMatrix
   * G = F' * F - F' * E * P * E' * F
   * g = F' * (b - E * P * E' * b)
   * Fixed size version
   */
  template <int N>  // N = 2 or 3
  static SymmetricBlockMatrix SchurComplement(
      const Eigen::MatrixXd& Fs, const Matrix& E,
      const Eigen::Matrix<double, N, N>& P, const Vector& b) {
    size_t m = Fs.rows() / kZDim;  // a single point is observed in m cameras

    // Create a SymmetricBlockMatrix
    size_t M1 = kDim * m + 1;
    std::vector<DenseIndex> dims(m + 1);  // this also includes the b term
    std::fill(dims.begin(), dims.end() - 1, kDim);
    dims.back() = 1;
    SymmetricBlockMatrix augmentedHessian(dims, Matrix::Zero(M1, M1));
    Eigen::MatrixXd EPEt = E * P * E.transpose();
    Eigen::MatrixXd G = Fs.transpose() * Fs - Fs.transpose() * EPEt * Fs;
    Eigen::VectorXd g = Fs.transpose() * (b - EPEt * b);

    for (size_t i = 0; i < m; i++) {
      augmentedHessian.setOffDiagonalBlock(i, m, g.segment<kDim>(kDim * i));
      augmentedHessian.setDiagonalBlock(i, G.block<kDim, kDim>(i * kDim, i * kDim));
      for (size_t j = i + 1; j < m; j++) {
        augmentedHessian.setOffDiagonalBlock(i, j, G.block<kDim, kDim>(i * kDim, j * kDim));
      }
    }

    augmentedHessian.diagonalBlock(m)(0, 0) += b.squaredNorm();
    return augmentedHessian;
  }

  template<int N> // N = 2 or 3
  static void ComputePointCovariance(Eigen::Matrix<double, N, N>& P,
      const Matrix& E, double lambda, bool diagonalDamping = false) {

    Matrix EtE = E.transpose() * E;

    if (diagonalDamping) { // diagonal of the hessian
      EtE.diagonal() += lambda * EtE.diagonal();
    } else {
      DenseIndex n = E.cols();
      EtE += lambda * Eigen::MatrixXd::Identity(n, n);
    }

    P = (EtE).inverse();
  }

  static SymmetricBlockMatrix SchurComplement(const Eigen::MatrixXd& Fblocks,
      const Matrix& E, const Vector& b, const double lambda = 0.0,
      bool diagonalDamping = false) {
    if (E.cols() == 2) {
      Matrix2 P;
      ComputePointCovariance<2>(P, E, lambda, diagonalDamping);
      return SchurComplement<2>(Fblocks, E, P, b);
    } else {
      Matrix3 P;
      ComputePointCovariance<3>(P, E, lambda, diagonalDamping);
      return SchurComplement<3>(Fblocks, E, P, b);
    }
  }

  /// linearize returns a Hessianfactor that is an approximation of error(p)
  boost::shared_ptr<RegularHessianFactor<kDim> > createHessianFactor(
      const Cameras& cameras, const double lambda = 0.0, bool diagonalDamping =
          false) const {
    size_t numKeys = this->keys_.size();
    // Create structures for Hessian Factors
    KeyVector js;
    std::vector<Matrix> Gs(numKeys * (numKeys + 1) / 2);
    std::vector<Vector> gs(numKeys);

    if (this->measured_.size() != cameras.size())
      throw std::runtime_error("SmartProjectionHessianFactor: this->measured_"
                               ".size() inconsistent with input");

    triangulateSafe(cameras);

    if (params_.degeneracyMode == ZERO_ON_DEGENERACY && !result_) {
      // failed: return"empty" Hessian
      for(Matrix& m: Gs)
        m = Matrix::Zero(kDim, kDim);
      for(Vector& v: gs)
        v = Vector::Zero(kDim);
      return boost::make_shared<RegularHessianFactor<kDim> >(this->keys_,
          Gs, gs, 0.0);
    }

    // Jacobian could be 3D Point3 OR 2D Unit3, difference is E.cols().
    Eigen::MatrixXd Fblocks; // 2m x 9m m is number of observations.
    Matrix E;
    Vector b;

    computeJacobiansWithTriangulatedPoint(Fblocks, E, b, toStateList(cameras));

    Base::noiseModel_->WhitenSystem(E, b);
    Fblocks = Base::noiseModel_->Whiten(Fblocks);

    // build augmented hessian
    SymmetricBlockMatrix augmentedHessian =
        SchurComplement(Fblocks, E, b, lambda, diagonalDamping);

    return boost::make_shared<RegularHessianFactor<kDim> >(this->keys_,
        augmentedHessian);
  }

  /// create factor
//  boost::shared_ptr<JacobianFactorQ<kDim, 2> > createJacobianQFactor(
//      const Cameras& cameras, double lambda) const {
//    if (triangulateForLinearize(cameras))
//      return Base::createJacobianQFactor(cameras, *result_, lambda);
//    else
//      // failed: return empty
//      return boost::make_shared<JacobianFactorQ<kDim, 2> >(this->keys_);
//  }

  /// Create a factor, takes values
//  boost::shared_ptr<JacobianFactorQ<kDim, 2> > createJacobianQFactor(
//      const Values& values, double lambda) const {
//    return createJacobianQFactor(this->cameras(values), lambda);
//  }

  /// different (faster) way to compute Jacobian factor
//  boost::shared_ptr<JacobianFactor> createJacobianSVDFactor(
//      const Cameras& cameras, double lambda) const {
//    if (triangulateForLinearize(cameras))
//      return Base::createJacobianSVDFactor(cameras, *result_, lambda);
//    else
//      // failed: return empty
//      return boost::make_shared<JacobianFactorSVD<kDim, 2> >(this->keys_);
//  }


  /**
   * Linearize to Gaussian Factor
   * @param values Values structure which must contain camera poses for this factor
   * @return a Gaussian factor
   */
  boost::shared_ptr<GaussianFactor> linearizeDamped(const Cameras& cameras,
      const double lambda = 0.0) const {
    // depending on flag set on construction we may linearize to different linear factors
    switch (params_.linearizationMode) {
    case HESSIAN:
      return createHessianFactor(cameras, lambda);
//    case JACOBIAN_SVD:
//      return createJacobianSVDFactor(cameras, lambda);
//    case JACOBIAN_Q:
//      return createJacobianQFactor(cameras, lambda);
    default:
      throw std::runtime_error("SmartFactorlinearize: unknown mode");
    }
  }

  /**
   * Linearize to Gaussian Factor
   * @param values Values structure which must contain camera poses for this factor
   * @return a Gaussian factor
   */
  boost::shared_ptr<GaussianFactor> linearizeDamped(const Values& values,
      const double lambda = 0.0) const {
    // depending on flag set on construction we may linearize to different linear factors
    Cameras cameras = this->cameras(values);
    return linearizeDamped(cameras, lambda);
  }

  /// linearize
  boost::shared_ptr<GaussianFactor> linearize(
      const Values& values) const override {
    return linearizeDamped(values);
  }

  void computeJacobians(
      Eigen::MatrixXd& Fs, Matrix& E, Vector& b,
      const std::vector<RiExtendedPose3,
                        Eigen::aligned_allocator<RiExtendedPose3>>& stateList,
      const Eigen::Vector3d& abrho) const {
    // Project into Camera set and calculate derivatives
    // As in expressionFactor, RHS vector b = - (h(x_bar) - z) = z-h(x_bar)
    // Indeed, nonlinear error |h(x_bar+dx)-z| ~ |h(x_bar) + A*dx - z|
    //                                         = |A*dx - (z-h(x_bar))|
    b = -unwhitenedError(stateList, abrho, Fs, E);
  }

  /// Compute reprojection errors [h(x)-z] = [cameras.project(p)-z] and
  /// derivatives
  Vector unwhitenedError(
      const std::vector<RiExtendedPose3,
                        Eigen::aligned_allocator<RiExtendedPose3>>& stateList,
      const Eigen::Vector3d& abrho,
      boost::optional<Matrix&> Fs = boost::none,
      boost::optional<Matrix&> E = boost::none) const {
    int m = stateList.size();
    okvis::kinematics::Transformation T_BC(
        Base::body_P_sensor_->translation(),
        Base::body_P_sensor_->rotation().toQuaternion());
    Eigen::MatrixXd fullF = Eigen::MatrixXd::Zero(kZDim * m, kDim * m);
    Eigen::MatrixXd fullE(kZDim * m, 3);
    Eigen::VectorXd fullResidual(kZDim * m);
    double variance = Base::noiseModel_->sigma() * Base::noiseModel_->sigma();
    Eigen::Vector2d variances(variance, variance);

    for (int j = 0; j < m; ++j) {
      if (j != anchorIndex_) {
        gtsam::RiProjectionFactorIDP factor = gtsam::RiProjectionFactorIDP(
            Base::keys_[j], Base::keys_[anchorIndex_],
            gtsam::Symbol('l', 1u), variances, Base::measured_.at(j),
            cameraGeometry_, T_BC, T_BC);
        Eigen::MatrixXd aHj, aHa, aHp;
        Eigen::Vector2d error = factor.evaluateError(
            stateList[j], stateList[anchorIndex_], abrho, aHj, aHa, aHp);

        fullF.block<kZDim, kDim>(kZDim * j, kDim * j) = aHj;
        fullF.block<kZDim, kDim>(kZDim * j, kDim * anchorIndex_) = aHa;
        fullE.block<kZDim, 3>(kZDim * j, 0) = aHp;
        fullResidual.segment<kZDim>(kZDim * j) = error;
      } else {
        gtsam::RiProjectionFactorIDPAnchor factorAnchor =
            gtsam::RiProjectionFactorIDPAnchor(gtsam::Symbol('l', 1u), variances,
                                               Base::measured_.at(j),
                                               cameraGeometry_, T_BC, T_BC);
        Eigen::MatrixXd aHp;
        Eigen::Vector2d error = factorAnchor.evaluateError(abrho, aHp);
        fullE.block<kZDim, 3>(kZDim * j, 0) = aHp;
        fullResidual.segment<kZDim>(kZDim * j) = error;
      }
    }
    if (Fs) {
      *Fs = fullF;
    }
    if (E) {
      *E = fullE;
    }
    return fullResidual;
  }

  /// Compute F, E only (called below in both vanilla and SVD versions)
  /// Assumes the point has been computed
  /// Note E can be 2m*3 or 2m*2, in case point is degenerate
  void computeJacobiansWithTriangulatedPoint(
      Eigen::MatrixXd& Fblocks, Matrix& E, Vector& b,
      const std::vector<RiExtendedPose3,
                        Eigen::aligned_allocator<RiExtendedPose3>>& stateList) const {
    if (!result_) {
      // Handle degeneracy
      // TODO check flag whether we should do this
      Eigen::Vector3d rayxy1;
      cameraGeometry_->backProject(this->measured_.at(anchorIndex_), &rayxy1);
      Eigen::Vector3d abrho(rayxy1[0], rayxy1[1], 0);
      computeJacobians(Fblocks, E, b, stateList, abrho);
      E = E.leftCols(2);
    } else {
      Eigen::Vector3d pW = result_->vector();
      gtsam::Pose3 T_WB(stateList.at(anchorIndex_).rotation(), stateList.at(anchorIndex_).position());
      gtsam::Pose3 T_WC = T_WB * *Base::body_P_sensor_;
      Eigen::Vector3d pC = T_WC.transformTo(pW);
      double zinv = 1.0 / pC[2];
      Eigen::Vector3d abrho(pC[0] * zinv, pC[1] * zinv, zinv);
      computeJacobians(Fblocks, E, b, stateList, abrho);
    }
  }

  /// Version that takes values, and creates the point
//  bool triangulateAndComputeJacobians(
//      std::vector<typename Base::MatrixZD, Eigen::aligned_allocator<typename Base::MatrixZD> >& Fblocks, Matrix& E, Vector& b,
//      const Values& values) const {
//    Cameras cameras = this->cameras(values);
//    bool nonDegenerate = triangulateForLinearize(cameras);
//    if (nonDegenerate)
//      computeJacobiansWithTriangulatedPoint(Fblocks, E, b, cameras);
//    return nonDegenerate;
//  }

  /// takes values
//  bool triangulateAndComputeJacobiansSVD(
//      std::vector<typename Base::MatrixZD, Eigen::aligned_allocator<typename Base::MatrixZD> >& Fblocks, Matrix& Enull, Vector& b,
//      const Values& values) const {
//    Cameras cameras = this->cameras(values);
//    bool nonDegenerate = triangulateForLinearize(cameras);
//    if (nonDegenerate)
//      Base::computeJacobiansSVD(Fblocks, Enull, b, cameras, *result_);
//    return nonDegenerate;
//  }

  /// Calculate vector of re-projection errors, before applying noise model
  Vector reprojectionErrorAfterTriangulation(const Values& values) const {
    Cameras cameras = this->cameras(values);
    bool nonDegenerate = triangulateForLinearize(cameras);
    if (nonDegenerate)
      return cameras.reprojectionError(*result_, Base::measured_);
    else
      return Vector::Zero(cameras.size() * 2);
  }

  /**
   * Calculate the error of the factor.
   * This is the log-likelihood, e.g. \f$ 0.5(h(x)-z)^2/\sigma^2 \f$ in case of Gaussian.
   * In this class, we take the raw prediction error \f$ h(x)-z \f$, ask the noise model
   * to transform it to \f$ (h(x)-z)^2/\sigma^2 \f$, and then multiply by 0.5.
   */
  double totalReprojectionError(const Cameras& cameras,
      boost::optional<Point3> externalPoint = boost::none) const {
    if (externalPoint)
      result_ = TriangulationResult(*externalPoint);
    else
      result_ = triangulateSafe(cameras);

    if (result_)
      // All good, just use version in base class
      return Base::totalReprojectionError(cameras, *result_);
    else if (params_.degeneracyMode == HANDLE_INFINITY) {
      // Otherwise, manage the exceptions with rotation-only factors
      Eigen::Vector3d rayxy1;
      cameraGeometry_->backProject(Base::measured_.at(anchorIndex_), &rayxy1);
      Unit3 backprojected(cameras.at(anchorIndex_).rotation().rotate(rayxy1));
      return Base::totalReprojectionError(cameras, backprojected);
    } else
      // if we don't want to manage the exceptions we discard the factor
      return 0.0;
  }

  /**
   * Collect all cameras involved in this factor
   * @param values Values structure which must contain camera poses corresponding
   * to keys involved in this factor
   * @return vector of Values
   */
  typename Base::Cameras cameras(const Values& values) const override {
    typename Base::Cameras cameras;
    for (const Key& k : this->keys_) {
      const RiExtendedPose3& exPose3 = values.at<RiExtendedPose3>(k);
      Pose3 T_WB = gtsam::Pose3(exPose3.rotation(), exPose3.position());
      const Pose3 world_P_sensor_k = T_WB * *Base::body_P_sensor_;
      cameras.emplace_back(world_P_sensor_k, K_);
    }
    return cameras;
  }

  /// Calculate total reprojection error
  double error(const Values& values) const override {
    if (this->active(values)) {
      return totalReprojectionError(cameras(values));
    } else { // else of active flag
      return 0.0;
    }
  }

  /** return the landmark */
  TriangulationResult point() const {
    return result_;
  }

  /** COMPUTE the landmark */
  TriangulationResult point(const Values& values) const {
    Cameras cameras = this->cameras(values);
    return triangulateSafe(cameras);
  }

  /// Is result valid?
  bool isValid() const { return result_.valid(); }

  /** return the degenerate state */
  bool isDegenerate() const { return result_.degenerate(); }

  /** return the cheirality status flag */
  bool isPointBehindCamera() const { return result_.behindCamera(); }

  /** return the outlier state */
  bool isOutlier() const { return result_.outlier(); }

  /** return the farPoint state */
  bool isFarPoint() const { return result_.farPoint(); }

 private:

  /// Serialization function
  friend class boost::serialization::access;
  template<class ARCHIVE>
  void serialize(ARCHIVE & ar, const unsigned int /*version*/) {
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Base);
    ar & BOOST_SERIALIZATION_NVP(params_);
    ar & BOOST_SERIALIZATION_NVP(result_);
    ar & BOOST_SERIALIZATION_NVP(cameraPosesTriangulation_);
  }
};

/// traits
template<class CAMERA>
struct traits<RiSmartProjectionFactor<CAMERA> > : public Testable<
    RiSmartProjectionFactor<CAMERA> > {
};

} // \ namespace gtsam
