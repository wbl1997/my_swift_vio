/**
 * @file    RiImuFactor.h
 * @brief   3-way IMU factor with right invariant errors.
 * @author  Jianzhu Huai
 */

#ifndef RIIMUFACTOR_H
#define RIIMUFACTOR_H

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/RiExtendedPose3.h>

#include <msckf/imu/ImuErrorModel.h>
#include <msckf/imu/ImuOdometry.h>

#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>

#include <Eigen/Core>

namespace gtsam {

/**
 * @brief The RiPreintegratedImuMeasurements class
 * Preintegrated IMU measurements with right invariant error formulation.
 * Notation:
 * x_i == (R_WB(t_i), v_WB(t_i), p_WB(t_i)), b_i = (ba_i, bg_i)
 * x_i = exp(\xi_i) \hat{x}_i
 * i <= j
 * z_{i:j} are IMU measurements from t_i to t_j.
 * x_{j|i} = f(x_i, z_{i:j})
 */
class RiPreintegratedImuMeasurements {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  RiPreintegratedImuMeasurements() {}

  RiPreintegratedImuMeasurements(
      const okvis::ImuMeasurementDeque& imuMeasurements,
      const okvis::ImuParameters& imuParams, const okvis::Time startTime,
      const okvis::Time finishTime)
      : imuMeasurements_(imuMeasurements),
        imuParams_(imuParams),
        ti_(startTime),
        tj_(finishTime), covariance_(Eigen::Matrix<double, 15, 15>::Zero()),
        jacobian_(Eigen::Matrix<double, 15, 15>::Identity()) {}

  RiPreintegratedImuMeasurements(const RiPreintegratedImuMeasurements& other)
      : imuMeasurements_(other.imuMeasurements()),
        imuParams_(other.imuParameters()),
        ti_(other.ti()),
        tj_(other.tj()),
        RiLin_(other.RotiLin()),
        deltaXij_(other.deltaXij_),
        covariance_(other.covariance_),
        jacobian_(other.jacobian_) {}

  RiPreintegratedImuMeasurements& operator=(
      const RiPreintegratedImuMeasurements& other) {
    if (this != &other) {
      imuMeasurements_ = other.imuMeasurements();
      imuParams_ = other.imuParameters();
      ti_ = other.ti();
      tj_ = other.tj();

      RiLin_ = other.RotiLin();
      deltaXij_ = other.deltaXij_;
      covariance_ = other.covariance_;
      jacobian_ = other.jacobian_;
    }
    return *this;
  }

  bool needToRedoPreintegration(const imuBias::ConstantBias& bias_i) const {
    Eigen::Vector3d deltaBg = bias_i.gyroscope() - imuParams_.g0;
    Eigen::Vector3d deltaBa = bias_i.accelerometer() - imuParams_.a0;
    return deltaBg.norm() * (tj_ - ti_).toSec() > kRelinThresholdGyro ||
           deltaBa.norm() * (tj_ - ti_).toSec() > kRelinThresholdAccelerometer;
  }

  /**
   * @brief redoPreintegration
   * @param x_i
   * @param bias_i
   * @param x_j
   */
  void redoPreintegration(const RiExtendedPose3& x_i, const imuBias::ConstantBias& bias_i);

  /**
   * @brief biasLin bias linearization point for the preintegrated IMU measurements.
   * @return
   */
  imuBias::ConstantBias biasLin() const {
    return imuBias::ConstantBias(imuParams_.a0, imuParams_.g0);
  }

  /**
   * @brief RotiLin linearization point for Rot at t_i.
   * @return
   */
  Rot3 RotiLin() const {
    return RiLin_;
  }

  const okvis::ImuMeasurementDeque& imuMeasurements() const {
    return imuMeasurements_;
  }

  const okvis::ImuParameters& imuParameters() const {
    return imuParams_;
  }

  okvis::Time ti() const {
    return ti_;
  }

  okvis::Time tj() const {
    return tj_;
  }

  double dt() const {
    return (tj_ - ti_).toSec();
  }

  double gravity() const {
    return imuParams_.g;
  }

  /**
   * @brief transformByGravity the transform due to gravity, eq 26 Brossard Associating Uncertainty ...
   * @return
   */
  RiExtendedPose3 transformByGravity() const {
    Eigen::Vector3d gW(0, 0, -imuParams_.g);
    double tij = (tj_ - ti_).toSec();
    return RiExtendedPose3(Rot3(), gW * tij, gW * tij * tij * 0.5);
  }

  RiExtendedPose3 predict(const RiExtendedPose3& x_i) const {
    Eigen::Vector3d gW(0, 0, -imuParams_.g);
    double tij = (tj_ - ti_).toSec();
    return RiExtendedPose3(
        x_i.rotation() * deltaXij_.rotation(),
        x_i.velocity() + x_i.rotation() * deltaXij_.velocity() + gW * tij,
        x_i.position() + x_i.rotation() * deltaXij_.position() +
            x_i.velocity() * tij + gW * tij * tij * 0.5);
  }

  GTSAM_EXPORT
  friend std::ostream& operator<<(std::ostream& os,
                                  const RiPreintegratedImuMeasurements& pim) {
    os << "dt:[" << pim.dt() << "]'\n";
    os << "Rij:[" << pim.deltaXij_.rotation().matrix() << "]'\n";
    os << "vij:[" << pim.deltaXij_.velocity().transpose() << "]'\n";
    os << "pij:[" << pim.deltaXij_.position().transpose() << "]'\n";
    os << "Ri_lin:[" << pim.RotiLin().matrix() << "]'\n";
    os << "ba_lin:[" << pim.biasLin().accelerometer().transpose() << "]'\n";
    os << "bg_lin:[" << pim.biasLin().gyroscope().transpose() << "]'\n";
    os << "gravity:[" << pim.gravity() << "]'\n";
    return os;
  }

  bool equals(const RiPreintegratedImuMeasurements& rhs, double tol = 1e-9) const {
    return
    gtsam::equal(dt(), rhs.dt(), tol) &&
    gtsam::equal(deltaXij_.rotation(), rhs.deltaXij_.rotation(), tol) &&
    gtsam::equal(deltaXij_.velocity(), rhs.deltaXij_.velocity(), tol) &&
    gtsam::equal(deltaXij_.position(), rhs.deltaXij_.position(), tol) &&
    gtsam::equal(RiLin_, rhs.RotiLin(), tol) &&
    gtsam::equal(biasLin(), rhs.biasLin(), tol) &&
    gtsam::equal(gravity(), rhs.gravity(), tol);
  }

 private:
  okvis::ImuMeasurementDeque imuMeasurements_;
  okvis::ImuParameters imuParams_; // This contains the linearization points for gyro and accelerometer biases.
  okvis::Time ti_;
  okvis::Time tj_;

  Rot3 RiLin_; // R_i lastly used in redoPreintegration.

public:
  RiExtendedPose3 deltaXij_; //! R_ij, v_ij, p_ij
  Eigen::Matrix<double, 15, 15> covariance_; // covariance of \xi_{j|i}.
  Eigen::Matrix<double, 15, 15> jacobian_; // jacobian of d(\xi_{j|i}) / d(\xi_i)

  static const double kRelinThresholdGyro;
  static const double kRelinThresholdAccelerometer;
};

class RiImuFactor : public NoiseModelFactor3<RiExtendedPose3, RiExtendedPose3,
                                             imuBias::ConstantBias> {
 private:
  typedef NoiseModelFactor3<RiExtendedPose3, RiExtendedPose3,
                            imuBias::ConstantBias>
      Base;

  mutable RiPreintegratedImuMeasurements pim_;
  mutable bool redo_;
  mutable int redoCounter_;

  mutable Eigen::Matrix<double, 9, 6> de_db_i_;
  mutable bool biasJacobianReady_; // is the bias Jacobian ready?

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  RiImuFactor() {}

  /**
   * @brief RiImuFactor e_{ij}. Its covariance is set to the covariance of pim
   * because X_j is usually initialized with X_{j|i}, and hence
   * de_{ij} / d\xi_{j|i} = - I_9.
   * @param xi
   * @param xj
   * @param bi
   * @param pim The preintegrated imu measurements for which redoPreintegration
   * should have been called at least once.
   */
  RiImuFactor(Key xi, Key xj, Key bi, const RiPreintegratedImuMeasurements& pim)
      : NoiseModelFactor3<RiExtendedPose3, RiExtendedPose3,
                          imuBias::ConstantBias>(
            noiseModel::Gaussian::Covariance(
                pim.covariance_.topLeftCorner<9, 9>()),
            xi, xj, bi),
        pim_(pim),
        redo_(false),
        redoCounter_(0),
        biasJacobianReady_(false) {}

  RiImuFactor(const RiImuFactor& other)
      : NoiseModelFactor3<RiExtendedPose3, RiExtendedPose3,
                          imuBias::ConstantBias>(
            noiseModel::Gaussian::Covariance(other.pim().covariance_),
            other.key1(), other.key2(), other.key3()),
        pim_(other.pim()),
        redo_(other.redo()),
        redoCounter_(other.redoCounter()),
        de_db_i_(other.de_db_i_),
        biasJacobianReady_(other.biasJacobianReady_) {}

  RiImuFactor& operator=(const RiImuFactor& rhs) {
    if (this != &rhs) {
      Base::operator=(rhs);
      pim_ = rhs.pim();
      redo_ = rhs.redo();
      redoCounter_ = rhs.redoCounter();
      de_db_i_ = rhs.de_db_i();
      biasJacobianReady_ = rhs.isBiasJacobianReady();
    }
    return *this;
  }

  virtual ~RiImuFactor(){};

  const RiPreintegratedImuMeasurements& pim() const {
    return pim_;
  }

  bool redo() const { return redo_; }

  int redoCounter() const { return redoCounter_; }

  const Eigen::Matrix<double, 9, 6>& de_db_i() const { return de_db_i_; }

  bool isBiasJacobianReady() const { return biasJacobianReady_; }

  // Error function. Given the current states, calculate the measurement error/residual
  gtsam::Vector evaluateError(
      const RiExtendedPose3& state_i, const RiExtendedPose3& state_j,
      const imuBias::ConstantBias& bias_i,
      boost::optional<Matrix&> H1 = boost::none,
      boost::optional<Matrix&> H2 = boost::none,
      boost::optional<Matrix&> H3 = boost::none) const override;

  /**
   * @brief evaluateErrorCheck evaluate error with gtsam preintegrated imu
   * measurements as a check.
   * @param state_i
   * @param state_j
   * @param bias_i
   * @return
   */
  gtsam::Vector evaluateErrorCheck(
      const RiExtendedPose3& state_i, const RiExtendedPose3& state_j,
      const imuBias::ConstantBias& bias_i) const;

  GTSAM_EXPORT
  friend std::ostream& operator<<(std::ostream& os, const RiImuFactor& factor) {
    os << factor.pim_ << " " << factor.redo_ << " " << factor.redoCounter_;
    return os;
  }

  void print(const std::string& s,
             const KeyFormatter& keyFormatter = DefaultKeyFormatter) const override {
    std::cout << s << "RiImuFactor(" << keyFormatter(this->key1()) << ","
              << keyFormatter(this->key2()) << ")\n";
    std::cout << "measured:\n" << *this << "\n";
    this->noiseModel_->print("  noise model: ");
  }

  bool equals(const NonlinearFactor& expected, double tol = 1e-9) const override {
    const RiImuFactor* e = dynamic_cast<const RiImuFactor*>(&expected);
    if (e == NULL) return false;
    return NoiseModelFactor3<RiExtendedPose3, RiExtendedPose3,
                             imuBias::ConstantBias>::equals(*e, tol) &&
           pim_.equals(e->pim(), tol);
  }
};  // RiImuFactor

}  // namespace gtsam

#endif  // RIIMUFACTOR_H
