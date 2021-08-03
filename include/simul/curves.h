#ifndef SIMUL_CURVE_H_
#define SIMUL_CURVE_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include "okvis/ImuMeasurements.hpp"
#include "okvis/kinematics/Transformation.hpp"

#include <swift_vio/memory.h>

namespace simul {
typedef std::vector<okvis::ImuMeasurement,
                    Eigen::aligned_allocator<okvis::ImuMeasurement>>
    ImuMeasurementVector;

enum class SimulatedTrajectoryType {
  Sinusoid = 0,
  Torus,
  Torus2,
  Ball,
  Squircle,
  Circle,
  Dot,
  WavyCircle,
  Motionless,
};

static const std::map<std::string, SimulatedTrajectoryType> trajectoryLabelToId{
    {"Sinusoid", SimulatedTrajectoryType::Sinusoid},
    {"Torus", SimulatedTrajectoryType::Torus},
    {"Torus2", SimulatedTrajectoryType::Torus2},
    {"Ball", SimulatedTrajectoryType::Ball},
    {"Squircle", SimulatedTrajectoryType::Squircle},
    {"Circle", SimulatedTrajectoryType::Circle},
    {"Dot", SimulatedTrajectoryType::Dot},
    {"WavyCircle", SimulatedTrajectoryType::WavyCircle},
    {"Motionless", SimulatedTrajectoryType::Motionless}};


// implements the horizontal circular and vertical sinusoidal
// motion of a body frame
// world frame x right, y forward, z up, sit at the circle center
// body frame, at each point on the curve,
// x outward along the radius, y tangent, z up
// imu frame coincides with body frame
class CircularSinusoidalTrajectory {
 protected:
  const double wz;    // parameter determining the angular rate of sinusoidal
                      // motion in the vertical
  double wxy;         // angular rate in the x-y plane
  const double rz;    // the radius of the sinusoidal vertical motion
  double rxy;         // radius of the circular motion
  const double freq;  // sampling frequency
  const double interval;   // reciprocal of freq
  const double maxThetaZ;  // maximum elevation angle of the camera optical axis
  const Eigen::Vector3d gw;  // gravity in the global frame

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CircularSinusoidalTrajectory(double _maxThetaZ = 0.2 * M_PI);
  CircularSinusoidalTrajectory(
      double imuFreq, Eigen::Vector3d ginw, double _maxThetaZ = 0.2 * M_PI);
  virtual ~CircularSinusoidalTrajectory() {}

  void getTrueInertialMeasurements(const okvis::Time tStart,
                                   const okvis::Time tEnd,
                                   okvis::ImuMeasurementDeque &imuMeasurements) const;

  void getTrueInertialMeasurements(okvis::Time t, Eigen::Vector3d *gyroscope,
                                   Eigen::Vector3d *accelerometer) const;

  /**
   * @brief computeLocalLinearAcceleration  $a_{WB}^B$ by applied force.
   * @warning Not efficient. You may want to use getTrueInertialMeasurements.
   * @param time
   * @return
   */
  Eigen::Vector3d computeLocalLinearAcceleration(const okvis::Time time) const;

  /**
   * @brief computeLocalAngularVelocity  \f$\omega_{WB}^B\f$
   * @warning Not efficient. You may want to use getTrueInertialMeasurements.
   * @param time
   * @return
   */
  Eigen::Vector3d computeLocalAngularVelocity(const okvis::Time time) const;

  void
  getTruePoses(const std::vector<okvis::Time> &times,
               Eigen::AlignedVector<okvis::kinematics::Transformation> &vT_WB);

  void getTrueVelocities(const std::vector<okvis::Time> &times,
                         Eigen::AlignedVector<Eigen::Vector3d> &velocities);

  void getSampleTimes(const okvis::Time tStart, const okvis::Time tEnd,
                      std::vector<okvis::Time> &vTime);

  // compute angular rate in the global frame, $\omega_{WB}^{W}$
  virtual Eigen::Vector3d computeGlobalAngularRate(const okvis::Time time) const;

  // $a_{WB}^W$ by applied force.
  virtual Eigen::Vector3d
  computeGlobalLinearAcceleration(const okvis::Time time) const;
  // $v_{WB}^W$
  virtual Eigen::Vector3d computeGlobalLinearVelocity(const okvis::Time time) const;

  // $T_{WB}$
  virtual okvis::kinematics::Transformation computeGlobalPose(
      const okvis::Time time) const;

  // compute angular rate in the global frame, $\omega_{WB}^{W}$, with numerical
  // differentiation.
  Eigen::Vector3d computeGlobalAngularRateNumeric(const okvis::Time time) const;

  // compute $a_{WB}^W$ caused by applied force with numerical differentiation.
  Eigen::Vector3d
  computeGlobalLinearAccelerationNumeric(const okvis::Time time) const;

  // compute $v_{WB}^W$ with numerical differentiation.
  Eigen::Vector3d
  computeGlobalLinearVelocityNumeric(const okvis::Time time) const;

  Eigen::Vector3d gravityVector() const { return gw; }
};

// Yarn torus
class TorusTrajectory : public CircularSinusoidalTrajectory {
 protected:
  const double wr;    // angular rate that the radius changes
  const double xosc;  // the oscillation mag in global x direction
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  TorusTrajectory();
  TorusTrajectory(double imuFreq, Eigen::Vector3d ginw);
  virtual ~TorusTrajectory() {}
  virtual Eigen::Vector3d computeGlobalLinearAcceleration(
      const okvis::Time time) const;

  virtual Eigen::Vector3d computeGlobalLinearVelocity(const okvis::Time time) const;

  virtual okvis::kinematics::Transformation computeGlobalPose(
      const okvis::Time time) const;
};

// Yarn ball
class SphereTrajectory : public CircularSinusoidalTrajectory {
 protected:
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SphereTrajectory(double _rxy = 37.0/19, double _maxThetaZ = 0.2 * M_PI);
  SphereTrajectory(double imuFreq, Eigen::Vector3d ginw,
                   double _rxy = 37.0/19, double _maxThetaZ = 0.2 * M_PI);
  virtual ~SphereTrajectory() {}
  virtual Eigen::Vector3d computeGlobalAngularRate(const okvis::Time time) const;

  virtual Eigen::Vector3d computeGlobalLinearAcceleration(
      const okvis::Time time) const;

  virtual Eigen::Vector3d computeGlobalLinearVelocity(const okvis::Time time) const;

  virtual okvis::kinematics::Transformation computeGlobalPose(
      const okvis::Time time) const;
};

// planar motion with constant velocity magnitude
class RoundedSquare : public CircularSinusoidalTrajectory {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  RoundedSquare();
  RoundedSquare(double imuFreq, Eigen::Vector3d ginw,
                okvis::Time startEpoch = okvis::Time(0, 0), double radius = 1.0,
                double sideLength = 6.0, double velocityNorm = 1.2);

  virtual Eigen::Vector3d computeGlobalAngularRate(const okvis::Time time) const;

  virtual Eigen::Vector3d computeGlobalLinearAcceleration(
      const okvis::Time time) const;

  virtual Eigen::Vector3d computeGlobalLinearVelocity(const okvis::Time time) const;

  virtual okvis::kinematics::Transformation computeGlobalPose(
      const okvis::Time time) const;

  std::vector<double> getEndEpochs() { return endEpochs_; }

 private:

  // decide time slot, endEpochs_[j-1] < time_into_period <= endEpochs_[j]
  void decideTimeSlot(double time_into_period, size_t* j,
                      double* time_into_slot) const ;

  void initDataStructures();

  double getPeriodRemainder(const okvis::Time time) const;

  okvis::Time startEpoch_; // reference time to start the motion

  const double radius_; // radius of four arcs at corners
  const double sideLength_; // contiguous to the arc of radius
  const double velocityNorm_; // magnitude of velocity, to ensure continuity in velocity

  okvis::Duration period_; // time to travel through the rounded square
  double omega_; // angular rate
  std::vector<double> endEpochs_; // end epochs for each segments
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      centers_;  // centers for corner arcs

  // beginPoints for the 5 line segments on four sides
  // the first side has two halves because the starting point is at its middle
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      beginPoints_;
};

class WavyCircle : public CircularSinusoidalTrajectory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  WavyCircle();
  WavyCircle(double imuFreq, Eigen::Vector3d ginw, double wallRadius = 5.0,
             double trajectoryRadius = 4.0, double wallHeight = 3,
             double frequencyNumber = 10, double velocityNorm = 1.2);

  virtual Eigen::Vector3d computeGlobalAngularRate(const okvis::Time time) const;

  virtual Eigen::Vector3d computeGlobalLinearAcceleration(
      const okvis::Time time) const;

  virtual Eigen::Vector3d computeGlobalLinearVelocity(
      const okvis::Time time) const;

  virtual okvis::kinematics::Transformation computeGlobalPose(
      const okvis::Time time) const;

  template<typename T>
  Eigen::Matrix<T, 3, 3> orientation(T t) const;

  template <typename T>
  bool operator()(const T* time, T* R_WB_coeffs) const {
    Eigen::Map<Eigen::Matrix<T, 3, 3>> R_WB(R_WB_coeffs);
    T theta = time[0] * T(angularRate_);
    R_WB = orientation(theta);
    return true;
  }

  double waveHeight() const {
    return waveHeight_;
  }

  double angularRate() const {
    return angularRate_;
  }
 private:
  double wallRadius_;
  double trajectoryRadius_;
  double wallHeight_;
  double frequencyNumber_;   // wave frequency
  double waveHeightCoeff_;  // decrease the coefficient to make more point
                             // visible.

  double velocity_;
  double angularRate_;
  double waveHeight_;

  Eigen::Vector3d position(double t) const;

  double nearestDepth() const;

  double computeWaveHeight() const;
};

class Motionless : public CircularSinusoidalTrajectory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Motionless() {}
  Motionless(double imuFreq, Eigen::Vector3d ginw)
      : CircularSinusoidalTrajectory(imuFreq, ginw) {}

  virtual Eigen::Vector3d computeGlobalAngularRate(
      const okvis::Time /*time*/) const {
    return Eigen::Vector3d::Zero();
  }

  virtual Eigen::Vector3d computeGlobalLinearAcceleration(
      const okvis::Time /*time*/) const {
    return Eigen::Vector3d::Zero() - gw;
  }

  virtual Eigen::Vector3d computeGlobalLinearVelocity(
      const okvis::Time /*time*/) const {
    return Eigen::Vector3d::Zero();
  }

  virtual okvis::kinematics::Transformation computeGlobalPose(
      const okvis::Time /*time*/) const {
    return okvis::kinematics::Transformation();
  }
};

template <typename T>
Eigen::Matrix<T, 3, 3> RotX(T theta) {
  T ct = cos(theta);
  T st = sin(theta);
  Eigen::Matrix<T, 3, 3> Rx;
  Rx << T(1), T(0), T(0),
  T(0), ct, st,
  T(0), -st, ct;
  return Rx;
}

/**
 * @brief rotMat2d This is in effect RotZ(theta + 90).
 * @param theta
 * @return
 */
Eigen::Matrix2d rotMat2d(double theta);

/**
 * @brief createSimulatedTrajectory factory method.
 * @param trajectoryType
 * @param rate
 * @param gravityNorm
 * @return
 */
std::shared_ptr<CircularSinusoidalTrajectory>
createSimulatedTrajectory(SimulatedTrajectoryType trajectoryType, int rate,
                          double gravityNorm);

} // namespace simul
#endif // SIMUL_CURVE_H_
