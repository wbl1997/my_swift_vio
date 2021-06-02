#include "simul/curves.h"

#include <ceres/internal/autodiff.h>

#include <okvis/kinematics/sophus_operators.hpp>

namespace simul {
CircularSinusoidalTrajectory::CircularSinusoidalTrajectory(double _maxThetaZ)
    : wz(17 * M_PI / 41),
      wxy(7 * M_PI / 37),
      rz(1),
      rxy(61.0 / 19),
      freq(100),
      interval(1 / freq),
      maxThetaZ(_maxThetaZ),
      gw(0, 0, -9.8) {}

CircularSinusoidalTrajectory::CircularSinusoidalTrajectory(double imuFreq,
                                                           Eigen::Vector3d ginw,
                                                           double _maxThetaZ)
    : wz(17 * M_PI / 41),
      wxy(7 * M_PI / 37),
      rz(1),
      rxy(61.0 / 19),
      freq(imuFreq),
      interval(1 / freq),
      maxThetaZ(_maxThetaZ),
      gw(ginw) {}

void CircularSinusoidalTrajectory::getTrueInertialMeasurements(
    const okvis::Time tStart, const okvis::Time tEnd,
    okvis::ImuMeasurementDeque& imuMeasurements) const {
  okvis::Time time = tStart;
  ImuMeasurementVector imuMeas;
  imuMeas.reserve((int)((tEnd - tStart).toSec() * freq + 1));
  for (; time < tEnd; time += okvis::Duration(interval)) {
    okvis::ImuMeasurement meas;
    meas.timeStamp = time;
    okvis::kinematics::Transformation T_WB = computeGlobalPose(time);
    meas.measurement.gyroscopes =
        T_WB.C().transpose() * computeGlobalAngularRate(time);
    meas.measurement.accelerometers =
        T_WB.C().transpose() * computeGlobalLinearAcceleration(time);
    imuMeas.push_back(meas);
  }
  imuMeasurements = okvis::ImuMeasurementDeque(imuMeas.begin(), imuMeas.end());
}

void CircularSinusoidalTrajectory::getTrueInertialMeasurements(
    okvis::Time time, Eigen::Vector3d *gyroscope,
    Eigen::Vector3d *accelerometer) const {
  okvis::kinematics::Transformation T_WB = computeGlobalPose(time);
  *gyroscope = T_WB.C().transpose() * computeGlobalAngularRate(time);
  *accelerometer = T_WB.C().transpose() * computeGlobalLinearAcceleration(time);
}

Eigen::Vector3d CircularSinusoidalTrajectory::computeLocalLinearAcceleration(
    const okvis::Time time) const {
  okvis::kinematics::Transformation T_WB = computeGlobalPose(time);
  return T_WB.C().transpose() * computeGlobalLinearAcceleration(time);
}

Eigen::Vector3d CircularSinusoidalTrajectory::computeLocalAngularVelocity(
    const okvis::Time time) const {
  okvis::kinematics::Transformation T_WB = computeGlobalPose(time);
  return T_WB.C().transpose() * computeGlobalAngularRate(time);
}

void CircularSinusoidalTrajectory::getTruePoses(
    const std::vector<okvis::Time> &times,
    Eigen::AlignedVector<okvis::kinematics::Transformation> &vT_WB) {
  vT_WB.clear();
  vT_WB.reserve(times.size());
  for (auto time : times) {
    vT_WB.push_back(computeGlobalPose(time));
  }
}

void CircularSinusoidalTrajectory::getTrueVelocities(const std::vector<okvis::Time> &times,
                                                     Eigen::AlignedVector<Eigen::Vector3d> &velocities) {
  velocities.clear();
  velocities.reserve(times.size());
  for (auto time : times) {
    velocities.push_back(computeGlobalLinearVelocity(time));
  }
}

void CircularSinusoidalTrajectory::getSampleTimes(
    const okvis::Time tStart, const okvis::Time tEnd,
    std::vector<okvis::Time>& vTime) {
  okvis::Time time = tStart;
  vTime.clear();
  vTime.reserve((int)((tEnd - tStart).toSec() * freq + 1));
  for (; time < tEnd; time += okvis::Duration(interval)) vTime.push_back(time);
}

// compute angular rate in the global frame
Eigen::Vector3d CircularSinusoidalTrajectory::computeGlobalAngularRate(
    const okvis::Time time) const {
  double dTime = time.toSec();
  double thetaZDot = maxThetaZ * cos(wz * dTime) * wz;

  double thetaXY = wxy * dTime;
  double wgx = -thetaZDot * sin(thetaXY);
  double wgy = thetaZDot * cos(thetaXY);
  double wgz = wxy;
  return Eigen::Vector3d(wgx, wgy, wgz);
}

Eigen::Vector3d CircularSinusoidalTrajectory::computeGlobalLinearAcceleration(
    const okvis::Time time) const {
  double dTime = time.toSec();
  double swzt = sin(wz * dTime);
  double cwzt = cos(wz * dTime);
  double thetaZ = maxThetaZ * swzt;
  double thetaZDot = maxThetaZ * cwzt * wz;
  double thetaZDDot = -maxThetaZ * swzt * wz * wz;

  double sThetaZ = sin(thetaZ);
  double cThetaZ = cos(thetaZ);

  double r = rxy - rz + rz * cThetaZ;
  double rDot = -rz * sThetaZ * thetaZDot;
  double rDDot = -rz * (cThetaZ * thetaZDot * thetaZDot + thetaZDDot * sThetaZ);

  double thetaXY = wxy * dTime;
  double sThetaXY = sin(thetaXY);
  double cThetaXY = cos(thetaXY);

  double xDDot =
      rDDot * cThetaXY - 2 * rDot * sThetaXY * wxy - r * cThetaXY * wxy * wxy;
  double yDDot =
      rDDot * sThetaXY + 2 * rDot * cThetaXY * wxy - r * sThetaXY * wxy * wxy;
  double zDDot = rz * (-sThetaZ * thetaZDot * thetaZDot + cThetaZ * thetaZDDot);
  return Eigen::Vector3d(xDDot, yDDot, zDDot) - gw;
}

Eigen::Vector3d CircularSinusoidalTrajectory::computeGlobalLinearVelocity(
    const okvis::Time time) const {
  double dTime = time.toSec();
  double swzt = sin(wz * dTime);
  double cwzt = cos(wz * dTime);
  double thetaZ = maxThetaZ * swzt;
  double thetaZDot = maxThetaZ * cwzt * wz;

  double sThetaZ = sin(thetaZ);
  double cThetaZ = cos(thetaZ);

  double r = rxy - rz + rz * cThetaZ;
  double rDot = -rz * sThetaZ * thetaZDot;

  double thetaXY = wxy * dTime;
  double sThetaXY = sin(thetaXY);
  double cThetaXY = cos(thetaXY);

  double xDot = rDot * cThetaXY - r * sThetaXY * wxy;
  double yDot = rDot * sThetaXY + r * cThetaXY * wxy;
  double zDot = rz * cThetaZ * thetaZDot;

  return Eigen::Vector3d(xDot, yDot, zDot);
}

okvis::kinematics::Transformation
CircularSinusoidalTrajectory::computeGlobalPose(const okvis::Time time) const {
  double dTime = time.toSec();
  double swzt = sin(wz * dTime);
  //  double cwzt = cos(wz * dTime);
  double thetaZ = maxThetaZ * swzt;

  double sThetaZ = sin(thetaZ);
  double cThetaZ = cos(thetaZ);

  double r = rxy - rz + rz * cThetaZ;

  double thetaXY = wxy * dTime;
  double sThetaXY = sin(thetaXY);
  double cThetaXY = cos(thetaXY);

  double x = r * cThetaXY;
  double y = r * sThetaXY;
  double z = rz * sThetaZ;
  return okvis::kinematics::Transformation(
      Eigen::Vector3d(x, y, z),
      Eigen::AngleAxisd(thetaXY, Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(thetaZ, Eigen::Vector3d::UnitY()));
}

Eigen::Vector3d CircularSinusoidalTrajectory::computeGlobalAngularRateNumeric(
    const okvis::Time time) const {
  double h = 1e-6;
  okvis::kinematics::Transformation T_WBdelta =
      computeGlobalPose(time + okvis::Duration(h));
  okvis::kinematics::Transformation T_WB = computeGlobalPose(time);
  okvis::kinematics::Transformation T_WBmdelta =
      computeGlobalPose(time - okvis::Duration(h));
  Eigen::Matrix3d Rdelta = (T_WBdelta.C() - T_WBmdelta.C()) / (2 * h);
  Eigen::Matrix3d OmegaW = Rdelta * T_WB.C().transpose();
  return okvis::kinematics::vee(OmegaW);
}

Eigen::Vector3d
CircularSinusoidalTrajectory::computeGlobalLinearAccelerationNumeric(
    const okvis::Time time) const {
  double h = 1e-6;
  Eigen::Vector3d v_WBdelta =
      computeGlobalLinearVelocity(time + okvis::Duration(h));
  Eigen::Vector3d v_WBmdelta =
      computeGlobalLinearVelocity(time - okvis::Duration(h));
  return (v_WBdelta - v_WBmdelta) / (2 * h) - gw;
}

Eigen::Vector3d
CircularSinusoidalTrajectory::computeGlobalLinearVelocityNumeric(
    const okvis::Time time) const {
  double h = 1e-6;
  okvis::kinematics::Transformation T_WBdelta =
      computeGlobalPose(time + okvis::Duration(h));
  okvis::kinematics::Transformation T_WBmdelta =
      computeGlobalPose(time - okvis::Duration(h));
  return (T_WBdelta.r() - T_WBmdelta.r()) / (2 * h);
}

TorusTrajectory::TorusTrajectory()
    : CircularSinusoidalTrajectory(), wr(19 * M_PI / 137), xosc(rxy - rz) {}

TorusTrajectory::TorusTrajectory(
    double imuFreq, Eigen::Vector3d ginw)
    : CircularSinusoidalTrajectory(imuFreq, ginw),
      wr(19 * M_PI / 137),
      xosc(rxy - rz) {}

Eigen::Vector3d TorusTrajectory::computeGlobalLinearAcceleration(
    const okvis::Time time) const {
  double dTime = time.toSec();
  double swzt = sin(wz * dTime);
  double cwzt = cos(wz * dTime);
  double thetaZ = maxThetaZ * swzt;
  double thetaZDot = maxThetaZ * cwzt * wz;
  double thetaZDDot = -maxThetaZ * swzt * wz * wz;

  double sThetaZ = sin(thetaZ);
  double cThetaZ = cos(thetaZ);

  double thetar = wr * dTime;
  double sthetar = sin(thetar);
  double cthetar = cos(thetar);
  double r = xosc * cthetar + rz * cThetaZ;
  double rDot = -xosc * sthetar * wr - rz * sThetaZ * thetaZDot;
  double rDDot = -xosc * cthetar * wr * wr -
                 rz * (cThetaZ * thetaZDot * thetaZDot + thetaZDDot * sThetaZ);

  double thetaXY = wxy * dTime;
  double sThetaXY = sin(thetaXY);
  double cThetaXY = cos(thetaXY);

  double xDDot =
      rDDot * cThetaXY - 2 * rDot * sThetaXY * wxy - r * cThetaXY * wxy * wxy;
  double yDDot =
      rDDot * sThetaXY + 2 * rDot * cThetaXY * wxy - r * sThetaXY * wxy * wxy;
  double zDDot = rz * (-sThetaZ * thetaZDot * thetaZDot + cThetaZ * thetaZDDot);
  return Eigen::Vector3d(xDDot, yDDot, zDDot) - gw;
}

Eigen::Vector3d TorusTrajectory::computeGlobalLinearVelocity(
    const okvis::Time time) const {
  double dTime = time.toSec();
  double swzt = sin(wz * dTime);
  double cwzt = cos(wz * dTime);
  double thetaZ = maxThetaZ * swzt;
  double thetaZDot = maxThetaZ * cwzt * wz;

  double sThetaZ = sin(thetaZ);
  double cThetaZ = cos(thetaZ);

  double thetar = wr * dTime;
  double sthetar = sin(thetar);
  double cthetar = cos(thetar);
  double r = xosc * cthetar + rz * cThetaZ;
  double rDot = -xosc * sthetar * wr - rz * sThetaZ * thetaZDot;

  double thetaXY = wxy * dTime;
  double sThetaXY = sin(thetaXY);
  double cThetaXY = cos(thetaXY);

  double xDot = rDot * cThetaXY - r * sThetaXY * wxy;
  double yDot = rDot * sThetaXY + r * cThetaXY * wxy;
  double zDot = rz * cThetaZ * thetaZDot;

  return Eigen::Vector3d(xDot, yDot, zDot);
}

okvis::kinematics::Transformation
TorusTrajectory::computeGlobalPose(const okvis::Time time) const {
  double dTime = time.toSec();
  double swzt = sin(wz * dTime);
  double thetaZ = maxThetaZ * swzt;

  double sThetaZ = sin(thetaZ);
  double cThetaZ = cos(thetaZ);

  double thetar = wr * dTime;
  double r = xosc * cos(thetar) + rz * cThetaZ;

  double thetaXY = wxy * dTime;
  double sThetaXY = sin(thetaXY);
  double cThetaXY = cos(thetaXY);

  double x = r * cThetaXY;
  double y = r * sThetaXY;
  double z = rz * sThetaZ;
  return okvis::kinematics::Transformation(
      Eigen::Vector3d(x, y, z),
      Eigen::AngleAxisd(thetaXY, Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(thetaZ, Eigen::Vector3d::UnitY()));
}

SphereTrajectory::SphereTrajectory(double _rxy, double _maxThetaZ)
    : CircularSinusoidalTrajectory(_maxThetaZ) {
  rxy = _rxy;
}

SphereTrajectory::SphereTrajectory(
    double imuFreq, Eigen::Vector3d ginw, double _rxy, double _maxThetaZ)
    : CircularSinusoidalTrajectory(imuFreq, ginw, _maxThetaZ) {
  rxy = _rxy;
}

Eigen::Vector3d SphereTrajectory::computeGlobalAngularRate(
    const okvis::Time time) const {
  double dTime = time.toSec();
  double thetaZDot = maxThetaZ * cos(wz * dTime) * wz;

  double thetaXY = M_PI * (1.0 - cos(wxy * dTime));
  double wgx = -thetaZDot * sin(thetaXY);
  double wgy = thetaZDot * cos(thetaXY);
  double wgz = M_PI * sin(wxy * dTime) * wxy;
  return Eigen::Vector3d(wgx, wgy, wgz);
}

Eigen::Vector3d SphereTrajectory::computeGlobalLinearAcceleration(
    const okvis::Time time) const {
  double dTime = time.toSec();
  double swzt = sin(wz * dTime);
  double cwzt = cos(wz * dTime);
  double thetaZ = maxThetaZ * swzt;
  double thetaZDot = maxThetaZ * cwzt * wz;
  double thetaZDDot = -maxThetaZ * swzt * wz * wz;

  double sThetaZ = sin(thetaZ);
  double cThetaZ = cos(thetaZ);

  double r = rxy - rz + rz * cThetaZ;
  double rDot = -rz * sThetaZ * thetaZDot;
  double rDDot = -rz * (cThetaZ * thetaZDot * thetaZDot + thetaZDDot * sThetaZ);

  double thetaXY = M_PI * (1.0 - cos(wxy * dTime));
  double sThetaXY = sin(thetaXY);
  double cThetaXY = cos(thetaXY);
  double thetaXYDot = M_PI * sin(wxy * dTime) * wxy;
  double thetaXYDDot = M_PI * cos(wxy * dTime) * wxy * wxy;

  double xDDot = rDDot * cThetaXY - 2 * rDot * sThetaXY * thetaXYDot -
                 r * cThetaXY * thetaXYDot * thetaXYDot -
                 r * sThetaXY * thetaXYDDot;
  double yDDot = rDDot * sThetaXY + 2 * rDot * cThetaXY * thetaXYDot -
                 r * sThetaXY * thetaXYDot * thetaXYDot +
                 r * cThetaXY * thetaXYDDot;
  double zDDot = rz * (-sThetaZ * thetaZDot * thetaZDot + cThetaZ * thetaZDDot);
  return Eigen::Vector3d(xDDot, yDDot, zDDot) - gw;
}

Eigen::Vector3d SphereTrajectory::computeGlobalLinearVelocity(
    const okvis::Time time) const {
  double dTime = time.toSec();
  double swzt = sin(wz * dTime);
  double cwzt = cos(wz * dTime);
  double thetaZ = maxThetaZ * swzt;
  double thetaZDot = maxThetaZ * cwzt * wz;

  double sThetaZ = sin(thetaZ);
  double cThetaZ = cos(thetaZ);

  double r = rxy - rz + rz * cThetaZ;
  double rDot = -rz * sThetaZ * thetaZDot;

  double thetaXY = M_PI * (1.0 - cos(wxy * dTime));
  double sThetaXY = sin(thetaXY);
  double cThetaXY = cos(thetaXY);
  double thetaXYDot = M_PI * sin(wxy * dTime) * wxy;

  double xDot = rDot * cThetaXY - r * sThetaXY * thetaXYDot;
  double yDot = rDot * sThetaXY + r * cThetaXY * thetaXYDot;
  double zDot = rz * cThetaZ * thetaZDot;

  return Eigen::Vector3d(xDot, yDot, zDot);
}

okvis::kinematics::Transformation
SphereTrajectory::computeGlobalPose(const okvis::Time time) const {
  double dTime = time.toSec();
  double swzt = sin(wz * dTime);
  double thetaZ = maxThetaZ * swzt;

  double sThetaZ = sin(thetaZ);
  double cThetaZ = cos(thetaZ);

  double r = rxy - rz + rz * cThetaZ;

  double thetaXY = M_PI * (1.0 - cos(wxy * dTime));
  double sThetaXY = sin(thetaXY);
  double cThetaXY = cos(thetaXY);

  double x = r * cThetaXY;
  double y = r * sThetaXY;
  double z = rz * sThetaZ;
  return okvis::kinematics::Transformation(
      Eigen::Vector3d(x, y, z),
      Eigen::AngleAxisd(thetaXY, Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(thetaZ, Eigen::Vector3d::UnitY()));
}

RoundedSquare::RoundedSquare()
    : CircularSinusoidalTrajectory(),
      radius_(1.0),
      sideLength_(6.0),
      velocityNorm_(1.2) {
  initDataStructures();
}

void RoundedSquare::initDataStructures() {
  CHECK_GT(radius_, 1e-8);
  period_.fromSec((2 * M_PI * radius_ + 4 * sideLength_) / velocityNorm_);
  omega_ = velocityNorm_ / radius_;
  // compute endEpochs
  // anticlockwise, start from (sideLength / 2 + r, 0)
  std::vector<double> segments{sideLength_ * 0.5,    radius_ * 0.5 * M_PI,
                               sideLength_,          radius_ * 0.5 * M_PI,
                               sideLength_,          radius_ * 0.5 * M_PI,
                               sideLength_,
                               radius_ * 0.5 * M_PI,  // fourth quadrant
                               sideLength_ * 0.5};

  double distance = 0;
  for (size_t j = 0; j < segments.size(); ++j) {
    distance += segments[j];
    endEpochs_.emplace_back(distance / velocityNorm_);
  }
  CHECK_NEAR(endEpochs_.back(), period_.toSec(), 1e-6);
  endEpochs_[endEpochs_.size() - 1] = period_.toSec();

  double half_d = sideLength_ * 0.5;

  centers_ =
      std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>{
          {half_d, half_d},
          {-half_d, half_d},
          {-half_d, -half_d},
          {half_d, -half_d}};

  beginPoints_ =
      std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>{
          {half_d + radius_, 0},
          {half_d, half_d + radius_},
          {-half_d - radius_, half_d},
          {-half_d, -half_d - radius_},
          {half_d + radius_, -half_d}};
}

RoundedSquare::RoundedSquare(double imuFreq, Eigen::Vector3d ginw,
                             okvis::Time startEpoch, double radius,
                             double sideLength, double velocityNorm)
    : CircularSinusoidalTrajectory(imuFreq, ginw),
      startEpoch_(startEpoch),
      radius_(radius),
      sideLength_(sideLength),
      velocityNorm_(velocityNorm) {
  initDataStructures();
}

okvis::kinematics::Transformation RoundedSquare::computeGlobalPose(
    const okvis::Time time) const {
  double remainder = getPeriodRemainder(time);
  size_t j;
  double delta_t;
  decideTimeSlot(remainder, &j, &delta_t);
  int half_j = j / 2;

  Eigen::Vector2d xny;
  double theta;         // body frame (FLU) in the world frame
  double motion_theta;  // radians elapsed by the circular motion

  Eigen::Matrix2d R_WB;
  Eigen::Vector2d v_B(velocityNorm_, 0);
  Eigen::Vector2d cs_theta;
  switch (j % 2) {
    case 0:
      theta = (half_j + 1) * M_PI * 0.5;
      R_WB = rotMat2d(theta);
      xny = beginPoints_[half_j] + R_WB * v_B * delta_t;
      break;
    case 1:
      motion_theta = M_PI * 0.5 * half_j + delta_t * omega_;
      cs_theta[0] = std::cos(motion_theta);
      cs_theta[1] = std::sin(motion_theta);
      xny = centers_[half_j] + radius_ * cs_theta;

      theta = motion_theta + M_PI * 0.5;
      R_WB = rotMat2d(theta);
      break;
    default:
      break;
  }
  Eigen::Matrix3d R_WB3d = Eigen::Matrix3d::Identity();
  R_WB3d.topLeftCorner<2, 2>() = R_WB;
  Eigen::Vector3d t_WB;
  t_WB << xny, 0;
  Eigen::Quaterniond q_WB(R_WB3d);
  if (q_WB.w() < 0) {
    q_WB.coeffs() *= -1;
  }
  return okvis::kinematics::Transformation(t_WB, q_WB);
}

Eigen::Vector3d RoundedSquare::computeGlobalAngularRate(
    const okvis::Time time) const {
  double remainder = getPeriodRemainder(time);
  size_t j;
  double delta_t;
  decideTimeSlot(remainder, &j, &delta_t);
  Eigen::Vector3d omega_W = Eigen::Vector3d::Zero();
  switch (j % 2) {
    case 1:
      omega_W[2] = omega_;
      break;
    case 0:
    default:
      break;
  }
  return omega_W;
}

Eigen::Vector3d RoundedSquare::computeGlobalLinearAcceleration(
    const okvis::Time time) const {
  double remainder = getPeriodRemainder(time);
  size_t j;
  double delta_t;
  decideTimeSlot(remainder, &j, &delta_t);
  int half_j = j / 2;

  double motion_theta;
  Eigen::Vector3d a_W = Eigen::Vector3d::Zero();
  Eigen::Vector2d cs_theta;

  switch (j % 2) {
    case 0:
      break;
    case 1:
      motion_theta = M_PI * 0.5 * half_j + delta_t * omega_;
      cs_theta[0] = std::cos(motion_theta);
      cs_theta[1] = std::sin(motion_theta);
      a_W.head<2>() = -radius_ * cs_theta * omega_ * omega_;
      break;
    default:
      break;
  }
  return a_W - gw;
}

Eigen::Vector3d RoundedSquare::computeGlobalLinearVelocity(
    const okvis::Time time) const {
  double remainder = getPeriodRemainder(time);
  size_t j;
  double delta_t;
  decideTimeSlot(remainder, &j, &delta_t);
  int half_j = j / 2;

  double theta;
  double motion_theta;
  Eigen::Vector3d v_W = Eigen::Vector3d::Zero();
  Eigen::Vector2d msc_theta;

  Eigen::Matrix2d R_WB;
  Eigen::Vector2d v_B(velocityNorm_, 0);

  switch (j % 2) {
    case 0:
      theta = (half_j + 1) * M_PI * 0.5;
      R_WB = rotMat2d(theta);
      v_W.head<2>() = R_WB * v_B;
      break;
    case 1:
      motion_theta = M_PI * 0.5 * half_j + delta_t * omega_;
      msc_theta[0] = -std::sin(motion_theta);
      msc_theta[1] = std::cos(motion_theta);
      v_W.head<2>() = radius_ * msc_theta * omega_;
      break;
    default:
      break;
  }
  return v_W;
}

Eigen::Matrix2d rotMat2d(double theta) {
  Eigen::Matrix2d mat;
  double ct = std::cos(theta);
  double st = std::sin(theta);
  mat << ct, -st, st, ct;
  return mat;
}

// decide time slot, endEpochs_[j-1] < time_into_period <= endEpochs_[j]
void RoundedSquare::decideTimeSlot(double time_into_period, size_t* j,
                                   double* time_into_slot) const {
  *j = 0;
  for (; *j < endEpochs_.size(); ++(*j)) {
    if (time_into_period < endEpochs_[*j]) break;
  }

  *time_into_slot =
      *j > 0 ? time_into_period - endEpochs_[(*j) - 1] : time_into_period;
}

double RoundedSquare::getPeriodRemainder(const okvis::Time time) const {
  CHECK_GE(time, startEpoch_)
      << "Query time should be greater than start epoch!";
  okvis::Duration elapsed = time - startEpoch_;
  return std::fmod(elapsed.toSec(), period_.toSec());
}

WavyCircle::WavyCircle() {}

WavyCircle::WavyCircle(double imuFreq, Eigen::Vector3d ginw, double wallRadius,
                       double trajectoryRadius, double wallHeight,
                       double frequencyNumber, double velocityNorm)
    : CircularSinusoidalTrajectory(imuFreq, ginw),
      wallRadius_(wallRadius),
      trajectoryRadius_(trajectoryRadius),
      wallHeight_(wallHeight),
      frequencyNumber_(frequencyNumber),
      waveHeightCoeff_(0.9),
      velocity_(velocityNorm),
      angularRate_(velocityNorm / trajectoryRadius) {
  waveHeight_ = computeWaveHeight();
}

Eigen::Vector3d WavyCircle::computeGlobalAngularRate(const okvis::Time time) const {
  double timeVal = time.toSec();
  Eigen::Matrix3d R_WB = orientation(timeVal * angularRate_);
  Eigen::Matrix<double, 9, 1> residual;
  const double* const parameters[] = {&timeVal};
  Eigen::Matrix<double, 9, 1> j;
  double * jacobians[] = {j.data()};
  ::ceres::internal::AutoDifferentiate<::ceres::internal::StaticParameterDims<1>>(
          *this, parameters, 9, residual.data(), jacobians);
  Eigen::Matrix3d Rprime = Eigen::Map<Eigen::Matrix3d>(j.data());
  Eigen::Matrix3d OmegaW = Rprime * R_WB.transpose();
  return okvis::kinematics::vee(OmegaW);
}

Eigen::Vector3d WavyCircle::computeGlobalLinearAcceleration(
    const okvis::Time time) const {
  double t = time.toSec() * angularRate_;
  Eigen::Vector3d a_WB_W;
  a_WB_W << -trajectoryRadius_ * angularRate_ * angularRate_ * std::cos(t),
      -trajectoryRadius_ * angularRate_ * angularRate_ * std::sin(t),
      -waveHeight_ * frequencyNumber_ * frequencyNumber_ * angularRate_ *
          angularRate_ * std::cos(frequencyNumber_ * t);
  return a_WB_W - gw;
}

Eigen::Vector3d WavyCircle::computeGlobalLinearVelocity(
    const okvis::Time time) const {
  double t = time.toSec() * angularRate_;
  Eigen::Vector3d v_WB_W;
  v_WB_W << - trajectoryRadius_ * angularRate_ * std::sin(t),
      trajectoryRadius_ * angularRate_ * std::cos(t),
      - waveHeight_ * frequencyNumber_ * angularRate_ * std::sin(frequencyNumber_ * t);
  return v_WB_W;
}

okvis::kinematics::Transformation WavyCircle::computeGlobalPose(
    const okvis::Time time) const {
  double t = time.toSec() * angularRate_;
  return okvis::kinematics::Transformation(position(t),
                                           Eigen::Quaterniond(orientation(t)));
}

Eigen::Vector3d WavyCircle::position(double t) const {
  Eigen::Vector3d t_WB_W;
  t_WB_W << trajectoryRadius_ * std::cos(t), trajectoryRadius_ * std::sin(t),
      waveHeight_ * std::cos(frequencyNumber_ * t);
  return t_WB_W;
}

double WavyCircle::nearestDepth() const {
  return std::sqrt(wallRadius_ * wallRadius_ -
                   trajectoryRadius_ * trajectoryRadius_);
}

double WavyCircle::computeWaveHeight() const {
  double halfz = 0.5 * wallHeight_;
  double nd = nearestDepth();
  double tan_vertical_half_Fov = halfz / nd;

  double wh = waveHeightCoeff_ *
              (tan_vertical_half_Fov * trajectoryRadius_ / frequencyNumber_);
  return wh;
}

template<typename T>
Eigen::Matrix<T, 3, 3> WavyCircle::orientation(T t) const {
  Eigen::Matrix<T, 3, 1> F(
      -T(trajectoryRadius_) * sin(t), T(trajectoryRadius_) * cos(t),
      -T(waveHeight_) * T(frequencyNumber_) * sin(T(frequencyNumber_) * t));
  F.normalize();
  Eigen::Matrix<T, 3, 1> L(-cos(t), -sin(t), T(0));
  Eigen::Matrix<T, 3, 1> U = F.cross(L);
  U.normalize();
  Eigen::Matrix<T, 3, 3> R_WB;
  R_WB.col(0) = F;
  R_WB.col(1) = L;
  R_WB.col(2) = U;
  // add rotation about another axis.
  Eigen::Matrix<T, 3, 3> Rx = RotX(T(30 * M_PI / 180) * sin(T(5) * t));
  R_WB = R_WB * Rx;
  return R_WB;
}

std::shared_ptr<CircularSinusoidalTrajectory>
createSimulatedTrajectory(SimulatedTrajectoryType trajectoryType, int rate,
                          double gravityNorm) {
  switch (trajectoryType) {
  case SimulatedTrajectoryType::Sinusoid:
    return std::shared_ptr<CircularSinusoidalTrajectory>(
        new simul::CircularSinusoidalTrajectory(
            rate, Eigen::Vector3d(0, 0, -gravityNorm)));

  case SimulatedTrajectoryType::Torus:
    return std::shared_ptr<CircularSinusoidalTrajectory>(
        new simul::TorusTrajectory(rate, Eigen::Vector3d(0, 0, -gravityNorm)));

  case SimulatedTrajectoryType::Squircle:
    return std::shared_ptr<CircularSinusoidalTrajectory>(
        new simul::RoundedSquare(rate, Eigen::Vector3d(0, 0, -gravityNorm)));

  case SimulatedTrajectoryType::Circle:
    return std::shared_ptr<CircularSinusoidalTrajectory>(
        new simul::RoundedSquare(rate, Eigen::Vector3d(0, 0, -gravityNorm),
                                 okvis::Time(0, 0), 1.0, 0, 0.8));

  case SimulatedTrajectoryType::Dot:
    return std::shared_ptr<CircularSinusoidalTrajectory>(
        new simul::RoundedSquare(rate, Eigen::Vector3d(0, 0, -gravityNorm),
                                 okvis::Time(0, 0), 1e-3, 0, 0.8e-3));

  case SimulatedTrajectoryType::WavyCircle:
    return std::shared_ptr<CircularSinusoidalTrajectory>(
        new simul::WavyCircle(rate, Eigen::Vector3d(0, 0, -gravityNorm)));

  case SimulatedTrajectoryType::Motionless:
    return std::shared_ptr<CircularSinusoidalTrajectory>(
        new simul::Motionless(rate, Eigen::Vector3d(0, 0, -gravityNorm)));

  case SimulatedTrajectoryType::Torus2:
    return std::shared_ptr<CircularSinusoidalTrajectory>(
        new simul::SphereTrajectory(rate, Eigen::Vector3d(0, 0, -gravityNorm)));

  case SimulatedTrajectoryType::Ball:
    return std::shared_ptr<CircularSinusoidalTrajectory>(
        new simul::SphereTrajectory(rate, Eigen::Vector3d(0, 0, -gravityNorm),
                                    1.0, 0.4 * M_PI));

  default:
    LOG(ERROR) << "Unknown trajectory id " << static_cast<int>(trajectoryType);
    return std::shared_ptr<CircularSinusoidalTrajectory>();
  }
}

}  // namespace simul
