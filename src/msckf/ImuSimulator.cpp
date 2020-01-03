#include "msckf/ImuSimulator.h"
#include "vio/Sample.h"

#include <glog/logging.h>

DEFINE_double(sim_sigma_g_c, 1.2e-3, "simulated gyro noise density");
DEFINE_double(sim_sigma_a_c, 8e-3, "simulated accelerometer noise density");
DEFINE_double(sim_sigma_gw_c, 2e-5, "simulated gyro bias noise density");
DEFINE_double(sim_sigma_aw_c, 5.5e-5, "simulated accelerometer bias noise density");

namespace imu {
CircularSinusoidalTrajectory::CircularSinusoidalTrajectory()
    : wz(17 * M_PI / 41),
      wxy(7 * M_PI / 37),
      rz(1),
      rxy(61.0 / 19),
      freq(100),
      interval(1 / freq),
      maxThetaZ(0.4 * M_PI),
      gw(0, 0, -9.8) {}

CircularSinusoidalTrajectory::CircularSinusoidalTrajectory(double imuFreq,
                                                           Eigen::Vector3d ginw)
    : wz(17 * M_PI / 41),
      wxy(7 * M_PI / 37),
      rz(1),
      rxy(61.0 / 19),
      freq(imuFreq),
      interval(1 / freq),
      maxThetaZ(0.4 * M_PI),
      gw(ginw) {}

void CircularSinusoidalTrajectory::getTrueInertialMeasurements(
    const okvis::Time tStart, const okvis::Time tEnd,
    okvis::ImuMeasurementDeque& imuMeasurements) {
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

void CircularSinusoidalTrajectory::getNoisyInertialMeasurements(
    const okvis::Time tStart, const okvis::Time tEnd,
    okvis::ImuMeasurementDeque& imuMeasurements) {
  // TODO: add noise and biases
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

void CircularSinusoidalTrajectory::getTruePoses(
    const okvis::Time tStart, const okvis::Time tEnd,
    std::vector<okvis::kinematics::Transformation>& vT_WB) {
  okvis::Time time = tStart;
  vT_WB.clear();
  vT_WB.reserve((int)((tEnd - tStart).toSec() * freq + 1));
  for (; time < tEnd; time += okvis::Duration(interval)) {
    vT_WB.push_back(computeGlobalPose(time));
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
    const okvis::Time time) {
  double dTime = time.toSec();
  double thetaZDot = maxThetaZ * cos(wz * dTime) * wz;

  double thetaXY = wxy * dTime;
  double wgx = -thetaZDot * sin(thetaXY);
  double wgy = thetaZDot * cos(thetaXY);
  double wgz = wxy;
  return Eigen::Vector3d(wgx, wgy, wgz);
}

Eigen::Vector3d CircularSinusoidalTrajectory::computeGlobalLinearAcceleration(
    const okvis::Time time) {
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
    const okvis::Time time) {
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
CircularSinusoidalTrajectory::computeGlobalPose(const okvis::Time time) {
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

TorusTrajectory::TorusTrajectory()
    : CircularSinusoidalTrajectory(), wr(19 * M_PI / 137), xosc(rxy - rz) {}

TorusTrajectory::TorusTrajectory(
    double imuFreq, Eigen::Vector3d ginw)
    : CircularSinusoidalTrajectory(imuFreq, ginw),
      wr(19 * M_PI / 137),
      xosc(rxy - rz) {}

Eigen::Vector3d TorusTrajectory::computeGlobalLinearAcceleration(
    const okvis::Time time) {
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
    const okvis::Time time) {
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
TorusTrajectory::computeGlobalPose(const okvis::Time time) {
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

SphereTrajectory::SphereTrajectory()
    : CircularSinusoidalTrajectory() {
  rxy = 37 / 19;
}

SphereTrajectory::SphereTrajectory(
    double imuFreq, Eigen::Vector3d ginw)
    : CircularSinusoidalTrajectory(imuFreq, ginw) {
  rxy = 37 / 19;
}

Eigen::Vector3d SphereTrajectory::computeGlobalAngularRate(
    const okvis::Time time) {
  double dTime = time.toSec();
  double thetaZDot = maxThetaZ * cos(wz * dTime) * wz;

  double thetaXY = M_PI * (1.0 - cos(wxy * dTime));
  double wgx = -thetaZDot * sin(thetaXY);
  double wgy = thetaZDot * cos(thetaXY);
  double wgz = M_PI * sin(wxy * dTime) * wxy;
  return Eigen::Vector3d(wgx, wgy, wgz);
}

Eigen::Vector3d SphereTrajectory::computeGlobalLinearAcceleration(
    const okvis::Time time) {
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
    const okvis::Time time) {
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
SphereTrajectory::computeGlobalPose(const okvis::Time time) {
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
      sideLength_(2.0),
      velocityNorm_(0.8) {
  initDataStructures();
}

void RoundedSquare::initDataStructures() {
  double eps = 1e-8;
  CHECK_GT(radius_, eps);
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
    const okvis::Time time) {
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
    const okvis::Time time) {
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
    const okvis::Time time) {
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
    const okvis::Time time) {
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
                                   double* time_into_slot) {
  *j = 0;
  for (; *j < endEpochs_.size(); ++(*j)) {
    if (time_into_period < endEpochs_[*j]) break;
  }

  *time_into_slot =
      *j > 0 ? time_into_period - endEpochs_[(*j) - 1] : time_into_period;
}

double RoundedSquare::getPeriodRemainder(const okvis::Time time) {
  CHECK_GE(time, startEpoch_)
      << "Query time should be greater than start epoch!";
  okvis::Duration elapsed = time - startEpoch_;
  return std::fmod(elapsed.toSec(), period_.toSec());
}

void initImuNoiseParams(
    okvis::ImuParameters* imuParameters, bool addPriorNoise,
    double sigma_bg, double sigma_ba, double std_Ta_elem,
    bool fixImuInternalParams) {
  imuParameters->g = 9.81;
  imuParameters->a_max = 1000.0;
  imuParameters->g_max = 1000.0;
  imuParameters->rate = 100;

  imuParameters->sigma_g_c = FLAGS_sim_sigma_g_c;
  imuParameters->sigma_a_c = FLAGS_sim_sigma_a_c;
  imuParameters->sigma_gw_c = FLAGS_sim_sigma_gw_c;
  imuParameters->sigma_aw_c = FLAGS_sim_sigma_aw_c;

  LOG(INFO) << "sigma_g_c " << FLAGS_sim_sigma_g_c
            << " sigma_a_c " << FLAGS_sim_sigma_a_c
            << " sigma_gw_c " << FLAGS_sim_sigma_gw_c
            << " sigma_aw_c " << FLAGS_sim_sigma_aw_c;

  imuParameters->tau = 600.0;

  imuParameters->sigma_bg = sigma_bg;
  imuParameters->sigma_ba = sigma_ba;

  if (fixImuInternalParams) {
    imuParameters->sigma_TGElement = 0;
    imuParameters->sigma_TSElement = 0;
    imuParameters->sigma_TAElement = 0;
  } else {
    // std for every element in shape matrix T_g
    imuParameters->sigma_TGElement = 5e-3;
    imuParameters->sigma_TSElement = 1e-3;
    imuParameters->sigma_TAElement = std_Ta_elem;
  }
  imuParameters->model_type = "BG_BA_TG_TS_TA";

  Eigen::Matrix<double, 9, 1> eye;
  eye << 1, 0, 0, 0, 1, 0, 0, 0, 1;

  if (addPriorNoise) {
    imuParameters->a0[0] = vio::gauss_rand(0, imuParameters->sigma_ba);
    imuParameters->a0[1] = vio::gauss_rand(0, imuParameters->sigma_ba);
    imuParameters->a0[2] = vio::gauss_rand(0, imuParameters->sigma_ba);
    imuParameters->g0[0] = vio::gauss_rand(0, imuParameters->sigma_bg);
    imuParameters->g0[1] = vio::gauss_rand(0, imuParameters->sigma_bg);
    imuParameters->g0[2] = vio::gauss_rand(0, imuParameters->sigma_bg);

    imuParameters->Tg0 =
        eye + vio::Sample::gaussian(imuParameters->sigma_TGElement, 9);
    imuParameters->Ts0 =
        vio::Sample::gaussian(imuParameters->sigma_TSElement, 9);
    imuParameters->Ta0 =
        eye + vio::Sample::gaussian(imuParameters->sigma_TAElement, 9);
  } else {
    imuParameters->a0.setZero();
    imuParameters->g0.setZero();

    imuParameters->Tg0 = eye;
    imuParameters->Ts0.setZero();
    imuParameters->Ta0 = eye;
  }
}


// TODO(jhuai): add a pure rotation test case in which the
// angular rate is governed by the pendulum dynamic equation
// see https://en.wikipedia.org/wiki/Pendulum_(mathematics)

}  // namespace imu
