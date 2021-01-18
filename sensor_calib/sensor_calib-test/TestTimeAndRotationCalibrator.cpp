#include "gmock/gmock.h"

#include "TimeAndRotationCalibrator.h"

#include <Eigen/Core>

#include "simul/curves.h"

#include "okvis/Time.hpp"
#include "okvis/kinematics/Transformation.hpp"

using namespace testing;

namespace sensor_calib::tests {
TEST(TimeAndRotationCalibrator, ImuCamera) {
  sensor_calib::TimeAndRotationCalibrator imuCamCalibrator;
  int rate = 200;
  int durationInSecs = 20;
  int checkPointInSecs[] = {1, 5, 10, 20};

  double nominalTimeOffsetInSecs = 0.5;
  okvis::kinematics::Transformation T_IG; // I - IMU, G - Generic sensor, say camera.
  T_IG.setRandom();
  okvis::kinematics::Transformation T_WWs; // W - World frame chosen by the target sensor, Ws, world frame for trajectory simulation.
  T_WWs.setRandom();

  std::shared_ptr<simul::CircularSinusoidalTrajectory> traj =
      simul::createSimulatedTrajectory(
        simul::SimulatedTrajectoryType::WavyCircle, rate, 9.80665);

  double samplingInterval = 1.0 / rate;
  int numSuccessfulCalibration = 0;
  int nextCheckIndex = 0;
  for (int tick = 1 + rate * nominalTimeOffsetInSecs; tick < rate * durationInSecs; ++tick) {
    okvis::Time time(tick * samplingInterval);
    Eigen::Vector3d gyro, accelerometer;
    traj->getTrueInertialMeasurements(time, &gyro, &accelerometer);
    okvis::Time targetTime = time - okvis::Duration(nominalTimeOffsetInSecs);
    okvis::kinematics::Transformation T_WsI = traj->computeGlobalPose(targetTime);
    okvis::kinematics::Transformation T_WG = T_WWs * T_WsI * T_IG;
    imuCamCalibrator.addImuAngularRate(time, gyro);
    imuCamCalibrator.addTargetOrientation(targetTime, T_WG.q());

    if (tick == rate * checkPointInSecs[nextCheckIndex]) {
      TimeAndRotationCalibrator::CalibrationStatus status = imuCamCalibrator.calibrate();
      ++nextCheckIndex;
      if (status == TimeAndRotationCalibrator::CalibrationStatus::Successful) {
        ++numSuccessfulCalibration;
        double timeOffset = imuCamCalibrator.relativeTimeOffset();
        EXPECT_NEAR(timeOffset, nominalTimeOffsetInSecs, 1e-5) << "Relative time offset estimated poorly!";
        Eigen::Quaterniond q_IG = imuCamCalibrator.relativeOrientation();
        Eigen::AngleAxisd qDelta(q_IG * T_IG.q().inverse());
        EXPECT_LT(qDelta.angle(), 1e-5) << "Relative orientation estimated poorly!";
      }
    }
  }
  EXPECT_GT(numSuccessfulCalibration, 0) << "None IMU-centric camera extrinsic calibration succeeds!";
}

TEST(TimeAndRotationCalibrator, ImuImu) {
  sensor_calib::TimeAndRotationCalibrator imuCamCalibrator;
  int rate = 200;
  int durationInSecs = 20;
  int checkPointInSecs[] = {1, 5, 10, 20};

  double nominalTimeOffsetInSecs = 0.5;
  okvis::kinematics::Transformation T_IG; // I - IMU, G - Generic sensor, say camera.
  T_IG.setRandom();

  std::shared_ptr<simul::CircularSinusoidalTrajectory> traj =
      simul::createSimulatedTrajectory(
        simul::SimulatedTrajectoryType::WavyCircle, rate, 9.80665);

  double samplingInterval = 1.0 / rate;
  int numSuccessfulCalibration = 0;
  int nextCheckIndex = 0;
  for (int tick = 1 + rate * nominalTimeOffsetInSecs; tick < rate * durationInSecs; ++tick) {
    okvis::Time time(tick * samplingInterval);
    Eigen::Vector3d gyroRef, accelerometerRef;
    traj->getTrueInertialMeasurements(time, &gyroRef, &accelerometerRef);

    Eigen::Vector3d gyro, accelerometer;
    okvis::Time targetTime = time - okvis::Duration(nominalTimeOffsetInSecs);
    gyro = traj->computeLocalAngularVelocity(targetTime);
    gyro = T_IG.q().conjugate() * gyro;

    imuCamCalibrator.addImuAngularRate(time, gyroRef);
    imuCamCalibrator.addTargetAngularRate(targetTime, gyro);

    if (tick == rate * checkPointInSecs[nextCheckIndex]) {
      TimeAndRotationCalibrator::CalibrationStatus status = imuCamCalibrator.calibrate();
      ++nextCheckIndex;
      if (status == TimeAndRotationCalibrator::CalibrationStatus::Successful) {
        ++numSuccessfulCalibration;
        double timeOffset = imuCamCalibrator.relativeTimeOffset();
        EXPECT_NEAR(timeOffset, nominalTimeOffsetInSecs, 1e-5) << "Relative time offset estimated poorly!";
        Eigen::Quaterniond q_IG = imuCamCalibrator.relativeOrientation();
        Eigen::AngleAxisd qDelta(q_IG * T_IG.q().inverse());
        EXPECT_LT(qDelta.angle(), 1e-5) << "Relative orientation estimated poorly!";
      }
    }
  }
  EXPECT_GT(numSuccessfulCalibration, 0) << "None IMU-centric camera extrinsic calibration succeeds!";
}


} // namespace sensor_calib::tests
