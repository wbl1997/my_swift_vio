#include "gmock/gmock.h"
#include "glog/logging.h"

#include "swift_vio/TimeAndRotationCalibrator.h"

#include <Eigen/Core>

#include "simul/curves.h"

#include "okvis/Time.hpp"
#include "okvis/kinematics/Transformation.hpp"

using namespace testing;

namespace swift_vio::tests {
void testTimeAndRotationCalibrator(bool orientationSensor) {
  swift_vio::TimeAndRotationCalibrator imuCamCalibrator;
  int rate = 200;
  int durationInSecs = 20;
  int checkPointInSecs[] = {5, 14, 16, 18};

  double nominalTimeOffsetInSecs = 0.723;
  okvis::kinematics::Transformation
      T_IG; // I - IMU, G - Generic sensor, say camera.
  T_IG.setRandom();
  okvis::kinematics::Transformation
      T_WWs; // W - World frame chosen by the target sensor, Ws, world frame for
             // trajectory simulation.
  T_WWs.setRandom();
  Eigen::Vector3d bias = Eigen::Vector3d::Random();

  std::shared_ptr<simul::CircularSinusoidalTrajectory> traj =
      simul::createSimulatedTrajectory(
          simul::SimulatedTrajectoryType::Torus2, rate,
          9.80665);

  double samplingInterval = 1.0 / rate;
  int numSuccessfulCalibration = 0;
  int nextCheckIndex = 0;
  int startTick = rate * 100;
  for (int tick = startTick + rate * nominalTimeOffsetInSecs;
       tick < startTick + rate * durationInSecs; ++tick) {
    okvis::Time time(tick * samplingInterval);
    Eigen::Vector3d gyro, accelerometer;
    traj->getTrueInertialMeasurements(time, &gyro, &accelerometer);
    imuCamCalibrator.addImuAngularRate(time, gyro + bias);

    if (orientationSensor) {
      if (tick % 10 == 0) {
        okvis::Time targetTime =
            time - okvis::Duration(nominalTimeOffsetInSecs);
        okvis::kinematics::Transformation T_WsI = traj->computeGlobalPose(time);
        okvis::kinematics::Transformation T_WG = T_WWs * T_WsI * T_IG;
        imuCamCalibrator.addTargetOrientation(targetTime, T_WG.q());
      }
    } else {
      if (tick % 4 == 0) {
        okvis::Time targetTime =
            time - okvis::Duration(nominalTimeOffsetInSecs);
        imuCamCalibrator.addTargetAngularRate(targetTime, T_IG.q().conjugate() * gyro);
      }
    }

    if (tick == startTick + rate * checkPointInSecs[nextCheckIndex]) {
      TimeAndRotationCalibrator::CalibrationStatus status =
          imuCamCalibrator.calibrate();
      ++nextCheckIndex;

      if (status == TimeAndRotationCalibrator::CalibrationStatus::Successful) {
        ++numSuccessfulCalibration;
        double timeOffset = imuCamCalibrator.relativeTimeOffset();
        EXPECT_NEAR(timeOffset, nominalTimeOffsetInSecs, 1e-4)
            << "Relative time offset estimated poorly!";
        Eigen::Quaterniond q_IG = imuCamCalibrator.relativeOrientation();
        Eigen::AngleAxisd qDelta(q_IG * T_IG.q().inverse());
        EXPECT_LT(qDelta.angle(), 5e-4)
            << "Relative orientation estimated poorly!"
            << "\nComputed R_IG\n"
            << q_IG.toRotationMatrix() << "\nTrue R_IG\n"
            << T_IG.C();
      }
    }
    if (tick % rate == 0) {
      int removed = imuCamCalibrator.slideWindow();
      if (nextCheckIndex >= 2) {
        EXPECT_GT(removed, 1);
      }
    }
  }
  LOG(INFO) << "#Successful calibrations " << numSuccessfulCalibration;
  EXPECT_GT(numSuccessfulCalibration, 0)
      << "None IMU-centric camera extrinsic calibration succeeds!";
}

TEST(TimeAndRotationCalibrator, ImuCamera) {
  testTimeAndRotationCalibrator(true);
}

TEST(TimeAndRotationCalibrator, ImuImu) {
  testTimeAndRotationCalibrator(false);
}

TEST(TimeAndRotationCalibrator, computeCovariance) {
  int durationInSecs = 20;
  double nominalTimeOffsetInSecs = 0.0;
  int rate = 100;
  std::shared_ptr<simul::CircularSinusoidalTrajectory> traj =
      simul::createSimulatedTrajectory(simul::SimulatedTrajectoryType::Torus2,
                                       rate, 9.80665);
  okvis::kinematics::Transformation
      T_IG; // I - IMU, G - Generic sensor, say camera.
            //  T_IG.setRandom();

  double samplingInterval = 1.0 / rate;

  Eigen::AlignedDeque<okvis::Measurement<Eigen::Vector3d>> x, y;

  for (int tick = 1 + rate * nominalTimeOffsetInSecs;
       tick < rate * durationInSecs; ++tick) {
    okvis::Time time(tick * samplingInterval);
    Eigen::Vector3d gyroRef, accelerometerRef;
    traj->getTrueInertialMeasurements(time, &gyroRef, &accelerometerRef);
    x.emplace_back(time, gyroRef);
    Eigen::Vector3d gyro, accelerometer;
    okvis::Time targetTime = time - okvis::Duration(nominalTimeOffsetInSecs);
    gyro = traj->computeLocalAngularVelocity(targetTime);
    gyro = T_IG.q().conjugate() * gyro;
    y.emplace_back(targetTime, gyro);
  }
  okvis::Time startTimestamp((1 + rate * nominalTimeOffsetInSecs) *
                             samplingInterval);
  double observationDuration = 8;
  std::vector<const Eigen::AlignedDeque<okvis::Measurement<Eigen::Vector3d>> *>
      xy{&x, &y};
  Eigen::Matrix3d C;
  int numUsedObservations =
      computeCovariance(startTimestamp, observationDuration, xy, &C);
  EXPECT_GT(numUsedObservations, observationDuration * rate - 1);
}

} // namespace swift_vio::tests
