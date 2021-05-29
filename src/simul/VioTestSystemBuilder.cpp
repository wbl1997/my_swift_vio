#include "simul/VioTestSystemBuilder.hpp"

#include <gflags/gflags.h>

#include <gtsam/SlidingWindowSmoother.hpp>

#include <swift_vio/TFVIO.hpp>
#include <swift_vio/VioEvaluationCallback.hpp>
#include <swift_vio/VioFactoryMethods.hpp>

#include <simul/CameraSystemCreator.hpp>

DEFINE_bool(zero_camera_intrinsic_param_noise, true,
            "Set the variance of the camera intrinsic parameters zero."
            " Otherwise, these parameters will be estimated by the filter.");

DEFINE_bool(zero_imu_intrinsic_param_noise, true,
            "Set the variance of the IMU augmented intrinsic parameters zero."
            " Otherwise, these parameters will be estimated by the filter.");

namespace simul {
void VioTestSystemBuilder::createVioSystem(
    const TestSetting& testSetting,
    const swift_vio::BackendParams& backendParams,
    SimulatedTrajectoryType trajectoryType,
    std::string projOptModelName,
    std::string extrinsicModelName,
    double td, double tr,
    std::string imuLogFile,
    std::string pointFile) {
  // The following parameters are in metric units.
  const double kDuration = 300.0;  // length of motion in seconds
  double pCB_std = 2e-2;
  double bg_std = 5e-3;
  double ba_std = 2e-2;
  double Tg_std = 5e-3;
  double Ts_std = 1e-3;
  double Ta_std = 5e-3;
  bool zeroCameraIntrinsicParamNoise = FLAGS_zero_camera_intrinsic_param_noise;
  bool zeroImuIntrinsicParamNoise = FLAGS_zero_imu_intrinsic_param_noise;

  okvis::ExtrinsicsEstimationParameters extrinsicsEstimationParameters;
  initCameraNoiseParams(&extrinsicsEstimationParameters, pCB_std,
                        zeroCameraIntrinsicParamNoise);

  okvis::ImuParameters imuParameters;
  if (testSetting.estimator_algorithm <
      swift_vio::EstimatorAlgorithm::HybridFilter) {
    imuParameters.model_type = "BG_BA";
  } else {
    imuParameters.model_type = "BG_BA_TG_TS_TA";
  }
  simul::initImuNoiseParams(testSetting.noisyInitialSpeedAndBiases,
                          testSetting.noisyInitialSensorParams, bg_std, ba_std,
                          Tg_std, Ts_std, Ta_std,
                          zeroImuIntrinsicParamNoise, &imuParameters);

  const okvis::Time tStart(100);
  const okvis::Time tEnd = tStart + okvis::Duration(kDuration);

  circularSinusoidalTrajectory = simul::createSimulatedTrajectory(
      trajectoryType, imuParameters.rate, imuParameters.g);

  circularSinusoidalTrajectory->getTruePoses(tStart, tEnd, ref_T_WS_list_);
  circularSinusoidalTrajectory->getSampleTimes(tStart, tEnd, times_);
  CHECK_EQ(ref_T_WS_list_.size(), times_.size())
      << "timestamps and true poses should have the same size!";

  circularSinusoidalTrajectory->getTrueInertialMeasurements(
      tStart - okvis::Duration(1), tEnd + okvis::Duration(1), imuMeasurements_);

  initialNavState_.std_p_WS = Eigen::Vector3d(1e-5, 1e-5, 1e-5);
  initialNavState_.std_q_WS = Eigen::Vector3d(M_PI / 180, M_PI / 180, 1e-5);
  initialNavState_.std_v_WS = Eigen::Vector3d(5e-2, 5e-2, 5e-2);
  okvis::Time startEpoch = times_.front();
  okvis::kinematics::Transformation truePose =
      circularSinusoidalTrajectory->computeGlobalPose(startEpoch);
  Eigen::Vector3d p_WS = truePose.r();
  Eigen::Vector3d v_WS = circularSinusoidalTrajectory->computeGlobalLinearVelocity(startEpoch);
  if (testSetting.noisyInitialSpeedAndBiases) {
    v_WS += vio::Sample::gaussian(1, 3).cwiseProduct(initialNavState_.std_v_WS);
  }

  initialNavState_.initWithExternalSource = true;
  initialNavState_.p_WS = p_WS;
  initialNavState_.q_WS = truePose.q();
  initialNavState_.v_WS = v_WS;

  if (testSetting.addImuNoise) {
      std::shared_ptr<std::ofstream> inertialStream;
      if (!imuLogFile.empty()) {
        inertialStream.reset(new std::ofstream(imuLogFile, std::ofstream::out));
        (*inertialStream)
            << "% timestamp, gx, gy, gz[rad/sec], acc x, acc y, acc "
               "z[m/s^2], gyro bias xyz, acc bias xyz, noisy gxyz, acc xyz"
            << std::endl;
      }
      simul::addNoiseToImuReadings(imuParameters, &imuMeasurements_, &trueBiases_,
                                 testSetting.sim_ga_noise_factor,
                                 testSetting.sim_ga_bias_noise_factor,
                                 inertialStream.get());
  } else {
    trueBiases_ = imuMeasurements_;
    for (size_t i = 0; i < imuMeasurements_.size(); ++i) {
      trueBiases_[i].measurement.gyroscopes.setZero();
      trueBiases_[i].measurement.accelerometers.setZero();
    }
  }
  // remove the padding part of trueBiases to prepare for computing bias rmse
  auto tempIter = trueBiases_.begin();
  for (; tempIter != trueBiases_.end(); ++tempIter) {
    if (fabs((tempIter->timeStamp - times_.front()).toSec()) < 1e-8) break;
  }
  CHECK_EQ(tempIter != trueBiases_.end(), true) << "No imu reading close to motion start epoch by 1e-8";
  trueBiases_.erase(trueBiases_.begin(), tempIter);

  evaluationCallback_.reset(new swift_vio::VioEvaluationCallback());
  std::shared_ptr<okvis::ceres::Map> mapPtr(new okvis::ceres::Map(evaluationCallback_.get()));
  // std::shared_ptr<okvis::ceres::Map> mapPtr(new okvis::ceres::Map());

  simul::CameraSystemCreator csc(testSetting.cameraModelId,
                                 testSetting.cameraOrientationId,
                                 projOptModelName, extrinsicModelName, td, tr);
  // reference camera system
  std::shared_ptr<okvis::cameras::CameraBase> cameraGeometry0;
  csc.createNominalCameraSystem(&cameraGeometry0, &trueCameraSystem_);

  // dummy camera to keep camera info secret from the estimator
  std::shared_ptr<okvis::cameras::CameraBase> cameraGeometry1;
  std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem1;
  csc.createDummyCameraSystem(&cameraGeometry1, &cameraSystem1);

  // camera system used for initilizing the estimator
  std::shared_ptr<okvis::cameras::CameraBase> cameraGeometry2;
  std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem2;
  if (testSetting.noisyInitialSensorParams) {
    csc.createNoisyCameraSystem(&cameraGeometry2, &cameraSystem2,
                                extrinsicsEstimationParameters);
  } else {
    csc.createNominalCameraSystem(&cameraGeometry2, &cameraSystem2);
  }

  estimator = swift_vio::createBackend(testSetting.estimator_algorithm,
                                       backendParams, mapPtr);

  okvis::Optimization optimOptions;
  optimOptions.useEpipolarConstraint = testSetting.useEpipolarConstraint;
  optimOptions.cameraObservationModelId = testSetting.cameraObservationModelId;
  optimOptions.getSmootherCovariance = false;
  optimOptions.numKeyframes = 5;
  optimOptions.numImuFrames = 3;
  estimator->setOptimizationOptions(optimOptions);

  swift_vio::PointLandmarkOptions plOptions;
  plOptions.minTrackLengthForSlam = 5;
  plOptions.landmarkModelId = testSetting.landmarkModelId;
  estimator->setPointLandmarkOptions(plOptions);

  frontend.reset(new SimulationFrontend(
      trueCameraSystem_->numCameras(), testSetting.addImageNoise, 60,
      testSetting.gridType, testSetting.landmarkRadius, pointFile));

  estimator->addImu(imuParameters);
  estimator->addCameraSystem(
      *cameraSystem2);  // init a noisy camera system in the estimator
  estimator->addCameraParameterStds(extrinsicsEstimationParameters);
}

} // namespace simul
