#include "msckf/VioTestSystemBuilder.hpp"
#include <msckf/CameraSystemCreator.hpp>
#include <msckf/GeneralEstimator.hpp>
#include <msckf/InvariantEKF.hpp>
#include <msckf/SlidingWindowSmoother.hpp>
#include <msckf/TFVIO.hpp>
#include <msckf/VioEvaluationCallback.hpp>

#include <gflags/gflags.h>

DEFINE_bool(zero_camera_intrinsic_param_noise, true,
            "Set the variance of the camera intrinsic parameters zero."
            " Otherwise, these parameters will be estimated by the filter."
            "If addSystemNoise is true, this parameter will have no effect.");

DEFINE_bool(zero_imu_intrinsic_param_noise, true,
            "Set the variance of the IMU augmented intrinsic parameters zero."
            " Otherwise, these parameters will be estimated by the filter."
            "If addSystemNoise is true, this parameter will have no effect.");

namespace simul {
void VioTestSystemBuilder::createVioSystem(
    const okvis::TestSetting& testSetting,
    SimulatedTrajectoryType trajectoryType,
    std::string projOptModelName,
    std::string extrinsicModelName,
    double td, double tr,
    std::string imuLogFile,
    std::string pointFile) {
  const double DURATION = 300.0;  // length of motion in seconds
  double pCB_std = 2e-2;
  double bg_std = 5e-3;
  double ba_std = 2e-2;
  double Tg_std = 5e-3;
  double Ts_std = 1e-3;
  double Ta_std = 5e-3;
  bool zeroCameraIntrinsicParamNoise = FLAGS_zero_camera_intrinsic_param_noise;
  bool zeroImuIntrinsicParamNoise = FLAGS_zero_imu_intrinsic_param_noise;
  if (testSetting.addSystemError) {
    zeroCameraIntrinsicParamNoise = false;
    zeroImuIntrinsicParamNoise = false;
  }
  okvis::ExtrinsicsEstimationParameters extrinsicsEstimationParameters;
  initCameraNoiseParams(&extrinsicsEstimationParameters, pCB_std,
                        zeroCameraIntrinsicParamNoise);

  okvis::ImuParameters imuParameters;
  simul::initImuNoiseParams(&imuParameters, testSetting.addPriorNoise,
                          testSetting.addSystemError, bg_std, ba_std,
                          Tg_std, Ts_std, Ta_std,
                          zeroImuIntrinsicParamNoise);
  imuModelType_ = imuParameters.model_type;

  const okvis::Time tStart(20);
  const okvis::Time tEnd(20 + DURATION);

  switch (trajectoryType) {
    case SimulatedTrajectoryType::Sinusoid:
      circularSinusoidalTrajectory.reset(
          new simul::CircularSinusoidalTrajectory(
              imuParameters.rate, Eigen::Vector3d(0, 0, -imuParameters.g)));
      break;
    case SimulatedTrajectoryType::Torus:
      circularSinusoidalTrajectory.reset(new simul::TorusTrajectory(
          imuParameters.rate, Eigen::Vector3d(0, 0, -imuParameters.g)));
      break;
    case SimulatedTrajectoryType::Squircle:
      circularSinusoidalTrajectory.reset(new simul::RoundedSquare(
          imuParameters.rate, Eigen::Vector3d(0, 0, -imuParameters.g)));
      break;
    case SimulatedTrajectoryType::Circle:
      circularSinusoidalTrajectory.reset(new simul::RoundedSquare(
          imuParameters.rate, Eigen::Vector3d(0, 0, -imuParameters.g),
          okvis::Time(0, 0), 1.0, 0, 0.8));
      break;
    case SimulatedTrajectoryType::Dot:
      circularSinusoidalTrajectory.reset(new simul::RoundedSquare(
          imuParameters.rate, Eigen::Vector3d(0, 0, -imuParameters.g),
          okvis::Time(0, 0), 1e-3, 0, 0.8e-3));
      break;
    case SimulatedTrajectoryType::WavyCircle:
      circularSinusoidalTrajectory.reset(new simul::WavyCircle(
          imuParameters.rate, Eigen::Vector3d(0, 0, -imuParameters.g)));
      break;
    case SimulatedTrajectoryType::Motionless:
      circularSinusoidalTrajectory.reset(new simul::Motionless(
          imuParameters.rate, Eigen::Vector3d(0, 0, -imuParameters.g)));
      break;
    case SimulatedTrajectoryType::Torus2:
      circularSinusoidalTrajectory.reset(new simul::SphereTrajectory(
          imuParameters.rate, Eigen::Vector3d(0, 0, -imuParameters.g)));
      break;
    case SimulatedTrajectoryType::Ball:
      circularSinusoidalTrajectory.reset(new simul::SphereTrajectory(
          imuParameters.rate, Eigen::Vector3d(0, 0, -imuParameters.g), 1.0,
          0.4 * M_PI));
      break;
    default:
      LOG(ERROR) << "Unknown trajectory id " << static_cast<int>(trajectoryType);
      break;
  }

  circularSinusoidalTrajectory->getTruePoses(tStart, tEnd, ref_T_WS_list_);
  circularSinusoidalTrajectory->getSampleTimes(tStart, tEnd, times_);
  CHECK_EQ(ref_T_WS_list_.size(), times_.size())
      << "timestamps and true poses should have the same size!";

  circularSinusoidalTrajectory->getTrueInertialMeasurements(
      tStart - okvis::Duration(1), tEnd + okvis::Duration(1), imuMeasurements_);

  initialNavState_.std_p_WS = Eigen::Vector3d(1e-8, 1e-8, 1e-8);
  initialNavState_.std_q_WS = Eigen::Vector3d(1e-8, 1e-8, 1e-8);
  initialNavState_.std_v_WS = Eigen::Vector3d(5e-2, 5e-2, 5e-2);
  okvis::Time startEpoch = times_.front();
  okvis::kinematics::Transformation truePose =
      circularSinusoidalTrajectory->computeGlobalPose(startEpoch);
  Eigen::Vector3d p_WS = truePose.r();
  Eigen::Vector3d v_WS = circularSinusoidalTrajectory->computeGlobalLinearVelocity(startEpoch);
  if (testSetting.addPriorNoise) {
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

  evaluationCallback_.reset(new msckf::VioEvaluationCallback());
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
  if (testSetting.addSystemError) {
    csc.createNoisyCameraSystem(&cameraGeometry2, &cameraSystem2,
                                extrinsicsEstimationParameters);
  } else {
    csc.createNominalCameraSystem(&cameraGeometry2, &cameraSystem2);
  }
  distortionType_ = cameraSystem2->cameraGeometry(0)->distortionType();

  okvis::VisualConstraints constraintScheme(okvis::OnlyReprojectionErrors);
  switch (testSetting.estimator_algorithm) {
    case okvis::EstimatorAlgorithm::OKVIS:
      estimator.reset(new okvis::Estimator(mapPtr));
      break;
    case okvis::EstimatorAlgorithm::General:
      estimator.reset(new okvis::GeneralEstimator(mapPtr));
      constraintScheme = okvis::OnlyTwoViewConstraints;
      break;
    case okvis::EstimatorAlgorithm::TFVIO:
      estimator.reset(new okvis::TFVIO(mapPtr));
      break;
    case okvis::EstimatorAlgorithm::InvariantEKF:
      estimator.reset(new okvis::InvariantEKF(mapPtr));
      break;
    case okvis::EstimatorAlgorithm::SlidingWindowSmoother:
      estimator.reset(new okvis::SlidingWindowSmoother(mapPtr));
      break;
    case okvis::EstimatorAlgorithm::MSCKF:
    default:
      estimator.reset(new okvis::MSCKF2(mapPtr));
      break;
  }
  estimator->setUseEpipolarConstraint(testSetting.useEpipolarConstraint);
  estimator->setCameraObservationModel(testSetting.cameraObservationModelId);
  okvis::PointLandmarkOptions plOptions;
  plOptions.landmarkModelId = testSetting.landmarkModelId;
  estimator->setPointLandmarkOptions(plOptions);

  int insideLandmarkModelId = estimator->landmarkModelId();
  int insideCameraObservationModelId = estimator->cameraObservationModelId();
  LOG(INFO) << "Present landmark model " << insideLandmarkModelId
            << " camera model " << insideCameraObservationModelId;

  frontend.reset(new okvis::SimulationFrontend(trueCameraSystem_->numCameras(),
                                               testSetting.addImageNoise, 60,
                                               constraintScheme,
                                               testSetting.gridType,
                                               testSetting.landmarkRadius,
                                               pointFile));

  estimator->addImu(imuParameters);
  estimator->addCameraSystem(
      *cameraSystem2);  // init a noisy camera system in the estimator
  estimator->addCameraParameterStds(extrinsicsEstimationParameters);
}

} // namespace simul
