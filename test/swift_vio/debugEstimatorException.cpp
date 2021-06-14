#include <gtest/gtest.h>

#include <simul/gflags.hpp>
#include <simul/VioSimTestSystem.hpp>
#include "../customized_terminate.hpp"

namespace {
void checkMSE(const Eigen::VectorXd &mse, const Eigen::VectorXd &desiredStdevs,
              const std::vector<std::string> &dimensionLabels) {
}

void checkNEES(const Eigen::Vector3d &nees) {
}
// invoke set_terminate as part of global constant initialization
static const bool SET_TERMINATE = std::set_terminate(customized_terminate);

}  // namespace


//TEST(MSCKF, TrajectoryLabel) {
int main(int argc, char ** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  FLAGS_logtostderr = 1;
  FLAGS_stderrthreshold = 0; // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
  FLAGS_colorlogtostderr = 1;

  LOG(INFO) << "Debugging HybridFilter";
  struct sigaction sigact;

  sigact.sa_sigaction = crit_err_hdlr;
  sigact.sa_flags = SA_RESTART | SA_SIGINFO;

  if (sigaction(SIGABRT, &sigact, (struct sigaction *)NULL) != 0) {
      std::cerr << "error setting handler for signal " << SIGABRT
                << " (" << strsignal(SIGABRT) << ")\n";
      exit(EXIT_FAILURE);
  }

  int cameraObservationModelId = 0;
  int landmarkModelId = FLAGS_sim_landmark_model;
  double landmarkRadius = 5;
  simul::SimImuParameters imuParams(
      FLAGS_sim_trajectory_label, true, FLAGS_noisyInitialSpeedAndBiases,
      FLAGS_noisyInitialSensorParams, FLAGS_fixImuIntrinsicParams,
      5e-3, 2e-2, 5e-3, 1e-3, 5e-3, FLAGS_sim_sigma_g_c, FLAGS_sim_sigma_gw_c,
      FLAGS_sim_sigma_a_c, FLAGS_sim_sigma_aw_c,
      FLAGS_sim_imu_noise_factor, FLAGS_sim_imu_bias_noise_factor);
  simul::SimVisionParameters visionParams(
      true, true, simul::SimCameraModelType::EUROC,
      simul::CameraOrientation::Forward, "FXY_CXY", "P_CB",
      FLAGS_fixCameraInternalParams, 2e-2, 0.0, FLAGS_sim_camera_time_offset_sec,
      FLAGS_sim_frame_readout_time_sec, FLAGS_noisyInitialSensorParams,
      simul::LandmarkGridType::FourWalls, landmarkRadius);
  simul::SimEstimatorParameters estimatorParams(
      "MSCKF", swift_vio::EstimatorAlgorithm::MSCKF, FLAGS_num_runs,
      cameraObservationModelId, landmarkModelId, false);
  swift_vio::BackendParams backendParams;
  simul::TestSetting testSetting(imuParams, visionParams,
                                 estimatorParams, backendParams, FLAGS_sim_real_data_dir);
  simul::VioSimTestSystem simSystem(checkMSE, checkNEES);
  simSystem.run(testSetting, FLAGS_log_dir);
  exit(EXIT_SUCCESS);
}
