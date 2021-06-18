#include <gtest/gtest.h>

#include <simul/gflags.hpp>
#include <simul/VioSimTestSystem.hpp>

DECLARE_string(log_dir); // FLAGS_log_dir can be passed in commandline as --log_dir=/some/log/dir


namespace {
void checkMSE(const Eigen::VectorXd &mse, const Eigen::VectorXd &desiredStdevs,
              const std::vector<std::string> &dimensionLabels) {
  for (int i = 0; i < mse.size(); ++i) {
    EXPECT_LT(mse[i], std::pow(desiredStdevs[i], 2))
        << dimensionLabels[i] + " MSE";
  }
}

void checkNEES(const Eigen::Vector3d &nees) {
  EXPECT_LT(nees[0], 8) << "Position NEES";
  EXPECT_LT(nees[1], 5) << "Orientation NEES";
  EXPECT_LT(nees[2], 10) << "Pose NEES";
}
}  // namespace

// {WavyCircle, Squircle, Circle, Dot, CircleWithFarPoints, Motionless} X
// {MSCKF with IDP and reprojection errors,
//  MSCKF with XYZ and reprojection errors,
//  MSCKF with IDP and reprojection errors and epipolar constraints for low parallax,
//  MSCKF with parallax angle and chordal distance,
//  MSCKF with parallax angle and reprojection errors,
//  TFVIO (roughly MSCKF with only epipolar constraints),
//  OKVIS}

TEST(DeadreckoningM, TrajectoryLabel) { 
  int cameraObservationModelId = 0;
  int landmarkModelId = 0;
  double landmarkRadius = 5;
  simul::SimImuParameters imuParams(
      FLAGS_sim_trajectory_label, true, FLAGS_noisyInitialSpeedAndBiases,
      FLAGS_noisyInitialSensorParams, FLAGS_fixImuIntrinsicParams,
      5e-3, 2e-2, 5e-3, 1e-3, 5e-3, FLAGS_sim_sigma_g_c, FLAGS_sim_sigma_gw_c,
      FLAGS_sim_sigma_a_c, FLAGS_sim_sigma_aw_c, FLAGS_sim_imu_noise_factor,
      FLAGS_sim_imu_bias_noise_factor, "BG_BA");
  simul::SimVisionParameters visionParams(
      true, false, simul::SimCameraModelType::EUROC,
      simul::CameraOrientation::Forward, "FIXED", "FIXED",
      FLAGS_fixCameraInternalParams, 0.0, 0.0,
      FLAGS_sim_camera_time_offset_sec,
      FLAGS_sim_frame_readout_time_sec,
      FLAGS_noisyInitialSensorParams, simul::LandmarkGridType::FourWalls,
      landmarkRadius);
  simul::SimEstimatorParameters estimatorParams(
      "DeadreckoningM", swift_vio::EstimatorAlgorithm::MSCKF, FLAGS_num_runs,
      cameraObservationModelId, landmarkModelId, false);
  swift_vio::BackendParams backendParams;
  simul::TestSetting testSetting(imuParams, visionParams,
                                 estimatorParams, backendParams, FLAGS_sim_real_data_dir);
  simul::VioSimTestSystem simSystem(checkMSE, checkNEES);
  simSystem.run(testSetting, FLAGS_log_dir);
}

TEST(DeadreckoningO, TrajectoryLabel) {
  int cameraObservationModelId = 0;
  int landmarkModelId = 0;
  double landmarkRadius = 5;
  simul::SimImuParameters imuParams(
      FLAGS_sim_trajectory_label, true, FLAGS_noisyInitialSpeedAndBiases,
      FLAGS_noisyInitialSensorParams, FLAGS_fixImuIntrinsicParams,
      5e-3, 2e-2, 5e-3, 1e-3, 5e-3, FLAGS_sim_sigma_g_c, FLAGS_sim_sigma_gw_c,
      FLAGS_sim_sigma_a_c, FLAGS_sim_sigma_aw_c,
      FLAGS_sim_imu_noise_factor, FLAGS_sim_imu_bias_noise_factor, "BG_BA");
  simul::SimVisionParameters visionParams(
      true, false, simul::SimCameraModelType::EUROC,
      simul::CameraOrientation::Forward, "FIXED", "FIXED",
      FLAGS_fixCameraInternalParams, 0.0, 0.0, FLAGS_sim_camera_time_offset_sec,
      FLAGS_sim_frame_readout_time_sec, FLAGS_noisyInitialSensorParams,
      simul::LandmarkGridType::FourWalls, landmarkRadius);
  simul::SimEstimatorParameters estimatorParams(
      "DeadreckoningO", swift_vio::EstimatorAlgorithm::OKVIS, FLAGS_num_runs,
      cameraObservationModelId, landmarkModelId, false);
  swift_vio::BackendParams backendParams;
  simul::TestSetting testSetting(imuParams, visionParams,
                                 estimatorParams, backendParams, "");
  simul::VioSimTestSystem simSystem(checkMSE, checkNEES);
  simSystem.run(testSetting, FLAGS_log_dir);
}

TEST(HybridFilter, TrajectoryLabel) {
  int cameraObservationModelId = 0;
  int landmarkModelId = FLAGS_sim_landmark_model;
  double landmarkRadius = 5;
  simul::SimImuParameters imuParams(
      FLAGS_sim_trajectory_label, true, FLAGS_noisyInitialSpeedAndBiases,
      FLAGS_noisyInitialSensorParams, FLAGS_fixImuIntrinsicParams,
      5e-3, 2e-2, 5e-3, 1e-3, 5e-3, FLAGS_sim_sigma_g_c, FLAGS_sim_sigma_gw_c,
      FLAGS_sim_sigma_a_c, FLAGS_sim_sigma_aw_c,
      FLAGS_sim_imu_noise_factor, FLAGS_sim_imu_bias_noise_factor, "BG_BA_TG_TS_TA");
  simul::SimVisionParameters visionParams(
      true, true, simul::SimCameraModelType::EUROC,
      simul::CameraOrientation::Forward, "FXY_CXY", "P_CB",
      FLAGS_fixCameraInternalParams, 2e-2, 0.0, FLAGS_sim_camera_time_offset_sec,
      FLAGS_sim_frame_readout_time_sec, FLAGS_noisyInitialSensorParams,
      simul::LandmarkGridType::FourWalls, landmarkRadius);
  simul::SimEstimatorParameters estimatorParams(
      "HybridFilter", swift_vio::EstimatorAlgorithm::HybridFilter, FLAGS_num_runs,
      cameraObservationModelId, landmarkModelId, false);
  swift_vio::BackendParams backendParams;
  simul::TestSetting testSetting(imuParams, visionParams,
                                 estimatorParams, backendParams, FLAGS_sim_real_data_dir);
  simul::VioSimTestSystem simSystem(checkMSE, checkNEES);
  simSystem.run(testSetting, FLAGS_log_dir);
}

TEST(CalibrationFilter, TrajectoryLabel) {
  int cameraObservationModelId = 0;
  int landmarkModelId = 0;
  double landmarkRadius = 5;
  ASSERT_TRUE(!FLAGS_sim_real_data_dir.empty()) << "CalibrationFilter must be tested with simulated calibration data!";
  simul::SimImuParameters imuParams(
      "None", false, FLAGS_noisyInitialSpeedAndBiases, false, true,
      5e-3, 2e-2, 5e-3, 1e-3, 5e-3, FLAGS_sim_sigma_g_c, FLAGS_sim_sigma_gw_c,
      FLAGS_sim_sigma_a_c, FLAGS_sim_sigma_aw_c,
      FLAGS_sim_imu_noise_factor, FLAGS_sim_imu_bias_noise_factor, "BG_BA");
  simul::SimVisionParameters visionParams(
      false, true, simul::SimCameraModelType::EUROC,
      simul::CameraOrientation::Forward, "FXY_CXY", "P_BC_Q_BC",
      false, 2e-2, 1e-2, 0.0, 0.0, FLAGS_noisyInitialSensorParams,
      simul::LandmarkGridType::FourWalls, landmarkRadius, true);
  simul::SimEstimatorParameters estimatorParams(
      "CalibrationFilter", swift_vio::EstimatorAlgorithm::CalibrationFilter, FLAGS_num_runs,
      cameraObservationModelId, landmarkModelId, false);
  swift_vio::BackendParams backendParams;
  simul::TestSetting testSetting(imuParams, visionParams,
                                 estimatorParams, backendParams, FLAGS_sim_real_data_dir);
  simul::VioSimTestSystem simSystem(checkMSE, checkNEES);
  simSystem.run(testSetting, FLAGS_log_dir);
}


TEST(MSCKF, TrajectoryLabel) {
  int cameraObservationModelId = 0;
  int landmarkModelId = FLAGS_sim_landmark_model;
  double landmarkRadius = 5;
  simul::SimImuParameters imuParams(
      FLAGS_sim_trajectory_label, true, FLAGS_noisyInitialSpeedAndBiases,
      FLAGS_noisyInitialSensorParams, FLAGS_fixImuIntrinsicParams,
      5e-3, 2e-2, 5e-3, 1e-3, 5e-3, FLAGS_sim_sigma_g_c, FLAGS_sim_sigma_gw_c,
      FLAGS_sim_sigma_a_c, FLAGS_sim_sigma_aw_c,
      FLAGS_sim_imu_noise_factor, FLAGS_sim_imu_bias_noise_factor, "BG_BA_TG_TS_TA");
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
}

TEST(MSCKF, HuaiThesis) {
  int cameraObservationModelId = 0;
  int landmarkModelId = FLAGS_sim_landmark_model;
  double landmarkRadius = 5;
  simul::SimImuParameters imuParams(
      "Ball", true, FLAGS_noisyInitialSpeedAndBiases,
      FLAGS_noisyInitialSensorParams, FLAGS_fixImuIntrinsicParams,
      5e-3, 2e-2, 5e-3, 1e-3, 5e-3, FLAGS_sim_sigma_g_c, FLAGS_sim_sigma_gw_c,
      FLAGS_sim_sigma_a_c, FLAGS_sim_sigma_aw_c,
      FLAGS_sim_imu_noise_factor, FLAGS_sim_imu_bias_noise_factor, "BG_BA_TG_TS_TA");
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
                                 estimatorParams, backendParams, "");
  simul::VioSimTestSystem simSystem(checkMSE, checkNEES);
  simSystem.run(testSetting, FLAGS_log_dir);
}

TEST(MSCKF, CircleFarPoints) {
  int cameraObservationModelId = 0;
  int landmarkModelId = FLAGS_sim_landmark_model;
  double landmarkRadius = 50;
  simul::SimImuParameters imuParams(
      "Circle", true, FLAGS_noisyInitialSpeedAndBiases,
      FLAGS_noisyInitialSensorParams, FLAGS_fixImuIntrinsicParams,
      5e-3, 2e-2, 5e-3, 1e-3, 5e-3, FLAGS_sim_sigma_g_c, FLAGS_sim_sigma_gw_c,
      FLAGS_sim_sigma_a_c, FLAGS_sim_sigma_aw_c,
      FLAGS_sim_imu_noise_factor, FLAGS_sim_imu_bias_noise_factor, "BG_BA_TG_TS_TA");
  simul::SimVisionParameters visionParams(
      true, true, simul::SimCameraModelType::EUROC,
      simul::CameraOrientation::Forward, "FXY_CXY", "P_CB",
      FLAGS_fixCameraInternalParams, 2e-2, 0.0, FLAGS_sim_camera_time_offset_sec,
      FLAGS_sim_frame_readout_time_sec, FLAGS_noisyInitialSensorParams,
      simul::LandmarkGridType::Cylinder, landmarkRadius);
  simul::SimEstimatorParameters estimatorParams(
      "MSCKF", swift_vio::EstimatorAlgorithm::MSCKF, FLAGS_num_runs,
      cameraObservationModelId, landmarkModelId, false);
  swift_vio::BackendParams backendParams;
  simul::TestSetting testSetting(imuParams, visionParams,
                                 estimatorParams, backendParams, "");
  simul::VioSimTestSystem simSystem(checkMSE, checkNEES);
  simSystem.run(testSetting, FLAGS_log_dir);
}

// When set computeOkvisNees true, OKVIS outputs consistent covariances indicated by NEES.
TEST(OKVIS, TrajectoryLabel) {
  int cameraObservationModelId = 0;
  int landmarkModelId = 0;
  double landmarkRadius = 5;
  simul::SimImuParameters imuParams(
      FLAGS_sim_trajectory_label, true, FLAGS_noisyInitialSpeedAndBiases,
      FLAGS_noisyInitialSensorParams, true,
      5e-3, 2e-2, 5e-3, 1e-3, 5e-3, FLAGS_sim_sigma_g_c, FLAGS_sim_sigma_gw_c,
      FLAGS_sim_sigma_a_c, FLAGS_sim_sigma_aw_c,
      FLAGS_sim_imu_noise_factor, FLAGS_sim_imu_bias_noise_factor, "BG_BA");
  simul::SimVisionParameters visionParams(
      true, true, simul::SimCameraModelType::EUROC,
      simul::CameraOrientation::Forward, "FIXED", "FIXED",
      true, 2e-2, 0.0, 0.0, 0.0, FLAGS_noisyInitialSensorParams,
      simul::LandmarkGridType::FourWalls, landmarkRadius);

  simul::SimEstimatorParameters estimatorParams(
      "OKVIS", swift_vio::EstimatorAlgorithm::OKVIS, FLAGS_num_runs,
      cameraObservationModelId, landmarkModelId, false, FLAGS_sim_compute_OKVIS_NEES);
  swift_vio::BackendParams backendParams;
  simul::TestSetting testSetting(imuParams, visionParams,
                                 estimatorParams, backendParams, FLAGS_sim_real_data_dir);
  simul::VioSimTestSystem simSystem(checkMSE, checkNEES);
  simSystem.run(testSetting, FLAGS_log_dir);
}

TEST(SlidingWindowSmoother, TrajectoryLabel) {
  int cameraObservationModelId = 0;
  int landmarkModelId = 0;
  double landmarkRadius = 5;
  simul::SimImuParameters imuParams(
      FLAGS_sim_trajectory_label, true, FLAGS_noisyInitialSpeedAndBiases,
      FLAGS_noisyInitialSensorParams, FLAGS_fixImuIntrinsicParams,
      5e-3, 2e-2, 5e-3, 1e-3, 5e-3, FLAGS_sim_sigma_g_c, FLAGS_sim_sigma_gw_c,
      FLAGS_sim_sigma_a_c, FLAGS_sim_sigma_aw_c,
      FLAGS_sim_imu_noise_factor, FLAGS_sim_imu_bias_noise_factor, "BG_BA");
  simul::SimVisionParameters visionParams(
      true, true, simul::SimCameraModelType::EUROC,
      simul::CameraOrientation::Forward, "FIXED", "FIXED",
      FLAGS_fixCameraInternalParams, 0.0, 0.0, FLAGS_sim_camera_time_offset_sec,
      FLAGS_sim_frame_readout_time_sec, FLAGS_noisyInitialSensorParams,
      simul::LandmarkGridType::FourWalls, landmarkRadius);
  simul::SimEstimatorParameters estimatorParams(
      "SlidingWindowSmoother", swift_vio::EstimatorAlgorithm::SlidingWindowSmoother, FLAGS_num_runs,
      cameraObservationModelId, landmarkModelId, false);
  swift_vio::BackendParams backendParams;
  simul::TestSetting testSetting(imuParams, visionParams,
                                 estimatorParams, backendParams, "");
  simul::VioSimTestSystem simSystem(checkMSE, checkNEES);
  simSystem.run(testSetting, FLAGS_log_dir);
}

TEST(RiSlidingWindowSmoother, TrajectoryLabel) {
  int cameraObservationModelId = 0;
  int landmarkModelId = 0;
  double landmarkRadius = 5;
  simul::SimImuParameters imuParams(
      FLAGS_sim_trajectory_label, true, FLAGS_noisyInitialSpeedAndBiases,
      FLAGS_noisyInitialSensorParams, FLAGS_fixImuIntrinsicParams,
      5e-3, 2e-2, 5e-3, 1e-3, 5e-3, FLAGS_sim_sigma_g_c, FLAGS_sim_sigma_gw_c,
      FLAGS_sim_sigma_a_c, FLAGS_sim_sigma_aw_c,
      FLAGS_sim_imu_noise_factor, FLAGS_sim_imu_bias_noise_factor, "BG_BA");
  simul::SimVisionParameters visionParams(
      true, true, simul::SimCameraModelType::EUROC,
      simul::CameraOrientation::Forward, "FIXED", "FIXED",
      FLAGS_fixCameraInternalParams, 0.0, 0.0, FLAGS_sim_camera_time_offset_sec,
      FLAGS_sim_frame_readout_time_sec, FLAGS_noisyInitialSensorParams,
      simul::LandmarkGridType::FourWalls, landmarkRadius);
  simul::SimEstimatorParameters estimatorParams(
      "RiSlidingWindowSmoother", swift_vio::EstimatorAlgorithm::RiSlidingWindowSmoother, FLAGS_num_runs,
      cameraObservationModelId, landmarkModelId, false);
  swift_vio::BackendParams backendParams;
  backendParams.backendModality_ = swift_vio::BackendModality::STRUCTURELESS;
  simul::TestSetting testSetting(imuParams, visionParams,
                                 estimatorParams, backendParams, "");
  simul::VioSimTestSystem simSystem(checkMSE, checkNEES);
  simSystem.run(testSetting, FLAGS_log_dir);
}

TEST(TFVIO, TrajectoryLabel) {
  int cameraObservationModelId = 1;
  int landmarkModelId = 0;
  double landmarkRadius = 5;
  simul::SimImuParameters imuParams(
      FLAGS_sim_trajectory_label, true, FLAGS_noisyInitialSpeedAndBiases,
      FLAGS_noisyInitialSensorParams, FLAGS_fixImuIntrinsicParams,
      5e-3, 2e-2, 5e-3, 1e-3, 5e-3, FLAGS_sim_sigma_g_c, FLAGS_sim_sigma_gw_c,
      FLAGS_sim_sigma_a_c, FLAGS_sim_sigma_aw_c,
      FLAGS_sim_imu_noise_factor, FLAGS_sim_imu_bias_noise_factor, "BG_BA_TG_TS_TA");
  simul::SimVisionParameters visionParams(
      true, true, simul::SimCameraModelType::EUROC,
      simul::CameraOrientation::Forward, "FXY_CXY", "P_CB",
      FLAGS_fixCameraInternalParams, 2e-2, 0.0, FLAGS_sim_camera_time_offset_sec,
      FLAGS_sim_frame_readout_time_sec, FLAGS_noisyInitialSensorParams,
      simul::LandmarkGridType::FourWalls, landmarkRadius);
  simul::SimEstimatorParameters estimatorParams(
      "TFVIO", swift_vio::EstimatorAlgorithm::TFVIO, FLAGS_num_runs,
      cameraObservationModelId, landmarkModelId, false);
  swift_vio::BackendParams backendParams;
  simul::TestSetting testSetting(imuParams, visionParams,
                                 estimatorParams, backendParams, "");
  simul::VioSimTestSystem simSystem(checkMSE, checkNEES);
  simSystem.run(testSetting, FLAGS_log_dir);
}

TEST(MSCKFWithEuclidean, TrajectoryLabel) {
  int cameraObservationModelId = 0;
  int landmarkModelId = 0;
  double landmarkRadius = 5;
  simul::SimImuParameters imuParams(
      FLAGS_sim_trajectory_label, true, FLAGS_noisyInitialSpeedAndBiases,
      FLAGS_noisyInitialSensorParams, FLAGS_fixImuIntrinsicParams,
      5e-3, 2e-2, 5e-3, 1e-3, 5e-3, FLAGS_sim_sigma_g_c, FLAGS_sim_sigma_gw_c,
      FLAGS_sim_sigma_a_c, FLAGS_sim_sigma_aw_c,
      FLAGS_sim_imu_noise_factor, FLAGS_sim_imu_bias_noise_factor, "BG_BA_TG_TS_TA");
  simul::SimVisionParameters visionParams(
      true, true, simul::SimCameraModelType::EUROC,
      simul::CameraOrientation::Forward, "FXY_CXY", "P_CB",
      FLAGS_fixCameraInternalParams, 2e-2, 0.0, FLAGS_sim_camera_time_offset_sec,
      FLAGS_sim_frame_readout_time_sec, FLAGS_noisyInitialSensorParams,
      simul::LandmarkGridType::FourWalls, landmarkRadius);
  simul::SimEstimatorParameters estimatorParams(
      "MSCKFWithEuclidean", swift_vio::EstimatorAlgorithm::MSCKF, FLAGS_num_runs,
      cameraObservationModelId, landmarkModelId, false);
  swift_vio::BackendParams backendParams;
  simul::TestSetting testSetting(imuParams, visionParams,
                                 estimatorParams, backendParams, "");
  simul::VioSimTestSystem simSystem(checkMSE, checkNEES);
  simSystem.run(testSetting, FLAGS_log_dir);
}

TEST(MSCKFWithPAP, TrajectoryLabel) {
  int cameraObservationModelId = 2;
  int landmarkModelId = 2;
  double landmarkRadius = 5;
  simul::SimImuParameters imuParams(
      FLAGS_sim_trajectory_label, true, FLAGS_noisyInitialSpeedAndBiases,
      FLAGS_noisyInitialSensorParams, FLAGS_fixImuIntrinsicParams,
      5e-3, 2e-2, 5e-3, 1e-3, 5e-3, FLAGS_sim_sigma_g_c, FLAGS_sim_sigma_gw_c,
      FLAGS_sim_sigma_a_c, FLAGS_sim_sigma_aw_c,
      FLAGS_sim_imu_noise_factor, FLAGS_sim_imu_bias_noise_factor, "BG_BA_TG_TS_TA");
  simul::SimVisionParameters visionParams(
      true, true, simul::SimCameraModelType::EUROC,
      simul::CameraOrientation::Forward, "FXY_CXY", "P_CB",
      FLAGS_fixCameraInternalParams, 2e-2, 0.0, FLAGS_sim_camera_time_offset_sec,
      FLAGS_sim_frame_readout_time_sec, FLAGS_noisyInitialSensorParams,
      simul::LandmarkGridType::FourWalls, landmarkRadius);
  simul::SimEstimatorParameters estimatorParams(
      "MSCKFWithPAP", swift_vio::EstimatorAlgorithm::MSCKF, FLAGS_num_runs,
      cameraObservationModelId, landmarkModelId, false);
  swift_vio::BackendParams backendParams;
  simul::TestSetting testSetting(imuParams, visionParams,
                                 estimatorParams, backendParams, "");
  simul::VioSimTestSystem simSystem(checkMSE, checkNEES);
  simSystem.run(testSetting, FLAGS_log_dir);
}

TEST(MSCKFWithReprojectionErrorPAP, TrajectoryLabel) {
  int cameraObservationModelId = 3;
  int landmarkModelId = 2;
  double landmarkRadius = 5;
  simul::SimImuParameters imuParams(
      FLAGS_sim_trajectory_label, true, FLAGS_noisyInitialSpeedAndBiases,
      FLAGS_noisyInitialSensorParams, FLAGS_fixImuIntrinsicParams,
      5e-3, 2e-2, 5e-3, 1e-3, 5e-3, FLAGS_sim_sigma_g_c, FLAGS_sim_sigma_gw_c,
      FLAGS_sim_sigma_a_c, FLAGS_sim_sigma_aw_c,
      FLAGS_sim_imu_noise_factor, FLAGS_sim_imu_bias_noise_factor, "BG_BA_TG_TS_TA");
  simul::SimVisionParameters visionParams(
      true, true, simul::SimCameraModelType::EUROC,
      simul::CameraOrientation::Forward, "FXY_CXY", "P_CB",
      FLAGS_fixCameraInternalParams, 2e-2, 0.0, FLAGS_sim_camera_time_offset_sec,
      FLAGS_sim_frame_readout_time_sec, FLAGS_noisyInitialSensorParams,
      simul::LandmarkGridType::FourWalls, landmarkRadius);
  simul::SimEstimatorParameters estimatorParams(
      "MSCKFWithReprojectionErrorPAP", swift_vio::EstimatorAlgorithm::MSCKF, FLAGS_num_runs,
      cameraObservationModelId, landmarkModelId, false);
  swift_vio::BackendParams backendParams;
  simul::TestSetting testSetting(imuParams, visionParams,
                                 estimatorParams, backendParams, "");
  simul::VioSimTestSystem simSystem(checkMSE, checkNEES);
  simSystem.run(testSetting, FLAGS_log_dir);
}

TEST(MSCKFWithPAP, SquircleBackward) {
  int cameraObservationModelId = 2;
  int landmarkModelId = 2;
  double landmarkRadius = 5;
  simul::SimImuParameters imuParams(
      "Squircle", true, FLAGS_noisyInitialSpeedAndBiases,
      FLAGS_noisyInitialSensorParams, FLAGS_fixImuIntrinsicParams,
      5e-3, 2e-2, 5e-3, 1e-3, 5e-3, FLAGS_sim_sigma_g_c, FLAGS_sim_sigma_gw_c,
      FLAGS_sim_sigma_a_c, FLAGS_sim_sigma_aw_c,
      FLAGS_sim_imu_noise_factor, FLAGS_sim_imu_bias_noise_factor, "BG_BA_TG_TS_TA");
  simul::SimVisionParameters visionParams(
      true, true, simul::SimCameraModelType::EUROC,
      simul::CameraOrientation::Forward, "FXY_CXY", "P_CB",
      FLAGS_fixCameraInternalParams, 2e-2, 0.0, FLAGS_sim_camera_time_offset_sec,
      FLAGS_sim_frame_readout_time_sec, FLAGS_noisyInitialSensorParams,
      simul::LandmarkGridType::FourWalls, landmarkRadius);
  simul::SimEstimatorParameters estimatorParams(
      "MSCKFWithPAP", swift_vio::EstimatorAlgorithm::MSCKF, FLAGS_num_runs,
      cameraObservationModelId, landmarkModelId, false);
  swift_vio::BackendParams backendParams;
  simul::TestSetting testSetting(imuParams, visionParams,
                                 estimatorParams, backendParams, "");
  simul::VioSimTestSystem simSystem(checkMSE, checkNEES);
  simSystem.run(testSetting, FLAGS_log_dir);
}

TEST(MSCKFWithPAP, SquircleSideways) {
  int cameraObservationModelId = 2;
  int landmarkModelId = 2;
  double landmarkRadius = 5;
  simul::SimImuParameters imuParams(
      "Squircle", true, FLAGS_noisyInitialSpeedAndBiases,
      FLAGS_noisyInitialSensorParams, FLAGS_fixImuIntrinsicParams,
      5e-3, 2e-2, 5e-3, 1e-3, 5e-3, FLAGS_sim_sigma_g_c, FLAGS_sim_sigma_gw_c,
      FLAGS_sim_sigma_a_c, FLAGS_sim_sigma_aw_c,
      FLAGS_sim_imu_noise_factor, FLAGS_sim_imu_bias_noise_factor, "BG_BA_TG_TS_TA");
  simul::SimVisionParameters visionParams(
      true, true, simul::SimCameraModelType::EUROC,
      simul::CameraOrientation::Right, "FXY_CXY", "P_CB",
      FLAGS_fixCameraInternalParams, 2e-2, 0.0, FLAGS_sim_camera_time_offset_sec,
      FLAGS_sim_frame_readout_time_sec, FLAGS_noisyInitialSensorParams,
      simul::LandmarkGridType::FourWalls, landmarkRadius);
  simul::SimEstimatorParameters estimatorParams(
      "MSCKFWithPAP", swift_vio::EstimatorAlgorithm::MSCKF, FLAGS_num_runs,
      cameraObservationModelId, landmarkModelId, false);
  swift_vio::BackendParams backendParams;
  simul::TestSetting testSetting(imuParams, visionParams,
                                 estimatorParams, backendParams, "");
  simul::VioSimTestSystem simSystem(checkMSE, checkNEES);
  simSystem.run(testSetting, FLAGS_log_dir);
}

TEST(MSCKFWithEpipolarConstraint, TrajectoryLabel) {
  int cameraObservationModelId = 0;
  int landmarkModelId = FLAGS_sim_landmark_model;
  double landmarkRadius = 5;
  simul::SimImuParameters imuParams(
      FLAGS_sim_trajectory_label, true, FLAGS_noisyInitialSpeedAndBiases,
      FLAGS_noisyInitialSensorParams, FLAGS_fixImuIntrinsicParams,
      5e-3, 2e-2, 5e-3, 1e-3, 5e-3, FLAGS_sim_sigma_g_c, FLAGS_sim_sigma_gw_c,
      FLAGS_sim_sigma_a_c, FLAGS_sim_sigma_aw_c,
      FLAGS_sim_imu_noise_factor, FLAGS_sim_imu_bias_noise_factor, "BG_BA_TG_TS_TA");
  simul::SimVisionParameters visionParams(
      true, true, simul::SimCameraModelType::EUROC,
      simul::CameraOrientation::Forward, "FXY_CXY", "P_CB",
      FLAGS_fixCameraInternalParams, 2e-2, 0.0, FLAGS_sim_camera_time_offset_sec,
      FLAGS_sim_frame_readout_time_sec, FLAGS_noisyInitialSensorParams,
      simul::LandmarkGridType::FourWalls, landmarkRadius);
  simul::SimEstimatorParameters estimatorParams(
      "MSCKFWithEpipolarConstraint", swift_vio::EstimatorAlgorithm::MSCKF, FLAGS_num_runs,
      cameraObservationModelId, landmarkModelId, true);
  swift_vio::BackendParams backendParams;
  simul::TestSetting testSetting(imuParams, visionParams,
                                 estimatorParams, backendParams, "");
  simul::VioSimTestSystem simSystem(checkMSE, checkNEES);
  simSystem.run(testSetting, FLAGS_log_dir);
}

TEST(MSCKFWithEpipolarConstraint, CircleFarPoints) {
  int cameraObservationModelId = 0;
  int landmarkModelId = FLAGS_sim_landmark_model;
  double landmarkRadius = 50;
  simul::SimImuParameters imuParams(
      "Circle", true, FLAGS_noisyInitialSpeedAndBiases,
      FLAGS_noisyInitialSensorParams, FLAGS_fixImuIntrinsicParams,
      5e-3, 2e-2, 5e-3, 1e-3, 5e-3, FLAGS_sim_sigma_g_c, FLAGS_sim_sigma_gw_c,
      FLAGS_sim_sigma_a_c, FLAGS_sim_sigma_aw_c,
      FLAGS_sim_imu_noise_factor, FLAGS_sim_imu_bias_noise_factor, "BG_BA_TG_TS_TA");
  simul::SimVisionParameters visionParams(
      true, true, simul::SimCameraModelType::EUROC,
      simul::CameraOrientation::Forward, "FXY_CXY", "P_CB",
      FLAGS_fixCameraInternalParams,
      2e-2, 0.0, FLAGS_sim_camera_time_offset_sec,
      FLAGS_sim_frame_readout_time_sec, FLAGS_noisyInitialSensorParams,
      simul::LandmarkGridType::Cylinder, landmarkRadius);
  simul::SimEstimatorParameters estimatorParams(
      "MSCKFWithEpipolarConstraint", swift_vio::EstimatorAlgorithm::MSCKF, FLAGS_num_runs,
      cameraObservationModelId, landmarkModelId, true);
  swift_vio::BackendParams backendParams;
  simul::TestSetting testSetting(imuParams, visionParams,
                                 estimatorParams, backendParams, "");
  simul::VioSimTestSystem simSystem(checkMSE, checkNEES);
  simSystem.run(testSetting, FLAGS_log_dir);
}
