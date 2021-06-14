#ifndef SIMUL_GFLAGS_HPP
#define SIMUL_GFLAGS_HPP
#include <gflags/gflags.h>

DECLARE_string(sim_real_data_dir);

DECLARE_bool(fixCameraInternalParams);

DECLARE_bool(fixImuIntrinsicParams);

DECLARE_bool(noisyInitialSpeedAndBiases);

DECLARE_bool(noisyInitialSensorParams);

DECLARE_bool(sim_compute_OKVIS_NEES);

DECLARE_int32(num_runs);

DECLARE_double(sim_camera_time_offset_sec);

DECLARE_double(sim_frame_readout_time_sec);

DECLARE_double(sim_sigma_g_c);

DECLARE_double(sim_sigma_a_c);

DECLARE_double(sim_sigma_gw_c);

DECLARE_double(sim_sigma_aw_c);

DECLARE_double(sim_imu_noise_factor);

DECLARE_double(sim_imu_bias_noise_factor);

DECLARE_string(sim_trajectory_label);

DECLARE_int32(sim_landmark_model);

DECLARE_int32(minTrackLengthForSlam);

DECLARE_int32(maxHibernationFrames);

DECLARE_int32(maxMatchKeyframes);

DECLARE_bool(allKeyframe);

DECLARE_bool(useMahalanobis);

DECLARE_double(maxPositionRmse);

#endif // SIMUL_GFLAGS_HPP
