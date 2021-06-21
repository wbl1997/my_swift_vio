#include <simul/gflags.hpp>


DEFINE_string(sim_real_data_dir, "", "Directory of simulated data in maplab csv format from real data!");

DEFINE_bool(fixCameraInternalParams, true,
            "Set the variance of the camera internal parameters (including intrinsics and temporal parameters) zero."
            " Otherwise, these parameters will be estimated by the filter.");

DEFINE_bool(fixImuIntrinsicParams, true,
            "Set the variance of the IMU augmented intrinsic parameters zero."
            " Otherwise, these parameters will be estimated by the filter.");

DEFINE_bool(
    noisyInitialSpeedAndBiases, true,
    "add noise to the initial value of velocity, gyro bias, accelerometer "
    "bias which is used to initialize an estimator.");

DEFINE_bool(noisyInitialSensorParams, false,
            "add noise to the initial value of sensor parameters, including "
            "camera extrinsic, intrinsic "
            "and temporal parameters, and IMU parameters except for biases "
            "which is used to initialize an estimator. But the noise may be "
            "zero by setting e.g. fixImuIntrinsicParams");

DEFINE_bool(sim_compute_OKVIS_NEES, false,
            "False to analyze OKVIS accuracy, true to analyze OKVIS consistency!");

DEFINE_int32(num_runs, 5, "How many times to run one simulation?");

DEFINE_double(
    sim_camera_time_offset_sec, 0.0,
    "image raw timestamp + camera time offset = image time in imu clock");

DEFINE_double(sim_frame_readout_time_sec, 0.0,
              "readout time for one frame in secs");

DEFINE_double(sim_sigma_g_c, 1.2e-3, "simulated gyro noise density");

DEFINE_double(sim_sigma_a_c, 8e-3, "simulated accelerometer noise density");

DEFINE_double(sim_sigma_gw_c, 2e-5, "simulated gyro bias noise density");

DEFINE_double(sim_sigma_aw_c, 5.5e-5, "simulated accelerometer bias noise density");

DEFINE_double(sim_imu_noise_factor, 1.0,
              "weaken the IMU noise added to IMU readings by this factor");

DEFINE_double(sim_imu_bias_noise_factor, 1.0,
              "weaken the IMU BIAS noise added to IMU readings by this factor");

DEFINE_string(sim_trajectory_label, "WavyCircle",
              "Ball has the most exciting motion, wavycircle is general");

DEFINE_int32(sim_landmark_model, 1,
             "Landmark model 0 for global homogeneous point, 1 for anchored "
             "inverse depth point, 2 for parallax angle parameterization");


DEFINE_int32(minTrackLengthForSlam, 6,
             "A feature has to meet the min track length to be included in the "
             "state vector.");

DEFINE_int32(maxHibernationFrames, 3,
             "A feature hibernate longer than or equal to this will be removed from the state vector.");

DEFINE_int32(maxMatchKeyframes, 3, "Max number of keyframes to match the current frame bundle to.");

DEFINE_bool(allKeyframe, false,
            "Treat all frames as keyframes. Paradoxically, this means using no "
            "keyframe scheme.");

DEFINE_bool(simUseMahalanobis, false,
            "Use Mahalanobis gating test to remove outliers.");

DEFINE_double(maxPositionRmse, 100, "If the final position RMSE is greater, then the run will be considered failed.");

DEFINE_string(
    sim_distortion_type, "RadialTangentialDistortion",
    "Distortion type for the simulated camera model when external sim data are "
    "not used. Candidate examples: RadialTangentialDistortion, EquidistantDistortion, FovDistortion");
