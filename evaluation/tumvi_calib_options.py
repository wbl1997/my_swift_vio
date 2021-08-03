#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# The best IMU parameters for stereo sliding window filters on TUM VI raw data.
# Note these noise parameters are inflated relative to those parameters for the TUM VI calibrated data.
SWF_TUMVI_RAW_IMU_PARAMETERS = {"sigma_g_c": 0.004,
                                "sigma_a_c": 0.07,
                                "sigma_gw_c": 4.4e-5,
                                "sigma_aw_c": 1.72e-3}

OKVIS_TUMVI_RAW_IMU_PARAMETERS = {"sigma_g_c": 0.004 * 5,
                                  "sigma_a_c": 0.07 * 5,
                                  "sigma_gw_c": 4.4e-5 * 2,
                                  "sigma_aw_c": 1.72e-3 * 2}

# The best IMU parameters for stereo sliding window filters on TUM VI calibrated data are found by a search.
#       &      Translation (\%) &  Rotation (deg/meter)
# SL-KSWF_n_01_01 &     31.934 &  0.135
# SL-KSWF_n_02_025 &     41.982 &  0.120
# SL-KSWF_n_0025_005 &     84470.155 &  0.313
# SL-KSWF_n_005_005 &     786.133 &  0.260
# SL-KSWF_n_005_01 &     42.373 &  0.151
SWF_TUMVI_IMU_PARAMETERS = {"sigma_g_c": 0.004 * 0.2,
                            "sigma_a_c": 0.07 * 0.2,
                            "sigma_gw_c": 4.4e-5 * 0.5,
                            "sigma_aw_c": 1.72e-3 * 0.5}

# The best IMU parameters for monocular sliding window filters on TUM VI calibrated data are found by a search.
#       &      Translation (\%) &  Rotation (deg/meter)
# SL-KSWF_005_01 &     37578.149 &  0.258
# SL-KSWF_01_01 &     70.471 &  0.186
# SL-KSWF_01_025 &     47.796 &  0.117
# SL-KSWF_02_025 &     42.107 &  0.118
SWF_TUMVI_MONO_IMU_PARAMETERS = {
    "sigma_g_c": 0.004 * 0.4,
    "sigma_a_c": 0.07 * 0.4,
    "sigma_gw_c": 4.4e-5 * 0.5,
    "sigma_aw_c": 1.72e-3 * 0.5
}


def tumvi_calibrated_swiftvio_options():
    """
    Compare okvis and KSWF on TUM VI calibrated 512 dataset
    Self-calibration should be disabled by default according to the passed configuration file.
    """
    algo_option_templates = {
        'KSWF': {"algo_code": "HybridFilter",
                 "extra_gflags": "--publish_via_ros=false",
                 "displayImages": "false",
                 "monocular_input": 0,
                 "numImuFrames": 5,
                 "loop_closure_method": 0,
                 "sigma_TGElement": 0.0,
                 "sigma_TSElement": 0.0,
                 "sigma_TAElement": 0.0,
                 "model_type": "BG_BA",
                 "extrinsic_opt_mode_main_camera": "p_BC_q_BC",
                 "extrinsic_opt_mode_other_camera": "p_BC_q_BC",
                 'projection_opt_mode': 'FX_CXY',
                 },
    }

    config_name_to_diffs = {
        ('KSWF-mono', 'KSWF'): {"sigma_g_c": 0.00016,
                                "sigma_a_c": 0.0028,
                                "sigma_gw_c": 2.2e-5,
                                "sigma_aw_c": 8.6e-4,
                                "monocular_input": 1,
                                "featureTrackingMethod": 0,
                                "stereoMatchWithEpipolarCheck": "true",
                                "epipolarDistanceThreshold": 5,
                                "minTrackLengthForSlam": 7,
                                "threshold": 35.0, },
        ('KSWF', 'KSWF'): {"sigma_g_c": 0.00016,
                           "sigma_a_c": 0.0028,
                           "sigma_gw_c": 2.2e-5,
                           "sigma_aw_c": 8.6e-4,
                           "featureTrackingMethod": 0,
                           "stereoMatchWithEpipolarCheck": "true",
                           "epipolarDistanceThreshold": 5,
                           "minTrackLengthForSlam": 7,
                           "threshold": 35.0, },
        ('SL-KSWF', 'KSWF'): {"algo_code": "MSCKF",
                              "sigma_g_c": 0.00016,
                              "sigma_a_c": 0.0028,
                              "sigma_gw_c": 2.2e-5,
                              "sigma_aw_c": 8.6e-4,
                              "featureTrackingMethod": 2,
                              "stereoMatchWithEpipolarCheck": "true",
                              "epipolarDistanceThreshold": 5,
                              "minTrackLengthForSlam": 7,
                              "threshold": 35.0, },
        ('OKVIS', 'KSWF'): {
            "algo_code": "OKVIS",
            "numImuFrames": 3,
            "timeLimit": -1,
            "landmarkModelId": 0,
            "extrinsic_opt_mode_main_camera": "p_BC_q_BC",
            "extrinsic_opt_mode_other_camera": "p_BC_q_BC",
        },
    }
    return algo_option_templates, config_name_to_diffs


def tumvi_raw_swiftvio_options():
    """
    Compare okvis and KSWF on TUM VI raw 512 dataset
    Full self-calibration should be enabled by default according to the passed configuration file.
    """
    algo_option_templates = {
        'KSWF': {"algo_code": "HybridFilter",
                 "extra_gflags": "--publish_via_ros=false --skip_first_seconds=0.0",
                 "displayImages": "false",
                 "monocular_input": 0,
                 "numImuFrames": 5,
                 "maxInStateLandmarks": 70,
                 "maxMarginalizedLandmarks": 50,
                 "maxHibernationFrames": 5,
                 "featureTrackingMethod": 0,
                 "extrinsic_opt_mode_main_camera": "p_CB",
                 "extrinsic_opt_mode_other_camera": "p_C0C_q_C0C",
                 "loop_closure_method": 0},
    }

    config_name_to_diffs = {
        ('SL-KSWF', 'KSWF'): {
            "algo_code": "MSCKF",
            **SWF_TUMVI_RAW_IMU_PARAMETERS,
        },
        ('KSWF-mono', 'KSWF'): {
            **SWF_TUMVI_MONO_IMU_PARAMETERS,
            "monocular_input": 1,
        },
        # This is the best performing setting for KSWF.
        ('KSWF', 'KSWF'): {
            **SWF_TUMVI_RAW_IMU_PARAMETERS,
        },
        ('KSWF_2', 'KSWF'): {
            **SWF_TUMVI_RAW_IMU_PARAMETERS,
            "featureTrackingMethod": 2,
        },
        ('KSWF_1_05', 'KSWF'): {
            "sigma_g_c": 0.004,
            "sigma_a_c": 0.07,
            "sigma_gw_c": 4.4e-5 * 0.5,
            "sigma_aw_c": 1.72e-3 * 0.5
        },
        # This is almost the best performing setting for OKVIS.
        ('OKVIS', 'KSWF'): {
            "algo_code": "OKVIS",
            "numImuFrames": 3,
            "timeLimit": -1,
            "landmarkModelId": 0,
            "extrinsic_opt_mode_main_camera": "p_BC_q_BC",
            "extrinsic_opt_mode_other_camera": "p_BC_q_BC",
            **OKVIS_TUMVI_RAW_IMU_PARAMETERS},
        # A potential candidate for OKVIS.
        ('OKVIS_1_05', 'KSWF'): {
            "algo_code": "OKVIS",
            "numImuFrames": 3,
            "timeLimit": -1,
            "landmarkModelId": 0,
            "extrinsic_opt_mode_main_camera": "p_BC_q_BC",
            "extrinsic_opt_mode_other_camera": "p_BC_q_BC",
            "sigma_g_c": 0.004 * 5,
            "sigma_a_c": 0.07 * 5,
            "sigma_gw_c": 4.4e-5 * 2 * 0.5,
            "sigma_aw_c": 1.72e-3 * 2 * 0.5
        },
        # The best performing setting for OKVIS.
        ('OKVIS_05_1', 'KSWF'): {
            "algo_code": "OKVIS",
            "numImuFrames": 3,
            "timeLimit": -1,
            "landmarkModelId": 0,
            "extrinsic_opt_mode_main_camera": "p_BC_q_BC",
            "extrinsic_opt_mode_other_camera": "p_BC_q_BC",
            "sigma_g_c": 0.004 * 5 * 0.5,
            "sigma_a_c": 0.07 * 5 * 0.5,
            "sigma_gw_c": 4.4e-5 * 2,
            "sigma_aw_c": 1.72e-3 * 2
        },
        ('KSWF_cal_cam', 'KSWF'): {
            **SWF_TUMVI_RAW_IMU_PARAMETERS,
            "sigma_TGElement": 0.0,
            "sigma_TSElement": 0.0,
            "sigma_TAElement": 0.0,
            "model_type": "BG_BA",
        },
        ('KSWF_cal_imu', 'KSWF'): {
            **SWF_TUMVI_RAW_IMU_PARAMETERS,
            "sigma_absolute_translation": 0.0,
            "sigma_absolute_orientation": 0.0,
            "sigma_c_relative_translation": 0.0,
            "sigma_c_relative_orientation": 0.0,
            "sigma_tr": 0.0,
            "sigma_td": 0.0,
            "sigma_focal_length": 0.0,
            "sigma_principal_point": 0.0,
            "sigma_distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        ('KSWF_fix_all', "KSWF"): {
            **SWF_TUMVI_RAW_IMU_PARAMETERS,

            "sigma_absolute_translation": 0.0,
            "sigma_absolute_orientation": 0.0,
            "sigma_c_relative_translation": 0.0,
            "sigma_c_relative_orientation": 0.0,
            "sigma_tr": 0.0,
            "sigma_td": 0.0,
            "sigma_focal_length": 0.0,
            "sigma_principal_point": 0.0,
            "sigma_distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
            "extrinsic_opt_mode_main_camera": "p_BC_q_BC",
            "extrinsic_opt_mode_other_camera": "p_BC_q_BC",
            "sigma_TGElement": 0.0,
            "sigma_TSElement": 0.0,
            "sigma_TAElement": 0.0,
            "model_type": "BG_BA",
        },
    }
    return algo_option_templates, config_name_to_diffs


def tumvi_calibrated_vinsmono_options():
    algo_option_templates = {
        'VINS': {"algo_code": "VINSMono",
                 "acc_n": 0.04,
                 "gyr_n": 0.004,
                 "acc_w": 0.0004,
                 "gyr_w": 2.0e-5,
                 "loop_closure": 0,
                 "output_path": "",
                 "launch_file": "tum.launch",
                 "estimate_extrinsic": 0,
                 "estimate_td": 0},
    }

    config_name_to_diffs = {
        ('VINS', 'VINS'): {},
    }
    return algo_option_templates, config_name_to_diffs


def tumvi_raw_vinsmono_options():
    algo_option_templates = {
        'VINS': {"algo_code": "VINSMono",
                 "acc_n": 0.04,
                 "gyr_n": 0.004,
                 "acc_w": 0.0004,
                 "gyr_w": 2.0e-5,
                 "loop_closure": 0,
                 "output_path": "",
                 "launch_file": "tum.launch",
                 "estimate_extrinsic": 1,
                 "estimate_td": 1},
    }

    config_name_to_diffs = {
        ('VINS', 'VINS'): {},
        ('VINS_1_2', 'VINS'): {
            "acc_n": 0.04,
            "gyr_n": 0.004,
            "acc_w": 0.0004 * 2,
            "gyr_w": 2.0e-5 * 2,
        },
        ('VINS_2_1', 'VINS'): {
            "acc_n": 0.04 * 2,
            "gyr_n": 0.004 * 2,
            "acc_w": 0.0004,
            "gyr_w": 2.0e-5,
        },
        ('VINS_2_2', 'VINS'): {
            "acc_n": 0.04 * 2,
            "gyr_n": 0.004 * 2,
            "acc_w": 0.0004 * 2,
            "gyr_w": 2.0e-5 * 2,
        },
        ('VINS_5_2', 'VINS'): {
            "acc_n": 0.04 * 5,
            "gyr_n": 0.004 * 5,
            "acc_w": 0.0004 * 2,
            "gyr_w": 2.0e-5 * 2,
        },
        ('VINS_5_5', 'VINS'): {
            "acc_n": 0.04 * 5,
            "gyr_n": 0.004 * 5,
            "acc_w": 0.0004 * 5,
            "gyr_w": 2.0e-5 * 5,
        },
        ('VINS_2_5', 'VINS'): {
            "acc_n": 0.04 * 2,
            "gyr_n": 0.004 * 2,
            "acc_w": 0.0004 * 5,
            "gyr_w": 2.0e-5 * 5,
        },
    }
    return algo_option_templates, config_name_to_diffs


def tumvi_calibrated_openvins_options():
    algo_option_templates = {
        'OpenVINS': {"algo_code": "OpenVINS",
                     "launch_file": "pgeneva_serial_tum.launch",
                     "gyroscope_noise_density": "0.00016",
                     "gyroscope_random_walk": "0.000022",
                     "accelerometer_noise_density": "0.0028",
                     "accelerometer_random_walk": "0.00086"
                     },
    }

    config_name_to_diffs = {
        ('OpenVINS-Mono', 'OpenVINS'): {
            "max_cameras": 1,
            "use_stereo": 'false'
        },
        ('OpenVINS-Stereo', 'OpenVINS'): {
            "max_cameras": 2,
            "use_stereo": 'true'
        },
    }
    return algo_option_templates, config_name_to_diffs


def tumvi_openvins_calib_cam_options():
    algo_option_templates = {
        'OpenVINS': {"algo_code": "OpenVINS",
                     "launch_file": "pgeneva_serial_tum_inaccurate.launch",
                     "gyroscope_noise_density": "0.00016",
                     "gyroscope_random_walk": "0.000022",
                     "accelerometer_noise_density": "0.0028",
                     "accelerometer_random_walk": "0.00086"
                     },
    }

    config_name_to_diffs = {
        ('OpenVINS-Mono', 'OpenVINS'): {
            "max_cameras": 1,
            "use_stereo": 'false'
        },
        ('OpenVINS-Stereo-Desc', 'OpenVINS'): {
            "max_cameras": 2,
            "use_stereo": 'true',
            "use_klt": "false",
        },
        ('OpenVINS-Stereo-KLT', 'OpenVINS'): {
            "max_cameras": 2,
            "use_stereo": 'true',
            "use_klt": "true",
        },
    }
    return algo_option_templates, config_name_to_diffs


def tumvi_raw_openvins_options():
    algo_option_templates = {
        'OpenVINS': {"algo_code": "OpenVINS",
                     "launch_file": "pgeneva_serial_tum_inaccurate.launch",
                     "gyroscope_noise_density": 0.00016,
                     "accelerometer_noise_density": 0.0028,
                     "gyroscope_random_walk": 0.000022,
                     "accelerometer_random_walk": 0.00086
                     },
    }

    config_name_to_diffs = {
        ('OpenVINS', 'OpenVINS'): {
            "max_cameras": 2,
            "use_stereo": 'true',
        },
        ('OpenVINS_2_1', 'OpenVINS'): {
            "max_cameras": 2,
            "use_stereo": 'true',
            "gyroscope_noise_density": 0.00016 * 2,
            "accelerometer_noise_density": 0.0028 * 2,
            "gyroscope_random_walk": 0.000022,
            "accelerometer_random_walk": 0.00086
        },
        ('OpenVINS_2_2', 'OpenVINS'): {
            "max_cameras": 2,
            "use_stereo": 'true',
            "gyroscope_noise_density": 0.00016 * 2,
            "accelerometer_noise_density": 0.0028 * 2,
            "gyroscope_random_walk": 0.000022 * 2,
            "accelerometer_random_walk": 0.00086 * 2
        },
        ('OpenVINS_2_5', 'OpenVINS'): {
            "max_cameras": 2,
            "use_stereo": 'true',
            "gyroscope_noise_density": 0.00016 * 2,
            "accelerometer_noise_density": 0.0028 * 2,
            "gyroscope_random_walk": 0.000022 * 5,
            "accelerometer_random_walk": 0.00086 * 5
        },
        ('OpenVINS_5_2', 'OpenVINS'): {
            "max_cameras": 2,
            "use_stereo": 'true',
            "gyroscope_noise_density": 0.00016 * 5,
            "accelerometer_noise_density": 0.0028 * 5,
            "gyroscope_random_walk": 0.000022 * 2,
            "accelerometer_random_walk": 0.00086 * 2
        },
        ('OpenVINS_5_5', 'OpenVINS'): {
            "max_cameras": 2,
            "use_stereo": 'true',
            "gyroscope_noise_density": 0.00016 * 5,
            "accelerometer_noise_density": 0.0028 * 5,
            "gyroscope_random_walk": 0.000022 * 5,
            "accelerometer_random_walk": 0.00086 * 5
        },
        ('OpenVINS_10_5', 'OpenVINS'): {
            "max_cameras": 2,
            "use_stereo": 'true',
            "gyroscope_noise_density": 0.00016 * 10,
            "accelerometer_noise_density": 0.0028 * 10,
            "gyroscope_random_walk": 0.000022 * 5,
            "accelerometer_random_walk": 0.00086 * 5
        },
        ('OpenVINS_10_10', 'OpenVINS'): {
            "max_cameras": 2,
            "use_stereo": 'true',
            "gyroscope_noise_density": 0.00016 * 10,
            "accelerometer_noise_density": 0.0028 * 10,
            "gyroscope_random_walk": 0.000022 * 10,
            "accelerometer_random_walk": 0.00086 * 10
        },
        ('OpenVINS-Mono', 'OpenVINS'): {
            "max_cameras": 1,
            "use_stereo": 'false'
        },
    }
    return algo_option_templates, config_name_to_diffs
