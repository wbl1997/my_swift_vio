#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def euroc_stasis_swiftvio_options():
    """
    Select the five MH sessions of EuRoC dataset and examine the drifts of sliding window filters in standstills.
    :return:
    """
    # case 1, KSWF (keyframe-based feature tracking, landmark in state)
    # case 2, SWF (framewise feature tracking, landmark in state)
    # case 3: SL-KSWF (keyframe-based feature tracking, landmark not in state)
    # case 4. SL-SWF (framewise feature tracking, landmark not in state)
    # case 5. OKVIS (keyframe-based feature tracking, landmark in state)
    algo_option_templates = {
        'KSWF': {"algo_code": "HybridFilter",
                 "extra_gflags": "--publish_via_ros=false",
                 "displayImages": "false",
                 "monocular_input": 1,
                 "numImuFrames": 4,
                 "sigma_absolute_translation": 0.02,
                 "sigma_absolute_orientation": 0.01,
                 "model_type": "BG_BA",
                 'projection_opt_mode': 'FIXED',
                 "extrinsic_opt_mode_main_camera": "P_CB",
                 "extrinsic_opt_mode_other_camera": "P_C0C_Q_C0C",
                 "sigma_td": 0.05,
                 "sigma_g_c": 12.0e-4 * 4,
                 "sigma_a_c": 8.0e-3 * 4,
                 "sigma_gw_c": 4.0e-6 * 4,
                 "sigma_aw_c": 4.0e-5 * 4,
                 "loop_closure_method": 0},
    }

    config_name_to_diffs = {
        ('KSWF', 'KSWF'): {},
        ('SWF', 'KSWF'): {"featureTrackingMethod": 2, },
        ('SL-KSWF', "KSWF"): {"algo_code": "MSCKF", },
        ('SL-SWF', 'KSWF'): {"algo_code": "MSCKF", "featureTrackingMethod": 2, },
        ('OKVIS', 'KSWF'): {
            "algo_code": "OKVIS",
            "numImuFrames": 3,
            "sigma_c_relative_translation": 1.0e-6,
            "sigma_c_relative_orientation": 1.0e-6,
            "timeLimit": -1,
        }
    }
    return algo_option_templates, config_name_to_diffs


def euroc_stasis_openvins_options():
    algo_option_templates = {
        'OpenVINS': {"algo_code": "OpenVINS",
                     "launch_file": "pgeneva_serial_eth.launch",
                     "gyroscope_noise_density": 1.6968e-04,
                     "gyroscope_random_walk": 1.9393e-05,
                     "accelerometer_noise_density": 2.0000e-3,
                     "accelerometer_random_walk": 3.0000e-3,
                     "calib_cam_extrinsics": "false",
                     "calib_cam_intrinsics": "false",
                     "calib_cam_timeoffset": "false",
                     },
    }

    config_name_to_diffs = {
        ('MSCKF-Mono', 'OpenVINS'): {
            "max_cameras": 1,
            "use_stereo": 'false',
            "max_slam": "0",
            "max_slam_in_update": "0",
        },
        ('SLAM-Mono', 'OpenVINS'): {
            "max_cameras": 1,
            "use_stereo": 'false',
            "max_slam": "75",
            "max_slam_in_update": "25",
        },
    }
    return algo_option_templates, config_name_to_diffs


def euroc_stasis_msckfmono_options():
    algo_option_templates = {
        'MSCKFMono': {"algo_code": "MSCKFMono",
                      "launch_file": "euroc.launch",
                      },
    }

    config_name_to_diffs = {
        ('MSCKFMono', 'MSCKFMono'): {},
    }
    return algo_option_templates, config_name_to_diffs


def euroc_stasis_rovio_options():
    algo_option_templates = {
        'ROVIO': {"algo_code": "ROVIO",
                  "launch_file": "rovio_rosbag_node.launch",
                  },
    }

    config_name_to_diffs = {
        ('ROVIOMono', 'ROVIO'): { "numcameras": 1},
    }
    return algo_option_templates, config_name_to_diffs
