#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def tumrs_calibrated_test_options():
    """
    Compare okvis and KSWF on TUM RS calibrated 640 dataset.
    Self-calibration should be disabled by default according to the passed configuration yaml.
    """
    algo_option_templates = {
        'KSWF': {"algo_code": "HybridFilter",
                 "extra_gflags": "--publish_via_ros=false",
                 "displayImages": "false",
                 "monocular_input": 0,
                 "numImuFrames": 5,
                 "loop_closure_method": 0},
    }

    config_name_to_diffs = {
        ('KSWF_03', 'KSWF'): {"sigma_g_c": 0.004 * 0.3,
                              "sigma_a_c": 0.07 * 0.3,
                              "sigma_gw_c": 0.000044 * 0.3,
                              "sigma_aw_c": 0.00172 * 0.3, },
        ('OKVIS', 'KSWF'): {
            "algo_code": "OKVIS",
            "numImuFrames": 3,
            "timeLimit": -1,
            "extrinsic_opt_mode_main_camera": "p_BC_q_BC",
            "extrinsic_opt_mode_other_camera": "p_BC_q_BC",
        },
    }
    return algo_option_templates, config_name_to_diffs


def tumrs_raw_test_options():
    """
    Compare okvis and KSWF on TUM RS raw 640 dataset
    Full self-calibration should be enabled by default according to the passed configuration yaml.
    """
    algo_option_templates = {
        'KSWF': {"algo_code": "MSCKF",
                 "extra_gflags": "--publish_via_ros=false",
                 "displayImages": "false",
                 "monocular_input": 0,
                 "loop_closure_method": 0},
    }

    config_name_to_diffs = {
        ('SL-KSWF_03', 'KSWF'): {
            "algo_code": "MSCKF",
            "numImuFrames": 5,
            "sigma_g_c": 0.004 * 0.3,
            "sigma_a_c": 0.07 * 0.3,
            "sigma_gw_c": 0.000044 * 0.3,
            "sigma_aw_c": 0.00172 * 0.3, },
        ('KSWF_03', 'KSWF'): {
            "sigma_g_c": 0.004 * 0.3,
            "sigma_a_c": 0.07 * 0.3,
            "sigma_gw_c": 0.000044 * 0.3,
            "sigma_aw_c": 0.00172 * 0.3, },
        ('KSWF_03_cal_cam', 'KSWF'): {
            "sigma_g_c": 0.004 * 0.3,
            "sigma_a_c": 0.07 * 0.3,
            "sigma_gw_c": 0.000044 * 0.3,
            "sigma_aw_c": 0.00172 * 0.3,

            "sigma_TGElement": 0.0,
            "sigma_TSElement": 0.0,
            "sigma_TAElement": 0.0, },
        ('KSWF_03_cal_imu', 'KSWF'): {
            "sigma_g_c": 0.004 * 0.3,
            "sigma_a_c": 0.07 * 0.3,
            "sigma_gw_c": 0.000044 * 0.3,
            "sigma_aw_c": 0.00172 * 0.3,

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
        ('KSWF_03_fix_all', "KSWF"): {
            "sigma_g_c": 0.004 * 0.3,
            "sigma_a_c": 0.07 * 0.3,
            "sigma_gw_c": 0.000044 * 0.3,
            "sigma_aw_c": 0.00172 * 0.3,

            "sigma_absolute_translation": 0.0,
            "sigma_absolute_orientation": 0.0,
            "sigma_c_relative_translation": 0.0,
            "sigma_c_relative_orientation": 0.0,
            "sigma_tr": 0.0,
            "sigma_td": 0.0,
            "sigma_focal_length": 0.0,
            "sigma_principal_point": 0.0,
            "sigma_distortion": [0.0, 0.0, 0.0, 0.0, 0.0],

            "sigma_TGElement": 0.0,
            "sigma_TSElement": 0.0,
            "sigma_TAElement": 0.0,
        },
        ('OKVIS', 'KSWF'): {
            "algo_code": "OKVIS",
            "numImuFrames": 3,
            "timeLimit": -1,
            "landmarkModelId": 0,
            "extrinsic_opt_mode_main_camera": "p_BC_q_BC",
            "extrinsic_opt_mode_other_camera": "p_BC_q_BC",
        }
    }
    return algo_option_templates, config_name_to_diffs
