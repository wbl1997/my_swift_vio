#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def tumrs_calibrated_test_options():
    """
    Compare okvis and KSWF on TUM RS calibrated 512 dataset
    Self-calibration is disabled.
    """
    algo_option_templates = {
        'KSWF': {"algo_code": "HybridFilter",
                 "extra_gflags": "--publish_via_ros=false",
                 "displayImages": "false",
                 "monocular_input": 0,
                 "loop_closure_method": 0,
                 'use_nominal_calib_value': False},
    }

    config_name_to_diffs = {
        # ('KSWF', 'KSWF'): {},
        ('KSWF_01', 'KSWF'): {
            "sigma_g_c": 0.004 * 0.1,
            "sigma_a_c": 0.07 * 0.1,
            "sigma_gw_c": 0.000044,
            "sigma_aw_c": 0.00172,},
        # ('KSWF_03', 'KSWF'): {"sigma_g_c": 0.004 * 0.3,
        #                       "sigma_a_c": 0.07 * 0.3,
        #                       "sigma_gw_c": 0.000044,
        #                       "sigma_aw_c": 0.00172, },
        # ('KSWF_3', "KSWF"): {"sigma_g_c": 0.004 * 3,
        #                      "sigma_a_c": 0.07 * 3,
        #                      "sigma_gw_c": 0.000044,
        #                      "sigma_aw_c": 0.00172, },
        # ('KSWF_10', 'KSWF'): {"sigma_g_c": 0.004 * 10,
        #                       "sigma_a_c": 0.07 * 10,
        #                       "sigma_gw_c": 0.000044,
        #                       "sigma_aw_c": 0.00172,},
        ('OKVIS_01', 'KSWF'): {
            "algo_code": "OKVIS",
            "numImuFrames": 3,
            "timeLimit": -1,
            "sigma_g_c": 0.004 * 0.1,
            "sigma_a_c": 0.07 * 0.1,
            "sigma_gw_c": 0.000044,
            "sigma_aw_c": 0.00172, },
        ('OKVIS_03', 'KSWF'): {
            "algo_code": "OKVIS",
            "numImuFrames": 3,
            "timeLimit": -1,
            "sigma_g_c": 0.004 * 0.3,
            "sigma_a_c": 0.07 * 0.3,
            "sigma_gw_c": 0.000044,
            "sigma_aw_c": 0.00172,},
        ('OKVIS', 'KSWF'): {
            "algo_code": "OKVIS",
            "numImuFrames": 3,
            "timeLimit": -1,
        },
        ('OKVIS_3', 'KSWF'): {
            "algo_code": "OKVIS",
            "numImuFrames": 3,
            "timeLimit": -1,
            "sigma_g_c": 0.004 * 3,
            "sigma_a_c": 0.07 * 3,
            "sigma_gw_c": 0.000044,
            "sigma_aw_c": 0.00172, },
        ('OKVIS_10', 'KSWF'): {
            "algo_code": "OKVIS",
            "numImuFrames": 3,
            "timeLimit": -1,
            "sigma_g_c": 0.004 * 10,
            "sigma_a_c": 0.07 * 10,
            "sigma_gw_c": 0.000044,
            "sigma_aw_c": 0.00172, },
    }
    return algo_option_templates, config_name_to_diffs


def tumrs_raw_test_options():
    """
    Compare okvis and KSWF on TUM RS calibrated 512 dataset
    Full self-calibration is enabled by default.
    """
    algo_option_templates = {
        'KSWF': {"algo_code": "HybridFilter",
                 "extra_gflags": "--publish_via_ros=false",
                 "displayImages": "false",
                 "monocular_input": 0,
                 "loop_closure_method": 0,
                 'use_nominal_calib_value': False},
    }

    config_name_to_diffs = {
        ('KSWF', 'KSWF'): {
            "sigma_g_c": 0.004 * 0.1,
            "sigma_a_c": 0.07 * 0.1,
            "sigma_gw_c": 0.000044,
            "sigma_aw_c": 0.00172,},
        ('KSWF_cal_cam', 'KSWF'): {
            "sigma_g_c": 0.004 * 0.1,
            "sigma_a_c": 0.07 * 0.1,
            "sigma_gw_c": 0.000044,
            "sigma_aw_c": 0.00172,
            "sigma_TGElement": 0.0,
            "sigma_TSElement": 0.0,
            "sigma_TAElement": 0.0,
        },
        ('KSWF_cal_imu', 'KSWF'): {
            "sigma_g_c": 0.004 * 0.1,
            "sigma_a_c": 0.07 * 0.1,
            "sigma_gw_c": 0.000044,
            "sigma_aw_c": 0.00172,

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
            "sigma_g_c": 0.004 * 0.1,
            "sigma_a_c": 0.07 * 0.1,
            "sigma_gw_c": 0.000044,
            "sigma_aw_c": 0.00172,

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
            "sigma_g_c": 0.004 * 0.1,
            "sigma_a_c": 0.07 * 0.1,
            "sigma_gw_c": 0.000044,
            "sigma_aw_c": 0.00172,
        }
    }
    return algo_option_templates, config_name_to_diffs
