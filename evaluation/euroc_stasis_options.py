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
