#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate several VIO method with ONE dataset, e.g., EUROC.
The evaluation procedure will run synchronously if publish_via_ros is true
because that case depends on roscore which does not accept nodes with the same name.
"""

import copy
import os
import sys

import dir_utility_functions
import parse_args
import rpg_eval_tool_wrap

import GroupPgoResults
import ResultsDirManager
import RunOneVioMethod
import utility_functions

from colorama import init, Fore
init(autoreset=True)


def find_all_bags_with_gt(data_dir, dataset_code):
    bag_list = []
    gt_list = []
    if dataset_code == "euroc":
        full_bag_list = dir_utility_functions.find_bags(data_dir, '.bag', discount_key='calibration')
        full_gt_list = dir_utility_functions.get_converted_euroc_gt_files(full_bag_list)

        # exclude V2_03
        for index, bagname in enumerate(full_bag_list):
            if 'V2_03_difficult' in bagname:
                continue
            else:
                bag_list.append(bagname)
                gt_list.append(full_gt_list[index])
    elif dataset_code == "uzh_fpv":
        bag_list = dir_utility_functions.find_bags_with_gt(data_dir, 'snapdragon_with_gt.bag')
        gt_list = dir_utility_functions.get_gt_file_for_bags(bag_list)
    elif dataset_code == "tum_vi":
        bag_list = dir_utility_functions.find_bags(data_dir, "_512_16.bag", "dataset-calib")
        gt_list = dir_utility_functions.get_gt_file_for_bags(bag_list)
    elif dataset_code == "tum_rs":
        bag_list = dir_utility_functions.find_bags(data_dir, ".bag")
        gt_list = dir_utility_functions.get_gt_file_for_bags(bag_list)
    elif dataset_code == "advio":
        bag_list = dir_utility_functions.find_bags(data_dir, "advio-")
        gt_list = dir_utility_functions.get_gt_file_for_bags(bag_list)
    elif dataset_code == "homebrew":
        bag_list = dir_utility_functions.find_bags(data_dir, "movie")
        gt_list = []

    print('For evaluation, #bags {} #gtlist {}'.format(len(bag_list), len(gt_list)))
    for index, gt in enumerate(gt_list):
        print('{}: {}'.format(bag_list[index], gt))
    return bag_list, gt_list


def euroc_stasis_test_options():
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
                 "loop_closure_method": 0,
                 'use_nominal_calib_value': False},
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
        }
    }
    return algo_option_templates, config_name_to_diffs


def sample_test_options():
    # python3.7 will remember insertion order of items, see
    # https://stackoverflow.com/questions/39980323/are-dictionaries-ordered-in-python-3-6
    algo_option_templates = {
        'OKVIS': {"algo_code": "OKVIS",
                  "extra_gflags": "--publish_via_ros=false",
                  "displayImages": "false",
                  "monocular_input": 1,
                  "loop_closure_method": 0,
                  'use_nominal_calib_value': False},
    }

    config_name_to_diffs = {
        ('KSWF', 'OKVIS'): {
            "algo_code": 'HybridFilter',
            "numImuFrames": 5,
            "sigma_absolute_translation": 0.02,
            "sigma_absolute_orientation": 0.01,
            "model_type": "BG_BA",
            'projection_opt_mode': 'FIXED',
            "extrinsic_opt_mode_main_camera": "P_CB",
            "extrinsic_opt_mode_other_camera": "P_C0C_Q_C0C",
        },
        ('KSWF_n', 'OKVIS'): {
            "algo_code": 'HybridFilter',
            "numImuFrames": 5,
            "monocular_input": 0,
            "sigma_absolute_translation": 0.02,
            "sigma_absolute_orientation": 0.01,
            "model_type": "BG_BA",
            'projection_opt_mode': 'FIXED',
            "extrinsic_opt_mode_main_camera": "P_CB",
            "extrinsic_opt_mode_other_camera": "P_C0C_Q_C0C",
        },
        ('OKVIS', 'OKVIS'): {
            "sigma_g_c": 12.0e-4 * 4,
            "sigma_a_c": 8.0e-3 * 4,
            "sigma_gw_c": 4.0e-6 * 4,
            "sigma_aw_c": 4.0e-5 * 4,
        },
        ('OKVIS_n', 'OKVIS'): {
            "monocular_input": 0,
            # We override P_C0C_Q_C0C as it conflicts with okvis estimator.
            "extrinsic_opt_mode_other_camera": "P_BC_Q_BC",
            "sigma_g_c": 12.0e-4 * 4,
            "sigma_a_c": 8.0e-3 * 4,
            "sigma_gw_c": 4.0e-6 * 4,
            "sigma_aw_c": 4.0e-5 * 4,
        },
        ('KSF', 'OKVIS'): {
            "algo_code": 'MSCKF',
            "numImuFrames": 5,
            "sigma_absolute_translation": 0.02,
            "sigma_absolute_orientation": 0.01,
            "model_type": "BG_BA",
            'projection_opt_mode': 'FIXED',
            "extrinsic_opt_mode_main_camera": "P_CB",
            "extrinsic_opt_mode_other_camera": "P_C0C_Q_C0C",
        },
        ('KSF_n', 'OKVIS'): {
            "algo_code": 'MSCKF',
            "numImuFrames": 5,
            "monocular_input": 0,
            "sigma_absolute_translation": 0.02,
            "sigma_absolute_orientation": 0.01,
            "model_type": "BG_BA",
            'projection_opt_mode': 'FIXED',
            "extrinsic_opt_mode_main_camera": "P_CB",
            "extrinsic_opt_mode_other_camera": "P_C0C_Q_C0C",
        },
        ('isam2-fls', 'OKVIS'): {
            "algo_code": "SlidingWindowSmoother",
            "extra_gflags": "--publish_via_ros=false",
        },
        ('ri-fls', 'OKVIS'): {
            "algo_code": "RiSlidingWindowSmoother",
            "extra_gflags": "--publish_via_ros=false --rifls_lock_jacobian=true",
        },
        ('ri-fls-exact', 'OKVIS'): {
            "algo_code": "RiSlidingWindowSmoother",
            "extra_gflags": "--publish_via_ros=false --rifls_lock_jacobian=false",
        },
    }
    return algo_option_templates, config_name_to_diffs

if __name__ == '__main__':
    args = parse_args.parse_args()
    bag_list, gt_list = find_all_bags_with_gt(args.data_dir, args.dataset_code)

    algo_option_templates, config_name_to_diffs = euroc_stasis_test_options()

    algoname_to_options = dict()
    for new_old_code, diffs in config_name_to_diffs.items():
        options = copy.deepcopy(algo_option_templates[new_old_code[1]])
        for key, value in diffs.items():
            options[key] = value
        algoname_to_options[new_old_code[0]] = options

    # rpg eval tool supports evaluating 6 algorithms at the same time, see len(PALLETE)
    MAX_ALGORITHMS_TO_EVALUATE = 6
    algoname_to_option_chunks = utility_functions.chunks(algoname_to_options,
                                                         MAX_ALGORITHMS_TO_EVALUATE)

    for index, minibatch in enumerate(algoname_to_option_chunks):
        algo_name_list = list(minibatch.keys())
        output_dir_suffix = ""
        if len(algoname_to_options) > MAX_ALGORITHMS_TO_EVALUATE:
            output_dir_suffix = "{}".format(index)
        minibatch_output_dir = args.output_dir + output_dir_suffix
        results_dir = os.path.join(minibatch_output_dir, "vio")
        eval_output_dir = os.path.join(minibatch_output_dir, "vio_eval")

        results_dir_manager = ResultsDirManager.ResultsDirManager(
            results_dir, bag_list, algo_name_list)
        results_dir_manager.create_results_dir()
        results_dir_manager.create_eval_config_yaml()
        results_dir_manager.create_eval_output_dir(eval_output_dir)
        results_dir_manager.save_config(minibatch, minibatch_output_dir)
        returncode = 0
        for name, options in minibatch.items():
            runner = RunOneVioMethod.RunOneVioMethod(
                args.catkin_ws, args.vio_config_yaml,
                options,
                args.num_trials, bag_list, gt_list,
                results_dir_manager.get_all_result_dirs(name),
                args.extra_lib_path, args.lcd_config_yaml,
                args.voc_file)
            rc = runner.run_method(name, args.pose_conversion_script, args.align_type, True)
            if rc != 0:
                returncode = rc
        # evaluate all VIO methods.
        if gt_list:
            rc, streamdata = rpg_eval_tool_wrap.run_rpg_evaluation(
                args.rpg_eval_tool_dir, results_dir_manager.get_eval_config_yaml(),
                args.num_trials, results_dir, eval_output_dir)
        else:
            rc = 1
            streamdata = "Skip trajectory evaluation because ground truth is unavailable."
        if rc != 0:
            print(Fore.RED + "Error code {} in run_rpg_evaluation: {}".format(rc, streamdata))

        rpg_eval_tool_wrap.check_eval_result(eval_output_dir, args.cmp_eval_output_dir)

        # also evaluate PGO results for every VIO method.
        for method_name, options in minibatch.items():
            if options["loop_closure_method"] == 0:
                continue
            method_results_dir = os.path.join(minibatch_output_dir, method_name + "_pgo")
            method_eval_output_dir = os.path.join(minibatch_output_dir, method_name + "_pgo_eval")

            gpr = GroupPgoResults.GroupPgoResults(
                results_dir, method_results_dir, method_eval_output_dir, method_name)
            gpr.copy_subdirs_for_pgo()

            rc, streamdata = rpg_eval_tool_wrap.run_rpg_evaluation(
                args.rpg_eval_tool_dir, gpr.get_eval_config_yaml(),
                args.num_trials, method_results_dir, method_eval_output_dir)
            if rc != 0:
                print(Fore.RED + "Error code {} in run_rpg_evaluation for method {}: {}".format(
                    rc, method_name, streamdata))

    print('Successfully finished testing methods in swift_vio project!')
    sys.exit(0)
