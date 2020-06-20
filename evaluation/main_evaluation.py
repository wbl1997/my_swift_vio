#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
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


def find_all_bags_with_gt(euroc_dir="", uzh_fpv_dir="", tum_vi_dir="", advio_dir=""):
    euroc_bag_list = dir_utility_functions.find_bags(euroc_dir, '.bag', discount_key='calibration')
    euroc_gt_list = dir_utility_functions.get_converted_euroc_gt_files(euroc_bag_list)

    # uzh_fpv_bag_list = dir_utility_functions.find_bags_with_gt(uzh_fpv_dir, 'snapdragon_with_gt.bag')
    # uzh_fpv_gt_list = dir_utility_functions.get_gt_file_for_bags(uzh_fpv_bag_list)

    # tumvi_bag_list = dir_utility_functions.find_bags(tum_vi_dir, "_512_16.bag", "dataset-calib")
    # tumvi_gt_list = dir_utility_functions.get_gt_file_for_bags(tumvi_bag_list)

    # advio_bag_list = dir_utility_functions.find_bags(advio_dir, "advio-")
    # advio_gt_list = dir_utility_functions.get_gt_file_for_bags(advio_bag_list)

    # exclude V2_03
    bag_list = []
    gt_list = []
    for index, bagname in enumerate(euroc_bag_list):
        if 'V2_03_difficult' in bagname:
            continue
        else:
            bag_list.append(bagname)
            gt_list.append(euroc_gt_list[index])

    print('For evaluation, #bags {} #gtlist {}'.format(len(bag_list), len(gt_list)))
    for index, gt in enumerate(gt_list):
        print('{}: {}'.format(bag_list[index], gt))
    return bag_list, gt_list


if __name__ == '__main__':
    args = parse_args.parse_args()

    bag_list, gt_list = find_all_bags_with_gt(
        args.euroc_dir, args.uzh_fpv_dir, args.tumvi_dir, args.advio_dir)

    # python3.7 will remember insertion order of items, see
    # https://stackoverflow.com/questions/39980323/are-dictionaries-ordered-in-python-3-6
    algo_option_templates = {
        'OKVIS': {"algo_code": "OKVIS",
                  "extra_gflags": "--publish_via_ros=false",
                  "numKeyframes": 5,
                  "numImuFrames": 3,
                  "monocular_input": 1,
                  "landmarkModelId": 1,
                  "anchorAtObservationTime": 0,
                  "model_type": "BG_BA",
                  'projection_opt_mode': 'FXY_CXY',
                  "extrinsic_opt_mode_main_camera": "P_BC_Q_BC",
                  "extrinsic_opt_mode_other_camera": "P_BC_Q_BC",
                  'sigma_TGElement': 0e-3,
                  'sigma_TSElement': 0e-3,
                  'sigma_TAElement': 0e-3,
                  "sigma_g_c": 12.0e-4,
                  "sigma_a_c": 8.0e-3,
                  "sigma_gw_c": 4.0e-6,
                  "sigma_aw_c": 4.0e-5,
                  "sigma_absolute_translation": 0.0,
                  "sigma_absolute_orientation": 0.0,
                  "sigma_td": 5e-3,
                  "sigma_tr": 0.0,
                  "sigma_focal_length": 0.0,
                  "sigma_principal_point": 0.0,
                  "sigma_distortion": "[0.0, 0.0, 0.0, 0.0]",
                  "stereoMatchWithEpipolarCheck": 1,
                  "epipolarDistanceThreshold": 2.5,
                  "maxOdometryConstraintForAKeyframe": 2,
                  "loop_closure_method": 0,
                  'use_nominal_calib_value': False},
    }

    config_name_to_diffs = {
        ('KSF', 'OKVIS'): {
            "algo_code": 'MSCKF',
            "numImuFrames": 5,
            "sigma_absolute_translation": 0.02,
            "sigma_absolute_orientation": 0.01,
        },
        ('KSF_n', 'OKVIS'): {
            "algo_code": 'MSCKF',
            "numImuFrames": 5,
            "monocular_input": 0,
            "sigma_absolute_translation": 0.02,
            "sigma_absolute_orientation": 0.01,
        },
        ('OKVIS_4_4', 'OKVIS'): {
            # we found that inflating IMU noise 4 times gives goog results for monocular OKVIS on EuRoC dataset.
            "sigma_g_c": 12.0e-4 * 4,
            "sigma_a_c": 8.0e-3 * 4,
            "sigma_gw_c": 4.0e-6 * 4,
            "sigma_aw_c": 4.0e-5 * 4,
        },
        ('OKVIS_n', 'OKVIS'): {
            "monocular_input": 0,
        },
    }

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
        rc, streamdata = rpg_eval_tool_wrap.run_rpg_evaluation(
            args.rpg_eval_tool_dir, results_dir_manager.get_eval_config_yaml(),
            args.num_trials, results_dir, eval_output_dir)
        if rc != 0:
            print(Fore.RED + "Error code {} in run_rpg_evaluation: {}".format(rc, streamdata))
            sys.exit(1)

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

    print('Successfully finished testing methods in msckf project!')
    sys.exit(0)
