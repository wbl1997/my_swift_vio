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

import euroc_stasis_options
import sample_eval_options
import tumrs_calib_options
import tumvi_calib_options


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


if __name__ == '__main__':
    args = parse_args.parse_args()
    bag_list, gt_list = find_all_bags_with_gt(args.data_dir, args.dataset_code)

    if args.test_case == "tumrs_calibrated":
        algo_option_templates, config_name_to_diffs = tumrs_calib_options.tumrs_calibrated_swiftvio_options()
    elif args.test_case == "tumrs_raw":
        algo_option_templates, config_name_to_diffs = tumrs_calib_options.tumrs_raw_swiftvio_options()
    elif args.test_case == "tumvi_calibrated":
        algo_option_templates, config_name_to_diffs = tumvi_calib_options.tumvi_calibrated_swiftvio_options()
    elif args.test_case == "tumvi_calibrated_vinsmono":
        algo_option_templates, config_name_to_diffs = tumvi_calib_options.tumvi_calibrated_vinsmono_options()
    elif args.test_case == "tumvi_calibrated_openvins":
        algo_option_templates, config_name_to_diffs = tumvi_calib_options.tumvi_calibrated_openvins_options()
    elif args.test_case == "tumvi_raw":
        algo_option_templates, config_name_to_diffs = tumvi_calib_options.tumvi_raw_swiftvio_options()
    elif args.test_case == "tumvi_raw_vinsmono":
        algo_option_templates, config_name_to_diffs = tumvi_calib_options.tumvi_raw_vinsmono_options()
    elif args.test_case == "tumvi_raw_openvins":
        algo_option_templates, config_name_to_diffs = tumvi_calib_options.tumvi_raw_openvins_options()
    elif args.test_case == "euroc_stasis":
        algo_option_templates, config_name_to_diffs = euroc_stasis_options.euroc_stasis_swiftvio_options()
    elif args.test_case == "euroc_stasis_openvins":
        algo_option_templates, config_name_to_diffs = euroc_stasis_options.euroc_stasis_openvins_options()
    else:
        algo_option_templates, config_name_to_diffs = sample_eval_options.sample_swiftvio_options()

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
            rc = runner.run_method(name, args.pose_conversion_script, args.align_type, True, args.dataset_code)
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
            print("Error code {} in run_rpg_evaluation: {}".format(rc, streamdata))

        rpg_eval_tool_wrap.check_eval_result(eval_output_dir, args.cmp_eval_output_dir)

        # also evaluate PGO results for every VIO method.
        for method_name, options in minibatch.items():
            if "loop_closure_method" not in options or options["loop_closure_method"] == 0:
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
                print("Error code {} in run_rpg_evaluation for method {}: {}".format(
                    rc, method_name, streamdata))

    print('Successfully finished testing methods in swift_vio project!')
    sys.exit(0)
