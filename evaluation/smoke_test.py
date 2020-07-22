"""
run several small tests to check the major modules of msckf

Test setting euroc MH_01_easy bag with ground truth

OKVIS_node okvis / msckf
OKVIS_node_synchronous okvis / msckf, feature tracking method 0/1/2

compare with ground truth
set threshold

"""

import os
import sys

import dir_utility_functions
import parse_args
import rpg_eval_tool_wrap
import utility_functions

import ResultsDirManager
import RunOneVioMethod

from colorama import init, Fore
init(autoreset=True)

if __name__ == '__main__':
    args = parse_args.parse_args()

    euroc_bag_list = dir_utility_functions.find_bags(args.euroc_dir, '.bag', discount_key='calibration')
    euroc_gt_list = dir_utility_functions.get_converted_euroc_gt_files(euroc_bag_list)

    advio_bag_list = dir_utility_functions.find_bags(args.advio_dir, '.bag')
    advio_gt_list = dir_utility_functions.get_gt_file_for_bags(advio_bag_list)

    bag_gt_list = sorted(zip(euroc_bag_list, euroc_gt_list))
    bag_list = [bag_gt_list[0][0]]
    gt_list = [bag_gt_list[0][1]]

    print('For evaluation, #bags {} #gtlist {}'.format(len(bag_list), len(gt_list)))
    for index, gt in enumerate(gt_list):
        print('{}: {}'.format(bag_list[index], gt))

    # python3.7 will remember insertion order of items, see
    # https://stackoverflow.com/questions/39980323/are-dictionaries-ordered-in-python-3-6
    algoname_to_options = {
        # We disable online extrinsic calibration for OKVIS by zeroing
        # sigma_absolute_translation and sigma_absolute_orientation.
        'MSCKF_BgBa': {"algo_code": "MSCKF",
                       "extra_gflags": "--publish_via_ros=true",
                       "numKeyframes": 5,
                       "numImuFrames": 5,
                       "monocular_input": 1,
                       "landmarkModelId": 1,
                       "anchorAtObservationTime": 0,
                       "model_type": "BG_BA",
                       "extrinsic_opt_mode_main_camera": "p_BC_q_BC",
                       "extrinsic_opt_mode_other_camera": "p_C0C_q_C0C",
                       "sigma_absolute_translation": "0.02",
                       "sigma_absolute_orientation": "0.01",},
        'MSCKF_n_aidp': {"algo_code": "MSCKF",
                         "extra_gflags": "--publish_via_ros=false",
                         "numKeyframes": 10,
                         "numImuFrames": 5,
                         "monocular_input": 0,
                         "landmarkModelId": 1,
                         "anchorAtObservationTime": 0,
                         "extrinsic_opt_mode_main_camera": "p_CB",
                         "extrinsic_opt_mode_other_camera": "p_C0C_q_C0C"},
        'MSCKF_n_hpp': {"algo_code": "MSCKF",
                        "extra_gflags": "--publish_via_ros=false",
                        "numKeyframes": 10,
                        "numImuFrames": 5,
                        "monocular_input": 0,
                        "landmarkModelId": 0,
                        "anchorAtObservationTime": 0,
                        "extrinsic_opt_mode_main_camera": "p_CB",
                        "extrinsic_opt_mode_other_camera": "p_C0C_q_C0C"},
        'MSCKF_n_aidp_T_BC': {"algo_code": "MSCKF",
                              "extra_gflags": "--publish_via_ros=false",
                              "numKeyframes": 10,
                              "numImuFrames": 5,
                              "monocular_input": 0,
                              "landmarkModelId": 1,
                              "anchorAtObservationTime": 0,
                              "extrinsic_opt_mode_main_camera": "p_CB",
                              "extrinsic_opt_mode_other_camera": "p_BC_q_BC"},
        'MSCKF_aidp': {"algo_code": "MSCKF",
                       "extra_gflags": "--publish_via_ros=false",
                       "numKeyframes": 10,
                       "numImuFrames": 5,
                       "monocular_input": 1,
                       "landmarkModelId": 1,
                       "anchorAtObservationTime": 0,
                       "extrinsic_opt_mode_main_camera": "p_CB",
                       "extrinsic_opt_mode_other_camera": "p_C0C_q_C0C"},
        'OKVIS': {"algo_code": "OKVIS",
                         "extra_gflags": "--publish_via_ros=false",
                         "numKeyframes": 5,
                         "numImuFrames": 3,
                         "monocular_input": 1,
                         "landmarkModelId": 0,
                         "anchorAtObservationTime": 0,
                         "extrinsic_opt_mode_main_camera": "p_BC_q_BC",
                         "extrinsic_opt_mode_other_camera": "p_BC_q_BC",
                         "sigma_absolute_translation": "0.0",
                         "sigma_absolute_orientation": "0.0"},
        'OKVIS_nframe': {"algo_code": "OKVIS",
                         "extra_gflags": "--publish_via_ros=false",
                         "numKeyframes": 5,
                         "numImuFrames": 3,
                         "monocular_input": 0,
                         "landmarkModelId": 0,
                         "anchorAtObservationTime": 0,
                         "extrinsic_opt_mode_main_camera": "p_BC_q_BC",
                         "extrinsic_opt_mode_other_camera": "p_BC_q_BC",
                         "sigma_absolute_translation": "0.0",
                         "sigma_absolute_orientation": "0.0"},
        # Jacobian relative to time come from observation frame and anchor frame.
        'MSCKF_n_aidp2': {"algo_code": "MSCKF",
                          "extra_gflags": "--publish_via_ros=false",
                          "numKeyframes": 10,
                          "numImuFrames": 5,
                          "monocular_input": 0,
                          "landmarkModelId": 1,
                          "anchorAtObservationTime": 1,
                          "extrinsic_opt_mode_main_camera": "p_CB",
                          "extrinsic_opt_mode_other_camera": "p_C0C_q_C0C"}
    }

    # 'MSCKF_i': AlgoConfig.create_algo_config(['MSCKF', '--use_IEKF=true', 10, 3]),
    # 'MSCKF_async': AlgoConfig.create_algo_config(['MSCKF', '', 10, 3])}

    # rpg eval tool supports evaluating 6 algorithms at the same time, see len(PALLETE)
    MAX_ALGORITHMS_TO_EVALUATE = 6
    algoname_to_options = utility_functions.resize_dict(algoname_to_options,
                                                        MAX_ALGORITHMS_TO_EVALUATE)

    algo_name_list = list(algoname_to_options.keys())

    results_dir = os.path.join(args.output_dir, "vio_smoke")
    eval_output_dir = os.path.join(args.output_dir, "vio_smoke_eval")

    results_dir_manager = ResultsDirManager.ResultsDirManager(
        results_dir, bag_list, algo_name_list)
    results_dir_manager.create_results_dir()
    results_dir_manager.create_eval_config_yaml()
    results_dir_manager.create_eval_output_dir(eval_output_dir)
    returncode = 0
    for name, options in algoname_to_options.items():
        runner = RunOneVioMethod.RunOneVioMethod(
            args.catkin_ws, args.vio_config_yaml,
            options,
            args.num_trials, bag_list, gt_list,
            results_dir_manager.get_all_result_dirs(name),
            args.extra_lib_path, args.lcd_config_yaml,
            args.voc_file)
        rc = runner.run_method(name, args.pose_conversion_script)
        if rc != 0:
            returncode = rc

    rc, streamdata = rpg_eval_tool_wrap.run_rpg_evaluation(
        args.rpg_eval_tool_dir, results_dir_manager.get_eval_config_yaml(),
        args.num_trials, results_dir, eval_output_dir)
    if rc != 0:
        print(Fore.RED + "Error code {} in run_rpg_evaluation: {}".format(rc, streamdata))
        sys.exit(1)

    rpg_eval_tool_wrap.check_eval_result(eval_output_dir, args.cmp_eval_output_dir)

    print('Successfully finished testing methods in msckf project!')
    sys.exit(0)
