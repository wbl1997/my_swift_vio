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

import AlgoConfig
import ResultsDirManager
import RunOneVioMethod

from colorama import init, Fore
init(autoreset=True)

if __name__ == '__main__':
    args = parse_args.parse_args()

    euroc_bags = dir_utility_functions.find_bags(args.euroc_dir, '.bag', discount_key='calibration')
    euroc_gt_list = dir_utility_functions.get_converted_euroc_gt_files(euroc_bags)
    bag_list = [euroc_bags[0]]
    gt_list = [euroc_gt_list[0]]
    
    print('For evaluation, #bags {} #gtlist {}'.format(len(bag_list), len(gt_list)))
    for index, gt in enumerate(gt_list):
        print('{}: {}'.format(bag_list[index], gt))

    # rpg eval tool supports evaluating 6 algorithms at the same time, see len(PALLETE)
    algo_name_code_flags_dict = {
        'OKVIS': AlgoConfig.create_algo_config(['OKVIS', '', 5, 3]),
        'OKVIS_nframe': AlgoConfig.create_algo_config(['OKVIS', '', 5, 3, 0]),
        'MSCKF_i': AlgoConfig.create_algo_config(['MSCKF', '--use_IEKF=true', 10, 3]),
        'MSCKF': AlgoConfig.create_algo_config(['MSCKF', '', 10, 3]),
        # 'MSCKF_brisk_b2b': AlgoConfig.create_algo_config(['MSCKF', '--feature_tracking_method=2', 10, 3]),
        'MSCKF_klt': AlgoConfig.create_algo_config(['MSCKF', '--feature_tracking_method=1', 10, 3]),
        'MSCKF_async': AlgoConfig.create_algo_config(['MSCKF', '', 10, 3])}

    algo_name_list = list(algo_name_code_flags_dict.keys())

    results_dir = os.path.join(args.output_dir, "msckf_smoke")
    eval_output_dir = os.path.join(args.output_dir, "msckf_smoke_eval")

    results_dir_manager = ResultsDirManager.ResultsDirManager(
        results_dir, bag_list, algo_name_list)
    results_dir_manager.create_results_dir()
    results_dir_manager.create_eval_config_yaml()
    results_dir_manager.create_eval_output_dir(eval_output_dir)
    returncode = 0
    for name, code_flags in algo_name_code_flags_dict.items():
        runner = RunOneVioMethod.RunOneVioMethod(
            args.catkin_ws, args.vio_config_yaml,
            code_flags,
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
