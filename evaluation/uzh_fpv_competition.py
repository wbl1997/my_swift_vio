
import os
import sys

import dir_utility_functions
import parse_args
import rpg_eval_tool_wrap

import ResultsDirManager
import RunOneVioMethod
import AlgoConfig

from colorama import init, Fore
init(autoreset=True)


def find_test_bags(uzh_fpv_dir):
    session_list = {

        'indoor_45_3_snapdragon.bag',
        'indoor_45_16_snapdragon.bag',
        'indoor_forward_11_snapdragon.bag',
        'indoor_forward_12_snapdragon.bag',
        'outdoor_forward_9_snapdragon.bag',
        'outdoor_forward_10_snapdragon.bag',
    }

    # indoor_forward_11_davis.bag
    # indoor_forward_12_davis.bag
    # indoor_45_3_davis.bag
    # indoor_45_16_davis.bag
    # outdoor_forward_9_davis.bag
    # outdoor_forward_10_davis.bag

    uzh_fpv_bags = []
    for dataname in session_list:
        uzh_fpv_bags.extend(dir_utility_functions.find_bags(uzh_fpv_dir, dataname))
    return uzh_fpv_bags


if __name__ == '__main__':
    args = parse_args.parse_args()

    bag_list = find_test_bags(args.uzh_fpv_dir)
    print('For evaluation, #bags {}'.format(len(bag_list)))

    for index, bagpath in enumerate(bag_list):
        print('{}: {}'.format(index, bag_list[index]))

    algo_name_code_flags_dict = {'OKVIS_nframe': AlgoConfig.create_algo_config(['OKVIS', '', 5, 3, 0]),
                                 'MSCKF': AlgoConfig.create_algo_config(['MSCKF', '', 10, 3]),
                                 'MSCKF_nframe': AlgoConfig.create_algo_config(['MSCKF', '', 10, 3, 0])}

    algo_name_list = list(algo_name_code_flags_dict.keys())

    results_dir = os.path.join(args.output_dir, "okvis")
    eval_output_dir = os.path.join(args.output_dir, "okvis_eval")

    results_dir_manager = ResultsDirManager.ResultsDirManager(results_dir, bag_list, algo_name_list)
    results_dir_manager.create_results_dir()
    results_dir_manager.create_eval_config_yaml()
    results_dir_manager.create_eval_output_dir(eval_output_dir)

    returncode = 0
    for name, code_flags in algo_name_code_flags_dict.items():
        runner = RunOneVioMethod.RunOneVioMethod(
            args.catkin_ws, args.vio_config_yaml,
            code_flags,
            args.num_trials, bag_list, None,
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

    print('Successfully finished processing uzh fpv competition dataset!')
    sys.exit(0)
