#!/usr/bin/env python3

"""
The evaluation procedure runs synchronously because it depends on roscore adn
roscore does not accept nodes with the same name.

Input Data Structure

The folder structure layout for EUROC data:
.
├── machine_hall
│   ├── MH_01_easy
│   │   ├── data.csv
│   │   ├── mav0
│   │   ├── MH_01_easy.bag
│   │   └── MH_01_easy.zip
│   ├── MH_02_easy
│   │   ├── data.csv
│   │   ├── MH_02_easy.bag
│   │   └── MH_02_easy.zip
│   ├── MH_03_medium
│   │   ├── data.csv
│   │   ├── MH_03_medium.bag
│   │   └── MH_03_medium.zip
│   ├── MH_04_difficult
│   │   ├── data.csv
│   │   ├── MH_04_difficult.bag
│   │   └── MH_04_difficult.zip
│   └── MH_05_difficult
│       ├── data.csv
│       ├── MH_05_difficult.bag
│       └── MH_05_difficult.zip
├── vicon_room1
│   ├── V1_01_easy
│   │   ├── data.csv
│   │   ├── V1_01_easy.bag
│   │   └── V1_01_easy.zip
│   ├── V1_02_medium
│   │   ├── data.csv
│   │   ├── V1_02_medium.bag
│   │   └── V1_02_medium.zip
│   └── V1_03_difficult
│       ├── data.csv
│       ├── V1_03_difficult.bag
│       └── V1_03_difficult.zip
└── vicon_room2
    ├── V2_01_easy
    │   ├── data.csv
    │   ├── V2_01_easy.bag
    │   └── V2_01_easy.zip
    ├── V2_02_medium
    │   ├── data.csv
    │   ├── V2_02_medium.bag
    │   └── V2_02_medium.zip
    └── V2_03_difficult
        ├── data.csv
        ├── V2_03_difficult.bag
        └── V2_03_difficult.zip

In each subfolder for each mission, data.csv is extracted from folder
state_groundtruth_estimate0 insider the zip file.

The folder structure layout for UZH-FPV dataset:
.
├── indoor_45_11_davis.bag
├── indoor_45_11_snapdragon.bag
├── indoor_45_12_davis_with_gt.bag
├── indoor_45_12_snapdragon_with_gt.bag
├── indoor_45_13_davis_with_gt.bag
├── indoor_45_13_snapdragon_with_gt.bag
├── indoor_45_14_davis_with_gt.bag
├── indoor_45_14_snapdragon_with_gt.bag
├── indoor_45_16_davis.bag
|
|
|


Metrics

Statistics for relatiive pose errors are saved in files like relative_error_statistics_16_0.yaml.
The statistics "trans_per: mean" with a unit of percentage and "rot_deg_per_m: mean"
 with a unit of deg/m are the metrics used by KITTI and UZH-FPV.
The relative pose errors are also box plotted in figures.
Note the middle line for these boxes corresponds to medians rather than means.
To save the translation ATE rmse table, turn on rmse_table.
To save overall relative pose errors in terms of translation percentage and
rotation angle per meter, turn on overall_odometry_error

"The relative pose errors at the sub-trajectory of lengths {40, 60, 80, 100, 120}
meters are computed. The average translation and rotation error over all
sequences are used for ranking." [UZH-FPV](http://rpg.ifi.uzh.ch/uzh-fpv.html)

"From all test sequences, our evaluation computes translational and rotational
errors for all possible subsequences of length (100,...,800) meters.
The evaluation table below ranks methods according to the average of those
values, where errors are measured in percent (for translation) and in degrees
per meter (for rotation)." [KITTI](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)

Coordinate Frames

For EUROC dataset, the ground truth is provided within the zip files under 
state_groundtruth_estimate0 in terms of T_WB where W is a quasi inertial frame,
 and B is the body frame defined to be the IMU sensor frame.

For UZH-FPV dataset, the ground truth is provided within the bag files under topic 
in terms of T_WB where W is a quasi inertial frame, and B is the body frame 
defined to be the IMU sensor frame. Note the Snapdragon flight and mDavis sensor
unit have different IMUs, and thus different ground truth.

The output of msckf/OKVIS are lists of T_WB where B is the IMU sensor frame
defined according to the used IMU model.

"""

import argparse
import os
import shutil

import sys
from colorama import init, Fore
init(autoreset=True)

import dir_utility_functions
import utility_functions
import ResultsDirManager
import RunOneVioMethod


ALGO_NAME_CODE_FLAGS_DICT = {
    'OKVIS': ['OKVIS', '', 5, 3],
    'MSCKF_kf': ['MSCKF', '', 10, 3],
    'MSCKF_brisk_b2b': ['MSCKF', '--feature_tracking_method=2', 10, 3],
    'MSCKF_klt': ['MSCKF', '--feature_tracking_method=1', 10, 3],
    'MSCKF_kf_async': ['MSCKF', '', 10, 3]}


def run_rpg_evaluation(rpg_eval_dir, eval_config_yaml, num_trials,
                       results_dir, eval_output_dir):
    """

    :param rpg_eval_dir:
    :param eval_config_yaml:
    :param num_trials: for each mission how many times an algorithm runs
    :param results_dir:
    :param eval_output_dir:
    :return:
    """
    analyze_traj_script = os.path.join(rpg_eval_dir, "scripts/analyze_trajectories.py")
    cmd = "python2 {} {} --output_dir={} --results_dir={} --platform laptop " \
          "--odometry_error_per_dataset --plot_trajectories --recalculate_errors " \
          "--rmse_table --rmse_boxplot --mul_trials={} --overall_odometry_error". \
        format(analyze_traj_script, eval_config_yaml, eval_output_dir, results_dir, num_trials)
    print('cmd to rpg eval tool\n{}'.format(cmd))
    return utility_functions.subprocess_cmd(cmd)

def check_eval_result(cmp_result_dir):
    """
    briefly check eval results from rpg evaluation
    :param cmp_result_dir: previous evaluation result dir
    :return:
    """
    if cmp_result_dir:
        file_list = os.listdir(cmp_result_dir)
        for fname in file_list:
            if 'laptop_rel_err_' in fname and '.txt' in fname:
                cmp_stat_file = os.path.join(cmp_result_dir, fname)
                break

        cmp_rel_stats = utility_functions.load_rel_error_stats(cmp_stat_file)
        print('cmp rel stats from {}'.format(cmp_stat_file))
        print(cmp_rel_stats)

def parse_args():
    parser = argparse.ArgumentParser(
        description='''Evaluate algorithms inside msckf project on many EUROC 
        and UZH-FPV missions with ground truth.''')
    parser.add_argument('vio_config_yaml', type=str,
                        help="path to yaml configuration file. Its content may be"
                        " modified in running the program.")
    parser.add_argument(
        '--euroc_dir', type=str,
        help="Folder containing the EUROC dataset with a structure layout "
             "depicted at the header. You need to extract data.csv for ground "
             "truth from the zip file beforehand.")
    parser.add_argument(
        '--uzh_fpv_dir', type=str,
        help="Folder containing the UZH-FPV dataset with a structure layout"
             " depicted at the header. You need to extract ground truth from "
             "the bag file beforehand. This can be done with bag_to_file.py"
             " under rpg_trajectory_evaluation")
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '../../rpg_trajectory_evaluation/results')
    parser.add_argument(
        '--output_dir',
        help="Folder to output vio results and evaluation results",
        default=default_path)

    rpg_tool_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '../../rpg_trajectory_evaluation')
    parser.add_argument(
        '--rpg_eval_tool_dir', help='folder of rpg_trajectory_evaluation package',
        default=rpg_tool_path)

    parser.add_argument(
        '--cmp_results_dir', help='base folder with the results to compare',
        default='')
    parser.add_argument(
        '--num_trials', type=int,
        help='number of trials for each mission each algorithm', default=1)

    pose_conversion_script_default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '../../vio_common/python/convert_pose_format.py')
    parser.add_argument(
        '--pose_conversion_script',
        help="Script to convert vio output csv to TUM RGBD format",
        default=pose_conversion_script_default)

    catkin_ws_default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '../../../')
    parser.add_argument('--catkin_ws', type=str,
                        help="path to the vio project workspace including executables and launch files",
                        default=catkin_ws_default)

    parser.add_argument(
        '--extra_lib_path',
        help='Export extra library path to LD_LIBRARY_PATH'
             ' so that locally installed libs can be loaded',
        default="$HOME/Documents/slam_devel/lib")
    args = parser.parse_args()
    return args

def check_args(args):
    assert os.path.exists(args.euroc_dir), "EUROC dir does not exist"
    assert os.path.exists(args.uzh_fpv_dir), "UZH-FPV dir does not exist"
    assert os.path.exists(args.rpg_eval_tool_dir), 'rpg_trajectory_evaluation does not exist'
    print('Input args to {}'.format(sys.argv[0]))
    for arg in vars(args):
        print(arg, getattr(args, arg))

def find_bags_with_gt(euroc_dir, uzh_fpv_dir):
    euroc_bags = dir_utility_functions.find_bags(euroc_dir, '.bag', discount_key='calibration')
    uzh_fpv_bags = dir_utility_functions.find_bags_with_gt(uzh_fpv_dir, 'snapdragon_with_gt.bag')
    uzh_fpv_gt_list = dir_utility_functions.get_uzh_fpv_gt_files(uzh_fpv_bags)
    euroc_gt_list = dir_utility_functions.get_converted_euroc_gt_files(euroc_bags)

    for gt_file in euroc_gt_list:
        if not os.path.isfile(gt_file):
            raise Exception(Fore.RED + "Ground truth file {} deos not exist. Do you "
                                       "forget to convert data.csv to data.txt, e.g.,"
                                       " with convert_euroc_gt_csv.py}".format(gt_file))

    for gt_file in uzh_fpv_gt_list:
        if not os.path.isfile(gt_file):
            raise Exception(Fore.RED + "Ground truth file {} deos not exist. Do you "
                                       "forget to extract gt from bag files, e.g.,"
                                       " with extract_uzh_fpv_gt.py}".format(gt_file))

    print(Fore.RED + "We use only one mission from each dataset for debugging purpose")
    bag_list = [euroc_bags[0], uzh_fpv_bags[0]]
    gt_list = [euroc_gt_list[0], uzh_fpv_gt_list[0]]

    # bag_list = euroc_bags
    # bag_list.extend(uzh_fpv_bags)
    # gt_list = euroc_gt_list
    # gt_list.extend(uzh_fpv_gt_list)
    return bag_list, gt_list

if __name__ == '__main__':
    args = parse_args()
    check_args(args)

    bag_list, gt_list = find_bags_with_gt(args.euroc_dir, args.uzh_fpv_dir)
    print('For evaluation, #bags {} #gtlist {}'.format(len(bag_list), len(gt_list)))
    for index, gt in enumerate(gt_list):
        print('{}: {}'.format(bag_list[index], gt))

    ALGO_NAME_CODE_FLAGS_DICT = {'MSCKF': ['MSCKF', '--use_IEKF=true', 10, 3]}
    # ALGO_NAME_CODE_FLAGS_DICT = {'OKVIS': ['OKVIS', '', 5, 3]}
    ALGO_NAME_CODE_FLAGS_DICT = {'MSCKF_kf_async': ['MSCKF', '--use_IEKF=true', 10, 3]}

    algo_name_list = list(ALGO_NAME_CODE_FLAGS_DICT.keys())

    results_dir = os.path.join(args.output_dir, "msckf_okvis")
    eval_output_dir = os.path.join(args.output_dir, "msckf_okvis_eval")
    dir_utility_functions.mkdir_p(eval_output_dir)

    results_dir_manager = ResultsDirManager.ResultsDirManager(results_dir, bag_list, algo_name_list)
    results_dir_manager.create_results_dir()
    results_dir_manager.create_eval_config_yaml()

    for name, code_flags in ALGO_NAME_CODE_FLAGS_DICT.items():
        runner = RunOneVioMethod.RunOneVioMethod(
            args.catkin_ws, args.vio_config_yaml,
            code_flags,
            args.num_trials, bag_list, gt_list,
            results_dir_manager.get_all_result_dirs(name),
            args.extra_lib_path)
        runner.run_method(name, args.pose_conversion_script)

    rc, streamdata = run_rpg_evaluation(args.rpg_eval_tool_dir, results_dir_manager.get_eval_config_yaml(),
                       args.num_trials,
                       results_dir, eval_output_dir)
    if rc != 0:
        print(Fore.RED + "Error code {} in run_rpg_evaluation: {}".format(rc, streamdata))
        sys.exit(1)

    check_eval_result(args.cmp_results_dir)

    print('Successfully finished testing methods in msckf project!')
    sys.exit(0)
