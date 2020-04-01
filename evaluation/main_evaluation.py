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

import os
import sys

import dir_utility_functions
import parse_args
import rpg_eval_tool_wrap

import ResultsDirManager
import RunOneVioMethod

from colorama import init, Fore
init(autoreset=True)

if __name__ == '__main__':
    args = parse_args.parse_args()

    bag_list, gt_list = dir_utility_functions.find_all_bags_with_gt(
        args.euroc_dir, args.uzh_fpv_dir)
    print('For evaluation, #bags {} #gtlist {}'.format(len(bag_list), len(gt_list)))
    for index, gt in enumerate(gt_list):
        print('{}: {}'.format(bag_list[index], gt))

    algo_name_code_flags_dict = {'OKVIS': ['OKVIS', '', 5, 3]}

    algo_name_list = list(algo_name_code_flags_dict.keys())

    results_dir = os.path.join(args.output_dir, "okvis")
    eval_output_dir = os.path.join(args.output_dir, "okvis_eval")
    dir_utility_functions.mkdir_p(eval_output_dir)

    results_dir_manager = ResultsDirManager.ResultsDirManager(results_dir, bag_list, algo_name_list)
    results_dir_manager.create_results_dir()
    results_dir_manager.create_eval_config_yaml()
    returncode = 0
    for name, code_flags in algo_name_code_flags_dict.items():
        runner = RunOneVioMethod.RunOneVioMethod(
            args.catkin_ws, args.vio_config_yaml,
            code_flags,
            args.num_trials, bag_list, gt_list,
            results_dir_manager.get_all_result_dirs(name),
            args.extra_lib_path)
        rc = runner.run_method(name, args.pose_conversion_script)
        if rc != 0:
            returncode = rc

    rc, streamdata = rpg_eval_tool_wrap.run_rpg_evaluation(
        args.rpg_eval_tool_dir, results_dir_manager.get_eval_config_yaml(),
        args.num_trials, results_dir, eval_output_dir)
    if rc != 0:
        print(Fore.RED + "Error code {} in run_rpg_evaluation: {}".format(rc, streamdata))
        sys.exit(1)

    rpg_eval_tool_wrap.check_eval_result(args.cmp_results_dir)

    print('Successfully finished testing methods in msckf project!')
    sys.exit(0)
