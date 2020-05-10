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
│   │   ├── data.txt
│   │   ├── MH_01_easy.bag
│   │   └── MH_01_easy.zip
│   ├── MH_02_easy
│   │   ├── data.csv
│   │   ├── data.txt
│   │   ├── MH_02_easy.bag
│   │   └── MH_02_easy.zip
│   ├── MH_03_medium
│   │   ├── data.csv
│   │   ├── data.txt
│   │   ├── MH_03_medium.bag
│   │   └── MH_03_medium.zip
│   ├── MH_04_difficult
│   │   ├── data.csv
│   │   ├── data.txt
│   │   ├── MH_04_difficult.bag
│   │   └── MH_04_difficult.zip
│   └── MH_05_difficult
│       ├── data.csv
│       ├── data.txt
│       ├── MH_05_difficult.bag
│       └── MH_05_difficult.zip
├── vicon_room1
│   ├── V1_01_easy
│   │   ├── data.csv
│   │   ├── data.txt
│   │   ├── V1_01_easy.bag
│   │   └── V1_01_easy.zip
│   ├── V1_02_medium
│   │   ├── data.csv
│   │   ├── data.txt
│   │   ├── V1_02_medium.bag
│   │   └── V1_02_medium.zip
│   └── V1_03_difficult
│       ├── data.csv
│       ├── data.txt
│       ├── V1_03_difficult.bag
│       └── V1_03_difficult.zip
└── vicon_room2
    ├── V2_01_easy
    │   ├── data.csv
    │   ├── data.txt
    │   ├── V2_01_easy.bag
    │   └── V2_01_easy.zip
    ├── V2_02_medium
    │   ├── data.csv
    │   ├── data.txt
    │   ├── V2_02_medium.bag
    │   └── V2_02_medium.zip
    └── V2_03_difficult
        ├── data.csv
        ├── data.txt
        ├── V2_03_difficult.bag
        └── V2_03_difficult.zip


In each subfolder for each mission, data.csv is extracted from folder
state_groundtruth_estimate0 insider the zip file.
And data.txt is converted in format from data.csv by convert_euroc_gt_csv.py.

The folder structure layout for UZH-FPV dataset:
.
├── indoor_45_12_snapdragon_with_gt.bag
├── indoor_45_12_snapdragon_with_gt.txt
├── indoor_45_13_davis_with_gt.bag
├── indoor_45_13_snapdragon_with_gt.bag
├── indoor_45_13_snapdragon_with_gt.txt
├── indoor_45_14_davis_with_gt.bag
├── indoor_45_14_snapdragon_with_gt.bag
├── indoor_45_14_snapdragon_with_gt.txt
├── indoor_45_16_davis.bag
├── indoor_45_16_snapdragon.bag
├── indoor_45_1_davis.bag
├── indoor_45_1_snapdragon.bag
├── indoor_45_2_davis_with_gt.bag
├── indoor_45_2_snapdragon_with_gt.bag
├── indoor_45_2_snapdragon_with_gt.txt
├── indoor_45_3_davis.bag
├── indoor_45_3_snapdragon.bag
├── indoor_45_4_davis_with_gt.bag
├── indoor_45_4_snapdragon_with_gt.bag
├── indoor_45_4_snapdragon_with_gt.txt
├── indoor_45_9_davis_with_gt.bag
├── indoor_45_9_snapdragon_with_gt.bag
├── indoor_45_9_snapdragon_with_gt.txt
├── indoor_forward_10_davis_with_gt.bag
├── indoor_forward_10_snapdragon_with_gt.bag
├── indoor_forward_10_snapdragon_with_gt.txt
├── indoor_forward_11_davis.bag
├── indoor_forward_11_snapdragon.bag
├── indoor_forward_12_davis.bag
├── indoor_forward_12_snapdragon.bag
├── indoor_forward_3_davis_with_gt.bag
├── indoor_forward_3_davis_with_gt.zip
├── indoor_forward_3_snapdragon_with_gt.bag
├── indoor_forward_3_snapdragon_with_gt.txt
├── indoor_forward_5_davis_with_gt.bag
├── indoor_forward_5_snapdragon_with_gt.bag
├── indoor_forward_5_snapdragon_with_gt.txt
├── indoor_forward_6_davis_with_gt.bag
├── indoor_forward_6_snapdragon_with_gt.bag
├── indoor_forward_6_snapdragon_with_gt.txt
├── indoor_forward_6_snapdragon_with_gt.zip
├── indoor_forward_7_davis_with_gt.bag
├── indoor_forward_7_snapdragon_with_gt.bag
├── indoor_forward_7_snapdragon_with_gt.txt
├── indoor_forward_8_davis.bag
├── indoor_forward_8_snapdragon.bag
├── indoor_forward_8_snapdragon.orig.bag
├── indoor_forward_9_davis_with_gt.bag
├── indoor_forward_9_snapdragon_with_gt.bag
├── indoor_forward_9_snapdragon_with_gt.txt
├── outdoor_45_1_davis_with_gt.bag
├── outdoor_45_1_snapdragon_with_gt.bag
├── outdoor_45_1_snapdragon_with_gt.txt
├── outdoor_45_2_davis.bag
├── outdoor_45_2_snapdragon.bag
├── outdoor_forward_10_davis.bag
├── outdoor_forward_10_snapdragon.bag
├── outdoor_forward_1_davis_with_gt.bag
├── outdoor_forward_1_snapdragon_with_gt.bag
├── outdoor_forward_1_snapdragon_with_gt.txt
├── outdoor_forward_2_davis.bag
├── outdoor_forward_2_snapdragon.bag
├── outdoor_forward_3_davis_with_gt.bag
├── outdoor_forward_3_snapdragon_with_gt.bag
├── outdoor_forward_3_snapdragon_with_gt.txt
├── outdoor_forward_5_davis_with_gt.bag
├── outdoor_forward_5_snapdragon_with_gt.bag
├── outdoor_forward_5_snapdragon_with_gt.txt
├── outdoor_forward_6_davis.bag
├── outdoor_forward_6_snapdragon.bag
├── outdoor_forward_9_davis.bag
├── outdoor_forward_9_snapdragon.bag
|

The bags are downloaded from uzh fpv dataset webpage and the corresponding txt
 files are extracted from the rosbags with ground truth using
 extract_uzh_fpv_gt.py which wraps bag_to_pose.py of the rpg evaluation tool.

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

import GroupPgoResults
import ResultsDirManager
import RunOneVioMethod
import utility_functions

from colorama import init, Fore
init(autoreset=True)


def find_all_bags_with_gt(euroc_dir, uzh_fpv_dir):
    euroc_bags = dir_utility_functions.find_bags(euroc_dir, '.bag', discount_key='calibration')
    euroc_gt_list = dir_utility_functions.get_converted_euroc_gt_files(euroc_bags)

    uzh_fpv_bags = dir_utility_functions.find_bags_with_gt(uzh_fpv_dir, 'snapdragon_with_gt.bag')
    uzh_fpv_gt_list = dir_utility_functions.get_uzh_fpv_gt_files(uzh_fpv_bags)

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

    bag_list = uzh_fpv_bags
    gt_list = uzh_fpv_gt_list

    # bag_list.extend(euroc_bags)
    # gt_list.extend(euroc_gt_list)
    return bag_list, gt_list


if __name__ == '__main__':
    args = parse_args.parse_args()

    bag_list, gt_list = find_all_bags_with_gt(
        args.euroc_dir, args.uzh_fpv_dir)
    print('For evaluation, #bags {} #gtlist {}'.format(len(bag_list), len(gt_list)))
    for index, gt in enumerate(gt_list):
        print('{}: {}'.format(bag_list[index], gt))

    # python3.7 will remember insertion order of items, see
    # https://stackoverflow.com/questions/39980323/are-dictionaries-ordered-in-python-3-6
    algoname_to_options = {
        # We disable online extrinsic calibration for OKVIS by zeroing
        # sigma_absolute_translation and sigma_absolute_orientation.
        # 'OKVIS': {"algo_code": "OKVIS",
        #           "extra_gflags": "",
        #           "numKeyframes": 5,
        #           "numImuFrames": 3,
        #           "monocular_input": 1,
        #           "landmarkModelId": 0,
        #           "anchorAtObservationTime": 0,
        #           "extrinsic_opt_mode_main_camera": "p_BC_q_BC",
        #           "extrinsic_opt_mode_other_camera": "p_BC_q_BC",
        #           "sigma_absolute_translation": "0.0",
        #           "sigma_absolute_orientation": "0.0"},
        # 'MSCKF_n_hpp': {"algo_code": "MSCKF",
        #                 "extra_gflags": "",
        #                 "numKeyframes": 10,
        #                 "numImuFrames": 5,
        #                 "monocular_input": 0,
        #                 "landmarkModelId": 0,
        #                 "anchorAtObservationTime": 0,
        #                 "extrinsic_opt_mode_main_camera": "p_CB",
        #                 "extrinsic_opt_mode_other_camera": "p_C0C_q_C0C",
        #                 "sigma_absolute_translation": "0.02",
        #                 "sigma_absolute_orientation": "0.01"
        #                 },
        # Jacobian relative to time come from observation frame and anchor frame.
        # 'MSCKF_n_aidp2': {"algo_code": "MSCKF",
        #                   "extra_gflags": "",
        #                   "numKeyframes": 10,
        #                   "numImuFrames": 5,
        #                   "monocular_input": 0,
        #                   "landmarkModelId": 1,
        #                   "anchorAtObservationTime": 1,
        #                   "extrinsic_opt_mode_main_camera": "p_CB",
        #                   "extrinsic_opt_mode_other_camera": "p_C0C_q_C0C",
        #                   "sigma_absolute_translation": "0.02",
        #                   "sigma_absolute_orientation": "0.01"
        #                   },
        'MSCKF_n_aidp_T_C0C': {"algo_code": "MSCKF",
                               "extra_gflags": "",
                               "numKeyframes": 10,
                               "numImuFrames": 5,
                               "monocular_input": 0,
                               "landmarkModelId": 1,
                               "anchorAtObservationTime": 0,
                               "extrinsic_opt_mode_main_camera": "p_CB",
                               "extrinsic_opt_mode_other_camera": "p_C0C_q_C0C",
                               "sigma_absolute_translation": "0.02",
                               "sigma_absolute_orientation": "0.01"
                               },
        'MSCKF_n_fix': {"algo_code": "MSCKF",
                        "extra_gflags": "",
                        "numKeyframes": 10,
                        "numImuFrames": 5,
                        "monocular_input": 0,
                        "landmarkModelId": 1,
                        "anchorAtObservationTime": 0,
                        "extrinsic_opt_mode_main_camera": "p_CB",
                        "extrinsic_opt_mode_other_camera": "p_C0C_q_C0C",
                        "sigma_absolute_translation": "0.0",
                        "sigma_absolute_orientation": "0.0",
                        "sigma_tr": "0.0",
                        "sigma_td": "0.0",
                        "sigma_focal_length": "0.0",
                        "sigma_principal_point": "0.0",
                        "sigma_distortion": "[0.00, 0.0, 0.0, 0.0]",
                        "sigma_TGElement": "0e-3",
                        "sigma_TSElement": "0e-4",
                        "sigma_TAElement": "0e-3",
                        "maxOdometryConstraintForAKeyframe": "3"
                        },
        'MSCKF_n_part_fix_imu': {"algo_code": "MSCKF",
                                 "extra_gflags": "",
                                 "numKeyframes": 10,
                                 "numImuFrames": 5,
                                 "monocular_input": 0,
                                 "landmarkModelId": 1,
                                 "anchorAtObservationTime": 0,
                                 "extrinsic_opt_mode_main_camera": "p_CB",
                                 "extrinsic_opt_mode_other_camera": "p_C0C_q_C0C",
                                 "sigma_tr": "0.0",
                                 "sigma_focal_length": "0.0",
                                 "sigma_principal_point": "0.0",
                                 "sigma_distortion": "[0.00, 0.0, 0.0, 0.0]",
                                 "sigma_TGElement": "0e-3",
                                 "sigma_TSElement": "0e-4",
                                 "sigma_TAElement": "0e-3",
                                 "maxOdometryConstraintForAKeyframe": "3"
                                 },
        'MSCKF_n_part_fix': {"algo_code": "MSCKF",
                             "extra_gflags": "",
                             "numKeyframes": 10,
                             "numImuFrames": 5,
                             "monocular_input": 0,
                             "landmarkModelId": 1,
                             "anchorAtObservationTime": 0,
                             "extrinsic_opt_mode_main_camera": "p_CB",
                             "extrinsic_opt_mode_other_camera": "p_C0C_q_C0C",
                             "sigma_tr": "0.0",
                             "sigma_focal_length": "0.0",
                             "sigma_principal_point": "0.0",
                             "sigma_distortion": "[0.00, 0.0, 0.0, 0.0]",
                             "maxOdometryConstraintForAKeyframe": "3"
                             },
        'MSCKF_n_part_fix_gc': {"algo_code": "MSCKF",
                                "extra_gflags": "",
                                "numKeyframes": 10,
                                "numImuFrames": 5,
                                "monocular_input": 0,
                                "landmarkModelId": 1,
                                "anchorAtObservationTime": 0,
                                "extrinsic_opt_mode_main_camera": "p_CB",
                                "extrinsic_opt_mode_other_camera": "p_C0C_q_C0C",
                                "sigma_tr": "0.0",
                                "sigma_focal_length": "0.0",
                                "sigma_principal_point": "0.0",
                                "sigma_distortion": "[0.00, 0.0, 0.0, 0.0]",
                                "maxOdometryConstraintForAKeyframe": "3",
                                "sigma_g_c": "0.015"
                                },
        # 'MSCKF_aidp': {"algo_code": "MSCKF",
        #                "extra_gflags": "",
        #                "numKeyframes": 10,
        #                "numImuFrames": 5,
        #                "monocular_input": 1,
        #                "landmarkModelId": 1,
        #                "anchorAtObservationTime": 0,
        #                "extrinsic_opt_mode_main_camera": "p_CB",
        #                "extrinsic_opt_mode_other_camera": "p_BC_q_BC",
        #                "sigma_absolute_translation": "0.02",
        #                "sigma_absolute_orientation": "0.01"
        #                },
        # 'OKVIS_nframe': {"algo_code": "OKVIS",
        #                  "extra_gflags": "",
        #                  "numKeyframes": 5,
        #                  "numImuFrames": 3,
        #                  "monocular_input": 0,
        #                  "landmarkModelId": 0,
        #                  "anchorAtObservationTime": 0,
        #                  "extrinsic_opt_mode_main_camera": "p_BC_q_BC",
        #                  "extrinsic_opt_mode_other_camera": "p_BC_q_BC",
        #                  "sigma_absolute_translation": "0.0",
        #                  "sigma_absolute_orientation": "0.0",
        #                  "loop_closure_method": 0},
    }

    # 'MSCKF_i': AlgoConfig.create_algo_config(['MSCKF', '--use_IEKF=true', 10, 3]),
    # 'MSCKF_brisk_b2b': AlgoConfig.create_algo_config(['MSCKF', '--feature_tracking_method=2', 10, 3]),
    # 'MSCKF_klt': AlgoConfig.create_algo_config(['MSCKF', '--feature_tracking_method=1', 10, 3]),
    # 'MSCKF_async': AlgoConfig.create_algo_config(['MSCKF', '', 10, 3])}

    # rpg eval tool supports evaluating 6 algorithms at the same time, see len(PALLETE)
    MAX_ALGORITHMS_TO_EVALUATE = 6
    algoname_to_options = utility_functions.resize_dict(algoname_to_options,
                                                        MAX_ALGORITHMS_TO_EVALUATE)

    algo_name_list = list(algoname_to_options.keys())

    results_dir = os.path.join(args.output_dir, "vio")
    eval_output_dir = os.path.join(args.output_dir, "vio_eval")

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
        rc = runner.run_method(name, args.pose_conversion_script, True)
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
    for method_name, options in algoname_to_options.items():
        method_results_dir = os.path.join(args.output_dir, method_name + "_pgo")
        method_eval_output_dir = os.path.join(args.output_dir, method_name + "_pgo_eval")

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
