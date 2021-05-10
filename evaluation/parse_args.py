#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Parse arguments for smoke_test, main_evaluation etc.

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

The output of swift_vio are lists of T_WB where B is the IMU sensor frame
defined according to the used IMU model.

"""

import argparse
import os
import sys

def check_common_args(args):
    assert os.path.isfile(args.vio_config_yaml)
    dataset_dir_list = [args.euroc_dir, args.uzh_fpv_dir, args.tumvi_dir,
                        args.advio_dir, args.homebrew_data_dir]
    dataset_status = list()
    for dataset_dir in dataset_dir_list:
        dataset_status.append(os.path.exists(dataset_dir))

    assert True in dataset_status, "None dataset dir exists"

    assert os.path.exists(args.rpg_eval_tool_dir), \
        'rpg_trajectory_evaluation does not exist'
    print('Input args to {}'.format(sys.argv[0]))
    for arg in vars(args):
        print(arg, getattr(args, arg))


def parse_args():
    parser = argparse.ArgumentParser(
        description='''Evaluate algorithms inside swift_vio project on EUROC 
        and UZH-FPV missions with ground truth.''')
    parser.add_argument('vio_config_yaml', type=str,
                        help="path to vio template config yaml. Its content "
                             "will NOT be modified in running the program.")
    parser.add_argument(
        '--euroc_dir', type=str,
        help="Folder containing the EUROC dataset with a structure layout "
             "depicted at the header. You need to extract data.csv for ground "
             "truth from the zip file beforehand.", default='')
    parser.add_argument(
        '--uzh_fpv_dir', type=str,
        help="Folder containing the UZH-FPV dataset with a structure layout"
             " depicted at the header. You need to extract ground truth from "
             "the bag file beforehand. This can be done with extract_uzh_fpv_gt.py",
             default='')
    parser.add_argument(
        '--tumvi_dir', type=str,
        help="Folder containing the TUM-VI dataset with a structure layout"
             " depicted at the header. You need to extract ground truth from "
             "the bag file beforehand. This can be done with extract_tum_vi_gt.py",
             default='')
    parser.add_argument(
        '--advio_dir', type=str,
        help="Folder containing the ADVIO dataset rosbags and ground truth files."
             " You need to extract convert ground truth from pose.csv."
             " This can be done with create_rosbags_for_advio.py",
             default='')
    parser.add_argument('--homebrew_data_dir', type=str,
                        help="Folder containing the homebrew rosbags", default='')

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
        '--cmp_eval_output_dir', help='base folder with the evaluation output to compare',
        default='')

    parser.add_argument(
        '--num_trials', type=int,
        help='number of trials for each mission each algorithm', default=1)

    parser.add_argument('--align_type', type=str,
                        default='posyaw', help='alignment type out of sim3 se3 posyaw none')

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
             ' so that locally installed gtsam libs can be loaded.\n'
             'It is not needed if gtsam is not used or gtsam is in /usr/local/lib',
        default="")

    lcd_config_default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      '../config/LcdParams.yaml')
    parser.add_argument('--lcd_config_yaml', type=str,
                        help="path to loop closure detector template config yaml. Its content "
                             "will NOT be modified in running the program.",
                        default=lcd_config_default)

    voc_file_default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '../vocabulary/ORBvoc.yml')

    parser.add_argument('--voc_file', type=str,
                        help="vocabulary full filename.",
                        default=voc_file_default)

    args = parser.parse_args()
    check_common_args(args)
    return args
