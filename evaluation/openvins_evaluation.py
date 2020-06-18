#!/usr/bin/env python3

"""
The evaluation procedure will run synchronously if publish_via_ros is true
because it depends on roscore which does not accept nodes with the same name.

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
imuthreshold={
    "dataset-room1_512_16": "0.75",
    "dataset-room2_512_16": "0.75",
    "dataset-room3_512_16": "0.75",
    "dataset-room4_512_16": "0.75",
    "dataset-room5_512_16": "0.75",
    "dataset-room6_512_16": "0.50",
    "dataset-corridor1_512_16": "0.70",
    "dataset-corridor2_512_16": "0.50",
    "dataset-corridor3_512_16": "0.70",
    "dataset-corridor4_512_16": "1.00",
    "dataset-corridor5_512_16": "0.70",
    "dataset-magistrale1_512_16": "0.50",
    "dataset-magistrale2_512_16": "0.70",
    "dataset-magistrale3_512_16": "0.70",
    "dataset-magistrale4_512_16": "0.50",
    "dataset-magistrale5_512_16": "1.00",
    "dataset-magistrale6_512_16": "0.70",
    "dataset-slides1_512_16": "0.50",
    "dataset-slides2_512_16": "0.50",
    "dataset-slides3_512_16": "0.70",
    "dataset-outdoors1_512_16": "0.50",
    "dataset-outdoors2_512_16": "0.70",
    "dataset-outdoors3_512_16": "0.75",
    "dataset-outdoors4_512_16": "0.70",
    "dataset-outdoors5_512_16": "0.75",
    "dataset-outdoors6_512_16": "0.75",
    "dataset-outdoors7_512_16": "0.75",
    "dataset-outdoors8_512_16": "0.50"
}



import os
import sys
import shutil

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
    euroc_bags = dir_utility_functions.find_bags(euroc_dir, '.bag', discount_key='calibration')
    euroc_gt_list = dir_utility_functions.get_converted_euroc_gt_files(euroc_bags)

    uzh_fpv_bags = dir_utility_functions.find_bags_with_gt(uzh_fpv_dir, 'snapdragon_with_gt.bag')
    uzh_fpv_gt_list = dir_utility_functions.get_gt_file_for_bags(uzh_fpv_bags)

    tumvi_bags = dir_utility_functions.find_bags(tum_vi_dir, "dataset-", "dataset-calib")
    tumvi_gt_list = dir_utility_functions.get_gt_file_for_bags(tumvi_bags)

    advio_bags = dir_utility_functions.find_bags(advio_dir, "advio-")
    advio_gt_list = dir_utility_functions.get_gt_file_for_bags(advio_bags)

    all_gt_list = euroc_gt_list
    all_gt_list.extend(uzh_fpv_gt_list)
    all_gt_list.extend(tumvi_gt_list)
    all_gt_list.extend(advio_gt_list)

    for gt_file in all_gt_list:
        if not os.path.isfile(gt_file):
            raise Exception(
                Fore.RED + "Ground truth file {} does not exist. Do you "
                           "forget to convert data.csv to data.txt or "
                           "extract ground truth from rosbags?".format(gt_file))

    if euroc_bags!=[]:
        dataset = "euroc"
        print("euroc")
        bag_list = euroc_bags
        gt_list = euroc_gt_list
    elif tumvi_bags!=[]:
        dataset = "tumvi"
        print("tumvi")
        bag_list = tumvi_bags
        gt_list = tumvi_gt_list
    # bag_list = uzh_fpv_bags
    # gt_list = uzh_fpv_gt_list

    return bag_list, gt_list, dataset


if __name__ == '__main__':
    args = parse_args.parse_args()

    bag_list, gt_list, dataset = find_all_bags_with_gt(
        args.euroc_dir, args.uzh_fpv_dir, args.tumvi_dir, args.advio_dir)

    print('For evaluation, #bags {} #gtlist {}'.format(len(bag_list), len(gt_list)))
    for index, gt in enumerate(gt_list):
        print('{}: {}'.format(bag_list[index], gt))

    # python3.7 will remember insertion order of items, see
    # https://stackoverflow.com/questions/39980323/are-dictionaries-ordered-in-python-3-6
    algoname_to_options = {
        # 'openvins-mono-imu-adapt': {"algo_code": "openvins-mono",
        #                   "sigma_a_c": 0.07,
        #                   "sigma_aw_c": 0.00172,
        #                   "sigma_g_c": 0.004,
        #                   "sigma_gw_c": 0.000044,
        #                   "coeff_c": 1,
        #                   "coeff_wc": 1
        #                      },
        'openvins-stereo-imu-adapt': {"algo_code": "openvins-stereo",
                          "sigma_a_c": 0.07,
                          "sigma_aw_c": 0.00172,
                          "sigma_g_c": 0.004,
                          "sigma_gw_c": 0.000044,
                          "coeff_c": 1,
                          "coeff_wc": 1
                             },                             
        # 'openvins-mono-0.5-0.5': {"algo_code": "openvins-mono",
        #                   "sigma_a_c": 0.07,
        #                   "sigma_aw_c": 0.00172,
        #                   "sigma_g_c": 0.004,
        #                   "sigma_gw_c": 0.000044,
        #                   "coeff_c": 0.5,
        #                   "coeff_wc": 0.5
        #                      },
        # 'openvins-mono-0.2-0.2': {"algo_code": "openvins-mono",
        #                   "sigma_a_c": 0.07,
        #                   "sigma_aw_c": 0.00172,
        #                   "sigma_g_c": 0.004,
        #                   "sigma_gw_c": 0.000044,
        #                   "coeff_c": 0.2,
        #                   "coeff_wc": 0.2
        #                      },
        # 'openvins-mono-0.1-0.1': {"algo_code": "openvins-mono",
        #                   "sigma_a_c": 0.07,
        #                   "sigma_aw_c": 0.00172,
        #                   "sigma_g_c": 0.004,
        #                   "sigma_gw_c": 0.000044,
        #                   "coeff_c": 0.1,
        #                   "coeff_wc": 0.1
        #                      },
        # 'openvins-mono-0.05-0.05': {"algo_code": "openvins-mono",
        #                   "sigma_a_c": 0.07,
        #                   "sigma_aw_c": 0.00172,
        #                   "sigma_g_c": 0.004,
        #                   "sigma_gw_c": 0.000044,
        #                   "coeff_c": 0.05,
        #                   "coeff_wc": 0.05
        #                      },                             
        # 'openvins-stereo-0.2-0.005': {"algo_code": "openvins-stereo",
        #                   "sigma_a_c": 0.07,
        #                   "sigma_aw_c": 0.00172,
        #                   "sigma_g_c": 0.004,
        #                   "sigma_gw_c": 0.000044,
        #                   "coeff_c": 0.2,
        #                   "coeff_wc": 0.005
        #                      },
        # 'openvins-stereo-0.2-0.01': {"algo_code": "openvins-stereo",
        #                   "sigma_a_c": 0.07,
        #                   "sigma_aw_c": 0.00172,
        #                   "sigma_g_c": 0.004,
        #                   "sigma_gw_c": 0.000044,
        #                   "coeff_c": 0.2,
        #                   "coeff_wc": 0.01
        #                      },
        # 'openvins-stereo-0.2-0.02': {"algo_code": "openvins-stereo",
        #                   "sigma_a_c": 0.07,
        #                   "sigma_aw_c": 0.00172,
        #                   "sigma_g_c": 0.004,
        #                   "sigma_gw_c": 0.000044,
        #                   "coeff_c": 0.2,
        #                   "coeff_wc": 0.02
        #                      },
        # 'openvins-stereo-x5': {"algo_code": "openvins-mono",
        #                   "sigma_a_c": 0.07,
        #                   "sigma_aw_c": 0.00172,
        #                   "sigma_g_c": 0.004,
        #                   "sigma_gw_c": 0.000044,
        #                   "coeff_c": 0.1,
        #                   "coeff_wc": 0.1
        #                      },
        # 'openvins-stereo-x6': {"algo_code": "openvins-mono",
        #                   "sigma_a_c": 0.07,
        #                   "sigma_aw_c": 0.00172,
        #                   "sigma_g_c": 0.004,
        #                   "sigma_gw_c": 0.000044,
        #                   "coeff_c": 0.1,
        #                   "coeff_wc": 0.05
        #                      },
        # 'openvins-stereo-x7': {"algo_code": "openvins-mono",
        #                   "sigma_a_c": 0.07,
        #                   "sigma_aw_c": 0.00172,
        #                   "sigma_g_c": 0.004,
        #                   "sigma_gw_c": 0.000044,
        #                   "coeff_c": 0.05,
        #                   "coeff_wc": 0.05
        #                      },
        # 'openvins-stereo-x8': {"algo_code": "openvins-mono",
        #                   "sigma_a_c": 0.07,
        #                   "sigma_aw_c": 0.00172,
        #                   "sigma_g_c": 0.004,
        #                   "sigma_gw_c": 0.000044,
        #                   "coeff_c": 0.02,
        #                   "coeff_wc": 0.05
        #                      },



        # 'openvins-stereo-x1': {"algo_code": "openvins-stereo",
        #                   "sigma_a_c": 0.07,
        #                   "sigma_aw_c": 0.00172,
        #                   "sigma_g_c": 0.004,
        #                   "sigma_gw_c": 0.000044,
        #                   "coeff_c": 0.1,
        #                   "coeff_wc": 0.1
        #                      },
    }

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

        results_dir = os.path.join(args.output_dir, "vio")
        eval_output_dir = os.path.join(args.output_dir, "vio_eval")

        results_dir_manager = ResultsDirManager.ResultsDirManager(
            results_dir, bag_list, algo_name_list)
        results_dir_manager.create_results_dir()
        results_dir_manager.create_eval_config_yaml()
        results_dir_manager.create_eval_output_dir(eval_output_dir)
        #results_dir_manager.save_config(minibatch, minibatch_output_dir)
        returncode = 0


        eval_cfg_template = os.path.join(args.catkin_ws, "src/msckf/evaluation/config/eval_cfg.yaml")

        for name, options in minibatch.items():
            for bag_index, bag_fullname in enumerate(bag_list):
                print(bag_fullname)
                bagname = os.path.basename(os.path.splitext(bag_fullname)[0])
                result_dir = results_dir_manager.get_result_dir(name, bag_fullname)

                if 'mono' in name:
                    max_cameras = 1
                    use_stereo = 'false'
                elif 'stereo' in name:
                    max_cameras = 2
                    use_stereo = 'true'


                for trial_index in range(args.num_trials):
                    if args.num_trials == 1:
                        index_str = ''
                    else:
                        index_str = '{}'.format(trial_index)

                    result_file = os.path.join(result_dir, 'stamped_traj_estimate{}.txt'.format(index_str))
                    setup_bash_file = os.path.join("/home/youkely/dev/openvins_ws", "devel/setup.bash")
                    print(setup_bash_file)
                    src_cmd = "cd ~\nsource {}\n".format(setup_bash_file)
                    bag_start = 0

                    if dataset == "euroc":
                        # launch_file = "pgeneva_ros_eth.launch"
                        launch_file = "pgeneva_serial_eth.launch"
                        init_imu_thresh = 1.5
                        if "MH_01" in bag_fullname:
                            bag_start = 40
                        elif "MH_02" in bag_fullname:
                            bag_start = 35
                        elif "MH_03" in bag_fullname:
                            bag_start = 15
                        elif "MH_04" in bag_fullname:
                            bag_start = 20
                        elif "MH_05" in bag_fullname:
                            bag_start = 20
                        eval_cfg_template = os.path.join(args.catkin_ws, "src/msckf/evaluation/config/eval_cfg.yaml")

                        launch_cmd = "roslaunch ov_msckf {} max_cameras:={} use_stereo:={} " \
                                     "bag:={} bag_start:={} init_imu_thresh:={} dosave:=true path_est:={}".format(
                            launch_file, max_cameras, use_stereo,
                            bag_fullname, bag_start, init_imu_thresh, result_file)

                    elif dataset == "tumvi":
                        # launch_file = "pgeneva_ros_tum.launch"
                        launch_file = "pgeneva_serial_tum.launch"
                        init_imu_thresh = imuthreshold[bagname]
                        eval_cfg_template = os.path.join(args.catkin_ws, "src/msckf/evaluation/config/eval_cfg_se3.yaml")

                        # launch_cmd = "roslaunch ov_msckf {} max_cameras:={} use_stereo:={} " \
                        #              "bag:={} bag_start:={} init_imu_thresh:={} dosave:=true path_est:={} " \
                        #              "gyroscope_noise_density:={} gyroscope_random_walk:={} " \
                        #              "accelerometer_noise_density:={} accelerometer_random_walk:={} ".format(
                        #     launch_file, max_cameras, use_stereo,
                        #     bag_fullname, bag_start, init_imu_thresh, result_file,
                        #     options["sigma_g_c"]*options["coeff_c"], options["sigma_gw_c"]*options["coeff_wc"], 
                        #     options["sigma_a_c"]*options["coeff_c"], options["sigma_aw_c"]*options["coeff_wc"])

                        launch_cmd = "roslaunch ov_msckf {} max_cameras:={} use_stereo:={} " \
                                     "bag:={} bag_start:={} init_imu_thresh:={} dosave:=true path_est:={}".format(
                            launch_file, max_cameras, use_stereo,
                            bag_fullname, bag_start, init_imu_thresh, result_file)


                    cmd = src_cmd + launch_cmd
                    src_wrap = os.path.join(result_dir, "source_wrap.sh")
                    with open(src_wrap, 'w') as stream:
                        stream.write('#!/bin/bash\n')
                        stream.write('{}\n'.format(cmd))
                    cmd = "chmod +x {wrap};{wrap}".format(wrap=src_wrap)

                    print(cmd)
                    rc, msg = utility_functions.subprocess_cmd(cmd)
                    if rc != 0:
                        err_msg = "Error code {} and msg {} in running vio method with cmd:\n{}".\
                            format(rc, msg, cmd)
                        print(err_msg)
                gt_file = os.path.join(result_dir, 'stamped_groundtruth.txt')
                shutil.copy2(gt_list[bag_index], gt_file)
                eval_cfg_file = os.path.join(result_dir, 'eval_cfg.yaml')
                shutil.copy2(eval_cfg_template, eval_cfg_file)

        #args.num_trials,
        # evaluate all VIO methods.
        rc, streamdata = rpg_eval_tool_wrap.run_rpg_evaluation(
            args.rpg_eval_tool_dir, results_dir_manager.get_eval_config_yaml(),
            args.num_trials, results_dir, eval_output_dir)
        if rc != 0:
            print(Fore.RED + "Error code {} in run_rpg_evaluation: {}".format(rc, streamdata))
            sys.exit(1)

        rpg_eval_tool_wrap.check_eval_result(eval_output_dir, args.cmp_eval_output_dir)

    print('Successfully finished testing methods in msckf project!')
    sys.exit(0)
