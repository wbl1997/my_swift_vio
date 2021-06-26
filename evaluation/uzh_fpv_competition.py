
import os
import sys

import dir_utility_functions
import parse_args

import ResultsDirManager
import RunOneVioMethod

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

    algoname_to_options = {
        # We disable online extrinsic calibration for OKVIS by zeroing
        # sigma_absolute_translation and sigma_absolute_orientation.
        # 'OKVIS_nframe': {"algo_code": "OKVIS",
        #                  "extra_gflags": "",
        #                  "numKeyframes": 5,
        #                  "numImuFrames": 3,
        #                  "monocular_input": 0,
        #                  "landmarkModelId": 0,
        #                  "extrinsic_opt_mode_main_camera": "p_BC_q_BC",
        #                  "extrinsic_opt_mode_other_camera": "p_BC_q_BC",
        #                  "sigma_absolute_translation": "0.0",
        #                  "sigma_absolute_orientation": "0.0"},
        'MSCKF_n_part_fix': {"algo_code": "MSCKF",
                             "extra_gflags": "",
                             "numKeyframes": 5,
                             "numImuFrames": 5,
                             "monocular_input": 0,
                             "landmarkModelId": 1,
                             "extrinsic_opt_mode_main_camera": "P_CB",
                             "extrinsic_opt_mode_other_camera": "P_BC_Q_BC",
                             "sigma_absolute_translation": 0.02,
                             "sigma_absolute_orientation": 0.01,
                             "sigma_tr": 0.0,
                             "sigma_focal_length": 0.0,
                             "sigma_principal_point": 0.0,
                             "sigma_distortion": "[0.00, 0.0, 0.0, 0.0]",
                             "maxOdometryConstraintForAKeyframe": 2,
                             "sigma_g_c": 0.05,
                             "g_max": 31.4,  # 0.9 x 2000 DPS
                             "a_max": 176.0,  # 1.1 x 16g
                             "g: 9": 9.807,  # zurich https://units.fandom.com/wiki/Gravity_of_Earth
                             "loop_closure_method": 0
                             },
    }

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
            args.num_trials, bag_list, None,
            results_dir_manager.get_all_result_dirs(name),
            args.extra_lib_path, args.lcd_config_yaml,
            args.voc_file)
        rc = runner.run_method(name, args.pose_conversion_script, "uzh_fpv")
        if rc != 0:
            returncode = rc

    if returncode != 0:
        print("Error code {} in run_uzh_fpv_competition".format(returncode))
    else:
        print('Successfully finished processing uzh fpv competition dataset!')
    sys.exit(returncode)
