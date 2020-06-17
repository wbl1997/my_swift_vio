import argparse
import os
import sys

def check_common_args(args):
    assert os.path.isfile(args.vio_config_yaml)
    euroc_exist = os.path.exists(args.euroc_dir)
    uzh_fpv_exist = os.path.exists(args.uzh_fpv_dir)
    homebrew_exist = os.path.exists(args.homebrew_data_dir)
    assert euroc_exist or uzh_fpv_exist or homebrew_exist, \
        "Data dirs for euroc, uzh-fpv, and homebrew do not exist"

    assert os.path.exists(args.rpg_eval_tool_dir), \
        'rpg_trajectory_evaluation does not exist'
    print('Input args to {}'.format(sys.argv[0]))
    for arg in vars(args):
        print(arg, getattr(args, arg))


def parse_args():
    parser = argparse.ArgumentParser(
        description='''Evaluate algorithms inside msckf project on many EUROC 
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
