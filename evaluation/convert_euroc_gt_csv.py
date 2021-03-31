#!/usr/bin/env python3

import os
import sys

import dir_utility_functions
import utility_functions

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: {} <euroc dir with a layout structure depicted in main_evaluation.py>'
              ' <path/to/vio_common/python/convert_pose_format.py>'.
              format(sys.argv[0]))
        sys.exit(1)
    script, euroc_dir, to_pose_script = sys.argv

    bag_list = dir_utility_functions.find_bags(euroc_dir, '', 'calibration')
    gt_list = dir_utility_functions.get_original_euroc_gt_files(bag_list)
    for gt_file in gt_list:
        baldname = os.path.splitext(gt_file)[0]
        cmd = "python3 {} {} --outfile={}.txt --output_delimiter=' ' --in_quat_order=wxyz".\
            format(to_pose_script, gt_file, baldname)
        utility_functions.subprocess_cmd(cmd)
    print('Converted ground truth for {} EUROC missions'.format(len(gt_list)))