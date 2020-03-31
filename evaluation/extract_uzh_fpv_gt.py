#!/usr/bin/env python3

import os
import sys

import dir_utility_functions
import utility_functions

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: {} <uzh_fpv_dir containing all bag files immediately>'
              ' <path/to/bag_to_pose.py>'.
              format(sys.argv[0]))
        sys.exit(1)
    script, uzh_fpv_dir, bag_to_pose_script = sys.argv

    bag_with_gt_list = dir_utility_functions.find_bags_with_gt(uzh_fpv_dir, 'snapdragon')
    print('bags with gt\n{}'.format('\n'.join(bag_with_gt_list)))

    for bagname in bag_with_gt_list:
        shortname = os.path.basename(os.path.splitext(bagname)[0])
        cmd = 'python2 {} {} /groundtruth/pose --msg_type=PoseStamped ' \
              '--output={}.txt'.format(bag_to_pose_script, bagname, shortname)
        utility_functions.subprocess_cmd(cmd)
    print('Extracted ground truth for {} missions'.format(len(bag_with_gt_list)))
