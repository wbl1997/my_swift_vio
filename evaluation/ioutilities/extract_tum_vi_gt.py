#!/usr/bin/env python3
"""
Extract gt from tum vi benchmark rosbags and save to txt files using
adapted bag_to_pose.py from rpg evaluation tool.
"""
import os
import sys

import dir_utility_functions
import utility_functions

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: {} <tum_vi_dir containing all bag files immediately>'
              ' <path/to/bag_to_pose.py>'.
              format(sys.argv[0]))
        sys.exit(1)
    script, tum_vi_dir, bag_to_pose_script = sys.argv

    bag_with_gt_list = dir_utility_functions.find_bags(tum_vi_dir, '512_16', 'dataset-calib')
    print('bags with gt\n{}'.format('\n'.join(bag_with_gt_list)))

    for bagname in bag_with_gt_list:
        shortname = os.path.basename(os.path.splitext(bagname)[0])
        cmd = 'python2 {} {} /vrpn_client/raw_transform --msg_type=TransformStamped ' \
              '--output={}.txt'.format(bag_to_pose_script, bagname, shortname)
        utility_functions.subprocess_cmd(cmd)
    print('Extracted ground truth for {} missions'.format(len(bag_with_gt_list)))
