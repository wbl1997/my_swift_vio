#!/usr/bin/env python3
import subprocess, yaml


def get_rosbag_duration(bag_fullname):
    info_dict = yaml.load(subprocess.Popen(['rosbag', 'info', '--yaml', bag_fullname],
                                           stdout=subprocess.PIPE).communicate()[0],
                          Loader=yaml.SafeLoader)
    return info_dict["duration"]
