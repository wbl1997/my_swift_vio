#!/usr/bin/env python3
"""
create rosbags for advio data sessions and convert the groundtruth file to TUM-RGBD format.
"""

import numpy as np
import os
import sys

import zipfile

import dir_utility_functions
import utility_functions

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: {} <advio-dir containing all zip files immediately>'
              ' </vio_common/python/kalibr_bagcreator.py> <output_dir>'.
              format(sys.argv[0]))
        sys.exit(1)
    script, advio_dir, bagcreator, output_dir = sys.argv

    shift_secs = 1000  # shift the start epoch to avoid 0 rospy time.

    zip_list = dir_utility_functions.find_zips(advio_dir, "advio-")
    zip_list.sort()
    print('Extracting zip files \n{}'.format('\n'.join(zip_list)))

    extract_dir_list = []
    dir_utility_functions.mkdir_p(output_dir)
    for zip_file in zip_list:
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                extract_dir = os.path.join(output_dir, os.path.basename(zip_file).split('.')[0])
                extract_dir_list.append(extract_dir)
                zip_ref.extractall(output_dir)
        except Exception as err:
            print("Error in unzipping {}".format(zip_file))
    print('Unzipped data for {} advio missions'.format(len(extract_dir_list)))

    convert_pose_script = os.path.join(os.path.dirname(bagcreator), "convert_pose_format.py")
    transform_traj_script = os.path.join(os.path.dirname(bagcreator), "transform_trajectory.py")
    cmd = "chmod +x {bc};chmod +x {cp};chmod +x {tt}".format(
        bc=bagcreator, cp=convert_pose_script, tt=transform_traj_script)
    utility_functions.subprocess_cmd(cmd)

    # correct frames.csv by subtracting the first timestamp, see
    # https://github.com/AaltoVision/ADVIO/issues/2
    for session_dir in extract_dir_list:
        iphone_dir = os.path.join(session_dir, "iphone")
        video_time_csv = os.path.join(iphone_dir, "frames.csv")
        timelist = np.loadtxt(video_time_csv, delimiter=",", skiprows=0)
        timelist[:, 0] -= timelist[0, 0]
        time_corrected_csv = os.path.join(iphone_dir, "frames_corrected.csv")
        np.savetxt(time_corrected_csv, timelist, fmt=['%.7f', '%d'], delimiter=",")

    bag_output_dir = os.path.join(output_dir, "iphone")
    dir_utility_functions.mkdir_p(bag_output_dir)
    for session_dir in extract_dir_list:
        iphone_dir = os.path.join(session_dir, "iphone")
        gyro_csv = os.path.join(iphone_dir, "gyro.csv")
        accel_csv = os.path.join(iphone_dir, "accelerometer.csv")
        video = os.path.join(iphone_dir, "frames.mov")
        video_time_csv = os.path.join(iphone_dir, "frames_corrected.csv")

        output_bag = os.path.join(bag_output_dir, os.path.basename(session_dir) + ".bag")
        cmd = "{bc} --video {v} --imu {g} {a} --video_time_file {t} " \
              "--shift_secs {s} --output_bag {o}".\
            format(bc=bagcreator, v=video, g=gyro_csv, a=accel_csv,
                   t=video_time_csv, s=shift_secs, o=output_bag)
        out_stream = open(os.path.join(session_dir, "bag_out.log"), 'w')
        err_stream = open(os.path.join(session_dir, "bag_err.log"), 'w')
        print("Running cmd: {}".format(cmd))
        utility_functions.subprocess_cmd(cmd, out_stream, err_stream)
        out_stream.close()
        err_stream.close()
    print('Created rosbags for {} missions'.format(len(extract_dir_list)))

    # The transform from the world frame of advio poses.csv to a world frame with z along negative gravity.
    R_Wzup_Wgt = np.array([[0, 0, -1],
                           [-1, 0, 0],
                           [0, 1, 0]])
    T_Wzup_Wgt = np.eye(4)
    T_Wzup_Wgt[0:3, 0:3] = R_Wzup_Wgt
    left_transform_txt = os.path.join(bag_output_dir, "T_Wzup_Wgt.txt")
    np.savetxt(left_transform_txt, T_Wzup_Wgt, fmt='%f', delimiter=",")


    for session_dir in extract_dir_list:
        ground_truth_csv = os.path.join(session_dir, "ground-truth", "pose.csv")
        original_gt_txt = os.path.join(bag_output_dir, os.path.basename(session_dir) + ".orig.txt")
        cmd = 'python3 {} {} --in_quat_order wxyz --outfile {} --shift_secs {} --output_delimiter=" ";' \
              .format(convert_pose_script, ground_truth_csv, original_gt_txt, shift_secs)

        transformed_gt_txt = os.path.join(bag_output_dir, os.path.basename(session_dir) + ".txt")
        cmd += 'python3 {} --left_transform={} {} --output_txt {}'.format(
            transform_traj_script, left_transform_txt, original_gt_txt, transformed_gt_txt)
        out_stream = open(os.path.join(session_dir, "gt_out.log"), 'w')
        err_stream = open(os.path.join(session_dir, "gt_err.log"), 'w')
        utility_functions.subprocess_cmd(cmd, out_stream, err_stream)
        out_stream.close()
        err_stream.close()
    print('Converted ground truth for {} missions'.format(len(extract_dir_list)))


