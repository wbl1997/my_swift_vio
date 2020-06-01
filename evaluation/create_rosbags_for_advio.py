#!/usr/bin/env python3
"""
create rosbags for advio data sessions and convert the groundtruth file to TUM-RGBD format.
"""
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

    zip_list = dir_utility_functions.find_zips(advio_dir, "advio-")
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

    bag_output_dir = os.path.join(output_dir, "iphone")
    dir_utility_functions.mkdir_p(bag_output_dir)
    for session_dir in extract_dir_list:
        iphone_dir = os.path.join(session_dir, "iphone")
        gyro_csv = os.path.join(iphone_dir, "gyro.csv")
        accel_csv = os.path.join(iphone_dir, "accelerometer.csv")
        video = os.path.join(iphone_dir, "frames.mov")
        video_time_csv = os.path.join(iphone_dir, "frames.csv")

        output_bag = os.path.join(bag_output_dir, os.path.basename(session_dir) + ".bag")
        cmd = "chmod +x {bc};{bc} --video {v} --imu {g} {a} --video_time_file {t} --output_bag {o}".\
            format(bc=bagcreator, v=video, g=gyro_csv, a=accel_csv, t=video_time_csv, o=output_bag)
        out_stream = open(os.path.join(session_dir, "bag_out.log"), 'w')
        err_stream = open(os.path.join(session_dir, "bag_err.log"), 'w')
        utility_functions.subprocess_cmd(cmd, out_stream, err_stream)
        out_stream.close()
        err_stream.close()

    print('Created rosbags for {} missions'.format(len(extract_dir_list)))

    convert_pose_script = os.path.join(os.path.dirname(bagcreator), "convert_pose_format.py")
    for session_dir in extract_dir_list:
        ground_truth_csv = os.path.join(session_dir, "ground-truth", "pose.csv")
        converted_txt = os.path.join(bag_output_dir, os.path.basename(session_dir) + ".txt")
        cmd = 'python3 {} {} --in_quat_order wxyz --outfile {} --output_delimiter=" "' \
              .format(convert_pose_script, ground_truth_csv, converted_txt)
        out_stream = open(os.path.join(session_dir, "gt_out.log"), 'w')
        err_stream = open(os.path.join(session_dir, "gt_err.log"), 'w')
        utility_functions.subprocess_cmd(cmd, out_stream, err_stream)
        out_stream.close()
        err_stream.close()
    print('Converted ground truth for {} missions'.format(len(extract_dir_list)))
