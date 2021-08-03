#!/usr/bin/env python3

'''
test VioConfigComposer
'''
from __future__ import print_function
import os

import VioConfigComposer


def test_get_calib_files():
    eval_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    composer = VioConfigComposer.VioConfigComposer(
        "../config/vio_common_params.yaml",
        '/data/euroc/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.bag',
        './new_config.yaml')
    imu_calib_file, camera_calib_files = composer.get_calib_files()
    assert imu_calib_file == os.path.join(eval_script_dir, "calibration/euroc/imu0.yaml")
    assert len(camera_calib_files) == 2
    assert camera_calib_files[0] == os.path.join(eval_script_dir, 'calibration/euroc/cam0.yaml')
    assert camera_calib_files[1] == os.path.join(eval_script_dir, 'calibration/euroc/cam1.yaml')

    composer_uzh = VioConfigComposer.VioConfigComposer(
        "../config/vio_common_params.yaml",
        '/data/uzh_fpv_drone_racing/indoor_45_3_snapdragon.bag',
        './new_config.yaml')
    imu_calib_file, camera_calib_files = composer_uzh.get_calib_files()
    assert imu_calib_file == os.path.join(eval_script_dir, 'calibration/uzh_fpv/imu-snapdragon_imu.yaml')
    assert len(camera_calib_files) == 1
    assert camera_calib_files[0] == os.path.join(
        eval_script_dir,
        'calibration/uzh_fpv/camchain-imucam-indoor_45_calib_snapdragon_imu.yaml')
