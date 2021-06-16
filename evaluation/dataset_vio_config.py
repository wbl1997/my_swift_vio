#!/usr/bin/python

"""
Generate configuration file to OKVIS for a variety of data.
The interface is main_vio_config.py.
author: Yukai Lin
author: Jianzhu Huai
"""
import argparse
import copy
import os
from ruamel.yaml import YAML

import numpy as np

# The suitable IMU parameters for stereo OKVIS, monocular MSCKF, and stereo MSCKF.
# We did not tune for these methods with a grid search.
# https://github.com/ethz-asl/okvis/blob/master/config/config_fpga_p2_euroc.yaml
OKVIS_EUROC_IMU_PARAMETERS = {"sigma_g_c": 12.0e-4,
                              "sigma_a_c": 8.0e-3,
                              "sigma_gw_c": 4.0e-6,
                              "sigma_aw_c": 4.0e-5}

# The best IMU parameters for monocular OKVIS on EUROC are found by a search.
#       &      Translation (\%) &  Rotation (deg/meter)
# OKVIS_1_1 &     135.089 &  0.138
# OKVIS_2_2 &     33.327 &  0.175
# OKVIS_2_4 &     6.651 &  0.054
# OKVIS_4_4 &     0.656 &  0.024
OKVIS_MONO_EUROC_IMU_PARAMETERS = {
    "sigma_g_c": 12.0e-4 * 4,
    "sigma_a_c": 8.0e-3 * 4,
    "sigma_gw_c": 4.0e-6 * 4,
    "sigma_aw_c": 4.0e-5 * 4,
}

# The best IMU parameters for stereo MSCKF on TUM VI are found by a search.
#       &      Translation (\%) &  Rotation (deg/meter)
# KSF_n_01_01 &     31.934 &  0.135
# KSF_n_02_025 &     41.982 &  0.120
# KSF_n_0025_005 &     84470.155 &  0.313
# KSF_n_005_005 &     786.133 &  0.260
# KSF_n_005_01 &     42.373 &  0.151
MSCKF_TUMVI_IMU_PARAMETERS = {"sigma_g_c": 0.004 * 0.1,
                              "sigma_a_c": 0.07 * 0.1,
                              "sigma_gw_c": 4.4e-5 * 0.1,
                              "sigma_aw_c": 1.72e-3 * 0.1}

# The best IMU parameters for monocular MSCKF on TUM VI are found by a search.
#       &      Translation (\%) &  Rotation (deg/meter)
# KSF_005_01 &     37578.149 &  0.258
# KSF_01_01 &     70.471 &  0.186
# KSF_01_025 &     47.796 &  0.117
# KSF_02_025 &     42.107 &  0.118
MSCKF_TUMVI_MONO_IMU_PARAMETERS = {
    "sigma_g_c": 0.004 * 0.2,
    "sigma_a_c": 0.07 * 0.2,
    "sigma_gw_c": 4.4e-5 * 0.25,
    "sigma_aw_c": 1.72e-3 * 0.25
}

TUMVI_PARAMETERS = {
    "cameras": [
        {"T_SC": [-0.99953071, 0.00744168, -0.02971511, 0.04536566,
                  0.0294408, -0.03459565, -0.99896766, -0.071996,
                  -0.00846201, -0.99937369, 0.03436032, -0.04478181,
                  0.0, 0.0, 0.0, 1.0],
         "image_dimension": [512, 512],
         "distortion_coefficients": [0.0034823894022493434, 0.0007150348452162257, -0.0020532361418706202,
                                     0.00020293673591811182],
         "distortion_type": "equidistant",
         "focal_length": [190.97847715128717, 190.9733070521226],
         "principal_point": [254.93170605935475, 256.8974428996504],
         "projection_opt_mode": "FXY_CXY",
         "extrinsic_opt_mode": "P_CB",
         "image_delay": 0.0,
         "image_readout_time": 0.00},
        {"T_SC":
             [-0.99951678, 0.00803569, -0.03002713, -0.05566603,
              0.03012473, 0.01231336, -0.9994703, -0.07010225,
              -0.0076617, -0.9998919, -0.01254948, -0.0475471,
              0., 0., 0., 1.],
         "image_dimension": [512, 512],
         "distortion_coefficients": [0.0034003170790442797, 0.001766278153469831, -0.00266312569781606,
                                     0.0003299517423931039],
         "distortion_type": "equidistant",
         "focal_length": [190.44236969414825, 190.4344384721956],
         "principal_point": [252.59949716835982, 254.91723064636983],
         "projection_opt_mode": "FXY_CXY",
         "extrinsic_opt_mode": "P_BC_Q_BC",
         "image_delay": 0.0,
         "image_readout_time": 0.00
         }],
    "imu_params": {
        'imu_rate': 200,
        'g_max': 7.8,
        'sigma_g_c': 0.004,
        'sigma_a_c': 0.07,
        'sigma_gw_c': 4.4e-5,
        'sigma_aw_c': 1.72e-3,
        'g': 9.80766,
        'sigma_TGElement': 5e-3,
        'sigma_TSElement': 1e-3,
        'sigma_TAElement': 5e-3, },
    'ceres_options': {
        'timeLimit': 1,  # in units of seconds, -1 means no limit.
    },
    "displayImages": "false",
    "publishing_options": {
        'publishLandmarks': "false", }
}

TUMVI_NOMINAL_PARAMETERS = {
    "cameras": [
        {"T_SC": [-1, 0.0, 0.0, 0.0,
                  0.0, 0.0, -1.0, 0.0,
                  0.0, -1.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 1.0],
         "image_dimension": [512, 512],
         "distortion_coefficients": [0.0, 0.0, 0.0, 0.0],
         "distortion_type": "equidistant",
         "focal_length": [190.0, 190.0],
         "principal_point": [256.0, 256.0],
         "projection_opt_mode": "FX_CXY",
         "extrinsic_opt_mode": "P_CB",
         "image_delay": 0.0,
         "image_readout_time": 0.00},
        {"T_SC":
             [-1.0, 0.0, 0.0, 0.0,
              0.0, 0.0, -1.0, 0.0,
              0.0, -1.0, 0.0, 0.0,
              0., 0., 0., 1.],
         "image_dimension": [512, 512],
         "distortion_coefficients": [0.0, 0.0, 0.0, 0.0],
         "distortion_type": "equidistant",
         "focal_length": [190.0, 190.0],
         "principal_point": [256.0, 256.0],
         "projection_opt_mode": "FX_CXY",
         "extrinsic_opt_mode": "P_BC_Q_BC",
         "image_delay": 0.0,
         "image_readout_time": 0.00
         }],
    "imu_params": {
        'imu_rate': 200,
        'g_max': 7.8,
        'sigma_g_c': 0.004,
        'sigma_a_c': 0.07,
        'sigma_gw_c': 4.4e-5,
        'sigma_aw_c': 1.72e-3,
        'g': 9.80766,
        'sigma_TGElement': 5e-3,
        'sigma_TSElement': 1e-3,
        'sigma_TAElement': 5e-3, },
    'ceres_options': {
        'timeLimit': 1,
    },
    "displayImages": "false",
    "publishing_options": {
        'publishLandmarks': "false", }
}

# The best IMU parameters for monocular MSCKF on ADVIO are found a search and visual check.
#       &      Translation (\%) &  Rotation (deg/meter)
# KSF_0.1_0.1 &     15.113 &  0.129
# KSF_0.2_0.1 &     14.167 &  0.125
# KSF_0.2_0.2 &     13.792 &  0.124
# KSF_0.5_0.2 &     13.795 &  0.126
# KSF_0.5_0.5 &     13.753 &  0.125
# KSF_1.0_0.5 &     15.405 &  0.127
# KSF_1.0_1.0 &     17.616 &  0.129
# KSF_2.0_1.0 &     21.493 &  0.134
# KSF_2.0_2.0 &     20.310 &  0.136
MSCKF_ADVIO_IMU_PARAMETERS = {
    'sigma_g_c': 2.4e-3 * 0.5,
    'sigma_a_c': 4.8e-2 * 0.5,
    'sigma_gw_c': 5.1e-5 * 0.2,
    'sigma_aw_c': 2.1e-4 * 0.2,
}

# The best IMU parameters for monocular OKVIS on ADVIO are found a search and visual check.
#       &      Translation (\%) &  Rotation (deg/meter)
# OKVIS_0.5_0.5 &     206955.716 &  0.863
# OKVIS_0.5_1.0 &     152459.404 &  0.814
# OKVIS_1.0_1.0 &     518639.908 &  0.874
# OKVIS_1.0_2.0 &     820545.006 &  0.858
# OKVIS_2.0_2.0 &     900615.497 &  0.892
# OKVIS_2.0_5.0 &     3500755.814 &  0.912
# OKVIS_5.0_10.0 &     2106334.534 &  0.860
# OKVIS_5.0_5.0 &     2591059.557 &  0.799
# OKVIS_10.0_10.0 &     1372010.014 &  0.474
OKVIS_ADVIO_IMU_PARAMETERS = {
    'sigma_g_c': 2.4e-3 * 2,
    'sigma_a_c': 4.8e-2 * 2,  # 4.8e-3 is the value provided by advio.
    'sigma_gw_c': 5.1e-5 * 2,
    'sigma_aw_c': 2.1e-4 * 2,
}

ADVIO_CAMERA_INTRINSIC_PARAMETERS = {
    range(1, 13): [1077.2, 1079.3, 362.14, 636.39, 0.0478, 0.0339, -0.0003, -0.0009],
    range(13, 18): [1082.4, 1084.4, 364.68, 643.31, 0.0366, 0.0803, 0.0007, -0.0002],
    range(18, 20): [1076.9, 1078.5, 360.96, 639.31, 0.0510, -0.0354, -0.0054, 0.0473],
    range(20, 24): [1081.1, 1082.1, 359.59, 640.79, 0.0556, -0.0454, 0.0009, -0.0018],
}

ADVIO_NOMINAL_CAMERA_INTRINSIC_PARAMETERS = {
    range(1, 24): [1080.0, 1080.0, 360.0, 640.0, 0.0, 0.0, 0.0, 0.0],
}

ADVIO_PARAMETERS = {
    "cameras":
        {"T_CpS": [0.9999763379093255, -0.004079205042965442, -0.005539287650170447, -0.008977668364731128,
                  -0.004066386342107199, -0.9999890330121858, 0.0023234365646622014, 0.07557012320238939,
                  -0.00554870467502187, -0.0023008567036498766, -0.9999819588046867, -0.005545773942541918,
                  0.0, 0.0, 0.0, 1.0],
         "image_dimension": [720, 1280],
         "distortion_coefficients": [],
         "distortion_type": "radial-tangential",
         "focal_length": [],
         "principal_point": [],
         "projection_opt_mode": "FX_CXY",
         "extrinsic_opt_mode": "P_CB",
         "image_delay": 0.0,
         "image_readout_time": 0.015},
    "imu_params": {
        "imu_rate": 100,
        'g_max': 7.8,
        'sigma_g_c': 2.4e-3,
        'sigma_a_c': 4.8e-2,  # 4.8e-3 is the value provided by advio.
        'sigma_gw_c': 5.1e-5,
        'sigma_aw_c': 2.1e-4,
        'g': 9.819,  # at Helsinki according to https://units.fandom.com/wiki/Gravity_of_Earth
        'sigma_TGElement': 5e-3,
        'sigma_TSElement': 1e-3,
        'sigma_TAElement': 5e-3, },
    'ceres_options': {
        'timeLimit': 1,
    },
    "displayImages": "false",
    "publishing_options": {
        'publishLandmarks': "false", }
}

ADVIO_NOMINAL_PARAMETERS = {
    "cameras":
        {"T_CpS": [1.0, 0.0, 0.0, 0.0,
                   0.0, -1.0, 0.0, 0.0,
                   0.0, 0.0, -1.0, 0.0,
                  0.0, 0.0, 0.0, 1.0],
         "image_dimension": [720, 1280],
         "distortion_coefficients": [],
         "distortion_type": "radial-tangential",
         "focal_length": [],
         "principal_point": [],
         "projection_opt_mode": "FX_CXY",
         "extrinsic_opt_mode": "P_CB",
         "image_delay": 0.0,
         "image_readout_time": 0.015},
    "imu_params": {
        "imu_rate": 100,
        'g_max': 7.8,
        'sigma_g_c': 2.4e-3,
        'sigma_a_c': 4.8e-2,  # 4.8e-3 is the value provided by advio.
        'sigma_gw_c': 5.1e-5,
        'sigma_aw_c': 2.1e-4,
        'g': 9.819,  # at Helsinki according to https://units.fandom.com/wiki/Gravity_of_Earth
        'sigma_TGElement': 5e-3,
        'sigma_TSElement': 1e-3,
        'sigma_TAElement': 5e-3, },
    'ceres_options': {
        'timeLimit': 1,
    },
    "displayImages": "false",
    "publishing_options": {
        'publishLandmarks': "false", }
}

def advio_transform_C_Cp_and_halve(use_nominal_value):
    """
    The given advio params are in Cp frame, this function applies
     T_CCp = [R_CCp, 0; 0, 1] on relevant quantities, and halve the camera size.
     R_CCp = [0, 1, 0; -1, 0, 0; 0, 0, 1]
    :return:
    """
    coeff = 0.5
    if use_nominal_value:
        swapped_intrinsics = copy.deepcopy(ADVIO_NOMINAL_CAMERA_INTRINSIC_PARAMETERS)
        swapped_parameters = copy.deepcopy(ADVIO_NOMINAL_PARAMETERS)
    else:
        swapped_intrinsics = copy.deepcopy(ADVIO_CAMERA_INTRINSIC_PARAMETERS)
        swapped_parameters = copy.deepcopy(ADVIO_PARAMETERS)

    for key in swapped_intrinsics.keys():
        oldvalue = swapped_intrinsics[key]
        swapped_intrinsics[key] =\
            [oldvalue[1] * coeff, oldvalue[0] * coeff,
             oldvalue[3] * coeff, oldvalue[2] * coeff,
             oldvalue[4], oldvalue[5], oldvalue[7], oldvalue[6]]

    T_CpS = np.array(swapped_parameters["cameras"]["T_CpS"]).reshape([4, 4])
    T_CCp = np.array([[0, 1, 0, 0],
                      [-1, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0.0, 0.0, 0.0, 1.0]])
    T_SC = np.linalg.inv(np.matmul(T_CCp, T_CpS))
    swapped_parameters["cameras"]["T_SC"] = sum(T_SC.tolist(), [])
    image_shape = swapped_parameters["cameras"]["image_dimension"]
    swapped_parameters["cameras"]["image_dimension"] = \
        [image_shape[1] * coeff, image_shape[0] * coeff]
    return swapped_parameters, swapped_intrinsics


def get_advio_sequence_unmber(bagname):
    barename = os.path.basename(bagname).split('.')[0]
    return int(barename.split('-')[1])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_template",
                        help="Input okvis config template yaml. It will not be modified.")
    parser.add_argument("format",
                        help="Input sensor config yaml format, 'kalibr' or 'euroc'. "
                             "kalibr format is used by UZH-FPV dataset. euroc "
                             "sensor yaml format is used inside euroc dataset")

    parser.add_argument("--camera_config_yamls", type=str, nargs='+',
                        help="A list of camera sensor config yamls",
                        required=False)

    parser.add_argument("--imu_config_yaml",
                        help="The IMU sensor config yaml",
                        required=False)

    parser.add_argument("--output_okvis_config",
                        help="Output config yaml accommodating the sensor parameters",
                        required=True)
    parser.add_argument("--algo_code",
                        help="Which algorithm to use, MSCKF or OKVIS. This only affects"
                             " IMU noise parameters in TUM VI config yaml.",
                        default="",
                        required=False)

    parser.add_argument("--rosbag",
                        help="Name of the rosbag for which the config will be used. "
                             "Only needed for advio to extract sequence number.",
                        default="",
                        required=False)

    parser.add_argument("--use_nominal_value",
                        help='Use nominal values to fill the config yaml?',
                        action='store_true')

    parser.add_argument("--monocular_input",
                        help='Use monocular input?',
                        action='store_true')

    args = parser.parse_args()
    return args


# That's how the output should should look like:
#
# base_topic: /
# cameras:
#     - {image_base_topic: cam0/, image_topic: image_raw, info_topic: camera_info,
#        T_SC:
#        [  0.0158177,  -0.999874 ,  -0.00133516, -0.01402297,
#           0.99987346,  0.01581992, -0.00166903, -0.07010539,
#           0.00168994, -0.00130859,  0.99999772,  0.00471241,
#           0. ,         0.    ,      0.       ,   1.       ],
#        image_dimension: [752, 480],
#        distortion_coefficients: [-0.0016170650137774234, 0.017615842489373677, -0.020236038143470282,
#     0.010279726211982943],
#        distortion_type: equidistant,
#        focal_length: [460.27046835000937, 458.7889758618953],
#        principal_point: [355.2403080101758, 232.60725397709305]}


def printCameraBlock(camConfig):
    T_SC = np.array(camConfig['T_SC']).reshape([4, 4])

    STRING_OUT = ""
    STRING_OUT += "    - {"
    STRING_OUT += "T_SC:\n"
    STRING_OUT+="       [ {0}, {1}, {2}, {3},\n".format(T_SC[0,0], T_SC[0,1], T_SC[0,2], T_SC[0,3])
    STRING_OUT+="         {0}, {1}, {2}, {3},\n".format(T_SC[1,0], T_SC[1,1], T_SC[1,2], T_SC[1,3])
    STRING_OUT+="         {0}, {1}, {2}, {3},\n".format(T_SC[2,0], T_SC[2,1], T_SC[2,2], T_SC[2,3])
    STRING_OUT += "          0.0, 0.0, 0.0, 1.0],\n"

    resolution = camConfig['image_dimension']
    STRING_OUT += "       image_dimension: [{}, {}],\n".format(resolution[0], resolution[1])

    dist_model = camConfig['distortion_type']
    dist_params = camConfig['distortion_coefficients']

    STRING_OUT += "       distortion_coefficients: [{0}, {1}, {2}, {3}],\n".format(dist_params[0], dist_params[1],
                                                                                   dist_params[2], dist_params[3])
    STRING_OUT += "       distortion_type: {0},\n".format(dist_model)

    focal_length = camConfig['focal_length']
    principal_point = camConfig['principal_point']
    STRING_OUT += "       focal_length: [{0}, {1}],\n".format(focal_length[0], focal_length[1])
    STRING_OUT += "       principal_point: [{0}, {1}],\n".format(principal_point[0], principal_point[1])
    STRING_OUT += "       projection_opt_mode: FXY_CXY,\n"
    STRING_OUT += "       extrinsic_opt_mode: P_CB,\n"
    STRING_OUT += "       image_delay: {},\n".format(camConfig["image_delay"])
    STRING_OUT += "       image_readout_time: {}".format(camConfig["image_readout_time"]) + "}\n"
    STRING_OUT += "\n"
    return STRING_OUT


def create_config_yaml(config_template, calib_format,
                       output_config, camera_config_yamls=None,
                       imu_config_yaml="",
                       algo_code="", bagname="", use_nominal_value=False,
                       monocular_input=False):
    """
    create configuration yaml to swift_vio for a dataset.
    :param config_template:
    :param calib_format: supported calibration format: uzh_fpv uses kalibr format,
     euroc format, tumvi format, amd advio format.
    :param camera_config_yamls:
    :param imu_config_yaml:
    :param algo_code: MSCKF or OKVIS, only for TUM-VI calibration format.
    :param bagname: only for ADVIO dataset to extract sequence number
    :param output_config:
    :return:
    """
    out_config = open(output_config, 'w')
    yaml = YAML()
    yaml.version = (1, 2)
    yaml.default_flow_style = None
    yaml.indent(mapping=4, sequence=6, offset=4)

    with open(config_template, 'r') as template_config:
        template_config.readline()
        template_data = yaml.load(template_config)

    # TODO(jhuai): move the following parameters into the corresponding yaml under swift_vio/config.
    imu_data = None
    if calib_format == 'kalibr':
        with open(camera_config_yamls[0], 'r') as camera_config:
            camera_data = yaml.load(camera_config)

        for cameraid in range(0, 2):
            camera_name = 'cam%d' % cameraid
            T_cam_imu = np.array(camera_data[camera_name]['T_cam_imu'])
            T_imu_cam = np.linalg.inv(T_cam_imu)
            template_data['cameras'][cameraid]['T_SC'] = sum(T_imu_cam.tolist(), [])
            template_data['cameras'][cameraid]['image_dimension'] = \
                camera_data[camera_name]['resolution']
            template_data['cameras'][cameraid]['distortion_coefficients'] = \
                camera_data[camera_name]['distortion_coeffs']
            template_data['cameras'][cameraid]['focal_length'] = \
                camera_data[camera_name]['intrinsics'][0:2]
            template_data['cameras'][cameraid]['principal_point'] = \
                camera_data[camera_name]['intrinsics'][2:4]
            if camera_data[camera_name]['distortion_model'] == 'radtan':
                template_data['cameras'][cameraid]['distortion_type'] = 'radialtangential'
            elif camera_data[camera_name]['distortion_model'] == 'equidistant':
                template_data['cameras'][cameraid]['distortion_type'] = 'equidistant'
            template_data['cameras'][cameraid]['image_delay'] = \
                camera_data[camera_name]['timeshift_cam_imu']

        with open(imu_config_yaml, 'r') as imu_config:
            imu_data = yaml.load(imu_config)['imu0']
        template_data['imu_params']['T_BS'] = sum(imu_data['T_i_b'], [])
        template_data['imu_params']['imu_rate'] = int(imu_data['update_rate'])
        template_data['imu_params']['sigma_a_c'] = imu_data['accelerometer_noise_density']
        template_data['imu_params']['sigma_aw_c'] = imu_data['accelerometer_random_walk']
        template_data['imu_params']['sigma_g_c'] = imu_data['gyroscope_noise_density']
        template_data['imu_params']['sigma_gw_c'] = imu_data['gyroscope_random_walk']
    elif calib_format == 'euroc':
        for cameraid in range(0, 2):
            with open(camera_config_yamls[cameraid], 'r') as camera_config:
                camera_data = yaml.load(camera_config)

            template_data['cameras'][cameraid]['T_SC'] = \
                camera_data['T_BS']['data']
            template_data['cameras'][cameraid]['image_dimension'] = \
                camera_data['resolution']
            template_data['cameras'][cameraid]['distortion_coefficients'] = \
                camera_data['distortion_coefficients']
            template_data['cameras'][cameraid]['focal_length'] = \
                camera_data['intrinsics'][0:2]
            template_data['cameras'][cameraid]['principal_point'] = \
                camera_data['intrinsics'][2:4]
            if camera_data['distortion_model'] == 'radial-tangential':
                template_data['cameras'][cameraid]['distortion_type'] = 'radialtangential'
            elif camera_data['distortion_model'] == 'equidistant':
                template_data['cameras'][cameraid]['distortion_type'] = 'equidistant'
        with open(imu_config_yaml, 'r') as imu_config:
            imu_data = yaml.load(imu_config)
        template_data['imu_params']['T_BS'] = imu_data['T_BS']['data']
        template_data['imu_params']['imu_rate'] = int(imu_data['rate_hz'])
        if monocular_input and algo_code == "OKVIS":
            for key in OKVIS_MONO_EUROC_IMU_PARAMETERS.keys():
                template_data["imu_params"][key] = OKVIS_MONO_EUROC_IMU_PARAMETERS[key]
        else:
            for key in OKVIS_EUROC_IMU_PARAMETERS.keys():
                template_data["imu_params"][key] = OKVIS_EUROC_IMU_PARAMETERS[key]
    elif calib_format == "tum_vi":
        used_TUMVI_parameters = TUMVI_PARAMETERS
        if use_nominal_value:
            used_TUMVI_parameters = TUMVI_NOMINAL_PARAMETERS
        for cameraid in range(0, 2):
            for key in used_TUMVI_parameters['cameras'][cameraid].keys():
                template_data['cameras'][cameraid][key] = used_TUMVI_parameters['cameras'][cameraid][key]
        for group in ["imu_params", "ceres_options", "publishing_options"]:
            for key in used_TUMVI_parameters[group].keys():
                template_data[group][key] = used_TUMVI_parameters[group][key]
        template_data["displayImages"] = used_TUMVI_parameters["displayImages"]
        if algo_code == "MSCKF" or algo_code == "HybridFilter":
            if monocular_input:
                for key in MSCKF_TUMVI_MONO_IMU_PARAMETERS.keys():
                    template_data["imu_params"][key] = MSCKF_TUMVI_MONO_IMU_PARAMETERS[key]
            else:
                for key in MSCKF_TUMVI_IMU_PARAMETERS.keys():
                    template_data["imu_params"][key] = MSCKF_TUMVI_IMU_PARAMETERS[key]
    elif calib_format == 'advio':
        swapped_parameters, swapped_intrinsics = advio_transform_C_Cp_and_halve(use_nominal_value)
        cameraid = 0
        template_data['cameras'][cameraid]['T_SC'] = \
            swapped_parameters["cameras"]['T_SC']
        seq_num = get_advio_sequence_unmber(bagname)
        intrinsics = []
        for key in swapped_intrinsics.keys():
            if seq_num in key:
                intrinsics = swapped_intrinsics[key]
                break

        template_data['cameras'][cameraid]['image_dimension'] = \
            swapped_parameters["cameras"]["image_dimension"]
        template_data['cameras'][cameraid]['distortion_coefficients'] = \
            intrinsics[4:]
        template_data['cameras'][cameraid]['focal_length'] = \
            intrinsics[0:2]
        template_data['cameras'][cameraid]['principal_point'] = \
            intrinsics[2:4]
        template_data['cameras'][cameraid]['distortion_type'] = 'radialtangential'

        template_data['cameras'][cameraid]['projection_opt_mode'] = \
            swapped_parameters["cameras"]["projection_opt_mode"]
        template_data['cameras'][cameraid]['extrinsic_opt_mode'] = \
            swapped_parameters["cameras"]["extrinsic_opt_mode"]
        template_data['cameras'][cameraid]['image_delay'] = \
            swapped_parameters["cameras"]["image_delay"]
        template_data['cameras'][cameraid]['image_readout_time'] = \
            swapped_parameters["cameras"]["image_readout_time"]
        template_data["camera_params"]["camera_rate"] = 60

        for group in ["imu_params", "ceres_options", "publishing_options"]:
            for key in swapped_parameters[group].keys():
                template_data[group][key] = swapped_parameters[group][key]
        template_data["displayImages"] = ADVIO_PARAMETERS["displayImages"]

        if algo_code == "MSCKF" or algo_code == "HybridFilter":
            for key in MSCKF_ADVIO_IMU_PARAMETERS.keys():
                template_data["imu_params"][key] = MSCKF_ADVIO_IMU_PARAMETERS[key]
        elif algo_code == "OKVIS":
            for key in OKVIS_ADVIO_IMU_PARAMETERS.keys():
                template_data["imu_params"][key] = OKVIS_ADVIO_IMU_PARAMETERS[key]

    elif calib_format == "homebrew":
        pass
    elif calib_format == 'tum_rs':
        pass

    # camera_config_str = printCameraBlock(template_data["cameras"][0])
    # camera_config_str += printCameraBlock(template_data["cameras"][1])
    # template_data['cameras'] = camera_config_str # dump simply keeps all newlines.
    # print("camera str \n{}\n".format(camera_config_str))

    yaml.dump(template_data, out_config)
    out_config.close()

