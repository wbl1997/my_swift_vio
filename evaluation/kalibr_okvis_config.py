#!/usr/bin/python

"""
author: Yukai Lin
author: Jianzhu Huai
"""
import argparse
from ruamel.yaml import YAML

import numpy as np

# https://github.com/ethz-asl/okvis/blob/master/config/config_fpga_p2_euroc.yaml
OKVIS_EUROC_IMU_PARAMETERS = {"sigma_g_c": 12.0e-4,
                              "sigma_a_c": 8.0e-3,
                              "sigma_gw_c": 4.0e-6,
                              "sigma_aw_c": 4.0e-5}

MSCKF_TUMVI_IMU_PARAMETERS = {"sigma_g_c": 2e-3,
                              "sigma_a_c": 1.75e-2,
                              "sigma_gw_c": 4.4e-5,
                              "sigma_aw_c": 1.72e-3}

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
        'timeLimit': -1,
    },
    "displayImages": "false",
    "publishing_options": {
        'publishLandmarks': "false", }
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("okvis_config_template",
                        help="Input okvis config template yaml. It will not be modified.")
    parser.add_argument("format",
                        help="Input sensor config yaml format, 'kalibr' or 'euroc'. "
                             "kalibr format is used by UZH-FPV dataset. euroc "
                             "sensor yaml format is used inside euroc dataset")

    parser.add_argument("--camera_config_yamls", type=str, nargs='+',
                        help="A list of camera sensor config yamls",
                        required=True)
    parser.add_argument("--imu_config_yaml",
                        help="The IMU sensor config yaml",
                        required=True)

    parser.add_argument("--output_okvis_config",
                        help="Output okvis config yaml accommodating the sensor parameters",
                        required=True)
    parser.add_argument("--algo_code",
                        help="Which algorithm to use, MSCKF or OKVIS. This only affects"
                             " IMU noise parameters in TUM VI config yaml.",
                        default="",
                        required=False)
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


def create_okvis_config_yaml(okvis_config_template, calib_format,
                             camera_config_yamls, imu_config_yaml,
                             algo_code,
                             output_okvis_config):
    out_config = open(output_okvis_config, 'w')
    yaml = YAML()
    yaml.version = (1, 2)
    yaml.default_flow_style = None
    yaml.indent(mapping=4, sequence=6, offset=4)

    with open(okvis_config_template, 'r') as template_config:
        template_config.readline()
        template_data = yaml.load(template_config)

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
        template_data['imu_params']['sigma_a_c'] = OKVIS_EUROC_IMU_PARAMETERS['sigma_a_c']
        template_data['imu_params']['sigma_aw_c'] =OKVIS_EUROC_IMU_PARAMETERS['sigma_aw_c']
        template_data['imu_params']['sigma_g_c'] = OKVIS_EUROC_IMU_PARAMETERS['sigma_g_c']
        template_data['imu_params']['sigma_gw_c'] = OKVIS_EUROC_IMU_PARAMETERS['sigma_gw_c']
    elif calib_format == "tum-vi":
        for cameraid in range(0, 2):
            for key in TUMVI_PARAMETERS['cameras'][cameraid].keys():
                template_data['cameras'][cameraid][key] = TUMVI_PARAMETERS['cameras'][cameraid][key]
        for group in ["imu_params", "ceres_options", "publishing_options"]:
            for key in TUMVI_PARAMETERS[group].keys():
                template_data[group][key] = TUMVI_PARAMETERS[group][key]
        template_data["displayImages"] = TUMVI_PARAMETERS["displayImages"]
        if algo_code == "MSCKF":
            for key in MSCKF_TUMVI_IMU_PARAMETERS.keys():
                template_data["imu_params"][key] = MSCKF_TUMVI_IMU_PARAMETERS[key]

    # camera_config_str = printCameraBlock(template_data["cameras"][0])
    # camera_config_str += printCameraBlock(template_data["cameras"][1])
    # template_data['cameras'] = camera_config_str # dump simply keeps all newlines.
    # print("camera str \n{}\n".format(camera_config_str))

    yaml.dump(template_data, out_config)
    out_config.close()

