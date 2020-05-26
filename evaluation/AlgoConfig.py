import utility_functions
import os

def create_algo_config(val_list):
    d = {"algo_code": val_list[0],
         "extra_gflags": val_list[1],
         "numKeyframes": val_list[2],
         "numImuFrames": val_list[3],
         "monocular_input": 1
         }
    if len(val_list) >= 5:
        d["monocular_input"] = val_list[4]
    return d


def default_algo_config():
    d = {"algo_code": "MSCKF",
         "extra_gflags": "",
         "numKeyframes": 10,
         "numImuFrames": 5,
         "monocular_input": 0,
         "landmarkModelId": 1,
         "anchorAtObservationTime": 0,
         "extrinsic_opt_mode_main_camera" : "p_CB",
         "extrinsic_opt_mode_other_camera": "p_C0C_q_C0C",
         "sigma_absolute_translation": "0.0",
         "sigma_absolute_orientation": "0.0",
         }
    return d


def sed_line_with_parameter(config_dict, param_name, padding, config_yaml):
    """

    :param config_dict:
    :param param_name:
    :param padding:
    :param config_yaml:
    :return:
    """
    if param_name in config_dict.keys():
        return r'sed -i "/{}/c\{}{}: {}" {};'.format(
            param_name, padding, param_name,
            config_dict[param_name], config_yaml)
    else:
        return ""


def apply_config_to_yaml(config_dict, vio_yaml, debug_output_dir):
    algo_code = config_dict["algo_code"]
    sed_algo = r'sed -i "/algorithm/c\    algorithm: {}" {};'. \
        format(algo_code, vio_yaml)
    sed_imuframes = r'sed -i "/numImuFrames:/c\numImuFrames: {}" {};'. \
        format(config_dict["numImuFrames"], vio_yaml)
    sed_keyframes = r'sed -i "/numKeyframes:/c\numKeyframes: {}" {};'. \
        format(config_dict["numKeyframes"], vio_yaml)
    sed_nframe = r'sed -i "/monocular_input:/c\monocular_input: {}" {};'. \
        format(config_dict["monocular_input"], vio_yaml)
    sed_display = r'sed -i "/displayImages:/c\displayImages: false" {};'. \
        format(vio_yaml)
    sed_cmd = sed_algo + sed_keyframes + sed_imuframes + sed_nframe + sed_display

    padding = " " * 4
    sed_cmd += sed_line_with_parameter(config_dict, "landmarkModelId", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "anchorAtObservationTime", padding, vio_yaml)

    # the fields for extrinsic_opt_mode will become scrabbled in the generated vio config file,
    # so we look for the string combination of extrinsic_opt_mode + default value.
    if "extrinsic_opt_mode_main_camera" in config_dict.keys():
        sed_extrinsicOptMode = 'sed -i "0,/extrinsic_opt_mode/ ' \
                               's/extrinsic_opt_mode: P_CB/extrinsic_opt_mode: {}/" {};'.format(
            config_dict["extrinsic_opt_mode_main_camera"], vio_yaml)
        sed_cmd += sed_extrinsicOptMode
    if "extrinsic_opt_mode_other_camera" in config_dict.keys():
        sed_extrinsicOptMode = \
            'sed -i "0,/extrinsic_opt_mode/! {{0,/extrinsic_opt_mode/ ' \
            's/extrinsic_opt_mode: P_C0C_Q_C0C/extrinsic_opt_mode: {}/}}" {};'.format(
            config_dict["extrinsic_opt_mode_other_camera"], vio_yaml)
        sed_cmd += sed_extrinsicOptMode

    sed_cmd += sed_line_with_parameter(config_dict, "sigma_absolute_translation", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "sigma_absolute_orientation", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "sigma_tr", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "sigma_td", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "sigma_focal_length", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "sigma_principal_point", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "sigma_distortion", padding, vio_yaml)

    sed_cmd += sed_line_with_parameter(config_dict, "sigma_TGElement", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "sigma_TSElement", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "sigma_TAElement", padding, vio_yaml)

    sed_cmd += sed_line_with_parameter(config_dict, "sigma_g_c", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "sigma_a_c", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "sigma_gw_c", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "sigma_aw_c", padding, vio_yaml)

    sed_cmd += sed_line_with_parameter(config_dict, "g_max", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "a_max", padding, vio_yaml)

    sed_gravity = ""
    param_name = "g: 9"
    if param_name in config_dict.keys():
        sed_gravity = r'sed -i "/{}/c\{}{}: {}" {};'.format(
            param_name, padding, "g",
            config_dict[param_name], vio_yaml)
    sed_cmd += sed_gravity

    sed_cmd += sed_line_with_parameter(config_dict, "maxOdometryConstraintForAKeyframe", padding, vio_yaml)

    out_stream = open(os.path.join(debug_output_dir, "sed_out.log"), 'w')
    err_stream = open(os.path.join(debug_output_dir, "sed_err.log"), 'w')
    utility_functions.subprocess_cmd(sed_cmd, out_stream, err_stream)
    out_stream.close()
    err_stream.close()


def apply_config_to_lcd_yaml(config_dict, lcd_yaml, debug_output_dir):
    sed_cmd = ""
    if "loop_closure_method" in config_dict.keys():
        sed_algo = r'sed -i "/loop_closure_method/c\loop_closure_method: {}" {};'. \
            format(config_dict["loop_closure_method"], lcd_yaml)
        sed_cmd = sed_algo

    out_stream = open(os.path.join(debug_output_dir, "sed_out.log"), 'w')
    err_stream = open(os.path.join(debug_output_dir, "sed_err.log"), 'w')
    utility_functions.subprocess_cmd(sed_cmd, out_stream, err_stream)
    out_stream.close()
    err_stream.close()


def doWePublishViaRos(config_dict):
    return "--publish_via_ros=true" in config_dict["extra_gflags"] or \
           "--publish_via_ros=1" in config_dict["extra_gflags"]
