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


def apply_config_to_swiftvio_yaml(config_dict, vio_yaml, debug_output_dir):
    algo_code = config_dict["algo_code"]
    sed_cmd = r'sed -i "/algorithm/c\    algorithm: {}" {};'. \
        format(algo_code, vio_yaml)

    padding = ''
    sed_cmd += sed_line_with_parameter(config_dict, "numImuFrames", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "numKeyframes", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "monocular_input", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "displayImages", padding, vio_yaml)

    padding = " " * 4
    sed_cmd += sed_line_with_parameter(config_dict, "landmarkModelId", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "cameraObservationModelId", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "useMahalanobisGating", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "maxProjectionErrorTol", padding, vio_yaml)

    # The fields for extrinsic_opt_mode will become scrabbled in the generated vio config file,
    # so we look for the string combination of extrinsic_opt_mode + default value.
    # Another solution is provided at
    # https://unix.stackexchange.com/questions/403271/sed-replace-only-the-second-match-word
    if "extrinsic_opt_mode_main_camera" in config_dict.keys():
        sed_extrinsicOptMode = 'sed -i "0,/extrinsic_opt_mode/ ' \
                               's/extrinsic_opt_mode: FIXED/extrinsic_opt_mode: {}/" {};'.format(
            config_dict["extrinsic_opt_mode_main_camera"], vio_yaml)
        sed_cmd += sed_extrinsicOptMode
    if "extrinsic_opt_mode_other_camera" in config_dict.keys():
        sed_extrinsicOptMode = \
            'sed -i "0,/extrinsic_opt_mode/! {{0,/extrinsic_opt_mode/ ' \
            's/extrinsic_opt_mode: FIXED/extrinsic_opt_mode: {}/}}" {};'.format(
            config_dict["extrinsic_opt_mode_other_camera"], vio_yaml)
        sed_cmd += sed_extrinsicOptMode

    sed_cmd += sed_line_with_parameter(config_dict, "sigma_absolute_translation", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "sigma_absolute_orientation", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "sigma_c_relative_translation", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "sigma_c_relative_orientation", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "sigma_tr", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "sigma_td", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "sigma_focal_length", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "sigma_principal_point", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "sigma_distortion", padding, vio_yaml)

    sed_cmd += sed_line_with_parameter(config_dict, "model_type", padding, vio_yaml)
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

    sed_cmd += sed_line_with_parameter(config_dict, "stereoMatchWithEpipolarCheck", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "epipolarDistanceThreshold", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "featureTrackingMethod", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "timeLimit", padding, vio_yaml)

    out_stream = open(os.path.join(debug_output_dir, "sed_out.log"), 'w')
    err_stream = open(os.path.join(debug_output_dir, "sed_err.log"), 'w')
    utility_functions.subprocess_cmd(sed_cmd, out_stream, err_stream)
    out_stream.close()
    err_stream.close()


def apply_config_to_vinsmono_yaml(config_dict, vio_yaml, debug_output_dir):
    padding = ''
    sed_cmd = ""
    sed_cmd += sed_line_with_parameter(config_dict, "acc_n", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "gyr_n", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "acc_w", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "gyr_w", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "loop_closure", padding, vio_yaml)

    sed_cmd += sed_line_with_parameter(config_dict, "vins_output_dir", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "estimate_extrinsic", padding, vio_yaml)
    sed_cmd += sed_line_with_parameter(config_dict, "estimate_td", padding, vio_yaml)

    out_stream = open(os.path.join(debug_output_dir, "sed_out.log"), 'w')
    err_stream = open(os.path.join(debug_output_dir, "sed_err.log"), 'w')
    utility_functions.subprocess_cmd(sed_cmd, out_stream, err_stream)
    out_stream.close()
    err_stream.close()


def apply_config_to_yaml(config_dict, vio_yaml, debug_output_dir):
    if config_dict["algo_code"] == "VINSMono":
        return apply_config_to_vinsmono_yaml(config_dict, vio_yaml, debug_output_dir)
    elif config_dict["algo_code"] in ["HybridFilter", "MSCKF", "OKVIS"]:
        return apply_config_to_swiftvio_yaml(config_dict, vio_yaml, debug_output_dir)


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
    return "extra_gflags" in config_dict and \
           ("--publish_via_ros=true" in config_dict["extra_gflags"] or \
            "--publish_via_ros=1" in config_dict["extra_gflags"])
