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


def apply_config_to_yaml(config_dict, vio_yaml, output_dir):
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
    if "landmarkModelId" in config_dict.keys():
        sed_lmkModel = r'sed -i "/landmarkModelId/c\    landmarkModelId: {}" {};'. \
            format(config_dict["landmarkModelId"], vio_yaml)
        sed_cmd += sed_lmkModel
    if "anchorAtObservationTime" in config_dict.keys():
        sed_anchorTime = r'sed -i "/anchorAtObservationTime/c\    anchorAtObservationTime: {}" {};'. \
            format(config_dict["anchorAtObservationTime"], vio_yaml)
        sed_cmd += sed_anchorTime
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

    if "sigma_absolute_translation" in config_dict.keys():
        sed_absTranslation = r'sed -i "/sigma_absolute_translation/c\    sigma_absolute_translation: {}" {};'.format(
            config_dict["sigma_absolute_translation"], vio_yaml)
        sed_cmd += sed_absTranslation

    if "sigma_absolute_orientation" in config_dict.keys():
        sed_absOrientation = r'sed -i "/sigma_absolute_orientation/c\    sigma_absolute_orientation: {}" {};'.format(
            config_dict["sigma_absolute_orientation"], vio_yaml)
        sed_cmd += sed_absOrientation

    out_stream = open(os.path.join(output_dir, "sed_out.log"), 'w')
    err_stream = open(os.path.join(output_dir, "sed_err.log"), 'w')
    utility_functions.subprocess_cmd(sed_cmd, out_stream, err_stream)
    out_stream.close()
    err_stream.close()
