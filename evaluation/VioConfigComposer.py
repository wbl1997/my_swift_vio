import os
import shutil

import dataset_parameters
import dataset_vio_config

"""If the bagname keyword is found in the bag's full path, then the 
corresponding calibration files will be used in creating the VIO config yaml"""
BAGKEY_CALIBRATION = {
    'euroc': ('calibration/euroc/imu0.yaml', 'calibration/euroc/cam0.yaml',
              'calibration/euroc/cam1.yaml'),
    'indoor_45': ('calibration/uzh_fpv/imu-snapdragon_imu.yaml',
                  'calibration/uzh_fpv/camchain-imucam-indoor_45_calib_snapdragon_imu.yaml'),
    'outdoor_45': ('calibration/uzh_fpv/imu-snapdragon_imu.yaml',
                   'calibration/uzh_fpv/camchain-imucam-outdoor_45_calib_snapdragon_imu.yaml'),
    'indoor_forward': ('calibration/uzh_fpv/imu-snapdragon_imu.yaml',
                       'calibration/uzh_fpv/camchain-imucam-indoor_forward_calib_snapdragon_imu.yaml'),
    'outdoor_forward': ('calibration/uzh_fpv/imu-snapdragon_imu.yaml',
                        'calibration/uzh_fpv/camchain-imucam-outdoor_forward_calib_snapdragon_imu.yaml'),
}


class VioConfigComposer(object):
    """compose a VIO config yaml for a data mission"""
    def __init__(self, vio_config_template, bag_fullname, vio_yaml_mission):
        """

        :param vio_config_template: vio config template yaml
        :param bag_fullname:
        :param vio_yaml_mission: customized yaml file for a data mission
        """
        self.vio_config_template = vio_config_template
        self.bag_fullname = bag_fullname
        self.vio_yaml_mission = vio_yaml_mission

    def get_calib_files(self):
        eval_script_dir = os.path.dirname(os.path.abspath(__file__))
        imu_calib_file = ""
        camera_calib_files = ""
        for bagkey in BAGKEY_CALIBRATION.keys():
            if bagkey in self.bag_fullname:
                imu_calib_file = os.path.join(eval_script_dir, BAGKEY_CALIBRATION[bagkey][0])
                camera_calib_files = [os.path.join(eval_script_dir, calib_file)
                                      for calib_file
                                      in BAGKEY_CALIBRATION[bagkey][1:]]
                break
        return imu_calib_file, camera_calib_files

    def create_config_for_mission(self, algo_code, use_nominal_value, monocular):
        dataset_type = dataset_parameters.dataset_code(self.bag_fullname)
        imu_calib_file, camera_calib_files = self.get_calib_files()
        calib_format = dataset_parameters.calibration_format(dataset_type)

        dataset_vio_config.create_config_yaml(
            self.vio_config_template, calib_format,
            self.vio_yaml_mission, camera_calib_files,
            imu_calib_file, algo_code, self.bag_fullname, use_nominal_value, monocular)
