import os
import shutil

import kalibr_okvis_config

"""If the bagname keyword is found in the bag's full path, then the 
corresponding calibration files will be used in creating okvis config yaml"""
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


def dataset_code(bagname):
    if 'snapdragon' in bagname or 'davis' in bagname:
        return 1
    if 'euroc' in bagname:
        return 0


class OkvisConfigComposer(object):
    """compose a okvis config yaml for a data mission"""
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
        for bagkey in BAGKEY_CALIBRATION.keys():
            if bagkey in self.bag_fullname:
                imu_calib_file = os.path.join(eval_script_dir, BAGKEY_CALIBRATION[bagkey][0])
                camera_calib_files = [os.path.join(eval_script_dir, calib_file)
                                      for calib_file
                                      in BAGKEY_CALIBRATION[bagkey][1:]]
                break
        return imu_calib_file, camera_calib_files

    def create_config_for_mission(self):
        dataset_type = dataset_code(self.bag_fullname)
        imu_calib_file, camera_calib_files = self.get_calib_files()
        if dataset_type == 0:
            calib_format = "euroc"
        elif dataset_type == 1:
            calib_format = "kalibr"
        else:
            raise Exception("Unknown dataset type {}".format(dataset_type))
        kalibr_okvis_config.create_okvis_config_yaml(
            self.vio_config_template, calib_format, camera_calib_files,
            imu_calib_file, self.vio_yaml_mission)
