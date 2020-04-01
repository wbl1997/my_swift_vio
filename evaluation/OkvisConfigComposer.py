import os
import shutil

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

    def create_config_for_euroc(self, imu_calib_file, camera_calib_files):
        """create vio_yaml_mission with vio_config_template and sensor
        calibration files for euroc data
        see https://github.com/MIT-SPARK/Kimera-VIO/blob/master/kalibr/config2kimeravio.py
        for how to load kalibr output
        """
        # TODO(jhuai)
        # imu_calib = load_kalibr_imu_calib()
        # cameras_calib = load_kalibr_cameras_calib()
        # okvis_config = OkvisConfigManager()
        # okvis_config.load_okvis_config(self.vio_config_template)
        # okvis_config.apply_imu_calib_euroc(imu_calib)
        # okvis_config.apply_cameras_calib_euroc(cameras_calib)
        # okvis_config.save(self.vio_yaml_mission)
        pass

    def create_config_for_uzh_fpv(self, imu_calib_file, camera_calib_file):
        """create vio_yaml_mission with vio_config_template and sensor
        calibration files for uzh-fpv data

        use the scripts inside Kimero-VIO to load kalibr output
        https://github.com/MIT-SPARK/Kimera-VIO/blob/master/kalibr/config2kimeravio.py

        python config2kimeravio.py -config_option "stereo-equi" -input_cam \
         /home/jhuai/Documents/docker/msckf_ws/src/msckf/evaluation/calibration/uzh_fpv/camchain-imucam-..indoor_45_calib_snapdragon_imu.yaml \
         -input_imu /home/jhuai/Documents/docker/msckf_ws/src/msckf/evaluation/calibration/uzh_fpv/imu-snapdragon_imu.yaml \
         -output /home/jhuai/Desktop/temp/indoor_45 -responsible "J. Huai" -date '03.31.2020' \
         -camera "Snapdragon Flight" -IMU "Snapdragon Flight"

        To deal with warnings of config2kimeravio.py "LoadWarning: calling yaml.load() without Loader=... is deprecated"
        use Loader like "yaml.load(f, Loader=yaml.FullLoader)".

        To test the function use test_okvis_config_composer.py in Nosetests or pytest of python2 or 3.
        A okvis template is at /msckf/config/config_fpga_p2_euroc_dissertation.yaml
        """
        # TODO(jhuai):
        # imu_calib = load_kalibr_imu_calib()
        # cameras_calib = load_kalibr_cameras_calib()
        # okvis_config = OkvisConfigManager()
        # okvis_config.load_okvis_config(self.vio_config_template)
        # okvis_config.apply_imu_calib_kalibr(imu_calib)
        # okvis_config.apply_cameras_calib_kalibr(cameras_calib)
        # okvis_config.save(self.vio_yaml_mission)
        pass

    def create_config_for_mission(self):
        # TODO(jhuai): adapt the vio yaml output to different dataset according to bag_fullname
        shutil.copy2(self.vio_config_template, self.vio_yaml_mission)

        dataset_type = dataset_code(self.bag_fullname)
        imu_calib_file, camera_calib_files = self.get_calib_files()
        if dataset_type == 0:
            self.create_config_for_euroc(imu_calib_file, camera_calib_files)
        elif dataset_type == 1:
            self.create_config_for_uzh_fpv(imu_calib_file, camera_calib_files)
        else:
            raise Exception("Unknown dataset type {}".format(dataset_type))

