import os
import shutil

import dir_utility_functions
import utility_functions
import OkvisConfigComposer
import RoscoreManager


ROS_TOPICS = [["/cam0/image_raw", "/cam1/image_raw", "/imu0"],
              ["/snappy_cam/stereo_l", "/snappy_cam/stereo_r", "/snappy_imu"]]




class RunOneVioMethod(object):
    """Run one vio method on a number of data missions"""
    def __init__(self, catkin_ws, vio_config_template,
                 algo_code_flags,
                 num_trials, bag_list, gt_list,
                 vio_output_dir_list, extra_library_path=''):
        """

        :param catkin_ws:
        :param vio_config_template:
        :param algo_code_flags: [algo_code, algo_cmd_flags, numkeyframes, numImuFrames, ]
        :param num_trials:
        :param bag_list:
        :param gt_list:
        :param vio_output_dir_list:
        :param extra_library_path: It is necessary when some libraries cannot be found.
        """

        self.catkin_ws = catkin_ws
        self.vio_config_template = vio_config_template
        self.algo_code_flags = algo_code_flags
        self.num_keyframes = algo_code_flags[2]
        self.num_imuframes = algo_code_flags[3]
        self.num_trials = num_trials
        self.bag_list = bag_list
        self.gt_list = gt_list
        self.output_dir_list = vio_output_dir_list
        self.algo_dir = os.path.dirname(vio_output_dir_list[0].rstrip('/'))
        self.custom_vio_config_list = []
        self.eval_cfg_template = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config/eval_cfg.yaml")
        self.extra_lib_path = extra_library_path

    def get_sync_exe(self):
        return os.path.join(self.catkin_ws, "devel/lib/msckf/okvis_node_synchronous")

    def get_async_lannch_file(self):
        return os.path.join(self.catkin_ws, "src/msckf/launch/okvis_node_rosbag.launch")

    def create_vio_config_yaml(self):
        """for each data mission, create a vio config yaml"""
        vio_yaml_list = []
        for bag_index, bag_fullname in enumerate(self.bag_list):
            output_dir_mission = self.output_dir_list[bag_index]
            vio_yaml_mission = os.path.join(output_dir_mission, os.path.basename(self.vio_config_template))

            config_composer = OkvisConfigComposer.OkvisConfigComposer(
                self.vio_config_template, bag_fullname, vio_yaml_mission)

            # apply sensor calibration parameters
            config_composer.create_config_for_mission()

            # apply algorithm parameters
            algo_code = self.algo_code_flags[0]
            sed_algo = r'sed -i "/algorithm/c\    algorithm: {}" {};'.\
                format(algo_code, vio_yaml_mission)
            sed_kf = r'sed -i "/numImuFrames:/c\numImuFrames: {}" {};'.\
                format(self.num_imuframes, vio_yaml_mission)
            sed_imuframes = r'sed -i "/numKeyframes:/c\numKeyframes: {}" {};'.\
                format(self.num_keyframes, vio_yaml_mission)
            sed_display = r'sed -i "/displayImages:/c\displayImages: false" {};'.\
                format(vio_yaml_mission)

            sed_cmd = sed_algo + sed_kf + sed_imuframes + sed_display
            utility_functions.subprocess_cmd(sed_cmd)
            vio_yaml_list.append(vio_yaml_mission)
        self.custom_vio_config_list = vio_yaml_list

    def create_sync_command(self, custom_vio_config, vio_trial_output_dir, bag_fullname):
        data_type = OkvisConfigComposer.dataset_code(bag_fullname)
        arg_topics = r'--camera_topics="{},{}" --imu_topic={}'.format(
            ROS_TOPICS[data_type][0], ROS_TOPICS[data_type][1], ROS_TOPICS[data_type][2])

        export_lib_cmd = "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{};". \
            format(self.extra_lib_path)
        cmd = "{} {} --output_dir={} --skip_first_seconds=0" \
              " --max_inc_tol=10.0 --dump_output_option=3" \
              " --bagname={} {} {}".format(
            self.get_sync_exe(), custom_vio_config, vio_trial_output_dir,
            bag_fullname, arg_topics,
            self.algo_code_flags[1])
        return export_lib_cmd + cmd

    def create_async_command(self, custom_vio_config, vio_trial_output_dir, bag_fullname):
        launch_file = "okvis_node_rosbag.launch"
        setup_bash_file = os.path.join(self.catkin_ws, "devel/setup.bash")
        arg_val_str = utility_functions.get_arg_value_from_gflags(
            self.algo_code_flags[1], "feature_tracking_method")
        if arg_val_str is None:
            arg_feature_method = ''
        else:
            arg_feature_method = "feature_tracking_method:={}".format(arg_val_str)
        data_type = OkvisConfigComposer.dataset_code(bag_fullname)
        arg_topics = "image_topic:={} image_topic1:={} imu_topic:={}".format(
            ROS_TOPICS[data_type][0], ROS_TOPICS[data_type][1], ROS_TOPICS[data_type][2])

        src_cmd = "cd {}\nsource {}\n".format(self.catkin_ws, setup_bash_file)
        export_lib_cmd = "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{}\n".\
            format(self.extra_lib_path)
        launch_cmd = "roslaunch msckf {} config_filename:={} output_dir:={}" \
                     " {} {} bag_file:={} start_into_bag:=0 play_rate:=1.0".format(
            launch_file, custom_vio_config, vio_trial_output_dir,
            arg_topics, arg_feature_method,
            bag_fullname)
        return src_cmd + export_lib_cmd + launch_cmd

    def run_method(self, algo_name, pose_conversion_script):
        '''
        run methods in the msckf project, mainly okvis, msckf.
        :param bag_list:
        :param gt_list: gt files will be copied to proper locations for evaluation.
        :param config_yaml: the yaml config file for running all algorithms, may be modified.
        :param results_dir: dir to put the estimation results
        :return:
        '''
        self.create_vio_config_yaml()

        mock = 'async' in algo_name
        roscore_manager = RoscoreManager.RosCoreManager(mock)
        roscore_manager.start_roscore()
        return_code = 0
        for bag_index, bag_fullname in enumerate(self.bag_list):
            output_dir_mission = self.output_dir_list[bag_index]
            gt_file = os.path.join(output_dir_mission, 'stamped_groundtruth.txt')
            shutil.copy2(self.gt_list[bag_index], gt_file)
            eval_config_file = os.path.join(output_dir_mission, 'eval_cfg.yaml')
            shutil.copy2(self.eval_cfg_template, eval_config_file)
            custom_vio_config = self.custom_vio_config_list[bag_index]
            for trial_index in range(self.num_trials):
                if self.num_trials == 1:
                    index_str = ''
                else:
                    index_str = '{}'.format(trial_index)

                output_dir_trial = os.path.join(
                    output_dir_mission, '{}{}'.format(self.algo_code_flags[0], index_str))
                dir_utility_functions.mkdir_p(output_dir_trial)

                if 'async' in algo_name:
                    cmd = self.create_async_command(custom_vio_config, output_dir_trial, bag_fullname)
                    # We put all commands in a bash script because source
                    # command is unavailable when running in python subprocess.
                    src_wrap = os.path.join(self.algo_dir, "source_wrap.sh")
                    with open(src_wrap, 'w') as stream:
                        stream.write('#!/bin/bash\n')
                        stream.write('{}\n'.format(cmd))
                    cmd = "chmod +x {wrap};{wrap}".format(wrap=src_wrap)
                else:
                    cmd = self.create_sync_command(custom_vio_config, output_dir_trial, bag_fullname)

                print('Running vio method with cmd\n{}\n'.format(cmd))
                rc, msg = utility_functions.subprocess_cmd(cmd)
                if rc != 0:
                    print('Error code {} with cmd:\n{}\nand error msg:{}\n'.format(rc, cmd, msg))
                    return_code = rc
                vio_estimate_csv = os.path.join(output_dir_trial, 'msckf_estimates.csv')
                converted_vio_file = os.path.join(
                    output_dir_mission, "stamped_traj_estimate{}.txt".format(index_str))
                cmd = "python3 {} {} --outfile={} --output_delimiter=' '". \
                    format(pose_conversion_script, vio_estimate_csv, converted_vio_file)
                utility_functions.subprocess_cmd(cmd)
        roscore_manager.stop_roscore()
        return return_code
