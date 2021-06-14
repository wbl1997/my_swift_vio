import os
import shutil
import textwrap
import warnings

import dataset_parameters

import dir_utility_functions
import rosbag_utility_functions
import rpg_eval_tool_wrap
import utility_functions

import AlgoConfig
import OkvisConfigComposer
import RoscoreManager


class RunOneVioMethod(object):
    """Run one vio method on a number of data missions"""
    def __init__(self, catkin_ws, vio_config_template,
                 algo_code_flags,
                 num_trials, bag_list, gt_list,
                 vio_output_dir_list, extra_library_path='',
                 lcd_config_template='',
                 voc_file = ''):
        """
        :param catkin_ws: workspace containing executables
        :param vio_config_template: the template config yaml for running this algorithm.
            New config yaml will be created for each data mission considering the
            sensor calibration parameters.
        :param algo_code_flags: {algo_code, algo_cmd_flags, numkeyframes, numImuFrames, }
        :param num_trials:
        :param bag_list:
        :param gt_list: If not None, gt files will be copied to proper locations for evaluation.
        :param vio_output_dir_list: list of dirs to put the estimation results for every data mission.
        :param extra_library_path: It is necessary when some libraries cannot be found.
        """

        self.catkin_ws = catkin_ws
        self.vio_config_template = vio_config_template
        self.lcd_config_template = lcd_config_template
        self.voc_file = voc_file
        self.algo_code_flags = algo_code_flags
        self.num_trials = num_trials
        self.bag_list = bag_list
        self.gt_list = gt_list
        self.output_dir_list = vio_output_dir_list
        self.algo_dir = os.path.dirname(vio_output_dir_list[0].rstrip('/'))
        self.custom_vio_config_list = []
        self.custom_lcd_config_list = []
        self.eval_cfg_template = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config/eval_cfg.yaml")
        self.extra_lib_path = extra_library_path

    def get_sync_exe(self):
        return os.path.join(self.catkin_ws, "devel/lib/swift_vio/swift_vio_node_synchronous")

    def get_async_launch_file(self):
        return os.path.join(self.catkin_ws, "src/swift_vio/launch/swift_vio_node_rosbag.launch")

    def create_vio_config_yaml(self):
        """for each data mission, create a vio config yaml"""
        vio_yaml_list = []
        for bag_index, bag_fullname in enumerate(self.bag_list):
            output_dir_mission = self.output_dir_list[bag_index]
            vio_yaml_mission = os.path.join(output_dir_mission, os.path.basename(self.vio_config_template))

            config_composer = OkvisConfigComposer.OkvisConfigComposer(
                self.vio_config_template, bag_fullname, vio_yaml_mission)

            # apply sensor calibration parameters
            if "use_nominal_calib_value" in self.algo_code_flags:
                config_composer.create_config_for_mission(
                    self.algo_code_flags["algo_code"], 
                    self.algo_code_flags["use_nominal_calib_value"], 
                    self.algo_code_flags["monocular_input"])
            else:
                config_composer.create_config_for_mission(
                    self.algo_code_flags["algo_code"], False,
                    self.algo_code_flags["monocular_input"])

            # apply algorithm parameters
            AlgoConfig.apply_config_to_yaml(
                self.algo_code_flags, vio_yaml_mission, output_dir_mission)

            vio_yaml_list.append(vio_yaml_mission)
        self.custom_vio_config_list = vio_yaml_list

    def create_lcd_config_yaml(self):
        """for each data mission, create a lcd config yaml"""
        lcd_yaml_list = []
        for bag_index, bag_fullname in enumerate(self.bag_list):
            output_dir_mission = self.output_dir_list[bag_index]
            lcd_yaml_mission = os.path.join(output_dir_mission, os.path.basename(self.lcd_config_template))
            shutil.copy2(self.lcd_config_template, lcd_yaml_mission)
            AlgoConfig.apply_config_to_lcd_yaml(
                self.algo_code_flags, lcd_yaml_mission, output_dir_mission)
            lcd_yaml_list.append(lcd_yaml_mission)
        self.custom_lcd_config_list = lcd_yaml_list

    def create_sync_command(self, custom_vio_config, custom_lcd_config,
                            vio_trial_output_dir, bag_fullname):
        data_type = dataset_parameters.dataset_code(bag_fullname)
        if data_type in  dataset_parameters.ROS_TOPICS.keys():
            arg_topics = r'--camera_topics="{},{}" --imu_topic={}'.format(
                dataset_parameters.ROS_TOPICS[data_type][0],
                dataset_parameters.ROS_TOPICS[data_type][1],
                dataset_parameters.ROS_TOPICS[data_type][2])
        else:
            arg_topics = r'--camera_topics="/cam0/image_raw" --imu_topic=/imu0'

        export_lib_cmd = ""
        if self.extra_lib_path:
            export_lib_cmd = "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{};".format(self.extra_lib_path)
        verbose_leak_sanitizer = True
        if verbose_leak_sanitizer:
            export_lib_cmd += "export ASAN_OPTIONS=fast_unwind_on_malloc=0;"

        cmd = "{} {} --lcd_params_yaml={} --output_dir={} --skip_first_seconds=0" \
              " --max_inc_tol=30.0 --dump_output_option=3" \
              " --bagname={} --vocabulary_path={} {} {}".format(
            self.get_sync_exe(), custom_vio_config, custom_lcd_config,
            vio_trial_output_dir,
            bag_fullname, self.voc_file, arg_topics,
            self.algo_code_flags["extra_gflags"])
        return export_lib_cmd + cmd

    def create_async_command(self, custom_vio_config,
                             custom_lcd_config,
                             vio_trial_output_dir, bag_fullname):
        launch_file = "swift_vio_node_rosbag.launch"
        setup_bash_file = os.path.join(self.catkin_ws, "devel/setup.bash")

        data_type = dataset_parameters.dataset_code(bag_fullname)
        arg_topics = "image_topic:={} image_topic1:={} imu_topic:={}".format(
            dataset_parameters.ROS_TOPICS[data_type][0],
            dataset_parameters.ROS_TOPICS[data_type][1],
            dataset_parameters.ROS_TOPICS[data_type][2])

        src_cmd = "cd {}\nsource {}\n".format(self.catkin_ws, setup_bash_file)
        export_lib_cmd = "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{}\n".\
            format(self.extra_lib_path)
        launch_cmd = "roslaunch swift_vio {} config_filename:={} lcd_config_filename:={} " \
                     "output_dir:={} {} bag_file:={} start_into_bag:=3 play_rate:=1.0".format(
            launch_file, custom_vio_config, custom_lcd_config, vio_trial_output_dir,
            arg_topics, bag_fullname)
        return src_cmd + export_lib_cmd + launch_cmd

    def timeout(self, bag_fullname):
        bag_duration = rosbag_utility_functions.get_rosbag_duration(bag_fullname)
        time_out = bag_duration * 4
        time_out = max(60 * 5, time_out)
        return time_out

    def run_method(self, algo_name, pose_conversion_script,
                   gt_align_type='posyaw', log_vio=True):
        '''
        run a method
        :return:
        '''
        self.create_vio_config_yaml()
        self.create_lcd_config_yaml()

        mock = 'async' in algo_name or (not AlgoConfig.doWePublishViaRos(self.algo_code_flags))
        roscore_manager = RoscoreManager.RosCoreManager(mock)
        roscore_manager.start_roscore()
        return_code = 0
        for bag_index, bag_fullname in enumerate(self.bag_list):
            output_dir_mission = self.output_dir_list[bag_index]
            gt_file = os.path.join(output_dir_mission, 'stamped_groundtruth.txt')
            if self.gt_list:
                shutil.copy2(self.gt_list[bag_index], gt_file)
            eval_config_file = os.path.join(output_dir_mission, 'eval_cfg.yaml')
            shutil.copy2(self.eval_cfg_template, eval_config_file)
            rpg_eval_tool_wrap.change_eval_cfg(eval_config_file, gt_align_type, -1)

            custom_vio_config = self.custom_vio_config_list[bag_index]
            custom_lcd_config = self.custom_lcd_config_list[bag_index]
            for trial_index in range(self.num_trials):
                if self.num_trials == 1:
                    index_str = ''
                else:
                    index_str = '{}'.format(trial_index)

                output_dir_trial = os.path.join(
                    output_dir_mission, '{}{}'.format(self.algo_code_flags["algo_code"], index_str))
                dir_utility_functions.mkdir_p(output_dir_trial)

                out_stream = open(os.path.join(output_dir_trial, "out.log"), 'w')
                err_stream = open(os.path.join(output_dir_trial, "err.log"), 'w')

                if 'async' in algo_name:
                    cmd = self.create_async_command(custom_vio_config, custom_lcd_config,
                                                    output_dir_trial, bag_fullname)
                    # We put all commands in a bash script because source
                    # command is unavailable when running in python subprocess.
                    src_wrap = os.path.join(output_dir_trial, "source_wrap.sh")
                    with open(src_wrap, 'w') as stream:
                        stream.write('#!/bin/bash\n')
                        stream.write('{}\n'.format(cmd))
                    cmd = "chmod +x {wrap};{wrap}".format(wrap=src_wrap)
                else:
                    cmd = self.create_sync_command(custom_vio_config, custom_lcd_config,
                                                   output_dir_trial, bag_fullname)

                user_msg = 'Running vio method with cmd\n{}\n'.format(cmd)
                print(textwrap.fill(user_msg, 120))
                out_stream.write(user_msg)
                time_out = self.timeout(bag_fullname)
                if log_vio:
                    rc, msg = utility_functions.subprocess_cmd(cmd, out_stream, err_stream, time_out)
                else:
                    rc, msg = utility_functions.subprocess_cmd(cmd, None, None, time_out)
                if rc != 0:
                    err_msg = "Return error code {} and msg {} in running vio method with cmd:\n{}".\
                        format(rc, msg, cmd)
                    warnings.warn(textwrap.fill(err_msg, 120))
                    return_code = rc
                vio_estimate_csv = os.path.join(output_dir_trial, 'swift_vio.csv')
                converted_vio_file = os.path.join(
                    output_dir_mission, "stamped_traj_estimate{}.txt".format(index_str))
                cmd = "python3 {} {} --outfile={} --output_delimiter=' '". \
                    format(pose_conversion_script, vio_estimate_csv, converted_vio_file)
                user_msg = 'Converting pose file with cmd\n{}\n'.format(cmd)
                print(textwrap.fill(user_msg, 120))
                out_stream.write(user_msg)
                utility_functions.subprocess_cmd(cmd, out_stream, err_stream)
                out_stream.close()
                err_stream.close()

        roscore_manager.stop_roscore()
        return return_code
