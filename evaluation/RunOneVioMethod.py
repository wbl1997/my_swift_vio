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
import VioConfigComposer
import RoscoreManager


def timeout(bag_fullname):
    bag_duration = rosbag_utility_functions.get_rosbag_duration(bag_fullname)
    time_out = bag_duration * 4
    time_out = max(60 * 5, time_out)
    return time_out


def append_ros_arg_if_exist(config_dict, parameter):
    if parameter in config_dict:
        return " {}:={}".format(parameter, config_dict[parameter])
    else:
        return ""


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
            sensor calibration parameters. It can be an empty string if not needed.
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
        self.eval_cfg_template = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config/eval_cfg.yaml")
        self.extra_lib_path = extra_library_path

    def get_sync_exe(self):
        return os.path.join(self.catkin_ws, "devel/lib/swift_vio/swift_vio_node_synchronous")

    def get_async_launch_file(self):
        return os.path.join(self.catkin_ws, "src/swift_vio/launch/swift_vio_node_rosbag.launch")

    def create_vio_config_yaml(self, dataset_code, bag_fullname, output_dir_mission):
        """for each data mission, create a vio config yaml"""
        if not self.vio_config_template:
            return ""
        vio_yaml_mission = os.path.join(output_dir_mission, os.path.basename(self.vio_config_template))
        config_composer = VioConfigComposer.VioConfigComposer(
            self.vio_config_template, bag_fullname, vio_yaml_mission)
        # apply sensor calibration parameters
        config_composer.create_config_for_mission(self.algo_code_flags["algo_code"], dataset_code)
        # apply algorithm parameters
        AlgoConfig.apply_config_to_yaml(self.algo_code_flags, vio_yaml_mission, output_dir_mission)
        return vio_yaml_mission

    def create_lcd_config_yaml(self, output_dir_mission):
        """for each data mission, create a lcd config yaml"""
        if self.lcd_config_template:
            lcd_yaml_mission = os.path.join(output_dir_mission, os.path.basename(self.lcd_config_template))
            shutil.copy2(self.lcd_config_template, lcd_yaml_mission)
            AlgoConfig.apply_config_to_lcd_yaml(
                self.algo_code_flags, lcd_yaml_mission, output_dir_mission)
            return lcd_yaml_mission
        else:
            return None

    def create_sync_command(self, custom_vio_config, custom_lcd_config,
                            vio_trial_output_dir, bag_fullname, dataset_code):
        """create synchronous commands for estimators in swift vio."""
        if dataset_code in dataset_parameters.ROS_TOPICS.keys():
            arg_topics = r'--camera_topics="{},{}" --imu_topic={}'.format(
                dataset_parameters.ROS_TOPICS[dataset_code][0],
                dataset_parameters.ROS_TOPICS[dataset_code][1],
                dataset_parameters.ROS_TOPICS[dataset_code][2])
        else:
            arg_topics = r'--camera_topics="/cam0/image_raw" --imu_topic=/imu0'

        export_lib_cmd = ""
        if self.extra_lib_path:
            export_lib_cmd = "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{};".format(self.extra_lib_path)
        verbose_leak_sanitizer = True
        if verbose_leak_sanitizer:
            export_lib_cmd += "export ASAN_OPTIONS=fast_unwind_on_malloc=0;"

        cmd = "{} {} --lcd_params_yaml={} --output_dir={}" \
              " --max_inc_tol=30.0 --dump_output_option=3" \
              " --bagname={} --vocabulary_path={} {} {}".format(
            self.get_sync_exe(), custom_vio_config, custom_lcd_config,
            vio_trial_output_dir,
            bag_fullname, self.voc_file, arg_topics,
            self.algo_code_flags["extra_gflags"])
        return export_lib_cmd + cmd

    def create_async_command(self, custom_vio_config,
                             custom_lcd_config,
                             vio_trial_output_dir, bag_fullname, dataset_code):
        """create asynchronous commands for estimators in swift vio."""
        setup_bash_file = os.path.join(self.catkin_ws, "devel/setup.bash")
        src_cmd = "cd {}\nsource {}\n".format(self.catkin_ws, setup_bash_file)

        launch_file = "swift_vio_node_rosbag.launch"
        arg_topics = "image_topic:={} image_topic1:={} imu_topic:={}".format(
            dataset_parameters.ROS_TOPICS[dataset_code][0],
            dataset_parameters.ROS_TOPICS[dataset_code][1],
            dataset_parameters.ROS_TOPICS[dataset_code][2])

        export_lib_cmd = "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{}\n".\
            format(self.extra_lib_path)
        launch_cmd = "roslaunch swift_vio {} config_filename:={} lcd_config_filename:={} " \
                     "output_dir:={} {} bag_file:={} start_into_bag:=3 play_rate:=1.0".format(
            launch_file, custom_vio_config, custom_lcd_config, vio_trial_output_dir,
            arg_topics, bag_fullname)
        cmd = src_cmd + export_lib_cmd + launch_cmd

        # We put all commands in a bash script because source
        # command is unavailable when running in python subprocess.
        src_wrap = os.path.join(vio_trial_output_dir, "source_wrap.sh")
        with open(src_wrap, 'w') as stream:
            stream.write('#!/bin/bash\n')
            stream.write('{}\n'.format(cmd))
        return "chmod +x {wrap};{wrap}".format(wrap=src_wrap)

    def create_vio_command(self, algo_name, dataset_code, bag_fullname,
                           custom_vio_config, custom_lcd_config, output_dir_trial):
        """create vio command for a variety of estimator depending on the algo_code"""
        if self.algo_code_flags["algo_code"] == "VINSMono":
            setup_bash_file = os.path.join(self.catkin_ws, "devel/setup.bash")
            src_cmd = "cd {}\nsource {}\n".format(self.catkin_ws, setup_bash_file)
            exe_cmd = "roslaunch vins_estimator {} config_path:={} output_dir:={} " \
                      "bag_file:={} bag_start:=0 bag_duration:=10000 " \
                      "rviz_file:={}/src/VINS-Mono/config/vins_rviz_config.rviz".format(
                          self.algo_code_flags["launch_file"], custom_vio_config,
                          output_dir_trial, bag_fullname, self.catkin_ws)
            cmd = src_cmd + exe_cmd

            # We put all commands in a bash script because source
            # command is unavailable when running in python subprocess.
            src_wrap = os.path.join(output_dir_trial, "source_wrap.sh")
            with open(src_wrap, 'w') as stream:
                stream.write('#!/bin/bash\n')
                stream.write('{}\n'.format(cmd))
            return "chmod +x {wrap};{wrap}".format(wrap=src_wrap)
        elif self.algo_code_flags["algo_code"] == "OpenVINS":
            setup_bash_file = os.path.join(self.catkin_ws, "devel/setup.bash")
            src_cmd = "cd {}\nsource {}\n".format(self.catkin_ws, setup_bash_file)
            if "bag_start" in self.algo_code_flags:
                bag_start = self.algo_code_flags["bag_start"]
            else:
                bag_start = dataset_parameters.decide_bag_start(self.algo_code_flags["algo_code"], bag_fullname)
            if "init_imu_thresh" in self.algo_code_flags:
                init_imu_thresh = self.algo_code_flags["init_imu_thresh"]
            else:
                init_imu_thresh = dataset_parameters.decide_initial_imu_threshold(
                    self.algo_code_flags["algo_code"], bag_fullname)
            result_file = os.path.join(output_dir_trial, 'stamped_traj_estimate.txt')
            exe_cmd = "roslaunch ov_msckf {} max_cameras:={} use_stereo:={} bag:={} " \
                      "bag_start:={} init_imu_thresh:={} dosave:=true path_est:={}".format(
                          self.algo_code_flags["launch_file"], self.algo_code_flags["max_cameras"],
                          self.algo_code_flags["use_stereo"], bag_fullname,
                          bag_start, init_imu_thresh, result_file)
            exe_cmd += append_ros_arg_if_exist(self.algo_code_flags, "gyroscope_noise_density")
            exe_cmd += append_ros_arg_if_exist(self.algo_code_flags, "gyroscope_random_walk")
            exe_cmd += append_ros_arg_if_exist(self.algo_code_flags, "accelerometer_noise_density")
            exe_cmd += append_ros_arg_if_exist(self.algo_code_flags, "accelerometer_random_walk")
            exe_cmd += append_ros_arg_if_exist(self.algo_code_flags, "calib_cam_extrinsics")
            exe_cmd += append_ros_arg_if_exist(self.algo_code_flags, "calib_cam_intrinsics")
            exe_cmd += append_ros_arg_if_exist(self.algo_code_flags, "calib_cam_timeoffset")
            exe_cmd += append_ros_arg_if_exist(self.algo_code_flags, "max_slam")
            exe_cmd += append_ros_arg_if_exist(self.algo_code_flags, "max_slam_in_update")

            cmd = src_cmd + exe_cmd
            # We put all commands in a bash script because source
            # command is unavailable when running in python subprocess.
            src_wrap = os.path.join(output_dir_trial, "source_wrap.sh")
            with open(src_wrap, 'w') as stream:
                stream.write('#!/bin/bash\n')
                stream.write('{}\n'.format(cmd))
            return "chmod +x {wrap};{wrap}".format(wrap=src_wrap)
        elif self.algo_code_flags["algo_code"] == "MSCKFMono":
            setup_bash_file = os.path.join(self.catkin_ws, "devel/setup.bash")
            src_cmd = "cd {}\nsource {}\n".format(self.catkin_ws, setup_bash_file)
            result_file = os.path.join(output_dir_trial, 'stamped_traj_estimate.txt')

            exe_cmd = "roslaunch msckf_mono {}  bagname:={} dosave:=true path_est:={} show_rviz:=true".format(
                self.algo_code_flags["launch_file"], bag_fullname, result_file)
            exe_cmd += dataset_parameters.msckf_mono_arg_of_dataset(bag_fullname)

            cmd = src_cmd + exe_cmd
            src_wrap = os.path.join(output_dir_trial, "source_wrap.sh")
            with open(src_wrap, 'w') as stream:
                stream.write('#!/bin/bash\n')
                stream.write('{}\n'.format(cmd))
            return "chmod +x {wrap};{wrap}".format(wrap=src_wrap)
        elif self.algo_code_flags["algo_code"] == "ROVIO":
            setup_bash_file = os.path.join(self.catkin_ws, "devel/setup.bash")
            src_cmd = "cd {}\nsource {}\n".format(self.catkin_ws, setup_bash_file)
            result_basename = os.path.join(output_dir_trial, 'stamped_traj_estimate')

            exe_cmd = "roslaunch rovio {}  bagname:={} numcameras:={} filename_out:={}".format(
                self.algo_code_flags["launch_file"], bag_fullname, self.algo_code_flags["numcameras"], result_basename)

            cmd = src_cmd + exe_cmd
            src_wrap = os.path.join(output_dir_trial, "source_wrap.sh")
            with open(src_wrap, 'w') as stream:
                stream.write('#!/bin/bash\n')
                stream.write('{}\n'.format(cmd))
            return "chmod +x {wrap};{wrap}".format(wrap=src_wrap)
        elif 'async' in algo_name:
            return self.create_async_command(custom_vio_config, custom_lcd_config,
                                             output_dir_trial, bag_fullname, dataset_code)
        else:
            return self.create_sync_command(custom_vio_config, custom_lcd_config,
                                            output_dir_trial, bag_fullname, dataset_code)

    @staticmethod
    def run_vio_command(cmd, bag_fullname, log_vio, out_stream, err_stream):
        user_msg = 'Running vio method with cmd\n{}\n'.format(cmd)
        print(textwrap.fill(user_msg, 120))
        out_stream.write(user_msg)
        time_out = timeout(bag_fullname)
        if log_vio:
            rc, msg = utility_functions.subprocess_cmd(cmd, out_stream, err_stream, time_out)
        else:
            rc, msg = utility_functions.subprocess_cmd(cmd, None, None, time_out)
        return rc, msg

    def create_convert_command(self, output_dir_mission, output_dir_trial, index_str, pose_conversion_script):
        if self.algo_code_flags["algo_code"] == "VINSMono":
            vio_estimate_csv = os.path.join(output_dir_trial, 'vins_result_no_loop.csv')
            converted_vio_file = os.path.join(
                output_dir_mission, "stamped_traj_estimate{}.txt".format(index_str))
            cmd = "chmod +x {};python3 {} {} {}".format(pose_conversion_script, pose_conversion_script,
                                                        vio_estimate_csv, converted_vio_file)
        elif self.algo_code_flags["algo_code"] in ["OpenVINS", "MSCKFMono"]:
            result_file = os.path.join(output_dir_trial, 'stamped_traj_estimate.txt')
            converted_vio_file = os.path.join(
                output_dir_mission, "stamped_traj_estimate{}.txt".format(index_str))
            cmd = "chmod +x {};python3 {} {} {}".format(pose_conversion_script, pose_conversion_script,
                                                        result_file, converted_vio_file)
        elif self.algo_code_flags["algo_code"] == "ROVIO":
            result_file = os.path.join(output_dir_trial, 'stamped_traj_estimate.bag')
            converted_vio_file = os.path.join(
                output_dir_mission, "stamped_traj_estimate{}.txt".format(index_str))
            cmd = "chmod +x {};python {} {} /rovio/odometry --msg_type Odometry --output {}".format(
                pose_conversion_script, pose_conversion_script,
                result_file, converted_vio_file)
        else:
            vio_estimate_csv = os.path.join(output_dir_trial, 'swift_vio.csv')
            converted_vio_file = os.path.join(
                output_dir_mission, "stamped_traj_estimate{}.txt".format(index_str))
            cmd = "python3 {} {} --outfile={} --output_delimiter=' '". \
                format(pose_conversion_script, vio_estimate_csv, converted_vio_file)
        return cmd

    @staticmethod
    def run_convert_command(cmd, out_stream, err_stream):
        user_msg = 'Converting pose file with cmd\n{}\n'.format(cmd)
        print(textwrap.fill(user_msg, 120))
        out_stream.write(user_msg)
        utility_functions.subprocess_cmd(cmd, out_stream, err_stream)
        out_stream.close()
        err_stream.close()

    def run_method(self, algo_name, pose_conversion_script,
                   gt_align_type='posyaw', log_vio=True, dataset_code="euroc"):
        '''
        run a method
        :return:
        '''
        # if algo_name has async, then the async script will start roscore, and no need to start it here.
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

            custom_vio_config = self.create_vio_config_yaml(dataset_code, bag_fullname, output_dir_mission)
            custom_lcd_config = self.create_lcd_config_yaml(output_dir_mission)

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

                cmd = self.create_vio_command(algo_name, dataset_code, bag_fullname,
                                              custom_vio_config, custom_lcd_config, output_dir_trial)

                rc, msg = RunOneVioMethod.run_vio_command(cmd, bag_fullname, log_vio, out_stream, err_stream)
                if rc != 0:
                    err_msg = "Return error code {} and msg {} in running vio method with cmd:\n{}". \
                        format(rc, msg, cmd)
                    warnings.warn(textwrap.fill(err_msg, 120))
                    return_code = rc

                cmd = self.create_convert_command(output_dir_mission, output_dir_trial,
                                                  index_str, pose_conversion_script)

                RunOneVioMethod.run_convert_command(cmd, out_stream, err_stream)

        roscore_manager.stop_roscore()
        return return_code
