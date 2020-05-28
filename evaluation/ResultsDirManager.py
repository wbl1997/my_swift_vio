import os

import dir_utility_functions

# actual bag filename : data name used in creating result dirs, data label used in plots.
# Data labels are not supposed to include underscores which causes latex interpretation error,
#  but can include hyphens.
BAGNAME_DATANAME_LABEL = {
    'MH_01_easy': ('MH_01', 'M1'),
    'MH_02_easy': ('MH_02', 'M2'),
    'MH_03_medium': ('MH_03', 'M3'),
    'MH_04_difficult': ('MH_04', 'M4'),
    'MH_05_difficult': ('MH_05', 'M5'),
    'V1_01_easy': ('V1_01', 'V11'),
    'V1_02_medium': ('V1_02', 'V12'),
    'V1_03_difficult': ('V1_03', 'V13'),
    'V2_01_easy': ('V2_01', 'V21'),
    'V2_02_medium': ('V2_02', 'V22'),
    'V2_03_difficult': ('V2_03', 'V23'),

    'indoor_45_2_snapdragon_with_gt': ('in_45_2', 'i2'),
    'indoor_45_4_snapdragon_with_gt': ('in_45_4', 'i4'),
    'indoor_45_9_snapdragon_with_gt': ('in_45_9', 'i9'),
    'indoor_45_12_snapdragon_with_gt': ('in_45_12', 'i12'),
    'indoor_45_13_snapdragon_with_gt': ('in_45_13', 'i13'),
    'indoor_45_14_snapdragon_with_gt': ('in_45_14', 'i14'),
    'outdoor_45_1_snapdragon_with_gt': ('out_45_1', 'o1'),
    'indoor_forward_3_snapdragon_with_gt': ('in_fwd_3', 'if3'),
    'indoor_forward_5_snapdragon_with_gt': ('in_fwd_5', 'if5'),
    'indoor_forward_6_snapdragon_with_gt': ('in_fwd_6', 'if6'),
    'indoor_forward_7_snapdragon_with_gt': ('in_fwd_7', 'if7'),
    'indoor_forward_9_snapdragon_with_gt': ('in_fwd_9', 'if9'),
    'indoor_forward_10_snapdragon_with_gt': ('in_fwd_10', 'if10'),
    'outdoor_forward_1_snapdragon_with_gt': ('out_fwd_1', 'of1'),
    'outdoor_forward_3_snapdragon_with_gt': ('out_fwd_3', 'of3'),
    'outdoor_forward_5_snapdragon_with_gt': ('out_fwd_5', 'of5'),
    'indoor_forward_11_snapdragon': ('in_fwd_11', 'if11'),
    'indoor_forward_12_snapdragon': ('in_fwd_12', 'if12'),
    'indoor_45_3_snapdragon': ('in_45_3', 'i3'),
    'indoor_45_16_snapdragon': ('in_45_16', 'i16'),
    'outdoor_forward_9_snapdragon': ('out_fwd_9', 'of9'),
    'outdoor_forward_10_snapdragon': ('out_fwd_10', 'of10'),

    'dataset-corridor1_512_16': ('corridor1', 'co1'),
    'dataset-corridor2_512_16': ('corridor2', 'co2'),
    'dataset-corridor3_512_16': ('corridor3', 'co3'),
    'dataset-corridor4_512_16': ('corridor4', 'co4'),
    'dataset-corridor5_512_16': ('corridor5', 'co5'),
    'dataset-magistrale1_512_16': ('magistrale1', 'ma1'),
    'dataset-magistrale2_512_16': ('magistrale2', 'ma2'),
    'dataset-magistrale3_512_16': ('magistrale3', 'ma3'),
    'dataset-magistrale4_512_16': ('magistrale4', 'ma4'),
    'dataset-magistrale5_512_16': ('magistrale5', 'ma5'),
    'dataset-magistrale6_512_16': ('magistrale6', 'ma6'),
    'dataset-outdoors1_512_16': ('outdoors1', 'out1'),
    'dataset-outdoors2_512_16': ('outdoors2', 'out2'),
    'dataset-outdoors3_512_16': ('outdoors3', 'out3'),
    'dataset-outdoors4_512_16': ('outdoors4', 'out4'),
    'dataset-outdoors5_512_16': ('outdoors5', 'out5'),
    'dataset-outdoors6_512_16': ('outdoors6', 'out6'),
    'dataset-outdoors7_512_16': ('outdoors7', 'out7'),
    'dataset-outdoors8_512_16': ('outdoors8', 'out8'),
    'dataset-room1_512_16': ('room1', 'rm1'), # only room sessions of tum vi dataset have throughout ground truth.
    'dataset-room2_512_16': ('room2', 'rm2'),
    'dataset-room3_512_16': ('room3', 'rm3'),
    'dataset-room4_512_16': ('room4', 'rm4'),
    'dataset-room5_512_16': ('room5', 'rm5'),
    'dataset-room6_512_16': ('room6', 'rm6'),
    'dataset-slides1_512_16': ('slides1', 'sl1'),
    'dataset-slides2_512_16': ('slides2', 'sl2'),
    'dataset-slides3_512_16': ('slides3', 'sl3'),
}


class ResultsDirManager(object):
    def __init__(self, results_dir, bag_list, algo_name_list):
        """

        :param results_dir:
        :param bag_list:
        :param algo_name_list: name of algorithms used for creating dirs
        """
        self.results_dir = results_dir
        self.bag_list = bag_list
        self.algo_name_list = algo_name_list
        self.platform = "laptop"

    def create_eval_output_dir(self, eval_output_dir):
        if os.path.isdir(eval_output_dir):
            dir_utility_functions.empty_dir(eval_output_dir)
        else:
            dir_utility_functions.mkdir_p(eval_output_dir)

    def create_results_dir(self):
        """create and clean necessary dirs and the config yaml for running evaluation"""
        if os.path.isdir(self.results_dir):
            dir_utility_functions.empty_dir(self.results_dir)
        else:
            dir_utility_functions.mkdir_p(self.results_dir)

        platform_dir = os.path.join(self.results_dir, self.platform)
        dir_utility_functions.mkdir_p(platform_dir)

        for algo_name in self.algo_name_list:
            algo_dir = os.path.join(platform_dir, algo_name)
            dir_utility_functions.mkdir_p(algo_dir)
            for bag_fullname in self.bag_list:
                data_result_dir = self.get_result_dir(algo_name, bag_fullname)
                dir_utility_functions.mkdir_p(data_result_dir)

    def create_eval_config_yaml(self):
        algo_label_list = [name.replace('_', '-') for name in self.algo_name_list]
        eval_config_yaml = self.get_eval_config_yaml()
        tab_indent = '  '
        with open(eval_config_yaml, 'w') as stream:
            stream.write('Datasets:\n')
            for bag_fullname in self.bag_list:
                bagname = os.path.basename(os.path.splitext(bag_fullname)[0])
                dir_data_name, plot_label = BAGNAME_DATANAME_LABEL[bagname]
                stream.write('{}{}:\n'.format(tab_indent, dir_data_name))
                stream.write('{}{}label: {}\n'.format(tab_indent, tab_indent, plot_label))
            stream.write('Algorithms:\n')
            for index, algo_name in enumerate(self.algo_name_list):
                stream.write('{}{}:\n'.format(tab_indent, algo_name))
                stream.write('{}{}fn: traj_est\n'.format(tab_indent, tab_indent))
                stream.write('{}{}label: {}\n'.format(tab_indent, tab_indent, algo_label_list[index]))
            stream.write('RelDistances: [40, 60, 80, 100, 120]\n')

    def get_result_dir(self, algo_name, bag_fullpath):
        bagname = os.path.basename(os.path.splitext(bag_fullpath)[0])
        dir_data_name, _ = BAGNAME_DATANAME_LABEL[bagname]
        return os.path.join(self.results_dir, self.platform, algo_name,
                            self.platform + "_" + algo_name + "_" + dir_data_name)

    def get_all_result_dirs(self, algo_name):
        return [self.get_result_dir(algo_name, bag_fullpath) for bag_fullpath in self.bag_list]

    def get_algo_dir(self, algo_name):
        return os.path.join(self.results_dir, self.platform, algo_name)

    def get_eval_config_yaml(self):
        return os.path.join(self.results_dir, "euroc_uzh_fpv_vio.yaml")
