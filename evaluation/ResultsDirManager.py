import os

import dir_utility_functions

# actual bag filename : data name used in creating result dirs, data label used in plots.
# Data labels are not supposed to include underscores which causes latex interpretation error,
#  but can include hyphens.
BAGNAME_DATANAME_LABEL = {
    'MH_01_easy': ('MH_01', 'MH01'),
    'MH_02_easy': ('MH_02', 'MH02'),
    'MH_03_medium': ('MH_03', 'MH03'),
    'MH_04_difficult': ('MH_04', 'MH04'),
    'MH_05_difficult': ('MH_05', 'MH05'),
    'V1_01_easy': ('V1_01', 'V101'),
    'V1_02_medium': ('V1_02', 'V102'),
    'V1_03_difficult': ('V1_03', 'V103'),
    'V2_01_easy': ('V2_01', 'V201'),
    'V2_02_medium': ('V2_02', 'V202'),
    'V2_03_difficult': ('V2_03', 'V203'),
    'indoor_45_2_snapdragon_with_gt': ('in_45_2', 'in-45-2'),
    'indoor_45_4_snapdragon_with_gt': ('in_45_4', 'in-45-4'),
    'indoor_45_9_snapdragon_with_gt': ('in_45_9', 'in-45-9'),
    'indoor_45_12_snapdragon_with_gt': ('in_45_12', 'in-45-12'),
    'indoor_45_13_snapdragon_with_gt': ('in_45_13', 'in-45-13'),
    'indoor_45_14_snapdragon_with_gt': ('in_45_14', 'in-45-14'),
    'outdoor_45_1_snapdragon_with_gt': ('out_45_1', 'out-45-1'),
    'indoor_forward_3_snapdragon_with_gt': ('in_fwd_3', 'in-fwd-3'),
    'indoor_forward_5_snapdragon_with_gt': ('in_fwd_5', 'in-fwd-5'),
    'indoor_forward_6_snapdragon_with_gt': ('in_fwd_6', 'in-fwd-6'),
    'indoor_forward_7_snapdragon_with_gt': ('in_fwd_7', 'in-fwd-7'),
    'indoor_forward_9_snapdragon_with_gt': ('in_fwd_9', 'in-fwd-9'),
    'indoor_forward_10_snapdragon_with_gt': ('in_fwd_10', 'in-fwd-10'),
    'outdoor_forward_1_snapdragon_with_gt': ('out_fwd_1', 'out-fwd-1'),
    'outdoor_forward_3_snapdragon_with_gt': ('out_fwd_3', 'out-fwd-3'),
    'outdoor_forward_5_snapdragon_with_gt': ('out_fwd_5', 'out-fwd-5')
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

    def create_results_dir(self):
        """create and clean necessary dirs and the config yaml for running evaluation"""
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
