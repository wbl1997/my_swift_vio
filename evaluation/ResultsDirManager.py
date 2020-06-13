import os

import dataset_parameters
import dir_utility_functions


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
            reply = input("Delete the content of " + eval_output_dir + "? [y(Y)/[n]] ")
            if reply == 'y' or reply == 'Y':
                 dir_utility_functions.empty_dir(eval_output_dir)
        else:
            dir_utility_functions.mkdir_p(eval_output_dir)

    def create_results_dir(self):
        """create and clean necessary dirs and the config yaml for running evaluation"""
        if os.path.isdir(self.results_dir):
            reply = input("Delete the content of " + self.results_dir + "? [y(Y)/[n]] ")
            if reply == 'y' or reply == 'Y':
                dir_utility_functions.empty_dir(self.results_dir)
        else:
            dir_utility_functions.makedirs_p(self.results_dir)

        platform_dir = os.path.join(self.results_dir, self.platform)
        dir_utility_functions.mkdir_p(platform_dir)

        for algo_name in self.algo_name_list:
            algo_dir = os.path.join(platform_dir, algo_name)
            dir_utility_functions.mkdir_p(algo_dir)
            for bag_fullname in self.bag_list:
                data_result_dir = self.get_result_dir(algo_name, bag_fullname)
                dir_utility_functions.mkdir_p(data_result_dir)

    def save_config(self, config_dict, directory):
        config_logfile = os.path.join(directory, "README.md")
        with open(config_logfile, 'w') as stream:
            for name, options in config_dict.items():
                stream.write("{}: {}\n".format(name, options))

    def create_eval_config_yaml(self):
        algo_label_list = [name.replace('_', '-') for name in self.algo_name_list]
        eval_config_yaml = self.get_eval_config_yaml()
        tab_indent = '  '
        with open(eval_config_yaml, 'w') as stream:
            stream.write('Datasets:\n')
            for bag_fullname in self.bag_list:
                bagname = os.path.basename(os.path.splitext(bag_fullname)[0])
                dir_data_name, plot_label = dataset_parameters.BAGNAME_DATANAME_LABEL[bagname]
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
        dir_data_name, _ = dataset_parameters.BAGNAME_DATANAME_LABEL[bagname]
        return os.path.join(self.results_dir, self.platform, algo_name,
                            self.platform + "_" + algo_name + "_" + dir_data_name)

    def get_all_result_dirs(self, algo_name):
        return [self.get_result_dir(algo_name, bag_fullpath) for bag_fullpath in self.bag_list]

    def get_algo_dir(self, algo_name):
        return os.path.join(self.results_dir, self.platform, algo_name)

    def get_eval_config_yaml(self):
        return os.path.join(self.results_dir, "euroc_uzh_fpv_vio.yaml")
