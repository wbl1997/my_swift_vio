import copy
import os
import shutil
import warnings
from ruamel.yaml import YAML

import dir_utility_functions


def create_empty_traj_estimate_txt(filename):
    with open(filename, "w") as stream:
        stream.write("# timestamp tx ty tz qx qy qz qw\n")


def find_original_config_yaml(original_result_dir):
    filename_list = os.listdir(original_result_dir)
    yaml_count = 0
    eval_config_yaml = ""
    for filename in filename_list:
        if filename.endswith(".yaml"):
            eval_config_yaml = os.path.join(original_result_dir, filename)
            yaml_count += 1
    if yaml_count > 1:
        warnings.warn("Multiple candidate config yaml found under {}".format(original_result_dir))
    return eval_config_yaml


class GroupPgoResults(object):
    """Group pgo results for one method into 3 groups (original algo,
    online pgo, final pgo) which are as if produced by 3 vio methods,
     so that rpg evaluation tool can be used."""

    def __init__(self, original_results_dir, results_dir, eval_output_dir, algo_name, platform="laptop"):

        self.results_dir = results_dir
        self.eval_output_dir = eval_output_dir
        self.algo_name = algo_name
        self.original_eval_config_yaml = find_original_config_yaml(original_results_dir)
        dir_utility_functions.make_or_empty_dir(results_dir)

        self.platform = platform
        self.platform_dir = os.path.join(results_dir, platform)
        dir_utility_functions.mkdir_p(self.platform_dir)

        dir_utility_functions.make_or_empty_dir(eval_output_dir)
        self.eval_config = None
        self.eval_config_yaml = os.path.join(results_dir,  os.path.basename(self.original_eval_config_yaml))
        self.create_config_yaml(self.original_eval_config_yaml, algo_name, self.eval_config_yaml)

    def create_config_yaml(self, original_yaml, algo_name, new_config_yaml):
        yaml = YAML()
        yaml.version = (1, 2)
        yaml.default_flow_style = None
        yaml.indent(mapping=4, sequence=6, offset=4)

        config = None
        with open(original_yaml, 'r') as stream:
            config = yaml.load(stream)

        algo_config_dict = dict(config["Algorithms"])
        # keep only one algorithm
        for key in list(algo_config_dict):
            if key != algo_name:
                del algo_config_dict[key]

        algo_config = algo_config_dict[algo_name]
        original_label = algo_config["label"]

        pgo_online = algo_name + "_pgo"
        algo_config_pgo = copy.deepcopy(algo_config)
        algo_config_pgo["label"] = original_label + "-pgo"
        algo_config_dict[pgo_online] = algo_config_pgo

        pgo_offline = algo_name + "_gpgo"
        algo_config_gpgo = copy.deepcopy(algo_config)
        algo_config_gpgo["label"] = original_label + "-gpgo"
        algo_config_dict[pgo_offline] = algo_config_gpgo

        config["Algorithms"] = algo_config_dict

        self.eval_config = config
        with open(new_config_yaml, "w") as stream:
            yaml.dump(config, stream)

    def copy_subdirs_for_pgo(self):
        """
        copy original traj_estimate, online_pgo and final_pgo of each trial of
        each data session of one algorithm to dirs for 3 algorithms:
        original_algo, online pgo and final pgo.
        :return:
        """
        original_platform_dir = os.path.join(os.path.dirname(self.original_eval_config_yaml), self.platform)
        original_algo_dir = os.path.join(original_platform_dir, self.algo_name)

        for algo_name in self.eval_config["Algorithms"].keys():
            algo_dir = os.path.join(self.platform_dir, algo_name)
            dir_utility_functions.mkdir_p(algo_dir)
            for session in self.eval_config["Datasets"].keys():
                session_dir = os.path.join(algo_dir, self.platform + "_" + algo_name + "_" + session)
                dir_utility_functions.mkdir_p(session_dir)

                original_session_dir = os.path.join(original_algo_dir,
                                                    self.platform + "_" + self.algo_name + "_" + session)
                # find the trials of this data session.
                trial_dir_list_ex = [f.path for f in os.scandir(original_session_dir) if
                                     f.is_dir()]
                trial_dir_list = []
                suffix_numbers = []
                for trial_dir in trial_dir_list_ex:
                    val, _ = dir_utility_functions.get_number_suffix(os.path.basename(trial_dir))
                    if val is not None:
                        trial_dir_list.append(trial_dir)
                        suffix_numbers.append(val)

                shutil.copy2(os.path.join(original_session_dir, "eval_cfg.yaml"),
                             os.path.join(session_dir, "eval_cfg.yaml"))
                shutil.copy2(os.path.join(original_session_dir, "stamped_groundtruth.txt"),
                             os.path.join(session_dir, "stamped_groundtruth.txt"))

                if algo_name == self.algo_name:
                    original_est_list = [f.path for f in os.scandir(original_session_dir) if
                                         f.is_file() and "stamped_traj_estimate" in f.path]
                    assert len(original_est_list) == len(suffix_numbers)
                    for original_est_file in original_est_list:
                        shutil.copy2(original_est_file,
                                     os.path.join(session_dir, os.path.basename(original_est_file)))

                elif algo_name.endswith("gpgo"):
                    for index, trial_dir in enumerate(trial_dir_list):
                        pgo_src_file = os.path.join(trial_dir, "final_pgo.csv")
                        pgo_dst_file = os.path.join(session_dir,
                                                    "stamped_traj_estimate{}.txt".format(suffix_numbers[index]))
                        if os.path.isfile(pgo_src_file):
                            shutil.copy2(pgo_src_file, pgo_dst_file)
                        else:
                            create_empty_traj_estimate_txt(pgo_dst_file)

                elif algo_name.endswith("pgo"):
                    for index, trial_dir in enumerate(trial_dir_list):
                        pgo_src_file = os.path.join(trial_dir, "online_pgo.csv")
                        pgo_dst_file = os.path.join(session_dir,
                                                    "stamped_traj_estimate{}.txt".format(suffix_numbers[index]))
                        if os.path.isfile(pgo_src_file):
                            shutil.copy2(pgo_src_file, pgo_dst_file)
                        else:
                            create_empty_traj_estimate_txt(pgo_dst_file)

    def get_eval_config_yaml(self):
        return self.eval_config_yaml
