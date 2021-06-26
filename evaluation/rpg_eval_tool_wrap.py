import os
import textwrap
import yaml

import utility_functions

def run_rpg_evaluation(rpg_eval_dir, eval_config_yaml, num_trials,
                       results_dir, eval_output_dir):
    """

    :param rpg_eval_dir:
    :param eval_config_yaml:
    :param num_trials: for each mission how many times an algorithm runs
    :param results_dir:
    :param eval_output_dir:
    :return:
    """
    analyze_traj_script = os.path.join(rpg_eval_dir, "scripts/analyze_trajectories.py")
    cmd = "python2 {} {} --output_dir={} --results_dir={} --platform laptop " \
          "--odometry_error_per_dataset --plot_trajectories --recalculate_errors " \
          "--rmse_table --rmse_boxplot --mul_trials={} --overall_odometry_error". \
        format(analyze_traj_script, eval_config_yaml, eval_output_dir, results_dir, num_trials)
    user_msg = 'cmd to rpg eval tool\n{}\n'.format(cmd)
    out_stream = open(os.path.join(eval_output_dir, "out.log"), 'w')
    err_stream = open(os.path.join(eval_output_dir, "err.log"), 'w')
    print(textwrap.fill(user_msg, 120))
    out_stream.write(user_msg)
    rc , err = utility_functions.subprocess_cmd(cmd, out_stream, err_stream)
    out_stream.close()
    err_stream.close()
    return rc, err


def find_and_load_rel_errors(eval_output_dir):
    """

    :param eval_output_dir:
    :return: {'algo_name': [rel_trans_err, rel_rot_err], ...}.
        Empty dict if not loaded successfully.
    """
    stat_file = ""
    file_list = os.listdir(eval_output_dir)
    # find the newest file meeting the requirements.
    for fname in file_list:
        if 'laptop_rel_err_' in fname and '.txt' in fname:
            stat_file = os.path.join(eval_output_dir, fname)

    cmp_rel_stats = {}
    if stat_file:
        cmp_rel_stats = utility_functions.load_rel_error_stats(stat_file)
    return cmp_rel_stats, stat_file


def check_eval_result(eval_result_dir, cmp_eval_output_dir):
    """
    briefly check eval results from rpg evaluation
    :param cmp_result_dir: previous evaluation result dir
    :return:
    """
    returncode = 0
    if os.path.isdir(cmp_eval_output_dir):
        cmp_rel_stats, cmp_stat_file = find_and_load_rel_errors(cmp_eval_output_dir)
        print('cmp rel stats from {}'.format(cmp_stat_file))
        print(cmp_rel_stats)

        rel_stats, stat_file = find_and_load_rel_errors(eval_result_dir)
        print('rel stats from {}'.format(stat_file))
        print(rel_stats)
        trans_rot_tolerance = [1e-3, 1e-3]
        for algo_name, cmp_trans_rot_err in cmp_rel_stats.items():
            if algo_name in rel_stats:
                trans_rot_err = rel_stats[algo_name]
                if trans_rot_err[0] > cmp_trans_rot_err[0] + trans_rot_tolerance[0] or \
                        trans_rot_err[1] > cmp_trans_rot_err[1] + trans_rot_tolerance[1]:
                    print("current rel trans rot error of algo "
                          "{} {} {} is greater than that for comparison {} {}".format(
                        algo_name, trans_rot_err[0], trans_rot_err[1], cmp_trans_rot_err[0], cmp_trans_rot_err[1]
                    ))
                    returncode = 1

    return returncode


def change_eval_cfg(eval_cfg_fn, align_type, align_n=-1):
    """
    apply configuration to a eval_cfg.yaml
    :param eval_cfg_fn: the cfg yaml on which the changes will be applied.
    :param align_type: one of sim3 se3 posyaw none
    :param align_n align first number of frames
    adapted from rpg_trajectory_evaluation/scripts/change_eval_cfg_recursive.py
    :return:
    """
    with open(eval_cfg_fn, 'r') as f:
        eval_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        eval_cfg['align_type'] = align_type
        eval_cfg['align_num_frames'] = align_n
    with open(eval_cfg_fn, 'w') as f:
        f.write(yaml.dump(eval_cfg, default_flow_style=False))
