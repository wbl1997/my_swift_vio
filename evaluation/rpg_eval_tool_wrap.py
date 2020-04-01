import os

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
    print('cmd to rpg eval tool\n{}'.format(cmd))
    return utility_functions.subprocess_cmd(cmd)

def check_eval_result(cmp_result_dir):
    """
    briefly check eval results from rpg evaluation
    :param cmp_result_dir: previous evaluation result dir
    :return:
    """
    if cmp_result_dir:
        file_list = os.listdir(cmp_result_dir)
        for fname in file_list:
            if 'laptop_rel_err_' in fname and '.txt' in fname:
                cmp_stat_file = os.path.join(cmp_result_dir, fname)
                break

        cmp_rel_stats = utility_functions.load_rel_error_stats(cmp_stat_file)
        print('cmp rel stats from {}'.format(cmp_stat_file))
        print(cmp_rel_stats)
