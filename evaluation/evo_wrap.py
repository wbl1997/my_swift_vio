import os

import dir_utility_functions
import utility_functions

"""Run evo to compute ATE.
The evo as of April 2020 does not support translation + yaw alignment.
But its ATE trans RMSE is very close to that obtained by rpg evaluation tool.
"""

def run_evo(results_dir, eval_output_dir):
    """
    If evo package is installed in a virtual env, you should call this function
    in the virtual env.

    :param results_dir:
    :param eval_output_dir: output dir of the evo evaluation result
    :return:
    """

    est_gt_list = dir_utility_functions.find_rpg_est_gt_pairs(results_dir)
    print("Find #est_gt_pair {}".format(len(est_gt_list)))
    # for est_gt in est_gt_list:
    #     print("est: {} gt: {}".format(est_gt[0], est_gt[1]))

    eval_zip_list = []
    for est_gt in est_gt_list:
        est_file = est_gt[0]
        gt_file = est_gt[1]
        mission_output_dir = os.path.dirname(est_file)
        eval_mission_output_dir = mission_output_dir.replace(results_dir, eval_output_dir)
        if not os.path.exists(eval_mission_output_dir):
            os.makedirs(eval_mission_output_dir)

        suffix = dir_utility_functions.get_rpg_est_file_suffix(est_file)

        eval_zip = os.path.join(eval_mission_output_dir, "evo{}.zip".format(suffix))
        out_stream = open(os.path.join(eval_mission_output_dir, "out.log"), 'a')
        err_stream = open(os.path.join(eval_mission_output_dir, "err.log"), 'a')

        cmd = "evo_ape tum {} {} -va --save_results {} --no_warnings".format(gt_file, est_file, eval_zip)
        print("Running evo cmd {}\n".format(cmd))
        out_stream.write("{}\n".format(cmd))
        utility_functions.subprocess_cmd(cmd, out_stream, err_stream)
        out_stream.close()
        err_stream.close()
        eval_zip_list.append(eval_zip)

    # put results together.
    output_table =  os.path.join(eval_output_dir, "ate_trans.csv")
    cmd = "evo_res {} \\\n --save_table {} --use_filenames".format(' \\\n '.join(eval_zip_list), output_table)

    user_msg = 'cmd to rpg eval tool\n{}\n'.format(cmd)
    out_stream = open(os.path.join(eval_output_dir, "out.log"), 'w')
    err_stream = open(os.path.join(eval_output_dir, "err.log"), 'w')
    print(user_msg)
    out_stream.write(user_msg)
    rc , err = utility_functions.subprocess_cmd(cmd, out_stream, err_stream)
    out_stream.close()
    err_stream.close()
    return rc, err

