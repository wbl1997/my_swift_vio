
from __future__ import print_function


import dir_utility_functions


def test_get_rpg_est_file_suffix():
    est_file = "/okvis_ref/laptop/OKVIS/laptop_OKVIS_MH_02/stamped_traj_estimate.txt"
    suffix = dir_utility_functions.get_rpg_est_file_suffix(est_file)
    assert suffix == ''
    est_file = "/okvis_ref/laptop/OKVIS/laptop_OKVIS_MH_02/stamped_traj_estimate2.txt"
    suffix = dir_utility_functions.get_rpg_est_file_suffix(est_file)
    assert suffix == "2"
