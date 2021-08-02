
from __future__ import print_function


import dir_utility_functions


def test_get_rpg_est_file_suffix():
    est_file = "/laptop/HybridFilter/laptop_HybridFilter_MH_02/stamped_traj_estimate.txt"
    suffix = dir_utility_functions.get_rpg_est_file_suffix(est_file)
    assert suffix == ''
    est_file = "/laptop/HybridFilter/laptop_HybridFilter_MH_02/stamped_traj_estimate2.txt"
    suffix = dir_utility_functions.get_rpg_est_file_suffix(est_file)
    assert suffix == "2"

def test_get_number_suffix():
    string_number = "HybridFilter3"
    val, string = dir_utility_functions.get_number_suffix(string_number)
    assert val == 3
    assert string == "3"

    string_number = "HybridFilter34"
    val, string = dir_utility_functions.get_number_suffix(string_number)
    assert val == 34
    assert string == "34"
    string_number = "HybridFilter345"
    val, string = dir_utility_functions.get_number_suffix(string_number)
    assert val == 345
    assert string == "345"

    string_number = "laptop_MSCKF_n_aidp_MH_01/stamped_traj_estimate0.txt"
    val, string = dir_utility_functions.get_number_suffix(string_number)
    assert val == 0
    assert string == "0"
    string_number = "laptop_MSCKF_n_aidp_MH_01/stamped_traj_estimate23.txt"
    val, string = dir_utility_functions.get_number_suffix(string_number)
    assert val == 23
    assert string == "23"
    string_number = "laptop_MSCKF_n_aidp_MH_01/stamped_traj_estimate01.txt"
    val, string = dir_utility_functions.get_number_suffix(string_number)
    assert val == 1
    assert string == "01"
