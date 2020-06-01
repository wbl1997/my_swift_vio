import errno
import os
import shutil
from colorama import init, Fore
init(autoreset=True)


def find_bags_with_gt(uzh_fpv_dir, bagname_key, discount_key='.orig.bag'):
    """This function finds ros bags named like 'xxx_with_gt.bag',
    works with uzh-fpv dataset."""
    filename_list = os.listdir(uzh_fpv_dir)
    bags_with_gt_list = []
    for filename in filename_list:
        if 'with_gt.bag' in filename and bagname_key in filename \
                and discount_key not in filename:
            bags_with_gt_list.append(os.path.join(uzh_fpv_dir, filename))
    return bags_with_gt_list


def find_bags(root_dir, bagname_key, discount_key='.orig.bag'):
    """find bags recursively under root_dir"""
    bag_list = []
    for dir_name, subdir_list, file_list in os.walk(root_dir):
        for fname in file_list:
            if discount_key:
                if fname.endswith('.bag') and bagname_key in fname \
                        and discount_key not in fname and \
                        discount_key not in dir_name:
                    bag_list.append(os.path.join(dir_name, fname))
            else:
                if fname.endswith('.bag') and bagname_key in fname:
                    bag_list.append(os.path.join(dir_name, fname))
    return bag_list


def find_zips(root_dir, name_key, discount_key='.orig.zip'):
    """find zip files recursively under root_dir"""
    zip_list = []
    for dir_name, subdir_list, file_list in os.walk(root_dir):
        for fname in file_list:
            if fname.endswith('.zip') and name_key in fname \
                    and discount_key not in fname and discount_key not in dir_name:
                zip_list.append(os.path.join(dir_name, fname))
    return zip_list


def get_gt_file_for_bags(bag_list):
    gt_list = []
    for bag in bag_list:
        gt_list.append(os.path.splitext(bag)[0] + ".txt")
    return gt_list


def get_original_euroc_gt_files(euroc_bag_list):
    gt_list = []
    for bag in euroc_bag_list:
        gt_list.append(os.path.join(os.path.dirname(bag), 'data.csv'))
    return gt_list


def get_converted_euroc_gt_files(euroc_bag_list):
    gt_list = []
    for bag in euroc_bag_list:
        gt_list.append(os.path.join(os.path.dirname(bag), 'data.txt'))
    return gt_list

def empty_dir(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def mkdir_p(dirname):
    try:
        os.mkdir(dirname)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def make_or_empty_dir(dirname):
    if os.path.isdir(dirname):
        empty_dir(dirname)
    else:
        mkdir_p(dirname)

def find_rpg_est_gt_pairs(results_dir):
    """
    Find all pairs of (stamped_groundtruth.txt, stamped_traj_estimateX.txt) under results_dir
    :param results_dir:
    :return:
    """
    est_gt_list = []
    for dir_name, subdir_list, file_list in os.walk(results_dir):
        for fname in file_list:
            if 'stamped_traj_estimate' in fname and fname.endswith('.txt'):
                gt_file = os.path.join(dir_name, "stamped_groundtruth.txt")
                assert os.path.isfile(gt_file), "gt file {} not found!".format(gt_file)
                est_gt_list.append((os.path.join(dir_name, fname), gt_file))
    return est_gt_list

def get_rpg_est_file_suffix(est_file):
    """

    :param est_file:
    :return:
    """
    key = "stamped_traj_estimate"
    ext = ".txt"
    index = est_file.find(key)
    assert index != -1, "wrong rpg est_file names"
    index += len(key)
    if est_file[index] is '.':
        return ''
    else:
        return est_file[index]

def get_number_suffix(string_number):
    end_index = -1
    start_index = -1
    index = len(string_number)
    while index > 0:
        c = string_number[index - 1]
        if c.isdigit():
            if end_index == -1:
                end_index = index
            start_index = index - 1
        else:
            if end_index == -1:
                pass
            else:
                break
        index = index - 1  # decrement index
    if end_index == -1:
        return None, ""
    segment = string_number[start_index:end_index]
    return int(segment), segment
