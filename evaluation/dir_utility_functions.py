import errno
import os
import shutil

def find_bags_with_gt(uzh_fpv_dir, bagname_key, discount_key='.orig.bag'):
    filename_list = os.listdir(uzh_fpv_dir)
    bags_with_gt_list = []
    for filename in filename_list:
        if 'with_gt.bag' in filename and bagname_key in filename \
                and discount_key not in filename:
            bags_with_gt_list.append(os.path.join(uzh_fpv_dir, filename))
    return bags_with_gt_list

def find_bags(root_dir, bagname_key, discount_key='.orig.bag'):
    bag_list = []
    for dir_name, subdir_list, file_list in os.walk(root_dir):
        for fname in file_list:
            if '.bag' in fname and bagname_key in fname \
                    and discount_key not in fname and discount_key not in dir_name:
                bag_list.append(os.path.join(dir_name, fname))
    return bag_list

def get_uzh_fpv_gt_files(uzh_fpv_bag_list):
    gt_list = []
    for bag in uzh_fpv_bag_list:
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

def emptyfolder(folder):
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
