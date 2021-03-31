#!/usr/bin/env python3

'''
This file computes the number of successful runs of one particular simulation
 based on the size of estimator output files.
'''
import math
import os
import sys

def is_estimator_output_file(filename, method_tag):
    basename = os.path.basename(filename)
    targetname = os.path.splitext(basename)[0]
    if method_tag in targetname:
        remain = targetname.replace(method_tag, '')
        remain = remain.strip('_')
        if remain.isdigit():
            return True
    return False

def main():
    if len(sys.argv) < 3:
        print('Usage: {} result_path method_tag'.format(sys.argv[0]))
    script, result_path, method_tag = sys.argv
    # find target files and get size
    file_list = []
    size_list = []
    for root, dirs, files in os.walk(result_path):
        for filename in files:
            if is_estimator_output_file(filename, method_tag):
                fullfilename = os.path.join(root, filename)
                size = os.path.getsize(fullfilename)
                file_list.append(fullfilename)
                size_list.append(size)
    
    expectedsize = max(size_list)
    print('Found #files {} and the maximum size is {}'.format(len(file_list), expectedsize))
    threshold = math.floor(expectedsize * 0.95 * 50.0/56)
    success = 0
    for index, item in enumerate(size_list):
        if item < threshold:
            print('Discount {} for its size {} is smaller than {}'.format(
                file_list[index], item, threshold))
        else:
            success += 1
    print('#Success runs {} out of {} runs for {}'.format(success, len(file_list), method_tag))

if __name__ == "__main__":
  main()
