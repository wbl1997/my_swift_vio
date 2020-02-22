#!/usr/bin/env python3


'''
 This module loads data from vinsmono or okvis output file,
 and plots 3d trajectories results.
'''

from __future__ import print_function

import os
import sys
import loadAndPlot3dPath

def draw_vinsmono_result(vins_result_dir):
  vinsmono_csv = os.path.join(vins_result_dir, 'vins_result_no_loop.csv')
  vinsmono_loop_csv = os.path.join(vins_result_dir, 'vins_result_loop.csv')
  
  vinsmono_data = loadAndPlot3dPath.load_file_with_nanosecs(vinsmono_csv)
  vinsmono_loop_data = loadAndPlot3dPath.load_file_with_nanosecs(vinsmono_loop_csv)
  
  result_fig  = os.path.join(vins_result_dir, 'p_GB.svg')
  loadAndPlot3dPath.plot3d_trajectory(vinsmono_data, [1,2,3], "vins mono", 
      cmp_data=vinsmono_loop_data, cmp_label="vins mono loop", result_fig=result_fig)
  print('p_GB saved at {}'.format(result_fig))  

def draw_okvis_result(okvis_csv):
  okvis_data = loadAndPlot3dPath.load_file_with_nanosecs(okvis_csv)
  output_dir = os.path.dirname(okvis_csv)
  result_fig  = os.path.join(output_dir, 'okvis_p_GB.svg')
  loadAndPlot3dPath.plot3d_trajectory(okvis_data, [2,3,4], "okvis", 
      cmp_data=None, result_fig=result_fig)
  print('p_GB saved at {}'.format(result_fig))  

def main():
  '''main function'''
  if len(sys.argv) < 2:
    print("Use 1: {} <vinsmono output dir>".format(sys.argv[0]))
    print("Use 2: {} <okvis output csv> okvis".format(sys.argv[0]))
    sys.exit(1)
  result_path = sys.argv[1]
  option = "vinsmono"
  if len(sys.argv) > 2:
    option = sys.argv[2]
  
  if option == 'okvis':
    draw_okvis_result(result_path)
  else:
    draw_vinsmono_result(result_path)
  
if __name__ == "__main__":
  main()
  
