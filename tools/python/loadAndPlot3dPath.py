#!/usr/bin/env python3


'''
 This module loads and plots 3d trajectory results.
'''

from __future__ import print_function

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
def parse_line_with_nanosecs(line, delimiter=','):
  line = line.rstrip(delimiter)
  rags = line.split(delimiter)
  rags[0] = rags[0].strip(' ')
  secs = int(rags[0][:-9])
  nanos = int(rags[0][-9:])
  time = secs + nanos * 10 ** (-9)
  result = [time]
  for val in rags[1:]:
    result.append(float(val))
  return result

def load_file_with_nanosecs(data_csv):
  data = list()
  with open(data_csv, 'r') as stream:
    for line in stream:
      if "#" in line or "%" in line:
        continue
      line = line.rstrip('\n').strip(' ')
      if line:
        data.append(parse_line_with_nanosecs(line))
  return np.asarray(data, dtype=np.float32)
  
def plot3d_trajectory(data, xyz_cols, label, cmp_data=None, cmp_label='', result_fig=''):
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.axis('equal')
  # ax.set_aspect('equal')

  x = data[:, xyz_cols[0]]
  y = data[:, xyz_cols[1]]
  z = data[:, xyz_cols[2]]
  ax.plot(x, y, z, label=label)
  if cmp_data is not None:
    x = cmp_data[:, xyz_cols[0]]
    y = cmp_data[:, xyz_cols[1]]
    z = cmp_data[:, xyz_cols[2]]
    ax.plot(x, y, z, label=cmp_label)
  ax.legend()
  ax.set_xlabel('x (m)')
  ax.set_ylabel('y (m)')
  ax.set_zlabel('z (m)')
  
  # https://python-decompiler.com/article/2012-12/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
  # Create cubic bounding box to simulate equal aspect ratio
  # max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
  # Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
  # Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
  # Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
  # # Comment or uncomment following both lines to test the fake bounding box:
  # for xb, yb, zb in zip(Xb, Yb, Zb):
  #     ax.plot([xb], [yb], [zb], 'w')

  plt.grid()
  # plt.show()

  if result_fig:
    os.remove(result_fig)
    plt.savefig(result_fig)

  
