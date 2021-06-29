#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Given output of KSF for a group of data sequences, compute and visualize
 statistics of estimated sensor parameters, e.g., bg ba Tg Ts Ta, and
 extrinsic, intrinsic, and temporal parameters for each camera.

 Currently this module works only with TUM VI to compare estimated parameters
  by KSF vs provided values from TUM VI.
 """
import argparse
import copy
import math
import os
import sys

import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

import dataset_vio_config

FORMAT = '.pdf'
PALETTE = plt.rcParams['axes.prop_cycle'].by_key()['color']

# ['b', 'g', 'r', 'c', 'k', 'y', 'm', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
#             'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'Purples', 'Oranges',]

# How to relate the IMU intrinsic parameters used by TUMVI (eq(1,2) in the
# TUMVI dataset paper) to KSF IMU parameters?
# We achieve this by express the inertial measurements in terms of the inertial
#  quantities in the camera frame. This is because definitions of the IMU
# sensor frame in TUMVI (denoted by S1) and KSF (denoted by S2) are different.
# for TUMVI model,
# a_m = Ma^-1 (R_S1C * a^C + b_a1)
# w_m = Mg^-1 (R_S1C * w^C + b_g1)
# for KSF model
# a_m = Ta * R_S2C * a^C + b_a2
# w_m = Tg * R_S2C * w^C + Ts * R_S2C * a^C + b_g2
# In practice, R_S2C is initialized with the nominal value of R_S1C and remain locked.
# As a result, we have the following relations between the TUM VI and KSF model.
# 1 for TUM VI 2 for KSF
# from KSF to TUM VI
# Ma = (Ta * R_S2C * R_S1C^T)^(-1)
# b_a1 = (Ta * R_S2C * R_S1C^T)^(-1) * b_a2
# Mg = (Tg * R_S2C * R_S1C^T)^(-1)
# b_g1 = (Tg * R_S2C * R_S1C^T)^(-1) * b_g2
# p_S1C0 = - R_S1C * p_C0S2
# p_S1C1 = R_S1C * R_S2C^T * p_S2C1
# R_S1C1 = R_S1C * R_S2C^T * R_S2C1
# from TUM VI to KSF
# Ma^(-1) * R_S1C * R_S2C^T = Ta
# Ma^(-1) * b_a1 = b_a2
# Mg^(-1) * R_S1C * R_S2C^T = Tg
# Mg^(-1) * b_g1 = b_g2
# - R_S1C^T * p_S1C0 = p_C0S2
# (R_S1C * R_S2C^T)^T * p_S1C1 = p_S2C1
# (R_S1C * R_S2C^T)^T * R_S1C1 = R_S2C1

TUMVI_IMU_INTRINSICS = {
    'Ma': np.array([[1.00422, 0, 0],
                    [-7.82123e-05, 1.00136, 0],
                    [-0.0097745, -0.000976476, 0.970467]]),
    'ba': np.array([-1.30318, -0.391441, 0.380509]),
    'Mg': np.array([[0.943611, 0.00148681, 0.000824366],
                    [0.000369694, 1.09413, -0.00273521],
                    [-0.00175252, 0.00834754, 1.01588]]),
    'bg': np.array([0.0283122, 0.00723077, 0.0165292]),
}

TUMVI_IMAGE_DELAY = 126788 * 1e-9  # nanoseconds

TUMVI_R_SC = np.array(dataset_vio_config.TUMVI_PARAMETERS['cameras'][0]['T_SC']).reshape(4, 4)[:3, :3]

Nominal_R_SC = np.array(dataset_vio_config.TUMVI_NOMINAL_PARAMETERS['cameras'][0]['T_SC']).reshape(4, 4)[:3, :3]

TUMVIS_R_NominalS = np.matmul(TUMVI_R_SC, np.transpose(Nominal_R_SC))

def quat2dcm(quaternion):
    """Returns direct cosine matrix from quaternion (Hamiltonian, [x y z w])
    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < np.finfo(float).eps * 4.0:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3]),
        (q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3]),
        (q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1])),
        dtype=np.float64)


def dcm2quat(matrix_3x3):
    """Return quaternion (Hamiltonian, [x y z w]) from rotation matrix.
    This algorithm comes from  "Quaternion Calculus and Fast Animation",
    Ken Shoemake, 1987 SIGGRAPH course notes
    (from Eigen)
    """
    q = np.empty((4, ), dtype=np.float64)
    M = np.array(matrix_3x3, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > 0.0:
        t = math.sqrt(t + 1.0)
        q[3] = 0.5 * t
        t = 0.5 / t
        q[0] = (M[2, 1] - M[1, 2]) * t
        q[1] = (M[0, 2] - M[2, 0]) * t
        q[2] = (M[1, 0] - M[0, 1]) * t
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = math.sqrt(M[i, i] - M[j, j] - M[k, k] + 1.0)
        q[i] = 0.5 * t
        t = 0.5 / t
        q[3] = (M[k, j] - M[j, k]) * t
        q[j] = (M[i, j] + M[j, i]) * t
        q[k] = (M[k, i] + M[i, k]) * t
    return q


def skew(omega):
    return np.array([[0, -omega[2], omega[1]],
                    [omega[2], 0, -omega[0]],
                    [-omega[1], omega[0], 0]])


def unskew(omega):
    """
    :param delta_rot_mat:
    :return:
    """
    return 0.5 * np.array(
        [omega[2, 1] - omega[1, 2],
        omega[0, 2] - omega[2, 0],
        omega[1, 0] - omega[0, 1]])


def simple_quat_average(quat_array):
    """
    :param quat_array: each row quat [x, y, z, w]
    :return:
    """
    # convert each quat to dcm
    dcm_array = []
    num_dcm = quat_array.shape[0]
    for row in quat_array:
        dcm_array.append(quat2dcm(row))
    theta_list = np.zeros((num_dcm, 3))
    ref_dcm = np.transpose(copy.deepcopy(dcm_array[0]))
    tol = 1e-6

    for j in range(5):
        for i in range(num_dcm):
            theta_list[i] = unskew(np.matmul(ref_dcm, dcm_array[i]) - np.eye(3))
        theta_avg = np.average(theta_list, axis = 0)
        if np.linalg.norm(theta_avg) < tol:
            print('theta changes {} very small at iter {}'.format(theta_avg, j))
            ref_dcm = np.transpose(np.matmul(np.transpose(ref_dcm), skew(theta_avg) + np.eye(3)))
            break
        ref_dcm = np.transpose(np.matmul(
            np.transpose(ref_dcm), skew(theta_avg) + np.eye(3)))

    theta_std = np.std(theta_list, axis=0) * 180 / math.pi
    return dcm2quat(np.transpose(ref_dcm)), theta_std


def compute_mean_and_std(component_label, data_array, column_range, unit, precision):
    mean = np.average(data_array[:, column_range], axis=0)
    std = np.std(data_array[:, column_range], axis=0)
    print('{}:{}'.format(component_label, unit))
    format_str = '{:.' + str(precision) + 'f} $\\pm$ {:.' + str(precision) + 'f}'
    for j in range(len(mean)):
        print(format_str.format(mean[j], std[j]))
    return mean, std


def compute_deviation_from_nominal_value(component_label, est, nominal):
    if component_label == 'q_BC1':
        return unskew(np.matmul(quat2dcm(nominal), np.transpose(quat2dcm(est))) - np.eye(3))
    else:
        return est - nominal

def get_ksf_model_parameters(tumvi_imu_parameters):
    ksf_intrinsics = {}
    invMa = np.linalg.inv(tumvi_imu_parameters['Ma'])
    invMg = np.linalg.inv(tumvi_imu_parameters['Mg'])
    ksf_intrinsics['bg'] = np.matmul(invMg, tumvi_imu_parameters['bg'])
    ksf_intrinsics['ba'] = np.matmul(invMa, tumvi_imu_parameters['ba'])
    R_S1S2 = np.matmul(TUMVI_R_SC, np.transpose(Nominal_R_SC))
    ksf_intrinsics['Tg'] = np.matmul(invMg, R_S1S2).reshape(9)
    ksf_intrinsics['Ts'] = np.zeros((3, 3)).reshape(9)
    ksf_intrinsics['Ta'] = np.matmul(invMa, R_S1S2).reshape(9)
    T_SC0 = np.array(dataset_vio_config.TUMVI_PARAMETERS['cameras'][0]['T_SC']).reshape([4, 4])
    T_SC1 = np.array(dataset_vio_config.TUMVI_PARAMETERS['cameras'][1]['T_SC']).reshape([4, 4])
    ksf_intrinsics['p_C0B'] = - np.matmul(np.transpose(T_SC0[:3, :3]), T_SC0[:3, 3])
    ksf_intrinsics['p_BC1'] = np.matmul(np.transpose(TUMVIS_R_NominalS), T_SC1[:3, 3])
    ksf_intrinsics['q_BC1'] = dcm2quat(np.matmul(np.transpose(TUMVIS_R_NominalS), T_SC1[:3, :3]))
    for j in range(2):
        suffix = str(j)
        fc = np.zeros(3)
        distort = np.zeros(4)
        tdtr = np.zeros(2)

        focal_length = dataset_vio_config.TUMVI_PARAMETERS['cameras'][j]['focal_length']
        fc[0] = (focal_length[0] + focal_length[1])*0.5
        fc[1:] = np.array(dataset_vio_config.TUMVI_PARAMETERS['cameras'][j]['principal_point'])
        distort = np.array(dataset_vio_config.TUMVI_PARAMETERS['cameras'][j]['distortion_coefficients'])
        tdtr[0] = TUMVI_IMAGE_DELAY
        tdtr[1] = dataset_vio_config.TUMVI_PARAMETERS['cameras'][j]["image_readout_time"]
        ksf_intrinsics['fc' + suffix] = fc
        ksf_intrinsics['distort' + suffix] = distort
        ksf_intrinsics['tdtr' + suffix] = tdtr

    return ksf_intrinsics


# copied from rpg trajectory evaluation toolbox plot_utils.py so as to use it in python3
def color_box(bp, color):
    elements = ['medians', 'boxes', 'caps', 'whiskers']
    # Iterate over each of the elements changing the color
    for elem in elements:
        [plt.setp(bp[elem][idx], color=color, linestyle='-', lw=1.0)
         for idx in range(len(bp[elem]))]
    return

def boxplot_compare(ax, xlabels,
                    data, data_colors):
    n_data = 1
    n_xlabel = len(xlabels)
    idx = 0
    w = 1 / (1.5 * n_data + 1.5)
    widths = [w for pos in np.arange(n_xlabel)]
    positions = [pos - 0.5 + 1.5 * w + idx * w
                 for pos in np.arange(n_xlabel)]
    # print("Positions: {0}".format(positions))
    bp = ax.boxplot(data, 0, '', positions=positions, widths=widths)
    color_box(bp, data_colors[idx])

    ax.set_xticks(np.arange(n_xlabel))
    ax.set_xticklabels(xlabels)
    xlims = ax.get_xlim()
    ax.set_xlim([xlims[0]-0.1, xlims[1]-0.1])


def boxplot_block_and_save(data_array, column_indices, xlabels, out_file,
                           ref_values, component_label, unit_label):
    """boxplot for parameters in a block and save the figure"""
    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(
        111, xlabel=component_label, ylabel='(' + unit_label + ')')
    config_colors = ['k', 'g']
    leg_labels = ['est - nominal', 'ref - nominal']
    boxplot_compare(ax, xlabels, data_array[:, column_indices],
                    config_colors)

    locs = ax.get_xticks()
    w = 1/3.0
    for index, loc in enumerate(locs):
        left_right = [loc- 0.5 * w, loc+0.5 * w]
        yvalues =  [ref_values[index],  ref_values[index]]
        ax.plot(left_right, yvalues, config_colors[1])
    leg_handles = []
    leg_line = mlines.Line2D([], [], color=config_colors[0])
    leg_handles.append(leg_line)
    leg_line = mlines.Line2D([], [], color=config_colors[1])
    leg_handles.append(leg_line)

    # ax.legend(leg_handles, leg_labels, bbox_to_anchor=(
        # 1.05, 1), loc=2, borderaxespad=0.)
    ax.legend(leg_handles, leg_labels)
    map(lambda x: x.set_visible(False), leg_handles)

    fig.tight_layout()
    # plt.show()
    fig.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close(fig)


def barplot_block_and_save(data_array, value_indices, sigma_indices, xlabels, out_file,
                           ref_values, component_label, component_label_code, unit_label):
    """
    draw the 3 sigma bounds for each column of the data_array, and also draw the reference value.
    :param data_array:
    :param value_indices: indices of values into the data array.
    :param sigma_indices: indices of sigmas into the data array.
    :param xlabels:
    :param out_file:
    :param ref_values: reference value array of the same length as value_indices.
    :param component_label:
    :param unit_label:
    :return:
    """
    print('{}: indices: {}, sigmas indices: {}.'.format(component_label, value_indices, sigma_indices))
    samples = data_array.shape[0]
    dimension = len(value_indices)

    xlist = []
    ylist = []
    yerrlist = []
    xticklist = []

    gap = 1.0
    xscale = 0.2
    for i in range(dimension):
        start = (gap + (samples + 1)* xscale) * i  # plus one for the reference values
        x = start + np.arange(0, samples, 1) * xscale
        y = data_array[:, value_indices[i]]
        yerr = data_array[:, sigma_indices[i]] * 3

        xlist.append(x)
        if samples % 2 == 0:
            xticklist.append(start + ((samples + 1) // 2 - 0.5) * xscale)
        else:
            xticklist.append(start + ((samples + 1) // 2) * xscale)
        ylist.append(y)
        yerrlist.append(yerr)

    fig, ax = plt.subplots()
    for i in range(dimension):
        ax.errorbar(xlist[i], ylist[i], yerr=yerrlist[i], ls='', ecolor=PALETTE[i],
                    capsize=2, elinewidth=2, markeredgewidth=2)
        ax.plot(xlist[i][-1] + xscale, ref_values[i], color=PALETTE[i], marker='x')

    ax.set_xticks(xticklist)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel(component_label_code + ' (' + unit_label + ')')

    fig.tight_layout()
    # plt.show()
    fig.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close(fig)


def draw_barplot_together(blockAIndex, blockBIndex, blockADim, blockBDim,
                          index_ranges, std_index_ranges, segment_list,
                          estimate_errors, out_file):
    error_indices = list(index_ranges[blockAIndex][:blockADim])
    error_indices.extend(index_ranges[blockBIndex][:blockBDim])

    std_indices = list(std_index_ranges[blockAIndex][:blockADim])
    std_indices.extend(std_index_ranges[blockBIndex][:blockBDim])

    labels = copy.deepcopy(segment_list[blockAIndex][kcolumn_labels][:blockADim])
    labels.extend(segment_list[blockBIndex][kcolumn_labels][:blockBDim])
    referrors = []
    dims = [blockADim, blockBDim]
    for index, block in enumerate([blockAIndex, blockBIndex]):
        refvalues = copy.deepcopy(segment_list[block][ktum_reference_value])
        nominal_value = segment_list[block][knominal_value]
        referror = compute_deviation_from_nominal_value(
            segment_list[block][kcomponent_label], refvalues, nominal_value)
        if isinstance(segment_list[block][kscale], list):
            for coeff_index, coeff in enumerate(segment_list[block][kscale]):
                referror[coeff_index] *= coeff
        else:
            referror *= segment_list[block][kscale]
        referrors.extend(referror[:dims[index]])

    component_label = segment_list[blockAIndex][kcomponent_label] + segment_list[blockBIndex][kcomponent_label]
    barplot_block_and_save(estimate_errors, error_indices, std_indices,
                           labels, out_file, referrors, component_label, '',
                           segment_list[blockAIndex][kerror_unit])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract the state vectors at the end of each output file of KSF, then"
                    "compute mean and std dev of estimated sensor parameters \nand compare"
                    " to the values provided by TUM VI.")
    parser.add_argument("vio_output_dir",
                        help="Output dir of KSF inside which the output csv files will be searched for.")
    parser.add_argument("plot_dir", help="Output dir of plots.")
    parser.add_argument("--seconds_to_end", type=float, default=0,
                        help="compute estimates at the time of seconds to end.")

    args = parser.parse_args()
    estimate_file_list = []
    for dir_name, subdir_list, file_list in os.walk(args.vio_output_dir):
        for fname in file_list:
            if 'swift_vio.csv' in fname:
                estimate_file_list.append(os.path.join(dir_name, fname))
    estimate_file_list.sort()
    print("Found {} calibration result files".format(len(estimate_file_list)))
    for index, fn in enumerate(estimate_file_list):
        print("{} {}".format(index, fn))

    data_list = list()
    frame_rate = 20
    estimate_index = int(args.seconds_to_end * frame_rate + 1)
    for estimate_file in estimate_file_list:
        data = np.loadtxt(estimate_file, delimiter=",", skiprows=1)
        data_list.append(data[-estimate_index])
    data_array = np.array(data_list)

    bg_index = 12
    ksf_params = get_ksf_model_parameters(TUMVI_IMU_INTRINSICS)
    # each element: component label, column label for each dimension of the component, scale,
    # tum reference value, unit, number of digits after dot to display, minimal dimension, nominal value
    kcomponent_label_code = 0
    kcolumn_labels = 1
    kscale = 2
    ktum_reference_value = 3
    kerror_unit = 4
    ksignificant_digits = 5
    kminimal_dim = 6
    knominal_value = 7
    kcomponent_label = 8
    kindex_list = 9
    kstd_index_list = 10

    segment_list = [(r'$\mathbf{b}_g$', [r'$x$', r'$y$', r'$z$'], 180 / math.pi, ksf_params['bg'], r'$^\circ/s$', 3, 3,
                     np.array([0, 0, 0]), 'bg'),
                    (r'$\mathbf{b}_a$', [r'$x$', r'$y$', r'$z$'], 1.0, ksf_params['ba'], r'$m/s^2$', 3, 3,
                     np.array([0, 0, 0]), 'ba'),
                    (r'$\mathbf{T}_g$',
                     [r'$1,1$', r'$1,2$', r'$1,3$', r'$2,1$', r'$2,2$', r'$2,3$', r'$3,1$', r'$3,2$', r'$3,3$'],
                     1000.0, ksf_params['Tg'], '0.001', 2, 9, np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]), 'Tg'),
                    (r'$\mathbf{T}_s$',
                     [r'$1,1$', r'$1,2$', r'$1,3$', r'$2,1$', r'$2,2$', r'$2,3$', r'$3,1$', r'$3,2$', r'$3,3$'],
                     1000.0, ksf_params['Ts'], r'$0.001 \frac{rad/s}{m/s^2}$', 2, 9,
                     np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), 'Ts'),
                    (r'$\mathbf{T}_a$',
                     [r'$1,1$', r'$1,2$', r'$1,3$', r'$2,1$', r'$2,2$', r'$2,3$', r'$3,1$', r'$3,2$', r'$3,3$'],
                     1000.0, ksf_params['Ta'], '0.001', 2, 9, np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]), 'Ta'),
                    ('', [r'$\mathbf{p}_{C0B}(1)$', r'$\mathbf{p}_{C0B}(2)$', r'$\mathbf{p}_{C0B}(3)$'], 100,
                     ksf_params['p_C0B'], r'$cm$', 2, 3, np.array([0, 0, 0]), 'p_C0B'),
                    ('', [r'$f^0$', r'$c_x^0$', r'$c_y^0$'], 1, ksf_params['fc0'], r'$px$', 2, 3, [190, 256, 256],
                     'fc0'),
                    ('', [r'$k_1^0$', r'$k_2^0$', r'$k_3^0$', r'$k_4^0$'], 1000, ksf_params['distort0'], '0.001', 2, 4,
                     np.array([0, 0, 0, 0]), 'distort0'),
                    ('', [r'$t_d^0$', r'$t_r^0$'], 1000, ksf_params['tdtr0'], r'$ms$', 2, 2, np.array([0, 0]), 'tdtr0'),
                    ('', [r'$\mathbf{p}_{BC1}(1)$', r'$\mathbf{p}_{BC1}(2)$', r'$\mathbf{p}_{BC1}$(3)'], 100,
                     ksf_params['p_BC1'], r'$cm$', 2, 3, np.array([0, 0, 0]), 'p_BC1'),
                    (r'$\mathbf{R}_{BC1}$', ['roll', 'pitch', 'yaw', 'w'],
                     180 / math.pi, ksf_params['q_BC1'], r'$^\circ$', 3, 3, dcm2quat(Nominal_R_SC), 'q_BC1'),
                    ('', [r'$f^1$', r'$c_x^1$', r'$c_y^1$'], 1, ksf_params['fc1'], r'$px$', 2, 3,
                     np.array([190, 256, 256]), 'fc1'),
                    ('', [r'$k_1^1$', r'$k_2^1$', r'$k_3^1$', r'$k_4^1$'], 1000, ksf_params['distort1'], '0.001', 2, 4,
                     np.array([0, 0, 0, 0]), 'distort1'),
                    ('', [r'$t_d^1$', r'$t_r^1$'], 1000, ksf_params['tdtr1'], r'$ms$', 2, 2, np.array([0, 0]), 'tdtr1')]

    index_ranges = list()
    start_index = bg_index
    for segment in segment_list:
        index_ranges.append(range(start_index, start_index + len(segment[kcolumn_labels])))
        start_index = index_ranges[-1][-1] + 1
    std_index_ranges = list()
    std_padding = 3 + 3 + 3 # pose and velocity
    std_start_index = index_ranges[-1][-1] + std_padding + 1
    for segment in segment_list:
        std_index_ranges.append(range(std_start_index, std_start_index + segment[kminimal_dim]))
        std_start_index = std_index_ranges[-1][-1] + 1

    # compute the difference between reference values and nominal values
    for index, segment in enumerate(segment_list):
        refvalues = copy.deepcopy(segment[ktum_reference_value])
        nominal_value = np.array(segment[knominal_value])
        referrors = compute_deviation_from_nominal_value(segment[kcomponent_label], refvalues, nominal_value)
        if isinstance(segment[kscale], list):
            for coeff_index, coeff in enumerate(segment[kscale]):
                referrors[coeff_index] *= coeff
        else:
            referrors *= segment[kscale]

        print('ref {}:{}'.format(segment[kcomponent_label], segment[kerror_unit]))
        format_str = '{:.' + str(segment[ksignificant_digits]) + 'f}'
        for j in range(len(referrors)):
            print(format_str.format(referrors[j]))

    # compute and visualize the differences between estimated values and nominal values.
    # The boxplot also visualizes the difference of the reference values to the nominal values.
    estimate_errors = copy.deepcopy(data_array)
    selected_indices = [4, 0, 1, 0, 3, 0]
    success_trials = [0, 5, 4, 3, 5, 5, 5]
    cum_indices = np.cumsum(success_trials)
    cum_chosen_indices = [cum_indices[i] + index for i, index in enumerate(selected_indices)]
    # cum_chosen_indices = range(0, len(estimate_file_list))
    print('chosen indices\n')
    for i, index in enumerate(cum_chosen_indices):
        print("{} {}".format(i, estimate_file_list[index]))
    for index, segment in enumerate(segment_list):
        refvalues = copy.deepcopy(segment[ktum_reference_value])
        nominal_value = segment[knominal_value]
        error_indices = index_ranges[index][0:segment[kminimal_dim]]

        for j in range(estimate_errors.shape[0]):
            estimate_errors[j, error_indices] = compute_deviation_from_nominal_value(
                segment[kcomponent_label], data_array[j, index_ranges[index]], nominal_value)

        referrors = compute_deviation_from_nominal_value(
            segment[kcomponent_label], refvalues, nominal_value)

        if isinstance(segment[kscale], list):
            for coeff_index, coeff in enumerate(segment[kscale]):
                estimate_errors[:, error_indices[coeff_index]] *= coeff
                referrors[coeff_index] *= coeff
                estimate_errors[:, std_index_ranges[index][coeff_index]] *= coeff
        else:
            estimate_errors[:, error_indices] *= segment[kscale]
            referrors *= segment[kscale]
            estimate_errors[:, std_index_ranges[index]] *= segment[kscale]
        out_file = args.plot_dir + '/' + segment[kcomponent_label] + FORMAT
        compute_mean_and_std(segment[kcomponent_label], estimate_errors, error_indices, segment[kerror_unit],
                             segment[ksignificant_digits])
        boxplot_block_and_save(estimate_errors, error_indices,
                               segment[kcolumn_labels][0:segment[kminimal_dim]],
                               out_file, referrors, segment[kcomponent_label],
                               segment[kerror_unit])

        # draw the 3\sigma bounds for estimated parameter values and (reference values - nominal values).
        out_file = args.plot_dir + '/std_' + segment[kcomponent_label] + FORMAT
        barplot_block_and_save(estimate_errors[cum_chosen_indices, :], error_indices, std_index_ranges[index],
                               segment[kcolumn_labels][0:segment[kminimal_dim]],
                               out_file, referrors, segment[kcomponent_label], segment[kcomponent_label_code],
                               segment[kerror_unit])

    # draw p_C0B and p_BC1 together
    out_file = args.plot_dir + '/std_p_C0B_p_BC1' + FORMAT
    draw_barplot_together(5, 9, 3, 3, index_ranges, std_index_ranges, segment_list,
                          estimate_errors[cum_chosen_indices, :], out_file)
    # draw fc0 fc1 together
    out_file = args.plot_dir + '/std_fc0_fc1' + FORMAT
    draw_barplot_together(6, 11, 3, 3, index_ranges, std_index_ranges, segment_list,
                          estimate_errors[cum_chosen_indices, :], out_file)
    # draw distort0 and distort1 together
    out_file = args.plot_dir + '/std_dist0_dist1' + FORMAT
    draw_barplot_together(7, 12, 2, 2, index_ranges, std_index_ranges, segment_list,
                          estimate_errors[cum_chosen_indices, :], out_file)
    # draw td tr together
    out_file = args.plot_dir + '/std_tdtr0_tdtr1' + FORMAT
    draw_barplot_together(8, 13, 2, 2, index_ranges, std_index_ranges, segment_list,
                          estimate_errors[cum_chosen_indices, :], out_file)
