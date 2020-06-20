#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Given output of KSF for a group of data sequences, compute and visualize
 statistics of estimated sensor parameters, e.g., bg ba Tg Ts Ta, and
 extrinsic, intrinsic, and temporal parameters for each camera.

 Currently this module works only with TUM VI to compare estimated parameters
  by KSF vs provided values from TUM VI.
 """

import copy
import math
import os
import sys

import numpy as np

import matplotlib.pyplot as plt

import kalibr_okvis_config

FORMAT = '.pdf'
PALLETE = ['b', 'g', 'r', 'c', 'k', 'y', 'm']

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

TUMVI_R_SC = np.array(kalibr_okvis_config.TUMVI_PARAMETERS['cameras'][0]['T_SC']).reshape(4, 4)[:3, :3]

Nominal_R_SC = np.array(kalibr_okvis_config.TUMVI_NOMINAL_PARAMETERS['cameras'][0]['T_SC']).reshape(4, 4)[:3, :3]

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
    T_SC0 = np.array(kalibr_okvis_config.TUMVI_PARAMETERS['cameras'][0]['T_SC']).reshape([4, 4])
    T_SC1 = np.array(kalibr_okvis_config.TUMVI_PARAMETERS['cameras'][1]['T_SC']).reshape([4, 4])
    ksf_intrinsics['p_C0B'] = - np.matmul(np.transpose(T_SC0[:3, :3]), T_SC0[:3, 3])
    ksf_intrinsics['p_BC1'] = np.matmul(np.transpose(TUMVIS_R_NominalS), T_SC1[:3, 3])
    ksf_intrinsics['q_BC1'] = dcm2quat(np.matmul(np.transpose(TUMVIS_R_NominalS), T_SC1[:3, :3]))
    for j in range(2):
        suffix = str(j)
        fc = np.zeros(3)
        distort = np.zeros(4)
        tdtr = np.zeros(2)

        focal_length = kalibr_okvis_config.TUMVI_PARAMETERS['cameras'][j]['focal_length']
        fc[0] = (focal_length[0] + focal_length[1])*0.5
        fc[1:] = np.array(kalibr_okvis_config.TUMVI_PARAMETERS['cameras'][j]['principal_point'])
        distort = np.array(kalibr_okvis_config.TUMVI_PARAMETERS['cameras'][j]['distortion_coefficients'])
        tdtr[0] = TUMVI_IMAGE_DELAY
        tdtr[1] = kalibr_okvis_config.TUMVI_PARAMETERS['cameras'][j]["image_readout_time"]
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

# copied from rpg trajectory evaluation toolbox plot_utils.py so as to use it in python3
def boxplot_compare(ax, xlabels,
                    data, data_labels, data_colors,
                    legend=True):
    n_data = 1
    n_xlabel = len(xlabels)
    leg_handles = []
    leg_labels = []
    idx = 0
    d = data
    # print("idx and d: {0} and {1}".format(idx, d))
    w = 1 / (1.5 * n_data + 1.5)
    widths = [w for pos in np.arange(n_xlabel)]
    positions = [pos - 0.5 + 1.5 * w + idx * w
                 for pos in np.arange(n_xlabel)]
    # print("Positions: {0}".format(positions))
    bp = ax.boxplot(d, 0, '', positions=positions, widths=widths)
    color_box(bp, data_colors[idx])
    # tmp, = plt.plot([1, 1], data_colors[idx])
    # leg_handles.append(tmp)
    leg_labels.append(data_labels[idx])
    idx += 1
    ax.set_xticks(np.arange(n_xlabel))
    ax.set_xticklabels(xlabels)
    xlims = ax.get_xlim()
    ax.set_xlim([xlims[0]-0.1, xlims[1]-0.1])
    if legend:
        # ax.legend(leg_handles, leg_labels, bbox_to_anchor=(
            # 1.05, 1), loc=2, borderaxespad=0.)
        ax.legend(leg_handles, leg_labels)
    map(lambda x: x.set_visible(False), leg_handles)


def boxplot_block_and_save(data_array, column_indices, xlabels, out_file,
                           ref_values, component_label, unit_label):
    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(
        111, xlabel=component_label, ylabel='(' + unit_label + ')')
    config_colors = ['k']
    legend_labels = list(column_indices)
    boxplot_compare(ax, xlabels, data_array[:, column_indices],
                       legend_labels, config_colors, legend=False)
    locs = ax.get_xticks()
    w = 1/3.0
    for index, loc in enumerate(locs):
        left_right = [loc- 0.5 * w, loc+0.5 * w]
        yvalues =  [ref_values[index],  ref_values[index]]
        ax.plot(left_right, yvalues, 'g-')

    fig.tight_layout()
    # plt.show()
    fig.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    _, output_dir = sys.argv
    estimate_file_list = []
    for dir_name, subdir_list, file_list in os.walk(output_dir):
        for fname in file_list:
            if 'msckf_estimates.csv' in fname:
                estimate_file_list.append(os.path.join(dir_name, fname))
    print("Found {} calibration result files: {}".format(len(estimate_file_list), '\n'.join(estimate_file_list)))
    data_list = list()
    for estimate_file in estimate_file_list:
        data = np.loadtxt(estimate_file, delimiter=",", skiprows=1)
        data_list.append(data[-1])

    data_array = np.array(data_list)

    bg_index = 12
    ksf_params = get_ksf_model_parameters(TUMVI_IMU_INTRINSICS)
    # each element: component label, column label for each dimension of the component, scale,
    # tum reference value, unit, number of digits after dot to display, minimal dimension, nominal value
    kcomponent_label = 0
    kcolumn_labels = 1
    kscale = 2
    ktum_reference_value = 3
    kerror_unit = 4
    ksignificant_digits = 5
    kminimal_dim = 6
    knominal_value = 7

    segment_list = [('bg', ['x', 'y', 'z'], 180 / math.pi, ksf_params['bg'], 'deg/s', 3, 3, np.array([0, 0, 0])),
                    ('ba', ['x', 'y', 'z'], 1.0, ksf_params['ba'], 'm/s^2', 3, 3, np.array([0, 0, 0])),
                    ('Tg', [1, 2, 3, 4, 5, 6, 7, 8, 9], 1000.0, ksf_params['Tg'], '0.001', 2, 9,
                     np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])),
                    ('Ts', [1, 2, 3, 4, 5, 6, 7, 8, 9], 1000.0, ksf_params['Ts'], '0.001 rad/s/(m/s^2)', 2, 9,
                     np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])),
                    ('Ta', [1, 2, 3, 4, 5, 6, 7, 8, 9], 1000.0, ksf_params['Ta'], '0.001', 2, 9,
                     np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])),
                    ('p_C0B', ['x', 'y', 'z'], 100, ksf_params['p_C0B'], 'cm', 2, 3, np.array([0, 0, 0])),
                    ('p_BC1', ['x', 'y', 'z'], 100, ksf_params['p_BC1'], 'cm', 2, 3, np.array([0, 0, 0])),
                    ('q_BC1', ['x', 'y', 'z', 'w'],
                     180 / math.pi, ksf_params['q_BC1'], 'deg', 3, 3, dcm2quat(Nominal_R_SC)),
                    ('fc0', ['f', 'cx', 'cy'], 1, ksf_params['fc0'], 'px', 2, 3, [190, 256, 256]),
                    ('distort0', ['k1', 'k2', 'k3', 'k4'], 1000, ksf_params['distort0'], '0.001', 2, 4,
                     np.array([0, 0, 0, 0])),
                    ('tdtr0', ['td', 'tr'], 1000, ksf_params['tdtr0'], 'ms', 2, 2, np.array([0, 0])),
                    ('fc1', ['f', 'cx', 'cy'], 1, ksf_params['fc1'], 'px', 2, 3, np.array([190, 256, 256])),
                    ('distort1', ['k1', 'k2', 'k3', 'k4'], 1000, ksf_params['distort1'], '0.001', 2, 4,
                     np.array([0, 0, 0, 0])),
                    ('tdtr1', ['td', 'tr'], 1000, ksf_params['tdtr1'], 'ms', 2, 2, np.array([0, 0]))]

    index_ranges = list()
    for segment in segment_list:
        if len(index_ranges):
            start = index_ranges[-1][-1] + 1
            index_ranges.append(range(start, start + len(segment[kcolumn_labels])))
        else:
            index_ranges.append(range(bg_index, bg_index + len(segment[kcolumn_labels])))

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

    for index, segment in enumerate(segment_list):
        refvalues = copy.deepcopy(segment[ktum_reference_value])
        nominal_value = segment[knominal_value]
        error_indices = index_ranges[index][0:segment[kminimal_dim]]
        # we put the estimate deviation from nominal value directly into the estimates data array.

        for j in range(data_array.shape[0]):
            data_array[j, error_indices] = compute_deviation_from_nominal_value(
                segment[kcomponent_label], data_array[j, index_ranges[index]], nominal_value)

        referrors = compute_deviation_from_nominal_value(
            segment[kcomponent_label], refvalues, nominal_value)

        if isinstance(segment[kscale], list):
            for coeff_index, coeff in enumerate(segment[kscale]):
                data_array[:, error_indices[coeff_index]] *= coeff
                referrors[coeff_index] *= coeff
        else:
            data_array[:, error_indices] *= segment[kscale]
            referrors *= segment[kscale]
        out_file = output_dir + '/' + segment[kcomponent_label] + FORMAT
        compute_mean_and_std(segment[kcomponent_label], data_array, error_indices, segment[kerror_unit],
                             segment[ksignificant_digits])
        boxplot_block_and_save(data_array, error_indices,
                               segment[kcolumn_labels][0:segment[kminimal_dim]],
                               out_file, referrors, segment[kcomponent_label],
                               segment[kerror_unit])
