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

import dataParams
import mathUtils
import OpenVinsIO

FORMAT = '.pdf'
PALETTE = plt.rcParams['axes.prop_cycle'].by_key()['color']

# ['b', 'g', 'r', 'c', 'k', 'y', 'm', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
#             'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'Purples', 'Oranges',]


def get_kswf_model_parameters(tumvi_imu_parameters):
    """
    convert TUM VI calibration result to KSWF model parameters.
    :param tumvi_imu_parameters:
    :return:
    """
    kswf_params = {}
    invMa = np.linalg.inv(tumvi_imu_parameters['Ma'])
    invMg = np.linalg.inv(tumvi_imu_parameters['Mg'])
    kswf_params['bg'] = np.matmul(invMg, tumvi_imu_parameters['bg'])
    kswf_params['ba'] = np.matmul(invMa, tumvi_imu_parameters['ba'])
    R_S1S2 = np.matmul(dataParams.TUMVI_R_SC0, np.transpose(dataParams.TUMVI_NominalS_R_C0))
    kswf_params['Tg'] = np.matmul(invMg, R_S1S2).reshape(9)
    kswf_params['Ts'] = np.zeros((3, 3)).reshape(9)
    kswf_params['Ta'] = np.matmul(invMa, R_S1S2).reshape(9)
    T_SC0 = np.array(dataParams.TUMVI_PARAMETERS['cameras'][0]['T_SC']).reshape([4, 4])
    T_SC1 = np.array(dataParams.TUMVI_PARAMETERS['cameras'][1]['T_SC']).reshape([4, 4])
    kswf_params['p_C0B'] = - np.matmul(np.transpose(T_SC0[:3, :3]), T_SC0[:3, 3])

    S_R_NominalS = np.matmul(dataParams.TUMVI_R_SC0, dataParams.TUMVI_NominalS_R_C0.T)
    kswf_params['p_BC1'] = np.matmul(S_R_NominalS.T, T_SC1[:3, 3])
    kswf_params['q_BC1'] = mathUtils.dcm2quat(np.matmul(S_R_NominalS.T, T_SC1[:3, :3]))
    for j in range(2):
        suffix = str(j)
        fc = np.zeros(3)

        focal_length = dataParams.TUMVI_PARAMETERS['cameras'][j]['focal_length']
        fc[0] = (focal_length[0] + focal_length[1])*0.5
        fc[1:] = np.array(dataParams.TUMVI_PARAMETERS['cameras'][j]['principal_point'])
        distort = np.array(dataParams.TUMVI_PARAMETERS['cameras'][j]['distortion_coefficients'])

        kswf_params['fc' + suffix] = fc
        kswf_params['distort' + suffix] = distort
        kswf_params['tdtr' + suffix] = [dataParams.TUMVI_IMAGE_DELAY,
                                        dataParams.TUMVI_PARAMETERS['cameras'][j]["image_readout_time"]]

    return kswf_params


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
    leg_labels = ['est - initial', 'ref - initial']
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
    fig.savefig(out_file, bbox_inches="tight", dpi=300, transparent=True)
    plt.close(fig)


def barplot_block_and_save(data_array, value_indices, sigma_indices, xlabels, out_file,
                           ref_values, component_label, component_label_code, unit_label,
                           aspect, row_ranges, leg_labels=None):
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
    :param aspect: figure aspect ratio. the ratio of y-unit to x-unit.
    :param row_ranges: draw group of rows in data_array with bars of different colors.
    :param group_labels: legend labels for each group including reference.
    :return:
    """

    print('{}: indices: {}, sigmas indices: {}.'.format(component_label, value_indices, sigma_indices))
    samples = data_array.shape[0]
    dimension = len(value_indices)
    if row_ranges is None:
        row_ranges = [range(0, samples)]

    xlist = []
    ylist = []
    yerrlist = []
    xticklist = []

    gap = 1.0
    xscale = 0.2
    start = 0
    for i in range(dimension):
        usedSamples = np.count_nonzero(data_array[:, sigma_indices[i]])
        x = start + np.arange(0, samples, 1) * xscale
        y = data_array[:, value_indices[i]]
        yerr = data_array[:, sigma_indices[i]] * 3  # +/-3 \sigma bounds

        xlist.append(x)
        if usedSamples % 2 == 0:
            xticklist.append(start + ((usedSamples + 1) // 2) * xscale)
        else:
            xticklist.append(start + ((usedSamples + 1) // 2) * xscale)
        ylist.append(y)
        yerrlist.append(yerr)
        start += (gap + (usedSamples + 1) * xscale)  # plus one for the reference values

    fig, ax = plt.subplots()
    usedColors = []
    for i in range(dimension):
        colors = []
        usedRowRange = row_ranges[0]
        for j, rowrange in enumerate(row_ranges):
            # skip rowranges with 0 std
            if np.all((yerrlist[i][rowrange] == 0)):
                continue
            ax.errorbar(xlist[i][rowrange], ylist[i][rowrange], yerr=yerrlist[i][rowrange],
                        ls='', ecolor=PALETTE[j],
                        capsize=0.0, elinewidth=1, markeredgewidth=0.0)
            colors.append(PALETTE[j])
            usedRowRange = rowrange
        if len(colors) > len(usedColors):
            usedColors = copy.deepcopy(colors)
        ax.plot(xlist[i][usedRowRange[-1]] + xscale, ref_values[i], color='r', marker='x', markersize=4)

    if leg_labels:
        leg_handles = []
        for c in usedColors:
            leg_line = mlines.Line2D([], [], color=c)
            leg_handles.append(leg_line)
        leg_line = plt.plot([], [], marker="x", ms=4, ls="", mec=None, color='r')[0]
        leg_handles.append(leg_line)
        ax.legend(leg_handles, leg_labels)

    ax.set_xticks(xticklist)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel(component_label_code + ' (' + unit_label + ')')
    ax.set_aspect(aspect)
    fig.tight_layout()
    # plt.show()
    fig.savefig(out_file, bbox_inches="tight", dpi=300, transparent=True)
    plt.close(fig)


def draw_barplot_together(blockAIndex, blockBIndex, blockADim, blockBDim,
                          index_ranges, std_index_ranges, SegmentList,
                          estimate_errors, out_file, aspect, row_ranges, leg_labels=None):
    error_indices = list(index_ranges[blockAIndex][:blockADim])
    error_indices.extend(index_ranges[blockBIndex][:blockBDim])

    std_indices = list(std_index_ranges[blockAIndex][:blockADim])
    std_indices.extend(std_index_ranges[blockBIndex][:blockBDim])

    labels = copy.deepcopy(SegmentList[blockAIndex][kcolumn_labels][:blockADim])
    labels.extend(SegmentList[blockBIndex][kcolumn_labels][:blockBDim])
    referrors = []
    dims = [blockADim, blockBDim]
    for index, block in enumerate([blockAIndex, blockBIndex]):
        refvalues = copy.deepcopy(SegmentList[block][ktum_reference_value])
        initial_value = SegmentList[block][kinitial_value]
        referror = mathUtils.compute_deviation_from_initial_value(
            SegmentList[block][kcomponent_label], refvalues, initial_value)
        if isinstance(SegmentList[block][kscale], list):
            for coeff_index, coeff in enumerate(SegmentList[block][kscale]):
                referror[coeff_index] *= coeff
        else:
            referror *= SegmentList[block][kscale]
        referrors.extend(referror[:dims[index]])

    component_label = SegmentList[blockAIndex][kcomponent_label] + SegmentList[blockBIndex][kcomponent_label]
    barplot_block_and_save(estimate_errors, error_indices, std_indices,
                           labels, out_file, referrors, component_label, '',
                           SegmentList[blockAIndex][kerror_unit], aspect, row_ranges, leg_labels)


def select_runs_for_3sigma_bounds_example():
    """Select the runs with small ATE rot and small ATE trans which will be used to plot the 3sigma bounds"""
    transerrors = [0.09022562756865582, 0.13114642737489238, 0.25622578793125195, 0.36267676472363924,
                   0.13306107509360549, 1000, 0.23050848389168876, 0.3218183154650858, 0.6003507083814829,
                   0.22879267809510662, 0.5719646823836804, 0.24911421900416536, 0.3408112171871047,
                   87.34580654604498, 644.3724045394401, 0.1332637773118743, 2.3374334160602688,
                   0.13391120417289787, 0.1044488889419483, 1.535957438210344, 0.10096161195107226,
                   0.6655764678454069, 0.08360377795799294, 0.10446224966712461, 0.08955341880147774,
                   0.15536441838692425,
                   0.1645168781777384, 0.17826573889485225, 0.06815480978239215, 0.34653887754949936]
    roterrors = [3.502730633663846, 4.042757989950087, 5.410597858239634, 11.017149785882092, 3.4981200002788104, 1000,
                 10.205941538071434, 11.543153038521977, 26.357714255405615, 8.814572915966675, 5.394925854219783,
                 4.528571781458833, 6.09258478360548, 148.25187832087994, 105.64793030011404, 4.457533097768242,
                 22.278626038474375, 3.9906416854988467, 5.13630378830125, 12.906875327849283, 4.967389016486302,
                 9.333908490980884, 5.095433611284956, 3.964432295162097, 4.244687943017181, 3.7912300430151045,
                 3.6987893821244606, 4.593540033699874, 3.1289402267398048, 5.723220901478907]
    dataindices = []
    for i in range(6):
        for j in range(5):
            dataindices.append((i, j))

    assert len(transerrors) == 30
    assert len(roterrors) == 30
    assert len(dataindices) == 30

    transindices = [x for _, x in sorted(zip(transerrors, dataindices))]
    rotindices = [x for _, x in sorted(zip(roterrors, dataindices))]
    print('Trial indices of ordered trans errors: \n{}\nTrial indices of ordered rot errors:\n{}'.format(transindices, rotindices))
    print('The candidate runs for plotting 3sigma bounds can be selected from the beginning part.')
    print('Note you have to deduce the indices in the main plot function which does not count the failed trials.')


def selected_runs_for_3sigma_bounds(estimate_file_list):
    selected_indices = [0, 3, 1, 3, 2, 3]  # actual #run index should be [0, 4, 1, 3, 2, 3]
    # selected_indices = [0, 3, 1, 2, 3, 3]  # actual #run index should be [0, 4, 1, 2, 3, 3]
    success_trials = [0, 5, 4, 3, 5, 5, 5]
    cum_indices = np.cumsum(success_trials)
    cum_chosen_indices = [cum_indices[i] + index for i, index in enumerate(selected_indices)]
    # cum_chosen_indices = range(0, len(estimate_file_list))
    print('chosen Swift VIO files to visualize 3 sigma bounds')
    for i, index in enumerate(cum_chosen_indices):
        print("{} {}".format(i, estimate_file_list[index]))
    return cum_chosen_indices


def loadSwiftVioResults(vio_output_dir, seconds_to_end):
    estimate_file_list = []
    for dir_name, subdir_list, file_list in os.walk(vio_output_dir):
        for fname in file_list:
            if 'swift_vio.csv' in fname:
                estimate_file_list.append(os.path.join(dir_name, fname))
    estimate_file_list.sort()
    print("Found {} calibration result files".format(len(estimate_file_list)))
    for index, fn in enumerate(estimate_file_list):
        print("{} {}".format(index, fn))

    data_list = list()
    rate = 0
    backrowindex = -1
    for estimate_file in estimate_file_list:
        data = np.loadtxt(estimate_file, delimiter=",", skiprows=1)
        if rate == 0:
            interval = np.mean(np.diff(data[:, 0]))
            rate = 1.0 / interval
            backrowindex = math.floor(seconds_to_end * rate + 1)
            print('Swift VIO state estimate backrow index {}'.format(backrowindex))
        print('Selected Swift VIO state at {}'.format(data[-backrowindex, 0]))
        data_list.append(data[-backrowindex])
    return np.array(data_list), estimate_file_list


SWIFTVIO_BG_INDEX = 12 # 0-based index of the first dimension of gyro bias.
kcomponent_label = 0
kcomponent_label_code = 1 # ylabel for bar plots, can be empty.
kcolumn_labels = 2 # column label for each dimension of the component
kscale = 3 # scale factor in drawing
ktum_reference_value = 4 # reference value provided by TUM VI authors.
kerror_unit = 5
ksignificant_digits = 6 # number of digits after dot to display
kminimal_dim = 7
kinitial_value = 8 # Initial value to the estimators.
kaspect = 9

# One output format of swift_vio.
ReferenceParameters = get_kswf_model_parameters(dataParams.TUMVI_IMU_INTRINSICS)
SegmentList = [('bg', r'$\mathbf{b}_g$', [r'$x$', r'$y$', r'$z$'], 180 / math.pi, ReferenceParameters['bg'], r'$^\circ/s$', 3, 3,
                 np.array([0.0, 0.0, 0.0]), 1.0),
                ('ba', r'$\mathbf{b}_a$', [r'$x$', r'$y$', r'$z$'], 1.0, ReferenceParameters['ba'], r'$m/s^2$', 3, 3,
                 np.array([0.0, 0.0, 0.0]), 1.6),
                ('Tg', r'$\mathbf{T}_g$',
                 [r'$1,1$', r'$1,2$', r'$1,3$', r'$2,1$', r'$2,2$', r'$2,3$', r'$3,1$', r'$3,2$', r'$3,3$'],
                 1000.0, ReferenceParameters['Tg'], '0.001', 2, 9, np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]), 0.1),
                ('Ts', r'$\mathbf{T}_s$',
                 [r'$1,1$', r'$1,2$', r'$1,3$', r'$2,1$', r'$2,2$', r'$2,3$', r'$3,1$', r'$3,2$', r'$3,3$'],
                 1000.0, ReferenceParameters['Ts'], r'$0.001 \frac{rad/s}{m/s^2}$', 2, 9,
                 np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), 2.0),
                ('Ta', r'$\mathbf{T}_a$',
                 [r'$1,1$', r'$1,2$', r'$1,3$', r'$2,1$', r'$2,2$', r'$2,3$', r'$3,1$', r'$3,2$', r'$3,3$'],
                 1000.0, ReferenceParameters['Ta'], '0.001', 2, 9, np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]), 0.17),
                ('p_C0B',  '', [r'$\mathbf{p}_{C0B}(1)$', r'$\mathbf{p}_{C0B}(2)$', r'$\mathbf{p}_{C0B}(3)$'], 100,
                 ReferenceParameters['p_C0B'], r'$cm$', 2, 3, np.array([0.0, 0.0, 0.0]), 0.5),
                ('fc0', '', [r'$f^0$', r'$c_x^0$', r'$c_y^0$'], 1, ReferenceParameters['fc0'], r'$px$', 2, 3, [190, 256, 256], 0.5),
                ('distort0', '', [r'$k_1^0$', r'$k_2^0$', r'$k_3^0$', r'$k_4^0$'], 1000, ReferenceParameters['distort0'], '0.001', 2, 4,
                 np.array([0, 0, 0, 0]), 0.5),
                ('tdtr0', '', [r'$t_d^0$', r'$t_r^0$'], 1000, ReferenceParameters['tdtr0'], r'$ms$', 2, 2, np.array([0, 0.02]), 0.5),
                ('p_BC1', '', [r'$\mathbf{p}_{BC1}(1)$', r'$\mathbf{p}_{BC1}(2)$', r'$\mathbf{p}_{BC1}$(3)'], 100,
                 ReferenceParameters['p_BC1'], r'$cm$', 2, 3, np.array([0.0, 0.0, 0.0]), 0.5),
                ('q_BC1', r'$\mathbf{R}_{BC1}$', ['roll', 'pitch', 'yaw', 'w'],
                 180 / math.pi, ReferenceParameters['q_BC1'], r'$^\circ$', 3, 3, mathUtils.dcm2quat(dataParams.TUMVI_NominalS_R_C0), 2.2),
                ('fc1', '', [r'$f^1$', r'$c_x^1$', r'$c_y^1$'], 1, ReferenceParameters['fc1'], r'$px$', 2, 3,
                 np.array([190, 256, 256]), 0.5),
                ('distort1', '', [r'$k_1^1$', r'$k_2^1$', r'$k_3^1$', r'$k_4^1$'], 1000, ReferenceParameters['distort1'], '0.001', 2, 4,
                 np.array([0, 0, 0, 0]), 0.5),
                ('tdtr1', '', [r'$t_d^1$', r'$t_r^1$'], 1000, ReferenceParameters['tdtr1'], r'$ms$', 2, 2, np.array([0, 0.02]), 0.5)]

# Another output format for BG_BA, P_BC_Q_BC, and P_BC_Q_BC.
# ref_T_SC0 = np.array(dataParams.TUMVI_PARAMETERS['cameras'][0]['T_SC']).reshape([4, 4])
# ref_T_SC1 = np.array(dataParams.TUMVI_PARAMETERS['cameras'][1]['T_SC']).reshape([4, 4])
#
# SegmentList = [
#     ('bg', r'$\mathbf{b}_g$', [r'$x$', r'$y$', r'$z$'], 180 / math.pi, ReferenceParameters['bg'], r'$^\circ/s$', 3, 3,
#      np.array([0.0, 0.0, 0.0]), 1.0),
#     ('ba', r'$\mathbf{b}_a$', [r'$x$', r'$y$', r'$z$'], 1.0, ReferenceParameters['ba'], r'$m/s^2$', 3, 3,
#      np.array([0.0, 0.0, 0.0]), 1.6),
#     ('p_BC0', '', [r'$\mathbf{p}_{BC0}(1)$', r'$\mathbf{p}_{BC0}(2)$', r'$\mathbf{p}_{BC0}$(3)'], 100,
#      ref_T_SC0[:3, 3], r'$cm$', 2, 3, np.array([0.0, 0.0, 0.0]), 0.5),
#     ('q_BC0', r'$\mathbf{R}_{BC0}$', ['roll', 'pitch', 'yaw', 'w'],
#      180 / math.pi, mathUtils.dcm2quat(ref_T_SC0[:3, :3]), r'$^\circ$', 3, 3,
#      mathUtils.dcm2quat(dataParams.TUMVI_NominalS_R_C0), 2.2),
#     ('fc0', '', [r'$f^0$', r'$c_x^0$', r'$c_y^0$'], 1, ReferenceParameters['fc0'], r'$px$', 2, 3, [190, 256, 256], 0.5),
#     ('distort0', '', [r'$k_1^0$', r'$k_2^0$', r'$k_3^0$', r'$k_4^0$'], 1000, ReferenceParameters['distort0'], '0.001', 2, 4,
#     np.array([0, 0, 0, 0]), 0.5),
#     ('tdtr0', '', [r'$t_d^0$', r'$t_r^0$'], 1000, ReferenceParameters['tdtr0'], r'$ms$', 2, 2, np.array([0, 0.02]), 0.5),
#     ('p_BC1', '', [r'$\mathbf{p}_{BC1}(1)$', r'$\mathbf{p}_{BC1}(2)$', r'$\mathbf{p}_{BC1}$(3)'], 100,
#      ref_T_SC1[:3, 3], r'$cm$', 2, 3, np.array([0.0, 0.0, 0.0]), 0.5),
#     ('q_BC1', r'$\mathbf{R}_{BC1}$', ['roll', 'pitch', 'yaw', 'w'],
#      180 / math.pi, mathUtils.dcm2quat(ref_T_SC1[:3, :3]), r'$^\circ$', 3, 3,
#      mathUtils.dcm2quat(dataParams.TUMVI_NominalS_R_C0), 2.2),
#     ('fc1', '', [r'$f^1$', r'$c_x^1$', r'$c_y^1$'], 1, ReferenceParameters['fc1'], r'$px$', 2, 3,
#      np.array([190, 256, 256]), 0.5),
#     ('distort1', '', [r'$k_1^1$', r'$k_2^1$', r'$k_3^1$', r'$k_4^1$'], 1000, ReferenceParameters['distort1'], '0.001', 2,
#     4, np.array([0, 0, 0, 0]), 0.5),
#     ('tdtr1', '', [r'$t_d^1$', r'$t_r^1$'], 1000, ReferenceParameters['tdtr1'], r'$ms$', 2, 2, np.array([0, 0.02]), 0.5)]

# Another output format for monocular setups on smartphones
# ref_T_SC0 = np.array([[0, -1, 0, 0],
#                       [-1, 0, 0, 0],
#                       [0, 0, -1, 0],
#                       [0.0, 0.0, 0.0, 1.0]])

# SegmentList = [
#     ('bg', r'$\mathbf{b}_g$', [r'$x$', r'$y$', r'$z$'], 180 / math.pi, np.array([0.0, 0.0, 0.0]), r'$^\circ/s$', 3, 3,
#      np.array([0.0, 0.0, 0.0]), 1.0),
#     ('ba', r'$\mathbf{b}_a$', [r'$x$', r'$y$', r'$z$'], 1.0, np.array([0.0, 0.0, 0.0]), r'$m/s^2$', 3, 3,
#      np.array([0.0, 0.0, 0.0]), 1.6),
#     ('Tg', r'$\mathbf{T}_g$',
#      [r'$1,1$', r'$1,2$', r'$1,3$', r'$2,1$', r'$2,2$', r'$2,3$', r'$3,1$', r'$3,2$', r'$3,3$'],
#      1000.0, np.array([1.0, 0, 0, 0, 1, 0, 0, 0, 1]), '0.001', 2, 9, np.array([1.0, 0, 0, 0, 1, 0, 0, 0, 1]), 0.1),
#     ('Ts', r'$\mathbf{T}_s$',
#      [r'$1,1$', r'$1,2$', r'$1,3$', r'$2,1$', r'$2,2$', r'$2,3$', r'$3,1$', r'$3,2$', r'$3,3$'],
#      1000.0, np.array([0.0, 0, 0, 0, 0, 0, 0, 0, 0]), r'$0.001 \frac{rad/s}{m/s^2}$', 2, 9,
#      np.array([0.0, 0, 0, 0, 0, 0, 0, 0, 0]), 2.0),
#     ('Ta', r'$\mathbf{T}_a$',
#      [r'$1,1$', r'$1,2$', r'$1,3$', r'$2,1$', r'$2,2$', r'$2,3$', r'$3,1$', r'$3,2$', r'$3,3$'],
#      1000.0, np.array([1.0, 0, 0, 0, 1, 0, 0, 0, 1]), '0.001', 2, 9, np.array([1.0, 0, 0, 0, 1, 0, 0, 0, 1]), 0.17),
#     ('p_C0B', '', [r'$\mathbf{p}_{C0B}(1)$', r'$\mathbf{p}_{C0B}(2)$', r'$\mathbf{p}_{C0B}$(3)'], 100,
#      np.array([0.0, 0.0, 0.0]), r'$cm$', 2, 3, np.array([0.0, 0.0, 0.0]), 0.5),
#     ('fc0', '', [r'$f^0$', r'$c_x^0$', r'$c_y^0$'], 1, np.array([980, 640, 360]), r'$px$', 2, 3, np.array([980, 640, 360]), 0.5),
#     ('distort0', '', [r'$k_1^0$', r'$k_2^0$', r'$k_3^0$', r'$k_4^0$'], 1000, np.array([0, 0, 0, 0]), '0.001', 2, 4,
#     np.array([0, 0, 0, 0]), 0.5),
#     ('tdtr0', '', [r'$t_d^0$', r'$t_r^0$'], 1000, np.array([0, 0]), r'$ms$', 2, 2, np.array([0, 0.0]), 0.5)]


def getSwiftVioEstimatesIndexRanges():
    index_ranges = list()
    start_index = SWIFTVIO_BG_INDEX
    for segment in SegmentList:
        index_ranges.append(range(start_index, start_index + len(segment[kcolumn_labels])))
        start_index = index_ranges[-1][-1] + 1
    return index_ranges


def getSwiftVioStdIndexRanges(std_start_index):
    std_index_ranges = list()
    std_padding = 3 + 3 + 3 # pose and velocity
    std_start_index = std_start_index + std_padding
    for segment in SegmentList:
        std_index_ranges.append(range(std_start_index, std_start_index + segment[kminimal_dim]))
        std_start_index = std_index_ranges[-1][-1] + 1
    return std_index_ranges


def convertToSwiftVioArray(paramDicts, index_ranges, std_index_ranges):
    data_array = np.zeros((len(paramDicts), std_index_ranges[-1][-1] + 1))
    for row, params in enumerate(paramDicts):
        for index, param_range in enumerate(index_ranges):
            data_array[row, param_range] = params[SegmentList[index][kcomponent_label]]
        for index, param_range in enumerate(std_index_ranges):
            data_array[row, param_range] = params['std_' + SegmentList[index][kcomponent_label]]
    return data_array


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract the state vectors at the end of each output file of KSWF, then"
                    "compute mean and std dev of estimated sensor parameters \nand compare"
                    " to the values provided by TUM VI.")
    parser.add_argument("vio_output_dir",
                        help="Output dir of Swift VIO in which swift_vio.csv will be searched for.")
    parser.add_argument("plot_dir", help="Output dir of plots.")

    parser.add_argument("--ov_output_dir",
                        help=("Output dir of Open VINS in which state_estimate.txt and "
                             "state_deviation.txt will be searched for."))
    parser.add_argument("--seconds_to_end", type=float, default=0,
                        help="compute estimates at the time of seconds to end.")

    args = parser.parse_args()
    sv_data_array, sv_estimate_file_list = loadSwiftVioResults(args.vio_output_dir, args.seconds_to_end)
    num_sv_rows = sv_data_array.shape[0]
    index_ranges = getSwiftVioEstimatesIndexRanges()
    std_index_ranges = getSwiftVioStdIndexRanges(index_ranges[-1][-1] + 1)

    numruns = 5
    if args.ov_output_dir:
        ov_estimates = OpenVinsIO.loadOpenVinsResults(args.ov_output_dir, args.seconds_to_end, numruns)
        ov_data_array = convertToSwiftVioArray(ov_estimates, index_ranges, std_index_ranges)

        data_array = np.vstack((sv_data_array, ov_data_array))
        ov_row_indices = range(num_sv_rows, data_array.shape[0])
    else:
        data_array = sv_data_array
        ov_row_indices = None

    # compute the difference between reference values and initial values
    referenceErrors = {}
    for index, segment in enumerate(SegmentList):
        refvalues = copy.deepcopy(segment[ktum_reference_value])
        initial_value = segment[kinitial_value]
        referrors = mathUtils.compute_deviation_from_initial_value(segment[kcomponent_label], refvalues, initial_value)
        if isinstance(segment[kscale], list):
            for coeff_index, coeff in enumerate(segment[kscale]):
                referrors[coeff_index] *= coeff
        else:
            referrors *= segment[kscale]

        print('ref {}:{}'.format(segment[kcomponent_label], segment[kerror_unit]))
        format_str = '{:.' + str(segment[ksignificant_digits]) + 'f}'
        for j in range(len(referrors)):
            print(format_str.format(referrors[j]))
        referenceErrors[segment[kcomponent_label]] = referrors

    # compute the differences between estimated values and initial values, and scale the std.
    estimate_errors = copy.deepcopy(data_array)
    for index, segment in enumerate(SegmentList):
        initial_value = segment[kinitial_value]
        referrors = referenceErrors[segment[kcomponent_label]]
        error_indices = index_ranges[index][0:segment[kminimal_dim]]
        for j in range(estimate_errors.shape[0]):
            estimate_errors[j, error_indices] = mathUtils.compute_deviation_from_initial_value(
                segment[kcomponent_label], data_array[j, index_ranges[index]], initial_value)

        if isinstance(segment[kscale], list):
            for coeff_index, coeff in enumerate(segment[kscale]):
                estimate_errors[:, error_indices[coeff_index]] *= coeff
                estimate_errors[:, std_index_ranges[index][coeff_index]] *= coeff
        else:
            estimate_errors[:, error_indices] *= segment[kscale]
            estimate_errors[:, std_index_ranges[index]] *= segment[kscale]

        mathUtils.compute_mean_and_std(segment[kcomponent_label], estimate_errors, error_indices, segment[kerror_unit],
                             segment[ksignificant_digits])

    chosen_indices = list(range(len(sv_estimate_file_list)))
    # chosen_indices = selected_runs_for_3sigma_bounds(sv_estimate_file_list)
    num_chosen_sv_rows = len(chosen_indices)
    group_ranges = [range(num_chosen_sv_rows)]

    if ov_row_indices:
        chosen_indices.extend(ov_row_indices)
        group_ranges.append(range(num_chosen_sv_rows, len(chosen_indices)))

    for index, segment in enumerate(SegmentList):
        initial_value = segment[kinitial_value]
        referrors = referenceErrors[segment[kcomponent_label]]
        error_indices = index_ranges[index][0:segment[kminimal_dim]]
        # boxplot state over different runs.
        out_file = args.plot_dir + '/' + segment[kcomponent_label] + FORMAT
        boxplot_block_and_save(estimate_errors[range(num_sv_rows), :], error_indices,
                               segment[kcolumn_labels][0:segment[kminimal_dim]],
                               out_file, referrors, segment[kcomponent_label],
                               segment[kerror_unit])

        # draw 3\sigma bounds for estimated parameters at the end of sessions and (reference values - initial values).
        out_file = args.plot_dir + '/std_' + segment[kcomponent_label] + FORMAT
        if index < 5: # only draw bias, scale, and misalignment for swift vio.
            barplot_block_and_save(estimate_errors[chosen_indices[:num_chosen_sv_rows], :], error_indices, std_index_ranges[index],
                                   segment[kcolumn_labels][0:segment[kminimal_dim]],
                                   out_file, referrors, segment[kcomponent_label], segment[kcomponent_label_code],
                                   segment[kerror_unit], segment[kaspect], [group_ranges[0]])
        else: # draw for both swift vio and openvins.
            barplot_block_and_save(estimate_errors[chosen_indices, :], error_indices, std_index_ranges[index],
                                   segment[kcolumn_labels][0:segment[kminimal_dim]],
                                   out_file, referrors, segment[kcomponent_label], segment[kcomponent_label_code],
                                   segment[kerror_unit], segment[kaspect], group_ranges)

    # draw p_C0B and p_BC1 together
    out_file = args.plot_dir + '/std_p_C0B_p_BC1' + FORMAT
    draw_barplot_together(5, 9, 3, 3, index_ranges, std_index_ranges, SegmentList,
                          estimate_errors[chosen_indices, :], out_file, 1.0, group_ranges)
    # draw fc0 fc1 together
    out_file = args.plot_dir + '/std_fc0_fc1' + FORMAT
    draw_barplot_together(6, 11, 3, 3, index_ranges, std_index_ranges, SegmentList,
                          estimate_errors[chosen_indices, :], out_file, 0.8, group_ranges)
    # draw distort0 and distort1 together
    out_file = args.plot_dir + '/std_dist0_dist1' + FORMAT
    draw_barplot_together(7, 12, 2, 2, index_ranges, std_index_ranges, SegmentList,
                          estimate_errors[chosen_indices, :], out_file, 0.55, group_ranges)
    # draw td tr together
    out_file = args.plot_dir + '/std_tdtr0_tdtr1' + FORMAT
    leg_labels = ["KSWF", "OpenVINS", "TUM VI Ref."]
    draw_barplot_together(8, 13, 2, 2, index_ranges, std_index_ranges, SegmentList,
                          estimate_errors[chosen_indices, :], out_file, 0.35, group_ranges, leg_labels)
