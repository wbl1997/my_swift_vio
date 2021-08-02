import csv
import math
import os
import sys

import numpy as np

import dataParams
import mathUtils

def loadcsv(dataFile):
    lineCount = 0
    columns = 0
    with open(dataFile, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            if lineCount == 0:
                vals = [float(x) for x in line.split()]
                columns = len(vals)
            lineCount += 1

    data = np.zeros((lineCount, columns))
    lineCount = 0
    with open(dataFile, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            vals = [float(x) for x in line.split()]
            data[lineCount, :] = vals
            lineCount += 1
    return data


OV_BG_RANGE = range(11, 14)
OV_BA_RANGE = range(14, 17)
OV_TD = 17
OV_FXY_CXY0_RANGE = range(19, 23)
OV_DISTORTION0_RANGE = range(23, 27)
OV_Q_BC0_RANGE = range(27, 31)
OV_P_BC0_RANGE = range(31, 34)
OV_FXY_CXY1_RANGE = range(34, 38)
OV_DISTORTION1_RANGE = range(38, 42)
OV_Q_BC1_RANGE = range(42, 46)
OV_P_BC1_RANGE = range(46, 49)

OV_STD_BG_RANGE = range(10, 13)
OV_STD_BA_RANGE = range(13, 16)
OV_STD_TD = 16
OV_STD_FXY_CXY0_RANGE = range(18, 22)
OV_STD_DISTORTION0_RANGE = range(22, 26)
OV_STD_Q_BC0_RANGE = range(26, 29)
OV_STD_P_BC0_RANGE = range(29, 32)
OV_STD_FXY_CXY1_RANGE = range(32, 36)
OV_STD_DISTORTION1_RANGE = range(36, 40)
OV_STD_Q_BC1_RANGE = range(40, 43)
OV_STD_P_BC1_RANGE = range(43, 46)


def convertOpenVinsToSwiftVioParams(ov_state, ov_std):
    """openvins params:
    T_BsC0, T_BsC1, camrea intrinsics, and td
    Swift vio params
    p_C0Bn, T_BnC1, camera intrinsics, and td
    R_BnBs = R_BnC * R_CBs
    Bn is the nominal IMU sensor frame derived from camera frame.
    Bs is the IMU sensor frame.
    """
    kswf_params = {}
    R_SC0 = mathUtils.quat2dcm(ov_state[OV_Q_BC0_RANGE])
    p_SC0 = ov_state[OV_P_BC0_RANGE]
    R_SC1 = mathUtils.quat2dcm(ov_state[OV_Q_BC1_RANGE])
    p_SC1 = ov_state[OV_P_BC1_RANGE]

    kswf_params['bg'] = ov_state[OV_BG_RANGE]
    kswf_params['ba'] = ov_state[OV_BA_RANGE]
    kswf_params['Tg'] = np.identity(3).reshape(9)
    kswf_params['Ts'] = np.zeros((3, 3)).reshape(9)
    kswf_params['Ta'] = np.identity(3).reshape(9)
    kswf_params['p_BC0'] = p_SC0
    kswf_params['q_BC0'] = ov_state[OV_Q_BC0_RANGE]
    kswf_params['p_C0B'] = - np.matmul(R_SC0.T, p_SC0)
    S_R_NominalS = np.matmul(R_SC0, np.transpose(dataParams.TUMVI_NominalS_R_C0))
    kswf_params['p_BC1'] = np.matmul(S_R_NominalS.T, p_SC1)
    kswf_params['q_BC1'] = mathUtils.dcm2quat(np.matmul(S_R_NominalS.T, R_SC1[:3, :3]))
    kswf_params['fc0'] = ov_state[OV_FXY_CXY0_RANGE]
    kswf_params['distort0'] = ov_state[OV_DISTORTION0_RANGE]
    kswf_params['tdtr0'] = [ov_state[OV_TD], 0]
    kswf_params['fc1'] = ov_state[OV_FXY_CXY1_RANGE]
    kswf_params['distort1'] = ov_state[OV_DISTORTION1_RANGE]
    kswf_params['tdtr1'] = [0, 0]

    # convert stds
    # openvins keeps covariance for p_CiS and q_CiS, their errors are dp_CiS and theta_CiS
    # p_CiS = hat{p}_CiS + dp_CiS
    # q_CiS = q(theta_CiS) * hat{q}_CiS
    # In Swift Vio, the covariance is for p_C0B, p_BC1, q_BC1, their errors are dp_C0B, dp_BC1, theta_BC1,
    # q_BC1 = q(theta_BC1) * hat{q}_BC1.
    kswf_params['std_bg'] = ov_std[OV_STD_BG_RANGE]
    kswf_params['std_ba'] = ov_std[OV_STD_BA_RANGE]
    kswf_params['std_Tg'] = np.zeros((3, 3)).reshape(9)
    kswf_params['std_Ts'] = np.zeros((3, 3)).reshape(9)
    kswf_params['std_Ta'] = np.zeros((3, 3)).reshape(9)
    kswf_params['std_p_C0B'] = ov_std[OV_STD_P_BC0_RANGE]

    kswf_params['std_p_BC0'] = ov_std[OV_STD_P_BC0_RANGE]  # Hack: ov_std[OV_STD_P_BC0_RANGE] is actually STD_P_C0B.
    kswf_params['std_q_BC0'] = ov_std[OV_STD_Q_BC0_RANGE]  # Hack: ov_std[OV_STD_Q_BC0_RANGE] is actually STD_Q_C0B.

    kswf_params['std_fc0'] = ov_std[OV_STD_FXY_CXY0_RANGE]
    kswf_params['std_distort0'] = ov_std[OV_STD_DISTORTION0_RANGE]
    kswf_params['std_tdtr0'] = [ov_std[OV_STD_TD], 0]
    kswf_params['std_fc1'] = ov_std[OV_STD_FXY_CXY1_RANGE]
    kswf_params['std_distort1'] = ov_std[OV_STD_DISTORTION1_RANGE]
    kswf_params['std_tdtr1'] = [0, 0]

    # convert covariance for T_BC1, ignoring the cross terms.
    cov = np.identity(9) # covariance for the errors of X = [R_C0S, R_C1S, p_C1S], dX = [theta_C0S, theta_C1S, p_C1S]
    for i, index in enumerate(OV_STD_Q_BC0_RANGE):
        cov[i, i] = pow(ov_std[index], 2)
    for i, index in enumerate(OV_STD_Q_BC1_RANGE):
        cov[i + 3, i + 3] = pow(ov_std[index], 2)
    for i, index in enumerate(OV_STD_P_BC1_RANGE):
        cov[i + 6, i + 6] = pow(ov_std[index], 2)

    dT_SnC1_dX = np.zeros((6, 9))
    R_C0C1 = np.matmul(R_SC0.T, R_SC1)
    p_C1S = - np.matmul(R_SC1.T, p_SC1)
    dT_SnC1_dX[:3, :3] = np.matmul(dataParams.TUMVI_NominalS_R_C0, mathUtils.skew(np.matmul(R_C0C1, p_C1S)))
    dT_SnC1_dX[:3, 3:6] = - np.matmul(dataParams.TUMVI_NominalS_R_C0, np.matmul(R_C0C1, mathUtils.skew(p_C1S)))
    dT_SnC1_dX[:3, 6:] = - np.matmul(dataParams.TUMVI_NominalS_R_C0, R_C0C1)
    dT_SnC1_dX[3:, :3] = dataParams.TUMVI_NominalS_R_C0
    dT_SnC1_dX[3:, 3:6] = - np.matmul(dataParams.TUMVI_NominalS_R_C0, R_C0C1)
    cov_T_SnC1 = np.matmul(np.matmul(dT_SnC1_dX, cov), dT_SnC1_dX.T)

    kswf_params['std_p_BC1'] = np.sqrt(cov_T_SnC1.diagonal()[:3])
    kswf_params['std_q_BC1'] = np.sqrt(cov_T_SnC1.diagonal()[3:])
    print("std_p_C0B {}".format( kswf_params['std_p_C0B']))
    print("std_p_BC1 {} from in sheet {}".format(kswf_params['std_p_BC1'], ov_std[OV_STD_P_BC1_RANGE]))
    print("std_q_BC1 {} from in sheet {}".format(kswf_params['std_q_BC1'], ov_std[OV_STD_Q_BC1_RANGE]))

    # remove fy and its std
    for i in range(2):
        estimatekey = "fc{:d}".format(i)
        stdkey = 'std_' + estimatekey
        keys = [estimatekey, stdkey]
        for key in keys:
            kswf_params[key] = [(kswf_params[key][0] + kswf_params[key][1]) * 0.5,
                                kswf_params[key][2], kswf_params[key][3]]
    return kswf_params


def loadOpenVinsResults(output_dir, seconds_to_end, numruns):
    estimate_file_list = []
    for dir_name, subdir_list, file_list in os.walk(output_dir):
        for fname in file_list:
            if 'state_estimate.txt' in fname:
                estimate_file_list.append(os.path.join(dir_name, fname))
    estimate_file_list.sort()

    selected_files = []
    for i in range(0, len(estimate_file_list), numruns):
        selected_files.append(estimate_file_list[i])
    print('Chosen OpenVINS calibration results files:')
    for index, fn in enumerate(selected_files):
        print("{} {}".format(index, fn))

    paramsList = []
    rate = 0.0
    for stateFile in selected_files:
        stdFile = os.path.join(os.path.dirname(stateFile), 'state_deviation.txt')
        states = loadcsv(stateFile)
        stds = loadcsv(stdFile)
        if rate == 0.0:
            interval = np.mean(np.diff(states[:, 0]))
            rate = 1.0 / interval
            backrowindex = math.floor(seconds_to_end * rate + 1)
            print('OpenVINS state estimate backrow index {}'.format(backrowindex))
        state = states[-backrowindex, :]
        print('Selected OpenVINS state at {}'.format(state[0]))
        std = stds[-backrowindex, :]
        assert state[0] == std[0]
        params = convertOpenVinsToSwiftVioParams(state, std)
        paramsList.append(params)

    return paramsList
