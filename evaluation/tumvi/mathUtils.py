import copy
import math
import numpy as np


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


def compute_deviation_from_initial_value(component_label, est, initial):
    if component_label == 'q_BC1':
        return unskew(np.matmul(quat2dcm(initial), np.transpose(quat2dcm(est))) - np.eye(3))
    else:
        return est - initial
