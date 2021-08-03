import numpy as np


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

# copied from https://vision.in.tum.de/data/datasets/visual-inertial-dataset
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

# The camera extrinsics are copied from the okvis config file provided by TUMVI at
# https://vision.in.tum.de/_media/data/datasets/visual-inertial-dataset/config_okvis_50_20.yaml.tar.gz
# The camera intrinsic parameters are estimated by running basalt_calibrate on
# tumvi_calib_data/dataset-calib-cam3_512_16.bag.
TUMVI_PARAMETERS = {
    "cameras": [
        {"T_SC": [-0.99953071, 0.00744168, -0.02971511, 0.04536566,
                  0.0294408, -0.03459565, -0.99896766, -0.071996,
                  -0.00846201, -0.99937369, 0.03436032, -0.04478181,
                  0.0, 0.0, 0.0, 1.0],
         "image_dimension": [512, 512],
         "distortion_coefficients": [0.0071903212354232789, -0.004483597406537407, 0.0011164152345498162,
                                     -0.00042033545473632523],
         "distortion_type": "equidistant",
         "focal_length": [191.10360934193845, 191.08897924484246],
         "principal_point": [254.96090765301757, 256.8868959188778],
         "projection_opt_mode": "FXY_CXY",
         "extrinsic_opt_mode": "P_CB",
         "image_delay": 0.0,
         "image_readout_time": 0.00},
        {"T_SC":
             [-0.99951678, 0.00803569, -0.03002713, -0.05566603,
              0.03012473, 0.01231336, -0.9994703, -0.07010225,
              -0.0076617, -0.9998919, -0.01254948, -0.0475471,
              0., 0., 0., 1.,],
         "image_dimension": [512, 512],
         "distortion_coefficients": [0.0076099391727948409, -0.004231474520440184, 0.0010904030371996857,
                                     -0.00043379644513004217],
         "distortion_type": "equidistant",
         "focal_length": [190.4520077890315, 190.4187691410897],
         "principal_point": [252.56390242046556, 255.0272611597151],
         "projection_opt_mode": "FXY_CXY",
         "extrinsic_opt_mode": "P_BC_Q_BC",
         "image_delay": 0.0,
         "image_readout_time": 0.00}],
    "imu_params": {
        'imu_rate': 200,
        'g_max': 7.8,
        'sigma_g_c': 0.004,
        'sigma_a_c': 0.07,
        'sigma_gw_c': 4.4e-5,
        'sigma_aw_c': 1.72e-3,
        'g': 9.80766,
        'sigma_TGElement': 5e-3,
        'sigma_TSElement': 1e-3,
        'sigma_TAElement': 5e-3, },
    'ceres_options': {
        'timeLimit': 1,  # in units of seconds, -1 means no limit.
    },
    "displayImages": "false",
    "publishing_options": {
        'publishLandmarks': "false", }
}

TUMVI_NOMINAL_PARAMETERS = {
    "cameras": [
        {"T_SC": [-1, 0.0, 0.0, 0.0,
                  0.0, 0.0, -1.0, 0.0,
                  0.0, -1.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 1.0],
         "image_dimension": [512, 512],
         "distortion_coefficients": [0.0, 0.0, 0.0, 0.0],
         "distortion_type": "equidistant",
         "focal_length": [190.0, 190.0],
         "principal_point": [256.0, 256.0],
         "projection_opt_mode": "FX_CXY",
         "extrinsic_opt_mode": "P_CB",
         "image_delay": 0.0,
         "image_readout_time": 0.00},
        {"T_SC":
             [-1.0, 0.0, 0.0, 0.0,
              0.0, 0.0, -1.0, 0.0,
              0.0, -1.0, 0.0, 0.0,
              0., 0., 0., 1.],
         "image_dimension": [512, 512],
         "distortion_coefficients": [0.0, 0.0, 0.0, 0.0],
         "distortion_type": "equidistant",
         "focal_length": [190.0, 190.0],
         "principal_point": [256.0, 256.0],
         "projection_opt_mode": "FX_CXY",
         "extrinsic_opt_mode": "P_BC_Q_BC",
         "image_delay": 0.0,
         "image_readout_time": 0.00
         }],
    "imu_params": {
        'imu_rate': 200,
        'g_max': 7.8,
        'sigma_g_c': 0.004,
        'sigma_a_c': 0.07,
        'sigma_gw_c': 4.4e-5,
        'sigma_aw_c': 1.72e-3,
        'g': 9.80766,
        'sigma_TGElement': 5e-3,
        'sigma_TSElement': 1e-3,
        'sigma_TAElement': 5e-3, },
    'ceres_options': {
        'timeLimit': 1,
    },
    "displayImages": "false",
    "publishing_options": {
        'publishLandmarks': "false", }
}

TUMVI_IMAGE_DELAY = 126788 * 1e-9  # nanoseconds

TUMVI_R_SC0 = np.array(TUMVI_PARAMETERS['cameras'][0]['T_SC']).reshape(4, 4)[:3, :3]

TUMVI_NominalS_R_C0 = np.array(TUMVI_NOMINAL_PARAMETERS['cameras'][0]['T_SC']).reshape(4, 4)[:3, :3]
