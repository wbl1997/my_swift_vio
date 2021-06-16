

ROS_TOPICS = {"euroc": ["/cam0/image_raw", "/cam1/image_raw", "/imu0"],
              "tum_vi": ["/cam0/image_raw", "/cam1/image_raw", "/imu0"],
              "tum_rs": ["/cam0/image_raw", "/cam1/image_raw", "/imu0"],
              "uzh_fpv": ["/snappy_cam/stereo_l", "/snappy_cam/stereo_r", "/snappy_imu"],
              "advio": ["/cam0/image_raw", "", "/imu0"]}

# actual bag filename : data name used in creating result dirs, data label used in plots.
# Data labels are not supposed to include underscores which causes latex interpretation error,
#  but can include hyphens.
BAGNAME_DATANAME_LABEL = {
    'MH_01_easy': ('MH_01', 'M1'),
    'MH_02_easy': ('MH_02', 'M2'),
    'MH_03_medium': ('MH_03', 'M3'),
    'MH_04_difficult': ('MH_04', 'M4'),
    'MH_05_difficult': ('MH_05', 'M5'),
    'V1_01_easy': ('V1_01', 'V11'),
    'V1_02_medium': ('V1_02', 'V12'),
    'V1_03_difficult': ('V1_03', 'V13'),
    'V2_01_easy': ('V2_01', 'V21'),
    'V2_02_medium': ('V2_02', 'V22'),
    'V2_03_difficult': ('V2_03', 'V23'),

    'indoor_45_2_snapdragon_with_gt': ('in_45_2', 'i2'),
    'indoor_45_4_snapdragon_with_gt': ('in_45_4', 'i4'),
    'indoor_45_9_snapdragon_with_gt': ('in_45_9', 'i9'),
    'indoor_45_12_snapdragon_with_gt': ('in_45_12', 'i12'),
    'indoor_45_13_snapdragon_with_gt': ('in_45_13', 'i13'),
    'indoor_45_14_snapdragon_with_gt': ('in_45_14', 'i14'),
    'outdoor_45_1_snapdragon_with_gt': ('out_45_1', 'o1'),
    'indoor_forward_3_snapdragon_with_gt': ('in_fwd_3', 'if3'),
    'indoor_forward_5_snapdragon_with_gt': ('in_fwd_5', 'if5'),
    'indoor_forward_6_snapdragon_with_gt': ('in_fwd_6', 'if6'),
    'indoor_forward_7_snapdragon_with_gt': ('in_fwd_7', 'if7'),
    'indoor_forward_9_snapdragon_with_gt': ('in_fwd_9', 'if9'),
    'indoor_forward_10_snapdragon_with_gt': ('in_fwd_10', 'if10'),
    'outdoor_forward_1_snapdragon_with_gt': ('out_fwd_1', 'of1'),
    'outdoor_forward_3_snapdragon_with_gt': ('out_fwd_3', 'of3'),
    'outdoor_forward_5_snapdragon_with_gt': ('out_fwd_5', 'of5'),
    'indoor_forward_11_snapdragon': ('in_fwd_11', 'if11'),
    'indoor_forward_12_snapdragon': ('in_fwd_12', 'if12'),
    'indoor_45_3_snapdragon': ('in_45_3', 'i3'),
    'indoor_45_16_snapdragon': ('in_45_16', 'i16'),
    'outdoor_forward_9_snapdragon': ('out_fwd_9', 'of9'),
    'outdoor_forward_10_snapdragon': ('out_fwd_10', 'of10'),

    'dataset-corridor1_512_16': ('corridor1', 'co1'),
    'dataset-corridor2_512_16': ('corridor2', 'co2'),
    'dataset-corridor3_512_16': ('corridor3', 'co3'),
    'dataset-corridor4_512_16': ('corridor4', 'co4'),
    'dataset-corridor5_512_16': ('corridor5', 'co5'),
    'dataset-magistrale1_512_16': ('magistrale1', 'ma1'),
    'dataset-magistrale2_512_16': ('magistrale2', 'ma2'),
    'dataset-magistrale3_512_16': ('magistrale3', 'ma3'),
    'dataset-magistrale4_512_16': ('magistrale4', 'ma4'),
    'dataset-magistrale5_512_16': ('magistrale5', 'ma5'),
    'dataset-magistrale6_512_16': ('magistrale6', 'ma6'),
    'dataset-outdoors1_512_16': ('outdoors1', 'out1'),
    'dataset-outdoors2_512_16': ('outdoors2', 'out2'),
    'dataset-outdoors3_512_16': ('outdoors3', 'out3'),
    'dataset-outdoors4_512_16': ('outdoors4', 'out4'),
    'dataset-outdoors5_512_16': ('outdoors5', 'out5'),
    'dataset-outdoors6_512_16': ('outdoors6', 'out6'),
    'dataset-outdoors7_512_16': ('outdoors7', 'out7'),
    'dataset-outdoors8_512_16': ('outdoors8', 'out8'),
    'dataset-room1_512_16': ('room1', 'rm1'), # only room sessions of tum vi dataset have throughout ground truth.
    'dataset-room2_512_16': ('room2', 'rm2'),
    'dataset-room3_512_16': ('room3', 'rm3'),
    'dataset-room4_512_16': ('room4', 'rm4'),
    'dataset-room5_512_16': ('room5', 'rm5'),
    'dataset-room6_512_16': ('room6', 'rm6'),
    'dataset-slides1_512_16': ('slides1', 'sl1'),
    'dataset-slides2_512_16': ('slides2', 'sl2'),
    'dataset-slides3_512_16': ('slides3', 'sl3'),
    # among the datasets, 07 and 18 are captured through glass elevators,
    # 14 is captured through steel elevator thus having 100% outliers.
    'advio-01': ('advio01', 'ad1'),
    'advio-02': ('advio02', 'ad2'),
    'advio-03': ('advio03', 'ad3'),
    'advio-04': ('advio04', 'ad4'),
    'advio-05': ('advio05', 'ad5'),
    'advio-06': ('advio06', 'ad6'),
    'advio-07': ('advio07', 'ad7'),
    'advio-08': ('advio08', 'ad8'),
    'advio-09': ('advio09', 'ad9'),
    'advio-10': ('advio10', 'ad10'),
    'advio-11': ('advio11', 'ad11'),
    'advio-12': ('advio12', 'ad12'),
    'advio-13': ('advio13', 'ad13'),
    'advio-14': ('advio14', 'ad14'),
    'advio-15': ('advio15', 'ad15'),
    'advio-16': ('advio16', 'ad16'),
    'advio-17': ('advio17', 'ad17'),
    'advio-18': ('advio18', 'ad18'),
    'advio-19': ('advio19', 'ad19'),
    'advio-20': ('advio20', 'ad20'),
    'advio-21': ('advio21', 'ad21'),
    'advio-22': ('advio22', 'ad22'),
    'advio-23': ('advio23', 'ad23')
}


def dataset_code(bagname):
    if 'snapdragon' in bagname or 'davis' in bagname:
        return "uzh_fpv"
    if 'euroc' in bagname:
        return "euroc"
    if 'TUM-VI' in bagname or 'tum-vi' in bagname:
        return "tum_vi"
    if 'rollingshutter' in bagname:
        return "tum_rs"
    if "ADVIO" in bagname or "advio" in bagname:
        return "advio"
    else:
        return "homebrew"


def calibration_format(dataset_type):
    if dataset_type == "uzh_fpv":
        calib_format = "kalibr"
    else:
        calib_format = dataset_type
    return calib_format
