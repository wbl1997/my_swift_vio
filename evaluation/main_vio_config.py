#!/usr/bin/python

import dataset_vio_config

if __name__ == "__main__":
    args = dataset_vio_config.parse_args()
    dataset_vio_config.create_config_yaml(
        args.config_template, args.format,
        args.output_okvis_config,
        args.camera_config_yamls, args.imu_config_yaml,
        args.algo_code, args.rosbag, args.use_nominal_value, args.monocular_input)
