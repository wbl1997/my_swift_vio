#!/usr/bin/python

import kalibr_okvis_config

if __name__ == "__main__":
    args = kalibr_okvis_config.parse_args()
    kalibr_okvis_config.create_okvis_config_yaml(
        args.okvis_config_template, args.format,
        args.output_okvis_config,
        args.camera_config_yamls, args.imu_config_yaml,
        args.algo_code, args.rosbag)
