import sys

import numpy as np

from ruamel.yaml import YAML

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: {} camchain-imucam.yaml".format(sys.argv[0]))
        exit(-1)
    camimuchain=sys.argv[1]

    yaml = YAML()
    yaml.version = (1, 2)
    yaml.default_flow_style = None
    yaml.indent(mapping=4, sequence=6, offset=4)

    with open(camimuchain, 'r') as camera_config:
        camera_data = yaml.load(camera_config)

    for cameraid in range(0, 2):
        camera_name = 'cam%d' % cameraid
        T_cam_imu = np.array(camera_data[camera_name]['T_cam_imu'])
        T_imu_cam = np.linalg.inv(T_cam_imu)
        print("T_imu_cam{}:\n{}".format(cameraid, T_imu_cam))
