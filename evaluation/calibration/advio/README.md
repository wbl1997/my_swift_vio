[ADVIO](https://github.com/AaltoVision/ADVIO)
# calibration
The calibration parameters are copied from 
[advio](https://github.com/AaltoVision/ADVIO/tree/master/calibration).

# Ground truth
E.g, advio/advio-01/ground-truth/pose.csv

## Pose T_W_S
W is a world frame.
S is the iPhone IMU sensor frame according to [eq. 1](https://arxiv.org/pdf/1807.09828.pdf).
## Format
t[sec], x, y, z, qw, qx, qy, qz
t usually starts from 0.

# Camera extrinsic and intrinsic parameters
The images are rotated 90 degree clockwise before they are saved to rosbags.
Then these rosbags are given to Kalibr to calibrate the camera intrinsic and extrinsic parameters.
The camera coordinate frame (C') after rotation relates to the conventional camera frame (C)
in that x_C' = - y_C, y_C' = x_C, z_C' = z_C.


The conventional coordinate frames of the camera (C) and the IMU (S) on a smartphone
has the below relation (see Fig 2.3 J. Huai thesis).
R_SC = [0, -1, 0;
        -1, 0, 0;
         0, 0, -1]
Compared to the conventional camera coordinate frame (C), the camera coordinate frame used in advio
relates to the IMU frame by
R_SC' = [1, 0, 0;
         0, -1, 0;
         0, 0, -1]

If you are using the conventional camera coordinate frame on a smartphone,
then caution should be exercised when interpreting the calibration results at
[here](https://github.com/AaltoVision/ADVIO/tree/master/calibration).
Specifically, you should modify the below entities,

swap(image width, image_height)
intrinsics(fx, fy, cx, cy), swap fx fy, swap cx cy
distortion(k1, k2, p1, p2), swap p1 p2
T_C'S ==> T_CS = [R_CC', t_CC'; 0, 1] * T_C'S

