## Use freak feature descriptor instead of brisk for feature description, and matching.

## There may be negative diagonal elements in the covariance matrix. Are these elements all are very small and
such occurrences are very rare, e.g. for the beginning of a test?
This has been solved by enforcing symmetry.

## if too many observations exist in one image (>height/2), then it is desirable to create a lookup table of [p_WS, q_WS, v_WS] for every few rows. 

## to test okvis_ros with your own data, check the following for best performance: 

keyframeInsertionOverlapThreshold_,
keyframeInsertionMatchingRatioThreshold_,
and the parameters used in setting file (*.yaml), esp, T_SC, IMU specs, timeLimit, camera_rate, imageDelay, detection threshold and octaves and maxNoKeypoints, and outputPath

In my opinion, timeLimit can be set roughly the reciprocal of camera_rate which determines the play speed in Player.cpp
set publishImuPropagatedState false to enable output keyframe states
you may want to decrease the detection threshold increase octaves and maxNoKeypoints to improve tracking continuity.
set output in the local hard drive rather than a USB drive may improve performance.
The IMU specs has a important role in convergence, esp sigma_g_c, you may want to enlarge (e.g. x2) the values from the spec sheet.

## referring to msckf_mono implemented by daniliidis group, 
use all feature tracks to update states at the last frame

## Use the IMU model defined in Rheder ICRA 2016 extending Kalibr
### A. In HybridFilter
w_m = T_g * w_b + T_s * a_b + b_g + n_g
a_m = T_a * a_b + b_a + n_a
where body frame {b} is aligned with the camera frame by a nominal R_SC in orientation,
but its origin is at the accelerometer triad intersection,
T_g, T_s, and T_a are fully populated.

### B. The simplified model is used in Jung et al. Observability Analysis of IMU Intrinsic Parameters,
w_m = N_g * w_b + b_g + n_g
a_m = N_a * a_b + b_a + n_a
where both N_g and N_a are fully populated.
The inverse model is given by
w_calibrated = M_g * w_m + b_g + n_g
a_calibrated = M_a * a_m + b_a + n_a
where M_g = N_g^{-1} and M_a = N_a^{-1}.
This is done for easy observability analysis.

### C. In ICRA 16 Extending Kalibr, for the reference IMU in a bundle of IMUs, its scaled misalignment model is given by
w_m = M_g * C_gyro_i * w_b + M_ga * C_gyro_i * a_b + b_g + n_g
a_m = M_a * C_i_w * (a_w - g_w) + b_a + n_a
where the i frame is the nominal frame of the accelerometer triad, and has an origin at the accelerometer triad intersection;
the b frame is identical to the i frame of the reference IMU;
the gyro frame is the nominal frame of the gyro triad, and assuming to have the same origin as i;
M_g and M_ga are fully populated, and M_a is a lower triangular matrix.

### D. In TUM VI dataset 2018 paper, the IMU model is defined by
w_calibrated = M_g * w_m + b_g
a_calibrated = M_a * a_m + b_a
where M_a is a lower triangular matrix, and M_g is fully populated.

### E. In Schneider 2019 Observability-aware self-calibration of visual and inertial sensors, 
the body frame coincides with the gyro frame, the IMU model is given by
w_m = T_g * w_b + b_g + n_g
a_m = T_a * C_a_g * C_g_w * (a_w - g_w) + b_a + n_a
where a is the accelerometer frame, g is the gyro frame,
and T_g and T_a are upper triangular matrices.
This model is used for observability analysis and degenerate motion identification in Yang 2020 "Online IMU intrinsic calibration: Is it necessary?"


## Design the coding structure for the swift_vio methods adaptive to different camera and IMU models,
minor thing: absorb the camera model parameters like T_SC0, tdLatestEstimate, etc, of the HybridFilter to the camera_rig_,

Foundation level: IMU camera landmark speedBias parameter blocks, expressed essentially by Eigen::Vectors
used by Optimizer and Filter

Connection level: IMU Model interface, Camera model interface, interfacing between the
rundimentary parameter blocks, and the actual models. It contains the IMU/Camera model type info which can be set by users.
are exposed to user settings, employed by the Algorithm to manage sensor models

Functional class level: the actual IMU/Camera model, IMU bias model, IMU bias + TgTsTa model, IMU bias + Tg Ts SM model, Camera projection + distortion model
are exposed to user by documentation

Use a underlying trajectory parameterization to serve poses, and underlying variable parameterization to serve values, sort of like Kalibr
Create factors with Jacobians that can be used for either filtering or optimization

## A mysterious issue
I0911 23:19:01.549983   955 HybridFrontend.cpp:128] Initialized!
QObject::~QObject: Timers cannot be stopped from another thread
terminate called after throwing an instance of 'boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::lock_error> >'
  what():  boost: mutex lock failed in pthread_mutex_lock: Invalid argument

The root cause for this issue has nothing to do with the report. It is caused by a seemingly innocuous exit(1) in the codebase.


## Use FREAK besides BRISK in the frontend for feature description. And update the immature landmark with depth filtering.

## The timing entries for waitForOptimization and waitForMatching are often unusually large.

## OKVIS leniently add triangulated landmarks to the estimator.
Some landmarks with small disparity may cause rank deficiency in computing the pose covarinace with ceres solver.
OKVIS addObservation() adds observations to the landmarkMap and as residuals to the ceres problem at the same time.
A better approach is to separate adding observations to feature tracks and adding residuals.

## ThreadedKFVio mock test failed in MockVioBackendInterface even with the github okvis master branch.

## The deadreckoning in okvis and that in swift_vio give different results, see testHybridFilter DeadreckoningM and DeadreckoningO.
Answer: Disabling marginalization in OKVIS deadreckoning makes its result similar to swift_vio deadreckoning.

## Vector access over the boundary when running command:
/media/jhuai/Seagate/jhuai/temp/swift_vio_ws/devel/lib/swift_vio/swift_vio_node_synchronous
/home/jhuai/Desktop/temp/swift_vio/vio/laptop/MSCKF_aidp/laptop_MSCKF_aidp_MH_01/config_fpga_p2_euroc.yaml
/home/jhuai/Desktop/temp/swift_vio/vio/laptop/MSCKF_aidp/laptop_MSCKF_aidp_MH_01/LcdParams.yaml
--output_dir=/home/jhuai/Desktop/temp/swift_vio/vio/laptop/MSCKF_aidp/laptop_MSCKF_aidp_MH_01/MSCKF4
--skip_first_seconds=0 --max_inc_tol=10.0 --dump_output_option=3
--bagname=/media/jhuai/OldWin8OS/jhuai/data/euroc/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.bag
--vocabulary_path=/media/jhuai/docker/swift_vio_ws/src/swift_vio/evaluation/../vocabulary/ORBvoc.yml
--camera_topics="/cam0/image_raw,/cam1/image_raw" --imu_topic=/imu0 

The below are the logs:
I0506 00:09:33.438650 13717 LoopClosureDetector.cpp:520] P3P inliers 11 out of 16 ransac iteration 23 max iterations 100
I0506 00:09:33.438757 13717 LoopClosureDetector.cpp:441] LoopClosureDetector: LOOP CLOSURE detected between loop keyframe 5326 with dbow id 21 and query keyframe 73619 with dbow id 131
I0506 00:09:33.446796 13715 MSCKF.cpp:1184] MSCKF receives #loop frames 1 but has not implemented relocalization yet!
I0506 00:09:33.473873 13695 swift_vio_node_synchronous.cpp:311] Progress: 25%  
I0506 00:09:33.513044 13717 LoopClosureDetector.cpp:493] knnmatch 3d landmarks 31 to 2d keypoints 149 correspondences 16
I0506 00:09:33.513265 13717 LoopClosureDetector.cpp:520] P3P inliers 13 out of 16 ransac iteration 9 max iterations 100
I0506 00:09:33.513334 13717 LoopClosureDetector.cpp:441] LoopClosureDetector: LOOP CLOSURE detected between loop keyframe 6967 with dbow id 24 and query keyframe 74336 with dbow id 132
I0506 00:09:33.521447 13715 MSCKF.cpp:1184] MSCKF receives #loop frames 1 but has not implemented relocalization yet!
I0506 00:09:33.704480 13717 LoopClosureDetector.cpp:493] knnmatch 3d landmarks 32 to 2d keypoints 159 correspondences 10
terminate called after throwing an instance of 'std::out_of_range'
  what():  vector::_M_range_check: __n (which is 5) >= this->size() (which is 5)
Aborted (core dumped)

I believe this issue has been resolved but not rememebering what is the exact solution.


## The progress percentage in building swift_vio jumps back and forth with catkin build.
This effect is not observed with OKVIS_ROS.
An comparison of the CMakeLists.txt between swift_vio and okvis_ros does not reveal suspicious differences.

## HybridFilter jumps forth and then back at the beginning
To reproduce,
```
/media/jhuai/Seagate/jhuai/temp/swift_vio_ws/devel/lib/swift_vio/swift_vio_node_synchronous 
/home/jhuai/Desktop/temp/advio_noise0/vio/laptop/KSF_0.5_0.2/laptop_KSF_0.5_0.2_advio15/config_fpga_p2_euroc.yaml 
/home/jhuai/Desktop/temp/advio_noise0/vio/laptop/KSF_0.5_0.2/laptop_KSF_0.5_0.2_advio15/LcdParams.yaml 
--output_dir=/home/jhuai/Desktop/temp/advio_noise0/vio/laptop/KSF_0.5_0.2/laptop_KSF_0.5_0.2_advio15/MSCKF0 
--skip_first_seconds=0 --max_inc_tol=30.0 --dump_output_option=3 
--bagname=/media/jhuai/OldWin8OS/jhuai/data/advio/rosbag/iphone/advio-15.bag 
--vocabulary_path=/media/jhuai/docker/swift_vio_ws/src/swift_vio/vocabulary/ORBvoc.yml 
--camera_topics="/cam0/image_raw," --imu_topic=/imu0 --publish_via_ros=false
```
Also check the impact of initWithoutEnoughParallax.
Improve this with better initial motion check.

## *** Error in `/home/jhuai/Documents/docker/swift_vio_ws/devel/lib/swift_vio/swift_vio_node_synchronous': malloc(): smallbin double linked list corrupted: 0x0000000002028420 ***
======= Backtrace: =========
/lib/x86_64-linux-gnu/libc.so.6(+0x777e5)[0x7ff5ecf407e5]
/lib/x86_64-linux-gnu/libc.so.6(+0x82651)[0x7ff5ecf4b651]
/lib/x86_64-linux-gnu/libc.so.6(__libc_malloc+0x54)[0x7ff5ecf4d184]
/usr/lib/nvidia-396/tls/libnvidia-tls.so.396.37(+0x24c0)[0x7ff5d9a1d4c0]
======= Memory map: ========
00400000-017ab000 r-xp 00000000 08:06 1587543                            /home/jhuai/Documents/docker/swift_vio_ws/devel/lib/swift_vio/swift_vio_node_synchronous
019aa000-019d5000 r--p 013aa000 08:06 1587543                            /home/jhuai/Documents/docker/swift_vio_ws/devel/lib/swift_vio/swift_vio_node_synchronous
019d5000-019d6000 rw-p 013d5000 08:06 1587543                            /home/jhuai/Documents/docker/swift_vio_ws/devel/lib/swift_vio/swift_vio_node_synchronous
019d6000-019d8000 rw-p 00000000 00:00 0 
01f6a000-030f0000 rw-p 00000000 00:00 0                                  [heap]

"The library is telling you that the memory metadata is corrupt. That won't happen by mere memory leak, 
you had to write to invalid pointer. Either you wrote to index out of bounds or you wrote to pointer after it was freed."

This error occasionally occurred on Ubuntu 16 with Eigen 3.3.4 built from source 
when the pose estimates drifted so much that there was almost no tracked features.

The same core dumped error (seg fault or aborted) occurred twice at about 34% of the TUM VI room3 sequence.

The command to reproduce the problem:
/home/jhuai/Documents/docker/swift_vio_ws/devel/lib/swift_vio/swift_vio_node_synchronous /home/
jhuai/Desktop/tumvi_calib0/vio/laptop/KSF_n_calibrated/laptop_KSF_n_calibrated_room3/config_fpga_p2_euroc.yaml
/home/jhuai/Desktop/tumvi_calib0/vio/laptop/KSF_n_calibrated/laptop_KSF_n_calibrated_room3/LcdParams.yaml
--output_dir=/home/jhuai/Desktop/tumvi_calib0/vio/laptop/KSF_n_calibrated/laptop_KSF_n_calibrated_room3/MSCKF
--skip_first_seconds=0 --max_inc_tol=30.0 --dump_output_option=3 --bagname=/media/jhuai/viola/jhuai/data/TUM-
VI/raw/room/dataset-room3_512_16.bag
--vocabulary_path=/home/jhuai/Documents/docker/swift_vio_ws/src/swift_vio/evaluation/../vocabulary/ORBvoc.yml
--camera_topics="/cam0/image_raw,/cam1/image_raw" --imu_topic=/imu0 --publish_via_ros=false

Or,
/home/jhuai/Documents/docker/swift_vio_ws/devel/lib/swift_vio/swift_vio_node_synchronous /home/
jhuai/Desktop/tumvi_calib0/vio/laptop/KSF_n_fix_TgTsTa/laptop_KSF_n_fix_TgTsTa_room3/config_fpga_p2_euroc.yaml
/home/jhuai/Desktop/tumvi_calib0/vio/laptop/KSF_n_fix_TgTsTa/laptop_KSF_n_fix_TgTsTa_room3/LcdParams.yaml
--output_dir=/home/jhuai/Desktop/tumvi_calib0/vio/laptop/KSF_n_fix_TgTsTa/laptop_KSF_n_fix_TgTsTa_room3/MSCKF0
--skip_first_seconds=0 --max_inc_tol=30.0 --dump_output_option=3 --bagname=/media/jhuai/viola/jhuai/data/TUM-
VI/raw/room/dataset-room3_512_16.bag
--vocabulary_path=/home/jhuai/Documents/docker/swift_vio_ws/src/swift_vio/evaluation/../vocabulary/ORBvoc.yml
--camera_topics="/cam0/image_raw,/cam1/image_raw" --imu_topic=/imu0 --publish_via_ros=false

Several errors possibly of the same origin can be found with the below command
```
RES_DIR=/keyframe_based_filter_2020/results
grep error $RES_DIR/tumvi-calibration/tumvi_calib0/log.txt -B 3 -A 10
```
The failure cases have the below configurations:
KSF_n_fix_TgTsTa_room3
KSF_n_fix_all_room1
KSF_n_fix_all_room3
KSF_n_loose_intrinsics_room3
KSF_n_loose_extrinsics_room3
They all fixed Tg Ts Ta and had a huge drift and possibly tracking failures.


## On EuRoC MH_04, RI-FLS throws gtsam::IndeterminantLinearSystemException working near a pose variable because of significant drift.

## Will marginalization cause the observable parameters, e.g., calibration parameters, to have wrongly optimistic covariance? 
I believe that marginalization will not cause overconfident covariance for these parameters.
To test, run HybridFilter or MSCKF in simulation to estimate camera extrinsic parameters, and compute their NEES.

