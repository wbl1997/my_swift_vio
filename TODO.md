4. use akaze features instead of brisk for feature detection, description, and matching. But it may require opencv 3.0 and later.
This may cause conflict with current opencv library used by msckf.

8. There may be negative diagonal elements in the covariance matrix. Are these elements all are very small and
such occurrences are very rare, e.g. for the beginning of a test?
This has been solved by enforcing symmetry.

9. if too many observations exist in one image (>height/2), then it is desirable to create a lookup table of [p_WS, q_WS, v_WS] for every few rows. 

10. gridded feature matching like in ORB-SLAM, use a motion model or inertial data to predict motion of camera and constrain search cells.
Then after 2d2d matching, use 5 point algorithm to
filter outliers if necessary (see SOFT: Stereo odometry based on careful feature selection and tracking).

15. to test okvis_ros with proprietary data, check the following for best performance: 

keyframeInsertionOverlapThreshold_,
keyframeInsertionMatchingRatioThreshold_,
and the parameters used in setting file (*.yaml), esp, T_SC, IMU specs, timeLimit, camera_rate, imageDelay, detection threshold and octaves and maxNoKeypoints, and outputPath

In my opinion, timeLimit can be set roughly the reciprocal of camera_rate which determines the play speed in Player.cpp
set publishImuPropagatedState false to enable output keyframe states
you may want to decrease the detection threshold increase octaves and maxNoKeypoints to improve tracking continuity.
set output in the local hard drive rather than a USB drive may improve performance.
The IMU specs has a important role in convergence, esp sigma_g_c, you may want to enlarge (e.g. x2) the values from the spec sheet.

18. I believe anchored inverse depth parameterization can handle these special
motions just like deferred triangulation SLAM(DTSLAM) by Herrena 2014, and Jia
Chao online camera gyroscope calibration.
The latter two used epipolar constraints to handle pure rotation.

19. The current implementation of hybridfilter has passed all the assertions, and after commenting
the conditions for inserting point features, making it a msckf2 filter, and it works fine.
But the result of the hybridfilter drift a lot. There are points sliding through the screen during
tests, I believe it is caused by wrong data associations.

20. In frontend hybridFilter match3d2d, the homogeneous point coordinates are used and are in anchor frame,
but in the frontend matching algorithm the homogeneous coordinates in the global frame is required,
see also doSetup() in viokeyframewindowmatchingalgorithm. the pointHomog and anchorStateId of a mappoint and
the homogeneous point parameter block has two fold meanings.
For MSCKF2, pointHomog and parameter block stores position in the global frame, but for HybridFilter,
they stores position in the anchor camera frame. So need to double check their usage, although
positions are only relevant in frontend matching.

29. implement observability constrained EKF for calibration, this is similar in spirit to first estimate
Jacobians in that it modifies the computed Jacobians only.
Answer: This requires deriving the observability constraint from scratch.
And it involves complicated corrections to the orientation linearization point.

30. add the calibration parameters to the original OKVIS estimator, use bool flags to indicate
if some parameters are fixed. Take a look at OKVIS camera IMU extrinsics for how to fix parameters.

31. implement square root Kalman filter for better numerical stability, referring to square root sliding window filter of MARS lab.

32. referring to msckf_mono implemented by daniliidis group, 
a, use all feature tracks to update states at the last frame

33. Use the IMU model defined in Rheder ICRA 2016 extending Kalibr
% In msckf2 implementation
% w_m = T_g * w_b + T_s * a_b + b_w + n_w
% a_m = T_a * a_b + b_a + n_a
% where body frame {b} is aligned with the camera frame by a nominal R_SC in orientation,
% but its origin is at the accelerometer triad intersection,
% T_g, T_s, and T_a are fully populated

% In ICRA 16 Extending Kalibr, ignoring the lever arm between
% accelerometers
% w_m = T_g * w_b + T_s * a_b + b_w + n_w
% a_m = T_a * a_b + b_a + n_a
% where b is aligned with the x-axis of the accelerometer triad, and has an
% origin at the accelerometer triad intersection
% T_g, T_s is fully populated, T_a is a lower triangular matrix

34. design the coding structure for the okvis and msckf methods adaptive to different camera and IMU models,
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

35. Solved: A mysterious issue
I0911 23:19:01.549983   955 HybridFrontend.cpp:128] Initialized!
QObject::~QObject: Timers cannot be stopped from another thread
terminate called after throwing an instance of 'boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::lock_error> >'
  what():  boost: mutex lock failed in pthread_mutex_lock: Invalid argument

The root cause for this issue has nothing to do with the report. It is caused by a seemingly innocuous exit(1) in the codebase.

39. develop iSAM with monocular, and multiple camera e.g., binocular, or stereo frontend, refer to Kimera stereo VIO implementation, 
my previous vin-csfm repo that depends on gtsam, and cpi closed_form preintegration by Eckenhoff, and 
the structureless factor from Forster RSS2017.
Refer to OKVIS, rovio, svo2, VINS Fusion, and MSCKF_VIO for developing stereo frontend.

41. use FREAK besides BRISK in the frontend for feature description.

42. Tune the feature descriptor extraction parameters, the number of keyframes for matching,
 the temporal window size, and the spatial window size.

43. Implement fixed lag smoother for mono and stereo data with or without IMU, refer to Kimera-VIO, [closed form preintegration](https://github.com/rpng/cpi), and 
examples [here](https://github.com/ganlumomo/VisualInertialOdometry)

44. The timing entries for waitForOptimization and waitForMatching are often unusually large.

49. Write a global bundle adjustment module with existing factors. Remember removing outliers during optimization steps, see Maplab.

50. OKVIS leniently add triangulated landmarks to the estimator.
Some landmarks with small disparity may cause rank deficiency in computing the pose covarinace with ceres solver.
OKVIS addObservation() adds observations to the landmarkMap and as residuals to the ceres problem at the same time.
A better approach is to separate adding observations to feature tracks and adding residuals.

51. To improve OKVIS NEES consistency, we can test exact first estimate Jacobian which means to use first estimates in
computing Jacobians for variables of fixed linearization points.
Refer to Basalt and GTSAM for alternate margialization strategies.

52. ThreadedKFVio mock test failed in MockVioBackendInterface even with the github okvis master branch.

53. The deadreckoning in okvis and that in MSCKF give different results, see testHybridFilter DeadreckoningM and DeadreckoningO.
Answer: Disabling marginalization in OKVIS deadreckoning makes its result similar to MSCKF deadreckoning.

54. Vector access over the boundary when running command:
/media/jhuai/Seagate/jhuai/temp/msckf_ws_rel/devel/lib/msckf/okvis_node_synchronous
/home/jhuai/Desktop/temp/msckf_rel/vio/laptop/MSCKF_aidp/laptop_MSCKF_aidp_MH_01/config_fpga_p2_euroc.yaml
/home/jhuai/Desktop/temp/msckf_rel/vio/laptop/MSCKF_aidp/laptop_MSCKF_aidp_MH_01/LcdParams.yaml
--output_dir=/home/jhuai/Desktop/temp/msckf_rel/vio/laptop/MSCKF_aidp/laptop_MSCKF_aidp_MH_01/MSCKF4
--skip_first_seconds=0 --max_inc_tol=10.0 --dump_output_option=3
--bagname=/media/jhuai/OldWin8OS/jhuai/data/euroc/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.bag
--vocabulary_path=/media/jhuai/docker/msckf_ws/src/msckf/evaluation/../vocabulary/ORBvoc.yml
--camera_topics="/cam0/image_raw,/cam1/image_raw" --imu_topic=/imu0 

The below are the logs:
I0506 00:09:33.438650 13717 LoopClosureDetector.cpp:520] P3P inliers 11 out of 16 ransac iteration 23 max iterations 100
I0506 00:09:33.438757 13717 LoopClosureDetector.cpp:441] LoopClosureDetector: LOOP CLOSURE detected between loop keyframe 5326 with dbow id 21 and query keyframe 73619 with dbow id 131
I0506 00:09:33.446796 13715 MSCKF2.cpp:1184] MSCKF receives #loop frames 1 but has not implemented relocalization yet!
I0506 00:09:33.473873 13695 okvis_node_synchronous.cpp:311] Progress: 25%  
I0506 00:09:33.513044 13717 LoopClosureDetector.cpp:493] knnmatch 3d landmarks 31 to 2d keypoints 149 correspondences 16
I0506 00:09:33.513265 13717 LoopClosureDetector.cpp:520] P3P inliers 13 out of 16 ransac iteration 9 max iterations 100
I0506 00:09:33.513334 13717 LoopClosureDetector.cpp:441] LoopClosureDetector: LOOP CLOSURE detected between loop keyframe 6967 with dbow id 24 and query keyframe 74336 with dbow id 132
I0506 00:09:33.521447 13715 MSCKF2.cpp:1184] MSCKF receives #loop frames 1 but has not implemented relocalization yet!
I0506 00:09:33.704480 13717 LoopClosureDetector.cpp:493] knnmatch 3d landmarks 32 to 2d keypoints 159 correspondences 10
terminate called after throwing an instance of 'std::out_of_range'
  what():  vector::_M_range_check: __n (which is 5) >= this->size() (which is 5)
Aborted (core dumped)

I believe this issue has been resolved but not rememebering what is the exact solution.

55. MSCKF jumps forth and then back at the beginning
To reproduce,
```
/media/jhuai/Seagate/jhuai/temp/msckf_ws_rel/devel/lib/msckf/okvis_node_synchronous 
/home/jhuai/Desktop/temp/msckf_advio_noise0/vio/laptop/KSF_0.5_0.2/laptop_KSF_0.5_0.2_advio15/config_fpga_p2_euroc.yaml 
/home/jhuai/Desktop/temp/msckf_advio_noise0/vio/laptop/KSF_0.5_0.2/laptop_KSF_0.5_0.2_advio15/LcdParams.yaml 
--output_dir=/home/jhuai/Desktop/temp/msckf_advio_noise0/vio/laptop/KSF_0.5_0.2/laptop_KSF_0.5_0.2_advio15/MSCKF0 
--skip_first_seconds=0 --max_inc_tol=30.0 --dump_output_option=3 
--bagname=/media/jhuai/OldWin8OS/jhuai/data/advio/rosbag/iphone/advio-15.bag 
--vocabulary_path=/media/jhuai/docker/msckf_ws/src/msckf/vocabulary/ORBvoc.yml 
--camera_topics="/cam0/image_raw," --imu_topic=/imu0 --publish_via_ros=false
```
Also check the impact of initWithoutEnoughParallax.

56. *** Error in `/home/jhuai/Documents/docker/msckf_ws/devel/lib/msckf/okvis_node_synchronous': malloc(): smallbin double linked list corrupted: 0x0000000002028420 ***
======= Backtrace: =========
/lib/x86_64-linux-gnu/libc.so.6(+0x777e5)[0x7ff5ecf407e5]
/lib/x86_64-linux-gnu/libc.so.6(+0x82651)[0x7ff5ecf4b651]
/lib/x86_64-linux-gnu/libc.so.6(__libc_malloc+0x54)[0x7ff5ecf4d184]
/usr/lib/nvidia-396/tls/libnvidia-tls.so.396.37(+0x24c0)[0x7ff5d9a1d4c0]
======= Memory map: ========
00400000-017ab000 r-xp 00000000 08:06 1587543                            /home/jhuai/Documents/docker/msckf_ws/devel/lib/msckf/okvis_node_synchronous
019aa000-019d5000 r--p 013aa000 08:06 1587543                            /home/jhuai/Documents/docker/msckf_ws/devel/lib/msckf/okvis_node_synchronous
019d5000-019d6000 rw-p 013d5000 08:06 1587543                            /home/jhuai/Documents/docker/msckf_ws/devel/lib/msckf/okvis_node_synchronous
019d6000-019d8000 rw-p 00000000 00:00 0 
01f6a000-030f0000 rw-p 00000000 00:00 0                                  [heap]

"The library is telling you that the memory metadata is corrupt. That won't happen by mere memory leak, 
you had to write to invalid pointer. Either you wrote to index out of bounds or you wrote to pointer after it was freed."

This error occasionally occurred on Ubuntu 16 with Eigen 3.3.4 built from source 
when the pose estimates drifted so much that there was almost no tracked features.

The same core dumped error (seg fault or aborted) occurred twice at about 34% of the TUM VI room3 sequence.

The command to reproduce the problem:
/home/jhuai/Documents/docker/msckf_ws/devel/lib/msckf/okvis_node_synchronous /home/
jhuai/Desktop/msckf_tumvi_calib0/vio/laptop/KSF_n_calibrated/laptop_KSF_n_calibrated_room3/config_fpga_p2_euroc.yaml
/home/jhuai/Desktop/msckf_tumvi_calib0/vio/laptop/KSF_n_calibrated/laptop_KSF_n_calibrated_room3/LcdParams.yaml
--output_dir=/home/jhuai/Desktop/msckf_tumvi_calib0/vio/laptop/KSF_n_calibrated/laptop_KSF_n_calibrated_room3/MSCKF2
--skip_first_seconds=0 --max_inc_tol=30.0 --dump_output_option=3 --bagname=/media/jhuai/viola/jhuai/data/TUM-
VI/raw/room/dataset-room3_512_16.bag
--vocabulary_path=/home/jhuai/Documents/docker/msckf_ws/src/msckf/evaluation/../vocabulary/ORBvoc.yml
--camera_topics="/cam0/image_raw,/cam1/image_raw" --imu_topic=/imu0 --publish_via_ros=false

Or,
/home/jhuai/Documents/docker/msckf_ws/devel/lib/msckf/okvis_node_synchronous /home/
jhuai/Desktop/msckf_tumvi_calib0/vio/laptop/KSF_n_fix_TgTsTa/laptop_KSF_n_fix_TgTsTa_room3/config_fpga_p2_euroc.yaml
/home/jhuai/Desktop/msckf_tumvi_calib0/vio/laptop/KSF_n_fix_TgTsTa/laptop_KSF_n_fix_TgTsTa_room3/LcdParams.yaml
--output_dir=/home/jhuai/Desktop/msckf_tumvi_calib0/vio/laptop/KSF_n_fix_TgTsTa/laptop_KSF_n_fix_TgTsTa_room3/MSCKF0
--skip_first_seconds=0 --max_inc_tol=30.0 --dump_output_option=3 --bagname=/media/jhuai/viola/jhuai/data/TUM-
VI/raw/room/dataset-room3_512_16.bag
--vocabulary_path=/home/jhuai/Documents/docker/msckf_ws/src/msckf/evaluation/../vocabulary/ORBvoc.yml
--camera_topics="/cam0/image_raw,/cam1/image_raw" --imu_topic=/imu0 --publish_via_ros=false

Several errors possibly of the same origin can be found with the below command
```
RES_DIR=/keyframe_based_filter_2020/results
grep error $RES_DIR/tumvi-calibration/msckf_tumvi_calib0/log.txt -B 3 -A 10
```
The failure cases have the below configurations:
KSF_n_fix_TgTsTa_room3
KSF_n_fix_all_room1
KSF_n_fix_all_room3
KSF_n_loose_intrinsics_room3
KSF_n_loose_extrinsics_room3
They all fixed Tg Ts Ta and had a huge drift and possibly tracking failures.


57. On EuRoC MH_04, RI-FLS throws gtsam::IndeterminantLinearSystemException working near a pose variable because of significant drift.


58. Will marginalization cause the observable parameters, e.g., calibration parameters, to have wrongly optimistic covariance? 
I believe that marginalization will not cause overconfident covariance for these parameters.
To test, run MSCKF in simulation to estimate camera extrinsic parameters, and compute their NEES.



