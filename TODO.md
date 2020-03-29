1, double checks : 
Is it necessary to update the camera intrinsic parameters and IMU-camera relative
position after state update? The cameraGeometry embedded in vioparameter structure
is used in ThreadedKFVio and framesynchronizer. If it is desired to use IMU data to help
feature matching, and to possibly improve feature tracking between frames,
these calibration parameters can be refreshed after filtering update.
But for feature matching, this may not contribute much.

c, imageDelay in config.yaml of okvis has the opposite sign of T_d in msckf2

2, assumptions: hybridFilter currently only support one camera + one IMU, because 
a, FrameSynchronizer::addNewFrame() called in ThreadedKFVio.cpp, line 309, may average timestamps for multiple frames in a multiframe.
b, places commented with //one camera assumption
hybridFilter assumes that t_d and t_r are constant variables at least in a short span of time,
therefore, for a world point observed in several frames, the t_d used in computing Jacobians for the image features are the same.

4. use akaze features instead of brisk for feature detection, description, and matching. But it may require opencv 3.0 and later. This may cause conflict with current opencv library used by msckf. 

8. There may be negative diagonal elements in the covariance matrix. Are these elements all are very small and such occurrences are very rare, e.g. for the beginning of a test? ---Solved by enforcing symmetry.

9. if too many observations exist in one image (>height/2), then it is desirable to create a lookup table of [p_WS, q_WS, v_WS] for every few rows. 

10. gridded feature matching like in ORB-SLAM, use a motion model or inertial data to predict motion of camera and constrain search cells. Then after 2d2d matching, use 5 point algorithm to filter outliers if necessary (see SOFT: Stereo odometry based on careful feature selection and tracking).

15. to test okvis_ros with proprietary data, check the following for best performance: 

keyframeInsertionOverlapThreshold_,
keyframeInsertionMatchingRatioThreshold_,
and the parameters used in setting file (*.yaml), esp, T_SC, IMU specs, timeLimit, camera_rate, imageDelay, detection threshold and octaves and maxNoKeypoints, and outputPath

In my opinion, timeLimit can be set roughly the reciprocal of camera_rate which determines the play speed in Player.cpp
set publishImuPropagatedState false to enable output keyframe states
you may want to decrease the detection threshold increase octaves and maxNoKeypoints to improve tracking continuity.
set output in the local hard drive rather than a USB drive may improve performance.
The IMU specs has a important role in convergence, esp sigma_g_c, you may want to enlarge (e.g. x2) the values from the spec sheet.

16. CAVEAT: don't put print functions in estimator::optimizationLoop which is already too strained

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
see also doSetup() in vioframematchingalgorithm. the pointHomog and anchorStateId of a mappoint and
the homogeneous point parameter block has two folds meanings.
For MSCKF2, pointHomog and parameter block stores position in the global frame, but for HybridFilter,
they stores position in the anchor camera frame. So need to double check their usage, although
positions are only relevant in frontend matching.

24. Do we fix p_BA_G and q_GA after a feature is initialized into the States?
If fixed, we need to modify slamFeatureJacobian to make it consistent,
if not fixed, we need to modify slamFeatureJacobian (primarily) and other places related to these two parameters(trivia).

25. compare the performance of msckf2 against Kalibr with multiple IMUs and multiple cameras for a publication.
Does Kalibr output the trajectory of the camera-IMU system? If so, this serves for better comparison.

28. Implement another version of MSCKF: Fix Tg Ts Ta p_b^c, fx fy cx cy k1 k2 p1 p2 td tr, only estimate
p_b^g, v_b^g, R_b^g, b_g, b_a. ---This can be achieved by set the noise of the fixed parameters to zero,
and set the noise of variable parameters nonzero.

29. implement observability constrained EKF for calibration, this is similar in spirit to first estimate
Jacobians in that it modifies the computed Jacobians only.

30. add the calibration parameters to the original OKVIS estimator, use bool flags to indicate
if some parameters are fixed. Take a look at OKVIS camera IMU extrinsics for how to fix parameters.

31. implement square root Kalman filter for better numerical stability, referring to square root sliding window filter of MARS lab.

32. referring to msckf_mono implemented by daniliidis group, 
a, use all feature tracks to update states at the last frame

6. how to deal with static motion, standstill like the beginning of MH_01_easy?
limit the number of states, and hence the size of the covariance matrix.
Also, what will triangulateAmapPoint return when given two identical observations?
It is not imperative to add point states into the state vector.

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

35. A mysterious issue
I0911 23:19:01.549983   955 HybridFrontend.cpp:128] Initialized!
QObject::~QObject: Timers cannot be stopped from another thread
terminate called after throwing an instance of 'boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::lock_error> >'
  what():  boost: mutex lock failed in pthread_mutex_lock: Invalid argument

The root cause for this issue has nothing to do with the report. It is caused by a seemingly innocuous exit(1) in the codebase.

36. In testHybridFilter, of DeadreckoningO some runs have zigzag ripples in the pose profile, 
some runs have results close to the pose profile of DeadreckoningM. 
Preliminary tests find that this issue with DeadreckoningO is not caused by applyMarginalizationStrategy.

38. develop compatibility with multiple IMU models

39. develop iSAM with monocular, and multiple camera e.g., binocular, or stereo frontend, refer to Kimera stereo VIO implementation, 
my previous vin-csfm repo that depends on gtsam, and cpi closed_form preintegration by Eckenhoff, and 
the structureless factor from Forster RSS2017.
Refer to OKVIS, rovio, svo2, VINS Fusion, and MSCKF_VIO for developing stereo frontend.

40. develop the invariant EKF, refer to Zhang Teng, and Ross Hartley thesis

41. use FREAK besides BRISK in the frontend for feature description.

42. Tune the feature descriptor extraction parameters, the number of keyframes for matching,
 the temporal window size, and the spatial window size.

43. Implement fixed lag smoother for mono and stereo data with or without IMU, refer to Kimera-VIO, [closed form preintegration](https://github.com/rpng/cpi), and 
examples [here](https://github.com/ganlumomo/VisualInertialOdometry)

44. The timing entries for waitForOptimization and waitForMatching are often unusually large.

