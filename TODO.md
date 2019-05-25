1, double checks : 
Is it necessary to update the camera intrinsic parameters and IMU-camera relative position after state update? The cameraGeometry embedded in vioparameter structure is used in hybridVio and framesynchronizer. If it is desired to use IMU data to help feature matching, and to possibly improve feature tracking between frames, these calibration parameters can be refreshed after filtering update. But for feature matching, this may not contribute much.

b, use Akaze features for feature detection and matching
c, imageDelay in config.yaml of okvis has the opposite sign of T_d in msckf2
d, make sure temporal_imu_data_overlap is greater than the maximum of possible T_d + T_r
2, assumptions: hybridFilter currently only support one camera + one IMU, because 
a, FrameSynchronizer::addNewFrame() called in hybridVio.cpp, line 309, may average timestamps for multiple frames in a multiframe.
b, places commented with //one camera assumption
hybridFilter assumes that t_d and t_r are constant variables at least in a short span of time, therefore, for a world point observed in several frames, the t_d used in computing Jacobians for the image features are the same.

3. Done! is it necessary to addParameterBlock to map? e.g., mapPtr_->addParameterBlock(poseParameterBlock,ceres::Map::Pose6d)? Yes, these blocks stores the states values.

4. use akaze features instead of brisk for feature detection, description, and matching. But it may require opencv 3.0 and later. This may cause conflict with current opencv library used by msckf2. 

5. make sure the z axis of the world frame is pointing up.

6, how to deal with static motion, standstill like the beginning of MH_01_easy? limit the number of states, and hence the size of the covariance matrix. Also, what will triangulateAmapPoint return when given two identical observations? 
Have to add point states into the state vector. Otherwise have to assume that all the frames undergoes motion.

7, make a multiplier to Tg Ts Ta elements, like 1000 as done in Yuksel's matlab code, for numerical consistency

8, there may be negative diagonal elements in the covariance matrix. Are these elements all are very small and such occurrences are very rare, e.g. for th ebeginning of a test?

9, if too many observations exist in one image (>height/2), then it is desirable to create a lookup table of [p_WS, q_WS, v_WS] for every few rows. 

10. gridded feature matching like in ORB-SLAM, use a motion model or inertial data to predict motion of camera and constrain search cells. Then after 2d2d matching, use 5 point algorithm to filter outliers if necessary (see SOFT: Stereo odometry based on careful feature selection and tracking).

11. compare the performance of msckf2 and okvis estimator in the test cases.

12. DONE! When compiling two independent packages in catkin workspace, both depending on a copy of the OKVIS library, some strange errors may occur. E.g., OKVIS_ROS has errors in Frontend.cpp which says no match operator= (operands are boost::shared_ptr and std::shared_ptr), MSCKF2 has error like no rule to build target libbrisk.a. These errors are caused by installing a different okvis for each package. These issues are solved by building msckf2 on top of the okvis library used by okvis_ros. In particular, error "not found file <Eigen/Core>" is because the old FindEigen.cmake does not support eigen3 well. Error "no match operator= (operands are boost::shared_ptr and std::shared_ptr)" is because a wrong version of opengv is downloaded from github. CMake warnings about putting devel folder in the build folder is because cmake needs the argument for devel directory. Right now, following the okvis_ros/CMakeLists.txt to make its CMakeLists.txt, the msckf2 can work well. Only one strange thing in CMakeLists.txt is that okvis libraries in target_link_libraries, (e.g., okvis_cv) cannot be replaced by its full name (in this case libokvis_cv.a). Otherwise many undefined function errors occur. Still sometimes catkin_make succeeded building a package, but when it is opened in qtcreator, the building in qtcreator fails. In this case, keep both the project in qtcreator and a terminal open, in the terminal run catkin_make, and then in qtcreator, build. It works many times.

14 I don't see much benefit of mahalanobis distance to defend outliers vs thresholding with projection discrepancy.

15 to test okvis_ros with proprietary data, check the following for best performance: 

keyframeInsertionOverlapThreshold_,
keyframeInsertionMatchingRatioThreshold_,
and the parameters used in setting file (*.yaml), esp, T_SC, IMU specs, timeLimit, camera_rate, imageDelay, detection threshold and octaves and maxNoKeypoints, and outputPath
In my opinion, timeLimit can be set roughly the reciprocal of camera_rate which determines the play speed in Player.cpp
set publishImuPropagatedState false to enable output keyframe states
you may want to decrease the detection threshold increase octaves and maxNoKeypoints to improve tracking continuity.
set output in the local hard drive rather than a USB drive may improve performance.
The IMU specs has a important role in convergence, esp sigma_g_c, you may want to enlarge (e.g. x2) the values from the spec sheet.

16 CAVEAT: don't put print functions in estimator::optimizationLoop which is already too strained

17 compare the performance of the Kalman filter with one bundled KF update for all feature tracks, with sequential KF update for every feature track

18 test the algorithm in static mode, pure rotation, and all infinity points, in simulation or real data. Make sure triangulation based on these observations are properly represented by the inverse depth parameterization. I believe anchored inverse depth parameterization can handle these special motions just like deferred triangulation SLAM(DTSLAM) by Herrena 2014, and Jia Chao online camera gyroscope calibration. The latter two used epipolar constraints to handle pure rotation. But Jia Chao's approach does not take care of translation in calibration.

19 The current implementation of hybridfilter has passed all the assertions, and after commenting the conditions for inserting point features, making it a msckf2 filter, and it works fine. But the result of the hybridfilter drift a lot. There are points sliding through the screen during tests, I believe it is caused by wrong data associations.

20. In frontend hybridFilter match3d2d, the homogeneous point coordinates are used and are in anchor frame,
but in the frontend matching algorithm the homogeneous coordinates in the global frame is required, see also doSetup() in vioframematchingalgorithm. the pointHomog and anchorStateId of a mappoint and the homogeneous point parameter block has two folds meanings.
For MSCKF2, pointHomog and parameter block stores position in the global frame, but for HybridFilter, they stores position in the anchor camera frame. So need to double check their usage, although positions are only relevant in frontend matching.

21. use round KLT tracking to replace the frontend dataassociationandinitialization. KLT avoids 3D-2D matching. This may be neneficial in cases where positions are landmarks are not well initialized.

22. In the future, it may be a good idea to compare the performance of a, visual odometry, b, inertial odometry, c, visual-inertial odometry, and global BA, see how much contribution are made to the result by each sensor, and answer the question of whether inertial data indeed helped in improving accuracy. This can be verified with my previous work ORBSLAM-DWO.

23. implement batch processing with global nonlinear optimization using msckf2 observations and initializations, and assume time offset changes linearly.

24. Done! write error metric, e.g., RMSE and NEES for evaluating msckf2 and batch processing performance.

24. DO we fix p_BA_G and q_GA after a feature is initialized into the States? If fixed, we need to modify computeHxf to make it consistent, if not fixed, we need to modify computeHxf (primarily) and other places related to these two parameters(trivia).

25. compare the performance of msckf2 against Kalibr with multiple IMUs and multiple cameras for a publication. Does Kalibr output the trajectory of the camera-IMU system? If so, this serves for better comparison.

26. in testHybridFilter's simulation, the std for noises used in covariance propagation should be slightly larger than the std used in sampling noises, becuase the process model involves many approximations other than these noise terms.


27. use ORB feature matching in frontend, see if it improves compared to KLT feature tracking which causes much drift now.

28. Implement another two versions of MSCKF, (1) Fix Tg Ts Ta p_b^c, fx fy cx cy k1 k2 p1 p2 td tr, only estimate p_b^g, v_b^g, R_b^g, b_g, b_a

use some bool flags to tell if some parameters are fixed, or some parameters are equal, like fx = fy. Need to synthesize the patterns to organize related parameters, eg, use a model to interface with the camera intrinsics, and another to deal with the IMU intrinsics, and yet another for camera-IMU extrinsics. Refer to colmap for an implementation example where template are used to handle different camera models.

29. implement observability constrained EKF for calibration

30. add the calibration parameters to the original OKVIS estimator, use bool flags to indicate if some parameters are fixed. Take a look at OKVIS camera IMU extrinsics for how to fix parameters.


