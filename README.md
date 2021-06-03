README                        {#mainpage}
======

Welcome to SWIFT VIO, short for Sliding WIndow FilTers for Visual Inertial Odometry.
There are several sliding window filters and smoothers inside swift_vio for visual inertial odometry.
The underlying data structures are largely borrowed from OKVIS.

* HybridFilter implements a keyframe-based hybrid of MSCKF and EKF-SLAM.
* MSCKF implements a keyframe-based MSCKF.
* SlidingWindowSmoother implements a fixed-lag smoother based on gtsam (without using keyframes).
* Estimator is modified from the OKVIS sliding window smoother.

## Build dependencies

This is a catkin package that wraps the pure CMake project.
You will need to install the following dependencies,

* ROS (currently supported: kinetic, and melodic). 
Read the ROS installation [instructions](http://wiki.ros.org/melodic/Installation/Ubuntu).

* google-glog + gflags,

```
sudo apt-get install libgoogle-glog-dev
```

* The following should get installed through ROS anyway:

```
sudo apt-get install libatlas-base-dev libeigen3-dev libsuitesparse-dev 
sudo apt-get install libboost-dev libboost-filesystem-dev
```

* catkin tools
```
sudo apt-get install python-catkin-tools
```

* gtest (**No operation required**)

The ros melodic desktop distro will install the source files for the three packages by default: googletest libgtest-dev google-mock.
The googletest package includes source for both googletest and googlemock.
*You do not need to cmake and install gtest libraries to /usr/lib.*.

* vio_common

```
cd swift_vio_ws/src
git clone https://github.com/JzHuai0108/vio_common.git
```

then clone the repository from github into your catkin workspace (assuming swift_vio_ws):

```
cd swift_vio_ws/src
git clone --recursive https://jzhuai@bitbucket.org/jzhuai/swift_vio.git
```

* supplant Eigen on Ubuntu 16
The system wide Eigen library in Ubuntu 18 is OK for swift_vio.
However, in Ubuntu 16, the system wide Eigen library (usually of version 3.2) does not 
meet the requirements of ceres solver used by swift_vio.
Therefore, a newer Eigen library (newer than 3.3.4) should be downloaded from 
[here](https://github.com/eigenteam/eigen-git-mirror/releases)
and installed in a local directory say $HOME/slam_devel by the below commands.

```
mkdir -p $HOME/slam_src
cd $HOME/slam_src
wget https://github.com/eigenteam/eigen-git-mirror/archive/3.3.4.zip
unzip 3.3.4.zip
mv eigen-git-mirror-3.3.4 eigen-3.3.4
cd $HOME/slam_src/eigen-3.3.4
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX="$HOME/slam_devel"
make install
```

* gtsam (optional)

```
sudo apt-get install libtbb-dev
cd $HOME/Documents/slam_src
git clone https://github.com/borglab/gtsam.git --recursive
cd gtsam
git checkout 6c85850147751d45cf9c595f1a7e623d239305fc
# 342f30d148fae84c92ff71705c9e50e0a3683bda(previously tested commit)
mkdir build
cd build

# In Ubuntu 16, to circumvent the incompatible system-wide Eigen, passing the local Eigen by EIGEN_INCLUDE_DIR is needed.
# GTSAM can be installed locally, e.g., at $HOME/slam_devel, but 
# /usr/local is recommended as it has no issue when debugging swift_vio in QtCreator.

cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release \
  -DGTSAM_TANGENT_PREINTEGRATION=OFF -DGTSAM_POSE3_EXPMAP=ON -DGTSAM_ROT3_EXPMAP=ON \
  -DGTSAM_USE_SYSTEM_EIGEN=ON ..
# -DEIGEN3_INCLUDE_DIR=$HOME/slam_devel/include/eigen3 -DEIGEN_INCLUDE_DIR=$HOME/slam_devel/include/eigen3 # in Ubuntu 16

make -j $(nproc) check # (optional, runs unit tests)
make -j $(nproc) install
```

## Build the project

```
cd swift_vio_ws/

export ROS_VERSION=melodic # kinetic
catkin init
catkin config --merge-devel # Necessary for catkin_tools >= 0.4.
catkin config --extend /opt/ros/$ROS_VERSION
catkin config --cmake-args -DUSE_ROS=ON -DBUILD_TESTS=ON \
 -DGTSAM_DIR=/usr/local/lib/cmake/GTSAM
# -DEIGEN3_INCLUDE_DIR=$HOME/slam_devel/include/eigen3 -DEIGEN_INCLUDE_DIR=$HOME/slam_devel/include/eigen3 # in Ubuntu 16

catkin build vio_common swift_vio -DUSE_ROS=ON -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release -j4
# -DCMAKE_PREFIX_PATH=/opt/ros/$ROS_VERSION
# -DDO_TIMING=ON
# -DUSE_SANITIZER=Address

```

### 1. CMAKE_PREFIX_PATH
After opening and building the project with QtCreator (see Section Debug the Project with QtCreator),
the following warning and associated errors may come up when the project is built again by catkin in a terminal.

"WARNING: Your workspace is configured to explicitly extend a workspace which
yields a CMAKE_PREFIX_PATH which is different from the cached CMAKE_PREFIX_PATH
used last time this workspace was built."

To suppress these errors, CMAKE_PREFIX_PATH needs to be specified.

### 2. CATKIN_ENABLE_TESTING
BUILD_TESTS=ON tells the program to build tests which depends on gmock and gtest. 
If the program is built outside a catkin environment, then we will automatically download and build gmock and gtest.
Otherwise, if the program is built by catkin, the ros stack provides gmock and gtest. 
Additional build of gmock and gtest will cause the error 
"add_library cannot create imported target "gmock" because another target with the same name already exists."
To tell if we are in catkin, CATKIN_ENABLE_TESTING=ON can be used. 
But since this the default value in catkin, we do not need to specify it.

### 3. DO_TIMING
Add this cmake flag to enable timing statistics.

### 4. Error "ceres-solver/include/ceres/jet.h:887:8: error: ‘ScalarBinaryOpTraits’ is not a class template".
This error arises when the system wide Eigen, e.g., on Ubuntu 16, is incompatible with ceres solver 14.0 
which requires Eigen version >= 3.3.
You need to pass EIGEN_INCLUDE_DIR and EIGEN3_INCLUDE_DIR to this package and also eschew PCL 
which depends on system wide Eigen.
In the end, the workspace should be clear of traces of system wide Eigen. That is, 
no /usr/include/eigen3 should appear when searching in the workspace.

## Build and run tests

* To build all tests
```
catkin build swift_vio --catkin-make-args run_tests # or
catkin build --make-args tests -- swift_vio
```

* To run all tests,
```
catkin build swift_vio --catkin-make-args run_tests # or
rosrun swift_vio swift_vio_test
```

* To run selected tests, e.g.,
```
rosrun swift_vio swift_vio_test --gtest_filter="*Eigen*"

# test HybridFilter
rosrun swift_vio swift_vio_test --log_dir="/swift_vio_sim" \
 --gtest_filter="*HybridFilter.TrajectoryLabel*" --num_runs=10 --noisyInitialSensorParams=true \
 --sim_camera_time_offset_sec=0.5 --sim_frame_readout_time_sec=0.03 --sim_trajectory_label=WavyCircle \
 --zero_imu_intrinsic_param_noise=false --zero_camera_intrinsic_param_noise=false
```

* To test RPGO,
```
cd swift_vio_ws/build/swift_vio/Kimera-RPGO
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
make check
```

* To run integration tests,
Download the evaluation workhorse [rpg_trajectory_evaluation](https://github.com/uzh-rpg/rpg_trajectory_evaluation.git), 
and install its dependencies,
```
pip2 install --upgrade pyyaml
pip2 install numpy matplotlib colorama ruamel.yaml
```
Because python scripts under evaluation/ directory defaults to use python3, 
you also need to install the python3 counterparts,
```
pip3 install numpy matplotlib colorama pyyaml ruamel.yaml
```

For tests, download the EuRoC dataset and optionally the UZH-FPV dataset.
```
swift_vio/evaluation/smoke_test.py tests the program with one data session from EuRoC dataset.
swift_vio/evaluation/main_evaluation.py tests the program with multiple data session from EuRoC and UZH-FPV dataset.
```

## Debug the project with QtCreator

Follow the below steps exactly, otherwise mysterious errors like missing generate_config file arise.

### 1. Build swift_vio as described earlier.

### 2. Open swift_vio with QtCreator

Open QtCreator by

```
source /opt/ros/melodic/setup.bash # or .zsh
/opt/Qt/Tools/QtCreator/bin/qtcreator
```

Then, open swift_vio_ws/src/swift_vio/CMakeLists.txt in QtCreator. 

If a dialog warning "the CMakeCache.txt file or the project configuration has changed",
comes up with two buttons, "Overwrite Changes in CMake" and 
"Apply Changes to Project" with the former being the default one. 
To avoid adding CMAKE_PREFIX_PATH in future builds with catkin in a terminal,
the "Apply Changes to Project" option is recommended.

To enable building test targets inside QtCreator, you may need to turn on 
"CATKIN_ENABLE_TESTING" in the CMake section of Building Settings and 
select the *_test target in a newly added Build Step from the Build Steps section.
The default target "all" may not emcompass building some test targets.

To solve the error about loading shared libraries libmetis.so, 
add the lib path of gtsam to LD_LIBRARY_PATH as below inside the terminal or 
the Run Environment in QtCreator (Projects > Build and Run > Run > Run Environment).

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
```

## Example running cases

### Parameter description
The parameters are divided into two groups, those that depend on operating system and file system, 
and those that are intrinsic to the program.
Parameters belonging to the first group are typically passed through command line gflags.
Parameters of the second group are typically configured through yaml.

**List of parameters through command line**
|  Command line arguments | Description | Default |
|---|---|---|
|  load_input_option |  0 subscribe to rostopics, 1 load video and IMU csv |  1 |
| dump_output_option | 0 only publish to rostopics, 1 also save nav states to csv, 2, also save nav states and extrinsic parameters, 3 also save nav states and all calibration parameters to csv. | 3 |
| use_IEKF | true iterated EKF, false EKF. For filters only | false |
| max_inc_tol | the maximum infinity norm of the filter correction. 10.0 for outdoors, 2.0 for indoors, though its value should be insensitive | 2.0 |
| head_tail | Use the fixed head and receding tail observations or the entire feature track to compose two-view constraints in TFVIO | false |
| image_noise_cov_multiplier | Scale up the image observation noise cov for TFVIO. 9 is recommended for the two-view constraint scheme based on an entire feature track, 25 is recommended for the two-view head_tail constraint scheme | 9 |

**Parameters through config yaml**
Remark: set sigma_{param} to zero to disable estimating "param" in filtering methods.

| Configuration parameters | Description | Default |
|---|---|---|
| extrinsic_opt_mode | For HybridFilter and TFVIO. "P_CB", estimate camera extrinsic translation, used with IMU TgTsTa 27 dim model [2]; "P_BC_R_BC", estimate camera extrinsics, used with say IMU BgBa model; "", fixed camera extrinsics | "" |
| projection_opt_mode | For HybridFilter and TFVIO. "FXY_CXY", estimate fx fy cx cy; "FX_CXY", estimate fx=fy, cx, cy; "FX", estimate fx=fy; "", camera projection intrinsic parameters are fixed | "" |
| down_scale | divide up the configured image width, image height, fx, fy, cx, cy by this divisor | 1 |
| distortion_type | "radialtangential", "equidistant", "radialtangential8", "fov" | REQUIRED |
| camera_rate | For processing data loaded from a video and an IMU csv, it determines the play speed. For debug mode, half of the normal camera rate is recommended, e.g., 15 Hz | 30 for Release, 15 for Debug |
| image_readout_time | time to read out an entire frame, i.e., the rolling shutter skew. 0 for global shutter, ~0.030 for rolling shutter | 0 |
| sigma_absolute_translation | The standard deviation [m] of the camera extrinsics translation. With OKVIS, e.g. 1.0e-10 for online-calib; With HybridFilter or TFVIO, 5e-2 for online extrinsic translation calib, 0 to fix the translation. |  0 |
| sigma_absolute_orientation | The standard deviation [rad] of the camera extrinsics orientation. With OKVIS, e.g. 1.0e-3 for online-calib; With HybridFilter or TFVIO and extrinsic_opt_mode is "P_BC_R_BC", 5e-2 for online extrinsic orientation calib, 0 to fix the extrinsic orientation | 0 |
| sigma_c_relative_translation | For OKVIS only, the std. dev. [m] of the cam. extr. transl. change between frames, e.g. 1.0e-6 for adaptive online calib (not less for numerics) | 0 |
| sigma_c_relative_orientation | For OKVIS only, the std. dev. [rad] of the cam. extr. orient. change between frames, e.g. 1.0e-6 for adaptive online calib (not less for numerics) | 0 |
| timestamp_tolerance | stereo frame out-of-sync tolerance [s] | 0.2/camera_rate, e.g., 0.005 |
| sigma_focal_length | For HybridFilter and TFVIO only, set to say 5.0 to estimate focal lengths, set to 0 to fix them | 0.0 |
| sigma_principal_point | For HybridFilter and TFVIO only, set to say 5.0 to estimate cx, cy, set to 0 to fix them | 0.0 |
| sigma_distortion | For HybridFilter and TFVIO only, set to nonzeros to estimate distortion parameters, set to 0s to fix them. Adapt to distortion types, e.g., [0.01, 0.003, 0.0, 0.0] for "radialtangential", e.g., [0.01] for "fov" | [0.0] * k |
| imageDelay | Used e.g., when loading data from a video and an IMU csv, image frame time in the video clock - imageDelay = frame time in the IMU clock | 0.0 |
| sigma_td | For HybridFilter and TFVIO only, set to say 5e-3 sec to estimate time delay between the camera and IMU, set to 0 to fix the delay as imageDelay | 0.0 |
| sigma_tr | For HybridFilter and TFVIO only, set to say 5e-3 sec to estimate the rolling shutter readout time, set to 0 to fix the readout time as image_readout_time | 0.0 |
| sigma_TGElement | For HybridFilter and TFVIO only, set to say 5e-3 to estimate the Tg matrix for gyros, set to 0 to fix the matrix as Identity | 0 |
| sigma_TSElement | For HybridFilter and TFVIO only, set to say 1e-3 to estimate the gravity sensitivity Ts matrix for gyros, set to 0 to fix the matrix as Zero | 0 |
| sigma_TAElement | For HybridFilter and TFVIO only, set to say 5e-3 to estimate the Ta matrix for accelerometers, set to 0 to fix the matrix as Identity | 0 |
| numKeyframes | For OKVIS, number of keyframes in optimisation window | 5 |
| numImuFrames | For OKVIS, number of average frames in optimisation window, 3 recommended; For HybridFilter and TFVIO, numKeyframes + numImuFrames is the maximum allowed cloned states in the entire state vector, 30 recommended for their sum. | 2 |
| featureTrackingMethod | 0 BRISK brute force in OKVIS with 3d2d RANSAC, 1 KLT back-to-back, 2 BRISK back-to-back | 0 |

### Process measurements from rostopics

1. Download a dataset of your choice from [here](
   http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets). 
   Assuming you downloaded MH_01_easy/. 
   You will find a corresponding calibration / estimator configuration in the 
   okvis/config folder.

2. Process a rosbag.

```
# In terminal 1
roscore

# In terminal 2
export SWIFT_VIO_WS=/path/to/swift_vio_ws
rosrun rviz rviz -d $SWIFT_VIO_WS/src/swift_vio/config/rviz.rviz

# In terminal 3
export SWIFT_VIO_WS=/path/to/swift_vio_ws
source $SWIFT_VIO_WS/devel/setup.bash
rosrun swift_vio swift_vio_node $SWIFT_VIO_WS/src/swift_vio/config/config_fpga_p2_euroc.yaml \
 --dump_output_option=0 --load_input_option=0 --output_dir=$HOME/Desktop/temp

# In terminal 4
rosbag play --pause --start=5.0 --rate=1.0 /path/to/euroc/MH_01_easy.bag /cam0/image_raw:=/camera0 /imu0:=/imu

```

3. Process the zip synchronously.

```
SWIFT_VIO_WS=/path/to/swift_vio_ws
cd $SWIFT_VIO_WS
source devel/setup.bash
rosrun swift_vio swift_vio_node_synchronous $SWIFT_VIO_WS/src/swift_vio/config/config_fpga_p2_euroc.yaml \
 --bagname=/path/to/euroc/MH_01_easy.bag --camera_topics="/cam0/image_raw,/cam1/image_raw" --imu_topic="/imu0" \
 --dump_output_option=3 --output_dir=$HOME/Desktop/temp

rosrun rviz rviz -d $SWIFT_VIO_WS/src/swift_vio/config/rviz.rviz
```

## Outputs and frames

In terms of coordinate frames and notation, 

* W denotes the OKVIS World frame (z up), 
* C\_i denotes the i-th camera frame, 
* S denotes the IMU sensor frame,
* B denotes a (user-specified) body frame.

For swift_vio, S and B are often used interchangeably.

The output of the okvis library is the pose T\_WS as a position r\_WS and quaternion 
q\_WS, followed by the velocity in World frame v\_W and gyro biases (b_g) as well as 
accelerometer biases (b_a). See the example application to get an idea on how to
use the estimator and its outputs (callbacks returning states).

The swift_vio_node ROS application will publish a configurable state -- see just below.

## Configuration files

The okvis/config folder contains example configuration files. Please read the
documentation of the individual parameters in the yaml file carefully. 
You have various options to trade-off accuracy and computational expense as well 
as to enable online calibration. You also have various options concerning the
things that will get published -- in particular weather or not landmarks should
be published (may be important to turn off for on-bard operation). Moreover, you 
can specify how the body frame is specified (T_BS) or define a custom World frame.
In other words, the final pose published will be 
T\_Wc\_B = T\_Wc\_W * T\_WS * T\_BS^(-1) . You have the option to express the
velocity as well as the rotation rates in either B, S, or Wc. 

## Run with data captured by a smartphone

* camera intrinsics can be read from the data sheet.
* camera extrinsics relative to the IMU can be set to the conventional value.
* The IMU noise parameters can be set roughly.


## Static program analysis with linter
The below instructions installs linter which requres python2 following the
guide at [here](https://github.com/ethz-asl/linter/tree/master).

```
sudo pip2 install yapf requests pylint
sudo apt install clang-format-6.0
cd $SLAM_TOOL_PATH
git clone https://github.com/JzHuai0108/linter.git
cd linter
git fetch origin
git checkout -b feature/onespacecppcomment

echo ". $(realpath setup_linter.sh)" >> ~/.bashrc
bash

cd swift_vio_ws/src/swift_vio
init_linter_git_hooks
```
