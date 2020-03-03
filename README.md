README                        {#mainpage}
======

Welcome to msckf/Hybrid Filter.

The msckf [1][2] has been fully implemented with the so-called first estimate Jacobian technique [2].
Its derivation can be found in [1] and [4].

The Hybrid filter [3] referring to a hybrid of MSCKF and EKF-SLAM is not yet fully implemented.
The filters are developed based on the OKVIS library.

[1] J. Huai, “Collaborative SLAM with crowdsourced data,” Ph.D., The Ohio State University, Columbus OH, 2017.

[2] M. Li, H. Yu, X. Zheng, and A. I. Mourikis, “High-fidelity sensor modeling and self-calibration in vision-aided inertial navigation,” in 2014 IEEE International Conference on Robotics and Automation (ICRA), Hong Kong, China, 2014, pp. 409–416.

[3] M. Li and A. I. Mourikis, “Optimization-based estimator design for vision-aided inertial navigation,” in Robotics: Science and Systems, 2013, pp. 241–248.

[4] M. Shelley, “Monocular Visual Inertial Odometry on a Mobile Device,” Master, Technical University of Munich, Germany, 2014.

Note that the codebase that you are provided here is free of charge and without 
any warranty. This is bleeding edge research software.

Also note that the quaternion standard has been adapted to match Eigen/ROS, 
thus some related mathematical description in [1,2] will not match the 
implementation here.

If you publish work that relates to this software, please cite at least [1].

## How do I get set up?

This is a catkin package that wraps the pure CMake project.

You will need to install the following dependencies,

* ROS (currently supported: hydro, jade, kinetic, and melodic). 
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

ros melodic will install the source files for the following three packages by default: googletest libgtest-dev google-mock.
The googletest package includes source for both googletest and googlemock.
*Do not manually cmake and install gtest libraries to /usr/lib. It is a bad practice*.

* vio_common

```
cd msckf_ws/src
git clone https://github.com/JzHuai0108/vio_common.git
```

then clone the repository from github into your catkin workspace (assuming msckf_ws):

```
cd msckf_ws/src
git clone --recursive https://JzHuai0108@bitbucket.org/JzHuai0108/msckf.git
```

## Build the project

```
cd msckf_ws/
if [[ "x$(nproc)" = "x1" ]] ; then export USE_PROC=1 ;
else export USE_PROC=$(($(nproc)/2)) ; 
fi

export ROS_VERSION=kinetic # melodic
catkin init
catkin config --merge-devel # Necessary for catkin_tools >= 0.4.
catkin config --extend /opt/ros/$ROS_VERSION
catkin config --cmake-args -DUSE_ROS=ON -DBUILD_TESTS=ON 
# -DEIGEN3_INCLUDE_DIR=$HOME/slam_devel/include/eigen3 -DEIGEN_INCLUDE_DIR=$HOME/slam_devel/include/eigen3
catkin build vio_common msckf -DUSE_ROS=ON -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release -j$USE_PROC 

```
Setting EIGEN_INCLUDE_DIR is necessary for Ubuntu 16.04 because
the system wide Eigen library (usually 3.2) does not 
meet the requirements of the ceres solver depended by this package.
Therefore, the eigen library 3.3.4 should be downloaded from 
[here](https://github.com/eigenteam/eigen-git-mirror/releases)
and installed in a local directory with the below commands.
In a low bandwidth environment, wget is significantly slower than 
downloading from the webpage with the browser.
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

TODO(jhuai): the progress percentage in building msckf jumps back and forth with catkin build.
This effect is not observed with OKVIS_ROS. 
An comparison of the CMakeLists.txt between msckf and okvis_ros does not reveal suspicious differences.

## Build and run tests
* To build all tests
```
catkin build msckf --catkin-make-args run_tests # or
catkin build --make-args tests -- msckf
```
* To run all tests,
```
catkin build msckf --catkin-make-args run_tests # or
rosrun msckf msckf_test
```

* To run selected tests, e.g.,
```
rosrun msckf msckf_test --gtest_filter="*Eigen*"
```

## Debug the project with QtCreator

Follow the below steps exactly, otherwise mysterious errors like missing generate_config file arise.

### 1. Build msckf with catkin

Build the project with instructions in Section Build the project. 
You may need to clean build and devel dirs under the workspace with

```
caktin clean -y
```

### 2. Open msckf with QtCreator

Open QtCreator, assuming msckf_ws is the workspace dir,

```
source /opt/ros/melodic/setup.bash # or .zsh depending on the terminal shell
/opt/Qt/Tools/QtCreator/bin/qtcreator
```

if you encounter the error in starting qtcreator 
"qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.",
open another terminal, try the below alternative approach.

```
source /opt/ros/kinetic/setup.zsh
/opt/Qt/Tools/QtCreator/bin/qtcreator
```

Then, open msckf_ws/src/msckf/CMakeLists.txt in QtCreator. 
For the first time, configure the DEFAULT output path for the project in QtCreator 
as msckf_ws/build/msckf.
For subsequent times, you may encounter a dialog warning that 
"the CMakeCache.txt file or the project configuration has changed",
and two options are given, "Overwrite Changes in CMake" and 
"Apply Changes to Project" with the former being the default one. 
To avoid building the project anew with every project opening later on,
the overwrite option is recommended.

QtCreator may not find cmake files for libraries like roscpp, 
set the path like /opt/ros/kinetic/share/roscpp/cmake. Doing similar changes for other not found ros libraries.

To enable debug mode, in the Build option panel, set CMake Build Type to Debug.

To start debugging, add commandline arguments in the Run option panel, then press the Run icon.

## Example running cases

### Parameter description

Parameters through command line
|  Command line arguments | Description | Default |
|---|---|---|
|  load_input_option |  0 subscribe to rostopics, 1 load video and IMU csv |  1 |
| dump_output_option | 0 publish to rostopics, 1 save nav states to csv, 3 save nav states and calibration parameters to csv. 0 and others do not work simultaneously | 3 |
| feature_tracking_method | 0 BRISK brute force in OKVIS with 3d2d RANSAC, 1 KLT back-to-back, 2 BRISK back-to-back | 0 |
| use_IEKF | true iterated EKF, false EKF. For filters only | false |
| max_inc_tol | the maximum infinity norm of the filter correction. 10.0 for outdoors, 2.0 for indoors, though its value should be insensitive | 2.0 |
| head_tail | Use the fixed head and receding tail observations or the entire feature track to compose two-view constraints in TF_VIO | false |
| image_noise_cov_multiplier | Scale up the image observation noise cov for TF_VIO. 9 is recommended for the two-view constraint scheme based on an entire feature track, 25 is recommended for the two-view head_tail constraint scheme | 9 |

Parameters through config yaml

Remark: set sigma_{param} to zero to disable estimating "param" in filtering methods.

| Configuration parameters | Description | Default |
|---|---|---|
| extrinsic_opt_mode | For MSCKF and TF_VIO. "P_CB", estimate camera extrinsic translation, used with IMU TgTsTa 27 dim model [2]; "P_BC_R_BC", estimate camera extrinsics, used with say IMU BgBa model; "", fixed camera extrinsics | "" |
| projection_opt_mode | For MSCKF and TF_VIO. "FXY_CXY", estimate fx fy cx cy; "FX_CXY", estimate fx=fy, cx, cy; "FX", estimate fx=fy; "", camera projection intrinsic parameters are fixed | "" |
| down_scale | divide up the configured image width, image height, fx, fy, cx, cy by this divisor | 1 |
| distortion_type | "radialtangential", "equidistant", "radialtangential8", "fov" | REQUIRED |
| camera_rate | For processing data loaded from a video and an IMU csv, it determines the play speed. For debug mode, half of the normal camera rate is recommended, e.g., 15 Hz | 30 for Release, 15 for Debug |
| image_readout_time | time to read out an entire frame, i.e., the rolling shutter skew. 0 for global shutter, ~0.030 for rolling shutter | 0 |
| sigma_absolute_translation | The standard deviation [m] of the camera extrinsics translation. With OKVIS, e.g. 1.0e-10 for online-calib; With MSCKF or TF_VIO, 5e-2 for online extrinsic translation calib, 0 to fix the translation. |  0 |
| sigma_absolute_orientation | The standard deviation [rad] of the camera extrinsics orientation. With OKVIS, e.g. 1.0e-3 for online-calib; With MSCKF or TF_VIO and extrinsic_opt_mode is "P_BC_R_BC", 5e-2 for online extrinsic orientation calib, 0 to fix the extrinsic orientation | 0 |
| sigma_c_relative_translation | For OKVIS only, the std. dev. [m] of the cam. extr. transl. change between frames, e.g. 1.0e-6 for adaptive online calib (not less for numerics) | 0 |
| sigma_c_relative_orientation | For OKVIS only, the std. dev. [rad] of the cam. extr. orient. change between frames, e.g. 1.0e-6 for adaptive online calib (not less for numerics) | 0 |
| timestamp_tolerance | stereo frame out-of-sync tolerance [s] | 0.2/camera_rate, e.g., 0.005 |
| sigma_focal_length | For MSCKF and TF_VIO only, set to say 5.0 to estimate focal lengths, set to 0 to fix them | 0.0 |
| sigma_principal_point | For MSCKF and TF_VIO only, set to say 5.0 to estimate cx, cy, set to 0 to fix them | 0.0 |
| sigma_distortion | For MSCKF and TF_VIO only, set to nonzeros to estimate distortion parameters, set to 0s to fix them. Adapt to distortion types, e.g., [0.01, 0.003, 0.0, 0.0] for "radialtangential", e.g., [0.01] for "fov" | [0.0] * k |
| imageDelay | Only used for loading data from a video and an IMU csv, image frame time in the video clock - imageDelay = frame time in the IMU clock | 0.0 |
| sigma_td | For MSCKF and TF_VIO only, set to say 5e-3 sec to estimate time delay between the camera and IMU, set to 0 to fix the delay as imageDelay | 0.0 |
| sigma_tr | For MSCKF and TF_VIO only, set to say 5e-3 sec to estimate the rolling shutter readout time, set to 0 to fix the readout time as image_readout_time | 0.0 |
| sigma_TGElement | For MSCKF and TF_VIO only, set to say 5e-3 to estimate the Tg matrix for gyros, set to 0 to fix the matrix as Identity | 0 |
| sigma_TSElement | For MSCKF and TF_VIO only, set to say 1e-3 to estimate the gravity sensitivity Ts matrix for gyros, set to 0 to fix the matrix as Zero | 0 |
| sigma_TAElement | For MSCKF and TF_VIO only, set to say 5e-3 to estimate the Ta matrix for accelerometers, set to 0 to fix the matrix as Identity | 0 |
| numKeyframes | For OKVIS, number of keyframes in optimisation window | 5 |
| numImuFrames | For OKVIS, number of average frames in optimisation window, 3 recommended; For MSCKF and TF_VIO, numKeyframes + numImuFrames is the maximum allowed cloned states in the entire state vector, 30 recommended for their sum. | 2 |

### Process measurements from a video and an IMU csv
```
msckf_ws/devel/lib/msckf/okvis_node $HOME/docker_documents/msckf_ws/src/msckf/config/config_parkinglot_jisun_s6.yaml 
 --output_dir=/media/$USER/Seagate/temp/parkinglot/
 --video_file="/media/$USER/Seagate/data/spin-lab/west_campus_parking_lot/Jisun/20151111_120342.mp4" 
 --imu_file="/media/$USER/Seagate/data/spin-lab/west_campus_parking_lot/Jisun/mystream_11_11_12_3_13.csv" 
 --start_index=18800 --finish_index=28900 --max_inc_tol=10.0
 --dump_output_option=0 --feature_tracking_method=0
```
The running program will exit once the sequence finishes.

### Process measurements from rostopics

1. Download a dataset of your choice from [here](
   http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets). 
   Assuming you downloaded MH_01_easy/. 
   You will find a corresponding calibration / estimator configuration in the 
   okvis/config folder.

2. Run the node

```
rosrun msckf okvis_node $HOME/docker_documents/msckf_ws/src/msckf/config/config_fpga_p2_euroc_dissertation.yaml 
 --dump_output_option=0 --load_input_option=0 --output_dir=$HOME/Desktop/temp 
 --feature_tracking_method=0

rosbag play --pause --start=5.0 --rate=1.0 /media/$USER/Seagate/$USER/data/euroc/MH_01_easy.bag /cam0/image_raw:=/camera0 /imu0:=/imu

rosrun rviz rviz -d $HOME/docker_documents/msckf_ws/src/msckf/config/rviz.rviz
```

In this case, the program will exit once the Ctrl+C is entered in the terminal that runs the msckf_node.
Note the program will not exit if Ctrl+C is entered in the terminal of roscore.

### Process the TUM VI dataset

```
$HOME/docker_documents/msckf_ws/src/msckf/config/config_tum_vi_50_20_msckf.yaml \
 --output_dir=$HOME/Seagate/data/TUM-VI/postprocessed/ \
 --max_inc_tol=10.0 --dump_output_option=0 \
 --feature_tracking_method=0 --load_input_option=0

```

## Outputs and frames

In terms of coordinate frames and notation, 

* W denotes the OKVIS World frame (z up), 
* C\_i denotes the i-th camera frame, 
* S denotes the IMU sensor frame,
* B denotes a (user-specified) body frame.

For MSCKF and TF_VIO, S and B are used interchangeably.

The output of the okvis library is the pose T\_WS as a position r\_WS and quaternion 
q\_WS, followed by the velocity in World frame v\_W and gyro biases (b_g) as well as 
accelerometer biases (b_a). See the example application to get an idea on how to
use the estimator and its outputs (callbacks returning states).

The okvis_node ROS application will publish a configurable state -- see just below.

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

## HEALTH WARNING: calibration

If you would like to run the software/library on your own hardware setup, be 
aware that good results (or results at all) may only be obtained with 
appropriate calibration of the 

* camera intrinsics,
* camera extrinsics (poses relative to the IMU), 
* knowledge about the IMU noise parameters,
* and ACCURATE TIME SYNCHRONISATION OF ALL SENSORS.

To perform a calibration yourself, we recommend the following:

* Get Kalibr by following the instructions here 
  https://github.com/ethz-asl/kalibr/wiki/installation . If you decide to build from 
	source and you run ROS kinetic checkout pull request 3:

    git fetch origin pull/3/head:request3
    git checkout request3

* Follow https://github.com/ethz-asl/kalibr/wiki/multiple-camera-calibration to 
  calibrate intrinsic and extrinsic parameters of the cameras. If you receive an 
  error message that the tool was unable to make an initial guess on focal 
  length, make sure that your recorded dataset contains frames that have the 
  whole calibration target in view.

* Follow https://github.com/ethz-asl/kalibr/wiki/camera-imu-calibration to get 
  estimates for the spatial parameters of the cameras with respect to the IMU.

* msckf/Hybrid filter does not support pure rotation, and cannot start from static mode, because it uses delayed triangulation. But it supports infinity points.

## Contribution guidelines

* Contact s.leutenegger@imperial.ac.uk to request access to the bitbucket 
  repository.

* Programming guidelines: please follow 
  https://github.com/ethz-asl/programming_guidelines/wiki/Cpp-Coding-Style-Guidelines .
	
* Writing tests: please write unit tests (gtest).

* Code review: please create a pull request for all changes proposed. The pull 
  request will be reviewed by an admin before merging.

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

cd msckf_ws/src/msckf
init_linter_git_hooks
```

## Support

The developpers will be happy to assist you or to consider bug reports / feature 
requests. But questions that can be answered reading this document will be 
ignored. Please contact s.leutenegger@imperial.ac.uk.
