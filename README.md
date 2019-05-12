README                        {#mainpage}
======

Welcome to MSCKF2/Hybrid Filter. 

This is the Jianzhu Huai's implementation of the [1] and [2] with detailed derivation in [3]. It is developed based on the OKVIS library.

[1] Li, M., Yu, H., Zheng, X., & Mourikis, A. I. (2014, May). High-fidelity sensor modeling and self-calibration in vision-aided inertial navigation. In 2014 IEEE International Conference on Robotics and Automation (ICRA) (pp. 409-416). IEEE.

[2] Li, M., & Mourikis, A. I. (2013, July). Optimization-based estimator design for vision-aided inertial navigation. In Proc. of the Robotics: Science and Systems Conference (pp. 241-248).

[3] Michael Andrew Shelley. Monocular Visual Inertial Odometry on a Mobile Device. Master thesis, Technical University of Munich, 2014.

Note that the codebase that you are provided here is free of charge and without 
any warranty. This is bleeding edge research software.

Also note that the quaternion standard has been adapted to match Eigen/ROS, 
thus some related mathematical description in [1,2] will not match the 
implementation here.

If you publish work that relates to this software, please cite at least [1].

## How do I get set up?

This is a catkin package that wraps the pure CMake project.

You will need to install the following dependencies,

* ROS (currently supported: hydro, indigo and jade). Read the instructions in 
  http://wiki.ros.org/indigo/Installation/Ubuntu. You will need the additional 
  package pcl-ros as (assuming indigo)

        sudo apt-get install ros-indigo-pcl-ros

* google-glog + gflags,

        sudo apt-get install libgoogle-glog-dev
   
* The following should get installed through ROS anyway:

        sudo apt-get install libatlas-base-dev libeigen3-dev libsuitesparse-dev 
        sudo apt-get install libopencv-dev libboost-dev libboost-filesystem-dev

then clone the repository from github into your catkin workspace:

    git clone --recursive https://github.com/JzHuai0108/msckf2.git

Also vio_common
```
cd workspace/src
git clone https://github.com/JzHuai0108/vio_common.git
# then open CMakeLists.txt of vio_common, on line 9, SET(USE_ROS True)
```

## Building the project

If you have installed okvis_ros in the catkin workspace, then you may need to disable that package by renaming its package.xml file in order to avoid confusing catkin.

From the catkin workspace root, type 
```
cd workspace/
catkin_make --pkg vio_common msckf2
```
You will find a demo application in devel/lib/msckf2/msckf2_node. It can process datasets in the ASL/ETH format.

If you are going to contribute to the project, please configure linter as follows.
The below instructions were tested on Ubuntu 16.04.
```
sudo pip2 install yapf requests
sudo apt install clang-format-6.0 pylint
# the below follows installation guide at https://github.com/ethz-asl/linter/tree/master
cd workspace/src/msckf2/tools/linter
echo ". $(realpath setup_linter.sh)" >> ~/.bashrc
bash

cd ../..
init_linter_git_hooks
```

## Debug the project with QtCreator

Follow the below steps exactly, otherwise mysterious errors like missing generate_config file.

Even following the below instructions, the qtcreator still will create binaries and
file structures different from that generated by running catkin_make.

### Build msckf2 with catkin_make

To prepare the workspace file structure, 
clean build and devel dirs under the workspace, 
then build the project
```
cd vins_ws/devel
rm -rf ./*
cd ../build
rm -rf ./*
catkin_make --pkg vio_common msckf2
```

### Build msckf2 with QtCreator

To begin with, open qtcreator, suppose vins_ws is the workspace dir,

```
source vins_ws/devel/setup.zsh
/opt/Qt/Tools/QtCreator/bin/qtcreator
```

if you encounter the error in starting qtcreator 
"qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.",
open another terminal, try the below alternative approach.

```
source /opt/ros/kinetic/setup.zsh
/opt/Qt/Tools/QtCreator/bin/qtcreator
```

Then, open vins_ws/src/msckf2/CMakeLists.txt in QtCreator,
uncomment SET(CATKIN_DEVEL_PREFIX /persist/vins_ws/devel) in CMakeLists.txt

For the first time, configure the DEFAULT output path for the project in QtCreator as vins_ws/build/msckf2. 
QtCreator may not find cmake files for libraries like roscpp, 
set the path like /opt/ros/kinetic/share/roscpp/cmake. Doing similar changes for other not found ros libraries.

To enable debug mode, in the Build option panel, set CMake Build Type to Debug

Build msckf2 in QtCreator

To start debugging, add commandline arguments in the Run option panel, then press the Run icon

Example running cases

```
rosrun msckf2 msckf2_node /home/jhuai/docker_documents/vins_ws/src/msckf2/config/config_parkinglot_jisun_s6.yaml
```

```
rosrun msckf2 msckf2_node /home/jhuai/docker_documents/vins_ws/src/msckf2/config/config_fpga_p2_euroc_dissertation.yaml --load_input_option=0
# use start to skip the static segment
rosbag play --pause --start=45.0 --rate=1.0 /media/jhuai/Seagate1/data/euroc/MH_01_easy.bag /cam0/image_raw:=/camera0 /imu0:=/imu

rosrun rviz rviz -d /home/jhuai/docker_documents/vins_ws/src/msckf2/config/rviz.rviz
```

In order to run a minimal working example, follow the steps below:

1. Download a dataset of your choice from 
   http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets. 
   Assuming you downloaded MH_01_easy/. 
   You will find a corresponding calibration / estimator configuration in the 
   okvis/config folder.

2. Run the app as

        ./okvis_apps path/to/okvis_ros/okvis/config/config_fpga_p2_euroc.yaml path/to/MH_01_easy/
				
You can also run a dataset processing ros node that will publish topics that can be visualized with rviz

    rosrun okvis_ros okvis_node_synchronous path/to/okvis_ros/okvis/config/config_fpga_p2_euroc.yaml path/to/MH_01_easy/

Use the rviz.rviz configuration in the okvis_ros/config/ directory to get the pose / 
landmark display.

If you want to run the live application connecting to a sensor, use the okvis_node 
application (modify the launch file launch/okvis_node.launch).

## Outputs and frames

In terms of coordinate frames and notation, 

* W denotes the OKVIS World frame (z up), 
* C\_i denotes the i-th camera frame, 
* S denotes the IMU sensor frame,
* B denotes a (user-specified) body frame.

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
	source and you run ROS indigo checkout pull request 3:

    git fetch origin pull/3/head:request3
    git checkout request3

* Follow https://github.com/ethz-asl/kalibr/wiki/multiple-camera-calibration to 
  calibrate intrinsic and extrinsic parameters of the cameras. If you receive an 
  error message that the tool was unable to make an initial guess on focal 
  length, make sure that your recorded dataset contains frames that have the 
  whole calibration target in view.

* Follow https://github.com/ethz-asl/kalibr/wiki/camera-imu-calibration to get 
  estimates for the spatial parameters of the cameras with respect to the IMU.

* MSCKF2/Hybrid filter does not support pure rotation, and cannot start from static mode, because it uses delayed triangulation. But it supports infinity points.

## Contribution guidelines

* Contact s.leutenegger@imperial.ac.uk to request access to the bitbucket 
  repository.

* Programming guidelines: please follow 
  https://github.com/ethz-asl/programming_guidelines/wiki/Cpp-Coding-Style-Guidelines .
	
* Writing tests: please write unit tests (gtest).

* Code review: please create a pull request for all changes proposed. The pull 
  request will be reviewed by an admin before merging.

## Support

The developpers will be happy to assist you or to consider bug reports / feature 
requests. But questions that can be answered reading this document will be 
ignored. Please contact s.leutenegger@imperial.ac.uk.
