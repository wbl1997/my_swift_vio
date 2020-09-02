cmake_minimum_required(VERSION 2.8.3)
project(msckf)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()
IF(CMAKE_BUILD_TYPE MATCHES Release)
  ADD_DEFINITIONS(-DNDEBUG)
ENDIF()

message(STATUS "Building MSCKF with ${CMAKE_BUILD_TYPE}")

option(DO_TIMING "Time routines." OFF)
if(NOT DO_TIMING)
  add_definitions(-DDEACTIVATE_TIMERS)
  message(STATUS "Deactivating timers.")
endif()

if(APPLE)
  # The clang compiler (on osx) is somehow much more strict
  # than the compilers on ubuntu and so this does not seem
  # possible on OSX just yet.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -std=c++11 -fPIC")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -std=c++11 -fPIC")
endif()

# https://github.com/StableCoder/cmake-scripts.git
include(./cmake/sanitizers.cmake)

if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "arm*")
  message(STATUS "ARM processor detected, will attempt to use NEON.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
else()
  message(STATUS "Assuming SSE instructions available.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mssse3")
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/okvis/cmake")
find_package(Eigen REQUIRED)
include_directories(${EIGEN_INCLUDE_DIR}) 

LIST(APPEND ROS_DEPENDENT_PACKAGES
  rosbag
  roscpp
  sensor_msgs
  image_transport
  cv_bridge
  dynamic_reconfigure
  geometry_msgs
  image_geometry
  tf
)

if($ENV{ROS_DISTRO} STREQUAL "melodic")
  LIST(APPEND ROS_DEPENDENT_PACKAGES
    pcl_conversions
    pcl_ros)
  add_definitions(-DHAVE_PCL)
elseif($ENV{ROS_DISTRO} STREQUAL "kinetic")
  message("Publishing with PCL point cloud is disabled with ROS kinetic because"
      " libpcl-dev on Ubuntu 16 depends on the system-wide Eigen \nwhich"
      " is incompatible with ceres solver 14.0.")
else()
  message("Unknown ROS distro:")
  message($ENV{ROS_DISTRO})
endif()

find_package(catkin REQUIRED  
  ${ROS_DEPENDENT_PACKAGES}
)

find_package(vio_common REQUIRED)

# generate dynamic reconfigure stuff
generate_dynamic_reconfigure_options(cfg/Camera.cfg)

catkin_package(
    DEPENDS
    CATKIN_DEPENDS
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
)

# The configure step fails at catkin_package() if this directory
# doesn't exist yet. Create it at configure time!
file(MAKE_DIRECTORY ${CATKIN_DEVEL_PREFIX}/include)

# let's pull in the okvis plain cmake project
set(BUILD_SHARED_LIBS OFF)
add_subdirectory(okvis)

# libvisensor
find_package(VISensor)

if(VISENSORDRIVER_FOUND)
  add_definitions(-DHAVE_LIBVISENSOR)
  message(STATUS "Found libvisensor in ROS interface.")
  include_directories(${VISensorDriver_INCLUDE_DIR})
else()
  message(WARNING "libvisensor not found")
endif()

find_package(GTSAM)
if(GTSAM_FOUND)
  add_definitions(-DHAVE_GTSAM)
  include_directories(${GTSAM_INCLUDE_DIR})
  list(APPEND LINK_LIBRARIES
    gtsam
    gtsam_unstable)
  message(STATUS "Found GTSAM at ${GTSAM_INCLUDE_DIR}.")
else()
  message(WARNING "GTSAM not found")
endif()

include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${CMAKE_BINARY_DIR}/include
)

# https://github.com/Crascit/DownloadProject/blob/master/CMakeLists.txt
if (CMAKE_VERSION VERSION_LESS 3.2)
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "")
else()
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "UPDATE_DISCONNECTED 1")
endif()

include("${PROJECT_SOURCE_DIR}/okvis/cmake/DownloadProject/DownloadProject.cmake")
download_project(PROJ                DBoW2
                 PREFIX ${CMAKE_CURRENT_BINARY_DIR}/DBoW2
                 GIT_REPOSITORY      https://github.com/dorian3d/DBoW2.git
                 GIT_TAG             master
                 ${UPDATE_DISCONNECTED_IF_AVAILABLE}
)

add_subdirectory(${DBoW2_SOURCE_DIR} ${DBoW2_BINARY_DIR})

# When using CMake 2.8.11 or later, header path dependencies
# should be automatically added to the targets.
# But this does not work for DBoW2.
# For earlier CMake versions, we have to explicitly add the
# required directories to the header search path ourselves.
# if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${DBoW2_SOURCE_DIR}/include")
# endif()

### Download and unzip the vocabularly file
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/vocabulary/ORBvoc.yml)
  message(STATUS "Downloading vocabulary file from dropbox.")
  file(DOWNLOAD
       http://www.dropbox.com/s/lyo0qgbdxn6eg6o/ORBvoc.zip?dl=1
       ${CMAKE_CURRENT_SOURCE_DIR}/vocabulary/ORBvoc.zip
       SHOW_PROGRESS
       STATUS orbvoc_download_success
       TIMEOUT 60
       EXPECTED_MD5 41c290ca05ab2b68548dc6542632d69f)
  if(orbvoc_download_success)
    message(STATUS "Unzipping vocabulary file.")

    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_SOURCE_DIR}/vocabulary/ORBvoc.zip
                            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/vocabulary/)
    message(STATUS "Moving vocabulary file.")
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory
                    ${CMAKE_CURRENT_SOURCE_DIR}/vocabulary/ORBvoc/
                    ${CMAKE_CURRENT_SOURCE_DIR}/vocabulary/)
    execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/vocabulary/ORBvoc/)
  else(orbvoc_download_success)
    message(STATUS "Failed to download vocabulary file. Please download manually.")
  endif(orbvoc_download_success)
else()
  message(STATUS "Vocabulary file exists, will not download.")
endif()

if(GTSAM_FOUND)
  add_subdirectory(Kimera-RPGO)
  include_directories(Kimera-RPGO)
  list(APPEND LINK_LIBRARIES KimeraRPGO)
endif()

# compile the ROS dependent stuff into a library
add_definitions(-fPIC -march=native -Wextra -Winit-self -Woverloaded-virtual
                -Wnon-virtual-dtor -Wsign-promo -Wno-long-long -std=c++0x)

LIST(APPEND SOURCEFILES
    src/feature_tracker/feature_tracker.cpp

    src/gtsam/ImuFactorTestHelpers.cpp

    src/io_wrap/CameraPoseVisualization.cpp
    src/io_wrap/CommonGflags.cpp
    src/io_wrap/PgoPublisher.cpp
    src/io_wrap/Player.cpp
    src/io_wrap/PoseViewer.cpp
    src/io_wrap/Publisher.cpp
    src/io_wrap/RosParametersReader.cpp
    src/io_wrap/StreamHelper.cpp
    src/io_wrap/Subscriber.cpp

    src/loop_closure/LcdThirdPartyWrapper.cpp
    src/loop_closure/LoopClosureDetectorParams.cpp

    src/msckf/CameraSystemCreator.cpp
    src/msckf/ConsistentEstimator.cpp
    src/msckf/GeneralEstimator.cpp
    src/msckf/FeatureTriangulation.cpp
    src/msckf/FilterHelper.cpp
    src/msckf/HybridFilter.cpp
    src/msckf/HybridFrontend.cpp
    src/msckf/ImuSimulator.cpp
    src/msckf/InvariantEKF.cpp
    src/msckf/MotionAndStructureStats.cpp
    src/msckf/MSCKF2.cpp
    src/msckf/PointLandmark.cpp
    src/msckf/PointLandmarkSimulationRS.cpp
    src/msckf/PreconditionedEkfUpdater.cpp
    src/msckf/SimulationFrontend.cpp

    src/msckf/TFVIO.cpp
    src/msckf/TwoViewPair.cpp
    src/msckf/VioEvaluationCallback.cpp
    src/msckf/VioFactoryMethods.cpp
    src/msckf/VioTestSystemBuilder.cpp
    src/msckf/numeric_ceres_residual_Jacobian.cpp
    src/msckf/triangulate.cpp
)

if(GTSAM_FOUND)
list(APPEND SOURCEFILES
    src/loop_closure/GtsamWrap.cpp
    src/loop_closure/LoopClosureDetector.cpp

    src/gtsam/ImuFrontEnd.cpp
    src/gtsam/ImuFrontEndParams.cpp
    src/gtsam/RiExtendedPose3.cpp
    src/gtsam/VioBackEndParams.cpp
    src/gtsam/SlidingWindowSmoother.cpp
)
endif(GTSAM_FOUND)

add_library(${PROJECT_NAME} STATIC
  ${SOURCEFILES}
)

add_dependencies(${PROJECT_NAME} okvis_multisensor_processing)

# make sure configure headers are built before any node using them
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg)

list(APPEND LINK_LIBRARIES
    ${catkin_LIBRARIES}
    okvis_util
    okvis_kinematics
    okvis_time
    okvis_cv
    okvis_common
    loop_closure
    okvis_ceres
    okvis_timing
    okvis_matcher
    okvis_frontend
    okvis_multisensor_processing
    ${vio_common_LIBRARIES}
    DBoW2
    gflags
    pthread
)

target_link_libraries(${PROJECT_NAME}  
  ${LINK_LIBRARIES}
)

if(VISENSORDRIVER_FOUND)
  target_link_libraries(${PROJECT_NAME} ${VISensorDriver_LIBRARY})
endif()

include_directories(
  okvis/okvis_util/include
  okvis/okvis_kinematics/include
  okvis/okvis_time/include
  okvis/okvis_cv/include
  okvis/okvis_common/include
  okvis/loop_closure/include
  okvis/okvis_ceres/include
  okvis/okvis_timing/include
  okvis/okvis_matcher/include
  okvis/okvis_frontend/include 
  okvis/okvis_multisensor_processing/include

  sophus
  ${vio_common_INCLUDE_DIRS}
)

# now the actual applications
add_executable(okvis_node src/apps/okvis_node.cpp src/apps/VioSystemWrap.cpp)
target_link_libraries(okvis_node ${PROJECT_NAME})

add_executable(okvis_node_synchronous src/apps/okvis_node_synchronous.cpp
    src/apps/VioSystemWrap.cpp)
target_link_libraries(okvis_node_synchronous ${PROJECT_NAME} )

add_executable(dataset_convertor src/apps/dataset_convertor.cpp)
target_link_libraries(dataset_convertor ${PROJECT_NAME} )

list(APPEND TEST_SRC_FILES
    test/test_main.cpp

    test/boost/TestBoostAccumulators.cpp
    test/boost/TestChiSquared.cpp

    test/eigen/TestEigen2CV.cpp
    test/eigen/TestEigenMatrixInitialization.cpp
    test/eigen/TestEigenQR.cpp
    test/eigen/TestEigenQuaternion.cpp

    test/io_wrap/TestVioDatasetPlayer.cpp
   
    test/loop_closure/TestLcdThirdPartyWrapper.cpp

    test/msckf/TestBoundedImuDeque.cpp
    test/msckf/TestCovarianceEstimation.cpp
    test/msckf/TestEpipolarJacobian.cpp
    test/msckf/TestEuclideanParamError.cpp
    test/msckf/TestExtrinsicJacobian.cpp
    test/msckf/TestFeatureInitialization.cpp
    test/msckf/TestFeatureJacobian.cpp
    test/msckf/TestFilterHelper.cpp
    test/msckf/TestHybridFilter.cpp
    test/msckf/TestImuOdometry.cpp
    test/msckf/TestImuSimulator.cpp
    test/msckf/TestMatrixSqrt.cpp
    test/msckf/TestMergeTwoLandmarks.cpp
    test/msckf/TestOkvisTriangulateFast.cpp
    test/msckf/TestOkvisTime.cpp
    test/msckf/TestPointLandmark.cpp
    test/msckf/TestPointLandmarkSimulationRS.cpp
    test/msckf/TestPointObsJacobian.cpp
    test/msckf/TestQuaternionFromSmallAngle.cpp
    test/msckf/TestRelativeMotionJacobian.cpp
    test/msckf/TestRsReprojectionError.cpp
    test/msckf/TestReprojectionErrorWithPap.cpp
    test/msckf/TestEpipolarFactor.cpp
    test/msckf/TestTriangulate.cpp
    test/msckf/TestTwoViewPair.cpp

    test/opencv/TestOpenCVDistortion.cpp

    test/ros/TestRosbag.cpp

    test/std/TestBinaryOperators.cpp
    test/std/TestCallMemberFunctionAtConstruction.cpp
    test/std/TestConstStaticInBaseClass.cpp
    test/std/TestDequeIterator.cpp
    test/std/TestDerivedClass.cpp
    test/std/TestMapKeyOrder.cpp
    test/std/TestModifyThisInConstMethod.cpp
    test/std/TestSizeof.cpp
    test/std/TestStdAccumulate.cpp
)

if(GTSAM_FOUND)
list(APPEND TEST_SRC_FILES
    test/gtsam/TestTriangulate.cpp
    test/gtsam/TestGtsamImuFactor.cpp
    test/gtsam/TestImuFrontEnd.cpp
    test/gtsam/TestIncrementalFixedLagSmoother.cpp
    test/gtsam/TestRiExtendedPose3.cpp
    test/loop_closure/TestGtsamBetweenFactor.cpp
    test/loop_closure/TestLoopClosureMethod.cpp
    test/loop_closure/TestRPGO.cpp
    test/loop_closure/TestSimpleProjectionFactor.cpp
)
endif(GTSAM_FOUND)

# In a catkin package, do not use or link gtest which is supported poorly.
if(CATKIN_ENABLE_TESTING)
  set(PROJECT_TEST_NAME ${PROJECT_NAME}_test)
  catkin_add_gtest(${PROJECT_TEST_NAME} ${TEST_SRC_FILES})
  target_link_libraries(${PROJECT_TEST_NAME} ${PROJECT_NAME})
endif()
