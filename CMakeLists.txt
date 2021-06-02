cmake_minimum_required(VERSION 2.8.3)
project(swift_vio)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()
IF(CMAKE_BUILD_TYPE MATCHES Release)
  ADD_DEFINITIONS(-DNDEBUG)
ENDIF()

message(STATUS "Building swift_vio with ${CMAKE_BUILD_TYPE}")

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

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/thirdparty/okvis/cmake")
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

catkin_package(
    DEPENDS
    CATKIN_DEPENDS
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
)

# The configure step fails at catkin_package() if this directory
# doesn't exist yet. Create it at configure time!
file(MAKE_DIRECTORY ${CATKIN_DEVEL_PREFIX}/include)

set(BUILD_SHARED_LIBS OFF)
add_subdirectory(thirdparty/okvis)

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

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${CMAKE_BINARY_DIR}/include
)

# https://github.com/Crascit/DownloadProject/blob/master/CMakeLists.txt
if (CMAKE_VERSION VERSION_LESS 3.2)
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "")
else()
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "UPDATE_DISCONNECTED 1")
endif()

include("${PROJECT_SOURCE_DIR}/thirdparty/okvis/cmake/DownloadProject/DownloadProject.cmake")
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
  add_subdirectory(thirdparty/Kimera-RPGO)
  include_directories(thirdparty/Kimera-RPGO)
  list(APPEND LINK_LIBRARIES KimeraRPGO)
endif()

# compile the ROS dependent stuff into a library
add_definitions(-fPIC -march=native -Wextra -Winit-self -Woverloaded-virtual
                -Wnon-virtual-dtor -Wsign-promo -Wno-long-long -std=c++0x)

LIST(APPEND SOURCEFILES
    src/feature_tracker/FeatureTracker.cpp

    src/gtsam/VioBackEndParams.cpp
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

    src/swift_vio/BaseFilter.cpp
    src/swift_vio/EkfUpdater.cpp
    src/swift_vio/FeatureTriangulation.cpp
    src/swift_vio/FilterHelper.cpp
    src/swift_vio/HybridFilter.cpp
    src/swift_vio/HybridFrontend.cpp
    src/swift_vio/MotionAndStructureStats.cpp
    src/swift_vio/MSCKF.cpp
    src/swift_vio/PointLandmark.cpp

    src/swift_vio/StatAccumulator.cpp
    src/swift_vio/TFVIO.cpp
    src/swift_vio/TwoViewPair.cpp
    src/swift_vio/VioEvaluationCallback.cpp
    src/swift_vio/VioFactoryMethods.cpp
    src/swift_vio/triangulate.cpp

    include/simul/SimParameters.h
    src/simul/CameraSystemCreator.cpp
    src/simul/curves.cpp
    src/simul/ImuNoiseSimulator.cpp
    src/simul/LandmarkGrid.cpp
    src/simul/numeric_ceres_residual_Jacobian.cpp
    src/simul/PointLandmarkSimulationRS.cpp
    src/simul/SimDataInterface.cpp
    src/simul/SimulationFrontend.cpp
    src/simul/VioSimTestSystem.cpp
)

if(GTSAM_FOUND)
list(APPEND SOURCEFILES
    src/loop_closure/GtsamWrap.cpp
    src/loop_closure/LoopClosureDetector.cpp

    src/gtsam/ImuFrontEnd.cpp
    src/gtsam/ImuFrontEndParams.cpp
    src/gtsam/RiProjectionFactorIDP.cpp
    src/gtsam/RiProjectionFactorIDPAnchor.cpp
    src/gtsam/RiExtendedPose3.cpp
    src/gtsam/RiExtendedPose3Prior.cpp
    src/gtsam/RiImuFactor.cpp
    src/gtsam/RiSlidingWindowSmoother.cpp
    src/gtsam/SlidingWindowSmoother.cpp
)
endif(GTSAM_FOUND)

add_library(${PROJECT_NAME} STATIC
  ${SOURCEFILES}
)

add_dependencies(${PROJECT_NAME} okvis_multisensor_processing)

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

include_directories(
  thirdparty/okvis/okvis_util/include
  thirdparty/okvis/okvis_kinematics/include
  thirdparty/okvis/okvis_time/include
  thirdparty/okvis/okvis_cv/include
  thirdparty/okvis/okvis_common/include
  thirdparty/okvis/loop_closure/include
  thirdparty/okvis/okvis_ceres/include
  thirdparty/okvis/okvis_timing/include
  thirdparty/okvis/okvis_matcher/include
  thirdparty/okvis/okvis_frontend/include
  thirdparty/okvis/okvis_multisensor_processing/include
  thirdparty/sophus
  ${vio_common_INCLUDE_DIRS}
)

# now the actual applications
add_executable(swift_vio_node src/apps/swift_vio_node.cpp src/apps/VioSystemWrap.cpp)
target_link_libraries(swift_vio_node ${PROJECT_NAME})

add_executable(swift_vio_node_synchronous src/apps/swift_vio_node_synchronous.cpp
    src/apps/VioSystemWrap.cpp)
target_link_libraries(swift_vio_node_synchronous ${PROJECT_NAME} )

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

    test/swift_vio/TestBoundedImuDeque.cpp
    test/swift_vio/TestCovarianceEstimation.cpp
    test/swift_vio/TestEkf.cpp
    test/swift_vio/TestEpipolarFactor.cpp
    test/swift_vio/TestEpipolarJacobian.cpp
    test/swift_vio/TestEuclideanParamError.cpp
    test/swift_vio/TestExtrinsicJacobian.cpp
    test/swift_vio/TestFeatureInitialization.cpp
    test/swift_vio/TestFeatureJacobian.cpp
    test/swift_vio/TestFilterHelper.cpp
    test/swift_vio/TestHybridFilter.cpp
    test/swift_vio/TestImuOdometry.cpp
    test/swift_vio/TestCurves.cpp
    test/swift_vio/TestMergeTwoLandmarks.cpp
    test/swift_vio/TestOkvisTriangulateFast.cpp
    test/swift_vio/TestOkvisTime.cpp
    test/swift_vio/TestPointLandmark.cpp
    test/swift_vio/TestPointLandmarkSimulationRS.cpp
    test/swift_vio/TestPointObsJacobian.cpp
    test/swift_vio/TestQuaternionFromSmallAngle.cpp
    test/swift_vio/TestRelativeMotionJacobian.cpp
    test/swift_vio/TestRsReprojectionError.cpp
    test/swift_vio/TestReprojectionErrorWithPap.cpp
    test/swift_vio/TestSE3Interpolation.cpp
    test/swift_vio/TestTriangulate.cpp
    test/swift_vio/TestTwoViewPair.cpp

    test/opencv/TestOpenCVDistortion.cpp

    test/ros/TestRosbag.cpp

    test/std/TestBinaryOperators.cpp
    test/std/TestCallMemberFunctionAtConstruction.cpp
    test/std/TestConstStaticInBaseClass.cpp
    test/std/TestDequeIterator.cpp
    test/std/TestDerivedClass.cpp
    test/std/TestMapKeyOrder.cpp
    test/std/TestModifyThisInConstMethod.cpp
    test/std/TestPointerCompare.cpp
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
    test/gtsam/TestRiExtendedPose3Prior.cpp
    test/gtsam/TestRiImuFactor.cpp
    test/gtsam/TestRiProjectionFactorIDP.cpp
    test/gtsam/TestRiSmartProjectionFactor.cpp
    test/gtsam/TestRiTransformPointJacobian.cpp
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
