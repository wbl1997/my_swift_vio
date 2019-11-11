cmake_minimum_required(VERSION 2.8.3)
project(msckf)

# { Begin: special handling for building with QtCreator
if(NOT CMAKE_SOURCE_DIR MATCHES "src$" AND NOT CMAKE_BINARY_DIR MATCHES "build$")
# If catkin_make is to be invoked, CMAKE_SOURCE_DIR will be like <workspace>/src
# and CMAKE_BINARY_DIR will be like <workspace>/build
# If Build within QtCreator is invoked, CMAKE_SOURCE_DIR will be like <workspace>/src/msckf
# and CMAKE_BINARY_DIR will be like <workspace>/build/msckf

get_filename_component(SUPER_SOURCE_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)
get_filename_component(SUPER_BINARY_DIR ${CMAKE_BINARY_DIR} DIRECTORY)
if(NOT SUPER_SOURCE_DIR MATCHES "src$" OR NOT SUPER_BINARY_DIR MATCHES "build$")
    message(WARNING "CMAKE_SOURCE_DIR is ${CMAKE_SOURCE_DIR} and"
        " CMAKE_BINARY_DIR is ${CMAKE_BINARY_DIR}.\n"
        "They should look like <workspace>/src/${PROJECT_NAME}"
        " and <workspace>/build/${PROJECT_NAME}")
endif()

get_filename_component(WORKSPACE_DIR ${SUPER_SOURCE_DIR} DIRECTORY)
set(CATKIN_DEVEL_PREFIX ${WORKSPACE_DIR}/devel) # this ensures binaries are put into correct location of catkin_ws
endif()
# } End
message("CATKIN_DEVEL_PREFIX ${CATKIN_DEVEL_PREFIX}")

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()
IF(CMAKE_BUILD_TYPE MATCHES Release)
  ADD_DEFINITIONS(-DNDEBUG)
ENDIF()

# ADD_DEFINITIONS(-DDEACTIVATE_TIMERS)

message(STATUS "Building MSCKF with ${CMAKE_BUILD_TYPE}")

set(OKVIS_SOURCE_PATH okvis)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mssse3")
MESSAGE ("ssse3 ON")

find_package(catkin REQUIRED  
  roscpp
  rosbag
  tf
  std_msgs
  sensor_msgs
  image_transport
  cv_bridge
  dynamic_reconfigure
  geometry_msgs
  image_geometry
)

find_package(vio_common)

set(CERES_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/include)

# generate dynamic reconfigure stuff
generate_dynamic_reconfigure_options(cfg/Camera.cfg)

catkin_package(
    DEPENDS
    CATKIN_DEPENDS rosbag tf roscpp cv_bridge message_runtime 
        std_msgs sensor_msgs vio_common
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
)

# The configure step fails at catkin_package() if this directory
# doesn't exist yet. Create it at configure time!
file(MAKE_DIRECTORY ${CATKIN_DEVEL_PREFIX}/include)

# let's pull in the okvis plain cmake project
set(BUILD_SHARED_LIBS OFF)
add_subdirectory(okvis)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
find_package(Eigen REQUIRED)

include_directories(
  include
  ${EIGEN_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
)

# compile the ROS dependent stuff into a library
add_definitions(-fPIC -march=native -Wextra -Winit-self -Woverloaded-virtual
                -Wnon-virtual-dtor -Wsign-promo -Wno-long-long -std=c++0x)

LIST(APPEND SOURCEFILES
    src/feature_tracker/feature_tracker.cpp

    src/io_wrap/Player.cpp
    src/io_wrap/PoseViewer.cpp
    src/io_wrap/Publisher.cpp
    src/io_wrap/RosParametersReader.cpp
    src/io_wrap/StreamHelper.cpp
    src/io_wrap/Subscriber.cpp

    src/okvis/ceres/CameraTimeParamBlock.cpp

    src/msckf/BoundedImuDeque.cpp
    src/msckf/CameraSystemCreator.cpp
    src/msckf/FilterHelper.cpp
    src/msckf/HybridFilter.cpp
    src/msckf/HybridFrameNoncentralAbsoluteAdapter.cpp
    src/msckf/HybridFrameRelativeAdapter.cpp
    src/msckf/HybridFrontend.cpp
    src/msckf/HybridVio.cpp
    src/msckf/ImuOdometry.cpp
    src/msckf/ImuSimulator.cpp
    src/msckf/MSCKF2.cpp
    src/msckf/PreconditionedEkfUpdater.cpp
    src/msckf/SimulationFrontend.cpp
    src/msckf/TFVIO.cpp
    src/msckf/TwoViewPair.cpp
    src/msckf/VioFrameMatchingAlgorithm.cpp
    src/msckf/numeric_ceres_residual_Jacobian.cpp
    src/msckf/triangulate.cpp
    src/msckf/triangulateFast.cpp
)

add_library(${PROJECT_NAME} STATIC
  ${SOURCEFILES}
)

include_directories(
  ${OKVIS_SOURCE_PATH}/okvis_util/include
  ${OKVIS_SOURCE_PATH}/okvis_kinematics/include
  ${OKVIS_SOURCE_PATH}/okvis_time/include
  ${OKVIS_SOURCE_PATH}/okvis_cv/include
  ${OKVIS_SOURCE_PATH}/okvis_common/include
  ${OKVIS_SOURCE_PATH}/okvis_ceres/include
  ${OKVIS_SOURCE_PATH}/okvis_timing/include
  ${OKVIS_SOURCE_PATH}/okvis_matcher/include
  ${OKVIS_SOURCE_PATH}/okvis_frontend/include
  ${OKVIS_SOURCE_PATH}/okvis_multisensor_processing/include
  sophus
  ${vio_common_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
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
    okvis_ceres
    okvis_timing
    okvis_matcher
    okvis_frontend
    okvis_multisensor_processing
    ${vio_common_LIBRARIES}
    gflags
    pthread
)

target_link_libraries(${PROJECT_NAME}  
  ${LINK_LIBRARIES}
)

# now the actual applications
add_executable(okvis_node src/apps/okvis_node.cpp)
target_link_libraries(okvis_node ${PROJECT_NAME})

add_executable(okvis_node_synchronous src/apps/okvis_node_synchronous.cpp)
target_link_libraries(okvis_node_synchronous ${PROJECT_NAME} )

list(APPEND TEST_SRC_FILES
    test/test_main.cpp

    test/boost/TestBoostAccumulators.cpp
    test/boost/TestChiSquared.cpp

    test/eigen/TestEigen2CV.cpp
    test/eigen/TestEigenMatrixInitialization.cpp
    test/eigen/TestEigenQR.cpp

    test/io_wrap/TestVioDatasetPlayer.cpp

    test/msckf/TestBoundedImuDeque.cpp
    test/msckf/TestEpipolarJacobian.cpp
    test/msckf/TestEuclideanParamError.cpp
    test/msckf/TestExtrinsicJacobian.cpp
    test/msckf/TestFeatureInitialization.cpp
    test/msckf/TestFilterHelper.cpp
    test/msckf/TestHybridFilter.cpp
    test/msckf/TestImuOdometry.cpp
    test/msckf/TestImuSimulator.cpp
    test/msckf/TestPointObsJacobian.cpp
    test/msckf/TestRelativeMotionJacobian.cpp
    test/msckf/TestRsReprojectionError.cpp
    test/msckf/TestEpipolarFactor.cpp
    test/msckf/TestTriangulate.cpp
    test/msckf/TestTwoViewPair.cpp

    test/opencv/TestOpenCVDistortion.cpp

    test/std/TestBinaryOperators.cpp
    test/std/TestDequeIterator.cpp
    test/std/TestDerivedClass.cpp
    test/std/TestMapKeyOrder.cpp
    test/std/TestSizeof.cpp
    test/std/TestStdAccumulate.cpp
    )

#QtCreator may not create the allTests target
#if(CATKIN_ENABLE_TESTING)
#  set(PROJECT_TEST_NAME ${PROJECT_NAME}_test)
#  catkin_add_gtest(${PROJECT_TEST_NAME} ${TEST_SRC_FILES})
#  target_link_libraries(${PROJECT_TEST_NAME} ${PROJECT_NAME} gtest gmock pthread)
#endif()

if(BUILD_TESTS)
# The below lines cause tr1 not found issue in Ubuntu 16.04
#  if(APPLE)
#    add_definitions(-DGTEST_HAS_TR1_TUPLE=1)
#  else()
#    add_definitions(-DGTEST_HAS_TR1_TUPLE=0)
#  endif(APPLE)
  enable_testing()
  set(PROJECT_TEST_NAME ${PROJECT_NAME}_test)
  add_executable(${PROJECT_TEST_NAME}
    ${TEST_SRC_FILES}
  )
  target_link_libraries(${PROJECT_TEST_NAME}
    ${PROJECT_NAME}
    gtest gmock
    pthread)
  add_test(test ${PROJECT_TEST_NAME})
endif()
