cmake_minimum_required(VERSION 2.8.3)
project(msckf2)

SET(USE_ROS TRUE)
SET(MSCKF2 TRUE) # build with msck2 or hybridfilter
SET(FIRST_ESTIMATE TRUE) # use the first estimate technique for propagation matrix and measurement Jacobian?
SET(KLT_MATCHER FALSE) #use calcOpticalFlowPyrLK to track features instead of brisk feature matching

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()
IF(CMAKE_BUILD_TYPE MATCHES Release)
  ADD_DEFINITIONS(-DNDEBUG)
ENDIF()

IF(MSCKF2)
  ADD_DEFINITIONS(-DUSE_MSCKF2)
ENDIF()
IF(USE_ROS)
  add_definitions(-DSLAM_USE_ROS)
ENDIF()

IF(FIRST_ESTIMATE)
  ADD_DEFINITIONS(-DUSE_FIRST_ESTIMATE)
ENDIF()

IF(KLT_MATCHER)
  ADD_DEFINITIONS(-DUSE_KLT)
ELSEIF(1)
  #ADD_DEFINITIONS(-DUSE_BRISK)
  #ADD_DEFINITIONS(-DDEACTIVATE_TIMERS)
ENDIF()

message(STATUS "Building MSCKF2 with ${CMAKE_BUILD_TYPE}")

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
    src/Subscriber.cpp  
    src/Player.cpp
    src/Publisher.cpp
    src/HybridVioParametersReader.cpp
    src/RosParametersReader.cpp
    src/IMUOdometry.cpp  
    src/CameraIntrinsicParamBlock.cpp
    src/CameraDistortionParamBlock.cpp
    src/CameraTimeParamBlock.cpp
    src/ShapeMatrixParamBlock.cpp
    src/CameraIntrinsicError.cpp   
    src/HybridVio.cpp
    src/initialpvandstd.cpp
    src/HybridFrontend.cpp
    src/triangulate.cpp
    src/triangulateFast.cpp
    src/VioFrameMatchingAlgorithm.cpp
    src/HybridFrameRelativeAdapter.cpp
    src/HybridFrameNoncentralAbsoluteAdapter.cpp
    src/ImuSimulator.cpp
    src/TrackResultReader.cpp

    include/okvis/Subscriber.hpp   
    include/okvis/Player.hpp
    include/okvis/Publisher.hpp
    include/okvis/HybridVioParametersReader.hpp
    include/okvis/RosParametersReader.hpp
    include/okvis/HybridVio.hpp
    include/okvis/odeHybrid.hpp
    include/okvis/ImuSimulator.h
)

IF(MSCKF2)
  LIST(APPEND SOURCEFILES
    include/okvis/msckf2.hpp
    src/msckf2.cpp
  )
ELSE()
  LIST(APPEND SOURCEFILES
    include/okvis/HybridFilter.hpp
    src/HybridFilter.cpp
  )
ENDIF()

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

target_link_libraries(${PROJECT_NAME}  
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
  pthread
  ${vio_common_LIBRARIES}
  gflags
)

# a wrapper to invoke a specific test
add_executable(invoke_one_test src/invoke_one_test.cpp )
target_link_libraries(invoke_one_test ${PROJECT_NAME} )

# now the actual applications
add_executable(msckf2_node src/okvis_node.cpp)
target_link_libraries(msckf2_node ${PROJECT_NAME})

catkin_add_gtest(allTests test/test_main.cpp)
target_link_libraries(allTests ${PROJECT_NAME})
