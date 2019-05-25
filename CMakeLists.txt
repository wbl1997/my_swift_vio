cmake_minimum_required(VERSION 2.8.3)
project(msckf2)

# { Begin: special handling for building with QtCreator
if(NOT CMAKE_SOURCE_DIR MATCHES "src$" AND NOT CMAKE_BINARY_DIR MATCHES "build$")
# If catkin_make is to be invoked, CMAKE_SOURCE_DIR will be like <workspace>/src
# and CMAKE_BINARY_DIR will be like <workspace>/build
# If Build within QtCreator is invoked, CMAKE_SOURCE_DIR will be like <workspace>/src/msckf2
# and CMAKE_BINARY_DIR will be like <workspace>/build/msckf2

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

SET(USE_ROS TRUE)
SET(MSCKF2 TRUE) # build with msck2 or hybridfilter
SET(FIRST_ESTIMATE TRUE) # use the first estimate technique for propagation matrix and measurement Jacobian?

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

#ADD_DEFINITIONS(-DDEACTIVATE_TIMERS)

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
    src/HybridFilter.cpp

    include/okvis/Subscriber.hpp   
    include/okvis/Player.hpp
    include/okvis/Publisher.hpp
    include/okvis/HybridVioParametersReader.hpp
    include/okvis/RosParametersReader.hpp
    include/okvis/HybridVio.hpp
    include/okvis/odeHybrid.hpp
    include/okvis/ImuSimulator.h
    include/okvis/HybridFilter.hpp
)

IF(MSCKF2)
  LIST(APPEND SOURCEFILES
    include/okvis/msckf2.hpp
    src/msckf2.cpp
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
