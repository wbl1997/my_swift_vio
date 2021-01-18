cmake_minimum_required(VERSION 3.10)

project(sensor_calib CXX)
set(CMAKE_CXX_STANDARD 17)

set(CERES_INCLUDE_DIR  ${PROJECT_SOURCE_DIR}/../../../build/msckf/include)
set(OKVIS_DIR ${PROJECT_SOURCE_DIR}/../okvis)

enable_testing()

add_subdirectory(libsensor_calib)
add_subdirectory(sensor_calib)
add_subdirectory(sensor_calib-test)
