list(APPEND CMAKE_MODULE_PATH "${OKVIS_DIR}/cmake")
find_package(Eigen REQUIRED)

find_package(Glog REQUIRED)

add_library(libsensor_calib TimeAndRotationCalibrator.cpp
    ${OKVIS_DIR}/okvis_time/src/Time.cpp
    ${OKVIS_DIR}/okvis_time/src/Duration.cpp)
target_include_directories(libsensor_calib PUBLIC ${EIGEN_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(libsensor_calib PUBLIC
    "${OKVIS_DIR}/okvis_ceres/include"
    "${OKVIS_DIR}/okvis_common/include"
    "${OKVIS_DIR}/okvis_kinematics/include"
    "${OKVIS_DIR}/okvis_time/include"
    ${CERES_INCLUDE_DIR}
    ${GLOG_INCLUDE_DIRS})

target_link_libraries(libsensor_calib PUBLIC ${GLOG_LIBRARY})
