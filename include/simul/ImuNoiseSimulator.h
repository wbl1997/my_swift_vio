#ifndef IMU_SIMULATOR_H_
#define IMU_SIMULATOR_H_

#include "okvis/ImuMeasurements.hpp"
#include "okvis/Parameters.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <fstream>
#include <iostream>
#include <vector>

namespace simul {
/**
 * @brief initImuNoiseParams
 * @param imuParameters
 * @param noisyInitialSpeedAndBiases
 * @param noisyInitialSensorParams
 * @param sigma_bg std dev of initial gyroscope bias.
 * @param sigma_ba std dev of initial accelerometer bias.
 * @param std_Ta_elem
 * @param fixImuInternalParams If true, set the noise of IMU intrinsic
 *     parameters (including misalignment shape matrices) to zeros in order
 *     to fix IMU intrinsic parameters in estimator.
 */
void initImuNoiseParams(
    bool noisyInitialSpeedAndBiases,
    bool noisyInitialSensorParams,
    double sigma_bg, double sigma_ba, double std_Tg_elem,
    double std_Ts_elem, double std_Ta_elem,
    bool fixImuInternalParams,
    okvis::ImuParameters* imuParameters);

/**
 * @brief addNoiseToImuReadings
 * @param imuParameters
 * @param imuMeasurements as input original perfect imu measurement,
 *     as output imu measurements with added bias and noise
 * @param trueBiases output added biases
 * @param inertialStream
 */
void addNoiseToImuReadings(const okvis::ImuParameters& imuParameters,
                           okvis::ImuMeasurementDeque* imuMeasurements,
                           okvis::ImuMeasurementDeque* trueBiases,
                           double gyroAccelNoiseFactor,
                           double gyroAccelBiasNoiseFactor,
                           std::ofstream* inertialStream);



} // namespace simul
#endif
