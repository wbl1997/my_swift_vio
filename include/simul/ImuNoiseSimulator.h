#ifndef IMU_SIMULATOR_H_
#define IMU_SIMULATOR_H_

#include "okvis/ImuMeasurements.hpp"
#include "okvis/Parameters.hpp"
#include <simul/SimParameters.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <fstream>
#include <iostream>
#include <vector>

namespace simul {
/**
 * @brief initImuNoiseParams
 * @param[in] simImuParameters
 * @param[out] imuParameters
 */
void initImuNoiseParams(
    const SimImuParameters& simImuParameters,
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
