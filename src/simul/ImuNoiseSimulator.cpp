#include "simul/ImuNoiseSimulator.h"

#include <glog/logging.h>

#include "vio/Sample.h"


namespace simul {
void initImuNoiseParams(
    const SimImuParameters& simParams,
    okvis::ImuParameters* imuParameters) {
  imuParameters->g = 9.81;
  imuParameters->a_max = 1000.0;
  imuParameters->g_max = 1000.0;
  imuParameters->rate = 100;

  imuParameters->sigma_g_c = simParams.sigma_g_c;
  imuParameters->sigma_a_c = simParams.sigma_a_c;
  imuParameters->sigma_gw_c = simParams.sigma_gw_c;
  imuParameters->sigma_aw_c = simParams.sigma_aw_c;

  imuParameters->tau = 600.0;

  imuParameters->sigma_bg = simParams.bg_std;
  imuParameters->sigma_ba = simParams.ba_std;

  if (simParams.fixImuIntrinsicParams) {
    imuParameters->sigma_TGElement = 0;
    imuParameters->sigma_TSElement = 0;
    imuParameters->sigma_TAElement = 0;
  } else {
    // std for every element in shape matrix T_g
    imuParameters->sigma_TGElement = simParams.Tg_std;
    imuParameters->sigma_TSElement = simParams.Ts_std;
    imuParameters->sigma_TAElement = simParams.Ta_std;
  }

  Eigen::Matrix<double, 9, 1> eye;
  eye << 1, 0, 0, 0, 1, 0, 0, 0, 1;

  if (simParams.noisyInitialSpeedAndBiases) {
    imuParameters->a0[0] = vio::gauss_rand(0, imuParameters->sigma_ba);
    imuParameters->a0[1] = vio::gauss_rand(0, imuParameters->sigma_ba);
    imuParameters->a0[2] = vio::gauss_rand(0, imuParameters->sigma_ba);
    imuParameters->g0[0] = vio::gauss_rand(0, imuParameters->sigma_bg);
    imuParameters->g0[1] = vio::gauss_rand(0, imuParameters->sigma_bg);
    imuParameters->g0[2] = vio::gauss_rand(0, imuParameters->sigma_bg);
  } else {
    imuParameters->a0.setZero();
    imuParameters->g0.setZero();
  }
  if (simParams.noisyInitialSensorParams) {
    imuParameters->Tg0 =
        eye + vio::Sample::gaussian(imuParameters->sigma_TGElement, 9);
    imuParameters->Ts0 =
        vio::Sample::gaussian(imuParameters->sigma_TSElement, 9);
    imuParameters->Ta0 =
        eye + vio::Sample::gaussian(imuParameters->sigma_TAElement, 9);
  } else {
    imuParameters->Tg0 = eye;
    imuParameters->Ts0.setZero();
    imuParameters->Ta0 = eye;
  }
}

void
addNoiseToImuReadings(const okvis::ImuParameters& imuParameters,
                           okvis::ImuMeasurementDeque* imuMeasurements,
                           okvis::ImuMeasurementDeque* trueBiases,
                           double gyroAccelNoiseFactor,
                           double gyroAccelBiasNoiseFactor,
                           std::ofstream* inertialStream) {
  double noiseFactor = gyroAccelNoiseFactor;
  double biasNoiseFactor = gyroAccelBiasNoiseFactor;
  LOG(INFO) << "noise downscale factor " << noiseFactor
            << " bias noise downscale factor " << biasNoiseFactor;
  double sqrtRate = std::sqrt(imuParameters.rate);
  double sqrtDeltaT = 1 / sqrtRate;
  *trueBiases = (*imuMeasurements);
  // The expected means of the prior of biases, imuParameters.g0 and a0,
  // fed to the estimator, are different from the true biases.
  Eigen::Vector3d bgk = Eigen::Vector3d::Zero();
  Eigen::Vector3d bak = Eigen::Vector3d::Zero();

  for (size_t i = 0; i < imuMeasurements->size(); ++i) {
    if (inertialStream) {
      Eigen::Vector3d porterGyro = imuMeasurements->at(i).measurement.gyroscopes;
      Eigen::Vector3d porterAcc = imuMeasurements->at(i).measurement.accelerometers;
      (*inertialStream) << imuMeasurements->at(i).timeStamp << " " << porterGyro[0]
                        << " " << porterGyro[1] << " " << porterGyro[2] << " "
                        << porterAcc[0] << " " << porterAcc[1] << " "
                        << porterAcc[2];
      (*inertialStream) << " " << bgk[0] << " " << bgk[1] << " " << bgk[2]
                        << " " << bak[0] << " " << bak[1] << " " << bak[2];
    }

    trueBiases->at(i).measurement.gyroscopes = bgk;
    trueBiases->at(i).measurement.accelerometers = bak;

    // eq 50, Oliver Woodman, An introduction to inertial navigation
    imuMeasurements->at(i).measurement.gyroscopes +=
        (bgk +
         vio::Sample::gaussian(imuParameters.sigma_g_c * sqrtRate * noiseFactor,
                               3));
    imuMeasurements->at(i).measurement.accelerometers +=
        (bak +
         vio::Sample::gaussian(imuParameters.sigma_a_c * sqrtRate * noiseFactor,
                               3));
    // eq 51, Oliver Woodman, An introduction to inertial navigation,
    // we do not divide sqrtDeltaT by sqrtT because sigma_gw_c is bias white noise density
    // for bias random walk (BRW) whereas eq 51 uses bias instability (BS) having the
    // same unit as the IMU measurements. also see eq 9 therein.
    bgk += vio::Sample::gaussian(
        imuParameters.sigma_gw_c * sqrtDeltaT * biasNoiseFactor, 3);
    bak += vio::Sample::gaussian(
        imuParameters.sigma_aw_c * sqrtDeltaT * biasNoiseFactor, 3);
    if (inertialStream) {
      Eigen::Vector3d porterGyro = imuMeasurements->at(i).measurement.gyroscopes;
      Eigen::Vector3d porterAcc = imuMeasurements->at(i).measurement.accelerometers;
      (*inertialStream) << " " << porterGyro[0] << " " << porterGyro[1] << " "
                        << porterGyro[2] << " " << porterAcc[0] << " "
                        << porterAcc[1] << " " << porterAcc[2] << std::endl;
    }
  }
}
}  // namespace simul
