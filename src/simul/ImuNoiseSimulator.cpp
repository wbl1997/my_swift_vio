#include "simul/ImuNoiseSimulator.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "vio/Sample.h"

DEFINE_double(sim_sigma_g_c, 1.2e-3, "simulated gyro noise density");
DEFINE_double(sim_sigma_a_c, 8e-3, "simulated accelerometer noise density");
DEFINE_double(sim_sigma_gw_c, 2e-5, "simulated gyro bias noise density");
DEFINE_double(sim_sigma_aw_c, 5.5e-5, "simulated accelerometer bias noise density");

namespace simul {
void initImuNoiseParams(
    okvis::ImuParameters* imuParameters, bool noisyInitialSpeedAndBiases,
    bool noisyInitialSensorParams,
    double sigma_bg, double sigma_ba,
    double std_Tg_elem,
    double std_Ts_elem,
    double std_Ta_elem,
    bool fixImuInternalParams) {
  imuParameters->g = 9.81;
  imuParameters->a_max = 1000.0;
  imuParameters->g_max = 1000.0;
  imuParameters->rate = 100;

  imuParameters->sigma_g_c = FLAGS_sim_sigma_g_c;
  imuParameters->sigma_a_c = FLAGS_sim_sigma_a_c;
  imuParameters->sigma_gw_c = FLAGS_sim_sigma_gw_c;
  imuParameters->sigma_aw_c = FLAGS_sim_sigma_aw_c;

  LOG(INFO) << "sigma_g_c " << FLAGS_sim_sigma_g_c
            << " sigma_a_c " << FLAGS_sim_sigma_a_c
            << " sigma_gw_c " << FLAGS_sim_sigma_gw_c
            << " sigma_aw_c " << FLAGS_sim_sigma_aw_c;

  imuParameters->tau = 600.0;

  imuParameters->sigma_bg = sigma_bg;
  imuParameters->sigma_ba = sigma_ba;

  if (fixImuInternalParams) {
    imuParameters->sigma_TGElement = 0;
    imuParameters->sigma_TSElement = 0;
    imuParameters->sigma_TAElement = 0;
  } else {
    // std for every element in shape matrix T_g
    imuParameters->sigma_TGElement = std_Tg_elem;
    imuParameters->sigma_TSElement = std_Ts_elem;
    imuParameters->sigma_TAElement = std_Ta_elem;
  }
  imuParameters->model_type = "BG_BA_TG_TS_TA";

  Eigen::Matrix<double, 9, 1> eye;
  eye << 1, 0, 0, 0, 1, 0, 0, 0, 1;

  if (noisyInitialSpeedAndBiases) {
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
  if (noisyInitialSensorParams) {
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

void addNoiseToImuReadings(const okvis::ImuParameters& imuParameters,
                           okvis::ImuMeasurementDeque* imuMeasurements,
                           okvis::ImuMeasurementDeque* trueBiases,
                           double gyroAccelNoiseFactor,
                           double gyroAccelBiasNoiseFactor,
                           std::ofstream* inertialStream) {
  double noiseFactor = gyroAccelNoiseFactor;
  double biasNoiseFactor = gyroAccelBiasNoiseFactor;
  LOG(INFO) << "noise downscale factor " << noiseFactor
            << " bias noise downscale factor " << biasNoiseFactor;
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

    double sqrtRate = std::sqrt(imuParameters.rate);
    double sqrtDeltaT = 1 / sqrtRate;
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
