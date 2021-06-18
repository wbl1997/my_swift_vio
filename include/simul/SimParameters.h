#ifndef SIMPARAMETERS_H
#define SIMPARAMETERS_H

#include <gtsam/VioBackEndParams.h>

#include <simul/curves.h>
#include <simul/CameraSystemCreator.hpp>
#include <simul/LandmarkGrid.h>

namespace simul {
struct SimImuParameters {
  std::string trajLabel;
  SimulatedTrajectoryType trajectoryId;
  bool addImuNoise;                ///< add noise to IMU readings?

  bool noisyInitialSpeedAndBiases; ///< add noise to the prior position,
                                   ///< quaternion, velocity, bias in gyro,
                                   ///< bias in accelerometer?
  bool noisyInitialSensorParams; ///< add system error to IMU on scale and
                                 ///< misalignment and g-sensitivity?
  bool fixImuIntrinsicParams; ///< lock IMU intrinsics which do not include biases?

  double bg_std;
  double ba_std;
  double Tg_std;
  double Ts_std;
  double Ta_std;
  double sigma_g_c;
  double sigma_gw_c;
  double sigma_a_c;
  double sigma_aw_c;

  //  Mmultiply the accelerometer and gyro noise root PSD by this reduction
  //  factor in generating noise. As a result, the std for noises used in
  //  covariance propagation is slightly larger than the std used in sampling
  //  noises.
  double sim_ga_noise_factor;

  //  Multiply the accelerometer and gyro BIAS noise root PSD by this
  //  reduction factor in generating noise.
  double sim_ga_bias_noise_factor;
  std::string imuModel;
  bool noisyInitialGravityDirection;

  SimImuParameters(std::string _trajLabel = "WavyCircle",
                   bool _addImuNoise = true,
                   bool _noisyInitialSpeedAndBiases = true,
                   bool _noisyInitialSensorParams = false,
                   bool _fixImuIntrinsicParams = true,
                   double _bg_std = 5e-3, double _ba_std = 2e-2,
                   double _Tg_std = 5e-3, double _Ts_std = 1e-3,
                   double _Ta_std = 5e-3, double _sigma_g_c = 1.2e-3,
                   double _sigma_gw_c = 2e-5, double _sigma_a_c = 8e-3,
                   double _sigma_aw_c = 5.5e-5,
                   double _sim_ga_noise_factor = 1.0,
                   double _sim_ga_bias_noise_factor = 1.0,
                   std::string _imuModel = "BG_BA")
      : trajLabel(_trajLabel),
        trajectoryId(simul::trajectoryLabelToId.find(trajLabel)->second),
        addImuNoise(_addImuNoise),
        noisyInitialSpeedAndBiases(_noisyInitialSpeedAndBiases),
        noisyInitialSensorParams(_noisyInitialSensorParams),
        fixImuIntrinsicParams(_fixImuIntrinsicParams),
        bg_std(_bg_std), ba_std(_ba_std), Tg_std(_Tg_std), Ts_std(_Ts_std),
        Ta_std(_Ta_std), sigma_g_c(_sigma_g_c), sigma_gw_c(_sigma_gw_c),
        sigma_a_c(_sigma_a_c), sigma_aw_c(_sigma_aw_c),
        sim_ga_noise_factor(_sim_ga_noise_factor),
        sim_ga_bias_noise_factor(_sim_ga_bias_noise_factor),
        imuModel(_imuModel), noisyInitialGravityDirection(false) {}

  std::string toString() const {
    std::stringstream ss;
    ss << "Trajectory label " << trajLabel << " addImuNoise " << addImuNoise
       << "\nnoisyInitialSpeedAndBiases " << noisyInitialSpeedAndBiases
       << " noisyInitialSensorParams " << noisyInitialSensorParams
       << "\nfixImuIntrinsicParams " << fixImuIntrinsicParams
       << " sim_ga_noise_factor " << sim_ga_noise_factor
       << " sim_ga_bias_noise_factor " << sim_ga_bias_noise_factor
       << " IMU model " << imuModel;
    return ss.str();
  }
};

struct SimVisionParameters {
  bool addImageNoise;   ///< add noise to image measurements in pixels?
  bool useImageObservs; ///< use image observations in an estimator?

  simul::SimCameraModelType cameraModelId;
  simul::CameraOrientation cameraOrientationId;
  std::string projOptModelName;
  std::string extrinsicModelName;

  bool fixCameraInternalParams;

  double sigma_abs_position;
  double sigma_abs_orientation;
  double timeOffset;
  double readoutTime;

  bool noisyInitialSensorParams; ///< add system error to camera on projection
                                 ///< and distortion parameters?
  LandmarkGridType gridType;
  double landmarkRadius; // radius of the cylinder on whose surface the
                         // landmarks are distributed.
  bool useTrueLandmarkPosition;

  SimVisionParameters(bool _addImageNoise = true, bool _useImageObservs = true,
                      simul::SimCameraModelType _cameraModelId =
                          simul::SimCameraModelType::EUROC,
                      simul::CameraOrientation _cameraOrientationId =
                          simul::CameraOrientation::Forward,
                      std::string _projOptModelName="FIXED",
                      std::string _extrinsicModelName="FIXED",
                      bool _fixCameraInternalParams = true,
                      double _sigma_abs_position = 2e-2, double _sigma_abs_orientation = 1e-2,
                      double _timeOffset = 0.0, double _readoutTime = 0.0,
                      bool _noisyInitialSensorParams = false,
                      LandmarkGridType _gridType = LandmarkGridType::FourWalls,
                      double _landmarkRadius = 5,
                      bool _useTrueLandmarkPosition = false)
      : addImageNoise(_addImageNoise), useImageObservs(_useImageObservs),
        cameraModelId(_cameraModelId),
        cameraOrientationId(_cameraOrientationId),
        projOptModelName(_projOptModelName), extrinsicModelName(_extrinsicModelName),
        fixCameraInternalParams(_fixCameraInternalParams),
        sigma_abs_position(_sigma_abs_position), sigma_abs_orientation(_sigma_abs_orientation),
        timeOffset(_timeOffset), readoutTime(_readoutTime),
        noisyInitialSensorParams(_noisyInitialSensorParams),
        gridType(_gridType), landmarkRadius(_landmarkRadius),
        useTrueLandmarkPosition(_useTrueLandmarkPosition) {}

  std::string toString() const {
    std::stringstream ss;
    ss << "addImageNoise " << addImageNoise << " useImageObservs "
       << useImageObservs << " camera geometry type "
       << static_cast<int>(cameraModelId) << " camera orientation type "
       << static_cast<int>(cameraOrientationId) << "\nprojOptModelName "
       << projOptModelName << " extrinsicModelName " << extrinsicModelName
       << "\nfixCameraInternalParams " << fixCameraInternalParams
       << " timeOffset " << timeOffset << " readoutTime " << readoutTime
       << " noisyInitialSensorParams " << noisyInitialSensorParams
       << "\nlandmark grid type " << static_cast<int>(gridType)
       << " landmark radius " << landmarkRadius;
    return ss.str();
  }
};

struct SimEstimatorParameters {
  std::string estimatorLabel;
  swift_vio::EstimatorAlgorithm estimator_algorithm;
  int numRuns;
  int cameraObservationModelId;
  int landmarkModelId;
  bool useEpipolarConstraint;
  bool computeOkvisNees;

  SimEstimatorParameters(std::string _estimatorLabel = "MSCKF",
                         swift_vio::EstimatorAlgorithm _estimator_algorithm =
                             swift_vio::EstimatorAlgorithm::MSCKF,
                         int _numRuns = 10, int _cameraObservationModelId = 0,
                         int _landmarkModelId = 0,
                         bool _useEpipolarConstraint = false,
                         bool _computeOkvisNees = false)
      : estimatorLabel(_estimatorLabel),
        estimator_algorithm(_estimator_algorithm), numRuns(_numRuns),
        cameraObservationModelId(_cameraObservationModelId),
        landmarkModelId(_landmarkModelId),
        useEpipolarConstraint(_useEpipolarConstraint),
        computeOkvisNees(_computeOkvisNees) {}

  std::string toString() const {
    std::stringstream ss;
    ss << "estimator_algorithm "
       << swift_vio::EstimatorAlgorithmIdToName(estimator_algorithm) << " #runs "
       << numRuns << "\ncamera observation model id "
       << cameraObservationModelId << " landmark model id " << landmarkModelId
       << " use epipolar constraint? " << useEpipolarConstraint
       << " compute OKVIS NEES? " << computeOkvisNees;

    return ss.str();
  }
};

struct TestSetting {
  SimImuParameters imuParams;
  SimVisionParameters visionParams;
  SimEstimatorParameters estimatorParams;
  swift_vio::BackendParams backendParams;
  std::string simDataDir; // external input in maplab csv format.

  TestSetting() {}

  TestSetting(const SimImuParameters &_imuParams,
              const SimVisionParameters &_visionParams,
              const SimEstimatorParameters &_estimatorParams,
              const swift_vio::BackendParams &_backendParams,
              const std::string _simDataDir)
      : imuParams(_imuParams), visionParams(_visionParams),
        estimatorParams(_estimatorParams),
        backendParams(_backendParams),
        simDataDir(_simDataDir) {
  }

  std::string toString() const {
    std::stringstream ss;
    ss << imuParams.toString() << "\n"
       << visionParams.toString() << "\n"
       << estimatorParams.toString() << "\nExternal input dir "
       << simDataDir;
    return ss.str();
  }
};

}  // namespace simul

#endif // SIMPARAMETERS_H
