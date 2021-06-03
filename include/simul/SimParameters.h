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
  bool zeroImuIntrinsicParamNoise; ///< lock IMU intrinsics which do not include biases?

  //  Mmultiply the accelerometer and gyro noise root PSD by this reduction
  //  factor in generating noise. As a result, the std for noises used in
  //  covariance propagation is slightly larger than the std used in sampling
  //  noises.
  double sim_ga_noise_factor;

  //  Multiply the accelerometer and gyro BIAS noise root PSD by this
  //  reduction factor in generating noise.
  double sim_ga_bias_noise_factor;

  SimImuParameters(std::string _trajLabel="WavyCircle", bool _addImuNoise = true,
                   bool _noisyInitialSpeedAndBiases = true,
                   bool _noisyInitialSensorParams = false,
                   bool _zeroImuIntrinsicParamNoise = false,
                   double _sim_ga_noise_factor = 1.0,
                   double _sim_ga_bias_noise_factor = 1.0)
      : trajLabel(_trajLabel),
        trajectoryId(simul::trajectoryLabelToId.find(trajLabel)->second),
        addImuNoise(_addImuNoise),
        noisyInitialSpeedAndBiases(_noisyInitialSpeedAndBiases),
        noisyInitialSensorParams(_noisyInitialSensorParams),
        zeroImuIntrinsicParamNoise(_zeroImuIntrinsicParamNoise),
        sim_ga_noise_factor(_sim_ga_noise_factor),
        sim_ga_bias_noise_factor(_sim_ga_bias_noise_factor) {}

  std::string toString() const {
    std::stringstream ss;
    ss << "Trajectory label " << trajLabel << " addImuNoise " << addImuNoise
       << "\nnoisyInitialSpeedAndBiases " << noisyInitialSpeedAndBiases
       << " noisyInitialSensorParams " << noisyInitialSensorParams
       << "\nzeroImuIntrinsicParamNoise " << zeroImuIntrinsicParamNoise
       << " sim_ga_noise_factor " << sim_ga_noise_factor
       << " sim_ga_bias_noise_factor " << sim_ga_bias_noise_factor;
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

  bool zeroCameraIntrinsicParamNoise;

  double timeOffset;
  double readoutTime;

  bool noisyInitialSensorParams; ///< add system error to camera on projection
                                 ///< and distortion parameters?
  LandmarkGridType gridType;
  double landmarkRadius; // radius of the cylinder on whose surface the
                         // landmarks are distributed.

  SimVisionParameters(bool _addImageNoise = true, bool _useImageObservs = true,
                      simul::SimCameraModelType _cameraModelId =
                          simul::SimCameraModelType::EUROC,
                      simul::CameraOrientation _cameraOrientationId =
                          simul::CameraOrientation::Forward,
                      std::string _projOptModelName="FIXED",
                      std::string _extrinsicModelName="FIXED",
                      bool _zeroCameraIntrinsicParamNoise = false,
                      double _timeOffset = 0.0, double _readoutTime = 0.0,
                      bool _noisyInitialSensorParams = false,
                      LandmarkGridType _gridType = LandmarkGridType::FourWalls,
                      double _landmarkRadius = 5)
      : addImageNoise(_addImageNoise), useImageObservs(_useImageObservs),
        cameraModelId(_cameraModelId),
        cameraOrientationId(_cameraOrientationId),
        projOptModelName(_projOptModelName), extrinsicModelName(_extrinsicModelName),
        zeroCameraIntrinsicParamNoise(_zeroCameraIntrinsicParamNoise),
        timeOffset(_timeOffset), readoutTime(_readoutTime),
        noisyInitialSensorParams(_noisyInitialSensorParams),
        gridType(_gridType), landmarkRadius(_landmarkRadius) {}

  std::string toString() const {
    std::stringstream ss;
    ss << "addImageNoise " << addImageNoise << " useImageObservs "
       << useImageObservs << " camera geometry type "
       << static_cast<int>(cameraModelId) << " camera orientation type "
       << static_cast<int>(cameraOrientationId) << "\nprojOptModelName "
       << projOptModelName << " extrinsicModelName " << extrinsicModelName
       << "\nzeroCameraIntrinsicParamNoise " << zeroCameraIntrinsicParamNoise
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
  std::string externalInputDir; // external input in maplab csv format.

  TestSetting() {}

  TestSetting(const SimImuParameters &_imuParams,
              const SimVisionParameters &_visionParams,
              const SimEstimatorParameters &_estimatorParams,
              const swift_vio::BackendParams &_backendParams,
              const std::string _externalInputDir)
      : imuParams(_imuParams), visionParams(_visionParams),
        estimatorParams(_estimatorParams),
        backendParams(_backendParams),
        externalInputDir(_externalInputDir) {
  }

  std::string toString() const {
    std::stringstream ss;
    ss << imuParams.toString() << "\n"
       << visionParams.toString() << "\n"
       << estimatorParams.toString() << "\nExternal input dir "
       << externalInputDir;
    return ss.str();
  }
};

}  // namespace simul

#endif // SIMPARAMETERS_H
