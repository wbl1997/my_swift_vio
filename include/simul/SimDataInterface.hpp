#ifndef INCLUDE_SWIFT_VIO_SIM_DATA_INTERFACE_HPP_
#define INCLUDE_SWIFT_VIO_SIM_DATA_INTERFACE_HPP_

#include <simul/ImuNoiseSimulator.h>
#include <simul/SimParameters.h>

#include <simul/curves.h>

#include <vio/VimapContainer.h>

namespace simul {

okvis::ImuSensorReadings interpolate(const okvis::ImuSensorReadings &left,
                                     const okvis::ImuSensorReadings &right,
                                     double ratio);

class SimDataInterface {
 protected:
  bool addImageNoise_;                ///< Add noise to image observations
  static const double imageNoiseMag_; // pixel unit

  bool addImuNoise_;

  okvis::Time startTime_;
  okvis::Time finishTime_;

  // reference variables of the same length used for computing RMSEs.
  std::vector<okvis::Time> times_;
  Eigen::AlignedVector<okvis::kinematics::Transformation> ref_T_WS_list_;
  Eigen::AlignedVector<Eigen::Vector3d> ref_v_WS_list_;
  okvis::ImuMeasurementDeque refBiases_;

  // imu meas. covering at least [start, finish].
  okvis::ImuMeasurementDeque imuMeasurements_;
  okvis::ImuParameters imuParameters_;

  // landmarks
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
      homogeneousPoints_;
  // Warn: To avoid conflicits with ids used in the backend estimator,
  // okvis::IdProvider should be used for generating Ids.
  std::vector<uint64_t> lmIds_;

  // variables for iterations
  std::vector<okvis::Time>::const_iterator refTimeIter_;
  size_t refIndex_;
  okvis::ImuMeasurementDeque::const_iterator refBiasIter_;

  okvis::ImuMeasurementDeque latestImuMeasurements_;
  okvis::Time lastRefNFrameTime_;

public:
  SimDataInterface(const okvis::ImuParameters& imuParams, bool addImageNoise) :
    addImageNoise_(addImageNoise), imuParameters_(imuParams) {}

  virtual ~SimDataInterface() {}

  /**
   * @brief rewind prepare for iteration at the start. Call this after resetImuBiases.
   * @return whether rewind is successful.
   */
  virtual bool rewind() = 0;

  virtual bool nextNFrame() = 0;

  /**
   * @brief addFeaturesToNFrame
   * @param[in] refCameraSystem
   * @param[out] multiFrame will be assigned keypoints for individual frames.
   * @param[out] keypointIndexForLandmarks keypoint indices for landmarks in frames.
   */
  virtual void addFeaturesToNFrame(
      std::shared_ptr<const okvis::cameras::NCameraSystem> refCameraSystem,
      std::shared_ptr<okvis::MultiFrame> multiFrame,
      std::vector<std::vector<int>> *keypointIndexForLandmarks) const = 0;

  virtual int expectedNumNFrames() const = 0;

  virtual void initializeLandmarkGrid(LandmarkGridType gridType,
                                      double landmarkRadius) = 0;

  /**
   * @brief reset and simulate IMU biases again.
   */
  void resetImuBiases(const okvis::ImuParameters &imuParameters,
                      const SimImuParameters &simParameters,
                      const std::string &imuLogFile);

  void saveRefMotion(const std::string &truthFile);

  void saveLandmarkGrid(const std::string &gridFile) const;

  okvis::Time currentTime() const {
    return *refTimeIter_;
  }

  okvis::kinematics::Transformation currentPose() const {
    return ref_T_WS_list_[refIndex_];
  }

  Eigen::Vector3d currentVelocity() const {
    return ref_v_WS_list_[refIndex_];
  }

  okvis::ImuSensorReadings currentBiases() const;

  okvis::ImuMeasurementDeque imuMeasurementsSinceLastNFrame() const {
    return latestImuMeasurements_;
  }

  const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
      &homogeneousPoints() const {
    return homogeneousPoints_;
  }

  const std::vector<uint64_t> &landmarkIds() { return lmIds_; }

  void navStateAtStart(okvis::kinematics::Transformation *T_WB,
                       Eigen::Vector3d *v_WB) const {
    *T_WB = ref_T_WS_list_.front();
    *v_WB = ref_v_WS_list_.front();
  }

  const okvis::ImuParameters& imuParameters() const {
    return imuParameters_;
  }
};

class SimFromRealData : public SimDataInterface {
public:
  SimFromRealData(const std::string& dataDir, const okvis::ImuParameters& imuParameters, bool addImageNoise);

  virtual ~SimFromRealData() {}

  void initializeLandmarkGrid(LandmarkGridType /*gridType*/,
                              double /*landmarkRadius*/) final;

  bool rewind() final;

  bool nextNFrame() final;

  void addFeaturesToNFrame(
      std::shared_ptr<const okvis::cameras::NCameraSystem> refCameraSystem,
      std::shared_ptr<okvis::MultiFrame> multiFrame,
      std::vector<std::vector<int>> *keypointIndexForLandmarks) const final;

  int expectedNumNFrames() const;

private:
  Eigen::Vector3d g_oldW_;  // gravity in the world frame used by the real data.
  okvis::kinematics::Transformation
      T_newW_oldW_; // To transform entities in the world frame used by the real
                    // data to a new world frame with z along negative gravity.

  vio::VimapContainer vimap_;
};

class CurveData : public SimDataInterface {
private:
  static const int kCameraIntervalRatio = 10; // #imu meas. for 1 camera nframe.
  const double kDuration = 300.0; // length of motion in seconds
  std::shared_ptr<CircularSinusoidalTrajectory> trajectory_;

public:

  CurveData(SimulatedTrajectoryType trajectoryType,
            const okvis::ImuParameters &imuParameters, bool addImageNoise);

  virtual ~CurveData() {}

  void initializeLandmarkGrid(LandmarkGridType gridType, double landmarkRadius) final;

  bool rewind() final;

  bool nextNFrame() final;

  void addFeaturesToNFrame(
      std::shared_ptr<const okvis::cameras::NCameraSystem> refCameraSystem,
      std::shared_ptr<okvis::MultiFrame> multiFrame,
      std::vector<std::vector<int>> *keypointIndexForLandmarks) const final;

  int expectedNumNFrames() const final {
    return times_.size() / kCameraIntervalRatio + 1;
  }
};

void saveCameraParameters(
    std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem,
    const std::string cameraFile);

/**
 * @brief loadImuYaml
 * @param[in] imuYaml imu yaml in the format of the Kalibr output.
 * @param[out] imuParams
 * @param[out] g_W gravity vector in the external world frame.
 */
void loadImuYaml(const std::string& imuYaml, okvis::ImuParameters* imuParams, Eigen::Vector3d *g_W);

} // namespace simul
#endif // INCLUDE_SWIFT_VIO_SIM_DATA_INTERFACE_HPP_
