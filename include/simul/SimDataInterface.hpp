#ifndef INCLUDE_SWIFT_VIO_SIM_DATA_INTERFACE_HPP_
#define INCLUDE_SWIFT_VIO_SIM_DATA_INTERFACE_HPP_

#include <simul/ImuNoiseSimulator.h>
#include <simul/SimParameters.h>

#include <simul/curves.h>

namespace simul {
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

  // landmarks
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
      homogeneousPoints_;
  std::vector<uint64_t> lmIds_;

  // variables for iterations
  std::vector<okvis::Time>::const_iterator refTimeIter_;
  size_t refIndex_;
  okvis::ImuMeasurementDeque::const_iterator refBiasIter_;

  okvis::ImuMeasurementDeque latestImuMeasurements_;
  okvis::Time lastRefNFrameTime_;

public:

  SimDataInterface(bool addImageNoise) : addImageNoise_(addImageNoise) {}

  virtual ~SimDataInterface() {}

  /**
   * @brief rewind prepare for iteration at the start.
   */
  virtual void rewind() = 0;

  virtual bool nextNFrame() = 0;

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

  okvis::ImuSensorReadings currentBiases() const {
    return refBiasIter_->measurement;
  }

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
};

class SimFromRealData : public SimDataInterface {
public:
  SimFromRealData(const std::string& dataDir, bool addImageNoise);

  virtual ~SimFromRealData() {}

  void initializeLandmarkGrid(LandmarkGridType /*gridType*/,
                              double /*landmarkRadius*/) final {}

  void rewind() final;

  bool nextNFrame() final;

  void addFeaturesToNFrame(
      std::shared_ptr<const okvis::cameras::NCameraSystem> refCameraSystem,
      std::shared_ptr<okvis::MultiFrame> multiFrame,
      std::vector<std::vector<int>> *keypointIndexForLandmarks) const final;

  int expectedNumNFrames() const { return 0; }
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

  void rewind() final;

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

} // namespace simul
#endif // INCLUDE_SWIFT_VIO_SIM_DATA_INTERFACE_HPP_
