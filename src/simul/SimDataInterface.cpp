#include <simul/SimDataInterface.hpp>
#include <swift_vio/IoUtil.hpp>
#include <swift_vio/imu/BoundedImuDeque.hpp>
#include <simul/PointLandmarkSimulationRS.hpp>

namespace simul {
const double SimDataInterface::imageNoiseMag_ = 1.0;

void SimDataInterface::resetImuBiases(const okvis::ImuParameters &imuParameters,
                                      const SimImuParameters &simParameters,
                                      const std::string &imuLogFile) {
  if (addImuNoise_) {
    std::shared_ptr<std::ofstream> inertialStream;
    if (!imuLogFile.empty()) {
      inertialStream.reset(new std::ofstream(imuLogFile, std::ofstream::out));
      (*inertialStream)
          << "% timestamp, gx, gy, gz[rad/sec], acc x, acc y, acc "
             "z[m/s^2], gyro bias xyz, acc bias xyz, noisy gxyz, acc xyz"
          << std::endl;
    }
    simul::addNoiseToImuReadings(imuParameters, &imuMeasurements_, &refBiases_,
                                 simParameters.sim_ga_noise_factor,
                                 simParameters.sim_ga_bias_noise_factor,
                                 inertialStream.get());
  } else {
    refBiases_ = imuMeasurements_;
    for (size_t i = 0; i < imuMeasurements_.size(); ++i) {
      refBiases_[i].measurement.gyroscopes.setZero();
      refBiases_[i].measurement.accelerometers.setZero();
    }
  }

  // remove the padding part of refBiases.
  auto it = refBiases_.begin();
  for (; it != refBiases_.end(); ++it) {
    if (fabs((it->timeStamp - times_.front()).toSec()) < 1e-8)
      break;
  }
  OKVIS_ASSERT_FALSE(std::runtime_error, it == refBiases_.end(),
                     "No imu reading close to motion start epoch by 1e-8");
  refBiases_.erase(refBiases_.begin(), it);
}

void SimDataInterface::saveRefMotion(const std::string &truthFile) {
  std::ofstream truthStream;
  truthStream.open(truthFile, std::ofstream::out);
  truthStream << "%state timestamp, frameIdInSource, T_WS(xyz, qxyzw), v_WS" << std::endl;

  rewind();

  uint64_t id = 0u;
  do {
    truthStream << currentTime() << " " << id << " " << std::setfill(' ')
                << currentPose().parameters().transpose().format(
                       swift_vio::kSpaceInitFmt)
                << " "
                << currentVelocity().transpose().format(
                       swift_vio::kSpaceInitFmt)
                << std::endl;
    ++id;
  } while (nextNFrame());
  truthStream.close();
}

void SimDataInterface::saveLandmarkGrid(const std::string &gridFile) const {
  return simul::saveLandmarkGrid(homogeneousPoints_, lmIds_, gridFile);
}

void saveCameraParameters(
    std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem,
    const std::string cameraFile) {
  std::ofstream stream(cameraFile);
  for (size_t i = 0u; i < cameraSystem->numCameras(); ++i) {
    Eigen::VectorXd allIntrinsics;
    cameraSystem->cameraGeometry(i)->getIntrinsics(allIntrinsics);
    stream << cameraSystem->T_SC(i)->parameters().transpose().format(
                  swift_vio::kSpaceInitFmt)
           << " " << allIntrinsics.transpose().format(swift_vio::kSpaceInitFmt)
           << " " << cameraSystem->cameraGeometry(i)->imageDelay() << " "
           << cameraSystem->cameraGeometry(i)->readoutTime() << std::endl;
  }
  stream.close();
}

SimFromRealData::SimFromRealData(const std::string &dataDir, bool addImageNoise)
    : SimDataInterface(addImageNoise) {
  // load maplab data

  // add image noise to feature observations.

  rewind();
}

void SimFromRealData::rewind() {}

bool SimFromRealData::nextNFrame() {
  return false;
}

void SimFromRealData::addFeaturesToNFrame(
    std::shared_ptr<const okvis::cameras::NCameraSystem> refCameraSystem,
    std::shared_ptr<okvis::MultiFrame> multiFrame,
    std::vector<std::vector<int>> *keypointIndexForLandmarks) const {

}

CurveData::CurveData(SimulatedTrajectoryType trajectoryType,
                     const okvis::ImuParameters& imuParameters,
                     bool addImageNoise) :
  SimDataInterface(addImageNoise) {
  startTime_ = okvis::Time(100);
  finishTime_ =  startTime_ + okvis::Duration(kDuration);

  trajectory_ = simul::createSimulatedTrajectory(
      trajectoryType, imuParameters.rate, imuParameters.g);

  trajectory_->getSampleTimes(startTime_, finishTime_, times_);
  trajectory_->getTruePoses(times_, ref_T_WS_list_);
  trajectory_->getTrueVelocities(times_, ref_v_WS_list_);
  trajectory_->getTrueInertialMeasurements(
      startTime_ - okvis::Duration(1), finishTime_ + okvis::Duration(1), imuMeasurements_);
  rewind();
}

void CurveData::initializeLandmarkGrid(LandmarkGridType gridType,
                                       double landmarkRadius) {
  double halfz = 1.5;
  bool addFloorCeiling = false;
  switch (gridType) {
  case LandmarkGridType::FourWalls:
    createBoxLandmarkGrid(&homogeneousPoints_, &lmIds_, halfz, addFloorCeiling);
    break;
  case LandmarkGridType::FourWallsFloorCeiling:
    halfz = 2.5;
    addFloorCeiling = true;
    createBoxLandmarkGrid(&homogeneousPoints_, &lmIds_, halfz, addFloorCeiling);
    break;
  case LandmarkGridType::Cylinder:
    createCylinderLandmarkGrid(&homogeneousPoints_, &lmIds_, landmarkRadius);
    break;
  }
}

void CurveData::rewind() {
  refIndex_ = 0u;
  refBiasIter_ = refBiases_.begin();
  refTimeIter_ = times_.begin();
  latestImuMeasurements_.clear();
  lastRefNFrameTime_ = currentTime();

  okvis::Time imuDataEndTime = currentTime() + okvis::Duration(1);
  okvis::Time imuDataBeginTime = lastRefNFrameTime_ - okvis::Duration(1);
  latestImuMeasurements_ = swift_vio::getImuMeasurements(
      imuDataBeginTime, imuDataEndTime, imuMeasurements_, nullptr);
}

bool CurveData::nextNFrame() {
  if (refIndex_ + kCameraIntervalRatio >= ref_T_WS_list_.size()) {
    return false;
  }

  refTimeIter_ += kCameraIntervalRatio;
  refIndex_ += kCameraIntervalRatio;
  refBiasIter_ += kCameraIntervalRatio;

  okvis::Time imuDataEndTime = currentTime() + okvis::Duration(1);
  okvis::Time imuDataBeginTime = lastRefNFrameTime_ - okvis::Duration(1);
  latestImuMeasurements_ = swift_vio::getImuMeasurements(
      imuDataBeginTime, imuDataEndTime, imuMeasurements_, nullptr);

  lastRefNFrameTime_ = currentTime();
  return true;
}

void CurveData::addFeaturesToNFrame(
    std::shared_ptr<const okvis::cameras::NCameraSystem> refCameraSystem,
    std::shared_ptr<okvis::MultiFrame> multiFrame,
    std::vector<std::vector<int>> *keypointIndexForLandmarks) const {
  std::vector<std::vector<size_t>> frameLandmarkIndices;
  PointLandmarkSimulationRS::projectLandmarksToNFrame(
      homogeneousPoints_, trajectory_, currentTime(),
      refCameraSystem, multiFrame, &frameLandmarkIndices, keypointIndexForLandmarks,
      addImageNoise_ ? &imageNoiseMag_ : nullptr);
}

} // namespace simul
