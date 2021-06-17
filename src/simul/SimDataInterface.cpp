#include <simul/SimDataInterface.hpp>

#include <random>

#include "okvis/IdProvider.hpp"

#include <swift_vio/IoUtil.hpp>
#include <swift_vio/imu/BoundedImuDeque.hpp>
#include <simul/PointLandmarkSimulationRS.hpp>

#include <gflags/gflags.h>

DEFINE_int32(sim_start_index, 0, "start index in real data");

namespace simul {
const double SimDataInterface::imageNoiseMag_ = 1.0;

okvis::ImuSensorReadings interpolate(const okvis::ImuSensorReadings &left,
                                     const okvis::ImuSensorReadings &right,
                                     double ratio) {
  Eigen::Vector3d gyro =
      left.gyroscopes + (right.gyroscopes - left.gyroscopes) * ratio;
  Eigen::Vector3d accel = left.accelerometers +
                          (right.accelerometers - left.accelerometers) * ratio;
  return okvis::ImuSensorReadings(gyro, accel);
}

void loadImuYaml(const std::string &imuYaml,
                 okvis::ImuParameters *imuParameters, Eigen::Vector3d *g_W) {
  cv::FileStorage file(imuYaml, cv::FileStorage::READ);
  OKVIS_ASSERT_TRUE(std::runtime_error, file.isOpened(),
                    "Could not open config file: " << imuYaml);

  imuParameters->sigma_a_c = file["accelerometer_noise_density"];
  imuParameters->sigma_aw_c = file["accelerometer_random_walk"];
  imuParameters->sigma_g_c = file["gyroscope_noise_density"];
  imuParameters->sigma_gw_c = file["gyroscope_random_walk"];
  imuParameters->rate = file["update_rate"];
  if (file["initial_gyro_bias"].isSeq()) {
    for (int i = 0; i < 3; ++i) {
      imuParameters->g0[i] = file["initial_gyro_bias"][i];
    }
  }
  if (file["initial_accelerometer_bias"].isSeq()) {
    for (int i = 0; i < 3; ++i) {
      imuParameters->a0[i] = file["initial_accelerometer_bias"][i];
    }
  }
  const cv::FileNode &gravityNode = file["gravity_in_target"];
  for (int i = 0; i < 3; ++i) {
    (*g_W)[i] = gravityNode[i];
  }
  imuParameters->g = g_W->norm();
  imuParameters->setGravityDirection(g_W->normalized());
  file.release();
}

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
}

void SimDataInterface::saveRefMotion(const std::string &truthFile) {
  std::ofstream truthStream;
  truthStream.open(truthFile, std::ofstream::out);
  truthStream << "%state timestamp, frameIdInSource, T_WS(xyz, qxyzw), v_WS" << std::endl;

  if (!rewind()) {
    LOG(WARNING) << "Skip saving reference motion because of failed rewinding!";
    return;
  }

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

okvis::ImuSensorReadings SimDataInterface::currentBiases() const {
  okvis::Time timenow = currentTime();
  if (refBiasIter_->timeStamp < timenow) {
    if ((timenow - refBiasIter_->timeStamp).toSec() < 1e-5) {
      return refBiasIter_->measurement;
    }
    auto left = refBiasIter_;
    auto right = refBiasIter_;
    ++right;
    if (right == refBiases_.end()) {
      return left->measurement;
    }
    OKVIS_ASSERT_TRUE(
        std::runtime_error, refBiasIter_->timeStamp < right->timeStamp,
        "Current bias timestamp should be close to current time!");
    double ratio = (timenow - left->timeStamp).toSec() /
                   (right->timeStamp - left->timeStamp).toSec();
    return interpolate(left->measurement, right->measurement, ratio);
  } else {  // refBiasIter_->timeStamp >= timenow
    if ((refBiasIter_->timeStamp - timenow).toSec() < 1e-5) {
      return refBiasIter_->measurement;
    }
    auto left = refBiasIter_;
    auto right = refBiasIter_;
    if (left == refBiases_.begin()) {
      return left->measurement;
    }
    --left;
    OKVIS_ASSERT_TRUE(
        std::runtime_error, refBiasIter_->timeStamp > left->timeStamp,
        "Current bias timestamp should be close to current time!");
    double ratio = (timenow - left->timeStamp).toSec() /
                   (right->timeStamp - left->timeStamp).toSec();
    return interpolate(left->measurement, right->measurement, ratio);
  }
  return refBiasIter_->measurement;
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

SimFromRealData::SimFromRealData(const std::string &dataDir, const okvis::ImuParameters& imuParameters,
                                 bool addImageNoise, bool addImuNoise)
    : SimDataInterface(imuParameters, addImageNoise, addImuNoise) {
  vimap_.loadVimapFromFolder(dataDir);
  const auto& imuData = vimap_.imuData();

  for (const auto& entry : imuData) {
    okvis::Time time(entry.sec_, entry.nsec_);
    okvis::ImuSensorReadings measurement(entry.w_, entry.a_);
    imuMeasurements_.emplace_back(time, measurement);
  }
  const int64_t secToNanos = 1000000000;
  const auto& times = vimap_.vertexTimestamps();
  times_.reserve(times.size() - FLAGS_sim_start_index);

  int index = 0;
  for (const auto &time : times) {
    if (index >= FLAGS_sim_start_index) {
      times_.emplace_back(time / secToNanos, time % secToNanos);
    }
    ++index;
  }

  startTime_ = times_.front();
  finishTime_ = times_.back();

  const auto &vertices = vimap_.vertices();
  ref_T_WS_list_.reserve(vertices.size());
  ref_v_WS_list_.reserve(vertices.size());
  index = 0;
  for (const auto &vertex : vertices) {
    if (index >= FLAGS_sim_start_index) {
      ref_T_WS_list_.emplace_back(vertex.p_WS_, vertex.q_WS_);
      ref_v_WS_list_.emplace_back(vertex.v_WS_);
    }
    ++index;
  }

  // Transform entities in the world frame used by the real data to
  // a new world frame with z along negative gravity.
  std::string imuYaml = dataDir + "/imu.yaml";
  loadImuYaml(imuYaml, &imuParameters_, &g_oldW_);
//  Eigen::Quaterniond q_newW_oldW;
//  swift_vio::alignZ(-g_oldW_, &q_newW_oldW);
//  T_newW_oldW_ = okvis::kinematics::Transformation(Eigen::Vector3d::Zero(), q_newW_oldW);
//  for (auto& T_WS : ref_T_WS_list_) {
//    T_WS = T_newW_oldW_ * T_WS;
//  }
//  for (auto &v_W : ref_v_WS_list_) {
//    v_W = q_newW_oldW._transformVector(v_W);
//  }
  if ((g_oldW_ - Eigen::Vector3d(0, 0, -9.80665)).norm() > 1.0) {
    imuParameters_.estimateGravityDirection = true;
    LOG(INFO) << "Estimate gravity direction because of unconventional gravity vector " << g_oldW_.transpose();
  }
}

void SimFromRealData::initializeLandmarkGrid(LandmarkGridType /*gridType*/,
                                             double /*landmarkRadius*/) {
  homogeneousPoints_.reserve(vimap_.landmarks().size());
  lmIds_.reserve(vimap_.landmarks().size());
  const auto &landmarks = vimap_.landmarks();
  for (const auto &lmk : landmarks) {
    Eigen::Vector4d hPoint;
    hPoint << lmk.position, 1.0;
    homogeneousPoints_.push_back(T_newW_oldW_ * hPoint);
    lmIds_.push_back(okvis::IdProvider::instance().newId());
  }
}

bool SimFromRealData::nextNFrame() {
  ++refIndex_;
  if (refIndex_ == ref_T_WS_list_.size()) {
    return false;
  }
  ++refTimeIter_;

  while (refBiasIter_ != refBiases_.end() && refBiasIter_->timeStamp < *refTimeIter_) {
    ++refBiasIter_;
  }
  if (refBiasIter_ == refBiases_.end()) {
    return false;
  }

  okvis::Time imuDataEndTime = currentTime() + okvis::Duration(1);
  okvis::Time imuDataBeginTime = lastRefNFrameTime_ - okvis::Duration(1);
  latestImuMeasurements_ = swift_vio::getImuMeasurements(
      imuDataBeginTime, imuDataEndTime, imuMeasurements_, nullptr);
  lastRefNFrameTime_ = currentTime();
  return true;
}

void SimFromRealData::addFeaturesToNFrame(
    std::shared_ptr<const okvis::cameras::NCameraSystem> refCameraSystem,
    std::shared_ptr<okvis::MultiFrame> multiFrame,
    std::vector<std::vector<int>> *keypointIndexForLandmarks) const {
  int numCameras = multiFrame->numFrames();
  keypointIndexForLandmarks->resize(numCameras);
  for (auto &keypointIndices : *keypointIndexForLandmarks) {
    keypointIndices.resize(vimap_.landmarks().size(), -1);
  }

  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<> d{0, imageNoiseMag_};

  const std::vector<vio::CornersInImage,
                    Eigen::aligned_allocator<vio::CornersInImage>>
      &keypointsInAllFrames = vimap_.validKeypoints();
  for (size_t camId = 0u; camId < vimap_.numberCameras(); ++camId) {
    const vio::CornersInImage &corners =
        keypointsInAllFrames.at((refIndex_ + FLAGS_sim_start_index) * vimap_.numberCameras() + camId);
    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(corners.corner_ids.size());
    for (size_t i = 0u; i < corners.corner_ids.size(); ++i) {
      const Eigen::Vector2d &uv = corners.corners.at(i);
      keypointIndexForLandmarks->at(corners.cam_id)
          .at(corners.corner_ids.at(i)) = i;
      if (addImageNoise_) {
        keypoints.emplace_back(uv[0] + d(gen), uv[1] + d(gen),
                               corners.radii[i]);
      } else {
        keypoints.emplace_back(uv[0], uv[1], corners.radii[i]);
      }
    }
    multiFrame->resetKeypoints(camId, keypoints);
  }

  // check
  for (size_t camId = 0u; camId < keypointIndexForLandmarks->size(); ++camId) {
    const auto &keypointIndices = keypointIndexForLandmarks->at(camId);
    int landmarkId = 0;
    for (auto index : keypointIndices) {
      if (index != -1) {
        OKVIS_ASSERT_EQ(
            std::runtime_error, landmarkId,
            keypointsInAllFrames[(refIndex_ + FLAGS_sim_start_index) * vimap_.numberCameras() + camId]
                .corner_ids[index],
            "Wrong landmark ID association!");
        Eigen::Vector2d keypoint;
        multiFrame->getKeypoint(camId, index, keypoint);
        Eigen::Vector2d projection(100, 100);
        Eigen::Vector4d hpC =
            (ref_T_WS_list_[refIndex_] * (*(refCameraSystem->T_SC(camId))))
                .inverse() *
            homogeneousPoints_[landmarkId];
        auto status =
            refCameraSystem->cameraGeometry(camId)->projectHomogeneous(
                hpC, &projection);
        if (status !=
                okvis::cameras::CameraBase::ProjectionStatus::Successful ||
            (keypoint - projection).norm() > 30) {
          LOG(INFO) << "keypoint " << keypoint.transpose() << " reprojection "
                    << projection.transpose() << " status " << (int)status;
        }
      }
      ++landmarkId;
    }
  }
}

bool SimFromRealData::rewind() {
  if (refBiases_.size() < 1) {
    LOG(WARNING) << "Call resetImuBiases to init IMU biases before going through the sim data!";
    return false;
  }

  refIndex_ = 0u;
  refTimeIter_ = times_.begin();
  latestImuMeasurements_.clear();
  lastRefNFrameTime_ = currentTime();

  refBiasIter_ = refBiases_.begin();
  while (refBiasIter_->timeStamp < *refTimeIter_) {
    ++refBiasIter_;
  }

  okvis::Time imuDataEndTime = currentTime() + okvis::Duration(1);
  okvis::Time imuDataBeginTime = lastRefNFrameTime_ - okvis::Duration(1);
  latestImuMeasurements_ = swift_vio::getImuMeasurements(
      imuDataBeginTime, imuDataEndTime, imuMeasurements_, nullptr);
  return true;
}

int SimFromRealData::expectedNumNFrames() const {
  return vimap_.vertices().size();
}

void SimFromRealData::navStateAtStart(swift_vio::InitialNavState* initialStateAndCov) const {
  initialStateAndCov->p_WS = ref_T_WS_list_.front().r();
  initialStateAndCov->q_WS = ref_T_WS_list_.front().q();
  initialStateAndCov->v_WS = ref_v_WS_list_.front();
  initialStateAndCov->std_p_WS = Eigen::Vector3d(0.1, 0.1, 0.1);
  initialStateAndCov->std_q_WS = Eigen::Vector3d(3 * M_PI / 180, 3 * M_PI / 180, M_PI / 180);
  initialStateAndCov->std_v_WS = Eigen::Vector3d(0.2, 0.2, 0.2);
  initialStateAndCov->initializeToCustomPose = true;
}

CurveData::CurveData(SimulatedTrajectoryType trajectoryType,
                     const okvis::ImuParameters& imuParameters,
                     bool addImageNoise, bool addImuNoise) :
  SimDataInterface(imuParameters, addImageNoise, addImuNoise) {
  startTime_ = okvis::Time(100);
  finishTime_ =  startTime_ + okvis::Duration(kDuration);

  trajectory_ = simul::createSimulatedTrajectory(
      trajectoryType, imuParameters.rate, imuParameters.g);

  trajectory_->getSampleTimes(startTime_, finishTime_, times_);
  trajectory_->getTruePoses(times_, ref_T_WS_list_);
  trajectory_->getTrueVelocities(times_, ref_v_WS_list_);
  trajectory_->getTrueInertialMeasurements(
      startTime_ - okvis::Duration(1), finishTime_ + okvis::Duration(1), imuMeasurements_);
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

bool CurveData::rewind() {
  if (refBiases_.size() < 1) {
    LOG(WARNING) << "Call resetImuBiases to init IMU biases before going through the sim data!";
    return false;
  }

  refIndex_ = 0u;
  refTimeIter_ = times_.begin();

  refBiasIter_ = refBiases_.begin();
  while (refBiasIter_->timeStamp < *refTimeIter_ &&
         (*refTimeIter_ - refBiasIter_->timeStamp).toSec() > 1e-3) {
    ++refBiasIter_;
  }
  OKVIS_ASSERT_LT(std::runtime_error,
                  (refBiasIter_->timeStamp - *refTimeIter_).toSec(), 1e-3,
                  "IMU data from curves are not perfectly synced with poses!");

  latestImuMeasurements_.clear();
  lastRefNFrameTime_ = currentTime();

  okvis::Time imuDataEndTime = currentTime() + okvis::Duration(1);
  okvis::Time imuDataBeginTime = lastRefNFrameTime_ - okvis::Duration(1);
  latestImuMeasurements_ = swift_vio::getImuMeasurements(
      imuDataBeginTime, imuDataEndTime, imuMeasurements_, nullptr);
  return true;
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

void CurveData::navStateAtStart(swift_vio::InitialNavState* initialStateAndCov) const {
  initialStateAndCov->p_WS = ref_T_WS_list_.front().r();
  initialStateAndCov->q_WS = ref_T_WS_list_.front().q();
  initialStateAndCov->v_WS = ref_v_WS_list_.front();
  initialStateAndCov->std_p_WS = Eigen::Vector3d(1e-5, 1e-5, 1e-5);
  initialStateAndCov->std_q_WS = Eigen::Vector3d(M_PI / 180, M_PI / 180, 1e-5);
  initialStateAndCov->std_v_WS = Eigen::Vector3d(5e-2, 5e-2, 5e-2);
  initialStateAndCov->initializeToCustomPose = true;
}

} // namespace simul
