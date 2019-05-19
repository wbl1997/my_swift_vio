#include <algorithm>
#include <map>
#include <memory>
#include <vector>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include <okvis/HybridVio.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ImuError.hpp>

// the following four are used only for debugging with ground truth
#include <okvis/ceres/CameraIntrinsicParamBlock.hpp>
#include <okvis/ceres/CameraDistortionParamBlock.hpp>
#include <okvis/ceres/CameraTimeParamBlock.hpp>
#include <okvis/ceres/ShapeMatrixParamBlock.hpp>

DEFINE_bool(
    use_mahalanobis, true,
    "use malalanobis gating test in optimize or a simple projection distance"
    " threshold in computing jacobians. true by default, set false"
    " in simulation as it may prune many valid correspondences");

/// \brief okvis Main namespace of this package.
namespace okvis {

static const int max_camera_input_queue_size = 10;
// was 0.02, overlap of imu data before and after two consecutive frames
// [seconds]
static const okvis::Duration temporal_imu_data_overlap(0.6);

#ifdef USE_MOCK
// Constructor for gmock.
HybridVio::HybridVio(okvis::VioParameters &parameters,
                     okvis::MockVioBackendInterface &estimator,
                     okvis::MockVioFrontendInterface &frontend)
    : speedAndBiases_propagated_(okvis::SpeedAndBiases::Zero()),
      imu_params_(parameters.imu), repropagationNeeded_(false),
      frameSynchronizer_(okvis::FrameSynchronizer(parameters)),
      lastAddedImageTimestamp_(okvis::Time(0, 0)), optimizationDone_(true),
      estimator_(estimator), frontend_(frontend), parameters_(parameters),
      maxImuInputQueueSize_(60) {
  init();
}
#else
// Constructor.
HybridVio::HybridVio(okvis::VioParameters &parameters)
    : speedAndBiases_propagated_(okvis::SpeedAndBiases::Zero()),
      imu_params_(parameters.imu), repropagationNeeded_(false),
      frameSynchronizer_(okvis::FrameSynchronizer(parameters)),
      lastAddedImageTimestamp_(okvis::Time(0, 0)), optimizationDone_(true),
      estimator_(parameters.sensors_information.imageReadoutTime),
      frontend_(parameters.nCameraSystem.numCameras(),
                parameters.input.voFeatureTracksFile),
      parameters_(parameters),
      maxImuInputQueueSize_(2 * max_camera_input_queue_size *
                            parameters.imu.rate /
                            parameters.sensors_information.cameraRate) {
  estimator_.resetInitialPVandStd(
      InitialPVandStd(parameters.initialState),
      parameters.initialState.bUseExternalInitState);
  setBlocking(false);
  init();
}
#endif

// Initialises settings and calls startThreads().
void HybridVio::init() {
  assert(parameters_.nCameraSystem.numCameras() > 0);
  numCameras_ = parameters_.nCameraSystem.numCameras();
  numCameraPairs_ = 1;

  frontend_.setBriskDetectionOctaves(parameters_.optimization.detectionOctaves);
  frontend_.setBriskDetectionThreshold(
      parameters_.optimization.detectionThreshold);
  frontend_.setBriskDetectionMaximumKeypoints(
      parameters_.optimization.maxNoKeypoints);

  // s.t. last_timestamp_ - overlap >= 0 (since okvis::time(-0.02)
  // returns big number)
  lastOptimizedStateTimestamp_ = okvis::Time(0.0) + temporal_imu_data_overlap;

  // s.t. last_timestamp_ - overlap >= 0 (since okvis::time(-0.02)
  // returns big number)
  lastAddedStateTimestamp_ = okvis::Time(0.0) + temporal_imu_data_overlap;

  estimator_.addImu(parameters_.imu);
  for (size_t i = 0; i < numCameras_; ++i) {
    estimator_.addCamera(parameters_.camera_extrinsics);
    cameraMeasurementsReceived_.emplace_back(
        std::shared_ptr<threadsafe::ThreadSafeQueue<
            std::shared_ptr<okvis::CameraMeasurement>>>(
            new threadsafe::ThreadSafeQueue<
                std::shared_ptr<okvis::CameraMeasurement>>()));
  }
  // set up windows so things don't crash on Mac OS
  if (parameters_.visualization.displayImages) {
    for (size_t im = 0; im < parameters_.nCameraSystem.numCameras(); im++) {
      std::stringstream windowname;
      windowname << "OKVIS camera " << im;
      cv::namedWindow(windowname.str());
    }
  }
  startThreads();
}

// Start all threads.
void HybridVio::startThreads() {
  // consumer threads
  for (size_t i = 0; i < numCameras_; ++i) {
    frameConsumerThreads_.emplace_back(&HybridVio::frameConsumerLoop, this, i);
  }
  for (size_t i = 0; i < numCameraPairs_; ++i) {
    keypointConsumerThreads_.emplace_back(&HybridVio::matchingLoop, this);
  }
  imuConsumerThread_ = std::thread(&HybridVio::imuConsumerLoop, this);
  positionConsumerThread_ = std::thread(&HybridVio::positionConsumerLoop,
                                        this);
  gpsConsumerThread_ = std::thread(&HybridVio::gpsConsumerLoop, this);
  magnetometerConsumerThread_ = std::thread(
      &HybridVio::magnetometerConsumerLoop, this);
  differentialConsumerThread_ = std::thread(
      &HybridVio::differentialConsumerLoop, this);

  // algorithm threads
  visualizationThread_ = std::thread(&HybridVio::visualizationLoop, this);
  optimizationThread_ = std::thread(&HybridVio::optimizationLoop, this);
  publisherThread_ = std::thread(&HybridVio::publisherLoop, this);
}

// Destructor. This calls Shutdown() for all threadsafe queues and
// joins all threads.
HybridVio::~HybridVio() {
  for (size_t i = 0; i < numCameras_; ++i) {
    cameraMeasurementsReceived_.at(i)->Shutdown();
  }
  keypointMeasurements_.Shutdown();
  matchedFrames_.Shutdown();
  imuMeasurementsReceived_.Shutdown();
  optimizationResults_.Shutdown();
  visualizationData_.Shutdown();
  imuFrameSynchronizer_.shutdown();
  positionMeasurementsReceived_.Shutdown();

  // consumer threads
  for (size_t i = 0; i < numCameras_; ++i) {
    frameConsumerThreads_.at(i).join();
  }
  for (size_t i = 0; i < numCameraPairs_; ++i) {
    keypointConsumerThreads_.at(i).join();
  }
  imuConsumerThread_.join();
  positionConsumerThread_.join();
  gpsConsumerThread_.join();
  magnetometerConsumerThread_.join();
  differentialConsumerThread_.join();
  visualizationThread_.join();
  optimizationThread_.join();
  publisherThread_.join();

  /*okvis::kinematics::Transformation endPosition;
  estimator_.get_T_WS(estimator_.currentFrameId(), endPosition);
  std::stringstream s;
  s << endPosition.r();
  LOG(INFO) << "Sensor end position:\n" << s.str();
  LOG(INFO) << "Distance to origin: " << endPosition.r().norm();*/
#ifndef DEACTIVATE_TIMERS
  LOG(INFO) << okvis::timing::Timing::print();
#endif
}

// Add a new image.
bool HybridVio::addImage(const okvis::Time &stamp, size_t cameraIndex,
                         const cv::Mat &image,
                         const std::vector<cv::KeyPoint> *keypoints,
                         int frameIdInSource, bool * /*asKeyframe*/) {
  assert(cameraIndex < numCameras_);

  if (lastAddedImageTimestamp_ > stamp
      && fabs((lastAddedImageTimestamp_ - stamp).toSec())
          > parameters_.sensors_information.frameTimestampTolerance) {
    LOG(ERROR)
        << "Received image from the past. Dropping the image.";
    return false;
  }
  lastAddedImageTimestamp_ = stamp;

  std::shared_ptr<okvis::CameraMeasurement> frame = std::make_shared<
      okvis::CameraMeasurement>();
  frame->measurement.image = image;
  frame->measurement.idInSource = frameIdInSource;
  frame->timeStamp = stamp;
  frame->sensorId = cameraIndex;


  if (keypoints != nullptr) {
    frame->measurement.deliversKeypoints = true;
    frame->measurement.keypoints = *keypoints;
  } else {
    frame->measurement.deliversKeypoints = false;
  }

  if (blocking_) {
    cameraMeasurementsReceived_[cameraIndex]->PushBlockingIfFull(frame, 1);
    return true;
  } else {
    cameraMeasurementsReceived_[cameraIndex]->PushNonBlockingDroppingIfFull(
        frame, max_camera_input_queue_size);
    size_t measSize = cameraMeasurementsReceived_[cameraIndex]->Size();
    if (measSize > max_camera_input_queue_size / 2)
      std::cout << "Warn: Exceptional camera meas size " << measSize
                << std::endl;
    return measSize == 1;
  }
}

// Add an abstracted image observation.
bool HybridVio::addKeypoints(
    const okvis::Time & /*stamp*/, size_t /*cameraIndex*/,
    const std::vector<cv::KeyPoint> & /*keypoints*/,
    const std::vector<uint64_t> & /*landmarkIds*/,
    const cv::Mat & /*descriptors*/,
    bool* /*asKeyframe*/) {
  OKVIS_THROW(Exception, "HybridVio::addKeypoints() not implemented anymore "
                         "since changes to _keypointMeasurements queue.");
  return false;
}

// Add an IMU measurement.
bool HybridVio::addImuMeasurement(const okvis::Time & stamp,
                                      const Eigen::Vector3d & alpha,
                                      const Eigen::Vector3d & omega) {
  okvis::ImuMeasurement imu_measurement;
  imu_measurement.measurement.accelerometers = alpha;
  imu_measurement.measurement.gyroscopes = omega;
  imu_measurement.timeStamp = stamp;

  if (blocking_) {
    imuMeasurementsReceived_.PushBlockingIfFull(imu_measurement, 1);
    return true;
  } else {
    imuMeasurementsReceived_.PushNonBlockingDroppingIfFull(
        imu_measurement, maxImuInputQueueSize_);
    if (imuMeasurementsReceived_.Size() > 1)
      return imuMeasurementsReceived_.Size() == 1;
  }
  return true;
}

// Add a position measurement.
void HybridVio::addPositionMeasurement(
    const okvis::Time &stamp, const Eigen::Vector3d &position,
    const Eigen::Vector3d &positionOffset,
    const Eigen::Matrix3d &positionCovariance) {
  okvis::PositionMeasurement position_measurement;
  position_measurement.measurement.position = position;
  position_measurement.measurement.positionOffset = positionOffset;
  position_measurement.measurement.positionCovariance = positionCovariance;
  position_measurement.timeStamp = stamp;

  if (blocking_) {
    positionMeasurementsReceived_.PushBlockingIfFull(position_measurement, 1);
    return;
  } else {
    positionMeasurementsReceived_.PushNonBlockingDroppingIfFull(
        position_measurement, maxPositionInputQueueSize_);
    return;
  }
}

// Add a GPS measurement.
void HybridVio::addGpsMeasurement(const okvis::Time &, double, double,
                                      double, const Eigen::Vector3d &,
                                      const Eigen::Matrix3d &) {
  OKVIS_THROW(Exception, "GPS measurements not supported")
}

// Add a magnetometer measurement.
void HybridVio::addMagnetometerMeasurement(const okvis::Time &,
                                           const Eigen::Vector3d &, double) {
  OKVIS_THROW(Exception, "Magnetometer measurements not supported")
}

// Add a static pressure measurement.
void HybridVio::addBarometerMeasurement(const okvis::Time &, double, double) {
  OKVIS_THROW(Exception, "Barometer measurements not supported")
}

// Add a differential pressure measurement.
void HybridVio::addDifferentialPressureMeasurement(const okvis::Time &,
                                                       double, double) {
  OKVIS_THROW(Exception, "Differential pressure measurements not supported")
}

// Set the blocking variable that indicates whether the addMeasurement()
// functions should return immediately (blocking=false),
// or only when the processing is complete.
void HybridVio::setBlocking(bool blocking) {
  blocking_ = blocking;
  // disable time limit for optimization
  if (blocking_) {
    std::lock_guard<std::mutex> lock(estimator_mutex_);
    estimator_.setOptimizationTimeLimit(
        -1.0, parameters_.optimization.max_iterations);
  }
}

// Loop to process frames from camera with index cameraIndex
void HybridVio::frameConsumerLoop(size_t cameraIndex) {
  std::shared_ptr<okvis::CameraMeasurement> frame;
  std::shared_ptr<okvis::MultiFrame> multiFrame;
  TimerSwitchable beforeDetectTimer(
      "1.1 frameLoopBeforeDetect" + std::to_string(cameraIndex), true);
  TimerSwitchable waitForFrameSynchronizerMutexTimer(
      "1.1.1 waitForFrameSynchronizerMutex" + std::to_string(cameraIndex),
      true);
  TimerSwitchable addNewFrameToSynchronizerTimer(
      "1.1.2 addNewFrameToSynchronizer" + std::to_string(cameraIndex), true);
  TimerSwitchable waitForStateVariablesMutexTimer(
      "1.1.3 waitForStateVariablesMutex" + std::to_string(cameraIndex), true);
  TimerSwitchable propagationTimer(
      "1.1.4 propagationTimer" + std::to_string(cameraIndex), true);
  TimerSwitchable detectTimer(
      "1.2 detectAndDescribe" + std::to_string(cameraIndex), true);
  TimerSwitchable afterDetectTimer(
      "1.3 afterDetect" + std::to_string(cameraIndex), true);
  TimerSwitchable waitForFrameSynchronizerMutexTimer2(
      "1.3.1 waitForFrameSynchronizerMutex2" + std::to_string(cameraIndex),
      true);
  TimerSwitchable waitForMatchingThreadTimer(
      "1.4 waitForMatchingThread" + std::to_string(cameraIndex), true);

  for (;;) {
    // get data and check for termination request
    if (cameraMeasurementsReceived_[cameraIndex]->PopBlocking(&frame) ==
        false) {
      return;
    }
    beforeDetectTimer.start();
    {  // lock the frame synchronizer
      waitForFrameSynchronizerMutexTimer.start();
      std::lock_guard<std::mutex> lock(frameSynchronizer_mutex_);
      waitForFrameSynchronizerMutexTimer.stop();
      // add new frame to frame synchronizer and get the MultiFrame container
      addNewFrameToSynchronizerTimer.start();
      multiFrame = frameSynchronizer_.addNewFrame(frame);
      addNewFrameToSynchronizerTimer.stop();
    } // unlock frameSynchronizer only now as we can be sure that not
    // two states are added for the same timestamp
    okvis::kinematics::Transformation T_WS;
    okvis::Time lastTimestamp;
    okvis::SpeedAndBiases speedAndBiases;
    // copy last state variables
    {
      waitForStateVariablesMutexTimer.start();
      std::lock_guard<std::mutex> lock(lastState_mutex_);
      waitForStateVariablesMutexTimer.stop();
      T_WS = lastOptimized_T_WS_;
      speedAndBiases = lastOptimizedSpeedAndBiases_;
      lastTimestamp = lastOptimizedStateTimestamp_;
    }

    // -- get relevant imu messages for new state
    okvis::Time imuDataEndTime = multiFrame->timestamp()
        + temporal_imu_data_overlap;
    okvis::Time imuDataBeginTime = lastTimestamp - temporal_imu_data_overlap;

    OKVIS_ASSERT_TRUE_DBG(Exception, imuDataBeginTime < imuDataEndTime,
                          "imu data end time is smaller than begin time.");

    // wait until all relevant imu messages have arrived and check for
    // termination request
    if (imuFrameSynchronizer_.waitForUpToDateImuData(
      okvis::Time(imuDataEndTime)) == false)  {
      return;
    }
    OKVIS_ASSERT_TRUE_DBG(
        Exception, imuDataEndTime < imuMeasurements_.back().timeStamp,
        "Waiting for up to date imu data seems to have failed!");

    okvis::ImuMeasurementDeque imuData = getImuMeasurments(imuDataBeginTime,
                                                           imuDataEndTime);

    // if imu_data is empty, either end_time > begin_time or
    // no measurements in timeframe, should not happen, as we waited
    // for measurements
    if (imuData.size() == 0) {
      beforeDetectTimer.stop();
      continue;
    }

    if (imuData.front().timeStamp > frame->timeStamp) {
      LOG(WARNING)
          << "Frame is newer than oldest IMU measurement. Dropping it.";
      beforeDetectTimer.stop();
      continue;
    }

    // get T_WC(camIndx) for detectAndDescribe()
    std::cout << "estimator numFrames in frame loop " << estimator_.numFrames()
              << std::endl;
    if (estimator_.numFrames() == 0) {
      // first frame ever
#ifdef USE_MSCKF2
      bool success = okvis::MSCKF2::initPoseFromImu(imuData, T_WS);
#else
      bool success = okvis::HybridFilter::initPoseFromImu(imuData, T_WS);
#endif
      std::cout << "estimator initialized using inertial data " << std::endl;
      {
        std::lock_guard<std::mutex> lock(lastState_mutex_);
        lastOptimized_T_WS_ = T_WS;
        lastOptimizedSpeedAndBiases_.setZero();
        lastOptimizedSpeedAndBiases_.segment<3>(3) = imu_params_.g0;
        lastOptimizedSpeedAndBiases_.segment<3>(6) = imu_params_.a0;
        lastOptimizedStateTimestamp_ = multiFrame->timestamp();
        //        printf("lastOptimizedStateTimestamp_ set to %ld "
        //               "for the first time\n", lastOptimizedStateTimestamp_);
      }
      OKVIS_ASSERT_TRUE_DBG(Exception, success,
          "pose could not be initialized from imu measurements.");
      if (!success) {
        beforeDetectTimer.stop();
        continue;
      }
    } else {
      // get old T_WS
      propagationTimer.start();
      okvis::ceres::ImuError::propagation(imuData, parameters_.imu, T_WS,
                                          speedAndBiases, lastTimestamp,
                                          multiFrame->timestamp());
      propagationTimer.stop();
    }
    okvis::kinematics::Transformation T_WC = T_WS
        * (*parameters_.nCameraSystem.T_SC(frame->sensorId));
    beforeDetectTimer.stop();

#ifdef USE_BRISK
    detectTimer.start();
    frontend_.detectAndDescribe(frame->sensorId, multiFrame, T_WC, nullptr);
    detectTimer.stop();
#endif

    afterDetectTimer.start();

    bool push = false;
    { // we now tell frame synchronizer that detectAndDescribe is done for
      // MF with our timestamp
      waitForFrameSynchronizerMutexTimer2.start();
      std::lock_guard<std::mutex> lock(frameSynchronizer_mutex_);
      waitForFrameSynchronizerMutexTimer2.stop();
      frameSynchronizer_.detectionEndedForMultiFrame(multiFrame->id());

      if (frameSynchronizer_.detectionCompletedForAllCameras(
          multiFrame->id())) {
        //        LOG(INFO) << "detection completed for multiframe with id "
        // << multi_frame->id();
        push = true;
      }
    }  // unlocking frame synchronizer
    afterDetectTimer.stop();
    if (push) {
      // use queue size 1 to propagate a congestion to the
      // _cameraMeasurementsReceived queue and check for termination request
      waitForMatchingThreadTimer.start();
      if (keypointMeasurements_.PushBlockingIfFull(multiFrame, 1) == false) {
        return;
      }
      waitForMatchingThreadTimer.stop();
    }
  }
}

// Loop that matches frames with existing frames.
void HybridVio::matchingLoop() {
  TimerSwitchable prepareToAddStateTimer("2.1 prepareToAddState", true);
  TimerSwitchable waitForOptimizationTimer("2.2 waitForOptimization", true);
  TimerSwitchable addStateTimer("2.3 addState", true);
  TimerSwitchable matchingTimer("2.4 matching", true);

  for (;;) {
    // get new frame
    std::shared_ptr<okvis::MultiFrame> frame;

    // get data and check for termination request
    if (keypointMeasurements_.PopBlocking(&frame) == false)
      return;

    prepareToAddStateTimer.start();
    // -- get relevant imu messages for new state
    okvis::Time imuDataEndTime = frame->timestamp() + temporal_imu_data_overlap;
    okvis::Time imuDataBeginTime = lastAddedStateTimestamp_
        - temporal_imu_data_overlap;
    if (imuDataBeginTime.toSec() == 0.0) { // first state not yet added
        imuDataBeginTime = frame->timestamp() - temporal_imu_data_overlap;
    }
    // at maximum Duration(.) sec of data is allowed, 1sec is
    // a conservative number
    if (imuDataEndTime - imuDataBeginTime > Duration(8)) {
      std::cout << imuDataEndTime << " " << imuDataBeginTime << " "
                << frame->timestamp() << " " << temporal_imu_data_overlap << " "
                << lastAddedStateTimestamp_ << std::endl;
      LOG(WARNING) << "Warn: Too long interval between two frames "
                   << lastAddedStateTimestamp_.toSec() << " and "
                   << frame->timestamp().toSec();
      imuDataBeginTime = imuDataEndTime - Duration(8);
    }
    OKVIS_ASSERT_TRUE_DBG(Exception, imuDataBeginTime < imuDataEndTime,
                          "imu data end time is smaller than begin time."
                              << "current frametimestamp " << frame->timestamp()
                              << " (id: " << frame->id() << "last timestamp "
                              << lastAddedStateTimestamp_
                              << " (id: " << estimator_.currentFrameId());

    // wait until all relevant imu messages have arrived and check for
    // termination request
    if (imuFrameSynchronizer_.waitForUpToDateImuData(
            okvis::Time(imuDataEndTime)) == false) {
      return;
      OKVIS_ASSERT_TRUE_DBG(
          Exception, imuDataEndTime < imuMeasurements_.back().timeStamp,
          "Waiting for up to date imu data seems to have failed!");
    }

    okvis::ImuMeasurementDeque imuData = getImuMeasurments(imuDataBeginTime,
                                                           imuDataEndTime);

    prepareToAddStateTimer.stop();
    // if imu_data is empty, either end_time > begin_time or
    // no measurements in timeframe, should not happen, as we waited
    // for measurements
    // this should only happen at the beginning of a test
    if (imuData.size() == 0)
      continue;

    // make sure that optimization of last frame is over.
    // TODO(sleuten): If we didn't actually 'pop' the _matchedFrames queue
    // until after optimization this would not be necessary
    {
      waitForOptimizationTimer.start();
      std::unique_lock<std::mutex> l(estimator_mutex_);
      while (!optimizationDone_)
        optimizationNotification_.wait(l);
      waitForOptimizationTimer.stop();
      addStateTimer.start();
      okvis::Time t0Matching = okvis::Time::now();
      bool asKeyframe = true; // for msckf2 always set a frame as keyframe
      if (estimator_.addStates(frame, imuData, asKeyframe)) {
        lastAddedStateTimestamp_ = frame->timestamp();
        addStateTimer.stop();
      } else {
        LOG(ERROR) << "Failed to add state! will drop multiframe.";
        addStateTimer.stop();
        continue;
      }

      // -- matching keypoints, initialising landmarks etc.
      okvis::kinematics::Transformation T_WS;
      estimator_.get_T_WS(frame->id(), T_WS);
      matchingTimer.start();
      frontend_.dataAssociationAndInitialization(estimator_, T_WS, parameters_,
                                                 map_, frame, &asKeyframe);

      matchingTimer.stop();
      if (asKeyframe) {
        estimator_.setKeyframe(frame->id(), asKeyframe);
      }
      if (!blocking_) {
        double timeLimit =
            parameters_.optimization.timeLimitForMatchingAndOptimization -
            (okvis::Time::now() - t0Matching).toSec();
        estimator_.setOptimizationTimeLimit(
            std::max<double>(0.0, timeLimit),
            parameters_.optimization.min_iterations);
      }
      optimizationDone_ = false;
    }  // unlock estimator_mutex_

    // use queue size 1 to propagate a congestion to the _matchedFrames queue
    if (matchedFrames_.PushBlockingIfFull(frame, 1) == false)
      return;
  }
}

// Loop to process IMU measurements.
void HybridVio::imuConsumerLoop() {
  okvis::ImuMeasurement data;
  TimerSwitchable processImuTimer("0 processImuMeasurements", true);
  for (;;) {
    // get data and check for termination request
    if (imuMeasurementsReceived_.PopBlocking(&data) == false)
      return;
    processImuTimer.start();
    okvis::Time start;
    const okvis::Time* end;  // do not need to copy end timestamp
    {
      std::lock_guard<std::mutex> imuLock(imuMeasurements_mutex_);
      OKVIS_ASSERT_TRUE(Exception,
                        imuMeasurements_.empty()
                        || imuMeasurements_.back().timeStamp < data.timeStamp,
                        "IMU measurement from the past received");
//      imuMeasurements_.push_back(data);

      if (parameters_.publishing.publishImuPropagatedState) {
        if (!repropagationNeeded_ && imuMeasurements_.size() > 0) {
          start = imuMeasurements_.back().timeStamp;
        } else if (repropagationNeeded_) {
          std::lock_guard<std::mutex> lastStateLock(lastState_mutex_);
          start = lastOptimizedStateTimestamp_;
          T_WS_propagated_ = lastOptimized_T_WS_;
          speedAndBiases_propagated_ = lastOptimizedSpeedAndBiases_;
          repropagationNeeded_ = false;
        } else {
          start = okvis::Time(0, 0);
        }
        end = &data.timeStamp;
      }
      imuMeasurements_.push_back(data);
    }  // unlock _imuMeasurements_mutex

    // notify other threads that imu data with timeStamp is here.
    imuFrameSynchronizer_.gotImuData(data.timeStamp);

    if (parameters_.publishing.publishImuPropagatedState) {
      Eigen::Matrix<double, 15, 15> covariance;
      Eigen::Matrix<double, 15, 15> jacobian;

      frontend_.propagation(imuMeasurements_, imu_params_, T_WS_propagated_,
                            speedAndBiases_propagated_, start, *end,
                            &covariance, &jacobian);

      OptimizationResults result;
      result.stamp = *end;
      result.T_WS = T_WS_propagated_;
      result.speedAndBiases = speedAndBiases_propagated_;
      result.omega_S = imuMeasurements_.back().measurement.gyroscopes
          - speedAndBiases_propagated_.segment<3>(3);
      for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
        result.vector_of_T_SCi.push_back(
            okvis::kinematics::Transformation(
                *parameters_.nCameraSystem.T_SC(i)));
      }
      result.onlyPublishLandmarks = false;
      optimizationResults_.PushNonBlockingDroppingIfFull(result, 1);
    }
    processImuTimer.stop();
  }
}

// Loop to process position measurements.
void HybridVio::positionConsumerLoop() {
  okvis::PositionMeasurement data;
  for (;;) {
    // get data and check for termination request
    if (positionMeasurementsReceived_.PopBlocking(&data) == false)
      return;
    // collect
    {
      std::lock_guard<std::mutex> positionLock(positionMeasurements_mutex_);
      positionMeasurements_.push_back(data);
    }
  }
}

// Loop to process GPS measurements.
void HybridVio::gpsConsumerLoop() {
}

// Loop to process magnetometer measurements.
void HybridVio::magnetometerConsumerLoop() {
}

// Loop to process differential pressure measurements.
void HybridVio::differentialConsumerLoop() {
}

// Loop that visualizes completed frames.
void HybridVio::visualizationLoop() {
  okvis::VioVisualizer visualizer_(parameters_);
  for (;;) {
    VioVisualizer::VisualizationData::Ptr new_data;
    if (visualizationData_.PopBlocking(&new_data) == false)
      return;
    // visualizer_.showDebugImages(new_data);
    std::vector<cv::Mat> out_images(parameters_.nCameraSystem.numCameras());
    for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
      out_images[i] = visualizer_.drawMatches(new_data, i);
    }
    displayImages_.PushNonBlockingDroppingIfFull(out_images, 1);
  }
}

// trigger display (needed because OSX won't allow threaded display)
void HybridVio::display() {
  std::vector<cv::Mat> out_images;
  if (displayImages_.Size() == 0)
    return;
  if (displayImages_.PopBlocking(&out_images) == false)
    return;
  // draw
  for (size_t im = 0; im < parameters_.nCameraSystem.numCameras(); im++) {
    std::stringstream windowname;
    windowname << "OKVIS camera " << im;
    cv::imshow(windowname.str(), out_images[im]);
  }
  cv::waitKey(1);
}

// Get a subset of the recorded IMU measurements.
okvis::ImuMeasurementDeque HybridVio::getImuMeasurments(
    okvis::Time& imuDataBeginTime, okvis::Time& imuDataEndTime) {
  // sanity checks:
  // if end time is smaller than begin time, return empty queue.
  // if begin time is larger than newest imu time, return empty queue.
  if (imuDataEndTime < imuDataBeginTime
      || imuDataBeginTime > imuMeasurements_.back().timeStamp)
    return okvis::ImuMeasurementDeque();

  std::lock_guard<std::mutex> lock(imuMeasurements_mutex_);
  // get iterator to imu data before previous frame
  okvis::ImuMeasurementDeque::iterator first_imu_package = imuMeasurements_
      .begin();
  okvis::ImuMeasurementDeque::iterator last_imu_package =
      imuMeasurements_.end();
  // TODO(jhuai): go backwards through queue. Is probably faster.
  for (auto iter = imuMeasurements_.begin(); iter != imuMeasurements_.end();
      ++iter) {
    // move first_imu_package iterator back until iter->timeStamp is higher
    // than requested begintime
    if (iter->timeStamp <= imuDataBeginTime)
      first_imu_package = iter;

    // set last_imu_package iterator as soon as we hit first timeStamp higher
    // than requested endtime & break
    if (iter->timeStamp >= imuDataEndTime) {
      last_imu_package = iter;
      // since we want to include this last imu measurement in returned Deque
      // we increase last_imu_package iterator once.
      ++last_imu_package;
      break;
    }
  }

  // create copy of imu buffer
  return okvis::ImuMeasurementDeque(first_imu_package, last_imu_package);
}

// Remove IMU measurements from the internal buffer.
int HybridVio::deleteImuMeasurements(const okvis::Time& eraseUntil) {
  std::lock_guard<std::mutex> lock(imuMeasurements_mutex_);
  if (imuMeasurements_.front().timeStamp > eraseUntil)
    return 0;

  okvis::ImuMeasurementDeque::iterator eraseEnd;
  int removed = 0;
  for (auto it = imuMeasurements_.begin(); it != imuMeasurements_.end(); ++it) {
    eraseEnd = it;
    if (it->timeStamp >= eraseUntil)
      break;
    ++removed;
  }

  imuMeasurements_.erase(imuMeasurements_.begin(), eraseEnd);

  return removed;
}

// Loop that performs the optimization and marginalisation.
void HybridVio::optimizationLoop() {
  TimerSwitchable optimizationTimer("3.1 optimization", true);
  TimerSwitchable marginalizationTimer("3.2 marginalization", true);
  TimerSwitchable afterOptimizationTimer("3.3 afterOptimization", true);

  for (;;) {
    std::shared_ptr<okvis::MultiFrame> frame_pairs;
    VioVisualizer::VisualizationData::Ptr visualizationDataPtr;
    okvis::Time deleteImuMeasurementsUntil(0, 0);
    if (matchedFrames_.PopBlocking(&frame_pairs) == false)
      return;

    OptimizationResults result;
    {
      std::lock_guard<std::mutex> l(estimator_mutex_);
      optimizationTimer.start();

      okvis::Time opt_frame_time = frame_pairs->timestamp();
      int frameIdInSource = -1;
      bool isKF = false;
      estimator_.getFrameId(frame_pairs->id(), frameIdInSource, isKF);
      printf("Optimizing at frame id %d at %d.%d\n", frameIdInSource,
             opt_frame_time.sec, opt_frame_time.nsec);

      estimator_.optimize(false);

      optimizationTimer.stop();

      deleteImuMeasurementsUntil =
          estimator_.firstStateTimestamp() - temporal_imu_data_overlap;

      marginalizationTimer.start();
      estimator_.applyMarginalizationStrategy();
      marginalizationTimer.stop();
      afterOptimizationTimer.start();

      // now actually remove measurements
      deleteImuMeasurements(deleteImuMeasurementsUntil);

      // saving optimized state and saving it in OptimizationResults struct
      {
        std::lock_guard<std::mutex> lock(lastState_mutex_);
        estimator_.get_T_WS(frame_pairs->id(), lastOptimized_T_WS_);
        estimator_.getSpeedAndBias(frame_pairs->id(), 0,
                                   lastOptimizedSpeedAndBiases_);
        lastOptimizedStateTimestamp_ = frame_pairs->timestamp();

        int frameIdInSource = -1;
        bool isKF = false;
        estimator_.getFrameId(frame_pairs->id(), frameIdInSource, isKF);
        // if we publish the state after each IMU propagation we do not need
        // to publish it here.
        if (!parameters_.publishing.publishImuPropagatedState) {
          result.T_WS = lastOptimized_T_WS_;
          result.speedAndBiases = lastOptimizedSpeedAndBiases_;
          result.stamp = lastOptimizedStateTimestamp_;
          result.onlyPublishLandmarks = false;
          result.frameIdInSource = frameIdInSource;
          result.isKeyframe = isKF;
          estimator_.getTgTsTaEstimate(result.vTgTsTa_);
          // returned poseId by getCameraCalibrationEstimate should be
          // frame_pairs->id()
          estimator_.getCameraCalibrationEstimate(result.vfckptdr_);
          estimator_.getVariance(result.vVariance_);
        } else {
          result.onlyPublishLandmarks = true;
        }
        estimator_.getLandmarks(result.landmarksVector);

        repropagationNeeded_ = true;
      }

      if (parameters_.visualization.displayImages) {
        // fill in information that requires access to estimator.
        visualizationDataPtr = VioVisualizer::VisualizationData::Ptr(
            new VioVisualizer::VisualizationData());
        visualizationDataPtr->observations.resize(frame_pairs->numKeypoints());
        okvis::MapPoint landmark;
        okvis::ObservationVector::iterator it = visualizationDataPtr
            ->observations.begin();
        for (size_t camIndex = 0; camIndex < frame_pairs->numFrames();
            ++camIndex) {
          for (size_t k = 0; k < frame_pairs->numKeypoints(camIndex); ++k) {
            OKVIS_ASSERT_TRUE_DBG(
                Exception, it != visualizationDataPtr->observations.end(),
                "Observation-vector not big enough");
            it->keypointIdx = k;
            frame_pairs->getKeypoint(camIndex, k, it->keypointMeasurement);
            frame_pairs->getKeypointSize(camIndex, k, it->keypointSize);
            it->cameraIdx = camIndex;
            it->frameId = frame_pairs->id();
            it->landmarkId = frame_pairs->landmarkId(camIndex, k);
            if (estimator_.isLandmarkAdded(it->landmarkId)) {
              estimator_.getLandmark(it->landmarkId, landmark);
              it->landmark_W = landmark.pointHomog;
              if (estimator_.isLandmarkInitialized(it->landmarkId))
                it->isInitialized = true;
              else
                it->isInitialized = false;
            } else {
              // set to infinity to tell visualizer that landmark is not added
              it->landmark_W = Eigen::Vector4d(0, 0, 0, 0);
            }
            ++it;
          }
        }
        visualizationDataPtr->keyFrames = estimator_.multiFrame(
            estimator_.currentKeyframeId());
        estimator_.get_T_WS(estimator_.currentKeyframeId(),
                            visualizationDataPtr->T_WS_keyFrame);
      }

      optimizationDone_ = true;
    }  // unlock mutex
    optimizationNotification_.notify_all();

    if (!parameters_.publishing.publishImuPropagatedState) {
      // adding further elements to result that do not access estimator.
      for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
          okvis::kinematics::Transformation T_SCA;
          estimator_.getCameraSensorStates(frame_pairs->id(), i, T_SCA);
          result.vector_of_T_SCi.push_back(T_SCA);
      }
    }
    optimizationResults_.Push(result);

    // adding further elements to visualization data that do not
    // access estimator
    if (parameters_.visualization.displayImages) {
      visualizationDataPtr->currentFrames = frame_pairs;
      visualizationData_.PushNonBlockingDroppingIfFull(visualizationDataPtr, 1);
    }
    afterOptimizationTimer.stop();
  }
}

// Loop that publishes the newest state and landmarks.
void HybridVio::publisherLoop() {
  for (;;) {
    // get the result data
    OptimizationResults result;
    if (optimizationResults_.PopBlocking(&result) == false)
      return;

    // call all user callbacks
    if (stateCallback_ && !result.onlyPublishLandmarks)
      stateCallback_(result.stamp, result.T_WS);
    if (!result.onlyPublishLandmarks && result.isKeyframe) {
      if (fullStateCallback_)
        fullStateCallback_(result.stamp, result.T_WS, result.speedAndBiases,
                           result.omega_S, result.frameIdInSource);
      else if (fullStateCallbackWithExtrinsics_)
        fullStateCallbackWithExtrinsics_(
            result.stamp, result.T_WS, result.speedAndBiases, result.omega_S,
            result.frameIdInSource, result.vector_of_T_SCi);
      else if (fullStateCallbackWithAllCalibration_)
        fullStateCallbackWithAllCalibration_(
            result.stamp, result.T_WS, result.speedAndBiases, result.omega_S,
            result.frameIdInSource, result.vector_of_T_SCi, result.vTgTsTa_,
            result.vfckptdr_, result.vVariance_);
    }
    if (landmarksCallback_ && !result.landmarksVector.empty())
      // TODO(gohlp): why two maps?
      landmarksCallback_(result.stamp, result.landmarksVector,
                         result.transferredLandmarks);
  }
}

// Set the callback to be called every time a new state is estimated.
void HybridVio::setFullStateCallbackWithAllCalibration(
    const FullStateCallbackWithAllCalibration
        &fullStateCallbackWithAllCalibration) {
  fullStateCallbackWithAllCalibration_ = fullStateCallbackWithAllCalibration;
}

}  // namespace okvis
