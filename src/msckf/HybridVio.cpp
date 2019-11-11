#include <msckf/HybridVio.hpp>

#include <algorithm>
#include <map>
#include <memory>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <msckf/ExtrinsicModels.hpp>
#include <msckf/FeatureTriangulation.hpp>
#include <msckf/MSCKF2.hpp>
#include <msckf/TFVIO.hpp>

#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ImuError.hpp>

DEFINE_int32(
    estimator_algorithm, 1,
    "0 and 1 for okvis optimization paired with ThreadedKFVio. "
    "0 okvis original estimator, 1 okvis general estimator, 2 pavio.\n"
    "4 for msckf, 5 for triangulation free vio, 6 for hybrid filter, "
    "4, 5, and 6 are paired with HybridVio.\n"
    "This flag will overwrite parameters.optimization.algorithm");

DECLARE_int32(feature_tracking_method);


msckf_vio::Feature::OptimizationConfig msckf_vio::Feature::optimization_config;

/// \brief okvis Main namespace of this package.
namespace okvis {

static const int max_camera_input_queue_size = 10;
// overlap of imu data before and after two consecutive frames
// [seconds] if too large, frame consumer loop will be blocked for too long
// by waiting for imu meas
static const okvis::Duration temporal_imu_data_overlap(0.02);

#ifdef USE_MOCK
// Constructor for gmock.
HybridVio::HybridVio(okvis::VioParameters &parameters,
                     okvis::MockVioBackendInterface &estimator,
                     okvis::MockVioFrontendInterface &frontend)
    : speedAndBiases_propagated_(okvis::SpeedAndBiases::Zero()),
      imu_params_(parameters.imu),
      repropagationNeeded_(false),
      frameSynchronizer_(okvis::FrameSynchronizer(parameters)),
      lastAddedImageTimestamp_(okvis::Time(0, 0)),
      optimizationDone_(true),
      estimator_(estimator),
      frontend_(frontend),
      parameters_(parameters),
      maxImuInputQueueSize_(60) {
  init();
}
#else
// Constructor.
HybridVio::HybridVio(okvis::VioParameters &parameters)
    : speedAndBiases_propagated_(okvis::SpeedAndBiases::Zero()),
      imu_params_(parameters.imu),
      repropagationNeeded_(false),
      frameSynchronizer_(okvis::FrameSynchronizer(parameters)),
      lastAddedImageTimestamp_(okvis::Time(0, 0)),
      optimizationDone_(true),
      frontend_(parameters.nCameraSystem.numCameras()),
      parameters_(parameters),
      maxImuInputQueueSize_(2 * max_camera_input_queue_size *
                            parameters.imu.rate /
                            parameters.sensors_information.cameraRate) {
  switch (FLAGS_estimator_algorithm) {
    case 1:
      estimator_.reset(
          new okvis::MSCKF2());
      break;
    case 2:
      estimator_.reset(new okvis::TFVIO());
      break;
    default:
      LOG(WARNING) << "There are bugs inside the present HybridFilter!";
      estimator_.reset(new okvis::HybridFilter());
      break;
  }
  estimator_->resetInitialPVandStd(
      InitialPVandStd(parameters.initialState));
  msckf_vio::Feature::optimization_config.translation_threshold =
      parameters.optimization.triangulationTranslationThreshold;
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
  lastOptimizedStateTimestamp_ = okvis::Time(0.0) + HybridFilter::half_window_;
  lastOptimizedTimeDelay_ = okvis::Duration(parameters_.imu.td0);
  // s.t. last_timestamp_ - overlap >= 0 (since okvis::time(-0.02)
  // returns big number)
  lastAddedStateTimestamp_ = okvis::Time(0.0) + HybridFilter::half_window_;

  estimator_->addImu(parameters_.imu);
  for (size_t i = 0; i < numCameras_; ++i) {
    // parameters_.camera_extrinsics is never set (default 0's)...
    // do they ever change?
    estimator_->addCamera(parameters_.camera_extrinsics,
                          parameters_.sensors_information.imageReadoutTime);
    cameraMeasurementsReceived_.emplace_back(
          std::shared_ptr<threadsafe::ThreadSafeQueue<std::shared_ptr<okvis::CameraMeasurement> > >
          (new threadsafe::ThreadSafeQueue<std::shared_ptr<okvis::CameraMeasurement> >()));
  }
  
  // set up windows so things don't crash on Mac OS
  if(parameters_.visualization.displayImages){
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
  positionConsumerThread_ = std::thread(&HybridVio::positionConsumerLoop, this);
  gpsConsumerThread_ = std::thread(&HybridVio::gpsConsumerLoop, this);
  magnetometerConsumerThread_ =
      std::thread(&HybridVio::magnetometerConsumerLoop, this);
  differentialConsumerThread_ =
      std::thread(&HybridVio::differentialConsumerLoop, this);

  // algorithm threads
  visualizationThread_ = std::thread(&HybridVio::visualizationLoop, this);
  optimizationThread_ = std::thread(&HybridVio::optimizationLoop, this);
  publisherThread_ = std::thread(&HybridVio::publisherLoop, this);
}

// Destructor. This calls Shutdown() for all threadsafe queues and joins all threads.
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
  estimator_->get_T_WS(estimator_->currentFrameId(), endPosition);
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
bool HybridVio::addKeypoints(const okvis::Time & /*stamp*/,
                             size_t /*cameraIndex*/,
                             const std::vector<cv::KeyPoint> & /*keypoints*/,
                             const std::vector<uint64_t> & /*landmarkIds*/,
                             const cv::Mat & /*descriptors*/,
                             bool * /*asKeyframe*/) {
  OKVIS_THROW(Exception,
              "HybridVio::addKeypoints() not implemented anymore "
              "since changes to _keypointMeasurements queue.");
  return false;
}

// Add an IMU measurement.
bool HybridVio::addImuMeasurement(const okvis::Time &stamp,
                                  const Eigen::Vector3d &alpha,
                                  const Eigen::Vector3d &omega) {
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
    return imuMeasurementsReceived_.Size() == 1;
  }
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
void HybridVio::addGpsMeasurement(const okvis::Time &, double, double, double,
                                  const Eigen::Vector3d &,
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
void HybridVio::addDifferentialPressureMeasurement(const okvis::Time &, double,
                                                   double) {
  OKVIS_THROW(Exception, "Differential pressure measurements not supported")
}

// Set the blocking variable that indicates whether the addMeasurement() functions
// should return immediately (blocking=false), or only when the processing is complete.
void HybridVio::setBlocking(bool blocking) {
  blocking_ = blocking;
  // disable time limit for optimization
  if(blocking_) {
    std::lock_guard<std::mutex> lock(estimator_mutex_);
    estimator_->setOptimizationTimeLimit(-1.0,parameters_.optimization.max_iterations);
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
    if (cameraMeasurementsReceived_[cameraIndex]->PopBlocking(&frame) == false) {
      return;
    }
    beforeDetectTimer.start();
    {  // lock the frame synchronizer
      waitForFrameSynchronizerMutexTimer.start();
      std::lock_guard<std::mutex> lock(frameSynchronizer_mutex_);
      waitForFrameSynchronizerMutexTimer.stop();
      // add new frame to frame synchronizer and get the MultiFrame containing it
      addNewFrameToSynchronizerTimer.start();
      multiFrame = frameSynchronizer_.addNewFrame(frame);
      addNewFrameToSynchronizerTimer.stop();
    }  // unlock frameSynchronizer only now as we can be sure that not two states are added for the same timestamp
    okvis::kinematics::Transformation T_WS;
    okvis::Time lastTimestamp;
    okvis::SpeedAndBiases speedAndBiases;
    okvis::Duration lastTimeDelay;
    // copy last state variables
    {
      waitForStateVariablesMutexTimer.start();
      std::lock_guard<std::mutex> lock(lastState_mutex_);
      waitForStateVariablesMutexTimer.stop();
      T_WS = lastOptimized_T_WS_;
      speedAndBiases = lastOptimizedSpeedAndBiases_;
      lastTimestamp = lastOptimizedStateTimestamp_;
      lastTimeDelay = lastOptimizedTimeDelay_;
    }

    // -- get relevant imu messages for new state
    okvis::Time imuDataEndTime =
        multiFrame->timestamp() + temporal_imu_data_overlap;
    okvis::Time imuDataBeginTime = lastTimestamp - HybridFilter::half_window_;

    OKVIS_ASSERT_TRUE_DBG(Exception, imuDataBeginTime < imuDataEndTime,
                          "imu data end time is smaller than begin time.");

    // wait until all relevant imu messages have arrived and check for termination request
    if (imuFrameSynchronizer_.waitForUpToDateImuData(
      okvis::Time(imuDataEndTime)) == false)  {
      return;
    }
    OKVIS_ASSERT_TRUE_DBG(Exception,
                          imuDataEndTime < imuMeasurements_.back().timeStamp,
                          "Waiting for up to date imu data seems to have failed!");

    okvis::ImuMeasurementDeque imuData = getImuMeasurements(
        imuDataBeginTime, imuDataEndTime, imuMeasurements_, &imuMeasurements_mutex_);

    // if imu_data is empty, either end_time > begin_time or
    // no measurements in timeframe, should not happen, as we waited for measurements
    if (imuData.size() == 0) {
      beforeDetectTimer.stop();
      continue;
    }

    if (imuData.front().timeStamp > frame->timeStamp) {
      LOG(WARNING) << "Frame is newer than oldest IMU measurement. Dropping it.";
      beforeDetectTimer.stop();
      continue;
    }

    // get T_WC(camIndx) for detectAndDescribe()
    if (estimator_->numFrames() == 0) {
      // first frame ever
      bool success = okvis::HybridFilter::initPoseFromImu(imuData, T_WS);
      LOG(INFO) << "estimator initialized using inertial data ";
      {
        std::lock_guard<std::mutex> lock(lastState_mutex_);
        lastOptimized_T_WS_ = T_WS;
        lastOptimizedSpeedAndBiases_.setZero();
        lastOptimizedSpeedAndBiases_.segment<3>(3) = imu_params_.g0;
        lastOptimizedSpeedAndBiases_.segment<3>(6) = imu_params_.a0;
        lastOptimizedStateTimestamp_ = multiFrame->timestamp();
        lastOptimizedTimeDelay_ = okvis::Duration(parameters_.imu.td0);
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
                                          speedAndBiases, lastTimestamp + lastTimeDelay,
                                          multiFrame->timestamp() + lastTimeDelay);
      propagationTimer.stop();
    }
    okvis::kinematics::Transformation T_WC = T_WS
        * (*parameters_.nCameraSystem.T_SC(frame->sensorId));
    beforeDetectTimer.stop();

    if (FLAGS_feature_tracking_method != 1) {
      detectTimer.start();
      frontend_.detectAndDescribe(frame->sensorId, multiFrame, T_WC, nullptr);
      detectTimer.stop();
    }

    afterDetectTimer.start();

    bool push = false;
    {  // we now tell frame synchronizer that detectAndDescribe is done for MF with our timestamp
      waitForFrameSynchronizerMutexTimer2.start();
      std::lock_guard<std::mutex> lock(frameSynchronizer_mutex_);
      waitForFrameSynchronizerMutexTimer2.stop();
      frameSynchronizer_.detectionEndedForMultiFrame(multiFrame->id());

      if (frameSynchronizer_.detectionCompletedForAllCameras(
          multiFrame->id())) {
//        LOG(INFO) << "detection completed for multiframe with id "<< multi_frame->id();
        push = true;
      }
    }  // unlocking frame synchronizer
    afterDetectTimer.stop();
    if (push) {
      // use queue size 1 to propagate a congestion to the _cameraMeasurementsReceived queue
      // and check for termination request
      waitForMatchingThreadTimer.start();
      if (keypointMeasurements_.PushBlockingIfFull(multiFrame, 1) == false) {
        waitForMatchingThreadTimer.stop();
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
    okvis::Time imuDataBeginTime =
        lastAddedStateTimestamp_ - HybridFilter::half_window_;
    if (imuDataBeginTime.toSec() == 0.0) {  // first state not yet added
      imuDataBeginTime = frame->timestamp() - HybridFilter::half_window_;
    }
    // at maximum Duration(.) sec of data is allowed
    if (imuDataEndTime - imuDataBeginTime > Duration(8)) {
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
                              << " (id: " << estimator_->currentFrameId());

    // wait until all relevant imu messages have arrived and check for termination request
    if (imuFrameSynchronizer_.waitForUpToDateImuData(
        okvis::Time(imuDataEndTime)) == false)
      return; OKVIS_ASSERT_TRUE_DBG(Exception,
        imuDataEndTime < imuMeasurements_.back().timeStamp,
        "Waiting for up to date imu data seems to have failed!");

    okvis::ImuMeasurementDeque imuData = getImuMeasurements(
        imuDataBeginTime, imuDataEndTime, imuMeasurements_, &imuMeasurements_mutex_);

    prepareToAddStateTimer.stop();
    // if imu_data is empty, either end_time > begin_time or
    // no measurements in timeframe, should not happen, as we waited for measurements
    if (imuData.size() == 0)
      continue;

    // make sure that optimization of last frame is over.
    // TODO If we didn't actually 'pop' the _matchedFrames queue until after optimization this would not be necessary
    {
      waitForOptimizationTimer.start();
      std::unique_lock<std::mutex> l(estimator_mutex_);
      while (!optimizationDone_)
        optimizationNotification_.wait(l);
      waitForOptimizationTimer.stop();
      addStateTimer.start();
      okvis::Time t0Matching = okvis::Time::now();
      bool asKeyframe = true;  // for msckf always set a frame as keyframe
      if (estimator_->addStates(frame, imuData, asKeyframe)) {
        lastAddedStateTimestamp_ = frame->timestamp();
        addStateTimer.stop();
      } else {
        LOG(ERROR) << "Failed to add state! will drop multiframe.";
        addStateTimer.stop();
        continue;
      }

      // -- matching keypoints, initialising landmarks etc.
      okvis::kinematics::Transformation T_WS;
      estimator_->get_T_WS(frame->id(), T_WS);
      matchingTimer.start();
      frontend_.dataAssociationAndInitialization(*estimator_, T_WS, parameters_,
                                                 map_, frame, &asKeyframe);
      matchingTimer.stop();
      if (asKeyframe)
        estimator_->setKeyframe(frame->id(), asKeyframe);
      if(!blocking_) {
        double timeLimit = parameters_.optimization.timeLimitForMatchingAndOptimization
                           -(okvis::Time::now()-t0Matching).toSec();
        estimator_->setOptimizationTimeLimit(std::max<double>(0.0, timeLimit),
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
  TimerSwitchable processImuTimer("0 processImuMeasurements",true);
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

      if (parameters_.publishing.publishImuPropagatedState) {
        if (!repropagationNeeded_ && imuMeasurements_.size() > 0) {
          start = imuMeasurements_.back().timeStamp;
        } else if (repropagationNeeded_) {
          std::lock_guard<std::mutex> lastStateLock(lastState_mutex_);
          start = lastOptimizedStateTimestamp_;
          T_WS_propagated_ = lastOptimized_T_WS_;
          speedAndBiases_propagated_ = lastOptimizedSpeedAndBiases_;
          repropagationNeeded_ = false;
        } else
          start = okvis::Time(0, 0);
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
                            speedAndBiases_propagated_, start, *end, &covariance,
                            &jacobian);
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
      optimizationResults_.PushNonBlockingDroppingIfFull(result,1);
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
void HybridVio::gpsConsumerLoop() {}

// Loop to process magnetometer measurements.
void HybridVio::magnetometerConsumerLoop() {}

// Loop to process differential pressure measurements.
void HybridVio::differentialConsumerLoop() {}

// Loop that visualizes completed frames.
void HybridVio::visualizationLoop() {
  okvis::VioVisualizer visualizer_(parameters_);
  for (;;) {
    VioVisualizer::VisualizationData::Ptr new_data;
    if (visualizationData_.PopBlocking(&new_data) == false)
      return;
    //visualizer_.showDebugImages(new_data);
    std::vector<cv::Mat> out_images(parameters_.nCameraSystem.numCameras());
    for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
      out_images[i] = visualizer_.drawMatches(new_data, i);
    }
	displayImages_.PushNonBlockingDroppingIfFull(out_images,1);
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
      int frameIdInSource = -1;
      bool isKF = false;
      estimator_->getFrameId(frame_pairs->id(), frameIdInSource, isKF);
      estimator_->optimize(1, 1, false);
      optimizationTimer.stop();

      deleteImuMeasurementsUntil =
          estimator_->oldestFrameTimestamp() - temporal_imu_data_overlap;

      marginalizationTimer.start();
      estimator_->setKeyframeRedundancyThresholds(
          parameters_.optimization.translationThreshold,
          parameters_.optimization.rotationThreshold,
          parameters_.optimization.trackingRateThreshold,
          parameters_.optimization.minTrackLength);
      estimator_->applyMarginalizationStrategy(
          parameters_.optimization.numKeyframes,
          parameters_.optimization.numImuFrames, result.transferredLandmarks);
      marginalizationTimer.stop();
      afterOptimizationTimer.start();

      // now actually remove measurements
      deleteImuMeasurements(deleteImuMeasurementsUntil,
                            imuMeasurements_, &imuMeasurements_mutex_);

      // saving optimized state and saving it in OptimizationResults struct
      {
        std::lock_guard<std::mutex> lock(lastState_mutex_);
        estimator_->get_T_WS(frame_pairs->id(), lastOptimized_T_WS_);
        estimator_->getSpeedAndBias(frame_pairs->id(), 0,
                                   lastOptimizedSpeedAndBiases_);
        estimator_->getTimeDelay(frame_pairs->id(), 0, &lastOptimizedTimeDelay_);
        lastOptimizedStateTimestamp_ = frame_pairs->timestamp();
        int frameIdInSource = -1;
        bool isKF= false;
        estimator_->getFrameId(frame_pairs->id(), frameIdInSource, isKF );
        // if we publish the state after each IMU propagation we do not need to publish it here.
        if (!parameters_.publishing.publishImuPropagatedState) {
          result.T_WS = lastOptimized_T_WS_;
          result.speedAndBiases = lastOptimizedSpeedAndBiases_;
          result.stamp = lastOptimizedStateTimestamp_;
          result.onlyPublishLandmarks = false;
          result.frameIdInSource = frameIdInSource;
          result.isKeyframe = isKF;
          for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
            okvis::kinematics::Transformation T_SCA;
            estimator_->getCameraSensorStates(frame_pairs->id(), i, T_SCA);
            result.vector_of_T_SCi.emplace_back(T_SCA);
          }
          estimator_->getTgTsTaEstimate(result.vTgTsTa_);
          estimator_->getCameraCalibrationEstimate(result.vfckptdr_);
          estimator_->getVariance(result.vVariance_);
        } else {
          result.onlyPublishLandmarks = true;
        }
        for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
          int extrinsic_opt_type = estimator_->getCameraExtrinsicOptType(i);
          Eigen::VectorXd optimized_coeffs;
          ExtrinsicModelToParamValues(extrinsic_opt_type,
                                      result.vector_of_T_SCi[i],
                                      &optimized_coeffs);
          result.opt_T_SCi_coeffs.emplace_back(optimized_coeffs);
        }
        estimator_->getLandmarks(result.landmarksVector);

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
            OKVIS_ASSERT_TRUE_DBG(Exception,it != visualizationDataPtr->observations.end(),"Observation-vector not big enough");
            it->keypointIdx = k;
            frame_pairs->getKeypoint(camIndex, k, it->keypointMeasurement);
            frame_pairs->getKeypointSize(camIndex, k, it->keypointSize);
            it->cameraIdx = camIndex;
            it->frameId = frame_pairs->id();
            it->landmarkId = frame_pairs->landmarkId(camIndex, k);
            if (estimator_->isLandmarkAdded(it->landmarkId)) {
              estimator_->getLandmark(it->landmarkId, landmark);
              it->landmark_W = landmark.pointHomog;
              if (estimator_->isLandmarkInitialized(it->landmarkId))
                it->isInitialized = true;
              else
                it->isInitialized = false;
            } else {
              it->landmark_W = Eigen::Vector4d(0, 0, 0, 0);  // set to infinity to tell visualizer that landmark is not added
            }
            ++it;
          }
        }
        visualizationDataPtr->keyFrames = estimator_->multiFrame(
            estimator_->currentKeyframeId());
        estimator_->get_T_WS(estimator_->currentKeyframeId(),
                            visualizationDataPtr->T_WS_keyFrame);
      }
      afterOptimizationTimer.stop();
      optimizationDone_ = true;
    }  // unlock mutex
    optimizationNotification_.notify_all();

    if (!parameters_.publishing.publishImuPropagatedState) {
      // adding further elements to result that do not access estimator.
      for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
//        result.vector_of_T_SCi.push_back(
//            okvis::kinematics::Transformation(
//                *parameters_.nCameraSystem.T_SC(i)));
      }
    }
    optimizationResults_.Push(result);

    // adding further elements to visualization data that do not access estimator
    if (parameters_.visualization.displayImages) {
      visualizationDataPtr->currentFrames = frame_pairs;
      visualizationData_.PushNonBlockingDroppingIfFull(visualizationDataPtr, 1);
    }    
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
            result.frameIdInSource, result.opt_T_SCi_coeffs, result.vTgTsTa_,
            result.vfckptdr_, result.vVariance_);
    }
    if (landmarksCallback_ && !result.landmarksVector.empty())
      // TODO(gohlp): why two maps?
      landmarksCallback_(result.stamp, result.landmarksVector,
                         result.transferredLandmarks);
  }
}

void HybridVio::saveStatistics(const std::string &filename) const {
  std::ofstream stream(filename, std::ios_base::app);
  if (!stream.is_open()) {
    LOG(WARNING) << "error in opening " << filename;
    return;
  }
  estimator_->printTrackLengthHistogram(stream);
  frontend_.printNumFeatureDistribution(stream);
  if (stream.is_open())
    stream.close();
}

}  // namespace okvis
