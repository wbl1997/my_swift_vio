#include <okvis/HybridFrontend.hpp>

#include <brisk/brisk.h>

#include <opencv2/imgproc/imgproc.hpp>

#include <glog/logging.h>

#include <okvis/ceres/ImuError.hpp>
#include <okvis/VioFrameMatchingAlgorithm.hpp>
#include <okvis/IdProvider.hpp>

// cameras and distortions
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>

// Kneip RANSAC
#include <opengv/sac/Ransac.hpp>
#include <okvis/HybridFrameAbsolutePoseSacProblem.hpp>
#include <okvis/HybridFrameRelativePoseSacProblem.hpp>
#include <okvis/HybridFrameRotationOnlySacProblem.hpp>

#include <okvis/HybridFrameRelativeAdapter.hpp>
#include <okvis/HybridFrameNoncentralAbsoluteAdapter.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

// Constructor.
HybridFrontend::HybridFrontend(size_t numCameras, std::string orbTrackOutput)
    : isInitialized_(false),
      numCameras_(numCameras),
      briskDetectionOctaves_(0),
      briskDetectionThreshold_(50.0),
      briskDetectionAbsoluteThreshold_(800.0),
      briskDetectionMaximumKeypoints_(450),
      briskDescriptionRotationInvariance_(true),
      briskDescriptionScaleInvariance_(false),
      briskMatchingThreshold_(60.0),
      matcher_(std::unique_ptr<okvis::DenseMatcher>(new okvis::DenseMatcher(4))),
      keyframeInsertionOverlapThreshold_(0.6),
      keyframeInsertionMatchingRatioThreshold_(0.2),
      myAccumulator(boost::accumulators::tag::density::num_bins = 20, boost::accumulators::tag::density::cache_size = 40)
{
  // create mutexes for feature detectors and descriptor extractors
  for (size_t i = 0; i < numCameras_; ++i) {
    featureDetectorMutexes_.push_back(
        std::unique_ptr<std::mutex>(new std::mutex()));
  }
  initialiseBriskFeatureDetectors();

  pTracker = new TrackResultReader(orbTrackOutput);
}

// Detection and descriptor extraction on a per image basis.
bool HybridFrontend::detectAndDescribe(size_t cameraIndex,
                                 std::shared_ptr<okvis::MultiFrame> frameOut,
                                 const okvis::kinematics::Transformation& T_WC,
                                 const std::vector<cv::KeyPoint> * keypoints) {
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIndex < numCameras_, "Camera index exceeds number of cameras.");
  std::lock_guard<std::mutex> lock(*featureDetectorMutexes_[cameraIndex]);

  // check there are no keypoints here
  OKVIS_ASSERT_TRUE(Exception, keypoints == nullptr, "external keypoints currently not supported")

  frameOut->setDetector(cameraIndex, featureDetectors_[cameraIndex]);
  frameOut->setExtractor(cameraIndex, descriptorExtractors_[cameraIndex]);

  frameOut->detect(cameraIndex);

  // ExtractionDirection == gravity direction in camera frame
  Eigen::Vector3d g_in_W(0, 0, -1);
  Eigen::Vector3d extractionDirection = T_WC.inverse().C() * g_in_W;
  frameOut->describe(cameraIndex, extractionDirection);

  // set detector/extractor to nullpointer? TODO
  return true;
}

// Matching as well as initialization of landmarks and state.
bool HybridFrontend::dataAssociationAndInitialization(
#ifdef USE_MSCKF2
    okvis::MSCKF2& estimator,
#else
    okvis::HybridFilter& estimator,
#endif
    okvis::kinematics::Transformation& /*T_WS_propagated*/, // TODO sleutenegger: why is this not used here?
    const okvis::VioParameters &params,
    const std::shared_ptr<okvis::MapPointVector> /*map*/, // TODO sleutenegger: why is this not used here?
    std::shared_ptr<okvis::MultiFrame> framesInOut,
    bool *asKeyframe) {
  // match new keypoints to existing landmarks/keypoints
  // initialise new landmarks (states)
  // outlier rejection by consistency check
  // RANSAC (2D2D / 3D2D)
  // decide keyframe
  // left-right stereo match & init

  // find distortion type
  okvis::cameras::NCameraSystem::DistortionType distortionType = params.nCameraSystem
      .distortionType(0);
  for (size_t i = 1; i < params.nCameraSystem.numCameras(); ++i) {
    OKVIS_ASSERT_TRUE(Exception,
                      distortionType == params.nCameraSystem.distortionType(i),
                      "mixed frame types are not supported yet");
  }

#ifdef USE_KLT

  // first frame? (did do addStates before, so 1 frame minimum in estimator)
  if (estimator.numFrames() <= 1)
      *asKeyframe = true;  // first frame needs to be keyframe

  bool rotationOnly = false;

  if (!isInitialized_) {
      if (!rotationOnly) {
          isInitialized_ = true;
          LOG(INFO) << "Initialized!";
      }
  }

  // keyframe decision, at the moment only landmarks that match with keyframe are initialised
  *asKeyframe = *asKeyframe || doWeNeedANewKeyframe(estimator, framesInOut);

  // match to last frame
  TimerSwitchable matchToLastFrameTimer("2.4.2 matchToLastFrame");
  switch (distortionType) {
  case okvis::cameras::NCameraSystem::RadialTangential: {
      matchToLastFrame<
              VioFrameMatchingAlgorithm<
              okvis::cameras::PinholeCamera<
              okvis::cameras::RadialTangentialDistortion> > >(
                  estimator, params, framesInOut->id(),
                  false);
      break;
  }
  case okvis::cameras::NCameraSystem::Equidistant: {
      matchToLastFrame<
              VioFrameMatchingAlgorithm<
              okvis::cameras::PinholeCamera<
              okvis::cameras::EquidistantDistortion> > >(
                  estimator, params, framesInOut->id(),
                  false);
      break;
  }
  case okvis::cameras::NCameraSystem::RadialTangential8: {
      matchToLastFrame<
              VioFrameMatchingAlgorithm<
              okvis::cameras::PinholeCamera<
              okvis::cameras::RadialTangentialDistortion8> > >(
                  estimator, params, framesInOut->id(),
                  false);

      break;
  }
  default:
      OKVIS_THROW(Exception, "Unsupported distortion type.")
              break;
  }
  matchToLastFrameTimer.stop();

#elif defined(USE_BRISK)
  // first frame? (did do addStates before, so 1 frame minimum in estimator)
  if (estimator.numFrames() > 1) {
    bool rotationOnly = false;


    if (!isInitialized_) {
      if (!rotationOnly) {
        isInitialized_ = true;
        LOG(INFO) << "Initialized!";
      }
    }

    // keyframe decision, at the moment only landmarks that match with keyframe are initialised
    *asKeyframe = *asKeyframe || doWeNeedANewKeyframe(estimator, framesInOut);

    // match to last frame
    TimerSwitchable matchToLastFrameTimer("2.4.2 matchToLastFrame");
    switch (distortionType) {
      case okvis::cameras::NCameraSystem::RadialTangential: {
        matchToLastFrame<
            VioFrameMatchingAlgorithm<
                okvis::cameras::PinholeCamera<
                    okvis::cameras::RadialTangentialDistortion> > >(
            estimator, params, framesInOut->id(),
            false);
        break;
      }
      case okvis::cameras::NCameraSystem::Equidistant: {
        matchToLastFrame<
            VioFrameMatchingAlgorithm<
                okvis::cameras::PinholeCamera<
                    okvis::cameras::EquidistantDistortion> > >(
            estimator, params, framesInOut->id(),
            false);
        break;
      }
      case okvis::cameras::NCameraSystem::RadialTangential8: {
        matchToLastFrame<
            VioFrameMatchingAlgorithm<
                okvis::cameras::PinholeCamera<
                    okvis::cameras::RadialTangentialDistortion8> > >(
            estimator, params, framesInOut->id(),
            false);

        break;
      }
      default:
        OKVIS_THROW(Exception, "Unsupported distortion type.")
        break;
    }
    matchToLastFrameTimer.stop();
  } else
    *asKeyframe = true;  // first frame needs to be keyframe
#else //use ORB_VO

  // first frame? (did do addStates before, so 1 frame minimum in estimator)
//  if (estimator.numFrames() <= 1)
  assert(*asKeyframe == true);  // first frame needs to be keyframe
  bool rotationOnly = false;
  if (!isInitialized_) {
      if (!rotationOnly) {
          isInitialized_ = true;
          LOG(INFO) << "Initialized!";
      }
  }

  // match to last frame
  TimerSwitchable matchToLastFrameTimer("2.4.2 matchToLastFrame");
  switch (distortionType) {
  case okvis::cameras::NCameraSystem::RadialTangential: {
      matchToLastFrame<
              VioFrameMatchingAlgorithm<
              okvis::cameras::PinholeCamera<
              okvis::cameras::RadialTangentialDistortion> > >(
                  estimator, params, framesInOut->id(),
                  false);
      break;
  }
  case okvis::cameras::NCameraSystem::Equidistant: {
      matchToLastFrame<
              VioFrameMatchingAlgorithm<
              okvis::cameras::PinholeCamera<
              okvis::cameras::EquidistantDistortion> > >(
                  estimator, params, framesInOut->id(),
                  false);
      break;
  }
  case okvis::cameras::NCameraSystem::RadialTangential8: {
      matchToLastFrame<
              VioFrameMatchingAlgorithm<
              okvis::cameras::PinholeCamera<
              okvis::cameras::RadialTangentialDistortion8> > >(
                  estimator, params, framesInOut->id(),
                  false);

      break;
  }
  default:
      OKVIS_THROW(Exception, "Unsupported distortion type.")
              break;
  }
  matchToLastFrameTimer.stop();

#endif
  return true;
}

// Propagates pose, speeds and biases with given IMU measurements.
bool HybridFrontend::propagation(const okvis::ImuMeasurementDeque & imuMeasurements,
                           const okvis::ImuParameters & imuParams,
                           okvis::kinematics::Transformation& T_WS_propagated,
                           okvis::SpeedAndBiases & speedAndBiases,
                           const okvis::Time& t_start, const okvis::Time& t_end,
                           Eigen::Matrix<double, 15, 15>* covariance,
                           Eigen::Matrix<double, 15, 15>* jacobian) const {
  if (imuMeasurements.size() < 2) {
    LOG(WARNING)
        << "- Skipping propagation as only one IMU measurement has been given to HybridFrontend."
        << " Normal when starting up.";
    return 0;
  }
  //huai: this is not replaced by IMUOdometry function because it performs good for propagating only states,
  // and it does not use imuErrorModel
  int measurements_propagated = okvis::ceres::ImuError::propagation(
              imuMeasurements, imuParams, T_WS_propagated, speedAndBiases, t_start,
      t_end, covariance, jacobian);

  return measurements_propagated > 0;
}

// Decision whether a new frame should be keyframe or not.
bool HybridFrontend::doWeNeedANewKeyframe(
#ifdef USE_MSCKF2
    const okvis::MSCKF2& estimator,
#else
    const okvis::HybridFilter& estimator,
#endif
    std::shared_ptr<okvis::MultiFrame> currentFrame) {

  if (estimator.numFrames() < 2) {
    // just starting, so yes, we need this as a new keyframe
    return true;
  }

  if (!isInitialized_)
    return false;

  double overlap = 0.0;
  double ratio = 0.0;

  // go through all the frames and try to match the initialized keypoints
  for (size_t im = 0; im < currentFrame->numFrames(); ++im) {

    // get the hull of all keypoints in current frame
    std::vector<cv::Point2f> frameBPoints, frameBHull;
    std::vector<cv::Point2f> frameBMatches, frameBMatchesHull;
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > frameBLandmarks;

    const size_t numB = currentFrame->numKeypoints(im);
    frameBPoints.reserve(numB);
    frameBLandmarks.reserve(numB);
    Eigen::Vector2d keypoint;
    for (size_t k = 0; k < numB; ++k) {
      currentFrame->getKeypoint(im, k, keypoint);
      // insert it
      frameBPoints.push_back(cv::Point2f(keypoint[0], keypoint[1]));
      // also remember matches
      if (currentFrame->landmarkId(im, k) != 0) {
        frameBMatches.push_back(cv::Point2f(keypoint[0], keypoint[1]));
      }
    }

    if (frameBPoints.size() < 3)
      continue;
    cv::convexHull(frameBPoints, frameBHull);
    if (frameBMatches.size() < 3)
      continue;
    cv::convexHull(frameBMatches, frameBMatchesHull);

    // areas
    double frameBArea = cv::contourArea(frameBHull);
    double frameBMatchesArea = cv::contourArea(frameBMatchesHull);

    // overlap area
    double overlapArea = frameBMatchesArea / frameBArea;
    // matching ratio inside overlap area: count
    int pointsInFrameBMatchesArea = 0;
    if (frameBMatchesHull.size() > 2) {
      for (size_t k = 0; k < frameBPoints.size(); ++k) {
        if (cv::pointPolygonTest(frameBMatchesHull, frameBPoints[k], false)
            > 0) {
          pointsInFrameBMatchesArea++;
        }
      }
    }
    double matchingRatio = double(frameBMatches.size())
        / double(pointsInFrameBMatchesArea);

    // calculate overlap score
    overlap = std::max(overlapArea, overlap);
    ratio = std::max(matchingRatio, ratio);
  }

  // take a decision
  if (overlap > keyframeInsertionOverlapThreshold_
      && ratio > keyframeInsertionMatchingRatioThreshold_)
    return false;
  else
    return true;
}

// Match a new multiframe to existing keyframes
template<class MATCHING_ALGORITHM>
int HybridFrontend::matchToKeyframes(
#ifdef USE_MSCKF2
    okvis::MSCKF2& estimator,
#else
    okvis::HybridFilter& estimator,
#endif
                               const okvis::VioParameters & params,
                               const uint64_t currentFrameId,
                               bool& rotationOnly,
                               bool usePoseUncertainty,
                               double* uncertainMatchFraction,
                               bool removeOutliers) {
  rotationOnly = true;
  if (estimator.numFrames() < 2) {
    // just starting, so yes, we need this as a new keyframe
    return 0;
  }

  int retCtr = 0;
  int numUncertainMatches = 0;

  // go through all the frames and try to match the initialized keypoints
  size_t kfcounter = 0;
  for (size_t age = 1; age < estimator.numFrames(); ++age) {
    uint64_t olderFrameId = estimator.frameIdByAge(age);
    if (!estimator.isKeyframe(olderFrameId))
      continue;
    for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {
      MATCHING_ALGORITHM matchingAlgorithm(estimator,
                                           MATCHING_ALGORITHM::Match3D2D,
                                           briskMatchingThreshold_,
                                           usePoseUncertainty);
      matchingAlgorithm.setFrames(olderFrameId, currentFrameId, im, im);

      // match 3D-2D
      matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
      retCtr += matchingAlgorithm.numMatches();
      numUncertainMatches += matchingAlgorithm.numUncertainMatches();

    }
    kfcounter++;
    if (kfcounter > 2)
      break;
  }

  kfcounter = 0;
  bool firstFrame = true;
  for (size_t age = 1; age < estimator.numFrames(); ++age) {
    uint64_t olderFrameId = estimator.frameIdByAge(age);
    if (!estimator.isKeyframe(olderFrameId))
      continue;
    for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {
      MATCHING_ALGORITHM matchingAlgorithm(estimator,
                                           MATCHING_ALGORITHM::Match2D2D,
                                           briskMatchingThreshold_,
                                           usePoseUncertainty);
      matchingAlgorithm.setFrames(olderFrameId, currentFrameId, im, im);

      // match 2D-2D for initialization of new (mono-)correspondences
      matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
      retCtr += matchingAlgorithm.numMatches();
      numUncertainMatches += matchingAlgorithm.numUncertainMatches();
    }

    // remove outliers
    // only do RANSAC 3D2D with most recent KF
    if (kfcounter == 0 && isInitialized_)
      runRansac3d2d(estimator, params.nCameraSystem,
                    estimator.multiFrame(currentFrameId), removeOutliers);

    bool rotationOnly_tmp = false;
    // do RANSAC 2D2D for initialization only
    if (!isInitialized_) {
      runRansac2d2d(estimator, params, currentFrameId, olderFrameId, true,
                    removeOutliers, rotationOnly_tmp);
    }
    if (firstFrame) {
      rotationOnly = rotationOnly_tmp;
      firstFrame = false;
    }

    kfcounter++;
    if (kfcounter > 1)
      break;
  }

  // calculate fraction of safe matches
  if (uncertainMatchFraction) {
    *uncertainMatchFraction = double(numUncertainMatches) / double(retCtr);
  }

  return retCtr;
}

// Match a new multiframe to the last frame.
template<class MATCHING_ALGORITHM>
int HybridFrontend::matchToLastFrame(
#ifdef USE_MSCKF2
    okvis::MSCKF2& estimator,
#else
    okvis::HybridFilter& estimator,
#endif
                               const okvis::VioParameters& params,
                               const uint64_t currentFrameId,
                               bool usePoseUncertainty,
                               bool removeOutliers) {

  int retCtr = 0;
  int matches3d2d(0), inliers3d2d(0), matches2d2d(0), inliers2d2d(0);

#ifdef USE_KLT

  cv::Size winSize(21,21);
  const int LEVELS = 3;
  bool withDerivatives=true;
  cv::Mat currentImage = estimator.multiFrame(currentFrameId)->getFrame(0);
  mPreviousPyramid = mCurrentPyramid;
  mPreviousFrameId = mCurrentFrameId;
  std::cout <<"current frame id "<< mCurrentFrameId<<std::endl;
  mCurrentFrameId = currentFrameId;
  mCurrentPyramid.clear();
  //build up the pyramid for KLT tracking
  // the pyramid often involves padding, even though image has no padding
  int maxLevel = cv::buildOpticalFlowPyramid(currentImage, mCurrentPyramid, winSize, LEVELS-1, withDerivatives);
  std::cout <<"number of frames "<< estimator.numFrames()<<std::endl;
  if (estimator.numFrames() <2) {

      TrailTracking_Start();
      std::shared_ptr<okvis::MultiFrame> frameB = estimator.multiFrame(currentFrameId);
      const size_t camIdB=0;
      frameB->resetKeypoints(camIdB, mvKeyPoints);

      return 0;
  }

  uint64_t lastFrameId = estimator.frameIdByAge(1);
  OKVIS_ASSERT_EQ(Exception, lastFrameId, mPreviousFrameId, "lastFrameId should equal to recorded in Frontend");
  for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {
      // match 2D-2D for initialization of new (mono-)correspondences
      matches2d2d = TrailTracking_Advance(estimator, lastFrameId, currentFrameId, im, im); // update feature tracks

      retCtr += matches2d2d;
      // If there are too few tracked points in the current-frame, some points are added by goodFeaturesToTrack(),
      // and, add this frame as keyframe, updates the tracker's current-frame-KeyFrame struct with any measurements made.
      if(matches2d2d< 0.5*mMaxFeaturesInFrame || (mFrameCounter%4==0 && matches2d2d< 0.6*mMaxFeaturesInFrame))
      {
        std::vector<cv::KeyPoint> vNewKPs;
        TrailTracking_DetectAndInsert(currentImage, matches2d2d, vNewKPs);
        mvKeyPoints.insert(mvKeyPoints.end(), vNewKPs.begin(), vNewKPs.end());
      }

      // create the feature list for the current frame based on welll tracked features
      std::shared_ptr<okvis::MultiFrame> frameB = estimator.multiFrame(currentFrameId);
      frameB->resetKeypoints(im, mvKeyPoints);
      UpdateEstimatorObservations(estimator, lastFrameId, currentFrameId, im, im);
  }
  ++mFrameCounter;
#elif defined(USE_BRISK)
  if (estimator.numFrames() < 2) {
    return 0;
  }

  uint64_t lastFrameId = estimator.frameIdByAge(1);

  // Huai{ comment the following 4 lines because no keyframe matching is performed in msckf
//  if (estimator.isKeyframe(lastFrameId)) {
//    // already done
//    return 0;
//  }
#define MATCH_3D2D
#ifdef MATCH_3D2D
  for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {
    MATCHING_ALGORITHM matchingAlgorithm(estimator,
                                         MATCHING_ALGORITHM::Match3D2D,
                                         briskMatchingThreshold_,
                                         usePoseUncertainty);
    matchingAlgorithm.setFrames(lastFrameId, currentFrameId, im, im);

    // match 3D-2D
    matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
    matches3d2d = matchingAlgorithm.numMatches();
    retCtr += matches3d2d;
  }

  inliers3d2d= runRansac3d2d(estimator, params.nCameraSystem,
                estimator.multiFrame(currentFrameId), removeOutliers);
#endif

  for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {
    MATCHING_ALGORITHM matchingAlgorithm(estimator,
                                         MATCHING_ALGORITHM::Match2D2D,
                                         briskMatchingThreshold_,
                                         usePoseUncertainty);
    matchingAlgorithm.setFrames(lastFrameId, currentFrameId, im, im);

    // match 2D-2D for initialization of new (mono-)correspondences
    if(1)
        matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
    else{
        matchingAlgorithm.doSetup();
        matchingAlgorithm.match();
    }
    matches2d2d = matchingAlgorithm.numMatches();

    retCtr += matches2d2d;
  }

  // remove outliers
  bool rotationOnly = false;
  if (!isInitialized_)
    inliers2d2d = runRansac2d2d(estimator, params, currentFrameId, lastFrameId, false,
                  removeOutliers, rotationOnly);
#else // load saved tracking result by an external module

  std::shared_ptr<okvis::MultiFrame> frameB = estimator.multiFrame(currentFrameId);

  cv::Mat currentImage = frameB->getFrame(0);
  okvis::Time currentTime = frameB->timestamp();

  mPreviousFrameId = mCurrentFrameId;
  std::cout <<"current frame id in frontend "<< mCurrentFrameId <<" timestamp "<<std::setprecision(12)<< currentTime.toSec() <<std::endl;
  mCurrentFrameId = currentFrameId;

  std::cout <<"number of frames used by estimator "<< estimator.numFrames()<<std::endl;

  std::vector<cv::KeyPoint> keypoints; //keypoints in the current image
  std::vector<size_t> mapPointIds; //map point ids in the current image , if not empty should of the same length as keypoints
  std::vector<Eigen::Vector3d> mapPointPositions; //map point positions in the current image, if not empty should of the same length as keypoints

  //if mapPointIds == 0, it means non correspondence
  pTracker->getNextFrame(currentTime.toSec(), keypoints, mapPointIds, mapPointPositions, currentFrameId);
  std::cout <<"#kp, #mp "<< keypoints.size() <<" "<< mapPointIds.size()<<std::endl;
  if(mapPointIds.empty())
  {
      return 0;
  }
  if(!mapPointIds.empty() && mlTrailers.empty()) // should be second keyframe in orb_vo
  {// initialize for the first frame
      size_t nToAdd = 200; //MaxInitialTrails
      mMaxFeaturesInFrame = nToAdd;
      size_t nToThrow = 100;//MinInitialTrails
      mMinFeaturesInFrame = nToThrow;
      mFrameCounter = 0;


      for(size_t i = 0; i<keypoints.size(); i++)
      {
          if(mapPointIds[i]!=0){
            Trailer t(keypoints[i].pt, keypoints[i].pt, 0, i, mapPointIds[i]);
            t.p_W = mapPointPositions[i];
            t.uCurrentFrameId = currentFrameId;
            mlTrailers.push_back(t);
          }
      }
      std::cout <<"Initializing at frameid "<<currentFrameId<<" with new trails "<< mlTrailers.size()<<std::endl;

      const size_t camIdB=0;
      frameB->resetKeypoints(camIdB, keypoints);
      return 0;
  }

  uint64_t lastFrameId = estimator.frameIdByAge(1);
  OKVIS_ASSERT_EQ(Exception, lastFrameId, mPreviousFrameId, "lastFrameId should equal to recorded in Frontend");
  for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {
      // match 2D-2D for initialization of new (mono-)correspondences
      matches2d2d = TrailTracking_Advance2(keypoints, mapPointIds, mapPointPositions, currentFrameId, currentImage); // update feature tracks
      retCtr += matches2d2d;

      frameB->resetKeypoints(im, keypoints);
      UpdateEstimatorObservations2(estimator, lastFrameId, currentFrameId, im, im);
  }
  ++mFrameCounter;
#endif
  std::cout<< "matchToLastFrame matches3d2d, inliers3d2d, matches2d2d "<<
              matches3d2d <<" "<< inliers3d2d <<" "<< matches2d2d << std::endl;
  return retCtr;
}

// Match the frames inside the multiframe to each other to initialise new landmarks.
template<class MATCHING_ALGORITHM>
void HybridFrontend::matchStereo(
#ifdef USE_MSCKF2
    okvis::MSCKF2& estimator,
#else
    okvis::HybridFilter& estimator,
#endif
                           std::shared_ptr<okvis::MultiFrame> multiFrame) {

  const size_t camNumber = multiFrame->numFrames();
  const uint64_t mfId = multiFrame->id();

  for (size_t im0 = 0; im0 < camNumber; im0++) {
    for (size_t im1 = im0 + 1; im1 < camNumber; im1++) {
      // first, check the possibility for overlap
      // FIXME: implement this in the Multiframe...!!

      // check overlap
      if(!multiFrame->hasOverlap(im0, im1)){
        continue;
      }

      MATCHING_ALGORITHM matchingAlgorithm(estimator,
                                           MATCHING_ALGORITHM::Match2D2D,
                                           briskMatchingThreshold_,
                                           false);  // TODO: make sure this is changed when switching back to uncertainty based matching
      matchingAlgorithm.setFrames(mfId, mfId, im0, im1);  // newest frame

      // match 2D-2D
      matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);

      // match 3D-2D
      matchingAlgorithm.setMatchingType(MATCHING_ALGORITHM::Match3D2D);
      matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);

      // match 2D-3D
      matchingAlgorithm.setFrames(mfId, mfId, im1, im0);  // newest frame
      matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
    }
  }

  // TODO: for more than 2 cameras check that there were no duplications!

  // TODO: ensure 1-1 matching.

  // TODO: no RANSAC ?

  for (size_t im = 0; im < camNumber; im++) {
    const size_t ksize = multiFrame->numKeypoints(im);
    for (size_t k = 0; k < ksize; ++k) {
      if (multiFrame->landmarkId(im, k) != 0) {
        continue;  // already identified correspondence
      }
      multiFrame->setLandmarkId(im, k, okvis::IdProvider::instance().newId());
    }
  }
}

// Perform 3D/2D RANSAC.
int HybridFrontend::runRansac3d2d(
#ifdef USE_MSCKF2
    okvis::MSCKF2& estimator,
#else
    okvis::HybridFilter& estimator,
#endif
                            const okvis::cameras::NCameraSystem& nCameraSystem,
                            std::shared_ptr<okvis::MultiFrame> currentFrame,
                            bool removeOutliers) {
  if (estimator.numFrames() < 2) {
    // nothing to match against, we are just starting up.
    return 1;
  }

  /////////////////////
  //   KNEIP RANSAC
  /////////////////////
  int numInliers = 0;

  // absolute pose adapter for Kneip toolchain
  opengv::absolute_pose::HybridFrameNoncentralAbsoluteAdapter adapter(estimator,
                                                                nCameraSystem,
                                                                currentFrame);

  size_t numCorrespondences = adapter.getNumberCorrespondences();
  if (numCorrespondences < 5)
    return numCorrespondences;

  // create a RelativePoseSac problem and RANSAC
  opengv::sac::Ransac<
      opengv::sac_problems::absolute_pose::HybridFrameAbsolutePoseSacProblem> ransac;
  std::shared_ptr<
      opengv::sac_problems::absolute_pose::HybridFrameAbsolutePoseSacProblem> absposeproblem_ptr(
      new opengv::sac_problems::absolute_pose::HybridFrameAbsolutePoseSacProblem(
          adapter,
          opengv::sac_problems::absolute_pose::HybridFrameAbsolutePoseSacProblem::Algorithm::GP3P));
  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ = 9;
  ransac.max_iterations_ = 50;
  // initial guess not needed...
  // run the ransac
  ransac.computeModel(0);

  // assign transformation
  numInliers = ransac.inliers_.size();
  if (numInliers >= 10) {

    // kick out outliers:
    std::vector<bool> inliers(numCorrespondences, false);
    for (size_t k = 0; k < ransac.inliers_.size(); ++k) {
      inliers.at(ransac.inliers_.at(k)) = true;
    }

    for (size_t k = 0; k < numCorrespondences; ++k) {
      if (!inliers[k]) {
        // get the landmark id:
        size_t camIdx = adapter.camIndex(k);
        size_t keypointIdx = adapter.keypointIndex(k);
        uint64_t lmId = currentFrame->landmarkId(camIdx, keypointIdx);

        // reset ID:
        currentFrame->setLandmarkId(camIdx, keypointIdx, 0);

        // remove observation
        if (removeOutliers) {
          estimator.removeObservation(lmId, currentFrame->id(), camIdx,
                                      keypointIdx);
        }
      }
    }
  }
  return numInliers;
}

// Perform 2D/2D RANSAC.
int HybridFrontend::runRansac2d2d(
#ifdef USE_MSCKF2
    okvis::MSCKF2& estimator,
#else
    okvis::HybridFilter& estimator,
#endif

                            const okvis::VioParameters& params,
                            uint64_t currentFrameId, uint64_t olderFrameId,
                            bool initializePose,
                            bool removeOutliers,
                            bool& rotationOnly) {
  // match 2d2d
  rotationOnly = false;
  const size_t numCameras = params.nCameraSystem.numCameras();

  size_t totalInlierNumber = 0;
  bool rotation_only_success = false;
  bool rel_pose_success = false;

  // run relative RANSAC
  for (size_t im = 0; im < numCameras; ++im) {

    // relative pose adapter for Kneip toolchain
    opengv::relative_pose::HybridFrameRelativeAdapter adapter(estimator,
                                                        params.nCameraSystem,
                                                        olderFrameId, im,
                                                        currentFrameId, im);

    size_t numCorrespondences = adapter.getNumberCorrespondences();

    if (numCorrespondences < 10)
      continue;  // won't generate meaningful results. let's hope the few correspondences we have are all inliers!!

    // try both the rotation-only RANSAC and the relative one:

    // create a RelativePoseSac problem and RANSAC
    typedef opengv::sac_problems::relative_pose::HybridFrameRotationOnlySacProblem HybridFrameRotationOnlySacProblem;
    opengv::sac::Ransac<HybridFrameRotationOnlySacProblem> rotation_only_ransac;
    std::shared_ptr<HybridFrameRotationOnlySacProblem> rotation_only_problem_ptr(
        new HybridFrameRotationOnlySacProblem(adapter));
    rotation_only_ransac.sac_model_ = rotation_only_problem_ptr;
    rotation_only_ransac.threshold_ = 9;
    rotation_only_ransac.max_iterations_ = 50;

    // run the ransac
    rotation_only_ransac.computeModel(0);

    // get quality
    int rotation_only_inliers = rotation_only_ransac.inliers_.size();
    float rotation_only_ratio = float(rotation_only_inliers)
        / float(numCorrespondences);

    // now the rel_pose one:
    typedef opengv::sac_problems::relative_pose::HybridFrameRelativePoseSacProblem HybridFrameRelativePoseSacProblem;
    opengv::sac::Ransac<HybridFrameRelativePoseSacProblem> rel_pose_ransac;
    std::shared_ptr<HybridFrameRelativePoseSacProblem> rel_pose_problem_ptr(
        new HybridFrameRelativePoseSacProblem(
            adapter, HybridFrameRelativePoseSacProblem::STEWENIUS));
    rel_pose_ransac.sac_model_ = rel_pose_problem_ptr;
    rel_pose_ransac.threshold_ = 9;     //(1.0 - cos(0.5/600));
    rel_pose_ransac.max_iterations_ = 50;

    // run the ransac
    rel_pose_ransac.computeModel(0);

    // assess success
    int rel_pose_inliers = rel_pose_ransac.inliers_.size();
    float rel_pose_ratio = float(rel_pose_inliers) / float(numCorrespondences);

    // decide on success and fill inliers
    std::vector<bool> inliers(numCorrespondences, false);
    if (rotation_only_ratio > rel_pose_ratio || rotation_only_ratio > 0.8) {
      if (rotation_only_inliers > 10) {
        rotation_only_success = true;
      }
      rotationOnly = true;
      totalInlierNumber += rotation_only_inliers;
      for (size_t k = 0; k < rotation_only_ransac.inliers_.size(); ++k) {
        inliers.at(rotation_only_ransac.inliers_.at(k)) = true;
      }
    } else {
      if (rel_pose_inliers > 10) {
        rel_pose_success = true;
      }
      totalInlierNumber += rel_pose_inliers;
      for (size_t k = 0; k < rel_pose_ransac.inliers_.size(); ++k) {
        inliers.at(rel_pose_ransac.inliers_.at(k)) = true;
      }
    }

    // failure?
    if (!rotation_only_success && !rel_pose_success) {
      continue;
    }

    // otherwise: kick out outliers!
    std::shared_ptr<okvis::MultiFrame> multiFrame = estimator.multiFrame(
        currentFrameId);

    for (size_t k = 0; k < numCorrespondences; ++k) {
      size_t idxB = adapter.getMatchKeypointIdxB(k);
      if (!inliers[k]) {

        uint64_t lmId = multiFrame->landmarkId(im, k);
        // reset ID:
        multiFrame->setLandmarkId(im, k, 0);
        // remove observation
        if (removeOutliers) {
          if (lmId != 0 && estimator.isLandmarkAdded(lmId)){
            estimator.removeObservation(lmId, currentFrameId, im, idxB);
          }
        }
      }
    }

    // initialize pose if necessary
    if (initializePose && !isInitialized_) {
      if (rel_pose_success)
        LOG(INFO)
            << "Initializing pose from 2D-2D RANSAC";
      else
        LOG(INFO)
            << "Initializing pose from 2D-2D RANSAC: orientation only";

      Eigen::Matrix4d T_C1C2_mat = Eigen::Matrix4d::Identity();

      okvis::kinematics::Transformation T_SCA, T_WSA, T_SC0, T_WS0;
      uint64_t idA = olderFrameId;
      uint64_t id0 = currentFrameId;
      estimator.getCameraSensorStates(idA, im, T_SCA);
      estimator.get_T_WS(idA, T_WSA);
      estimator.getCameraSensorStates(id0, im, T_SC0);
      estimator.get_T_WS(id0, T_WS0);
      if (rel_pose_success) {
        // update pose
        // if the IMU is used, this will be quickly optimized to the correct scale. Hopefully.
        T_C1C2_mat.topLeftCorner<3, 4>() = rel_pose_ransac.model_coefficients_;

        //initialize with projected length according to motion prior.

        okvis::kinematics::Transformation T_C1C2 = T_SCA.inverse()
            * T_WSA.inverse() * T_WS0 * T_SC0;
        T_C1C2_mat.topRightCorner<3, 1>() = T_C1C2_mat.topRightCorner<3, 1>()
            * std::max(
                0.0,
                double(
                    T_C1C2_mat.topRightCorner<3, 1>().transpose()
                        * T_C1C2.r()));
      } else {
        // rotation only assigned...
        T_C1C2_mat.topLeftCorner<3, 3>() = rotation_only_ransac
            .model_coefficients_;
      }

      // set.
      estimator.set_T_WS(
          id0,
          T_WSA * T_SCA * okvis::kinematics::Transformation(T_C1C2_mat)
              * T_SC0.inverse());
    }
  }

  if (rel_pose_success || rotation_only_success)
    return totalInlierNumber;
  else {
    rotationOnly = true;  // hack...
    return -1;
  }

  return 0;
}

// (re)instantiates feature detectors and descriptor extractors. Used after settings changed or at startup.
void HybridFrontend::initialiseBriskFeatureDetectors() {
  for (auto it = featureDetectorMutexes_.begin();
      it != featureDetectorMutexes_.end(); ++it) {
    (*it)->lock();
  }
  featureDetectors_.clear();
  descriptorExtractors_.clear();
  for (size_t i = 0; i < numCameras_; ++i) {
    featureDetectors_.push_back(
        std::shared_ptr<cv::FeatureDetector>(
#ifdef __ARM_NEON__
            new cv::GridAdaptedFeatureDetector( 
            new cv::FastFeatureDetector(briskDetectionThreshold_),
                briskDetectionMaximumKeypoints_, 7, 4 ))); // from config file, except the 7x4...
#else
            new brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator>(
                briskDetectionThreshold_, briskDetectionOctaves_, 
                briskDetectionAbsoluteThreshold_,
                briskDetectionMaximumKeypoints_)));
#endif
    descriptorExtractors_.push_back(
        std::shared_ptr<cv::DescriptorExtractor>(
            new brisk::BriskDescriptorExtractor(
                briskDescriptionRotationInvariance_,
                briskDescriptionScaleInvariance_)));
  }
  for (auto it = featureDetectorMutexes_.begin();
      it != featureDetectorMutexes_.end(); ++it) {
    (*it)->unlock();
  }
}

/**
 * The current frame is to be the first frame
 */
bool HybridFrontend::TrailTracking_Start()
{
  // detect good features
  std::vector<cv::Point2f> vfPoints;
  size_t nToAdd = 200; //MaxInitialTrails
  mMaxFeaturesInFrame = nToAdd;
  size_t nToThrow = 100;//MinInitialTrails
  mMinFeaturesInFrame = nToThrow;
  mFrameCounter = 0;
  int nSubPixWinWidth= 7; //SubPixWinWidth
  cv::Size subPixWinSize(nSubPixWinWidth,nSubPixWinWidth);

  cv::TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10, 0.03);
  cv::goodFeaturesToTrack(
      mCurrentPyramid[0], vfPoints, nToAdd, 
      0.01, 10, cv::Mat(), 
      3, false, 0.04);
  cv::cornerSubPix(mCurrentPyramid[0], vfPoints, subPixWinSize, cv::Size(-1,-1), termcrit);
  if(vfPoints.size()<nToThrow)
  {
    OKVIS_ASSERT_GT(Exception, vfPoints.size(), nToThrow, "No. features in the first frame is too few");
    return false;
  }
  mvKeyPoints.clear();
  mvKeyPoints.reserve(mMaxFeaturesInFrame+50);
  mlTrailers.clear();
  for(size_t i = 0; i<vfPoints.size(); i++)
  {
      Trailer t(vfPoints[i], vfPoints[i], 0, i);
      mlTrailers.push_back(t);
      mvKeyPoints.push_back(cv::KeyPoint(vfPoints[i], 8.f));
  }
  std::cout <<"create new points "<< mvKeyPoints.size()<<" at the start "<<std::endl;
  return true;
}

/**
 * Steady-state trail tracking: Advance from the previous frame, update mlTrailers, remove duds.
 * (1) track the points kept in mlTrailers which contains MapPoints in the last few keyframes and points just detected from the last keyframe
 * since these points appear in the last frame which is stored, we can track these points in the current frame by using KLT tracker
 * @return number of good trails
 */
int HybridFrontend::TrailTracking_Advance(
#ifdef USE_MSCKF2
        okvis::MSCKF2& estimator,
#else
        okvis::HybridFilter& estimator,
#endif
        uint64_t mfIdA, uint64_t mfIdB, size_t camIdA, size_t camIdB)
{
    typedef okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion> CAMERA_GEOMETRY_T;
    int nGoodTrails = 0;

    std::vector<cv::Point2f> vfPoints[3];//vfPoints[0-1] for forward searching, vfPoints[2-3] for backward searching,
    std::vector<uchar> status[2]; //status[0] forward searching indicator, status[1] backward searching indicator
    std::vector<float> err;

    std::shared_ptr<okvis::MultiFrame> frameA = estimator.multiFrame(mfIdA);
    std::shared_ptr<okvis::MultiFrame> frameB = estimator.multiFrame(mfIdB);

    // calculate the relative transformations and uncertainties
    // TODO donno, if and what we need here - I'll see

    okvis::kinematics::Transformation T_CaCb;
    okvis::kinematics::Transformation T_CbCa;
    okvis::kinematics::Transformation T_SaCa;
    okvis::kinematics::Transformation T_SbCb;
    okvis::kinematics::Transformation T_WSa;
    okvis::kinematics::Transformation T_WSb;
    okvis::kinematics::Transformation T_SaW;
    okvis::kinematics::Transformation T_SbW;
    okvis::kinematics::Transformation T_WCa;
    okvis::kinematics::Transformation T_WCb;
    okvis::kinematics::Transformation T_CaW;
    okvis::kinematics::Transformation T_CbW;

    estimator.getCameraSensorStates(mfIdA, camIdA, T_SaCa);
    estimator.getCameraSensorStates(mfIdB, camIdB, T_SbCb);
    estimator.get_T_WS(mfIdA, T_WSa);
    estimator.get_T_WS(mfIdB, T_WSb);
    T_SaW = T_WSa.inverse();
    T_SbW = T_WSb.inverse();
    T_WCa = T_WSa * T_SaCa;
    T_WCb = T_WSb * T_SbCb;
    T_CaW = T_WCa.inverse();
    T_CbW = T_WCb.inverse();
    T_CaCb = T_WCa.inverse() * T_WCb;
    T_CbCa = T_CaCb.inverse();

    // assume frameA and frameB of the same size
    float imageW = (float)(frameA->geometryAs<CAMERA_GEOMETRY_T>(camIdA)->imageWidth());
    float imageH = (float)(frameA->geometryAs<CAMERA_GEOMETRY_T>(camIdA)->imageHeight());

    size_t mMeasAttempted = 0;
    vfPoints[0].reserve(mlTrailers.size());
    vfPoints[1].reserve(mlTrailers.size());
    for(std::list<Trailer>::const_iterator it = mlTrailers.begin(); it!=mlTrailers.end();++it)
    {
        //TODO: predict point position with its world position when it->mLandmarkId is not 0.
        // To implement it cf. VioFrameMatchingAlgorithm doSetup

            //predict pose in the current pose by rotation compensation,
            //assume the camera tranlates very small compared to the point depth
            // if it's not projected successfully, we alias it to prevent it being tracked
//            Eigen::Vector2d kptA(it->fCurrPose.x, it->fCurrPose.y);
//            Eigen::Vector3d dirVecA;
//            if (!frameA->geometryAs<CAMERA_GEOMETRY_T>(camIdA)->backProject(
//                kptA, &dirVecA)) {
//                vfPoints[0].push_back(cv::Point2f(-1,-1)); //pose in last frame
//                vfPoints[1].push_back(cv::Point2f(-1,-1)); //preventing detect anything
//                continue;
//            }
//            Eigen::Vector3d dirVecB = T_CbCa.C()*dirVecA;
//            Eigen::Vector2d kptB;
//            if(frameB->geometryAs<CAMERA_GEOMETRY_T>(camIdB)->project(dirVecB, &kptB)
//                    != okvis::cameras::CameraBase::ProjectionStatus::Successful) {
//                vfPoints[0].push_back(cv::Point2f(-1,-1)); //pose in last frame
//                vfPoints[1].push_back(cv::Point2f(-1,-1)); //preventing detect anything
//                continue;
//            }

            vfPoints[0].push_back(it->fCurrPose);//point in previous frame
//            vfPoints[1].push_back(cv::Point2f((float)kptB[0],(float)kptB[1]));
            vfPoints[1].push_back(it->fCurrPose);
            ++mMeasAttempted;
    }

    int nWinWidth= 15;//WinWidth
    cv::Size winSize(nWinWidth,nWinWidth);
    cv::TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10, 0.03);
    cv::calcOpticalFlowPyrLK(mPreviousPyramid, mCurrentPyramid, vfPoints[0], vfPoints[1], status[0], err, winSize,
                3, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW, 0.001);
    //what is returned for points failed to track? the initial position
    // if a point is out of boundary, it will ended up failing to track
    vfPoints[2]=vfPoints[0];
    cv::calcOpticalFlowPyrLK(mCurrentPyramid, mPreviousPyramid, vfPoints[1], vfPoints[2], status[1], err, winSize,
                3, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW, 0.001);

    mvKeyPoints.clear();
    mvKeyPoints.reserve(mMaxFeaturesInFrame+100); //keypoints in frame B
    std::list<Trailer>::iterator it = mlTrailers.begin();
    for(size_t which=0; which<status[0].size(); ++which){// note status[0].size() is constant, but not mlTrailers.size()
        if(status[0][which]){
            cv::Point2f delta=vfPoints[2][which] - vfPoints[0][which];
            bool bInImage=(vfPoints[0][which].x>=0.f)&&(vfPoints[0][which].y>=0.f)&&
                    (vfPoints[1][which].x>=0.f)&&(vfPoints[1][which].y>=0.f)&&
                (vfPoints[0][which].x<=(imageW-1.f))&&(vfPoints[0][which].y<=(imageH-1.f))&&
                (vfPoints[1][which].x<=(imageW-1.f))&&(vfPoints[1][which].y<=(imageH-1.f));
            if(!status[1][which]||!bInImage || (delta.dot(delta)) > 2)
                status[0][which] = 0;

//           if(!bInImage){
//                    std::cout<<"vPoint[0][which] "<< vfPoints[0][which].x <<" "<<vfPoints[0][which].y <<" "<<
//                               vfPoints[1][which].x <<" "<<vfPoints[1][which].y <<" "<<std::endl;
//            }

            it->fCurrPose=vfPoints[1][which];
        }

        if(!status[0][which]) // Erase from list of trails if not found this frame.
            it=mlTrailers.erase(it);
        else{
            it->uOldKeyPointIdx = it->uKeyPointIdx;
            it->uKeyPointIdx = nGoodTrails;
            ++it->uTrackLength;

            mvKeyPoints.push_back(cv::KeyPoint(it->fCurrPose, 8.f));
            ++it;
            ++nGoodTrails;
        }
    }

    assert(mlTrailers.end()==it && (int)mvKeyPoints.size()== nGoodTrails);

    /*//added by huai for debug
        cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
        cv::Mat image=mCurrentKF.pyramid[0].clone(); //CAUTION: do not draw on pyramid[0], ruining the data
        for(list<Trailer>::const_iterator it = mlTrailers.begin(); it!=mlTrailers.end();++it){
            line(image, it->fInitPose, it->fCurrPose, CV_RGB(0,0,0));
            circle( image, it->fCurrPose, 3, Scalar(255,0,0), -1, 8);
        }
        //CVDRGB2OpenCVRGB(mimFrameRGB, image);
        cv::imshow( "Display window", image );                   // Show our image inside it.
        cv::waitKey(0);
    */

    return nGoodTrails;
}

int HybridFrontend::TrailTracking_Advance2(const std::vector<cv::KeyPoint>& keypoints, const std::vector<size_t>& mapPointIds,
                                           const std::vector<Eigen::Vector3d>& mapPointPositions, uint64_t mfIdB,
                                           const cv::Mat currentImage){
    int nGoodTrails =0;
    int nTrackedFeatures = 0;
    //build a map of mappoint ids for accelerating searching
    std::map<size_t, std::list<Trailer>::iterator> mpId2Trailer;

    for(std::list<Trailer>::iterator iter = mlTrailers.begin(), itEnd = mlTrailers.end(); iter!= itEnd; ++iter)
    {
        mpId2Trailer[iter->uExternalLmId] = iter;
    }

    std::list<Trailer> tempTrailers;
    auto it = mapPointIds.begin();
    size_t counter =0;
    while(it!= mapPointIds.end()){
        if(*it ==0){
            ++it;
            ++counter;
            continue;
        }
        auto iterMap = mpId2Trailer.find(*it);
        if(iterMap!= mpId2Trailer.end()){
            std::list<Trailer>::iterator tp = iterMap->second;
            tp->uOldKeyPointIdx = tp->uKeyPointIdx;
            tp->uKeyPointIdx = counter;
            tp->fCurrPose= keypoints[counter].pt;
            tp->p_W = mapPointPositions[counter];
            tp->uCurrentFrameId = mfIdB;
            ++(tp->uTrackLength);
            ++nGoodTrails;
        }else{ // a new trail
            Trailer t(keypoints[counter].pt, keypoints[counter].pt, 0, counter, *it);
            t.p_W = mapPointPositions[counter];
            t.uCurrentFrameId = mfIdB;
            tempTrailers.push_back(t);
        }
        ++nTrackedFeatures;
        ++it;
        ++counter;
    }

    assert(counter == mapPointIds.size());
    mlTrailers.insert(mlTrailers.end(), tempTrailers.begin(), tempTrailers.end());

    //added by huai for debug
//    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
//    cv::Mat image= currentImage.clone(); //CAUTION: do not draw on pyramid[0], ruining the data
//    for(std::list<Trailer>::const_iterator it = mlTrailers.begin(); it!=mlTrailers.end();++it){
//        cv::line(image, it->fInitPose, it->fCurrPose, CV_RGB(0,0,0));
//        cv::circle( image, it->fCurrPose, 3, cv::Scalar(255,0,0), -1, 8);
//    }
//    //CVDRGB2OpenCVRGB(mimFrameRGB, image);
//    cv::imshow( "Display window", image );                   // Show our image inside it.
//    cv::waitKey(30);

    myAccumulator(nTrackedFeatures);
    return nGoodTrails;
}

// detect some points from just inserted frame, curKF, and insert them into mlTrailers
int HybridFrontend::TrailTracking_DetectAndInsert(const cv::Mat &currentFrame, int nInliers, std::vector<cv::KeyPoint> & vNewKPs)
{
    vNewKPs.clear();
    vNewKPs.reserve(500);
    const size_t startIndex = mvKeyPoints.size(); // how many keypoints already in the current frame?
    // detect good features
    std::vector<cv::Point2f> vfPoints;
    int nToAdd=mMaxFeaturesInFrame-nInliers+50;
    int nSubPixWinWidth=7; //SubPixWinWidth
    cv::Size subPixWinSize(nSubPixWinWidth,nSubPixWinWidth);
    cv::TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10, 0.03);
    //create mask
    if(mMask.empty())
        mMask.create(currentFrame.size(), CV_8UC1);
    mMask= cv::Scalar(255);
    for (std::list<Trailer>::const_iterator lTcIt=mlTrailers.begin();lTcIt!=mlTrailers.end(); ++lTcIt)
    {
        cv::circle(mMask, lTcIt->fCurrPose, 11, cv::Scalar(0), CV_FILLED);
    }
    double minDist=std::min(currentFrame.size().width/100+1.0, 15.0);
    cv::goodFeaturesToTrack(
        currentFrame, vfPoints, nToAdd, 
        0.01, minDist, mMask,
        3, false, 0.04);
    cornerSubPix(currentFrame, vfPoints, subPixWinSize, cv::Size(-1,-1), termcrit);

    std::cout << "Detected additional " << vfPoints.size() << " points" << std::endl;
    /*//use grid for even distribution. Emprically, without even distribution, rotations are already very good
    float bucket_width=50, bucket_height=50, max_features_bin=4;
    cv::Mat grid((int)ceil(currentFrame.size().width/bucket_width), (int)ceil(currentFrame.size().height/bucket_height), CV_8UC1, Scalar::all(0));
    for (list<Trailer>::const_iterator lTcIt=mlTrailers.begin();lTcIt!=mlTrailers.end(); ++lTcIt)
    {
        ++grid.at<unsigned char>((int)(lTcIt->fCurrPose.x/bucket_width),(int)(lTcIt->fCurrPose.y/bucket_height));
    }
    // feature bucketing: keeps only max_features per bucket, where the domain
    // is split into buckets of size (bucket_width,bucket_height)
    */
    size_t jack=0;
    for(size_t i = 0; i<vfPoints.size(); i++)
    {
        /*if(grid.at<unsigned char>((int)(vfPoints[i].x/bucket_width),(int)(vfPoints[i].y/bucket_height))>=max_features_bin)
            continue;
        else
            ++grid.at<unsigned char>((int)(vfPoints[i].x/bucket_width),(int)(vfPoints[i].y/bucket_height));*/
        Trailer t(vfPoints[i], vfPoints[i], 0, startIndex + jack);
        mlTrailers.push_back(t);
        vNewKPs.push_back(cv::KeyPoint(vfPoints[i], 8.f));
        ++jack;
    }

    /*//added by huai for debug
        cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
        cv::Mat image=mCurrentKF.pyramid[0].clone(); //CAUTION: do not draw on pyramid[0], ruining the data
        for(list<Trailer>::const_iterator i = mlTrailers.begin(); i!=mlTrailers.end();++i){
        //	line(image, i->fInitPose, i->fCurrPose, CV_RGB(0,0,0));
            circle( image, i->fCurrPose, 3, Scalar(255,0,0), -1, 8);
        }
        //CVDRGB2OpenCVRGB(mimFrameRGB, image);
        cv::imshow( "Display window", image );                   // Show our image inside it.
        cv::waitKey(0);
    */
    return vNewKPs.size();
}

void HybridFrontend::UpdateEstimatorObservations(
#ifdef USE_MSCKF2
        okvis::MSCKF2& estimator,
#else
        okvis::HybridFilter& estimator,
#endif
        uint64_t mfIdA, uint64_t mfIdB, size_t camIdA, size_t camIdB)
{
    // update estimator's observations and keypoints
    typedef okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion > camera_geometry_t;
    std::shared_ptr<okvis::MultiFrame> frameA = estimator.multiFrame(mfIdA);
    std::shared_ptr<okvis::MultiFrame> frameB = estimator.multiFrame(mfIdB);
    for(auto it= mlTrailers.begin(); it!= mlTrailers.end(); ++it)
    {
        if(it->uLandmarkId!=0){
            estimator.addObservation<camera_geometry_t>(it->uLandmarkId, mfIdB, camIdB, it->uKeyPointIdx);
        }
        else if(it->uTrackLength==2){//two observations, we skip those for new features of one observation
            uint64_t lmId = okvis::IdProvider::instance().newId();
            //TODO: do triangulation        
//            double sigmaR = 1e-3;
//            Eigen::Vector3d e1 = backProject(Eigen::Vector3d(it->fInitPose.x, it->fInitPose.y, 1)).normalized();
//            Eigen::Vector3d e2 = (T_AW.C()*T_BW.C().transpose()*obsDirections.back()).normalized();
//            okvis::kinematics::Transformation T_MW = T_BW;
//            Eigen::Vector3d p2 = T_AW.r() - (T_AW.q()*T_BW.q().conjugate())._transformVector(T_BW.r());
//            Eigen::Matrix<double,4,1> hP_A = triangulateFastLocal(Eigen::Vector3d(0,0,0),  // center of A in A coordinates
//                             e1, p2, // center of B in A coordinates
//                             e2, sigmaR, isValid, isParallel);

            Eigen::Matrix<double,4,1> hP_W;
            hP_W.setOnes(); //random initialization
            bool canBeInitialized=true;
            estimator.addLandmark(lmId, hP_W);
            OKVIS_ASSERT_TRUE(Exception, estimator.isLandmarkAdded(lmId),
                              lmId<<" not added, bug");
            estimator.setLandmarkInitialized(lmId, canBeInitialized);

            frameA->setLandmarkId(camIdA, it->uOldKeyPointIdx, lmId);
            estimator.addObservation<camera_geometry_t>(lmId, mfIdA, camIdA, it->uOldKeyPointIdx);

            it->uLandmarkId = lmId;
            frameB->setLandmarkId(camIdB, it->uKeyPointIdx, lmId);

            estimator.addObservation<camera_geometry_t>(lmId, mfIdB, camIdB, it->uKeyPointIdx);
        }
    }
}

void HybridFrontend::UpdateEstimatorObservations2(
#ifdef USE_MSCKF2
        okvis::MSCKF2& estimator,
#else
        okvis::HybridFilter& estimator,
#endif
        uint64_t mfIdA, uint64_t mfIdB, size_t camIdA, size_t camIdB)
{
    const size_t maxTrackLength = 15;  //break the track once it hits this number
    // update estimator's observations and keypoints
    typedef okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion > camera_geometry_t;
    std::shared_ptr<okvis::MultiFrame> frameA = estimator.multiFrame(mfIdA);
    std::shared_ptr<okvis::MultiFrame> frameB = estimator.multiFrame(mfIdB);
    int deletedTrails = 0;
    for(auto it= mlTrailers.begin(); it!= mlTrailers.end();)
    {
        if(it->uCurrentFrameId< mfIdB)
        {
            if(it->uTrackLength>1)
                ++deletedTrails;

            it= mlTrailers.erase(it);
            continue;
        }
        else
            assert(it->uCurrentFrameId == mfIdB);

        if(it->uLandmarkId!=0){
            if(it->uTrackLength<maxTrackLength)
                estimator.addObservation<camera_geometry_t>(it->uLandmarkId, mfIdB, camIdB, it->uKeyPointIdx);
            else//break up the trail
            {
                it->fInitPose =it->fCurrPose;
                it->uLandmarkId =0;
                it->uTrackLength =1;
            }
        }
        else if(it->uTrackLength==2){//two observations, we skip those for new features of one observation
            uint64_t lmId = okvis::IdProvider::instance().newId();

            Eigen::Matrix<double,4,1> hP_W;
            hP_W.head<3>() = it->p_W; //up to a scale factor
            hP_W[3] =1;
            bool canBeInitialized=true;
            estimator.addLandmark(lmId, hP_W);
            OKVIS_ASSERT_TRUE(Exception, estimator.isLandmarkAdded(lmId),
                              lmId<<" not added, bug");
            estimator.setLandmarkInitialized(lmId, canBeInitialized);

            frameA->setLandmarkId(camIdA, it->uOldKeyPointIdx, lmId);
            estimator.addObservation<camera_geometry_t>(lmId, mfIdA, camIdA, it->uOldKeyPointIdx);

            it->uLandmarkId = lmId;
            frameB->setLandmarkId(camIdB, it->uKeyPointIdx, lmId);
            estimator.addObservation<camera_geometry_t>(lmId, mfIdB, camIdB, it->uKeyPointIdx);
        }
        else
            assert(it->uTrackLength==1);
        ++it;
    }
    std::cout <<"Trails after deleted "<< deletedTrails<<" "<< mlTrailers.size()<<std::endl;
}

void HybridFrontend::printNumFeatureDistribution(std::ofstream & stream){

    if(!stream.is_open()){
        std::cerr <<"cannot write feature distribution without proper stream "<<std::endl;
    }
    histogram_type hist = boost::accumulators::density(myAccumulator);

    double total = 0.0;
    stream<<"histogram of number of features in images (bin lower bound, value)"<< std::endl;
    for( size_t i = 0; i < hist.size(); i++ )
    {
        stream << hist[i].first << " " << hist[i].second << std::endl;
        total += hist[i].second;
    }
    std::cout << "Total of densities: " << total << " should be 1."<<std::endl;
}

}  // namespace okvis
