#include <msckf/HybridFrontend.hpp>
#include <msckf/implementation/HybridFrontend.hpp>

#include <algorithm>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <brisk/brisk.h>

#include <opencv2/imgproc/imgproc.hpp>

#include <okvis/VioKeyframeWindowMatchingAlgorithm.hpp>
#include <okvis/IdProvider.hpp>
#include <okvis/ceres/ImuError.hpp>

// cameras and distortions
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/FovDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>

// Kneip RANSAC
#include <opengv/sac_problems/absolute_pose/FrameAbsolutePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/FrameRelativePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/FrameRotationOnlySacProblem.hpp>
#include <msckf/FrameTranslationOnlySacProblem.hpp>

#include <opengv/absolute_pose/FrameNoncentralAbsoluteAdapter.hpp>
#include <opengv/relative_pose/FrameRelativeAdapter.hpp>

#include <opengv/sac/Ransac.hpp>

DEFINE_int32(feature_tracking_method, 0,
             "0 default okvis brisk keyframe and back-to-back frame matching, "
             "1 KLT back-to-back frame matching, "
             "2 brisk back-to-back frame matching");

/// \brief okvis Main namespace of this package.
namespace okvis {

// Constructor.
HybridFrontend::HybridFrontend(size_t numCameras)
    : isInitialized_(false),
      numCameras_(numCameras),
      briskDetectionOctaves_(0),
      briskDetectionThreshold_(50.0),
      briskDetectionAbsoluteThreshold_(800.0),
      briskDetectionMaximumKeypoints_(450),
      briskDescriptionRotationInvariance_(true),
      briskDescriptionScaleInvariance_(false),
      briskMatchingThreshold_(60.0),
      matcher_(
          std::unique_ptr<okvis::DenseMatcher>(new okvis::DenseMatcher(4))),
      keyframeInsertionOverlapThreshold_(0.6),
      keyframeInsertionMatchingRatioThreshold_(0.2) {
  // create mutexes for feature detectors and descriptor extractors
  for (size_t i = 0; i < numCameras_; ++i) {
    featureDetectorMutexes_.push_back(
        std::unique_ptr<std::mutex>(new std::mutex()));
  }
  initialiseBriskFeatureDetectors();
}

// Detection and descriptor extraction on a per image basis.
bool HybridFrontend::detectAndDescribe(
    size_t cameraIndex, std::shared_ptr<okvis::MultiFrame> frameOut,
    const okvis::kinematics::Transformation& T_WC,
    const std::vector<cv::KeyPoint>* keypoints) {
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
    okvis::HybridFilter& estimator,
    // TODO(sleutenegger): why is this not used here?
    okvis::kinematics::Transformation& /*T_WS_propagated*/,
    const okvis::VioParameters& params,
    // TODO(sleutenegger): why is this not used here?
    const std::shared_ptr<okvis::MapPointVector> /*map*/,
    std::shared_ptr<okvis::MultiFrame> framesInOut,
    bool* asKeyframe) {
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
  int num3dMatches = 0;
  // always set asKeyframe true for back-to-back feature tracking methods
  if (FLAGS_feature_tracking_method == 1) {
    *asKeyframe = true;
    int requiredMatches = 5;
    bool rotationOnly = false;
    // match to last frame
    TimerSwitchable matchToLastFrameTimer("2.4.2 matchToLastFrameKLT");
    switch (distortionType) {
      case okvis::cameras::NCameraSystem::RadialTangential: {
        num3dMatches = matchToLastFrameKLT<
            okvis::cameras::PinholeCamera<
                okvis::cameras::RadialTangentialDistortion> >(
            estimator, params, framesInOut,
            rotationOnly, false);
        break;
      }
      case okvis::cameras::NCameraSystem::Equidistant: {
        num3dMatches = matchToLastFrameKLT<
            okvis::cameras::PinholeCamera<
                okvis::cameras::EquidistantDistortion> >(
            estimator, params, framesInOut,
            rotationOnly, false);
        break;
      }
      case okvis::cameras::NCameraSystem::RadialTangential8: {
        num3dMatches = matchToLastFrameKLT<
            okvis::cameras::PinholeCamera<
                okvis::cameras::RadialTangentialDistortion8> >(
            estimator, params, framesInOut,
            rotationOnly, false);

        break;
      }
      case okvis::cameras::NCameraSystem::FOV: {
        num3dMatches = matchToLastFrameKLT<
            okvis::cameras::PinholeCamera<
                okvis::cameras::FovDistortion> >(
            estimator, params, framesInOut,
            rotationOnly, false);
        break;
      }
      default:
        OKVIS_THROW(Exception, "Unsupported distortion type.")
        break;
    }
    matchToLastFrameTimer.stop();
    if (!isInitialized_) {
        isInitialized_ = true;
        LOG(INFO) << "Frontend initialized!";
    }
    if (num3dMatches <= requiredMatches) {
      LOG(WARNING) << "Tracking last frame failure. Number of 3d2d-matches: " << num3dMatches;
    }
    if (estimator.numFrames() > 1) {
        uint64_t currentFrameId = framesInOut->id();
        uint64_t lastFrameId = estimator.currentKeyframeId();
        bool removeOutliers = false;
        runRansac2d2d(estimator, params, currentFrameId, lastFrameId,
                      removeOutliers, asKeyframe);
    }
    return true;
  }

  // first frame? (did do addStates before, so 1 frame minimum in estimator)
  if (estimator.numFrames() > 1) {
    int requiredMatches = 5;
    double uncertainMatchFraction = 0;
    bool rotationOnly = false;

    if (FLAGS_feature_tracking_method == 0) {
    // match to last keyframe
    TimerSwitchable matchKeyframesTimer("2.4.1 matchToKeyframes");
    switch (distortionType) {
      case okvis::cameras::NCameraSystem::RadialTangential: {
        num3dMatches = matchToKeyframes<
            VioKeyframeWindowMatchingAlgorithm<
                okvis::cameras::PinholeCamera<
                    okvis::cameras::RadialTangentialDistortion> > >(
            estimator, params, framesInOut->id(), rotationOnly, false,
            &uncertainMatchFraction);
        break;
      }
      case okvis::cameras::NCameraSystem::Equidistant: {
        num3dMatches = matchToKeyframes<
            VioKeyframeWindowMatchingAlgorithm<
                okvis::cameras::PinholeCamera<
                    okvis::cameras::EquidistantDistortion> > >(
            estimator, params, framesInOut->id(), rotationOnly, false,
            &uncertainMatchFraction);
        break;
      }
      case okvis::cameras::NCameraSystem::RadialTangential8: {
        num3dMatches = matchToKeyframes<
            VioKeyframeWindowMatchingAlgorithm<
                okvis::cameras::PinholeCamera<
                    okvis::cameras::RadialTangentialDistortion8> > >(
            estimator, params, framesInOut->id(), rotationOnly, false,
            &uncertainMatchFraction);
        break;
      }
      case okvis::cameras::NCameraSystem::FOV: {
        num3dMatches = matchToKeyframes<
            VioKeyframeWindowMatchingAlgorithm<
                okvis::cameras::PinholeCamera<
                    okvis::cameras::FovDistortion> > >(
            estimator, params, framesInOut->id(), rotationOnly, false,
            &uncertainMatchFraction);
        break;
      }
      default:
        OKVIS_THROW(Exception, "Unsupported distortion type.")
        break;
    }
    matchKeyframesTimer.stop();
    if (!isInitialized_) {
//      if (!rotationOnly) {
        isInitialized_ = true;
        LOG(INFO) << "Initialized!";
//      }
    }

    if (num3dMatches <= requiredMatches) {
      LOG(WARNING) << "Tracking failure. Number of 3d2d-matches: " << num3dMatches;
    }

    // keyframe decision, at the moment only landmarks that match with keyframe are initialised
    *asKeyframe = *asKeyframe || doWeNeedANewKeyframe(estimator, framesInOut);
    } else {
        if (!isInitialized_) {
    //      if (!rotationOnly) {
            // TODO(jhuai): should check the motion and then initialize the filter
            // Adding states before proper initialization of a filter like MSCKF
            // greatly complicates states management.
            // The following link may hints on solutions.
            // https://github.com/ethz-asl/okvis/blob/master/okvis_multisensor_processing/src/ThreadedKFVio.cpp#L735
            isInitialized_ = true;
            LOG(INFO) << "Frontend initialized!";
    //      }
        }
        *asKeyframe = true;
    }
    // match to last frame
    TimerSwitchable matchToLastFrameTimer("2.4.2 matchToLastFrame");
    switch (distortionType) {
      case okvis::cameras::NCameraSystem::RadialTangential: {
        num3dMatches = matchToLastFrame<
            VioKeyframeWindowMatchingAlgorithm<okvis::cameras::PinholeCamera<
                okvis::cameras::RadialTangentialDistortion> > >(
            estimator, params, framesInOut->id(),
            rotationOnly, false);
        break;
      }
      case okvis::cameras::NCameraSystem::Equidistant: {
        num3dMatches = matchToLastFrame<
            VioKeyframeWindowMatchingAlgorithm<okvis::cameras::PinholeCamera<
                okvis::cameras::EquidistantDistortion> > >(
            estimator, params, framesInOut->id(),
            rotationOnly, false);
        break;
      }
      case okvis::cameras::NCameraSystem::RadialTangential8: {
        num3dMatches = matchToLastFrame<
            VioKeyframeWindowMatchingAlgorithm<okvis::cameras::PinholeCamera<
                okvis::cameras::RadialTangentialDistortion8> > >(
            estimator, params, framesInOut->id(),
            rotationOnly, false);

        break;
      }
      case okvis::cameras::NCameraSystem::FOV: {
        num3dMatches = matchToLastFrame<
            VioKeyframeWindowMatchingAlgorithm<okvis::cameras::PinholeCamera<
                okvis::cameras::FovDistortion> > >(
            estimator, params, framesInOut->id(),
            rotationOnly, false);
        break;
      }
      default:
        OKVIS_THROW(Exception, "Unsupported distortion type.")
        break;
    }
    matchToLastFrameTimer.stop();
    if (FLAGS_feature_tracking_method != 0 && num3dMatches <= requiredMatches) {
      LOG(WARNING) << "Tracking last frame failure. Number of 3d2d-matches: " << num3dMatches;
    }
  } else
    *asKeyframe = true;  // first frame needs to be keyframe

  // do stereo match to get new landmarks
  TimerSwitchable matchStereoTimer("2.4.3 matchStereo");
  switch (distortionType) {
    case okvis::cameras::NCameraSystem::RadialTangential: {
      matchStereo<
          VioKeyframeWindowMatchingAlgorithm<
              okvis::cameras::PinholeCamera<
                  okvis::cameras::RadialTangentialDistortion> > >(estimator,
                                                                  framesInOut);
      break;
    }
    case okvis::cameras::NCameraSystem::Equidistant: {
      matchStereo<
          VioKeyframeWindowMatchingAlgorithm<
              okvis::cameras::PinholeCamera<
                  okvis::cameras::EquidistantDistortion> > >(estimator,
                                                             framesInOut);
      break;
    }
    case okvis::cameras::NCameraSystem::RadialTangential8: {
      matchStereo<
          VioKeyframeWindowMatchingAlgorithm<
              okvis::cameras::PinholeCamera<
                  okvis::cameras::RadialTangentialDistortion8> > >(estimator,
                                                                   framesInOut);
      break;
    }
    case okvis::cameras::NCameraSystem::FOV: {
      matchStereo<
          VioKeyframeWindowMatchingAlgorithm<
              okvis::cameras::PinholeCamera<
                  okvis::cameras::FovDistortion> > >(estimator,
                                                     framesInOut);
      break;
    }
    default:
      OKVIS_THROW(Exception, "Unsupported distortion type.")
      break;
  }
  matchStereoTimer.stop();

  return true;
}

// Propagates pose, speeds and biases with given IMU measurements.
bool HybridFrontend::propagation(
    const okvis::ImuMeasurementDeque& imuMeasurements,
    const okvis::ImuParameters& imuParams,
    okvis::kinematics::Transformation& T_WS_propagated,
    okvis::SpeedAndBiases& speedAndBiases,
    const okvis::Time& t_start, const okvis::Time& t_end,
    Eigen::Matrix<double, 15, 15>* covariance,
    Eigen::Matrix<double, 15, 15>* jacobian) const {
  if (imuMeasurements.size() < 2) {
    LOG(WARNING)
        << "- Skipping propagation as only one IMU measurement has been given to frontend."
        << " Normal when starting up.";
    return 0;
  }
  // huai: this is not replaced by IMUOdometry function because it performs good
  // for propagating only states,
  // and it does not use imuErrorModel
  int measurements_propagated = okvis::ceres::ImuError::propagation(
      imuMeasurements, imuParams, T_WS_propagated, speedAndBiases, t_start,
      t_end, covariance, jacobian);

  return measurements_propagated > 0;
}

// Decision whether a new frame should be keyframe or not.
bool HybridFrontend::doWeNeedANewKeyframe(
    const okvis::HybridFilter& estimator,
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
    double matchingRatio = static_cast<double>(frameBMatches.size()) /
                           static_cast<double>(pointsInFrameBMatchesArea);

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
template <class MATCHING_ALGORITHM>
int HybridFrontend::matchToKeyframes(
    okvis::HybridFilter& estimator,
    const okvis::VioParameters& params,
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
    } else if (kfcounter == 0) {
      bool asKeyframe;
      runRansac2d2d(estimator, params, currentFrameId, olderFrameId, false,
                       &asKeyframe);
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
    *uncertainMatchFraction =
        static_cast<double>(numUncertainMatches) / static_cast<double>(retCtr);
  }

  return retCtr;
}

// Match a new multiframe to the last frame.
template <class CAMERA_GEOMETRY_T>
int HybridFrontend::matchToLastFrameKLT(
    okvis::HybridFilter& estimator,
    const okvis::VioParameters& /*params*/,
    std::shared_ptr<okvis::MultiFrame> framesInOut,
    bool& rotationOnly,
    bool /*usePoseUncertainty*/, bool /*removeOutliers*/) {
  int retCtr = 0;
  rotationOnly = true;

  if (estimator.numFrames() == 1) {
    loadParameters<CAMERA_GEOMETRY_T>(framesInOut, estimator, &featureTracker_);
    featureTracker_.initialize();
  } else {
    uint64_t fIdB = framesInOut->id();
    okvis::kinematics::Transformation T_WSb;
    estimator.get_T_WS(fIdB, T_WSb);

    uint64_t fIdA = estimator.frameIdByAge(1);
    okvis::kinematics::Transformation T_WSa;
    estimator.get_T_WS(fIdA, T_WSa);

    std::vector<cv::Matx33f> R_CkCkm1_list;
    for (size_t camId = 0; camId < framesInOut->numFrames(); ++camId) {
      okvis::kinematics::Transformation T_SC;
      estimator.getCameraSensorStates(fIdA, camId, T_SC);
      Eigen::Quaterniond q_CkCkm1 =
          (T_WSb.q() * T_SC.q()).conjugate() * (T_WSa.q() * T_SC.q());
      Eigen::Matrix3f R_CkCkm1 = q_CkCkm1.toRotationMatrix().cast<float>();
      cv::Matx33f mat_R_CkCkm1;
      cv::eigen2cv(R_CkCkm1, mat_R_CkCkm1);
      R_CkCkm1_list.emplace_back(mat_R_CkCkm1);
    }
    featureTracker_.setRelativeOrientation(R_CkCkm1_list);
  }

  featureTracker_.stereoCallback(
      framesInOut->getFrame(0),
      framesInOut->numFrames() > 1 ?
          framesInOut->getFrame(1) :
          cv::Mat(),
      feature_tracker::MessageHeader{framesInOut->timestamp()});

//  featureTracker_.drawFeaturesMono();

  std::vector<feature_tracker::FeatureIDType> curr_ids(0);
  featureTracker_.getCurrentFeatureIds(&curr_ids);
  retCtr += curr_ids.size();

  std::vector<std::vector<cv::KeyPoint>> curr_keypoints(2);
  featureTracker_.getCurrentKeypoints(&curr_keypoints[0], &curr_keypoints[1]);

  featureTracker_.prepareForNextFrame(); // clear many things for next frame

  for (size_t im = 0; im < framesInOut->numFrames(); ++im) {
    framesInOut->resetKeypoints(im, curr_keypoints[im]);
  }

  addConstraintToEstimator<CAMERA_GEOMETRY_T>(
      curr_ids, framesInOut, estimator);
  return retCtr;
}

// Match a new multiframe to the last frame.
template <class MATCHING_ALGORITHM>
int HybridFrontend::matchToLastFrame(
    okvis::HybridFilter& estimator,
    const okvis::VioParameters& params, const uint64_t currentFrameId,
    bool& rotationOnly,
    bool usePoseUncertainty, bool removeOutliers) {
  if (estimator.numFrames() < 2) {
    // just starting, so yes, we need this as a new keyframe
    return 0;
  }

  uint64_t lastFrameId = estimator.frameIdByAge(1);

  if (FLAGS_feature_tracking_method == 0 &&
      estimator.isKeyframe(lastFrameId)) {
    // already done
    return 0;
  }

  int retCtr = 0;

  for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {
    MATCHING_ALGORITHM matchingAlgorithm(estimator,
                                         MATCHING_ALGORITHM::Match3D2D,
                                         briskMatchingThreshold_,
                                         usePoseUncertainty);
    matchingAlgorithm.setFrames(lastFrameId, currentFrameId, im, im);

    // match 3D-2D
    matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
    retCtr += matchingAlgorithm.numMatches();
  }

  runRansac3d2d(estimator, params.nCameraSystem,
                estimator.multiFrame(currentFrameId), removeOutliers);

  for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {
    MATCHING_ALGORITHM matchingAlgorithm(estimator,
                                         MATCHING_ALGORITHM::Match2D2D,
                                         briskMatchingThreshold_,
                                         usePoseUncertainty);
    matchingAlgorithm.setFrames(lastFrameId, currentFrameId, im, im);

    // match 2D-2D for initialization of new (mono-)correspondences
    matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
    retCtr += matchingAlgorithm.numMatches();
  }

  // remove outliers
  rotationOnly = false;
  if (!isInitialized_)
    runRansac2d2d(estimator, params, currentFrameId, lastFrameId, false,
                  removeOutliers, rotationOnly);

  return retCtr;
}

// Match the frames inside the multiframe to each other to initialise
// new landmarks.
template <class MATCHING_ALGORITHM>
void HybridFrontend::matchStereo(
    okvis::HybridFilter& estimator,
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

  // TODO(sleuten): for more than 2 cameras check there were no duplications!

  // TODO(sleuten): ensure 1-1 matching.

  // TODO(sleuten): no RANSAC ?

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
    okvis::HybridFilter& estimator,
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
  opengv::absolute_pose::FrameNoncentralAbsoluteAdapter adapter(
      estimator, nCameraSystem, currentFrame);

  size_t numCorrespondences = adapter.getNumberCorrespondences();
  if (numCorrespondences < 5)
    return numCorrespondences;

  // create a RelativePoseSac problem and RANSAC
  opengv::sac::Ransac<
      opengv::sac_problems::absolute_pose ::FrameAbsolutePoseSacProblem>
      ransac;
  std::shared_ptr<
      opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem>
      absposeproblem_ptr(
          new opengv::sac_problems::absolute_pose::
              FrameAbsolutePoseSacProblem(
                  adapter,
                  opengv::sac_problems::absolute_pose::
                      FrameAbsolutePoseSacProblem::Algorithm::GP3P));
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
    okvis::HybridFilter& estimator,
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
    opengv::relative_pose::FrameRelativeAdapter adapter(
        estimator, params.nCameraSystem, olderFrameId, im, currentFrameId, im);

    size_t numCorrespondences = adapter.getNumberCorrespondences();

    if (numCorrespondences < 10)
      continue;  // won't generate meaningful results. let's hope the few correspondences we have are all inliers!!

    // try both the rotation-only RANSAC and the relative one:

    // create a RelativePoseSac problem and RANSAC
    typedef opengv::sac_problems::relative_pose::
        FrameRotationOnlySacProblem FrameRotationOnlySacProblem;
    opengv::sac::Ransac<FrameRotationOnlySacProblem> rotation_only_ransac;
    std::shared_ptr<FrameRotationOnlySacProblem>
        rotation_only_problem_ptr(
            new FrameRotationOnlySacProblem(adapter));
    rotation_only_ransac.sac_model_ = rotation_only_problem_ptr;
    rotation_only_ransac.threshold_ = 9;
    rotation_only_ransac.max_iterations_ = 50;

    // run the ransac
    rotation_only_ransac.computeModel(0);

    // get quality
    int rotation_only_inliers = rotation_only_ransac.inliers_.size();
    float rotation_only_ratio = static_cast<float>(rotation_only_inliers) /
                                static_cast<float>(numCorrespondences);

    // now the rel_pose one:
    typedef opengv::sac_problems::relative_pose::
        FrameRelativePoseSacProblem FrameRelativePoseSacProblem;
    opengv::sac::Ransac<FrameRelativePoseSacProblem> rel_pose_ransac;
    std::shared_ptr<FrameRelativePoseSacProblem> rel_pose_problem_ptr(
        new FrameRelativePoseSacProblem(
            adapter, FrameRelativePoseSacProblem::STEWENIUS));
    rel_pose_ransac.sac_model_ = rel_pose_problem_ptr;
    rel_pose_ransac.threshold_ = 9;     //(1.0 - cos(0.5/600));
    rel_pose_ransac.max_iterations_ = 50;

    // run the ransac
    rel_pose_ransac.computeModel(0);

    // assess success
    int rel_pose_inliers = rel_pose_ransac.inliers_.size();
    float rel_pose_ratio = static_cast<float>(rel_pose_inliers) /
                           static_cast<float>(numCorrespondences);

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

        uint64_t lmId = multiFrame->landmarkId(im, idxB);
        // reset ID:
        multiFrame->setLandmarkId(im, idxB, 0);
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

int HybridFrontend::runRansac2d2d(okvis::HybridFilter& estimator,
                                  const okvis::VioParameters& params,
                                  uint64_t currentFrameId,
                                  uint64_t olderFrameId, bool removeOutliers,
                                  bool* asKeyframe) {
  const size_t numCameras = params.nCameraSystem.numCameras();
  RelativeMotionType rmt = UNCERTAIN_MOTION;
  size_t totalInlierNumber = 0;
  bool rotation_only_success = false;
  bool translation_only_success = false;
  double maxOverlap = 0.0;
  double maxMatchRatio = 0.0;
  std::shared_ptr<okvis::MultiFrame> frameBPtr =
      estimator.multiFrame(currentFrameId);
  // run relative RANSAC
  for (size_t im = 0; im < numCameras; ++im) {
    // relative pose adapter for Kneip toolchain
    opengv::relative_pose::FrameRelativeAdapter adapter(
        estimator, params.nCameraSystem, olderFrameId, im, currentFrameId, im);
    double overlap;
    double matchRatio;

    adapter.computeMatchStats(frameBPtr, im, &overlap, &matchRatio);
    maxOverlap = std::max(overlap, maxOverlap);
    maxMatchRatio = std::max(matchRatio, maxMatchRatio);
    size_t numCorrespondences = adapter.getNumberCorrespondences();

    if (numCorrespondences < 10) { // won't generate meaningful results.
      estimator.multiFrame(currentFrameId)
          ->setRelativeMotion(im, olderFrameId, rmt);
      continue;
    }
    // try both the rotation-only RANSAC and the relative one:

    // create a RelativePoseSac problem and RANSAC
    typedef opengv::sac_problems::relative_pose::
        FrameRotationOnlySacProblem FrameRotationOnlySacProblem;
    opengv::sac::Ransac<FrameRotationOnlySacProblem> rotation_only_ransac;
    std::shared_ptr<FrameRotationOnlySacProblem>
        rotation_only_problem_ptr(
            new FrameRotationOnlySacProblem(adapter));
    rotation_only_ransac.sac_model_ = rotation_only_problem_ptr;
    rotation_only_ransac.threshold_ = 9;
    rotation_only_ransac.max_iterations_ = 50;

    // run the ransac
    rotation_only_ransac.computeModel(0);

    // get quality
    int rotation_only_inliers = rotation_only_ransac.inliers_.size();
    float rotation_only_ratio = static_cast<float>(rotation_only_inliers) /
                                static_cast<float>(numCorrespondences);

    // now the translation only one:
    okvis::kinematics::Transformation T_SaCa, T_SbCb;
    estimator.getCameraSensorStates(olderFrameId, im, T_SaCa);
    estimator.getCameraSensorStates(currentFrameId, im, T_SbCb);
    okvis::kinematics::Transformation T_WSa, T_WSb;
    estimator.get_T_WS(olderFrameId, T_WSa);
    estimator.get_T_WS(currentFrameId, T_WSb);
    adapter.setR12((T_WSa.C() * T_SaCa.C()).transpose() * T_WSb.C() *
                   T_SbCb.C());

    typedef opengv::sac_problems::relative_pose::
        FrameTranslationOnlySacProblem FrameTranslationOnlySacProblem;
    opengv::sac::Ransac<FrameTranslationOnlySacProblem> translation_only_ransac;
    std::shared_ptr<FrameTranslationOnlySacProblem> translation_only_problem_ptr(
        new FrameTranslationOnlySacProblem(adapter));

    translation_only_ransac.sac_model_ = translation_only_problem_ptr;
    translation_only_ransac.threshold_ = 9;     //(1.0 - cos(0.5/600));
    translation_only_ransac.max_iterations_ = 50;

    // run the ransac
    translation_only_ransac.computeModel(0);

    // assess success
    int translation_only_inliers = translation_only_ransac.inliers_.size();
    float translation_only_ratio = static_cast<float>(translation_only_inliers) /
                           static_cast<float>(numCorrespondences);
    // TODO(jhuai): assess the distribution of inliers,
    // so that we can determine the motion type more accurately
    // successful ransac should have evenly distributed inliers, esp for rotation only

    // decide on success and fill inliers
    std::vector<bool> inliers(numCorrespondences, false);
    if (rotation_only_ratio > translation_only_ratio || rotation_only_ratio > 0.8) {
      if (rotation_only_inliers > 10) {
        rotation_only_success = true;
      }
      rmt = ROTATION_ONLY;
      totalInlierNumber += rotation_only_inliers;
      for (size_t k = 0; k < rotation_only_ransac.inliers_.size(); ++k) {
        inliers.at(rotation_only_ransac.inliers_.at(k)) = true;
      }
    } else {
      if (translation_only_inliers > 10) {
        translation_only_success = true;
      }
      rmt = RELATIVE_POSE;
      totalInlierNumber += translation_only_inliers;
      for (size_t k = 0; k < translation_only_ransac.inliers_.size(); ++k) {
        inliers.at(translation_only_ransac.inliers_.at(k)) = true;
      }
    }

    estimator.multiFrame(currentFrameId)
        ->setRelativeMotion(im, olderFrameId, rmt);

    // failure?
    if (!rotation_only_success && !translation_only_success) {
      LOG(INFO) << "2D-2D RANSAC failed";
      continue;
    }
//    else {
//      okvis::Time olderTime = estimator.timestamp(olderFrameId);
//      okvis::Time currentTime = estimator.timestamp(currentFrameId);
//      okvis::Duration gap = currentTime - olderTime;
//      LOG(INFO) << "2D-2D RANSAC "
//                << (translation_only_success ? "rel pose" : "rot only")
//                << " frame distance in time " << gap.toSec();
//    }

    // otherwise: kick out outliers!
    std::shared_ptr<okvis::MultiFrame> multiFrame = estimator.multiFrame(
        currentFrameId);

    for (size_t k = 0; k < numCorrespondences; ++k) {
      size_t idxB = adapter.getMatchKeypointIdxB(k);
      if (!inliers[k]) {
        uint64_t lmId = multiFrame->landmarkId(im, idxB);
        // reset ID:
        multiFrame->setLandmarkId(im, idxB, 0);
        // remove observation
        if (removeOutliers) {
          if (lmId != 0 && estimator.isLandmarkAdded(lmId)){
            estimator.removeObservation(lmId, currentFrameId, im, idxB);
          }
        }
      }
    }
  }

  if (totalInlierNumber <= 15 || rmt == UNCERTAIN_MOTION) {
    *asKeyframe = true;
  }
  if (isInitialized_) {
    if (maxOverlap > keyframeInsertionOverlapThreshold_ &&
        maxMatchRatio > keyframeInsertionMatchingRatioThreshold_) {
      *asKeyframe = *asKeyframe;
    } else {
      *asKeyframe = true;
    }
  }

  if (translation_only_success || rotation_only_success) {
    return totalInlierNumber;
  } else {
    return -1;
  }
}

// (re)instantiates feature detectors and descriptor extractors. Used after
// settings changed or at startup.
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

void HybridFrontend::printNumFeatureDistribution(std::ofstream& /*stream*/) const {
// TODO(jhuai): featureTracker records #feature of every frame and
// then print the histogram
//  featureTracker_.printNumFeatureDistribution(stream);
}

}  // namespace okvis
