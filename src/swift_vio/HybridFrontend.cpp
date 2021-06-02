#include <swift_vio/HybridFrontend.hpp>
#include <swift_vio/implementation/HybridFrontend.hpp>

#include <algorithm>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <swift_vio/FeatureTriangulation.hpp>
#include <swift_vio/FrameMatchingStats.hpp>
#include <swift_vio/FrameTranslationOnlySacProblem.hpp>

#include <okvis/StereoMatchingAlgorithm.hpp>
#include <okvis/VioKeyframeWindowMatchingAlgorithm.hpp>
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

#include <opengv/absolute_pose/FrameNoncentralAbsoluteAdapter.hpp>
#include <opengv/relative_pose/FrameRelativeAdapter.hpp>

#include <opengv/sac/Ransac.hpp>

using namespace okvis;

namespace swift_vio {
HybridFrontend::HybridFrontend(size_t numCameras, const FrontendOptions& frontendOptions)
    : okvis::Frontend(numCameras, frontendOptions) {

}

bool HybridFrontend::dataAssociationAndInitialization(
    okvis::Estimator& estimator,
    // TODO(sleutenegger): why is this not used here?
    okvis::kinematics::Transformation& /*T_WS_propagated*/,
    const okvis::VioParameters& params,
    // TODO(sleutenegger): why is this not used here?
    const std::shared_ptr<okvis::MapPointVector> /*map*/,
    std::shared_ptr<okvis::MultiFrame> nframes,
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
  if (frontendOptions_.featureTrackingMethod == 1) {
    int requiredMatches = 5;
    bool rotationOnly = false;
    // match to last frame
    TimerSwitchable matchToLastFrameTimer("2.4.2 matchToLastFrameKLT");
    switch (distortionType) {
      case okvis::cameras::NCameraSystem::RadialTangential: {
        num3dMatches = matchToLastFrameKLT<
            okvis::cameras::PinholeCamera<
                okvis::cameras::RadialTangentialDistortion> >(
            estimator, params, nframes,
            rotationOnly, false);
        break;
      }
      case okvis::cameras::NCameraSystem::Equidistant: {
        num3dMatches = matchToLastFrameKLT<
            okvis::cameras::PinholeCamera<
                okvis::cameras::EquidistantDistortion> >(
            estimator, params, nframes,
            rotationOnly, false);
        break;
      }
      case okvis::cameras::NCameraSystem::RadialTangential8: {
        num3dMatches = matchToLastFrameKLT<
            okvis::cameras::PinholeCamera<
                okvis::cameras::RadialTangentialDistortion8> >(
            estimator, params, nframes,
            rotationOnly, false);

        break;
      }
      case okvis::cameras::NCameraSystem::FOV: {
        num3dMatches = matchToLastFrameKLT<
            okvis::cameras::PinholeCamera<
                okvis::cameras::FovDistortion> >(
            estimator, params, nframes,
            rotationOnly, false);
        break;
      }
      default:
        OKVIS_THROW(Exception, "Unsupported distortion type.")
        break;
    }
    matchToLastFrameTimer.stop();
    if (!isInitialized_) {
      if (!rotationOnly) {
        isInitialized_ = true;
        LOG(INFO) << "Initialized frontend with KLT.";
      }
    }

    if (num3dMatches <= requiredMatches) {
      LOG(WARNING) << "Tracking last frame failure. Number of 3d2d-matches: " << num3dMatches;
    }
    if (estimator.numFrames() > 1) {
      uint64_t currentFrameId = nframes->id();
      uint64_t lastKeyframeId = estimator.currentKeyframeId();
      bool removeOutliers = false;
      checkMotionByRansac2d2d(estimator, params, currentFrameId, lastKeyframeId,
                              removeOutliers, asKeyframe);
    } else {
      *asKeyframe = true;
    }
    return true;
  }

  // first frame? (did do addStates before, so 1 frame minimum in estimator)
  if (estimator.numFrames() > 1) {
    int requiredMatches = 5;
    double uncertainMatchFraction = 0;
    bool rotationOnly = false;

    if (frontendOptions_.featureTrackingMethod == 0) {
    // match to last keyframe
    TimerSwitchable matchKeyframesTimer("2.4.1 matchToKeyframes");
    switch (distortionType) {
      case okvis::cameras::NCameraSystem::RadialTangential: {
        num3dMatches = matchToKeyframes<
            VioKeyframeWindowMatchingAlgorithm<
                okvis::cameras::PinholeCamera<
                    okvis::cameras::RadialTangentialDistortion> > >(
            estimator, params, nframes->id(), rotationOnly, false,
            &uncertainMatchFraction);
        break;
      }
      case okvis::cameras::NCameraSystem::Equidistant: {
        num3dMatches = matchToKeyframes<
            VioKeyframeWindowMatchingAlgorithm<
                okvis::cameras::PinholeCamera<
                    okvis::cameras::EquidistantDistortion> > >(
            estimator, params, nframes->id(), rotationOnly, false,
            &uncertainMatchFraction);
        break;
      }
      case okvis::cameras::NCameraSystem::RadialTangential8: {
        num3dMatches = matchToKeyframes<
            VioKeyframeWindowMatchingAlgorithm<
                okvis::cameras::PinholeCamera<
                    okvis::cameras::RadialTangentialDistortion8> > >(
            estimator, params, nframes->id(), rotationOnly, false,
            &uncertainMatchFraction);
        break;
      }
      case okvis::cameras::NCameraSystem::FOV: {
        num3dMatches = matchToKeyframes<
            VioKeyframeWindowMatchingAlgorithm<
                okvis::cameras::PinholeCamera<
                    okvis::cameras::FovDistortion> > >(
            estimator, params, nframes->id(), rotationOnly, false,
            &uncertainMatchFraction);
        break;
      }
      default:
        OKVIS_THROW(Exception, "Unsupported distortion type.")
        break;
    }
    matchKeyframesTimer.stop();
    if (!isInitialized_) {
      if (!rotationOnly) {
        isInitialized_ = true;
        LOG(INFO) << "Initialized frontend.";
      }
    }

    if (num3dMatches <= requiredMatches) {
      LOG(WARNING) << "Tracking failure. Number of 3d2d-matches: " << num3dMatches;
    }

    // keyframe decision, at the moment only landmarks that match with keyframe are initialised
    *asKeyframe = *asKeyframe || doWeNeedANewKeyframe(estimator, nframes);
    }
    // match to last frame
    TimerSwitchable matchToLastFrameTimer("2.4.2 matchToLastFrame");
    switch (distortionType) {
      case okvis::cameras::NCameraSystem::RadialTangential: {
        num3dMatches = matchToLastFrame<
            VioKeyframeWindowMatchingAlgorithm<okvis::cameras::PinholeCamera<
                okvis::cameras::RadialTangentialDistortion> > >(
            estimator, params, nframes->id(),
            rotationOnly, false);
        break;
      }
      case okvis::cameras::NCameraSystem::Equidistant: {
        num3dMatches = matchToLastFrame<
            VioKeyframeWindowMatchingAlgorithm<okvis::cameras::PinholeCamera<
                okvis::cameras::EquidistantDistortion> > >(
            estimator, params, nframes->id(),
            rotationOnly, false);
        break;
      }
      case okvis::cameras::NCameraSystem::RadialTangential8: {
        num3dMatches = matchToLastFrame<
            VioKeyframeWindowMatchingAlgorithm<okvis::cameras::PinholeCamera<
                okvis::cameras::RadialTangentialDistortion8> > >(
            estimator, params, nframes->id(),
            rotationOnly, false);

        break;
      }
      case okvis::cameras::NCameraSystem::FOV: {
        num3dMatches = matchToLastFrame<
            VioKeyframeWindowMatchingAlgorithm<okvis::cameras::PinholeCamera<
                okvis::cameras::FovDistortion> > >(
            estimator, params, nframes->id(),
            rotationOnly, false);
        break;
      }
      default:
        OKVIS_THROW(Exception, "Unsupported distortion type.")
        break;
    }
    matchToLastFrameTimer.stop();
    if (frontendOptions_.featureTrackingMethod != 0) {
      if (!isInitialized_) {
        if (!rotationOnly) {
          isInitialized_ = true;
          LOG(INFO) << "Initialized frontend.";
        }
      }
      if (num3dMatches <= requiredMatches) {
        LOG(WARNING) << "Tracking last frame failure. Number of 3d2d-matches: " << num3dMatches;
      }
      // At the moment, landmarks that match with last frame are initialised so
      // checking overlap and matching ratio will not work as well as checking relative motion.
      *asKeyframe = *asKeyframe || doWeNeedANewKeyframePosterior(estimator, nframes);
    }
  } else
    *asKeyframe = true;  // first frame needs to be keyframe

  // do stereo match to get new landmarks
  if (frontendOptions_.stereoMatchWithEpipolarCheck) {
    matchStereoWithEpipolarCheckSwitch(distortionType, estimator, nframes);
  } else {
    matchStereoSwitch(distortionType, estimator, nframes);
  }

  return true;
}



bool HybridFrontend::doWeNeedANewKeyframePosterior(
    const okvis::Estimator& estimator,
    std::shared_ptr<okvis::MultiFrame> currentFrame) {
  if (estimator.numFrames() < 2) {
    // just starting, so yes, we need this as a new keyframe
    return true;
  }

  if (!isInitialized_)
    return false;

  uint64_t latestKeyframeId = estimator.currentKeyframeId();
  std::shared_ptr<okvis::MultiFrame> latestKeyframePtr =
      estimator.multiFrame(latestKeyframeId);

  double overlap = 0.0;
  double ratio = 0.0;

  // go through all the frames and try to match the initialized keypoints
  for (size_t im = 0; im < currentFrame->numFrames(); ++im) {
    okvis::Matches matches;
    opengv::findMatches(estimator, latestKeyframePtr, im, currentFrame, im,
                        &matches);
    double overlapArea;
    double matchingRatio;
    opengv::computeMatchStats(matches, currentFrame, im, &overlapArea,
                              &matchingRatio);
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


// Match a new multiframe to the last frame.
template <class CAMERA_GEOMETRY_T>
int HybridFrontend::matchToLastFrameKLT(
    okvis::Estimator& estimator,
    const okvis::VioParameters& /*params*/,
    std::shared_ptr<okvis::MultiFrame> nframes,
    bool& rotationOnly,
    bool /*usePoseUncertainty*/, bool /*removeOutliers*/) {
  int retCtr = 0;
  rotationOnly = false;

  if (estimator.numFrames() == 1) {
    loadParameters<CAMERA_GEOMETRY_T>(nframes, estimator, &featureTracker_);
    featureTracker_.initialize();
  } else {
    uint64_t fIdB = nframes->id();
    okvis::kinematics::Transformation T_WSb;
    estimator.get_T_WS(fIdB, T_WSb);

    uint64_t fIdA = estimator.frameIdByAge(1);
    okvis::kinematics::Transformation T_WSa;
    estimator.get_T_WS(fIdA, T_WSa);

    std::vector<cv::Matx33f> R_CkCkm1_list;
    for (size_t camId = 0; camId < nframes->numFrames(); ++camId) {
      okvis::kinematics::Transformation T_SC;
      estimator.getCameraSensorExtrinsics(fIdA, camId, T_SC);
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
      nframes->image(0),
      nframes->numFrames() > 1 ?
          nframes->image(1) :
          cv::Mat(),
      feature_tracker::MessageHeader{nframes->timestamp()});

//  featureTracker_.drawFeaturesMono();

  std::vector<feature_tracker::FeatureIDType> curr_ids(0);
  featureTracker_.getCurrentFeatureIds(&curr_ids);
  retCtr += curr_ids.size();

  std::vector<std::vector<cv::KeyPoint>> curr_keypoints(2);
  featureTracker_.getCurrentKeypoints(&curr_keypoints[0], &curr_keypoints[1]);

  featureTracker_.prepareForNextFrame(); // clear many things for next frame

  for (size_t im = 0; im < nframes->numFrames(); ++im) {
    nframes->resetKeypoints(im, curr_keypoints[im]);
  }

  addConstraintToEstimator<CAMERA_GEOMETRY_T>(
      curr_ids, nframes, estimator);
  return retCtr;
}

// Match a new multiframe to the last frame.
template <class MATCHING_ALGORITHM>
int HybridFrontend::matchToLastFrame(
    okvis::Estimator& estimator,
    const okvis::VioParameters& params, const uint64_t currentFrameId,
    bool& rotationOnly,
    bool usePoseUncertainty, bool removeOutliers) {
  if (estimator.numFrames() < 2) {
    // just starting, so yes, we need this as a new keyframe
    return 0;
  }

  uint64_t lastFrameId = estimator.frameIdByAge(1);

  if (frontendOptions_.featureTrackingMethod == 0 &&
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

int HybridFrontend::checkMotionByRansac2d2d(okvis::Estimator& estimator,
                                            const okvis::VioParameters& params,
                                            uint64_t currentFrameId,
                                            uint64_t olderFrameId,
                                            bool removeOutliers,
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
    // This is about 3 pixel in image.
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
    estimator.getCameraSensorExtrinsics(olderFrameId, im, T_SaCa);
    estimator.getCameraSensorExtrinsics(currentFrameId, im, T_SbCb);
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
    // This is about 3 pixels in image. More info at getSelectedDistancesToModel().
    translation_only_ransac.threshold_ = 9;
    translation_only_ransac.max_iterations_ = 50;

    // run the ransac
    translation_only_ransac.computeModel(0);

    // assess success
    int translation_only_inliers = translation_only_ransac.inliers_.size();
    float translation_only_ratio = static_cast<float>(translation_only_inliers) /
                           static_cast<float>(numCorrespondences);

    // decide on success and fill inliers
    std::vector<bool> inliers(numCorrespondences, false);
    if (rotation_only_ratio > translation_only_ratio || rotation_only_ratio > 0.8) {
      if (rotation_only_inliers > 10) {
        rotation_only_success = true;
      }
      rmt = okvis::ROTATION_ONLY;
      totalInlierNumber += rotation_only_inliers;
      for (size_t k = 0; k < rotation_only_ransac.inliers_.size(); ++k) {
        inliers.at(rotation_only_ransac.inliers_.at(k)) = true;
      }
    } else {
      if (translation_only_inliers > 10) {
        translation_only_success = true;
      }
      rmt = okvis::RELATIVE_POSE;
      totalInlierNumber += translation_only_inliers;
      for (size_t k = 0; k < translation_only_ransac.inliers_.size(); ++k) {
        inliers.at(translation_only_ransac.inliers_.at(k)) = true;
      }
    }

    // failure?
    if (!rotation_only_success && !translation_only_success) {
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
  }

  if (totalInlierNumber <= 15 || rmt == okvis::UNCERTAIN_MOTION) {
    *asKeyframe = true;
  }
  if (isInitialized_) {
    if (maxOverlap > keyframeInsertionOverlapThreshold_ &&
        maxMatchRatio > keyframeInsertionMatchingRatioThreshold_) {
      *asKeyframe = *asKeyframe;
    } else {
      *asKeyframe = true;
    }
  } else {
    *asKeyframe = false;
  }

  if (translation_only_success || rotation_only_success) {
    return totalInlierNumber;
  } else {
    return -1;
  }
}

bool HybridFrontend::isDescriptorBasedMatching() const {
  return frontendOptions_.featureTrackingMethod != 1;
}

void HybridFrontend::setLandmarkTriangulationParameters(double triangulationTranslationThreshold,
                                                      double triangulationMaxDepth) {
  swift_vio::Feature::optimization_config.translation_threshold =
      triangulationTranslationThreshold;
  swift_vio::Feature::optimization_config.max_depth =
      triangulationMaxDepth;
}
}  // namespace swift_vio
