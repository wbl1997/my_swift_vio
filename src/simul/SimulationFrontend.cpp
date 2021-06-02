#include <simul/SimulationFrontend.hpp>

#include <algorithm>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <vio/Sample.h>

#include <swift_vio/implementation/HybridFrontend.hpp>

#include <okvis/IdProvider.hpp>

// cameras and distortions
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/FovDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>

#include <simul/PointLandmarkSimulationRS.hpp>

namespace simul {

const double SimulationFrontend::fourthRoot2_ = 1.1892071150;
const int SimulationFrontend::kMaxMatchKeyframes = 2;
const double SimulationFrontend::kMinKeyframeDistance = 0.4;
const double SimulationFrontend::kMinKeyframeAngle = 10 * M_PI / 180;

SimulationFrontend::SimulationFrontend(
    const std::vector<Eigen::Vector4d,
                      Eigen::aligned_allocator<Eigen::Vector4d>>
        &homogeneousPoints,
    const std::vector<uint64_t> &lmIds, size_t numCameras, int maxTrackLength)
    : isInitialized_(true), numCameras_(numCameras),
      maxTrackLength_(maxTrackLength), homogeneousPoints_(homogeneousPoints),
      lmIds_(lmIds) {}

int SimulationFrontend::dataAssociationAndInitialization(
    okvis::Estimator& estimator, const std::vector<std::vector<int>>& keypointIndices,
    std::shared_ptr<okvis::MultiFrame> nframes, bool* asKeyframe) {
  okvis::cameras::NCameraSystem::DistortionType distortionType =
      nframes->cameraSystem().distortionType(0);
  int requiredMatches = 5;
  int trackedFeatures = 0;
  if (estimator.numFrames() > 1) {
    // Find matches between a previous keyframe and current frame.
    std::vector<LandmarkKeypointMatch> landmarkKeyframeMatches;
    int numKeyframes = 0;
    for (size_t age = 1; age < estimator.numFrames(); ++age) {
      uint64_t olderFrameId = estimator.frameIdByAge(age);
      if (!estimator.isKeyframe(olderFrameId))
        continue;

      auto rit = nframeList_.rbegin();
      for (; rit != nframeList_.rend(); ++rit) {
        if (rit->nframe_->id() == olderFrameId) {
          OKVIS_ASSERT_TRUE(Exception, rit->isKeyframe_,
                            "Inconsistent frontend and backend frame status!");
          break;
        }
      }
      matchToFrame(rit->keypointIndices_, keypointIndices, rit->nframe_->id(),
                   nframes->id(), &landmarkKeyframeMatches);
      switch (distortionType) {
      case okvis::cameras::NCameraSystem::RadialTangential: {
        trackedFeatures += addMatchToEstimator<okvis::cameras::PinholeCamera<
            okvis::cameras::RadialTangentialDistortion>>(
            estimator, rit->nframe_, nframes, landmarkKeyframeMatches);
        break;
      }
      case okvis::cameras::NCameraSystem::Equidistant: {
        trackedFeatures += addMatchToEstimator<okvis::cameras::PinholeCamera<
            okvis::cameras::EquidistantDistortion>>(
            estimator, rit->nframe_, nframes, landmarkKeyframeMatches);
        break;
      }
      case okvis::cameras::NCameraSystem::RadialTangential8: {
        trackedFeatures += addMatchToEstimator<okvis::cameras::PinholeCamera<
            okvis::cameras::RadialTangentialDistortion8>>(
            estimator, rit->nframe_, nframes, landmarkKeyframeMatches);
        break;
      }
      case okvis::cameras::NCameraSystem::FOV: {
        trackedFeatures += addMatchToEstimator<
            okvis::cameras::PinholeCamera<okvis::cameras::FovDistortion>>(
            estimator, rit->nframe_, nframes, landmarkKeyframeMatches);
        break;
      }
      default:
        OKVIS_THROW(Exception, "Unsupported distortion type.")
        break;
      }
      ++numKeyframes;

      if (numKeyframes >= kMaxMatchKeyframes) {
        break;
      }
    }
    // find matches between the previous frame and current frame.
    uint64_t lastFrameId = estimator.frameIdByAge(1);
    if (!estimator.isKeyframe(lastFrameId)) {
      auto lastNFrame = nframeList_.rbegin();
      OKVIS_ASSERT_EQ(Exception, lastNFrame->nframe_->id(), lastFrameId,
                      "Inconsistent frontend and backend frame status!");

      std::vector<LandmarkKeypointMatch> landmarkFrameMatches;
      matchToFrame(lastNFrame->keypointIndices_, keypointIndices,
                   lastNFrame->nframe_->id(), nframes->id(),
                   &landmarkFrameMatches);
      switch (distortionType) {
      case okvis::cameras::NCameraSystem::RadialTangential: {
        trackedFeatures += addMatchToEstimator<okvis::cameras::PinholeCamera<
            okvis::cameras::RadialTangentialDistortion>>(
            estimator, lastNFrame->nframe_, nframes, landmarkFrameMatches);
        break;
      }
      case okvis::cameras::NCameraSystem::Equidistant: {
        trackedFeatures += addMatchToEstimator<okvis::cameras::PinholeCamera<
            okvis::cameras::EquidistantDistortion>>(
            estimator, lastNFrame->nframe_, nframes, landmarkFrameMatches);
        break;
      }
      case okvis::cameras::NCameraSystem::RadialTangential8: {
        trackedFeatures += addMatchToEstimator<okvis::cameras::PinholeCamera<
            okvis::cameras::RadialTangentialDistortion8>>(
            estimator, lastNFrame->nframe_, nframes, landmarkFrameMatches);
        break;
      }
      case okvis::cameras::NCameraSystem::FOV: {
        trackedFeatures += addMatchToEstimator<
            okvis::cameras::PinholeCamera<okvis::cameras::FovDistortion>>(
            estimator, lastNFrame->nframe_, nframes, landmarkFrameMatches);
        break;
      }
      default:
        OKVIS_THROW(Exception, "Unsupported distortion type.")
        break;
      }
    }
    if (trackedFeatures <= requiredMatches) {
      LOG(WARNING) << "Tracking landmarks failure. Number of 3d2d-matches: "
                   << trackedFeatures;
    }
  }

  okvis::kinematics::Transformation current_T_WS;
  estimator.get_T_WS(nframes->id(), current_T_WS);
  *asKeyframe = *asKeyframe || doWeNeedANewKeyframe(estimator, nframes);
  if (*asKeyframe) {
    previousKeyframePose_ = current_T_WS;
  }
  nframeList_.emplace_back(nframes, current_T_WS, keypointIndices, *asKeyframe);
  uint64_t oldestFrameId = estimator.oldestFrameId();
  while (nframeList_.front().nframe_->id() < oldestFrameId) {
    nframeList_.pop_front();
  }

  return trackedFeatures;
}

bool SimulationFrontend::doWeNeedANewKeyframe(
    const okvis::Estimator& estimator,
    std::shared_ptr<okvis::MultiFrame> currentFrame) const {
  if (estimator.numFrames() < 2) {
    return true;
  }

  if (!isInitialized_)
    return false;

  okvis::kinematics::Transformation current_T_WS;
  estimator.get_T_WS(currentFrame->id(), current_T_WS);
  okvis::kinematics::Transformation T_SpSc = previousKeyframePose_.inverse() * current_T_WS;
  double distance = T_SpSc.r().norm();
  double rotAngle = std::acos(T_SpSc.q().w()) * 2;
  if (distance > kMinKeyframeDistance || rotAngle > kMinKeyframeAngle) {
    return true;
  } else {
    return false;
  }
}

void SimulationFrontend::printNumFeatureDistribution(std::ofstream& /*stream*/) const {
// TODO(jhuai): featureTracker records #feature of every frame and
// then print the histogram
//  featureTracker_.printNumFeatureDistribution(stream);
}

// TODO(jhuai): make it work with multiple cameras.
template <class CAMERA_GEOMETRY_T>
int SimulationFrontend::addMatchToEstimator(
    okvis::Estimator& estimator, std::shared_ptr<okvis::MultiFrame> prevFrames,
    std::shared_ptr<okvis::MultiFrame> currFrames,
    const std::vector<LandmarkKeypointMatch>& landmarkMatches) const {
  int trackedFeatures = 0;
  for (auto landmarkMatch : landmarkMatches) {
    uint64_t lmIdPrevious = prevFrames->landmarkId(
        landmarkMatch.previousKeypoint.cameraIndex,
        landmarkMatch.previousKeypoint.keypointIndex);
    uint64_t lmIdCurrent =
        currFrames->landmarkId(landmarkMatch.currentKeypoint.cameraIndex,
                               landmarkMatch.currentKeypoint.keypointIndex);
    if (lmIdPrevious != 0 && estimator.isLandmarkAdded(lmIdPrevious)) {
      if (lmIdCurrent != 0) { // avoid duplicates.
        if (lmIdPrevious != lmIdCurrent) {
          LOG(WARNING) << "Different landmarks " << lmIdPrevious << " and "
                    << lmIdCurrent << " are involved in a feature match!";
        }
        continue;
      }
      CHECK_EQ(lmIdPrevious, landmarkMatch.landmarkId);
      size_t numObs = estimator.numObservations(lmIdPrevious);
      double sam = vio::Sample::uniform();
      if (sam >
          static_cast<double>(numObs) / static_cast<double>(maxTrackLength_)) {
        currFrames->setLandmarkId(landmarkMatch.currentKeypoint.cameraIndex,
                                  landmarkMatch.currentKeypoint.keypointIndex,
                                  lmIdPrevious);
        estimator.addObservation<CAMERA_GEOMETRY_T>(
            lmIdPrevious, currFrames->id(),
            landmarkMatch.currentKeypoint.cameraIndex,
            landmarkMatch.currentKeypoint.keypointIndex);
        ++trackedFeatures;
      } // else pass
    } else {
      // This happens when either both observations are not associated to any landmark yet, or
      // a landmark added earlier has been marginalized from the estimator but
      // its observations in the multiframe is not nullified yet.
      okvis::KeypointIdentifier IdA = landmarkMatch.previousKeypoint;
      okvis::KeypointIdentifier IdB = landmarkMatch.currentKeypoint;

      okvis::kinematics::Transformation T_WSa;
      okvis::kinematics::Transformation T_WSb;
      // Use estimated values rather than reference ones to triangulate the landmark.
      estimator.get_T_WS(IdA.frameId, T_WSa);
      estimator.get_T_WS(IdB.frameId, T_WSb);

      okvis::kinematics::Transformation T_SaCa;
      estimator.getCameraSensorExtrinsics(IdA.frameId, IdA.cameraIndex, T_SaCa);

      okvis::kinematics::Transformation T_SbCb;
      estimator.getCameraSensorExtrinsics(IdB.frameId, IdB.cameraIndex, T_SbCb);
      okvis::kinematics::Transformation T_WCa = T_WSa * T_SaCa;
      okvis::kinematics::Transformation T_CaCb =
          T_WCa.inverse() * (T_WSb * T_SbCb);

      okvis::MultiFramePtr frameA = prevFrames;
      okvis::MultiFramePtr frameB = currFrames;
      okvis::triangulation::ProbabilisticStereoTriangulator<CAMERA_GEOMETRY_T>
          probabilisticStereoTriangulator(frameA, frameB, IdA.cameraIndex,
                                          IdB.cameraIndex, T_CaCb);

      Eigen::Vector4d hP_Ca;
      bool canBeInitialized;  // It is essentially if two rays are NOT parallel.

      double fA = frameA->geometryAs<CAMERA_GEOMETRY_T>(IdA.cameraIndex)
                      ->focalLengthU();
      double keypointAStdDev;
      frameA->getKeypointSize(IdA.cameraIndex, IdA.keypointIndex,
                              keypointAStdDev);
      keypointAStdDev = 0.8 * keypointAStdDev / 12.0;
      double raySigma = fourthRoot2_ * keypointAStdDev / fA;

      // valid tells if all involved Chi2's are small enough.
      bool valid = probabilisticStereoTriangulator.stereoTriangulate(
          IdA.keypointIndex, IdB.keypointIndex, hP_Ca, canBeInitialized,
          raySigma, true);

      if (valid) {
        // For filtering methods with delayed initialization, landmarks need
        // not be initialized successfully at construction.
        prevFrames->setLandmarkId(landmarkMatch.previousKeypoint.cameraIndex,
                                  landmarkMatch.previousKeypoint.keypointIndex,
                                  landmarkMatch.landmarkId);
        currFrames->setLandmarkId(landmarkMatch.currentKeypoint.cameraIndex,
                                  landmarkMatch.currentKeypoint.keypointIndex,
                                  landmarkMatch.landmarkId);

        Eigen::Vector4d hP_W = homogeneousPoints_[landmarkMatch.landmarkIdInVector];
        // Use estimated landmark position because true position does not
        // affect VIO results much.
        hP_W = T_WCa * hP_Ca;
        bool inserted = estimator.addLandmark(landmarkMatch.landmarkId, hP_W);
        estimator.setLandmarkInitialized(landmarkMatch.landmarkId,
                                         canBeInitialized);
        estimator.addObservation<CAMERA_GEOMETRY_T>(
            landmarkMatch.landmarkId, IdA.frameId, IdA.cameraIndex,
            IdA.keypointIndex);
        estimator.addObservation<CAMERA_GEOMETRY_T>(
            landmarkMatch.landmarkId, IdB.frameId, IdB.cameraIndex,
            IdB.keypointIndex);
        trackedFeatures += 2;
      } // else pass
    }
  }
  return trackedFeatures;
}

// TODO(jhuai): make it work with multiple camera images.
int SimulationFrontend::matchToFrame(
    const std::vector<std::vector<int>>& previousKeypointIndices,
    const std::vector<std::vector<int>>& currentKeypointIndices,
    const uint64_t prevFrameId, const uint64_t currFrameId,
    std::vector<LandmarkKeypointMatch>* landmarkMatches) const {
  landmarkMatches->clear();
  int numMatches = 0;
  for (size_t im = 0; im < previousKeypointIndices.size(); ++im) {
    const std::vector<int>& previousLm2Kp = previousKeypointIndices[im];
    const std::vector<int>& currentLm2Kp = currentKeypointIndices[im];
    CHECK_EQ(previousLm2Kp.size(), currentLm2Kp.size())
        << "Number of scene landmarks should be constant";
    for (size_t lm = 0; lm < previousLm2Kp.size(); ++lm) {
      if (previousLm2Kp[lm] != -1 && currentLm2Kp[lm] != -1) {
        LandmarkKeypointMatch lmKpMatch;
        lmKpMatch.previousKeypoint =
            okvis::KeypointIdentifier(prevFrameId, im, previousLm2Kp[lm]);
        lmKpMatch.currentKeypoint =
            okvis::KeypointIdentifier(currFrameId, im, currentLm2Kp[lm]);
        lmKpMatch.landmarkId = lmIds_[lm];
        lmKpMatch.landmarkIdInVector = lm;
        landmarkMatches->push_back(lmKpMatch);
        ++numMatches;
      }
    }
  }
  return numMatches;
}
}  // namespace simul
