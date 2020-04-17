#include <msckf/SimulationFrontend.hpp>

#include <algorithm>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <vio/Sample.h>

#include <okvis/IdProvider.hpp>

// cameras and distortions
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/FovDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>

#include <msckf/PointLandmarkSimulation.hpp>
#include <msckf/implementation/HybridFrontend.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

const double SimulationFrontend::imageNoiseMag_ = 1.0;

const double SimulationFrontend::fourthRoot2_ = 1.1892071150;

const double SimulationFrontend::kRangeThreshold = 20;

void saveLandmarkGrid(
    const std::vector<Eigen::Vector4d,
                      Eigen::aligned_allocator<Eigen::Vector4d>>&
        homogeneousPoints,
    const std::vector<uint64_t>& lmIds, std::string pointFile) {
  // save these points into file
  if (pointFile.size()) {
    std::ofstream pointStream(pointFile, std::ofstream::out);
    pointStream << "%id, x, y, z in the world frame " << std::endl;
    auto iter = homogeneousPoints.begin();
    for (auto it = lmIds.begin(); it != lmIds.end(); ++it, ++iter)
      pointStream << *it << " " << (*iter)[0] << " " << (*iter)[1] << " "
                  << (*iter)[2] << std::endl;
    pointStream.close();
    assert(iter == homogeneousPoints.end());
  }
}

void createBoxLandmarkGrid(
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
        *homogeneousPoints,
    std::vector<uint64_t> *lmIds, std::string pointFile = "") {
  const double xyLimit = 5, zLimit = 1.5,
      xyIncrement = 1.0, zIncrement = 0.5, offsetNoiseMag = 0.0;
  // four walls
  double x(xyLimit), y(xyLimit), z(zLimit);
  for (y = -xyLimit; y <= xyLimit; y += xyIncrement) {
    for (z = -zLimit; z <= zLimit; z += zIncrement) {
      homogeneousPoints->push_back(
          Eigen::Vector4d(x + vio::gauss_rand(0, offsetNoiseMag),
                          y + vio::gauss_rand(0, offsetNoiseMag),
                          z + vio::gauss_rand(0, offsetNoiseMag), 1));
      lmIds->push_back(okvis::IdProvider::instance().newId());
    }
  }

  for (y = -xyLimit; y <= xyLimit; y += xyIncrement) {
    for (z = -zLimit; z <= zLimit; z += zIncrement) {
      homogeneousPoints->push_back(
          Eigen::Vector4d(y + vio::gauss_rand(0, offsetNoiseMag),
                          x + vio::gauss_rand(0, offsetNoiseMag),
                          z + vio::gauss_rand(0, offsetNoiseMag), 1));
      lmIds->push_back(okvis::IdProvider::instance().newId());
    }
  }

  x = -xyLimit;
  for (y = -xyLimit; y <= xyLimit; y += xyIncrement) {
    for (z = -zLimit; z <= zLimit; z += zIncrement) {
      homogeneousPoints->push_back(
          Eigen::Vector4d(x + vio::gauss_rand(0, offsetNoiseMag),
                          y + vio::gauss_rand(0, offsetNoiseMag),
                          z + vio::gauss_rand(0, offsetNoiseMag), 1));
      lmIds->push_back(okvis::IdProvider::instance().newId());
    }
  }

  for (y = -xyLimit; y <= xyLimit; y += xyIncrement) {
    for (z = -zLimit; z <= zLimit; z += zIncrement) {
      homogeneousPoints->push_back(
          Eigen::Vector4d(y + vio::gauss_rand(0, offsetNoiseMag),
                          x + vio::gauss_rand(0, offsetNoiseMag),
                          z + vio::gauss_rand(0, offsetNoiseMag), 1));
      lmIds->push_back(okvis::IdProvider::instance().newId());
    }
  }
  saveLandmarkGrid(*homogeneousPoints, *lmIds, pointFile);
}

void createCylinderLandmarkGrid(
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>*
        homogeneousPoints,
    std::vector<uint64_t>* lmIds, double radius, std::string pointFile = "") {
  const int numSteps = 40;
  double zmin = -1.5, zmax = 1.5;
  double zstep = 0.5;
  if (radius >= SimulationFrontend::kRangeThreshold) {
    zmin = -1.5;
    zmax = 3.5;
    zstep = 0.8;
  }
  double step = 2 * M_PI / numSteps;
  for (int j = 0; j < numSteps; ++j) {
    double theta = step * j;
    double px = radius * sin(theta);
    double py = radius * cos(theta);
    for (double pz = zmin; pz < zmax; pz += zstep) {
      homogeneousPoints->emplace_back(px, py, pz, 1.0);
      lmIds->push_back(okvis::IdProvider::instance().newId());
    }
  }
  saveLandmarkGrid(*homogeneousPoints, *lmIds, pointFile);
}

void addLandmarkNoise(
    const std::vector<Eigen::Vector4d,
                      Eigen::aligned_allocator<Eigen::Vector4d>>&
        homogeneousPoints,
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>*
        noisyHomogeneousPoints,
    double axisSigma = 0.1) {
  *noisyHomogeneousPoints = homogeneousPoints;
  for (auto iter = noisyHomogeneousPoints->begin();
       iter != noisyHomogeneousPoints->end(); ++iter) {
    iter->head<3>() = iter->head<3>() + Eigen::Vector3d::Random() * axisSigma;
  }
}

void initCameraNoiseParams(
    okvis::ExtrinsicsEstimationParameters* cameraNoiseParams,
    double sigma_abs_position, bool fixCameraInteranlParams) {
  cameraNoiseParams->sigma_absolute_translation = sigma_abs_position;
  cameraNoiseParams->sigma_absolute_orientation = 0;
  cameraNoiseParams->sigma_c_relative_translation = 0;
  cameraNoiseParams->sigma_c_relative_orientation = 0;
  if (fixCameraInteranlParams) {
    cameraNoiseParams->sigma_focal_length = 0;
    cameraNoiseParams->sigma_principal_point = 0;
    cameraNoiseParams->sigma_distortion.resize(5, 0);
    cameraNoiseParams->sigma_td = 0;
    cameraNoiseParams->sigma_tr = 0;
  } else {
    cameraNoiseParams->sigma_focal_length = 5;
    cameraNoiseParams->sigma_principal_point = 5;
    cameraNoiseParams->sigma_distortion =
        std::vector<double>{5e-2, 1e-2, 1e-3, 1e-3, 1e-3};
    cameraNoiseParams->sigma_td = 5e-3;
    cameraNoiseParams->sigma_tr = 5e-3;
  }
}

// Constructor.
SimulationFrontend::SimulationFrontend(
    size_t numCameras, bool addImageNoise,
    int maxTrackLength, double landmarkRadius,
    VisualConstraints constraintScheme,
    std::string pointFile)
    : isInitialized_(true), numCameras_(numCameras),
      addImageNoise_(addImageNoise),
      maxTrackLength_(maxTrackLength),
      constraintScheme_(constraintScheme) {
  if (landmarkRadius < SimulationFrontend::kRangeThreshold) {
    createBoxLandmarkGrid(&homogeneousPoints_, &lmIds_, pointFile);
  } else {
    createCylinderLandmarkGrid(&homogeneousPoints_, &lmIds_,
                               landmarkRadius,
                               pointFile);
  }
  double axisSigma = 0.1;
  addLandmarkNoise(homogeneousPoints_, &noisyHomogeneousPoints_, axisSigma);
}

int SimulationFrontend::dataAssociationAndInitialization(
    okvis::Estimator& estimator, okvis::kinematics::Transformation& T_WS_ref,
    std::shared_ptr<const okvis::cameras::NCameraSystem> cameraSystemRef,
    std::shared_ptr<okvis::MultiFrame> framesInOut, bool* asKeyframe) {
  // find distortion type
  okvis::cameras::NCameraSystem::DistortionType distortionType =
      cameraSystemRef->distortionType(0);
  for (size_t i = 1; i < cameraSystemRef->numCameras(); ++i) {
    OKVIS_ASSERT_TRUE(Exception,
                      distortionType == cameraSystemRef->distortionType(i),
                      "mixed frame types are not supported yet");
  }
  int requiredMatches = 5;

  std::vector<std::vector<size_t>> frameLandmarkIndices;
  std::vector<std::vector<int>> keypointIndices;
      PointLandmarkSimulation::projectLandmarksToNFrame(
          homogeneousPoints_, T_WS_ref, cameraSystemRef, framesInOut,
          &frameLandmarkIndices, &keypointIndices,
          addImageNoise_ ? &imageNoiseMag_ : nullptr);

  int trackedFeatures = 0;
  if (estimator.numFrames() > 1) {
    // find matches between the previous keyframe and current frame
    // Matching to last Keyframe should not encounter zombie landmark
    // ids for filters because they remove disappearing landmarks from
    // landmarkMap_ as well as landmark Ids from multiframe.
    std::vector<LandmarkKeypointMatch> landmarkKeyframeMatches;
    matchToFrame(previousKeyframeKeypointIndices_, keypointIndices,
                 previousKeyframe_->id(), framesInOut->id(),
                 &landmarkKeyframeMatches);
    switch (distortionType) {
      case okvis::cameras::NCameraSystem::RadialTangential: {
        trackedFeatures += addMatchToEstimator<okvis::cameras::PinholeCamera<
            okvis::cameras::RadialTangentialDistortion>>(
            estimator, previousKeyframe_, framesInOut, previousKeyframePose_,
            T_WS_ref, landmarkKeyframeMatches);
        break;
      }
      case okvis::cameras::NCameraSystem::Equidistant: {
        trackedFeatures += addMatchToEstimator<okvis::cameras::PinholeCamera<
            okvis::cameras::EquidistantDistortion>>(
            estimator, previousKeyframe_, framesInOut, previousKeyframePose_,
            T_WS_ref, landmarkKeyframeMatches);
        break;
      }
      case okvis::cameras::NCameraSystem::RadialTangential8: {
        trackedFeatures += addMatchToEstimator<okvis::cameras::PinholeCamera<
            okvis::cameras::RadialTangentialDistortion8>>(
            estimator, previousKeyframe_, framesInOut, previousKeyframePose_,
            T_WS_ref, landmarkKeyframeMatches);
        break;
      }
      case okvis::cameras::NCameraSystem::FOV: {
        trackedFeatures += addMatchToEstimator<
            okvis::cameras::PinholeCamera<okvis::cameras::FovDistortion>>(
            estimator, previousKeyframe_, framesInOut, previousKeyframePose_,
            T_WS_ref, landmarkKeyframeMatches);
        break;
      }
      default:
        OKVIS_THROW(Exception, "Unsupported distortion type.")
        break;
    }
    // find matches between the previous frame and current frame
    if (previousKeyframe_->id() != previousFrame_->id()) {
      std::vector<LandmarkKeypointMatch> landmarkFrameMatches;
      matchToFrame(previousFrameKeypointIndices_, keypointIndices,
                   previousFrame_->id(), framesInOut->id(),
                   &landmarkFrameMatches);
      switch (distortionType) {
        case okvis::cameras::NCameraSystem::RadialTangential: {
          trackedFeatures += addMatchToEstimator<okvis::cameras::PinholeCamera<
              okvis::cameras::RadialTangentialDistortion>>(
              estimator, previousFrame_, framesInOut, previousFramePose_,
              T_WS_ref, landmarkFrameMatches);
          break;
        }
        case okvis::cameras::NCameraSystem::Equidistant: {
          trackedFeatures += addMatchToEstimator<okvis::cameras::PinholeCamera<
              okvis::cameras::EquidistantDistortion>>(
              estimator, previousFrame_, framesInOut, previousFramePose_,
              T_WS_ref, landmarkFrameMatches);
          break;
        }
        case okvis::cameras::NCameraSystem::RadialTangential8: {
          trackedFeatures += addMatchToEstimator<okvis::cameras::PinholeCamera<
              okvis::cameras::RadialTangentialDistortion8>>(
              estimator, previousFrame_, framesInOut, previousFramePose_,
              T_WS_ref, landmarkFrameMatches);
          break;
        }
        case okvis::cameras::NCameraSystem::FOV: {
          trackedFeatures += addMatchToEstimator<
              okvis::cameras::PinholeCamera<okvis::cameras::FovDistortion>>(
              estimator, previousFrame_, framesInOut, previousFramePose_,
              T_WS_ref, landmarkFrameMatches);
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

  *asKeyframe = *asKeyframe || doWeNeedANewKeyframe(estimator, framesInOut, T_WS_ref);
  if (asKeyframe) {
    previousKeyframe_ = framesInOut;
    previousKeyframePose_ = T_WS_ref;
    previousKeyframeKeypointIndices_ = keypointIndices;
  }
  previousFrame_ = framesInOut;
  previousFramePose_ = T_WS_ref;
  previousFrameKeypointIndices_ = keypointIndices;
  return trackedFeatures;
}

// Decision whether a new frame should be keyframe or not.
// count the number of common landmarks in the current frame and the previous keyframe
bool SimulationFrontend::doWeNeedANewKeyframe(
    const okvis::Estimator& estimator,
    std::shared_ptr<okvis::MultiFrame> /*currentFrame*/,
    const okvis::kinematics::Transformation& T_WS) const {
  if (estimator.numFrames() < 2) {
    // just starting, so yes, we need this as a new keyframe
    return true;
  }

  if (!isInitialized_)
    return false;

  okvis::kinematics::Transformation T_SpSc = previousKeyframePose_.inverse() * T_WS;
  double distance = T_SpSc.r().norm();
  double rotAngle = std::acos(T_SpSc.q().w()) * 2;
  if (distance > 0.4 || rotAngle > 10 * M_PI / 180) {
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

template <class CAMERA_GEOMETRY_T>
int SimulationFrontend::addMatchToEstimator(
    okvis::Estimator& estimator, std::shared_ptr<okvis::MultiFrame> prevFrames,
    std::shared_ptr<okvis::MultiFrame> currFrames,
    const okvis::kinematics::Transformation& T_WSp_ref,
    const okvis::kinematics::Transformation& T_WSc_ref,
    const std::vector<LandmarkKeypointMatch>& landmarkMatches) const {
  int trackedFeatures = 0;
  for (auto landmarkMatch : landmarkMatches) {
    uint64_t lmIdPrevious = prevFrames->landmarkId(
        landmarkMatch.previousKeypoint.cameraIndex,
        landmarkMatch.previousKeypoint.keypointIndex);
    uint64_t lmIdCurrent =
        currFrames->landmarkId(landmarkMatch.currentKeypoint.cameraIndex,
                               landmarkMatch.currentKeypoint.keypointIndex);
    if (lmIdPrevious != 0) {
      if (lmIdCurrent != 0) { // avoid duplicates
        LOG(INFO) << "Potential duplicates found " << lmIdPrevious
                  << " should == " << lmIdCurrent;
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
        switch (constraintScheme_) {
          case OnlyReprojectionErrors:
            estimator.addObservation<CAMERA_GEOMETRY_T>(
                lmIdPrevious, currFrames->id(),
                landmarkMatch.currentKeypoint.cameraIndex,
                landmarkMatch.currentKeypoint.keypointIndex);
            break;
          case OnlyTwoViewConstraints:
            estimator.addEpipolarConstraint<CAMERA_GEOMETRY_T>(
                lmIdPrevious, currFrames->id(),
                landmarkMatch.currentKeypoint.cameraIndex,
                landmarkMatch.currentKeypoint.keypointIndex,
                singleTwoViewConstraint_);
            break;
          case TwoViewAndReprojection:
            break;
        }
        ++trackedFeatures;
      } // else do nothing
    } else {
      okvis::KeypointIdentifier IdA = landmarkMatch.previousKeypoint;
      okvis::KeypointIdentifier IdB = landmarkMatch.currentKeypoint;

      okvis::kinematics::Transformation T_WSa = T_WSp_ref;
      okvis::kinematics::Transformation T_WSb = T_WSc_ref;
      // TODO(jhuai): do we use estimates or reference values?
      estimator.get_T_WS(IdA.frameId, T_WSa);
      estimator.get_T_WS(IdB.frameId, T_WSb);

      okvis::kinematics::Transformation T_SaCa;
      estimator.getCameraSensorStates(IdA.frameId, IdA.cameraIndex, T_SaCa);

      okvis::kinematics::Transformation T_SbCb;
      estimator.getCameraSensorStates(IdB.frameId, IdB.cameraIndex, T_SbCb);
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
        estimator.addLandmark(landmarkMatch.landmarkId, T_WCa * hP_Ca);
        estimator.setLandmarkInitialized(landmarkMatch.landmarkId,
                                         canBeInitialized);
        switch (constraintScheme_) {
          case OnlyReprojectionErrors:
            estimator.addObservation<CAMERA_GEOMETRY_T>(
                landmarkMatch.landmarkId, IdA.frameId, IdA.cameraIndex,
                IdA.keypointIndex);
            estimator.addObservation<CAMERA_GEOMETRY_T>(
                landmarkMatch.landmarkId, IdB.frameId, IdB.cameraIndex,
                IdB.keypointIndex);
            break;
          case OnlyTwoViewConstraints:
            if (estimator.numObservations(landmarkMatch.landmarkId) + 1 <
                estimator.minTrackLength()) {
              estimator.addLandmarkObservation(landmarkMatch.landmarkId,
                                               IdA.frameId, IdA.cameraIndex,
                                               IdA.keypointIndex);
            } else {
              estimator.addEpipolarConstraint<CAMERA_GEOMETRY_T>(
                  landmarkMatch.landmarkId, IdA.frameId, IdA.cameraIndex,
                  IdA.keypointIndex, singleTwoViewConstraint_);
            }
            estimator.addEpipolarConstraint<CAMERA_GEOMETRY_T>(
                landmarkMatch.landmarkId, IdB.frameId, IdB.cameraIndex,
                IdB.keypointIndex, singleTwoViewConstraint_);
            break;
          case TwoViewAndReprojection:
            break;
        }
        trackedFeatures += 2;
      } // else do nothing
    }
  }
  return trackedFeatures;
}

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
            KeypointIdentifier(prevFrameId, im, previousLm2Kp[lm]);
        lmKpMatch.currentKeypoint =
            KeypointIdentifier(currFrameId, im, currentLm2Kp[lm]);
        lmKpMatch.landmarkId = lmIds_[lm];
        lmKpMatch.landmarkIdInVector = lm;
        landmarkMatches->push_back(lmKpMatch);
        ++numMatches;
      }
    }
  }
  return numMatches;
}
}  // namespace okvis
