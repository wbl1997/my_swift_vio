#include <msckf/SimulationFrontend.hpp>
#include <msckf/implementation/HybridFrontend.hpp>

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

DECLARE_int32(estimator_algorithm);

/// \brief okvis Main namespace of this package.
namespace okvis {

const double SimulationFrontend::imageNoiseMag_ = 1.0;

void create_landmark_grid(
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
        *homogeneousPoints,
    std::vector<uint64_t> *lmIds, std::string pointFile = "") {
  //        const double xyLimit = 10, zLimit = 5,
  //            xyzIncrement = 0.5, offsetNoiseMag = 0.1;
  //        const double xyLimit = 5, zLimit = 2.5,
  //            xyzIncrement = 0.25, offsetNoiseMag = 0.05;
  const double xyLimit = 5, zLimit = 2.5, xyzIncrement = 0.5,
               offsetNoiseMag = 0.05;
  // four walls
  double x(xyLimit), y(xyLimit), z(zLimit);
  for (y = -xyLimit; y <= xyLimit; y += xyzIncrement) {
    for (z = -zLimit; z <= zLimit; z += xyzIncrement) {
      homogeneousPoints->push_back(
          Eigen::Vector4d(x + vio::gauss_rand(0, offsetNoiseMag),
                          y + vio::gauss_rand(0, offsetNoiseMag),
                          z + vio::gauss_rand(0, offsetNoiseMag), 1));
      lmIds->push_back(okvis::IdProvider::instance().newId());
    }
  }

  for (y = -xyLimit; y <= xyLimit; y += xyzIncrement) {
    for (z = -zLimit; z <= zLimit; z += xyzIncrement) {
      homogeneousPoints->push_back(
          Eigen::Vector4d(y + vio::gauss_rand(0, offsetNoiseMag),
                          x + vio::gauss_rand(0, offsetNoiseMag),
                          z + vio::gauss_rand(0, offsetNoiseMag), 1));
      lmIds->push_back(okvis::IdProvider::instance().newId());
    }
  }

  x = -xyLimit;
  for (y = -xyLimit; y <= xyLimit; y += xyzIncrement) {
    for (z = -zLimit; z <= zLimit; z += xyzIncrement) {
      homogeneousPoints->push_back(
          Eigen::Vector4d(x + vio::gauss_rand(0, offsetNoiseMag),
                          y + vio::gauss_rand(0, offsetNoiseMag),
                          z + vio::gauss_rand(0, offsetNoiseMag), 1));
      lmIds->push_back(okvis::IdProvider::instance().newId());
    }
  }

  for (y = -xyLimit; y <= xyLimit; y += xyzIncrement) {
    for (z = -zLimit; z <= zLimit; z += xyzIncrement) {
      homogeneousPoints->push_back(
          Eigen::Vector4d(y + vio::gauss_rand(0, offsetNoiseMag),
                          x + vio::gauss_rand(0, offsetNoiseMag),
                          z + vio::gauss_rand(0, offsetNoiseMag), 1));
      lmIds->push_back(okvis::IdProvider::instance().newId());
    }
  }
//  // top
//  z = zLimit;
//  for (y = -xyLimit; y <= xyLimit; y += xyzIncrement) {
//    for (x = -xyLimit; x <= xyLimit; x += xyzIncrement) {
//      homogeneousPoints->push_back(
//          Eigen::Vector4d(x + vio::gauss_rand(0, offsetNoiseMag),
//                          y + vio::gauss_rand(0, offsetNoiseMag),
//                          z + vio::gauss_rand(0, offsetNoiseMag), 1));
//      lmIds->push_back(okvis::IdProvider::instance().newId());
//    }
//  }
//  // bottom
//  z = -zLimit;
//  for (y = -xyLimit; y <= xyLimit; y += xyzIncrement) {
//    for (x = -xyLimit; x <= xyLimit; x += xyzIncrement) {
//      homogeneousPoints->push_back(
//          Eigen::Vector4d(x + vio::gauss_rand(0, offsetNoiseMag),
//                          y + vio::gauss_rand(0, offsetNoiseMag),
//                          z + vio::gauss_rand(0, offsetNoiseMag), 1));
//      lmIds->push_back(okvis::IdProvider::instance().newId());
//    }
//  }

  // save these points into file
  if (pointFile.size()) {
    std::ofstream pointStream(pointFile, std::ofstream::out);
    pointStream << "%id, x, y, z in the world frame " << std::endl;
    auto iter = homogeneousPoints->begin();
    for (auto it = lmIds->begin(); it != lmIds->end(); ++it, ++iter)
      pointStream << *it << " " << (*iter)[0] << " " << (*iter)[1] << " "
                  << (*iter)[2] << std::endl;
    pointStream.close();
    assert(iter == homogeneousPoints->end());
  }
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
// Constructor.
SimulationFrontend::SimulationFrontend(
    size_t numCameras, bool addImageNoise,
    int maxTrackLength, std::string pointFile)
    : isInitialized_(true), numCameras_(numCameras),
      addImageNoise_(addImageNoise),
      maxTrackLength_(maxTrackLength) {
  create_landmark_grid(&homogeneousPoints_, &lmIds_, pointFile);

  double axisSigma = 0.1;
  addLandmarkNoise(homogeneousPoints_, &noisyHomogeneousPoints_, axisSigma);
}

// Matching as well as initialization of landmarks and state.
int SimulationFrontend::dataAssociationAndInitialization(
    okvis::Estimator& estimator, okvis::kinematics::Transformation& T_WS_ref,
    std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystemRef,
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

  size_t numFrames = framesInOut->numFrames();
  std::vector<std::vector<size_t>> frame_landmark_index;
  std::vector<std::vector<cv::KeyPoint>> frame_keypoints;
  // project landmarks onto frames of framesInOut
  for (size_t i = 0; i < framesInOut->numFrames(); ++i) {
    std::vector<size_t> lmk_indices;
    std::vector<cv::KeyPoint> keypoints;
    for (size_t j = 0; j < homogeneousPoints_.size(); ++j) {
      Eigen::Vector2d projection;
      Eigen::Vector4d point_C = cameraSystemRef->T_SC(i)->inverse() *
                                T_WS_ref.inverse() * homogeneousPoints_[j];
      okvis::cameras::CameraBase::ProjectionStatus status =
          cameraSystemRef->cameraGeometry(i)->projectHomogeneous(point_C,
                                                                 &projection);
      if (status == okvis::cameras::CameraBase::ProjectionStatus::Successful) {
        Eigen::Vector2d measurement(projection);
        if (addImageNoise_) {
          measurement[0] += vio::gauss_rand(0, imageNoiseMag_);
          measurement[1] += vio::gauss_rand(0, imageNoiseMag_);
        }
        keypoints.emplace_back(measurement[0], measurement[1], 8.0);
        lmk_indices.emplace_back(j);
      }
    }
    frame_landmark_index.emplace_back(lmk_indices);
    frame_keypoints.emplace_back(keypoints);
    framesInOut->resetKeypoints(i, keypoints);
  }

  // add landmark observations in the framesInOut to the estimator
  int trackedFeatures = 0;
  for (size_t i = 0; i < numFrames; ++i) {
    const std::vector<size_t>& lmk_indices = frame_landmark_index[i];
    const std::vector<cv::KeyPoint>& keypoints = frame_keypoints[i];
    for (size_t k = 0; k < keypoints.size(); ++k) {
      size_t j = lmk_indices[k];
      size_t numObs = estimator.numObservations(lmIds_[j]);
      if (numObs == 0) {
        if (isFilteringMethod(FLAGS_estimator_algorithm)) {
          // use dummy values to keep info secret from the estimator
          Eigen::Vector4d unknown = Eigen::Vector4d::Zero();
          estimator.addLandmark(lmIds_[j], unknown);
        } else {
          // TODO(jhuai): use more sophisticated schemes to simulate frontend
          //                  Eigen::Vector2d kp{keypoints[k].pt.x,
          //                  keypoints[k].pt.y}; Eigen::Vector3d xy1; bool
          //                  backProjOk =
          //                      cameraSystemRef->cameraGeometry(i)->backProject(kp,
          //                      &xy1);
          //                  if (backProjOk) {
          //                    Eigen::Vector3d infinityGuess =
          //                        (T_WS_ref *
          //                        (*(cameraSystemRef->T_SC(i)))).inverse() *
          //                        xy1 * 1e3;
          //                    estimator->addLandmark(lmIds_[j],
          //                    noisyHomogeneousPoints_[j]);
          //                  } else {
          //                    continue;
          //                  }
          estimator.addLandmark(lmIds_[j], homogeneousPoints_[j]);
        }
        estimator.addObservation<okvis::cameras::PinholeCamera<
            okvis::cameras::RadialTangentialDistortion>>(
            lmIds_[j], framesInOut->id(), i, k);
        ++trackedFeatures;
      } else {
        double sam = vio::Sample::uniform();
        if (sam > static_cast<double>(numObs) /
                      static_cast<double>(maxTrackLength_)) {
          estimator.addObservation<okvis::cameras::PinholeCamera<
              okvis::cameras::RadialTangentialDistortion>>(
              lmIds_[j], framesInOut->id(), i, k);
          ++trackedFeatures;
        }
      }
    }
  }

// need to modify addConstraintToEstimator which assumes curr_ids are the same for stereo frames
//  switch (distortionType) {
//    case okvis::cameras::NCameraSystem::RadialTangential: {
//      //          curr_ids: current keypoint's corresponding landmark ids
//      addConstraintToEstimator<okvis::cameras::PinholeCamera<
//          okvis::cameras::RadialTangentialDistortion>>(frame_landmark_index, framesInOut,
//                                                       estimator);
//      break;
//    }
//    case okvis::cameras::NCameraSystem::Equidistant: {
//      addConstraintToEstimator<
//          okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>>(
//          curr_ids, framesInOut, estimator);

//      break;
//    }
//    case okvis::cameras::NCameraSystem::RadialTangential8: {
//      addConstraintToEstimator<okvis::cameras::PinholeCamera<
//          okvis::cameras::RadialTangentialDistortion8>>(curr_ids, framesInOut,
//                                                        estimator);

//      break;
//    }
//    case okvis::cameras::NCameraSystem::FOV: {
//      addConstraintToEstimator<
//          okvis::cameras::PinholeCamera<okvis::cameras::FovDistortion>>(
//          curr_ids, framesInOut, estimator);

//      break;
//    }
//    default:
//      OKVIS_THROW(Exception, "Unsupported distortion type.")
//      break;
//  }

  *asKeyframe = *asKeyframe || doWeNeedANewKeyframe(estimator, framesInOut);
  if (trackedFeatures <= requiredMatches) {
    LOG(WARNING) << "Tracking landmarks failure. Number of 3d2d-matches: "
                 << trackedFeatures;
  }
  return trackedFeatures;
}

// Decision whether a new frame should be keyframe or not.
// count the number of common landmarks in the current frame and the previous keyframe
bool SimulationFrontend::doWeNeedANewKeyframe(
    const okvis::Estimator& estimator,
    std::shared_ptr<okvis::MultiFrame> currentFrame) {

  if (estimator.numFrames() < 2) {
    // just starting, so yes, we need this as a new keyframe
    return true;
  }

  if (!isInitialized_)
    return false;

  double overlap = 0.0;
  double ratio = 0.0;
  return true;
}

void SimulationFrontend::printNumFeatureDistribution(std::ofstream& /*stream*/) const {
// TODO(jhuai): featureTracker records #feature of every frame and
// then print the histogram
//  featureTracker_.printNumFeatureDistribution(stream);
}

}  // namespace okvis
