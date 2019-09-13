

#include <msckf/HybridFrameNoncentralAbsoluteAdapter.hpp>
#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>

// cameras and distortions
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/FovDistortion.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>

// Constructor.
opengv::absolute_pose::HybridFrameNoncentralAbsoluteAdapter::
    HybridFrameNoncentralAbsoluteAdapter(
        const okvis::HybridFilter& estimator,
        const okvis::cameras::NCameraSystem& nCameraSystem,
        std::shared_ptr<okvis::MultiFrame> frame) {

  size_t numCameras = nCameraSystem.numCameras();

  // find distortion type
  okvis::cameras::NCameraSystem::DistortionType distortionType =
      nCameraSystem.distortionType(0);
  for (size_t i = 1; i < nCameraSystem.numCameras(); ++i) {
    OKVIS_ASSERT_TRUE(Exception,
                      distortionType == nCameraSystem.distortionType(i),
                      "mixed frame types are not supported yet");
  }

  for (size_t im = 0; im < numCameras; ++im) {
    // store transformation. note: the T_SC estimates might actually slightly
    // differ, but we ignore this here.
    camOffsets_.push_back(frame->T_SC(im)->r());
    camRotations_.push_back(frame->T_SC(im)->C());

    // iterate through all the keypoints
    const size_t numK = frame->numKeypoints(im);
    int noCorrespondences = 0;
    for (size_t k = 0; k < numK; ++k) {
      uint64_t lmId = frame->landmarkId(im, k);

      // check if in the map and good enough
      if (lmId == 0 || !estimator.isLandmarkAdded(lmId)) continue;
      okvis::MapPoint landmark;
      estimator.getLandmark(lmId, landmark);
      if (landmark.observations.size() < 2) continue;

      // get it
      const Eigen::Vector4d hp = landmark.pointHomog;

      // check if not at infinity
      if (fabs(hp[3]) < 1.0e-8) continue;

      // add landmark here
      points_.push_back(hp.head<3>() / hp[3]);

      // also add bearing vector
      Eigen::Vector3d bearing;
      Eigen::Vector2d keypoint;
      frame->getKeypoint(im, k, keypoint);
      double keypointStdDev;
      frame->getKeypointSize(im, k, keypointStdDev);
      keypointStdDev = 0.8 * keypointStdDev / 12.0;
      double fu = 1.0;
      switch (distortionType) {
        case okvis::cameras::NCameraSystem::RadialTangential: {
          frame
              ->geometryAs<okvis::cameras::PinholeCamera<
                  okvis::cameras::RadialTangentialDistortion> >(im)
              ->backProject(keypoint, &bearing);
          fu = frame
                   ->geometryAs<okvis::cameras::PinholeCamera<
                       okvis::cameras::RadialTangentialDistortion> >(im)
                   ->focalLengthU();
          break;
        }
        case okvis::cameras::NCameraSystem::RadialTangential8: {
          frame
              ->geometryAs<okvis::cameras::PinholeCamera<
                  okvis::cameras::RadialTangentialDistortion8> >(im)
              ->backProject(keypoint, &bearing);
          fu = frame
                   ->geometryAs<okvis::cameras::PinholeCamera<
                       okvis::cameras::RadialTangentialDistortion8> >(im)
                   ->focalLengthU();
          break;
        }
        case okvis::cameras::NCameraSystem::Equidistant: {
          frame
              ->geometryAs<okvis::cameras::PinholeCamera<
                  okvis::cameras::EquidistantDistortion> >(im)
              ->backProject(keypoint, &bearing);
          fu = frame
                   ->geometryAs<okvis::cameras::PinholeCamera<
                       okvis::cameras::EquidistantDistortion> >(im)
                   ->focalLengthU();
          break;
        }
        case okvis::cameras::NCameraSystem::FOV: {
          frame
              ->geometryAs<okvis::cameras::PinholeCamera<
                  okvis::cameras::FovDistortion> >(im)
              ->backProject(keypoint, &bearing);
          fu = frame
                   ->geometryAs<okvis::cameras::PinholeCamera<
                       okvis::cameras::FovDistortion> >(im)
                   ->focalLengthU();
          break;
        }
        default:
          OKVIS_THROW(Exception, "Unsupported distortion type")
          break;
      }

      // also store sigma angle
      sigmaAngles_.push_back(sqrt(2) * keypointStdDev * keypointStdDev /
                             (fu * fu));

      bearing.normalize();
      bearingVectors_.push_back(bearing);

      // count
      noCorrespondences++;

      // store camera index
      camIndices_.push_back(im);

      // store keypoint index
      keypointIndices_.push_back(k);
    }
  }
}

// Retrieve the bearing vector of a correspondence.
opengv::bearingVector_t
opengv::absolute_pose::HybridFrameNoncentralAbsoluteAdapter::getBearingVector(
    size_t index) const {
  assert(index < bearingVectors_.size());
  return bearingVectors_[index];
}

// Retrieve the world point of a correspondence.
opengv::point_t
opengv::absolute_pose::HybridFrameNoncentralAbsoluteAdapter::getPoint(
    size_t index) const {
  assert(index < bearingVectors_.size());
  return points_[index];
}

// Retrieve the position of a camera of a correspondence seen from the viewpoint
// origin.
opengv::translation_t
opengv::absolute_pose::HybridFrameNoncentralAbsoluteAdapter::getCamOffset(
    size_t index) const {
  return camOffsets_[camIndices_[index]];
}

// Retrieve the rotation from a camera of a correspondence to the viewpoint
// origin.
opengv::rotation_t
opengv::absolute_pose::HybridFrameNoncentralAbsoluteAdapter::getCamRotation(
    size_t index) const {
  return camRotations_[camIndices_[index]];
}

// Get the number of correspondences. These are keypoints that have a
// corresponding landmark which is added to the estimator,
// has more than one observation and not at infinity.
size_t opengv::absolute_pose::HybridFrameNoncentralAbsoluteAdapter::
    getNumberCorrespondences() const {
  return points_.size();
}

// Obtain the angular standard deviation in [rad].
double
opengv::absolute_pose::HybridFrameNoncentralAbsoluteAdapter::getSigmaAngle(
    size_t index) {
  return sigmaAngles_[index];
}
