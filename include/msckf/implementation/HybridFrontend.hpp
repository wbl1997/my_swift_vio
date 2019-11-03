#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <okvis/triangulation/ProbabilisticStereoTriangulator.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
template <class CAMERA_GEOMETRY_T>
void loadParameters(const std::shared_ptr<okvis::MultiFrame> multiframe,
                    const okvis::HybridFilter& estimator,
                    feature_tracker::FeatureTracker* feature_tracker) {
  // Camera calibration parameters
  // TODO(jhuai): modify here along with functions in feature_tracker.cpp
  // to deal with multiple distortion models
  int camId = 0;
  std::string distortionName =
      multiframe->geometryAs<CAMERA_GEOMETRY_T>(camId)->distortionType();
  feature_tracker->cam0_distortion_model =
      distortionName.compare("RadialTangentialDistortion") == 0 ? "radtan"
                                                                : "cult";

  uint32_t width =
      multiframe->geometryAs<CAMERA_GEOMETRY_T>(camId)->imageWidth();
  uint32_t height =
      multiframe->geometryAs<CAMERA_GEOMETRY_T>(camId)->imageHeight();
  feature_tracker->cam0_resolution[0] = width;
  feature_tracker->cam0_resolution[1] = height;

  Eigen::VectorXd intrinsics;
  multiframe->geometryAs<CAMERA_GEOMETRY_T>(camId)->getIntrinsics(intrinsics);
  feature_tracker->cam0_intrinsics[0] = intrinsics[0];
  feature_tracker->cam0_intrinsics[1] = intrinsics[1];
  feature_tracker->cam0_intrinsics[2] = intrinsics[2];
  feature_tracker->cam0_intrinsics[3] = intrinsics[3];

  int numProjIntrinsics = CAMERA_GEOMETRY_T::NumProjectionIntrinsics;
  feature_tracker->cam0_distortion_coeffs[0] = intrinsics[numProjIntrinsics];
  feature_tracker->cam0_distortion_coeffs[1] =
      intrinsics[numProjIntrinsics + 1];
  feature_tracker->cam0_distortion_coeffs[2] =
      intrinsics[numProjIntrinsics + 2];
  feature_tracker->cam0_distortion_coeffs[3] =
      intrinsics[numProjIntrinsics + 3];

  uint64_t fId = multiframe->id();
  okvis::kinematics::Transformation T_SC;
  estimator.getCameraSensorStates(fId, camId, T_SC);
  Eigen::Matrix3d R_SC = T_SC.C();
  Eigen::Vector3d t_SC = T_SC.r();
  cv::eigen2cv(R_SC, feature_tracker->R_cam0_imu);
  cv::eigen2cv(t_SC, feature_tracker->t_cam0_imu);

  if (multiframe->numFrames() > 1) {
    int camId = 1;
    std::string distortionName =
        multiframe->geometryAs<CAMERA_GEOMETRY_T>(camId)->distortionType();
    feature_tracker->cam1_distortion_model =
        distortionName.compare("RadialTangentialDistortion") == 0 ? "radtan"
                                                                  : "cult";
    uint32_t width =
        multiframe->geometryAs<CAMERA_GEOMETRY_T>(camId)->imageWidth();
    uint32_t height =
        multiframe->geometryAs<CAMERA_GEOMETRY_T>(camId)->imageHeight();
    feature_tracker->cam1_resolution[0] = width;
    feature_tracker->cam1_resolution[1] = height;

    Eigen::VectorXd intrinsics;
    multiframe->geometryAs<CAMERA_GEOMETRY_T>(camId)->getIntrinsics(intrinsics);
    feature_tracker->cam1_intrinsics[0] = intrinsics[0];
    feature_tracker->cam1_intrinsics[1] = intrinsics[1];
    feature_tracker->cam1_intrinsics[2] = intrinsics[2];
    feature_tracker->cam1_intrinsics[3] = intrinsics[3];

    int numProjIntrinsics = CAMERA_GEOMETRY_T::NumProjectionIntrinsics;
    feature_tracker->cam1_distortion_coeffs[0] = intrinsics[numProjIntrinsics];
    feature_tracker->cam1_distortion_coeffs[1] =
        intrinsics[numProjIntrinsics + 1];
    feature_tracker->cam1_distortion_coeffs[2] =
        intrinsics[numProjIntrinsics + 2];
    feature_tracker->cam1_distortion_coeffs[3] =
        intrinsics[numProjIntrinsics + 3];

    okvis::kinematics::Transformation T_SC;
    estimator.getCameraSensorStates(fId, camId, T_SC);
    Eigen::Matrix3d R_SC = T_SC.C();
    Eigen::Vector3d t_SC = T_SC.r();
    cv::eigen2cv(R_SC, feature_tracker->R_cam1_imu);
    cv::eigen2cv(t_SC, feature_tracker->t_cam1_imu);
  }
  feature_tracker->processor_config.monocular = multiframe->numFrames() == 1;
}

template <class CAMERA_GEOMETRY_T>
int addConstraintToEstimator(
    const std::vector<feature_tracker::FeatureIDType>& curr_ids,
    std::shared_ptr<okvis::MultiFrame> framesInOut,
    okvis::HybridFilter& estimator) {
  uint64_t fIdB = framesInOut->id();
  okvis::kinematics::Transformation T_WSb;
  // T_WS is available because addStates precedes dataAssociationAndInitialization
  estimator.get_T_WS(fIdB, T_WSb);
  const double fourthRoot2 = 1.1892071150; // sqrt(sqrt(2))
  for (size_t im = 0; im < framesInOut->numFrames(); ++im) {
    int keypointIndex = 0;
    okvis::kinematics::Transformation T_SbCb;
    estimator.getCameraSensorStates(fIdB, im, T_SbCb);

    for (auto lmId : curr_ids) {
      bool added = estimator.isLandmarkAdded(lmId);
      framesInOut->setLandmarkId(im, keypointIndex, lmId);
      if (added) {
        bool initialized = estimator.isLandmarkInitialized(lmId);
        if (!initialized) {
          // triangulate the landmark with the head and tail observations
          // which presumably have the largest disparity
          okvis::KeypointIdentifier IdA;
          estimator.getLandmarkHeadObs(lmId, &IdA);
          okvis::kinematics::Transformation T_WSa;
          estimator.get_T_WS(IdA.frameId, T_WSa);

          okvis::kinematics::Transformation T_SaCa;
          estimator.getCameraSensorStates(IdA.frameId, IdA.cameraIndex, T_SaCa);
          okvis::kinematics::Transformation T_WCa = T_WSa * T_SaCa;
          okvis::kinematics::Transformation T_CaCb = T_WCa.inverse() * (T_WSb * T_SbCb);

          okvis::MultiFramePtr frameA = estimator.multiFrame(IdA.frameId);
          okvis::MultiFramePtr frameB = framesInOut;
          okvis::triangulation::ProbabilisticStereoTriangulator<
              CAMERA_GEOMETRY_T>
              probabilisticStereoTriangulator(frameA, frameB, IdA.cameraIndex,
                                              im, T_CaCb);


          Eigen::Vector4d hP_Ca;
          bool canBeInitialized;  // It is essentially if two rays are NOT parallel.

          double fA = frameA->geometryAs<CAMERA_GEOMETRY_T>(IdA.cameraIndex)->focalLengthU();
          double keypointAStdDev;
          frameA->getKeypointSize(IdA.cameraIndex, IdA.keypointIndex, keypointAStdDev);
          keypointAStdDev = 0.8 * keypointAStdDev / 12.0;
          double raySigma = fourthRoot2 * keypointAStdDev / fA;

          // valid tells if all involved Chi2's are small enough.
          bool valid = probabilisticStereoTriangulator.stereoTriangulate(
              IdA.keypointIndex, keypointIndex, hP_Ca, canBeInitialized,
              raySigma, true);

          estimator.addLandmarkObservation(lmId, fIdB, im, keypointIndex);
          if (valid && canBeInitialized) {
            estimator.setLandmark(lmId, T_WCa * hP_Ca);
            estimator.setLandmarkInitialized(lmId, true);
            estimator.replaceEpipolarWithReprojectionErrors<CAMERA_GEOMETRY_T>(lmId);
          } else {
            estimator.addEpipolarConstraint<CAMERA_GEOMETRY_T>(lmId);
          }
        } else { // The landmark has been initialized.
          estimator.addObservation<CAMERA_GEOMETRY_T>(lmId, fIdB, im,
                                                      keypointIndex);
        }
      } else { // add a landmark with only one observation.
        Eigen::Matrix<double, 4, 1> hP_W;
        hP_W.setZero();
        // addLandmark is alright because ceres solver can single it out if
        // the landmark has no related residual terms.
        // But do not use addObservation which adds a residual to the solver
        // because we only have one observation for the landmark which
        // is not properly initialized
        estimator.addLandmark(lmId, hP_W);
        OKVIS_ASSERT_TRUE(std::runtime_error, estimator.isLandmarkAdded(lmId),
                          lmId << " not added, bug");
        estimator.setLandmarkInitialized(lmId, false);
        estimator.addLandmarkObservation(lmId, fIdB, im, keypointIndex);
      }
      ++keypointIndex;
    }
  }
  return 0;
}
}  // namespace okvis
