#include <gtest/gtest.h>

#include <gtsam/VioBackEndParams.h>
#include <io_wrap/StreamHelper.hpp>
#include <msckf/VioTestSystemBuilder.hpp>
namespace {
void computeNavErrors(
    const okvis::Estimator* estimator,
    const okvis::kinematics::Transformation& T_WS,
    const Eigen::Vector3d& v_WS_true,
    Eigen::VectorXd* navError) {
  okvis::kinematics::Transformation T_WS_est;
  uint64_t currFrameId = estimator->currentFrameId();
  estimator->get_T_WS(currFrameId, T_WS_est);
  Eigen::Vector3d delta = T_WS.r() - T_WS_est.r();
  Eigen::Vector3d alpha = vio::unskew3d(T_WS.C() * T_WS_est.C().transpose() -
                                        Eigen::Matrix3d::Identity());
  Eigen::Matrix<double, 6, 1> deltaPose;
  deltaPose << delta, alpha;

  int index = 0;
  navError->head<3>() = delta;
  index += 3;
  navError->segment<3>(index) = alpha;
  index += 3;
  okvis::SpeedAndBias speedAndBias_est;
  estimator->getSpeedAndBias(currFrameId, 0, speedAndBias_est);
  Eigen::Vector3d deltaV = speedAndBias_est.head<3>() - v_WS_true;
  navError->segment<3>(index) = deltaV;
}

int examinLandmarkStackedJacobian(const okvis::MapPoint& mapPoint,
                                   std::shared_ptr<okvis::MSCKF2> estimator) {
  Eigen::MatrixXd H_oi[2];
  Eigen::Matrix<double, Eigen::Dynamic, 1> r_oi[2];
  Eigen::MatrixXd R_oi[2];
  bool jacOk1 =
      estimator->featureJacobian(mapPoint, H_oi[0], r_oi[0], R_oi[0], nullptr);
  bool jacOk2 = estimator->featureJacobianGeneric(mapPoint, H_oi[1], r_oi[1],
                                                  R_oi[1], nullptr);
  EXPECT_EQ(jacOk1, jacOk2) << "featureJacobian status";
  if (jacOk1 && jacOk2) {
    Eigen::MatrixXd information = R_oi[0].inverse();
    Eigen::LLT<Eigen::MatrixXd> lltOfInformation(information);
    Eigen::MatrixXd squareRootInformation =
        lltOfInformation.matrixL().transpose();
    const int cameraMinimalDimWoTime =
        estimator->cameraParamsMinimalDimen() - 2;
    const int obsMinus3 = H_oi[1].rows();
    H_oi[0] = squareRootInformation * H_oi[0];
    r_oi[0] = squareRootInformation * r_oi[0];
    R_oi[0] =
        squareRootInformation * R_oi[0] * squareRootInformation.transpose();
    Eigen::MatrixXd H_diff = H_oi[1] - H_oi[0];
    Eigen::VectorXd r_diff = r_oi[1] - r_oi[0];
    Eigen::MatrixXd R_diff = R_oi[1] - R_oi[0];

    EXPECT_LT(H_diff.topLeftCorner(obsMinus3, cameraMinimalDimWoTime)
                  .lpNorm<Eigen::Infinity>(),
              1e-6)
        << "H_oi camera params\nH_diff camera params without time\n"
        << H_diff.topLeftCorner(obsMinus3, cameraMinimalDimWoTime);

    Eigen::MatrixXd H_time_diff =
        H_diff.block(0, cameraMinimalDimWoTime, obsMinus3, 2);
    Eigen::MatrixXd H_time_diff_spikes =
        (H_time_diff.cwiseAbs().array() > 0.1)
            .select(H_time_diff.cwiseAbs().array(), 0.0);
    Eigen::MatrixXd H_time_ratio = H_time_diff_spikes.cwiseQuotient(
        (H_oi[0] + H_oi[1])
            .block(0, cameraMinimalDimWoTime, obsMinus3, 2)
            .cwiseAbs());
    EXPECT_LT(H_time_ratio.lpNorm<Eigen::Infinity>(), 4e-1)
        << "H_oi camera td tr\nH_time_diff_spikes\n"
        << H_time_diff_spikes
        << "\n(H_oi[0] + H_oi[1]).block(0, cameraMinimalDimWoTime, obsMinus3, "
           "2)\n"
        << (H_oi[0] + H_oi[1]).block(0, cameraMinimalDimWoTime, obsMinus3, 2)
        << "\nH_time_ratio\n"
        << H_time_ratio;
    EXPECT_LT(H_diff
                  .block(0, cameraMinimalDimWoTime + 2, obsMinus3,
                         H_diff.cols() - cameraMinimalDimWoTime - 2)
                  .lpNorm<Eigen::Infinity>(),
              1e-1)
        << "H_diff rest XBj\nH_oi[1]\n"
        << H_oi[1] << "\nH_oi[0]\n"
        << H_oi[0] << "\nDiff\n"
        << H_diff;

    EXPECT_LT(r_diff.lpNorm<Eigen::Infinity>(), 1e-6) << "r_oi";
    EXPECT_LT(R_diff.lpNorm<Eigen::Infinity>(), 1e-6) << "R_oi";
    return 1;
  }
  return 0;
}

int examineLandmarkMeasurementJacobian(
    const okvis::MapPoint& mapPoint, std::shared_ptr<okvis::MSCKF2> estimator) {
  Eigen::MatrixXd H_oi;
  Eigen::Matrix<double, Eigen::Dynamic, 1> r_oi;
  Eigen::MatrixXd R_oi;
  bool jacOk = estimator->featureJacobianGeneric(mapPoint, H_oi, r_oi,
                                                  R_oi, nullptr);

  // init landmark parameterization

  // calculate Jacobians for at least three measurements
  // compare Jacobians against auto diff

  return jacOk;
}

} // namespace

enum class EstimatorCheck {
  FEATURE_TRACK_JACOBIAN=0,
  FEATURE_MEASUREMENT_JACOBIAN,
  NAVSTATE_COVARIANCE,
};

class EstimatorTest {
public:
  EstimatorTest(std::string _projOptModelName, std::string _extrinsicModelName,
                std::string _outputFile, double _timeOffset, double _readoutTime,
                int _cameraObservationModelId = 0, int _landmarkModelId = 0) :
    projOptModelName(_projOptModelName),
    extrinsicModelName(_extrinsicModelName), outputFile(_outputFile),
    timeOffset(_timeOffset), readoutTime(_readoutTime),
    cameraObservationModelId(_cameraObservationModelId), landmarkModelId(_landmarkModelId) {

  }

  void SetUp() {
    testSetting = okvis::TestSetting(
        true, noisyInitialSpeedAndBiases, false, true, true, noise_factor,
        noise_factor, algorithm, useEpipolarConstraint,
        cameraObservationModelId, landmarkModelId, cameraModelId,
        cameraOrientationId, okvis::LandmarkGridType::FourWalls,
        landmarkRadius);
    vioSystemBuilder.createVioSystem(testSetting, backendParams, trajectoryType,
                                     projOptModelName, extrinsicModelName,
                                     timeOffset, readoutTime, "", "");

    times = vioSystemBuilder.sampleTimes();

    trueBiases = vioSystemBuilder.trueBiases();
    ref_T_WS_list = vioSystemBuilder.ref_T_WS_list();
    imuMeasurements = vioSystemBuilder.imuMeasurements();
    estimator = vioSystemBuilder.mutableEstimator();

    frontend = vioSystemBuilder.mutableFrontend();
    cameraSystem0 = vioSystemBuilder.trueCameraSystem();
    cameraGeometry0 = cameraSystem0->cameraGeometry(0);

    if (!debugStream.is_open()) {
      debugStream.open(outputFile, std::ofstream::out);
      std::string headerLine;
      okvis::StreamHelper::composeHeaderLine(
          vioSystemBuilder.imuModelType(), {extrinsicModelName},
          {projOptModelName}, {vioSystemBuilder.distortionType()},
          okvis::FULL_STATE_WITH_ALL_CALIBRATION, &headerLine);
      debugStream << headerLine << std::endl;
    }
  }

  void Run(EstimatorCheck checkCase) {
    std::vector<uint64_t> multiFrameIds;
    size_t kale = 0; // imu data counter
    bool bStarted = false;
    int frameCount = -1;                // number of frames used in estimator
    int trackedFeatures = 0;            // feature tracks observed in a frame
    const int cameraIntervalRatio = 10; // number imu meas for 1 camera frame

    Eigen::VectorXd navError(9);
    okvis::Time lastKFTime = times.front();
    okvis::ImuMeasurementDeque::const_iterator trueBiasIter = trueBiases.begin();
    for (auto iter = times.begin(), iterEnd = times.end(); iter != iterEnd;
         iter += cameraIntervalRatio, kale += cameraIntervalRatio,
              trueBiasIter += cameraIntervalRatio) {
      okvis::kinematics::Transformation T_WS(ref_T_WS_list[kale]);
      // assemble a multi-frame
      std::shared_ptr<okvis::MultiFrame> mf(new okvis::MultiFrame);
      uint64_t id = okvis::IdProvider::instance().newId();
      mf->setId(id);
      okvis::Time frameStamp = *iter - okvis::Duration(timeOffset);
      mf->setTimestamp(frameStamp);

      // The reference cameraSystem will be used for triangulating landmarks in
      // the frontend which provides observations to the estimator.
      mf->resetCameraSystemAndFrames(*cameraSystem0);
      mf->setTimestamp(0u, frameStamp);

      // reference ID will be and stay the first frame added.
      multiFrameIds.push_back(id);

      okvis::Time currentKFTime = *iter;
      okvis::Time imuDataEndTime = currentKFTime + okvis::Duration(1);
      okvis::Time imuDataBeginTime = lastKFTime - okvis::Duration(1);
      okvis::ImuMeasurementDeque imuSegment = okvis::getImuMeasurements(
          imuDataBeginTime, imuDataEndTime, imuMeasurements, nullptr);
      bool asKeyframe = true;
      // add it in the window to create a new time instance
      if (!bStarted) {
        bStarted = true;
        frameCount = 0;
        estimator->setInitialNavState(vioSystemBuilder.initialNavState());
        estimator->addStates(mf, imuSegment, asKeyframe);
      } else {
        estimator->addStates(mf, imuSegment, asKeyframe);
        ++frameCount;
      }

      // add landmark observations
      trackedFeatures = 0;
      if (testSetting.useImageObservs) {
        trackedFeatures = frontend->dataAssociationAndInitialization(
            *estimator, vioSystemBuilder.sinusoidalTrajectory(), *iter,
            cameraSystem0, mf, &asKeyframe);
        estimator->setKeyframe(mf->id(), asKeyframe);
      }

      examineCase(checkCase, frameCount);

      size_t maxIterations = 10u;
      size_t numThreads = 2u;
      estimator->optimize(maxIterations, numThreads, false);
      okvis::Optimization sharedOptConfig;
      size_t numKeyFrames = 5u;
      size_t numImuFrames = 3u;
      estimator->setKeyframeRedundancyThresholds(
          sharedOptConfig.translationThreshold,
          sharedOptConfig.rotationThreshold,
          sharedOptConfig.trackingRateThreshold, sharedOptConfig.minTrackLength,
          numKeyFrames, numImuFrames);
      okvis::MapPointVector removedLandmarks;

      estimator->applyMarginalizationStrategy(numKeyFrames, numImuFrames,
                                              removedLandmarks);
      estimator->print(debugStream);
      debugStream << std::endl;

      Eigen::Vector3d v_WS_true =
          vioSystemBuilder.sinusoidalTrajectory()->computeGlobalLinearVelocity(
              *iter);

      computeNavErrors(estimator.get(), T_WS, v_WS_true, &navError);

      lastKFTime = currentKFTime;
    } // every keyframe

    LOG(INFO) << "Finishes with last added frame " << frameCount
              << " of tracked features " << trackedFeatures;
    EXPECT_LT(navError.head<3>().lpNorm<Eigen::Infinity>(), 0.5)
        << "Final position error";
    EXPECT_LT(navError.segment<3>(3).lpNorm<Eigen::Infinity>(), 0.2)
        << "Final orientation error";
    EXPECT_LT(navError.tail<3>().lpNorm<Eigen::Infinity>(), 0.2)
        << "Final velocity error";
  }

  void examineCase(EstimatorCheck checkCase, int frameCount) {
    if ((checkCase == EstimatorCheck::FEATURE_TRACK_JACOBIAN ||
         checkCase == EstimatorCheck::FEATURE_MEASUREMENT_JACOBIAN) &&
        frameCount == 100)
      examineJacobian(checkCase ==
                      EstimatorCheck::FEATURE_MEASUREMENT_JACOBIAN);

    if (checkCase == EstimatorCheck::NAVSTATE_COVARIANCE && frameCount % 10 == 0) {
      examineCovariance(frameCount);
    }
  }

  /**
   * @brief examineJacobian Check if the Jacobian of a feature measurement or
   *  the Jacobian of a feature track are correct.
   * @warning Assume the MSCKF2 estimator is used.
   * @param examineMeasurementJacobian true measurement Jacobian, false track
   * Jacobian.
   */
  void examineJacobian(bool examineMeasurementJacobian) const {
    std::shared_ptr<okvis::MSCKF2> filter =
        std::dynamic_pointer_cast<okvis::MSCKF2>(estimator);
    {
      okvis::StatePointerAndEstimateList currentStates;
      filter->cloneFilterStates(&currentStates);
      Eigen::VectorXd deltaX;
      filter->boxminusFromInput(currentStates, &deltaX);
      EXPECT_LT(deltaX.lpNorm<Eigen::Infinity>(), 1e-8);
    }
    okvis::PointMap landmarkMap;
    size_t numLandmarks = estimator->getLandmarks(landmarkMap);
    int featureJacobianLandmarkCount = 0;
    int measurementJacobianCount = 0;
    for (auto mpIter = landmarkMap.begin(); mpIter != landmarkMap.end();
         ++mpIter) {
      uint64_t latestFrameId = estimator->currentFrameId();
      const okvis::MapPoint &mapPoint = mpIter->second;
      auto obsIter = std::find_if(mapPoint.observations.begin(),
                                  mapPoint.observations.end(),
                                  okvis::IsObservedInNFrame(latestFrameId));
      if (obsIter != mapPoint.observations.end()) {
        continue; // only examine observations of landmarks disappeared in
                  // current frame.
      }
      if (examineMeasurementJacobian) {
        int examined =
            examineLandmarkMeasurementJacobian(mapPoint, filter);
        measurementJacobianCount += examined;
      } else {
        int examined = examinLandmarkStackedJacobian(mapPoint, filter);
        featureJacobianLandmarkCount += examined;
      }
    }
    LOG(INFO) << "Examined " << featureJacobianLandmarkCount
              << " stacked landmark Jacobians, and "
              << measurementJacobianCount << " measurement Jacobians of "
              << numLandmarks << " landmarks";
  }

  /**
   * @brief examineCovariance Check if nav state covariance computed by ceres
   * solver and okvis are the same.
   * @warning Assume the OKVIS estimator is used.
   */
  void examineCovariance(int frameCount) {
    std::shared_ptr<okvis::Estimator> optimizer =
        std::dynamic_pointer_cast<okvis::Estimator>(estimator);
    Eigen::MatrixXd navStateCov;
    Eigen::MatrixXd navStateCovCeres;
    bool statusCeres = optimizer->computeCovarianceCeres(&navStateCovCeres, ::ceres::CovarianceAlgorithmType::DENSE_SVD);
    estimator->computeCovariance(&navStateCov);

    if (!statusCeres) {
      std::cerr << "ceres solver fails to compute covariance. Skip covariance comparison!\n";
      return;
    }
    Eigen::MatrixXd covDiff = navStateCovCeres - navStateCov;
    EXPECT_LT((covDiff.topLeftCorner<6, 6>()).lpNorm<Eigen::Infinity>(), 1e-5)
        << "covariances of pose by OKVIS and ceres solver not close enough.\n"
        << "Frame count: " << frameCount << ". okvis pose cov\n"
        << navStateCov.topLeftCorner<6, 6>() << "\nceres pose cov\n"
        << navStateCovCeres.topLeftCorner<6, 6>() << "\ndiff\n"
        << covDiff.topLeftCorner<6, 6>();
    EXPECT_LT((covDiff.topRightCorner<6, 9>()).lpNorm<Eigen::Infinity>(), 5e-5)
        << "covariances of pose and speed and biases by OKVIS and ceres solver "
           "not close enough.\n"
        << "Frame count: " << frameCount
        << ". okvis pose and speed and biases cov\n"
        << navStateCov.topRightCorner<6, 9>()
        << "\nceres pose and speed and biases cov\n"
        << navStateCovCeres.topRightCorner<6, 9>() << "\ndiff\n"
        << covDiff.topRightCorner<6, 9>();
    EXPECT_LT((covDiff.bottomRightCorner<9, 9>()).lpNorm<Eigen::Infinity>(), 1e-4)
        << "covariances of speed and biases by OKVIS and ceres solver not "
           "close enough.\n"
        << "Frame count: " << frameCount << ". okvis speed and biases cov\n"
        << navStateCov.bottomRightCorner<9, 9>()
        << "\nceres speed and biases cov\n"
        << navStateCovCeres.bottomRightCorner<9, 9>() << "\ndiff\n"
        << covDiff.bottomRightCorner<9, 9>();
  }

  ~EstimatorTest() {
  }

public:
  // input arguments
  std::string projOptModelName;
  std::string extrinsicModelName;
  std::string outputFile;
  double timeOffset;
  double readoutTime;
  int cameraObservationModelId = 0;
  int landmarkModelId = 0;
  okvis::EstimatorAlgorithm algorithm = okvis::EstimatorAlgorithm::MSCKF;

private:

  // internal default system parameters
  simul::VioTestSystemBuilder vioSystemBuilder;
  bool noisyInitialSpeedAndBiases = true;
  bool useEpipolarConstraint = false;
  double noise_factor = 1.0;
  simul::SimCameraModelType cameraModelId = simul::SimCameraModelType::EUROC;
  simul::CameraOrientation cameraOrientationId =
      simul::CameraOrientation::Forward;
  double landmarkRadius = 5;
  okvis::TestSetting testSetting;
  simul::SimulatedTrajectoryType trajectoryType =
      simul::SimulatedTrajectoryType::Torus;
  okvis::BackendParams backendParams;


  std::ofstream debugStream; // record state history of a trial
  std::shared_ptr<okvis::Estimator> estimator;
  std::vector<okvis::Time> times;
  okvis::ImuMeasurementDeque imuMeasurements;
  okvis::ImuMeasurementDeque trueBiases;
  std::vector<okvis::kinematics::Transformation> ref_T_WS_list;

  std::shared_ptr<okvis::SimulationFrontend> frontend;
  std::shared_ptr<const okvis::cameras::NCameraSystem> cameraSystem0;
  std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry0;
};

TEST(EstimatorTest, FeatureJacobianFixedModels) {
  std::string projOptModelName = "FIXED";
  std::string extrinsicModelName = "FIXED";
  std::string outputFile = FLAGS_log_dir + "/MSCKF_Torus_Fixed.txt";

  EstimatorTest test(projOptModelName, extrinsicModelName, outputFile, 0.0, 0.0);
  test.SetUp();
  test.Run(EstimatorCheck::FEATURE_TRACK_JACOBIAN);
}

TEST(EstimatorTest, FeatureJacobianVariableParams) {
  std::string projOptModelName = "FXY_CXY";
  std::string extrinsicModelName = "P_CB";
  std::string outputFile = FLAGS_log_dir + "/MSCKF_Torus.txt";
  EstimatorTest test(projOptModelName, extrinsicModelName, outputFile, 0.0, 0.0);
  test.SetUp();
  test.Run(EstimatorCheck::FEATURE_TRACK_JACOBIAN);
}

TEST(EstimatorTest, FeatureJacobianVariableTime) {
  // also inspect the Jacobians relative to velocity
  std::string projOptModelName = "FXY_CXY";
  std::string extrinsicModelName = "P_CB";
  std::string outputFile = FLAGS_log_dir + "/MSCKF_Torus_RS.txt";

  EstimatorTest test(projOptModelName, extrinsicModelName, outputFile, 0.0, 0.01);
  test.SetUp();
  test.Run(EstimatorCheck::FEATURE_TRACK_JACOBIAN);
}

TEST(EstimatorTest, FeatureJacobianSingleChordalDistance) {
  std::string projOptModelName = "FXY_CXY";
  std::string extrinsicModelName = "P_CB";
  std::string outputFile = FLAGS_log_dir + "/MSCKF_Torus_RS_Chordal.txt";
  EstimatorTest test(projOptModelName, extrinsicModelName, outputFile, 0.0, 0.0, 0, 0);
  test.SetUp();
  test.Run(EstimatorCheck::FEATURE_MEASUREMENT_JACOBIAN);
}

TEST(EstimatorTest, FeatureJacobianChordalDistance) {
  std::string projOptModelName = "FXY_CXY";
  std::string extrinsicModelName = "P_CB";
  std::string outputFile = FLAGS_log_dir + "/MSCKF_Torus_Chordal.txt";
  EstimatorTest test(projOptModelName, extrinsicModelName, outputFile, 0.0, 0.0, 2, 2);
  test.SetUp();
  test.Run(EstimatorCheck::FEATURE_MEASUREMENT_JACOBIAN);
}

TEST(EstimatorTest, OkvisCovariance) {
  std::string projOptModelName = "FIXED";
  std::string extrinsicModelName = "FIXED";
  std::string outputFile = FLAGS_log_dir + "/OKVIS_Torus_Fixed.txt";
  EstimatorTest test(projOptModelName, extrinsicModelName, outputFile, 0.0, 0.0);
  test.algorithm = okvis::EstimatorAlgorithm::OKVIS;
  test.SetUp();
  test.Run(EstimatorCheck::NAVSTATE_COVARIANCE);
}

