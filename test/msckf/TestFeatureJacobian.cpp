#include <gtest/gtest.h>

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

void testPointLandmarkJacobian(std::string projOptModelName,
                               std::string extrinsicModelName,
                               std::string outputFile,
                               double timeOffset,
                               double readoutTime,
                               int cameraObservationModelId = 0,
                               int landmarkModelId = 0,
                               bool examineMeasurementJacobian = false) {
  simul::VioTestSystemBuilder vioSystemBuilder;
  bool noisyInitialSpeedAndBiases = true;
  bool useEpipolarConstraint = false;
  double noise_factor = 1.0;
  simul::SimCameraModelType cameraModelId = simul::SimCameraModelType::EUROC;
  simul::CameraOrientation cameraOrientationId = simul::CameraOrientation::Forward;
  double landmarkRadius = 5;
  okvis::TestSetting testSetting(true, noisyInitialSpeedAndBiases, false, true, true, noise_factor,
                                 noise_factor, okvis::EstimatorAlgorithm::MSCKF,
                                 useEpipolarConstraint,
                                 cameraObservationModelId, landmarkModelId,
                                 cameraModelId, cameraOrientationId,
                                 okvis::LandmarkGridType::FourWalls, landmarkRadius);
  simul::SimulatedTrajectoryType trajectoryType = simul::SimulatedTrajectoryType::Torus;


  vioSystemBuilder.createVioSystem(testSetting, trajectoryType,
                                   projOptModelName, extrinsicModelName,
                                   timeOffset, readoutTime,
                                   "", "");
  std::vector<uint64_t> multiFrameIds;
  size_t kale = 0;  // imu data counter
  bool bStarted = false;
  int frameCount = -1;               // number of frames used in estimator
  int trackedFeatures = 0;  // feature tracks observed in a frame
  const int cameraIntervalRatio = 10; // number imu meas for 1 camera frame

  std::vector<okvis::Time> times = vioSystemBuilder.sampleTimes();
  okvis::Time lastKFTime = times.front();
  okvis::ImuMeasurementDeque trueBiases = vioSystemBuilder.trueBiases();
  okvis::ImuMeasurementDeque::const_iterator trueBiasIter =
      trueBiases.begin();
  std::vector<okvis::kinematics::Transformation> ref_T_WS_list =
      vioSystemBuilder.ref_T_WS_list();
  okvis::ImuMeasurementDeque imuMeasurements = vioSystemBuilder.imuMeasurements();
  std::shared_ptr<okvis::Estimator> genericEstimator = vioSystemBuilder.mutableEstimator();
  std::shared_ptr<okvis::MSCKF2> estimator = std::dynamic_pointer_cast<okvis::MSCKF2>(genericEstimator);

  std::shared_ptr<okvis::SimulationFrontend> frontend = vioSystemBuilder.mutableFrontend();
  std::shared_ptr<const okvis::cameras::NCameraSystem> cameraSystem0 =
      vioSystemBuilder.trueCameraSystem();
  std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry0 =
      cameraSystem0->cameraGeometry(0);

  std::ofstream debugStream;  // record state history of a trial
  if (!debugStream.is_open()) {
    debugStream.open(outputFile, std::ofstream::out);
    std::string headerLine;
    okvis::StreamHelper::composeHeaderLine(
          vioSystemBuilder.imuModelType(),
          {extrinsicModelName},
          {projOptModelName},
          {vioSystemBuilder.distortionType()},
          okvis::FULL_STATE_WITH_ALL_CALIBRATION,
          &headerLine);
    debugStream << headerLine << std::endl;
  }

  Eigen::VectorXd navError(9);

  const int kNavImuCamParamDim =
      okvis::ceres::ode::kNavErrorStateDim +
      okvis::ImuModelGetMinimalDim(
          okvis::ImuModelNameToId(vioSystemBuilder.imuModelType())) +
      estimator->cameraParamsMinimalDimen();

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

      ASSERT_EQ(
          estimator->getEstimatedVariableMinimalDim(),
          kNavImuCamParamDim + okvis::HybridFilter::kClonedStateMinimalDimen)
          << "Initial cov with one cloned state has a wrong dim";
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

    if (frameCount == 100) {
      {
        okvis::StatePointerAndEstimateList currentStates;
        estimator->cloneFilterStates(&currentStates);
        Eigen::VectorXd deltaX;
        estimator->boxminusFromInput(currentStates, &deltaX);
        EXPECT_LT(deltaX.lpNorm<Eigen::Infinity>(), 1e-8);
      }
      okvis::PointMap landmarkMap;
      size_t numLandmarks = estimator->getLandmarks(landmarkMap);
      int featureJacobianLandmarkCount = 0;
      int measurementJacobianCount = 0;
      for (auto mpIter = landmarkMap.begin(); mpIter != landmarkMap.end(); ++mpIter) {
        uint64_t latestFrameId = estimator->currentFrameId();
        const okvis::MapPoint& mapPoint = mpIter->second;
        auto obsIter = std::find_if(mapPoint.observations.begin(), mapPoint.observations.end(),
                                    okvis::IsObservedInNFrame(latestFrameId));
        if (obsIter != mapPoint.observations.end()) {
          continue; // only examine observations of landmarks disappeared in current frame.
        }
        if (examineMeasurementJacobian) {
          int examined = examineLandmarkMeasurementJacobian(mapPoint, estimator);
          measurementJacobianCount += examined;
        } else {
          int examined = examinLandmarkStackedJacobian(mapPoint, estimator);
          featureJacobianLandmarkCount += examined;
        }
      }
      LOG(INFO) << "Examined " << featureJacobianLandmarkCount
                << " stacked landmark Jacobians, and "
                << measurementJacobianCount << " measurement Jacobians of "
                << numLandmarks << " landmarks";
    }

    size_t maxIterations = 10u;
    size_t numThreads = 2u;
    estimator->optimize(maxIterations, numThreads, false);
    okvis::Optimization sharedOptConfig;
    size_t numKeyFrames = 5u;
    size_t numImuFrames = 3u;
    estimator->setKeyframeRedundancyThresholds(
        sharedOptConfig.translationThreshold,
        sharedOptConfig.rotationThreshold,
        sharedOptConfig.trackingRateThreshold,
        sharedOptConfig.minTrackLength,
        numKeyFrames,
        numImuFrames);
    okvis::MapPointVector removedLandmarks;

    estimator->applyMarginalizationStrategy(numKeyFrames, numImuFrames, removedLandmarks);
    estimator->print(debugStream);
    debugStream << std::endl;

    // TODO(jhuai): Check loop query keyframe message for msckf has proper
    // covariance matrices.
    // estimator->getLoopQueryKeyframeMessage();
    // check number of landmarks, landmark positions are in camera frame,
    // the constraint list, poses, etc.

    Eigen::Vector3d v_WS_true = vioSystemBuilder.sinusoidalTrajectory()
                                    ->computeGlobalLinearVelocity(*iter);

    computeNavErrors(estimator.get(), T_WS, v_WS_true, &navError);

    lastKFTime = currentKFTime;
  }  // every keyframe

  LOG(INFO) << "Finishes with last added frame " << frameCount
            << " of tracked features " << trackedFeatures;
  EXPECT_LT(navError.head<3>().lpNorm<Eigen::Infinity>(), 0.3)
      << "Final position error";
  EXPECT_LT(navError.segment<3>(3).lpNorm<Eigen::Infinity>(), 0.08)
      << "Final orientation error";
  EXPECT_LT(navError.tail<3>().lpNorm<Eigen::Infinity>(), 0.1)
      << "Final velocity error";
}

TEST(MSCKF, FeatureJacobianFixedModels) {
  std::string projOptModelName = "FIXED";
  std::string extrinsicModelName = "FIXED";
  std::string outputFile = FLAGS_log_dir + "/MSCKF_Torus_Fixed.txt";
  testPointLandmarkJacobian(projOptModelName, extrinsicModelName, outputFile, 0.0, 0.0);
}

TEST(MSCKF, FeatureJacobianVariableParams) {
  std::string projOptModelName = "FXY_CXY";
  std::string extrinsicModelName = "P_CB";
  std::string outputFile = FLAGS_log_dir + "/MSCKF_Torus.txt";
  testPointLandmarkJacobian(projOptModelName, extrinsicModelName, outputFile, 0.0, 0.0);
}

TEST(MSCKF, FeatureJacobianVariableTime) {
  // also inspect the Jacobians relative to velocity
  std::string projOptModelName = "FXY_CXY";
  std::string extrinsicModelName = "P_CB";
  std::string outputFile = FLAGS_log_dir + "/MSCKF_Torus_RS.txt";
  testPointLandmarkJacobian(projOptModelName, extrinsicModelName, outputFile, 0.0, 0.01);
}

TEST(MSCKF, FeatureJacobianSingleChordalDistance) {
  std::string projOptModelName = "FXY_CXY";
  std::string extrinsicModelName = "P_CB";
  std::string outputFile = FLAGS_log_dir + "/MSCKF_Torus_RS_Chordal.txt";
  testPointLandmarkJacobian(projOptModelName, extrinsicModelName, outputFile, 0.0, 0.0, 0, 0, true);
}

TEST(MSCKF, FeatureJacobianChordalDistance) {
  std::string projOptModelName = "FXY_CXY";
  std::string extrinsicModelName = "P_CB";
  std::string outputFile = FLAGS_log_dir + "/MSCKF_Torus_Chordal.txt";
  testPointLandmarkJacobian(projOptModelName, extrinsicModelName, outputFile, 0.0, 0.0, 2, 2, true);
}
