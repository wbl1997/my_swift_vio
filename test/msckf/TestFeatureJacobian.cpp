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
}

TEST(MSCKF, FeatureJacobian) {
  simul::VioTestSystemBuilder vioSystemBuilder;
  bool addPriorNoise = true;
  int32_t estimatorType = 4;
  okvis::TestSetting testSetting(true, addPriorNoise, false, true, true, 0.5, 0.5, estimatorType);
  int trajectoryId = 0; // Torus
  std::string projOptModelName = "FXY_CXY";
  std::string extrinsicModelName = "P_CB";
  int projOptModelId = okvis::ProjectionOptNameToId(projOptModelName);
  int extrinsicModelId = okvis::ExtrinsicModelNameToId(extrinsicModelName);

  int cameraOrientation = 0;
  std::shared_ptr<std::ofstream> inertialStream;
  vioSystemBuilder.createVioSystem(testSetting, trajectoryId,
                                   projOptModelName, extrinsicModelName,
                                   cameraOrientation, inertialStream,
                                   "");

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
  std::shared_ptr<okvis::MSCKF2> estimator =
      std::dynamic_pointer_cast<okvis::MSCKF2>(genericEstimator);
  std::shared_ptr<okvis::SimulationFrontend> frontend = vioSystemBuilder.mutableFrontend();
  std::shared_ptr<const okvis::cameras::NCameraSystem> cameraSystem0 =
      vioSystemBuilder.trueCameraSystem();
  std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry0 =
      cameraSystem0->cameraGeometry(0);

  std::string outputFile = FLAGS_log_dir + "/MSCKF_Torus.txt";
  std::ofstream debugStream;  // record state history of a trial
  if (!debugStream.is_open()) {
    debugStream.open(outputFile, std::ofstream::out);
    std::string headerLine;
    okvis::StreamHelper::composeHeaderLine(
          vioSystemBuilder.imuModelType(),
          projOptModelName,
          extrinsicModelName,
          vioSystemBuilder.distortionType(),
          okvis::FULL_STATE_WITH_ALL_CALIBRATION,
          &headerLine);
    debugStream << headerLine << std::endl;
  }

  Eigen::VectorXd navError(9);

  for (auto iter = times.begin(), iterEnd = times.end(); iter != iterEnd;
       iter += cameraIntervalRatio, kale += cameraIntervalRatio,
       trueBiasIter += cameraIntervalRatio) {
    okvis::kinematics::Transformation T_WS(ref_T_WS_list[kale]);
    // assemble a multi-frame
    std::shared_ptr<okvis::MultiFrame> mf(new okvis::MultiFrame);
    uint64_t id = okvis::IdProvider::instance().newId();
    mf->setId(id);

    mf->setTimestamp(*iter);
    // The reference cameraSystem will be used for triangulating landmarks in
    // the frontend which provides observations to the estimator.
    mf->resetCameraSystemAndFrames(*cameraSystem0);

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
      estimator->resetInitialNavState(vioSystemBuilder.initialNavState());
      estimator->addStates(mf, imuSegment, asKeyframe);
        const int kDistortionCoeffDim =
            okvis::cameras::RadialTangentialDistortion::NumDistortionIntrinsics;
        const int kNavImuCamParamDim =
            okvis::ceres::ode::NavErrorStateDim +
            okvis::ImuModelGetMinimalDim(
                okvis::ImuModelNameToId(vioSystemBuilder.imuModelType())) +
            2 + kDistortionCoeffDim +
            okvis::ExtrinsicModelGetMinimalDim(extrinsicModelId) +
            okvis::ProjectionOptGetMinimalDim(projOptModelId);
        ASSERT_EQ(estimator->getEstimatedVariableMinimalDim(),
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
          *estimator, T_WS, cameraSystem0, mf, &asKeyframe);
      estimator->setKeyframe(mf->id(), asKeyframe);
    }

    if (frameCount == 100) {
      okvis::PointMap landmarkMap;
      size_t numLandmarks = estimator->getLandmarks(landmarkMap);
      int mpCount = 0;
      for (auto mpIter = landmarkMap.begin(); mpIter != landmarkMap.end();
           ++mpIter) {
        uint64_t latestFrameId = estimator->currentFrameId();
        const okvis::MapPoint& mapPoint = mpIter->second;
        auto obsIter = std::find_if(mapPoint.observations.begin(), mapPoint.observations.end(),
                                    okvis::IsObservedInFrame(latestFrameId));
        if (obsIter != mapPoint.observations.end()) {
          continue;
        }
        Eigen::MatrixXd H_oi[2];
        Eigen::Matrix<double, Eigen::Dynamic, 1> r_oi[2];
        Eigen::MatrixXd R_oi[2];
        estimator->featureJacobian(mpIter->second, H_oi[0], r_oi[0], R_oi[0], nullptr);
//        int landmarkModelId = 0;
//        int observationModelId = 0;
//        estimator->featureJacobian(mpIter->second, H_oi[1], r_oi[1], R_oi[1], landmarkModelId, observationModelId, nullptr);
//        EXPECT_LT((H_oi[1] - H_oi[0]).lpNorm<Eigen::Infinity>(), 1e-6) << "H_oi";
//        EXPECT_LT((r_oi[1] - r_oi[0]).lpNorm<Eigen::Infinity>(), 1e-6) << "r_oi";
//        EXPECT_LT((R_oi[1] - R_oi[0]).lpNorm<Eigen::Infinity>(), 1e-6) << "R_oi";
        ++mpCount;
      }
      std::cout << "examined " << mpCount << " out of " << numLandmarks << " landmarks\n";
    }

    size_t maxIterations = 10u;
    size_t numThreads = 2u;
    estimator->optimize(maxIterations, numThreads, false);
    okvis::Optimization sharedOptConfig;
    estimator->setKeyframeRedundancyThresholds(
        sharedOptConfig.translationThreshold,
        sharedOptConfig.rotationThreshold,
        sharedOptConfig.trackingRateThreshold,
        sharedOptConfig.minTrackLength);
    okvis::MapPointVector removedLandmarks;
    size_t numKeyFrames = 5u;
    size_t numImuFrames = 20u;

    estimator->applyMarginalizationStrategy(numKeyFrames, numImuFrames, removedLandmarks);
    estimator->print(debugStream);
    debugStream << std::endl;

    Eigen::Vector3d v_WS_true = vioSystemBuilder.sinusoidalTrajectory()
                                    ->computeGlobalLinearVelocity(*iter);

    computeNavErrors(estimator.get(), T_WS, v_WS_true, &navError);

    lastKFTime = currentKFTime;
  }  // every keyframe

  LOG(INFO) << "Finishes with last added frame " << frameCount
            << " of tracked features " << trackedFeatures << std::endl;
  EXPECT_LT(navError.head<3>().lpNorm<Eigen::Infinity>(), 0.3)
      << "Final position error";
  EXPECT_LT(navError.segment<3>(3).lpNorm<Eigen::Infinity>(), 0.08)
      << "Final orientation error";
  EXPECT_LT(navError.tail<3>().lpNorm<Eigen::Infinity>(), 0.1)
      << "Final velocity error";
}
