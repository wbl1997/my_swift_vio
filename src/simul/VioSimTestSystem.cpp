#include "simul/VioSimTestSystem.hpp"

#include <gtsam/SlidingWindowSmoother.hpp>

#include <swift_vio/StatAccumulator.h>
#include <swift_vio/TFVIO.hpp>
#include <swift_vio/VioEvaluationCallback.hpp>
#include <swift_vio/VioFactoryMethods.hpp>

#include <simul/CameraSystemCreator.hpp>
#include <simul/gflags.hpp>

namespace simul {
typedef boost::iterator_range<std::vector<std::pair<double, double>>::iterator>
    HistogramType;

void outputFeatureHistogram(const std::string &featureHistFile,
                            const HistogramType &hist) {
  std::ofstream featureHistStream(featureHistFile, std::ios_base::out);
  double total = 0.0;
  featureHistStream << "Histogram of number of features in images (bin "
                    << "lower bound, value)" << std::endl;
  for (size_t i = 0; i < hist.size(); i++) {
    featureHistStream << hist[i].first << " " << hist[i].second << std::endl;
    total += hist[i].second;
  }
  featureHistStream.close();
  if (std::fabs(total - 1.0) > 1e-4) {
    std::cerr << "Total of feature histogram densities " << total << " != 1.\n";
  }
}

Eigen::Matrix<double, 6, 1>
computeNormalizedErrors(const Eigen::VectorXd &errors,
                        const Eigen::MatrixXd &covariance) {
  Eigen::Matrix<double, 6, 1> normalizedSquaredError;
  Eigen::Vector3d deltaP = errors.head<3>();
  Eigen::Vector3d alpha = errors.segment<3>(3);

  normalizedSquaredError[0] =
      deltaP.transpose() * covariance.topLeftCorner<3, 3>().inverse() * deltaP;
  normalizedSquaredError[1] =
      alpha.transpose() * covariance.block<3, 3>(3, 3).inverse() * alpha;

  Eigen::Matrix<double, 6, 1> deltaPose = errors.head<6>();
  Eigen::Matrix<double, 6, 1> tempPoseError =
      covariance.topLeftCorner<6, 6>().ldlt().solve(deltaPose);
  normalizedSquaredError[2] = deltaPose.transpose() * tempPoseError;

  Eigen::Vector3d deltaV = errors.segment<3>(6);
  Eigen::Vector3d deltaBg = errors.segment<3>(9);
  Eigen::Vector3d deltaBa = errors.segment<3>(12);
  normalizedSquaredError[3] =
      deltaV.transpose() * covariance.block<3, 3>(6, 6).inverse() * deltaV;
  normalizedSquaredError[4] =
      deltaBg.transpose() * covariance.block<3, 3>(9, 9).inverse() * deltaBg;
  normalizedSquaredError[5] =
      deltaBa.transpose() * covariance.block<3, 3>(12, 12).inverse() * deltaBa;
  return normalizedSquaredError;
}

void VioSimTestSystem::createSensorSystem(const TestSetting &testSetting) {
  initCameraNoiseParams(testSetting.visionParams.sigma_abs_position,
                        testSetting.visionParams.sigma_abs_orientation,
                        &refCameraNoiseParameters_);
  // create initial camera system used by the estimator.
  if (testSetting.simDataDir.empty()) {
    simul::CameraSystemCreator csc(testSetting.visionParams.cameraModelId,
                                   testSetting.visionParams.cameraOrientationId,
                                   testSetting.visionParams.projOptModelName,
                                   testSetting.visionParams.extrinsicModelName,
                                   testSetting.visionParams.timeOffset,
                                   testSetting.visionParams.readoutTime);
    csc.createNominalCameraSystem(&refCameraSystem_);
  } else {
    std::string cameraImuYaml =
        testSetting.simDataDir + "/camchain_imucam.yaml";
    refCameraSystem_ = loadCameraSystemYaml(cameraImuYaml);
    refCameraSystem_->setProjectionOptMode(0, testSetting.visionParams.projOptModelName);
    refCameraSystem_->setExtrinsicOptMode(0, testSetting.visionParams.extrinsicModelName);
  }

  if (testSetting.estimatorParams.estimator_algorithm <
      swift_vio::EstimatorAlgorithm::HybridFilter) {
    refImuParameters_.model_type = "BG_BA"; // use BG_BA for smoothers.
  } else {
    refImuParameters_.model_type = "BG_BA_TG_TS_TA";
  }
  simul::initImuNoiseParams(testSetting.imuParams, &refImuParameters_);
}

void VioSimTestSystem::createEstimator(const TestSetting &testSetting) {
  simData_->navStateAtStart(&refNavState_);

  initialNavState_ = refNavState_;
  if (testSetting.imuParams.noisyInitialSpeedAndBiases) {
    initialNavState_.v_WS =
        refNavState_.v_WS +
        vio::Sample::gaussian(1, 3).cwiseProduct(refNavState_.std_v_WS);
    initialImuParameters_.a0 =
        refImuParameters_.a0 +
        vio::Sample::gaussian(refImuParameters_.sigma_ba, 3);
    initialImuParameters_.g0 =
        refImuParameters_.g0 +
        vio::Sample::gaussian(refImuParameters_.sigma_bg, 3);
  }

  if (testSetting.visionParams.noisyInitialSensorParams) {
    initialCameraSystem_ = createNoisyCameraSystem(
        refCameraSystem_, refCameraNoiseParameters_);
  } else {
    initialCameraSystem_ = refCameraSystem_->deepCopy();
  }

  initialCameraNoiseParameters_ = refCameraNoiseParameters_;
  if (testSetting.visionParams.fixCameraInternalParams) {
    initialCameraNoiseParameters_.sigma_focal_length = 0;
    initialCameraNoiseParameters_.sigma_principal_point = 0;
    for (size_t i = 0u;
         i < initialCameraNoiseParameters_.sigma_distortion.size(); ++i) {
      initialCameraNoiseParameters_.sigma_distortion[i] = 0;
    }
    initialCameraNoiseParameters_.sigma_td = 0;
    initialCameraNoiseParameters_.sigma_tr = 0;
  }

  initialImuParameters_ = refImuParameters_;
  if (testSetting.imuParams.noisyInitialSensorParams) {
    initialImuParameters_.Tg0 =
        refImuParameters_.Tg0 +
        vio::Sample::gaussian(refImuParameters_.sigma_TGElement, 9);
    initialImuParameters_.Ts0 =
        refImuParameters_.Ts0 +
        vio::Sample::gaussian(refImuParameters_.sigma_TSElement, 9);
    initialImuParameters_.Ta0 =
        refImuParameters_.Ta0 +
        vio::Sample::gaussian(refImuParameters_.sigma_TAElement, 9);
  }

  if (testSetting.imuParams.fixImuIntrinsicParams) {
    initialImuParameters_.sigma_TGElement = 0;
    initialImuParameters_.sigma_TSElement = 0;
    initialImuParameters_.sigma_TAElement = 0;
  }

  evaluationCallback_.reset(new swift_vio::VioEvaluationCallback());
  std::shared_ptr<okvis::ceres::Map> mapPtr(
      new okvis::ceres::Map(evaluationCallback_.get()));
  // std::shared_ptr<okvis::ceres::Map> mapPtr(new okvis::ceres::Map());

  estimator_ = swift_vio::createBackend(testSetting.estimatorParams.estimator_algorithm,
                                        testSetting.backendParams, mapPtr);

  okvis::Optimization optimOptions;
  optimOptions.useEpipolarConstraint = testSetting.estimatorParams.useEpipolarConstraint;
  optimOptions.cameraObservationModelId = testSetting.estimatorParams.cameraObservationModelId;
  optimOptions.computeOkvisNees = testSetting.estimatorParams.computeOkvisNees;
  optimOptions.numKeyframes = 5;
  optimOptions.numImuFrames = 3;
  if (testSetting.simDataDir.empty()) {
    optimOptions.useMahalanobisGating = FLAGS_useMahalanobis;
  } else {
    optimOptions.useMahalanobisGating = false;
  }
  estimator_->setOptimizationOptions(optimOptions);

  swift_vio::PointLandmarkOptions plOptions;
  plOptions.minTrackLengthForSlam = FLAGS_minTrackLengthForSlam;
  plOptions.maxHibernationFrames = FLAGS_maxHibernationFrames;
  plOptions.maxInStateLandmarks = 80;
  plOptions.landmarkModelId = testSetting.estimatorParams.landmarkModelId;
  estimator_->setPointLandmarkOptions(plOptions);

  estimator_->addImu(initialImuParameters_);
  estimator_->addCameraSystem(*initialCameraSystem_);
  estimator_->addCameraParameterStds(initialCameraNoiseParameters_);

  estimatedCameraSystem_ = initialCameraSystem_->deepCopy();
}

void VioSimTestSystem::run(const simul::TestSetting &testSetting,
                           const std::string &outputPath) {
  swift_vio::StatAccumulator neesAccumulator;
  swift_vio::StatAccumulator rmseAccumulator;

  // number of features tracked in a frame.
  boost::accumulators::accumulator_set<
      double, boost::accumulators::features<boost::accumulators::tag::count,
                                            boost::accumulators::tag::density>>
      frameFeatureTally(boost::accumulators::tag::density::num_bins = 20,
                        boost::accumulators::tag::density::cache_size = 40);
  std::string featureHistFile = outputPath + "/FeatureHist.txt";

  okvis::timing::Timer filterTimer("msckf timer", true);

  std::string testIdentifier = testSetting.estimatorParams.estimatorLabel + "_" + testSetting.imuParams.trajLabel;
  LOG(INFO) << "Estimator algorithm: " << testSetting.estimatorParams.estimatorLabel << " trajectory "
            << testSetting.imuParams.trajLabel;
  std::string pathEstimatorTrajectory = outputPath + "/" + testIdentifier;
  std::string neesFile = pathEstimatorTrajectory + "_NEES.txt";
  std::string rmseFile = pathEstimatorTrajectory + "_RMSE.txt";
  std::string metadataFile = pathEstimatorTrajectory + "_metadata.txt";
  std::string headerLine;
  std::ofstream metaStream;
  metaStream.open(metadataFile, std::ofstream::out);

  bool verbose = false;

  createSensorSystem(testSetting);

  if (testSetting.simDataDir.empty()) {
    simData_ = std::shared_ptr<SimDataInterface>(
        new CurveData(testSetting.imuParams.trajectoryId, refImuParameters_,
                      testSetting.visionParams.addImageNoise));
  } else {
    simData_ = std::shared_ptr<SimDataInterface>(new SimFromRealData(
        testSetting.simDataDir, refImuParameters_, testSetting.visionParams.addImageNoise));
    refImuParameters_ = simData_->imuParameters();
  }

  LOG(INFO) << "IMU rate " << refImuParameters_.rate << " sigma_g_c " << refImuParameters_.sigma_g_c
            << " sigma_gw_c " << refImuParameters_.sigma_gw_c << " sigma_a_c "
            << refImuParameters_.sigma_a_c << " sigma_aw_c " << refImuParameters_.sigma_aw_c;

  simData_->initializeLandmarkGrid(testSetting.visionParams.gridType,
                                   testSetting.visionParams.landmarkRadius);

  std::string pointFile = outputPath + "/" + testSetting.imuParams.trajLabel + "_Points.txt";
  simData_->saveLandmarkGrid(pointFile);

  std::string imuSampleFile = outputPath + "/" + testSetting.imuParams.trajLabel + "_IMU.txt";
  simData_->resetImuBiases(refImuParameters_, testSetting.imuParams, imuSampleFile);

  std::string truthFile = outputPath + "/" + testSetting.imuParams.trajLabel + ".txt";
  simData_->saveRefMotion(truthFile);

  std::string cameraFile = outputPath + "/cameraSystem.txt";
  saveCameraParameters(refCameraSystem_, cameraFile);

  for (int run = 0; run < testSetting.estimatorParams.numRuns; ++run) {
    verbose = neesAccumulator.succeededRuns() == 0;
    filterTimer.start();

    srand((unsigned int)time(0)); // comment out to make tests deterministic.

    LOG(INFO) << "Run " << run << " " << testIdentifier << " "
              << testSetting.toString();

    std::stringstream ss;
    ss << run;
    std::string outputFile = pathEstimatorTrajectory + "_" + ss.str() + ".txt";

    SimFrontendOptions frontendOptions(60, FLAGS_maxMatchKeyframes);
    frontend_.reset(new SimulationFrontend(simData_->homogeneousPoints(),
                                           simData_->landmarkIds(),
                                           refCameraSystem_->numCameras(), frontendOptions));

    createEstimator(testSetting);

    std::ofstream debugStream;
    debugStream.open(outputFile, std::ofstream::out);
    headerLine = estimator_->headerLine();
    debugStream << headerLine << std::endl;

    bool hasStarted = false;
    int frameCount = 0;     // number of frames used in estimator
    int trackedFeatures = 0; // feature tracks observed in a frame
    bool runSuccessful = true;

    simData_->resetImuBiases(refImuParameters_, testSetting.imuParams, "");
    simData_->rewind();

    int expectedNumFrames = simData_->expectedNumNFrames();
    neesAccumulator.refreshBuffer(expectedNumFrames);
    rmseAccumulator.refreshBuffer(expectedNumFrames);
    try {
      do {
        okvis::Time refNFrameTime = simData_->currentTime();
        okvis::kinematics::Transformation T_WS_ref = simData_->currentPose();
        Eigen::Vector3d v_WS_ref = simData_->currentVelocity();
        okvis::ImuSensorReadings refBiases = simData_->currentBiases();
        okvis::ImuMeasurementDeque imuSegment =
            simData_->imuMeasurementsSinceLastNFrame();

        // assemble a multi-frame
        std::shared_ptr<okvis::MultiFrame> mf(new okvis::MultiFrame);
        uint64_t id = okvis::IdProvider::instance().newId();
        mf->setId(id);
        okvis::Time frameStamp = refNFrameTime - okvis::Duration(testSetting.visionParams.timeOffset);
        mf->setTimestamp(frameStamp);
        estimator_->getEstimatedCameraSystem(estimatedCameraSystem_);
        mf->resetCameraSystemAndFrames(*estimatedCameraSystem_);
        mf->setTimestamp(0u, frameStamp);

        VLOG(1) << "Processing frame " << id << " of index " << frameCount;

        bool asKeyframe = false;
        if (!hasStarted) {
          hasStarted = true;
          estimator_->setInitialNavState(initialNavState_);
          asKeyframe = true;
          estimator_->addStates(mf, imuSegment, asKeyframe);
        } else {
          asKeyframe = false;
          estimator_->addStates(mf, imuSegment, asKeyframe);
        }
        ++frameCount;
        if (FLAGS_allKeyframe) {
          asKeyframe = true;
        }

        // add landmark observations
        trackedFeatures = 0;
        if (testSetting.visionParams.useImageObservs) {
          std::vector<std::vector<int>> keypointIndices;
          simData_->addFeaturesToNFrame(refCameraSystem_, mf, &keypointIndices);

          trackedFeatures = frontend_->dataAssociationAndInitialization(
              *estimator_, keypointIndices, mf, &asKeyframe);
          estimator_->setKeyframe(mf->id(), asKeyframe);
        }

        frameFeatureTally(trackedFeatures);

        size_t maxIterations = 10u;
        size_t numThreads = 2u;
        estimator_->optimize(maxIterations, numThreads, false);

        okvis::MapPointVector removedLandmarks;
        estimator_->applyMarginalizationStrategy(removedLandmarks);
        estimator_->printStatesAndStdevs(debugStream);
        debugStream << std::endl;

        Eigen::MatrixXd covariance;
        estimator_->computeCovariance(&covariance);
        Eigen::VectorXd errors;
        estimator_->computeErrors(T_WS_ref, v_WS_ref, refBiases, refCameraSystem_,
                                  &errors);
        Eigen::VectorXd squaredError = errors.cwiseAbs2();
        Eigen::VectorXd normalizedSquaredError =
            computeNormalizedErrors(errors, covariance);

        if (errors.head<3>().lpNorm<Eigen::Infinity>() > FLAGS_maxPositionRmse) {
          runSuccessful = false;
        }

        neesAccumulator.push_back(refNFrameTime, normalizedSquaredError);
        rmseAccumulator.push_back(refNFrameTime, squaredError);
      } while (simData_->nextNFrame());

      Eigen::VectorXd desiredStdevs;
      std::vector<std::string> dimensionLabels;
      estimator_->getDesiredStdevs(&desiredStdevs, &dimensionLabels);
      checkMseCallback_(rmseAccumulator.lastValue(), desiredStdevs, dimensionLabels);
      checkNeesCallback_(neesAccumulator.lastValue());

      if (runSuccessful) {
        neesAccumulator.accumulate();
        rmseAccumulator.accumulate();
      }

      std::stringstream messageStream;
      messageStream << "Run " << run << " finishes with #processed frames " << frameCount
                    << " #tracked features in last frame " << trackedFeatures
                    << " #keyframes " << frontend_->numKeyframes() << ". Successful? " << runSuccessful;
      LOG(INFO) << messageStream.str();
      metaStream << messageStream.str() << std::endl;

      // output track length distribution
      std::string trackStatFile =
          pathEstimatorTrajectory + "_trackstat_" + ss.str() + ".txt";
      std::ofstream trackStatStream(trackStatFile, std::ios_base::out);
      estimator_->printTrackLengthHistogram(trackStatStream);
      trackStatStream.close();
    } catch (std::exception &e) {
      std::stringstream messageStream;
      messageStream << "Run " << run << " aborts with #processed frames " << frameCount
                    << " #tracked features in last frame " << trackedFeatures
                    << " #keyframes " << frontend_->numKeyframes() << " and error: " << e.what();
      LOG(INFO) << messageStream.str();
      metaStream << messageStream.str() << std::endl;
      if (debugStream.is_open()) {
        debugStream.close();
      }
    }
    double elapsedTime = filterTimer.stop();
    std::stringstream sstream;
    sstream << "Run " << run << " used " << elapsedTime << " seconds.";
    LOG(INFO) << sstream.str();
    metaStream << sstream.str() << std::endl;
  }  // next run

  HistogramType hist = boost::accumulators::density(frameFeatureTally);
  outputFeatureHistogram(featureHistFile, hist);

  int numSucceededRuns = neesAccumulator.succeededRuns();
  std::stringstream message;
  message << "#successful runs " << numSucceededRuns << " out of "
          << testSetting.estimatorParams.numRuns << " runs.";

  std::string neesHeaderLine =
      "%state timestamp, NEES of p_WS, \\alpha_WS, T_WS, v_WS, b_g, b_a";
  neesAccumulator.computeMean();
  neesAccumulator.dump(neesFile, neesHeaderLine);

  rmseAccumulator.computeRootMean();
  rmseAccumulator.dump(rmseFile, headerLine);

  LOG(INFO) << message.str();
  metaStream << message.str() << std::endl;
  metaStream.close();
}
} // namespace simul
