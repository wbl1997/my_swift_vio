#include "simul/VioSimTestSystem.hpp"

#include <gtsam/SlidingWindowSmoother.hpp>

#include <swift_vio/StatAccumulator.h>
#include <swift_vio/TFVIO.hpp>
#include <swift_vio/VioEvaluationCallback.hpp>
#include <swift_vio/VioFactoryMethods.hpp>

#include <simul/CameraSystemCreator.hpp>

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

okvis::ImuParameters VioSimTestSystem::createSensorSystem(const TestSetting &testSetting) {
  // The following parameters are in metric units.
  initCameraNoiseParams(testSetting.visionParams.sigma_abs_position,
                        testSetting.visionParams.sigma_abs_orientation,
                        testSetting.visionParams.fixCameraInternalParams,
                        &extrinsicsEstimationParameters_);

  if (testSetting.simDataDir.empty()) {
    simul::CameraSystemCreator csc(testSetting.visionParams.cameraModelId,
                                   testSetting.visionParams.cameraOrientationId,
                                   testSetting.visionParams.projOptModelName,
                                   testSetting.visionParams.extrinsicModelName,
                                   testSetting.visionParams.timeOffset,
                                   testSetting.visionParams.readoutTime);

    csc.createNominalCameraSystem(&refCameraSystem_);

    if (testSetting.visionParams.noisyInitialSensorParams) {
      initialCameraSystem_ = createNoisyCameraSystem(refCameraSystem_, extrinsicsEstimationParameters_);
    } else {
      initialCameraSystem_ = refCameraSystem_->deepCopy();
    }
  } else {
    std::string cameraImuYaml =
        testSetting.simDataDir + "/camchain_imucam.yaml";
    refCameraSystem_ = loadCameraSystemYaml(cameraImuYaml);
    if (testSetting.visionParams.noisyInitialSensorParams) {
      initialCameraSystem_ = createNoisyCameraSystem(
          refCameraSystem_, extrinsicsEstimationParameters_);
    } else {
      initialCameraSystem_ = refCameraSystem_->deepCopy();
    }
    initialCameraSystem_->setProjectionOptMode(0, testSetting.visionParams.projOptModelName);
    initialCameraSystem_->setExtrinsicOptMode(0, testSetting.visionParams.extrinsicModelName);
  }
  estimatedCameraSystem_ = initialCameraSystem_->deepCopy();

  okvis::ImuParameters imuParameters;
  if (testSetting.estimatorParams.estimator_algorithm <
      swift_vio::EstimatorAlgorithm::HybridFilter) {
    imuParameters.model_type = "BG_BA"; // use BG_BA for smoothers.
  } else {
    imuParameters.model_type = "BG_BA_TG_TS_TA";
  }
  simul::initImuNoiseParams(testSetting.imuParams, &imuParameters);
  return imuParameters;
}

void VioSimTestSystem::createEstimator(const TestSetting &testSetting) {
  initialNavState_.std_p_WS = Eigen::Vector3d(1e-5, 1e-5, 1e-5);
  initialNavState_.std_q_WS = Eigen::Vector3d(M_PI / 180, M_PI / 180, 1e-5);
  initialNavState_.std_v_WS = Eigen::Vector3d(5e-2, 5e-2, 5e-2);

  okvis::kinematics::Transformation T_WS;
  Eigen::Vector3d v_WS;
  simData_->navStateAtStart(&T_WS, &v_WS);
  Eigen::Vector3d p_WS = T_WS.r();

  if (testSetting.imuParams.noisyInitialSpeedAndBiases) {
    v_WS += vio::Sample::gaussian(1, 3).cwiseProduct(initialNavState_.std_v_WS);
  }

  initialNavState_.initWithExternalSource = true;
  initialNavState_.p_WS = p_WS;
  initialNavState_.q_WS = T_WS.q();
  initialNavState_.v_WS = v_WS;

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
  estimator_->setOptimizationOptions(optimOptions);

  swift_vio::PointLandmarkOptions plOptions;
  plOptions.minTrackLengthForSlam = 5;
  plOptions.landmarkModelId = testSetting.estimatorParams.landmarkModelId;
  estimator_->setPointLandmarkOptions(plOptions);

  estimator_->addImu(simData_->imuParameters());
  estimator_->addCameraSystem(*initialCameraSystem_);
  estimator_->addCameraParameterStds(extrinsicsEstimationParameters_);
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

  okvis::ImuParameters imuParameters = createSensorSystem(testSetting);

  if (testSetting.simDataDir.empty()) {
    simData_ = std::shared_ptr<SimDataInterface>(
        new CurveData(testSetting.imuParams.trajectoryId, imuParameters,
                      testSetting.visionParams.addImageNoise));
  } else {
    simData_ = std::shared_ptr<SimDataInterface>(new SimFromRealData(
        testSetting.simDataDir, imuParameters, testSetting.visionParams.addImageNoise));
    imuParameters = simData_->imuParameters();
  }

  LOG(INFO) << "sigma_g_c " << imuParameters.sigma_g_c << " sigma_gw_c " << imuParameters.sigma_gw_c
            << " sigma_a_c " << imuParameters.sigma_a_c << " sigma_aw_c " << imuParameters.sigma_aw_c;

  simData_->initializeLandmarkGrid(testSetting.visionParams.gridType,
                                   testSetting.visionParams.landmarkRadius);

  std::string pointFile = outputPath + "/" + testSetting.imuParams.trajLabel + "_Points.txt";
  simData_->saveLandmarkGrid(pointFile);

  std::string imuSampleFile = outputPath + "/" + testSetting.imuParams.trajLabel + "_IMU.txt";
  simData_->resetImuBiases(imuParameters, testSetting.imuParams, imuSampleFile);

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

    frontend_.reset(new SimulationFrontend(simData_->homogeneousPoints(),
                                           simData_->landmarkIds(),
                                           refCameraSystem_->numCameras(), 60));

    createEstimator(testSetting);

    std::ofstream debugStream;
    debugStream.open(outputFile, std::ofstream::out);
    headerLine = estimator_->headerLine();
    debugStream << headerLine << std::endl;

    bool hasStarted = false;
    int frameCount = 0;     // number of frames used in estimator
    int keyframeCount = 0;
    int trackedFeatures = 0; // feature tracks observed in a frame

    simData_->resetImuBiases(imuParameters, testSetting.imuParams, "");
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
          estimator_->addStates(mf, imuSegment, true);
          asKeyframe = true;
        } else {
          estimator_->addStates(mf, imuSegment, false);
        }
        ++frameCount;
        // add landmark observations
        trackedFeatures = 0;
        if (testSetting.visionParams.useImageObservs) {
          std::vector<std::vector<int>> keypointIndices;
          simData_->addFeaturesToNFrame(refCameraSystem_, mf, &keypointIndices);

          trackedFeatures = frontend_->dataAssociationAndInitialization(
              *estimator_, keypointIndices, mf, &asKeyframe);
          estimator_->setKeyframe(mf->id(), asKeyframe);
        }

        if (asKeyframe) {
          ++keyframeCount;
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

        neesAccumulator.push_back(refNFrameTime, normalizedSquaredError);
        rmseAccumulator.push_back(refNFrameTime, squaredError);
      } while (simData_->nextNFrame());

      neesAccumulator.accumulate();
      rmseAccumulator.accumulate();

      Eigen::VectorXd desiredStdevs;
      std::vector<std::string> dimensionLabels;
      estimator_->getDesiredStdevs(&desiredStdevs, &dimensionLabels);
      checkMseCallback_(rmseAccumulator.lastValue(), desiredStdevs, dimensionLabels);
      checkNeesCallback_(neesAccumulator.lastValue());
      std::stringstream messageStream;
      messageStream << "Run " << run << " finishes with #processed frames " << frameCount
                    << " #tracked features in last frame " << trackedFeatures
                    << " #keyframes " << keyframeCount;
      LOG(INFO) << messageStream.str();
      metaStream << messageStream.str() << std::endl;
    } catch (std::exception &e) {
      LOG(INFO) << "Run " << run << " aborts with #processed frames " << frameCount
                << " #keyframes " << keyframeCount << " and error: " << e.what();
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
