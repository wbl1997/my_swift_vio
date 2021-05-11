#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/density.hpp>
#include <boost/accumulators/statistics/stats.hpp>

#include <gtest/gtest.h>


#include <vio/eigen_utils.h>
#include <vio/Sample.h>

#include <gtsam/VioBackEndParams.h>

#include <io_wrap/StreamHelper.hpp>

#include <okvis/IdProvider.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/RelativePoseError.hpp>
#include <okvis/ceres/ReprojectionError.hpp>
#include <okvis/ceres/SpeedAndBiasError.hpp>
#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>

#include <swift_vio/ceres/CameraTimeParamBlock.hpp>
#include <swift_vio/ceres/EuclideanParamBlock.hpp>
#include <swift_vio/PointLandmarkModels.hpp>
#include <swift_vio/ProjParamOptModels.hpp>
#include <swift_vio/StatAccumulator.h>

#include <simul/VioTestSystemBuilder.hpp>

DEFINE_bool(
    noisyInitialSpeedAndBiases, true,
    "add noise to the initial value of velocity, gyro bias, accelerometer "
    "bias which is used to initialize an estimator.");

DEFINE_bool(noisyInitialSensorParams, false,
            "add noise to the initial value of sensor parameters, including "
            "camera extrinsic, intrinsic "
            "and temporal parameters, and IMU parameters except for biases "
            "which is used to initialize an estimator. But the noise may be "
            "zero by setting e.g. zero_imu_intrinsic_param_noise");

DEFINE_int32(num_runs, 5, "How many times to run one simulation?");

DECLARE_bool(use_mahalanobis);

DEFINE_double(
    sim_camera_time_offset_sec, 0.0,
    "image raw timestamp + camera time offset = image time in imu clock");

DEFINE_double(
    sim_frame_readout_time_sec, 0.0,
    "readout time for one frame in secs");

DEFINE_double(sim_imu_noise_factor, 1.0,
              "weaken the IMU noise added to IMU readings by this factor");

DEFINE_double(sim_imu_bias_noise_factor, 1.0,
              "weaken the IMU BIAS noise added to IMU readings by this factor");

DEFINE_string(sim_trajectory_label, "WavyCircle",
              "Ball has the most exciting motion, wavycircle is general");

typedef boost::iterator_range<std::vector<std::pair<double, double>>::iterator>
    HistogramType;

void outputFeatureHistogram(const std::string& featureHistFile,
                            const HistogramType& hist) {
  std::ofstream featureHistStream(featureHistFile, std::ios_base::out);
  double total = 0.0;
  featureHistStream << "Histogram of number of features in images (bin "
              << "lower bound, value)" << std::endl;
  for (size_t i = 0; i < hist.size(); i++) {
    featureHistStream << hist[i].first << " " << hist[i].second << std::endl;
    total += hist[i].second;
  }
  featureHistStream.close();
  EXPECT_NEAR(total, 1.0, 1e-5)
      << "Total of densities: " << total << " should be 1.";
}

inline bool isFilteringMethod(swift_vio::EstimatorAlgorithm algorithmId) {
  return algorithmId >= swift_vio::EstimatorAlgorithm::MSCKF;
}

/**
 * @brief computeErrors
 * @param estimator
 * @param T_WS
 * @param v_WS_true
 * @param refMeasurement
 * @param refCameraGeometry
 * @param normalizedSquaredError normalized squared error in position, orientation, and pose
 * @param squaredError squared error in xyz, \alpha, v_WS, bg, ba, Tg, Ts, Ta, p_CB,
 *  (fx, fy), (cx, cy), k1, k2, p1, p2, td, tr
 */
void computeErrors(
    const okvis::Estimator* estimator,
    swift_vio::EstimatorAlgorithm estimatorAlgorithm,
    const okvis::kinematics::Transformation& T_WS,
    const Eigen::Vector3d& v_WS_true,
    const okvis::ImuSensorReadings& refMeasurement,
    std::shared_ptr<const okvis::cameras::CameraBase> refCameraGeometry,
    const int projOptModelId, Eigen::VectorXd* normalizedSquaredError,
    Eigen::VectorXd* squaredError) {
  okvis::kinematics::Transformation T_WS_est;
  uint64_t currFrameId = estimator->currentFrameId();
  estimator->get_T_WS(currFrameId, T_WS_est);
  Eigen::Vector3d delta = T_WS.r() - T_WS_est.r();
  Eigen::Vector3d alpha = vio::unskew3d(T_WS.C() * T_WS_est.C().transpose() -
                                        Eigen::Matrix3d::Identity());
  Eigen::Matrix<double, 6, 1> deltaPose;
  deltaPose << delta, alpha;

  okvis::SpeedAndBias speedAndBiasEstimate;
  estimator->getSpeedAndBias(currFrameId, 0, speedAndBiasEstimate);
  Eigen::Vector3d deltaV = speedAndBiasEstimate.head<3>() - v_WS_true;
  Eigen::Vector3d deltaBg = speedAndBiasEstimate.segment<3>(3) - refMeasurement.gyroscopes;
  Eigen::Vector3d deltaBa = speedAndBiasEstimate.tail<3>() - refMeasurement.accelerometers;

  Eigen::MatrixXd covariance;
  estimator->computeCovariance(&covariance);

  normalizedSquaredError->resize(3 + 3);
  (*normalizedSquaredError)[0] =
      delta.transpose() * covariance.topLeftCorner<3, 3>().inverse() * delta;
  (*normalizedSquaredError)[1] =
      alpha.transpose() * covariance.block<3, 3>(3, 3).inverse() * alpha;
  Eigen::Matrix<double, 6, 1> tempPoseError =
      covariance.topLeftCorner<6, 6>().ldlt().solve(deltaPose);
  (*normalizedSquaredError)[2] = deltaPose.transpose() * tempPoseError;

  (*normalizedSquaredError)[3] =
      deltaV.transpose() * covariance.block<3, 3>(6, 6).inverse() * deltaV;
  (*normalizedSquaredError)[4] =
      deltaBg.transpose() * covariance.block<3, 3>(9, 9).inverse() * deltaBg;
  (*normalizedSquaredError)[5] =
      deltaBa.transpose() * covariance.block<3, 3>(12, 12).inverse() * deltaBa;

  Eigen::Matrix<double, 9, 1> eye;
  eye << 1, 0, 0, 0, 1, 0, 0, 0, 1;
  int projOptModelDim = swift_vio::ProjectionOptGetMinimalDim(projOptModelId);
  squaredError->resize(51 + projOptModelDim);
  int index = 0;
  squaredError->head<3>() = delta.cwiseAbs2();
  index += 3;
  squaredError->segment<3>(index) = alpha.cwiseAbs2();
  index += 3;
  squaredError->segment<3>(index) = deltaV.cwiseAbs2();
  index += 3;
  squaredError->segment<3>(index) = deltaBg.cwiseAbs2();
  index += 3;
  squaredError->segment<3>(index) = deltaBa.cwiseAbs2();
  index += 3;

  bool isFilter = isFilteringMethod(estimatorAlgorithm);
  if (isFilter) {
    Eigen::Matrix<double, 27, 1> extraParamDeviation =
        estimator->computeImuAugmentedParamsError();
    squaredError->segment<27>(index) = extraParamDeviation.cwiseAbs2();
    index += 27;
  } else {
    squaredError->segment<27>(index).setZero();
    index += 27;
  }
  Eigen::Matrix<double, 3, 1> p_CB_est;
  okvis::kinematics::Transformation T_SC_est;
  estimator->getSensorStateEstimateAs<okvis::ceres::PoseParameterBlock>(
      currFrameId, 0, okvis::HybridFilter::SensorStates::Camera,
      okvis::HybridFilter::CameraSensorStates::T_SCi, T_SC_est);
  p_CB_est = T_SC_est.inverse().r();
  squaredError->segment<3>(index) = p_CB_est.cwiseAbs2();
  index += 3;
  Eigen::VectorXd intrinsics_true;
  refCameraGeometry->getIntrinsics(intrinsics_true);
  const int nDistortionCoeffDim =
      okvis::cameras::RadialTangentialDistortion::NumDistortionIntrinsics;
  Eigen::VectorXd distIntrinsic_true =
      intrinsics_true.tail<nDistortionCoeffDim>();
  if (isFilter) {
    Eigen::Matrix<double, Eigen::Dynamic, 1> projectionIntrinsic;
    if (projOptModelDim > 0) {
      estimator->getSensorStateEstimateAs<okvis::ceres::EuclideanParamBlock>(
          currFrameId, 0, okvis::HybridFilter::SensorStates::Camera,
          okvis::HybridFilter::CameraSensorStates::Intrinsics,
          projectionIntrinsic);
      Eigen::VectorXd local_opt_params;
      okvis::ProjectionOptGlobalToLocal(projOptModelId, intrinsics_true,
                                        &local_opt_params);

      squaredError->segment(index, projOptModelDim) =
          (projectionIntrinsic - local_opt_params).cwiseAbs2();
      index += projOptModelDim;
    }

    Eigen::Matrix<double, Eigen::Dynamic, 1> cameraDistortion_est(
        nDistortionCoeffDim);
    estimator->getSensorStateEstimateAs<okvis::ceres::EuclideanParamBlock>(
        currFrameId, 0, okvis::HybridFilter::SensorStates::Camera,
        okvis::HybridFilter::CameraSensorStates::Distortion,
        cameraDistortion_est);
    squaredError->segment(index, nDistortionCoeffDim) =
        (cameraDistortion_est - distIntrinsic_true).cwiseAbs2();
    index += nDistortionCoeffDim;

    double timeDelayEstimate(0.0), readoutTimeEstimate(0.0);
    estimator->getSensorStateEstimateAs<okvis::ceres::CameraTimeParamBlock>(
        currFrameId, 0, okvis::HybridFilter::SensorStates::Camera,
        okvis::HybridFilter::CameraSensorStates::TD, timeDelayEstimate);
    double delta_td = refCameraGeometry->imageDelay() - timeDelayEstimate;
    (*squaredError)[index] = delta_td * delta_td;
    ++index;

    estimator->getSensorStateEstimateAs<okvis::ceres::CameraTimeParamBlock>(
        currFrameId, 0, okvis::HybridFilter::SensorStates::Camera,
        okvis::HybridFilter::CameraSensorStates::TR, readoutTimeEstimate);
    double deltaReadoutTime = refCameraGeometry->readoutTime() - readoutTimeEstimate;
    (*squaredError)[index] = deltaReadoutTime * deltaReadoutTime;
    ++index;
  } else {
    squaredError->segment(index, projOptModelDim + nDistortionCoeffDim + 2)
        .setZero();
  }
}

void check_tail_mse(
    const Eigen::VectorXd& mse_tail, int projOptModelId) {
  int index = 0;
  EXPECT_LT(mse_tail.head<3>().norm(), std::pow(0.3, 2)) << "Position MSE";
  index = 3;
  EXPECT_LT(mse_tail.segment<3>(index).norm(), std::pow(0.08, 2)) << "Orientation MSE";
  index += 3;
  EXPECT_LT(mse_tail.segment<3>(index).norm(), std::pow(0.1, 2)) << "Velocity MSE";
  index += 3;
  EXPECT_LT(mse_tail.segment<3>(index).norm(), std::pow(0.002, 2)) << "Gyro bias MSE";
  index += 3;
  EXPECT_LT(mse_tail.segment<3>(index).norm(), std::pow(0.02, 2)) << "Accelerometer bias MSE";
  index += 3;
  EXPECT_LT(mse_tail.segment<9>(index).norm(), std::pow(4e-3, 2)) << "Tg MSE";
  index += 9;
  EXPECT_LT(mse_tail.segment<9>(index).norm(), std::pow(1e-3, 2)) << "Ts MSE";
  index += 9;
  EXPECT_LT(mse_tail.segment<9>(index).norm(), std::pow(5e-3, 2)) << "Ta MSE";
  index += 9;
  EXPECT_LT(mse_tail.segment<3>(index).norm(), std::pow(0.04, 2)) << "p_CB MSE";
  index += 3;
  int projIntrinsicDim = swift_vio::ProjectionOptGetMinimalDim(projOptModelId);
  EXPECT_LT(mse_tail.segment(index, projIntrinsicDim).norm(), std::pow(1, 2)) << "fxy cxy MSE";
  index += projIntrinsicDim;
  EXPECT_LT(mse_tail.segment<4>(index).norm(), std::pow(0.002, 2)) << "k1 k2 p1 p2 MSE";
  index += 4;
  EXPECT_LT(mse_tail.segment<2>(index).norm(), std::pow(1e-3, 2)) << "td tr MSE";
  index += 2;
}

void check_tail_nees(const Eigen::Vector3d &nees_tail) {
  EXPECT_LT(nees_tail[0], 8) << "Position NEES";
  EXPECT_LT(nees_tail[1], 5) << "Orientation NEES";
  EXPECT_LT(nees_tail[2], 10) << "Pose NEES";
}

/**
 * @brief testHybridFilterSinusoid
 * @param testSetting
 * @param outputPath
 * @param estimatorLabel The arbitrary label to identity the tested method.
 * @param trajLabel The trajectory name must be one of the predefined ones.
 * @param numRuns repetition of the same test.
 */
void testHybridFilterSinusoid(
    const simul::TestSetting& testSetting, const std::string& outputPath,
    std::string estimatorLabel, std::string trajLabel, int numRuns,
    const swift_vio::BackendParams& backendParams = swift_vio::BackendParams()) {
  swift_vio::EstimatorAlgorithm estimatorAlgorithm = testSetting.estimator_algorithm;

  swift_vio::StatAccumulator neesAccumulator;
  swift_vio::StatAccumulator rmseAccumulator;

  std::string truthFile = outputPath + "/" + trajLabel + ".txt";
  std::ofstream truthStream;

  // number of features tracked in a frame
  boost::accumulators::accumulator_set<
      double, boost::accumulators::features<boost::accumulators::tag::count,
                                            boost::accumulators::tag::density>>
      frameFeatureTally(boost::accumulators::tag::density::num_bins = 20,
                        boost::accumulators::tag::density::cache_size = 40);
  std::string featureHistFile = outputPath + "/FeatureHist.txt";

  okvis::timing::Timer filterTimer("msckf timer", true);

  std::string methodIdentifier =  estimatorLabel + "_" + trajLabel;
  LOG(INFO) << "Estimator algorithm: " << estimatorLabel
            << " trajectory " << trajLabel;
  std::string pathEstimatorTrajectory = outputPath + "/" + methodIdentifier;
  std::string neesFile = pathEstimatorTrajectory + "_NEES.txt";
  std::string rmseFile = pathEstimatorTrajectory + "_RMSE.txt";
  std::string metadataFile = pathEstimatorTrajectory + "_metadata.txt";
  std::ofstream metaStream;
  metaStream.open(metadataFile, std::ofstream::out);

  // only output the ground truth and data for the first successful trial
  bool verbose = false;

  std::string projOptModelName = "FIXED";
  std::string extrinsicModelName = "FIXED";
  if (isFilteringMethod(estimatorAlgorithm)) {
    projOptModelName = "FXY_CXY";
    extrinsicModelName = "P_CB";
  }
  for (int run = 0; run < numRuns; ++run) {
    verbose = neesAccumulator.succeededRuns() == 0;
    filterTimer.start();

    srand((unsigned int)time(0)); // comment out to make tests deterministic

    LOG(INFO) << "Run " << run << " " << methodIdentifier << " " << testSetting.toString();

    std::string pointFile = outputPath + "/" + trajLabel + "_Points.txt";
    std::string imuSampleFile = outputPath + "/" + trajLabel + "_IMU.txt";
    
    if (verbose) {
      truthStream.open(truthFile, std::ofstream::out);
      truthStream << "%state timestamp, frameIdInSource, T_WS(xyz, xyzw), "
                     "v_WS, bg, ba, Tg, Ts, Ta, "
                     "p_CB, fx, fy, cx, cy, k1, k2, p1, p2, td, tr"
                  << std::endl;
    }

    std::stringstream ss;
    ss << run;
    std::string outputFile = pathEstimatorTrajectory + "_" + ss.str() + ".txt";
    std::string trackStatFile = pathEstimatorTrajectory + "_trackstat_" + ss.str() + ".txt";

    simul::VioTestSystemBuilder vioSystemBuilder;
    double timeOffset = FLAGS_sim_camera_time_offset_sec;
    double readoutTime = FLAGS_sim_frame_readout_time_sec;

    simul::SimulatedTrajectoryType trajectoryId =
        simul::trajectoryLabelToId.find(trajLabel)->second;
    vioSystemBuilder.createVioSystem(testSetting, backendParams, trajectoryId,
                                     projOptModelName, extrinsicModelName,
                                     timeOffset, readoutTime,
                                     verbose ? imuSampleFile : "",
                                     verbose ? pointFile : "");
    int projOptModelId = swift_vio::ProjectionOptNameToId(projOptModelName);
    int extrinsicModelId = swift_vio::ExtrinsicModelNameToId(extrinsicModelName);

    std::ofstream debugStream;  // record state history of a trial
    if (!debugStream.is_open()) {
      debugStream.open(outputFile, std::ofstream::out);
      std::string headerLine;
      swift_vio::StreamHelper::composeHeaderLine(
          vioSystemBuilder.imuModelType(), {extrinsicModelName},
          {projOptModelName}, {vioSystemBuilder.distortionType()},
          swift_vio::FULL_STATE_WITH_ALL_CALIBRATION, &headerLine);
      debugStream << headerLine << std::endl;
    }

    std::vector<uint64_t> multiFrameIds;
    size_t poseIndex = 0u;
    bool hasStarted = false;
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
    std::shared_ptr<okvis::Estimator> estimator = vioSystemBuilder.mutableEstimator();

    std::shared_ptr<simul::SimulationFrontend> frontend = vioSystemBuilder.mutableFrontend();
    std::shared_ptr<const okvis::cameras::NCameraSystem> cameraSystem0 =
        vioSystemBuilder.trueCameraSystem();
    std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry0 =
        cameraSystem0->cameraGeometry(0);

    int expectedNumFrames = times.size() / cameraIntervalRatio + 1;
    neesAccumulator.refreshBuffer(expectedNumFrames);
    rmseAccumulator.refreshBuffer(expectedNumFrames);
    try {
      for (auto iter = times.begin(), iterEnd = times.end(); iter != iterEnd;
           iter += cameraIntervalRatio, poseIndex += cameraIntervalRatio,
           trueBiasIter += cameraIntervalRatio) {
        okvis::kinematics::Transformation T_WS(ref_T_WS_list[poseIndex]);
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
//        LOG(INFO) << "Processing frame " << id << " of index " << frameCount;
        okvis::Time currentKFTime = *iter;
        okvis::Time imuDataEndTime = currentKFTime + okvis::Duration(1);
        okvis::Time imuDataBeginTime = lastKFTime - okvis::Duration(1);
        okvis::ImuMeasurementDeque imuSegment = swift_vio::getImuMeasurements(
            imuDataBeginTime, imuDataEndTime, imuMeasurements, nullptr);
        bool asKeyframe = true;
        // add it in the window to create a new time instance
        if (!hasStarted) {
          hasStarted = true;
          frameCount = 0;
          estimator->setInitialNavState(vioSystemBuilder.initialNavState());
          estimator->addStates(mf, imuSegment, asKeyframe);
          if (isFilteringMethod(estimatorAlgorithm)) {
            const int kDistortionCoeffDim =
                okvis::cameras::RadialTangentialDistortion::NumDistortionIntrinsics;
            const int kNavImuCamParamDim =
                swift_vio::ode::kNavErrorStateDim +
                swift_vio::ImuModelGetMinimalDim(
                    swift_vio::ImuModelNameToId(vioSystemBuilder.imuModelType())) +
                2 + kDistortionCoeffDim +
                swift_vio::ExtrinsicModelGetMinimalDim(extrinsicModelId) +
                swift_vio::ProjectionOptGetMinimalDim(projOptModelId);
            ASSERT_EQ(estimator->getEstimatedVariableMinimalDim(),
                      kNavImuCamParamDim + swift_vio::HybridFilter::kClonedStateMinimalDimen)
                << "Initial cov with one cloned state has a wrong dim";
          }
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
        frameFeatureTally(trackedFeatures);
        size_t maxIterations = 10u;
        size_t numThreads = 2u;
        estimator->optimize(maxIterations, numThreads, false);
        okvis::Optimization sharedOptConfig;
        sharedOptConfig.getCovariance = true;
        sharedOptConfig.numKeyframes = 5;
        sharedOptConfig.numImuFrames = 3;
        estimator->setOptimizationOptions(sharedOptConfig);

        okvis::MapPointVector removedLandmarks;
        estimator->applyMarginalizationStrategy(sharedOptConfig.numKeyframes,
            sharedOptConfig.numImuFrames, removedLandmarks);
        estimator->print(debugStream);
        debugStream << std::endl;

        Eigen::Vector3d v_WS_true = vioSystemBuilder.sinusoidalTrajectory()
                                        ->computeGlobalLinearVelocity(*iter);
        if (verbose) {
          Eigen::VectorXd allIntrinsics;
          cameraGeometry0->getIntrinsics(allIntrinsics);
          std::shared_ptr<const okvis::kinematics::Transformation> T_SC_0
              = cameraSystem0->T_SC(0);

          truthStream << *iter << " " << id << " " << std::setfill(' ')
                      << T_WS.parameters().transpose().format(okvis::kSpaceInitFmt)
                      << " " << v_WS_true.transpose().format(okvis::kSpaceInitFmt)
                      << " " << trueBiasIter->measurement.gyroscopes.transpose().format(okvis::kSpaceInitFmt)
                      << " " << trueBiasIter->measurement.accelerometers.transpose().format(okvis::kSpaceInitFmt)
                      << "1 0 0 0 1 0 0 0 1 "
                      << "0 0 0 0 0 0 0 0 0 "
                      << "1 0 0 0 1 0 0 0 1 "
                      << T_SC_0->inverse().r().transpose().format(okvis::kSpaceInitFmt)
                      << " " << allIntrinsics.transpose().format(okvis::kSpaceInitFmt)
                      << " " << cameraGeometry0->imageDelay()
                      << " " << cameraGeometry0->readoutTime() << std::endl;
        }

        Eigen::VectorXd normalizedSquaredError;
        Eigen::VectorXd squaredError;
        computeErrors(estimator.get(), estimatorAlgorithm, T_WS, v_WS_true,
                      trueBiasIter->measurement, cameraGeometry0,
                      projOptModelId, &normalizedSquaredError, &squaredError);

        neesAccumulator.push_back(*iter, normalizedSquaredError);
        rmseAccumulator.push_back(*iter, squaredError);
        lastKFTime = currentKFTime;
      }  // every frame

      neesAccumulator.accumulate();
      rmseAccumulator.accumulate();
      check_tail_mse(rmseAccumulator.lastValue(), projOptModelId);
      check_tail_nees(neesAccumulator.lastValue());
      std::stringstream messageStream;
      messageStream << "Run " << run << " finishes with last added frame "
                    << frameCount << " of tracked features " << trackedFeatures;
      LOG(INFO) << messageStream.str();
      metaStream << messageStream.str() << std::endl;
      // output track length distribution
      std::ofstream trackStatStream(trackStatFile, std::ios_base::out);
      estimator->printTrackLengthHistogram(trackStatStream);
      trackStatStream.close();
      // end output track length distribution
      if (truthStream.is_open())
        truthStream.close();

    } catch (std::exception &e) {
      if (truthStream.is_open()) truthStream.close();
      LOG(INFO) << "Run and last added frame " << run << " " << frameCount << " "
                << e.what();
      if (debugStream.is_open()) {
          debugStream.close();
      }
//      unlink(outputFile.c_str());
    }
    double elapsedTime = filterTimer.stop();
    std::stringstream sstream;
    sstream << "Run " << run << " using time [sec] " << elapsedTime;
    LOG(INFO) << sstream.str();
    metaStream << sstream.str() << std::endl;
  }  // next run

  HistogramType hist = boost::accumulators::density(frameFeatureTally);
  outputFeatureHistogram(featureHistFile, hist);

  int numSucceededRuns = neesAccumulator.succeededRuns();
  EXPECT_GT(numSucceededRuns, 0)
      << "number of successful runs " << numSucceededRuns << " out of runs " << numRuns;
  std::string neesHeaderLine = "%state timestamp, NEES of p_WS, \\alpha_WS, T_WS";
  neesAccumulator.computeMean();
  neesAccumulator.dump(neesFile, neesHeaderLine);
  std::string rmseHeaderLine;
  swift_vio::StreamHelper::composeHeaderLine(
      "BG_BA_TG_TS_TA", {extrinsicModelName}, {projOptModelName},
      {"RadialTangentialDistortion"}, swift_vio::FULL_STATE_WITH_ALL_CALIBRATION,
      &rmseHeaderLine, false);
  rmseAccumulator.computeRootMean();
  rmseAccumulator.dump(rmseFile, rmseHeaderLine);

  metaStream << "#successful runs " << numSucceededRuns << " out of runs " << numRuns << std::endl;
  metaStream.close();
}

// FLAGS_log_dir can be passed in commandline as --log_dir=/some/log/dir
// {WavyCircle, Squircle, Circle, Dot, CircleWithFarPoints, Motionless} X
// {MSCKF with IDP and reprojection errors,
//  MSCKF with XYZ and reprojection errors,
//  MSCKF with IDP and reprojection errors and epipolar constraints for low parallax,
//  MSCKF with parallax angle and chordal distance,
//  MSCKF with parallax angle and reprojection errors,
//  TFVIO (roughly MSCKF with only epipolar constraints),
//  OKVIS, General estimator}

TEST(DeadreckoningM, TrajectoryLabel) {
  bool addImageNoise = true;
  bool useImageObservation = false;
  bool useEpipolarConstraint = false;
  int cameraObservationModelId = 0;
  int landmarkModelId = 1;
  double landmarkRadius = 5;
  simul::TestSetting testSetting(
      true, FLAGS_noisyInitialSpeedAndBiases, FLAGS_noisyInitialSensorParams,
      addImageNoise, useImageObservation, FLAGS_sim_imu_noise_factor,
      FLAGS_sim_imu_bias_noise_factor, swift_vio::EstimatorAlgorithm::MSCKF,
      useEpipolarConstraint, cameraObservationModelId, landmarkModelId,
      simul::SimCameraModelType::EUROC, simul::CameraOrientation::Forward,
      okvis::LandmarkGridType::FourWalls, landmarkRadius);
  testHybridFilterSinusoid(testSetting, FLAGS_log_dir, "DeadreckoningM",
                           FLAGS_sim_trajectory_label, FLAGS_num_runs);
}

TEST(DeadreckoningO, TrajectoryLabel) {
  bool addImageNoise = true;
  bool useImageObservation = false;
  bool useEpipolarConstraint = false;
  int cameraObservationModelId = 0;
  int landmarkModelId = 0;
  double landmarkRadius = 5;
  simul::TestSetting testSetting(
      true, FLAGS_noisyInitialSpeedAndBiases, FLAGS_noisyInitialSensorParams,
      addImageNoise, useImageObservation, FLAGS_sim_imu_noise_factor,
      FLAGS_sim_imu_bias_noise_factor, swift_vio::EstimatorAlgorithm::OKVIS,
      useEpipolarConstraint, cameraObservationModelId, landmarkModelId,
      simul::SimCameraModelType::EUROC, simul::CameraOrientation::Forward,
      okvis::LandmarkGridType::FourWalls, landmarkRadius);
  testHybridFilterSinusoid(testSetting, FLAGS_log_dir, "DeadreckoningO",
                           FLAGS_sim_trajectory_label, FLAGS_num_runs);
}

TEST(HybridFilter, TrajectoryLabel) {
  bool addImageNoise = true;
  bool useImageObservation = true;
  bool useEpipolarConstraint = false;
  int cameraObservationModelId = 0;
  int landmarkModelId = 1;
  double landmarkRadius = 5;
  simul::TestSetting testSetting(
      true, FLAGS_noisyInitialSpeedAndBiases, FLAGS_noisyInitialSensorParams,
      addImageNoise, useImageObservation, FLAGS_sim_imu_noise_factor,
      FLAGS_sim_imu_bias_noise_factor, swift_vio::EstimatorAlgorithm::HybridFilter,
      useEpipolarConstraint, cameraObservationModelId, landmarkModelId,
      simul::SimCameraModelType::EUROC, simul::CameraOrientation::Forward,
      okvis::LandmarkGridType::FourWalls, landmarkRadius);
  testHybridFilterSinusoid(testSetting, FLAGS_log_dir, "HybridFilter",
                           FLAGS_sim_trajectory_label, FLAGS_num_runs);
}

TEST(MSCKF, TrajectoryLabel) {
  bool addImageNoise = true;
  bool useImageObservation = true;
  bool useEpipolarConstraint = false;
  int cameraObservationModelId = 0;
  int landmarkModelId = 1;
  double landmarkRadius = 5;
  simul::TestSetting testSetting(
      true, FLAGS_noisyInitialSpeedAndBiases, FLAGS_noisyInitialSensorParams,
      addImageNoise, useImageObservation, FLAGS_sim_imu_noise_factor,
      FLAGS_sim_imu_bias_noise_factor, swift_vio::EstimatorAlgorithm::MSCKF,
      useEpipolarConstraint, cameraObservationModelId, landmarkModelId,
      simul::SimCameraModelType::EUROC, simul::CameraOrientation::Forward,
      okvis::LandmarkGridType::FourWalls, landmarkRadius);
  testHybridFilterSinusoid(testSetting, FLAGS_log_dir, "MSCKF",
                           FLAGS_sim_trajectory_label, FLAGS_num_runs);
}

TEST(MSCKF, HuaiThesis) {
  bool addImageNoise = true;
  bool useImageObservation = true;
  bool useEpipolarConstraint = false;
  int cameraObservationModelId = 0;
  int landmarkModelId = 1;
  double landmarkRadius = 5;
  simul::TestSetting testSetting(
      true, FLAGS_noisyInitialSpeedAndBiases, FLAGS_noisyInitialSensorParams,
      addImageNoise, useImageObservation, FLAGS_sim_imu_noise_factor,
      FLAGS_sim_imu_bias_noise_factor, swift_vio::EstimatorAlgorithm::MSCKF,
      useEpipolarConstraint, cameraObservationModelId, landmarkModelId,
      simul::SimCameraModelType::EUROC, simul::CameraOrientation::Forward,
      okvis::LandmarkGridType::FourWallsFloorCeiling, landmarkRadius);
  testHybridFilterSinusoid(testSetting, FLAGS_log_dir, "MSCKF", "Ball",
                           FLAGS_num_runs);
}

TEST(MSCKF, CircleFarPoints) {
  bool addImageNoise = true;
  bool useImageObservation = true;
  int cameraObservationModelId = 0;
  int landmarkModelId = 1;
  bool useEpipolarConstraint = false;
  double landmarkRadius = 50;
  simul::TestSetting testSetting(
      true, FLAGS_noisyInitialSpeedAndBiases, FLAGS_noisyInitialSensorParams,
      addImageNoise, useImageObservation, FLAGS_sim_imu_noise_factor,
      FLAGS_sim_imu_bias_noise_factor, swift_vio::EstimatorAlgorithm::MSCKF,
      useEpipolarConstraint, cameraObservationModelId, landmarkModelId,
      simul::SimCameraModelType::EUROC, simul::CameraOrientation::Forward,
      okvis::LandmarkGridType::Cylinder, landmarkRadius);
  testHybridFilterSinusoid(testSetting, FLAGS_log_dir, "MSCKF", "Circle",
                           FLAGS_num_runs);
}

TEST(General, TrajectoryLabel) {
  bool addImageNoise = true;
  bool useImageObservation = true;
  bool useEpipolarConstraint = false;
  int cameraObservationModelId = 0;
  int landmarkModelId = 0;
  double landmarkRadius = 5;
  simul::TestSetting testSetting(
      true, FLAGS_noisyInitialSpeedAndBiases, FLAGS_noisyInitialSensorParams,
      addImageNoise, useImageObservation, FLAGS_sim_imu_noise_factor,
      FLAGS_sim_imu_bias_noise_factor, swift_vio::EstimatorAlgorithm::General,
      useEpipolarConstraint, cameraObservationModelId, landmarkModelId,
      simul::SimCameraModelType::EUROC, simul::CameraOrientation::Forward,
      okvis::LandmarkGridType::FourWalls, landmarkRadius);
  testHybridFilterSinusoid(testSetting, FLAGS_log_dir, "General",
                           FLAGS_sim_trajectory_label, FLAGS_num_runs);
}

// It is also possible to test OKVIS with an infinity time horizon by
// disabling the applyMarginalizationStrategy line in testHybridFilterSinusoid.
// That setting has been shown to output consistent covariance to the actual
// errors up to 10 secs simulation, however, it is very time-consuming,
// much slower than iSAM2.
TEST(OKVIS, TrajectoryLabel) {
  bool addImageNoise = true;
  bool useImageObservation = true;
  bool useEpipolarConstraint = false;
  int cameraObservationModelId = 0;
  int landmarkModelId = 0;
  double landmarkRadius = 5;
  simul::TestSetting testSetting(
      true, FLAGS_noisyInitialSpeedAndBiases, FLAGS_noisyInitialSensorParams,
      addImageNoise, useImageObservation, FLAGS_sim_imu_noise_factor,
      FLAGS_sim_imu_bias_noise_factor, swift_vio::EstimatorAlgorithm::OKVIS,
      useEpipolarConstraint, cameraObservationModelId, landmarkModelId,
      simul::SimCameraModelType::EUROC, simul::CameraOrientation::Forward,
      okvis::LandmarkGridType::FourWalls, landmarkRadius);
  testHybridFilterSinusoid(testSetting, FLAGS_log_dir, "OKVIS",
                           FLAGS_sim_trajectory_label, FLAGS_num_runs);
}

TEST(SlidingWindowSmoother, TrajectoryLabel) {
  bool addImageNoise = true;
  bool useImageObservation = true;
  bool useEpipolarConstraint = false;
  int cameraObservationModelId = 0;
  int landmarkModelId = 0;
  double landmarkRadius = 5;
  simul::TestSetting testSetting(
      true, FLAGS_noisyInitialSpeedAndBiases, FLAGS_noisyInitialSensorParams,
      addImageNoise, useImageObservation, FLAGS_sim_imu_noise_factor,
      FLAGS_sim_imu_bias_noise_factor,
      swift_vio::EstimatorAlgorithm::SlidingWindowSmoother, useEpipolarConstraint,
      cameraObservationModelId, landmarkModelId,
      simul::SimCameraModelType::EUROC, simul::CameraOrientation::Forward,
      okvis::LandmarkGridType::FourWalls, landmarkRadius);
  testHybridFilterSinusoid(testSetting, FLAGS_log_dir, "SlidingWindowSmoother",
                           FLAGS_sim_trajectory_label, FLAGS_num_runs);
}

TEST(RiSlidingWindowSmoother, TrajectoryLabel) {
  bool addImageNoise = true;
  bool useImageObservation = true;
  bool useEpipolarConstraint = false;
  int cameraObservationModelId = 0;
  int landmarkModelId = 0;
  double landmarkRadius = 5;
  simul::TestSetting testSetting(
      true, FLAGS_noisyInitialSpeedAndBiases, FLAGS_noisyInitialSensorParams,
      addImageNoise, useImageObservation, FLAGS_sim_imu_noise_factor,
      FLAGS_sim_imu_bias_noise_factor,
      swift_vio::EstimatorAlgorithm::RiSlidingWindowSmoother, useEpipolarConstraint,
      cameraObservationModelId, landmarkModelId,
      simul::SimCameraModelType::EUROC, simul::CameraOrientation::Forward,
      okvis::LandmarkGridType::FourWalls, landmarkRadius);
  swift_vio::BackendParams backendParams;
  backendParams.backendModality_ = okvis::BackendModality::STRUCTURELESS;
  testHybridFilterSinusoid(testSetting, FLAGS_log_dir, "RiSlidingWindowSmoother",
                           FLAGS_sim_trajectory_label, FLAGS_num_runs, backendParams);
}

TEST(TFVIO, TrajectoryLabel) {
  bool addImageNoise = true;
  bool useImageObservation = true;
  bool useEpipolarConstraint = false;
  int cameraObservationModelId = 1;
  int landmarkModelId = 0;
  double landmarkRadius = 5;
  simul::TestSetting testSetting(
      true, FLAGS_noisyInitialSpeedAndBiases, FLAGS_noisyInitialSensorParams,
      addImageNoise, useImageObservation, FLAGS_sim_imu_noise_factor,
      FLAGS_sim_imu_bias_noise_factor, swift_vio::EstimatorAlgorithm::TFVIO,
      useEpipolarConstraint, cameraObservationModelId, landmarkModelId,
      simul::SimCameraModelType::EUROC, simul::CameraOrientation::Forward,
      okvis::LandmarkGridType::FourWalls, landmarkRadius);
  testHybridFilterSinusoid(testSetting, FLAGS_log_dir, "TFVIO",
                           FLAGS_sim_trajectory_label, FLAGS_num_runs);
}

TEST(MSCKFWithEuclidean, TrajectoryLabel) {
  bool addImageNoise = true;
  bool useImageObservation = true;
  bool useEpipolarConstraint = false;
  int cameraObservationModelId = 0;
  int landmarkModelId = 0;
  double landmarkRadius = 5;
  simul::TestSetting testSetting(
      true, FLAGS_noisyInitialSpeedAndBiases, FLAGS_noisyInitialSensorParams,
      addImageNoise, useImageObservation, FLAGS_sim_imu_noise_factor,
      FLAGS_sim_imu_bias_noise_factor, swift_vio::EstimatorAlgorithm::MSCKF,
      useEpipolarConstraint, cameraObservationModelId, landmarkModelId,
      simul::SimCameraModelType::EUROC, simul::CameraOrientation::Forward,
      okvis::LandmarkGridType::FourWalls, landmarkRadius);
  testHybridFilterSinusoid(testSetting, FLAGS_log_dir, "MSCKFWithEuclidean",
                           FLAGS_sim_trajectory_label, FLAGS_num_runs);
}

TEST(MSCKFWithPAP, TrajectoryLabel) {
  bool addImageNoise = true;
  bool useImageObservation = true;
  bool useEpipolarConstraint = false;
  int cameraObservationModelId = 2;
  int landmarkModelId = 2;
  double landmarkRadius = 5;
  simul::TestSetting testSetting(
      true, FLAGS_noisyInitialSpeedAndBiases, FLAGS_noisyInitialSensorParams,
      addImageNoise, useImageObservation, FLAGS_sim_imu_noise_factor,
      FLAGS_sim_imu_bias_noise_factor, swift_vio::EstimatorAlgorithm::MSCKF,
      useEpipolarConstraint, cameraObservationModelId, landmarkModelId,
      simul::SimCameraModelType::EUROC, simul::CameraOrientation::Forward,
      okvis::LandmarkGridType::FourWalls, landmarkRadius);
  testHybridFilterSinusoid(testSetting, FLAGS_log_dir, "MSCKFWithPAP",
                           FLAGS_sim_trajectory_label, FLAGS_num_runs);
}

TEST(MSCKFWithReprojectionErrorPAP, TrajectoryLabel) {
  bool addImageNoise = true;
  bool useImageObservation = true;
  bool useEpipolarConstraint = false;
  int cameraObservationModelId = 3;
  int landmarkModelId = 2;
  double landmarkRadius = 5;
  simul::TestSetting testSetting(
      true, FLAGS_noisyInitialSpeedAndBiases, FLAGS_noisyInitialSensorParams,
      addImageNoise, useImageObservation, FLAGS_sim_imu_noise_factor,
      FLAGS_sim_imu_bias_noise_factor, swift_vio::EstimatorAlgorithm::MSCKF,
      useEpipolarConstraint, cameraObservationModelId, landmarkModelId,
      simul::SimCameraModelType::EUROC, simul::CameraOrientation::Forward,
      okvis::LandmarkGridType::FourWalls, landmarkRadius);
  testHybridFilterSinusoid(testSetting, FLAGS_log_dir,
                           "MSCKFWithReprojectionErrorPAP",
                           FLAGS_sim_trajectory_label, FLAGS_num_runs);
}

TEST(MSCKFWithPAP, SquircleBackward) {
  bool addImageNoise = true;
  bool useImageObservation = true;
  bool useEpipolarConstraint = false;
  int cameraObservationModelId = 2;
  int landmarkModelId = 2;
  double landmarkRadius = 5;
  simul::TestSetting testSetting(
      true, FLAGS_noisyInitialSpeedAndBiases, FLAGS_noisyInitialSensorParams,
      addImageNoise, useImageObservation, FLAGS_sim_imu_noise_factor,
      FLAGS_sim_imu_bias_noise_factor, swift_vio::EstimatorAlgorithm::MSCKF,
      useEpipolarConstraint, cameraObservationModelId, landmarkModelId,
      simul::SimCameraModelType::EUROC, simul::CameraOrientation::Backward,
      okvis::LandmarkGridType::FourWalls, landmarkRadius);
  testHybridFilterSinusoid(testSetting, FLAGS_log_dir, "MSCKFWithPAP",
                           "Squircle", FLAGS_num_runs);
}

TEST(MSCKFWithPAP, SquircleSideways) {
  bool addImageNoise = true;
  bool useImageObservation = true;
  bool useEpipolarConstraint = false;
  int cameraObservationModelId = 2;
  int landmarkModelId = 2;
  double landmarkRadius = 5;
  simul::TestSetting testSetting(
      true, FLAGS_noisyInitialSpeedAndBiases, FLAGS_noisyInitialSensorParams,
      addImageNoise, useImageObservation, FLAGS_sim_imu_noise_factor,
      FLAGS_sim_imu_bias_noise_factor, swift_vio::EstimatorAlgorithm::MSCKF,
      useEpipolarConstraint, cameraObservationModelId, landmarkModelId,
      simul::SimCameraModelType::EUROC, simul::CameraOrientation::Right,
      okvis::LandmarkGridType::FourWalls, landmarkRadius);
  testHybridFilterSinusoid(testSetting, FLAGS_log_dir, "MSCKFWithPAP",
                           "Squircle", FLAGS_num_runs);
}

TEST(MSCKFWithEpipolarConstraint, TrajectoryLabel) {
  bool addImageNoise = true;
  bool useImageObservation = true;
  bool useEpipolarConstraint = false;
  int cameraObservationModelId = 0;
  int landmarkModelId = 1;
  double landmarkRadius = 5;
  simul::TestSetting testSetting(
      true, FLAGS_noisyInitialSpeedAndBiases, FLAGS_noisyInitialSensorParams,
      addImageNoise, useImageObservation, FLAGS_sim_imu_noise_factor,
      FLAGS_sim_imu_bias_noise_factor, swift_vio::EstimatorAlgorithm::MSCKF,
      useEpipolarConstraint, cameraObservationModelId, landmarkModelId,
      simul::SimCameraModelType::EUROC, simul::CameraOrientation::Forward,
      okvis::LandmarkGridType::FourWalls, landmarkRadius);
  testHybridFilterSinusoid(testSetting, FLAGS_log_dir,
                           "MSCKFWithEpipolarConstraint",
                           FLAGS_sim_trajectory_label, FLAGS_num_runs);
}

TEST(MSCKFWithEpipolarConstraint, CircleFarPoints) {
  bool addImageNoise = true;
  bool useImageObservation = true;
  bool useEpipolarConstraint = true;
  int cameraObservationModelId = 0;
  int landmarkModelId = 1;
  double landmarkRadius = 50;
  simul::TestSetting testSetting(
      true, FLAGS_noisyInitialSpeedAndBiases, FLAGS_noisyInitialSensorParams,
      addImageNoise, useImageObservation, FLAGS_sim_imu_noise_factor,
      FLAGS_sim_imu_bias_noise_factor, swift_vio::EstimatorAlgorithm::MSCKF,
      useEpipolarConstraint, cameraObservationModelId, landmarkModelId,
      simul::SimCameraModelType::EUROC, simul::CameraOrientation::Forward,
      okvis::LandmarkGridType::Cylinder, landmarkRadius);
  testHybridFilterSinusoid(testSetting, FLAGS_log_dir,
                           "MSCKFWithEpipolarConstraint", "Circle",
                           FLAGS_num_runs);
}
