#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/density.hpp>
#include <boost/accumulators/statistics/stats.hpp>

#include <gtest/gtest.h>

#include <vio/Sample.h>

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

#include <msckf/CameraTimeParamBlock.hpp>
#include <msckf/EuclideanParamBlock.hpp>

#include <msckf/ImuOdometry.h>
#include <msckf/ProjParamOptModels.hpp>
#include <msckf/VioTestSystemBuilder.hpp>

// values used in TestEstimator.cpp
//imuParameters.sigma_g_c = 6.0e-4;
//imuParameters.sigma_a_c = 2.0e-3;
//imuParameters.sigma_gw_c = 3.0e-6;
//imuParameters.sigma_aw_c = 2.0e-5;

DEFINE_bool(
    add_prior_noise, true,
    "add noise to initial states, including velocity, gyro bias, accelerometer "
    "bias, imu misalignment matrices, extrinsic parameters, camera projection "
    "and distortion intrinsic parameters, td, tr");

DECLARE_bool(use_mahalanobis);
DECLARE_int32(estimator_algorithm);

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

inline bool isFilteringMethod(int algorithmId) {
  return algorithmId >= 4;
}

/**
 * @brief computeErrors
 * @param estimator
 * @param T_WS
 * @param v_WS_true
 * @param ref_measurement
 * @param ref_camera_geometry
 * @param normalizedError nees in position, nees in orientation, nees in pose
 * @param rmsError rmse in xyz, \alpha, v_WS, bg, ba, Tg, Ts, Ta, p_CB,
 *  (fx, fy), (cx, cy), k1, k2, p1, p2, td, tr
 */
void computeErrors(
    const okvis::Estimator* estimator,
    const okvis::kinematics::Transformation& T_WS,
    const Eigen::Vector3d& v_WS_true,
    const okvis::ImuSensorReadings& ref_measurement,
    std::shared_ptr<const okvis::cameras::CameraBase> ref_camera_geometry,
    const int projOptModelId, Eigen::Vector3d* normalizedError,
    Eigen::VectorXd* rmsError) {
  int projOptModelDim = okvis::ProjectionOptGetMinimalDim(projOptModelId);
  rmsError->resize(51 + projOptModelDim);

  okvis::kinematics::Transformation T_WS_est;
  uint64_t currFrameId = estimator->currentFrameId();
  estimator->get_T_WS(currFrameId, T_WS_est);
  Eigen::Vector3d delta = T_WS.r() - T_WS_est.r();
  Eigen::Vector3d alpha = vio::unskew3d(T_WS.C() * T_WS_est.C().transpose() -
                                        Eigen::Matrix3d::Identity());
  Eigen::Matrix<double, 6, 1> deltaPose;
  deltaPose << delta, alpha;
  Eigen::MatrixXd covariance;
  bool isFilter = isFilteringMethod(FLAGS_estimator_algorithm);

  estimator->computeCovariance(&covariance);

  (*normalizedError)[0] =
      delta.transpose() * covariance.topLeftCorner<3, 3>().inverse() * delta;
  (*normalizedError)[1] =
      alpha.transpose() * covariance.block<3, 3>(3, 3).inverse() * alpha;
  Eigen::Matrix<double, 6, 1> tempPoseError =
      covariance.topLeftCorner<6, 6>().ldlt().solve(deltaPose);
  (*normalizedError)[2] = deltaPose.transpose() * tempPoseError;

  Eigen::Matrix<double, 9, 1> eye;
  eye << 1, 0, 0, 0, 1, 0, 0, 0, 1;
  int index = 0;
  rmsError->head<3>() = delta.cwiseAbs2();
  index += 3;
  rmsError->segment<3>(index) = alpha.cwiseAbs2();
  index += 3;
  okvis::SpeedAndBias speedAndBias_est;
  estimator->getSpeedAndBias(currFrameId, 0, speedAndBias_est);
  Eigen::Vector3d deltaV = speedAndBias_est.head<3>() - v_WS_true;
  rmsError->segment<3>(index) = deltaV.cwiseAbs2();
  index += 3;
  rmsError->segment<3>(index) =
      (speedAndBias_est.segment<3>(3) - ref_measurement.gyroscopes).cwiseAbs2();
  index += 3;
  rmsError->segment<3>(index) =
      (speedAndBias_est.tail<3>() - ref_measurement.accelerometers).cwiseAbs2();
  index += 3;

  if (isFilter) {
    Eigen::Matrix<double, 27, 1> extraParamDeviation =
        estimator->computeImuAugmentedParamsError();
    rmsError->segment<27>(index) = extraParamDeviation.cwiseAbs2();
    index += 27;
  } else {
    rmsError->segment<27>(index).setZero();
    index += 27;
  }
  Eigen::Matrix<double, 3, 1> p_CB_est;
  okvis::kinematics::Transformation T_SC_est;
  estimator->getSensorStateEstimateAs<okvis::ceres::PoseParameterBlock>(
      currFrameId, 0, okvis::HybridFilter::SensorStates::Camera,
      okvis::HybridFilter::CameraSensorStates::T_SCi, T_SC_est);
  p_CB_est = T_SC_est.inverse().r();
  rmsError->segment<3>(index) = p_CB_est.cwiseAbs2();
  index += 3;
  Eigen::VectorXd intrinsics_true;
  ref_camera_geometry->getIntrinsics(intrinsics_true);
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

      rmsError->segment(index, projOptModelDim) =
          (projectionIntrinsic - local_opt_params).cwiseAbs2();
      index += projOptModelDim;
    }

    Eigen::Matrix<double, Eigen::Dynamic, 1> cameraDistortion_est(
        nDistortionCoeffDim);
    estimator->getSensorStateEstimateAs<okvis::ceres::EuclideanParamBlock>(
        currFrameId, 0, okvis::HybridFilter::SensorStates::Camera,
        okvis::HybridFilter::CameraSensorStates::Distortion,
        cameraDistortion_est);
    rmsError->segment(index, nDistortionCoeffDim) =
        (cameraDistortion_est - distIntrinsic_true).cwiseAbs2();
    index += nDistortionCoeffDim;

    double td_est(0.0), tr_est(0.0);
    estimator->getSensorStateEstimateAs<okvis::ceres::CameraTimeParamBlock>(
        currFrameId, 0, okvis::HybridFilter::SensorStates::Camera,
        okvis::HybridFilter::CameraSensorStates::TD, td_est);
    (*rmsError)[index] = td_est * td_est;
    ++index;

    estimator->getSensorStateEstimateAs<okvis::ceres::CameraTimeParamBlock>(
        currFrameId, 0, okvis::HybridFilter::SensorStates::Camera,
        okvis::HybridFilter::CameraSensorStates::TR, tr_est);
    (*rmsError)[index] = tr_est * tr_est;
    ++index;
  } else {
    rmsError->segment(index, projOptModelDim + nDistortionCoeffDim + 2)
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
  int projIntrinsicDim = okvis::ProjectionOptGetMinimalDim(projOptModelId);
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

std::map<int, std::string> estimatorIdToLabel{
  {0, "OKVIS"},
  {1, "General"},
  {4, "MSCKF"},
  {5, "TFVIO"},
  {6, "PAVIO"},
  {9, "Uncanny"},
};

std::map<std::string, int> estimatorLabelToId{
  {"OKVIS", 0},
  {"General", 1},
  {"MSCKF", 4},
  {"TFVIO", 5},
  {"PAVIO", 6},
  {"Uncanny", 9},
};

std::vector<std::string> trajectoryIdToLabel{
    "Torus", "Ball", "Squircle", "Circle", "Dot", "Uncanny",
};

std::map<std::string, int> trajectoryLabelToId{
    {"Torus", 0},  {"Ball", 1}, {"Squircle", 2},
    {"Circle", 3}, {"Dot", 4},  {"Uncanny", 5},
};

/**
 * @brief testHybridFilterSinusoid
 * @param outputPath
 * @param runs
 * @param trajectoryId: 0: yarn torus, 1: yarn ball, 2: rounded square,
 *     3: circle, 4: dot
 * @param cameraOrientationId
 */
void testHybridFilterSinusoid(const std::string& outputPath,
                              const int runs = 100,
                              const int trajectoryId=1,
                              simul::CameraOrientation cameraOrientationId=simul::CameraOrientation::Forward) {

  // definition of NEES in Huang et al. 2007 Generalized Analysis and
  // Improvement of the consistency of EKF-based SLAM
  // https://pdfs.semanticscholar.org/4881/2a9d4a2ae5eef95939cbee1119e9f15633e8.pdf
  // each entry, timestamp, nees in position, orientation, and pose, the
  // expected NEES is 6 for pose error, see Li ijrr high precision
  // nees for one run, neesSum for multiple runs
  std::vector<std::pair<okvis::Time, Eigen::Vector3d>> nees, neesSum;

  // each entry state timestamp, rmse in xyz, \alpha, v_WS, bg, ba, Tg, Ts, Ta,
  // p_CB, fx, fy, cx, cy, k1, k2, p1, p2, td, tr
  // rmse for one run, rmseSum for multiple runs
  std::vector<std::pair<okvis::Time, Eigen::VectorXd>,
      Eigen::aligned_allocator<std::pair<okvis::Time, Eigen::VectorXd>>>
      rmse, rmseSum;

  std::string trajLabel = trajectoryIdToLabel[trajectoryId];
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
  std::string estimatorLabel = estimatorIdToLabel[FLAGS_estimator_algorithm];

  LOG(INFO) << "Estimator algorithm: " << FLAGS_estimator_algorithm
            << " " << estimatorLabel << " trajectory " << trajLabel;
  std::string neesFile = outputPath + "/" + estimatorLabel + "_" +
      trajLabel + "_NEES.txt";
  std::string rmseFile = outputPath + "/" + estimatorLabel + "_" +
      trajLabel + "_RMSE.txt";

  // only output the ground truth and data for the first successful trial
  bool bVerbose = false;
  int successRuns = 0;

  std::string projOptModelName = "FIXED";
  std::string extrinsicModelName = "FIXED";
  if (FLAGS_add_prior_noise) {
    projOptModelName = "FXY_CXY";
    extrinsicModelName = "P_CB";
  }
  for (int run = 0; run < runs; ++run) {
    bVerbose = successRuns == 0;
    filterTimer.start();

    srand((unsigned int)time(0)); // comment out to make tests deterministic
    okvis::TestSetting cases[] = {
        okvis::TestSetting(true, FLAGS_add_prior_noise, false, true, true,
        0.5, 0.5, FLAGS_estimator_algorithm)};
    size_t c = 0;
    LOG(INFO) << "Run " << run << " " << cases[c].print() << " algo " << estimatorLabel;

    std::string pointFile = outputPath + "/" + trajLabel + "_Points.txt";
    std::string imuSampleFile = outputPath + "/" + trajLabel + "_IMU.txt";

    if (bVerbose) {
      truthStream.open(truthFile, std::ofstream::out);
      truthStream << "%state timestamp, frameIdInSource, T_WS(xyz, xyzw), "
                     "v_WS, bg, ba, Tg, Ts, Ta, "
                     "p_CB, fx, fy, cx, cy, k1, k2, p1, p2, td, tr"
                  << std::endl;
    }

    std::stringstream ss;
    ss << run;
    std::string outputFile = outputPath + "/" + estimatorLabel + "_" +
                             trajLabel + "_" + ss.str() + ".txt";
    std::string trackStatFile = outputPath + "/" + estimatorLabel + "_" +
                                trajLabel + "_trackstat_" + ss.str() + ".txt";

    simul::VioTestSystemBuilder vioSystemBuilder;
    double timeOffset = 0.0;
    double readoutTime = 0.0;
    int cameraModelId = 0;
    vioSystemBuilder.createVioSystem(cases[c], trajectoryId,
                                     projOptModelName, extrinsicModelName,
                                     cameraModelId,
                                     cameraOrientationId, timeOffset, readoutTime,
                                     bVerbose ? imuSampleFile : "",
                                     bVerbose ? pointFile : "");
    int projOptModelId = okvis::ProjectionOptNameToId(projOptModelName);
    int extrinsicModelId = okvis::ExtrinsicModelNameToId(extrinsicModelName);

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
    std::shared_ptr<okvis::Estimator> estimator = vioSystemBuilder.mutableEstimator();
    int landmarkModelId = 2;
    int cameraObservationModelId = 2;
    std::shared_ptr<okvis::HybridFilter> hybridFilter =
        std::dynamic_pointer_cast<okvis::HybridFilter>(estimator);
    if (hybridFilter) {
      int insideLandmarkModelId = hybridFilter->landmarkModelId();
      int insideCameraObservationModelId = hybridFilter->cameraObservationModelId();
      LOG(INFO) << "Previous landmark model " << insideLandmarkModelId
                << " camera model " << insideCameraObservationModelId;

      hybridFilter->setLandmarkModel(landmarkModelId);
      hybridFilter->setCameraObservationModel(cameraObservationModelId);
      insideLandmarkModelId = hybridFilter->landmarkModelId();
      insideCameraObservationModelId = hybridFilter->cameraObservationModelId();
      LOG(INFO) << "Present landmark model " << insideLandmarkModelId
                << " camera model " << insideCameraObservationModelId;
    }

    std::shared_ptr<okvis::SimulationFrontend> frontend = vioSystemBuilder.mutableFrontend();
    std::shared_ptr<const okvis::cameras::NCameraSystem> cameraSystem0 =
        vioSystemBuilder.trueCameraSystem();
    std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry0 =
        cameraSystem0->cameraGeometry(0);
    nees.clear();
    rmse.clear();
    try {
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
          if (isFilteringMethod(FLAGS_estimator_algorithm)) {
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
          }
        } else {
          estimator->addStates(mf, imuSegment, asKeyframe);
          ++frameCount;
        }

        // add landmark observations
        trackedFeatures = 0;
        if (cases[c].useImageObservs) {
          trackedFeatures = frontend->dataAssociationAndInitialization(
              *estimator, T_WS, cameraSystem0, mf, &asKeyframe);
          estimator->setKeyframe(mf->id(), asKeyframe);
        }
        frameFeatureTally(trackedFeatures);
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
        size_t numImuFrames = 5u;
        if (isFilteringMethod(FLAGS_estimator_algorithm)) {
          numImuFrames = 20u;
        }
        estimator->applyMarginalizationStrategy(numKeyFrames, numImuFrames, removedLandmarks);
        estimator->print(debugStream);
        debugStream << std::endl;

        Eigen::Vector3d v_WS_true = vioSystemBuilder.sinusoidalTrajectory()
                                        ->computeGlobalLinearVelocity(*iter);
        if (bVerbose) {
          Eigen::VectorXd allIntrinsics;
          cameraGeometry0->getIntrinsics(allIntrinsics);
          std::shared_ptr<const okvis::kinematics::Transformation> T_SC_0
              = cameraSystem0->T_SC(0);
          Eigen::IOFormat SpaceInitFmt(Eigen::StreamPrecision,
                                       Eigen::DontAlignCols, " ", " ", "", "",
                                       "", "");
          truthStream << *iter << " " << id << " " << std::setfill(' ')
                      << T_WS.parameters().transpose().format(SpaceInitFmt)
                      << " " << v_WS_true.transpose().format(SpaceInitFmt)
                      << " 0 0 0 0 0 0 "
                      << "1 0 0 0 1 0 0 0 1 "
                      << "0 0 0 0 0 0 0 0 0 "
                      << "1 0 0 0 1 0 0 0 1 "
                      << T_SC_0->inverse().r().transpose().format(SpaceInitFmt)
                      << " " << allIntrinsics.transpose().format(SpaceInitFmt)
                      << " 0 0" << std::endl;
        }

        Eigen::Vector3d normalizedError;
        Eigen::VectorXd rmsError;
        computeErrors(estimator.get(), T_WS, v_WS_true,
                      trueBiasIter->measurement, cameraGeometry0,
                      projOptModelId, &normalizedError, &rmsError);

        nees.push_back(std::make_pair(*iter, normalizedError));
        rmse.push_back(std::make_pair(*iter, rmsError));
        lastKFTime = currentKFTime;
      }  // every keyframe

      if (neesSum.empty()) {
        neesSum = nees;
        rmseSum = rmse;
      } else {
        for (size_t jack = 0; jack < neesSum.size(); ++jack) {
          neesSum[jack].second += nees[jack].second;
          rmseSum[jack].second += rmse[jack].second;
        }
      }

      check_tail_mse(rmse.back().second, projOptModelId);
      check_tail_nees(nees.back().second);

      LOG(INFO) << "Run " << run << " finishes with last added frame " << frameCount
                << " of tracked features " << trackedFeatures << std::endl;

      // output track length distribution
      std::ofstream trackStatStream(trackStatFile, std::ios_base::out);
      estimator->printTrackLengthHistogram(trackStatStream);
      trackStatStream.close();
      // end output track length distribution
      if (truthStream.is_open())
        truthStream.close();

      ++successRuns;
    } catch (std::exception &e) {
      if (truthStream.is_open()) truthStream.close();
      LOG(INFO) << "Run and last added frame " << run << " " << frameCount << " "
                << e.what();
      // revert the accumulated errors and delete the corresponding file
      if (debugStream.is_open()) {
          debugStream.close();
      }
//      unlink(outputFile.c_str());
    }
    double elapsedTime = filterTimer.stop();
    LOG(INFO) << "Run " << run << " using time [sec] " << elapsedTime;
  }  // next run

  HistogramType hist = boost::accumulators::density(frameFeatureTally);
  outputFeatureHistogram(featureHistFile, hist);

  for (auto it = neesSum.begin(); it != neesSum.end(); ++it)
    (it->second) /= successRuns;
  std::ofstream neesStream;
  neesStream.open(neesFile, std::ofstream::out);
  neesStream << "%state timestamp, NEES of p_WS, \\alpha_WS, T_WS "
             << std::endl;
  for (auto it = neesSum.begin(); it != neesSum.end(); ++it)
    neesStream << it->first << " " << it->second.transpose() << std::endl;
  neesStream.close();

  EXPECT_GT(successRuns, 0)
      << "number of successful runs " << successRuns << " out of runs " << runs;
  for (auto it = rmseSum.begin(); it != rmseSum.end(); ++it)
    it->second = ((it->second) / successRuns).cwiseSqrt();

  std::ofstream rmseStream;
  rmseStream.open(rmseFile, std::ofstream::out);
  std::string headerLine;
  okvis::StreamHelper::composeHeaderLine(
        "BG_BA_TG_TS_TA", projOptModelName,
        extrinsicModelName, "RadialTangentialDistortion",
        okvis::FULL_STATE_WITH_ALL_CALIBRATION,
        &headerLine, false);
  rmseStream << headerLine << std::endl;
  for (auto it = rmseSum.begin(); it != rmseSum.end(); ++it)
    rmseStream << it->first << " " << it->second.transpose() << std::endl;
  rmseStream.close();
}

// FLAGS_log_dir can be passed in commandline as --log_dir=/some/log/dir
#ifndef TEST_ALGO_TRAJ
#define TEST_ALGO_TRAJ(algo_name, traj_name, repeat_times_value, \
                       camera_orientation_value)                 \
  {                                                              \
    int32_t old_algorithm = FLAGS_estimator_algorithm;           \
    FLAGS_estimator_algorithm = estimatorLabelToId[#algo_name];  \
    testHybridFilterSinusoid(FLAGS_log_dir, repeat_times_value,  \
                             trajectoryLabelToId[#traj_name],    \
                             camera_orientation_value);          \
    FLAGS_estimator_algorithm = old_algorithm;                   \
  }
#endif

TEST(MSCKF, Ball)
TEST_ALGO_TRAJ(MSCKF, Ball, 5, simul::CameraOrientation::Forward)

TEST(OKVIS, Ball)
TEST_ALGO_TRAJ(OKVIS, Ball, 5, simul::CameraOrientation::Forward)

TEST(PAVIO, Ball)
TEST_ALGO_TRAJ(PAVIO, Ball, 5, simul::CameraOrientation::Forward)

TEST(TFVIO, Ball)
TEST_ALGO_TRAJ(TFVIO, Ball, 5, simul::CameraOrientation::Forward)

TEST(General, Torus)
TEST_ALGO_TRAJ(General, Torus, 5, simul::CameraOrientation::Forward)

TEST(MSCKF, Torus)
TEST_ALGO_TRAJ(MSCKF, Torus, 5, simul::CameraOrientation::Forward)

TEST(OKVIS, Torus)
TEST_ALGO_TRAJ(OKVIS, Torus, 5, simul::CameraOrientation::Forward)

TEST(PAVIO, Torus)
TEST_ALGO_TRAJ(PAVIO, Torus, 5, simul::CameraOrientation::Forward)

TEST(TFVIO, Torus)
TEST_ALGO_TRAJ(TFVIO, Torus, 5, simul::CameraOrientation::Forward)

TEST(MSCKF, Squircle)
TEST_ALGO_TRAJ(MSCKF, Squircle, 5, simul::CameraOrientation::Forward)

TEST(OKVIS, Squircle)
TEST_ALGO_TRAJ(OKVIS, Squircle, 5, simul::CameraOrientation::Forward)

TEST(PAVIO, Squircle)
TEST_ALGO_TRAJ(PAVIO, Squircle, 5, simul::CameraOrientation::Forward)

TEST(TFVIO, Squircle)
TEST_ALGO_TRAJ(TFVIO, Squircle, 5, simul::CameraOrientation::Forward)

TEST(MSCKF, Circle)
TEST_ALGO_TRAJ(MSCKF, Circle, 5, simul::CameraOrientation::Forward)

TEST(OKVIS, Circle)
TEST_ALGO_TRAJ(OKVIS, Circle, 5, simul::CameraOrientation::Forward)

TEST(PAVIO, Circle)
TEST_ALGO_TRAJ(PAVIO, Circle, 5, simul::CameraOrientation::Forward)

TEST(TFVIO, Circle)
TEST_ALGO_TRAJ(TFVIO, Circle, 5, simul::CameraOrientation::Forward)

TEST(MSCKF, Dot)
TEST_ALGO_TRAJ(MSCKF, Dot, 5, simul::CameraOrientation::Forward)

TEST(TFVIO, Dot)
TEST_ALGO_TRAJ(TFVIO, Dot, 5, simul::CameraOrientation::Forward)

TEST(PAVIO, Dot)
TEST_ALGO_TRAJ(PAVIO, Dot, 5, simul::CameraOrientation::Forward)

TEST(OKVIS, Dot)
TEST_ALGO_TRAJ(OKVIS, Dot, 5, simul::CameraOrientation::Forward)
