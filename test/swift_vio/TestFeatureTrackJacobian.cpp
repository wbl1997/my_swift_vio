/**
 * @file   TestFeatureJacobian
 * @author Jianzhu Huai
 * @date
 *
 * @brief  Test Jacobians of feature tracks (a list of camera observations) used by filters.
 */

#include <gtest/gtest.h>

#include <gtsam/VioBackEndParams.h>
#include <vio/eigen_utils.h>

#include <io_wrap/StreamHelper.hpp>
#include <simul/VioSimTestSystem.hpp>


// TODO(jhuai): This test is broken.

namespace {
int examinLandmarkStackedJacobian(const okvis::MapPoint& mapPoint,
                                   std::shared_ptr<swift_vio::MSCKF> estimator) {
  swift_vio::PointLandmark pointLandmark;
  Eigen::MatrixXd H_oi[2];
  Eigen::Matrix<double, Eigen::Dynamic, 1> r_oi[2];
  Eigen::MatrixXd R_oi[2];
  bool jacOk1 =
      estimator->featureJacobian(mapPoint, &pointLandmark, H_oi[0], r_oi[0], R_oi[0], nullptr);
  bool jacOk2 =
      estimator->featureJacobianGeneric(mapPoint, &pointLandmark, H_oi[1], r_oi[1], R_oi[1]);
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

enum class EstimatorCheck {
  FEATURE_TRACK_JACOBIAN=0,
  NAVSTATE_COVARIANCE,
};

void checkMSE(const Eigen::VectorXd & /*mse*/,
              const Eigen::VectorXd & /*desiredStdevs*/,
              const std::vector<std::string> & /*dimensionLabels*/) {}

void checkNEES(const Eigen::Vector3d & /*nees*/) {}

}  // namespace

class EstimatorTest {
public:
  EstimatorTest(std::string _projOptModelName, std::string _extrinsicModelName,
                std::string _outputFile, double _timeOffset,
                double _readoutTime, int _cameraObservationModelId = 0,
                int _landmarkModelId = 0)
      : projOptModelName(_projOptModelName),
        extrinsicModelName(_extrinsicModelName),
        outputFile(_outputFile),
        timeOffset(_timeOffset),
        readoutTime(_readoutTime),
        cameraObservationModelId(_cameraObservationModelId),
        landmarkModelId(_landmarkModelId), simSystem(checkMSE, checkNEES) {}

  void SetUp() {
    bool addImuNoise = true;
    bool noisyInitialSpeedAndBiases = true;
    bool useEpipolarConstraint = false;
    double noise_factor = 1.0;
    simul::SimCameraModelType cameraModelId = simul::SimCameraModelType::EUROC;
    simul::CameraOrientation cameraOrientationId =
        simul::CameraOrientation::Forward;
    double landmarkRadius = 5;
    swift_vio::BackendParams backendParams;

    swift_vio::EstimatorAlgorithm algorithm = swift_vio::EstimatorAlgorithm::MSCKF;

    simul::SimImuParameters imuParams(
        "Torus", addImuNoise, noisyInitialSpeedAndBiases, false, true, 5e-3,
        2e-2, 5e-3, 1e-3, 5e-3, 1.2e-3, 2e-5, 8e-3, 5.5e-5, noise_factor,
        noise_factor);
    simul::SimVisionParameters visionParams(
        true, true, cameraModelId, cameraOrientationId,
        "RadialTangentialDistortion", "FXY_CXY", "P_CB", true, 0.02, 0.0,
        timeOffset, readoutTime, false, simul::LandmarkGridType::FourWalls,
        landmarkRadius);
    simul::SimEstimatorParameters estimatorParams(
        "MSCKF", algorithm, 1,
        cameraObservationModelId, landmarkModelId, useEpipolarConstraint);
    testSetting = simul::TestSetting(imuParams, visionParams,
                                   estimatorParams, backendParams, "");
  }

  void Run(EstimatorCheck checkCase) {
    // TODO(jhuai): loop through the optimization and callback the health check.
    int frameCount = 100;
    examineCase(checkCase, frameCount);
  }

  void examineCase(EstimatorCheck checkCase, int frameCount) {
    if (checkCase == EstimatorCheck::FEATURE_TRACK_JACOBIAN && frameCount == 100)
      examineJacobian();

    if (checkCase == EstimatorCheck::NAVSTATE_COVARIANCE && frameCount % 10 == 0) {
      examineCovariance(frameCount);
    }
  }

  /**
   * @brief examineJacobian Check if the Jacobian of a feature measurement or
   *  the Jacobian of a feature track are correct.
   * @warning Assume the MSCKF estimator is used.
   * Jacobian.
   */
  void examineJacobian() const {
    std::shared_ptr<swift_vio::MSCKF> filter =
        std::dynamic_pointer_cast<swift_vio::MSCKF>(estimator);
    {
      swift_vio::StatePointerAndEstimateList currentStates;
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

      int examined = examinLandmarkStackedJacobian(mapPoint, filter);
      featureJacobianLandmarkCount += examined;
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

  int cameraObservationModelId;
  int landmarkModelId;
  swift_vio::EstimatorAlgorithm algorithm = swift_vio::EstimatorAlgorithm::MSCKF;

private:
  // internal default system parameters
  simul::VioSimTestSystem simSystem;

  simul::TestSetting testSetting;

  std::ofstream debugStream; // record state history of a trial
  std::shared_ptr<okvis::Estimator> estimator;
  std::vector<okvis::Time> times;
  okvis::ImuMeasurementDeque imuMeasurements;
  okvis::ImuMeasurementDeque trueBiases;
  Eigen::AlignedVector<okvis::kinematics::Transformation> ref_T_WS_list;

  std::shared_ptr<simul::SimulationFrontend> frontend;
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

TEST(EstimatorTest, OkvisCovariance) {
  std::string projOptModelName = "FIXED";
  std::string extrinsicModelName = "FIXED";
  std::string outputFile = FLAGS_log_dir + "/OKVIS_Torus_Fixed.txt";
  EstimatorTest test(projOptModelName, extrinsicModelName, outputFile, 0.0, 0.0);
  test.algorithm = swift_vio::EstimatorAlgorithm::OKVIS;
  test.SetUp();
  test.Run(EstimatorCheck::NAVSTATE_COVARIANCE);
}

