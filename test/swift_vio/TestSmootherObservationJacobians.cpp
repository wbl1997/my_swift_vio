/**
 * @file   TestSmootherObservationJacobians
 * @author Jianzhu Huai
 * @date
 *
 * @brief  Test Jacobians of camera observations used by sliding window smoothers based on ceres solver.
 * These observations include ReprojectionErrorWithPap and ChordalDistance which
 *  uses PointSharedData and host and target frames.
 */

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "SmootherJacobianUtilities.h"
#include <swift_vio/ceres/ReprojectionErrorWithPap.hpp>

namespace {
void setupPoseOptProblem(bool perturbPose, bool rollingShutter,
                         bool noisyKeypoint, int cameraObservationModelId,
                         bool R_WCnmf = false) {
  // srand((unsigned int) time(0));
  swift_vio::CameraObservationOptions coo;
  coo.perturbPose = perturbPose;
  coo.rollingShutter = rollingShutter;
  coo.noisyKeypoint = noisyKeypoint;
  coo.cameraObservationModelId = cameraObservationModelId;

  swift_vio::CameraObservationJacobianTest jacTest(coo);

  okvis::ImuParameters imuParameters;
  double imuFreq = imuParameters.rate;
  Eigen::Vector3d ginw(0, 0, -imuParameters.g);
  okvis::Time startEpoch(1.0);
  okvis::Time endEpoch(3.0);
  std::shared_ptr<simul::CircularSinusoidalTrajectory> cameraMotion(
      new simul::RoundedSquare(imuFreq, ginw, okvis::Time(0, 0), 1.0, 6.0, 0.8));

  okvis::ImuMeasurementDeque imuMeasurements;
  cameraMotion->getTrueInertialMeasurements(startEpoch - okvis::Duration(1),
                                            endEpoch + okvis::Duration(1),
                                            imuMeasurements);
  jacTest.addNavStatesAndExtrinsic(cameraMotion, startEpoch, 1.0);

  double cameraTimeOffset(0.0);
  jacTest.addImuAugmentedParameterBlocks(startEpoch);
  jacTest.addImuInfo(imuMeasurements, imuParameters, cameraTimeOffset);

  std::shared_ptr<swift_vio::DistortedPinholeCameraGeometry> cameraGeometry =
      std::static_pointer_cast<swift_vio::DistortedPinholeCameraGeometry>(
          swift_vio::DistortedPinholeCameraGeometry::createTestObject());

  Eigen::VectorXd intrinsicParams;
  cameraGeometry->getIntrinsics(intrinsicParams);
  double tr = 0;
  if (jacTest.coo_.rollingShutter) {
    tr = 0.03;
  }
  cameraGeometry->setReadoutTime(tr);
  cameraGeometry->setImageDelay(cameraTimeOffset);
  jacTest.addCameraParameterBlocks(intrinsicParams, startEpoch, tr, cameraTimeOffset);

  const size_t numberTrials = 200;
  std::vector<std::shared_ptr<swift_vio::PointLandmark>> visibleLandmarks;
  Eigen::AlignedVector<Eigen::AlignedVector<Eigen::Vector2d>> pointObservationList;
  jacTest.createLandmarksAndObservations(cameraGeometry, &visibleLandmarks, &pointObservationList, numberTrials);

  int numberLandmarks = visibleLandmarks.size();
  std::vector<uint64_t> frameIds = jacTest.frameIds();
  std::vector<std::shared_ptr<okvis::ceres::PoseParameterBlock>> poseBlocks =
      jacTest.poseBlocks();
  LOG(INFO) << "Number landmarks " << numberLandmarks;

  double imageHeight = cameraGeometry->imageHeight();
  for (int i = 0; i < numberLandmarks; ++i) {
    jacTest.addLandmark(visibleLandmarks[i]);

    // create PointSharedData with IMU measurements
    std::shared_ptr<swift_vio::PointSharedData> pointDataPtr(
        new swift_vio::PointSharedData());
    for (int j = 0; j < 3; ++j) {
      double kpN = pointObservationList[i][j][1] / imageHeight - 0.5;
      okvis::KeypointIdentifier kpi(frameIds[j], 0, i);
      std::shared_ptr<const okvis::ceres::ParameterBlock> T_WBj_ptr(poseBlocks[j]);
      okvis::Time imageStamp = jacTest.getImageTimestamp(j, 0);
      pointDataPtr->addKeypointObservation(kpi, T_WBj_ptr, kpN, imageStamp);
    }

    jacTest.propagatePoseAndVelocityForMapPoint(pointDataPtr);

    std::vector<swift_vio::AnchorFrameIdentifier> anchorIds{{frameIds[0], 0, 0}, {frameIds[1], 0, 1}};
    pointDataPtr->setAnchors(anchorIds);

    bool useFirstEstimate = true;
    pointDataPtr->computePoseAndVelocityForJacobians(useFirstEstimate);
    pointDataPtr->computeSharedJacobians(cameraObservationModelId);

    // add landmark observations (residuals) to the problem.
    for (int observationIndex = 0; observationIndex < 3; ++observationIndex) {
      std::shared_ptr<::ceres::CostFunction> costFunctionPtr;
      std::shared_ptr<okvis::ceres::ErrorInterface> errorInterface;
      switch (cameraObservationModelId) {
        case swift_vio::cameras::kChordalDistanceId: {
          std::shared_ptr<okvis::ceres::ChordalDistance<swift_vio::DistortedPinholeCameraGeometry,
                                        swift_vio::ProjectionOptFXY_CXY,
                                        swift_vio::Extrinsic_p_CB>>
              localCostFunctionPtr(
                  new okvis::ceres::ChordalDistance<
                      swift_vio::DistortedPinholeCameraGeometry,
                      swift_vio::ProjectionOptFXY_CXY, swift_vio::Extrinsic_p_CB>(
                      cameraGeometry, pointObservationList[i][observationIndex],
                      swift_vio::kCovariance, observationIndex, pointDataPtr, R_WCnmf));
          costFunctionPtr = std::static_pointer_cast<::ceres::CostFunction>(localCostFunctionPtr);
          errorInterface = std::static_pointer_cast<okvis::ceres::ErrorInterface>(localCostFunctionPtr);
          break;
        }
        case swift_vio::cameras::kReprojectionErrorWithPapId: {
          std::shared_ptr<okvis::ceres::ReprojectionErrorWithPap<swift_vio::DistortedPinholeCameraGeometry,
                                                 swift_vio::ProjectionOptFXY_CXY,
                                                 swift_vio::Extrinsic_p_CB>>
              localCostFunctionPtr(
                  new okvis::ceres::ReprojectionErrorWithPap<
                      swift_vio::DistortedPinholeCameraGeometry,
                      swift_vio::ProjectionOptFXY_CXY, swift_vio::Extrinsic_p_CB>(
                      cameraGeometry, pointObservationList[i][observationIndex],
                      swift_vio::kCovariance, observationIndex, pointDataPtr));
          costFunctionPtr = std::static_pointer_cast<::ceres::CostFunction>(localCostFunctionPtr);
          errorInterface = std::static_pointer_cast<okvis::ceres::ErrorInterface>(localCostFunctionPtr);
          break;
        }
      }
      CHECK(costFunctionPtr) << "Null cost function not allowed!";
      if (observationIndex == 2) {
        jacTest.addResidual(costFunctionPtr, observationIndex, i);
      }
      if (i % 20 == 0) {
        jacTest.verifyJacobians(errorInterface, observationIndex, i, pointDataPtr, cameraGeometry,
                                pointObservationList[i][observationIndex]);
      }
    }
  }
  jacTest.solveAndCheck();
}
}  // namespace

TEST(CeresErrorTerms, ReprojectionErrorWithPapNoiseFree) {
   LOG(INFO)<< "1";
  setupPoseOptProblem(false, false, false,
                      swift_vio::cameras::kReprojectionErrorWithPapId);
}

TEST(CeresErrorTerms, ChordalDistanceNoiseFree) {
  LOG(INFO)<< "2";
  setupPoseOptProblem(false, false, false, swift_vio::cameras::kChordalDistanceId);
}

TEST(CeresErrorTerms, ChordalDistance_R_WC_NoiseFree) {
  setupPoseOptProblem(false, false, false, swift_vio::cameras::kChordalDistanceId,
                      true);
}

TEST(CeresErrorTerms, ReprojectionErrorWithPapNoisy) {
  LOG(INFO)<< "3";
  setupPoseOptProblem(true, true, true,
                      swift_vio::cameras::kReprojectionErrorWithPapId);
}

TEST(CeresErrorTerms, ChordalDistanceNoisy) {
  LOG(INFO)<< "4";
  setupPoseOptProblem(true, true, true,
                      swift_vio::cameras::kChordalDistanceId);
}
