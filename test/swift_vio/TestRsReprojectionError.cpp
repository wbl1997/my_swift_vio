#include <glog/logging.h>
#include <gtest/gtest.h>

#include <ceres/ceres.h>

#include <swift_vio/ceres/CameraTimeParamBlock.hpp>
#include <swift_vio/ceres/EuclideanParamBlock.hpp>
#include <swift_vio/ceres/EuclideanParamBlockSized.hpp>
#include <swift_vio/ExtrinsicModels.hpp>
#include <swift_vio/ProjParamOptModels.hpp>
#include <swift_vio/ceres/RsReprojectionError.hpp>
#include <swift_vio/ceres/RSCameraReprojectionError.hpp>

#include <okvis/assert_macros.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/ceres/HomogeneousPointError.hpp>
#include <okvis/ceres/HomogeneousPointLocalParameterization.hpp>
#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/ReprojectionError.hpp>
#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/Time.hpp>

#include <simul/curves.h>
#include <simul/numeric_ceres_residual_Jacobian.hpp>
#include <simul/PointLandmarkSimulationRS.hpp>
#include "SmootherJacobianUtilities.h"

// When readout time, tr is non zero, analytic, numeric and automatic Jacobians
// of the rolling shutter reprojection factor are roughly the same.
// Surprisingly, if tr is zero, automatic Jacobians of the rolling shutter
// reprojection factor relative to the time offset and readout time are zeros
// and disagree with the values supported by both numeric and analytic
// approaches. Other than that, the rest Jacobians by the three method when tr
// is zero are roughly the same.

void setupPoseOptProblem(bool perturbPose, bool rollingShutter,
                         bool noisyKeypoint, int cameraObservationModelId) {
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
  std::shared_ptr<simul::CircularSinusoidalTrajectory> cameraMotion(
      new simul::WavyCircle(imuFreq, ginw));
  okvis::ImuMeasurementDeque imuMeasurements;

  okvis::Time startEpoch(2.0);
  okvis::Time endEpoch(5.0);
  cameraMotion->getTrueInertialMeasurements(startEpoch - okvis::Duration(1),
                                            endEpoch + okvis::Duration(1),
                                            imuMeasurements);
  jacTest.addNavStatesAndExtrinsic(cameraMotion, startEpoch, 0.3);
  //jacTest.addNavStatesAndExtrinsich(cameraMotion, startEpoch, 0.3);

  double tdAtCreation(0.0);  // camera time offset used in initializing the state time.
  double initialCameraTimeOffset(0.0);  // camera time offset's initial estimate.
  double cameraTimeOffset(0.0);  // true camera time offset.
  jacTest.addImuAugmentedParameterBlocks(startEpoch);
  jacTest.addImuInfo(imuMeasurements, imuParameters, tdAtCreation);

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
  jacTest.addCameraParameterBlocks(intrinsicParams, startEpoch, tr, initialCameraTimeOffset);

  // get some random points
  const size_t numberTrials = 200;
  std::vector<std::shared_ptr<swift_vio::PointLandmark>> visibleLandmarks;
  Eigen::AlignedVector<Eigen::AlignedVector<Eigen::Vector2d>> pointObservationList;
  jacTest.createLandmarksAndObservations(cameraGeometry, &visibleLandmarks, &pointObservationList, numberTrials);

  std::cout << "created " << visibleLandmarks.size()
            << " visible points and add respective reprojection error terms... "
            << std::endl;


  for (size_t i = 0u; i < visibleLandmarks.size(); ++i) {
    jacTest.addLandmark(visibleLandmarks[i]);
    for (size_t j = 0; j < pointObservationList[i].size(); ++j) {
      std::shared_ptr<okvis::ImuMeasurementDeque> imuMeasDequePtr(
          new okvis::ImuMeasurementDeque(imuMeasurements));

      if (coo.cameraObservationModelId ==
          swift_vio::cameras::kRsReprojectionErrorId)
      {
        std::shared_ptr<::ceres::CostFunction> costFunctionPtr;
        std::shared_ptr<okvis::ceres::ErrorInterface> errorInterface;

        std::shared_ptr<okvis::ceres::RsReprojectionError<
            swift_vio::DistortedPinholeCameraGeometry,
            swift_vio::ProjectionOptFXY_CXY, swift_vio::Extrinsic_p_BC_q_BC>>
            localCostFunctionPtr(
                new okvis::ceres::RsReprojectionError<
                    swift_vio::DistortedPinholeCameraGeometry,
                    swift_vio::ProjectionOptFXY_CXY, swift_vio::Extrinsic_p_BC_q_BC>(
                    cameraGeometry, pointObservationList[i][j], swift_vio::kCovariance,
                    imuMeasDequePtr,
                    std::shared_ptr<const Eigen::Matrix<double, 6, 1>>(),
                    jacTest.stateEpoch(j), tdAtCreation, imuParameters.g));
        costFunctionPtr = std::static_pointer_cast<::ceres::CostFunction>(localCostFunctionPtr);
        errorInterface = std::static_pointer_cast<okvis::ceres::ErrorInterface>(localCostFunctionPtr);
        jacTest.addResidual(costFunctionPtr, j, i);

        std::shared_ptr<swift_vio::PointSharedData> pointDataPtr;
        if (i % 20 == 0 && j == 2)
        {
          jacTest.verifyJacobians(errorInterface, j, i, pointDataPtr,
                                  cameraGeometry, pointObservationList[i][j]);
        }
      }
      else if (coo.cameraObservationModelId ==
               swift_vio::cameras::kRSCameraReprojectionErrorId)
      {
        std::shared_ptr<::ceres::CostFunction> costFunctionPtr;
        std::shared_ptr<okvis::ceres::ErrorInterface> errorInterface;

        std::shared_ptr<okvis::ceres::RSCameraReprojectionError<
            swift_vio::DistortedPinholeCameraGeometry>>
            localCostFunctionPtr(
                new okvis::ceres::RSCameraReprojectionError<swift_vio::DistortedPinholeCameraGeometry>(
                    pointObservationList[i][j], swift_vio::kCovariance, cameraGeometry,
                    imuMeasDequePtr, imuParameters,
                    jacTest.stateEpoch(j),
                    //jacTest.stateEpoch(j)
                    okvis::Time( jacTest.stateEpoch(j) - okvis::Duration(tdAtCreation))
                    ));
        costFunctionPtr = std::static_pointer_cast<::ceres::CostFunction>(localCostFunctionPtr);
        errorInterface = std::static_pointer_cast<okvis::ceres::ErrorInterface>(localCostFunctionPtr);
        jacTest.addResidual(costFunctionPtr, j, i);

        std::shared_ptr<swift_vio::PointSharedData> pointDataPtr;
        if (i % 20 == 0 && j == 2)
        {
          jacTest.verifyJacobians(errorInterface, j, i, pointDataPtr, cameraGeometry, pointObservationList[i][j]);
        }
      }
    }
  }
  std::cout << "Successfully constructed ceres solver pose optimization problem." << std::endl;
  jacTest.solveAndCheck();
}


TEST(CeresErrorTerms, RsReprojectionErrorNoiseFree) {
  LOG(INFO)<< "1";
  setupPoseOptProblem(false, false, false, swift_vio::cameras::kRsReprojectionErrorId);
}

TEST(CeresErrorTerms, RsReprojectionErrorNoisy) {
  LOG(INFO)<< "2";
  setupPoseOptProblem(true, true, true, swift_vio::cameras::kRsReprojectionErrorId);
}


TEST(CeresErrorTerms, RSCameraReprojectionErrorNoiseFree) {
  LOG(INFO)<< "3";
  setupPoseOptProblem(false, false, false, swift_vio::cameras::kRSCameraReprojectionErrorId);
}

TEST(CeresErrorTerms, RSCameraReprojectionErrorNoisy) {
  LOG(INFO)<< "4";
  setupPoseOptProblem(true, true, true, swift_vio::cameras::kRSCameraReprojectionErrorId);
}/**/