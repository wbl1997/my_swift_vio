#include "SmootherJacobianUtilities.h"
#include <gtest/gtest.h>

#include <swift_vio/ceres/ReprojectionErrorWithPap.hpp>
#include <swift_vio/ceres/RsReprojectionError.hpp>
#include <swift_vio/ceres/RSCameraReprojectionError.hpp>

namespace swift_vio {
const double NumericJacobianPAP::h = 1e-5;

uint64_t CameraObservationJacobianTest::addNavStatesAndExtrinsic(
    std::shared_ptr<const simul::CircularSinusoidalTrajectory> cameraMotion,
    okvis::Time startEpoch, double timeGapBetweenStates) {
  okvis::kinematics::Transformation T_disturb;
  T_disturb.setRandom(1, 0.02);
  const size_t numFrames = 3;  // main host frame, associate host frame, and observing frame (aka. target frame).
  frameIds_.reserve(numFrames);
  stateEpochs_.reserve(numFrames);
  poseBlocks_.reserve(numFrames);
  speedAndBiasBlocks_.reserve(numFrames);
  for (size_t f = 0u; f < numFrames; ++f) {
    okvis::Time stateEpoch =
        startEpoch + okvis::Duration(f * timeGapBetweenStates, 0);
    okvis::kinematics::Transformation T_WB =
        cameraMotion->computeGlobalPose(stateEpoch);
    Eigen::Vector3d v_WB =
        cameraMotion->computeGlobalLinearVelocity(stateEpoch);
    okvis::SpeedAndBiases speedAndBias;
    speedAndBias.head<3>() = v_WB;
    speedAndBias.tail<6>().setZero();
    if (f == 2) { // target frame
      ref_T_WB_ = T_WB;
      if (coo_.perturbPose) {
        T_WB = T_WB * T_disturb;
      }
      initial_T_WB_ = T_WB;
    }
    frameIds_.push_back(nextBlockIndex_);
    stateEpochs_.push_back(stateEpoch);

    std::shared_ptr<okvis::ceres::PoseParameterBlock> poseParameterBlock(
        new okvis::ceres::PoseParameterBlock(T_WB, nextBlockIndex_++,
                                             stateEpoch));
    problem_->AddParameterBlock(poseParameterBlock->parameters(),
                                poseParameterBlock->dimension(),
                                poseLocalParameterization_.get());
    poseBlocks_.push_back(poseParameterBlock);

    std::shared_ptr<okvis::ceres::SpeedAndBiasParameterBlock> speedAndBiasBlock(
        new okvis::ceres::SpeedAndBiasParameterBlock(
            speedAndBias, nextBlockIndex_++, stateEpoch));
    problem_->AddParameterBlock(speedAndBiasBlock->parameters(),
                                speedAndBiasBlock->dimension());
    speedAndBiasBlocks_.push_back(speedAndBiasBlock);
    if (f == 2) {
      problem_->SetParameterBlockVariable(poseParameterBlock->parameters());
    } else {
      problem_->SetParameterBlockConstant(poseParameterBlock->parameters());
    }
    problem_->SetParameterBlockConstant(speedAndBiasBlock->parameters());
  }

  okvis::kinematics::Transformation T_BC(
      simul::create_T_BC(simul::CameraOrientation::Right, 0));
  std::shared_ptr<okvis::ceres::PoseParameterBlock> extrinsicsParameterBlock(
      new okvis::ceres::PoseParameterBlock(T_BC, nextBlockIndex_++,
                                           startEpoch));
  problem_->AddParameterBlock(extrinsicsParameterBlock->parameters(),
                              extrinsicsParameterBlock->dimension(),
                              extrinsicLocalParameterization_.get());
  extrinsicBlock_ = extrinsicsParameterBlock;
  problem_->SetParameterBlockConstant(extrinsicsParameterBlock->parameters());

 if (coo_.cameraObservationModelId ==
      swift_vio::cameras::kRSCameraReprojectionErrorId)
  {
      std::shared_ptr<okvis::ceres::PoseParameterBlock> extrinsicsParameterBlockh(
          new okvis::ceres::PoseParameterBlock(T_BC, nextBlockIndex_++,
                                               startEpoch));
      problem_->AddParameterBlock(extrinsicsParameterBlockh->parameters(),
                                  extrinsicsParameterBlockh->dimension(),
                                  extrinsicLocalParameterization_.get());
      extrinsicBlockh_ = extrinsicsParameterBlockh;
      problem_->SetParameterBlockConstant(extrinsicsParameterBlockh->parameters());
  }
return nextBlockIndex_;
}


uint64_t CameraObservationJacobianTest::addCameraParameterBlocks(
    const Eigen::VectorXd &intrinsicParams, okvis::Time startEpoch, double tr,
    double timeOffset) {
    if (coo_.cameraObservationModelId ==
        swift_vio::cameras::kRSCameraReprojectionErrorId)
    {
        int projOptModelId = swift_vio::ProjectionOptNameToId(coo_.projOptModelName);
        Eigen::VectorXd projIntrinsics;
        swift_vio::ProjectionOptGlobalToLocal(projOptModelId, intrinsicParams,
                                              &projIntrinsics);

        std::shared_ptr<okvis::ceres::EuclideanParamBlock> intrinsicParamBlock(
            new okvis::ceres::EuclideanParamBlock(intrinsicParams, nextBlockIndex_++,
                                                  startEpoch, kProjIntrinsicDim+kDistortionDim));

        std::shared_ptr<okvis::ceres::CameraTimeParamBlock> trParamBlock(
            new okvis::ceres::CameraTimeParamBlock(tr, nextBlockIndex_++,
                                                   startEpoch));
        std::shared_ptr<okvis::ceres::CameraTimeParamBlock> tdParamBlock(
            new okvis::ceres::CameraTimeParamBlock(timeOffset, nextBlockIndex_++,
                                                   startEpoch));


        problem_->AddParameterBlock(intrinsicParamBlock->parameters(),
                                    kProjIntrinsicDim+kDistortionDim);
        problem_->SetParameterBlockConstant(intrinsicParamBlock->parameters());
        
        problem_->AddParameterBlock(trParamBlock->parameters(), 1);
        problem_->SetParameterBlockConstant(trParamBlock->parameters());

        problem_->AddParameterBlock(tdParamBlock->parameters(),
                                    okvis::ceres::CameraTimeParamBlock::Dimension);
        problem_->SetParameterBlockConstant(tdParamBlock->parameters());

        cameraParameterBlocks_.push_back(intrinsicParamBlock);
        cameraParameterBlocks_.push_back(trParamBlock);
        cameraParameterBlocks_.push_back(tdParamBlock);

        //initialization Tg,Ts,Ta:They are not used for the time being, 
        //and the initialization method is not professional
        std::shared_ptr<okvis::ceres::EuclideanParamBlock> TgParameterBlocks(
            new okvis::ceres::EuclideanParamBlock(intrinsicParams, nextBlockIndex_++,
                                                  startEpoch, kProjIntrinsicDim + kDistortionDim));
        std::shared_ptr<okvis::ceres::EuclideanParamBlock> TsParameterBlocks(
            new okvis::ceres::EuclideanParamBlock(intrinsicParams, nextBlockIndex_++,
                                                  startEpoch, kProjIntrinsicDim + kDistortionDim));
        std::shared_ptr<okvis::ceres::EuclideanParamBlock> TaParameterBlocks(
            new okvis::ceres::EuclideanParamBlock(intrinsicParams, nextBlockIndex_++,
                                                  startEpoch, kProjIntrinsicDim + kDistortionDim));
        TgParameterBlocks_ = TgParameterBlocks;
        TsParameterBlocks_ = TsParameterBlocks;
        TaParameterBlocks_ = TaParameterBlocks;

        return nextBlockIndex_;
    }
    else
    {
        int projOptModelId = swift_vio::ProjectionOptNameToId(coo_.projOptModelName);
        Eigen::VectorXd projIntrinsics;
        swift_vio::ProjectionOptGlobalToLocal(projOptModelId, intrinsicParams,
                                              &projIntrinsics);

        Eigen::VectorXd distortion = intrinsicParams.tail(kDistortionDim);
        std::shared_ptr<okvis::ceres::EuclideanParamBlock> projectionParamBlock(
            new okvis::ceres::EuclideanParamBlock(projIntrinsics, nextBlockIndex_++,
                                                  startEpoch, kProjIntrinsicDim));
        std::shared_ptr<okvis::ceres::EuclideanParamBlock> distortionParamBlock(
            new okvis::ceres::EuclideanParamBlock(distortion, nextBlockIndex_++,
                                                  startEpoch, kDistortionDim));

        std::shared_ptr<okvis::ceres::CameraTimeParamBlock> trParamBlock(
            new okvis::ceres::CameraTimeParamBlock(tr, nextBlockIndex_++,
                                                   startEpoch));
        std::shared_ptr<okvis::ceres::CameraTimeParamBlock> tdParamBlock(
            new okvis::ceres::CameraTimeParamBlock(timeOffset, nextBlockIndex_++,
                                                   startEpoch));

        problem_->AddParameterBlock(projectionParamBlock->parameters(),
                                    kProjIntrinsicDim);
        problem_->SetParameterBlockConstant(projectionParamBlock->parameters());
        problem_->AddParameterBlock(distortionParamBlock->parameters(),
                                    kDistortionDim);
        problem_->SetParameterBlockConstant(distortionParamBlock->parameters());
        problem_->AddParameterBlock(trParamBlock->parameters(), 1);
        problem_->SetParameterBlockConstant(trParamBlock->parameters());

        problem_->AddParameterBlock(tdParamBlock->parameters(),
                                    okvis::ceres::CameraTimeParamBlock::Dimension);
        problem_->SetParameterBlockConstant(tdParamBlock->parameters());

        cameraParameterBlocks_.push_back(projectionParamBlock);
        cameraParameterBlocks_.push_back(distortionParamBlock);
        cameraParameterBlocks_.push_back(trParamBlock);
        cameraParameterBlocks_.push_back(tdParamBlock);

        return nextBlockIndex_;
    }
}

void CameraObservationJacobianTest::createLandmarksAndObservations(
    std::shared_ptr<const DistortedPinholeCameraGeometry> cameraGeometry,
    std::vector<std::shared_ptr<swift_vio::PointLandmark>> *visibleLandmarks,
    Eigen::AlignedVector<Eigen::AlignedVector<Eigen::Vector2d>>
        *pointObservationList,
    int numberTrials) {
  visibleLandmarks->reserve(numberTrials);
  pointObservationList->reserve(numberTrials);

  Eigen::AlignedVector<okvis::kinematics::Transformation> true_T_WB_list =
      truePoses();
  okvis::kinematics::Transformation T_WBm = true_T_WB_list[0];  // main host frame.
  okvis::kinematics::Transformation T_BC = extrinsicBlock_->estimate();  // camera extrinsics.

  for (int i = 1; i < numberTrials; ++i) {
    Eigen::Vector4d hpCm = cameraGeometry->createRandomVisibleHomogeneousPoint(
        double(i % 10) * 3 + 2.0);
    Eigen::Vector4d hpW = T_WBm * T_BC * hpCm;
    Eigen::AlignedVector<Eigen::Vector3d> observationsxy1;
    observationsxy1.reserve(3);
    Eigen::AlignedVector<Eigen::Vector2d> imageObservations;
    imageObservations.reserve(3);
    std::vector<size_t> anchorObsIndices{0, 1};  // main host frame and associate host frame.

    bool projectOk = true;
    for (int j = 0; j < 3; ++j) {
      okvis::kinematics::Transformation T_WBj = true_T_WB_list[j];
      Eigen::Vector4d hpCj = (T_WBj * T_BC).inverse() * hpW;
      Eigen::Vector2d imagePoint;
      okvis::cameras::CameraBase::ProjectionStatus status =
          cameraGeometry->projectHomogeneous(hpCj, &imagePoint);
      if (status != okvis::cameras::CameraBase::ProjectionStatus::Successful) {
        projectOk = false;
        break;
      }

      if (coo_.noisyKeypoint) {
        imagePoint += Eigen::Vector2d::Random();
      }

      Eigen::Vector3d xy1;
      bool backProjectOk = cameraGeometry->backProject(imagePoint, &xy1);
      if (!backProjectOk) {
        projectOk = false;
        break;
      }

      observationsxy1.push_back(xy1);
      imageObservations.push_back(imagePoint);
    }

    if (!projectOk) {
      continue;
    }

    swift_vio::TriangulationStatus status;
    std::shared_ptr<swift_vio::PointLandmark> pl;
    if (coo_.cameraObservationModelId ==
            swift_vio::cameras::kChordalDistanceId ||
        coo_.cameraObservationModelId ==
            swift_vio::cameras::kReprojectionErrorWithPapId) {
      pl.reset(new swift_vio::PointLandmark(
          swift_vio::ParallaxAngleParameterization::kModelId));
      Eigen::AlignedVector<okvis::kinematics::Transformation> T_BCs{T_BC};
      std::vector<size_t> camIndices(true_T_WB_list.size(), 0u);
      Eigen::AlignedVector<okvis::kinematics::Transformation> T_WCa_list{
          true_T_WB_list[anchorObsIndices[0]] * T_BC};
      status = pl->initialize(true_T_WB_list, observationsxy1, T_BCs,
                              T_WCa_list, camIndices, anchorObsIndices);
    } else if (coo_.cameraObservationModelId ==
               swift_vio::cameras::kRSCameraReprojectionErrorId) {
      status.triangulationOk = true;
      pl.reset(new swift_vio::PointLandmark(
          swift_vio::InverseDepthParameterization::kModelId));
      pl->setEstimate(hpCm);
    } else if (coo_.cameraObservationModelId ==
               swift_vio::cameras::kRsReprojectionErrorId) {
      pl.reset(new swift_vio::PointLandmark(
          okvis::ceres::HomogeneousPointLocalParameterization::kModelId));
      status.triangulationOk = true;
      pl->setEstimate(hpW);
    }

    if (!status.triangulationOk) {
      continue;
    }
    pointObservationList->push_back(imageObservations);
    visibleLandmarks->push_back(pl);
  }
}

void CameraObservationJacobianTest::addLandmark(
    std::shared_ptr<swift_vio::PointLandmark> pl) {
  if (pl->modelId() == swift_vio::ParallaxAngleParameterization::kModelId) {
    problem_->AddParameterBlock(
        pl->data(), swift_vio::ParallaxAngleParameterization::kGlobalDim);
    problem_->SetParameterBlockConstant(pl->data());
    problem_->SetParameterization(pl->data(), papLocalParameterization_.get());
  } else if (pl->modelId() ==
                 swift_vio::InverseDepthParameterization::kModelId) {
    problem_->AddParameterBlock(pl->data(), swift_vio::InverseDepthParameterization::kGlobalDim);
    problem_->SetParameterBlockConstant(pl->data());
    problem_->SetParameterization(pl->data(),
                                  inverseDepthLocalParameterization_.get());
  }else if (pl->modelId() ==
           okvis::ceres::HomogeneousPointLocalParameterization::kModelId)
  {
      problem_->AddParameterBlock(
          pl->data(), okvis::ceres::HomogeneousPointParameterBlock::Dimension);
      problem_->SetParameterBlockConstant(pl->data());
      problem_->SetParameterization(pl->data(),
                                    homogeneousPointLocalParameterization_.get());
  }
  visibleLandmarks_.push_back(pl);
}

void CameraObservationJacobianTest::addImuInfo(
    const okvis::ImuMeasurementDeque &entireImuList,
    const okvis::ImuParameters &imuParameters, double cameraTimeOffset) {
  imuParameters_ = imuParameters;
  positionAndVelocityLin_.reserve(3);
  imuWindowList_.reserve(3);
  tdAtCreationList_.reserve(3);
  for (int j = 0; j < 3; ++j) {
    tdAtCreationList_.push_back(cameraTimeOffset);
    std::shared_ptr<Eigen::Matrix<double, 6, 1>> positionAndVelocity(
        new Eigen::Matrix<double, 6, 1>());
    positionAndVelocity->head<3>() = poseBlocks_.at(j)->estimate().r();
    positionAndVelocity->tail<3>() =
        speedAndBiasBlocks_.at(j)->estimate().head<3>();
    positionAndVelocityLin_.push_back(positionAndVelocity);
    okvis::Time centerTime = stateEpochs_.at(j);
    okvis::Duration halfSide(0.5);
    std::shared_ptr<okvis::ImuMeasurementDeque> window(
        new okvis::ImuMeasurementDeque());
    *window = swift_vio::getImuMeasurements(
        centerTime - halfSide, centerTime + halfSide, entireImuList, nullptr);
    imuWindowList_.push_back(window);
  }
}

uint64_t CameraObservationJacobianTest::addImuAugmentedParameterBlocks(
    okvis::Time stateEpoch) {
  Eigen::Matrix<double, 9, 1> eye;
  eye << 1, 0, 0, 0, 1, 0, 0, 0, 1;
  Eigen::Matrix<double, 9, 1> zerovec = Eigen::Matrix<double, 9, 1>::Zero();
  std::shared_ptr<okvis::ceres::ShapeMatrixParamBlock> tgBlockPtr(
      new okvis::ceres::ShapeMatrixParamBlock(eye, nextBlockIndex_++,
                                              stateEpoch));
  problem_->AddParameterBlock(tgBlockPtr->parameters(), 9);
  problem_->SetParameterBlockConstant(tgBlockPtr->parameters());

  std::shared_ptr<okvis::ceres::ShapeMatrixParamBlock> tsBlockPtr(
      new okvis::ceres::ShapeMatrixParamBlock(zerovec, nextBlockIndex_++,
                                              stateEpoch));
  problem_->AddParameterBlock(tsBlockPtr->parameters(), 9);
  problem_->SetParameterBlockConstant(tsBlockPtr->parameters());

  std::shared_ptr<okvis::ceres::ParameterBlock> taBlockPtr;
  if (coo_.cameraObservationModelId == swift_vio::cameras::kRSCameraReprojectionErrorId) {
    Eigen::Matrix<double, 6, 1> scaleAndMisalignment;
    scaleAndMisalignment << 1, 1, 1, 0, 0, 0;
    taBlockPtr.reset(new okvis::ceres::EuclideanParamBlockSized<6>(
                       scaleAndMisalignment, nextBlockIndex_++, stateEpoch));
    problem_->AddParameterBlock(taBlockPtr->parameters(), 6);
  } else {
    taBlockPtr.reset(new okvis::ceres::ShapeMatrixParamBlock(eye, nextBlockIndex_++, stateEpoch));
    problem_->AddParameterBlock(taBlockPtr->parameters(), 9);
  }
  problem_->SetParameterBlockConstant(taBlockPtr->parameters());
  imuAugmentedBlocks_.push_back(tgBlockPtr);
  imuAugmentedBlocks_.push_back(tsBlockPtr);
  imuAugmentedBlocks_.push_back(taBlockPtr);
  return nextBlockIndex_;
}

void CameraObservationJacobianTest::propagatePoseAndVelocityForMapPoint(
    std::shared_ptr<swift_vio::PointSharedData> pointDataPtr) const {
  std::vector<std::pair<uint64_t, size_t>> frameIds = pointDataPtr->frameIds();
  CHECK(frameIds[0].first == frameIds_[0] && frameIds[2].first == frameIds_[2]);
  for (int observationIndex = 0; observationIndex < 3; ++observationIndex) {
    pointDataPtr->setImuInfo(observationIndex, stateEpochs_[observationIndex],
                             imuWindowList_[observationIndex],
                             positionAndVelocityLin_[observationIndex]);
    std::shared_ptr<const okvis::ceres::SpeedAndBiasParameterBlock>
        parameterBlockPtr(speedAndBiasBlocks_[observationIndex]);
    pointDataPtr->setVelocityParameterBlockPtr(observationIndex,
                                               parameterBlockPtr);
  }
  pointDataPtr->setCameraTimeParameterPtrs({cameraParameterBlocks_[3]},
                                           {cameraParameterBlocks_[2]});

  pointDataPtr->setImuAugmentedParameterPtrs(imuAugmentedBlocks_,
                                             &imuParameters_);
  pointDataPtr->computePoseAndVelocityAtObservation();
}

void CameraObservationJacobianTest::addResidual(
    std::shared_ptr<::ceres::CostFunction> costFunctionPtr,
    int observationIndex,  int landmarkIndex, int hostIndex) {
  if (coo_.cameraObservationModelId == swift_vio::cameras::kChordalDistanceId ||
      coo_.cameraObservationModelId ==
          swift_vio::cameras::kReprojectionErrorWithPapId) {
    switch (observationIndex) {
    case 0: // main
      problem_->AddResidualBlock(
          costFunctionPtr.get(), NULL, (double *)(nullptr),
          poseBlocks_[0]->parameters(), poseBlocks_[1]->parameters(),
          visibleLandmarks_[landmarkIndex]->data(),
          extrinsicBlock_->parameters(),
          cameraParameterBlocks_[0]->parameters(),
          cameraParameterBlocks_[1]->parameters(),
          cameraParameterBlocks_[2]->parameters(),
          cameraParameterBlocks_[3]->parameters(), (double *)(nullptr),
          speedAndBiasBlocks_[0]->parameters(),
          speedAndBiasBlocks_[1]->parameters());
      break;
    case 1: // associate
      problem_->AddResidualBlock(
          costFunctionPtr.get(), NULL, (double *)(nullptr),
          poseBlocks_[0]->parameters(), poseBlocks_[1]->parameters(),
          visibleLandmarks_[landmarkIndex]->data(),
          extrinsicBlock_->parameters(),
          cameraParameterBlocks_[0]->parameters(),
          cameraParameterBlocks_[1]->parameters(),
          cameraParameterBlocks_[2]->parameters(),
          cameraParameterBlocks_[3]->parameters(), (double *)(nullptr),
          speedAndBiasBlocks_[0]->parameters(),
          speedAndBiasBlocks_[1]->parameters());
      break;
    default:
      const double *const parameters[] = {
          poseBlocks_[observationIndex]->parameters(),
          poseBlocks_[0]->parameters(),
          poseBlocks_[1]->parameters(),
          visibleLandmarks_[landmarkIndex]->data(),
          extrinsicBlock_->parameters(),
          cameraParameterBlocks_[0]->parameters(),
          cameraParameterBlocks_[1]->parameters(),
          cameraParameterBlocks_[2]->parameters(),
          cameraParameterBlocks_[3]->parameters(),
          speedAndBiasBlocks_[observationIndex]->parameters(),
          speedAndBiasBlocks_[0]->parameters(),
          speedAndBiasBlocks_[1]->parameters()};
      Eigen::Vector3d residual;
      costFunctionPtr->Evaluate(parameters, residual.data(), nullptr);

      problem_->AddResidualBlock(
          costFunctionPtr.get(), NULL,
          poseBlocks_[observationIndex]->parameters(),
          poseBlocks_[0]->parameters(), poseBlocks_[1]->parameters(),
          visibleLandmarks_[landmarkIndex]->data(),
          extrinsicBlock_->parameters(),
          cameraParameterBlocks_[0]->parameters(),
          cameraParameterBlocks_[1]->parameters(),
          cameraParameterBlocks_[2]->parameters(),
          cameraParameterBlocks_[3]->parameters(),
          speedAndBiasBlocks_[observationIndex]->parameters(),
          speedAndBiasBlocks_[0]->parameters(),
          speedAndBiasBlocks_[1]->parameters());
      break;
    }
  } else if (coo_.cameraObservationModelId ==
             swift_vio::cameras::kRsReprojectionErrorId) {
    problem_->AddResidualBlock(
        costFunctionPtr.get(), NULL, poseBlocks_[observationIndex]->parameters(),
        visibleLandmarks_[landmarkIndex]->data(), extrinsicBlock_->parameters(), 
        cameraParameterBlocks_[0]->parameters(),
        cameraParameterBlocks_[1]->parameters(),
        cameraParameterBlocks_[2]->parameters(),
        cameraParameterBlocks_[3]->parameters(),
        speedAndBiasBlocks_[observationIndex]->parameters());
  } else if (coo_.cameraObservationModelId ==
             swift_vio::cameras::kRSCameraReprojectionErrorId) {
    // TODO(jhuai): Binliang
    if (observationIndex>hostIndex)
    {
        problem_->AddResidualBlock(
            costFunctionPtr.get(), NULL,
            poseBlocks_[observationIndex]->parameters(),
            visibleLandmarks_[landmarkIndex]->data(),
            poseBlocks_[hostIndex]->parameters(),
            extrinsicBlock_->parameters(),
            extrinsicBlockh_->parameters(),
            cameraParameterBlocks_[0]->parameters(),
            cameraParameterBlocks_[1]->parameters(),
            cameraParameterBlocks_[2]->parameters(), 
            speedAndBiasBlocks_[observationIndex]->parameters(),
            TgParameterBlocks_->parameters(),
            TsParameterBlocks_->parameters(),
            TaParameterBlocks_->parameters() 
            );
    }
  }
  costFunctions_.push_back(costFunctionPtr);  // remember in order to avert premature cost function destruction.
}

void CameraObservationJacobianTest::verifyJacobians(
    std::shared_ptr<const okvis::ceres::ErrorInterface> costFunctionPtr,
    int observationIndex, int landmarkIndex,
    std::shared_ptr<swift_vio::PointSharedData> pointDataPtr,
    std::shared_ptr<DistortedPinholeCameraGeometry> cameraGeometry,
    const Eigen::Vector2d &imagePoint, int hostIndex) const {
  if (coo_.cameraObservationModelId == swift_vio::cameras::kChordalDistanceId ||
      coo_.cameraObservationModelId == swift_vio::cameras::kReprojectionErrorWithPapId) {
    verifyJacobiansPAP(costFunctionPtr, observationIndex, landmarkIndex,
                       pointDataPtr, cameraGeometry, imagePoint);
  } else if (coo_.cameraObservationModelId ==
             swift_vio::cameras::kRsReprojectionErrorId) {
    verifyJacobiansHPP(costFunctionPtr, observationIndex, landmarkIndex,
                       pointDataPtr, cameraGeometry, imagePoint);
  }
  else if (coo_.cameraObservationModelId ==
           swift_vio::cameras::kRSCameraReprojectionErrorId)
  {
      if (observationIndex>hostIndex)
      {
          verifyJacobiansAIDP(costFunctionPtr, observationIndex, landmarkIndex,
                              pointDataPtr, cameraGeometry, imagePoint, hostIndex);
      }
  }
}

void CameraObservationJacobianTest::verifyJacobiansPAP(
    std::shared_ptr<const okvis::ceres::ErrorInterface> costFunctionPtr,
    int observationIndex, int landmarkIndex,
    std::shared_ptr<swift_vio::PointSharedData> pointDataPtr,
    std::shared_ptr<DistortedPinholeCameraGeometry> cameraGeometry,
    const Eigen::Vector2d &imagePoint) const {
  // compare numerical and analytic Jacobians and residuals
  std::cout << "Main anchor 0 associate anchor 1 observationIndex "
            << observationIndex << " landmark index " << landmarkIndex
            << std::endl;
  const double *const parameters[] = {
      poseBlocks_[observationIndex]->parameters(),
      poseBlocks_[0]->parameters(),
      poseBlocks_[1]->parameters(),
      visibleLandmarks_[landmarkIndex]->data(),
      extrinsicBlock_->parameters(),
      cameraParameterBlocks_[0]->parameters(),
      cameraParameterBlocks_[1]->parameters(),
      cameraParameterBlocks_[2]->parameters(),
      cameraParameterBlocks_[3]->parameters(),
      speedAndBiasBlocks_[observationIndex]->parameters(),
      speedAndBiasBlocks_[0]->parameters(),
      speedAndBiasBlocks_[1]->parameters()};

  const int krd = swift_vio::cameras::CameraObservationModelResidualDim(
      coo_.cameraObservationModelId);
  Eigen::VectorXd residuals(krd);
  Eigen::AlignedVector<Eigen::Matrix<double, -1, 7, Eigen::RowMajor>>
      de_deltaTWB(3, Eigen::Matrix<double, -1, 7, Eigen::RowMajor>(krd, 7));
  Eigen::AlignedVector<Eigen::Matrix<double, -1, 9, Eigen::RowMajor>>
      de_dSpeedAndBias(3,
                       Eigen::Matrix<double, -1, 9, Eigen::RowMajor>(krd, 9));
  Eigen::AlignedVector<Eigen::Matrix<double, -1, 6, Eigen::RowMajor>>
      de_deltaTWB_minimal(
          3, Eigen::Matrix<double, -1, 6, Eigen::RowMajor>(krd, 6));
  Eigen::AlignedVector<Eigen::Matrix<double, -1, 9, Eigen::RowMajor>>
      de_dSpeedAndBias_minimal(
          3, Eigen::Matrix<double, -1, 9, Eigen::RowMajor>(krd, 9));

  Eigen::Matrix<double, -1, 6, Eigen::RowMajor> de_dPoint(krd, 6);
  Eigen::Matrix<double, -1, 3, Eigen::RowMajor> de_dPoint_minimal(krd, 3);
  Eigen::Matrix<double, -1, 7, Eigen::RowMajor> de_dExtrinsic(krd, 7);
  Eigen::Matrix<double, -1, kExtrinsicMinimalDim, Eigen::RowMajor>
      de_dExtrinsic_minimal(krd, kExtrinsicMinimalDim);

  Eigen::Matrix<double, -1, kProjIntrinsicDim, Eigen::RowMajor>
      de_dProjectionIntrinsic(krd, kProjIntrinsicDim);
  Eigen::Matrix<double, -1, kDistortionDim, Eigen::RowMajor> de_dDistortion(
      krd, kDistortionDim);
  Eigen::Matrix<double, -1, 1> de_dtr(krd, 1);

  Eigen::Matrix<double, -1, kProjIntrinsicDim, Eigen::RowMajor>
      de_dProjectionIntrinsic_minimal(krd, kProjIntrinsicDim);
  Eigen::Matrix<double, -1, kDistortionDim, Eigen::RowMajor>
      de_dDistortion_minimal(krd, kDistortionDim);
  Eigen::Matrix<double, -1, 1> de_dtr_minimal(krd, 1);

  Eigen::Matrix<double, -1, 1> de_dtd(krd, 1);
  Eigen::Matrix<double, -1, 1> de_dtd_minimal(krd, 1);

  double *jacobians[] = {de_deltaTWB[0].data(),
                         de_deltaTWB[1].data(),
                         de_deltaTWB[2].data(),
                         de_dPoint.data(),
                         de_dExtrinsic.data(),
                         de_dProjectionIntrinsic.data(),
                         de_dDistortion.data(),
                         de_dtr.data(),
                         de_dtd.data(),
                         de_dSpeedAndBias[0].data(),
                         de_dSpeedAndBias[1].data(),
                         de_dSpeedAndBias[2].data()};

  double *jacobiansMinimal[] = {de_deltaTWB_minimal[0].data(),
                                de_deltaTWB_minimal[1].data(),
                                de_deltaTWB_minimal[2].data(),
                                de_dPoint_minimal.data(),
                                de_dExtrinsic_minimal.data(),
                                de_dProjectionIntrinsic_minimal.data(),
                                de_dDistortion_minimal.data(),
                                de_dtr_minimal.data(),
                                de_dtd_minimal.data(),
                                de_dSpeedAndBias_minimal[0].data(),
                                de_dSpeedAndBias_minimal[1].data(),
                                de_dSpeedAndBias_minimal[2].data()};

  costFunctionPtr->EvaluateWithMinimalJacobians(parameters, residuals.data(),
                                                jacobians, jacobiansMinimal);
  if (isZeroResidualExpected()) {
    EXPECT_LT(residuals.lpNorm<Eigen::Infinity>(), 1e-6)
        << "Without noise, residual should be fairly close to zero!";
  }

  double tol = 1e-8;
  // analytic full vs minimal
  ARE_MATRICES_CLOSE(de_dProjectionIntrinsic, de_dProjectionIntrinsic_minimal,
                     tol);
  ARE_MATRICES_CLOSE(de_dDistortion, de_dDistortion_minimal, tol);
  ARE_MATRICES_CLOSE(de_dtr, de_dtr_minimal, tol);
  ARE_MATRICES_CLOSE(de_dtd, de_dtd_minimal, tol);
  for (int s = 0; s < 3; ++s) {
    ARE_MATRICES_CLOSE(de_dSpeedAndBias[s], de_dSpeedAndBias_minimal[s], tol);
  }

  tol = 1e-3;
  Eigen::Matrix<double, -1, 3> de_dSpeed =
      de_dSpeedAndBias[0].topLeftCorner(krd, 3);
  Eigen::Matrix<double, -1, 3> de_dGyroBias =
      de_dSpeedAndBias[0].block(0, 3, krd, 3);
  Eigen::Matrix<double, -1, 3> de_dAccelBias =
      de_dSpeedAndBias[0].topRightCorner(krd, 3);

  // compute the numeric diff and check
  Eigen::AlignedVector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      de_deltaTWB_numeric(3,
                          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>(krd, 7));
  Eigen::AlignedVector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      de_deltaTWB_minimal_numeric(
          3, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                           Eigen::RowMajor>(krd, 6));
  Eigen::AlignedVector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      de_dSpeedAndBias_numeric(
          3, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                           Eigen::RowMajor>(krd, 9));

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      de_dPoint_numeric(krd, 6);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      de_dPoint_minimal_numeric(krd, 3);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      de_dExtrinsic_numeric(krd, 7);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      de_dExtrinsic_minimal_numeric(krd, kExtrinsicMinimalDim);

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      de_dProjectionIntrinsic_numeric(krd, kProjIntrinsicDim);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      de_dDistortion_numeric(krd, kDistortionDim);
  Eigen::Matrix<double, -1, 1> de_dtr_numeric(krd, 1);
  Eigen::Matrix<double, -1, 1> de_dtd_numeric(krd, 1);

  // compute numeric Jacobians
  NumericJacobianPAP nj(costFunctionPtr, poseBlocks_,
                     visibleLandmarks_[landmarkIndex], extrinsicBlock_,
                     cameraParameterBlocks_, speedAndBiasBlocks_,
                     positionAndVelocityLin_, pointDataPtr, cameraGeometry,
                     imagePoint, observationIndex, krd);

  // main and associate frame
  for (int j = 1; j < 3; ++j) {
    nj.computeNumericJacobianForPose(j - 1, &de_deltaTWB_numeric[j],
                                     &de_deltaTWB_minimal_numeric[j]);
    nj.computeNumericJacobianForSpeedAndBias(j - 1,
                                             &de_dSpeedAndBias_numeric[j]);
  }
  // observing frame
  nj.computeNumericJacobianForPose(observationIndex, &de_deltaTWB_numeric[0],
                                   &de_deltaTWB_minimal_numeric[0]);
  nj.computeNumericJacobianForSpeedAndBias(observationIndex,
                                           &de_dSpeedAndBias_numeric[0]);

  nj.computeNumericJacobianForPoint(&de_dPoint_numeric,
                                    &de_dPoint_minimal_numeric);
  nj.computeNumericJacobianForExtrinsic(&de_dExtrinsic_numeric,
                                        &de_dExtrinsic_minimal_numeric);
  nj.computeNumericJacobianForProjectionIntrinsic(
      &de_dProjectionIntrinsic_numeric);
  nj.computeNumericJacobianForDistortion(&de_dDistortion_numeric);
  nj.computeNumericJacobianForReadoutTime(&de_dtr_numeric);
  nj.computeNumericJacobianForCameraDelay(&de_dtd_numeric);
  double poseTol = 5e-3;
  ARE_MATRICES_CLOSE(de_deltaTWB_numeric[0], de_deltaTWB[0], poseTol);
  ARE_MATRICES_CLOSE(de_deltaTWB_numeric[1], de_deltaTWB[1], 1e-2);
  ARE_MATRICES_CLOSE(de_deltaTWB_numeric[2].leftCols<3>(),
                     de_deltaTWB[2].leftCols<3>(), poseTol);
  if (observationIndex != 1) {
    EXPECT_LT(de_deltaTWB_numeric[2].rightCols<4>().lpNorm<Eigen::Infinity>(),
              5e-2);
  }
  ARE_MATRICES_CLOSE(de_deltaTWB_minimal_numeric[0], de_deltaTWB_minimal[0],
                     1e-2);
  ARE_MATRICES_CLOSE(de_deltaTWB_minimal_numeric[1], de_deltaTWB_minimal[1],
                     5e-3);
  ARE_MATRICES_CLOSE(de_deltaTWB_minimal_numeric[2].leftCols<3>(),
                     de_deltaTWB_minimal[2].leftCols<3>(), 5e-3);
  if (observationIndex != 1) {
    EXPECT_LT(
        de_deltaTWB_minimal_numeric[2].rightCols<3>().lpNorm<Eigen::Infinity>(),
        5e-2);
  }
  ARE_MATRICES_CLOSE(de_dPoint_numeric, de_dPoint, tol);
  ARE_MATRICES_CLOSE(de_dPoint_minimal_numeric, de_dPoint_minimal, tol);
  ARE_MATRICES_CLOSE(de_dExtrinsic_numeric, de_dExtrinsic, tol);
  ARE_MATRICES_CLOSE(de_dExtrinsic_minimal_numeric, de_dExtrinsic_minimal,
                     5e-3);

  ARE_MATRICES_CLOSE(de_dProjectionIntrinsic_numeric, de_dProjectionIntrinsic,
                     1e-3);
  ARE_MATRICES_CLOSE(de_dDistortion_numeric, de_dDistortion, 1e-3);
  ARE_MATRICES_CLOSE(de_dtr_numeric, de_dtr, 5e-2);
  ARE_MATRICES_CLOSE(de_dtd_numeric, de_dtd, 1e-3);
  for (int j = 0; j < 3; ++j) {
    Eigen::Matrix<double, -1, 3> de_dSpeed_numeric =
        de_dSpeedAndBias_numeric[0].topLeftCorner(krd, 3);
    Eigen::Matrix<double, -1, 3> de_dGyroBias_numeric =
        de_dSpeedAndBias_numeric[0].block(0, 3, krd, 3);
    Eigen::Matrix<double, -1, 3> de_dAccelBias_numeric =
        de_dSpeedAndBias_numeric[0].topRightCorner(krd, 3);

    ARE_MATRICES_CLOSE(de_dSpeed_numeric, de_dSpeed, tol);
    EXPECT_LT(de_dGyroBias.lpNorm<Eigen::Infinity>(), 1e-9);
    EXPECT_LT((de_dGyroBias_numeric).lpNorm<Eigen::Infinity>(), 3)
        << "de_dGyroBias_numeric\n"
        << de_dGyroBias_numeric;
    EXPECT_LT((de_dAccelBias_numeric - de_dAccelBias).lpNorm<Eigen::Infinity>(),
              5e-3)
        << "de_dAccelBias_numeric\n"
        << de_dAccelBias_numeric << "\nde_dAccelBias\n"
        << de_dAccelBias;
  }
}

void CameraObservationJacobianTest::verifyJacobiansHPP(
    std::shared_ptr<const okvis::ceres::ErrorInterface> errorPtr,
    int observationIndex, int landmarkIndex,
    std::shared_ptr<swift_vio::PointSharedData> /*pointDataPtr*/,
    std::shared_ptr<DistortedPinholeCameraGeometry> /*cameraGeometry*/,
    const Eigen::Vector2d &/*imagePoint*/) const {
  // compare Jacobians obtained by analytic diff and auto diff.
  double const *const parameters[] = {
      poseBlocks_[observationIndex]->parameters(),
      visibleLandmarks_[landmarkIndex]->data(),
      extrinsicBlock_->parameters(),
      cameraParameterBlocks_[0]->parameters(),
      cameraParameterBlocks_[1]->parameters(),
      cameraParameterBlocks_[2]->parameters(),
      cameraParameterBlocks_[3]->parameters(),
      speedAndBiasBlocks_[observationIndex]->parameters()};
  Eigen::Vector2d residuals;
  Eigen::Vector2d residuals_auto;

  Eigen::Matrix<double, 2, 7, Eigen::RowMajor> duv_deltaTWS_auto;
  Eigen::Matrix<double, 2, 6, Eigen::RowMajor> duv_deltaTWS_minimal_auto;
  Eigen::Matrix<double, 2, 4, Eigen::RowMajor> duv_deltahpW_auto;
  Eigen::Matrix<double, 2, 3, Eigen::RowMajor> duv_deltahpW_minimal_auto;
  Eigen::Matrix<double, 2, 7, Eigen::RowMajor> duv_deltaTSC_auto;
  Eigen::Matrix<double, 2, 6, Eigen::RowMajor> duv_deltaTSC_minimal_auto;

  Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::RowMajor>
      duv_proj_intrinsic_auto(2, kProjIntrinsicDim);
  Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::RowMajor> duv_distortion_auto(
      2, kDistortionDim);
  Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::RowMajor> duv_tr_auto(2, 1);
  Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::RowMajor>
      duv_proj_intrinsic_minimal_auto(2, kProjIntrinsicDim);
  Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::RowMajor>
      duv_distortion_minimal_auto(2, kDistortionDim);
  Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::RowMajor> duv_tr_minimal_auto(
      2, 1);

  Eigen::Matrix<double, 2, 1> duv_td_auto;
  Eigen::Matrix<double, 2, 1> duv_td_minimal_auto;
  Eigen::Matrix<double, 2, 9, Eigen::RowMajor> duv_sb_auto;
  Eigen::Matrix<double, 2, 9, Eigen::RowMajor> duv_sb_minimal_auto;
  double *jacobiansAD[] = {
      duv_deltaTWS_auto.data(),   duv_deltahpW_auto.data(),
      duv_deltaTSC_auto.data(),   duv_proj_intrinsic_auto.data(),
      duv_distortion_auto.data(), duv_tr_auto.data(),
      duv_td_auto.data(),         duv_sb_auto.data()};
  double *jacobiansMinimalAD[] = {duv_deltaTWS_minimal_auto.data(),
                                  duv_deltahpW_minimal_auto.data(),
                                  duv_deltaTSC_minimal_auto.data(),
                                  duv_proj_intrinsic_minimal_auto.data(),
                                  duv_distortion_minimal_auto.data(),
                                  duv_tr_minimal_auto.data(),
                                  duv_td_minimal_auto.data(),
                                  duv_sb_minimal_auto.data()};

  Eigen::Matrix<double, 2, 7, Eigen::RowMajor> duv_deltaTWS;
  Eigen::Matrix<double, 2, 6, Eigen::RowMajor> duv_deltaTWS_minimal;
  Eigen::Matrix<double, 2, 4, Eigen::RowMajor> duv_deltahpW;
  Eigen::Matrix<double, 2, 3, Eigen::RowMajor> duv_deltahpW_minimal;
  Eigen::Matrix<double, 2, 7, Eigen::RowMajor> duv_deltaTSC;
  Eigen::Matrix<double, 2, 6, Eigen::RowMajor> duv_deltaTSC_minimal;

  Eigen::Matrix<double, 2, kProjIntrinsicDim, Eigen::RowMajor>
      duv_proj_intrinsic;
  Eigen::Matrix<double, 2, kDistortionDim, Eigen::RowMajor> duv_distortion;
  Eigen::Matrix<double, 2, 1> duv_tr;

  Eigen::Matrix<double, 2, kProjIntrinsicDim, Eigen::RowMajor>
      duv_proj_intrinsic_minimal;
  Eigen::Matrix<double, 2, kDistortionDim, Eigen::RowMajor>
      duv_distortion_minimal;
  Eigen::Matrix<double, 2, 1> duv_tr_minimal;

  Eigen::Matrix<double, 2, 1> duv_td;
  Eigen::Matrix<double, 2, 1> duv_td_minimal;
  Eigen::Matrix<double, 2, 9, Eigen::RowMajor> duv_sb;
  Eigen::Matrix<double, 2, 9, Eigen::RowMajor> duv_sb_minimal;

  double *jacobians[] = {duv_deltaTWS.data(),   duv_deltahpW.data(),
                         duv_deltaTSC.data(),   duv_proj_intrinsic.data(),
                         duv_distortion.data(), duv_tr.data(),
                         duv_td.data(),         duv_sb.data()};
  double *jacobiansMinimal[] = {
      duv_deltaTWS_minimal.data(),   duv_deltahpW_minimal.data(),
      duv_deltaTSC_minimal.data(),   duv_proj_intrinsic_minimal.data(),
      duv_distortion_minimal.data(), duv_tr_minimal.data(),
      duv_td_minimal.data(),         duv_sb_minimal.data()};

  std::shared_ptr<const okvis::ceres::RsReprojectionError<
      DistortedPinholeCameraGeometry, swift_vio::ProjectionOptFXY_CXY,
      swift_vio::Extrinsic_p_BC_q_BC>> costFuncPtr =
      std::static_pointer_cast<const okvis::ceres::RsReprojectionError<
          DistortedPinholeCameraGeometry, swift_vio::ProjectionOptFXY_CXY,
          swift_vio::Extrinsic_p_BC_q_BC>>(errorPtr);  

  costFuncPtr->EvaluateWithMinimalJacobians(parameters, residuals.data(),
                                            jacobians, jacobiansMinimal);
  costFuncPtr->EvaluateWithMinimalJacobiansAutoDiff(
      parameters, residuals_auto.data(), jacobiansAD, jacobiansMinimalAD);

  if (isZeroResidualExpected()) {
    EXPECT_TRUE(residuals.isMuchSmallerThan(1, 1e-8))
        << "Unusually large residual " << residuals.transpose();
    EXPECT_TRUE(residuals_auto.isMuchSmallerThan(1, 1e-8))
        << "Unusually large residual " << residuals.transpose();
  }
  double tol = 1e-8;
  // analytic full vs minimal
  ARE_MATRICES_CLOSE(duv_proj_intrinsic, duv_proj_intrinsic_minimal, tol);
  ARE_MATRICES_CLOSE(duv_distortion, duv_distortion_minimal, tol);
  ARE_MATRICES_CLOSE(duv_tr, duv_tr_minimal, tol);
  ARE_MATRICES_CLOSE(duv_td, duv_td_minimal, tol);
  ARE_MATRICES_CLOSE(duv_sb, duv_sb_minimal, tol);

  // automatic vs analytic
  tol = 1e-3;
  ARE_MATRICES_CLOSE(residuals_auto, residuals, tol);
  ARE_MATRICES_CLOSE(duv_deltaTWS_auto, duv_deltaTWS, tol);
  ARE_MATRICES_CLOSE(duv_deltahpW_auto, duv_deltahpW, tol);
  ARE_MATRICES_CLOSE(duv_deltaTSC_auto, duv_deltaTSC, tol);
  ARE_MATRICES_CLOSE(duv_proj_intrinsic_auto, duv_proj_intrinsic, 4e-3);
  ARE_MATRICES_CLOSE(duv_distortion_auto, duv_distortion, 4e-3);

  if (coo_.rollingShutter) {
    ARE_MATRICES_CLOSE(duv_tr_auto, duv_tr, 1e-1);
    ARE_MATRICES_CLOSE(duv_td_auto, duv_td, 1e-1);
  }

  Eigen::Matrix<double, 2, 3> duv_ds_auto = duv_sb_auto.topLeftCorner<2, 3>();
  Eigen::Matrix<double, 2, 3> duv_ds = duv_sb.topLeftCorner<2, 3>();
  ARE_MATRICES_CLOSE(duv_ds_auto, duv_ds, tol);
  Eigen::Matrix<double, 2, 3> duv_dbg_auto = duv_sb_auto.block<2, 3>(0, 3);
  Eigen::Matrix<double, 2, 3> duv_dbg = duv_sb.block<2, 3>(0, 3);
  EXPECT_LT((duv_dbg_auto - duv_dbg).lpNorm<Eigen::Infinity>(), 5e-2)
      << "duv_dbg_auto:\n"
      << duv_dbg_auto << "\nduv_dbg\n"
      << duv_dbg;
  Eigen::Matrix<double, 2, 3> duv_dba_auto = duv_sb_auto.topRightCorner<2, 3>();
  Eigen::Matrix<double, 2, 3> duv_dba = duv_sb.topRightCorner<2, 3>();
  EXPECT_LT((duv_dba_auto - duv_dba).lpNorm<Eigen::Infinity>(), 5e-3)
      << "duv_dba_auto\n"
      << duv_dba_auto << "\nduv_dba\n"
      << duv_dba;

  ARE_MATRICES_CLOSE(duv_deltaTWS_minimal_auto, duv_deltaTWS_minimal, 5e-3);
  ARE_MATRICES_CLOSE(duv_deltahpW_minimal_auto, duv_deltahpW_minimal, tol);
  ARE_MATRICES_CLOSE(duv_deltaTSC_minimal_auto, duv_deltaTSC_minimal, tol);

  ARE_MATRICES_CLOSE(duv_proj_intrinsic_minimal_auto,
                     duv_proj_intrinsic_minimal, 4e-3);
  ARE_MATRICES_CLOSE(duv_distortion_minimal_auto, duv_distortion_minimal, 4e-3);
  if (coo_.rollingShutter) {
    ARE_MATRICES_CLOSE(duv_tr_minimal_auto, duv_tr_minimal, 1e-1);
    ARE_MATRICES_CLOSE(duv_td_minimal_auto, duv_td_minimal, 1e-1);
  }
  ARE_MATRICES_CLOSE(duv_sb_minimal_auto, duv_sb_minimal_auto, tol);

  // compute the numeric diff and check
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      duv_deltaTWS_numeric(2, 7);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      duv_deltaTWS_minimal_numeric(2, 6);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      duv_deltahpW_numeric(2, 4);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      duv_deltahpW_minimal_numeric(2, 3);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      duv_deltaTSC_numeric(2, 7);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      duv_deltaTSC_minimal_numeric(2, 6);

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      duv_proj_intrinsic_numeric(2, kProjIntrinsicDim);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      duv_distortion_numeric(2, kDistortionDim);
  Eigen::Matrix<double, 2, 1> duv_tr_numeric;

  Eigen::Matrix<double, 2, 1> duv_td_numeric;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      duv_sb_numeric(2, 9);

  simul::computeNumericJacPose(*poseBlocks_[observationIndex], errorPtr.get(), parameters,
                               residuals, &duv_deltaTWS_minimal_numeric, true);
  simul::computeNumericJacPose(*poseBlocks_[observationIndex], errorPtr.get(), parameters,
                               residuals, &duv_deltaTWS_numeric, false);
  simul::computeNumericJacPoint(*visibleLandmarks_[landmarkIndex], *homogeneousPointLocalParameterization_,
                                errorPtr.get(), parameters, residuals,
                                &duv_deltahpW_minimal_numeric, true);
  simul::computeNumericJacPoint(*visibleLandmarks_[landmarkIndex], *homogeneousPointLocalParameterization_,
                                errorPtr.get(), parameters, residuals,
                                &duv_deltahpW_numeric, false);

  simul::computeNumericJacPose(*extrinsicBlock_, errorPtr.get(),
                               parameters, residuals,
                               &duv_deltaTSC_minimal_numeric, true);
  simul::computeNumericJacPose(*extrinsicBlock_, errorPtr.get(),
                               parameters, residuals, &duv_deltaTSC_numeric,
                               false);
  simul::computeNumericJac(*cameraParameterBlocks_[0], errorPtr.get(), parameters,
                           residuals, &duv_proj_intrinsic_numeric);

  simul::computeNumericJac(*cameraParameterBlocks_[1], errorPtr.get(), parameters,
                           residuals, &duv_distortion_numeric);

  simul::computeNumericJac<Eigen::Matrix<double, 2, 1>>(
      *cameraParameterBlocks_[2], errorPtr.get(), parameters, residuals, &duv_tr_numeric);

  simul::computeNumericJac<Eigen::Matrix<double, 2, 1>>(
      *cameraParameterBlocks_[3], errorPtr.get(), parameters, residuals, &duv_td_numeric);

  simul::computeNumericJac(*speedAndBiasBlocks_[observationIndex], errorPtr.get(), parameters, residuals,
                           &duv_sb_numeric);

  ARE_MATRICES_CLOSE(duv_deltaTWS_numeric, duv_deltaTWS, tol);
  ARE_MATRICES_CLOSE(duv_deltaTWS_minimal_numeric, duv_deltaTWS_minimal, 5e-3);
  ARE_MATRICES_CLOSE(duv_deltahpW_numeric, duv_deltahpW, tol);
  ARE_MATRICES_CLOSE(duv_deltahpW_minimal_numeric, duv_deltahpW_minimal, tol);
  ARE_MATRICES_CLOSE(duv_deltaTSC_numeric, duv_deltaTSC, tol);
  ARE_MATRICES_CLOSE(duv_deltaTSC_minimal_numeric, duv_deltaTSC_minimal, 5e-3);

  ARE_MATRICES_CLOSE(duv_proj_intrinsic_numeric, duv_proj_intrinsic, 4e-3);
  ARE_MATRICES_CLOSE(duv_distortion_numeric, duv_distortion, 4e-3);
  ARE_MATRICES_CLOSE(duv_tr_numeric, duv_tr, 1e-1);
  ARE_MATRICES_CLOSE(duv_td_numeric, duv_td, 1e-1);

  Eigen::Matrix<double, 2, 3> duv_ds_numeric =
      duv_sb_numeric.topLeftCorner<2, 3>();
  ARE_MATRICES_CLOSE(duv_ds_numeric, duv_ds, tol);
  Eigen::Matrix<double, 2, 3> duv_dbg_numeric =
      duv_sb_numeric.block<2, 3>(0, 3);
  EXPECT_LT((duv_dbg_numeric - duv_dbg).lpNorm<Eigen::Infinity>(), 5e-2)
      << "duv_dbg_numeric\n"
      << duv_dbg_numeric << "\nduv_dbg\n"
      << duv_dbg;
  Eigen::Matrix<double, 2, 3> duv_dba_numeric =
      duv_sb_numeric.topRightCorner<2, 3>();
  EXPECT_LT((duv_dba_numeric - duv_dba).lpNorm<Eigen::Infinity>(), 5e-3)
      << "duv_dba_numeric\n"
      << duv_dba_numeric << "\nduv_dba\n"
      << duv_dba;
}

void CameraObservationJacobianTest::verifyJacobiansAIDP(
    std::shared_ptr<const okvis::ceres::ErrorInterface> errorPtr,
    int observationIndex, int landmarkIndex, 
    std::shared_ptr<swift_vio::PointSharedData> pointDataPtr,
    std::shared_ptr<DistortedPinholeCameraGeometry> cameraGeometry,
    const Eigen::Vector2d & imagePoint, int hostIndex) const {
    // TODO(jhuai): binliang
    // compare Jacobians obtained by analytic diff and auto diff.

        double const *const parameters[] = {
        poseBlocks_[observationIndex]->parameters(),
        visibleLandmarks_[landmarkIndex]->data(),
        poseBlocks_[hostIndex]->parameters(),
        extrinsicBlock_->parameters(),
        extrinsicBlockh_->parameters(),
        cameraParameterBlocks_[0]->parameters(),
        cameraParameterBlocks_[1]->parameters(),
        cameraParameterBlocks_[2]->parameters(),   
        speedAndBiasBlocks_[observationIndex]->parameters(),
        TgParameterBlocks_->parameters(),
        TsParameterBlocks_->parameters(),
        TaParameterBlocks_->parameters()};   
    
    Eigen::Vector2d residuals;
    Eigen::Vector2d residuals_auto;

    Eigen::Matrix<double, 2, 7, Eigen::RowMajor> duv_deltaTWSt_auto;
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> duv_deltaTWSt_minimal_auto;
    Eigen::Matrix<double, 2, 4, Eigen::RowMajor> duv_deltahpCh_auto;
    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> duv_deltahpCh_minimal_auto;  
    Eigen::Matrix<double, 2, 7, Eigen::RowMajor> duv_deltaTSCt_auto;
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> duv_deltaTSCt_minimal_auto;

    Eigen::Matrix<double, 2, 7, Eigen::RowMajor> duv_deltaTWSh_auto;
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> duv_deltaTWSh_minimal_auto;
    Eigen::Matrix<double, 2, 7, Eigen::RowMajor> duv_deltaTSCh_auto;
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> duv_deltaTSCh_minimal_auto;

    const int kNumIntrinsics = kProjIntrinsicDim + kDistortionDim;   
    //Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::RowMajor> duv_proj_intrinsic_auto(2, kProjIntrinsicDim);
    //Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::RowMajor> duv_distortion_auto(2, kDistortionDim);
    Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::RowMajor> duv_intrinsic_auto(2, kNumIntrinsics);
    //Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::RowMajor> duv_proj_intrinsic_minimal_auto(2, kProjIntrinsicDim);
    //Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::RowMajor> duv_distortion_minimal_auto(2, kDistortionDim);
    Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::RowMajor> duv_intrinsic_minimal_auto(2, kNumIntrinsics);

    Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::RowMajor> duv_tr_auto( 2, 1);
    Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::RowMajor> duv_tr_minimal_auto( 2, 1);
    Eigen::Matrix<double, 2, 1> duv_td_auto;
    Eigen::Matrix<double, 2, 1> duv_td_minimal_auto;
    Eigen::Matrix<double, 2, 9, Eigen::RowMajor> duv_sb_auto;
    Eigen::Matrix<double, 2, 9, Eigen::RowMajor> duv_sb_minimal_auto;

    Eigen::Matrix<double, 2, 9, Eigen::RowMajor> duv_Tg_auto;
    Eigen::Matrix<double, 2, 9, Eigen::RowMajor> duv_Tg_minimal_auto;
    Eigen::Matrix<double, 2, 9, Eigen::RowMajor> duv_Ts_auto;
    Eigen::Matrix<double, 2, 9, Eigen::RowMajor> duv_Ts_minimal_auto;
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> duv_Ta_auto;
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> duv_Ta_minimal_auto;

    double *jacobiansAD[] = {
        duv_deltaTWSt_auto.data(), 
        duv_deltahpCh_auto.data(),
        duv_deltaTWSh_auto.data(), 
        duv_deltaTSCt_auto.data(),duv_deltaTSCh_auto.data(),
        duv_intrinsic_auto.data(),
        duv_tr_auto.data(),
        duv_td_auto.data(), 
        duv_sb_auto.data(),
        duv_Tg_auto.data(),duv_Ts_auto.data(), duv_Ta_auto.data()
        };
    double *jacobiansMinimalAD[] = {
        duv_deltaTWSt_minimal_auto.data(), 
        duv_deltahpCh_minimal_auto.data(),
        duv_deltaTWSh_minimal_auto.data(),
        duv_deltaTSCt_minimal_auto.data(), duv_deltaTSCh_minimal_auto.data(),
        duv_intrinsic_minimal_auto.data(),
        duv_tr_minimal_auto.data(),
        duv_td_minimal_auto.data(),
        duv_sb_minimal_auto.data(),
        duv_Tg_minimal_auto.data(),duv_Ts_minimal_auto.data(), duv_Ta_minimal_auto.data()
        };  

    Eigen::Matrix<double, 2, 7, Eigen::RowMajor> duv_deltaTWSt;
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> duv_deltaTWSt_minimal;
    Eigen::Matrix<double, 2, 4, Eigen::RowMajor> duv_deltahpCh;
    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> duv_deltahpCh_minimal;  
    Eigen::Matrix<double, 2, 7, Eigen::RowMajor> duv_deltaTSCt;
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> duv_deltaTSCt_minimal;

    Eigen::Matrix<double, 2, 7, Eigen::RowMajor> duv_deltaTWSh;
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> duv_deltaTWSh_minimal;
    Eigen::Matrix<double, 2, 7, Eigen::RowMajor> duv_deltaTSCh;
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> duv_deltaTSCh_minimal;

    //Eigen::Matrix<double, 2, kProjIntrinsicDim, Eigen::RowMajor> duv_proj_intrinsic;
    //Eigen::Matrix<double, 2, kDistortionDim, Eigen::RowMajor> duv_distortion;
    Eigen::Matrix<double, 2, kNumIntrinsics, Eigen::RowMajor> duv_intrinsic;
    //Eigen::Matrix<double, 2, kProjIntrinsicDim, Eigen::RowMajor> duv_proj_intrinsic_minimal;
    //Eigen::Matrix<double, 2, kDistortionDim, Eigen::RowMajor> duv_distortion_minimal;
    Eigen::Matrix<double, 2, kNumIntrinsics, Eigen::RowMajor> duv_intrinsic_minimal;

    Eigen::Matrix<double, 2, 1> duv_tr;
    Eigen::Matrix<double, 2, 1> duv_tr_minimal;
    Eigen::Matrix<double, 2, 1> duv_td;
    Eigen::Matrix<double, 2, 1> duv_td_minimal;
    Eigen::Matrix<double, 2, 9, Eigen::RowMajor> duv_sb;
    Eigen::Matrix<double, 2, 9, Eigen::RowMajor> duv_sb_minimal;

    Eigen::Matrix<double, 2, 9, Eigen::RowMajor> duv_Tg;
    Eigen::Matrix<double, 2, 9, Eigen::RowMajor> duv_Tg_minimal;
    Eigen::Matrix<double, 2, 9, Eigen::RowMajor> duv_Ts;
    Eigen::Matrix<double, 2, 9, Eigen::RowMajor> duv_Ts_minimal;
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> duv_Ta;
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> duv_Ta_minimal;

    double *jacobians[] = {   
        duv_deltaTWSt.data(), duv_deltahpCh.data(),
        duv_deltaTWSh.data(), 
        duv_deltaTSCt.data(),duv_deltaTSCh.data(),
        duv_intrinsic.data(),
        duv_tr.data(),
        duv_td.data(), 
        duv_sb.data(),
        duv_Tg.data(),duv_Ts.data(), duv_Ta.data()
        };
    double *jacobiansMinimal[] = {   
        duv_deltaTWSt_minimal.data(), duv_deltahpCh_minimal.data(),
        duv_deltaTWSh_minimal.data(),
        duv_deltaTSCt_minimal.data(), duv_deltaTSCh_minimal.data(),
        duv_intrinsic_minimal.data(),
        duv_tr_minimal.data(),
        duv_td_minimal.data(),
        duv_sb_minimal.data(),
        duv_Tg_minimal.data(),duv_Ts_minimal.data(), duv_Ta_minimal.data()
        };

    std::shared_ptr<const okvis::ceres::RSCameraReprojectionError<
        DistortedPinholeCameraGeometry>>                    
        costFuncPtr =
            std::static_pointer_cast<const okvis::ceres::RSCameraReprojectionError<
                DistortedPinholeCameraGeometry>>(errorPtr);

    costFuncPtr->EvaluateWithMinimalJacobians(parameters, residuals.data(),
                                              jacobians, jacobiansMinimal);
    costFuncPtr->EvaluateWithMinimalJacobiansAutoDiff(
        parameters, residuals_auto.data(), jacobiansAD, jacobiansMinimalAD);

    if (isZeroResidualExpected())
    {
        EXPECT_TRUE(residuals.isMuchSmallerThan(1, 1e-8))
            << "Unusually large residual " << residuals.transpose();
        EXPECT_TRUE(residuals_auto.isMuchSmallerThan(1, 1e-8))
            << "Unusually large residual " << residuals.transpose();
    }
    double tol = 1e-8;
    // analytic full vs minimal
    //ARE_MATRICES_CLOSE(duv_proj_intrinsic, duv_proj_intrinsic_minimal, tol);
    //ARE_MATRICES_CLOSE(duv_distortion, duv_distortion_minimal, tol);
    ARE_MATRICES_CLOSE(duv_intrinsic, duv_intrinsic_minimal, tol);
    ARE_MATRICES_CLOSE(duv_tr, duv_tr_minimal, tol);
    ARE_MATRICES_CLOSE(duv_td, duv_td_minimal, tol);
    ARE_MATRICES_CLOSE(duv_sb, duv_sb_minimal, tol);

    // automatic vs analytic
    tol = 1e-3;
    ARE_MATRICES_CLOSE(residuals_auto, residuals, tol);
    ARE_MATRICES_CLOSE(duv_deltaTWSt_auto, duv_deltaTWSt, tol);
    ARE_MATRICES_CLOSE(duv_deltaTWSh_auto, duv_deltaTWSh, tol);
    ARE_MATRICES_CLOSE(duv_deltahpCh_auto, duv_deltahpCh, tol);
    ARE_MATRICES_CLOSE(duv_deltaTSCt_auto, duv_deltaTSCt, tol);
    ARE_MATRICES_CLOSE(duv_deltaTSCh_auto, duv_deltaTSCh, tol);
    ARE_MATRICES_CLOSE(duv_intrinsic_auto, duv_intrinsic, 4e-3);
    //ARE_MATRICES_CLOSE(duv_proj_intrinsic_auto, duv_proj_intrinsic, 4e-3);
    //ARE_MATRICES_CLOSE(duv_distortion_auto, duv_distortion, 4e-3);

    if (coo_.rollingShutter)
    {
        ARE_MATRICES_CLOSE(duv_tr_auto, duv_tr, 1e-1);
        ARE_MATRICES_CLOSE(duv_td_auto, duv_td, 1e-1);
    }

    Eigen::Matrix<double, 2, 3> duv_ds_auto = duv_sb_auto.topLeftCorner<2, 3>();
    Eigen::Matrix<double, 2, 3> duv_ds = duv_sb.topLeftCorner<2, 3>();
    ARE_MATRICES_CLOSE(duv_ds_auto, duv_ds, tol);
    Eigen::Matrix<double, 2, 3> duv_dbg_auto = duv_sb_auto.block<2, 3>(0, 3);
    Eigen::Matrix<double, 2, 3> duv_dbg = duv_sb.block<2, 3>(0, 3);
    EXPECT_LT((duv_dbg_auto - duv_dbg).lpNorm<Eigen::Infinity>(), 5e-2)
        << "duv_dbg_auto:\n"
        << duv_dbg_auto << "\nduv_dbg\n"
        << duv_dbg;
    Eigen::Matrix<double, 2, 3> duv_dba_auto = duv_sb_auto.topRightCorner<2, 3>();
    Eigen::Matrix<double, 2, 3> duv_dba = duv_sb.topRightCorner<2, 3>();
    EXPECT_LT((duv_dba_auto - duv_dba).lpNorm<Eigen::Infinity>(), 5e-3)
        << "duv_dba_auto\n"
        << duv_dba_auto << "\nduv_dba\n"
        << duv_dba;

    ARE_MATRICES_CLOSE(duv_deltaTWSt_minimal_auto, duv_deltaTWSt_minimal, 5e-3);
    ARE_MATRICES_CLOSE(duv_deltaTWSh_minimal_auto, duv_deltaTWSh_minimal, 5e-3);
    ARE_MATRICES_CLOSE(duv_deltahpCh_minimal_auto, duv_deltahpCh_minimal, tol);
    ARE_MATRICES_CLOSE(duv_deltaTSCt_minimal_auto, duv_deltaTSCt_minimal, tol);
    ARE_MATRICES_CLOSE(duv_deltaTSCh_minimal_auto, duv_deltaTSCh_minimal, tol);

    //ARE_MATRICES_CLOSE(duv_proj_intrinsic_minimal_auto,duv_proj_intrinsic_minimal, 4e-3);
    //ARE_MATRICES_CLOSE(duv_distortion_minimal_auto, duv_distortion_minimal, 4e-3);
    ARE_MATRICES_CLOSE(duv_intrinsic_minimal_auto,duv_intrinsic_minimal, 4e-3);
    if (coo_.rollingShutter)
    {
        ARE_MATRICES_CLOSE(duv_tr_minimal_auto, duv_tr_minimal, 1e-1);
        ARE_MATRICES_CLOSE(duv_td_minimal_auto, duv_td_minimal, 1e-1);
    }
    ARE_MATRICES_CLOSE(duv_sb_minimal_auto, duv_sb_minimal_auto, tol); 
}


void CameraObservationJacobianTest::solveAndCheck() {
  ::ceres::Solver::Options options;
  // options.check_gradients=true;
  // options.numeric_derivative_relative_step_size = 1e-6;
  // options.gradient_check_relative_precision=1e-2;
  options.minimizer_progress_to_stdout = false;
  ::FLAGS_stderrthreshold =
      google::WARNING; // enable console warnings (Jacobian verification)
  ::ceres::Solver::Summary summary;
  Solve(options, problem_.get(), &summary);

  // print some infos about the optimization
  LOG(INFO) << summary.BriefReport();

  LOG(INFO) << "initial T_WB:\n" << initial_T_WB_.T() << "\n"
            << "optimized T_WB:\n" << poseBlocks_[2]->estimate().T() << "\n"
            << "correct T_WB:\n" << ref_T_WB_.T();

  // make sure it converged
  EXPECT_LT(2 * (ref_T_WB_.q() * poseBlocks_[2]->estimate().q().inverse())
                    .vec()
                    .norm(),
            1e-2)
      << "quaternions not close enough";

  EXPECT_LT((ref_T_WB_.r() - poseBlocks_[2]->estimate().r()).norm(), 1e-1)
      << "translation not close enough";

}

void NumericJacobianPAP::computeNumericJacobianForPoint(
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        *de_dPoint,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        *de_dPoint_minimal) {
  Eigen::VectorXd residual(krd_);

  if (krd_ == 2 && observationIndex_ != 0) {
    Eigen::Vector2d mimicResidual;
    computeReprojectionWithPapResidual(&mimicResidual, nullptr);
    EXPECT_LT((mimicResidual - refResidual_).lpNorm<Eigen::Infinity>(), 1e-8);
  }

  Eigen::VectorXd originalValue = pointLandmark_->estimate();
  swift_vio::ParallaxAngleParameterization pap;
  Eigen::VectorXd delta;
  delta.resize(pap.LocalSize());
  for (int j = 0; j < pap.LocalSize(); ++j) {
    delta.setZero();
    delta[j] = h;
    pap.Plus(pointLandmark_->data(), delta.data(), pointLandmark_->data());
    bool status = costFunctionPtr_->EvaluateWithMinimalJacobians(
        parameters_, residual.data(), nullptr, nullptr);
    EXPECT_TRUE(status);
    de_dPoint_minimal->col(j) = (residual - refResidual_) / h;
    pointLandmark_->setEstimate(originalValue);
  }

  Eigen::Matrix<double, 3, 6, Eigen::RowMajor> jLift;
  swift_vio::ParallaxAngleParameterization::liftJacobian(
      pointLandmark_->data(), jLift.data());
  *de_dPoint = (*de_dPoint_minimal) * jLift;
}
}  // namespace swift_vio
