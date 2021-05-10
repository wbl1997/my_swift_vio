#include <glog/logging.h>
#include <gtest/gtest.h>

#include <ceres/ceres.h>

#include <swift_vio/memory.h>
#include <swift_vio/CameraRig.hpp>

#include <swift_vio/CameraTimeParamBlock.hpp>
#include <swift_vio/ChordalDistance.hpp>
#include <swift_vio/EuclideanParamBlock.hpp>
#include <swift_vio/EuclideanParamBlockSized.hpp>
#include <swift_vio/ExtrinsicModels.hpp>
#include <swift_vio/ParallaxAnglePoint.hpp>
#include <swift_vio/PointLandmark.hpp>
#include <swift_vio/PointLandmarkModels.hpp>
#include <swift_vio/ProjParamOptModels.hpp>
#include <swift_vio/ReprojectionErrorWithPap.hpp>
#include <swift_vio/imu/BoundedImuDeque.hpp>

#include <okvis/FrameTypedefs.hpp>
#include <okvis/Time.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/ceres/HomogeneousPointError.hpp>
#include <okvis/ceres/HomogeneousPointLocalParameterization.hpp>
#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>
#include <okvis/kinematics/Transformation.hpp>

#include <simul/CameraSystemCreator.hpp>
#include <simul/curves.h>
#include <simul/numeric_ceres_residual_Jacobian.hpp>

namespace {

class CameraObservationOptions {
 public:
  CameraObservationOptions()
      : perturbPose(false),
        rollingShutter(false),
        noisyKeypoint(false),
        cameraObservationModelId(okvis::cameras::kChordalDistanceId),
        projOptModelName("FXY_CXY"),
        extrinsicOptModelName("P_BC_Q_BC") {}

  bool perturbPose;
  bool rollingShutter;
  bool noisyKeypoint;
  int cameraObservationModelId;

  std::string projOptModelName;
  std::string extrinsicOptModelName;
};

typedef okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>
    DistortedPinholeCameraGeometry;
const int kDistortionDim =
    DistortedPinholeCameraGeometry::distortion_t::NumDistortionIntrinsics;
const int kProjIntrinsicDim = okvis::ProjectionOptFXY_CXY::kNumParams;
const int kExtrinsicMinimalDim = 3;
Eigen::Matrix2d covariance = Eigen::Matrix2d::Identity() / 0.36;
Eigen::Matrix2d squareRootInformation = Eigen::Matrix2d::Identity() * 0.6;

class CameraObservationJacobianTest {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CameraObservationJacobianTest(CameraObservationOptions coo)
      : coo_(coo),
        nextBlockIndex_(1u),
        papLocalParameterization_(new swift_vio::ParallaxAngleParameterization()) {
    ::ceres::Problem::Options problemOptions;
    problemOptions.local_parameterization_ownership =
        ::ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
    problemOptions.loss_function_ownership =
        ::ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
    problemOptions.cost_function_ownership =
        ::ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
    problem_.reset(new ::ceres::Problem(problemOptions));
    int extrinsicModelId = okvis::ExtrinsicModelNameToId(coo_.extrinsicOptModelName);
    if (extrinsicModelId == 1) {
      extrinsicLocalParameterization_.reset(new okvis::Extrinsic_p_CB());
    } else {
      extrinsicLocalParameterization_.reset(
          new okvis::ceres::PoseLocalParameterization());
    }
    poseLocalParameterization_.reset(new okvis::ceres::PoseLocalParameterization());
  }

  bool isZeroResidualExpected() const {
    return !(coo_.perturbPose || coo_.rollingShutter || coo_.noisyKeypoint);
  }

  okvis::Time lastStateEpoch() const { return stateEpochs_.back(); }

  std::vector<uint64_t> frameIds() const { return frameIds_; }

  Eigen::AlignedVector<okvis::kinematics::Transformation> truePoses() const {
    Eigen::AlignedVector<okvis::kinematics::Transformation> T_WB_list;
    T_WB_list.reserve(3);
    for (int j = 0; j < 3; ++j) {
        T_WB_list.push_back(poseBlocks_[j]->estimate());
    }
    T_WB_list[2] = expected_T_WB_;
    return T_WB_list;
  }

  std::vector<std::shared_ptr<okvis::ceres::PoseParameterBlock>> poseBlocks() const {
    return poseBlocks_;
  }

  std::shared_ptr<okvis::ceres::PoseParameterBlock> extrinsicBlock()
      const {
    return extrinsicBlock_;
  }

  uint64_t addNavStatesAndExtrinsic(
      std::shared_ptr<const simul::CircularSinusoidalTrajectory> cameraMotion,
      okvis::Time startEpoch);

  uint64_t addImuAugmentedParameterBlocks(okvis::Time stateEpoch);

  uint64_t addCameraParameterBlocks(const Eigen::VectorXd& intrinsicParams,
                                    okvis::Time startEpoch, double timeOffset);

  void addLandmark(std::shared_ptr<swift_vio::PointLandmark> pl);

  void addImuInfo(const okvis::ImuMeasurementDeque& entireImuList,
                  const okvis::ImuParameters& imuParameters,
                  double timeOffset);

  void propagatePoseAndVelocityForMapPoint(
      std::shared_ptr<swift_vio::PointSharedData> pointDataPtr) const;

  void addResidual(std::shared_ptr<::ceres::CostFunction> costFunctionPtr, int observationIndex,
                   int landmarkIndex);

  void verifyJacobians(std::shared_ptr<const okvis::ceres::ErrorInterface> costFunctionPtr,
                       int observationIndex, int landmarkIndex,
                       std::shared_ptr<swift_vio::PointSharedData> pointDataPtr,
                       std::shared_ptr<DistortedPinholeCameraGeometry> cameraGeometry,
                       const Eigen::Vector2d& imagePoint) const;

  void solveAndCheck();

  inline okvis::Time getImageTimestamp(int observationIndex, int /*cameraIdx*/) const {
    return stateEpochs_[observationIndex] - okvis::Duration(tdAtCreationList_[observationIndex]);
  }

  CameraObservationOptions coo_;

 private:
  // main anchor, associate anchor, observing frame j
  std::vector<okvis::Time> stateEpochs_;
  std::vector<uint64_t> frameIds_;
  std::vector<std::shared_ptr<okvis::ceres::PoseParameterBlock>> poseBlocks_;
  okvis::kinematics::Transformation initial_T_WB_;
  okvis::kinematics::Transformation expected_T_WB_;
  std::vector<std::shared_ptr<okvis::ceres::SpeedAndBiasParameterBlock>>
      speedAndBiasBlocks_;

  // linearization points.
  std::vector<std::shared_ptr<Eigen::Matrix<double, 6, 1>>>
      positionAndVelocityLp_;
  std::vector<std::shared_ptr<okvis::ImuMeasurementDeque>> imuWindowList_;
  std::vector<double> tdAtCreationList_;
  std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>
      imuAugmentedBlocks_;
  okvis::ImuParameters imuParameters_;
  /// camera parameters
  std::shared_ptr<okvis::ceres::PoseParameterBlock> extrinsicBlock_;
  // projection intrinsics, distortion, readout time, time offset
  std::vector<std::shared_ptr<okvis::ceres::ParameterBlock>>
      cameraParameterBlocks_;

  std::shared_ptr<::ceres::Problem> problem_;
  uint64_t nextBlockIndex_;

  std::vector<std::shared_ptr<swift_vio::PointLandmark>> visibleLandmarks_;

  std::shared_ptr<::ceres::LocalParameterization> papLocalParameterization_;
  std::shared_ptr<::ceres::LocalParameterization> extrinsicLocalParameterization_;
  std::shared_ptr<::ceres::LocalParameterization> poseLocalParameterization_;

};

uint64_t CameraObservationJacobianTest::addNavStatesAndExtrinsic(
    std::shared_ptr<const simul::CircularSinusoidalTrajectory> cameraMotion,
    okvis::Time startEpoch) {
  okvis::kinematics::Transformation T_disturb;
  T_disturb.setRandom(1, 0.02);

  frameIds_.reserve(3);
  stateEpochs_.reserve(3);
  poseBlocks_.reserve(3);
  speedAndBiasBlocks_.reserve(3);
  for (int f = 0; f < 3; ++f) {
    // main anchor, associate anchor, and observing frame.
    okvis::Time stateEpoch = startEpoch + okvis::Duration(f, 0);
    okvis::kinematics::Transformation T_WB =
        cameraMotion->computeGlobalPose(stateEpoch);
    Eigen::Vector3d v_WB =
        cameraMotion->computeGlobalLinearVelocity(stateEpoch);
    okvis::SpeedAndBiases speedAndBias;
    speedAndBias.head<3>() = v_WB;
    speedAndBias.tail<6>().setZero();
    if (f == 2) {  // j frame
      expected_T_WB_ = T_WB;
      if (coo_.perturbPose) {
        T_WB = T_WB * T_disturb;
      }
      initial_T_WB_ = T_WB;
    }
    frameIds_.push_back(nextBlockIndex_);
    std::shared_ptr<okvis::ceres::PoseParameterBlock> poseParameterBlock(
        new okvis::ceres::PoseParameterBlock(T_WB, nextBlockIndex_++,
                                             stateEpoch));
    problem_->AddParameterBlock(poseParameterBlock->parameters(),
                               poseParameterBlock->dimension(),
                               poseLocalParameterization_.get());
    poseBlocks_.push_back(poseParameterBlock);
    stateEpochs_.push_back(stateEpoch);

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
  return nextBlockIndex_;
}

uint64_t CameraObservationJacobianTest::addCameraParameterBlocks(
    const Eigen::VectorXd& intrinsicParams, okvis::Time startEpoch,
    double timeOffset) {
  int projOptModelId = okvis::ProjectionOptNameToId(coo_.projOptModelName);
  Eigen::VectorXd projIntrinsics;
  okvis::ProjectionOptGlobalToLocal(projOptModelId, intrinsicParams,
                                    &projIntrinsics);

  Eigen::VectorXd distortion = intrinsicParams.tail(kDistortionDim);
  std::shared_ptr<okvis::ceres::EuclideanParamBlock> projectionParamBlock(
      new okvis::ceres::EuclideanParamBlock(projIntrinsics, nextBlockIndex_++,
                                            startEpoch, kProjIntrinsicDim));
  std::shared_ptr<okvis::ceres::EuclideanParamBlock> distortionParamBlock(
      new okvis::ceres::EuclideanParamBlock(distortion, nextBlockIndex_++,
                                            startEpoch, kDistortionDim));

  double tr = 0;
  if (coo_.rollingShutter) {
    tr = 0.033;
  }
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

void CameraObservationJacobianTest::addLandmark(
    std::shared_ptr<swift_vio::PointLandmark> pl) {
  problem_->AddParameterBlock(pl->data(),
                             swift_vio::ParallaxAngleParameterization::kGlobalDim);
  problem_->SetParameterBlockConstant(pl->data());
  problem_->SetParameterization(pl->data(), papLocalParameterization_.get());
  visibleLandmarks_.push_back(pl);
}

void CameraObservationJacobianTest::addImuInfo(
    const okvis::ImuMeasurementDeque& entireImuList,
    const okvis::ImuParameters& imuParameters,
    double timeOffset) {
  imuParameters_ = imuParameters;
  positionAndVelocityLp_.reserve(3);
  imuWindowList_.reserve(3);
  tdAtCreationList_.reserve(3);
  for (int j = 0; j < 3; ++j) {
    tdAtCreationList_.push_back(timeOffset);
    std::shared_ptr<Eigen::Matrix<double, 6, 1>> positionAndVelocity(
        new Eigen::Matrix<double, 6, 1>());
    positionAndVelocity->head<3>() = poseBlocks_.at(j)->estimate().r();
    positionAndVelocity->tail<3>() =
        speedAndBiasBlocks_.at(j)->estimate().head<3>();
    positionAndVelocityLp_.push_back(positionAndVelocity);
    okvis::Time centerTime = stateEpochs_.at(j);
    okvis::Duration halfSide(0.5);
    std::shared_ptr<okvis::ImuMeasurementDeque> window(
        new okvis::ImuMeasurementDeque());
    *window = okvis::getImuMeasurements(
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

  std::shared_ptr<okvis::ceres::ShapeMatrixParamBlock> taBlockPtr(
      new okvis::ceres::ShapeMatrixParamBlock(eye, nextBlockIndex_++,
                                              stateEpoch));
  problem_->AddParameterBlock(taBlockPtr->parameters(), 9);
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
                             positionAndVelocityLp_[observationIndex]);
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
    std::shared_ptr<::ceres::CostFunction> costFunctionPtr, int observationIndex,
    int landmarkIndex) {
  switch (observationIndex) {
    case 0: // main
      problem_->AddResidualBlock(
          costFunctionPtr.get(), NULL, (double*)(nullptr),
          poseBlocks_[0]->parameters(), poseBlocks_[1]->parameters(),
          visibleLandmarks_[landmarkIndex]->data(), extrinsicBlock_->parameters(),
          cameraParameterBlocks_[0]->parameters(),
          cameraParameterBlocks_[1]->parameters(),
          cameraParameterBlocks_[2]->parameters(),
          cameraParameterBlocks_[3]->parameters(),
          (double*)(nullptr),
          speedAndBiasBlocks_[0]->parameters(),
          speedAndBiasBlocks_[1]->parameters());
      break;
    case 1: // associate
      problem_->AddResidualBlock(
          costFunctionPtr.get(), NULL, (double*)(nullptr),
          poseBlocks_[0]->parameters(), poseBlocks_[1]->parameters(),
          visibleLandmarks_[landmarkIndex]->data(), extrinsicBlock_->parameters(),
          cameraParameterBlocks_[0]->parameters(),
          cameraParameterBlocks_[1]->parameters(),
          cameraParameterBlocks_[2]->parameters(),
          cameraParameterBlocks_[3]->parameters(),
          (double*)(nullptr),
          speedAndBiasBlocks_[0]->parameters(),
          speedAndBiasBlocks_[1]->parameters());
      break;
    default:
      const double * const parameters[] = {
        poseBlocks_[observationIndex]->parameters(),
                  poseBlocks_[0]->parameters(), poseBlocks_[1]->parameters(),
                  visibleLandmarks_[landmarkIndex]->data(), extrinsicBlock_->parameters(),
                  cameraParameterBlocks_[0]->parameters(),
                  cameraParameterBlocks_[1]->parameters(),
                  cameraParameterBlocks_[2]->parameters(),
                  cameraParameterBlocks_[3]->parameters(),
                  speedAndBiasBlocks_[observationIndex]->parameters(),
                  speedAndBiasBlocks_[0]->parameters(),
                  speedAndBiasBlocks_[1]->parameters()
      };
      Eigen::Vector3d residual;
      costFunctionPtr->Evaluate(parameters, residual.data(), nullptr);

      problem_->AddResidualBlock(
          costFunctionPtr.get(), NULL, poseBlocks_[observationIndex]->parameters(),
          poseBlocks_[0]->parameters(), poseBlocks_[1]->parameters(),
          visibleLandmarks_[landmarkIndex]->data(), extrinsicBlock_->parameters(),
          cameraParameterBlocks_[0]->parameters(),
          cameraParameterBlocks_[1]->parameters(),
          cameraParameterBlocks_[2]->parameters(),
          cameraParameterBlocks_[3]->parameters(),
          speedAndBiasBlocks_[observationIndex]->parameters(),
          speedAndBiasBlocks_[0]->parameters(),
          speedAndBiasBlocks_[1]->parameters());
      break;
  }
}

class NumericJacobian {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 NumericJacobian(
     std::shared_ptr<const okvis::ceres::ErrorInterface> costFunctionPtr,
     const std::vector<std::shared_ptr<okvis::ceres::PoseParameterBlock>>&
         poseBlocks,
     std::shared_ptr<swift_vio::PointLandmark> pointLandmark,
     std::shared_ptr<okvis::ceres::PoseParameterBlock> extrinsicBlock,
     const std::vector<std::shared_ptr<okvis::ceres::ParameterBlock>>&
         cameraParameterBlocks,
     const std::vector<
         std::shared_ptr<okvis::ceres::SpeedAndBiasParameterBlock>>&
         speedAndBiasBlocks,
     const std::vector<std::shared_ptr<Eigen::Matrix<double, 6, 1>>>&
         positionAndVelocityLpList,
     std::shared_ptr<swift_vio::PointSharedData> pointDataPtr,
     std::shared_ptr<DistortedPinholeCameraGeometry> cameraGeometry,
     Eigen::Vector2d imagePoint,
     int observationIndex,
     int krd)
     : costFunctionPtr_(costFunctionPtr),
       poseBlocks_(poseBlocks),
       pointLandmark_(pointLandmark),
       extrinsicBlock_(extrinsicBlock),
       cameraParameterBlocks_(cameraParameterBlocks),
       speedAndBiasBlocks_(speedAndBiasBlocks),
       positionAndVelocityLpList_(positionAndVelocityLpList),
       parameters_{poseBlocks_[observationIndex]->parameters(),
                   poseBlocks_[0]->parameters(),
                   poseBlocks_[1]->parameters(),
                   pointLandmark_->data(),
                   extrinsicBlock_->parameters(),
                   cameraParameterBlocks_[0]->parameters(),
                   cameraParameterBlocks_[1]->parameters(),
                   cameraParameterBlocks_[2]->parameters(),
                   cameraParameterBlocks_[3]->parameters(),
                   speedAndBiasBlocks_[observationIndex]->parameters(),
                   speedAndBiasBlocks_[0]->parameters(),
                   speedAndBiasBlocks_[1]->parameters()},
       pointDataPtr_(pointDataPtr),
       cameraGeometryBase_(cameraGeometry),
       imagePoint_(imagePoint),
       observationIndex_(observationIndex),
       krd_(krd) {
   refResidual_.resize(krd);
   costFunctionPtr_->EvaluateWithMinimalJacobians(
         parameters_, refResidual_.data(), nullptr, nullptr);
 }

 // scheme:
 // adjust the parameter block a bit
 // update pointDataPtr
 // compute residual
 // compute Jacobian
 // reset the parameter block back and recompute pointDataPtr

 void computeNumericJacobianForPose(
     int majIndex,
     Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>*
         de_deltaTWB,
     Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>*
         de_deltaTWB_minimal) {
   Eigen::VectorXd residual(krd_);
   okvis::kinematics::Transformation ref_T_WB = poseBlocks_.at(majIndex)->estimate();
   Eigen::Matrix<double, 6, 1> delta;
   for (int j = 0; j < 6; ++j) {
     delta.setZero();
     delta[j] = h;
     okvis::kinematics::Transformation T_WB = ref_T_WB;
     T_WB.oplus(delta);
     poseBlocks_.at(majIndex)->setEstimate(T_WB);
     positionAndVelocityLpList_[majIndex]->head<3>() = T_WB.r();
     pointDataPtr_->computePoseAndVelocityAtObservation();
     bool useFirstEstimate = true;
     pointDataPtr_->computePoseAndVelocityForJacobians(useFirstEstimate);
     int cameraObservationModelId = 2;
     pointDataPtr_->computeSharedJacobians(cameraObservationModelId);
     costFunctionPtr_->EvaluateWithMinimalJacobians(
           parameters_, residual.data(), nullptr, nullptr);
     de_deltaTWB_minimal->col(j) = (residual - refResidual_) / h;

     // reset
     poseBlocks_.at(majIndex)->setEstimate(ref_T_WB);
     positionAndVelocityLpList_[majIndex]->head<3>() = ref_T_WB.r();
     pointDataPtr_->computePoseAndVelocityAtObservation();
     pointDataPtr_->computePoseAndVelocityForJacobians(useFirstEstimate);
     pointDataPtr_->computeSharedJacobians(cameraObservationModelId);
   }
   Eigen::Matrix<double, 6, 7, Eigen::RowMajor> jLift;
   okvis::ceres::PoseLocalParameterization::liftJacobian(ref_T_WB.parameters().data(), jLift.data());
   *de_deltaTWB = (*de_deltaTWB_minimal) * jLift;
 }

 void computeNumericJacobianForSpeedAndBias(
     int majIndex, Eigen::Matrix<double, Eigen::Dynamic, -1, Eigen::RowMajor>*
                       de_dSpeedAndBias) {
   Eigen::VectorXd residual(krd_);
   okvis::SpeedAndBias refSpeedAndBias = speedAndBiasBlocks_.at(majIndex)->estimate();
   for (int j = 0; j < 9; ++j) {
     okvis::SpeedAndBias speedAndBias = refSpeedAndBias;
     speedAndBias[j] += h;
     speedAndBiasBlocks_.at(majIndex)->setEstimate(speedAndBias);
     positionAndVelocityLpList_[majIndex]->tail<3>() = speedAndBias.head<3>();
     pointDataPtr_->computePoseAndVelocityAtObservation();
     bool useFirstEstimate = true;
     pointDataPtr_->computePoseAndVelocityForJacobians(useFirstEstimate);
     int cameraObservationModelId = 2;
     pointDataPtr_->computeSharedJacobians(cameraObservationModelId);
     costFunctionPtr_->EvaluateWithMinimalJacobians(
           parameters_, residual.data(), nullptr, nullptr);
     de_dSpeedAndBias->col(j) = (residual - refResidual_) / h;

     // reset
     speedAndBiasBlocks_.at(majIndex)->setEstimate(refSpeedAndBias);
     positionAndVelocityLpList_[majIndex]->tail<3>() = refSpeedAndBias.head<3>();
     pointDataPtr_->computePoseAndVelocityAtObservation();
     pointDataPtr_->computePoseAndVelocityForJacobians(useFirstEstimate);
     pointDataPtr_->computeSharedJacobians(cameraObservationModelId);
   }
 }

 /**
  * @brief computeReprojectionWithPapResidual mimics Evaluate() of ReprojectionErrorWithPap.
  * @warning Only use it with ReprojectionErrorWithPap.
  * @param residual
  * @param de_dPap
  * @return residual computation Ok or not?
  */
 bool computeReprojectionWithPapResidual(
     Eigen::Vector2d* residual,
     Eigen::Matrix<double, 2, 3, Eigen::RowMajor>* de_dPap) const {
   LWF::ParallaxAnglePoint pap;
   pap.set(parameters_[3]);
   Eigen::Matrix<double, 3, 1> t_BC_B(parameters_[4][0], parameters_[4][1],
                                      parameters_[4][2]);
   Eigen::Quaternion<double> q_BC(parameters_[4][6], parameters_[4][3],
                                  parameters_[4][4], parameters_[4][5]);
   std::pair<Eigen::Vector3d, Eigen::Quaterniond> pair_T_BC(t_BC_B, q_BC);

   // compute N_{i,j}.
   okvis::kinematics::Transformation T_WBtij =
       pointDataPtr_->T_WBtij(observationIndex_);
   okvis::kinematics::Transformation T_WBtmi =
       pointDataPtr_->T_WBtij(0);
   okvis::kinematics::Transformation T_WBtai =
       pointDataPtr_->T_WBtij(1);

   swift_vio::TransformMultiplyJacobian T_WCtij_jacobian(
       std::make_pair(T_WBtij.r(), T_WBtij.q()), pair_T_BC);
   swift_vio::TransformMultiplyJacobian T_WCtmi_jacobian(
       std::make_pair(T_WBtmi.r(), T_WBtmi.q()), pair_T_BC);
   swift_vio::TransformMultiplyJacobian T_WCtai_jacobian(
       std::make_pair(T_WBtai.r(), T_WBtai.q()), pair_T_BC);
   std::pair<Eigen::Vector3d, Eigen::Quaterniond> pair_T_WCtij =
       T_WCtij_jacobian.multiply();
   std::pair<Eigen::Vector3d, Eigen::Quaterniond> pair_T_WCtmi =
       T_WCtmi_jacobian.multiply();
   std::pair<Eigen::Vector3d, Eigen::Quaterniond> pair_T_WCtai =
       T_WCtai_jacobian.multiply();

   swift_vio::DirectionFromParallaxAngleJacobian NijFunction(
       pair_T_WCtmi, pair_T_WCtai.first, pair_T_WCtij.first, pap);
   Eigen::Vector3d Nij = NijFunction.evaluate();
   Eigen::Vector3d NijC = pair_T_WCtij.second.conjugate() * Nij;
   Eigen::Vector2d imagePoint;
   Eigen::Matrix<double, 2, 3> pointJacobian;
   Eigen::Matrix2Xd intrinsicsJacobian;
   okvis::cameras::CameraBase::ProjectionStatus projectStatus =
       cameraGeometryBase_->project(NijC, &imagePoint, &pointJacobian,
                               &intrinsicsJacobian);

   Eigen::Matrix3d R_CtijW = pair_T_WCtij.second.toRotationMatrix().transpose();
   Eigen::Matrix<double, 3, 3> dNC_dN = R_CtijW;
   Eigen::Matrix<double, 2, 3> de_dN = pointJacobian * dNC_dN;
   Eigen::Matrix<double, 3, 2> dN_dni;
   NijFunction.dN_dni(&dN_dni);
   Eigen::Matrix<double, 3, 1> dN_dthetai;
   NijFunction.dN_dthetai(&dN_dthetai);
   Eigen::Matrix<double, 3, 3> dN_dntheta;
   dN_dntheta.topLeftCorner<3, 2>() = dN_dni;
   dN_dntheta.col(2) = dN_dthetai;
   if (de_dPap) {
     *de_dPap = squareRootInformation * de_dN * dN_dntheta;
   }

   bool projectOk = projectStatus ==
                    okvis::cameras::CameraBase::ProjectionStatus::Successful;
   Eigen::Vector2d error = imagePoint - imagePoint_;
   *residual = squareRootInformation * error;
   return projectOk;
 }

 void computeNumericJacobianForPoint(
     Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>*
         de_dPoint,
     Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>*
         de_dPoint_minimal) {
   Eigen::Vector3d delta;
   Eigen::VectorXd residual(krd_);

   if (krd_ == 2 && observationIndex_ != 0) {
     Eigen::Vector2d mimicResidual;
     computeReprojectionWithPapResidual(&mimicResidual, nullptr);
     EXPECT_LT((mimicResidual - refResidual_).lpNorm<Eigen::Infinity>(), 1e-8);
   }

   LWF::ParallaxAnglePoint refPap;
   refPap.set(pointLandmark_->data());
   for (int j = 0; j < 3; ++j) {
       delta.setZero();
       delta[j] = h;
       swift_vio::ParallaxAngleParameterization pap;
       pap.Plus(pointLandmark_->data(), delta.data(), pointLandmark_->data());
       bool status = costFunctionPtr_->EvaluateWithMinimalJacobians(
             parameters_, residual.data(), nullptr, nullptr);
       EXPECT_TRUE(status);
       de_dPoint_minimal->col(j) = (residual - refResidual_) / h;
       // reset
       refPap.copy(pointLandmark_->data());
   }

   Eigen::Matrix<double, 3, 6, Eigen::RowMajor> jLift;
   swift_vio::ParallaxAngleParameterization::liftJacobian(pointLandmark_->data(), jLift.data());
   *de_dPoint = (*de_dPoint_minimal) * jLift;
 }

 void computeNumericJacobianForExtrinsic(
     Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>*
         de_dExtrinsic,
     Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>*
         de_dExtrinsic_minimal) {
   Eigen::VectorXd residual(krd_);
   okvis::kinematics::Transformation ref_T_BC = extrinsicBlock_->estimate();
   Eigen::Matrix<double, 6, 1> delta;
   for (int j = 0; j < 3; ++j) {
     delta.setZero();
     delta[j] = h;
     std::pair<Eigen::Vector3d, Eigen::Quaterniond> T_BC(ref_T_BC.r(), ref_T_BC.q());
     okvis::Extrinsic_p_CB::oplus(delta.data(), &T_BC);

     extrinsicBlock_->setEstimate(okvis::kinematics::Transformation(T_BC.first, T_BC.second));

     costFunctionPtr_->EvaluateWithMinimalJacobians(
           parameters_, residual.data(), nullptr, nullptr);

     de_dExtrinsic_minimal->col(j) = (residual - refResidual_) / h;

     // reset
     extrinsicBlock_->setEstimate(ref_T_BC);
   }
   Eigen::Matrix<double, 3, 7, Eigen::RowMajor> jLift;
   okvis::Extrinsic_p_CB::liftJacobian(ref_T_BC.parameters().data(), jLift.data());
   *de_dExtrinsic = (*de_dExtrinsic_minimal) * jLift;
 }

 void computeNumericJacobianForProjectionIntrinsic(
     Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>*
         de_dProjectionIntrinsic) {
   Eigen::VectorXd residual(krd_);
   Eigen::Matrix<double, kProjIntrinsicDim, 1> refProjectionIntrinsic =
       Eigen::Map<Eigen::Matrix<double, kProjIntrinsicDim, 1>>(
           cameraParameterBlocks_[0]->parameters());
   Eigen::VectorXd refIntrinsics(kProjIntrinsicDim + kDistortionDim);
   cameraGeometryBase_->getIntrinsics(refIntrinsics);

   for (int j = 0; j < kProjIntrinsicDim; ++j) {
     Eigen::Map<Eigen::Matrix<double, kProjIntrinsicDim, 1>>
         projectionIntrinsic(cameraParameterBlocks_[0]->parameters());
     projectionIntrinsic[j] += h;
     Eigen::VectorXd intrinsics = refIntrinsics;
     intrinsics[j] += h;
     cameraGeometryBase_->setIntrinsics(intrinsics);
     costFunctionPtr_->EvaluateWithMinimalJacobians(
         parameters_, residual.data(), nullptr, nullptr);
     de_dProjectionIntrinsic->col(j) = (residual - refResidual_) / h;
     // reset
     projectionIntrinsic[j] = refProjectionIntrinsic[j];
     cameraGeometryBase_->setIntrinsics(refIntrinsics);
   }
 }

 void computeNumericJacobianForDistortion(
     Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>*
         de_dDistortion) {
   Eigen::VectorXd residual(krd_);
   Eigen::Matrix<double, kDistortionDim, 1> refDistortion =
       Eigen::Map<Eigen::Matrix<double, kDistortionDim, 1>>(
           cameraParameterBlocks_[1]->parameters());
   Eigen::VectorXd refIntrinsics(kProjIntrinsicDim + kDistortionDim);
   cameraGeometryBase_->getIntrinsics(refIntrinsics);

   for (int j = 0; j < kDistortionDim; ++j) {
     Eigen::Map<Eigen::Matrix<double, kDistortionDim, 1>>
         distortion(cameraParameterBlocks_[1]->parameters());
     distortion[j] += h;
     Eigen::VectorXd intrinsics = refIntrinsics;
     intrinsics[j + kProjIntrinsicDim] += h;
     cameraGeometryBase_->setIntrinsics(intrinsics);
     costFunctionPtr_->EvaluateWithMinimalJacobians(
         parameters_, residual.data(), nullptr, nullptr);
     de_dDistortion->col(j) = (residual - refResidual_) / h;
     // reset
     distortion[j] = refDistortion[j];
     cameraGeometryBase_->setIntrinsics(refIntrinsics);
   }
 }

 void computeNumericJacobianForReadoutTime(
     Eigen::Matrix<double, Eigen::Dynamic, 1>* de_dtr) {
   Eigen::VectorXd residual(krd_);
   double refReadoutTime = cameraParameterBlocks_[2]->parameters()[0];
   cameraParameterBlocks_[2]->parameters()[0] += h;
   pointDataPtr_->computePoseAndVelocityAtObservation();
   bool useFirstEstimate = true;
   pointDataPtr_->computePoseAndVelocityForJacobians(useFirstEstimate);
   int cameraObservationModelId = 2;
   pointDataPtr_->computeSharedJacobians(cameraObservationModelId);
   costFunctionPtr_->EvaluateWithMinimalJacobians(
         parameters_, residual.data(), nullptr, nullptr);
   *de_dtr = (residual - refResidual_) / h;
   // reset
   cameraParameterBlocks_[2]->parameters()[0] = refReadoutTime;
   pointDataPtr_->computePoseAndVelocityAtObservation();
   pointDataPtr_->computePoseAndVelocityForJacobians(useFirstEstimate);
   pointDataPtr_->computeSharedJacobians(cameraObservationModelId);
 }

 void computeNumericJacobianForCameraDelay(
     Eigen::Matrix<double, Eigen::Dynamic, 1>* de_dtd) {
   Eigen::VectorXd residual(krd_);
   double refCameraDelay = cameraParameterBlocks_[3]->parameters()[0];
   cameraParameterBlocks_[3]->parameters()[0] += h;
   pointDataPtr_->computePoseAndVelocityAtObservation();
   bool useFirstEstimate = true;
   pointDataPtr_->computePoseAndVelocityForJacobians(useFirstEstimate);
   int cameraObservationModelId = 2;
   pointDataPtr_->computeSharedJacobians(cameraObservationModelId);
   costFunctionPtr_->EvaluateWithMinimalJacobians(
         parameters_, residual.data(), nullptr, nullptr);
   *de_dtd = (residual - refResidual_) / h;
   // reset
   cameraParameterBlocks_[3]->parameters()[0] = refCameraDelay;
   pointDataPtr_->computePoseAndVelocityAtObservation();
   pointDataPtr_->computePoseAndVelocityForJacobians(useFirstEstimate);
   pointDataPtr_->computeSharedJacobians(cameraObservationModelId);
 }

 std::shared_ptr<const okvis::ceres::ErrorInterface> costFunctionPtr_;
 std::vector<std::shared_ptr<okvis::ceres::PoseParameterBlock>> poseBlocks_;
 std::shared_ptr<swift_vio::PointLandmark> pointLandmark_;

 /// camera parameters
 std::shared_ptr<okvis::ceres::PoseParameterBlock> extrinsicBlock_;
 // projection intrinsics, distortion, readout time, time offset
 std::vector<std::shared_ptr<okvis::ceres::ParameterBlock>>
     cameraParameterBlocks_;
 std::vector<std::shared_ptr<okvis::ceres::SpeedAndBiasParameterBlock>>
     speedAndBiasBlocks_;

 // linearization points.
 std::vector<std::shared_ptr<Eigen::Matrix<double, 6, 1>>>
     positionAndVelocityLpList_;
 const double* const parameters_[12];
 std::shared_ptr<swift_vio::PointSharedData> pointDataPtr_;
 std::shared_ptr<DistortedPinholeCameraGeometry> cameraGeometryBase_;
 Eigen::Vector2d imagePoint_;
 const int observationIndex_; // 0 main anchor, 1 associate anchor, 2 other observing frame
 const int krd_;
 Eigen::VectorXd refResidual_;
 static const double h;
};

const double NumericJacobian::h = 1e-5;

void CameraObservationJacobianTest::verifyJacobians(
    std::shared_ptr<const okvis::ceres::ErrorInterface> costFunctionPtr,
    int observationIndex,
    int landmarkIndex,
    std::shared_ptr<swift_vio::PointSharedData> pointDataPtr,
    std::shared_ptr<DistortedPinholeCameraGeometry> cameraGeometry,
    const Eigen::Vector2d& imagePoint) const {
  // compare numerical and analytic Jacobians and residuals
  std::cout << "Main anchor 0 associate anchor 1 observationIndex "
            << observationIndex << " landmark index " << landmarkIndex << std::endl;
  const double* const parameters[] = {
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


  const int krd = okvis::cameras::CameraObservationModelResidualDim(
      coo_.cameraObservationModelId);
  Eigen::VectorXd residuals(krd);
  Eigen::AlignedVector<Eigen::Matrix<double, -1, 7, Eigen::RowMajor>> de_deltaTWB(
      3, Eigen::Matrix<double, -1, 7, Eigen::RowMajor>(krd, 7));
  Eigen::AlignedVector<Eigen::Matrix<double, -1, 9, Eigen::RowMajor>> de_dSpeedAndBias(
      3, Eigen::Matrix<double, -1, 9, Eigen::RowMajor>(krd, 9));
  Eigen::AlignedVector<Eigen::Matrix<double, -1, 6, Eigen::RowMajor>>
      de_deltaTWB_minimal(
          3, Eigen::Matrix<double, -1, 6, Eigen::RowMajor>(krd, 6));
  Eigen::AlignedVector<Eigen::Matrix<double, -1, 9, Eigen::RowMajor>>
      de_dSpeedAndBias_minimal(
          3, Eigen::Matrix<double, -1, 9, Eigen::RowMajor>(krd, 9));

  Eigen::Matrix<double, -1, 6, Eigen::RowMajor> de_dPoint(krd, 6);
  Eigen::Matrix<double, -1, 3, Eigen::RowMajor> de_dPoint_minimal(krd, 3);
  Eigen::Matrix<double, -1, 7, Eigen::RowMajor> de_dExtrinsic(krd, 7);
  Eigen::Matrix<double, -1, kExtrinsicMinimalDim, Eigen::RowMajor> de_dExtrinsic_minimal(
      krd, kExtrinsicMinimalDim);

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

  double* jacobians[] = {de_deltaTWB[0].data(),
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

  double* jacobiansMinimal[] = {de_deltaTWB_minimal[0].data(),
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
  NumericJacobian nj(costFunctionPtr, poseBlocks_, visibleLandmarks_[landmarkIndex],
                     extrinsicBlock_, cameraParameterBlocks_, speedAndBiasBlocks_,
                     positionAndVelocityLp_, pointDataPtr, cameraGeometry,
                     imagePoint, observationIndex, krd);

  // main and associate frame
  for (int j = 1; j < 3; ++j) {
    nj.computeNumericJacobianForPose(j-1, &de_deltaTWB_numeric[j],
                                     &de_deltaTWB_minimal_numeric[j]);
    nj.computeNumericJacobianForSpeedAndBias(
        j-1, &de_dSpeedAndBias_numeric[j]);
  }
  // observing frame
  nj.computeNumericJacobianForPose(observationIndex, &de_deltaTWB_numeric[0],
                                   &de_deltaTWB_minimal_numeric[0]);
  nj.computeNumericJacobianForSpeedAndBias(
      observationIndex, &de_dSpeedAndBias_numeric[0]);

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
  ARE_MATRICES_CLOSE(de_deltaTWB_numeric[2].leftCols<3>(), de_deltaTWB[2].leftCols<3>(), poseTol);
  if (observationIndex != 1) {
    EXPECT_LT(de_deltaTWB_numeric[2].rightCols<4>().lpNorm<Eigen::Infinity>(), 5e-2);
  }
  ARE_MATRICES_CLOSE(de_deltaTWB_minimal_numeric[0], de_deltaTWB_minimal[0],
                     1e-2);
  ARE_MATRICES_CLOSE(de_deltaTWB_minimal_numeric[1], de_deltaTWB_minimal[1],
                     5e-3);
  ARE_MATRICES_CLOSE(de_deltaTWB_minimal_numeric[2].leftCols<3>(), de_deltaTWB_minimal[2].leftCols<3>(),
                     5e-3);
  if (observationIndex != 1) {
    EXPECT_LT(de_deltaTWB_minimal_numeric[2].rightCols<3>().lpNorm<Eigen::Infinity>(), 5e-2);
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

void CameraObservationJacobianTest::solveAndCheck() {
  ::ceres::Solver::Options options;
  // options.check_gradients=true;
  // options.numeric_derivative_relative_step_size = 1e-6;
  // options.gradient_check_relative_precision=1e-2;
  options.minimizer_progress_to_stdout = false;
  ::FLAGS_stderrthreshold =
      google::WARNING;  // enable console warnings (Jacobian verification)
  ::ceres::Solver::Summary summary;
  Solve(options, problem_.get(), &summary);

  // print some infos about the optimization
  LOG(INFO) << summary.BriefReport();

  LOG(INFO) << "initial T_WB : " << initial_T_WB_.T() << "\n"
            << "optimized T_WB : " << poseBlocks_[2]->estimate().T() << "\n"
            << "correct T_WB : " << expected_T_WB_.T();

  // make sure it converged
  EXPECT_LT(2 * (expected_T_WB_.q() * poseBlocks_[2]->estimate().q().inverse())
                    .vec()
                    .norm(),
            1e-2)
      << "quaternions not close enough";
  EXPECT_LT((expected_T_WB_.r() - poseBlocks_[2]->estimate().r()).norm(), 1e-1)
      << "translation not close enough";
}

void setupPoseOptProblem(bool perturbPose, bool rollingShutter,
                         bool noisyKeypoint, int cameraObservationModelId,
                         bool R_WCnmf = false) {
  // srand((unsigned int) time(0));
  CameraObservationOptions coo;
  coo.perturbPose = perturbPose;
  coo.rollingShutter = rollingShutter;
  coo.noisyKeypoint = noisyKeypoint;
  coo.cameraObservationModelId = cameraObservationModelId;

  CameraObservationJacobianTest jacTest(coo);

  okvis::ImuParameters imuParameters;
  double imuFreq = imuParameters.rate;
  Eigen::Vector3d ginw(0, 0, -imuParameters.g);
  okvis::Time startEpoch(1.0);
  okvis::Time endEpoch(3.0);  // poses separated by 1 sec.
  std::shared_ptr<simul::CircularSinusoidalTrajectory> cameraMotion(
      new simul::RoundedSquare(imuFreq, ginw, okvis::Time(0, 0), 1.0, 6.0, 0.8));

  okvis::ImuMeasurementDeque imuMeasurements;
  cameraMotion->getTrueInertialMeasurements(startEpoch - okvis::Duration(1),
                                            endEpoch + okvis::Duration(1),
                                            imuMeasurements);
  jacTest.addNavStatesAndExtrinsic(cameraMotion, startEpoch);

  double timeOffset(0.0);
  jacTest.addImuAugmentedParameterBlocks(startEpoch);
  jacTest.addImuInfo(imuMeasurements, imuParameters, timeOffset);

  std::shared_ptr<DistortedPinholeCameraGeometry> cameraGeometry =
      std::static_pointer_cast<DistortedPinholeCameraGeometry>(
          DistortedPinholeCameraGeometry::createTestObject());

  Eigen::VectorXd intrinsicParams;
  cameraGeometry->getIntrinsics(intrinsicParams);
  double imageHeight = cameraGeometry->imageHeight();

  jacTest.addCameraParameterBlocks(intrinsicParams, startEpoch, timeOffset);

  const size_t numberTrials = 100;
  std::vector<std::shared_ptr<swift_vio::PointLandmark>> visibleLandmarks;
  Eigen::AlignedVector<Eigen::AlignedVector<Eigen::Vector2d>> pointObservationList;
  visibleLandmarks.reserve(numberTrials);
  pointObservationList.reserve(numberTrials);

  Eigen::AlignedVector<okvis::kinematics::Transformation> true_T_WB_list =
      jacTest.truePoses();
  okvis::kinematics::Transformation T_BC = jacTest.extrinsicBlock()->estimate();

  for (size_t i = 1; i < numberTrials; ++i) {
    Eigen::Vector4d pCm = cameraGeometry->createRandomVisibleHomogeneousPoint(
        double(i % 10) * 3 + 2.0);
    okvis::kinematics::Transformation T_WBm = true_T_WB_list[0];
    Eigen::AlignedVector<Eigen::Vector3d> observationsxy1;
    observationsxy1.reserve(3);
    Eigen::AlignedVector<Eigen::Vector2d> imageObservations;
    imageObservations.reserve(3);
    std::vector<size_t> anchorObsIndices{0, 1};

    bool projectOk = true;
    for (int j = 0; j < 3; ++j) {
      okvis::kinematics::Transformation T_WBj = true_T_WB_list[j];
      Eigen::Vector4d pCj = (T_WBj * T_BC).inverse() * (T_WBm * (T_BC * pCm));
      Eigen::Vector2d imagePoint;
      okvis::cameras::CameraBase::ProjectionStatus status =
          cameraGeometry->projectHomogeneous(pCj, &imagePoint);
      if (status != okvis::cameras::CameraBase::ProjectionStatus::Successful) {
        projectOk = false;
        break;
      }

      Eigen::Vector3d xy1;
      bool backProjectOk = cameraGeometry->backProject(imagePoint, &xy1);
      if (!backProjectOk) {
        projectOk = false;
        break;
      }
      if (coo.noisyKeypoint) {
        imagePoint += Eigen::Vector2d::Random();
      }
      observationsxy1.push_back(xy1);
      imageObservations.push_back(imagePoint);
    }
    if (!projectOk) {
      continue;
    }
    std::shared_ptr<swift_vio::PointLandmark> pl(new swift_vio::PointLandmark(
        swift_vio::ParallaxAngleParameterization::kModelId));
    Eigen::AlignedVector<okvis::kinematics::Transformation> T_BCs{T_BC};
    std::vector<size_t> camIndices(true_T_WB_list.size(), 0u);
    Eigen::AlignedVector<okvis::kinematics::Transformation> T_WCa_list{
      true_T_WB_list[anchorObsIndices[0]] * T_BC};
    swift_vio::TriangulationStatus status = pl->initialize(
        true_T_WB_list, observationsxy1, T_BCs, T_WCa_list,
        camIndices, anchorObsIndices);
    if (!status.triangulationOk) {
      continue;
    }
    pointObservationList.push_back(imageObservations);
    visibleLandmarks.push_back(pl);
  }

  int numberLandmarks = visibleLandmarks.size();
  std::vector<uint64_t> frameIds = jacTest.frameIds();
  std::vector<std::shared_ptr<okvis::ceres::PoseParameterBlock>> poseBlocks =
      jacTest.poseBlocks();
  LOG(INFO) << "Number landmarks " << numberLandmarks;
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

    std::vector<okvis::AnchorFrameIdentifier> anchorIds{{frameIds[0], 0, 0}, {frameIds[1], 0, 1}};
    pointDataPtr->setAnchors(anchorIds);

    bool useFirstEstimate = true;
    pointDataPtr->computePoseAndVelocityForJacobians(useFirstEstimate);
    pointDataPtr->computeSharedJacobians(cameraObservationModelId);

    // add landmark observations (residuals) to the problem.
    for (int observationIndex = 0; observationIndex < 3; ++observationIndex) {
      std::shared_ptr<::ceres::CostFunction> costFunctionPtr;
      std::shared_ptr<okvis::ceres::ErrorInterface> errorInterface;
      switch (cameraObservationModelId) {
        case okvis::cameras::kChordalDistanceId: {
          std::shared_ptr<okvis::ceres::ChordalDistance<DistortedPinholeCameraGeometry,
                                        okvis::ProjectionOptFXY_CXY,
                                        okvis::Extrinsic_p_CB>>
              localCostFunctionPtr(
                  new okvis::ceres::ChordalDistance<
                      DistortedPinholeCameraGeometry,
                      okvis::ProjectionOptFXY_CXY, okvis::Extrinsic_p_CB>(
                      cameraGeometry, pointObservationList[i][observationIndex],
                      covariance, observationIndex, pointDataPtr, R_WCnmf));
          costFunctionPtr = std::static_pointer_cast<::ceres::CostFunction>(localCostFunctionPtr);
          errorInterface = std::static_pointer_cast<okvis::ceres::ErrorInterface>(localCostFunctionPtr);
          break;
        }
        case okvis::cameras::kReprojectionErrorWithPapId: {
          std::shared_ptr<okvis::ceres::ReprojectionErrorWithPap<DistortedPinholeCameraGeometry,
                                                 okvis::ProjectionOptFXY_CXY,
                                                 okvis::Extrinsic_p_CB>>
              localCostFunctionPtr(
                  new okvis::ceres::ReprojectionErrorWithPap<
                      DistortedPinholeCameraGeometry,
                      okvis::ProjectionOptFXY_CXY, okvis::Extrinsic_p_CB>(
                      cameraGeometry, pointObservationList[i][observationIndex],
                      covariance, observationIndex, pointDataPtr));
          costFunctionPtr = std::static_pointer_cast<::ceres::CostFunction>(localCostFunctionPtr);
          errorInterface = std::static_pointer_cast<okvis::ceres::ErrorInterface>(localCostFunctionPtr);
          break;
        }
      }
      CHECK(costFunctionPtr) << "Null cost function not allowed!";
      if (observationIndex == 2) {
        // TODO(jhuai): when one of the anchor is the observing frame,
        // duplicate parameters are not allowed by ceres cost function.
        jacTest.addResidual(costFunctionPtr, observationIndex, i);
      }
      if (i % 20 == 0) {
        jacTest.verifyJacobians(errorInterface, observationIndex, i, pointDataPtr, cameraGeometry,
                                pointObservationList[i][observationIndex]);
      }
    }
  }
  // TODO(jhuai): Pure virtual function called and segfault at cost_function->Evaluate() inside ceres solver.
//  jacTest.solveAndCheck();
}
}  // namespace

TEST(ReprojectionErrorWithPap, NoiseFree) {
  setupPoseOptProblem(false, false, false,
                      okvis::cameras::kReprojectionErrorWithPapId);
}

TEST(ChordalDistance, NoiseFree) {
  setupPoseOptProblem(false, false, false, okvis::cameras::kChordalDistanceId);
}

TEST(ChordalDistanceRWC, NoiseFree) {
  setupPoseOptProblem(false, false, false, okvis::cameras::kChordalDistanceId,
                      true);
}

TEST(ReprojectionErrorWithPap, Noisy) {
  setupPoseOptProblem(true, true, true,
                      okvis::cameras::kReprojectionErrorWithPapId);
}

TEST(ChordalDistance, Noisy) {
  setupPoseOptProblem(true, true, true,
                      okvis::cameras::kChordalDistanceId);
}
