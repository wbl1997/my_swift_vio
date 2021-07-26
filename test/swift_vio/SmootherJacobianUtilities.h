#ifndef SMOOTHERJACOBIANUTILITIES_H
#define SMOOTHERJACOBIANUTILITIES_H

#include <ceres/ceres.h>

#include <swift_vio/CameraRig.hpp>
#include <swift_vio/memory.h>

#include <swift_vio/ExtrinsicModels.hpp>
#include <swift_vio/FrameTypedefs.hpp>
#include <swift_vio/ParallaxAnglePoint.hpp>
#include <swift_vio/PointLandmark.hpp>
#include <swift_vio/PointLandmarkModels.hpp>
#include <swift_vio/ProjParamOptModels.hpp>
#include <swift_vio/ceres/CameraTimeParamBlock.hpp>
#include <swift_vio/ceres/ChordalDistance.hpp>
#include <swift_vio/ceres/EuclideanParamBlock.hpp>
#include <swift_vio/ceres/EuclideanParamBlockSized.hpp>
#include <swift_vio/imu/BoundedImuDeque.hpp>

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
namespace swift_vio {
class CameraObservationOptions {
public:
  CameraObservationOptions()
      : perturbPose(false), rollingShutter(false), noisyKeypoint(false),
        cameraObservationModelId(swift_vio::cameras::kChordalDistanceId),
        projOptModelName("FXY_CXY"), extrinsicOptModelName("P_BC_Q_BC") {}

  bool perturbPose;
  bool rollingShutter;
  bool noisyKeypoint;
  int cameraObservationModelId;

  std::string projOptModelName;
  std::string extrinsicOptModelName;
};

typedef okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>
    DistortedPinholeCameraGeometry;
static const int kDistortionDim =
    DistortedPinholeCameraGeometry::distortion_t::NumDistortionIntrinsics;
static const int kProjIntrinsicDim =
    swift_vio::ProjectionOptFXY_CXY::kNumParams;
static const int kExtrinsicMinimalDim = 3;

static const Eigen::Matrix2d kCovariance = Eigen::Matrix2d::Identity() / 0.36;
static const Eigen::Matrix2d kSquareRootInformation =
    Eigen::Matrix2d::Identity() * 0.6;

class CameraObservationJacobianTest {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CameraObservationJacobianTest(CameraObservationOptions coo)
      : coo_(coo), nextBlockIndex_(1u),
        papLocalParameterization_(
            new swift_vio::ParallaxAngleParameterization()) {
    ::ceres::Problem::Options problemOptions;
    problemOptions.local_parameterization_ownership =
        ::ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
    problemOptions.loss_function_ownership =
        ::ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
    problemOptions.cost_function_ownership =
        ::ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
    problem_.reset(new ::ceres::Problem(problemOptions));
    int extrinsicModelId =
        swift_vio::ExtrinsicModelNameToId(coo_.extrinsicOptModelName);
    if (extrinsicModelId == 1) {
      extrinsicLocalParameterization_.reset(new swift_vio::Extrinsic_p_CB());
    } else {
      extrinsicLocalParameterization_.reset(
          new okvis::ceres::PoseLocalParameterization());
    }
    poseLocalParameterization_.reset(
        new okvis::ceres::PoseLocalParameterization());
    homogeneousPointLocalParameterization_.reset(
          new okvis::ceres::HomogeneousPointLocalParameterization());
    inverseDepthLocalParameterization_.reset(
          new swift_vio::InverseDepthParameterization());

    costFunctions_.reserve(200);
  }

  bool isZeroResidualExpected() const {
    return !(coo_.perturbPose || coo_.rollingShutter || coo_.noisyKeypoint);
  }

  std::vector<uint64_t> frameIds() const { return frameIds_; }

  Eigen::AlignedVector<okvis::kinematics::Transformation> truePoses() const {
    Eigen::AlignedVector<okvis::kinematics::Transformation> T_WB_list;
    T_WB_list.reserve(3);
    for (int j = 0; j < 3; ++j) {
      T_WB_list.push_back(poseBlocks_[j]->estimate());
    }
    T_WB_list[2] = ref_T_WB_;
    return T_WB_list;
  }

  std::vector<std::shared_ptr<okvis::ceres::PoseParameterBlock>>
  poseBlocks() const {
    return poseBlocks_;
  }

  inline okvis::Time getImageTimestamp(int observationIndex,
                                       int /*cameraIdx*/) const {
    return stateEpochs_[observationIndex] -
           okvis::Duration(tdAtCreationList_[observationIndex]);
  }

  okvis::Time stateEpoch(int frameIndex) const {
    return stateEpochs_[frameIndex];
  }

  uint64_t addNavStatesAndExtrinsic(
      std::shared_ptr<const simul::CircularSinusoidalTrajectory> cameraMotion,
      okvis::Time startEpoch, double timeGapBetweenStates);

  uint64_t addImuAugmentedParameterBlocks(okvis::Time stateEpoch);

  uint64_t addCameraParameterBlocks(const Eigen::VectorXd &intrinsicParams,
                                    okvis::Time startEpoch, double frameReadoutTime,
                                    double cameraTimeOffset);

  void createLandmarksAndObservations(
      std::shared_ptr<const DistortedPinholeCameraGeometry> cameraGeometry,
      std::vector<std::shared_ptr<swift_vio::PointLandmark>> *visibleLandmarks,
      Eigen::AlignedVector<Eigen::AlignedVector<Eigen::Vector2d>>
          *pointObservationList, int numberTrials);

  void addLandmark(std::shared_ptr<swift_vio::PointLandmark> pl);

  void addImuInfo(const okvis::ImuMeasurementDeque &entireImuList,
                  const okvis::ImuParameters &imuParameters, double cameraTimeOffset);

  void propagatePoseAndVelocityForMapPoint(
      std::shared_ptr<swift_vio::PointSharedData> pointDataPtr) const;

  void addResidual(std::shared_ptr<::ceres::CostFunction> costFunctionPtr,
                   int observationIndex, int landmarkIndex);

  void verifyJacobians(
      std::shared_ptr<const okvis::ceres::ErrorInterface> costFunctionPtr,
      int observationIndex, int landmarkIndex,
      std::shared_ptr<swift_vio::PointSharedData> pointDataPtr,
      std::shared_ptr<DistortedPinholeCameraGeometry> cameraGeometry,
      const Eigen::Vector2d &imagePoint) const;

  void verifyJacobiansAIDP(
      std::shared_ptr<const okvis::ceres::ErrorInterface> costFunctionPtr,
      int observationIndex, int landmarkIndex,
      std::shared_ptr<swift_vio::PointSharedData> pointDataPtr,
      std::shared_ptr<DistortedPinholeCameraGeometry> cameraGeometry,
      const Eigen::Vector2d &imagePoint) const;

  void verifyJacobiansHPP(
      std::shared_ptr<const okvis::ceres::ErrorInterface> costFunctionPtr,
      int observationIndex, int landmarkIndex,
      std::shared_ptr<swift_vio::PointSharedData> pointDataPtr,
      std::shared_ptr<DistortedPinholeCameraGeometry> cameraGeometry,
      const Eigen::Vector2d &imagePoint) const;

  void verifyJacobiansPAP(
      std::shared_ptr<const okvis::ceres::ErrorInterface> costFunctionPtr,
      int observationIndex, int landmarkIndex,
      std::shared_ptr<swift_vio::PointSharedData> pointDataPtr,
      std::shared_ptr<DistortedPinholeCameraGeometry> cameraGeometry,
      const Eigen::Vector2d &imagePoint) const;

  void solveAndCheck();

  CameraObservationOptions coo_;

private:
  // Details for the main host frame, associate host frame, and observing target frame.
  std::vector<okvis::Time> stateEpochs_;
  std::vector<uint64_t> frameIds_;
  std::vector<std::shared_ptr<okvis::ceres::PoseParameterBlock>> poseBlocks_;
  std::vector<std::shared_ptr<okvis::ceres::SpeedAndBiasParameterBlock>>
      speedAndBiasBlocks_;
  std::vector<std::shared_ptr<Eigen::Matrix<double, 6, 1>>>
      positionAndVelocityLin_; // linearization points
  std::vector<std::shared_ptr<okvis::ImuMeasurementDeque>> imuWindowList_;
  std::vector<double> tdAtCreationList_;  // time delay at creating a state + frame raw time by the camera = state time for the frame (per the IMU clock).

  okvis::kinematics::Transformation initial_T_WB_;  // initial pose for the target frame to the estimator.
  okvis::kinematics::Transformation ref_T_WB_;  // true pose for the target frame.

  std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>
      imuAugmentedBlocks_;
  okvis::ImuParameters imuParameters_;

  std::vector<std::shared_ptr<::ceres::CostFunction>> costFunctions_;

  /// camera parameters
  std::shared_ptr<okvis::ceres::PoseParameterBlock> extrinsicBlock_;
  // projection intrinsics, distortion, readout time, time offset
  std::vector<std::shared_ptr<okvis::ceres::ParameterBlock>>
      cameraParameterBlocks_;

  std::shared_ptr<::ceres::Problem> problem_;
  uint64_t nextBlockIndex_;

  std::vector<std::shared_ptr<swift_vio::PointLandmark>> visibleLandmarks_;

  std::shared_ptr<::ceres::LocalParameterization> papLocalParameterization_;
  std::shared_ptr<::ceres::LocalParameterization>
      extrinsicLocalParameterization_;
  std::shared_ptr<::ceres::LocalParameterization> poseLocalParameterization_;
  std::shared_ptr<::ceres::LocalParameterization> homogeneousPointLocalParameterization_;
  std::shared_ptr<::ceres::LocalParameterization> inverseDepthLocalParameterization_;
};

class NumericJacobianPAP {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  NumericJacobianPAP(
      std::shared_ptr<const okvis::ceres::ErrorInterface> costFunctionPtr,
      const std::vector<std::shared_ptr<okvis::ceres::PoseParameterBlock>>
          &poseBlocks,
      std::shared_ptr<swift_vio::PointLandmark> pointLandmark,
      std::shared_ptr<okvis::ceres::PoseParameterBlock> extrinsicBlock,
      const std::vector<std::shared_ptr<okvis::ceres::ParameterBlock>>
          &cameraParameterBlocks,
      const std::vector<
          std::shared_ptr<okvis::ceres::SpeedAndBiasParameterBlock>>
          &speedAndBiasBlocks,
      const std::vector<std::shared_ptr<Eigen::Matrix<double, 6, 1>>>
          &positionAndVelocityLpList,
      std::shared_ptr<swift_vio::PointSharedData> pointDataPtr,
      std::shared_ptr<DistortedPinholeCameraGeometry> cameraGeometry,
      Eigen::Vector2d imagePoint, int observationIndex, int krd)
      : costFunctionPtr_(costFunctionPtr), poseBlocks_(poseBlocks),
        pointLandmark_(pointLandmark), extrinsicBlock_(extrinsicBlock),
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
        pointDataPtr_(pointDataPtr), cameraGeometryBase_(cameraGeometry),
        imagePoint_(imagePoint), observationIndex_(observationIndex),
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
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          *de_deltaTWB,
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          *de_deltaTWB_minimal) {
    Eigen::VectorXd residual(krd_);
    okvis::kinematics::Transformation ref_T_WB =
        poseBlocks_.at(majIndex)->estimate();
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
    okvis::ceres::PoseLocalParameterization::liftJacobian(
        ref_T_WB.parameters().data(), jLift.data());
    *de_deltaTWB = (*de_deltaTWB_minimal) * jLift;
  }

  void computeNumericJacobianForSpeedAndBias(
      int majIndex, Eigen::Matrix<double, Eigen::Dynamic, -1, Eigen::RowMajor>
                        *de_dSpeedAndBias) {
    Eigen::VectorXd residual(krd_);
    okvis::SpeedAndBias refSpeedAndBias =
        speedAndBiasBlocks_.at(majIndex)->estimate();
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
      positionAndVelocityLpList_[majIndex]->tail<3>() =
          refSpeedAndBias.head<3>();
      pointDataPtr_->computePoseAndVelocityAtObservation();
      pointDataPtr_->computePoseAndVelocityForJacobians(useFirstEstimate);
      pointDataPtr_->computeSharedJacobians(cameraObservationModelId);
    }
  }

  /**
   * @brief computeReprojectionWithPapResidual mimics Evaluate() of
   * ReprojectionErrorWithPap.
   * @warning Only use it with ReprojectionErrorWithPap.
   * @param residual
   * @param de_dPap
   * @return residual computation Ok or not?
   */
  bool computeReprojectionWithPapResidual(
      Eigen::Vector2d *residual,
      Eigen::Matrix<double, 2, 3, Eigen::RowMajor> *de_dPap) const {
    swift_vio::ParallaxAnglePoint pap;
    pap.set(parameters_[3]);
    Eigen::Matrix<double, 3, 1> t_BC_B(parameters_[4][0], parameters_[4][1],
                                       parameters_[4][2]);
    Eigen::Quaternion<double> q_BC(parameters_[4][6], parameters_[4][3],
                                   parameters_[4][4], parameters_[4][5]);
    std::pair<Eigen::Vector3d, Eigen::Quaterniond> pair_T_BC(t_BC_B, q_BC);

    // compute N_{i,j}.
    okvis::kinematics::Transformation T_WBtij =
        pointDataPtr_->T_WBtij(observationIndex_);
    okvis::kinematics::Transformation T_WBtmi = pointDataPtr_->T_WBtij(0);
    okvis::kinematics::Transformation T_WBtai = pointDataPtr_->T_WBtij(1);

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

    Eigen::Matrix3d R_CtijW =
        pair_T_WCtij.second.toRotationMatrix().transpose();
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
      *de_dPap = kSquareRootInformation * de_dN * dN_dntheta;
    }

    bool projectOk = projectStatus ==
                     okvis::cameras::CameraBase::ProjectionStatus::Successful;
    Eigen::Vector2d error = imagePoint - imagePoint_;
    *residual = kSquareRootInformation * error;
    return projectOk;
  }

  void computeNumericJacobianForPoint(
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          *de_dPoint,
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          *de_dPoint_minimal);

  void computeNumericJacobianForExtrinsic(
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          *de_dExtrinsic,
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          *de_dExtrinsic_minimal) {
    Eigen::VectorXd residual(krd_);
    okvis::kinematics::Transformation ref_T_BC = extrinsicBlock_->estimate();
    Eigen::Matrix<double, 6, 1> delta;
    for (int j = 0; j < 3; ++j) {
      delta.setZero();
      delta[j] = h;
      std::pair<Eigen::Vector3d, Eigen::Quaterniond> T_BC(ref_T_BC.r(),
                                                          ref_T_BC.q());
      swift_vio::Extrinsic_p_CB::oplus(delta.data(), &T_BC);

      extrinsicBlock_->setEstimate(
          okvis::kinematics::Transformation(T_BC.first, T_BC.second));

      costFunctionPtr_->EvaluateWithMinimalJacobians(
          parameters_, residual.data(), nullptr, nullptr);

      de_dExtrinsic_minimal->col(j) = (residual - refResidual_) / h;

      // reset
      extrinsicBlock_->setEstimate(ref_T_BC);
    }
    Eigen::Matrix<double, 3, 7, Eigen::RowMajor> jLift;
    swift_vio::Extrinsic_p_CB::liftJacobian(ref_T_BC.parameters().data(),
                                            jLift.data());
    *de_dExtrinsic = (*de_dExtrinsic_minimal) * jLift;
  }

  void computeNumericJacobianForProjectionIntrinsic(
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          *de_dProjectionIntrinsic) {
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
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          *de_dDistortion) {
    Eigen::VectorXd residual(krd_);
    Eigen::Matrix<double, kDistortionDim, 1> refDistortion =
        Eigen::Map<Eigen::Matrix<double, kDistortionDim, 1>>(
            cameraParameterBlocks_[1]->parameters());
    Eigen::VectorXd refIntrinsics(kProjIntrinsicDim + kDistortionDim);
    cameraGeometryBase_->getIntrinsics(refIntrinsics);

    for (int j = 0; j < kDistortionDim; ++j) {
      Eigen::Map<Eigen::Matrix<double, kDistortionDim, 1>> distortion(
          cameraParameterBlocks_[1]->parameters());
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
      Eigen::Matrix<double, Eigen::Dynamic, 1> *de_dtr) {
    Eigen::VectorXd residual(krd_);
    double refReadoutTime = cameraParameterBlocks_[2]->parameters()[0];
    cameraParameterBlocks_[2]->parameters()[0] += h;
    pointDataPtr_->computePoseAndVelocityAtObservation();
    bool useFirstEstimate = true;
    pointDataPtr_->computePoseAndVelocityForJacobians(useFirstEstimate);
    int cameraObservationModelId = 2;
    pointDataPtr_->computeSharedJacobians(cameraObservationModelId);
    costFunctionPtr_->EvaluateWithMinimalJacobians(parameters_, residual.data(),
                                                   nullptr, nullptr);
    *de_dtr = (residual - refResidual_) / h;
    // reset
    cameraParameterBlocks_[2]->parameters()[0] = refReadoutTime;
    pointDataPtr_->computePoseAndVelocityAtObservation();
    pointDataPtr_->computePoseAndVelocityForJacobians(useFirstEstimate);
    pointDataPtr_->computeSharedJacobians(cameraObservationModelId);
  }

  void computeNumericJacobianForCameraDelay(
      Eigen::Matrix<double, Eigen::Dynamic, 1> *de_dtd) {
    Eigen::VectorXd residual(krd_);
    double refCameraDelay = cameraParameterBlocks_[3]->parameters()[0];
    cameraParameterBlocks_[3]->parameters()[0] += h;
    pointDataPtr_->computePoseAndVelocityAtObservation();
    bool useFirstEstimate = true;
    pointDataPtr_->computePoseAndVelocityForJacobians(useFirstEstimate);
    int cameraObservationModelId = 2;
    pointDataPtr_->computeSharedJacobians(cameraObservationModelId);
    costFunctionPtr_->EvaluateWithMinimalJacobians(parameters_, residual.data(),
                                                   nullptr, nullptr);
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
  const double *const parameters_[12];
  std::shared_ptr<swift_vio::PointSharedData> pointDataPtr_;
  std::shared_ptr<DistortedPinholeCameraGeometry> cameraGeometryBase_;
  Eigen::Vector2d imagePoint_;
  const int observationIndex_; // 0 main anchor, 1 associate anchor, 2 other
                               // observing frame
  const int krd_;
  Eigen::VectorXd refResidual_;
  static const double h;
};
}  // namespace swift_vio
#endif // SMOOTHERJACOBIANUTILITIES_H
