
/**
 * @file implementation/HybridFilter.hpp
 * @brief Header implementation file for the HybridFilter class.
 * @author Jianzhu Huai
 */
#include <glog/logging.h>

#include <msckf/ChordalDistance.hpp>
#include <msckf/EpipolarFactor.hpp>
#include <msckf/PointLandmark.hpp>
#include <msckf/PointLandmarkModels.hpp>
#include <msckf/ProjParamOptModels.hpp>
#include <msckf/RsReprojectionError.hpp>
#include <msckf/ReprojectionErrorWithPap.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
template <class CameraGeometry, class ProjectionIntrinsicModel,
          class ExtrinsicModel, class PointLandmarkModel, class ImuModel>
msckf::MeasurementJacobianStatus
HybridFilter::computeCameraObservationJacobians(
    const msckf::PointLandmark& pointLandmark,
    std::shared_ptr<const okvis::cameras::CameraBase> baseCameraGeometry,
    const Eigen::Vector2d& obs, const Eigen::Matrix2d& obsCov,
    int observationIndex,
    std::shared_ptr<const msckf::PointSharedData> pointDataPtr,
    Eigen::MatrixXd* J_X, Eigen::Matrix<double, Eigen::Dynamic, 3>* J_pfi,
    Eigen::Matrix<double, Eigen::Dynamic, 2>* J_n, Eigen::VectorXd* residual) const {
  std::shared_ptr<okvis::ceres::ErrorInterface> observationError;
  msckf::MeasurementJacobianStatus status =
      msckf::MeasurementJacobianStatus::GeneralProjectionFailed;
  std::shared_ptr<const okvis::ceres::ParameterBlock> poseParamBlockPtr =
      pointDataPtr->poseParameterBlockPtr(observationIndex);
  std::shared_ptr<const okvis::ceres::ParameterBlock> sbParamBlockPtr =
      pointDataPtr->speedAndBiasParameterBlockPtr(observationIndex);
  uint64_t poseId = pointDataPtr->frameId(observationIndex);
  int camIdx = pointDataPtr->cameraIndex(observationIndex);
  auto statesIter = statesMap_.find(poseId);
  const okvis::Estimator::States& stateInQuestion = statesIter->second;
  uint64_t extrinsicBlockId =
      stateInQuestion.sensors.at(okvis::Estimator::SensorStates::Camera)
          .at(camIdx)
          .at(okvis::Estimator::CameraSensorStates::T_SCi)
          .id;
  std::shared_ptr<const okvis::ceres::ParameterBlock> extrinsicParamBlockPtr =
      mapPtr_->parameterBlockPtr(extrinsicBlockId);

  uint64_t intrinsicId =
      stateInQuestion.sensors.at(okvis::Estimator::SensorStates::Camera)
          .at(camIdx)
          .at(okvis::Estimator::CameraSensorStates::Intrinsics)
          .id;
  std::shared_ptr<const okvis::ceres::ParameterBlock> projectionParamBlockPtr =
      mapPtr_->parameterBlockPtr(intrinsicId);

  uint64_t distortionId =
      stateInQuestion.sensors.at(okvis::Estimator::SensorStates::Camera)
          .at(camIdx)
          .at(okvis::Estimator::CameraSensorStates::Distortion)
          .id;
  std::shared_ptr<const okvis::ceres::ParameterBlock> distortionParamBlockPtr =
      mapPtr_->parameterBlockPtr(distortionId);

  std::shared_ptr<const okvis::ceres::ParameterBlock> tdParamBlockPtr =
      pointDataPtr->cameraTimeDelayParameterBlockPtr(camIdx);
  std::shared_ptr<const okvis::ceres::ParameterBlock> trParamBlockPtr =
      pointDataPtr->frameReadoutTimeParameterBlockPtr(camIdx);
  std::shared_ptr<const CameraGeometry> argCameraGeometry =
      std::static_pointer_cast<const CameraGeometry>(baseCameraGeometry);

  switch (cameraObservationModelId_) {
    case okvis::cameras::kReprojectionErrorId: {
      typedef okvis::ceres::RsReprojectionError<
          CameraGeometry, ProjectionIntrinsicModel, ExtrinsicModel,
          PointLandmarkModel, ImuModel>
          RsReprojectionErrorModel;
      std::shared_ptr<const okvis::ImuMeasurementDeque> imuMeasDequePtr =
          statesIter->second.imuReadingWindow;
      CHECK_GT(imuMeasDequePtr->size(), 0u)
          << "The IMU measurement does not exist";
      std::shared_ptr<const Eigen::Matrix<double, 6, 1>> posVelFirstEstimate =
          statesIter->second.linearizationPoint;
      okvis::Time stateEpoch = statesIter->second.timestamp;
      double tdAtCreation = statesIter->second.tdAtCreation;
      double gravity = pointDataPtr->gravityNorm();
      observationError.reset(new RsReprojectionErrorModel(
          argCameraGeometry, obs, obsCov, imuMeasDequePtr, posVelFirstEstimate,
          stateEpoch, tdAtCreation, gravity));
      double const* const parameters[] = {poseParamBlockPtr->parameters(),
                                          pointLandmark.data(),
                                          extrinsicParamBlockPtr->parameters(),
                                          projectionParamBlockPtr->parameters(),
                                          distortionParamBlockPtr->parameters(),
                                          trParamBlockPtr->parameters(),
                                          tdParamBlockPtr->parameters(),
                                          sbParamBlockPtr->parameters()};
      Eigen::Matrix<double, 2, 7, Eigen::RowMajor> duv_deltaTWS;
      Eigen::Matrix<double, 2, 6, Eigen::RowMajor> duv_deltaTWS_minimal;
      Eigen::Matrix<double, 2, 4, Eigen::RowMajor> duv_deltahpW;
      Eigen::Matrix<double, 2, 3, Eigen::RowMajor> duv_deltahpW_minimal;
      Eigen::Matrix<double, 2, 7, Eigen::RowMajor> duv_dExtrinsic;
      Eigen::Matrix<double, 2, ExtrinsicModel::kNumParams, Eigen::RowMajor>
          duv_dExtrinsic_minimal;
      typename RsReprojectionErrorModel::ProjectionIntrinsicJacType
          duv_proj_intrinsic;
      typename RsReprojectionErrorModel::DistortionJacType duv_distortion;
      Eigen::Matrix<double, 2, 1> duv_tr;
      typename RsReprojectionErrorModel::ProjectionIntrinsicJacType
          duv_proj_intrinsic_minimal;
      typename RsReprojectionErrorModel::DistortionJacType
          duv_distortion_minimal;
      Eigen::Matrix<double, 2, 1> duv_tr_minimal;
      Eigen::Matrix<double, 2, 1> duv_td;
      Eigen::Matrix<double, 2, 1> duv_td_minimal;
      Eigen::Matrix<double, 2, 9, Eigen::RowMajor> duv_sb;
      Eigen::Matrix<double, 2, 9, Eigen::RowMajor> duv_sb_minimal;
      double* jacobians[] = {duv_deltaTWS.data(),   duv_deltahpW.data(),
                             duv_dExtrinsic.data(), duv_proj_intrinsic.data(),
                             duv_distortion.data(), duv_tr.data(),
                             duv_td.data(),         duv_sb.data()};
      double* jacobiansMinimal[] = {
          duv_deltaTWS_minimal.data(),   duv_deltahpW_minimal.data(),
          duv_dExtrinsic_minimal.data(), duv_proj_intrinsic_minimal.data(),
          duv_distortion_minimal.data(), duv_tr_minimal.data(),
          duv_td_minimal.data(),         duv_sb_minimal.data()};
      bool evaluateOk = observationError->EvaluateWithMinimalJacobians(
          parameters, residual->data(), jacobians, jacobiansMinimal);
      if (evaluateOk) {
        status = msckf::MeasurementJacobianStatus::Successful;
      }
      int cameraParamsDim = cameraParamsMinimalDimen();
      J_X->setZero();
      if (fixCameraExtrinsicParams_[camIdx]) {
        if (fixCameraIntrinsicParams_[camIdx]) {
          J_X->topLeftCorner(2, cameraParamsDim) << duv_td_minimal,
              duv_tr_minimal;
        } else {
          J_X->topLeftCorner(2, cameraParamsDim) << duv_proj_intrinsic_minimal,
              duv_distortion_minimal, duv_td_minimal, duv_tr_minimal;
        }
      } else {
        if (fixCameraIntrinsicParams_[camIdx]) {
          J_X->topLeftCorner(2, cameraParamsDim) << duv_dExtrinsic_minimal,
              duv_td_minimal, duv_tr_minimal;
        } else {
          J_X->topLeftCorner(2, cameraParamsDim) << duv_dExtrinsic_minimal,
              duv_proj_intrinsic_minimal, duv_distortion_minimal,
              duv_td_minimal, duv_tr_minimal;
        }
      }

      int orderInCov = stateInQuestion.orderInCov;
      J_X->block<2, kClonedStateMinimalDimen>(
          0,
          cameraParamsDim + kClonedStateMinimalDimen * orderInCov)
          << duv_deltaTWS_minimal,
          duv_sb_minimal.topLeftCorner<2, 3>();
      *J_pfi = duv_deltahpW.topLeftCorner<2, 3>();
      *J_n = Eigen::Matrix2d::Identity();
      *residual = -(*residual);
      break;
    }
    case okvis::cameras::kChordalDistanceId: {
      typedef okvis::ceres::ChordalDistance<
          CameraGeometry, ProjectionIntrinsicModel, ExtrinsicModel,
          msckf::ParallaxAngleParameterization, ImuModel>
          CameraErrorModel;
      bool R_WCnmf = false;
      observationError.reset(new CameraErrorModel(
          argCameraGeometry, obs, obsCov, observationIndex, pointDataPtr, R_WCnmf));
      std::vector<int> anchorObservationIndices =
          pointDataPtr->anchorObservationIds();
      std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>
          anchorPoseBlockPtrs;
      anchorPoseBlockPtrs.reserve(2);
      std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>
          anchorSpeedAndBiasBlockPtrs;
      anchorSpeedAndBiasBlockPtrs.reserve(2);
      for (auto anchorObsId : anchorObservationIndices) {
        anchorPoseBlockPtrs.push_back(
            pointDataPtr->poseParameterBlockPtr(anchorObsId));
        anchorSpeedAndBiasBlockPtrs.push_back(
            pointDataPtr->speedAndBiasParameterBlockPtr(anchorObsId));
      }

      double const* const parameters[] = {
          poseParamBlockPtr->parameters(),
          anchorPoseBlockPtrs[0]->parameters(),
          anchorPoseBlockPtrs[1]->parameters(),
          pointLandmark.data(),
          extrinsicParamBlockPtr->parameters(),
          projectionParamBlockPtr->parameters(),
          distortionParamBlockPtr->parameters(),
          trParamBlockPtr->parameters(),
          tdParamBlockPtr->parameters(),
          sbParamBlockPtr->parameters(),
          anchorSpeedAndBiasBlockPtrs[0]->parameters(),
          anchorSpeedAndBiasBlockPtrs[1]->parameters()};

      const int krd = CameraErrorModel::kNumResiduals;
      const int kPoseNumber = 3;
      // The elements are de_dTWBj, de_dTWBmi, de_dTWBai.
      std::vector<Eigen::Matrix<double, krd, 7, Eigen::RowMajor>,
                  Eigen::aligned_allocator<
                      Eigen::Matrix<double, krd, 7, Eigen::RowMajor>>>
          de_dTWB(kPoseNumber);
      std::vector<Eigen::Matrix<double, krd, 6, Eigen::RowMajor>,
                  Eigen::aligned_allocator<
                      Eigen::Matrix<double, krd, 6, Eigen::RowMajor>>>
          de_dTWB_minimal(kPoseNumber);
      std::vector<Eigen::Matrix<double, krd, 9, Eigen::RowMajor>,
                  Eigen::aligned_allocator<
                      Eigen::Matrix<double, krd, 9, Eigen::RowMajor>>>
          de_dSpeedAndBias(kPoseNumber);
      std::vector<Eigen::Matrix<double, krd, 9, Eigen::RowMajor>,
                  Eigen::aligned_allocator<
                      Eigen::Matrix<double, krd, 9, Eigen::RowMajor>>>
          de_dSpeedAndBias_minimal(kPoseNumber);

      Eigen::Matrix<double, krd, PointLandmarkModel::kGlobalDim,
                    Eigen::RowMajor>
          de_dPoint;
      Eigen::Matrix<double, krd, PointLandmarkModel::kLocalDim, Eigen::RowMajor>
          de_dPoint_minimal;
      Eigen::Matrix<double, krd, ExtrinsicModel::kGlobalDim, Eigen::RowMajor>
          de_dExtrinsic;
      Eigen::Matrix<double, krd, ExtrinsicModel::kNumParams, Eigen::RowMajor>
          de_dExtrinsic_minimal;

      typename CameraErrorModel::ProjectionIntrinsicJacType de_dproj_intrinsic;
      typename CameraErrorModel::ProjectionIntrinsicJacType
          de_dproj_intrinsic_minimal;
      typename CameraErrorModel::DistortionJacType de_ddistortion;
      typename CameraErrorModel::DistortionJacType de_ddistortion_minimal;
      Eigen::Matrix<double, krd, 1> de_dtr;
      Eigen::Matrix<double, krd, 1> de_dtr_minimal;
      Eigen::Matrix<double, krd, 1> de_dtd;
      Eigen::Matrix<double, krd, 1> de_dtd_minimal;

      double* jacobians[] = {de_dTWB[0].data(),
                             de_dTWB[1].data(),
                             de_dTWB[2].data(),
                             de_dPoint.data(),
                             de_dExtrinsic.data(),
                             de_dproj_intrinsic.data(),
                             de_ddistortion.data(),
                             de_dtr.data(),
                             de_dtd.data(),
                             de_dSpeedAndBias[0].data(),
                             de_dSpeedAndBias[1].data(),
                             de_dSpeedAndBias[2].data()};
      double* jacobiansMinimal[] = {de_dTWB_minimal[0].data(),
                                    de_dTWB_minimal[1].data(),
                                    de_dTWB_minimal[2].data(),
                                    de_dPoint_minimal.data(),
                                    de_dExtrinsic_minimal.data(),
                                    de_dproj_intrinsic_minimal.data(),
                                    de_ddistortion_minimal.data(),
                                    de_dtr_minimal.data(),
                                    de_dtd_minimal.data(),
                                    de_dSpeedAndBias_minimal[0].data(),
                                    de_dSpeedAndBias_minimal[1].data(),
                                    de_dSpeedAndBias_minimal[2].data()};
      bool evaluateOk = observationError->EvaluateWithMinimalJacobians(
          parameters, residual->data(), jacobians, jacobiansMinimal);
      const std::vector<okvis::AnchorFrameIdentifier>& anchorIds = pointDataPtr->anchorIds();
      if (evaluateOk) {
        status = msckf::MeasurementJacobianStatus::Successful;
      } else {
        if (anchorIds[0].frameId_ == poseId) {
          status = msckf::MeasurementJacobianStatus::MainAnchorProjectionFailed;
        } else if (anchorIds[1].frameId_ == poseId) {
          status =
              msckf::MeasurementJacobianStatus::AssociateAnchorProjectionFailed;
        }
      }
      int cameraParamsDim = cameraParamsMinimalDimen();
      J_X->setZero();
      if (fixCameraExtrinsicParams_[camIdx]) {
        if (fixCameraIntrinsicParams_[camIdx]) {
          J_X->topLeftCorner(krd, cameraParamsDim) << de_dtd_minimal,
              de_dtr_minimal;
        } else {
          J_X->topLeftCorner(krd, cameraParamsDim)
              << de_dproj_intrinsic_minimal,
              de_ddistortion_minimal, de_dtd_minimal, de_dtr_minimal;
        }
      } else {
        if (fixCameraIntrinsicParams_[camIdx]) {
          J_X->topLeftCorner(krd, cameraParamsDim) << de_dExtrinsic_minimal,
              de_dtd_minimal, de_dtr_minimal;
        } else {
          J_X->topLeftCorner(krd, cameraParamsDim) << de_dExtrinsic_minimal,
              de_dproj_intrinsic_minimal, de_ddistortion_minimal,
              de_dtd_minimal, de_dtr_minimal;
        }
      }

      std::vector<uint64_t> jmaFrameIds{poseId, anchorIds[0].frameId_, anchorIds[1].frameId_};
      for (int f = 0; f < 3; ++f) {
        uint64_t frameId = jmaFrameIds[f];
        auto smIter = statesMap_.find(frameId);
        J_X->block<krd, kClonedStateMinimalDimen>(
            0, cameraParamsDim +
                   kClonedStateMinimalDimen * smIter->second.orderInCov)
            << de_dTWB_minimal[f],
            de_dSpeedAndBias_minimal[f].template topLeftCorner<krd, 3>();
      }

      *J_pfi = de_dPoint_minimal;
      *J_n = Eigen::Matrix2d::Identity();
      *residual = -(*residual);
      break;
    }
    case okvis::cameras::kReprojectionErrorWithPapId: {
        typedef okvis::ceres::ReprojectionErrorWithPap<
            CameraGeometry, ProjectionIntrinsicModel, ExtrinsicModel,
            msckf::ParallaxAngleParameterization, ImuModel>
            CameraErrorModel;
        observationError.reset(new CameraErrorModel(
            argCameraGeometry, obs, obsCov, observationIndex, pointDataPtr));
        std::vector<int> anchorObservationIndices =
            pointDataPtr->anchorObservationIds();
        std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>
            anchorPoseBlockPtrs;
        anchorPoseBlockPtrs.reserve(2);
        std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>
            anchorSpeedAndBiasBlockPtrs;
        anchorSpeedAndBiasBlockPtrs.reserve(2);
        for (auto anchorObsId : anchorObservationIndices) {
          anchorPoseBlockPtrs.push_back(
              pointDataPtr->poseParameterBlockPtr(anchorObsId));
          anchorSpeedAndBiasBlockPtrs.push_back(
              pointDataPtr->speedAndBiasParameterBlockPtr(anchorObsId));
        }

        double const* const parameters[] = {
            poseParamBlockPtr->parameters(),
            anchorPoseBlockPtrs[0]->parameters(),
            anchorPoseBlockPtrs[1]->parameters(),
            pointLandmark.data(),
            extrinsicParamBlockPtr->parameters(),
            projectionParamBlockPtr->parameters(),
            distortionParamBlockPtr->parameters(),
            trParamBlockPtr->parameters(),
            tdParamBlockPtr->parameters(),
            sbParamBlockPtr->parameters(),
            anchorSpeedAndBiasBlockPtrs[0]->parameters(),
            anchorSpeedAndBiasBlockPtrs[1]->parameters()};

        const int krd = CameraErrorModel::kNumResiduals;
        const int kPoseNumber = 3;
        // The elements are de_dTWBj, de_dTWBmi, de_dTWBai.
        std::vector<Eigen::Matrix<double, krd, 7, Eigen::RowMajor>,
                    Eigen::aligned_allocator<
                        Eigen::Matrix<double, krd, 7, Eigen::RowMajor>>>
            de_dTWB(kPoseNumber);
        std::vector<Eigen::Matrix<double, krd, 6, Eigen::RowMajor>,
                    Eigen::aligned_allocator<
                        Eigen::Matrix<double, krd, 6, Eigen::RowMajor>>>
            de_dTWB_minimal(kPoseNumber);
        std::vector<Eigen::Matrix<double, krd, 9, Eigen::RowMajor>,
                    Eigen::aligned_allocator<
                        Eigen::Matrix<double, krd, 9, Eigen::RowMajor>>>
            de_dSpeedAndBias(kPoseNumber);
        std::vector<Eigen::Matrix<double, krd, 9, Eigen::RowMajor>,
                    Eigen::aligned_allocator<
                        Eigen::Matrix<double, krd, 9, Eigen::RowMajor>>>
            de_dSpeedAndBias_minimal(kPoseNumber);

        Eigen::Matrix<double, krd, PointLandmarkModel::kGlobalDim,
                      Eigen::RowMajor>
            de_dPoint;
        Eigen::Matrix<double, krd, PointLandmarkModel::kLocalDim, Eigen::RowMajor>
            de_dPoint_minimal;
        Eigen::Matrix<double, krd, ExtrinsicModel::kGlobalDim, Eigen::RowMajor>
            de_dExtrinsic;
        Eigen::Matrix<double, krd, ExtrinsicModel::kNumParams, Eigen::RowMajor>
            de_dExtrinsic_minimal;

        typename CameraErrorModel::ProjectionIntrinsicJacType de_dproj_intrinsic;
        typename CameraErrorModel::ProjectionIntrinsicJacType
            de_dproj_intrinsic_minimal;
        typename CameraErrorModel::DistortionJacType de_ddistortion;
        typename CameraErrorModel::DistortionJacType de_ddistortion_minimal;
        Eigen::Matrix<double, krd, 1> de_dtr;
        Eigen::Matrix<double, krd, 1> de_dtr_minimal;
        Eigen::Matrix<double, krd, 1> de_dtd;
        Eigen::Matrix<double, krd, 1> de_dtd_minimal;

        double* jacobians[] = {de_dTWB[0].data(),
                               de_dTWB[1].data(),
                               de_dTWB[2].data(),
                               de_dPoint.data(),
                               de_dExtrinsic.data(),
                               de_dproj_intrinsic.data(),
                               de_ddistortion.data(),
                               de_dtr.data(),
                               de_dtd.data(),
                               de_dSpeedAndBias[0].data(),
                               de_dSpeedAndBias[1].data(),
                               de_dSpeedAndBias[2].data()};
        double* jacobiansMinimal[] = {de_dTWB_minimal[0].data(),
                                      de_dTWB_minimal[1].data(),
                                      de_dTWB_minimal[2].data(),
                                      de_dPoint_minimal.data(),
                                      de_dExtrinsic_minimal.data(),
                                      de_dproj_intrinsic_minimal.data(),
                                      de_ddistortion_minimal.data(),
                                      de_dtr_minimal.data(),
                                      de_dtd_minimal.data(),
                                      de_dSpeedAndBias_minimal[0].data(),
                                      de_dSpeedAndBias_minimal[1].data(),
                                      de_dSpeedAndBias_minimal[2].data()};
        bool evaluateOk = observationError->EvaluateWithMinimalJacobians(
            parameters, residual->data(), jacobians, jacobiansMinimal);
        const std::vector<okvis::AnchorFrameIdentifier>& anchorIds = pointDataPtr->anchorIds();
        if (evaluateOk) {
          status = msckf::MeasurementJacobianStatus::Successful;
        } else {
          if (anchorIds[0].frameId_ == poseId) {
            status = msckf::MeasurementJacobianStatus::MainAnchorProjectionFailed;
          } else if (anchorIds[1].frameId_ == poseId) {
            status =
                msckf::MeasurementJacobianStatus::AssociateAnchorProjectionFailed;
          }
        }
        int cameraParamsDim = cameraParamsMinimalDimen();
        J_X->setZero();
        if (fixCameraExtrinsicParams_[camIdx]) {
          if (fixCameraIntrinsicParams_[camIdx]) {
            J_X->topLeftCorner(krd, cameraParamsDim) << de_dtd_minimal,
                de_dtr_minimal;
          } else {
            J_X->topLeftCorner(krd, cameraParamsDim)
                << de_dproj_intrinsic_minimal,
                de_ddistortion_minimal, de_dtd_minimal, de_dtr_minimal;
          }
        } else {
          if (fixCameraIntrinsicParams_[camIdx]) {
            J_X->topLeftCorner(krd, cameraParamsDim) << de_dExtrinsic_minimal,
                de_dtd_minimal, de_dtr_minimal;
          } else {
            J_X->topLeftCorner(krd, cameraParamsDim) << de_dExtrinsic_minimal,
                de_dproj_intrinsic_minimal, de_ddistortion_minimal,
                de_dtd_minimal, de_dtr_minimal;
          }
        }

        std::vector<uint64_t> jmaFrameIds{poseId, anchorIds[0].frameId_, anchorIds[1].frameId_};
        for (int f = 0; f < 3; ++f) {
          uint64_t frameId = jmaFrameIds[f];
          auto smIter = statesMap_.find(frameId);
          J_X->block<krd, kClonedStateMinimalDimen>(
              0, cameraParamsDim +
                     kClonedStateMinimalDimen * smIter->second.orderInCov)
              << de_dTWB_minimal[f],
              de_dSpeedAndBias_minimal[f].template topLeftCorner<krd, 3>();
        }

        *J_pfi = de_dPoint_minimal;
        *J_n = Eigen::Matrix2d::Identity();
        *residual = -(*residual);
        break;
      }
    case okvis::cameras::kTangentDistanceId:
      break;
    default:
      MODEL_DOES_NOT_EXIST_EXCEPTION
      break;
  }
  return status;
}
}  // namespace okvis
