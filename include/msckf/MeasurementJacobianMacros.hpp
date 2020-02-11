#ifndef INCLUDE_MSCKF_MEASUREMENT_JACOBIAN_MACROS_HPP_
#define INCLUDE_MSCKF_MEASUREMENT_JACOBIAN_MACROS_HPP_
// Macros used by measurementJacobianGeneric().
// warn: This file is to be ONLY included by cpp file of the above function.
#ifndef POINT_LANDMARK_MODEL_CHAIN_CASE
#define POINT_LANDMARK_MODEL_CHAIN_CASE(                                      \
    ImuModel, ExtrinsicModel, CameraGeometry, ProjectionIntrinsicModel,       \
    PointLandmarkModel)                                                       \
  case PointLandmarkModel::kModelId:                                          \
    switch (cameraObservationModelId_) {                                      \
      case okvis::cameras::kReprojectionErrorId: {                            \
        typedef okvis::ceres::RsReprojectionError<                            \
            CameraGeometry, ProjectionIntrinsicModel, ExtrinsicModel,         \
            PointLandmarkModel, ImuModel>                                     \
            RsReprojectionErrorModel;                                         \
        std::shared_ptr<const CameraGeometry> argCameraGeometry =             \
            std::static_pointer_cast<const CameraGeometry>(                   \
                baseCameraGeometry);                                          \
        std::shared_ptr<const okvis::ImuMeasurementDeque> imuMeasDequePtr =   \
            statesIter->second.imuReadingWindow;                              \
        OKVIS_ASSERT_GT(Exception, imuMeasDequePtr->size(), 0u,               \
                        "the IMU measurement does not exist");                \
        std::shared_ptr<const Eigen::Matrix<double, 6, 1>>                    \
            posVelFirstEstimate = statesIter->second.linearizationPoint;      \
        okvis::Time stateEpoch = statesIter->second.timestamp;                \
        double tdAtCreation = statesIter->second.tdAtCreation;                \
        double gravity = imuParametersVec_.at(0).g;                           \
        observationError.reset(new RsReprojectionErrorModel(                  \
            argCameraGeometry, obs, obsCov, imuMeasDequePtr,                  \
            posVelFirstEstimate, stateEpoch, tdAtCreation, gravity));         \
        double const* const parameters[] = {                                  \
            poseParamBlockPtr->parameters(),                                  \
            pointLandmark.data(),                                             \
            extrinsicParamBlockPtr->parameters(),                             \
            projectionParamBlockPtr->parameters(),                            \
            distortionParamBlockPtr->parameters(),                            \
            trParamBlockPtr->parameters(),                                    \
            tdParamBlockPtr->parameters(),                                    \
            sbParamBlockPtr->parameters()};                                   \
        Eigen::Matrix<double, 2, 7, Eigen::RowMajor> duv_deltaTWS;            \
        Eigen::Matrix<double, 2, 6, Eigen::RowMajor> duv_deltaTWS_minimal;    \
        Eigen::Matrix<double, 2, 4, Eigen::RowMajor> duv_deltahpW;            \
        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> duv_deltahpW_minimal;    \
        Eigen::Matrix<double, 2, 7, Eigen::RowMajor> duv_dExtrinsic;          \
        Eigen::Matrix<double, 2, ExtrinsicModel::kNumParams, Eigen::RowMajor> \
            duv_dExtrinsic_minimal;                                           \
        RsReprojectionErrorModel::ProjectionIntrinsicJacType                  \
            duv_proj_intrinsic;                                               \
        RsReprojectionErrorModel::DistortionJacType duv_distortion;           \
        Eigen::Matrix<double, 2, 1> duv_tr;                                   \
        RsReprojectionErrorModel::ProjectionIntrinsicJacType                  \
            duv_proj_intrinsic_minimal;                                       \
        RsReprojectionErrorModel::DistortionJacType duv_distortion_minimal;   \
        Eigen::Matrix<double, 2, 1> duv_tr_minimal;                           \
        Eigen::Matrix<double, 2, 1> duv_td;                                   \
        Eigen::Matrix<double, 2, 1> duv_td_minimal;                           \
        Eigen::Matrix<double, 2, 9, Eigen::RowMajor> duv_sb;                  \
        Eigen::Matrix<double, 2, 9, Eigen::RowMajor> duv_sb_minimal;          \
        double* jacobians[] = {                                               \
            duv_deltaTWS.data(),   duv_deltahpW.data(),                       \
            duv_dExtrinsic.data(), duv_proj_intrinsic.data(),                 \
            duv_distortion.data(), duv_tr.data(),                             \
            duv_td.data(),         duv_sb.data()};                            \
        double* jacobiansMinimal[] = {                                        \
            duv_deltaTWS_minimal.data(),   duv_deltahpW_minimal.data(),       \
            duv_dExtrinsic_minimal.data(), duv_proj_intrinsic_minimal.data(), \
            duv_distortion_minimal.data(), duv_tr_minimal.data(),             \
            duv_td_minimal.data(),         duv_sb_minimal.data()};            \
        evaluateOk = observationError->EvaluateWithMinimalJacobians(          \
            parameters, residual->data(), jacobians, jacobiansMinimal);       \
        if (!evaluateOk) {                                                    \
          status = MeasurementJacobianStatus::GeneralProjectionFailed;        \
        }                                                                     \
        int cameraParamsDim = cameraParamsMinimalDimen();                     \
        J_X->setZero();                                                       \
        if (fixCameraExtrinsicParams_[camIdx]) {                              \
          if (fixCameraIntrinsicParams_[camIdx]) {                            \
            J_X->topLeftCorner(2, cameraParamsDim) << duv_td_minimal,         \
                duv_tr_minimal;                                               \
          } else {                                                            \
            J_X->topLeftCorner(2, cameraParamsDim)                            \
                << duv_proj_intrinsic_minimal,                                \
                duv_distortion_minimal, duv_td_minimal, duv_tr_minimal;       \
          }                                                                   \
        } else {                                                              \
          if (fixCameraIntrinsicParams_[camIdx]) {                            \
            J_X->topLeftCorner(2, cameraParamsDim) << duv_dExtrinsic_minimal, \
                duv_td_minimal, duv_tr_minimal;                               \
          } else {                                                            \
            J_X->topLeftCorner(2, cameraParamsDim) << duv_dExtrinsic_minimal, \
                duv_proj_intrinsic_minimal, duv_distortion_minimal,           \
                duv_td_minimal, duv_tr_minimal;                               \
          }                                                                   \
        }                                                                     \
        std::map<uint64_t, int>::const_iterator poseCovIndexIter =            \
            mStateID2CovID_.find(poseId);                                     \
        J_X->block<2, kClonedStateMinimalDimen>(                              \
            0, cameraParamsDim +                                              \
                   kClonedStateMinimalDimen * poseCovIndexIter->second)       \
            << duv_deltaTWS_minimal,                                          \
            duv_sb_minimal.topLeftCorner<2, 3>();                             \
        *J_pfi = duv_deltahpW.topLeftCorner<2, 3>();                          \
        *J_n = Eigen::Matrix2d::Identity();                                   \
        *residual = -(*residual);                                             \
        break;                                                                \
      }                                                                       \
      case okvis::cameras::kTangentDistanceId:                                \
        break;                                                                \
      case okvis::cameras::kChordalDistanceId: {                              \
        typedef okvis::ceres::ChordalDistance<                                \
            CameraGeometry, ProjectionIntrinsicModel, ExtrinsicModel,         \
            msckf::ParallaxAngleParameterization, ImuModel>                   \
            ChordalDistanceErrorModel;                                        \
        std::shared_ptr<const CameraGeometry> argCameraGeometry =             \
            std::static_pointer_cast<const CameraGeometry>(                   \
                baseCameraGeometry);                                          \
        observationError.reset(new ChordalDistanceErrorModel(                 \
            argCameraGeometry, obs, obsCov, observationIndex, pointDataPtr)); \
        break;                                                                \
      }                                                                       \
      default:                                                                \
        MODEL_DOES_NOT_EXIST_EXCEPTION                                        \
        break;                                                                \
    }                                                                         \
    break;
#endif

#ifndef PROJECTION_INTRINSIC_MODEL_CHAIN_CASE
#define PROJECTION_INTRINSIC_MODEL_CHAIN_CASE(                                \
    ImuModel, ExtrinsicModel, CameraGeometry, ProjectionIntrinsicModel)       \
  case ProjectionIntrinsicModel::kModelId:                                    \
    switch (landmarkModelId_) {                                               \
      POINT_LANDMARK_MODEL_CHAIN_CASE(                                        \
          ImuModel, ExtrinsicModel, CameraGeometry, ProjectionIntrinsicModel, \
          msckf::HomogeneousPointParameterization)                            \
      POINT_LANDMARK_MODEL_CHAIN_CASE(                                        \
          ImuModel, ExtrinsicModel, CameraGeometry, ProjectionIntrinsicModel, \
          msckf::InverseDepthParameterization)                                \
      POINT_LANDMARK_MODEL_CHAIN_CASE(                                        \
          ImuModel, ExtrinsicModel, CameraGeometry, ProjectionIntrinsicModel, \
          msckf::ParallaxAngleParameterization)                               \
      default:                                                                \
        MODEL_DOES_NOT_EXIST_EXCEPTION                                        \
        break;                                                                \
    }                                                                         \
    break;
#endif

#ifndef DISTORTION_MODEL_CHAIN_CASE
#define DISTORTION_MODEL_CHAIN_CASE(ImuModel, ExtrinsicModel, CameraGeometry)  \
  switch (projOptModelId) {                                                    \
    PROJECTION_INTRINSIC_MODEL_CHAIN_CASE(                                     \
        ImuModel, ExtrinsicModel, CameraGeometry, ProjectionOptFXY_CXY)        \
    PROJECTION_INTRINSIC_MODEL_CHAIN_CASE(ImuModel, ExtrinsicModel,            \
                                          CameraGeometry, ProjectionOptFX_CXY) \
    PROJECTION_INTRINSIC_MODEL_CHAIN_CASE(ImuModel, ExtrinsicModel,            \
                                          CameraGeometry, ProjectionOptFX)     \
    default:                                                                   \
      MODEL_DOES_NOT_APPLY_EXCEPTION                                           \
      break;                                                                   \
  }                                                                            \
  break;
#endif

#ifndef EXTRINSIC_MODEL_CHAIN_CASE
#define EXTRINSIC_MODEL_CHAIN_CASE(ImuModel, ExtrinsicModel)              \
  case ExtrinsicModel::kModelId:                                          \
    switch (distortionType) {                                             \
      case okvis::cameras::NCameraSystem::Equidistant:                    \
        DISTORTION_MODEL_CHAIN_CASE(                                      \
            ImuModel, ExtrinsicModel,                                     \
            okvis::cameras::PinholeCamera<                                \
                okvis::cameras::EquidistantDistortion>)                   \
      case okvis::cameras::NCameraSystem::RadialTangential:               \
        DISTORTION_MODEL_CHAIN_CASE(                                      \
            ImuModel, ExtrinsicModel,                                     \
            okvis::cameras::PinholeCamera<                                \
                okvis::cameras::RadialTangentialDistortion>)              \
      case okvis::cameras::NCameraSystem::RadialTangential8:              \
        DISTORTION_MODEL_CHAIN_CASE(                                      \
            ImuModel, ExtrinsicModel,                                     \
            okvis::cameras::PinholeCamera<                                \
                okvis::cameras::RadialTangentialDistortion8>)             \
      case okvis::cameras::NCameraSystem::FOV:                            \
        DISTORTION_MODEL_CHAIN_CASE(                                      \
            ImuModel, ExtrinsicModel,                                     \
            okvis::cameras::PinholeCamera<okvis::cameras::FovDistortion>) \
      default:                                                            \
        MODEL_DOES_NOT_APPLY_EXCEPTION                                    \
        break;                                                            \
    }                                                                     \
    break;
#endif

#ifndef IMU_MODEL_CHAIN_CASE
#define IMU_MODEL_CHAIN_CASE(ImuModel)                          \
  case ImuModel::kModelId:                                      \
    switch (extrinsicModelId) {                                 \
      EXTRINSIC_MODEL_CHAIN_CASE(ImuModel, Extrinsic_p_CB)      \
      EXTRINSIC_MODEL_CHAIN_CASE(ImuModel, Extrinsic_p_BC_q_BC) \
      default:                                                  \
        MODEL_DOES_NOT_APPLY_EXCEPTION                          \
        break;                                                  \
    }                                                           \
    break;
#endif

#endif // INCLUDE_MSCKF_MEASUREMENT_JACOBIAN_MACROS_HPP_
