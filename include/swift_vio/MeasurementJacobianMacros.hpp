#ifndef INCLUDE_SWIFT_VIO_MEASUREMENT_JACOBIAN_MACROS_HPP_
#define INCLUDE_SWIFT_VIO_MEASUREMENT_JACOBIAN_MACROS_HPP_

// Macros used by measurementJacobianGeneric().
// warn: This file is to be ONLY included by cpp file of the above function.
#ifndef PROJECTION_INTRINSIC_MODEL_CHAIN_CASE
#define PROJECTION_INTRINSIC_MODEL_CHAIN_CASE(                                \
    ImuModel, ExtrinsicModel, CameraGeometry, ProjectionIntrinsicModel)       \
  case ProjectionIntrinsicModel::kModelId:                                    \
    switch (pointLandmarkOptions_.landmarkModelId) {                                               \
      case okvis::ceres::HomogeneousPointLocalParameterization::kModelId:                 \
        status = computeCameraObservationJacobians<                           \
            CameraGeometry, ProjectionIntrinsicModel, ExtrinsicModel,         \
            okvis::ceres::HomogeneousPointLocalParameterization, ImuModel>(               \
            pointLandmark, baseCameraGeometry, obs, obsCov, observationIndex, \
            pointDataPtr, J_X, J_pfi, J_n, residual);                         \
        break;                                                                \
      case swift_vio::InverseDepthParameterization::kModelId:                     \
        status = computeCameraObservationJacobians<                           \
            CameraGeometry, ProjectionIntrinsicModel, ExtrinsicModel,         \
            swift_vio::InverseDepthParameterization, ImuModel>(                   \
            pointLandmark, baseCameraGeometry, obs, obsCov, observationIndex, \
            pointDataPtr, J_X, J_pfi, J_n, residual);                         \
        break;                                                                \
      case swift_vio::ParallaxAngleParameterization::kModelId:                    \
        status = computeCameraObservationJacobians<                           \
            CameraGeometry, ProjectionIntrinsicModel, ExtrinsicModel,         \
            swift_vio::ParallaxAngleParameterization, ImuModel>(                  \
            pointLandmark, baseCameraGeometry, obs, obsCov, observationIndex, \
            pointDataPtr, J_X, J_pfi, J_n, residual);                         \
        break;                                                                \
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
        ImuModel, ExtrinsicModel, CameraGeometry, swift_vio::ProjectionOptFXY_CXY)        \
    PROJECTION_INTRINSIC_MODEL_CHAIN_CASE(ImuModel, ExtrinsicModel,            \
                                          CameraGeometry, swift_vio::ProjectionOptFX_CXY) \
    PROJECTION_INTRINSIC_MODEL_CHAIN_CASE(ImuModel, ExtrinsicModel,            \
                                          CameraGeometry, swift_vio::ProjectionOptFX)     \
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
      EXTRINSIC_MODEL_CHAIN_CASE(ImuModel, swift_vio::Extrinsic_p_CB)      \
      EXTRINSIC_MODEL_CHAIN_CASE(ImuModel, swift_vio::Extrinsic_p_BC_q_BC) \
      default:                                                  \
        MODEL_DOES_NOT_APPLY_EXCEPTION                          \
        break;                                                  \
    }                                                           \
    break;
#endif

#endif // INCLUDE_SWIFT_VIO_MEASUREMENT_JACOBIAN_MACROS_HPP_
