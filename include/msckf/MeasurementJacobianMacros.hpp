#ifndef INCLUDE_MSCKF_MEASUREMENT_JACOBIAN_MACROS_HPP_
#define INCLUDE_MSCKF_MEASUREMENT_JACOBIAN_MACROS_HPP_

// Macros used by measurementJacobianGeneric().
// warn: This file is to be ONLY included by cpp file of the above function.
#ifndef PROJECTION_INTRINSIC_MODEL_CHAIN_CASE
#define PROJECTION_INTRINSIC_MODEL_CHAIN_CASE(                                \
    ImuModel, ExtrinsicModel, CameraGeometry, ProjectionIntrinsicModel)       \
  case ProjectionIntrinsicModel::kModelId:                                    \
    switch (landmarkModelId_) {                                               \
      case msckf::HomogeneousPointParameterization::kModelId:                 \
        status = computeCameraObservationJacobians<                           \
            CameraGeometry, ProjectionIntrinsicModel, ExtrinsicModel,         \
            msckf::HomogeneousPointParameterization, ImuModel>(               \
            pointLandmark, baseCameraGeometry, obs, obsCov, observationIndex, \
            pointDataPtr, J_X, J_pfi, J_n, residual);                         \
        break;                                                                \
      case msckf::InverseDepthParameterization::kModelId:                     \
        status = computeCameraObservationJacobians<                           \
            CameraGeometry, ProjectionIntrinsicModel, ExtrinsicModel,         \
            msckf::InverseDepthParameterization, ImuModel>(                   \
            pointLandmark, baseCameraGeometry, obs, obsCov, observationIndex, \
            pointDataPtr, J_X, J_pfi, J_n, residual);                         \
        break;                                                                \
      case msckf::ParallaxAngleParameterization::kModelId:                    \
        status = computeCameraObservationJacobians<                           \
            CameraGeometry, ProjectionIntrinsicModel, ExtrinsicModel,         \
            msckf::ParallaxAngleParameterization, ImuModel>(                  \
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
