#include <swift_vio/imu/ImuRig.hpp>

namespace swift_vio {
int ImuRig::addImu(const okvis::ImuParameters& imuParams) {
  int modelId = ImuModelNameToId(imuParams.model_type);
  Eigen::Matrix<double, Eigen::Dynamic, 1> extraParams;
  switch (modelId) {
    case Imu_BG_BA_TG_TS_TA::kModelId:
      extraParams.resize(Imu_BG_BA_TG_TS_TA::kAugmentedDim, 1);
      extraParams.head<9>() = imuParams.Tg0;
      extraParams.segment<9>(9) = imuParams.Ts0;
      extraParams.segment<9>(18) = imuParams.Ta0;
      break;
    case ScaledMisalignedImu::kModelId:
      extraParams = ScaledMisalignedImu::getNominalAugmentedParams<double>();
      break;
    default:
      extraParams.resize(0);
      break;
  }
  imus_.emplace_back(modelId, imuParams.g0, imuParams.a0, extraParams);
  return static_cast<int>(imus_.size()) - 1;
}

void getImuAugmentedStatesEstimate(
    std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>
        imuAugmentedParameterPtrs,
    Eigen::Matrix<double, Eigen::Dynamic, 1>* extraParams, int imuModelId) {
  switch (imuModelId) {
    case Imu_BG_BA::kModelId:
      break;
    case Imu_BG_BA_TG_TS_TA::kModelId: {
      extraParams->resize(27, 1);
      std::shared_ptr<const okvis::ceres::ShapeMatrixParamBlock> tgParamBlockPtr =
          std::static_pointer_cast<const okvis::ceres::ShapeMatrixParamBlock>(
              imuAugmentedParameterPtrs[0]);
      Eigen::Matrix<double, 9, 1> sm = tgParamBlockPtr->estimate();
      extraParams->head<9>() = sm;

      std::shared_ptr<const okvis::ceres::ShapeMatrixParamBlock> tsParamBlockPtr =
          std::static_pointer_cast<const okvis::ceres::ShapeMatrixParamBlock>(
              imuAugmentedParameterPtrs[1]);
      sm = tsParamBlockPtr->estimate();
      extraParams->segment<9>(9) = sm;

      std::shared_ptr<const okvis::ceres::ShapeMatrixParamBlock> taParamBlockPtr =
          std::static_pointer_cast<const okvis::ceres::ShapeMatrixParamBlock>(
              imuAugmentedParameterPtrs[2]);
      sm = taParamBlockPtr->estimate();
      extraParams->segment<9>(18) = sm;
    } break;
    case ScaledMisalignedImu::kModelId:
      LOG(WARNING) << "get state estimate not implemented for imu model "
                   << imuModelId;
      break;
  }
}
}  // namespace swift_vio
