
/**
 * @file implementation/Estimator.hpp
 * @brief Header implementation file for the Estimator class.
 * @author Jianzhu Huai
 */

#include <swift_vio/ceres/EpipolarFactor.hpp>
#include <swift_vio/ProjParamOptModels.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
template <class GEOMETRY_TYPE>
::ceres::ResidualBlockId Estimator::addPointFrameResidual(
    uint64_t landmarkId, const KeypointIdentifier& kpi) {
  // get the keypoint measurement
  okvis::MultiFramePtr multiFramePtr = multiFramePtrMap_.at(kpi.frameId);
  Eigen::Vector2d measurement;
  multiFramePtr->getKeypoint(kpi.cameraIndex, kpi.keypointIndex, measurement);
  Eigen::Matrix2d information = Eigen::Matrix2d::Identity();
  double size = 1.0;
  multiFramePtr->getKeypointSize(kpi.cameraIndex, kpi.keypointIndex, size);
  information *= 64.0 / (size * size);

  std::shared_ptr<const GEOMETRY_TYPE> cameraGeometry =
      cameraRig_.template geometryAs<GEOMETRY_TYPE>(kpi.cameraIndex);

  std::shared_ptr<ceres::ReprojectionError<GEOMETRY_TYPE>> reprojectionError(
      new ceres::ReprojectionError<GEOMETRY_TYPE>(cameraGeometry, kpi.cameraIndex,
                                                  measurement, information));

  ::ceres::ResidualBlockId retVal = mapPtr_->addResidualBlock(
      reprojectionError,
      cauchyLossFunctionPtr_ ? cauchyLossFunctionPtr_.get() : NULL,
      mapPtr_->parameterBlockPtr(kpi.frameId),
      mapPtr_->parameterBlockPtr(landmarkId),
      mapPtr_->parameterBlockPtr(statesMap_.at(kpi.frameId)
                                     .sensors.at(SensorStates::Camera)
                                     .at(kpi.cameraIndex)
                                     .at(CameraSensorStates::T_XCi)
                                     .id));
  return retVal;
}

template<class PARAMETER_BLOCK_T>
bool Estimator::getSensorStateEstimateAs(
    uint64_t poseId, int sensorIdx, int sensorType, int stateType,
    typename PARAMETER_BLOCK_T::estimate_t & state) const
{
#if 0
  PARAMETER_BLOCK_T stateParameterBlock;
  if (!getSensorStateParameterBlockAs(poseId, sensorIdx, sensorType, stateType,
                                      stateParameterBlock)) {
    return false;
  }
  state = stateParameterBlock.estimate();
  return true;
#else
  // convert base class pointer with various levels of checking
  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr;
  if (!getSensorStateParameterBlockPtr(poseId, sensorIdx, sensorType, stateType,
                                       parameterBlockPtr)) {
      return false;
  }
#ifndef NDEBUG
  std::shared_ptr<PARAMETER_BLOCK_T> derivedParameterBlockPtr =
          std::dynamic_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
  if(!derivedParameterBlockPtr) {
      std::shared_ptr<PARAMETER_BLOCK_T> info(new PARAMETER_BLOCK_T);
      OKVIS_THROW_DBG(Exception,"wrong pointer type requested: requested "
                      <<info->typeInfo()<<" but is of type"
                      <<parameterBlockPtr->typeInfo())
              return false;
  }
  state = derivedParameterBlockPtr->estimate();
#else
  state = std::static_pointer_cast<PARAMETER_BLOCK_T>(
              parameterBlockPtr)->estimate();
#endif
  return true;
#endif
}
}  // namespace okvis
