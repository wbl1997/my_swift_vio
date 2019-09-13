
/// \brief okvis Main namespace of this package.
namespace okvis {

// Add an observation to a landmark.
template<class GEOMETRY_TYPE>
::ceres::ResidualBlockId HybridFilter::addObservation(uint64_t landmarkId,
                                                   uint64_t poseId,
                                                   size_t camIdx,
                                                   size_t keypointIdx) {
  OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId),
                        "landmark not added");

  // avoid double observations
  okvis::KeypointIdentifier kid(poseId, camIdx, keypointIdx);
  if (landmarksMap_.at(landmarkId).observations.find(kid)
      != landmarksMap_.at(landmarkId).observations.end()) {
    return NULL;
  }

  // get the keypoint measurement
  okvis::MultiFramePtr multiFramePtr = multiFramePtrMap_.at(poseId);
  Eigen::Vector2d measurement;
  multiFramePtr->getKeypoint(camIdx, keypointIdx, measurement);
  Eigen::Matrix2d information = Eigen::Matrix2d::Identity();
  double size = 1.0;
  multiFramePtr->getKeypointSize(camIdx, keypointIdx, size);
  information *= 64.0 / (size * size);

  // create error term
  std::shared_ptr < ceres::ReprojectionError
      < GEOMETRY_TYPE
          >> reprojectionError(
              new ceres::ReprojectionError<GEOMETRY_TYPE>(
                  multiFramePtr->template geometryAs<GEOMETRY_TYPE>(camIdx),
                  camIdx, measurement, information));

  ::ceres::ResidualBlockId retVal = mapPtr_->addResidualBlock(
      reprojectionError,
      cauchyLossFunctionPtr_ ? cauchyLossFunctionPtr_.get() : NULL,
      mapPtr_->parameterBlockPtr(poseId),
      mapPtr_->parameterBlockPtr(landmarkId),
      mapPtr_->parameterBlockPtr(
          statesMap_.at(poseId).sensors.at(SensorStates::Camera).at(camIdx).at(
              CameraSensorStates::T_SCi).id));

  // remember
  landmarksMap_.at(landmarkId).observations.insert(
      std::pair<okvis::KeypointIdentifier, uint64_t>(
          kid, reinterpret_cast<uint64_t>(retVal)));

  return retVal;
}

template<class PARAMETER_BLOCK_T>
bool HybridFilter::getGlobalStateEstimateAs(
        uint64_t poseId, int stateType,
        typename PARAMETER_BLOCK_T::estimate_t & state) const
{
    PARAMETER_BLOCK_T stateParameterBlock;
    if (!getGlobalStateParameterBlockAs(poseId, stateType, stateParameterBlock)) {
        return false;
    }
    state = stateParameterBlock.estimate();
    return true;
}

template<class PARAMETER_BLOCK_T>
bool HybridFilter::getSensorStateEstimateAs(
        uint64_t poseId, int sensorIdx, int sensorType, int stateType,
        typename PARAMETER_BLOCK_T::estimate_t & state) const
{
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
}

template<class PARAMETER_BLOCK_T>
bool HybridFilter::setGlobalStateEstimateAs(
        uint64_t poseId, int stateType,
        const typename PARAMETER_BLOCK_T::estimate_t & state)
{
    // check existence in states set
    if (statesMap_.find(poseId) == statesMap_.end()) {
        OKVIS_THROW_DBG(Exception,"pose with id = "<<poseId<<" does not exist.")
                return false;
    }

    // obtain the parameter block ID
    uint64_t id = statesMap_.at(poseId).global.at(stateType).id;
    if (!mapPtr_->parameterBlockExists(id)) {
        OKVIS_THROW_DBG(Exception,"pose with id = "<<poseId<<" does not exist.")
                return false;
    }

    std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr = mapPtr_
            ->parameterBlockPtr(id);
#ifndef NDEBUG
    std::shared_ptr<PARAMETER_BLOCK_T> derivedParameterBlockPtr =
            std::dynamic_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
    if(!derivedParameterBlockPtr) {
        OKVIS_THROW_DBG(Exception,"wrong pointer type requested.")
                return false;
    }
    derivedParameterBlockPtr->setEstimate(state);
#else
    std::static_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr)->setEstimate(
                state);
#endif
    return true;
}


template<class PARAMETER_BLOCK_T>
bool HybridFilter::getGlobalStateParameterBlockAs(
        uint64_t poseId, int stateType,
        PARAMETER_BLOCK_T & stateParameterBlock) const
{
    // convert base class pointer with various levels of checking
    std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr;
    if (!getGlobalStateParameterBlockPtr(poseId, stateType, parameterBlockPtr)) {
        return false;
    }
#ifndef NDEBUG
    std::shared_ptr<PARAMETER_BLOCK_T> derivedParameterBlockPtr =
            std::dynamic_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
    if(!derivedParameterBlockPtr) {
        LOG(INFO) << "--"<<parameterBlockPtr->typeInfo();
        std::shared_ptr<PARAMETER_BLOCK_T> info(new PARAMETER_BLOCK_T);
        OKVIS_THROW_DBG(Exception,"wrong pointer type requested: requested "
                        <<info->typeInfo()<<" but is of type"
                        <<parameterBlockPtr->typeInfo())
                return false;
    }
    stateParameterBlock = *derivedParameterBlockPtr;
#else
    stateParameterBlock = *std::static_pointer_cast<PARAMETER_BLOCK_T>(
                parameterBlockPtr);
#endif
    return true;
}

}  // namespace okvis
