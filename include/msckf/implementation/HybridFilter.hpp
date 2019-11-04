
/// \brief okvis Main namespace of this package.
namespace okvis {
template <class CAMERA_GEOMETRY_T>
bool HybridFilter::replaceEpipolarWithReprojectionErrors(uint64_t lmId) {

  PointMap::iterator it = landmarksMap_.find(lmId);
  std::map<okvis::KeypointIdentifier, uint64_t>& obsMap = it->second.observations;
  // remove all previous (epipolar constraint) residual blocks for
  // this landmark if exist, use the ResidualBlockId which is the map value

  // add all observations as reprojection errors
  return true;
}

template <class CAMERA_GEOMETRY_T>
bool HybridFilter::addEpipolarConstraint(uint64_t lmId, bool removeExisting) {
  PointMap::const_iterator it = landmarksMap_.find(lmId);
  size_t numObs = 0;
  if (it != landmarksMap_.end())
    numObs = it->second.observations.size();

  if (numObs >= minTrackLength_) {
    if (removeExisting) {
      //  remove previous head tail constraints for this landmark
    }
    //  add an epipolar constraint head_tail, record the residualBlockId in the obs map

  }
  return true;
}

}  // namespace okvis
