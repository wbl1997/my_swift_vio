/**
 * @file FrameTypedefs.cpp
 * @brief This file contains implementations for MapPoint.
 * @author Jianzhu Huai
 */

#include <okvis/FrameTypedefs.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
uint64_t MapPoint::shouldChangeAnchor(
    const std::vector<uint64_t> &sortedToRemoveStateIds) const {
  if (anchorStateId != 0u) {
    auto iter = std::lower_bound(sortedToRemoveStateIds.begin(),
                                 sortedToRemoveStateIds.end(), anchorStateId);
    if (iter == sortedToRemoveStateIds.end() ||
        *iter != anchorStateId) { // Not found in the state ids.
      return 0u;
    } else {
      // find the last observation in an frame not among removed state ids.
      for (std::map<okvis::KeypointIdentifier, uint64_t>::const_reverse_iterator
               riter = observations.rbegin();
           riter != observations.rend(); ++riter) {
        auto it = std::lower_bound(sortedToRemoveStateIds.begin(),
                                   sortedToRemoveStateIds.end(),
                                   riter->first.frameId);
        if (it == sortedToRemoveStateIds.end() ||
            *it != riter->first.frameId) {
          return riter->first.frameId;
        }
      }
      return 0u;
    }
  } else { // Not an anchored landmark.
    return 0u;
  }
}

bool MapPoint::goodForMarginalization(size_t minCulledFrames) const {
  if (observations.size() < minCulledFrames)
    return false;
  switch (status.measurementType) {
  case swift_vio::FeatureTrackStatus::kPremature:
    return true;

  case swift_vio::FeatureTrackStatus::kMsckfTrack:
  case swift_vio::FeatureTrackStatus::kSlamInitialization:
    return status.measurementFate != swift_vio::FeatureTrackStatus::kSuccessful;
  default:
    return false;
  }
}

void MapPoint::updateStatus(uint64_t currentFrameId,
                            size_t minTrackLengthForMsckf,
                            size_t minTrackLengthForSlam) {
  bool newlyObserved = trackedInCurrentFrame(currentFrameId);
  status.updateTrackStat(newlyObserved);
  swift_vio::FeatureTrackStatus::MeasurementType measurementType(
      swift_vio::FeatureTrackStatus::kPremature);
  if (newlyObserved) {
    if (status.inState) {
      measurementType = swift_vio::FeatureTrackStatus::kSlamObservation;
    } else {
      if (observations.size() >= minTrackLengthForSlam) {
        measurementType = swift_vio::FeatureTrackStatus::kSlamInitialization;
      }
    }
  } else {
    if (observations.size() >= minTrackLengthForMsckf) {
      measurementType = swift_vio::FeatureTrackStatus::kMsckfTrack;
    }
  }
  status.measurementType = measurementType;
  status.measurementFate = swift_vio::FeatureTrackStatus::kUndetermined;
}

} // namespace okvis
