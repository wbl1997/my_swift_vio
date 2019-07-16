#include "feature_tracker/FeatureTrail.h"
namespace feature_tracker {

FeatureTrail::FeatureTrail(cv::Point2f fInitP, cv::Point2f fCurrP,
                           uint64_t lmId, size_t kpIdx)
    : uLandmarkId(lmId),
      fInitPose(fInitP),
      fCurrPose(fCurrP),
      uKeyPointIdx(kpIdx),
      uOldKeyPointIdx(1e6),
      uTrackLength(1),
      uCurrentFrameId(0) {}

FeatureTrail::FeatureTrail(cv::Point2f fInitP, cv::Point2f fCurrP,
                           uint64_t lmId, size_t kpIdx, size_t externalLmId)
    : uLandmarkId(lmId),
      fInitPose(fInitP),
      fCurrPose(fCurrP),
      uKeyPointIdx(kpIdx),
      uOldKeyPointIdx(1e6),
      uTrackLength(1),
      uExternalLmId(externalLmId),
      uCurrentFrameId(0) {}

// the copy constructor and assignment operator are only used for copying
// trailers into a vector,so do not increase their uID
FeatureTrail::FeatureTrail(const FeatureTrail& b)
    : uLandmarkId(b.uLandmarkId),
      fInitPose(b.fInitPose),
      fCurrPose(b.fCurrPose),
      uKeyPointIdx(b.uKeyPointIdx),
      uOldKeyPointIdx(b.uOldKeyPointIdx),
      uTrackLength(b.uTrackLength),
      p_W(b.p_W),
      uExternalLmId(b.uExternalLmId),
      uCurrentFrameId(b.uCurrentFrameId) {}
FeatureTrail& FeatureTrail::operator=(const FeatureTrail& b) {
  if (this != &b) {
    uLandmarkId = b.uLandmarkId;
    fInitPose = b.fInitPose;
    fCurrPose = b.fCurrPose;
    uKeyPointIdx = b.uKeyPointIdx;
    uOldKeyPointIdx = b.uOldKeyPointIdx;
    uTrackLength = b.uTrackLength;
    p_W = b.p_W;
    uExternalLmId = b.uExternalLmId;
    uCurrentFrameId = b.uCurrentFrameId;
  }
  return *this;
}
FeatureTrail::~FeatureTrail() {}
}  // namespace feature_tracker
