#ifndef FEATURE_TRACKER_FEATURE_TRAIL_H
#define FEATURE_TRACKER_FEATURE_TRAIL_H

#include <opencv2/core.hpp>
namespace feature_tracker {
struct FeatureTrail {
  FeatureTrail(cv::Point2f fInitP, cv::Point2f fCurrP, uint64_t lmId,
               size_t kpIdx);

  FeatureTrail(cv::Point2f fInitP, cv::Point2f fCurrP, uint64_t lmId,
               size_t kpIdx, size_t externalLmId);

  // the copy constructor and assignment operator are only used for copying
  // trailers into a vector,so do not increase their uID
  FeatureTrail(const FeatureTrail& b);
  FeatureTrail& operator=(const FeatureTrail& b);
  ~FeatureTrail();

 public:
  uint64_t uLandmarkId;  // There are two cases a Trail is created. (1) some
                         // MapPoints have projection on the current frame,
  // they initialize some trails with pointer to these MapPoints, and push back
  // to mFeatureTrailList, (2) when there are inadequate tracked points in
  // current frame, this frame is added as a KeyFrame. Some points are detected
  // from it, and added to mFeatureTrailList as starting trailers. when another
  // keyframe is reached, these trailers with null pHingeMP will create a
  // MapPoint. Therefore, the new MapPoint exists between the last KeyFrame and
  // the new KeyFrame
  cv::Point2f fInitPose;  // initial position of the trail in srcKF, which is
                          // also pHingeMP->pSourceKeyFrame
  cv::Point2f fCurrPose;  // current position of the trail in current frame, to
                          // store the predicted position
  // by MapPoint or rotation compensation and the tracked position by KLT
  size_t
      uKeyPointIdx;  // id of the key point in the features of the current frame
  size_t
      uOldKeyPointIdx;   // intermediate variable, initialized when used, id of
                         // the key point in the features of the previous frame
  size_t uTrackLength;   // how many times the features is tracked, start from 1
                         // for first observation
  cv::Point3d p_W;       // an dummy variable position of the point in the world
                         // frame only for visualization
  size_t uExternalLmId;  // landmark id used by an external program e.g. orb_vo
  uint64_t uCurrentFrameId;
};
}  // namespace feature_tracker
#endif  // FEATURE_TRACKER_FEATURE_TRAILER_H
