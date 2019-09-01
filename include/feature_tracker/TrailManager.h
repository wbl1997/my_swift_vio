#ifndef FEATURE_TRACKER_TRAIL_MANAGER_H
#define FEATURE_TRACKER_TRAIL_MANAGER_H

#include <boost/accumulators/accumulators.hpp>

#include <opencv2/core.hpp>

#include <msckf/MSCKF2.hpp>
#include "feature_tracker/FeatureTrail.h"
#include "feature_tracker/TrackResultReader.h"

namespace feature_tracker {
typedef boost::accumulators::accumulator_set<
    double, boost::accumulators::features<boost::accumulators::tag::count,
                                          boost::accumulators::tag::density>>
    MyAccumulator;
typedef boost::iterator_range<std::vector<std::pair<double, double>>::iterator>
    histogram_type;

class TrailManager {
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  TrailManager(const std::string& orbTrackOutput);
  ~TrailManager();
  // for feature tracking by KLT optic flow
  bool initialize();
  int detectAndInsert(const cv::Mat& currentFrame, int nInliers,
                      std::vector<cv::KeyPoint>&);
  int advance(
      okvis::HybridFilter& estimator,
      uint64_t mfIdA, uint64_t mfIdB, size_t camIdA, size_t camIdB);

  void updateEstimatorObservations(
      okvis::HybridFilter& estimator,
      uint64_t mfIdA, uint64_t mfIdB, size_t camIdA, size_t camIdB);

  // for feature tracks provided by an external module
  /**
   * @brief initialize2
   * @param keypoints  keypoints in the current image
   * @param mapPointIds map point ids in the current image,
   *     if not empty should of the same length as keypoints
   * @param mapPointPositions map point positions in the current image
   *     if not empty should of the same length as keypoints
   */
  void initialize2(const std::vector<cv::KeyPoint>& keypoints,
                   const std::vector<size_t>& mapPointIds,
                   const std::vector<Eigen::Vector3d,
                                     Eigen::aligned_allocator<Eigen::Vector3d>>&
                       mapPointPositions,
                   const uint64_t currentFrameId);
  /**
   * @brief advance2 works with external feature associations
   * @param keypoints
   * @param mapPointIds
   * @param mapPointPositions
   * @return
   */
  int advance2(const std::vector<cv::KeyPoint>& keypoints,
               const std::vector<size_t>& mapPointIds,
               const std::vector<Eigen::Vector3d,
                                 Eigen::aligned_allocator<Eigen::Vector3d>>&
                   mapPointPositions,
               uint64_t mfIdB);

  /**
   * @brief updateEstimatorObservations2 works with external feature
   * associations
   * @param estimator
   * @param mfIdA
   * @param mfIdB
   * @param camIdA
   * @param camIdB
   */
  void updateEstimatorObservations2(
      okvis::HybridFilter& estimator,
      uint64_t mfIdA, uint64_t mfIdB, size_t camIdA, size_t camIdB);

  // output the distribution of number of features in images
  void printNumFeatureDistribution(std::ofstream& stream) const;

  std::vector<cv::KeyPoint> getCurrentKeypoints() const;

  bool needToDetectMorePoints(int matches2d2d);

  const std::list<FeatureTrail>& getFeatureTrailList() const;

  std::vector<cv::Mat> mCurrentPyramid, mPreviousPyramid;

  TrackResultReader* pTracker;

 private:
  std::list<FeatureTrail> mFeatureTrailList;  // reference to lTrailers in map
  size_t mMaxFeaturesInFrame, mMinFeaturesInFrame;
  size_t mFrameCounter;  // how many frames have been processed by the frontend
  cv::Mat mMask;  // mask used to protect existing features during initializing
                  // features
  //  size_t mTrackedPoints; // number of well tracked points in the current
  //  frame
  std::vector<cv::KeyPoint> mvKeyPoints;  // key points in the current frame

  // an accumulator for number of features distribution
  // TODO(jhuai): for now only implemented for features tracked by an external
  // module, ORB-VO. The accumulator for KLT and OKVIS feature tracking  is to
  // be implemented
  MyAccumulator myAccumulator;
};
}  // namespace feature_tracker
#endif  // FEATURE_TRACKER_TRAIL_MANAGER_H
