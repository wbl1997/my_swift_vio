
#ifndef INCLUDE_OKVIS_HYBRID_FRONTEND_HPP_
#define INCLUDE_OKVIS_HYBRID_FRONTEND_HPP_

#include <mutex>
#include <okvis/assert_macros.hpp>
#ifdef USE_MSCKF2
  #include <okvis/msckf2.hpp>
#else
  #include <okvis/HybridFilter.hpp>
#endif
#include <okvis/timing/Timer.hpp>
#include <okvis/DenseMatcher.hpp>

#include <okvis/TrackResultReader.h>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/density.hpp>
#include <boost/accumulators/statistics/stats.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

typedef boost::accumulators::accumulator_set<double, boost::accumulators::features<boost::accumulators::tag::density> > MyAccumulator;
typedef boost::iterator_range<std::vector<std::pair<double, double> >::iterator > histogram_type;

struct Trailer{
    Trailer(cv::Point2f fInitP, cv::Point2f fCurrP, uint64_t lmId, size_t kpIdx):
        uLandmarkId(lmId), fInitPose(fInitP),fCurrPose(fCurrP),
        uKeyPointIdx(kpIdx), uOldKeyPointIdx(1e6), uTrackLength(1), uCurrentFrameId(0){}

    Trailer(cv::Point2f fInitP, cv::Point2f fCurrP, uint64_t lmId, size_t kpIdx, size_t externalLmId):
        uLandmarkId(lmId), fInitPose(fInitP),fCurrPose(fCurrP),
        uKeyPointIdx(kpIdx), uOldKeyPointIdx(1e6), uTrackLength(1), uExternalLmId(externalLmId), uCurrentFrameId(0){}

    // the copy constructor and assignment operator are only used for copying trailers into a vector,so do not increase their uID
    Trailer(const Trailer& b):
        uLandmarkId(b.uLandmarkId), fInitPose(b.fInitPose),fCurrPose(b.fCurrPose),
        uKeyPointIdx(b.uKeyPointIdx), uOldKeyPointIdx(b.uOldKeyPointIdx),
        uTrackLength(b.uTrackLength), p_W(b.p_W), uExternalLmId(b.uExternalLmId), uCurrentFrameId(b.uCurrentFrameId){}
    Trailer& operator=(const Trailer& b)
    {
        if(this != &b){
            uLandmarkId=b.uLandmarkId;
            fInitPose=b.fInitPose;
            fCurrPose=b.fCurrPose;
            uKeyPointIdx=b.uKeyPointIdx;
            uOldKeyPointIdx=b.uOldKeyPointIdx;
            uTrackLength=b.uTrackLength;
            p_W= b.p_W;
            uExternalLmId=b.uExternalLmId;
            uCurrentFrameId= b.uCurrentFrameId;
        }
        return *this;
    }
    ~Trailer(){
    }
public:
    uint64_t uLandmarkId; //There are two cases a Trail is created. (1) some MapPoints have projection on the current frame,
    // they initialize some trails with pointer to these MapPoints, and push back to mlTrailers, (2) when there are inadequate
    // tracked points in current frame, this frame is added as a KeyFrame. Some points are detected from it, and added to mlTrailers as starting trailers.
    // when another keyframe is reached, these trailers with null pHingeMP will create a MapPoint. Therefore, the new MapPoint exists
    // between the last KeyFrame and the new KeyFrame
    cv::Point2f fInitPose; //initial position of the trail in srcKF, which is also pHingeMP->pSourceKeyFrame
    cv::Point2f fCurrPose; //current position of the trail in current frame, to store the predicted position
    // by MapPoint or rotation compensation and the tracked position by KLT
    size_t uKeyPointIdx; // id of the key point in the features of the current frame
    size_t uOldKeyPointIdx; // intermediate variable, initialized when used, id of the key point in the features of the previous frame
    size_t uTrackLength; // how many times the features is tracked, start from 1 for first observation
    Eigen::Vector3d p_W;  // an dummy variable position of the point in the world frame only for visualization
    size_t uExternalLmId; //landmark id used by an external program e.g. orb_vo
    uint64_t uCurrentFrameId;
};

/**
 * @brief A frontend using BRISK features
 */
class HybridFrontend {
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

#ifdef DEACTIVATE_TIMERS
  typedef okvis::timing::DummyTimer TimerSwitchable;
#else
  typedef okvis::timing::Timer TimerSwitchable;
#endif

  /**
   * @brief Constructor.
   * @param numCameras Number of cameras in the sensor configuration.
   * @param orbTrackOutput Optional, path to the VO feature track output, used to replace brisk detector and matcher
   */
  HybridFrontend(size_t numCameras, std::string orbTrackOutput= "");
  virtual ~HybridFrontend() {
  }

  ///@{
  /**
   * @brief Detection and descriptor extraction on a per image basis.
   * @remark This method is threadsafe.
   * @param cameraIndex Index of camera to do detection and description.
   * @param frameOut    Multiframe containing the frames.
   *                    Resulting keypoints and descriptors are saved in here.
   * @param T_WC        Pose of camera with index cameraIndex at image capture time.
   * @param[in] keypoints If the keypoints are already available from a different source, provide them here
   *                      in order to skip detection.
   * @warning Using keypoints from a different source is not yet implemented.
   * @return True if successful.
   */
  virtual bool detectAndDescribe(size_t cameraIndex,
                                 std::shared_ptr<okvis::MultiFrame> frameOut,
                                 const okvis::kinematics::Transformation& T_WC,
                                 const std::vector<cv::KeyPoint> * keypoints);

  /**
   * @brief Matching as well as initialization of landmarks and state.
   * @warning This method is not threadsafe.
   * @warning This method uses the estimator. Make sure to not access it in another thread.
   * @param estimator       HybridFilter.
   * @param T_WS_propagated Pose of sensor at image capture time.
   * @param params          Configuration parameters.
   * @param map             Unused.
   * @param framesInOut     Multiframe including the descriptors of all the keypoints.
   * @param[out] asKeyframe Should the frame be a keyframe?
   * @return True if successful.
   */
  virtual bool dataAssociationAndInitialization(
#ifdef USE_MSCKF2
      okvis::MSCKF2& estimator,
#else
      okvis::HybridFilter& estimator,
#endif
      okvis::kinematics::Transformation& T_WS_propagated,
      const okvis::VioParameters & params,
      const std::shared_ptr<okvis::MapPointVector> map,
      std::shared_ptr<okvis::MultiFrame> framesInOut, bool* asKeyframe);

  /**
   * @brief Propagates pose, speeds and biases with given IMU measurements.
   * @see okvis:IMUOdometry::propagation()
   * @remark This method is threadsafe.
   * @param[in] imuMeasurements All the IMU measurements.
   * @param[in] imuParams The parameters to be used.
   * @param[inout] T_WS_propagated Start pose.
   * @param[inout] speedAndBiases Start speed and biases.
   * @param[in] t_start Start time.
   * @param[in] t_end End time.
   * @param[out] covariance Covariance for GIVEN start states.
   * @param[out] jacobian Jacobian w.r.t. start states.
   * @return True on success.
   */
  virtual bool propagation(const okvis::ImuMeasurementDeque & imuMeasurements,
                           const okvis::ImuParameters & imuParams,
                           okvis::kinematics::Transformation& T_WS_propagated,
                           okvis::SpeedAndBiases & speedAndBiases,
                           const okvis::Time& t_start, const okvis::Time& t_end,
                           Eigen::Matrix<double, 15, 15>* covariance,
                           Eigen::Matrix<double, 15, 15>* jacobian) const;

  ///@}
  /// @name Getters related to the BRISK detector
  /// @{

  /// @brief Get the number of octaves of the BRISK detector.
  size_t getBriskDetectionOctaves() const {
    return briskDetectionOctaves_;
  }

  /// @brief Get the detection threshold of the BRISK detector.
  double getBriskDetectionThreshold() const {
    return briskDetectionThreshold_;
  }

  /// @brief Get the absolute threshold of the BRISK detector.
  double getBriskDetectionAbsoluteThreshold() const {
    return briskDetectionAbsoluteThreshold_;
  }

  /// @brief Get the maximum amount of keypoints of the BRISK detector.
  size_t getBriskDetectionMaximumKeypoints() const {
    return briskDetectionMaximumKeypoints_;
  }

  ///@}
  /// @name Getters related to the BRISK descriptor
  /// @{

  /// @brief Get the rotation invariance setting of the BRISK descriptor.
  bool getBriskDescriptionRotationInvariance() const {
    return briskDescriptionRotationInvariance_;
  }

  /// @brief Get the scale invariance setting of the BRISK descriptor.
  bool getBriskDescriptionScaleInvariance() const {
    return briskDescriptionScaleInvariance_;
  }

  ///@}
  /// @name Other getters
  /// @{

  /// @brief Get the matching threshold.
  double getBriskMatchingThreshold() const {
    return briskMatchingThreshold_;
  }

  /// @brief Get the area overlap threshold under which a new keyframe is inserted.
  float getKeyframeInsertionOverlapThershold() const {
    return keyframeInsertionOverlapThreshold_;
  }

  /// @brief Get the matching ratio threshold under which a new keyframe is inserted.
  float getKeyframeInsertionMatchingRatioThreshold() const {
    return keyframeInsertionMatchingRatioThreshold_;
  }

  /// @brief Returns true if the initialization has been completed (RANSAC with actual translation)
  bool isInitialized() {
    return isInitialized_;
  }

  /// @}
  /// @name Setters related to the BRISK detector
  /// @{

  /// @brief Set the number of octaves of the BRISK detector.
  void setBriskDetectionOctaves(size_t octaves) {
    briskDetectionOctaves_ = octaves;
    initialiseBriskFeatureDetectors();
  }

  /// @brief Set the detection threshold of the BRISK detector.
  void setBriskDetectionThreshold(double threshold) {
    briskDetectionThreshold_ = threshold;
    initialiseBriskFeatureDetectors();
  }

  /// @brief Set the absolute threshold of the BRISK detector.
  void setBriskDetectionAbsoluteThreshold(double threshold) {
    briskDetectionAbsoluteThreshold_ = threshold;
    initialiseBriskFeatureDetectors();
  }

  /// @brief Set the maximum number of keypoints of the BRISK detector.
  void setBriskDetectionMaximumKeypoints(size_t maxKeypoints) {
    briskDetectionMaximumKeypoints_ = maxKeypoints;
    initialiseBriskFeatureDetectors();
  }

  /// @}
  /// @name Setters related to the BRISK descriptor
  /// @{

  /// @brief Set the rotation invariance setting of the BRISK descriptor.
  void setBriskDescriptionRotationInvariance(bool invariance) {
    briskDescriptionRotationInvariance_ = invariance;
    initialiseBriskFeatureDetectors();
  }

  /// @brief Set the scale invariance setting of the BRISK descriptor.
  void setBriskDescriptionScaleInvariance(bool invariance) {
    briskDescriptionScaleInvariance_ = invariance;
    initialiseBriskFeatureDetectors();
  }

  ///@}
  /// @name Other setters
  /// @{

  /// @brief Set the matching threshold.
  void setBriskMatchingThreshold(double threshold) {
    briskMatchingThreshold_ = threshold;
  }

  /// @brief Set the area overlap threshold under which a new keyframe is inserted.
  void setKeyframeInsertionOverlapThreshold(float threshold) {
    keyframeInsertionOverlapThreshold_ = threshold;
  }

  /// @brief Set the matching ratio threshold under which a new keyframe is inserted.
  void setKeyframeInsertionMatchingRatioThreshold(float threshold) {
    keyframeInsertionMatchingRatioThreshold_ = threshold;
  }

  //output the distribution of number of features in images
  void printNumFeatureDistribution(std::ofstream & stream);
  /// @}

 private:

  /**
   * @brief   feature detectors with the current settings.
   *          The vector contains one for each camera to ensure that there are no problems with parallel detection.
   * @warning Lock with featureDetectorMutexes_[cameraIndex] when using the detector.
   */
  std::vector<std::shared_ptr<cv::FeatureDetector> > featureDetectors_;
  /**
   * @brief   feature descriptors with the current settings.
   *          The vector contains one for each camera to ensure that there are no problems with parallel detection.
   * @warning Lock with featureDetectorMutexes_[cameraIndex] when using the descriptor.
   */
  std::vector<std::shared_ptr<cv::DescriptorExtractor> > descriptorExtractors_;
  /// Mutexes for feature detectors and descriptors.
  std::vector<std::unique_ptr<std::mutex> > featureDetectorMutexes_;

  bool isInitialized_;        ///< Is the pose initialised?
  const size_t numCameras_;   ///< Number of cameras in the configuration.

  /// @name BRISK detection parameters
  /// @{

  size_t briskDetectionOctaves_;            ///< The set number of brisk octaves.
  double briskDetectionThreshold_;          ///< The set BRISK detection threshold.
  double briskDetectionAbsoluteThreshold_;  ///< The set BRISK absolute detection threshold.
  size_t briskDetectionMaximumKeypoints_;   ///< The set maximum number of keypoints.

  /// @}
  /// @name BRISK descriptor extractor parameters
  /// @{

  bool briskDescriptionRotationInvariance_; ///< The set rotation invariance setting.
  bool briskDescriptionScaleInvariance_;    ///< The set scale invariance setting.

  ///@}
  /// @name BRISK matching parameters
  ///@{

  double briskMatchingThreshold_; ///< The set BRISK matching threshold.

  ///@}

  std::unique_ptr<okvis::DenseMatcher> matcher_; ///< Matcher object.

  /**
   * @brief If the hull-area around all matched keypoints of the current frame (with existing landmarks)
   *        divided by the hull-area around all keypoints in the current frame is lower than
   *        this threshold it should be a new keyframe.
   * @see   doWeNeedANewKeyframe()
   */
  float keyframeInsertionOverlapThreshold_;  //0.6
  /**
   * @brief If the number of matched keypoints of the current frame with an older frame
   *        divided by the amount of points inside the convex hull around all keypoints
   *        is lower than the threshold it should be a keyframe.
   * @see   doWeNeedANewKeyframe()
   */
  float keyframeInsertionMatchingRatioThreshold_;  //0.2

  /**
   * @brief Decision whether a new frame should be keyframe or not.
   * @param estimator     const reference to the estimator.
   * @param currentFrame  Keyframe candidate.
   * @return True if it should be a new keyframe.
   */
  bool doWeNeedANewKeyframe(
#ifdef USE_MSCKF2
    const okvis::MSCKF2& estimator,
#else
    const okvis::HybridFilter& estimator,
#endif
                            std::shared_ptr<okvis::MultiFrame> currentFrame);  // based on some overlap area heuristics

  /**
   * @brief Match a new multiframe to existing keyframes
   * @tparam MATCHING_ALGORITHM Algorithm to match new keypoints to existing landmarks
   * @warning As this function uses the estimator it is not threadsafe
   * @param      estimator              HybridFilter.
   * @param[in]  params                 Parameter struct.
   * @param[in]  currentFrameId         ID of the current frame that should be matched against keyframes.
   * @param[out] rotationOnly           Was the rotation only RANSAC motion model good enough to
   *                                    explain the motion between the new frame and the keyframes?
   * @param[in]  usePoseUncertainty     Use the pose uncertainty for the matching.
   * @param[out] uncertainMatchFraction Return the fraction of uncertain matches. Set to nullptr if not interested.
   * @param[in]  removeOutliers         Remove outliers during RANSAC.
   * @return The number of matches in total.
   */
  template<class MATCHING_ALGORITHM>
  int matchToKeyframes(
#ifdef USE_MSCKF2
    okvis::MSCKF2& estimator,
#else
    okvis::HybridFilter& estimator,
#endif
                       const okvis::VioParameters& params,
                       const uint64_t currentFrameId, bool& rotationOnly,
                       bool usePoseUncertainty = true,
                       double* uncertainMatchFraction = 0,
                       bool removeOutliers = true);  // for wide-baseline matches (good initial guess)

  /**
   * @brief Match a new multiframe to the last frame.
   * @tparam MATCHING_ALGORITHM Algorithm to match new keypoints to existing landmarks
   * @warning As this function uses the estimator it is not threadsafe.
   * @param estimator           HybridFilter.
   * @param params              Parameter struct.
   * @param currentFrameId      ID of the current frame that should be matched against the last one.
   * @param usePoseUncertainty  Use the pose uncertainty for the matching.
   * @param removeOutliers      Remove outliers during RANSAC.
   * @return The number of matches in total.
   */
  template<class MATCHING_ALGORITHM>
  int matchToLastFrame(
#ifdef USE_MSCKF2
    okvis::MSCKF2& estimator,
#else
    okvis::HybridFilter& estimator,
#endif
                       const okvis::VioParameters& params,
                       const uint64_t currentFrameId,
                       bool usePoseUncertainty = true,
                       bool removeOutliers = true);

  /**
   * @brief Match the frames inside the multiframe to each other to initialise new landmarks.
   * @tparam MATCHING_ALGORITHM Algorithm to match new keypoints to existing landmarks.
   * @warning As this function uses the estimator it is not threadsafe.
   * @param estimator   HybridFilter.
   * @param multiFrame  Multiframe containing the frames to match.
   */
  template<class MATCHING_ALGORITHM>
  void matchStereo(
#ifdef USE_MSCKF2
    okvis::MSCKF2& estimator,
#else
    okvis::HybridFilter& estimator,
#endif
                   std::shared_ptr<okvis::MultiFrame> multiFrame);

  /**
   * @brief Perform 3D/2D RANSAC.
   * @warning As this function uses the estimator it is not threadsafe.
   * @param estimator       HybridFilter.
   * @param nCameraSystem   Camera configuration and parameters.
   * @param currentFrame    Frame with the new potential matches.
   * @param removeOutliers  Remove observation of outliers in estimator.
   * @return Number of inliers.
   */
  int runRansac3d2d(
#ifdef USE_MSCKF2
    okvis::MSCKF2& estimator,
#else
    okvis::HybridFilter& estimator,
#endif
                    const okvis::cameras::NCameraSystem &nCameraSystem,
                    std::shared_ptr<okvis::MultiFrame> currentFrame,
                    bool removeOutliers);

  /**
   * @brief Perform 2D/2D RANSAC.
   * @warning As this function uses the estimator it is not threadsafe.
   * @param estimator         HybridFilter.
   * @param params            Parameter struct.
   * @param currentFrameId    ID of the new multiframe containing matches with the frame with ID olderFrameId.
   * @param olderFrameId      ID of the multiframe to which the current frame has been matched against.
   * @param initializePose    If the pose has not yet been initialised should the function try to initialise it.
   * @param removeOutliers    Remove observation of outliers in estimator.
   * @param[out] rotationOnly Was the rotation only RANSAC model enough to explain the matches.
   * @return Number of inliers.
   */
  int runRansac2d2d(
#ifdef USE_MSCKF2
    okvis::MSCKF2& estimator,
#else
    okvis::HybridFilter& estimator,
#endif
                    const okvis::VioParameters& params, uint64_t currentFrameId,
                    uint64_t olderFrameId, bool initializePose,
                    bool removeOutliers, bool &rotationOnly);

  /// (re)instantiates feature detectors and descriptor extractors. Used after settings changed or at startup.
  void initialiseBriskFeatureDetectors();


  ///for KLT tracking
  bool TrailTracking_Start();
  int TrailTracking_DetectAndInsert(const cv::Mat &currentFrame, int nInliers, std::vector<cv::KeyPoint>&);
  int TrailTracking_Advance(
  #ifdef USE_MSCKF2
          okvis::MSCKF2& estimator,
  #else
          okvis::HybridFilter& estimator,
  #endif
          uint64_t mfIdA, uint64_t mfIdB, size_t camIdA, size_t camIdB);

  /**
   * @brief TrailTracking_Advance2
   * @param keypoints
   * @param mapPointIds
   * @param mapPointPositions
   * @param currentImage
   * @return
   */
  int TrailTracking_Advance2(const std::vector<cv::KeyPoint>& keypoints, const std::vector<size_t>& mapPointIds,
  const std::vector<Eigen::Vector3d>& mapPointPositions, uint64_t mfIdB, const cv::Mat currentImage);

  void UpdateEstimatorObservations(
  #ifdef USE_MSCKF2
          okvis::MSCKF2& estimator,
  #else
          okvis::HybridFilter& estimator,
  #endif
          uint64_t mfIdA, uint64_t mfIdB, size_t camIdA, size_t camIdB);

  // works for orb_vo
  void UpdateEstimatorObservations2(
  #ifdef USE_MSCKF2
          okvis::MSCKF2& estimator,
  #else
          okvis::HybridFilter& estimator,
  #endif
          uint64_t mfIdA, uint64_t mfIdB, size_t camIdA, size_t camIdB);


  std::list<Trailer> mlTrailers; // reference to lTrailers in map
  std::vector<cv::Mat> mCurrentPyramid, mPreviousPyramid;
  uint64_t mPreviousFrameId, mCurrentFrameId;
  size_t mMaxFeaturesInFrame, mMinFeaturesInFrame;
  size_t mFrameCounter; //how many frames have been processed by the frontend
  cv::Mat mMask; // mask used to protect existing features during initializing
                 // features
  //  size_t mTrackedPoints; // number of well tracked points in the current
  //  frame
  std::vector<cv::KeyPoint> mvKeyPoints; //key points in the current frame

  TrackResultReader* pTracker;
  // an accumulator for number of features distribution
  // TODO(jhuai): for now only implemented for features tracked by an external
  // module, ORB-VO, for KLT and OKVIS feature tracking to be implemented
  MyAccumulator myAccumulator;
};

}  // namespace okvis

#endif // INCLUDE_OKVIS_HYBRID_FRONTEND_HPP_
