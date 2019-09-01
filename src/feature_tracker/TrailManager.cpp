#include "feature_tracker/TrailManager.h"
#include <okvis/IdProvider.hpp>
namespace feature_tracker {

TrailManager::TrailManager(const std::string& orbTrackOutput)
    : myAccumulator(boost::accumulators::tag::density::num_bins = 20,
                    boost::accumulators::tag::density::cache_size = 40) {
  if (!orbTrackOutput.empty()) {
    pTracker = new TrackResultReader(orbTrackOutput);
  } else {
    pTracker = nullptr;
  }
}

TrailManager::~TrailManager() {
  if (pTracker != nullptr) {
    delete pTracker;
    pTracker = nullptr;
  }
}
/**
 * The current frame is to be the first frame
 */
bool TrailManager::initialize() {
  // detect good features
  std::vector<cv::Point2f> vfPoints;
  size_t nToAdd = 200;  // MaxInitialTrails
  mMaxFeaturesInFrame = nToAdd;
  size_t nToThrow = 100;  // MinInitialTrails
  mMinFeaturesInFrame = nToThrow;
  mFrameCounter = 0;
  int nSubPixWinWidth = 7;  // SubPixWinWidth
  cv::Size subPixWinSize(nSubPixWinWidth, nSubPixWinWidth);

  cv::TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10, 0.03);
  cv::goodFeaturesToTrack(mCurrentPyramid[0], vfPoints, nToAdd, 0.01, 10,
                          cv::Mat(), 3, false, 0.04);
  cv::cornerSubPix(mCurrentPyramid[0], vfPoints, subPixWinSize,
                   cv::Size(-1, -1), termcrit);
  if (vfPoints.size() < nToThrow) {
    OKVIS_ASSERT_GT(Exception, vfPoints.size(), nToThrow,
                    "No. features in the first frame is too few");
    return false;
  }
  mvKeyPoints.clear();
  mvKeyPoints.reserve(mMaxFeaturesInFrame + 50);
  mFeatureTrailList.clear();
  for (size_t i = 0; i < vfPoints.size(); i++) {
    FeatureTrail t(vfPoints[i], vfPoints[i], 0, i);
    mFeatureTrailList.push_back(t);
    mvKeyPoints.push_back(cv::KeyPoint(vfPoints[i], 8.f));
  }
  std::cout << "create new points " << mvKeyPoints.size() << " at the start "
            << std::endl;
  return true;
}

/**
 * Steady-state trail tracking: Advance from the previous frame,
 * update mFeatureTrailList, remove duds.
 * (1) track the points kept in mFeatureTrailList which contains MapPoints in
 * the last few keyframes and points just detected from the last keyframe since
 * these points appear in the last frame which is stored, we can track these
 * points in the current frame by using KLT tracker
 * @return number of good trails
 */
int TrailManager::advance(
    okvis::HybridFilter& estimator,
    uint64_t mfIdA, uint64_t mfIdB, size_t camIdA, size_t camIdB) {
  typedef okvis::cameras::PinholeCamera<
      okvis::cameras::RadialTangentialDistortion>
      CAMERA_GEOMETRY_T;
  int nGoodTrails = 0;

  // vfPoints[0-1] for forward searching,
  // vfPoints[2-3] for backward searching
  std::vector<cv::Point2f> vfPoints[3];
  // status[0] forward searching indicator
  // status[1] backward searching indicator
  std::vector<uchar> status[2];
  std::vector<float> err;

  std::shared_ptr<okvis::MultiFrame> frameA = estimator.multiFrame(mfIdA);
  std::shared_ptr<okvis::MultiFrame> frameB = estimator.multiFrame(mfIdB);

  // calculate the relative transformations and uncertainties
  // TODO(sleuten): donno, if and what we need here - I'll see

  okvis::kinematics::Transformation T_CaCb;
  okvis::kinematics::Transformation T_CbCa;
  okvis::kinematics::Transformation T_SaCa;
  okvis::kinematics::Transformation T_SbCb;
  okvis::kinematics::Transformation T_WSa;
  okvis::kinematics::Transformation T_WSb;
  okvis::kinematics::Transformation T_SaW;
  okvis::kinematics::Transformation T_SbW;
  okvis::kinematics::Transformation T_WCa;
  okvis::kinematics::Transformation T_WCb;
  okvis::kinematics::Transformation T_CaW;
  okvis::kinematics::Transformation T_CbW;

  estimator.getCameraSensorStates(mfIdA, camIdA, T_SaCa);
  estimator.getCameraSensorStates(mfIdB, camIdB, T_SbCb);
  estimator.get_T_WS(mfIdA, T_WSa);
  estimator.get_T_WS(mfIdB, T_WSb);
  T_SaW = T_WSa.inverse();
  T_SbW = T_WSb.inverse();
  T_WCa = T_WSa * T_SaCa;
  T_WCb = T_WSb * T_SbCb;
  T_CaW = T_WCa.inverse();
  T_CbW = T_WCb.inverse();
  T_CaCb = T_WCa.inverse() * T_WCb;
  T_CbCa = T_CaCb.inverse();

  // assume frameA and frameB of the same size
  float imageW = static_cast<float>(
      frameA->geometryAs<CAMERA_GEOMETRY_T>(camIdA)->imageWidth());
  float imageH = static_cast<float>(
      frameA->geometryAs<CAMERA_GEOMETRY_T>(camIdA)->imageHeight());

  size_t mMeasAttempted = 0;
  vfPoints[0].reserve(mFeatureTrailList.size());
  vfPoints[1].reserve(mFeatureTrailList.size());
  for (std::list<FeatureTrail>::const_iterator it = mFeatureTrailList.begin();
       it != mFeatureTrailList.end(); ++it) {
    // TODO(jhuai): predict point position with its world position when
    // it->mLandmarkId is not 0.
    // To implement it cf. VioFrameMatchingAlgorithm doSetup

    // predict pose in the current pose by rotation compensation,
    // assume the camera tranlates very small compared to the point depth
    // if it's not projected successfully, we alias it to prevent it
    // being tracked
    //            Eigen::Vector2d kptA(it->fCurrPose.x, it->fCurrPose.y);
    //            Eigen::Vector3d dirVecA;
    //            if
    //            (!frameA->geometryAs<CAMERA_GEOMETRY_T>(camIdA)->backProject(
    //                kptA, &dirVecA)) {
    //                // pose in last frame
    //                vfPoints[0].push_back(cv::Point2f(-1,-1));
    //                // preventing detect anything
    //                vfPoints[1].push_back(cv::Point2f(-1,-1));
    //                continue;
    //            }
    //            Eigen::Vector3d dirVecB = T_CbCa.C()*dirVecA;
    //            Eigen::Vector2d kptB;
    //            if(frameB->geometryAs<CAMERA_GEOMETRY_T>(camIdB)
    //                ->project(dirVecB, &kptB) !=
    //                okvis::cameras::CameraBase::
    //                ProjectionStatus::Successful) {
    //                // pose in last frame
    //                vfPoints[0].push_back(cv::Point2f(-1,-1));
    //                // preventing detect anything
    //                vfPoints[1].push_back(cv::Point2f(-1,-1));
    //                continue;
    //            }

    vfPoints[0].push_back(it->fCurrPose);  // point in previous frame
    //            vfPoints[1].push_back(cv::Point2f((float)kptB[0],(float)kptB[1]));
    vfPoints[1].push_back(it->fCurrPose);
    ++mMeasAttempted;
  }

  int nWinWidth = 15;  // WinWidth
  cv::Size winSize(nWinWidth, nWinWidth);
  cv::TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10, 0.03);
  cv::calcOpticalFlowPyrLK(mPreviousPyramid, mCurrentPyramid, vfPoints[0],
                           vfPoints[1], status[0], err, winSize, 3, termcrit,
                           cv::OPTFLOW_USE_INITIAL_FLOW, 0.001);
  // what is returned for points failed to track? the initial position
  // if a point is out of boundary, it will ended up failing to track
  vfPoints[2] = vfPoints[0];
  cv::calcOpticalFlowPyrLK(mCurrentPyramid, mPreviousPyramid, vfPoints[1],
                           vfPoints[2], status[1], err, winSize, 3, termcrit,
                           cv::OPTFLOW_USE_INITIAL_FLOW, 0.001);

  mvKeyPoints.clear();
  mvKeyPoints.reserve(mMaxFeaturesInFrame + 100);  // keypoints in frame B
  std::list<FeatureTrail>::iterator it = mFeatureTrailList.begin();
  // note status[0].size() is constant, but not mFeatureTrailList.size()
  for (size_t which = 0; which < status[0].size(); ++which) {
    if (status[0][which]) {
      cv::Point2f delta = vfPoints[2][which] - vfPoints[0][which];
      bool bInImage =
          (vfPoints[0][which].x >= 0.f) && (vfPoints[0][which].y >= 0.f) &&
          (vfPoints[1][which].x >= 0.f) && (vfPoints[1][which].y >= 0.f) &&
          (vfPoints[0][which].x <= (imageW - 1.f)) &&
          (vfPoints[0][which].y <= (imageH - 1.f)) &&
          (vfPoints[1][which].x <= (imageW - 1.f)) &&
          (vfPoints[1][which].y <= (imageH - 1.f));
      if (!status[1][which] || !bInImage || (delta.dot(delta)) > 2)
        status[0][which] = 0;

      //           if(!bInImage){
      //               std::cout << "vPoint[0][which] "<< vfPoints[0][which].x
      //               <<" "
      //                         << vfPoints[0][which].y << " "
      //                         << vfPoints[1][which].x << " "
      //                         << vfPoints[1][which].y <<" "<<std::endl;
      //           }

      it->fCurrPose = vfPoints[1][which];
    }
    // Erase from list of trails if not found this frame.
    if (!status[0][which]) {
      it = mFeatureTrailList.erase(it);
    } else {
      it->uOldKeyPointIdx = it->uKeyPointIdx;
      it->uKeyPointIdx = nGoodTrails;
      ++it->uTrackLength;

      mvKeyPoints.push_back(cv::KeyPoint(it->fCurrPose, 8.f));
      ++it;
      ++nGoodTrails;
    }
  }

  assert(mFeatureTrailList.end() == it &&
         static_cast<int>(mvKeyPoints.size()) == nGoodTrails);

  /*//added by huai for debug
  cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
  // CAUTION: do not draw on pyramid[0], ruining the data
  cv::Mat image=mCurrentKF.pyramid[0].clone();
  for (list<FeatureTrail>::const_iterator it = mFeatureTrailList.begin();
      it!=mFeatureTrailList.end();++it) {
      line(image, it->fInitPose, it->fCurrPose, CV_RGB(0,0,0));
      circle( image, it->fCurrPose, 3, Scalar(255,0,0), -1, 8);
  }
  //CVDRGB2OpenCVRGB(mimFrameRGB, image);
  cv::imshow( "Display window", image );
  cv::waitKey(0);
  */
  ++mFrameCounter;
  return nGoodTrails;
}

int TrailManager::advance2(
    const std::vector<cv::KeyPoint>& keypoints,
    const std::vector<size_t>& mapPointIds,
    const std::vector<Eigen::Vector3d,
                      Eigen::aligned_allocator<Eigen::Vector3d>>&
        mapPointPositions,
    uint64_t mfIdB) {
  int nGoodTrails = 0;
  int nTrackedFeatures = 0;
  // build a map of mappoint ids for accelerating searching
  std::map<size_t, std::list<FeatureTrail>::iterator> mpId2Trailer;

  for (std::list<FeatureTrail>::iterator iter = mFeatureTrailList.begin(),
                                         itEnd = mFeatureTrailList.end();
       iter != itEnd; ++iter) {
    mpId2Trailer[iter->uExternalLmId] = iter;
  }

  std::list<FeatureTrail> tempTrailers;
  auto it = mapPointIds.begin();
  size_t counter = 0;
  while (it != mapPointIds.end()) {
    if (*it == 0) {
      ++it;
      ++counter;
      continue;
    }
    auto iterMap = mpId2Trailer.find(*it);
    if (iterMap != mpId2Trailer.end()) {
      std::list<FeatureTrail>::iterator tp = iterMap->second;
      tp->uOldKeyPointIdx = tp->uKeyPointIdx;
      tp->uKeyPointIdx = counter;
      tp->fCurrPose = keypoints[counter].pt;
      tp->p_W.x = mapPointPositions[counter][0];
      tp->p_W.y = mapPointPositions[counter][1];
      tp->p_W.z = mapPointPositions[counter][2];
      tp->uCurrentFrameId = mfIdB;
      ++(tp->uTrackLength);
      ++nGoodTrails;
    } else {  // a new trail
      FeatureTrail t(keypoints[counter].pt, keypoints[counter].pt, 0, counter,
                     *it);
      t.p_W.x = mapPointPositions[counter][0];
      t.p_W.y = mapPointPositions[counter][1];
      t.p_W.z = mapPointPositions[counter][2];
      t.uCurrentFrameId = mfIdB;
      tempTrailers.push_back(t);
    }
    ++nTrackedFeatures;
    ++it;
    ++counter;
  }

  assert(counter == mapPointIds.size());
  mFeatureTrailList.insert(mFeatureTrailList.end(), tempTrailers.begin(),
                           tempTrailers.end());
  myAccumulator(nTrackedFeatures);
  ++mFrameCounter;
  return nGoodTrails;
}

// detect some points from just inserted frame, curKF, and insert them
// into mFeatureTrailList
int TrailManager::detectAndInsert(const cv::Mat& currentFrame, int nInliers,
                                  std::vector<cv::KeyPoint>& vNewKPs) {
  vNewKPs.clear();
  vNewKPs.reserve(500);
  // how many keypoints already in the current frame?
  const size_t startIndex = mvKeyPoints.size();
  // detect good features
  std::vector<cv::Point2f> vfPoints;
  int nToAdd = mMaxFeaturesInFrame - nInliers + 50;
  int nSubPixWinWidth = 7;  // SubPixWinWidth
  cv::Size subPixWinSize(nSubPixWinWidth, nSubPixWinWidth);
  cv::TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10, 0.03);
  // create mask
  if (mMask.empty()) mMask.create(currentFrame.size(), CV_8UC1);
  mMask = cv::Scalar(255);
  for (std::list<FeatureTrail>::const_iterator lTcIt =
           mFeatureTrailList.begin();
       lTcIt != mFeatureTrailList.end(); ++lTcIt) {
    cv::circle(mMask, lTcIt->fCurrPose, 11, cv::Scalar(0), CV_FILLED);
  }
  double minDist = std::min(currentFrame.size().width / 100 + 1.0, 15.0);
  cv::goodFeaturesToTrack(currentFrame, vfPoints, nToAdd, 0.01, minDist, mMask,
                          3, false, 0.04);
  cornerSubPix(currentFrame, vfPoints, subPixWinSize, cv::Size(-1, -1),
               termcrit);

  std::cout << "Detected additional " << vfPoints.size() << " points"
            << std::endl;
  /*//use grid for even distribution. Emprically, without even distribution,
  // rotations are already very good
  float bucket_width=50, bucket_height=50, max_features_bin=4;
  cv::Mat grid((int)ceil(currentFrame.size().width/bucket_width),
               (int)ceil(currentFrame.size().height/bucket_height),
               CV_8UC1, Scalar::all(0));
  for (list<FeatureTrail>::const_iterator lTcIt=mFeatureTrailList.begin();
      lTcIt!=mFeatureTrailList.end(); ++lTcIt) {
      ++grid.at<unsigned char>((int)(lTcIt->fCurrPose.x/bucket_width),
                               (int)(lTcIt->fCurrPose.y/bucket_height));
  }
  // feature bucketing: keeps only max_features per bucket, where the domain
  // is split into buckets of size (bucket_width,bucket_height)
  */
  size_t jack = 0u;
  for (size_t i = 0; i < vfPoints.size(); i++) {
    /*if (grid.at<unsigned char>((int)(vfPoints[i].x/bucket_width),
        (int)(vfPoints[i].y/bucket_height))>=max_features_bin)
        continue;
    else
        ++grid.at<unsigned char>((int)(vfPoints[i].x/bucket_width),
            (int)(vfPoints[i].y/bucket_height));*/
    FeatureTrail t(vfPoints[i], vfPoints[i], 0, startIndex + jack);
    mFeatureTrailList.push_back(t);
    vNewKPs.push_back(cv::KeyPoint(vfPoints[i], 8.f));
    ++jack;
  }

  /*//added by huai for debug
      cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
      // CAUTION: do not draw on pyramid[0], ruining the data
      cv::Mat image=mCurrentKF.pyramid[0].clone();
      for (list<FeatureTrail>::const_iterator i = mFeatureTrailList.begin();
          i!=mFeatureTrailList.end();++i) {
      //	line(image, i->fInitPose, i->fCurrPose, CV_RGB(0,0,0));
          circle( image, i->fCurrPose, 3, Scalar(255,0,0), -1, 8);
      }
      //CVDRGB2OpenCVRGB(mimFrameRGB, image);
      cv::imshow( "Display window", image );
      cv::waitKey(0);
  */
  mvKeyPoints.insert(mvKeyPoints.end(), vNewKPs.begin(), vNewKPs.end());
  return vNewKPs.size();
}

void TrailManager::updateEstimatorObservations(
    okvis::HybridFilter& estimator,
    uint64_t mfIdA, uint64_t mfIdB, size_t camIdA, size_t camIdB) {
  // update estimator's observations and keypoints
  typedef okvis::cameras::PinholeCamera<
      okvis::cameras::RadialTangentialDistortion>
      camera_geometry_t;
  std::shared_ptr<okvis::MultiFrame> frameA = estimator.multiFrame(mfIdA);
  std::shared_ptr<okvis::MultiFrame> frameB = estimator.multiFrame(mfIdB);
  for (auto it = mFeatureTrailList.begin(); it != mFeatureTrailList.end();
       ++it) {
    if (it->uLandmarkId != 0) {
      frameB->setLandmarkId(camIdB, it->uKeyPointIdx, it->uLandmarkId);
      estimator.addObservation<camera_geometry_t>(it->uLandmarkId, mfIdB,
                                                  camIdB, it->uKeyPointIdx);
    } else if (it->uTrackLength == 2) {
      uint64_t lmId = okvis::IdProvider::instance().newId();
      Eigen::Matrix<double, 4, 1> hP_W;
      hP_W.setOnes();  // random initialization
      bool canBeInitialized = true;
      estimator.addLandmark(lmId, hP_W);
      OKVIS_ASSERT_TRUE(Exception, estimator.isLandmarkAdded(lmId),
                        lmId << " not added, bug");
      estimator.setLandmarkInitialized(lmId, canBeInitialized);

      frameA->setLandmarkId(camIdA, it->uOldKeyPointIdx, lmId);
      estimator.addObservation<camera_geometry_t>(lmId, mfIdA, camIdA,
                                                  it->uOldKeyPointIdx);

      it->uLandmarkId = lmId;
      frameB->setLandmarkId(camIdB, it->uKeyPointIdx, lmId);

      estimator.addObservation<camera_geometry_t>(lmId, mfIdB, camIdB,
                                                  it->uKeyPointIdx);
    }
  }
}

void TrailManager::updateEstimatorObservations2(
    okvis::HybridFilter& estimator,
    uint64_t mfIdA, uint64_t mfIdB, size_t camIdA, size_t camIdB) {
  const size_t maxTrackLength = 15;  // break the track once it hits this val
  // update estimator's observations and keypoints
  typedef okvis::cameras::PinholeCamera<
      okvis::cameras::RadialTangentialDistortion>
      camera_geometry_t;
  std::shared_ptr<okvis::MultiFrame> frameA = estimator.multiFrame(mfIdA);
  std::shared_ptr<okvis::MultiFrame> frameB = estimator.multiFrame(mfIdB);
  int deletedTrails = 0;
  for (auto it = mFeatureTrailList.begin(); it != mFeatureTrailList.end();) {
    if (it->uCurrentFrameId < mfIdB) {
      if (it->uTrackLength > 1) ++deletedTrails;
      it = mFeatureTrailList.erase(it);
      continue;
    } else {
      assert(it->uCurrentFrameId == mfIdB);
    }

    if (it->uLandmarkId != 0) {
      if (it->uTrackLength < maxTrackLength) {
        estimator.addObservation<camera_geometry_t>(it->uLandmarkId, mfIdB,
                                                    camIdB, it->uKeyPointIdx);
      } else {  // break up the trail
        it->fInitPose = it->fCurrPose;
        it->uLandmarkId = 0;
        it->uTrackLength = 1;
      }
    } else if (it->uTrackLength == 2) {
      // two observations, we skip those for new features of an observation
      uint64_t lmId = okvis::IdProvider::instance().newId();

      Eigen::Matrix<double, 4, 1> hP_W;
      hP_W[0] = it->p_W.x;
      hP_W[1] = it->p_W.y;
      hP_W[2] = it->p_W.z;
      hP_W[3] = 1;
      bool canBeInitialized = true;
      estimator.addLandmark(lmId, hP_W);
      OKVIS_ASSERT_TRUE(Exception, estimator.isLandmarkAdded(lmId),
                        lmId << " not added, bug");
      estimator.setLandmarkInitialized(lmId, canBeInitialized);

      frameA->setLandmarkId(camIdA, it->uOldKeyPointIdx, lmId);
      estimator.addObservation<camera_geometry_t>(lmId, mfIdA, camIdA,
                                                  it->uOldKeyPointIdx);

      it->uLandmarkId = lmId;
      frameB->setLandmarkId(camIdB, it->uKeyPointIdx, lmId);
      estimator.addObservation<camera_geometry_t>(lmId, mfIdB, camIdB,
                                                  it->uKeyPointIdx);
    } else {
      assert(it->uTrackLength == 1);
    }
    ++it;
  }
  std::cout << "Trails after deleted " << deletedTrails << " "
            << mFeatureTrailList.size() << std::endl;
}

void TrailManager::printNumFeatureDistribution(std::ofstream& stream) const {
  if (!stream.is_open()) {
    std::cerr << "cannot write feature distribution without proper stream "
              << std::endl;
  }
  std::size_t num_samples = boost::accumulators::count(myAccumulator);
  if (num_samples == 0) {
    stream << "Skip computing the histogram of number of features in "
              "images as no sample is stored!"
           << std::endl;
    return;
  }
  histogram_type hist = boost::accumulators::density(myAccumulator);

  double total = 0.0;
  stream << "histogram of number of features in images "
            "(bin lower bound, value)"
         << std::endl;
  for (size_t i = 0; i < hist.size(); i++) {
    stream << hist[i].first << " " << hist[i].second << std::endl;
    total += hist[i].second;
  }
  std::cout << "Total of densities: " << total << " should be 1." << std::endl;
}

std::vector<cv::KeyPoint> TrailManager::getCurrentKeypoints() const {
  return mvKeyPoints;
}

bool TrailManager::needToDetectMorePoints(int matches2d2d) {
  return matches2d2d < 0.5 * mMaxFeaturesInFrame ||
         (mFrameCounter % 4 == 0 && matches2d2d < 0.6 * mMaxFeaturesInFrame);
}

const std::list<FeatureTrail>& TrailManager::getFeatureTrailList() const {
  return mFeatureTrailList;
}

void TrailManager::initialize2(
    const std::vector<cv::KeyPoint>& keypoints,
    const std::vector<size_t>& mapPointIds,
    const std::vector<Eigen::Vector3d,
                      Eigen::aligned_allocator<Eigen::Vector3d>>&
        mapPointPositions,
    const uint64_t currentFrameId) {
  // should be second keyframe in orb_vo
  // initialize for the first frame
  size_t nToAdd = 200;  // MaxInitialTrails
  mMaxFeaturesInFrame = nToAdd;
  size_t nToThrow = 100;  // MinInitialTrails
  mMinFeaturesInFrame = nToThrow;
  mFrameCounter = 0;

  for (size_t i = 0; i < keypoints.size(); i++) {
    if (mapPointIds[i] != 0) {
      feature_tracker::FeatureTrail t(keypoints[i].pt, keypoints[i].pt, 0, i,
                                      mapPointIds[i]);
      t.p_W.x = mapPointPositions[i][0];
      t.p_W.y = mapPointPositions[i][1];
      t.p_W.z = mapPointPositions[i][2];
      t.uCurrentFrameId = currentFrameId;
      mFeatureTrailList.push_back(t);
    }
  }
  std::cout << "Initializing at frameid " << currentFrameId
            << " with new trails " << mFeatureTrailList.size() << std::endl;
}
}  // namespace feature_tracker
