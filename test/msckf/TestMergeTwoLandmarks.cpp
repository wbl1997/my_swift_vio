#include "gtest/gtest.h"
#include <okvis/FrameTypedefs.hpp>
#include <okvis/MultiFrame.hpp>

TEST(StdC, swap) {
  int a = 100, b = 200;
  int &r1 = a, &r2 = b;
  std::swap(r1, r2);

  EXPECT_EQ(r1, a);
  EXPECT_EQ(r2, b);
  EXPECT_EQ(a, 200);
  EXPECT_EQ(b, 100);
}

TEST(StdC, swapComplex) {
  std::map<int, int> a{{1, 2}, {3, 4}}, b{{5, 6}, {7, 8}};
  std::map<int, int>::iterator itl = a.begin(), itr = b.begin();
  std::map<int, int>&refl = a, &refr = b;
  std::swap(refl, refr);
  std::swap(itl, itr);

  std::map<int, int>::const_iterator itera = a.begin();

  for (auto iter = itl; iter != refl.end(); ++iter, ++itera) {
    EXPECT_EQ(iter->first, itera->first);
    EXPECT_EQ(iter->second, itera->second);
  }
  std::map<int, int>::const_iterator iterb = b.begin();
  for (auto iter = itr; iter != refr.end(); ++iter, ++iterb) {
    EXPECT_EQ(iter->first, iterb->first);
    EXPECT_EQ(iter->second, iterb->second);
  }
}

class MergeTwoLandmarksTest : public testing::Test {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  MergeTwoLandmarksTest() {
    okvis::MapPoint mpa;
    mpa.observations.emplace(okvis::KeypointIdentifier(200, 0, 0), 100);
    mpa.observations.emplace(okvis::KeypointIdentifier(300, 0, 0), 101);
    landmarksMap_.emplace(1, mpa);
    okvis::MapPoint mpb;
    mpb.observations.emplace(okvis::KeypointIdentifier(200, 1, 0), 102);
    mpb.observations.emplace(okvis::KeypointIdentifier(300, 1, 0), 103);
    mpb.observations.emplace(okvis::KeypointIdentifier(400, 1, 0), 104);
    mpb.observations.emplace(okvis::KeypointIdentifier(500, 1, 0), 105);
    landmarksMap_.emplace(2, mpb);
  }

  void merge(uint64_t lmIdA, uint64_t lmIdB) {
    okvis::PointMap::iterator lmItA = landmarksMap_.find(lmIdA);
    okvis::PointMap::iterator lmItB = landmarksMap_.find(lmIdB);
    if (lmItB->second.observations.size() > lmItA->second.observations.size()) {
      std::swap(lmIdA, lmIdB);
      std::swap(lmItA, lmItB);
    }
    std::map<okvis::KeypointIdentifier, uint64_t>& obsMapA =
        lmItA->second.observations;
    std::map<okvis::KeypointIdentifier, uint64_t>& obsMapB =
        lmItB->second.observations;
    EXPECT_EQ(obsMapA.size(), 4u);
    EXPECT_EQ(obsMapB.size(), 2u);
    for (std::map<okvis::KeypointIdentifier, uint64_t>::iterator obsIter =
             obsMapB.begin();
         obsIter != obsMapB.end(); ++obsIter) {
      if (obsIter->second != 0u) {
        //      mapPtr_->removeResidualBlock(
        //          reinterpret_cast<::ceres::ResidualBlockId>(obsIter->second));
        obsIter->second = 0u;
      }
      // reset landmark ids for relevant keypoints in multiframe.
      const okvis::KeypointIdentifier& kpi = obsIter->first;
      //    okvis::MultiFramePtr multiFramePtr =
      //    multiFramePtrMap_.at(kpi.frameId); auto iterA =
      //    std::find_if(obsMapA.begin(), obsMapA.end(),
      //                              okvis::IsObservedInFrame(kpi.frameId,
      //                              kpi.cameraIndex));
      //    if (iterA != obsMapA.end()) {
      //      multiFramePtr->setLandmarkId(kpi.cameraIndex, kpi.keypointIndex,
      //      0u); continue;
      //    }
      //    multiFramePtr->setLandmarkId(kpi.cameraIndex, kpi.keypointIndex,
      //    lmIdA);

      //    Eigen::Vector2d measurement;
      //    multiFramePtr->getKeypoint(kpi.cameraIndex, kpi.keypointIndex,
      //    measurement); Eigen::Matrix2d information =
      //    Eigen::Matrix2d::Identity(); double size = 1.0;
      //    multiFramePtr->getKeypointSize(kpi.cameraIndex, kpi.keypointIndex,
      //    size); information *= 64.0 / (size * size);

      //    std::shared_ptr<const okvis::cameras::CameraBase> baseCameraGeometry
      //    =
      //        camera_rig_.getCameraGeometry(kpi.cameraIndex);
      //    std::shared_ptr<const CAMERA_GEOMETRY_T> argCameraGeometry =
      //        std::static_pointer_cast<const
      //        CAMERA_GEOMETRY_T>(baseCameraGeometry);
      //    ::ceres::ResidualBlockId retVal =
      //        addPointFrameResidual(lmIdA, kpi.frameId, kpi.cameraIndex,
      //        measurement,
      //                              information, argCameraGeometry);
      //    CHECK_NE(reinterpret_cast<uint64_t>(retVal), 0u);
      // remember
      uint64_t retVal = 100 + rand() % 37;
      obsMapA.emplace(kpi, retVal);
    }
    //  mapPtr_->removeParameterBlock(lmIdB);
    landmarksMap_.erase(lmItB);
    // check
    lmItB = landmarksMap_.find(lmIdB);
    EXPECT_TRUE(lmItB == landmarksMap_.end());
    lmItA = landmarksMap_.find(lmIdA);
    EXPECT_TRUE(lmItA != landmarksMap_.end());
    if (lmItA != landmarksMap_.end()) {
      obsMapA = lmItA->second.observations;
      EXPECT_EQ(obsMapA.size(), 6u);
      size_t i = 0;
      for (auto obsIter = obsMapA.begin(); obsIter != obsMapA.end();
           ++obsIter, ++i) {
        EXPECT_GT(obsIter->second, 0u);
      }
    }
  }
  void SetUp() override {

  };

 private:
  okvis::PointMap landmarksMap_;

};

TEST_F(MergeTwoLandmarksTest, mergeTwoLandmarks) {
  merge(1u, 2u);
}
