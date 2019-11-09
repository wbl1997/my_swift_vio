#include "gtest/gtest.h"
#include <msckf/EpipolarFactor.hpp>
#include <msckf/ExtrinsicModels.hpp>

#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/PinholeCamera.hpp>

TEST(CeresErrorTerms, EpipolarFactor) {
  typedef okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>
      DistortedPinholeCameraGeometry;
  const int distortionDim = DistortedPinholeCameraGeometry::distortion_t::NumDistortionIntrinsics;
  const int projIntrinsicDim = okvis::ProjectionOptFXY_CXY::kNumParams;
  std::shared_ptr<DistortedPinholeCameraGeometry> cameraGeometry =
      std::static_pointer_cast<DistortedPinholeCameraGeometry>(
          DistortedPinholeCameraGeometry::createTestObject());

  okvis::ceres::EpipolarFactor<DistortedPinholeCameraGeometry,
      okvis::ProjectionOptFXY_CXY, okvis::Extrinsic_p_SC_q_SC> epiFactor;
  EXPECT_EQ(5, 5);
}
