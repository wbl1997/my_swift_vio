#include <gtest/gtest.h>
#include <msckf/VioTestSystemBuilder.hpp>

TEST(MSCKF, FeatureJacobian) {
  simul::VioTestSystemBuilder vioSystemBuilder;
  bool addPriorNoise = true;
  okvis::TestSetting testSetting(true, addPriorNoise, false, true, true);
  int trajectoryId = 0; // Torus
  std::string projOptModelName = "FXY_CXY";
  std::string extrinsicModelName = "P_CB";
  int cameraOrientation = 0;
  std::shared_ptr<std::ofstream> inertialStream;
  vioSystemBuilder.createVioSystem(testSetting, trajectoryId,
                                   projOptModelName, extrinsicModelName,
                                   cameraOrientation, inertialStream,
                                   "");
}
