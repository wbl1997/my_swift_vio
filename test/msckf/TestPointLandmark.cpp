#include <gtest/gtest.h>
#include <msckf/PointLandmark.hpp>

TEST(PointObservations, EraseBadObservations) {
  std::vector<std::pair<uint64_t, int>> dudIds{
      {100, 0}, {200, 1}, {300, 1}, {400, 0}};
  std::vector<uint64_t> candidateFrameIds{150, 200, 250, 350, 400, 450};
  msckf::eraseBadObservations(dudIds, &candidateFrameIds);
  std::vector<uint64_t> expectedFrameIds{150, 250, 350, 450};
  int count = 0;
  for (auto element : candidateFrameIds) {
    EXPECT_EQ(element, expectedFrameIds[count]);
    ++count;
  }
}
