
#include <gtest/gtest.h>
#include "swift_vio/TwoViewPair.hpp"

TEST(TwoViewPair, MixedPairs) {
  for (int j = 8; j < 10; ++j) {
    std::vector<std::pair<int, int>> frame_pairs =
        swift_vio::TwoViewPair::getFramePairs(j);
    std::vector<std::pair<int, int>> frame_pairs_expected =
        swift_vio::TwoViewPair::pairs_lut[j];
    for (int k = 0; k < (int)frame_pairs_expected.size(); ++k) {
      EXPECT_EQ(frame_pairs[k].first, frame_pairs_expected[k].first);
      EXPECT_EQ(frame_pairs[k].second, frame_pairs_expected[k].second);
    }
  }
}
