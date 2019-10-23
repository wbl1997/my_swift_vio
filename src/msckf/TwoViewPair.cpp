#include "msckf/TwoViewPair.hpp"

namespace okvis {
const std::vector<std::vector<std::pair<int, int>>> TwoViewPair::pairs_lut{
    {},  // place holder
    {},  // place holder
    {{0, 1}},
    {{0, 2}},
    {{0, 2}, {1, 3}, {0, 3}},
    {{0, 2}, {1, 3}, {2, 4}, {0, 3}, {1, 4}, {0, 4}},
    {{0, 3}, {1, 4}, {2, 5}, {0, 4}, {1, 5}, {0, 5}},
    {{0, 3}, {1, 4}, {2, 5}, {3, 6}, {0, 4}, {1, 5}, {2, 6}, {0, 5}, {1, 6}},
    {{0, 4}, {1, 5}, {2, 6}, {3, 7}, {0, 5}, {1, 6}, {2, 7}, {0, 6}, {1, 7}},
    {{0, 4},
     {1, 5},
     {2, 6},
     {3, 7},
     {4, 8},
     {0, 5},
     {1, 6},
     {2, 7},
     {3, 8},
     {0, 6},
     {1, 7},
     {2, 8}},
};

std::vector<std::pair<int, int>> TwoViewPair::getFramePairs(
    int numFeatures, const TWO_VIEW_CONSTRAINT_SCHEME scheme) {
  std::vector<std::pair<int, int>> framePairs;
  framePairs.reserve(2 * numFeatures);
  int j = 0;
  int halfFeatures = numFeatures / 2;
  switch (scheme) {
    case FIXED_HEAD_RECEDING_TAIL:
      for (j = 0; j < numFeatures - 1; ++j) {
        framePairs.emplace_back(0, j + 1);
      }
      break;
    case FIXED_TAIL_RECEDING_HEAD:
      for (j = 0; j < numFeatures - 1; ++j) {
        framePairs.emplace_back(numFeatures - 1, j);
      }
      break;
    case FIXED_MIDDLE:
      for (j = 0; j < halfFeatures; ++j) {
        framePairs.emplace_back(j, halfFeatures);
      }
      for (j = halfFeatures + 1; j < numFeatures; ++j) {
        framePairs.emplace_back(j, halfFeatures);
      }
      break;
    case MAX_GAP_EVEN_CHANCE:
    default:
      if (numFeatures < static_cast<int>(pairs_lut.size())) {
        framePairs = pairs_lut[numFeatures];
      } else {
        // more constraints does not mean better results
        //        for (j = 0; j < numFeatures - halfFeatures + 1; ++j) {
        //          framePairs.emplace_back(j, halfFeatures - 1 + j);
        //        }
        for (j = 0; j < numFeatures - halfFeatures; ++j) {
          framePairs.emplace_back(j, halfFeatures + j);
        }
        for (j = 0; j < numFeatures - halfFeatures - 1; ++j) {
          framePairs.emplace_back(j, halfFeatures + 1 + j);
        }
        for (j = 0; j < numFeatures - halfFeatures - 2; ++j) {
          framePairs.emplace_back(j, halfFeatures + 2 + j);
        }
      }
      break;
  }
  return framePairs;
}
}  // namespace okvis
