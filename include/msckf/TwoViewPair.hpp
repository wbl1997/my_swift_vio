#ifndef INCLUDE_MSCKF_TWO_VIEW_PAIR_HPP_
#define INCLUDE_MSCKF_TWO_VIEW_PAIR_HPP_

#include <vector>

namespace okvis {
class TwoViewPair {
 public:
  // for views less than 8, use the look up table
  const static std::vector<std::vector<std::pair<int, int>>> pairs_lut;

  enum TWO_VIEW_CONSTRAINT_SCHEME {
    FIXED_HEAD_RECEDING_TAIL = 0,
    FIXED_TAIL_RECEDING_HEAD,
    MAX_GAP_EVEN_CHANCE,
  };

  /**
   * @brief getFramePairs generate pairs of frames to form two-view constraints
   *     Available schemes,
   *  FIXED_HEAD_RECEDING_TAIL: [0, 1], [0, 2], ..., [0, n-1]
   *  FIXED_TAIL_RECEDING_HEAD: [n-1, n-2], [n-1, n-3], ..., [n-1, 0]
   *  MAX_GAP_EVEN_CHANCE: predefined LUT for less than 8,
   *  for n>=8, {(i, i + n/2 - 1)} + {(i, i + n/2)} +
   *  {(i, i + n/2 + 1)} + {(i, i + n/2 + 2)}
   * @param numFeatures number of observations
   * @return pairs of 0-based frame indices
   */
  static std::vector<std::pair<int, int>> getFramePairs(
      int numFeatures,
      const TWO_VIEW_CONSTRAINT_SCHEME scheme = MAX_GAP_EVEN_CHANCE);
};
}  // namespace okvis
#endif  // INCLUDE_MSCKF_TWO_VIEW_PAIR_HPP_
