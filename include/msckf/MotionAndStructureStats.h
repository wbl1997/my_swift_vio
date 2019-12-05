#ifndef INCLUDE_MSCKF_MOTION_AND_STRUCTURE_STATS_H_
#define INCLUDE_MSCKF_MOTION_AND_STRUCTURE_STATS_H_

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

namespace msckf {
class MotionAndStructureStats {
 public:
  MotionAndStructureStats();
  void startUpdatingSceneDepth();
  /**
   * @brief addLandmarkDepth
   * @param zA depth in anchor frame A
   */
  void addLandmarkDepth(double zA);
  void finishUpdatingSceneDepth();
  double medianSceneDepth() const { return medianSceneDepth_; }

 private:
  double medianSceneDepth_;
  boost::accumulators::accumulator_set<
      double, boost::accumulators::stats<boost::accumulators::tag::median(
                  boost::accumulators::with_p_square_quantile)> >
      acc_;
};
} // namespace msckf
#endif // INCLUDE_MSCKF_MOTION_AND_STRUCTURE_STATS_H_
