#include <msckf/MotionAndStructureStats.h>

namespace msckf {
MotionAndStructureStats::MotionAndStructureStats() : medianSceneDepth_(1e2)
{

}

void MotionAndStructureStats::startUpdatingSceneDepth() {
    acc_ = boost::accumulators::accumulator_set<
        double, boost::accumulators::stats<boost::accumulators::tag::median(
                    boost::accumulators::with_p_square_quantile)> >();
}

void MotionAndStructureStats::addLandmarkDepth(double zA) {
    acc_(1.0/zA);
}

void MotionAndStructureStats::finishUpdatingSceneDepth() {
    double invDepth = boost::accumulators::median(acc_);
    double filteredInvDepth = 0.5 / medianSceneDepth_ + 0.5 * invDepth;
    medianSceneDepth_ = 1.0 / filteredInvDepth;
}

} // namespace msckf
