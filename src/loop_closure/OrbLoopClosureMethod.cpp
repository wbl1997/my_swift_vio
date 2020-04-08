#include "loop_closure/OrbLoopClosureMethod.hpp"
namespace okvis {
OrbLoopClosureMethod::OrbLoopClosureMethod()
{

}

OrbLoopClosureMethod::OrbLoopClosureMethod(const LoopClosureParameters& parameters) {

}

OrbLoopClosureMethod::~OrbLoopClosureMethod() {

}

bool OrbLoopClosureMethod::detectLoop(
    std::shared_ptr<LoopQueryKeyframeMessage> queryKeyframe,
    std::shared_ptr<KeyframeInDatabase> queryKeyframeInDB,
    std::shared_ptr<LoopFrameAndMatches> loopFrameAndMatches) {
  return LoopClosureMethod::detectLoop(queryKeyframe, queryKeyframeInDB,
                                       loopFrameAndMatches);
}

bool OrbLoopClosureMethod::addConstraintsAndOptimize(
    std::shared_ptr<KeyframeInDatabase> queryKeyframeInDB,
    std::shared_ptr<LoopFrameAndMatches> loopFrameAndMatches) {
  return LoopClosureMethod::addConstraintsAndOptimize(queryKeyframeInDB,
                                                      loopFrameAndMatches);
}

} // namespace okvis
