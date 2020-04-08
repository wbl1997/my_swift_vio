/**
 * @file
 * @brief
 */

#ifndef INCLUDE_LOOP_CLOSURE_ORB_LOOP_CLOSURE_METHOD_HPP_
#define INCLUDE_LOOP_CLOSURE_ORB_LOOP_CLOSURE_METHOD_HPP_

#include <DBoW2/DBoW2.h>

#include <okvis/KeyframeForLoopDetection.hpp>
#include <okvis/LoopFrameAndMatches.hpp>
#include <okvis/LoopClosureMethod.hpp>
#include <okvis/LoopClosureParameters.hpp>

namespace okvis {
class OrbLoopClosureMethod : public okvis::LoopClosureMethod {
 public:
  OrbLoopClosureMethod();
  explicit OrbLoopClosureMethod(const LoopClosureParameters& parameters);
  virtual ~OrbLoopClosureMethod();

  virtual bool detectLoop(
      std::shared_ptr<LoopQueryKeyframeMessage> queryKeyframe,
      std::shared_ptr<KeyframeInDatabase> queryKeyframeInDB,
      std::shared_ptr<LoopFrameAndMatches> loopFrameAndMatches) final;

  virtual bool addConstraintsAndOptimize(
      std::shared_ptr<KeyframeInDatabase> queryKeyframeInDB,
      std::shared_ptr<LoopFrameAndMatches> loopFrameAndMatches) final;

 private:
  // ORB extraction and matching members
  cv::Ptr<cv::ORB> orb_feature_detector_;
  cv::Ptr<cv::DescriptorMatcher> orb_feature_matcher_;

  // BoW and Loop Detection database and members
  std::unique_ptr<OrbDatabase> db_BoW_;

};
}  // namespace okvis
#endif  // INCLUDE_LOOP_CLOSURE_ORB_LOOP_CLOSURE_METHOD_HPP_
