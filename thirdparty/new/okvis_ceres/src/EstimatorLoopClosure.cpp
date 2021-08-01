/**
 * @file EstimatorLoopClosure.cpp
 * @brief implementation of estimator functions related to loop closure
 */

#include <map>
#include <memory>

#include <okvis/Estimator.hpp>
#include <loop_closure/KeyframeForLoopDetection.hpp>

namespace okvis {
// TODO(jhuai): Add heuristic rules to throttle loop query keyframes.
// 1, minimum time gap, 2, minimum distance, 3, minimum number of keypoints
// while keeping the keyframe of the previous message in the sliding window.
bool Estimator::getLoopQueryKeyframeMessage(
    okvis::MultiFramePtr multiFrame,
    std::shared_ptr<swift_vio::LoopQueryKeyframeMessage>* queryKeyframe) const {
  auto riter = statesMap_.rbegin();
  if (!riter->second.isKeyframe) {
    return false;
  }
  okvis::kinematics::Transformation T_WBr;
  get_T_WS(riter->first, T_WBr);

  uint64_t queryKeyframeId = riter->first;
  queryKeyframe->reset(new swift_vio::LoopQueryKeyframeMessage(
      queryKeyframeId, riter->second.timestamp, T_WBr, multiFrame));

  getOdometryConstraintsForKeyframe(*queryKeyframe);

  // add 3d landmarks observed in query keyframe's first frame,
  // and corresponding indices into the 2d keypoint list.
  // The local camera frame will be used as their coordinate frame.
  const std::vector<uint64_t>& landmarkIdList =
      multiFrame->getLandmarkIds(swift_vio::LoopQueryKeyframeMessage::kQueryCameraIndex);
  size_t numKeypoints = landmarkIdList.size();
  auto& keypointIndexForLandmarkList =
      (*queryKeyframe)->keypointIndexForLandmarkListMutable();
  keypointIndexForLandmarkList.reserve(numKeypoints / 4);
  auto& landmarkPositionList = (*queryKeyframe)->landmarkPositionListMutable();
  landmarkPositionList.reserve(numKeypoints / 4);
  int keypointIndex = 0;

  okvis::kinematics::Transformation T_BrW = T_WBr.inverse();
  for (const uint64_t landmarkId : landmarkIdList) {
    if (landmarkId != 0) {
      auto result = landmarksMap_.find(landmarkId);
      if (result != landmarksMap_.end() && result->second.quality > 1e-6) {
        keypointIndexForLandmarkList.push_back(keypointIndex);
        Eigen::Vector4d hp_W = result->second.pointHomog;
        Eigen::Vector4d hp_B = T_BrW * hp_W;
        landmarkPositionList.push_back(hp_B);
      }
    }
    ++keypointIndex;
  }
  return true;
}

bool Estimator::getOdometryConstraintsForKeyframe(
    std::shared_ptr<swift_vio::LoopQueryKeyframeMessage> queryKeyframe) const {
  int j = 0;
  auto& odometryConstraintList = queryKeyframe->odometryConstraintListMutable();
  odometryConstraintList.reserve(poseGraphOptions_.maxOdometryConstraintForAKeyframe);
  okvis::kinematics::Transformation T_WBr = queryKeyframe->T_WB_;
  queryKeyframe->setZeroCovariance();
  auto riter = statesMap_.rbegin();
  for (++riter;  // skip the last frame which is queryKeyframe.
       riter != statesMap_.rend() && j < poseGraphOptions_.maxOdometryConstraintForAKeyframe;
       ++riter) {
    if (riter->second.isKeyframe) {
      okvis::kinematics::Transformation T_WBn;
      get_T_WS(riter->first, T_WBn);
      okvis::kinematics::Transformation T_BnBr = T_WBn.inverse() * T_WBr;
      std::shared_ptr<swift_vio::NeighborConstraintMessage> odometryConstraint(
          new swift_vio::NeighborConstraintMessage(
              riter->first, riter->second.timestamp, T_BnBr, T_WBn));
      odometryConstraint->core_.squareRootInfo_.setIdentity();
      odometryConstraintList.emplace_back(odometryConstraint);
      ++j;
    }
  }
  return true;
}
} // namespace okvis
