#include <msckf/PointLandmark.hpp>

#include <msckf/FeatureTriangulation.hpp>
#include <msckf/ParallaxAnglePoint.hpp>
#include <msckf/PointLandmarkModels.hpp>
#include <okvis/FrameTypedefs.hpp>

namespace msckf {

/**
 * @brief decideAnchors
 * @param mp
 * @param involvedFrameIds
 * @param landmarkModelId
 * @param anchorIds
 * @param anchorSeqIds id of anchors relative to mappoint observations
 */
void decideAnchors(const okvis::MapPoint& mp,
                   const std::vector<uint64_t>* involvedFrameIds,
                   int landmarkModelId, std::vector<uint64_t>* anchorIds,
                   std::vector<int>* anchorSeqIds) {
  uint64_t anchorId;
  if (involvedFrameIds != nullptr) {
    switch (landmarkModelId) {
      case msckf::ParallaxAngleParameterization::kModelId:
        // greedily choose the head and tail observation
        anchorId = involvedFrameIds->front();
        anchorIds->push_back(anchorId);
        [[fallthrough]];
      case msckf::InverseDepthParameterization::kModelId:
        anchorId = involvedFrameIds->back();
        anchorIds->push_back(anchorId);
        break;
      case msckf::HomogeneousPointParameterization::kModelId:
      default:
        break;
    }
    const std::map<okvis::KeypointIdentifier, uint64_t>& obsMap =
        mp.observations;
    for (auto aid : *anchorIds) {
      std::map<okvis::KeypointIdentifier, uint64_t>::const_iterator anchorIter =
          std::find_if(obsMap.begin(), obsMap.end(), IsObservedInFrame(aid));
      int anchorSeqId = std::distance(obsMap.begin(), anchorIter);
      anchorSeqIds->push_back(anchorSeqId);
    }
  } else {
    switch (landmarkModelId) {
      case msckf::ParallaxAngleParameterization::kModelId:
        // TODO(jhuai): is there an efficient way to find the ray pair of max angle?
        // greedily choose the head and tail observation
        anchorId = mp.observations.begin()->first.frameId;
        anchorIds->push_back(anchorId);
        anchorSeqIds->push_back(0);
        [[fallthrough]];
      case msckf::InverseDepthParameterization::kModelId:
        anchorId = mp.observations.rbegin()->first.frameId;
        anchorIds->push_back(anchorId);
        anchorSeqIds->push_back(mp.observations.size() - 1);
        break;
      case msckf::HomogeneousPointParameterization::kModelId:
      default:
        break;
    }
  }
}

PointLandmark::PointLandmark(int modelId) : modelId_(modelId) {
}

TriangulationStatus PointLandmark::initialize(
    const std::vector<
        okvis::kinematics::Transformation,
        Eigen::aligned_allocator<okvis::kinematics::Transformation>>& T_WSs,
    const std::vector<Eigen::Vector3d,
                      Eigen::aligned_allocator<Eigen::Vector3d>>& obsDirections,
    const okvis::kinematics::Transformation& T_BC0,
    const std::vector<int>& anchorSeqIds) {
  std::vector<okvis::kinematics::Transformation,
              Eigen::aligned_allocator<okvis::kinematics::Transformation>>
      cam_states(obsDirections.size());
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      measurements(obsDirections.size());
  int jack = 0;
  for (auto obs3 : obsDirections) {
    measurements[jack] = obs3.head<2>();
    ++jack;
  }
  int joel = 0;
  switch (anchorSeqIds.size()) {
    case 1: { // AIDP
      // Ca will play the role of W
      okvis::kinematics::Transformation T_CaW =
          (T_WSs.at(anchorSeqIds[0]) * T_BC0).inverse();
      for (auto T_WS : T_WSs) {
        okvis::kinematics::Transformation T_WCi = T_WS * T_BC0;
        cam_states[joel] = T_CaW * T_WCi;
        ++joel;
      }
      break;
    }
    case 2: {
      Eigen::Vector3d d_m = obsDirections[anchorSeqIds[0]].normalized();
      Eigen::Vector3d d_a = obsDirections[anchorSeqIds[1]].normalized();
      Eigen::Vector3d W_d_m = T_WSs[anchorSeqIds[0]].C() * (T_BC0.C() * d_m);
      Eigen::Vector3d W_d_a = T_WSs[anchorSeqIds[1]].C() * (T_BC0.C() * d_a);
      double cos_theta = W_d_m.dot(W_d_a);
      LWF::ParallaxAnglePoint pap(d_m, cos_theta);
      pap.copy(&parameters_);
      TriangulationStatus status;
      status.triangulationOk = true;
      status.chi2Small = true;
      status.flipped = false;
      status.raysParallel = false;
      return status;
    }
    case 0:
    default:
      for (auto iter = T_WSs.begin(); iter != T_WSs.end(); ++iter, ++joel) {
        cam_states[joel] = *iter * T_BC0;
      }
      break;
  }
  msckf_vio::Feature feature(measurements, cam_states);
  feature.initializePosition();
  parameters_.reserve(4);
  parameters_.insert(parameters_.end(), feature.position.data(),
                     feature.position.data() + 3);
  parameters_.push_back(1.0);
  TriangulationStatus status;
  status.triangulationOk = feature.is_initialized;
  status.chi2Small = feature.is_chi2_small;
  status.flipped = feature.is_flipped;
  status.raysParallel = feature.is_parallel;
  return status;
}


} // namespace msckf
