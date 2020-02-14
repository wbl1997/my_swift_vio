#include <msckf/PointLandmark.hpp>

#include <msckf/FeatureTriangulation.hpp>
#include <msckf/ParallaxAnglePoint.hpp>
#include <msckf/PointLandmarkModels.hpp>
#include <okvis/FrameTypedefs.hpp>

namespace msckf {

/**
 * @brief decideAnchors
 * @param mp
 * @param orderedCulledFrameIds
 * @param landmarkModelId
 * @param anchorIds
 * @param anchorSeqIds id of anchors relative to mappoint observations
 */
void decideAnchors(const std::vector<std::pair<uint64_t, int>>& frameIds,
                   const std::vector<uint64_t>& orderedCulledFrameIds,
                   int landmarkModelId, std::vector<uint64_t>* anchorIds,
                   std::vector<int>* anchorSeqIds) {
  uint64_t anchorId;
  switch (landmarkModelId) {
    case msckf::ParallaxAngleParameterization::kModelId:
      // greedily choose the head and tail observation
      anchorId = orderedCulledFrameIds.front();
      anchorIds->push_back(anchorId);
      [[fallthrough]];
    case msckf::InverseDepthParameterization::kModelId:
      anchorId = orderedCulledFrameIds.back();
      anchorIds->push_back(anchorId);
      break;
    case msckf::HomogeneousPointParameterization::kModelId:
    default:
      break;
  }
  for (auto aid : *anchorIds) {
    std::vector<std::pair<uint64_t, int>>::const_iterator anchorIter =
        std::find_if(frameIds.begin(), frameIds.end(),
                     [aid](const std::pair<uint64_t, int>& s) {
                       return s.first == aid;
                     });
    int anchorSeqId = std::distance(frameIds.begin(), anchorIter);
    anchorSeqIds->push_back(anchorSeqId);
  }
}

void decideAnchors(const std::vector<std::pair<uint64_t, int>>& frameIds,
                   int landmarkModelId, std::vector<uint64_t>* anchorIds,
                   std::vector<int>* anchorSeqIds) {
  uint64_t anchorId;
  switch (landmarkModelId) {
    case msckf::ParallaxAngleParameterization::kModelId:
      // TODO(jhuai): is there an efficient way to find the ray pair of max
      // angle? For now, we greedily choose the head and tail observation.
      anchorId = frameIds.front().first;
      anchorIds->push_back(anchorId);
      anchorSeqIds->push_back(0);
      [[fallthrough]];
    case msckf::InverseDepthParameterization::kModelId:
      anchorId = frameIds.back().first;
      anchorIds->push_back(anchorId);
      anchorSeqIds->push_back(frameIds.size() - 1);
      break;
    case msckf::HomogeneousPointParameterization::kModelId:
    default:
      break;
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
      for (auto iter = T_WSs.begin(); iter != T_WSs.end(); ++iter, ++joel) {
        cam_states[joel] = *iter * T_BC0;
      }
      LWF::ParallaxAnglePoint pap;
      TriangulationStatus status;
      status.triangulationOk =
          pap.initializePosition(obsDirections, cam_states, anchorSeqIds);
      pap.copy(&parameters_);
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
