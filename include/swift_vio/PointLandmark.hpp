#ifndef INCLUDE_SWIFT_VIO_POINT_LANDMARK_HPP_
#define INCLUDE_SWIFT_VIO_POINT_LANDMARK_HPP_

#include <vector>
#include <okvis/FrameTypedefs.hpp>

namespace swift_vio {

struct TriangulationStatus {
  bool triangulationOk; // True if the landmark is in front of every camera.
  bool chi2Small;
  bool raysParallel; // True if rotation compensated observation directions are parallel.
  bool flipped; // True if the landmark is flipped to be in front of every camera.
  bool lackObservations; // True if #obs is too few.
  TriangulationStatus()
      : triangulationOk(false),
        chi2Small(true),
        raysParallel(false),
        flipped(false),
        lackObservations(false)
  {}
};

enum class MeasurementJacobianStatus {
  Successful = 0,
  GeneralProjectionFailed = 1,
  MainAnchorProjectionFailed = 2,
  AssociateAnchorProjectionFailed = 3,
};

void decideAnchors(const std::vector<std::pair<uint64_t, size_t>>& frameIdentifiers,
                   const std::vector<uint64_t>& orderedCulledFrameIds,
                   int landmarkModelId, std::vector<okvis::AnchorFrameIdentifier>* anchorIds);

void decideAnchors(const std::vector<std::pair<uint64_t, size_t>>& frameIdentifiers,
                   int landmarkModelId, std::vector<okvis::AnchorFrameIdentifier>* anchorIds);

inline int eraseBadObservations(const std::vector<std::pair<uint64_t, int>>& dudIds,
                                std::vector<uint64_t>* candidateFrameIds) {
  int numErased = 0;
  for (auto dud : dudIds) {
    uint64_t frameId = dud.first;
    auto iter =
        std::find_if(candidateFrameIds->begin(), candidateFrameIds->end(),
                     [frameId](const uint64_t& s) { return s == frameId; });
    if (iter != candidateFrameIds->end()) {
      candidateFrameIds->erase(iter);
      ++numErased;
    }
  }
  return numErased;
}

class PointLandmark {
public:
  PointLandmark() {}

  explicit PointLandmark(int modelId);

 /**
  * @brief initialize
  * @param T_WSs
  * @param obsDirections a list of image coordinates at z=1, [x, y, 1]
  * @param T_BC0
  * @param anchorSeqId main anchor, associate anchor id
  * @return
  */
 TriangulationStatus initialize(
     const std::vector<
         okvis::kinematics::Transformation,
         Eigen::aligned_allocator<okvis::kinematics::Transformation>>& T_WSs,
     const std::vector<Eigen::Vector3d,
                       Eigen::aligned_allocator<Eigen::Vector3d>>&
         obsDirections,
     const std::vector<
         okvis::kinematics::Transformation,
         Eigen::aligned_allocator<okvis::kinematics::Transformation>>& T_BCs,
     const std::vector<
         okvis::kinematics::Transformation,
         Eigen::aligned_allocator<okvis::kinematics::Transformation>>& T_WCa_list,
     const std::vector<size_t>& cameraIndices,
     const std::vector<size_t>& anchorSeqIds);

 double* data() {
   return parameters_.data();
 }

 const double* data() const {
   return parameters_.data();
 }

 int modelId() const {
   return modelId_;
 }

 void setModelId(int modelId) {
   modelId_ = modelId;
 }

private:
  int modelId_;
  std::vector<double> parameters_;
};
} // namespace swift_vio
#endif // INCLUDE_SWIFT_VIO_POINT_LANDMARK_HPP_
