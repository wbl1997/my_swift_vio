#ifndef INCLUDE_MSCKF_POINT_LANDMARK_HPP_
#define INCLUDE_MSCKF_POINT_LANDMARK_HPP_

#include <vector>
#include <okvis/FrameTypedefs.hpp>

namespace msckf {

struct TriangulationStatus {
  bool triangulationOk; // True if the landmark is in front of every camera.
  bool chi2Small;
  bool raysParallel; // True if rotation compensated observation directions are parallel.
  bool flipped; // True if the landmark is flipped to be in front of every camera.
  bool lackObservations; // True if #obs is less than minTrackLength.
  TriangulationStatus()
      : triangulationOk(false),
        chi2Small(true),
        raysParallel(false),
        flipped(false),
        lackObservations(false)
  {}
};

struct IsObservedInFrame {
  IsObservedInFrame(uint64_t x) : frameId(x) {}
  bool operator()(
      const std::pair<okvis::KeypointIdentifier, uint64_t> &v) const {
    return v.first.frameId == frameId;
  }

 private:
  uint64_t frameId;  ///< Multiframe ID.
};

void decideAnchors(const std::vector<std::pair<uint64_t, int>>& frameIdentifiers,
                   const std::vector<uint64_t>& orderedCulledFrameIds,
                   int landmarkModelId, std::vector<uint64_t>* anchorIds,
                   std::vector<int>* anchorSeqIds);

void decideAnchors(const std::vector<std::pair<uint64_t, int>>& frameIdentifiers,
                   int landmarkModelId, std::vector<uint64_t>* anchorIds,
                   std::vector<int>* anchorSeqIds);

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
 explicit PointLandmark(int modelId);

 /**
  * @brief initialize
  * @param T_WSs
  * @param obsDirections a list of [x, y, 1]
  * @param T_BC0
  * @param anchorSeqId main anchor, associate anchor id
  * @return
  */
 TriangulationStatus initialize(
     const std::vector<
         okvis::kinematics::Transformation,
         Eigen::aligned_allocator<okvis::kinematics::Transformation>>& T_WSs,
     const std::vector<Eigen::Vector3d,
                       Eigen::aligned_allocator<Eigen::Vector3d>>& obsDirections,
     const okvis::kinematics::Transformation& T_BC0,
     const std::vector<int>& anchorSeqId);

 double* data() {
   return parameters_.data();
 }

 const double* data() const {
   return parameters_.data();
 }

 int modelId() const {
   return modelId_;
 }

private:
  int modelId_;
  std::vector<double> parameters_;
};
}
#endif // INCLUDE_MSCKF_POINT_LANDMARK_HPP_
