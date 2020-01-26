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
        chi2Small(false),
        raysParallel(false),
        flipped(false) {}
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

void decideAnchors(const okvis::MapPoint& mp,
                   const std::vector<uint64_t>* involvedFrameIds,
                   int landmarkModelId, std::vector<uint64_t>* anchorIds,
                   std::vector<int>* anchorSeqIds);

class PointLandmark {
public:
 PointLandmark(int modelId=0);

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

private:
  int modelId_;
  std::vector<double> parameters_;
};
}
#endif // INCLUDE_MSCKF_POINT_LANDMARK_HPP_
