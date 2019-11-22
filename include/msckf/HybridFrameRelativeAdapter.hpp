
#ifndef INCLUDE_OKVIS_OPENGV_HYBRIDHybridFrameRelativeAdapter_HPP_
#define INCLUDE_OKVIS_OPENGV_HYBRIDHybridFrameRelativeAdapter_HPP_

#include <stdlib.h>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/types.hpp>
#include <vector>

#include <msckf/MSCKF2.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/cameras/NCameraSystem.hpp>
/**
 * \brief Namespace for classes extending the OpenGV library.
 */
namespace opengv {
/**
 * \brief The namespace for the relative pose methods.
 */
namespace relative_pose {

/// \brief Adapter for relative pose RANSAC (2D2D)
class HybridFrameRelativeAdapter : public RelativeAdapterBase {
 private:
  using RelativeAdapterBase::_R12;
  using RelativeAdapterBase::_t12;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  /**
   * @brief Constructor
   * @param estimator     The estimator.
   * @param nCameraSystem Camera configuration and parameters.
   * @param multiFrameIdA The first multiframe.
   * @param camIdA        The camera index for the first multiframe
   *                      in order to access the relevant frame.
   * @param multiFrameIdB The second multiframe.
   * @param camIdB        The camera index for the second multiframe.
   *                      in order to access the relevant frame.
   */
  HybridFrameRelativeAdapter(
      const okvis::HybridFilter& estimator,
      const okvis::cameras::NCameraSystem& nCameraSystem,
      uint64_t multiFrameIdA, size_t camIdA, uint64_t multiFrameIdB,
      size_t camIdB);

  virtual ~HybridFrameRelativeAdapter() {}

  /// @name Algorithm input
  /// @{

  /**
   * \brief Retrieve the bearing vector of a correspondence in viewpoint 1.
   * \param[in] index The serialized index of the correspondence.
   * \return The corresponding bearing vector.
   */
  virtual opengv::bearingVector_t getBearingVector1(size_t index) const;
  /**
   * \brief Retrieve the bearing vector of a correspondence in viewpoint 2.
   * \param[in] index The serialized index of the correspondence.
   * \return The corresponding bearing vector.
   */
  virtual opengv::bearingVector_t getBearingVector2(size_t index) const;
  /**
   * \brief Retrieve the position of a camera of a correspondence in viewpoint
   *        1 seen from the origin of the viewpoint.
   * \param[in] index The serialized index of the correspondence.
   * \return The position of the corresponding camera seen from the viewpoint
   *         origin.
   */
  virtual opengv::translation_t getCamOffset1(size_t index) const;
  /**
   * \brief Retrieve the rotation from a camera of a correspondence in
   *        viewpoint 1 to the viewpoint origin.
   * \param[in] index The serialized index of the correspondence.
   * \return The rotation from the corresponding camera back to the viewpoint
   *         origin.
   */
  virtual opengv::rotation_t getCamRotation1(size_t index) const;
  /**
   * \brief Retrieve the position of a camera of a correspondence in viewpoint
   *        2 seen from the origin of the viewpoint.
   * \param[in] index The serialized index of the correspondence.
   * \return The position of the corresponding camera seen from the viewpoint
   *         origin.
   */
  virtual opengv::translation_t getCamOffset2(size_t index) const;
  /**
   * \brief Retrieve the rotation from a camera of a correspondence in
   *        viewpoint 2 to the viewpoint origin.
   * \param[in] index The serialized index of the correspondence.
   * \return The rotation from the corresponding camera back to the viewpoint
   *         origin.
   */
  virtual opengv::rotation_t getCamRotation2(size_t index) const;
  /**
   * \brief Retrieve the number of correspondences.
   * \return The number of correspondences.
   */
  virtual size_t getNumberCorrespondences() const;

  /// @}

  // custom:
  /**
   * @brief Obtain the angular standard deviation of the correspondence in frame
   * 1 in [rad].
   * @param index The index of the correspondence.
   * @return The standard deviation in [rad].
   */
  double getSigmaAngle1(size_t index);
  /**
   * @brief Obtain the angular standard deviation of the correspondence in frame
   * 2 in [rad].
   * @param index The index of the correspondence.
   * @return The standard deviation in [rad].
   */
  double getSigmaAngle2(size_t index);
  /**
   * @brief Get the keypoint index in frame 1 of a correspondence.
   * @param index The serialized index of the correspondence.
   * @return The keypoint index of the correspondence in frame 1.
   */
  size_t getMatchKeypointIdxA(size_t index) { return matches_.at(index).idxA; }
  /**
   * @brief Get the keypoint index in frame 2 of a correspondence.
   * @param index The serialized index of the correspondence.
   * @return The keypoint index of the correspondence in frame 2.
   */
  size_t getMatchKeypointIdxB(size_t index) { return matches_.at(index).idxB; }
  /**
   * \brief Retrieve the weight of a correspondence. The weight is supposed to
   *        reflect the quality of a correspondence, and typically is between
   *        0 and 1.
   * \warning This is not implemented and always returns 1.0.
   */
  virtual double getWeight(size_t) const {
    return 1.0;
  }  // TODO : figure out, if this is needed

  void computeMatchStats(std::shared_ptr<okvis::MultiFrame> frameBPtr,
                         size_t camIdx, double* overlap,
                         double* matchRatio) const;

 private:
  /// The bearing vectors of the correspondences in frame 1.
  opengv::bearingVectors_t bearingVectors1_;
  /// The bearing vectors of the correspondences in frame 2.
  opengv::bearingVectors_t bearingVectors2_;
  /// The matching keypoints of both frames.
  okvis::Matches matches_;

  // also store individual uncertainties
  /// The standard deviations of the bearing vectors of frame 1 in [rad].
  std::vector<double> sigmaAngles1_;
  /// The standard deviations of the bearing vectors of frame 2' in [rad].
  std::vector<double> sigmaAngles2_;
};

}  // namespace relative_pose
}  // namespace opengv

#endif /* INCLUDE_OKVIS_OPENGV_HybridFrameRelativeAdapter_HPP_ */
