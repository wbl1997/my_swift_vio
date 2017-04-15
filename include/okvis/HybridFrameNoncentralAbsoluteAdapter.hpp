
#ifndef INCLUDE_OKVIS_OPENGV_HYBRIDFRAMENONCENTRALABSOLUTEADAPTER_HPP_
#define INCLUDE_OKVIS_OPENGV_HYBRIDFRAMENONCENTRALABSOLUTEADAPTER_HPP_

#include <stdlib.h>
#include <vector>
#include <memory>
#include <opengv/types.hpp>
#include <opengv/absolute_pose/NoncentralAbsoluteAdapter.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/cameras/NCameraSystem.hpp>

#ifdef USE_MSCKF2
#include <okvis/msckf2.hpp>
#else
#include <okvis/HybridFilter.hpp>
#endif

/**
 * \brief Namespace for classes extending the OpenGV library.
 */
namespace opengv {
/**
 * \brief The namespace for the absolute pose methods.
 */
namespace absolute_pose {

/// \brief Adapter for absolute pose RANSAC (3D2D) with non-central cameras,
///        i.e. could be a multi-camera-setup.
class HybridFrameNoncentralAbsoluteAdapter : public AbsoluteAdapterBase {
 private:
  using AbsoluteAdapterBase::_t;
  using AbsoluteAdapterBase::_R;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief type for describing matches.
  typedef std::vector<int> matches_t;

  /**
   * @brief Constructor.
   * @param estimator     Estimator.
   * @param nCameraSystem Camera configuration and parameters.
   * @param frame         The multiframe.
   */
  HybridFrameNoncentralAbsoluteAdapter(
#ifdef USE_MSCKF2
      const okvis::MSCKF2 & estimator,
#else
      const okvis::HybridFilter & estimator,
#endif
      const okvis::cameras::NCameraSystem & nCameraSystem,
      std::shared_ptr<okvis::MultiFrame> frame);

  virtual ~HybridFrameNoncentralAbsoluteAdapter() {
  }

  /// @name Algorithm input
  /// @{

  /**
   * \brief Retrieve the bearing vector of a correspondence.
   * \param[in] index The serialized index of the correspondence.
   * \return The corresponding bearing vector.
   */
  virtual opengv::bearingVector_t getBearingVector(size_t index) const;

  /**
   * \brief Retrieve the position of a camera of a correspondence
   *        seen from the viewpoint origin.
   * \param[in] index The serialized index of the correspondence.
   * \return The position of the corresponding camera seen from the viewpoint
   *         origin.
   */
  virtual opengv::translation_t getCamOffset(size_t index) const;

  /**
   * \brief Retrieve the rotation from a camera of a correspondence to the
   *        viewpoint origin.
   * \param[in] index The serialized index of the correspondence.
   * \return The rotation from the corresponding camera back to the viewpoint
   *         origin.
   */
  virtual opengv::rotation_t getCamRotation(size_t index) const;

  /**
   * \brief Retrieve the world point of a correspondence.
   * \param[in] index The serialized index of the correspondence.
   * \return The corresponding world point.
   */
  virtual opengv::point_t getPoint(size_t index) const;

  /**
   * @brief Get the number of correspondences. These are keypoints that have a
   *        corresponding landmark which is added to the estimator,
   *        has more than one observation and not at infinity.
   * @return Number of correspondences.
   */
  virtual size_t getNumberCorrespondences() const;

  /// @}

  // lestefan: some additional accessors
  /**
   * @brief Get the camera index for a specific correspondence.
   * @param index The serialized index of the correspondence.
   * @return Camera index of the correspondence.
   */
  int camIndex(size_t index) const {return camIndices_.at(index);}

  /**
   * @brief Get the keypoint index for a specific correspondence
   * @param index The serialized index of the correspondence.
   * @return Keypoint index belonging to the correspondence.
   */
  int keypointIndex(size_t index) const {return keypointIndices_.at(index);}

  /**
   * \brief Retrieve the weight of a correspondence. The weight is supposed to
   *        reflect the quality of a correspondence, and typically is between
   *        0 and 1.
   * \warning This is not implemented and always returns 1.0.
   */
  virtual double getWeight(size_t) const {
    return 1.0;
  }  // TODO : figure out if needed...

  // custom:
  /**
   * @brief Obtain the angular standard deviation in [rad].
   * @param index The index of the correspondence.
   * @return The standard deviation in [rad].
   */
  double getSigmaAngle(size_t index);

 private:
  /// The bearing vectors of the correspondences.
  opengv::bearingVectors_t bearingVectors_;

  /// The world coordinates of the correspondences.
  opengv::points_t points_;

  /// The camera indices of the correspondences.
  std::vector<size_t> camIndices_;

  /// The keypoint indices of the correspondences.
  std::vector<size_t> keypointIndices_;

  /// The position of the cameras seen from the viewpoint origin
  opengv::translations_t camOffsets_;

  /// The rotation of the cameras to the viewpoint origin.
  opengv::rotations_t camRotations_;

  /// The standard deviations of the bearing vectors in [rad].
  std::vector<double> sigmaAngles_;

};

}
}

#endif /* INCLUDE_OKVIS_OPENGV_FRAMENONCENTRALABSOLUTEADAPTER_HPP_ */
