/**
 * @file HybridFrameRotationOnlySacProblem.hpp
 * @brief Header file for the HybridFrameRotationOnlySacProblem class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_OPENGV_HYBRIDFRAMEROTATIONONLYSACPROBLEM_HPP_
#define INCLUDE_OKVIS_OPENGV_HYBRIDFRAMEROTATIONONLYSACPROBLEM_HPP_

#include <msckf/HybridFrameRelativeAdapter.hpp>
#include <okvis/assert_macros.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac_problems/relative_pose/RotationOnlySacProblem.hpp>
#include <opengv/types.hpp>

/**
 * \brief Namespace for classes extending the OpenGV library.
 */
namespace opengv {
/**
 * \brief The namespace for the sample consensus problems.
 */
namespace sac_problems {
/**
 * \brief The namespace for the relative pose methods.
 */
namespace relative_pose {

/**
 * \brief Functions for fitting a rotation-only model to a set of bearing-vector
 *        correspondences (using twopt_rotationOnly). The viewpoints are assumed
 * to be separated by rotation only. Used within a random-sample paradigm for
 *        rejecting outlier correspondences.
 */
class HybridFrameRotationOnlySacProblem : public RotationOnlySacProblem {
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  typedef RotationOnlySacProblem base_t;

  /** The type of adapter that is expected by the methods */
  using base_t::adapter_t;
  /** The model we are trying to fit (rotation) */
  using base_t::model_t;

  /**
   * \brief Constructor.
   * \param[in] adapter Visitor holding bearing vector correspondences etc.
   * @warning Only okvis::relative_pose::HybridFrameRelativeAdapter supported.
   */
  HybridFrameRotationOnlySacProblem(adapter_t& adapter)
      : base_t(adapter),
        adapterDerived_(
            *static_cast<opengv::relative_pose::HybridFrameRelativeAdapter*>(
                &_adapter)) {
    OKVIS_ASSERT_TRUE(
        Exception,
        dynamic_cast<opengv::relative_pose::HybridFrameRelativeAdapter*>(
            &_adapter),
        "only opengv::absolute_pose::HybridFrameRelativeAdapter supported");
  }

  /**
   * \brief Constructor.
   * \param[in] adapter Visitor holding bearing vector correspondences etc.
   * \param[in] indices A vector of indices to be used from all available
   *                    correspondences.
   * @warning Only okvis::relative_pose::HybridFrameRelativeAdapter supported.
   */
  HybridFrameRotationOnlySacProblem(adapter_t& adapter,
                                    const std::vector<int>& indices)
      : base_t(adapter, indices),
        adapterDerived_(
            *static_cast<opengv::relative_pose::HybridFrameRelativeAdapter*>(
                &_adapter)) {
    OKVIS_ASSERT_TRUE(
        Exception,
        dynamic_cast<opengv::relative_pose::HybridFrameRelativeAdapter*>(
            &_adapter),
        "only opengv::absolute_pose::HybridFrameRelativeAdapter supported");
  }

  virtual ~HybridFrameRotationOnlySacProblem() {}

  /**
   * \brief Compute the distances of all samples whith respect to given model
   *        coefficients.
   * \param[in] model The coefficients of the model hypothesis.
   * \param[in] indices The indices of the samples of which we compute
   * distances. \param[out] scores The resulting distances of the selected
   * samples. Low distances mean a good fit.
   */
  virtual void getSelectedDistancesToModel(const model_t& model,
                                           const std::vector<int>& indices,
                                           std::vector<double>& scores) const {
    for (size_t i = 0; i < indices.size(); i++) {
      bearingVector_t f1 = adapterDerived_.getBearingVector1(indices[i]);
      bearingVector_t f2 = adapterDerived_.getBearingVector2(indices[i]);

      // unrotate bearing-vector f2
      bearingVector_t f2_unrotated = model * f2;

      // unrotate bearing-vector f1
      bearingVector_t f1_unrotated = model.transpose() * f1;

      point_t error1 = (f2_unrotated - f1);
      point_t error2 = (f1_unrotated - f2);
      double error_squared1 = error1.transpose() * error1;
      double error_squared2 = error2.transpose() * error2;
      scores.push_back(
          error_squared1 * 0.5 / adapterDerived_.getSigmaAngle1(indices[i]) +
          error_squared2 * 0.5 / adapterDerived_.getSigmaAngle2(indices[i]));
    }
  }

 protected:
  /// The adapter holding the bearing, correspondences etc.
  opengv::relative_pose::HybridFrameRelativeAdapter& adapterDerived_;
};

}  // namespace relative_pose
}  // namespace sac_problems
}  // namespace opengv

#endif /* INCLUDE_OKVIS_OPENGV_FRAMEROTATIONONLYPOSESACPROBLEM_HPP_ */
