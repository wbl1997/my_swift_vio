/**
 * @file   RiSlidingWindowSmoother.hpp
 * @brief  Declaration of RiSlidingWindowSmoother which wraps
 * gtsam::FixedLagSmoother for VIO with right invariant error formulation.
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_GTSAM_RI_SLIDING_WINDOW_SMOOTHER_HPP_
#define INCLUDE_GTSAM_RI_SLIDING_WINDOW_SMOOTHER_HPP_

#include <gtsam/SlidingWindowSmoother.hpp>

#ifdef HAVE_GTSAM

namespace okvis {
/**
 * RiSlidingWindowSmoother builds upon gtsam FixedLagSmoother with right invariant errors.
 */
class RiSlidingWindowSmoother : public SlidingWindowSmoother {
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  RiSlidingWindowSmoother(const okvis::BackendParams& backendParams);

  /**
   * @brief Constructor if a ceres map is already available.
   * @param mapPtr Shared pointer to ceres map.
   */
  RiSlidingWindowSmoother(const okvis::BackendParams& backendParams,
                        std::shared_ptr<okvis::ceres::Map> mapPtr);

  virtual ~RiSlidingWindowSmoother();

  void addInitialPriorFactors() override;

  /**
   * @brief addImuValues add values for the last navigation state variable.
   */
  void addImuValues() override;

  /**
   * @brief addImuFactor add the IMU factor for the last navigation state variable.
   */
  void addImuFactor() override;

 protected:
  /**
   * @brief addLandmarkToGraph add a new landmark to the graph.
   * @param landmarkId
   */
  void addLandmarkToGraph(uint64_t landmarkId, const Eigen::Vector3d& pW) override;

  /**
   * @brief updateLandmarkInGraph add observations for an existing landmark.
   * @param landmarkId
   */
  void updateLandmarkInGraph(uint64_t landmarkId) override;

  /**
   * @brief update state variables with FixedLagSmoother estimates.
   */
  void updateStates() override;


  /**
   * @brief gtsamMarginalCovariance compute marginal covariance blocks for the
   * lastest nav state and biases with gtsam::Marginals. It works with
   * Incremental and BatchFixedLagSmoother.
   * @param[in, out] cov
   * @return
   */
  bool gtsamMarginalCovariance(Eigen::MatrixXd* cov) const final;

};
}  // namespace okvis
#else
namespace okvis {
  typedef SlidingWindowSmoother RiSlidingWindowSmoother;
}  // namespace okvis
#endif // # ifdef HAVE_GTSAM
#endif /* INCLUDE_GTSAM_RI_SLIDING_WINDOW_SMOOTHER_HPP_ */
