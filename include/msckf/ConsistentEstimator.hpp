
/**
 * @file msckf/ConsistentEstimator.hpp
 * @brief Header file for the ConsistentEstimator class. This does all the backend work.
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_MSCKF_CONSISTENT_ESTIMATOR_HPP_
#define INCLUDE_MSCKF_CONSISTENT_ESTIMATOR_HPP_

#include <memory>
#include <mutex>
#include <array>

#include <ceres/ceres.h>
#include <okvis/kinematics/Transformation.hpp>

#include <okvis/assert_macros.hpp>
#include <okvis/VioBackendInterface.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Variables.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>
#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>
#include <okvis/ceres/Map.hpp>
#include <okvis/ceres/MarginalizationError.hpp>
#include <okvis/ceres/ReprojectionError.hpp>
#include <okvis/ceres/CeresIterationCallback.hpp>
#include <okvis/Estimator.hpp>

#include <msckf/CameraRig.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
class ConsistentEstimator : public Estimator
{
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief The default constructor.
   */
  ConsistentEstimator();

  /**
   * @brief Constructor if a ceres map is already available.
   * @param mapPtr Shared pointer to ceres map.
   */
  ConsistentEstimator(std::shared_ptr<okvis::ceres::Map> mapPtr);
  virtual ~ConsistentEstimator();

  /**
   * @brief addReprojectionFactors see descriptions in Estimator.
   * @return
   */
  bool addReprojectionFactors();

  /**
   * @brief Start ceres optimization.
   * @param[in] numIter Maximum number of iterations.
   * @param[in] numThreads Number of threads.
   * @param[in] verbose Print out optimization progress and result, if true.
   */
  void optimize(size_t numIter, size_t numThreads = 1, bool verbose = false) override;

};
}  // namespace okvis

#endif /* #ifndef INCLUDE_MSCKF_CONSISTENT_ESTIMATOR_HPP_ */
