
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
   * @brief triangulateWithDisparityCheck custom triangulation by using DLT with disparity check
   * @param lmkId
   * @return
   */
  bool triangulateWithDisparityCheck(uint64_t lmkId, Eigen::Vector4d* hpW,
                                     double focalLength, double raySigmaScalar) const;

  /**
   * @brief addLandmarkToGraph add all observations of a landmark as factors to the estimator.
   * @param landmarkId
   * @return true if successfully added.
   */
  bool addLandmarkToGraph(uint64_t landmarkId, const Eigen::Vector4d& hpW);

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

  /**
   * @brief Applies the dropping/marginalization strategy according to the RSS'13/IJRR'14 paper.
   *        The new number of frames in the window will be numKeyframes+numImuFrames.
   * @param numKeyframes Number of keyframes.
   * @param numImuFrames Number of frames in IMU window.
   * @param removedLandmarks Get the landmarks that were removed by this operation.
   * @return True if successful.
   */
  bool applyMarginalizationStrategy(
      size_t numKeyframes, size_t numImuFrames,
      okvis::MapPointVector& removedLandmarks) override;

  /**
   * @brief computeCovariance the block covariance for the navigation state
   * with an abuse of notation, i.e., diag(cov(p, q), cov(v), cov(bg),
   * cov(ba))
   * @param[in, out] cov pointer to the covariance matrix which will be
   * resized in this function.
   * @return true if covariance is successfully computed.
   */
  bool computeCovariance(Eigen::MatrixXd* cov) const override;
};

/**
 * @brief hasMultipleObservationsInOneImage
 * @param mapPoint
 * @return
 */
bool hasMultipleObservationsInOneImage(const MapPoint& mapPoint);

}  // namespace okvis

#endif /* #ifndef INCLUDE_MSCKF_CONSISTENT_ESTIMATOR_HPP_ */
