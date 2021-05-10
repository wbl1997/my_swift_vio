#ifndef INCLUDE_OKVIS_TFVIO_HPP_
#define INCLUDE_OKVIS_TFVIO_HPP_

#include <array>
#include <memory>
#include <mutex>

#include <ceres/ceres.h>
#include <okvis/kinematics/Transformation.hpp>

#include <swift_vio/HybridFilter.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/Variables.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/CeresIterationCallback.hpp>
#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>
#include <okvis/ceres/Map.hpp>
#include <okvis/ceres/MarginalizationError.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/ReprojectionError.hpp>
#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
//! TFVIO class investigates the effects of epipolar constraints on VIO filter.
//! The best strategies for VIO filtering with epipolar constraints found by TFVIO have
//! been integrated into MSCKF which has extra features like redundant frame marginalization.
//! As a result, to obtain the experiment result for TFVIO, we can use MSCKF
//! with useEpipolarConstraint true and epipolar_sigma_keypoint_size a very large value.

class TFVIO : public HybridFilter {
  // landmarks are not in the EKF states in contrast to HybridFilter
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief The default constructor.
   */
  TFVIO();

  /**
   * @brief Constructor if a ceres map is already available.
   * @param mapPtr Shared pointer to ceres map.
   */
  TFVIO(std::shared_ptr<okvis::ceres::Map> mapPtr);

  virtual ~TFVIO();

  /**
   * @brief Applies the dropping/marginalization strategy according to the
   * RSS'13/IJRR'14 paper. The new number of frames in the window will be
   * numKeyframes+numImuFrames.
   * @return True if successful.
   */
  bool
  applyMarginalizationStrategy(size_t numKeyframes, size_t numImuFrames,
                               okvis::MapPointVector &removedLandmarks) final;

  void optimize(size_t numIter, size_t numThreads = 1,
                bool verbose = false) final;

  int computeStackedJacobianAndResidual(
      Eigen::MatrixXd *T_H, Eigen::Matrix<double, Eigen::Dynamic, 1> *r_q,
      Eigen::MatrixXd *R_q) final;

private:
  uint64_t getMinValidStateId() const;

  // minimum of the ids of the states that have tracked features
  uint64_t minValidStateId_;
};

}  // namespace okvis

#endif /* INCLUDE_OKVIS_TFVIO_HPP_ */
