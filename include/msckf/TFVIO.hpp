#ifndef INCLUDE_OKVIS_TFVIO_HPP_
#define INCLUDE_OKVIS_TFVIO_HPP_

#include <array>
#include <memory>
#include <mutex>

#include <ceres/ceres.h>
#include <okvis/kinematics/Transformation.hpp>

#include <msckf/HybridFilter.hpp>
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

#include <okvis/timing/Timer.hpp>
#include "msckf/InitialPVandStd.hpp"
#include "vio/CsvReader.h"
#include <vio/ImuErrorModel.h>

/// \brief okvis Main namespace of this package.
namespace okvis {

//! The estimator class
/*!
 The estimator class. This does all the backend work.
 Frames:
 W: World
 B: Body
 C: Camera
 S: Sensor (IMU)
 */
class TFVIO : public HybridFilter {
  // landmarks are not in the EKF states in contrast to HybridFilter
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief The default constructor.
   */
  TFVIO(const double readoutTime);

  /**
   * @brief Constructor if a ceres map is already available.
   * @param mapPtr Shared pointer to ceres map.
   */
  TFVIO(std::shared_ptr<okvis::ceres::Map> mapPtr,
         const double readoutTime = 0.0);

  virtual ~TFVIO();

  /**
   * @brief add a state to the state map
   * @param multiFrame Matched multiFrame.
   * @param imuMeasurements IMU measurements from last state to new one.
   * imuMeasurements covers at least the current state and the last state in
   * time, with an extension on both sides.
   * @param asKeyframe Is this new frame a keyframe?
   * @return True if successful.
   * If it is the first state, initialize it and the covariance matrix. In
   * initialization, please make sure the world frame has z axis in negative
   * gravity direction which is assumed in the IMU propagation Only one IMU is
   * supported for now
   */
  virtual bool addStates(okvis::MultiFramePtr multiFrame,
                         const okvis::ImuMeasurementDeque& imuMeasurements,
                         bool asKeyframe) final;

  /**
   * @brief Applies the dropping/marginalization strategy according to the
   * RSS'13/IJRR'14 paper. The new number of frames in the window will be
   * numKeyframes+numImuFrames.
   * @return True if successful.
   */
  virtual bool applyMarginalizationStrategy(
      size_t numKeyframes, size_t numImuFrames,
      okvis::MapPointVector& removedLandmarks) final;

  virtual void optimize(size_t numIter, size_t numThreads = 1,
                        bool verbose = false) final;

 private:
  uint64_t getMinValidStateID() const;

  /**
   * @brief featureJacobian
   * @param mp MapPoint from which all observations are retrieved
   * @param Hi de_dX
   * @param ri residual 0 - \hat{e}
   * @param Ri cov(\hat{e})
   * @return true if Jacobian is computed successfully
   */
  bool featureJacobian(const MapPoint& mp,
                       Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>* Hi,
                       Eigen::Matrix<double, Eigen::Dynamic, 1>* ri,
                       Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>* Ri) const;

  int computeStackedJacobianAndResidual(
      Eigen::MatrixXd* T_H, Eigen::Matrix<double, Eigen::Dynamic, 1>* r_q,
      Eigen::MatrixXd* R_q) const;

};

}  // namespace okvis

#endif /* INCLUDE_OKVIS_TFVIO_HPP_ */
