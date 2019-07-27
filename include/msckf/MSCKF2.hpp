#ifndef INCLUDE_OKVIS_MSCKF2_HPP_
#define INCLUDE_OKVIS_MSCKF2_HPP_

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

//#include <okvis/ceres/ImuError.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>

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
class MSCKF2 : public HybridFilter {
  // landmarks are not in the EKF states in contrast to HybridFilter
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief The default constructor.
   */
  MSCKF2(const double readoutTime);

  /**
   * @brief Constructor if a ceres map is already available.
   * @param mapPtr Shared pointer to ceres map.
   */
  MSCKF2(std::shared_ptr<okvis::ceres::Map> mapPtr,
         const double readoutTime = 0.0);

  virtual ~MSCKF2();

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
  virtual bool applyMarginalizationStrategy() final;

  virtual void optimize(size_t numIter, size_t numThreads = 1,
                        bool verbose = false) final;

 protected:
  // set intermediate variables which are used for computing Jacobians of
  // feature point observations
  virtual void retrieveEstimatesOfConstants() final;

  virtual void updateStates(
      const Eigen::Matrix<double, Eigen::Dynamic, 1>& deltaX) final;

 private:
  uint64_t getMinValidStateID() const;

  /**
   * @brief computeHoi, compute the marginalized Jacobian for a feature i's
   * track assume the number of observations of the map points is at least two
   * @param hpbid homogeneous point parameter block id of the map point
   * @param mp mappoint
   * @param r_oi residuals
   * @param H_oi Jacobians of feature observations w.r.t variables related to
   * camera intrinsics, camera poses (13+9(m-1))
   * @param R_oi covariance matrix of these observations
   * r_oi H_oi and R_oi are values after marginalizing H_fi
   * @return true if succeeded in computing the residual and Jacobians
   */
  bool computeHoi(const uint64_t hpbid, const MapPoint& mp,
                  Eigen::Matrix<double, Eigen::Dynamic, 1>& r_oi,
                  Eigen::MatrixXd& H_oi, Eigen::MatrixXd& R_oi) const;

  int computeStackedJacobianAndResidual(
      Eigen::MatrixXd* T_H, Eigen::Matrix<double, Eigen::Dynamic, 1>* r_q,
      Eigen::MatrixXd* R_q) const;
};

}  // namespace okvis

#endif /* INCLUDE_OKVIS_MSCKF2_HPP_ */
