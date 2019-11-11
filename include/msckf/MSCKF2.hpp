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

#include <okvis/timing/Timer.hpp>
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

  bool measurementJacobianAIDP(
      const Eigen::Vector4d& ab1rho,
      const std::shared_ptr<okvis::cameras::CameraBase> tempCameraGeometry,
      const Eigen::Vector2d& obs, uint64_t poseId, int camIdx,
      uint64_t anchorId, const okvis::kinematics::Transformation& T_WBa,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* H_x,
      Eigen::Matrix<double, 2, 3>* J_pfi, Eigen::Vector2d* residual) const;

  bool measurementJacobian(
      const Eigen::Vector4d& v4Xhomog,
      const std::shared_ptr<okvis::cameras::CameraBase> tempCameraGeometry,
      const Eigen::Vector2d& obs, uint64_t poseId, int camIdx,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* J_Xc,
      Eigen::Matrix<double, 2, 9>* J_XBj, Eigen::Matrix<double, 2, 3>* J_pfi,
      Eigen::Vector2d* residual) const;

 private:
  uint64_t getMinValidStateID() const;

  /**
   * @brief compute the marginalized Jacobian for a feature i's
   * track assume the number of observations of the map points is at least two
   * @param hpbid homogeneous point parameter block id of the map point
   * @param mp mappoint
   * @param r_oi residuals
   * @param H_oi Jacobians of feature observations w.r.t variables related to
   * camera intrinsics, camera poses (13+9(m-1))
   * @param R_oi covariance matrix of these observations
   * r_oi H_oi and R_oi are values after marginalizing H_fi
   * @param involved_frame_ids if not null, all the included frames must observe mp
   * @return true if succeeded in computing the residual and Jacobians
   */
  bool featureJacobian(const MapPoint& mp,
                  Eigen::MatrixXd& H_oi,
                  Eigen::Matrix<double, Eigen::Dynamic, 1>& r_oi,
                  Eigen::MatrixXd& R_oi,
                  const std::vector<uint64_t>* involved_frame_ids=nullptr) const;

  int computeStackedJacobianAndResidual(
      Eigen::MatrixXd* T_H, Eigen::Matrix<double, Eigen::Dynamic, 1>* r_q,
      Eigen::MatrixXd* R_q) const;

  void findRedundantCamStates(
      std::vector<uint64_t>* rm_cam_state_ids);

  // param: max number of cloned frame vector states
  // return number of marginalized frames
  int marginalizeRedundantFrames(size_t maxClonedStates);


  // minimum number of culled frames in each prune frame state
  // step if cloned states size hit maxClonedStates_
  // should be at least 3 for the monocular case so that
  // the marginalized observations can contribute innovation to the states,
  // see Sun 2017 Robust stereo appendix D
  static const int minCulledFrames_ = 3;
};

}  // namespace okvis

#endif /* INCLUDE_OKVIS_MSCKF2_HPP_ */
