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

/// \brief okvis Main namespace of this package.
namespace okvis {
enum class MeasurementJacobianStatus {
  Successful = 0,
  GeneralProjectionFailed = 1,
  MainAnchorProjectionFailed = 2,
  AssociateAnchorProjectionFailed = 3,
};

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
  MSCKF2();

  /**
   * @brief Constructor if a ceres map is already available.
   * @param mapPtr Shared pointer to ceres map.
   */
  MSCKF2(std::shared_ptr<okvis::ceres::Map> mapPtr);

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
  /**
   * @brief measurementJacobianAIDP
   * @warning Both poseId and anchorId are older than the latest frame Id.
   * @param ab1rho
   * @param tempCameraGeometry
   * @param obs
   * @param observationIndex index of the observation inside the point's shared data.
   * @param pointDataPtr shared data of the point.
   * @param anchorId
   * @param T_WBa
   * @param H_x Jacobians of the image observation relative to the camera parameters and cloned states.
   *     It ought to be allocated in advance.
   * @param J_pfi Jacobian of the image observation relative to [\alpha, \beta, \rho].
   * @param residual
   * @return
   */
  bool measurementJacobianAIDP(
      const Eigen::Vector4d& ab1rho,
      std::shared_ptr<const okvis::cameras::CameraBase> tempCameraGeometry,
      const Eigen::Vector2d& obs, int observationIndex,
      std::shared_ptr<const msckf::PointSharedData> pointDataPtr,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* H_x,
      Eigen::Matrix<double, 2, 3>* J_pfi, Eigen::Vector2d* residual) const;

  bool measurementJacobian(
      const Eigen::Vector4d& v4Xhomog,
      std::shared_ptr<const okvis::cameras::CameraBase> tempCameraGeometry,
      const Eigen::Vector2d& obs, int observationIndex,
      std::shared_ptr<const msckf::PointSharedData> pointData,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* J_Xc,
      Eigen::Matrix<double, 2, 9>* J_XBj, Eigen::Matrix<double, 2, 3>* J_pfi,
      Eigen::Vector2d* residual) const;

  MeasurementJacobianStatus measurementJacobianGeneric(
      const msckf::PointLandmark& pointLandmark,
      std::shared_ptr<const okvis::cameras::CameraBase> tempCameraGeometry,
      const Eigen::Vector2d& obs, int observationIndex,
      const Eigen::Matrix2d& obsCovariance,
      std::shared_ptr<const msckf::PointSharedData> pointDataPtr,
      Eigen::MatrixXd* J_X,
      Eigen::Matrix<double, Eigen::Dynamic, 3>* J_pfi,
      Eigen::Matrix<double, Eigen::Dynamic, 2>* J_n,
      Eigen::VectorXd* residual) const;

  bool featureJacobianGeneric(
      const MapPoint& mp, Eigen::MatrixXd& H_oi,
      Eigen::Matrix<double, Eigen::Dynamic, 1>& r_oi, Eigen::MatrixXd& R_oi,
      std::vector<uint64_t>* involved_frame_ids) const;

  /**
   * @brief compute the marginalized Jacobian for a feature i's track.
   * @warning The map point is not observed in the latest frame.
   * @warning The number of observations of the map points is at least two.
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
                  std::vector<uint64_t>* involved_frame_ids=nullptr) const;

private:
  int computeStackedJacobianAndResidual(
      Eigen::MatrixXd* T_H, Eigen::Matrix<double, Eigen::Dynamic, 1>* r_q,
      Eigen::MatrixXd* R_q) const;

  void findRedundantCamStates(
      std::vector<uint64_t>* rm_cam_state_ids);

  // param: max number of cloned frame vector states
  // return number of marginalized frames
  int marginalizeRedundantFrames(size_t maxClonedStates);

  // use epipolar constraints in case of low disparity or triangulation failure?
  bool useEpipolarConstraint_;

  // minimum number of culled frames in each prune frame state
  // step if cloned states size hit maxClonedStates_
  // should be at least 3 for the monocular case so that
  // the marginalized observations can contribute innovation to the states,
  // see Sun 2017 Robust stereo appendix D
  static const int minCulledFrames_ = 3;
};

}  // namespace okvis

#endif /* INCLUDE_OKVIS_MSCKF2_HPP_ */
