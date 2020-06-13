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
   * @brief measurementJacobian
   * @warning Both poseId and anchorId should be older than the latest frame Id.
   * @param homogeneousPoint, if landmarkModel is AIDP,
   * \f$[\alpha, \beta, 1, \rho] = [X, Y, Z, 1]^C_a / Z^C_a\f$,
   * if landmarkModel is HPP, \f$[X,Y,Z,1]^W\f$.
   * \f $[\alpha, \beta, 1]^T = \rho p_{C{t(i, a)}} \f$ or
   * \f $[\alpha, \beta, 1]^T = \rho p_{C{t(a)}} \f$
   * @param obs
   * @param observationIndex index of the observation inside the point's shared data.
   * @param pointDataPtr shared data of the point.
   * @param anchorId
   * @param T_WBa
   * @param J_x Jacobians of the image observation relative to the camera parameters and cloned states.
   *     It ought to be allocated in advance.
   * @param J_pfi Jacobian of the image observation relative to [\alpha, \beta, \rho].
   * @param residual
   * @return
   */
  bool measurementJacobian(
      const Eigen::Vector4d& homogeneousPoint,
      const Eigen::Vector2d& obs, size_t observationIndex,
      std::shared_ptr<const msckf::PointSharedData> pointDataPtr,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* J_x,
      Eigen::Matrix<double, 2, 3>* J_pfi, Eigen::Vector2d* residual) const;

  /**
   * @brief measurementJacobianAIDPMono
   * @warning legacy method to check measurementJacobian in monocular case
   * with anchored inverse depth parameterization.
   * @param ab1rho
   * @param obs
   * @param observationIndex
   * @param pointDataPtr
   * @param H_x
   * @param J_pfi
   * @param residual
   * @return
   */
  bool measurementJacobianAIDPMono(
      const Eigen::Vector4d& ab1rho,
      const Eigen::Vector2d& obs, size_t observationIndex,
      std::shared_ptr<const msckf::PointSharedData> pointDataPtr,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* H_x,
      Eigen::Matrix<double, 2, 3>* J_pfi, Eigen::Vector2d* residual) const;

  /**
   * @brief measurementJacobianHPPMono
   * @warning legacy method to check measurementJacobian
   * in monocular homogeneous parameterization case.
   * @param v4Xhomog
   * @param obs
   * @param observationIndex
   * @param pointData
   * @param J_Xc
   * @param J_XBj
   * @param J_pfi
   * @param residual
   * @return
   */
  bool measurementJacobianHPPMono(
      const Eigen::Vector4d& v4Xhomog,
      const Eigen::Vector2d& obs, int observationIndex,
      std::shared_ptr<const msckf::PointSharedData> pointData,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* J_Xc,
      Eigen::Matrix<double, 2, 9>* J_XBj, Eigen::Matrix<double, 2, 3>* J_pfi,
      Eigen::Vector2d* residual) const;

  /**
   * @brief Generic measurement Jacobian to handle different camera measurement
   * factors, e.g., reprojection error, chordal distance, etc.
   * @warning only supports one camera.
   * @param pointLandmark
   * @param tempCameraGeometry
   * @param obs
   * @param obsCovariance
   * @param observationIndex
   * @param pointDataPtr
   * @param J_X
   * @param J_pfi
   * @param J_n
   * @param residual
   * @return
   */
  msckf::MeasurementJacobianStatus measurementJacobianGeneric(
      const msckf::PointLandmark& pointLandmark,
      std::shared_ptr<const okvis::cameras::CameraBase> tempCameraGeometry,
      const Eigen::Vector2d& obs,
      const Eigen::Matrix2d& obsCovariance,
      int observationIndex,
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

  virtual void setKeyframeRedundancyThresholds(double dist, double angle,
                                               double trackingRate,
                                               size_t minTrackLength,
                                               size_t numKeyframes,
                                               size_t numImuFrames) final;

 private:
  int computeStackedJacobianAndResidual(
      Eigen::MatrixXd* T_H, Eigen::Matrix<double, Eigen::Dynamic, 1>* r_q,
      Eigen::MatrixXd* R_q) const;

  void findRedundantCamStates(
      std::vector<uint64_t>* rm_cam_state_ids,
      size_t numImuFrames);

  /**
   * @brief marginalizeRedundantFrames
   * @param numKeyframes
   * @param numImuFrames
   * @return number of marginalized frames
   */
  int marginalizeRedundantFrames(size_t numKeyframes, size_t numImuFrames);

  // minimum number of culled frames in each prune frame state
  // step if cloned states size hit maxClonedStates_
  // should be at least 3 for the monocular case so that
  // the marginalized observations can contribute innovation to the states,
  // see Sun 2017 Robust stereo appendix D
  size_t minCulledFrames_;
};

/**
 * @brief compute the Jacobians of T_BC relative to extrinsic parameters.
 * Perturbation in T_BC is defined by kinematics::oplus.
 * Perturbation in extrinsic parameters are defined by extrinsic models.
 * @param T_BCi Transform from i camera frame to body frame.
 * @param T_BC0 Transform from main camera frame to body frame.
 * @param cameraExtrinsicModelId
 * @param mainCameraExtrinsicModelId
 * @param[out] dT_BCi_dExtrinsics list of Jacobians for T_BC.
 * @param[in, out] involvedCameraIndices observation camera index, and main camera index if T_C0Ci extrinsic model is used.
 * @pre involvedCameraIndices has exactly one camera index for i camera frame.
 */
void computeExtrinsicJacobians(
    const okvis::kinematics::Transformation& T_BCi,
    const okvis::kinematics::Transformation& T_BC0,
    int cameraExtrinsicModelId, int mainCameraExtrinsicModelId,
    AlignedVector<Eigen::MatrixXd>* dT_BCi_dExtrinsics,
    std::vector<size_t>* involvedCameraIndices,
    size_t mainCameraIndex);

}  // namespace okvis

#endif /* INCLUDE_OKVIS_MSCKF2_HPP_ */
