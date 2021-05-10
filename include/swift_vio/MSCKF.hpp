#ifndef INCLUDE_SWIFT_VIO_MSCKF_HPP_
#define INCLUDE_SWIFT_VIO_MSCKF_HPP_

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

/**
 * @brief The MSCKF class implement the MSCKF with first estimate Jacobian technique.
 * It does not include landmarks in the state vector but supports iterative EKF.
 */
class MSCKF : public HybridFilter {
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief The default constructor.
   */
  MSCKF();

  /**
   * @brief Constructor if a ceres map is already available.
   * @param mapPtr Shared pointer to ceres map.
   */
  MSCKF(std::shared_ptr<okvis::ceres::Map> mapPtr);

  virtual ~MSCKF();

  void optimize(size_t numIter, size_t numThreads = 1,
                bool verbose = false) final;

  /**
   * @brief measurementJacobianAIDPMono
   * @obsolete legacy method to check measurementJacobian in monocular case
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
      std::shared_ptr<const swift_vio::PointSharedData> pointDataPtr,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* H_x,
      Eigen::Matrix<double, 2, 3>* J_pfi, Eigen::Vector2d* residual) const;

  /**
   * @brief measurementJacobianHPPMono
   * @obsolete legacy method to check measurementJacobian
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
      std::shared_ptr<const swift_vio::PointSharedData> pointData,
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
  swift_vio::MeasurementJacobianStatus measurementJacobianGeneric(
      const swift_vio::PointLandmark &pointLandmark,
      std::shared_ptr<const okvis::cameras::CameraBase> tempCameraGeometry,
      const Eigen::Vector2d &obs, const Eigen::Matrix2d &obsCovariance,
      int observationIndex,
      std::shared_ptr<const swift_vio::PointSharedData> pointDataPtr,
      Eigen::MatrixXd *J_X, Eigen::Matrix<double, Eigen::Dynamic, 3> *J_pfi,
      Eigen::Matrix<double, Eigen::Dynamic, 2> *J_n,
      Eigen::VectorXd *residual) const;

  bool
  featureJacobian(const MapPoint &mp, swift_vio::PointLandmark *pointLandmark,
                  Eigen::MatrixXd &H_oi,
                  Eigen::Matrix<double, Eigen::Dynamic, 1> &r_oi,
                  Eigen::MatrixXd &R_oi,
                  Eigen::Matrix<double, Eigen::Dynamic, 3> *pH_fi = nullptr,
                  std::vector<uint64_t> *involved_frame_ids = nullptr) const override;

  bool featureJacobianGeneric(
      const MapPoint &mp, swift_vio::PointLandmark *pointLandmark,
      Eigen::MatrixXd &H_oi,
      Eigen::Matrix<double, Eigen::Dynamic, 1> &r_oi, Eigen::MatrixXd &R_oi,
      Eigen::Matrix<double, Eigen::Dynamic, 3> *pH_fi = nullptr,
      std::vector<uint64_t> *involved_frame_ids = nullptr) const;

  int computeStackedJacobianAndResidual(
      Eigen::MatrixXd *T_H, Eigen::Matrix<double, Eigen::Dynamic, 1> *r_q,
      Eigen::MatrixXd *R_q) final;

};


}  // namespace okvis

#endif /* INCLUDE_SWIFT_VIO_MSCKF_HPP_ */
