#ifndef INCLUDE_OKVIS_HYBRID_FILTER_HPP_
#define INCLUDE_OKVIS_HYBRID_FILTER_HPP_

#include <array>
#include <memory>
#include <mutex>

#include <ceres/ceres.h>
#include <okvis/kinematics/Transformation.hpp>

#include <okvis/FrameTypedefs.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/Variables.hpp>
#include <okvis/VioBackendInterface.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/CeresIterationCallback.hpp>
#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>
#include <okvis/ceres/Map.hpp>
#include <okvis/ceres/MarginalizationError.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/ReprojectionError.hpp>
#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>
#include <msckf/EuclideanParamBlockSized.hpp>
#include <okvis/Estimator.hpp>
#include <okvis/timing/Timer.hpp>

#include <msckf/BaseFilter.h>
#include <msckf/CameraRig.hpp>
#include <msckf/MotionAndStructureStats.h>
#include <msckf/PointLandmark.hpp>
#include <msckf/PointSharedData.hpp>
#include <msckf/memory.h>
#include <msckf/imu/ImuOdometry.h>

/// \brief okvis Main namespace of this package.
namespace okvis {

enum RetrieveObsSeqType {
    ENTIRE_TRACK=0,
    LATEST_TWO,
    HEAD_TAIL,
};


//! The HybridFilter that uses short feature track observations in the MSCKF
//!  manner and long feature track observations in the SLAM manner, i.e.,
//! put into the state vector.
/*!
 The estimator class does all the backend work.
 Frames:
 W: World
 C: Camera
 S: Sensor (IMU)
 B: Body, defined by the IMU model, e.g., Imu_BG_BA, usually defined close to S.
 Its relation to the camera frame is modeled by the extrinsic model, e.g., Extrinsic_p_CB.
 */
class HybridFilter : public Estimator, public BaseFilter {
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief The default constructor.
   */
  HybridFilter();

  /**
   * @brief Constructor if a ceres map is already available.
   * @param mapPtr Shared pointer to ceres map.
   */
  HybridFilter(std::shared_ptr<okvis::ceres::Map> mapPtr);

  virtual ~HybridFilter();

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
                         const okvis::ImuMeasurementDeque &imuMeasurements,
                         bool asKeyframe) final;


  /**
   * @obsolete the Jacobians only supports one camera.
   * @brief Applies the dropping/marginalization strategy according to the
   * RSS'13/IJRR'14 paper. The new number of frames in the window will be
   * numKeyframes+numImuFrames.
   * @return True if successful.
   */
  virtual bool applyMarginalizationStrategy(
      size_t numKeyframes, size_t numImuFrames,
      okvis::MapPointVector &removedLandmarks);

  /**
   * @obsolete the Jacobians only supports one camera.
   * @brief optimize
   * @param numIter
   * @param numThreads
   * @param verbose
   */
  virtual void optimize(size_t numIter, size_t numThreads = 1,
                        bool verbose = false);

  /// @name Getters
  ///\{

  int getEstimatedVariableMinimalDim() const final {
    return covariance_.rows();
  }

  bool computeCovariance(Eigen::MatrixXd* cov) const final {
    *cov = covariance_;
    return true;
  }
  ///@}

  /**
   * @brief gatherMapPointObservations
   * @param mp
   * @param pointDataPtr
   * @param obsDirections Each observation is in image plane z=1, (\bar{x}, \bar{y}, 1).
   * @param obsInPixel
   * @param vSigmai
   * @param orderedBadFrameIds[out] Ids of frames that have bad back
   *     projections of the mappoint.
   * @param seqType
   * @return number of gathered valid observations
   */
  size_t gatherMapPointObservations(
      const MapPoint &mp,
      msckf::PointSharedData *pointDataPtr,
      std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
          *obsDirections,
      std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
          *obsInPixel,
      std::vector<double> *vSigmai,
      std::vector<std::pair<uint64_t, int>>* orderedBadFrameIds,
      RetrieveObsSeqType seqType = ENTIRE_TRACK) const;

  /**
   * @brief hasLowDisparity check if a feature track has low disparity at its endpoints
   * @param obsDirections [x, y, 1]
   * @param T_WSs assume the poses are either optimized or propagated with IMU data
   * @return true if low disparity at its endpoints
   */
  bool hasLowDisparity(
      const std::vector<Eigen::Vector3d,
                        Eigen::aligned_allocator<Eigen::Vector3d>>
          &obsDirections,
      const std::vector<
          okvis::kinematics::Transformation,
          Eigen::aligned_allocator<okvis::kinematics::Transformation>> &T_WSs,
      const AlignedVector<okvis::kinematics::Transformation>& T_BCs,
      const std::vector<size_t>& camIndices) const;

  bool isPureRotation(const MapPoint& mp) const;

  void propagatePoseAndVelocityForMapPoint(msckf::PointSharedData* pointDataPtr) const;

  /**
   * @brief triangulateAMapPoint initialize a landmark with proper
   * parameterization. If not PAP, then it does not support rays which arise
   * from static mode, pure rotation, or points at infinity. Assume the same
   * camera model for all observations, and rolling shutter effect is not
   * accounted for in triangulation.
   * This function will determine the anchor frames if applicable.
   * @param mp
   * @param obsInPixel
   * @param frameIds, id of frames observing this feature in the ascending order
   *    because the MapPoint.observations is an ordinary ordered map
   * @param pointLandmark, stores [X,Y,Z,1] in the global frame,
   *  or [X,Y,Z,1] in the anchor frame,
   *  or [cos\theta, sin\theta, n_x, n_y, n_z] depending on anchorSeqId.
   * @param vSigmai, the diagonal elements of the observation noise matrix, in
   *    pixels, size 2Nx1
   * @param pointDataPtr[in/out] shared data of the map point for computing
   * poses and velocity at observation.
   * The anchor frames will be set depending on the landmarkModelId_.
   * If 0, none anchor.
   * else if 1, last frame (of orderedCulledFrameIds if not null) observing the map point
   * else if 2, last frame (of orderedCulledFrameIds if not null) is main anchor,
   * first frame is associate anchor.
   * @param orderedCulledFrameIds[in/out] Ordered Ids of frames to be used for
   * update, e.g., in marginalization. Some frame Ids may be erased in this
   * function due to bad back projection.
   * @return true if triangulation successful
   */
  msckf::TriangulationStatus triangulateAMapPoint(
      const MapPoint &mp,
      std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
          &obsInPixel,
      msckf::PointLandmark& pointLandmark,
      std::vector<double> &vSigmai,
      msckf::PointSharedData* pointDataPtr,
      std::vector<uint64_t>* orderedCulledFrameIds,
      bool checkDisparity = false) const;

  /**
   * @obsolete the Jacobians only supports one camera.
   * @brief slamFeatureJacobian, compute the residual and Jacobians for a SLAM feature i
   * observed in current frame
   * @param hpbid homogeneous point parameter block id of the map point
   * @param mp mappoint
   * @param r_i residual of the observation of the map point in the latest frame
   * @param H_x Jacobian w.r.t variables related to camera intrinsics, camera
   * poses (13+9m)
   * @param H_f Jacobian w.r.t variables of features (3s_k)
   * @param R_i covariance matrix of this observation (2x2)
   * @return true if succeeded in computing the residual and Jacobians
   */
  bool slamFeatureJacobian(const uint64_t hpbid, const MapPoint &mp,
                           Eigen::Matrix<double, 2, 1> &r_i,
                           Eigen::Matrix<double, 2, Eigen::Dynamic> &H_x,
                           Eigen::Matrix<double, 2, Eigen::Dynamic> &H_f,
                           Eigen::Matrix2d &R_i);

  /**
   * @obsolete The jacobians do not support ncameras.
   * @brief compute the marginalized Jacobian for a feature i's track
   * assume the number of observations of the map points is at least two
   * @param mp mappoint
   * @param H_oi Jacobians of feature observations w.r.t variables related to
   camera intrinsics, camera poses (13+9(m-1)-3)
   * @param r_oi residuals
   * @param R_oi covariance matrix of these observations
   * @param ab1rho [\alpha, \beta, 1, \rho] of the point in the anchor frame,
   representing either an ordinary point or a ray
   * @param pH_fi pointer to the Jacobian of feature observations w.r.t the
   feature parameterization,[\alpha, \beta, \rho]
   * if pH_fi is NULL, r_oi H_oi and R_oi are values after marginalizing H_fi,
   H_oi is of size (2n-3)x(13+9(m-1)-3);
   * otherwise, H_oi is of size 2nx(13+9(m-1)-3)
   * @return true if succeeded in computing the residual and Jacobians
   */
  bool featureJacobian(
      const MapPoint &mp, Eigen::MatrixXd &H_oi,
      Eigen::Matrix<double, Eigen::Dynamic, 1> &r_oi, Eigen::MatrixXd &R_oi,
      Eigen::Vector4d &ab1rho,
      Eigen::Matrix<double, Eigen::Dynamic, 3> *pH_fi =
          (Eigen::Matrix<double, Eigen::Dynamic, 3> *)(NULL)) const;

  void cloneFilterStates(StatePointerAndEstimateList *currentStates) const override;

  void boxminusFromInput(
      const StatePointerAndEstimateList& refState,
      Eigen::Matrix<double, Eigen::Dynamic, 1>* deltaX) const override;

  void updateStates(const Eigen::Matrix<double, Eigen::Dynamic, 1> &deltaX) override;

  /// print out the most recent state vector and the stds of its elements.
  /// It can be called in the optimizationLoop, but a better way to save
  /// results is to save in the publisher loop
  virtual bool print(std::ostream &stream) const final;

  virtual void printTrackLengthHistogram(std::ostream &stream) const final;

  void getCameraTimeParameterPtrs(
      std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>
          *cameraDelayParameterPtrs,
      std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>
          *cameraReadoutTimeParameterPtrs) const;

  std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>
      getImuAugmentedParameterPtrs() const;

  virtual void getCameraCalibrationEstimate(
      Eigen::Matrix<double, Eigen::Dynamic, 1>* cameraParams) const final;

  virtual void getImuAugmentedStatesEstimate(
      Eigen::Matrix<double, Eigen::Dynamic, 1>* extraParams) const final;

  bool getStateStd(Eigen::Matrix<double, Eigen::Dynamic, 1>* stateStd) const final;

  void setKeyframeRedundancyThresholds(double dist, double angle,
                                       double trackingRate,
                                       size_t minTrackLength,
                                       size_t numKeyframes,
                                       size_t numImuFrames) override;

  void initCovariance();

  void initCameraParamCovariance(int camIdx);

  void addCovForClonedStates();

  /**
   * @brief minimalDimOfAllCameraParams
   * @warning call this no earlier than first call of addStates().
   * @return
   */
  inline size_t minimalDimOfAllCameraParams() const {
    size_t totalCamDim = statesMap_.rbegin()
                             ->second.sensors.at(SensorStates::Camera)
                             .back()
                             .at(CameraSensorStates::TR)
                             .startIndexInCov +
                         1;
    size_t totalImuDim = statesMap_.rbegin()
                             ->second.sensors.at(SensorStates::Camera)
                             .at(0u)
                             .at(CameraSensorStates::T_SCi)
                             .startIndexInCov;
    return totalCamDim - totalImuDim;
  }

  /**
   * @brief cameraParamsMinimalDimFast
   * @warning call this no earlier than first call of addStates().
   * @param camIdx
   * @return
   */
  size_t cameraParamsMinimalDimFast(size_t camIdx) const {
    size_t totalInclusiveDim = statesMap_.rbegin()
                                   ->second.sensors.at(SensorStates::Camera)
                                   .at(camIdx)
                                   .at(CameraSensorStates::TR)
                                   .startIndexInCov +
                               1;
    size_t totalExclusiveDim = statesMap_.rbegin()
                                   ->second.sensors.at(SensorStates::Camera)
                                   .at(camIdx)
                                   .at(CameraSensorStates::T_SCi)
                                   .startIndexInCov;
    return totalInclusiveDim - totalExclusiveDim;
  }

  /**
   * @brief minimal dim of camera parameters and all cloned states including the last
   * inserted one and all landmarks.
   * Ex: C_p_B, f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, [k_3], t_d, t_r,
   * C0_p_Ci, C0_q_Ci, f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, [k_3], t_d, t_r,
   * \pi_{B_i}(=[p_{B_i}^G, q_{B_i}^G, v_{B_i}^G]), l_i
   * @warning call this no earlier than first call of addStates().
   * @return
   */
  inline size_t cameraParamPoseAndLandmarkMinimalDimen() const {
    return minimalDimOfAllCameraParams() +
           kClonedStateMinimalDimen * statesMap_.size() +
           3 * mInCovLmIds.size();
  }

  inline size_t startIndexOfClonedStates() const {
    size_t dim = okvis::ceres::ode::kNavErrorStateDim + imu_rig_.getImuParamsMinimalDim(0);
    for (size_t j = 0; j < camera_rig_.numberCameras(); ++j) {
      dim += cameraParamsMinimalDimen(j);
    }
    return dim;
  }

  /**
   * @brief startIndexOfClonedStatesFast
   * @warning call this no earlier than first call of addStates().
   * @return
   */
  inline size_t startIndexOfClonedStatesFast() const {
    return statesMap_.rbegin()
               ->second.sensors.at(SensorStates::Camera)
               .back()
               .at(CameraSensorStates::TR)
               .startIndexInCov +
           1;
  }

  inline size_t startIndexOfCameraParams(size_t camIdx = 0u) const {
    size_t dim = okvis::ceres::ode::kNavErrorStateDim + imu_rig_.getImuParamsMinimalDim(0);
    for (size_t i = 0u; i < camIdx; ++i) {
      dim += cameraParamsMinimalDimen(i);
    }
    return dim;
  }

  /**
   * @brief startIndexOfCameraParamsFast
   * @warning call this no earlier than first call of addStates().
   * @param camIdx
   * @return
   */
  inline size_t startIndexOfCameraParamsFast(
      size_t camIdx,
      CameraSensorStates camParamBlockName = CameraSensorStates::T_SCi) const {
    return statesMap_.rbegin()
        ->second.sensors.at(SensorStates::Camera)
        .at(camIdx)
        .at(camParamBlockName)
        .startIndexInCov;
  }

  /**
   * @brief intraStartIndexOfCameraParams
   * @warning call this no earlier than first call of addStates().
   * @param camIdx
   * @return
   */
  inline size_t intraStartIndexOfCameraParams(
      size_t camIdx,
      CameraSensorStates camParamBlockName = CameraSensorStates::T_SCi) const {
    size_t totalInclusiveDim = statesMap_.rbegin()
                                   ->second.sensors.at(SensorStates::Camera)
                                   .at(camIdx)
                                   .at(camParamBlockName)
                                   .startIndexInCov;
    size_t totalExclusiveDim = statesMap_.rbegin()
                                   ->second.sensors.at(SensorStates::Camera)
                                   .at(0u)
                                   .at(CameraSensorStates::T_SCi)
                                   .startIndexInCov;
    return totalInclusiveDim - totalExclusiveDim;
  }

  /**
   * @brief navStateAndImuParamsMinimalDim
   * @warning assume only one IMU is used.
   * @param imuIdx
   * @return
   */
  inline size_t navStateAndImuParamsMinimalDim(size_t imuIdx = 0u) {
    return okvis::ceres::ode::kNavErrorStateDim +
           imu_rig_.getImuParamsMinimalDim(imuIdx);
  }

  // error state: \delta p, \alpha for q, \delta v
  // state: \pi_{B_i}(=[p_{B_i}^G, q_{B_i}^G, v_{B_i}^G])
  static const int kClonedStateMinimalDimen = 9;

 protected:
  virtual bool getOdometryConstraintsForKeyframe(
      std::shared_ptr<okvis::LoopQueryKeyframeMessage> queryKeyframe) const final;

  // using latest state estimates set imu_rig_ and camera_rig_ which are then
  // used in computing Jacobians of all feature observations
  void updateSensorRigs();

  void cloneImuAugmentedStates(
      const States &stateInQuestion,
      StatePointerAndEstimateList *currentStates) const;

  void cloneCameraParameterStates(
      const States &stateInQuestion,
      StatePointerAndEstimateList *currentStates,
      size_t camIdx) const;

  uint64_t getMinValidStateId() const;

  void addImuAugmentedStates(const okvis::Time stateTime, int imu_id,
                             SpecificSensorStatesContainer* imuInfo);

  /**
   * @brief updateImuRig update imu_rig_ from states.
   * @param
   */
  void updateImuRig();

  void updateCovarianceIndex();

  /**
   * @brief updateImuAugmentedStates update states with correction.
   * @param stateInQuestion
   * @param deltaAugmentedParams
   */
  void updateImuAugmentedStates(const States& stateInQuestion,
                                const Eigen::VectorXd deltaAugmentedParams);

  void updateCameraSensorStates(const States& stateInQuestion,
                                const Eigen::VectorXd& deltaX);

  void usePreviousCameraParamBlocks(
      std::map<uint64_t, States>::const_reverse_iterator prevStateRevIter,
      size_t cameraIndex, SpecificSensorStatesContainer* cameraInfos) const;

  void initializeCameraParamBlocks(okvis::Time stateEpoch, size_t cameraIndex,
                                   SpecificSensorStatesContainer *cameraInfos);

  // epipolar measurement used by filters for computing Jacobians
  // https://en.cppreference.com/w/cpp/language/nested_types
  class EpipolarMeasurement {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EpipolarMeasurement(
        const HybridFilter& filter,
        const uint32_t imageHeight,
        int camIdx, int extrinsicModelId, int minExtrinsicDim, int minProjDim,
        int minDistortDim);

    void prepareTwoViewConstraint(
        const std::vector<Eigen::Vector3d,
                          Eigen::aligned_allocator<Eigen::Vector3d>>
            &obsDirections,
        const std::vector<Eigen::Vector2d,
                          Eigen::aligned_allocator<Eigen::Vector2d>>
            &obsInPixels,
        const std::vector<
            Eigen::Matrix<double, 3, Eigen::Dynamic>,
            Eigen::aligned_allocator<Eigen::Matrix<double, 3, Eigen::Dynamic>>>
            &dfj_dXcam,
        const std::vector<int> &index_vec);

    /**
     @ @obsolete the Jacobians do not support ncameras.
     * @brief measurementJacobian
     *     The Jacobians for state variables except for states related to time
     *     can be computed with automatic differentiation.
     * z = h(X, n) \\
     * X = \hat{X} \oplus \delta \chi \\
     * e.g., R = exp(\delta \theta)\hat{R}\\
     * r = z - h(\hat{X}, 0) = J_x \delta \chi + J_n n \\
     * J_x = \lim_{\delta \chi \rightarrow 0} \frac{h(\hat{x}\oplus\delta\chi) - h(\hat{x})}{\delta \chi}
     * The last equation can be used in ceres AutoDifferentiate function for computing the Jacobian.
     * @param H_xjk
     * @param H_fjk
     * @param residual
     * @return
     */
    bool measurementJacobian(
        std::shared_ptr<const msckf::PointSharedData> pointDataPtr,
        const std::vector<int> observationIndexPairs,
        Eigen::Matrix<double, 1, Eigen::Dynamic> *H_xjk,
        std::vector<Eigen::Matrix<double, 1, 3>,
                    Eigen::aligned_allocator<Eigen::Matrix<double, 1, 3>>>
            *H_fjk,
        double *residual) const;

   private:
    const HybridFilter& filter_;
    const int camIdx_;
    const uint32_t imageHeight_;
    const int extrinsicModelId_;
    const int minExtrinsicDim_;
    const int minProjDim_;
    const int minDistortDim_;

    std::vector<uint64_t> frameId2;
    std::vector<okvis::kinematics::Transformation,
                Eigen::aligned_allocator<okvis::kinematics::Transformation>>
        T_WS2;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        obsDirection2;
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
        obsInPixel2;
    std::vector<
        Eigen::Matrix<double, 3, Eigen::Dynamic>,
        Eigen::aligned_allocator<Eigen::Matrix<double, 3, Eigen::Dynamic>>>
        dfj_dXcam2;
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>
        cov_fj2;
  };

  /**
   * @brief featureJacobian
   * @warn Number of columns of Hi equals to the required variable dimen.
   * @param mp MapPoint from which all observations are retrieved
   * @param Hi de_dX.
   * @param ri residual 0 - \hat{e}
   * @param Ri cov(\hat{e})
   * @return true if Jacobian is computed successfully
   */
  bool featureJacobianEpipolar(
      const MapPoint &mp,
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> *Hi,
      Eigen::Matrix<double, Eigen::Dynamic, 1> *ri,
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> *Ri,
      RetrieveObsSeqType seqType) const;

  template <class CameraGeometry, class ProjectionIntrinsicModel,
  class ExtrinsicModel, class PointLandmarkModel, class ImuModel>
  msckf::MeasurementJacobianStatus computeCameraObservationJacobians(
      const msckf::PointLandmark& pointLandmark,
      std::shared_ptr<const okvis::cameras::CameraBase> baseCameraGeometry,
      const Eigen::Vector2d& obs,
      const Eigen::Matrix2d& obsCov, int observationIndex,
      std::shared_ptr<const msckf::PointSharedData> pointDataPtr,
      Eigen::MatrixXd* J_X, Eigen::Matrix<double, Eigen::Dynamic, 3>* J_pfi,
      Eigen::Matrix<double, Eigen::Dynamic, 2>* J_n,
      Eigen::VectorXd* residual) const;


  /// the error vector corresponds to states x_B | x_imu | x_c | \pi{B_{N-m}}
  /// ... \pi{B_{N-1}} following Li icra 2014 x_B = [^{G}p_B] ^{G}q_B ^{G}v_B
  /// b_g b_a]

  // minimum of the ids of the states that have tracked features
  uint64_t minValidStateId_;

  mutable okvis::timing::Timer triangulateTimer;
  mutable okvis::timing::Timer computeHTimer;
  okvis::timing::Timer updateLandmarksTimer;

  // for each point in the state vector/covariance,
  // its landmark id which points to the parameter block
  std::deque<uint64_t> mInCovLmIds;

  // maximum number of consecutive observations until a landmark is added as a
  // state, but can be set dynamically as done in
  // Li, icra14 optimization based ...
  static const size_t maxTrackLength_ = 12;
  // i.e., max cloned states in the cov matrix

  std::vector<size_t>
      mTrackLengthAccumulator;  // histogram of the track lengths, start from
                                // 0,1,2, to a fixed number
  msckf::MotionAndStructureStats slamStats_;
  double trackingRate_;
  // Threshold for determine keyframes
  double translationThreshold_;
  double rotationThreshold_;
  double trackingRateThreshold_;

  size_t numImuFrames_;
  size_t numKeyframes_;
};
}  // namespace okvis
#include <msckf/implementation/HybridFilter.hpp>
#endif /* INCLUDE_OKVIS_HYBRID_FILTER_HPP_ */
