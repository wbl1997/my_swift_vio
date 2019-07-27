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

//#include <okvis/ceres/ImuError.hpp>
#include <msckf/CameraRig.h>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>

#include <okvis/timing/Timer.hpp>

#include <vio/CsvReader.h>
#include <vio/ImuErrorModel.h>

#include "msckf/InitialPVandStd.hpp"

/// \brief okvis Main namespace of this package.
namespace okvis {

//! The estimator class
/*!
 The estimator class. This does all the backend work.
 Frames:
 W: World
 B: Body, usu. tied to S and denoted by S in this codebase
 C: Camera
 S: Sensor (IMU), S frame is defined such that its rotation component is
     fixed to the nominal value of R_SC0 and its origin is at the
     accelerometer intersection as discussed in Huai diss. In this case, the
     remaining misalignment between the conventional IMU frame (A) and the C
     frame will be absorbed into T_a, the IMU accelerometer misalignment matrix

     w_m = T_g * w_B + T_s * a_B + b_w + n_w
     a_m = T_a * a_B + b_a + n_a = S * M * R_AB * a_B + b_a + n_a

     The conventional IMU frame has origin at the accelerometers intersection
     and x-axis aligned with accelerometer x.
 */
class HybridFilter : public VioBackendInterface {
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief The default constructor.
   */
  HybridFilter(const double readoutTime);

  /**
   * @brief Constructor if a ceres map is already available.
   * @param mapPtr Shared pointer to ceres map.
   */
  HybridFilter(std::shared_ptr<okvis::ceres::Map> mapPtr,
               const double readoutTime = 0.0);

  virtual ~HybridFilter();

  /// @name Sensor configuration related
  ///@{
  /**
   * @brief Add a camera to the configuration. Sensors can only be added and
   * never removed.
   * @param extrinsicsEstimationParameters The parameters that tell how to
   * estimate extrinsics.
   * @return Index of new camera.
   */
  int addCamera(const okvis::ExtrinsicsEstimationParameters
                    &extrinsicsEstimationParameters);

  /**
   * @brief Add an IMU to the configuration.
   * @warning Currently there is only one IMU supported.
   * @param imuParameters The IMU parameters.
   * @return index of IMU.
   */
  int addImu(const okvis::ImuParameters &imuParameters);

  /**
   * @brief Remove all cameras from the configuration
   */
  void clearCameras();

  /**
   * @brief Remove all IMUs from the configuration.
   */
  void clearImus();

  /// @}

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
                         bool asKeyframe);

  /**
   * @brief Prints state information to buffer.
   * @param poseId The pose Id for which to print.
   * @param buffer The puffer to print into.
   */
  void printStates(uint64_t poseId, std::ostream &buffer) const;

  /**
   * @brief Add a landmark.
   * @param landmarkId ID of the new landmark.
   * @param landmark Homogeneous coordinates of landmark in W-frame.
   * @return True if successful.
   */
  bool addLandmark(uint64_t landmarkId, const Eigen::Vector4d &landmark);

  /**
   * @brief Add an observation to a landmark.
   * \tparam GEOMETRY_TYPE The camera geometry type for this observation.
   * @param landmarkId ID of landmark.
   * @param poseId ID of pose where the landmark was observed.
   * @param camIdx ID of camera frame where the landmark was observed.
   * @param keypointIdx ID of keypoint corresponding to the landmark.
   * @return Residual block ID for that observation.
   */
  template <class GEOMETRY_TYPE>
  ::ceres::ResidualBlockId addObservation(uint64_t landmarkId, uint64_t poseId,
                                          size_t camIdx, size_t keypointIdx);
  /**
   * @brief Remove an observation from a landmark, if available.
   * @param landmarkId ID of landmark.
   * @param poseId ID of pose where the landmark was observed.
   * @param camIdx ID of camera frame where the landmark was observed.
   * @param keypointIdx ID of keypoint corresponding to the landmark.
   * @return True if observation was present and successfully removed.
   */
  bool removeObservation(uint64_t landmarkId, uint64_t poseId, size_t camIdx,
                         size_t keypointIdx);

  /**
   * @brief Applies the dropping/marginalization strategy according to the
   * RSS'13/IJRR'14 paper. The new number of frames in the window will be
   * numKeyframes+numImuFrames.
   * @return True if successful.
   */
  virtual bool applyMarginalizationStrategy();

  /**
   * @brief Initialise pose from IMU measurements. For convenience as static.
   * @param[in]  imuMeasurements The IMU measurements to be used for this.
   * @param[out] T_WS initialised pose.
   * @return True if successful.
   */
  static bool initPoseFromImu(const okvis::ImuMeasurementDeque &imuMeasurements,
                              okvis::kinematics::Transformation &T_WS);

  virtual void optimize(size_t numIter, size_t numThreads = 1,
                        bool verbose = false);

  /**
   * @brief Set a time limit for the optimization process.
   * @param[in] timeLimit Time limit in seconds. If timeLimit < 0 the time limit
   * is removed.
   * @param[in] minIterations minimum iterations the optimization process should
   * do disregarding the time limit.
   * @return True if successful.
   */
  bool setOptimizationTimeLimit(double timeLimit, int minIterations);

  /**
   * @brief Checks whether the landmark is added to the estimator.
   * @param landmarkId The ID.
   * @return True if added.
   */
  bool isLandmarkAdded(uint64_t landmarkId) const {
    bool isAdded = landmarksMap_.find(landmarkId) != landmarksMap_.end();
    OKVIS_ASSERT_TRUE_DBG(
        Exception, isAdded == mapPtr_->parameterBlockExists(landmarkId),
        "id=" << landmarkId << " inconsistent. isAdded = " << isAdded);
    return isAdded;
  }

  /**
   * @brief Checks whether the landmark is initialized.
   * @param landmarkId The ID.
   * @return True if initialised.
   */
  bool isLandmarkInitialized(uint64_t landmarkId) const;

  /// @name Getters
  ///\{
  /**
   * @brief Get a specific landmark.
   * @param[in]  landmarkId ID of desired landmark.
   * @param[out] mapPoint Landmark information, such as quality, coordinates
   * etc.
   * @return True if successful.
   */
  bool getLandmark(uint64_t landmarkId, okvis::MapPoint &mapPoint) const;

  /**
   * @brief Get a copy of all the landmarks as a PointMap.
   * @param[out] landmarks The landmarks.
   * @return number of landmarks.
   */
  size_t getLandmarks(okvis::PointMap &landmarks) const;

  /**
   * @brief Get a copy of all the landmark in a MapPointVector. This is for
   * legacy support. Use getLandmarks(okvis::PointMap&) if possible.
   * @param[out] landmarks A vector of all landmarks.
   * @see getLandmarks().
   * @return number of landmarks.
   */
  size_t getLandmarks(okvis::MapPointVector &landmarks) const;

  /**
   * @brief Get a multiframe.
   * @param frameId ID of desired multiframe.
   * @return Shared pointer to multiframe.
   */
  okvis::MultiFramePtr multiFrame(uint64_t frameId) const {
    OKVIS_ASSERT_TRUE_DBG(
        Exception, multiFramePtrMap_.find(frameId) != multiFramePtrMap_.end(),
        "Requested multi-frame does not exist in estimator.");
    return multiFramePtrMap_.at(frameId);
  }

  /**
   * @brief Get pose for a given pose ID.
   * @param[in]  poseId ID of desired pose.
   * @param[out] T_WS Homogeneous transformation of this pose.
   * @return True if successful.
   */
  bool get_T_WS(uint64_t poseId, okvis::kinematics::Transformation &T_WS) const;

  // the following access the optimization graph, so are not very fast.
  // Feel free to implement caching for them...
  /**
   * @brief Get speeds and IMU biases for a given pose ID.
   * @warning This accesses the optimization graph, so not very fast.
   * @param[in]  poseId ID of pose to get speeds and biases for.
   * @param[in]  imuIdx index of IMU to get biases for. As only one IMU is
   * supported this is always 0.
   * @param[out] speedAndBias Speed And bias requested.
   * @return True if successful.
   */
  bool getSpeedAndBias(uint64_t poseId, uint64_t imuIdx,
                       okvis::SpeedAndBiases &speedAndBias) const;

  /**
   * @brief Get camera states for a given pose ID.
   * @warning This accesses the optimization graph, so not very fast.
   * @param[in]  poseId ID of pose to get camera state for.
   * @param[in]  cameraIdx index of camera to get state for.
   * @param[out] T_SCi Homogeneous transformation from sensor (IMU) frame to
   * camera frame.
   * @return True if successful.
   */
  bool getCameraSensorStates(uint64_t poseId, size_t cameraIdx,
                             okvis::kinematics::Transformation &T_SCi) const;

  /// @brief Get the number of states/frames in the estimator.
  /// \return The number of frames.
  size_t numFrames() const { return statesMap_.size(); }

  /// @brief Get the number of landmarks in the estimator
  /// \return The number of landmarks.
  size_t numLandmarks() const { return landmarksMap_.size(); }

  /// @brief Get the ID of the current keyframe.
  /// \return The ID of the current keyframe.
  uint64_t currentKeyframeId() const;

  /**
   * @brief Get the ID of an older frame.
   * @param[in] age age of desired frame. 0 would be the newest frame added to
   * the state.
   * @return ID of the desired frame or 0 if parameter age was out of range.
   */
  uint64_t frameIdByAge(size_t age) const;

  /// @brief Get the ID of the newest frame added to the state.
  /// \return The ID of the current frame.
  uint64_t currentFrameId() const;

  ///@}

  /**
   * @brief Checks if a particular frame is a keyframe.
   * @param[in] frameId ID of frame to check.
   * @return True if the frame is a keyframe.
   */
  bool isKeyframe(uint64_t frameId) const {
    return statesMap_.at(frameId).isKeyframe;
  }

  /**
   * @brief Checks if a particular frame is still in the IMU window.
   * @param[in] frameId ID of frame to check.
   * @return True if the frame is in IMU window.
   */
  bool isInImuWindow(uint64_t frameId) const;

  /// @name Getters
  /// @{
  /**
   * @brief Get the timestamp for a particular frame.
   * @param[in] frameId ID of frame.
   * @return Timestamp of frame.
   */
  okvis::Time timestamp(uint64_t frameId) const {
    return statesMap_.at(frameId).timestamp;
  }

  ///@}
  /// @name Setters
  ///@{
  /**
   * @brief Set pose for a given pose ID.
   * @warning This accesses the optimization graph, so not very fast.
   * @param[in] poseId ID of the pose that should be changed.
   * @param[in] T_WS new homogeneous transformation.
   * @return True if successful.
   */
  bool set_T_WS(uint64_t poseId, const okvis::kinematics::Transformation &T_WS);

  /**
   * @brief Set the speeds and IMU biases for a given pose ID.
   * @warning This accesses the optimization graph, so not very fast.
   * @param[in] poseId ID of the pose to change corresponding speeds and biases
   * for.
   * @param[in] imuIdx index of IMU to get biases for. As only one IMU is
   * supported this is always 0.
   * @param[in] speedAndBias new speeds and biases.
   * @return True if successful.
   */
  bool setSpeedAndBias(uint64_t poseId, size_t imuIdx,
                       const okvis::SpeedAndBiases &speedAndBias);

  /**
   * @brief Set the transformation from sensor to camera frame for a given pose
   * ID.
   * @warning This accesses the optimization graph, so not very fast.
   * @param[in] poseId ID of the pose to change corresponding camera states for.
   * @param[in] cameraIdx Index of camera to set state for.
   * @param[in] T_SCi new homogeneous transformation from sensor (IMU) to camera
   * frame.
   * @return True if successful.
   */
  bool setCameraSensorStates(uint64_t poseId, size_t cameraIdx,
                             const okvis::kinematics::Transformation &T_SCi);

  /// @brief Set the homogeneous coordinates for a landmark.
  /// @param[in] landmarkId The landmark ID.
  /// @param[in] landmark Homogeneous coordinates of landmark in W-frame.
  /// @return True if successful.
  bool setLandmark(uint64_t landmarkId, const Eigen::Vector4d &landmark);

  /// @brief Set the landmark initialization state.
  /// @param[in] landmarkId The landmark ID.
  /// @param[in] initialized Whether or not initialised.
  void setLandmarkInitialized(uint64_t landmarkId, bool initialized);

  /// @brief Set whether a frame is a keyframe or not.
  /// @param[in] frameId The frame ID.
  /// @param[in] isKeyframe Whether or not keyrame.
  void setKeyframe(uint64_t frameId, bool isKeyframe) {
    statesMap_.at(frameId).isKeyframe = isKeyframe;
  }

  /// @brief set ceres map
  /// @param[in] mapPtr The pointer to the okvis::ceres::Map.
  void setMap(std::shared_ptr<okvis::ceres::Map> mapPtr) { mapPtr_ = mapPtr; }
  ///@}

 private:
 public:  // huai
  /**
   * @brief Remove an observation from a landmark.
   * @param residualBlockId Residual ID for this landmark.
   * @return True if successful.
   */
  bool removeObservation(::ceres::ResidualBlockId residualBlockId);

  /// \brief StateInfo This configures the state vector ordering
  struct StateInfo {
    /// \brief Constructor
    /// @param[in] id The Id.
    /// @param[in] isRequired Whether or not we require the state.
    /// @param[in] exists Whether or not this exists in the ceres problem.
    StateInfo(uint64_t id = 0, bool isRequired = true, bool exists = false)
        : id(id), isRequired(isRequired), exists(exists) {}
    uint64_t id;       ///< The ID.
    bool isRequired;   ///< Whether or not we require the state.
    bool exists;       ///< Whether or not this exists in the ceres problem.
    uint64_t idInCov;  ///< start id of the state within the covariance matrix
    int minimalDim;    ///< minimal dimension of this state, which can acctually
                       ///< replace member "exists"
  };

  /// \brief GlobalStates The global states enumerated
  enum GlobalStates {
    T_WS = 0,           ///< Pose.
    MagneticZBias = 1,  ///< Magnetometer z-bias, currently unused
    Qff = 2,            ///< QFF (pressure at sea level), currently unused
    T_GW = 3            ///< Alignment of global frame, currently unused
  };

  /// \brief SensorStates The sensor-internal states enumerated
  enum SensorStates {
    Camera = 0,      ///< Camera
    Imu,             ///< IMU
    Position,        ///< Position, currently unused
    Gps,             ///< GPS, currently unused
    Magnetometer,    ///< Magnetometer, currently unused
    StaticPressure,  ///< Static pressure, currently unused
    DynamicPressure  ///< Dynamic pressure, currently unused
  };

  /// \brief CameraSensorStates The camera-internal states enumerated
  enum CameraSensorStates {
    T_SCi = 0,   ///< Extrinsics as T_SC, in MSCKF, only p_SC is changing, R_SC
                 ///< is contant
    Intrinsic,   ///< Intrinsics, for pinhole camera, fx ,fy, cx, cy
    Distortion,  ///< Distortion coefficients, for radial tangential distoriton
                 ///< of pinhole cameras, k1, k2, p1, p2, [k3], this ordering is
                 ///< OpenCV style
    TD,          ///< time delay of the image timestamp with respect to the IMU
                 ///< timescale, Raw t_Ci + t_d = t_Ci in IMU time,
    TR  ///< t_r is the read out time of a whole frames of a rolling shutter
        ///< camera
  };

  /// \brief ImuSensorStates The IMU-internal states enumerated
  /// \warning This is slightly inconsistent, since the velocity should be
  /// global.
  enum ImuSensorStates {
    SpeedAndBias =
        0,  ///< Speed and biases as v in /*S*/W-frame, then b_g and b_a
    TG,     ///< T_g, T_s, T_a in row major order as defined in Li icra14 high
            ///< fedeltiy
    TS,
    TA
  };

  /// \brief PositionSensorStates, currently unused
  enum PositionSensorStates {
    T_PiW = 0,  ///< position sensor frame to world, currently unused
    PositionSensorB_t_BA = 1  ///< antenna offset, currently unused
  };

  /// \brief GpsSensorStates, currently unused
  enum GpsSensorStates {
    GpsB_t_BA = 0  ///< antenna offset, currently unused
  };

  /// \brief MagnetometerSensorStates, currently unused
  enum MagnetometerSensorStates {
    MagnetometerBias = 0  ///< currently unused
  };

  /// \brief GpsSensorStates, currently unused
  enum StaticPressureSensorStates {
    StaticPressureBias = 0  ///< currently unused
  };

  /// \brief GpsSensorStates, currently unused
  enum DynamicPressureSensorStates {
    DynamicPressureBias = 0  ///< currently unused
  };

  // getters
  bool getGlobalStateParameterBlockPtr(
      uint64_t poseId, int stateType,
      std::shared_ptr<ceres::ParameterBlock> &stateParameterBlockPtr) const;
  template <class PARAMETER_BLOCK_T>
  bool getGlobalStateParameterBlockAs(
      uint64_t poseId, int stateType,
      PARAMETER_BLOCK_T &stateParameterBlock) const;
  template <class PARAMETER_BLOCK_T>
  bool getGlobalStateEstimateAs(
      uint64_t poseId, int stateType,
      typename PARAMETER_BLOCK_T::estimate_t &state) const;

  bool getSensorStateParameterBlockPtr(
      uint64_t poseId, int sensorIdx, int sensorType, int stateType,
      std::shared_ptr<ceres::ParameterBlock> &stateParameterBlockPtr) const;
  template <class PARAMETER_BLOCK_T>
  bool getSensorStateParameterBlockAs(
      uint64_t poseId, int sensorIdx, int sensorType, int stateType,
      PARAMETER_BLOCK_T &stateParameterBlock) const;
  template <class PARAMETER_BLOCK_T>
  bool getSensorStateEstimateAs(
      uint64_t poseId, int sensorIdx, int sensorType, int stateType,
      typename PARAMETER_BLOCK_T::estimate_t &state) const;

  // setters
  template <class PARAMETER_BLOCK_T>
  bool setGlobalStateEstimateAs(
      uint64_t poseId, int stateType,
      const typename PARAMETER_BLOCK_T::estimate_t &state);

  template <class PARAMETER_BLOCK_T>
  bool setSensorStateEstimateAs(
      uint64_t poseId, int sensorIdx, int sensorType, int stateType,
      const typename PARAMETER_BLOCK_T::estimate_t &state) {
    // check existence in states set
    if (statesMap_.find(poseId) == statesMap_.end()) {
      OKVIS_THROW_DBG(Exception,
                      "pose with id = " << poseId << " does not exist.")
      return false;
    }

    // obtain the parameter block ID
    uint64_t id = statesMap_.at(poseId)
                      .sensors.at(sensorType)
                      .at(sensorIdx)
                      .at(stateType)
                      .id;
    if (!mapPtr_->parameterBlockExists(id)) {
      OKVIS_THROW_DBG(Exception,
                      "pose with id = " << poseId << " does not exist.")
      return false;
    }

    std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr =
        mapPtr_->parameterBlockPtr(id);
#ifndef NDEBUG
    std::shared_ptr<PARAMETER_BLOCK_T> derivedParameterBlockPtr =
        std::dynamic_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
    if (!derivedParameterBlockPtr) {
      OKVIS_THROW_DBG(Exception, "wrong pointer type requested.")
      return false;
    }
    derivedParameterBlockPtr->setEstimate(state);
#else
    std::static_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr)
        ->setEstimate(state);
#endif
    return true;
  }

  // the following are just fixed-size containers for related parameterBlockIds:
  typedef std::array<StateInfo, 6>
      GlobalStatesContainer;  ///< Container for global states.
  typedef std::vector<StateInfo>
      SpecificSensorStatesContainer;  ///< Container for sensor states. The
                                      ///< dimension can vary from sensor to
                                      ///< sensor...
  typedef std::array<std::vector<SpecificSensorStatesContainer>, 7>
      AllSensorStatesContainer;  ///< Union of all sensor states.

  /// \brief States This summarizes all the possible states -- i.e. their ids:
  /// t_j = t_{j_0} - imageDelay + t_{d_j}
  /// here t_{j_0} is the raw timestamp of image j,
  /// t_{d_j} is the current estimated time offset between the visual and
  /// inertial data, after correcting the initial time offset
  /// imageDelay.Therefore, t_{d_j} is set 0 at the beginning t_j is the
  /// timestamp of the state, remains constant after initialization t_{f_i} =
  /// t_j - t_{d_j} + t_d + (v-N/2)t_r/N here t_d and t_r are the true time
  /// offset and image readout time t_{f_i} is the time feature i is observed
  struct States {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    States() : isKeyframe(false), id(0), tdAtCreation(0) {}
    // _timestamp = image timestamp - imageDelay + _tdAtCreation
    States(bool isKeyframe, uint64_t id, okvis::Time _timestamp,
           okvis::Duration _tdAtCreation = okvis::Duration())
        : isKeyframe(isKeyframe),
          id(id),
          timestamp(_timestamp),
          tdAtCreation(_tdAtCreation) {}
    GlobalStatesContainer global;
    AllSensorStatesContainer sensors;
    bool isKeyframe;
    uint64_t id;
    const okvis::Time timestamp;         // t_j, fixed once initialized
    const okvis::Duration tdAtCreation;  // t_{d_j}, fixed once initialized
    Eigen::Matrix<double, 6, 1>
        linearizationPoint;  /// first estimate of position r_WB and velocity
                             /// v_WB
  };

  // the following keeps track of all the states at different time instances
  // (key=poseId)
  std::map<uint64_t, States, std::less<uint64_t>,
           Eigen::aligned_allocator<std::pair<const uint64_t, States>>>
      statesMap_;  ///< Buffer for currently considered states.
  std::map<uint64_t, okvis::MultiFramePtr>
      multiFramePtrMap_;  ///< remember all needed okvis::MultiFrame.
  std::shared_ptr<okvis::ceres::Map> mapPtr_;  ///< The underlying okvis::Map.

  // this is the reference pose
  uint64_t referencePoseId_;  ///< The pose ID of the reference (currently not
                              ///< changing)

  // the following are updated after the optimization
  okvis::PointMap
      landmarksMap_;  ///< Contains all the current landmarks (synched after
                      ///< optimisation). maps landmarkId(i.e., homogeneous
                      ///< point parameter block id) to MapPoint pointer
  mutable std::mutex statesMutex_;  ///< Regulate access of landmarksMap_.

  // parameters
  std::vector<okvis::ExtrinsicsEstimationParameters,
              Eigen::aligned_allocator<okvis::ExtrinsicsEstimationParameters>>
      extrinsicsEstimationParametersVec_;  ///< Extrinsics parameters.
  std::vector<okvis::ImuParameters,
              Eigen::aligned_allocator<okvis::ImuParameters>>
      imuParametersVec_;  ///< IMU parameters.

  // loss function for reprojection errors
  std::shared_ptr<::ceres::LossFunction>
      cauchyLossFunctionPtr_;  ///< Cauchy loss.
  std::shared_ptr<::ceres::LossFunction>
      huberLossFunctionPtr_;  ///< Huber loss.

  // the marginalized error term
  std::shared_ptr<ceres::MarginalizationError>
      marginalizationErrorPtr_;  ///< The marginalisation class
  ::ceres::ResidualBlockId
      marginalizationResidualId_;  ///< Remembers the marginalisation object's
                                   ///< Id

  // ceres iteration callback object
  std::unique_ptr<okvis::ceres::CeresIterationCallback>
      ceresCallback_;  ///< Maybe there was a callback registered, store it
                       ///< here.

 protected:
  // set intermediate variables which are used for computing Jacobians of
  // feature point observations
  virtual void retrieveEstimatesOfConstants();
  virtual void updateStates(
      const Eigen::Matrix<double, Eigen::Dynamic, 1> &deltaX);

public:
  okvis::Time firstStateTimestamp();

  void gatherPoseObservForTriang(
      const MapPoint &mp,
      const cameras::PinholeCamera<cameras::RadialTangentialDistortion>
          &cameraGeometry,
      std::vector<uint64_t> *frameIds,
      std::vector<okvis::kinematics::Transformation> *T_WSs,
      std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
          *obsDirections,
      std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
          *obsInPixel,
      std::vector<double> *vR_oi, const uint64_t &hpbid) const;

  /**
   * @brief triangulateAMapPoint, does not support rays which arise from static
   * mode, pure rotation, or points at infinity. Assume the same camera model
   * for all observations, and rolling shutter effect is not accounted for
   * @param mp
   * @param obsInPixel
   * @param frameIds, id of frames observing this feature in the ascending order
   * because the MapPoint.observations is an ordinary ordered map
   * @param v4Xhomog, stores either [X,Y,Z,1] in the global frame
   * @param vR_oi, the diagonal elements of the observation noise matrix, in
   * pixels, size 2Nx1
   * @param cameraGeometry, used for point projection
   * @param T_SC0
   * @param hpbid
   * @return v4Xhomog
   */
  bool triangulateAMapPoint(
      const MapPoint &mp,
      std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
          &obsInPixel,
      std::vector<uint64_t> &frameIds, Eigen::Vector4d &v4Xhomog,
      std::vector<double> &vR_oi,
      const okvis::cameras::PinholeCamera<
          okvis::cameras::RadialTangentialDistortion> &cameraGeometry,
      const okvis::kinematics::Transformation &T_SC0, const uint64_t &hpbid,
      bool use_AIDP = false) const;
  /**
   * @brief computeHxf, compute the residual and Jacobians for a SLAM feature i
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
  bool computeHxf(const uint64_t hpbid, const MapPoint &mp,
                  Eigen::Matrix<double, 2, 1> &r_i,
                  Eigen::Matrix<double, 2, Eigen::Dynamic> &H_x,
                  Eigen::Matrix<double, 2, Eigen::Dynamic> &H_f,
                  Eigen::Matrix2d &R_i);

  /**
   * @brief computeHoi, compute the marginalized Jacobian for a feature i's
   track
   * assume the number of observations of the map points is at least two
   * @param hpbid homogeneous point parameter block id of the map point
   * @param mp mappoint
   * @param r_oi residuals
   * @param H_oi Jacobians of feature observations w.r.t variables related to
   camera intrinsics, camera poses (13+9(m-1)-3)
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
  bool computeHoi(const uint64_t hpbid, const MapPoint &mp,
                  Eigen::Matrix<double, Eigen::Dynamic, 1> &r_oi,
                  Eigen::MatrixXd &H_oi, Eigen::MatrixXd &R_oi,
                  Eigen::Vector4d &ab1rho,
                  Eigen::Matrix<double, Eigen::Dynamic, 3> *pH_fi =
                      (Eigen::Matrix<double, Eigen::Dynamic, 3> *)(NULL)) const;

  /// OBSOLETE: check states by comparing current estimates with the ground
  /// truth. To use this function, make sure the ground truth is linked
  /// correctly This function can be called in the optimizationLoop
  void checkStates();

  /// print out the most recent state vector and the stds of its elements. This
  /// function can be called in the optimizationLoop, but a better way to save
  /// results is use the publisher loop
  bool print(const std::string) const;

  bool print(std::ostream &mDebug) const;

  void printTrackLengthHistogram(std::ostream &mDebug) const;

  /// reset initial condition
  inline void resetInitialPVandStd(const InitialPVandStd &rhs,
                                   bool bUseExternalPose = false) {
    pvstd_ = rhs;
    mbUseExternalInitialPose = bUseExternalPose;
  }

  size_t numObservations(uint64_t landmarkId);
  /**
   * @brief getCameraCalibrationEstimate, get the latest estimate of camera
   * calibration parameters
   * @param vfckptdr
   * @return the last pose id
   */
  uint64_t getCameraCalibrationEstimate(Eigen::Matrix<double, 10, 1> &vfckptdr);
  /**
   * @brief getTgTsTaEstimate, get the lastest estimate of Tg Ts Ta with entries
   * in row major order
   * @param vTGTSTA
   * @return the last pose id
   */
  uint64_t getTgTsTaEstimate(Eigen::Matrix<double, 27, 1> &vTGTSTA);

  /**
   * @brief get variance for evolving states(p_GB, q_GB, v_GB, bg, ba), Tg Ts
   * Ta, p_CB, fxy, cxy, k1, k2, p1, p2, td, tr
   * @param variances
   */
  void getVariance(Eigen::Matrix<double, 55, 1> &variances);

  bool getFrameId(uint64_t poseId, int &frameIdInSource, bool &isKF) const;

  size_t covDim_;  ///< rows(cols) of covariance, dynamically changes
  Eigen::MatrixXd
      covariance_;  ///< covariance of the error vector of all states, error is
                    ///< defined as \tilde{x} = x - \hat{x} except for rotations
  /// the error vector corresponds to states x_B | x_imu | x_c | \pi{B_{N-m}}
  /// ... \pi{B_{N-1}} following Li icra 2014 x_B = [^{G}p_B] ^{G}q_B ^{G}v_B
  /// b_g b_a]
  const double imageReadoutTime;  // time to read out one image of the rolling
                                  // shutter camera

  // map from state ID to segments of imu measurements, the imu measurements
  // covers the last state and current state of the id and extends on both sides
  std::map<uint64_t, okvis::ImuMeasurementDeque, std::less<uint64_t>,
           Eigen::aligned_allocator<
               std::pair<const uint64_t, okvis::ImuMeasurementDeque>>>
      mStateID2Imu;

  // intermediate variables used for computeHoi and computeHxf, refresh them
  // with retrieveEstimatesOfConstants before calling it
  size_t numCamPosePointStates_;  // the variables in the states involved in
                                  // compute Jacobians for a feature, including
                                  // the camera intrinsics, all cloned states,
                                  // and all feature states

  // intermediate variables used for computeHoi, refresh them with
  // retrieveEstimatesOfConstants before calling it
  size_t nVariableDim_;  // local dimension of variables used in computing
                         // feature Jacobians, including the camera intrinsics,
                         // all cloned states except the most recent one

  std::map<uint64_t, int>
      mStateID2CovID_;  // maps state id to the ordered cloned states in the
                        // covariance matrix

  // transformation from the camera frame to the sensor frame
  kinematics::Transformation T_SC0_;

  double tdLatestEstimate;
  double trLatestEstimate;

  // an evolving camera rig to temporarily store the optimized camera
  // raw parameters and to interface with the camera models
  okvis::cameras::CameraRig camera_rig_;

  Eigen::Matrix<double, 27, 1> vTGTSTA_;
  IMUErrorModel<double> iem_;

  // intermediate variable, refreshed in optimize();
  enum ResidualizeCase {
    NotInState_NotTrackedNow = 0,
    NotToAdd_TrackedNow,
    ToAdd_TrackedNow,
    InState_NotTrackedNow,
    InState_TrackedNow
  };
  std::vector<std::pair<size_t, ResidualizeCase>>
      mLandmarkID2Residualize;  // each landmark's case of residualize,
  // 0 means a point not in the states is not tracked in current frame,
  // 1 means a point not in states is tracked in current frame but not to be
  // included in states, 2 a point not in states is tracked in current frame and
  // to be included in states 3 a point in the states is not tracked in current
  // frame, 4 a points in states is tracked in current frame MSCKF only handles
  // the first two cases

  uint64_t minValidStateID;  // the minimum of the ids of the states that have
                             // tracked features

  mutable okvis::timing::Timer triangulateTimer;
  mutable okvis::timing::Timer computeHTimer;
  okvis::timing::Timer computeKalmanGainTimer;
  okvis::timing::Timer updateStatesTimer;
  okvis::timing::Timer updateCovarianceTimer;
  okvis::timing::Timer updateLandmarksTimer;

  std::deque<uint64_t>
      mInCovLmIds;  // for each point in the state vector/covariance,
  // its landmark id which points to the parameter block,

  size_t mM;  // number of \delta(pos, \theta, vel) states in the whole state
              // vector
  InitialPVandStd pvstd_;
  bool mbUseExternalInitialPose;  // do we use external pose for initialization

  // maximum number of consecutive observations until a landmark is added as a
  // state, but can be set dynamically as done in Li, icra14 optimization based
  // ...
  static const size_t mMaxM = 12;  // specific to HybridFilter

  std::vector<size_t>
      mTrackLengthAccumulator;  // histogram of the track lengths, start from
                                // 0,1,2, to a fixed number
};

const double chi2_95percentile[] = {
    0,   0,  // for easy reference
    1,   3.841458821, 2,   5.991464547, 3,   7.814727903, 4,   9.487729037,
    5,   11.07049769, 6,   12.59158724, 7,   14.06714045, 8,   15.50731306,
    9,   16.9189776,  10,  18.30703805, 11,  19.67513757, 12,  21.02606982,
    13,  22.36203249, 14,  23.6847913,  15,  24.99579014, 16,  26.2962276,
    17,  27.58711164, 18,  28.86929943, 19,  30.14352721, 20,  31.41043284,
    21,  32.67057334, 22,  33.92443847, 23,  35.17246163, 24,  36.4150285,
    25,  37.65248413, 26,  38.88513866, 27,  40.11327207, 28,  41.33713815,
    29,  42.5569678,  30,  43.77297183, 31,  44.98534328, 32,  46.19425952,
    33,  47.39988392, 34,  48.60236737, 35,  49.80184957, 36,  50.99846017,
    37,  52.19231973, 38,  53.38354062, 39,  54.57222776, 40,  55.75847928,
    41,  56.94238715, 42,  58.12403768, 43,  59.30351203, 44,  60.48088658,
    45,  61.65623338, 46,  62.82962041, 47,  64.00111197, 48,  65.1707689,
    49,  66.33864886, 50,  67.50480655, 51,  68.66929391, 52,  69.83216034,
    53,  70.99345283, 54,  72.15321617, 55,  73.31149303, 56,  74.46832416,
    57,  75.62374847, 58,  76.77780316, 59,  77.93052381, 60,  79.08194449,
    61,  80.23209785, 62,  81.38101519, 63,  82.52872654, 64,  83.67526074,
    65,  84.8206455,  66,  85.96490744, 67,  87.1080722,  68,  88.25016442,
    69,  89.39120787, 70,  90.53122543, 71,  91.67023918, 72,  92.80827038,
    73,  93.9453396,  74,  95.08146667, 75,  96.21667075, 76,  97.35097038,
    77,  98.48438346, 78,  99.61692732, 79,  100.7486187, 80,  101.879474,
    81,  103.0095087, 82,  104.1387382, 83,  105.2671773, 84,  106.3948402,
    85,  107.521741,  86,  108.647893,  87,  109.7733094, 88,  110.8980028,
    89,  112.0219857, 90,  113.1452701, 91,  114.2678677, 92,  115.3897897,
    93,  116.5110473, 94,  117.6316511, 95,  118.7516118, 96,  119.8709393,
    97,  120.9896437, 98,  122.1077346, 99,  123.2252215, 100, 124.3421134,
    101, 125.4584194, 102, 126.5741482, 103, 127.6893083, 104, 128.8039079,
    105, 129.9179553, 106, 131.0314583, 107, 132.1444245, 108, 133.2568617,
    109, 134.3687771, 110, 135.4801779, 111, 136.5910712, 112, 137.7014639,
    113, 138.8113626, 114, 139.9207739, 115, 141.0297043, 116, 142.13816,
    117, 143.2461473, 118, 144.353672,  119, 145.4607402, 120, 146.5673576,
    121, 147.6735298, 122, 148.7792623, 123, 149.8845606, 124, 150.98943,
    125, 152.0938757, 126, 153.1979027, 127, 154.3015162, 128, 155.4047209,
    129, 156.5075216, 130, 157.6099231, 131, 158.71193,   132, 159.8135469,
    133, 160.914778,  134, 162.0156279, 135, 163.1161008, 136, 164.2162009,
    137, 165.3159322, 138, 166.4152989, 139, 167.514305,  140, 168.6129543,
    141, 169.7112506, 142, 170.8091977, 143, 171.9067993, 144, 173.0040591,
    145, 174.1009806, 146, 175.1975673, 147, 176.2938226, 148, 177.38975,
    149, 178.4853527, 150, 179.5806342, 151, 180.6755974, 152, 181.7702457,
    153, 182.8645822, 154, 183.9586098, 155, 185.0523317, 156, 186.1457508,
    157, 187.2388699, 158, 188.3316921, 159, 189.42422,   160, 190.5164565,
    161, 191.6084043, 162, 192.7000662, 163, 193.7914446, 164, 194.8825424,
    165, 195.973362,  166, 197.0639059, 167, 198.1541768, 168, 199.2441769,
    169, 200.3339088, 170, 201.4233749, 171, 202.5125774, 172, 203.6015187,
    173, 204.6902011, 174, 205.7786268, 175, 206.866798,  176, 207.954717,
    177, 209.0423859, 178, 210.1298067, 179, 211.2169816, 180, 212.3039127,
    181, 213.390602,  182, 214.4770515, 183, 215.5632632, 184, 216.649239,
    185, 217.7349809, 186, 218.8204907, 187, 219.9057703, 188, 220.9908216,
    189, 222.0756464, 190, 223.1602465, 191, 224.2446237, 192, 225.3287798,
    193, 226.4127164, 194, 227.4964352, 195, 228.579938,  196, 229.6632264,
    197, 230.7463021, 198, 231.8291667, 199, 232.9118218, 200, 233.9942689,
    201, 235.0765096, 202, 236.1585456, 203, 237.2403782, 204, 238.322009,
    205, 239.4034395, 206, 240.4846711, 207, 241.5657054, 208, 242.6465436,
    209, 243.7271874, 210, 244.8076379, 211, 245.8878967, 212, 246.9679651,
    213, 248.0478444, 214, 249.127536,  215, 250.2070412, 216, 251.2863613,
    217, 252.3654975, 218, 253.4444512, 219, 254.5232236, 220, 255.601816,
    221, 256.6802295, 222, 257.7584655, 223, 258.836525,  224, 259.9144094,
    225, 260.9921196, 226, 262.069657,  227, 263.1470227, 228, 264.2242178,
    229, 265.3012434, 230, 266.3781007, 231, 267.4547907, 232, 268.5313145,
    233, 269.6076732, 234, 270.6838679};

/**
 * @brief Does a vector contain a certain element.
 * @tparam Class of a vector element.
 * @param vector Vector to search element in.
 * @param query Element to search for.
 * @return True if query is an element of vector.
 */
template <class T>
bool vectorContains(const std::vector<T> &vector, const T &query) {
  for (size_t i = 0; i < vector.size(); ++i) {
    if (vector[i] == query) {
      return true;
    }
  }
  return false;
}

}  // namespace okvis

#include "implementation/HybridFilter.hpp"

#endif /* INCLUDE_OKVIS_HYBRID_FILTER_HPP_ */
