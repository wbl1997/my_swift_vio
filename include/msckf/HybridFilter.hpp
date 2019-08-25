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
#include <msckf/CameraRig.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>

#include <okvis/timing/Timer.hpp>

#include <vio/CsvReader.h>
#include <vio/ImuErrorModel.h>

#include "msckf/InitialPVandStd.hpp"
#include "msckf/BoundedImuDeque.hpp"

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
  virtual bool applyMarginalizationStrategy(
      size_t numKeyframes, size_t numImuFrames,
      okvis::MapPointVector& removedLandmarks);

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
  okvis::BoundedImuDeque mStateID2Imu;

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

  // The window centered at a stateEpoch for retrieving the inertial data
  // which is used for propagating the camera pose to epochs in the window,
  // i.e., timestamps of observations in a rolling shutter image.
  // A value greater than (t_d + t_r)/2 is recommended.
  // Note camera observations in MSCKF will not occur at the latest frame.
  static const okvis::Duration half_window_;
};

/**
 * Chi-square thresholds based on the DOF of state (chi2(0.95,DOF))
 * degrees from 0, 1, 2, ...
 */
const double chi2_95percentile[] = {
      0,  // for easy reference at degree 0
      3.841459,   5.991465,   7.814728,   9.487729,  11.070498,  12.591587,  14.067140,  15.507313,  16.918978,  18.307038,
     19.675138,  21.026070,  22.362032,  23.684791,  24.995790,  26.296228,  27.587112,  28.869299,  30.143527,  31.410433,
     32.670573,  33.924438,  35.172462,  36.415029,  37.652484,  38.885139,  40.113272,  41.337138,  42.556968,  43.772972,
     44.985343,  46.194260,  47.399884,  48.602367,  49.801850,  50.998460,  52.192320,  53.383541,  54.572228,  55.758479,
     56.942387,  58.124038,  59.303512,  60.480887,  61.656233,  62.829620,  64.001112,  65.170769,  66.338649,  67.504807,
     68.669294,  69.832160,  70.993453,  72.153216,  73.311493,  74.468324,  75.623748,  76.777803,  77.930524,  79.081944,
     80.232098,  81.381015,  82.528727,  83.675261,  84.820645,  85.964907,  87.108072,  88.250164,  89.391208,  90.531225,
     91.670239,  92.808270,  93.945340,  95.081467,  96.216671,  97.350970,  98.484383,  99.616927, 100.748619, 101.879474,
    103.009509, 104.138738, 105.267177, 106.394840, 107.521741, 108.647893, 109.773309, 110.898003, 112.021986, 113.145270,
    114.267868, 115.389790, 116.511047, 117.631651, 118.751612, 119.870939, 120.989644, 122.107735, 123.225221, 124.342113,
    125.458419, 126.574148, 127.689308, 128.803908, 129.917955, 131.031458, 132.144425, 133.256862, 134.368777, 135.480178,
    136.591071, 137.701464, 138.811363, 139.920774, 141.029704, 142.138160, 143.246147, 144.353672, 145.460740, 146.567358,
    147.673530, 148.779262, 149.884561, 150.989430, 152.093876, 153.197903, 154.301516, 155.404721, 156.507522, 157.609923,
    158.711930, 159.813547, 160.914778, 162.015628, 163.116101, 164.216201, 165.315932, 166.415299, 167.514305, 168.612954,
    169.711251, 170.809198, 171.906799, 173.004059, 174.100981, 175.197567, 176.293823, 177.389750, 178.485353, 179.580634,
    180.675597, 181.770246, 182.864582, 183.958610, 185.052332, 186.145751, 187.238870, 188.331692, 189.424220, 190.516457,
    191.608404, 192.700066, 193.791445, 194.882542, 195.973362, 197.063906, 198.154177, 199.244177, 200.333909, 201.423375,
    202.512577, 203.601519, 204.690201, 205.778627, 206.866798, 207.954717, 209.042386, 210.129807, 211.216982, 212.303913,
    213.390602, 214.477052, 215.563263, 216.649239, 217.734981, 218.820491, 219.905770, 220.990822, 222.075646, 223.160247,
    224.244624, 225.328780, 226.412716, 227.496435, 228.579938, 229.663226, 230.746302, 231.829167, 232.911822, 233.994269,
    235.076510, 236.158546, 237.240378, 238.322009, 239.403439, 240.484671, 241.565705, 242.646544, 243.727187, 244.807638,
    245.887897, 246.967965, 248.047844, 249.127536, 250.207041, 251.286361, 252.365498, 253.444451, 254.523224, 255.601816,
    256.680230, 257.758465, 258.836525, 259.914409, 260.992120, 262.069657, 263.147023, 264.224218, 265.301243, 266.378101,
    267.454791, 268.531314, 269.607673, 270.683868, 271.759900, 272.835769, 273.911478, 274.987027, 276.062417, 277.137650,
    278.212725, 279.287644, 280.362409, 281.437019, 282.511477, 283.585782, 284.659936, 285.733940, 286.807794, 287.881501,
    288.955059, 290.028471, 291.101737, 292.174858, 293.247835, 294.320669, 295.393360, 296.465910, 297.538319, 298.610588,
    299.682719, 300.754710, 301.826565, 302.898282, 303.969864, 305.041310, 306.112622, 307.183800, 308.254846, 309.325759,
    310.396541, 311.467192, 312.537713, 313.608105, 314.678368, 315.748503, 316.818512, 317.888393, 318.958149, 320.027780,
    321.097286, 322.166669, 323.235928, 324.305065, 325.374080, 326.442974, 327.511748, 328.580401, 329.648936, 330.717351,
    331.785649, 332.853829, 333.921892, 334.989839, 336.057670, 337.125386, 338.192988, 339.260476, 340.327850, 341.395112,
    342.462262, 343.529300, 344.596226, 345.663043, 346.729749, 347.796346, 348.862834, 349.929214, 350.995485, 352.061650,
    353.127708, 354.193659, 355.259504, 356.325245, 357.390880, 358.456412, 359.521839, 360.587163, 361.652385, 362.717504,
    363.782521, 364.847437, 365.912253, 366.976967, 368.041582, 369.106097, 370.170513, 371.234831, 372.299051, 373.363173,
    374.427197, 375.491125, 376.554957, 377.618692, 378.682332, 379.745878, 380.809328, 381.872684, 382.935947, 383.999116,
    385.062192, 386.125175, 387.188067, 388.250867, 389.313575, 390.376192, 391.438719, 392.501156, 393.563503, 394.625760,
    395.687929, 396.750009, 397.812000, 398.873904, 399.935720, 400.997450, 402.059092, 403.120648, 404.182118, 405.243502,
    406.304801, 407.366015, 408.427145, 409.488190, 410.549151, 411.610029, 412.670823, 413.731535, 414.792164, 415.852711,
    416.913176, 417.973559, 419.033862, 420.094083, 421.154224, 422.214284, 423.274265, 424.334166, 425.393988, 426.453731,
    427.513395, 428.572980, 429.632488, 430.691918, 431.751271, 432.810546, 433.869745, 434.928867, 435.987913, 437.046882,
    438.105777, 439.164596, 440.223339, 441.282008, 442.340603, 443.399123, 444.457570, 445.515942, 446.574242, 447.632468,
    448.690621, 449.748702, 450.806711, 451.864647, 452.922512, 453.980305, 455.038027, 456.095679, 457.153259, 458.210769,
    459.268209, 460.325579, 461.382879, 462.440110, 463.497272, 464.554365, 465.611389, 466.668344, 467.725232, 468.782052,
    469.838804, 470.895488, 471.952105, 473.008656, 474.065139, 475.121556, 476.177907, 477.234192, 478.290411, 479.346565,
    480.402653, 481.458676, 482.514634, 483.570528, 484.626357, 485.682122, 486.737823, 487.793460, 488.849033, 489.904544,
    490.959991, 492.015375, 493.070697, 494.125956, 495.181153, 496.236287, 497.291360, 498.346372, 499.401322, 500.456210,
    501.511038, 502.565805, 503.620511, 504.675157, 505.729742, 506.784268, 507.838733, 508.893140, 509.947486, 511.001774,
    512.056002, 513.110172, 514.164283, 515.218335, 516.272329, 517.326265, 518.380143, 519.433964, 520.487727, 521.541432,
    522.595081, 523.648672, 524.702207, 525.755685, 526.809107, 527.862472, 528.915781, 529.969035, 531.022232, 532.075374,
    533.128461, 534.181492, 535.234469, 536.287390, 537.340257, 538.393069, 539.445827, 540.498531, 541.551181, 542.603777,
    543.656319, 544.708807, 545.761243, 546.813625, 547.865954, 548.918230, 549.970453, 551.022624, 552.074743, 553.126809
};

}  // namespace okvis

#include "implementation/HybridFilter.hpp"

#endif /* INCLUDE_OKVIS_HYBRID_FILTER_HPP_ */
