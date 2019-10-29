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
#include <okvis/ceres/EuclideanParamBlockSized.hpp>

#include <msckf/ImuOdometry.h>
#include <msckf/CameraRig.hpp>
#include <okvis/timing/Timer.hpp>

#include <vio/CsvReader.h>
#include <vio/ImuErrorModel.h>

#include "msckf/BoundedImuDeque.hpp"
#include "msckf/InitialPVandStd.hpp"

/// \brief okvis Main namespace of this package.
namespace okvis {

namespace ceres {
typedef EuclideanParamBlockSized<9> ShapeMatrixParamBlock;
}

enum RetrieveObsSeqType {
    ENTIRE_TRACK=0,
    LATEST_TWO,
    HEAD_TAIL,
};

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
      okvis::MapPointVector &removedLandmarks);

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

  bool getTimeDelay(uint64_t poseId, int camIdx, okvis::Duration *td) const;

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


  int getCameraExtrinsicOptType(size_t cameraIdx) const;

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
  /// imageDelay. Therefore, t_{d_j} is set 0 at the beginning t_j is the
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
  // set latest estimates to intermediate variables for the assumed constant
  // states which are commonly used in computing Jacobians of all feature
  // observations
  void retrieveEstimatesOfConstants();

  void updateStates(const Eigen::Matrix<double, Eigen::Dynamic, 1> &deltaX);

 public:
  okvis::Time firstStateTimestamp();

  size_t gatherPoseObservForTriang(
      const MapPoint &mp,
      const std::shared_ptr<cameras::CameraBase> cameraGeometry,
      std::vector<uint64_t> *frameIds,
      std::vector<okvis::kinematics::Transformation,
                  Eigen::aligned_allocator<okvis::kinematics::Transformation>>
          *T_WSs,
      std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
          *obsDirections,
      std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
          *obsInPixel,
      std::vector<double> *vSigmai,
      RetrieveObsSeqType seqType=ENTIRE_TRACK) const;

  /**
   * @brief triangulateAMapPoint, does not support rays which arise from static
   * mode, pure rotation, or points at infinity. Assume the same camera model
   * for all observations, and rolling shutter effect is not accounted for
   * @param mp
   * @param obsInPixel
   * @param frameIds, id of frames observing this feature in the ascending order
   *    because the MapPoint.observations is an ordinary ordered map
   * @param v4Xhomog, stores [X,Y,Z,1] in the global frame or the anchor frame
   *    depending on anchorSeqId
   * @param vSigmai, the diagonal elements of the observation noise matrix, in
   *    pixels, size 2Nx1
   * @param cameraGeometry, used for point projection
   * @param T_SC0
   * @param anchorSeqId index of the anchor frame in the ordered observation map
   *    -1 by default meaning that AIDP is not used
   * @return true if triangulation successful
   */
  bool triangulateAMapPoint(
      const MapPoint &mp,
      std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
          &obsInPixel,
      std::vector<uint64_t> &frameIds, Eigen::Vector4d &v4Xhomog,
      std::vector<double> &vSigmai,
      const std::shared_ptr<okvis::cameras::CameraBase> cameraGeometry,
      const okvis::kinematics::Transformation &T_SC0,
      int anchorSeqId = -1) const;
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
   * @brief compute the marginalized Jacobian for a feature i's
   track
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

  /**
   * @brief measurementJacobian for one epipolar constraint
   * @param tempCameraGeometry
   * @param frameId2
   * @param T_WS2
   * @param obsDirection2
   * @param obsInPixel2
   * @param imagePointNoiseStd2
   * @param camIdx
   * @param H_xjk has the proper size upon calling this func
   * @param H_fjk is an empty vector upon calling this func
   * @param cov_fjk is an empty vector upon entering this func
   * @param residual
   * @return
   */
  bool measurementJacobianEpipolar(
      const std::shared_ptr<okvis::cameras::CameraBase> tempCameraGeometry,
      const std::vector<uint64_t>& frameId2,
      const std::vector<
          okvis::kinematics::Transformation,
          Eigen::aligned_allocator<okvis::kinematics::Transformation>>& T_WS2,
      const std::vector<Eigen::Vector3d,
                        Eigen::aligned_allocator<Eigen::Vector3d>>&
          obsDirection2,
      const std::vector<Eigen::Vector2d,
                        Eigen::aligned_allocator<Eigen::Vector2d>>& obsInPixel2,
      const std::vector<double>& imagePointNoiseStd2, int camIdx,
      Eigen::Matrix<double, 1, Eigen::Dynamic>* H_xjk,
      std::vector<Eigen::Matrix<double, 1, 3>,
                  Eigen::aligned_allocator<Eigen::Matrix<double, 1, 3>>>* H_fjk,
      std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>*
          cov_fjk,
      double* residual) const;


  /// print out the most recent state vector and the stds of its elements. This
  /// function can be called in the optimizationLoop, but a better way to save
  /// results is use the publisher loop
  bool print(const std::string) const;

  bool print(std::ostream &stream) const;

  void printTrackLengthHistogram(std::ostream &stream) const;

  /// reset initial condition
  inline void resetInitialPVandStd(const InitialPVandStd &rhs,
                                   bool bUseExternalPose = false) {
    pvstd_ = rhs;
    useExternalInitialPose_ = bUseExternalPose;
  }

  size_t numObservations(uint64_t landmarkId);
  /**
   * @brief getCameraCalibrationEstimate get the latest estimate of camera
   * calibration parameters
   * @param vfckptdr
   * @return the last pose id
   */
  uint64_t getCameraCalibrationEstimate(Eigen::Matrix<double, Eigen::Dynamic, 1> &vfckptdr);
  /**
   * @brief getTgTsTaEstimate, get the lastest estimate of Tg Ts Ta with entries
   * in row major order
   * @param vTGTSTA
   * @return the last pose id
   */
  uint64_t getTgTsTaEstimate(Eigen::Matrix<double, Eigen::Dynamic, 1> &vTGTSTA);

  /**
   * @brief get variance for nav, imu, camera extrinsic, intrinsic, td, tr
   * @param variances
   */
  void getVariance(Eigen::Matrix<double, Eigen::Dynamic, 1> &variances) const;

  bool getFrameId(uint64_t poseId, int &frameIdInSource, bool &isKF) const;

  void setKeyframeRedundancyThresholds(double dist, double angle,
                                       double trackingRate,
                                       size_t minTrackLength);

  // will remove state parameter blocks and all of their related residuals
  okvis::Time removeState(uint64_t stateId);

  void initCovariance(int camIdx = 0);

  // currently only support one camera
  void initCameraParamCovariance(int camIdx = 0);

  void addCovForClonedStates();

  // camera parameters and all cloned states including the last inserted
  // and all landmarks.
  // p_B^C, f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, [k_3], t_d, t_r,
  // \pi_{B_i}(=[p_{B_i}^G, q_{B_i}^G, v_{B_i}^G]), l_i
  inline int cameraParamPoseAndLandmarkMinimalDimen() const {
    return cameraParamsMinimalDimen() +
           kClonedStateMinimalDimen * statesMap_.size() +
           3 * mInCovLmIds.size();
  }

  inline int cameraParamsMinimalDimen() const {
    const int camIdx = 0;
    return camera_rig_.getCameraParamsMininalDimen(camIdx);
  }

  inline int startIndexOfClonedStates() const {
    const int camIdx = 0;
    return ceres::ode::OdoErrorStateDim +
           camera_rig_.getCameraParamsMininalDimen(camIdx);
  }

  inline int startIndexOfCameraParams() const {
    return ceres::ode::OdoErrorStateDim;
  }

  // error state: \delta p, \alpha for q, \delta v
  // state: \pi_{B_i}(=[p_{B_i}^G, q_{B_i}^G, v_{B_i}^G])
  static const int kClonedStateMinimalDimen = 9;

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

  // minimum of the ids of the states that have tracked features
  uint64_t minValidStateID;

  mutable okvis::timing::Timer triangulateTimer;
  mutable okvis::timing::Timer computeHTimer;
  okvis::timing::Timer computeKalmanGainTimer;
  okvis::timing::Timer updateStatesTimer;
  okvis::timing::Timer updateCovarianceTimer;
  okvis::timing::Timer updateLandmarksTimer;

  // for each point in the state vector/covariance,
  // its landmark id which points to the parameter block
  std::deque<uint64_t> mInCovLmIds;

  InitialPVandStd pvstd_;
  bool useExternalInitialPose_;  // do we use external pose for initialization

  // maximum number of consecutive observations until a landmark is added as a
  // state, but can be set dynamically as done in
  // Li, icra14 optimization based ...
  static const size_t maxTrackLength_ = 12;
  // i.e., max cloned states in the cov matrix

  size_t minTrackLength_;
  // i.e., min observs to triang a landmark for the monocular case

  std::vector<size_t>
      mTrackLengthAccumulator;  // histogram of the track lengths, start from
                                // 0,1,2, to a fixed number

  double trackingRate_;
  // Threshold for determine keyframes
  double translationThreshold_;
  double rotationThreshold_;
  double trackingRateThreshold_;

  // The window centered at a stateEpoch for retrieving the inertial data
  // which is used for propagating the camera pose to epochs in the window,
  // i.e., timestamps of observations in a rolling shutter image.
  // A value greater than (t_d + t_r)/2 is recommended.
  // Note camera observations in MSCKF will not occur at the latest frame.
  static const okvis::Duration half_window_;
};

struct IsObservedInFrame {
  IsObservedInFrame(uint64_t x) : frameId(x) {}
  bool operator()(
      const std::pair<okvis::KeypointIdentifier, uint64_t> &v) const {
    return v.first.frameId == frameId;
  }

 private:
  uint64_t frameId;  ///< Multiframe ID.
};

/**
 * @brief obsDirectionJacobian
 * @param obsInPixel [u, v] affected with noise in image
 * @param cameraGeometry
 * @param pixelNoiseStd
 * @param dfj_dXcam
 * @param cov_fj
 */
void obsDirectionJacobian(
    const Eigen::Vector3d& obsInPixel,
    const std::shared_ptr<okvis::cameras::CameraBase> cameraGeometry,
    int projOptModelId,
    double pixelNoiseStd,
    Eigen::Matrix<double, 3, Eigen::Dynamic>* dfj_dXcam,
    Eigen::Matrix3d* cov_fj);

}  // namespace okvis

#include "implementation/HybridFilter.hpp"

#endif /* INCLUDE_OKVIS_HYBRID_FILTER_HPP_ */
