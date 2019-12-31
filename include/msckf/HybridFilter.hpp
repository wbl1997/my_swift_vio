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

#include <msckf/CameraRig.hpp>
#include <msckf/ImuOdometry.h>
#include <msckf/MotionAndStructureStats.h>

#include <vio/CsvReader.h>
#include <vio/ImuErrorModel.h>


/// \brief okvis Main namespace of this package.
namespace okvis {

enum RetrieveObsSeqType {
    ENTIRE_TRACK=0,
    LATEST_TWO,
    HEAD_TAIL,
};

struct TriangulationStatus {
  bool triangulationOk; // True if the landmark is in front of every camera.
  bool chi2Small;
  bool raysParallel; // True if rotation compensated observation directions are parallel.
  bool flipped; // True if the landmark is flipped to be in front of every camera.
  bool lackObservations; // True if #obs is less than minTrackLength.
  TriangulationStatus()
      : triangulationOk(false),
        chi2Small(false),
        raysParallel(false),
        flipped(false) {}
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
class HybridFilter : public Estimator {
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
   * @brief Applies the dropping/marginalization strategy according to the
   * RSS'13/IJRR'14 paper. The new number of frames in the window will be
   * numKeyframes+numImuFrames.
   * @return True if successful.
   */
  virtual bool applyMarginalizationStrategy(
      size_t numKeyframes, size_t numImuFrames,
      okvis::MapPointVector &removedLandmarks);

  virtual void optimize(size_t numIter, size_t numThreads = 1,
                        bool verbose = false);

  /// @name Getters
  ///\{

  bool getTimeDelay(uint64_t poseId, int camIdx, okvis::Duration *td) const;

  int getCameraExtrinsicOptType(size_t cameraIdx) const;

  virtual int getEstimatedVariableMinimalDim() const final {
    return covariance_.rows();
  }

  virtual void computeCovariance(Eigen::MatrixXd* cov) const final {
    *cov = covariance_;
  }
  ///@}

 public:
  /**
   * @brief gatherPoseObservForTriang
   * @param mp
   * @param cameraGeometry
   * @param frameIds
   * @param T_WSs
   * @param obsDirections Each observation is in image plane z=1, (\bar{x}, \bar{y}, 1).
   * @param obsInPixel
   * @param vSigmai
   * @param seqType
   * @return number of gathered valid observations
   */
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
      const kinematics::Transformation &T_SC0) const;

  bool isPureRotation(const MapPoint& mp) const;

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
  TriangulationStatus triangulateAMapPoint(
      const MapPoint &mp,
      std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
          &obsInPixel,
      std::vector<uint64_t> &frameIds, Eigen::Vector4d &v4Xhomog,
      std::vector<double> &vSigmai,
      const std::shared_ptr<okvis::cameras::CameraBase> cameraGeometry,
      const okvis::kinematics::Transformation &T_SC0, int anchorSeqId = -1,
      bool checkDisparity = false) const;
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


  /// print out the most recent state vector and the stds of its elements.
  /// It can be called in the optimizationLoop, but a better way to save
  /// results is to save in the publisher loop
  virtual bool print(std::ostream &stream) const final;

  virtual void printTrackLengthHistogram(std::ostream &stream) const final;

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

  virtual void setKeyframeRedundancyThresholds(double dist, double angle,
                                       double trackingRate,
                                       size_t minTrackLength) final;

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

 protected:
  // set latest estimates to intermediate variables for the assumed constant
  // states which are commonly used in computing Jacobians of all feature
  // observations
  void retrieveEstimatesOfConstants();

  void updateStates(const Eigen::Matrix<double, Eigen::Dynamic, 1> &deltaX);

  uint64_t getMinValidStateID() const;

  // epipolar measurement used by filters for computing Jacobians
  // https://en.cppreference.com/w/cpp/language/nested_types
  class EpipolarMeasurement {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EpipolarMeasurement(
        const HybridFilter& filter,
        const std::shared_ptr<okvis::cameras::CameraBase> tempCameraGeometry,
        int camIdx, int extrinsicModelId, int minExtrinsicDim, int minProjDim,
        int minDistortDim);

    void prepareTwoViewConstraint(
        const std::vector<uint64_t> &frameIds,
        const std::vector<
            okvis::kinematics::Transformation,
            Eigen::aligned_allocator<okvis::kinematics::Transformation>> &T_WSs,
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
        const std::vector<Eigen::Matrix3d,
                          Eigen::aligned_allocator<Eigen::Matrix3d>> &cov_fj,
        const std::vector<int> &index_vec);

    bool measurementJacobian(
        Eigen::Matrix<double, 1, Eigen::Dynamic> *H_xjk,
        std::vector<Eigen::Matrix<double, 1, 3>,
                    Eigen::aligned_allocator<Eigen::Matrix<double, 1, 3>>>
            *H_fjk,
        double *residual) const;

   private:
    const HybridFilter& filter_;
    const std::shared_ptr<okvis::cameras::CameraBase> tempCameraGeometry_;
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

  Eigen::MatrixXd
      covariance_;  ///< covariance of the error vector of all states, error is
                    ///< defined as \tilde{x} = x - \hat{x} except for rotations
  /// the error vector corresponds to states x_B | x_imu | x_c | \pi{B_{N-m}}
  /// ... \pi{B_{N-1}} following Li icra 2014 x_B = [^{G}p_B] ^{G}q_B ^{G}v_B
  /// b_g b_a]

  std::map<uint64_t, int>
      mStateID2CovID_;  // maps state id to the ordered cloned states in the
                        // covariance matrix

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

}  // namespace okvis

#endif /* INCLUDE_OKVIS_HYBRID_FILTER_HPP_ */
