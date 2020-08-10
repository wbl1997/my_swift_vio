#ifndef INCLUDE_OKVIS_SLIDING_WINDOW_SMOOTHER_HPP_
#define INCLUDE_OKVIS_SLIDING_WINDOW_SMOOTHER_HPP_

#include <memory>

#include <gflags/gflags.h>

#include <okvis/Estimator.hpp>
#include <okvis/timing/Timer.hpp>

#ifdef HAVE_GTSAM
#define INCREMENTAL_SMOOTHER

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>

#include <gtsam_unstable/nonlinear/BatchFixedLagSmoother.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtsam_unstable/slam/SmartStereoProjectionPoseFactor.h>

#include "gtsam/VioBackEndParams.h"
#include "gtsam/ImuFrontEnd.h"

/// \brief okvis Main namespace of this package.
namespace okvis {
#ifdef INCREMENTAL_SMOOTHER
typedef gtsam::IncrementalFixedLagSmoother Smoother;
#else
typedef gtsam::BatchFixedLagSmoother Smoother;
#endif


//using LandmarkId = long int;  // -1 for invalid landmarks. // int would be too
//                              // small if it is 16 bits!
using SmartStereoFactor = gtsam::SmartStereoProjectionPoseFactor;
//using LandmarkIdSmartFactorMap =
//    std::unordered_map<LandmarkId, SmartStereoFactor::shared_ptr>;
//using Slot = long int;
//using SmartFactorMap =
//    gtsam::FastMap<LandmarkId, std::pair<SmartStereoFactor::shared_ptr, Slot>>;

/**
 * SlidingWindowSmoother builds upon gtsam FixedLagSmoother.
 */
class SlidingWindowSmoother : public Estimator {
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief The default constructor.
   */
  SlidingWindowSmoother(const okvis::BackendParams& backendParams);

  void setupSmoother(const okvis::BackendParams& backendParams);

  /**
   * @brief Constructor if a ceres map is already available.
   * @param mapPtr Shared pointer to ceres map.
   */
  SlidingWindowSmoother(const okvis::BackendParams& backendParams,
                        std::shared_ptr<okvis::ceres::Map> mapPtr);

  virtual ~SlidingWindowSmoother();

  void addInitialPriorFactors();

  /**
   * @brief addImuValues add values for the last navigation state variable.
   */
  void addImuValues();

  /**
   * @brief addImuFactor add the IMU factor for the last navigation state variable.
   */
  void addImuFactor();

  int addImu(const okvis::ImuParameters & imuParameters) override;

  void addCameraExtrinsicFactor();

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
   * gravity direction which is assumed in the IMU propagation Only one IMU
   * is supported for now
   */
  virtual bool addStates(okvis::MultiFramePtr multiFrame,
                         const okvis::ImuMeasurementDeque& imuMeasurements,
                         bool asKeyframe) final;

  virtual void optimize(size_t numIter, size_t numThreads = 1,
                        bool verbose = false) final;

  /**
   * @brief Applies the dropping/marginalization strategy.
   * The new number of frames in the window will be
   * numKeyframes+numImuFrames.
   * @return True if successful.
   */
  virtual bool applyMarginalizationStrategy(
      size_t numKeyframes, size_t numImuFrames,
      okvis::MapPointVector& removedLandmarks) final;

  /**
   * @brief computeCovariance the block covariance for the navigation state
   * with an abuse of notation, i.e., diag(cov(p, q), cov(v), cov(bg),
   * cov(ba))
   * @param[in, out] cov pointer to the covariance matrix which will be
   * resized in this function.
   * @return true if covariance is successfully computed.
   */
  virtual bool computeCovariance(Eigen::MatrixXd* cov) const;

 private:
  uint64_t getMinValidStateId() const;

  void addLandmarkToGraph(uint64_t landmarkId);

  void updateLandmarkInGraph(uint64_t landmarkId);

 protected:
  okvis::BackendParams backendParams_;
  okvis::ImuParams imuParams_;

  // IMU frontend.
  std::unique_ptr<okvis::ImuFrontEnd> imuFrontend_;

  // State covariance. (initialize to zero)
  gtsam::Matrix state_covariance_lkf_ = Eigen::MatrixXd::Zero(15, 15);

  // Vision params.
  gtsam::SmartStereoProjectionParams smart_factors_params_;
  gtsam::SharedNoiseModel smart_noise_;
  // Pose of the left camera wrt body
  const gtsam::Pose3 B_Pose_leftCam_;
  // Stores calibration, baseline.
  const gtsam::Cal3_S2Stereo::shared_ptr stereo_cal_;

  // State.
  //!< current state of the system.
  gtsam::Values state_;

  // ISAM2 smoother
  std::shared_ptr<Smoother> smoother_;

  // Values
  //!< new states to be added
  gtsam::Values new_values_;

  // Factors.
  //!< New factors to be added
  gtsam::NonlinearFactorGraph new_imu_prior_and_other_factors_;
  //!< landmarkId -> {SmartFactorPtr}
//  LandmarkIdSmartFactorMap new_smart_factors_;
  //!< landmarkId -> {SmartFactorPtr, SlotIndex}
//  SmartFactorMap old_smart_factors_;
  // if SlotIndex is -1, means that the factor has not been inserted yet in
  // the graph

  okvis::timing::Timer addLandmarkFactorsTimer;
  okvis::timing::Timer isam2UpdateTimer;
  okvis::timing::Timer computeCovarianceTimer;
  okvis::timing::Timer marginalizeTimer;
  okvis::timing::Timer updateLandmarksTimer;

  std::vector<size_t>
      mTrackLengthAccumulator;  // histogram of the track lengths, start from
                                // 0,1,2, to a fixed number
  double trackingRate_;
};

}  // namespace okvis

#else
namespace okvis {
  typedef Estimator SlidingWindowSmoother;
}  // namespace okvis

#endif // # ifdef HAVE_GTSAM
#endif /* INCLUDE_OKVIS_SLIDING_WINDOW_SMOOTHER_HPP_ */
