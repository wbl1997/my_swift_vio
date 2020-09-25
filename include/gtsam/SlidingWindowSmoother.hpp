/**
 * @file   SlidingWindowSmoother.hpp
 * @brief  Declaration of SlidingWindowSmoother which wraps gtsam::FixedLagSmoother for VIO.
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_GTSAM_SLIDING_WINDOW_SMOOTHER_HPP_
#define INCLUDE_GTSAM_SLIDING_WINDOW_SMOOTHER_HPP_

#include <memory>

#include <gflags/gflags.h>

#include <okvis/Estimator.hpp>
#include <okvis/timing/Timer.hpp>
#include <msckf/memory.h>

#ifdef HAVE_GTSAM
#define INCREMENTAL_SMOOTHER

#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>

#include <gtsam_unstable/nonlinear/BatchFixedLagSmoother.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtsam_unstable/slam/SmartStereoProjectionPoseFactor.h>

#include "gtsam/VioBackEndParams.h"
#include "gtsam/ImuFrontEnd.h"

namespace okvis {
#ifdef INCREMENTAL_SMOOTHER
typedef gtsam::IncrementalFixedLagSmoother Smoother;
#else
typedef gtsam::BatchFixedLagSmoother Smoother;
#endif

using LandmarkId = uint64_t;
using LandmarkIdSmartFactorMap = std::unordered_map<
    LandmarkId,
    gtsam::SmartProjectionFactor<gtsam::PinholePose<gtsam::Cal3DS2>>::shared_ptr>;
using Slot = long int;
using SmartFactorMap = gtsam::FastMap<
    LandmarkId, std::pair<gtsam::SmartProjectionFactor<
                              gtsam::PinholePose<gtsam::Cal3DS2>>::shared_ptr,
                          Slot>>;
using PointsWithIdMap = std::unordered_map<LandmarkId, gtsam::Point3>;
enum class LandmarkType { SMART = 0, PROJECTION };
using LmkIdToLmkTypeMap = std::unordered_map<LandmarkId, LandmarkType>;

// Set parameters for smart factors.
void setSmartFactorsParams(
    gtsam::SharedNoiseModel* smart_noise,
    gtsam::SmartProjectionParams* smart_factors_params,
    double smart_noise_sigma,
    double rank_tolerance,
    double landmark_distance_threshold,
    double retriangulation_threshold,
    double outlier_rejection);

// Set parameters for all types of factors.
void setFactorsParams(
    const BackendParams& vio_params,
    gtsam::SharedNoiseModel* smart_noise,
    gtsam::SmartProjectionParams* smart_factors_params,
    gtsam::SharedNoiseModel* no_motion_prior_noise,
    gtsam::SharedNoiseModel* zero_velocity_prior_noise,
    gtsam::SharedNoiseModel* constant_velocity_prior_noise);

/**
 * SlidingWindowSmoother builds upon gtsam FixedLagSmoother.
 */
class SlidingWindowSmoother : public Estimator {
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SlidingWindowSmoother(const okvis::BackendParams& backendParams);

  void setupSmoother(const okvis::BackendParams& backendParams);

  /**
   * @brief Constructor if a ceres map is already available.
   * @param mapPtr Shared pointer to ceres map.
   */
  SlidingWindowSmoother(const okvis::BackendParams& backendParams,
                        std::shared_ptr<okvis::ceres::Map> mapPtr);

  virtual ~SlidingWindowSmoother();

  virtual void addInitialPriorFactors();

  /**
   * @brief addImuValues add values for the last navigation state variable.
   */
  virtual void addImuValues();

  /**
   * @brief addImuFactor add the IMU factor for the last navigation state variable.
   */
  virtual void addImuFactor();

  int addImu(const okvis::ImuParameters & imuParameters) override;

  void addCameraExtrinsicFactor();

  void addCameraSystem(const okvis::cameras::NCameraSystem& cameras) override;
  /**
   * @brief add  nav states, biases to the factor graph and record their ids
   * in statesMap_.
   * @param multiFrame Matched multiFrame.
   * @param imuMeasurements IMU measurements from last state to new one.
   * imuMeasurements covers at least the current state and the last state in
   * time, with an extension on both sides.
   * @param asKeyframe Is this new frame a keyframe?
   * @return True if successful.
   */
  bool addStates(okvis::MultiFramePtr multiFrame,
                         const okvis::ImuMeasurementDeque& imuMeasurements,
                         bool asKeyframe) final;

  void optimize(size_t numIter, size_t numThreads = 1,
                        bool verbose = false) final;

  size_t getLandmarks(okvis::MapPointVector& landmarks) const override;

  /**
   * @brief Remove unused landmarks and state variables to keep pace with the
   * internal smoother. After this function, statesMap_ and landmarksMap_ should
   * be consistent with the NonlinearFactorGraph. State variables not involved
   * in the NonlinearFactorGraph will be erased. Landmarks whose observations
   * are outside of the sliding window will be erased.
   * @return True if successful.
   */
  bool applyMarginalizationStrategy(
      size_t numKeyframes, size_t numImuFrames,
      okvis::MapPointVector& removedLandmarks) final;

  /**
   * @brief computeCovariance the block covariance for the navigation state
   * with an abuse of notation, i.e., diag(cov(p, q), cov(v), cov(bg),
   * cov(ba))
   * @warning only works for IncrementalFixedLagSmoother
   * @param[in, out] cov pointer to the covariance matrix which will be
   * resized in this function.
   * @return true if covariance is successfully computed.
   */
  bool computeCovariance(Eigen::MatrixXd* cov) const override;

  bool getStateStd(
      Eigen::Matrix<double, Eigen::Dynamic, 1>* stateStd) const final;

  bool print(std::ostream& stream) const override;

  /**
   * @brief getMapLmkIdsTo3dPointsInTimeHorizon Get valid 3D points and corresponding lmk id.
   * @warning  It only works for BackendModality::Structureless for now.
   * @param lmk_id_to_lmk_type_map
   * @param min_age
   * @return
   */
  PointsWithIdMap getMapLmkIdsTo3dPointsInTimeHorizon(
      LmkIdToLmkTypeMap* lmk_id_to_lmk_type_map = nullptr,
      const size_t& min_age = 2) const;

 protected:
  /**
   * @brief getMinValidStateId get minimum id of nav state variables in the NonlinearFactorGraph.
   * @return
   */
  uint64_t getMinValidStateId() const;

  /**
   * @brief addLandmarkToGraph add a new landmark to the graph.
   * @param landmarkId
   */
  virtual bool addLandmarkToGraph(uint64_t landmarkId, const Eigen::Vector4d& hpW);

  /**
   * @brief updateLandmarkInGraph add observations for an existing landmark.
   * @param landmarkId
   */
  virtual void updateLandmarkInGraph(uint64_t landmarkId);

  /**
   * @brief addLandmarkSmartFactorToGraph Adds the smart factor of a landmark to
   * the graph for the first time.
   * @param lm_id
   */
  virtual void addLandmarkSmartFactorToGraph(const LandmarkId& lm_id);

  /**
   * @brief updateLandmarkSmartFactorInGraph: Update the smart factor for
   * a landmark already in the graph.
   * @param lmk_id
   */
  virtual void updateLandmarkSmartFactorInGraph(const LandmarkId& lmk_id);

  void assembleNewFactorsAndDeleteSlots(
      gtsam::NonlinearFactorGraph* new_factors_tmp,
      gtsam::FactorIndices* delete_slots,
      std::vector<LandmarkId>* lmk_ids_of_new_smart_factors_tmp) const;

  /**
   * @brief update state variables with FixedLagSmoother estimates.
   */
  virtual void updateStates();

  /**
   * @brief updateSmoother add new factors and/or values to the
   * NonlinearFactorGraph and call FixedLagSmoother::update which internally
   * calls marginalizeLeaves. After this function, the oldest variables in statesMap_
   * may be not longer in the NonlinearFactorGraph.
   * @warning
   * @param result
   * @param new_factors_tmp
   * @param new_values
   * @param timestamps
   * @param delete_slots
   * @return False if the update failed, true otw.
   */
  bool updateSmoother(
      gtsam::FixedLagSmoother::Result* result,
      const gtsam::NonlinearFactorGraph& new_factors_tmp =
          gtsam::NonlinearFactorGraph(),
      const gtsam::Values& new_values = gtsam::Values(),
      const std::map<gtsam::Key, double>& timestamps =
          gtsam::FixedLagSmoother::KeyTimestampMap(),
      const gtsam::FactorIndices& delete_slots = gtsam::FactorIndices());

  // Here are 3 approaches to compute covariance blocks for the lastest pose,
  // velocity, and biases. Preliminary tests show that they have similar time
  // consumption but iSAM2 is a bit faster than gtsam::Marginals. And their
  // results are almost identical, iSAM2 difference from the other two methods
  // grows over time, reaching the the order of magnitude 1e-3 in position after
  // 300s.
  /**
   * @brief iSAM2MarginalCovariance compute marginal covariance blocks for the
   * latest nav state and biases with iSAM2.
   * @warning only works with IncrementalFixedLagSmoother.
   * @param[in, out] cov
   * @return
   */
  bool iSAM2MarginalCovariance(Eigen::MatrixXd* cov) const;

  /**
   * @brief gtsamMarginalCovariance compute marginal covariance blocks for the
   * lastest nav state and biases with gtsam::Marginals. It works with
   * Incremental and BatchFixedLagSmoother.
   * @param[in, out] cov
   * @return
   */
  virtual bool gtsamMarginalCovariance(Eigen::MatrixXd* cov) const;

  /**
   * @brief gtsamJointMarginalCovariance compute the joint marginal covariance
   * for the lastest nav state and biases with gtsam::Marginals. It works with
   * Incremental and BatchFixedLagSmoother.
   * @param[in, out] cov
   * @return
   */
  bool gtsamJointMarginalCovariance(Eigen::MatrixXd* cov) const;

  void printSmootherInfo(const gtsam::NonlinearFactorGraph& new_factors_tmp,
                         const gtsam::FactorIndices& delete_slots,
                         const std::string& message = "CATCHING EXCEPTION",
                         const bool& showDetails = false) const;

  void cleanCheiralityLmk(
      const gtsam::Symbol& lmk_symbol,
      gtsam::NonlinearFactorGraph* new_factors_tmp_cheirality,
      gtsam::Values* new_values_cheirality,
      std::map<gtsam::Key, double>* timestamps_cheirality,
      gtsam::FactorIndices* delete_slots_cheirality,
      const gtsam::NonlinearFactorGraph& graph,
      const gtsam::NonlinearFactorGraph& new_factors_tmp,
      const gtsam::Values& new_values,
      const std::map<gtsam::Key, double>& timestamps,
      const gtsam::FactorIndices& delete_slots);

  bool deleteLmkFromFeatureTracks(uint64_t lmkId);

  void updateNewSmartFactorsSlots(
      const std::vector<LandmarkId>& lmk_ids_of_new_smart_factors_tmp,
      SmartFactorMap* old_smart_factors);


  /**
   * @brief triangulateSafe wrap of gtsam::triangulateSafe.
   * It can be used as an alternative to triangulateWithDisparityCheck.
   * @param lmkId
   * @return
   */
  bool triangulateSafe(uint64_t lmkId, Eigen::Vector4d* hpW) const;

  /**
   * @brief triangulateWithDisparityCheck custom triangulation by using DLT with disparity check
   * @param lmkId
   * @return
   */
  bool triangulateWithDisparityCheck(uint64_t lmkId, Eigen::Vector4d* hpW,
                                     double focalLength, double raySigmaScalar) const;

 protected:
  okvis::BackendParams backendParams_;
  okvis::ImuParams imuParams_;

  // IMU frontend integrates IMU measurements into factors.
  std::unique_ptr<okvis::ImuFrontEnd> imuFrontend_;

  // Vision params.
  gtsam::SmartProjectionParams smart_factors_params_;
  gtsam::SharedNoiseModel smart_noise_;

  //! No motion factors settings.
  gtsam::SharedNoiseModel zero_velocity_prior_noise_;
  gtsam::SharedNoiseModel no_motion_prior_noise_;
  gtsam::SharedNoiseModel constant_velocity_prior_noise_;

  // Stores calibration, baseline.
//  const gtsam::Cal3_S2Stereo::shared_ptr stereo_cal_;

  //!< current state variables of the system calculated by the FixedLagSmoother.
  gtsam::Values state_;

  // Covariance of position, orientation, velocity, bg, ba.
  Eigen::MatrixXd covariance_;

  std::shared_ptr<Smoother> smoother_;

  // camera intrinsic parameters with distortion.
  boost::shared_ptr<gtsam::Cal3DS2> cal0_;
  // Pose of the left camera wrt body.
  gtsam::Pose3 body_P_cam0_;

  //!< new states to be added
  gtsam::Values new_values_;

  //!< New factors to be added
  gtsam::NonlinearFactorGraph new_imu_prior_and_other_factors_;

  gtsam::NonlinearFactorGraph new_reprojection_factors_;

  //!< The mapping from nav state id to the ids of landmarks added to the graph
  //!< along with the nav state.
  //! As a result, they will be marginalized at the same time by the smoother.
  std::unordered_map<uint64_t, std::vector<uint64_t>> navStateToLandmarks_;

  // Data structures used to update a SmartFactor of a landmark in the
  // NonlinearFactorGraph as a new observation for the landmark arrives.
  //!< landmarkId -> {SmartFactorPtr}
  LandmarkIdSmartFactorMap new_smart_factors_;
  //!< landmarkId -> {SmartFactorPtr, SlotIndex}
  SmartFactorMap old_smart_factors_;
  // if SlotIndex is -1, means that the factor has not been inserted yet in
  // the graph

  okvis::timing::Timer addLandmarkFactorsTimer;
  okvis::timing::Timer isam2UpdateTimer;
  okvis::timing::Timer computeCovarianceTimer;
  okvis::timing::Timer marginalizeTimer;
  okvis::timing::Timer updateLandmarksTimer;

  //! Number of Cheirality exceptions
  size_t counter_of_exceptions_ = 0;

  // histogram of the track lengths, starting from 0, 1, 2 up to a fixed number.
  std::vector<size_t> mTrackLengthAccumulator;
  // what percentage of landmarks in the landmarksMap_ are observed in the
  // current frame?
  double trackingRate_;
};


void deleteAllFactorsWithKeyFromFactorGraph(
    const gtsam::Key& key,
    const gtsam::NonlinearFactorGraph& new_factors_tmp,
    gtsam::NonlinearFactorGraph* factor_graph_output);

// Returns if the key in timestamps could be removed or not.
bool deleteKeyFromTimestamps(const gtsam::Key& key,
                             const std::map<gtsam::Key, double>& timestamps,
                             std::map<gtsam::Key, double>* timestamps_output);


// Returns if the key in timestamps could be removed or not.
bool deleteKeyFromValues(const gtsam::Key& key,
                         const gtsam::Values& values,
                         gtsam::Values* values_output);

// Find all slots of factors that have the given key in the list of keys.
void findSlotsOfFactorsWithKey(
    const gtsam::Key& key,
    const gtsam::NonlinearFactorGraph& graph,
    std::vector<size_t>* slots_of_factors_with_key);

}  // namespace okvis
#else
namespace okvis {
class SlidingWindowSmoother : public Estimator {
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SlidingWindowSmoother(const okvis::BackendParams& backendParams) : Estimator() {}

  /**
   * @brief Constructor if a ceres map is already available.
   * @param mapPtr Shared pointer to ceres map.
   */
  SlidingWindowSmoother(const okvis::BackendParams& backendParams,
                        std::shared_ptr<okvis::ceres::Map> mapPtr) : Estimator(mapPtr) {}
}
}  // namespace okvis
#endif // # ifdef HAVE_GTSAM

#endif /* INCLUDE_GTSAM_SLIDING_WINDOW_SMOOTHER_HPP_ */
