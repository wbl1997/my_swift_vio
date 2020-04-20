/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   LoopClosureDetector.cpp
 * @brief  Pipeline for detection and reporting of Loop Closures between frames
 * @author Marcus Abate, Luca Carlone
 */

#pragma once

#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv/cv.hpp>
#include <opencv2/features2d.hpp>

#include <DBoW2/DBoW2.h>

#include <Eigen/Core>
#include "loop_closure/LcdThirdPartyWrapper.h"
#include "loop_closure/LoopClosureDetector-definitions.h"
#include "loop_closure/LoopClosureDetectorParams.h"
#include <okvis/KeyframeForLoopDetection.hpp>

#include <okvis/LoopFrameAndMatches.hpp>
#include <okvis/LoopClosureMethod.hpp>

/* ------------------------------------------------------------------------ */
// Forward declare KimeraRPGO, a private dependency.
namespace KimeraRPGO {
class RobustSolver;
}

namespace VIO {
// Add compatibility for c++11's lack of make_unique.
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}


/* ------------------------------------------------------------------------ */
class LoopClosureDetector : public okvis::LoopClosureMethod {
 public:
  POINTER_TYPEDEFS(LoopClosureDetector);
  DELETE_COPY_CONSTRUCTORS(LoopClosureDetector);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /* ------------------------------------------------------------------------ */
  /** @brief Constructor: detects loop-closures and updates internal PGO.
   * @param[in] lcd_params Parameters for the instance of LoopClosureDetector.
   *  instantiated and output/statistics are logged at every spinOnce().
   */
  LoopClosureDetector(std::shared_ptr<LoopClosureDetectorParams> lcd_params);

  /* ------------------------------------------------------------------------ */
  virtual ~LoopClosureDetector();

  virtual std::shared_ptr<okvis::KeyframeInDatabase> initializeKeyframeInDatabase(
      size_t dbowId,
      const okvis::LoopQueryKeyframeMessage& queryKeyframe) const final;

  virtual bool addConstraintsAndOptimize(
      const okvis::KeyframeInDatabase& queryKeyframeInDB,
      std::shared_ptr<const okvis::LoopFrameAndMatches> loopFrameAndMatches) final;

  inline std::shared_ptr<const OrbDatabase> getBoWDatabase() const { return db_BoW_; }

  void detectAndDescribe(
      const okvis::LoopQueryKeyframeMessage& query_keyframe,
      OrbDescriptorVec* descriptors_vec);

  /* ------------------------------------------------------------------------ */
  /** @brief Runs all checks on a frame and determines whether it a loop-closure
      with a previous frame or not. Fills the LoopResult with this information.
   * @param[in] stereo_frame A stereo_frame that has already been "rewritten" by
   *  the pipeline to have ORB features and keypoints.
   * @param[out] result A pointer to the LoopResult that is filled with the
   *  result of the loop-closure detection stage.
   * @return True if the frame is declared a loop-closure with a previous frame,
   *  false otherwise.
   */
  virtual bool detectLoop(
      std::shared_ptr<const okvis::LoopQueryKeyframeMessage> queryKeyframe,
      std::shared_ptr<okvis::KeyframeInDatabase>& queryKeyframeInDB,
      std::shared_ptr<okvis::LoopFrameAndMatches>& loopFrameAndMatches) final;

  /* ------------------------------------------------------------------------ */
  /** @brief Verify that the geometry between two frames is close enough to be
      considered a match, and generate a monocular transformation between them.
   * @param[in] query_id The frame ID of the query image in the database.
   * @param[in] match_id The frame ID of the match image in the databse.
   * @param[out] camCur_T_camRef_mono The pose between the match frame and the
   *  query frame, in the coordinates of the match frame.
   * @return True if the verification check passes, false otherwise.
   */
  bool geometricVerificationCheck(
      const okvis::LoopQueryKeyframeMessage& queryKeyframe,
      const FrameId query_id, const FrameId match_id,
      std::shared_ptr<okvis::LoopFrameAndMatches>* loopFrameAndMatches);

  /* ------------------------------------------------------------------------ */
  /** @brief Returns the values of the PGO, which is the full trajectory of the
   *  PGO.
   * @return The gtsam::Values (poses) of the PGO.
   */
  const gtsam::Values getPGOTrajectory() const;

  /* ------------------------------------------------------------------------ */
  /** @brief Returns the Nonlinear-Factor-Graph from the PGO.
   * @return The gtsam::NonlinearFactorGraph of the optimized trajectory from
   *  the PGO.
   */
  const gtsam::NonlinearFactorGraph getPGOnfg() const;

  /* @brief Set the vocabulary of the BoW detector.
   * @param[in] voc An OrbVocabulary object.
   */
  void setVocabulary(const OrbVocabulary& voc);

  /* ------------------------------------------------------------------------ */
  /* @brief Prints parameters and other statistics on the LoopClosureDetector.
   */
  void print() const;

  /* ------------------------------------------------------------------------ */
  /** @brief Adds an odometry factor to the PGO and optimizes the trajectory.
   *  No actual optimization is performed on the RPGO side for odometry.
   * @param[in] factor An OdometryFactor representing the backend's guess for
   *  odometry between two consecutive keyframes.
   */
  void addOdometryFactors(const okvis::KeyframeInDatabase& keyframeInDB);

  void initializePGO(); ///< for test only.

  /* ------------------------------------------------------------------------ */
  /** @brief Initializes the RobustSolver member with an initial prior factor,
   *  which can be the first OdometryFactor given by the backend.
   * @param[in] factor An OdometryFactor representing the pose between the
   *  initial state of the vehicle and the first keyframe.
   */
  void initializePGO(const OdometryFactor& factor);

  std::shared_ptr<const LoopClosureDetectorParams> loopClosureParameters() const {
    return lcd_params_;
  }

  std::shared_ptr<LoopClosureDetectorParams> loopClosureParameters() {
    return lcd_params_;
  }

 private:
  /* ------------------------------------------------------------------------ */
  /** @brief Computes the indices of keypoints that match between two frames.
   * @param[in] query_id The frame ID of the query frame in the database.
   * @param[in] match_id The frame ID of the match frame in the database.
   * @param[out] i_query A vector of indices that match in the query frame.
   * @param[out] i_match A vector of indices that match in the match frame.
   * @param[in] cut_matches If true, Lowe's Ratio Test will be used to cut
   *  out bad matches before sending output.
   */
  void computeMatchedIndices(const FrameId& query_id,
                             const FrameId& match_id,
                             std::vector<int>* i_query,
                             std::vector<int>* i_match,
                             bool cut_matches = false) const;

 private:
  // Parameter members
  std::shared_ptr<LoopClosureDetectorParams> lcd_params_;

  std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry_;

  // ORB extraction and matching members
  cv::Ptr<cv::ORB> orb_feature_detector_;
  cv::Ptr<cv::DescriptorMatcher> descriptor_matcher_;

  // BoW and Loop Detection database and members
  std::shared_ptr<OrbDatabase> db_BoW_;

  // Store latest computed objects for temporal matching and nss scoring
  LcdThirdPartyWrapper::Ptr lcd_tp_wrapper_;
  DBoW2::BowVector latest_bowvec_;

  // Store camera parameters and StereoFrame stuff once
  gtsam::Pose3 B_Pose_camLrect_;

  // Robust PGO members
  std::unique_ptr<KimeraRPGO::RobustSolver> pgo_;
//  std::vector<gtsam::Pose3> W_Pose_Blkf_estimates_;

 private:
  // Lcd typedefs
  using DMatchVec = std::vector<cv::DMatch>;
//  using AdapterMono = opengv::relative_pose::CentralRelativeAdapter;
//  using SacProblemMono =
//      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem;
//  using AdapterStereo = opengv::point_cloud::PointCloudAdapter;
//  using SacProblemStereo =
//      opengv::sac_problems::point_cloud::PointCloudSacProblem;
};  // class LoopClosureDetector

enum class LoopClosureMethodType {
  Mock = 0u,
  //! Bag of Words approach
  OrbBoW = 1u,
};

//class LcdModule : public MIMOPipelineModule<LcdInput, LcdOutput> {
// public:
//  KIMERA_POINTER_TYPEDEFS(LcdModule);
//  KIMERA_DELETE_COPY_CONSTRUCTORS(LcdModule);
//  using LcdFrontendInput = FrontendOutput::Ptr;
//  using LcdBackendInput = BackendOutput::Ptr;

//  LcdModule(bool parallel_run, LoopClosureDetector::UniquePtr lcd)
//      : MIMOPipelineModule<LcdInput, LcdOutput>("Lcd", parallel_run),
//        frontend_queue_("lcd_frontend_queue"),
//        backend_queue_("lcd_backend_queue"),
//        lcd_(std::move(lcd)) {}
//  virtual ~LcdModule() = default;

//  //! Callbacks to fill queues: they should be all lighting fast.
//  inline void fillFrontendQueue(const LcdFrontendInput& frontend_payload) {
//    frontend_queue_.push(frontend_payload);
//  }
//  inline void fillBackendQueue(const LcdBackendInput& backend_payload) {
//    backend_queue_.push(backend_payload);
//  }

// protected:
//  //! Synchronize input queues.
//  inline InputUniquePtr getInputPacket() override {
//    // TODO(X): this is the same or very similar to the Mesher getInputPacket.
//    LcdBackendInput backend_payload;
//    bool queue_state = false;
//    if (PIO::parallel_run_) {
//      queue_state = backend_queue_.popBlocking(backend_payload);
//    } else {
//      queue_state = backend_queue_.pop(backend_payload);
//    }
//    if (!queue_state) {
//      LOG_IF(WARNING, PIO::parallel_run_)
//          << "Module: " << name_id_ << " - Backend queue is down";
//      VLOG_IF(1, !PIO::parallel_run_)
//          << "Module: " << name_id_ << " - Backend queue is empty or down";
//      return nullptr;
//    }
//    CHECK(backend_payload);
//    const Timestamp& timestamp = backend_payload->W_State_Blkf_.timestamp_;

//    // Look for the synchronized packet in frontend payload queue
//    // This should always work, because it should not be possible to have
//    // a backend payload without having a frontend one first!
//    LcdFrontendInput frontend_payload = nullptr;
//    PIO::syncQueue(timestamp, &frontend_queue_, &frontend_payload);
//    CHECK(frontend_payload);
//    CHECK(frontend_payload->is_keyframe_);

//    // Push the synced messages to the lcd's input queue
//    std::shared_ptr<const okvis::MultiFrame> stereo_keyframe = frontend_payload->stereo_frame_lkf_;
//    const gtsam::Pose3& body_pose = backend_payload->W_State_Blkf_.pose_;
//    return VIO::make_unique<LcdInput>(
//        timestamp, backend_payload->cur_kf_id_, stereo_keyframe, body_pose);
//  }

//  OutputUniquePtr spinOnce(LcdInput::UniquePtr input) override {
//    return lcd_->spinOnce(*input);
//  }

//  //! Called when general shutdown of PipelineModule is triggered.
//  void shutdownQueues() override {
//    LOG(INFO) << "Shutting down queues for: " << name_id_;
//    frontend_queue_.shutdown();
//    backend_queue_.shutdown();
//  }

//  //! Checks if the module has work to do (should check input queues are empty)
//  bool hasWork() const override {
//    // We don't check frontend queue because it runs faster than backend queue.
//    return !backend_queue_.empty();
//  }

// private:
//  //! Input Queues
//  ThreadsafeQueue<LcdFrontendInput> frontend_queue_;
//  ThreadsafeQueue<LcdBackendInput> backend_queue_;

//  //! Lcd implementation
//  LoopClosureDetector::UniquePtr lcd_;
//};

}  // namespace VIO
