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

struct LoopKeyframeMetadata {
  LoopKeyframeMetadata(size_t dbowId,
                       const okvis::kinematics::Transformation& vio_T_WB)
      : dbowId_(dbowId), vio_T_WB_(vio_T_WB) {}
  const size_t dbowId_;
  const okvis::kinematics::Transformation vio_T_WB_;
};

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
      std::shared_ptr<const okvis::LoopFrameAndMatches> loopFrameAndMatches,
      okvis::PgoResult& pgoResult) final;

  inline std::shared_ptr<const OrbDatabase> getBoWDatabase() const { return db_BoW_; }

  void detectAndDescribe(
      const okvis::LoopQueryKeyframeMessage& query_keyframe,
      OrbDescriptorVec* descriptors_vec);

  /* ------------------------------------------------------------------------ */
  /** @brief Runs all checks on a frame and determines whether it a loop-closure
      with a previous frame or not. Fills loopFrameAndMatches with this information.
   * @param[in] queryKeyframe. A pointer to the keyframe message provided by VIO estimator.
   * @param[out] queryKeyframeInDB A pointer to the keyframe saved in database.
   * @param[out] loopFrameAndMatches A pointer to the struct that is filled with the
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

  /**
   * @brief createMatchedKeypoints which will be used by VIO estimator for relocalisation.
   * @param[in, out] loopFrameAndMatches the loop frame and its matches message
   */
  void createMatchedKeypoints(okvis::LoopFrameAndMatches* loopFrameAndMatches) const;

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

  gtsam::Pose3 getPgoPoseEstimate(size_t dbowId) const;

  /* @brief Set the vocabulary of the BoW detector.
   * @param[in] voc An OrbVocabulary object.
   */
  void setVocabulary(const OrbVocabulary& voc);

  /* ------------------------------------------------------------------------ */
  /* @brief Prints parameters and other statistics on the LoopClosureDetector.
   */
  void print() const;

  /* ------------------------------------------------------------------------ */
  /** @brief Adds odometry factors to the PGO. No optimization is performed
   * in RPGO for odometry factors.
   * @param[in]
   */
  void addOdometryFactors(const okvis::KeyframeInDatabase& keyframeInDB);

  void initializePGO(); ///< for test only.

  std::shared_ptr<const LoopClosureDetectorParams> loopClosureParameters() const {
    return lcd_params_;
  }

  std::shared_ptr<LoopClosureDetectorParams> loopClosureParameters() {
    return lcd_params_;
  }

  virtual void saveFinalPgoResults() final;

 private:
  // Parameter members
  std::shared_ptr<LoopClosureDetectorParams> lcd_params_;

  // ORB extraction and matching members
  cv::Ptr<cv::ORB> orb_feature_detector_;
  cv::Ptr<cv::DescriptorMatcher> descriptor_matcher_;

  // BoW and Loop Detection database and members
  std::shared_ptr<OrbDatabase> db_BoW_;

  // Store latest computed objects for temporal matching and nss scoring
  LcdThirdPartyWrapper::Ptr lcd_tp_wrapper_;
  DBoW2::BowVector latest_bowvec_;

  // Robust PGO members
  std::unique_ptr<KimeraRPGO::RobustSolver> pgo_;

  std::shared_ptr<LoopKeyframeMetadata>
      latestLoopKeyframe_;  ///< The latest keyframe has been optimized by PGO,
                            ///< is used for correcting online pose estimates.

 private:
  using DMatchVec = std::vector<cv::DMatch>;
};  // class LoopClosureDetector

enum class LoopClosureMethodType {
  Mock = 0u,
  //! Bag of Words approach
  OrbBoW = 1u,
};
}  // namespace VIO
