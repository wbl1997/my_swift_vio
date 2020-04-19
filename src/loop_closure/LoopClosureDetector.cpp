/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   LoopClosureDetector.cpp
 * @brief  Pipeline for detection and reporting of Loop Closures between frames.
 * @author Marcus Abate
 * @author Antoni Rosinol
 * @author Luca Carlone
 */

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <opengv/point_cloud/PointCloudAdapter.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/point_cloud/PointCloudSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>

#include <KimeraRPGO/RobustSolver.h>

#include "loop_closure/LoopClosureDetector.h"
#include "loop_closure/GtsamWrap.hpp"

#include <gtsam/inference/Symbol.h>

//#include "utils/Statistics.h"
//#include "kimera-vio/utils/Timer.h"
//#include "kimera-vio/utils/UtilsOpenCV.h"

DEFINE_string(vocabulary_path,
              "vocabulary/ORBvoc.yml",
              "Path to BoW vocabulary file for LoopClosureDetector module.");

/** Verbosity settings: (cumulative with every increase in level)
      0: Runtime errors and warnings, spin start and frequency are reported.
      1: Loop closure detections are reported as warnings.
      2: Loop closure failures are reported as errors.
      3: Statistics are reported at relevant steps.
**/

namespace VIO {

/* ------------------------------------------------------------------------ */
LoopClosureDetector::LoopClosureDetector(
    std::shared_ptr<LoopClosureDetectorParams> lcd_params)
    : okvis::LoopClosureMethod(),
      lcd_params_(lcd_params),
//      set_intrinsics_(false),
      orb_feature_detector_(),
      descriptor_matcher_(),
      db_BoW_(nullptr),
      lcd_tp_wrapper_(nullptr),
      latest_bowvec_(),
      B_Pose_camLrect_(),
      pgo_(nullptr) {
  // Initialize the ORB feature detector object:
  orb_feature_detector_ = cv::ORB::create(lcd_params_->nfeatures_,
                                          lcd_params_->scale_factor_,
                                          lcd_params_->nlevels_,
                                          lcd_params_->edge_threshold_,
                                          lcd_params_->first_level_,
                                          lcd_params_->WTA_K_,
                                          lcd_params_->score_type_,
                                          lcd_params_->patch_sze_,
                                          lcd_params_->fast_threshold_);

  // Initialize our feature matching object:
  descriptor_matcher_ =
      cv::DescriptorMatcher::create(lcd_params_->matcher_type_);

  // Load ORB vocabulary:
  std::ifstream f_vocab(FLAGS_vocabulary_path.c_str());
  CHECK(f_vocab.good()) << "LoopClosureDetector: Incorrect vocabulary path: "
                        << FLAGS_vocabulary_path;
  f_vocab.close();

  OrbVocabulary vocab;
  LOG(INFO) << "LoopClosureDetector:: Loading vocabulary from "
            << FLAGS_vocabulary_path;
  vocab.load(FLAGS_vocabulary_path);
  LOG(INFO) << "Loaded vocabulary with " << vocab.size() << " visual words.";

  // Initialize the thirdparty wrapper:
  lcd_tp_wrapper_ = VIO::make_unique<LcdThirdPartyWrapper>(lcd_params);
  // Initialize db_BoW_:
  db_BoW_ = std::make_shared<OrbDatabase>(vocab);
  // Initialize pgo_:
  // TODO(marcus): parametrize the verbosity of PGO params
  KimeraRPGO::RobustSolverParams pgo_params;
  // TODO(jhuai): Pcm3D uses PoseWithCovariance to check consistency of loop constraints
  // where as PcmSimple3D uses Pose for this purpose.
  // pgo_params.setPcm3DParams(pgo_odom_mahal_threshold_, pgo_lc_mahal_threshold_);
  pgo_params.setPcmSimple3DParams(lcd_params_->pgo_trans_threshold_,
                                  lcd_params_->pgo_rot_threshold_,
                                  KimeraRPGO::Verbosity::QUIET);
  pgo_ = VIO::make_unique<KimeraRPGO::RobustSolver>(pgo_params);
}

LoopClosureDetector::~LoopClosureDetector() {
  LOG(INFO) << "LoopClosureDetector desctuctor called.";
}

/* ------------------------------------------------------------------------ */
bool LoopClosureDetector::addConstraintsAndOptimize(
    std::shared_ptr<okvis::KeyframeInDatabase> queryKeyframeInDB,
    std::shared_ptr<okvis::LoopFrameAndMatches> loopFrameAndMatches) {
  // Update the PGO with the backend VIO estimate.
  bool tryToSimplify = true;
  const Eigen::Matrix<double, 6, 6>& cov_z = queryKeyframeInDB->cov_vio_T_WB_;
  Eigen::Matrix<double, 6, 6, Eigen::RowMajor> de_dz;
  PriorFactorPose3Wrap pfw(GtsamWrap::toPose3(queryKeyframeInDB->vio_T_WB_));
  Eigen::Matrix<double, 6, 1> residual;
  pfw.toMeasurementJacobian(&de_dz, &residual);
  Eigen::Matrix<double, 6, 6> cov_e = de_dz * cov_z * de_dz.transpose();
  gtsam::SharedNoiseModel shared_noise_model =
      gtsam::noiseModel::Gaussian::Covariance(cov_e, tryToSimplify);

  OdometryFactor odom_factor(
      queryKeyframeInDB->dbowId_, GtsamWrap::toPose3(queryKeyframeInDB->vio_T_WB_), shared_noise_model);

  // Initialize PGO with first frame if needed.
  if (queryKeyframeInDB->constraintList().size() == 0u) {
    initializePGO(odom_factor);
    return true;
  }

  addOdometryFactorAndOptimize(queryKeyframeInDB);
  // compute between factor error covariance with J * cov_T_BlBq_ * J'
  gtsam::SharedNoiseModel noiseModel = createRobustNoiseModel(
      loopFrameAndMatches->relativePoseCovariance());
  LoopClosureFactor lc_factor(loopFrameAndMatches->id_,
                              loopFrameAndMatches->queryKeyframeId_,
                              GtsamWrap::toPose3(loopFrameAndMatches->T_BlBq_),
                              noiseModel);

  VLOG(1) << "LoopClosureDetector: LOOP CLOSURE detected from keyframe "
          << loopFrameAndMatches->id_ << " to keyframe "
          << loopFrameAndMatches->queryKeyframeId_;
  addLoopClosureFactorAndOptimize(lc_factor);

  // TODO(jhuai): Construct output payload.
  // option 1: save the pose for the newly added keyframe
  // option 2: save the entire pose graph estimates for all keyframes
  return false;
}

void LoopClosureDetector::detectAndDescribe(
    std::shared_ptr<const okvis::LoopQueryKeyframeMessage> query_keyframe,
    OrbDescriptorVec* descriptors_vec) {
  std::vector<cv::KeyPoint> keypoints;
  OrbDescriptor descriptors_mat;

  // Extract ORB features and construct descriptors_vec.
  orb_feature_detector_->detectAndCompute(
      query_keyframe->queryImage(), cv::Mat(), keypoints, descriptors_mat);

  int L = orb_feature_detector_->descriptorSize();
  descriptors_vec->resize(descriptors_mat.rows);

  for (size_t i = 0; i < descriptors_vec->size(); i++) {
    descriptors_vec->at(i) = cv::Mat(1, L, descriptors_mat.type());  // one row only
    descriptors_mat.row(i).copyTo(descriptors_vec->at(i).row(0));
  }
}

std::shared_ptr<okvis::KeyframeInDatabase>
LoopClosureDetector::initializeKeyframeInDatabase(
    size_t dbowId,
    std::shared_ptr<okvis::LoopQueryKeyframeMessage> queryKeyframe) const {
  std::shared_ptr<okvis::KeyframeInDatabase> queryKeyframeInDB =
      LoopClosureMethod::initializeKeyframeInDatabase(dbowId, queryKeyframe);
  if (lcd_params_->pgo_uniform_weight_ ||
      std::fabs(queryKeyframe->cov_T_WB_(0, 0)) < 1e-8) {
    // This second condition applies to estimators that cannot provide
    // covariance for poses.
    return queryKeyframeInDB;
  } else {
    size_t j = 0u;
    for (auto constraint : queryKeyframe->odometryConstraintList()) {
      Eigen::Matrix<double, 6, 6> cov_T_BqBn;
      constraint->computeRelativePoseCovariance(
          queryKeyframe->T_WB_, queryKeyframe->cov_T_WB_, &cov_T_BqBn);

      VIO::BetweenFactorPose3Wrap bfWrap(GtsamWrap::toPose3(constraint->core_.T_BrB_));
      Eigen::Matrix<double, 6, 6, Eigen::RowMajor> de_dz;
      Eigen::Matrix<double, 6, 1> autoResidual;
      bfWrap.toMeasurmentJacobian(&de_dz, &autoResidual);

      queryKeyframeInDB->setCovRawError(j, de_dz * cov_T_BqBn * de_dz.transpose());
      ++j;
    }
    return queryKeyframeInDB;
  }
}

bool LoopClosureDetector::detectLoop(
    std::shared_ptr<okvis::LoopQueryKeyframeMessage> input,
    std::shared_ptr<okvis::KeyframeInDatabase>& queryKeyframeInDB,
    std::shared_ptr<okvis::LoopFrameAndMatches>& loopFrameAndMatches) {
  // One time initialization from camera parameters.
  if (!set_intrinsics_) {
    setIntrinsics(input->cameraGeometry());
  }

  size_t dbowId = db_frames_.size();
  queryKeyframeInDB = initializeKeyframeInDatabase(dbowId, input);
  db_frames_.push_back(queryKeyframeInDB);
  vioIdToDbowId_.emplace(queryKeyframeInDB->id_, dbowId);
  // Process the StereoFrame and check for a loop closure with previous ones.
  LoopResult loop_result;

  OrbDescriptorVec descriptors_vec;
  detectAndDescribe(input, &descriptors_vec);
  FrameId frame_id = dbowId;
  loop_result.query_id_ = frame_id;

  // Create BOW representation of descriptors.
  DBoW2::BowVector bow_vec;
  DCHECK(db_BoW_);
  db_BoW_->getVocabulary()->transform(descriptors_vec, bow_vec);

  int max_possible_match_id = frame_id - lcd_params_->dist_local_;
  if (max_possible_match_id < 0) max_possible_match_id = 0;

  // Query for BoW vector matches in database.
  DBoW2::QueryResults query_result;
  db_BoW_->query(bow_vec,
                 query_result,
                 lcd_params_->max_db_results_,
                 max_possible_match_id);

  // Add current BoW vector to database.
  db_BoW_->add(bow_vec);

  if (query_result.empty()) {
    loop_result.status_ = LCDStatus::NO_MATCHES;
  } else {
    double nss_factor = 1.0;
    if (lcd_params_->use_nss_) {
      nss_factor = db_BoW_->getVocabulary()->score(bow_vec, latest_bowvec_);
      if (latest_bowvec_.size() == 0) {
        LOG(INFO) << "When ref bowvec is empty, the score is expected to be 0 ? " << nss_factor;
      }
    }

    if (lcd_params_->use_nss_ && nss_factor < lcd_params_->min_nss_factor_) {
      loop_result.status_ = LCDStatus::LOW_NSS_FACTOR;
    } else {
      // Remove low scores from the QueryResults based on nss.
      DBoW2::QueryResults::iterator query_it =
          lower_bound(query_result.begin(),
                      query_result.end(),
                      DBoW2::Result(0, lcd_params_->alpha_ * nss_factor),
                      DBoW2::Result::geq);
      if (query_it != query_result.end()) {
        query_result.resize(query_it - query_result.begin());
      }

      // Begin grouping and checking matches.
      if (query_result.empty()) {
        loop_result.status_ = LCDStatus::LOW_SCORE;
      } else {
        // Set best candidate to highest scorer.
        loop_result.match_id_ = query_result[0].Id;

        // Compute islands in the matches.
        std::vector<MatchIsland> islands;
        lcd_tp_wrapper_->computeIslands(&query_result, &islands);

        if (islands.empty()) {
          loop_result.status_ = LCDStatus::NO_GROUPS;
        } else {
          // Find the best island grouping using MatchIsland sorting.
          const MatchIsland& best_island =
              *std::max_element(islands.begin(), islands.end());

          // Run temporal constraint check on this best island.
          bool pass_temporal_constraint =
              lcd_tp_wrapper_->checkTemporalConstraint(frame_id, best_island);

          if (!pass_temporal_constraint) {
            loop_result.status_ = LCDStatus::FAILED_TEMPORAL_CONSTRAINT;
          } else {
            // Perform geometric verification check.
//            gtsam::Pose3 camCur_T_camRef_mono;
            okvis::kinematics::Transformation bodyMatch_T_bodyQuery;
            bool pass_geometric_verification = geometricVerificationCheck(
                frame_id, best_island.best_id_, &bodyMatch_T_bodyQuery);

            if (!pass_geometric_verification) {
              loop_result.status_ = LCDStatus::FAILED_GEOM_VERIFICATION;
            } else {
//              gtsam::Pose3 bodyCur_T_bodyRef_stereo;
//              bool pass_3d_pose_compute =
//                  recoverPose(loop_result.query_id_,
//                              loop_result.match_id_,
//                              camCur_T_camRef_mono,
//                              &bodyCur_T_bodyRef_stereo);

//              if (!pass_3d_pose_compute) {
//                loop_result.status_ = LCDStatus::FAILED_POSE_RECOVERY;
//              } else {
                loop_result.relative_pose_ = GtsamWrap::toPose3(bodyMatch_T_bodyQuery); // TODO(jhuai): should not it be bodyRef_T_bodyCur_stereo?
                loop_result.status_ = LCDStatus::LOOP_DETECTED;
//              }
            }
          }
        }
      }
    }
  }

  // Update latest bowvec for normalized similarity scoring (NSS).
  if (static_cast<int>(frame_id + 1) > lcd_params_->dist_local_) {
    latest_bowvec_ = bow_vec;
  } else {
    VLOG(0) << "LoopClosureDetector: Not enough frames processed.";
  }

  // Try to find a loop and update the PGO with the result if available.
  if (loop_result.isLoop()) {
    loopFrameAndMatches.reset(new okvis::LoopFrameAndMatches(
        loop_result.match_id_, loop_result.query_id_,
        GtsamWrap::toTransform(loop_result.relative_pose_.inverse())));
    loopFrameAndMatches->T_WB_ = GtsamWrap::toTransform(getPgoOptimizedPose(loop_result.match_id_));
    loopFrameAndMatches->setPoseCovariance(db_frames_[loop_result.match_id_]->cov_vio_T_WB_);
    VLOG(0) << "LoopClosureDetector: LOOP CLOSURE detected from keyframe "
            << loop_result.match_id_ << " to keyframe "
            << loop_result.query_id_;
  } else {
    loopFrameAndMatches.reset();
    VLOG(0) << "LoopClosureDetector: No loop closure detected. Reason: "
            << LoopResult::asString(loop_result.status_);
  }
  lcd_tp_wrapper_->setLatestQueryId(dbowId);
  return loop_result.isLoop();
}

/* ------------------------------------------------------------------------ */
bool LoopClosureDetector::geometricVerificationCheck(
    const FrameId& query_id,
    const FrameId& match_id,
    okvis::kinematics::Transformation* bodyMatch_T_bodyQuery) {
  // wirte an additional constructor for FrameNonCentralAbsolutePoseAdaptor
  // and follow runRansac3d2d to estimate the relative pose of the query
  // camera frame in the loop match frame which has 3d landmark in it camera frame.
  // Remember also transform the camera frame pose to body frame poses.

  // Also compute the measurement Jacobians and use the covariance inversion rule to
  // compute the covariance of the error in observation z and in turn the covariance of
  // e = log(z^-1 x^-1 y).

  CHECK_NOTNULL(bodyMatch_T_bodyQuery);
  switch (lcd_params_->geom_check_) {
    case GeomVerifOption::NISTER: {
//      return geometricVerificationNister(
//          query_id, match_id, bodyMatch_T_bodyQuery);
        return true;
    }
    case GeomVerifOption::NONE: {
      return true;
    }
    default: {
      LOG(FATAL) << "LoopClosureDetector: Incorrect geom_check_ option: "
                 << std::to_string(static_cast<int>(lcd_params_->geom_check_));
    }
  }

  return false;
}

/* ------------------------------------------------------------------------ */
bool LoopClosureDetector::recoverPose(const FrameId& query_id,
                                      const FrameId& match_id,
                                      const gtsam::Pose3& camCur_T_camRef_mono,
                                      gtsam::Pose3* bodyCur_T_bodyRef_stereo) {
  CHECK_NOTNULL(bodyCur_T_bodyRef_stereo);

  bool passed_pose_recovery = false;

  switch (lcd_params_->pose_recovery_option_) {
//    case PoseRecoveryOption::RANSAC_ARUN: {
//      passed_pose_recovery =
//          recoverPoseArun(query_id, match_id, bodyCur_T_bodyRef_stereo);
//      break;
//    }
//    case PoseRecoveryOption::GIVEN_ROT: {
//      passed_pose_recovery = recoverPoseGivenRot(
//          query_id, match_id, camCur_T_camRef_mono, bodyCur_T_bodyRef_stereo);
//      break;
//    }
    default: {
      LOG(FATAL) << "LoopClosureDetector: Incorrect pose recovery option: "
                 << std::to_string(
                        static_cast<int>(lcd_params_->pose_recovery_option_));
    }
  }

  // Use the rotation obtained from 5pt method if needed.
  // TODO(marcus): check that the rotations are close to each other!
  if (lcd_params_->use_mono_rot_ &&
      lcd_params_->pose_recovery_option_ != PoseRecoveryOption::GIVEN_ROT) {
    gtsam::Pose3 bodyCur_T_bodyRef_mono;
    transformCameraPoseToBodyPose(camCur_T_camRef_mono,
                                  &bodyCur_T_bodyRef_mono);

    const gtsam::Rot3& bodyCur_R_bodyRef_stereo =
        bodyCur_T_bodyRef_mono.rotation();
    const gtsam::Point3& bodyCur_t_bodyRef_stereo =
        bodyCur_T_bodyRef_stereo->translation();

    *bodyCur_T_bodyRef_stereo =
        gtsam::Pose3(bodyCur_R_bodyRef_stereo, bodyCur_t_bodyRef_stereo);
  }

  return passed_pose_recovery;
}

/* ------------------------------------------------------------------------ */
//const gtsam::Pose3 LoopClosureDetector::getWPoseMap() const {
//  if (W_Pose_Blkf_estimates_.size() > 1) {
//    CHECK(pgo_);
//    const gtsam::Pose3& w_Pose_Bkf_estim = W_Pose_Blkf_estimates_.back();
//    const gtsam::Pose3& w_Pose_Bkf_optimal =
//        pgo_->calculateEstimate().at<gtsam::Pose3>(
//            W_Pose_Blkf_estimates_.size() - 1);

//    return w_Pose_Bkf_optimal.between(w_Pose_Bkf_estim);
//  }

//  return gtsam::Pose3();
//}

const gtsam::Pose3 LoopClosureDetector::getPgoOptimizedPose(size_t pose_key) const {
  return pgo_->calculateEstimate().at<gtsam::Pose3>(pose_key);
}

/* ------------------------------------------------------------------------ */
const gtsam::Values LoopClosureDetector::getPGOTrajectory() const {
  CHECK(pgo_);
  return pgo_->calculateEstimate();
}

/* ------------------------------------------------------------------------ */
const gtsam::NonlinearFactorGraph LoopClosureDetector::getPGOnfg() const {
  CHECK(pgo_);
  return pgo_->getFactorsUnsafe();
}

/* ------------------------------------------------------------------------ */
// TODO(marcus): this should be parsed from CameraParams directly
void LoopClosureDetector::setIntrinsics(std::shared_ptr<const okvis::cameras::CameraBase> cam0) {
//  const CameraParams& cam_param = stereo_frame.getLeftFrame().cam_param_;
//  const CameraParams::Intrinsics& intrinsics = cam_param.intrinsics_;
  cameraGeometry_ = cam0;
  lcd_params_->image_width_ = cam0->imageWidth();
  lcd_params_->image_height_ = cam0->imageHeight();
  Eigen::VectorXd intrinsics;
  cam0->getIntrinsics(intrinsics);
  lcd_params_->focal_length_ = intrinsics[0];
  lcd_params_->principle_point_ = cv::Point2d(intrinsics[2], intrinsics[3]);
  set_intrinsics_ = true;
}

/* ------------------------------------------------------------------------ */
void LoopClosureDetector::setDatabase(const OrbDatabase& db) {
  db_BoW_ = std::make_shared<OrbDatabase>(db);
}

/* ------------------------------------------------------------------------ */
void LoopClosureDetector::setVocabulary(const OrbVocabulary& voc) {
  db_BoW_->setVocabulary(voc);
}

/* ------------------------------------------------------------------------ */
void LoopClosureDetector::print() const {
  // TODO(marcus): implement
}

/* ------------------------------------------------------------------------ */
//void LoopClosureDetector::rewriteStereoFrameFeatures(
//    const std::vector<cv::KeyPoint>& keypoints,
//    StereoFrame* stereo_frame) const {
//  CHECK_NOTNULL(stereo_frame);

//  // Populate frame keypoints with ORB features instead of the normal
//  // VIO features that came with the StereoFrame.
//  Frame* left_frame_mutable = stereo_frame->getLeftFrameMutable();
//  Frame* right_frame_mutable = stereo_frame->getRightFrameMutable();
//  CHECK_NOTNULL(left_frame_mutable);
//  CHECK_NOTNULL(right_frame_mutable);

//  // Clear all relevant fields.
//  left_frame_mutable->keypoints_.clear();
//  left_frame_mutable->versors_.clear();
//  left_frame_mutable->scores_.clear();
//  right_frame_mutable->keypoints_.clear();
//  right_frame_mutable->versors_.clear();
//  right_frame_mutable->scores_.clear();
//  stereo_frame->keypoints_3d_.clear();
//  stereo_frame->keypoints_depth_.clear();
//  stereo_frame->left_keypoints_rectified_.clear();
//  stereo_frame->right_keypoints_rectified_.clear();

//  // Reserve space in all relevant fields
//  left_frame_mutable->keypoints_.reserve(keypoints.size());
//  left_frame_mutable->versors_.reserve(keypoints.size());
//  left_frame_mutable->scores_.reserve(keypoints.size());
//  right_frame_mutable->keypoints_.reserve(keypoints.size());
//  right_frame_mutable->versors_.reserve(keypoints.size());
//  right_frame_mutable->scores_.reserve(keypoints.size());
//  stereo_frame->keypoints_3d_.reserve(keypoints.size());
//  stereo_frame->keypoints_depth_.reserve(keypoints.size());
//  stereo_frame->left_keypoints_rectified_.reserve(keypoints.size());
//  stereo_frame->right_keypoints_rectified_.reserve(keypoints.size());

//  // stereo_frame->setIsRectified(false);

//  // Add ORB keypoints.
//  for (const cv::KeyPoint& keypoint : keypoints) {
//    left_frame_mutable->keypoints_.push_back(keypoint.pt);
//    left_frame_mutable->versors_.push_back(
//        Frame::calibratePixel(keypoint.pt, left_frame_mutable->cam_param_));
//    left_frame_mutable->scores_.push_back(1.0);
//  }

//  // Automatically match keypoints in right image with those in left.
//  stereo_frame->sparseStereoMatching();

//  size_t num_kp = keypoints.size();
//  CHECK_EQ(left_frame_mutable->keypoints_.size(), num_kp);
//  CHECK_EQ(left_frame_mutable->versors_.size(), num_kp);
//  CHECK_EQ(left_frame_mutable->scores_.size(), num_kp);
//  CHECK_EQ(stereo_frame->keypoints_3d_.size(), num_kp);
//  CHECK_EQ(stereo_frame->keypoints_depth_.size(), num_kp);
//  CHECK_EQ(stereo_frame->left_keypoints_rectified_.size(), num_kp);
//  CHECK_EQ(stereo_frame->right_keypoints_rectified_.size(), num_kp);
//}

/* ------------------------------------------------------------------------ */
//cv::Mat LoopClosureDetector::computeAndDrawMatchesBetweenFrames(
//    const cv::Mat& query_img,
//    const cv::Mat& match_img,
//    const FrameId& query_id,
//    const FrameId& match_id,
//    bool cut_matches) const {
//  std::vector<std::vector<cv::DMatch>> matches;
//  std::vector<cv::DMatch> good_matches;

//  // Use the Lowe's Ratio Test only if asked.
//  double lowe_ratio = 1.0;
//  if (cut_matches) lowe_ratio = lcd_params_->lowe_ratio_;

//  // TODO(marcus): this can use computeMatchedIndices() as well...
//  descriptor_matcher_->knnMatch(db_frames_[query_id].descriptors_mat_,
//                                 db_frames_[match_id].descriptors_mat_,
//                                 matches,
//                                 2u);

//  for (const std::vector<cv::DMatch>& match : matches) {
//    if (match.at(0).distance < lowe_ratio * match.at(1).distance) {
//      good_matches.push_back(match[0]);
//    }
//  }

//  // Draw matches.
//  cv::Mat img_matches;
//  cv::drawMatches(query_img,
//                  db_frames_.at(query_id).keypoints_,
//                  match_img,
//                  db_frames_.at(match_id).keypoints_,
//                  good_matches,
//                  img_matches,
//                  cv::Scalar(255, 0, 0),
//                  cv::Scalar(255, 0, 0));

//  return img_matches;
//}

/* ------------------------------------------------------------------------ */
void LoopClosureDetector::transformCameraPoseToBodyPose(
    const gtsam::Pose3& camCur_T_camRef,
    gtsam::Pose3* bodyCur_T_bodyRef) const {
  CHECK_NOTNULL(bodyCur_T_bodyRef);
  *bodyCur_T_bodyRef =
      B_Pose_camLrect_ * camCur_T_camRef * B_Pose_camLrect_.inverse();
}

/* ------------------------------------------------------------------------ */
void LoopClosureDetector::transformBodyPoseToCameraPose(
    const gtsam::Pose3& bodyCur_T_bodyRef,
    gtsam::Pose3* camCur_T_camRef) const {
  CHECK_NOTNULL(camCur_T_camRef);
  *camCur_T_camRef =
      B_Pose_camLrect_.inverse() * bodyCur_T_bodyRef * B_Pose_camLrect_;
}

/* ------------------------------------------------------------------------ */
void LoopClosureDetector::computeMatchedIndices(const FrameId& query_id,
                                                const FrameId& match_id,
                                                std::vector<FrameId>* i_query,
                                                std::vector<FrameId>* i_match,
                                                bool cut_matches) const {
  CHECK_NOTNULL(i_query);
  CHECK_NOTNULL(i_match);

  // Get two best matches between frame descriptors.
  std::vector<DMatchVec> matches;
  double lowe_ratio = 1.0;
  if (cut_matches) lowe_ratio = lcd_params_->lowe_ratio_;

  descriptor_matcher_->knnMatch(db_frames_[query_id]->frontendDescriptors(),
                                 db_frames_[match_id]->frontendDescriptors(),
                                 matches,
                                 2u);

  // We reserve instead of resize because some of the matches will be pruned.
  const size_t& n_matches = matches.size();
  i_query->reserve(n_matches);
  i_match->reserve(n_matches);
  for (size_t i = 0; i < n_matches; i++) {
    const DMatchVec& match = matches[i];
    CHECK_EQ(match.size(), 2);
    if (match[0].distance < lowe_ratio * match[1].distance) {
      i_query->push_back(match[0].queryIdx);
      i_match->push_back(match[0].trainIdx);
    }
  }
}

/* ------------------------------------------------------------------------ */
// TODO(marcus): both geometrticVerification and recoverPose run the matching
// alg. this is wasteful. Store the matched indices as latest for use in the
// compute step
bool LoopClosureDetector::geometricVerificationNister(
    const FrameId& query_id,
    const FrameId& match_id,
    gtsam::Pose3* camCur_T_camRef_mono) {
  CHECK_NOTNULL(camCur_T_camRef_mono);

  // Find correspondences between keypoints.
  std::vector<FrameId> i_query, i_match;
  computeMatchedIndices(query_id, match_id, &i_query, &i_match, true);

//  BearingVectors query_versors, match_versors;

//  CHECK_EQ(i_query.size(), i_match.size());
//  query_versors.resize(i_query.size());
//  match_versors.resize(i_match.size());
//  for (size_t i = 0; i < i_match.size(); i++) {
//    query_versors[i] = (db_frames_[query_id].versors_[i_query[i]]);
//    match_versors[i] = (db_frames_[match_id].versors_[i_match[i]]);
//  }

//  // Recover relative pose between frames, with translation up to a scalar.
//  if (static_cast<int>(match_versors.size()) >=
//      lcd_params_->min_correspondences_) {
//    AdapterMono adapter(match_versors, query_versors);

//    // Use RANSAC to solve the central-relative-pose problem.
//    opengv::sac::Ransac<SacProblemMono> ransac;

//    ransac.sac_model_ =
//        std::make_shared<SacProblemMono>(adapter,
//                                         SacProblemMono::Algorithm::NISTER,
//                                         lcd_params_->ransac_randomize_mono_);
//    ransac.max_iterations_ = lcd_params_->max_ransac_iterations_mono_;
//    ransac.probability_ = lcd_params_->ransac_probability_mono_;
//    ransac.threshold_ = lcd_params_->ransac_threshold_mono_;

//    // Compute transformation via RANSAC.
//    bool ransac_success = ransac.computeModel();
//    VLOG(3) << "ransac 5pt size of input: " << query_versors.size()
//            << "\nransac 5pt inliers: " << ransac.inliers_.size()
//            << "\nransac 5pt iterations: " << ransac.iterations_;


//    if (!ransac_success) {
//      VLOG(3) << "LoopClosureDetector Failure: RANSAC 5pt could not solve.";
//    } else {
//      double inlier_percentage =
//          static_cast<double>(ransac.inliers_.size()) / query_versors.size();

//      if (inlier_percentage >= lcd_params_->ransac_inlier_threshold_mono_) {
//        if (ransac.iterations_ < lcd_params_->max_ransac_iterations_mono_) {
//          opengv::transformation_t transformation = ransac.model_coefficients_;
//          *camCur_T_camRef_mono =
//              UtilsOpenCV::openGvTfToGtsamPose3(transformation);

//          return true;
//        }
//      }
//    }
//  }

  return false;
}

/* ------------------------------------------------------------------------ */
//bool LoopClosureDetector::recoverPoseArun(const FrameId& query_id,
//                                          const FrameId& match_id,
//                                          gtsam::Pose3* bodyCur_T_bodyRef) {
//  CHECK_NOTNULL(bodyCur_T_bodyRef);

//  // Find correspondences between frames.
//  std::vector<FrameId> i_query, i_match;
//  computeMatchedIndices(query_id, match_id, &i_query, &i_match, false);

//  BearingVectors f_ref, f_cur;

//  // Fill point clouds with matched 3D keypoints.
//  CHECK_EQ(i_query.size(), i_match.size());
//  f_ref.resize(i_match.size());
//  f_cur.resize(i_query.size());
//  for (size_t i = 0; i < i_match.size(); i++) {
//    f_cur[i] = (db_frames_[query_id].keypoints_3d_.at(i_query[i]));
//    f_ref[i] = (db_frames_[match_id].keypoints_3d_.at(i_match[i]));
//  }

//  AdapterStereo adapter(f_ref, f_cur);
//  opengv::transformation_t transformation;

//  // Compute transform using RANSAC 3-point method (Arun).
//  opengv::sac::Ransac<SacProblemStereo> ransac;
//  ransac.sac_model_ = std::make_shared<SacProblemStereo>(
//      adapter, lcd_params_->ransac_randomize_stereo_);
//  ransac.max_iterations_ = lcd_params_->max_ransac_iterations_stereo_;
//  ransac.probability_ = lcd_params_->ransac_probability_stereo_;
//  ransac.threshold_ = lcd_params_->ransac_threshold_stereo_;

//  // Compute transformation via RANSAC.
//  bool ransac_success = ransac.computeModel();
//  VLOG(3) << "ransac 3pt size of input: " << f_ref.size()
//          << "\nransac 3pt inliers: " << ransac.inliers_.size()
//          << "\nransac 3pt iterations: " << ransac.iterations_;

//  if (!ransac_success) {
//    VLOG(3) << "LoopClosureDetector Failure: RANSAC 3pt could not solve.";
//  } else {
//    double inlier_percentage =
//        static_cast<double>(ransac.inliers_.size()) / f_ref.size();

//    if (inlier_percentage >= lcd_params_->ransac_inlier_threshold_stereo_) {
//      if (ransac.iterations_ < lcd_params_->max_ransac_iterations_stereo_) {
//        transformation = ransac.model_coefficients_;

//        // Transform pose from camera frame to body frame.
//        gtsam::Pose3 camCur_T_camRef =
//            UtilsOpenCV::openGvTfToGtsamPose3(transformation);
//        transformCameraPoseToBodyPose(camCur_T_camRef, bodyCur_T_bodyRef);

//        return true;
//      }
//    }
//  }

//  return false;
//}

/* ------------------------------------------------------------------------ */
//bool LoopClosureDetector::recoverPoseGivenRot(
//    const FrameId& query_id,
//    const FrameId& match_id,
//    const gtsam::Pose3& camCur_T_camRef_mono,
//    gtsam::Pose3* bodyCur_T_bodyRef) {
//  CHECK_NOTNULL(bodyCur_T_bodyRef);

//  const gtsam::Rot3& R = camCur_T_camRef_mono.rotation();

//  // Find correspondences between frames.
//  std::vector<FrameId> i_query, i_match;
//  computeMatchedIndices(query_id, match_id, &i_query, &i_match, true);

//  // Fill point clouds with matched 3D keypoints.
//  const size_t& n_matches = i_match.size();
//  CHECK_EQ(i_query.size(), n_matches);

//  if (n_matches > 0) {
//    std::vector<double> x_coord(n_matches);
//    std::vector<double> y_coord(n_matches);
//    std::vector<double> z_coord(n_matches);

//    for (size_t i = 0; i < n_matches; i++) {
//      const gtsam::Vector3& keypoint_cur =
//          db_frames_[query_id].keypoints_3d_.at(i_query[i]);
//      const gtsam::Vector3& keypoint_ref =
//          db_frames_[match_id].keypoints_3d_.at(i_match[i]);

//      gtsam::Vector3 rotated_keypoint_diff = keypoint_ref - (R * keypoint_cur);
//      x_coord[i] = rotated_keypoint_diff[0];
//      y_coord[i] = rotated_keypoint_diff[1];
//      z_coord[i] = rotated_keypoint_diff[2];
//    }

//    CHECK_EQ(x_coord.size(), n_matches);
//    CHECK_EQ(y_coord.size(), n_matches);
//    CHECK_EQ(z_coord.size(), n_matches);

//    // TODO(marcus): decide between median check and scaling factor
//    std::sort(x_coord.begin(), x_coord.end());
//    std::sort(y_coord.begin(), y_coord.end());
//    std::sort(z_coord.begin(), z_coord.end());

//    gtsam::Point3 scaled_t(
//        x_coord.at(std::floor(static_cast<int>(x_coord.size() / 2))),
//        y_coord.at(std::floor(static_cast<int>(y_coord.size() / 2))),
//        z_coord.at(std::floor(static_cast<int>(z_coord.size() / 2))));

//    // Transform pose from camera frame to body frame.
//    gtsam::Pose3 camCur_T_camRef_stereo(R, scaled_t);
//    transformCameraPoseToBodyPose(camCur_T_camRef_stereo, bodyCur_T_bodyRef);

//    return true;
//  }

//  return false;

  // TODO(marcus): input should alwasy be with unit translation, no need to
  // check
  // gtsam::Point3 unit_t = camCur_T_camRef_mono.translation() /
  // camCur_T_camRef_mono.translation().norm();
  // // Get sacling factor for translation by averaging across point cloud.
  // double scaling_factor = 0.0;
  // for (size_t i=0; i<f_ref.size(); i++) {
  //   gtsam::Vector3 keypoint_ref = f_ref[i];
  //   gtsam::Vector3 keypoint_cur = f_cur[i];
  //
  //   gtsam::Vector3 rotated_keypoint_diff =
  //       keypoint_ref - (R*keypoint_cur);
  //
  //   double cur_scaling_factor = rotated_keypoint_diff.dot(unit_t);
  //   if (cur_scaling_factor < 0) {
  //     cur_scaling_factor *= -1.0;
  //   }
  //   scaling_factor += cur_scaling_factor;
  // }
  // scaling_factor /= f_ref.size();
  //
  // if (scaling_factor < 0) {
  //   scaling_factor *= -1.0;
  // }
  //
  // gtsam::Point3 scaled_t(unit_t[0] * scaling_factor,
  //     unit_t[1] * scaling_factor, unit_t[2] * scaling_factor);
//}

/* ------------------------------------------------------------------------ */
//void LoopClosureDetector::initializePGO() {
//  gtsam::NonlinearFactorGraph init_nfg;
//  gtsam::Values init_val;
//  init_val.insert(gtsam::Symbol(0), gtsam::Pose3());

//  CHECK(pgo_);
//  pgo_->update(init_nfg, init_val);
//}

/* ------------------------------------------------------------------------ */
void LoopClosureDetector::initializePGO(const OdometryFactor& factor) {
  gtsam::NonlinearFactorGraph init_nfg;
  gtsam::Values init_val;

  init_val.insert(gtsam::Symbol(0), factor.W_Pose_Blkf_);

  init_nfg.add(gtsam::PriorFactor<gtsam::Pose3>(
      gtsam::Symbol(0), factor.W_Pose_Blkf_, factor.noise_));

  CHECK(pgo_);
  pgo_->update(init_nfg, init_val);
}

/* ------------------------------------------------------------------------ */
// TODO(marcus): only add nodes if they're x dist away from previous node
// TODO(marcus): consider making the keys of OdometryFactor minus one each so
// that the extra check in here isn't needed...
void LoopClosureDetector::addOdometryFactorAndOptimize(
    std::shared_ptr<const okvis::KeyframeInDatabase> keyframeInDB) {
  auto constraintList = keyframeInDB->constraintList();

  auto firstNeighbor = constraintList.at(0);
  size_t dbowIdLastKf = vioIdToDbowId_.find(firstNeighbor->id_)->second;
  CHECK_EQ(dbowIdLastKf + 1, keyframeInDB->dbowId_);

  gtsam::NonlinearFactorGraph nfgSequentialOdometry;
  gtsam::Values valueSequentialOdometry;
  valueSequentialOdometry.insert(gtsam::Symbol(keyframeInDB->dbowId_),
                                 GtsamWrap::toPose3(keyframeInDB->T_WB_));

  gtsam::SharedNoiseModel noiseModel = createRobustNoiseModel(firstNeighbor->covRawError_);
  nfgSequentialOdometry.add(gtsam::BetweenFactor<gtsam::Pose3>(gtsam::Symbol(dbowIdLastKf),
                                             gtsam::Symbol(keyframeInDB->dbowId_),
                                             GtsamWrap::toPose3(firstNeighbor->T_BrB_.inverse()),
                                             noiseModel));
  pgo_->update(nfgSequentialOdometry, valueSequentialOdometry);

  // non-sequential odometry constraints.
  gtsam::NonlinearFactorGraph nfg;
  gtsam::Values value;
  for (auto iter = ++constraintList.begin(); iter != constraintList.end(); ++iter) {
    size_t dbowIdOldKf = vioIdToDbowId_.find((*iter)->id_)->second;
    gtsam::SharedNoiseModel noiseModel = createRobustNoiseModel((*iter)->covRawError_);
    nfg.add(gtsam::BetweenFactor<gtsam::Pose3>(
        gtsam::Symbol(dbowIdOldKf), gtsam::Symbol(keyframeInDB->dbowId_),
        GtsamWrap::toPose3((*iter)->T_BrB_.inverse()), noiseModel));
  }
  pgo_->update(nfg, value, KimeraRPGO::FactorType::NONSEQUENTIAL_ODOMETRY);
}

/* ------------------------------------------------------------------------ */
void LoopClosureDetector::addLoopClosureFactorAndOptimize(
    const LoopClosureFactor& factor) {
  gtsam::NonlinearFactorGraph nfg;
  // TODO(jhuai): watch out degeneracy in covariance, we may directly pass square root info.
  nfg.add(gtsam::BetweenFactor<gtsam::Pose3>(gtsam::Symbol(factor.ref_key_),
                                             gtsam::Symbol(factor.cur_key_),
                                             factor.ref_Pose_cur_,
                                             factor.noise_));

  CHECK(pgo_);
  pgo_->update(nfg);
}

}  // namespace VIO
