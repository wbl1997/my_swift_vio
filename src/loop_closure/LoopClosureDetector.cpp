/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   LoopClosureDetector.cpp
 * @brief  Pipeline for detection and optimization of Loop Closures between frames.
 * @author Marcus Abate
 * @author Jianzhu Huai
 */
#include "loop_closure/LoopClosureDetector.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <Eigen/Core>

#include "msckf/ceres/tiny_solver.h"

#include <eigen/matrix_sqrt.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/absolute_pose/FrameNoncentralAbsoluteAdapter.hpp>
#include <opengv/sac_problems/absolute_pose/FrameAbsolutePoseSacProblem.hpp>

#include <KimeraRPGO/RobustSolver.h>

#include "loop_closure/GtsamWrap.hpp"

#include <gtsam/inference/Symbol.h>

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
    const okvis::KeyframeInDatabase& queryKeyframeInDB,
    std::shared_ptr<const okvis::LoopFrameAndMatches> loopFrameAndMatches) {
  // Initialize PGO with first frame if needed.
  if (queryKeyframeInDB.constraintList().size() == 0u) {
    const Eigen::Matrix<double, 6, 6>& cov_z = queryKeyframeInDB.cov_vio_T_WB_;
    Eigen::Matrix<double, 6, 6, Eigen::RowMajor> de_dz;
    PriorFactorPose3Wrap pfw(GtsamWrap::toPose3(queryKeyframeInDB.vio_T_WB_));
    Eigen::Matrix<double, 6, 1> residual;
    pfw.toMeasurementJacobian(&de_dz, &residual);
    Eigen::Matrix<double, 6, 6> cov_e = de_dz * cov_z * de_dz.transpose();
    bool tryToSimplify = true;
    gtsam::SharedNoiseModel shared_noise_model =
        gtsam::noiseModel::Gaussian::Covariance(cov_e, tryToSimplify);

    OdometryFactor odom_factor(queryKeyframeInDB.dbowId_,
                               GtsamWrap::toPose3(queryKeyframeInDB.vio_T_WB_),
                               shared_noise_model);
    initializePGO(odom_factor);
    return true;
  }

  addOdometryFactors(queryKeyframeInDB);

  if (loopFrameAndMatches) {
    VLOG(1) << "LoopClosureDetector: Loop closure detected from keyframe "
            << loopFrameAndMatches->id_ << " to keyframe "
            << loopFrameAndMatches->queryKeyframeId_;

    gtsam::SharedNoiseModel noiseModel =
        createRobustNoiseModelSqrtR(loopFrameAndMatches->relativePoseSqrtInfo());
    gtsam::NonlinearFactorGraph nfg;
    nfg.add(gtsam::BetweenFactor<gtsam::Pose3>(gtsam::Symbol(loopFrameAndMatches->dbowId_),
                                               gtsam::Symbol(loopFrameAndMatches->queryKeyframeDbowId_),
                                               GtsamWrap::toPose3(loopFrameAndMatches->T_BlBq_),
                                               noiseModel));

    pgo_->update(nfg);
  }
  // TODO(jhuai): Construct output payload.
  // option 1: save the pose for the newly added keyframe
  // option 2: save the entire pose graph estimates for all keyframes
  return false;
}

void LoopClosureDetector::detectAndDescribe(
    const okvis::LoopQueryKeyframeMessage& query_keyframe,
    OrbDescriptorVec* descriptors_vec) {
  std::vector<cv::KeyPoint> keypoints;
  OrbDescriptor descriptors_mat;

  // Extract ORB features and construct descriptors_vec.
  orb_feature_detector_->detectAndCompute(
      query_keyframe.queryImage(), cv::Mat(), keypoints, descriptors_mat);

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
    const okvis::LoopQueryKeyframeMessage& queryKeyframe) const {
  std::shared_ptr<okvis::KeyframeInDatabase> queryKeyframeInDB =
      LoopClosureMethod::initializeKeyframeInDatabase(dbowId, queryKeyframe);
  if (lcd_params_->pgo_uniform_weight_ ||
      std::fabs(queryKeyframe.cov_T_WB_(0, 0)) < 1e-8) {
    // This second condition applies to estimators that cannot provide
    // covariance for poses.
    return queryKeyframeInDB;
  } else {
    size_t j = 0u;
    for (auto constraint : queryKeyframe.odometryConstraintList()) {
      Eigen::Matrix<double, 6, 6> cov_T_BnBq;
      constraint->computeRelativePoseCovariance(
          queryKeyframe.T_WB_, queryKeyframe.cov_T_WB_, &cov_T_BnBq);

      VIO::BetweenFactorPose3Wrap bfWrap(GtsamWrap::toPose3(constraint->core_.T_BBr_));
      Eigen::Matrix<double, 6, 6, Eigen::RowMajor> de_dz;
      Eigen::Matrix<double, 6, 1> autoResidual;
      bfWrap.toMeasurmentJacobian(&de_dz, &autoResidual);

      queryKeyframeInDB->setSquareRootInfoFromCovariance(
            j, de_dz * cov_T_BnBq * de_dz.transpose());
      ++j;
    }
    return queryKeyframeInDB;
  }
}

bool LoopClosureDetector::detectLoop(
    std::shared_ptr<const okvis::LoopQueryKeyframeMessage> input,
    std::shared_ptr<okvis::KeyframeInDatabase>& queryKeyframeInDB,
    std::shared_ptr<okvis::LoopFrameAndMatches>& loopFrameAndMatches) {
  // One time initialization from camera parameters.
//  if (!set_intrinsics_) {
//    setIntrinsics(input->cameraGeometry());
//  }

  size_t dbowId = db_frames_.size();
  queryKeyframeInDB = initializeKeyframeInDatabase(dbowId, *input);
  db_frames_.push_back(queryKeyframeInDB);
  vioIdToDbowId_.emplace(queryKeyframeInDB->id_, dbowId);
  // Process the StereoFrame and check for a loop closure with previous ones.
  LoopResult loop_result;

  OrbDescriptorVec descriptors_vec;
  detectAndDescribe(*input, &descriptors_vec);
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
            bool pass_geometric_verification = geometricVerificationCheck(
                  *input, frame_id, best_island.best_id_, &loopFrameAndMatches);

            if (!pass_geometric_verification) {
              loop_result.status_ = LCDStatus::FAILED_GEOM_VERIFICATION;
            } else {
              loop_result.status_ = LCDStatus::LOOP_DETECTED;
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

  if (loop_result.isLoop()) {
    VLOG(0) << "LoopClosureDetector: LOOP CLOSURE detected from keyframe "
            << loop_result.match_id_ << " to keyframe "
            << loop_result.query_id_;
  } else {
    VLOG(0) << "LoopClosureDetector: No loop closure detected. Reason: "
            << LoopResult::asString(loop_result.status_);
  }
  lcd_tp_wrapper_->setLatestQueryId(dbowId);
  return loop_result.isLoop();
}

/* ------------------------------------------------------------------------ */
bool LoopClosureDetector::geometricVerificationCheck(
    const okvis::LoopQueryKeyframeMessage& queryKeyframe,
    const FrameId query_id, const FrameId match_id,
    std::shared_ptr<okvis::LoopFrameAndMatches>* loopFrameAndMatches) {
  std::shared_ptr<const okvis::KeyframeInDatabase> loopFrame =
      db_frames_[match_id];
  std::vector<DMatchVec> matches;
  // match descriptors associated with earlier landmarks to descriptors of query
  // keyframe.
  descriptor_matcher_->knnMatch(loopFrame->frontendDescriptors(),
                                queryKeyframe.getDescriptors(), matches, 2u);
  double lowe_ratio = lcd_params_->lowe_ratio_;

  const size_t n_matches = matches.size();
  std::vector<size_t> pointIndices;
  std::vector<size_t> keypointIndices;
  pointIndices.reserve(n_matches);
  keypointIndices.reserve(n_matches);
  for (size_t i = 0; i < n_matches; i++) {
    const DMatchVec& match = matches[i];
    if (match[0].distance < lowe_ratio * match[1].distance) {
      pointIndices.push_back(match[0].queryIdx);
      keypointIndices.push_back(match[0].trainIdx);
    }
  }
  // RANSAC 3d 2d WITH OpenGV, alternatively, use opencv solvePnPRansac().
  // The camera intrinsics are carried inside nframe.
  // If the estimator estimates the camera parameters, it is possible to
  // update these parameters in nframe before passing it to the loop closure
  // module.

  loopFrameAndMatches->reset();
  int numInliers = 0;

  // absolute pose adapter for Kneip toolchain
  opengv::absolute_pose::FrameNoncentralAbsoluteAdapter adapter(
      loopFrame->landmarkPositionList(), pointIndices, keypointIndices,
      okvis::LoopQueryKeyframeMessage::kQueryCameraIndex,
      queryKeyframe.NFrame());

  size_t numCorrespondences = adapter.getNumberCorrespondences();
  if (numCorrespondences >= 5) {
    // create a RelativePoseSac problem and RANSAC
    opengv::sac::Ransac<
        opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem>
        ransac;
    std::shared_ptr<
        opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem>
        absposeproblem_ptr(
            new opengv::sac_problems::absolute_pose::
                FrameAbsolutePoseSacProblem(
                    adapter, opengv::sac_problems::absolute_pose::
                                 FrameAbsolutePoseSacProblem::Algorithm::GP3P));
    ransac.sac_model_ = absposeproblem_ptr;
    ransac.threshold_ = 9;
    ransac.max_iterations_ = lcd_params_->max_ransac_iterations_stereo_;
    // initial guess not needed...
    // run the ransac
    bool ransac_success = ransac.computeModel(0);

    if (!ransac_success) {
      VLOG(3) << "LoopClosureDetector Failure: RANSAC 3D/2D could not solve.";
    } else {
      // assign transformation
      numInliers = ransac.inliers_.size();
      double inlier_percentage =
          static_cast<double>(numInliers) / numCorrespondences;

      if (inlier_percentage >= lcd_params_->ransac_inlier_threshold_stereo_) {
        if (ransac.iterations_ < lcd_params_->max_ransac_iterations_stereo_) {
          Eigen::Matrix4d T_BlBq_mat = Eigen::Matrix4d::Identity();
          T_BlBq_mat.topLeftCorner<3, 4>() = ransac.model_coefficients_;
          okvis::kinematics::Transformation T_BlBq(T_BlBq_mat);

          // collect inliers
          AlignedVector<Eigen::Vector3d> pointInliers;
          AlignedVector<Eigen::Vector3d> bearingInliers;
          adapter.getInlierPoints(ransac.inliers_, &pointInliers);
          adapter.getInlierBearings(ransac.inliers_, &bearingInliers);

          // LM optimization of T_BlBq.
          StackedProjectionFactorDynamic stackedProjectionFactor(
                pointInliers, bearingInliers, *queryKeyframe.NFrame()->T_SC(
                  okvis::LoopQueryKeyframeMessage::kQueryCameraIndex));
          Eigen::Matrix<double, 7, 1> estimated_T_BlBq_coeffs = T_BlBq.coeffs();

          GtsamPose3Parameterization localParameterization;
          msckf::ceres::TinySolver<StackedProjectionFactorDynamic> solver(&localParameterization);
          solver.options.max_num_iterations = 15;
          solver.Solve(stackedProjectionFactor, &estimated_T_BlBq_coeffs);
          okvis::kinematics::Transformation estimated_T_WS;
          estimated_T_WS.setCoeffs(estimated_T_BlBq_coeffs);
          LOG(INFO) << "T_BlBq: opengv:" << T_BlBq.coeffs().transpose()
                    << "\nTiny Solver: " << estimated_T_WS.coeffs().transpose();
          if (solver.summary.status !=
                           msckf::ceres::TinySolver<
                               StackedProjectionFactorDynamic>::Status::HIT_MAX_ITERATIONS) {
          // compute info of T_BlBq.
          Eigen::Matrix<double, -1, 6> jacColMajor(numInliers * 2, 6);
          Eigen::Matrix<double, -1, 1> residuals(numInliers * 2, 1);
          stackedProjectionFactor(T_BlBq.coeffs().data(), residuals.data(), jacColMajor.data());

          // info (inverse of covariance) of T_BlBq's perturbation as defined in
          // gtsam::Pose3.
          Eigen::Matrix<double, 6, 6> lambda_B = jacColMajor.transpose() * jacColMajor;

          Eigen::Matrix<double, 6, 6> choleskyFactor;
          sm::eigen::computeMatrixSqrt(lambda_B, choleskyFactor);
          loopFrameAndMatches->reset(new okvis::LoopFrameAndMatches(
              db_frames_[match_id]->id_, db_frames_[match_id]->stamp_, match_id,
              db_frames_[query_id]->id_, db_frames_[query_id]->stamp_, query_id,
              T_BlBq));
          gtsam::Values estimates = pgo_->calculateEstimate();
          bool keyExist = estimates.exists(match_id);
          if (keyExist) {
             gtsam::Pose3 pgo_T_WBl = estimates.at<gtsam::Pose3>(match_id);
             (*loopFrameAndMatches)->pgo_T_WBl_ = GtsamWrap::toTransform(pgo_T_WBl);
          } else {
             (*loopFrameAndMatches)->pgo_T_WBl_ = loopFrame->vio_T_WB_;
             LOG(WARNING) << "Pose of dbow key " << match_id << " not found in PGO estimates!";
          }

          (*loopFrameAndMatches)->setPoseCovariance(loopFrame->cov_vio_T_WB_);
          // The perturbation in T_BlBq equals the perturbation in
          // gtsam::BetweenFactor's unwhitened error by first order
          // approximation. so we let them have the same covariance/sqrt info.
          (*loopFrameAndMatches)
              ->setRelativePoseSqrtInfo(choleskyFactor.transpose());

          // Also record loop factor in queryKeyframeInDB constraint list.
          std::shared_ptr<okvis::KeyframeInDatabase> queryFrameInDB =
              db_frames_[query_id];
          std::shared_ptr<const okvis::KeyframeInDatabase> matchFrameInDB =
              db_frames_[match_id];
          std::shared_ptr<okvis::NeighborConstraintInDatabase> constraint(
                new  okvis::NeighborConstraintInDatabase(
                  matchFrameInDB->id_, matchFrameInDB->stamp_,
                  T_BlBq, okvis::PoseConstraintType::LoopClosure));
          constraint->squareRootInfo_ = choleskyFactor.transpose();
          queryFrameInDB->addLoopConstraint(constraint);
          }
        }
      }
    }
  }

  return loopFrameAndMatches == nullptr;
}


/* ------------------------------------------------------------------------ */
const gtsam::Values LoopClosureDetector::getPGOTrajectory() const {
  return pgo_->calculateEstimate();
}

/* ------------------------------------------------------------------------ */
const gtsam::NonlinearFactorGraph LoopClosureDetector::getPGOnfg() const {
  return pgo_->getFactorsUnsafe();
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
void LoopClosureDetector::computeMatchedIndices(const FrameId& query_id,
                                                const FrameId& match_id,
                                                std::vector<int>* i_query,
                                                std::vector<int>* i_match,
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


void LoopClosureDetector::initializePGO() {
  gtsam::NonlinearFactorGraph init_nfg;
  gtsam::Values init_val;
  init_val.insert(gtsam::Symbol(0), gtsam::Pose3());

  CHECK(pgo_);
  pgo_->update(init_nfg, init_val);
}

/* ------------------------------------------------------------------------ */
void LoopClosureDetector::initializePGO(const OdometryFactor& factor) {
  gtsam::NonlinearFactorGraph init_nfg;
  gtsam::Values init_val;

  init_val.insert(gtsam::Symbol(0), factor.W_Pose_Blkf_);

  init_nfg.add(gtsam::PriorFactor<gtsam::Pose3>(
      gtsam::Symbol(0), factor.W_Pose_Blkf_, factor.noise_));

  pgo_->update(init_nfg, init_val);
}

/* ------------------------------------------------------------------------ */
// TODO(marcus): only add nodes if they're x dist away from previous node
void LoopClosureDetector::addOdometryFactors(
    const okvis::KeyframeInDatabase& keyframeInDB) {
  auto constraintList = keyframeInDB.constraintList();
  auto firstNeighbor = constraintList.at(0);
  size_t dbowIdLastKf = vioIdToDbowId_.find(firstNeighbor->id_)->second;
  CHECK_EQ(dbowIdLastKf + 1, keyframeInDB.dbowId_);

  gtsam::NonlinearFactorGraph nfgSequentialOdometry;
  gtsam::Values valueSequentialOdometry;
  // We do not use pgo estimates to correct vio estimates for initializing
  // a pose because its constraint will pull it to the correct pose during PGO.
  valueSequentialOdometry.insert(gtsam::Symbol(keyframeInDB.dbowId_),
                                 GtsamWrap::toPose3(keyframeInDB.vio_T_WB_));

  gtsam::SharedNoiseModel noiseModel = createRobustNoiseModelSqrtR(firstNeighbor->squareRootInfo_);
  nfgSequentialOdometry.add(gtsam::BetweenFactor<gtsam::Pose3>(gtsam::Symbol(dbowIdLastKf),
                                             gtsam::Symbol(keyframeInDB.dbowId_),
                                             GtsamWrap::toPose3(firstNeighbor->T_BBr_),
                                             noiseModel));
  // no optimization will be performed.
  pgo_->update(nfgSequentialOdometry, valueSequentialOdometry);

  // non-sequential odometry constraints.
  gtsam::NonlinearFactorGraph nfg;
  gtsam::Values value;
  for (auto iter = ++constraintList.begin(); iter != constraintList.end(); ++iter) {
    size_t dbowIdOldKf = vioIdToDbowId_.find((*iter)->id_)->second;
    gtsam::SharedNoiseModel noiseModel = createRobustNoiseModelSqrtR((*iter)->squareRootInfo_);
    nfg.add(gtsam::BetweenFactor<gtsam::Pose3>(
        gtsam::Symbol(dbowIdOldKf), gtsam::Symbol(keyframeInDB.dbowId_),
        GtsamWrap::toPose3((*iter)->T_BBr_), noiseModel));
  }
  // no optimization will be performed.
  pgo_->update(nfg, value, KimeraRPGO::FactorType::NONSEQUENTIAL_ODOMETRY);
}


}  // namespace VIO
