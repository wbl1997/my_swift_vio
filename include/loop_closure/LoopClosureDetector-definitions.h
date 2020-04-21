/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   LoopClosureDetector-definitions.h
 * @brief  Definitions for LoopClosureDetector
 * @author Marcus Abate
 * @author Antoni Rosinol
 * @author Luca Carlone
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

//#include "kimera-vio/common/vio_types.h"
//#include "kimera-vio/frontend/StereoFrame.h"
#include "okvis/class_macros.hpp"
#include <okvis/Time.hpp>
#include <opencv2/core.hpp>
#include <Eigen/Core>

namespace VIO {

typedef cv::Mat OrbDescriptor;
typedef std::vector<OrbDescriptor> OrbDescriptorVec;
typedef uint64_t FrameId;
typedef okvis::Time Timestamp;
typedef std::unordered_map<FrameId, Timestamp> FrameIDTimestampMap;


enum class GeomVerifOption : int { NISTER, NONE };

enum class PoseRecoveryOption : int { RANSAC_ARUN, GIVEN_ROT };

//struct LCDFrame {
//  LCDFrame() {}
//  LCDFrame(const Timestamp& timestamp,
//           const FrameId& id,
//           const FrameId& id_kf,
//           const std::vector<cv::KeyPoint>& keypoints,
//           const std::vector<gtsam::Vector3>& keypoints_3d,
//           const OrbDescriptorVec& descriptors_vec,
//           const OrbDescriptor& descriptors_mat,
//           const BearingVectors& versors)
//      : timestamp_(timestamp),
//        id_(id),
//        id_kf_(id_kf),
//        keypoints_(keypoints),
//        keypoints_3d_(keypoints_3d),
//        descriptors_vec_(descriptors_vec),
//        descriptors_mat_(descriptors_mat),
//        versors_(versors) {}

//  Timestamp timestamp_;
//  FrameId id_;
//  FrameId id_kf_;
//  std::vector<cv::KeyPoint> keypoints_;
//  std::vector<gtsam::Vector3> keypoints_3d_;
//  OrbDescriptorVec descriptors_vec_;
//  OrbDescriptor descriptors_mat_;
//  BearingVectors versors_;
//};  // struct LCDFrame

struct MatchIsland {
  MatchIsland()
      : start_id_(0),
        end_id_(0),
        island_score_(0),
        best_id_(0),
        best_score_(0) {}

  MatchIsland(const FrameId& start, const FrameId& end)
      : start_id_(start),
        end_id_(end),
        island_score_(0),
        best_id_(0),
        best_score_(0) {}

  MatchIsland(const FrameId& start, const FrameId& end, const double& score)
      : start_id_(start),
        end_id_(end),
        island_score_(score),
        best_id_(0),
        best_score_(0) {}

  inline bool operator<(const MatchIsland& other) const {
    return island_score_ < other.island_score_;
  }

  inline bool operator>(const MatchIsland& other) const {
    return island_score_ > other.island_score_;
  }

  inline size_t size() const { return end_id_ - start_id_ + 1; }

  inline void clear() {
    start_id_ = 0;
    end_id_ = 0;
    island_score_ = 0;
    best_id_ = 0;
    best_score_ = 0;
  }

  FrameId start_id_;
  FrameId end_id_;
  double island_score_;
  FrameId best_id_;
  double best_score_;
};  // struct MatchIsland

class LCDStatus {
 public:
  enum {
    LOOP_DETECTED = 0,
    NO_MATCHES,
    LOW_NSS_FACTOR,
    LOW_SCORE,
    NO_GROUPS,
    FAILED_TEMPORAL_CONSTRAINT,
    FAILED_GEOM_VERIFICATION,
    FAILED_POSE_RECOVERY
  } status_;

  inline bool isLoop() const { return status_ == LCDStatus::LOOP_DETECTED; }

  std::string asString() {
    std::string status_str = "";
    switch (status_) {
      case LCDStatus::LOOP_DETECTED: {
        status_str = "LOOP_DETECTED";
        break;
      }
      case LCDStatus::NO_MATCHES: {
        status_str = "NO_MATCHES";
        break;
      }
      case LCDStatus::LOW_NSS_FACTOR: {
        status_str = "LOW_NSS_FACTOR";
        break;
      }
      case LCDStatus::LOW_SCORE: {
        status_str = "LOW_SCORE";
        break;
      }
      case LCDStatus::NO_GROUPS: {
        status_str = "NO_GROUPS";
        break;
      }
      case LCDStatus::FAILED_TEMPORAL_CONSTRAINT: {
        status_str = "FAILED_TEMPORAL_CONSTRAINT";
        break;
      }
      case LCDStatus::FAILED_GEOM_VERIFICATION: {
        status_str = "FAILED_GEOM_VERIFICATION";
        break;
      }
      case LCDStatus::FAILED_POSE_RECOVERY: {
        status_str = "FAILED_POSE_RECOVERY";
        break;
      }
    }
    return status_str;
  }
};

struct LcdDebugInfo {
  LcdDebugInfo() = default;

  Timestamp timestamp_;
  LCDStatus loop_result_;

  size_t mono_input_size_;
  size_t mono_inliers_;
  int mono_iter_;

  size_t stereo_input_size_;
  size_t stereo_inliers_;
  int stereo_iter_;

  size_t pgo_size_;
  size_t pgo_lc_count_;
  size_t pgo_lc_inliers_;
};  // struct LcdDebugInfo

struct LoopClosureFactor {
  LoopClosureFactor(const FrameId& ref_key,
                    const FrameId& cur_key,
                    const gtsam::Pose3& ref_Pose_cur,
                    const gtsam::SharedNoiseModel& noise)
      : ref_key_(ref_key),
        cur_key_(cur_key),
        ref_Pose_cur_(ref_Pose_cur),
        noise_(noise) {}

  const FrameId ref_key_;
  const FrameId cur_key_;
  const gtsam::Pose3 ref_Pose_cur_;
  const gtsam::SharedNoiseModel noise_;
};  // struct LoopClosureFactor

//struct LcdInput {
//  KIMERA_POINTER_TYPEDEFS(LcdInput);
//  KIMERA_DELETE_COPY_CONSTRUCTORS(LcdInput);
//  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//  LcdInput(const Timestamp& timestamp_kf,
//           const FrameId& cur_kf_id,
//           const StereoFrame& stereo_frame,
//           const gtsam::Pose3& W_Pose_Blkf)
//      : timestamp_kf_(timestamp_kf),
//        cur_kf_id_(cur_kf_id),
//        stereo_frame_(stereo_frame),
//        W_Pose_Blkf_(W_Pose_Blkf) {}

//  const Timestamp timestamp_kf_;
//  const FrameId cur_kf_id_;
//  const StereoFrame stereo_frame_;
//  const gtsam::Pose3 W_Pose_Blkf_;
//};

struct LcdOutput {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  POINTER_TYPEDEFS(LcdOutput);
  DELETE_COPY_CONSTRUCTORS(LcdOutput);


  LcdOutput(bool is_loop_closure,
            const Timestamp& timestamp_kf,
            const Timestamp& timestamp_query,
            const Timestamp& timestamp_match,
            const FrameId& id_match,
            const FrameId& id_recent,
            const gtsam::Pose3& relative_pose,
            const gtsam::Pose3& W_Pose_Map,
            const gtsam::Values& states,
            const gtsam::NonlinearFactorGraph& nfg)
      : is_loop_closure_(is_loop_closure),
        timestamp_kf_(timestamp_kf),
        timestamp_query_(timestamp_query),
        timestamp_match_(timestamp_match),
        id_match_(id_match),
        id_recent_(id_recent),
        relative_pose_(relative_pose),
        W_Pose_Map_(W_Pose_Map),
        states_(states),
        nfg_(nfg) {}

  LcdOutput()
      : is_loop_closure_(false),
        timestamp_kf_(0),
        timestamp_query_(0),
        timestamp_match_(0),
        id_match_(0),
        id_recent_(0),
        relative_pose_(gtsam::Pose3()),
        W_Pose_Map_(gtsam::Pose3()),
        states_(gtsam::Values()),
        nfg_(gtsam::NonlinearFactorGraph()) {}

  // TODO(marcus): inlude stats/score of match
  bool is_loop_closure_;
  Timestamp timestamp_kf_;
  Timestamp timestamp_query_;
  Timestamp timestamp_match_;
  FrameId id_match_;
  FrameId id_recent_;
  gtsam::Pose3 relative_pose_;
  gtsam::Pose3 W_Pose_Map_;
  gtsam::Values states_;
  gtsam::NonlinearFactorGraph nfg_;
};

}  // namespace VIO
