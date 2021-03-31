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

enum class LoopClosureMethodType {
  Mock = 0u,
  //! Bag of Words approach
  OrbBoW = 1u,
};

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

}  // namespace VIO
