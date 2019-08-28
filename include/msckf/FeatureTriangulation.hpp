﻿/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

/* notation specific to this header file
 * T_cam0_cam1 takes a vector from the cam0 frame to the cam1 frame.
 * R_cam0_cam1, t_cam0_cam1 are defined accordingly
 */

#ifndef MSCKF_VIO_FEATURE_HPP_
#define MSCKF_VIO_FEATURE_HPP_

#include <iostream>
#include <map>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

// #include "math_utils.hpp"

namespace msckf_vio {

typedef uint64_t StateIDType;
struct CAMState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // An unique identifier for the CAM state.
  StateIDType id;

  // Orientation
  // Take a vector from the world frame to the camera frame.
  Eigen::Quaterniond orientation;

  // Position of the camera frame in the world frame.
  Eigen::Vector3d position;

  CAMState(): id(0u),
    orientation(1, 0, 0, 0),
    position(Eigen::Vector3d::Zero()) {}

  CAMState(
      const StateIDType& new_id,
      const Eigen::Quaterniond& R_C_W, const Eigen::Vector3d& t_W_C):
      id(new_id),
    orientation(R_C_W),
    position(t_W_C) {}

  CAMState(const CAMState& rhs)
      : id(rhs.id), orientation(rhs.orientation), position(rhs.position) {}

  CAMState(CAMState&& rhs)
      : id{std::move(rhs.id)},
        orientation{std::move(rhs.orientation)},
        position{std::move(rhs.position)} {}

  CAMState& operator= (const CAMState& rhs) {
      if (this != &rhs) {
          id = rhs.id;
          orientation = rhs.orientation;
          position = rhs.position;
      }
      return *this;
  }
  CAMState& operator= (CAMState&& rhs) {
      std::swap(id, rhs.id);
      std::swap(orientation, rhs.orientation);
      std::swap(position, rhs.position);
      return *this;
  }
};

typedef std::vector<CAMState, Eigen::aligned_allocator<
        CAMState > > CamStateServer;

/*
 * @brief Feature Salient part of an image. Please refer
 *    to the Appendix of "A Multi-State Constraint Kalman
 *    Filter for Vision-aided Inertial Navigation" for how
 *    the 3d position of a feature is initialized.
 */
struct Feature {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef uint64_t FeatureIDType;

  /*
   * @brief OptimizationConfig Configuration parameters
   *    for 3d feature position optimization.
   */
  struct OptimizationConfig {
    double translation_threshold;
    double huber_epsilon;
    double estimation_precision;
    double initial_damping;
    int outer_loop_max_iteration;
    int inner_loop_max_iteration;

    OptimizationConfig():
      translation_threshold(0.2),
      huber_epsilon(0.01),
      estimation_precision(5e-7),
      initial_damping(1e-3),
      outer_loop_max_iteration(10),
      inner_loop_max_iteration(10) {
      return;
    }
  };

  // Constructors for the struct.
  Feature(): id(0), position(Eigen::Vector3d::Zero()),
    is_initialized(false) {}

  Feature(const FeatureIDType& new_id): id(new_id),
    position(Eigen::Vector3d::Zero()),
    is_initialized(false) {}

  /**
   * @brief Feature
   * @param observations_in observation directions [x, y, 1]
   * @param cam_states_in array of camera poses, corresponding to every observations
   */
  Feature(const std::vector<Eigen::Vector2d,
                            Eigen::aligned_allocator<Eigen::Vector2d>>&
              observations_in,
          const CamStateServer& cam_states_in)
      : id(0),
        observations(observations_in),
        cam_states(cam_states_in),
        position(Eigen::Vector3d::Zero()),
        is_initialized(false) {}

  inline void setObservations(
      const std::vector<Eigen::Vector2d,
                        Eigen::aligned_allocator<Eigen::Vector2d>>&
          observations_in) {
    observations = observations_in;
  }

  /*
   * @param cam_states: A vector containing the camera poses.
   */
  inline void setCamStates(const CamStateServer& cam_states_in) {
    cam_states = cam_states_in;
  }

  /*
   * @brief cost Compute the cost of the camera observations
   * @param T_c0_c1 A rigid body transformation takes
   *    a vector in c0 frame to ci frame.
   * @param x The current estimation.
   * @param z The ith measurement of the feature j in ci frame.
   * @return e The cost of this observation.
   */
  inline void cost(const Eigen::Isometry3d& T_c0_ci,
      const Eigen::Vector3d& x, const Eigen::Vector2d& z,
      double& e) const;

  /*
   * @brief jacobian Compute the Jacobian of the camera observation
   * @param T_c0_c1 A rigid body transformation takes
   *    a vector in c0 frame to ci frame.
   * @param x The current estimation.
   * @param z The actual measurement of the feature in ci frame.
   * @return J The computed Jacobian.
   * @return r The computed residual.
   * @return w Weight induced by huber kernel.
   */
  inline void jacobian(const Eigen::Isometry3d& T_c0_ci,
      const Eigen::Vector3d& x, const Eigen::Vector2d& z,
      Eigen::Matrix<double, 2, 3>& J, Eigen::Vector2d& r,
      double& w) const;

  /*
   * @brief generateInitialGuess Compute the initial guess of
   *    the feature's 3d position using only two views.
   * @param T_c1_c2: A rigid body transformation taking
   *    a vector from c2 frame to c1 frame.
   * @param z1: feature observation in c1 frame.
   * @param z2: feature observation in c2 frame.
   * @return p: Computed feature position in c1 frame.
   */
  inline void generateInitialGuess(
      const Eigen::Isometry3d& T_c1_c2, const Eigen::Vector2d& z1,
      const Eigen::Vector2d& z2, Eigen::Vector3d& p) const;

  /*
   * @brief checkMotion Check the input camera poses to ensure
   *    there is enough translation to triangulate the feature
   *    positon.
   * @return True if the translation between the input camera
   *    poses is sufficient.
   */
  inline bool checkMotion() const;

  /*
   * @brief InitializePosition Intialize the feature position
   *    based on all current available measurements.
   * @return The computed 3d position is used to set the position
   *    member variable. Note the resulted position is in world
   *    frame.
   * @return True if the estimated 3d position of the feature
   *    is valid.
   */
  inline bool initializePosition();


  // An unique identifier for the feature.
  // In case of long time running, the variable
  // type of id is set to FeatureIDType in order
  // to avoid duplication.
  FeatureIDType id;

  // Store the observations of the features.
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<
      Eigen::Vector2d>> observations;

  CamStateServer cam_states;

  // 3d postion of the feature in the world frame.
  Eigen::Vector3d position;

  // A indicator to show if the 3d postion of the feature
  // has been initialized or not.
  bool is_initialized;

  // Optimization configuration for solving the 3d position.
  static OptimizationConfig optimization_config;

};

typedef Feature::FeatureIDType FeatureIDType;
typedef std::map<FeatureIDType, Feature, std::less<int>,
        Eigen::aligned_allocator<
        std::pair<const FeatureIDType, Feature> > > MapServer;


void Feature::cost(const Eigen::Isometry3d& T_c0_ci,
    const Eigen::Vector3d& x, const Eigen::Vector2d& z,
    double& e) const {
  // Compute hi1, hi2, and hi3 as Equation (37).
  const double& alpha = x(0);
  const double& beta = x(1);
  const double& rho = x(2);

  Eigen::Vector3d h = T_c0_ci.linear()*
    Eigen::Vector3d(alpha, beta, 1.0) + rho*T_c0_ci.translation();
  double& h1 = h(0);
  double& h2 = h(1);
  double& h3 = h(2);

  // Predict the feature observation in ci frame.
  Eigen::Vector2d z_hat(h1/h3, h2/h3);

  // Compute the residual.
  e = (z_hat-z).squaredNorm();
  return;
}

void Feature::jacobian(const Eigen::Isometry3d& T_c0_ci,
    const Eigen::Vector3d& x, const Eigen::Vector2d& z,
    Eigen::Matrix<double, 2, 3>& J, Eigen::Vector2d& r,
    double& w) const {

  // Compute hi1, hi2, and hi3 as Equation (37).
  const double& alpha = x(0);
  const double& beta = x(1);
  const double& rho = x(2);

  Eigen::Vector3d h = T_c0_ci.linear()*
    Eigen::Vector3d(alpha, beta, 1.0) + rho*T_c0_ci.translation();
  double& h1 = h(0);
  double& h2 = h(1);
  double& h3 = h(2);

  // Compute the Jacobian.
  Eigen::Matrix3d W;
  W.leftCols<2>() = T_c0_ci.linear().leftCols<2>();
  W.rightCols<1>() = T_c0_ci.translation();

  J.row(0) = 1/h3*W.row(0) - h1/(h3*h3)*W.row(2);
  J.row(1) = 1/h3*W.row(1) - h2/(h3*h3)*W.row(2);

  // Compute the residual.
  Eigen::Vector2d z_hat(h1/h3, h2/h3);
  r = z_hat - z;

  // Compute the weight based on the residual.
  double e = r.norm();
  if (e <= optimization_config.huber_epsilon)
    w = 1.0;
  else
    w = std::sqrt(2.0*optimization_config.huber_epsilon / e);

  return;
}

void Feature::generateInitialGuess(
    const Eigen::Isometry3d& T_c1_c2, const Eigen::Vector2d& z1,
    const Eigen::Vector2d& z2, Eigen::Vector3d& p) const {
  // Construct a least square problem to solve the depth.
  Eigen::Vector3d m = T_c1_c2.linear() * Eigen::Vector3d(z1(0), z1(1), 1.0);

  Eigen::Vector2d A(0.0, 0.0);
  A(0) = m(0) - z2(0)*m(2);
  A(1) = m(1) - z2(1)*m(2);

  Eigen::Vector2d b(0.0, 0.0);
  b(0) = z2(0)*T_c1_c2.translation()(2) - T_c1_c2.translation()(0);
  b(1) = z2(1)*T_c1_c2.translation()(2) - T_c1_c2.translation()(1);

  // Solve for the depth.
  double depth = (A.transpose() * A).inverse() * A.transpose() * b;
  p(0) = z1(0) * depth;
  p(1) = z1(1) * depth;
  p(2) = depth;
  return;
}

bool Feature::checkMotion() const {
  Eigen::Isometry3d first_cam_pose;
  first_cam_pose.linear() = cam_states.front().orientation.toRotationMatrix().transpose();
  first_cam_pose.translation() =
    cam_states.front().position;

  Eigen::Isometry3d last_cam_pose;
  last_cam_pose.linear() =
      cam_states.back().orientation.toRotationMatrix().transpose();
  last_cam_pose.translation() =
    cam_states.back().position;

  // Get the direction of the feature when it is first observed.
  // This direction is represented in the world frame.
  Eigen::Vector3d feature_direction(
      observations.front()(0),
      observations.front()(1), 1.0);
  feature_direction = feature_direction / feature_direction.norm();
  feature_direction = first_cam_pose.linear()*feature_direction;

  // Compute the translation between the first frame
  // and the last frame. We assume the first frame and
  // the last frame will provide the largest motion to
  // speed up the checking process.
  Eigen::Vector3d translation = last_cam_pose.translation() -
    first_cam_pose.translation();
  double parallel_translation =
    translation.transpose()*feature_direction;
  Eigen::Vector3d orthogonal_translation = translation -
    parallel_translation*feature_direction;

  if (orthogonal_translation.norm() >
      optimization_config.translation_threshold)
    return true;
  else return false;
}

bool Feature::initializePosition() {
  // Organize camera poses and feature observations properly.
  std::vector<Eigen::Isometry3d,
    Eigen::aligned_allocator<Eigen::Isometry3d> > cam_poses(0);
  std::vector<Eigen::Vector2d,
    Eigen::aligned_allocator<Eigen::Vector2d> > measurements(0);
  int obs_index = 0;
  for (std::vector<Eigen::Vector2d, Eigen::aligned_allocator<
       Eigen::Vector2d>>::const_iterator iter = observations.begin();
       iter != observations.end(); ++iter, ++obs_index) {
    // Add the measurement.
    measurements.push_back(*iter);

    // This camera pose will take a vector from this camera frame
    // to the world frame.
    Eigen::Isometry3d cam0_pose;
    cam0_pose.linear() = cam_states[obs_index].orientation.toRotationMatrix().transpose();
    cam0_pose.translation() = cam_states[obs_index].position;
    cam_poses.push_back(cam0_pose);
  }

  // All camera poses should be modified such that it takes a
  // vector from the first camera frame in the buffer to this
  // camera frame.
  Eigen::Isometry3d T_c0_w = cam_poses[0];
  for (auto& pose : cam_poses)
    pose = pose.inverse() * T_c0_w;

  // Generate initial guess
  Eigen::Vector3d initial_position(0.0, 0.0, 0.0);
  generateInitialGuess(cam_poses[cam_poses.size()-1], measurements[0],
      measurements[measurements.size()-1], initial_position);
  Eigen::Vector3d solution(
      initial_position(0)/initial_position(2),
      initial_position(1)/initial_position(2),
      1.0/initial_position(2));

  // Apply Levenberg-Marquart method to solve for the 3d position.
  double lambda = optimization_config.initial_damping;
  int inner_loop_cntr = 0;
  int outer_loop_cntr = 0;
  bool is_cost_reduced = false;
  double delta_norm = 0;

  // Compute the initial cost.
  double total_cost = 0.0;
  int no_cam_poses = static_cast<int>(cam_poses.size());
  for (int i = 0; i < no_cam_poses; ++i) {
    double this_cost = 0.0;
    cost(cam_poses[i], solution, measurements[i], this_cost);
    total_cost += this_cost;
  }

  // Outer loop.
  do {
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    int no_cam_poses = static_cast<int>(cam_poses.size());
    for (int i = 0; i < no_cam_poses; ++i) {
      Eigen::Matrix<double, 2, 3> J;
      Eigen::Vector2d r;
      double w;

      jacobian(cam_poses[i], solution, measurements[i], J, r, w);

      if (w == 1) {
        A += J.transpose() * J;
        b += J.transpose() * r;
      } else {
        double w_square = w * w;
        A += w_square * J.transpose() * J;
        b += w_square * J.transpose() * r;
      }
    }

    // Inner loop.
    // Solve for the delta that can reduce the total cost.
    do {
      Eigen::Matrix3d damper = lambda * Eigen::Matrix3d::Identity();
      Eigen::Vector3d delta = (A+damper).ldlt().solve(b);
      Eigen::Vector3d new_solution = solution - delta;
      delta_norm = delta.norm();

      double new_cost = 0.0;
      int no_cam_poses = static_cast<int>(cam_poses.size());
      for (int i = 0; i < no_cam_poses; ++i) {
        double this_cost = 0.0;
        cost(cam_poses[i], new_solution, measurements[i], this_cost);
        new_cost += this_cost;
      }

      if (new_cost < total_cost) {
        is_cost_reduced = true;
        solution = new_solution;
        total_cost = new_cost;
        lambda = lambda/10 > 1e-10 ? lambda/10 : 1e-10;
      } else {
        is_cost_reduced = false;
        lambda = lambda*10 < 1e12 ? lambda*10 : 1e12;
      }

    } while (inner_loop_cntr++ <
        optimization_config.inner_loop_max_iteration && !is_cost_reduced);

    inner_loop_cntr = 0;

  } while (outer_loop_cntr++ <
      optimization_config.outer_loop_max_iteration &&
      delta_norm > optimization_config.estimation_precision);

  // Covert the feature position from inverse depth
  // representation to its 3d coordinate.
  Eigen::Vector3d final_position(solution(0)/solution(2),
      solution(1)/solution(2), 1.0/solution(2));

  // Check if the solution is valid. Make sure the feature
  // is in front of every camera frame observing it.
  bool is_valid_solution = true;
  for (const auto& pose : cam_poses) {
    Eigen::Vector3d position =
      pose.linear()*final_position + pose.translation();
    if (position(2) <= 0) {
      is_valid_solution = false;
      break;
    }
  }

  // Convert the feature position to the world frame.
  position = T_c0_w.linear()*final_position + T_c0_w.translation();

  if (is_valid_solution)
    is_initialized = true;

  return is_valid_solution;
}
} // namespace msckf_vio

#endif // MSCKF_VIO_FEATURE_HPP_
