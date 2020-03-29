/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

/* notation specific to this header file
 * T_A_B takes a vector from the A frame to the B frame.
 * R_A_B, t_A_B are defined accordingly
 */

#ifndef MSCKF_VIO_FEATURE_HPP_
#define MSCKF_VIO_FEATURE_HPP_

#include <iostream>
#include <map>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <okvis/kinematics/Transformation.hpp>
#include <okvis/triangulation/stereo_triangulation.hpp>

namespace msckf_vio {

namespace {
template <class T>
const T clamp(const T& v, const T& lo, const T& hi) {
  assert(!(hi < lo));
  return (v < lo) ? lo : (hi < v) ? hi : v;
}
}  // namespace

/*
 * @brief Feature Salient part of an image. Please refer
 *    to the Appendix of "A Multi-State Constraint Kalman
 *    Filter for Vision-aided Inertial Navigation" for how
 *    the 3d position of a feature is initialized.
 */
struct Feature {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef uint64_t FeatureIDType;
  typedef std::vector<
      okvis::kinematics::Transformation,
      Eigen::aligned_allocator<okvis::kinematics::Transformation>>
      CamStateServer;
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

    double min_depth;
    double max_depth; // 10 is recommended for indoor, 1000 for outdoor

    OptimizationConfig():
      translation_threshold(0.2),
      huber_epsilon(0.01),
      estimation_precision(5e-7),
      initial_damping(1e-3),
      outer_loop_max_iteration(10),
      inner_loop_max_iteration(10),
      min_depth(0.1),
      max_depth(1000) {
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
   * @brief cost Compute the cost of the camera observations
   * @param T_c0_c1 A rigid body transformation takes
   *    a vector in c0 frame to ci frame.
   * @param x The current estimation.
   * @param z The ith measurement of the feature j in ci frame.
   * @return e The cost of this observation.
   */
  inline void cost(const okvis::kinematics::Transformation& T_c0_ci,
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
  inline void jacobian(const okvis::kinematics::Transformation& T_c0_ci,
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
      const okvis::kinematics::Transformation& T_c1_c2, const Eigen::Vector2d& z1,
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
   *    This function handles points at infinity, rotation only motion.
   * @return The computed 3d position is used to set the position
   *    member variable. Note the resulted position is in world
   *    frame.
   * @return True if the estimated 3d position of the feature
   *    is valid.
   */
  inline bool initializePosition();

  inline static void setMaxDepth(double maxDepth);

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

  bool is_chi2_small; // is the chi2 small enough?
  bool is_parallel; // is the observations rays parallel?
  // is flipping the landmark position needed? e.g., in case of forward motion.
  bool is_flipped;

  // Optimization configuration for solving the 3d position.
  static OptimizationConfig optimization_config;

};

typedef Feature::FeatureIDType FeatureIDType;
typedef std::map<FeatureIDType, Feature, std::less<int>,
        Eigen::aligned_allocator<
        std::pair<const FeatureIDType, Feature> > > MapServer;


void Feature::cost(const okvis::kinematics::Transformation& T_c0_ci,
    const Eigen::Vector3d& x, const Eigen::Vector2d& z,
    double& e) const {
  // Compute hi1, hi2, and hi3 as Equation (37).
  const double& alpha = x(0);
  const double& beta = x(1);
  const double& rho = x(2);

  Eigen::Vector3d h = T_c0_ci.C()*
    Eigen::Vector3d(alpha, beta, 1.0) + rho*T_c0_ci.r();
  double& h1 = h(0);
  double& h2 = h(1);
  double& h3 = h(2);

  // Predict the feature observation in ci frame.
  Eigen::Vector2d z_hat(h1/h3, h2/h3);

  // Compute the residual.
  e = (z_hat-z).squaredNorm();
  return;
}

void Feature::jacobian(const okvis::kinematics::Transformation& T_c0_ci,
    const Eigen::Vector3d& x, const Eigen::Vector2d& z,
    Eigen::Matrix<double, 2, 3>& J, Eigen::Vector2d& r,
    double& w) const {

  // Compute hi1, hi2, and hi3 as Equation (37).
  const double& alpha = x(0);
  const double& beta = x(1);
  const double& rho = x(2);

  Eigen::Vector3d h = T_c0_ci.C()*
    Eigen::Vector3d(alpha, beta, 1.0) + rho*T_c0_ci.r();
  double& h1 = h(0);
  double& h2 = h(1);
  double& h3 = h(2);

  // Compute the Jacobian.
  Eigen::Matrix3d W;
  W.leftCols<2>() = T_c0_ci.C().leftCols<2>();
  W.rightCols<1>() = T_c0_ci.r();

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
    const okvis::kinematics::Transformation& T_c1_c2, const Eigen::Vector2d& z1,
    const Eigen::Vector2d& z2, Eigen::Vector3d& p) const {
  // Construct a least square problem to solve the depth.
  Eigen::Vector3d m = T_c1_c2.C() * Eigen::Vector3d(z1(0), z1(1), 1.0);

  Eigen::Vector2d A(0.0, 0.0);
  A(0) = m(0) - z2(0)*m(2);
  A(1) = m(1) - z2(1)*m(2);

  Eigen::Vector2d b(0.0, 0.0);
  b(0) = z2(0)*T_c1_c2.r()(2) - T_c1_c2.r()(0);
  b(1) = z2(1)*T_c1_c2.r()(2) - T_c1_c2.r()(1);

  // Solve for the depth.
  double depth = (A.transpose() * A).inverse() * A.transpose() * b;
  p(0) = z1(0) * depth;
  p(1) = z1(1) * depth;
  p(2) = depth;
  return;
}

bool Feature::checkMotion() const {
  okvis::kinematics::Transformation first_cam_pose = cam_states.front();
  okvis::kinematics::Transformation last_cam_pose = cam_states.back();

  // Get the direction of the feature when it is first observed.
  // This direction is represented in the world frame.
  Eigen::Vector3d feature_direction(
      observations.front()(0),
      observations.front()(1), 1.0);
  feature_direction.normalize();
  feature_direction = first_cam_pose.C()*feature_direction;

  // Compute the translation between the first frame
  // and the last frame. We assume the first frame and
  // the last frame will provide the largest motion to
  // speed up the checking process.
  Eigen::Vector3d translation = last_cam_pose.r() -
    first_cam_pose.r();
  double parallel_translation =
    translation.transpose()*feature_direction;
  Eigen::Vector3d orthogonal_translation = translation -
    parallel_translation*feature_direction;

  if (orthogonal_translation.norm() >
      optimization_config.translation_threshold)
    return true;
  else return false;
}

static double raySigma() {
  int kpSize = 8;
  int fx = 640;
  double keypointAStdDev = kpSize;
  keypointAStdDev = 0.8 * keypointAStdDev / 12.0;
  return sqrt(sqrt(2)) * keypointAStdDev / fx;
}

static bool isValidSolution(
    const std::vector<okvis::kinematics::Transformation,
                      Eigen::aligned_allocator<okvis::kinematics::Transformation>>& T_w_ci,
    const Eigen::Vector3d& pinw) {
  // Check if the solution is valid. Make sure the feature
  // is in front of every camera frame observing it.
  bool is_valid_solution = true;
  for (const auto& pose : T_w_ci) {
    Eigen::Vector3d position = pose.C() * pinw + pose.r();
    if (position(2) <= 0) {
      return false;
    }
  }
  return is_valid_solution;
}
// TODO(jhuai): use the LM method inside ceres TinySolver and directly
// returns anchored inverse depth parameters if needed.
bool Feature::initializePosition() {
  // Organize camera poses and feature observations properly.
  std::vector<okvis::kinematics::Transformation,
    Eigen::aligned_allocator<okvis::kinematics::Transformation> > cam_poses =
      cam_states;
  const std::vector<Eigen::Vector2d,
    Eigen::aligned_allocator<Eigen::Vector2d> > &measurements = observations;

  // All camera poses should be modified such that it takes a
  // vector from the first camera frame in the buffer to this
  // camera frame.
  okvis::kinematics::Transformation T_c0_w = cam_poses[0];
  for (auto& pose : cam_poses)
    pose = pose.inverse() * T_c0_w;

  // Generate initial guess
//  Eigen::Vector3d initial_position(0.0, 0.0, 0.0);
//  generateInitialGuess(cam_poses[cam_poses.size()-1], measurements[0],
//      measurements[measurements.size()-1], initial_position);
//  Eigen::Vector3d solution(
//      initial_position(0)/initial_position(2),
//      initial_position(1)/initial_position(2),
//      1.0/initial_position(2));

  // triangulate the point w.r.t the first camera frame as the world frame
  Eigen::Vector3d obsDir[2];
  obsDir[0].head<2>() = measurements[0];
  obsDir[0][2] = 1.0;
  obsDir[1].head<2>() = measurements[measurements.size()-1];
  obsDir[1][2] = 1.0;

  okvis::kinematics::Transformation T_c0_ce = cam_poses[cam_poses.size()-1];
  obsDir[1] = (T_c0_ce.C().transpose() * obsDir[1]).eval();
  Eigen::Vector4d homogeneousPoint = okvis::triangulation::triangulateFast(
      Eigen::Vector3d::Zero(),  // center of A in W coordinates
      obsDir[0].normalized(),
      - T_c0_ce.C().transpose() * T_c0_ce.r(),  // center of B in W coordinates
      obsDir[1].normalized(),
      raySigma(), is_chi2_small, is_parallel, is_flipped);
  homogeneousPoint /= homogeneousPoint[3];

  // This value should be dependent on the max allowed scene depth
  // Very small depths may occur under pure rotation.
  homogeneousPoint[2] = homogeneousPoint[2] < optimization_config.min_depth
                            ? optimization_config.min_depth
                            : homogeneousPoint[2];
  double invDepth = 1.0 / homogeneousPoint[2];

  // Too large chi2 may be due to wrong association.
  // But subsequent nonlinear opt may save this case, so we stick with it.
//  if (!is_chi2_small) {
//      position = T_c0_w.linear()*homogeneousPoint.head<3>() + T_c0_w.translation();
//      is_initialized = false;
//      return false;
//  }

//  if (is_flipped) { // Forward motion causes ambiguity. Let's bail out because
//      // doing nonlinear opt may revert the landmark position to the wrong side.
//      position = T_c0_w.C()*homogeneousPoint.head<3>() + T_c0_w.r();
//      is_initialized = isValidSolution(cam_poses, homogeneousPoint.head<3>());
//      return is_initialized;
//  }

  Eigen::Vector3d solution;  // landmark position in c0 frame
  solution << homogeneousPoint[0] * invDepth,
      homogeneousPoint[1] * invDepth, invDepth;
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

  // clamp the depth to avoid numerical issues
  solution(2) = clamp(solution(2), 1/optimization_config.max_depth, 1/optimization_config.min_depth);
  // Covert the feature position from inverse depth
  // representation to its 3d coordinate.
  Eigen::Vector3d final_position(solution(0)/solution(2),
      solution(1)/solution(2), 1.0/solution(2));

  is_initialized = isValidSolution(cam_poses, final_position);

  // Convert the feature position to the world frame.
  position = T_c0_w.C()*final_position + T_c0_w.r();

  return is_initialized;
}

void Feature::setMaxDepth(double maxDepth) {
  optimization_config.max_depth = maxDepth;
}
} // namespace msckf_vio

#endif // MSCKF_VIO_FEATURE_HPP_
