
#include <swift_vio/FeatureTriangulation.hpp>

namespace swift_vio {

Feature::OptimizationConfig Feature::optimization_config;

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

Eigen::Vector4d triangulateHomogeneousDLT(
    const Eigen::AlignedVector<Eigen::Vector3d>& obsDirections,
    const Eigen::AlignedVector<okvis::kinematics::Transformation>& T_CWs) {
  size_t K = obsDirections.size();
  Eigen::MatrixXd A(2 * K, 4);
  for (size_t k = 0; k < K; ++k) {
    A.row(2 * k).head<3>() =
        obsDirections[k][0] * T_CWs[k].C().row(2) - T_CWs[k].C().row(0);
    A(2 * k, 3) = obsDirections[k][0] * T_CWs[k].r()[2] - T_CWs[k].r()[0];
    A.row(2 * k + 1).head<3>() =
        obsDirections[k][1] * T_CWs[k].C().row(2) - T_CWs[k].C().row(1);
    A(2 * k + 1, 3) = obsDirections[k][1] * T_CWs[k].r()[2] - T_CWs[k].r()[1];
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svdM(A, Eigen::ComputeThinV);
  Eigen::Matrix<double, 4, 1> singularValues = svdM.singularValues();
  if (singularValues[2] < 1e-4) {
    std::cerr << "DLT result is invalid as the third singular value is too small.";
    return svdM.matrixV().col(2);
  } else {
    return svdM.matrixV().col(3);
  }
}

bool hasLowDisparity(
    const Eigen::AlignedVector<Eigen::Vector3d>& obsDirections,
    const Eigen::AlignedVector<okvis::kinematics::Transformation>& T_CWs,
    const std::vector<double>& imageNoiseStd,
    double focalLength, double raySigmaScalar) {
  double keypointAStdDev = (imageNoiseStd.front() + imageNoiseStd.back()) * 0.5;
  const double fourthRoot2 = 1.1892071150;
  double raySigma = fourthRoot2 * keypointAStdDev / focalLength;
  Eigen::Vector3d rayA_inA = obsDirections.front().normalized();
  Eigen::Vector3d rayB_inB = obsDirections.back().normalized();
  Eigen::Vector3d rayB_inA =
      T_CWs.front().C() * T_CWs.back().C().transpose() * rayB_inB;
  if ((rayA_inA.cross(rayB_inB)).norm() < raySigmaScalar * raySigma ||
      (rayA_inA.cross(rayB_inA)).norm() < raySigmaScalar * raySigma) {
    return true;
  } else {
    return false;
  }
}
} // namespace swift_vio


