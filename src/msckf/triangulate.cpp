#include "msckf/triangulate.h"
#include <iostream>

#define HAVE_CERES
#ifdef HAVE_CERES
#include "ceres/ceres.h"  //LM and Dogleg optimization
#include "ceres/rotation.h"
#include "glog/logging.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
#endif

Eigen::Vector4d Get_X_from_xP_lin(
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>&
        vV3Xn,
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>& vSE3,
    Eigen::Matrix<double, Eigen::Dynamic, 1>* pSingular) {
  assert(vV3Xn.size() == vSE3.size());
  int K = vV3Xn.size();
  Eigen::MatrixXd A(2 * K, 4);
  Eigen::MatrixXd rhs(2 * K, 1);

  for (int k = 0; k < K; ++k) {
    rhs(2 * k) = 0;
    rhs(2 * k + 1) = 0;
    A.row(2 * k).head<3>() = vV3Xn[k][0] * vSE3[k].rotationMatrix().row(2) -
                             vSE3[k].rotationMatrix().row(0);
    A(2 * k, 3) =
        vV3Xn[k][0] * vSE3[k].translation()[2] - vSE3[k].translation()[0];
    A.row(2 * k + 1).head<3>() = vV3Xn[k][1] * vSE3[k].rotationMatrix().row(2) -
                                 vSE3[k].rotationMatrix().row(1);
    A(2 * k + 1, 3) =
        vV3Xn[k][1] * vSE3[k].translation()[2] - vSE3[k].translation()[1];
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svdM(A, Eigen::ComputeThinV);
  if (pSingular) *pSingular = svdM.singularValues();
  Eigen::Vector4d v4Xhomog = svdM.matrixV().col(3);
  return v4Xhomog;
}

Eigen::Vector4d Get_X_from_xP_lin(
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>&
        vV3Xn,
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>& vSE3,
    bool& isValid, bool& isParallel) {
  assert(vV3Xn.size() == vSE3.size());
  int K = vV3Xn.size();
  Eigen::MatrixXd A(2 * K, 4);
  Eigen::MatrixXd rhs(2 * K, 1);

  for (int k = 0; k < K; ++k) {
    rhs(2 * k) = 0;
    rhs(2 * k + 1) = 0;
    A.row(2 * k).head<3>() = vV3Xn[k][0] * vSE3[k].rotationMatrix().row(2) -
                             vSE3[k].rotationMatrix().row(0);
    A(2 * k, 3) =
        vV3Xn[k][0] * vSE3[k].translation()[2] - vSE3[k].translation()[0];
    A.row(2 * k + 1).head<3>() = vV3Xn[k][1] * vSE3[k].rotationMatrix().row(2) -
                                 vSE3[k].rotationMatrix().row(1);
    A(2 * k + 1, 3) =
        vV3Xn[k][1] * vSE3[k].translation()[2] - vSE3[k].translation()[1];
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svdM(A, Eigen::ComputeThinV);
  Eigen::Matrix<double, Eigen::Dynamic, 1> pSingular = svdM.singularValues();
  if (std::fabs(pSingular[2]) < 1e-8)
    isValid = false;
  else
    isValid = true;
  Eigen::Vector4d v4Xhomog = svdM.matrixV().col(3);
  if (std::fabs(v4Xhomog[3]) < 1e-6)
    isParallel = true;
  else
    isParallel = false;
  return v4Xhomog;
}

// based on Hartley's decription in "Triangulation"
Eigen::Vector3d triangulate2View_midpoint(
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
        vV2Xn,
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>& vSE3) {
  Eigen::MatrixXd A(3, 2);
  Eigen::Matrix3d rel_rot =
      (vSE3[0].unit_quaternion() * vSE3[1].unit_quaternion().conjugate())
          .toRotationMatrix();
  A.col(0) = unproject2d(vV2Xn[0]);
  A.col(1) = -rel_rot * unproject2d(vV2Xn[1]);
  Eigen::Vector3d b = vSE3[0].translation() - rel_rot * vSE3[1].translation();
  Eigen::JacobiSVD<Eigen::MatrixXd> svdM(
      A, Eigen::ComputeThinU | Eigen::ComputeThinV);

  Eigen::Vector2d v3Xhomog = svdM.solve(b);
  Eigen::Matrix3d Ap;
  Ap.col(0) = A.col(0);
  Ap.col(1) = -A.col(1);
  Ap.col(2) = -vSE3[0].translation() - rel_rot * vSE3[1].translation();
  Eigen::Matrix3d rot =
      vSE3[0].unit_quaternion().conjugate().toRotationMatrix();
  return rot * Ap * unproject2d(v3Xhomog) / 2;
}
Eigen::Matrix<double, 2, 3> householder(Eigen::Vector3d x) {
  // Compute a 2x3 matrix where the rows are orthogonal to x and orthogonal to
  // each other.
  assert(x.norm() > 1e-8);
  Eigen::Vector3d a(0, 0, 0);
  // Vector3d::Index   minIndexRow, minIndexCol;
  Eigen::Vector3d absx = x.cwiseAbs();
  Eigen::MatrixXd::Index minIndex;
  absx.minCoeff(&minIndex);
  a(minIndex) = 1;
  Eigen::Vector3d u = x.cross(a);
  u.normalize();
  Eigen::Vector3d v = x.cross(u);
  v.normalize();
  Eigen::Matrix<double, 3, 2> res;
  res.col(0) = u;
  res.col(1) = v;
  return res.transpose();
}
Eigen::Vector3d triangulate_midpoint(
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
        vV2Xn,
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>& vSE3,
    int num_elements) {
  //  Triangulate a landmark from two or more views using the midpoint method.
  assert(vV2Xn.size() == vSE3.size());
  Eigen::Matrix3d jtj = Eigen::Matrix3d::Zero();
  Eigen::Vector3d jtr = Eigen::Vector3d::Zero();
  num_elements = num_elements == 0 ? vSE3.size() : num_elements;
  for (int jack = 0; jack < num_elements; ++jack) {
    Eigen::Matrix<double, 2, 3> h = householder(unproject2d(vV2Xn[jack]));
    Eigen::Matrix<double, 2, 3> a = h * vSE3[jack].rotationMatrix();
    Eigen::Matrix<double, 2, 1> b = -h * vSE3[jack].translation();
    jtj += a.transpose() * a;
    jtr += a.transpose() * b;
  }
  return jtj.colPivHouseholderQr().solve(jtr);
}

// each observation is in image plane z=1, (\bar{x}, \bar{y}, 1), each of
// frame_poses is Tw2c(i) old_point stores initial position and optimized
// position refine point position with Gauss Newton method \Delta{x}=
// -(J^TJ)^{-1}J^T (y-f(x)), J=\frac{\partial y-f(x)}{\partial x} x_{new}= x+
// \Delta{x}
#undef POINT_OPTIMIZER_DEBUG
void triangulate_refine_GN(
    const std::vector<Eigen::Vector3d,
                      Eigen::aligned_allocator<Eigen::Vector3d>>& obs,
    const std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>&
        frame_poses,
    Eigen::Vector3d& old_point, int n_iter) {
  Eigen::Vector3d pos = old_point;
  double chi2 = 0.0;
  Eigen::Matrix3d A;
  Eigen::Vector3d b;
  double max_elem = norm_max(old_point);
  for (int i = 0; i < n_iter; ++i) {
    A.setZero();
    b.setZero();
    double new_chi2 = 0.0;

    // compute residuals
    std::vector<Sophus::SE3d,
                Eigen::aligned_allocator<Sophus::SE3d>>::const_iterator
        it_poses = frame_poses.begin();
    for (std::vector<Eigen::Vector3d,
                     Eigen::aligned_allocator<Eigen::Vector3d>>::const_iterator
             it = obs.begin();
         it != obs.end(); ++it, ++it_poses) {
      Eigen::Matrix<double, 2, 3> J;
      const Eigen::Vector3d p_in_f((*it_poses) * pos);
      jacobian_xyz2uv(p_in_f, it_poses->rotationMatrix(), J);
      const Eigen::Vector2d e(project2d(*it) - project2d(p_in_f));
      new_chi2 += e.squaredNorm();
      A.noalias() += J.transpose() * J;
      b.noalias() -= J.transpose() * e;
    }
    // pdop= pixel_variance*sqrt(A.inverse().trace())/cam->errorMultiplier2();

    // solve linear system
    const Eigen::Vector3d dp(A.ldlt().solve(b));
    // check if error increased
    if ((i > 0 && new_chi2 > chi2) || (bool)std::isnan((double)dp[0])) {
#ifdef POINT_OPTIMIZER_DEBUG
      cout << "it " << i << "\t FAILURE \t new_chi2 = " << new_chi2 << endl;
#endif
      pos = old_point;  // roll-back
      break;
    }
    // update the model
    Eigen::Vector3d new_point = pos + dp;
    old_point = pos;
    pos = new_point;
    chi2 = new_chi2;
#ifdef POINT_OPTIMIZER_DEBUG
    cout << "it " << i << "\t Success \t new_chi2 = " << new_chi2
         << "\t norm(b) = " << norm_max(b) << endl;
#endif
    // stop when converged
    if (norm_max(dp) <= (max_elem + EPS) * EPS) break;
  }
  old_point = pos;
#ifdef POINT_OPTIMIZER_DEBUG
  cout << endl;
#endif
}

#ifdef HAVE_CERES
// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 7 parameters: 4 for rotation, 3 for translation, qxyzw,
// txyz assume observations are at the z=1 image plane Warning: use templated
// whole class causes type conversion issues in autodiff
struct ReprojectionError {
  ReprojectionError(Sophus::SE3Group<double> _Tw2c,
                    Eigen::Matrix<double, 2, 1> _observed)
      : Tw2c(_Tw2c), observed(_observed) {}
  template <typename T>
  bool operator()(const T* const point, T* residuals) const {
    typename Eigen::Matrix<T, 3, 1, Eigen::ColMajor>::ConstMapType v3point(
        point);
    Eigen::Matrix<T, 3, 1> p = Tw2c.cast<T>() * v3point;
    T xp = p[0] / p[2];
    T yp = p[1] / p[2];
    // The error is the difference between the predicted and observed position.
    residuals[0] = xp - T(observed[0]);
    residuals[1] = yp - T(observed[1]);
    return true;
  }
  // Factory to hide the construction of the CostFunction object from
  // the client code.

  static ceres::CostFunction* Create(
      const Sophus::SE3Group<double> Tw2c,
      const Eigen::Matrix<double, 2, 1> observed) {
    return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3>(
        new ReprojectionError(Tw2c, observed)));
  }
  Sophus::SE3Group<double> Tw2c;
  Eigen::Matrix<double, 2, 1> observed;
};
void triangulate_refine(
    const std::vector<Eigen::Vector3d,
                      Eigen::aligned_allocator<Eigen::Vector3d>>& obs,
    const std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>&
        frame_poses,
    Eigen::Vector3d& old_point, int n_iter, int algorithm) {
  // Create residuals for each observation in the bundle adjustment problem. The
  // parameters for cameras and points are added automatically.
  ceres::Problem problem;
  for (size_t i = 0; i < obs.size(); ++i) {
    // Each Residual block takes a point and a camera as input and outputs a 2
    // dimensional residual. Internally, the cost function stores the observed
    // image location and compares the reprojection against the observation.
    Eigen::Vector2d v2InPlane = project2d(obs[i]);
    ceres::CostFunction* cost_function =
        ReprojectionError::Create(frame_poses[i], v2InPlane);
    problem.AddResidualBlock(cost_function, NULL /* squared loss */,
                             old_point.data());
  }
  // Make Ceres automatically detect the bundle structure. Note that the
  // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
  // for standard bundle adjustment problems.
  ceres::Solver::Options options;
  options.trust_region_strategy_type =
      algorithm == 0 ? ceres::LEVENBERG_MARQUARDT : ceres::DOGLEG;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  options.max_num_iterations = n_iter;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  //    std::cout << summary.FullReport() << "\n";
}

class ReprojectionErrorJ : public ceres::SizedCostFunction<2, 3> {
 public:
  ReprojectionErrorJ(Sophus::SE3Group<double> _Tw2c,
                     Eigen::Matrix<double, 2, 1> _observed)
      : Tw2c(_Tw2c), observed(_observed) {}
  virtual ~ReprojectionErrorJ() {}

  virtual bool Evaluate(double const* const* point, double* residuals,
                        double** jacobians) const {
    typename Eigen::Matrix<double, 3, 1, Eigen::ColMajor>::ConstMapType v3point(
        point[0]);
    Eigen::Matrix<double, 3, 1> p = Tw2c * v3point;
    double xp = p[0] / p[2];
    double yp = p[1] / p[2];
    // The error is the difference between the predicted and observed position.
    residuals[0] = observed[0] - xp;
    residuals[1] = observed[1] - yp;

    if (jacobians != NULL && jacobians[0] != NULL) {
      Eigen::Matrix<double, 2, 3> point_jac;
      jacobian_xyz2uv(p, Tw2c.rotationMatrix(), point_jac);
      for (int iota = 0; iota < 2; ++iota) {
        for (int jack = 0; jack < 3; ++jack) {
          jacobians[0][iota * 3 + jack] = point_jac(iota, jack);
        }
      }
    }
    return true;
  }
  Sophus::SE3Group<double> Tw2c;
  Eigen::Matrix<double, 2, 1> observed;
};
void triangulate_refineJ(
    const std::vector<Eigen::Vector3d,
                      Eigen::aligned_allocator<Eigen::Vector3d>>& obs,
    const std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>&
        frame_poses,
    Eigen::Vector3d& old_point, int n_iter, int algorithm) {
  // Create residuals for each observation in the bundle adjustment problem. The
  // parameters for cameras and points are added automatically.
  ceres::Problem problem;
  for (size_t i = 0; i < obs.size(); ++i) {
    // Each Residual block takes a point and a camera as input and outputs a 2
    // dimensional residual. Internally, the cost function stores the observed
    // image location and compares the reprojection against the observation.
    Eigen::Vector2d v2InPlane = project2d(obs[i]);
    ceres::CostFunction* cost_function =
        new ReprojectionErrorJ(frame_poses[i], v2InPlane);
    problem.AddResidualBlock(cost_function, NULL /* squared loss */,
                             old_point.data());
  }
  // Make Ceres automatically detect the bundle structure. Note that the
  // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
  // for standard bundle adjustment problems.
  ceres::Solver::Options options;
  options.trust_region_strategy_type =
      algorithm == 0 ? ceres::LEVENBERG_MARQUARDT : ceres::DOGLEG;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  options.max_num_iterations = n_iter;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  //    std::cout << summary.FullReport() << "\n";
}
#endif
