#include "msckf/TwoViewGeometry.hpp"
namespace okvis {
double TwoViewGeometry::computeErrorEssentialMat(okvis::kinematics::Transformation T_ji,
                                Eigen::Vector3d bearing_i,
                                Eigen::Vector3d bearing_j,
                                double focal_length) {
  Eigen::Matrix3d E_ji = okvis::kinematics::crossMx(T_ji.r()) * T_ji.C();
  computeErrorEssentialMat(E_ji, bearing_i, bearing_j, focal_length);
}

/**
 * @brief deviationFromEpipolarLine compute deviation in pixels from the epipolar line.
 * see https://github.com/opencv/opencv/blob/master/modules/calib3d/src/fundam.cpp
 * @param E_ji t_ji X R_ji. Epipolar constraint is p_j' * t_ji X R_ji * p_i = 0.
 * @param bearing_i [x, y, 1] undistorted image coordinate at z=1 for point in image i
 * @param bearing_j [x, y, 1] undistorted image coordinate at z=1 for point in image j
 * @param focal_length nominal focal length to convert the epipolar line error into error of pixel unit.
 * @return squared distance to epipolar line. Distance has a unit of pixels.
 */
double TwoViewGeometry::computeErrorEssentialMat(Eigen::Matrix3d E_ji,
                                Eigen::Vector3d bearing_i,
                                Eigen::Vector3d bearing_j,
                                double focal_length) {
  Eigen::Vector3d abc = E_ji * bearing_i;
  double d1, d2, s1, s2;
  s2 = 1. / (abc[0] * abc[0] + abc[1] * abc[1]);
  d2 = bearing_j.dot(abc);

  abc = E_ji.transpose() * bearing_j;
  s1 = 1. / (abc[0] * abc[0] + abc[1] * abc[1]);
  d1 = bearing_i.dot(abc);
  double f2 = focal_length * focal_length;
  return (float)std::max(d1 * d1 * s1 * f2, d2 * d2 * s2 * f2);
}
} // namespace okvis
