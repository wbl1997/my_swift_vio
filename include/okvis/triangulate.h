#ifndef TRIANGULATE_H_
#define TRIANGULATE_H_
// direct linear triangulation method see Hartley and Zisserman 2003
// given a bunch of normalized image measurements of a point, and the camera
// poses, Pw2ck, estimate the point's position in the world implemented with
// TooN library
/*TooN::Vector<3> Get_X_from_xP_lin(std::vector<TooN::Vector<2> >&
vV2Xn,std::vector<TooN::SE3<double>>& vSE3){ assert(vV2Xn.size()==vSE3.size());
        int K=vV2Xn.size();
        Matrix<-1, 4> A(2*K, 4);
        //A=Zeros;
        for( int k = 0; k<K; ++k){
                A[2*k].slice<0,3>()=vV2Xn[k][0]*vSE3[k].get_rotation().get_matrix()[2]-vSE3[k].get_rotation().get_matrix()[0];
                A[2*k][3]=vV2Xn[k][0]*vSE3[k].get_translation()[2]-vSE3[k].get_translation()[0];
                A[2*k+1].slice<0,3>()=vV2Xn[k][1]*vSE3[k].get_rotation().get_matrix()[2]-vSE3[k].get_rotation().get_matrix()[1];
                A[2*k+1][3]=vV2Xn[k][1]*vSE3[k].get_translation()[2]-vSE3[k].get_translation()[1];
        }
        TooN::SVD<-1,4> svdM(2*K,4);
        svdM.compute(A);
        TooN::Matrix<4> m4temp=svdM.get_VT();
        TooN::Vector<4> v4Xhomog=svdM.get_VT()[3];
        return v4Xhomog.slice<0,3>()/v4Xhomog[3];
}
void TestGet_X_from_xP_lin()
{
        TooN::Vector<3> point=makeVector(1.5,3,25);
        vector< SE3<double> > vse3CFromW(3);
        vector< TooN::Vector<2> > vV2ImPlane(3);
        SO3<double> so3M(makeVector(0.0,0.0,1.0), makeVector(0.3, 0.05, 2.5));
        vse3CFromW[0].get_rotation()=so3M;
        vse3CFromW[0].get_translation()=makeVector(0.0, 0.0, 0.7);
        SO3<double> so3N(makeVector(0.0,0.0,1.0),  makeVector(0, 0.1, 1));
        vse3CFromW[1].get_rotation()=so3N;
        vse3CFromW[1].get_translation()=makeVector(1.0, -0.3, 3.0);
        SO3<double> so3P(makeVector(0.0,0.0,1.0),  makeVector(-0.5, 0.3, 5));
        vse3CFromW[2].get_rotation()=so3P;
        vse3CFromW[2].get_translation()=makeVector(1.0, -0.2, -5.0);
        for (int i=0; i<3; ++i){
                TooN::Vector<3> v3Cam = vse3CFromW[i]* point;
                if(v3Cam[2] < 0.001){
                        cerr<<"Some point "<<i<<" is behind the image"<<endl;
                        return;
                }
                vV2ImPlane[i]=project(v3Cam);
        }
        TooN::Vector<3> res=Get_X_from_xP_lin(vV2ImPlane, vse3CFromW);
        cout<<"residual squared "<<(res-point)*(res-point)<<endl;
}*/

#include <Eigen/SVD>
#include <sophus/se3.hpp>
#include <vector>

/**
 * @brief direct linear triangulation method see Hartley and Zisserman 2003,
 * implemented with Eigen and Sophus, however this function does not take care
 * of observations of infinity point, and repetitive static observations of a
 * point
 * @param vV2Xn, each observation is in image plane z=1, (\bar{x}, \bar{y}, 1)
 * @param vSE3, each of frame_poses is Tw2c(i)
 * @param pSingular, output the singular values for debugging if not NULL
 * @return
 */
Eigen::Vector4d Get_X_from_xP_lin(
    std::vector<Eigen::Vector3d,
                Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>& vV2Xn,
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>& vSE3,
    Eigen::Matrix<double, Eigen::Dynamic, 1>* pSingular =
        (Eigen::Matrix<double, Eigen::Dynamic, 1>*)(NULL));

/**
 * @brief direct linear triangulation method see Hartley and Zisserman 2003,
 * implemented with Eigen and Sophus
 * @param vV3Xn, each observation is in image plane z=1, (\bar{x}, \bar{y}, 1)
 * @param vSE3,  each of the frame_poses is Tw2c(i)
 * @param isValid, checks if the A matrix is rank deficient(<3 in this case).
 * Rank deficiency can occur with identical observations of a point, or pure
 * rotation
 * @param isParallel, checks if the point is at or close to infinity
 * @return
 */
Eigen::Vector4d Get_X_from_xP_lin(
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>&
        vV3Xn,
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>& vSE3,
    bool& isValid, bool& isParallel);

// two view triangulation using mid-point method as discussed in R Hartley,
// sec 5.3, "Triangulation", COMPUTER VISION AND IMAGE UNDERSTANDING, Vol. 68,
// No. 2, November, pp. 146–157, 1997
Eigen::Vector3d triangulate2View_midpoint(
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
        vV2Xn,
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>& vSE3);
// based on alex flint's implementation of triangulation,
// https://github.com/alexflint/triangulation/blob/master/triangulate.py
// midpoint triangulation of 2 or multiple rays
// it is verified (1)if only two views is used to triangulate a point,
// triangulate_midpoint and triangulate2view_midpoint gives identical result (2)
// triangulate_midpoint gives identical result as its python counterpart in alex
// flint's triangulation package
Eigen::Vector3d triangulate_midpoint(
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
        vV2Xn,
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>& vSE3,
    int how_many = 0);

void triangulate_refine_GN(
    const std::vector<Eigen::Vector3d,
                      Eigen::aligned_allocator<Eigen::Vector3d>>& obs,
    const std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>&
        frame_poses,
    Eigen::Vector3d& old_point, int n_iter);
// autodiff Jacobian
void triangulate_refine(
    const std::vector<Eigen::Vector3d,
                      Eigen::aligned_allocator<Eigen::Vector3d>>& obs,
    const std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>&
        frame_poses,
    Eigen::Vector3d& old_point, int n_iter, int algorithm = 0);
// analytic Jacobian
void triangulate_refineJ(
    const std::vector<Eigen::Vector3d,
                      Eigen::aligned_allocator<Eigen::Vector3d>>& obs,
    const std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>&
        frame_poses,
    Eigen::Vector3d& old_point, int n_iter, int algorithm = 0);
template <typename scalar>
scalar CalcMHWScore(std::vector<scalar>& scores) {
  scalar median;
  size_t size = scores.size();

  sort(scores.begin(), scores.end());

  if (size % 2 == 0) {
    median = (scores[size / 2 - 1] + scores[size / 2]) / 2;
  } else {
    median = scores[size / 2];
  }
  return median;
}
template <typename T>
inline Eigen::Matrix<T, 2, 1> project2d(const Eigen::Matrix<T, 3, 1>& v) {
  return v.template head<2>() / v[2];
}
template <typename T>
inline Eigen::Matrix<T, 3, 1> unproject2d(const Eigen::Matrix<T, 2, 1>& v) {
  return Eigen::Matrix<T, 3, 1>(v[0], v[1], T(1.0));
}
/// Jacobian of point projection on unit plane (focal length = 1) in frame (f).
/// J=\frac{\partial y-f(x)}{\partial x}, f(x)= project2d(Rx+t)
inline static void jacobian_xyz2uv(const Eigen::Vector3d& p_in_f,
                                   const Eigen::Matrix3d& R_f_w,
                                   Eigen::Matrix<double, 2, 3>& point_jac) {
  const double z_inv = 1.0 / p_in_f[2];
  const double z_inv_sq = z_inv * z_inv;
  point_jac(0, 0) = z_inv;
  point_jac(0, 1) = 0.0;
  point_jac(0, 2) = -p_in_f[0] * z_inv_sq;
  point_jac(1, 0) = 0.0;
  point_jac(1, 1) = z_inv;
  point_jac(1, 2) = -p_in_f[1] * z_inv_sq;
  point_jac = -point_jac * R_f_w;
}
template <class Derived>
inline typename Derived::Scalar norm_max(const Eigen::MatrixBase<Derived>& v) {
  typename Derived::Scalar max = typename Derived::Scalar(-1);
  for (int i = 0; i < v.size(); ++i) {
    typename Derived::Scalar abs = fabs(v[i]);
    if (abs > max) {
      max = abs;
    }
  }
  return max;
}
const double EPS = 1e-8;
// to refine triangulated results, use LM or Dogleg solver or GN as provided by
// Ceres solver this test verified that (1) the result using autodiff and
// analytical diff is identical for both LM and Dogleg; (2) Gauss-Newton method
// is very competitive for structure only point optimization compared to LM and
// Dogleg, given good initial estimate such as obtained by two view mid point
void TestGet_X_from_xP_lin();
// this test verifies that CostFunction operator() should only take variables
// and residuals as arguments, excluding fixed parameters and observations
int test_powell();
#endif
