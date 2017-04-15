#include <okvis/triangulateFast.hpp>
#include <Eigen/Geometry> //cross product
#include <Eigen/LU> //inversewithcheck
namespace okvis{

// Triangulate the intersection of two rays.
Eigen::Vector4d triangulateFastLocal(const Eigen::Vector3d& p1,
                                     const Eigen::Vector3d& e1,
                                     const Eigen::Vector3d& p2,
                                     const Eigen::Vector3d& e2, double sigma,
                                     bool& isValid, bool& isParallel) {
    isParallel = false; // This should be the default.
    // But parallel and invalid is not the same. Points at infinity are valid and parallel.
    isValid = false; // hopefully this will be reset to true.

    // stolen and adapted from the Kneip toolchain geometric_vision/include/geometric_vision/triangulation/impl/triangulation.hpp
    Eigen::Vector3d t12 = p2 - p1;

    // check parallel
    /*if (t12.dot(e1) - t12.dot(e2) < 1.0e-12) {
    if ((e1.cross(e2)).norm() < 6 * sigma) {
      isValid = true;  // check parallel
      isParallel = true;
      return (Eigen::Vector4d((e1[0] + e2[0]) / 2.0, (e1[1] + e2[1]) / 2.0,
                              (e1[2] + e2[2]) / 2.0, 1e-2).normalized());
    }
  }*/

    Eigen::Vector2d b;
    b[0] = t12.dot(e1);
    b[1] = t12.dot(e2);
    Eigen::Matrix2d A;
    A(0, 0) = e1.dot(e1);
    A(1, 0) = e1.dot(e2);
    A(0, 1) = -A(1, 0);
    A(1, 1) = -e2.dot(e2);

    if (A(1, 0) < 0.0) {
        A(1, 0) = -A(1, 0);
        A(0, 1) = -A(0, 1);
        // wrong viewing direction
    };

    bool invertible;
    Eigen::Matrix2d A_inverse;
    A.computeInverseWithCheck(A_inverse, invertible, 1.0e-6);
    Eigen::Vector2d lambda = A_inverse * b;
    if (!invertible) {
        isParallel = true; // let's note this.
        // parallel. that's fine. but A is not invertible. so handle it separately.
        if ((e1.cross(e2)).norm() < 6 * sigma){
            isValid = true;  // check parallel
        }
        return (Eigen::Vector4d((e1[0] + e2[0]) / 2.0, (e1[1] + e2[1]) / 2.0,
                (e1[2] + e2[2]) / 2.0, 1e-3).normalized());
    }

    Eigen::Vector3d xm = lambda[0] * e1 + p1;
    Eigen::Vector3d xn = lambda[1] * e2 + p2;
    Eigen::Vector3d midpoint = (xm + xn) / 2.0;

    // check it
    Eigen::Vector3d error = midpoint - xm;
    Eigen::Vector3d diff = midpoint - (p1 + 0.5 * t12);
    const double diff_sq = diff.dot(diff);
    const double chi2 = error.dot(error) * (1.0 / (diff_sq * sigma * sigma));

    isValid = true;
    if (chi2 > 9) {
        isValid = false;  // reject large chi2-errors
    }

    // flip if necessary //HUai: flipped often results in divergence after Guass optimization
//    if (diff.dot(e1) < 0) {
//        std::cout <<"flipped "<<std::endl;
//        midpoint = (p1 + 0.5 * t12) - diff;
//    }


    return Eigen::Vector4d(midpoint[0], midpoint[1], midpoint[2], 1.0).normalized();
}

}
