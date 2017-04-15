#ifndef _TRIANGULATE_FAST_HPP_
#define _TRIANGULATE_FAST_HPP_
#include <Eigen/Core>

namespace okvis {

//this function stolen from okvis, the only change is disabling flipping signs
Eigen::Vector4d triangulateFastLocal(const Eigen::Vector3d& p1,
                                     const Eigen::Vector3d& e1,
                                     const Eigen::Vector3d& p2,
                                     const Eigen::Vector3d& e2, double sigma,
                                     bool& isValid, bool& isParallel);

}
#endif
