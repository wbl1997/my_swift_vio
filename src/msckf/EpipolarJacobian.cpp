#include "msckf/EpipolarJacobian.hpp"
#include "vio/eigen_utils.h"

namespace okvis {
EpipolarJacobian::EpipolarJacobian(const Eigen::Quaterniond& R_CjCk,
                                   const Eigen::Vector3d& t_CjCk,
                                   const Eigen::Vector3d& fj,
                                   const Eigen::Vector3d& fk)
    : R_CjCk_(R_CjCk), t_CjCk_(t_CjCk), fj_(fj), fk_(fk) {

}
double EpipolarJacobian::evaluate() const {
    return (R_CjCk_*fk_).dot(t_CjCk_.cross(fj_));
}
void EpipolarJacobian::de_dtheta_CjCk(Eigen::Matrix<double, 1, 3>* jac) const {
    *jac = (R_CjCk_*fk_).transpose() * vio::skew3d(t_CjCk_.cross(fj_));
}
void EpipolarJacobian::de_dfj(Eigen::Matrix<double, 1, 3>* jac) const {
    *jac = (R_CjCk_*fk_).transpose() * vio::skew3d(t_CjCk_);
}
void EpipolarJacobian::de_dt_CjCk(Eigen::Matrix<double, 1, 3>* jac) const {
    *jac = - (R_CjCk_*fk_).transpose() * vio::skew3d(fj_);
};
void EpipolarJacobian::de_dfk(Eigen::Matrix<double, 1, 3>* jac) const {
    *jac = (t_CjCk_.cross(fj_)).transpose() * R_CjCk_.toRotationMatrix();
}
}  // namespace okvis
