#ifndef INCLUDE_MSCKF_EPIPOLAR_JACOBIAN_HPP_
#define INCLUDE_MSCKF_EPIPOLAR_JACOBIAN_HPP_
#include <Eigen/Geometry>

#include <vio/eigen_utils.h>

namespace okvis {
class EpipolarJacobian
{
public:
    inline EpipolarJacobian(const Eigen::Quaterniond& R_CjCk,
                            const Eigen::Vector3d& t_CjCk,
                            const Eigen::Vector3d& fj,
                            const Eigen::Vector3d& fk);
    inline double evaluate() const;
    inline void de_dtheta_CjCk(Eigen::Matrix<double, 1, 3>* jac) const;
    inline void de_dfj(Eigen::Matrix<double, 1, 3>* jac) const;
    inline void de_dt_CjCk(Eigen::Matrix<double, 1, 3>* jac) const;
    inline void de_dfk(Eigen::Matrix<double, 1, 3>* jac) const;

private:
    const Eigen::Quaterniond R_CjCk_;
    const Eigen::Vector3d t_CjCk_;
    const Eigen::Vector3d fj_; // z = 1
    const Eigen::Vector3d fk_; // z = 1
};

inline EpipolarJacobian::EpipolarJacobian(const Eigen::Quaterniond& R_CjCk,
                                   const Eigen::Vector3d& t_CjCk,
                                   const Eigen::Vector3d& fj,
                                   const Eigen::Vector3d& fk)
    : R_CjCk_(R_CjCk), t_CjCk_(t_CjCk), fj_(fj), fk_(fk) {

}
inline double EpipolarJacobian::evaluate() const {
    return (R_CjCk_*fk_).dot(t_CjCk_.cross(fj_));
}
inline void EpipolarJacobian::de_dtheta_CjCk(Eigen::Matrix<double, 1, 3>* jac) const {
    *jac = (R_CjCk_*fk_).transpose() * vio::skew3d(t_CjCk_.cross(fj_));
}
inline void EpipolarJacobian::de_dfj(Eigen::Matrix<double, 1, 3>* jac) const {
    *jac = (R_CjCk_*fk_).transpose() * vio::skew3d(t_CjCk_);
}
inline void EpipolarJacobian::de_dt_CjCk(Eigen::Matrix<double, 1, 3>* jac) const {
    *jac = - (R_CjCk_*fk_).transpose() * vio::skew3d(fj_);
};
inline void EpipolarJacobian::de_dfk(Eigen::Matrix<double, 1, 3>* jac) const {
    *jac = (t_CjCk_.cross(fj_)).transpose() * R_CjCk_.toRotationMatrix();
}
} // namespace okvis
#endif // INCLUDE_MSCKF_EPIPOLAR_JACOBIAN_HPP_
