#ifndef INCLUDE_MSCKF_EPIPOLAR_JACOBIAN_HPP_
#define INCLUDE_MSCKF_EPIPOLAR_JACOBIAN_HPP_
#include <Eigen/Geometry>

namespace okvis {
class EpipolarJacobian
{
public:
    EpipolarJacobian(const Eigen::Quaterniond& R_CjCk,
                            const Eigen::Vector3d& t_CjCk,
                            const Eigen::Vector3d& fj,
                            const Eigen::Vector3d& fk);
    double evaluate() const;
    void de_dtheta_CjCk(Eigen::Matrix<double, 1, 3>* jac) const;
    void de_dfj(Eigen::Matrix<double, 1, 3>* jac) const;
    void de_dt_CjCk(Eigen::Matrix<double, 1, 3>* jac) const;
    void de_dfk(Eigen::Matrix<double, 1, 3>* jac) const;

private:
    const Eigen::Quaterniond R_CjCk_;
    const Eigen::Vector3d t_CjCk_;
    const Eigen::Vector3d fj_; // z = 1
    const Eigen::Vector3d fk_; // z = 1
};
} // namespace okvis
#endif // INCLUDE_MSCKF_EPIPOLAR_JACOBIAN_HPP_
