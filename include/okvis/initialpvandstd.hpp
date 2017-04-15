#ifndef INITIALPVANDSTD_HPP
#define INITIALPVANDSTD_HPP

#include <okvis/Parameters.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry> //quaterniond

namespace okvis {

struct InitialPVandStd{
    // S represents the nominal IMU sensor frame realized with the camera frame and the intersection of three accelerometers
    // W represents the world frame with z along the negative gravity direction and has
    // minimal rotation relative to the S frame at the initialization epoch
    okvis::Time stateTime; //epoch for the initialization values
    Eigen::Vector3d p_WS;
    Eigen::Quaterniond q_WS;
    Eigen::Vector3d v_WS;
    Eigen::Vector3d std_p_WS;
    Eigen::Vector3d std_q_WS; // std of $\delta \theta$ which is expressed in the world frame
    Eigen::Vector3d std_v_WS;

    InitialPVandStd();

    InitialPVandStd(const okvis::InitialState & rhs);

    void updatePose(const okvis::kinematics::Transformation & T_WS, const okvis::Time state_time);

    InitialPVandStd& operator = (const InitialPVandStd & other);

};

}//namespace
#endif // INITIALPVANDSTD_HPP

