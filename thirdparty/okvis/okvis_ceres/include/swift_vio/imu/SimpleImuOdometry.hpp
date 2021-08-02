#ifndef INCLUDE_SWIFT_VIO_SIMPLE_IMU_ODOMETRY_HPP_
#define INCLUDE_SWIFT_VIO_SIMPLE_IMU_ODOMETRY_HPP_

#include <Eigen/Core>
#include <swift_vio/Measurements.hpp>

#include <okvis/kinematics/Transformation.hpp>
#include <okvis/kinematics/operators.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/ImuMeasurements.hpp>
#include <okvis/Variables.hpp>
#include <okvis/assert_macros.hpp>

namespace swift_vio {
namespace ode {
template<typename Scalar>
int predictStates(const GenericImuMeasurementDeque<Scalar> & imuMeasurements,
            const Scalar gravityMag,
            std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>> &T_WS,
            Eigen::Matrix<Scalar, 9,1>& speedBgBa,
            const Scalar t_start, const Scalar t_end)
{
    Eigen::Matrix<Scalar, 3,1> r_old(T_WS.first), r_new(r_old), v_old(speedBgBa.template head<3>()), v_new(v_old);
    Eigen::Quaternion<Scalar> q_old(T_WS.second.conjugate()), q_new(q_old); //rotation from world to sensor
    assert(imuMeasurements.front().timeStamp <= t_start && imuMeasurements.back().timeStamp >= t_end);

    Scalar time = t_start;
    Scalar end = t_end;
    Scalar Delta_t = Scalar(0);
    bool hasStarted = false;
    int i = 0;
    Scalar nexttime;
    for(typename GenericImuMeasurementDeque<Scalar>::const_iterator it = imuMeasurements.begin();
          it != imuMeasurements.end(); ++it) {

      Eigen::Matrix<Scalar,3,1> omega_S_0 = it->template gyroscopes;
      Eigen::Matrix<Scalar,3,1> acc_S_0 = it->template accelerometers;
      Eigen::Matrix<Scalar,3,1> omega_S_1 = (it + 1)->template gyroscopes;
      Eigen::Matrix<Scalar,3,1> acc_S_1 = (it + 1)->template accelerometers;

      // time delta
      if ((it + 1) == imuMeasurements.end()) {
        nexttime = t_end;
      } else
        nexttime = (it + 1)->timeStamp;
      Scalar dt = nexttime - time;

      if (end < nexttime) {
        Scalar interval = nexttime - it->timeStamp;
        nexttime = t_end;
        dt = nexttime - time;
        if (dt == 0.0)
            break;
        const Scalar r = dt / interval;
        omega_S_1 = ((Scalar(1.0) - r) * omega_S_0 + r * omega_S_1).eval();
        acc_S_1 = ((Scalar(1.0) - r) * acc_S_0 + r * acc_S_1).eval();
      }

      if (dt <= Scalar(0.0)) {
        continue;
      }
      Delta_t += dt;

      if (!hasStarted) {
        hasStarted = true;
        const Scalar r = dt / (nexttime - it->timeStamp);
        omega_S_0 = (r * omega_S_0 + (Scalar(1.0) - r) * omega_S_1).eval();
        acc_S_0 = (r * acc_S_0 + (Scalar(1.0) - r) * acc_S_1).eval();
      }

      // actual propagation
      Eigen::Matrix<Scalar,3,1> a_est = Scalar(0.5)*(acc_S_0+acc_S_1) - speedBgBa.template tail<3>();
      Eigen::Matrix<Scalar,3,1> w_est = Scalar(0.5)*(omega_S_0+omega_S_1) - speedBgBa.template segment<3>(3);
      Eigen::Matrix<Scalar,3,1> gW(Scalar(0), Scalar(0), -gravityMag);

      Eigen::Quaternion<Scalar> qb = okvis::kinematics::rvec2quat(-w_est*dt);
      q_new=qb*q_old;

      Eigen::Matrix<Scalar,3,1> vel_inc1=(q_old.conjugate()._transformVector(a_est*dt)+q_new.conjugate()._transformVector(a_est*dt))*Scalar(0.5);
      Eigen::Matrix<Scalar,3,1> vel_inc2=gW*dt;

      v_new= v_old + vel_inc1+vel_inc2;
      r_new= r_old + (v_new+v_old)*dt*Scalar(0.5);

      time = nexttime;
      ++i;

      if (nexttime == t_end)
        break;

      r_old=r_new;
      v_old=v_new;
      q_old=q_new;
    }
    assert(nexttime == t_end);

    T_WS.second = q_new.conjugate();
    T_WS.first = r_new;
    speedBgBa. template head<3>()=v_new;
    return i;
}

/**
 * @brief predictStates accumulate the rotation vector given gyro measurements.
 * i.e., \f$ \int_{t_1}^{t_2} \hat{\omega} - \mathbf{b}_g \f$
 */
template <typename Scalar>
int predictStates(
    const std::deque<
        okvis::Measurement<Eigen::Vector3d>,
        Eigen::aligned_allocator<okvis::Measurement<Eigen::Vector3d>>>
        &gyroMeasurements,
    Eigen::Vector3d *rotVector, const Eigen::Matrix<Scalar, 3, 1> &gyroBias,
    const okvis::Time t_start, const okvis::Time t_end) {
  rotVector->setZero();
  assert(gyroMeasurements.front().timeStamp <= t_start &&
         gyroMeasurements.back().timeStamp >= t_end);

  okvis::Time time = t_start;
  okvis::Time end = t_end;
  Scalar Delta_t = Scalar(0);
  bool hasStarted = false;
  int i = 0;
  okvis::Time nexttime;
  for (typename std::deque<okvis::Measurement<Eigen::Vector3d>,
                           Eigen::aligned_allocator<okvis::Measurement<
                               Eigen::Vector3d>>>::const_iterator it =
           gyroMeasurements.begin();
       it != gyroMeasurements.end(); ++it) {
    Eigen::Matrix<Scalar, 3, 1> omega_S_0 = it->measurement;
    Eigen::Matrix<Scalar, 3, 1> omega_S_1 = (it + 1)->measurement;
    // time delta
    if ((it + 1) == gyroMeasurements.end()) {
      nexttime = t_end;
    } else
      nexttime = (it + 1)->timeStamp;

    Scalar dt = (Scalar)(nexttime - time).toSec();
    if (end < nexttime) {
      Scalar interval = (Scalar)(nexttime - it->timeStamp).toSec();
      nexttime = t_end;
      dt = (Scalar)(nexttime - time).toSec();
      if (dt == 0.0)
        break;
      const Scalar r = dt / interval;
      omega_S_1 = ((Scalar(1.0) - r) * omega_S_0 + r * omega_S_1).eval();
    }

    if (dt <= Scalar(0.0)) {
      continue;
    }
    Delta_t += dt;

    if (!hasStarted) {
      hasStarted = true;
      const Scalar r = dt / ((Scalar)(nexttime - it->timeStamp).toSec());
      omega_S_0 = (r * omega_S_0 + (Scalar(1.0) - r) * omega_S_1).eval();
    }

    // actual propagation
    Eigen::Matrix<Scalar, 3, 1> w_est =
        Scalar(0.5) * (omega_S_0 + omega_S_1) - gyroBias;
    (*rotVector) += w_est * dt;

    time = nexttime;
    ++i;

    if (nexttime == t_end)
      break;
  }
  assert(nexttime == t_end);
  return i;
}

// time_pair[0] timestamp of the provided state values. time_pair[0] >= time_pair[1],
template<typename Scalar>
int predictStatesBackward(const GenericImuMeasurementDeque<Scalar> & imuMeasurements,
                    const Scalar gravityMag,
                    std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>> &T_WS,
                    Eigen::Matrix<Scalar, 9,1>& speedBgBa,
                    const Scalar t_start, const Scalar t_end)
{
    Eigen::Matrix<Scalar, 3,1> r_old(T_WS.first), r_new(r_old), v_old(speedBgBa.template head<3>()), v_new(v_old);
    Eigen::Quaternion<Scalar> q_old(T_WS.second.conjugate()), q_new(q_old); //rotation from world to sensor
    assert(imuMeasurements.front().timeStamp <= t_end && imuMeasurements.back().timeStamp >= t_start);

    Scalar time = t_start;
    Scalar end = t_end;
    Scalar Delta_t = Scalar(0);
    bool hasStarted = false;
    int i = 0;
    Scalar nexttime;
    for(typename GenericImuMeasurementDeque<Scalar>::const_reverse_iterator it = imuMeasurements.rbegin();
          it != imuMeasurements.rend(); ++it) {

      Eigen::Matrix<Scalar,3,1> omega_S_0 = it->template gyroscopes;
      Eigen::Matrix<Scalar,3,1> acc_S_0 = it->template accelerometers;
      Eigen::Matrix<Scalar,3,1> omega_S_1 = (it + 1)->template gyroscopes;
      Eigen::Matrix<Scalar,3,1> acc_S_1 = (it + 1)->template accelerometers;

      // time delta
      if ((it + 1) == imuMeasurements.rend()) {
        nexttime = t_end;
      } else
        nexttime = (it + 1)->timeStamp;
      Scalar dt = nexttime - time;

      if (end > nexttime) {
        Scalar interval = nexttime - it->timeStamp;
        nexttime = t_end;
        dt = nexttime - time;
        if (dt == 0.0)
            break;
        const Scalar r = dt / interval;
        omega_S_1 = ((Scalar(1.0) - r) * omega_S_0 + r * omega_S_1).eval();
        acc_S_1 = ((Scalar(1.0) - r) * acc_S_0 + r * acc_S_1).eval();
      }

      if (dt >= Scalar(0.0)) {
        continue;
      }
      Delta_t += dt;

      if (!hasStarted) {
        hasStarted = true;
        const Scalar r = dt / (nexttime - it->timeStamp);
        omega_S_0 = (r * omega_S_0 + (Scalar(1.0) - r) * omega_S_1).eval();
        acc_S_0 = (r * acc_S_0 + (Scalar(1.0) - r) * acc_S_1).eval();
      }

      // actual propagation

      Eigen::Matrix<Scalar,3,1> a_est = Scalar(0.5)*(acc_S_0+acc_S_1) - speedBgBa.template tail<3>();
      Eigen::Matrix<Scalar,3,1> w_est = Scalar(0.5)*(omega_S_0+omega_S_1) - speedBgBa.template segment<3>(3);
      Eigen::Matrix<Scalar,3,1> gW(Scalar(0), Scalar(0), -gravityMag);

      Eigen::Quaternion<Scalar> qb = okvis::kinematics::rvec2quat(-w_est*dt);
      q_new=qb*q_old;

      Eigen::Matrix<Scalar,3,1> vel_inc1=(q_old.conjugate()._transformVector(a_est*dt)+q_new.conjugate()._transformVector(a_est*dt))*Scalar(0.5);
      Eigen::Matrix<Scalar,3,1> vel_inc2=gW*dt;

      v_new= v_old + vel_inc1+vel_inc2;
      r_new= r_old + (v_new+v_old)*dt*Scalar(0.5);

      time = nexttime;
      ++i;

      if (nexttime == t_end)
        break;

      r_old=r_new;
      v_old=v_new;
      q_old=q_new;
    }
    assert(nexttime == t_end);

    T_WS.second = q_new.conjugate();
    T_WS.first = r_new;
    speedBgBa.template head<3>()=v_new;
    return i;
}

template<typename Scalar>
int predictStates(const okvis::ImuMeasurementDeque & imuMeasurements,
            const Scalar gravityMag,
            std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>> &T_WS,
            Eigen::Matrix<Scalar, 9,1>& speedBgBa,
            const okvis::Time t_start, const okvis::Time t_end)
{
    Eigen::Matrix<Scalar, 3,1> r_old(T_WS.first), r_new(r_old), v_old(speedBgBa.template head<3>()), v_new(v_old);
    Eigen::Quaternion<Scalar> q_old(T_WS.second.conjugate()), q_new(q_old); //rotation from world to sensor
    assert(imuMeasurements.front().timeStamp <= t_start && imuMeasurements.back().timeStamp >= t_end);

    okvis::Time time = t_start;
    okvis::Time end = t_end;
    Scalar Delta_t = Scalar(0);
    bool hasStarted = false;
    int i = 0;
    okvis::Time nexttime;
    for(typename okvis::ImuMeasurementDeque::const_iterator it = imuMeasurements.begin();
          it != imuMeasurements.end(); ++it) {

      Eigen::Matrix<Scalar,3,1> omega_S_0 = it->measurement.gyroscopes;
      Eigen::Matrix<Scalar,3,1> acc_S_0 = it->measurement.accelerometers;
      Eigen::Matrix<Scalar,3,1> omega_S_1 = (it + 1)->measurement.gyroscopes;
      Eigen::Matrix<Scalar,3,1> acc_S_1 = (it + 1)->measurement.accelerometers;

      // time delta
      if ((it + 1) == imuMeasurements.end()) {
        nexttime = t_end;
      } else
        nexttime = (it + 1)->timeStamp;

      Scalar dt = (Scalar)(nexttime - time).toSec();

      if (end < nexttime) {
        Scalar interval = (Scalar)(nexttime - it->timeStamp).toSec();
        nexttime = t_end;
        dt =(Scalar)(nexttime - time).toSec();
        if (dt == 0.0)
            break;
        const Scalar r = dt / interval;
        omega_S_1 = ((Scalar(1.0) - r) * omega_S_0 + r * omega_S_1).eval();
        acc_S_1 = ((Scalar(1.0) - r) * acc_S_0 + r * acc_S_1).eval();
      }

      if (dt <= Scalar(0.0)) {
        continue;
      }
      Delta_t += dt;

      if (!hasStarted) {
        hasStarted = true;
        const Scalar r = dt / ((Scalar)(nexttime - it->timeStamp).toSec());
        omega_S_0 = (r * omega_S_0 + (Scalar(1.0) - r) * omega_S_1).eval();
        acc_S_0 = (r * acc_S_0 + (Scalar(1.0) - r) * acc_S_1).eval();
      }

      // actual propagation
      Eigen::Matrix<Scalar,3,1> a_est = Scalar(0.5)*(acc_S_0+acc_S_1) - speedBgBa.template tail<3>();
      Eigen::Matrix<Scalar,3,1> w_est = Scalar(0.5)*(omega_S_0+omega_S_1) - speedBgBa.template segment<3>(3);
      Eigen::Matrix<Scalar,3,1> gW(Scalar(0), Scalar(0), -gravityMag);

      Eigen::Quaternion<Scalar> qb = okvis::kinematics::rvec2quat(-w_est*dt);
      q_new=qb*q_old;

      Eigen::Matrix<Scalar,3,1> vel_inc1=(q_old.conjugate()._transformVector(a_est*dt)+q_new.conjugate()._transformVector(a_est*dt))*Scalar(0.5);
      Eigen::Matrix<Scalar,3,1> vel_inc2=gW*dt;

      v_new= v_old + vel_inc1+vel_inc2;
      r_new= r_old + (v_new+v_old)*dt*Scalar(0.5);

      time = nexttime;
      ++i;

      if (nexttime == t_end)
        break;

      r_old=r_new;
      v_old=v_new;
      q_old=q_new;
    }
    assert(nexttime == t_end);

    T_WS.second = q_new.conjugate();
    T_WS.first = r_new;
    speedBgBa. template head<3>()=v_new;
    return i;
}

// time_pair[0] timestamp of the provided state values. time_pair[0] >= time_pair[1],
template<typename Scalar>
int predictStatesBackward(const okvis::ImuMeasurementDeque & imuMeasurements,
                    const Scalar gravityMag,
                    std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>> &T_WS,
                    Eigen::Matrix<Scalar, 9,1>& speedBgBa,
                    const okvis::Time t_start, const okvis::Time t_end)
{
    Eigen::Matrix<Scalar, 3,1> r_old(T_WS.first), r_new(r_old), v_old(speedBgBa.template head<3>()), v_new(v_old);
    Eigen::Quaternion<Scalar> q_old(T_WS.second.conjugate()), q_new(q_old); //rotation from world to sensor
    assert(imuMeasurements.front().timeStamp <= t_end && imuMeasurements.back().timeStamp >= t_start);
    // if this assertion fails during optimization, it often means the time offset variables diverge. Solution:
    // either try to enlarge the range of imu reading segment, or decrease the time variables' std to effectively lock them

    okvis::Time time = t_start;
    okvis::Time end = t_end;
    Scalar Delta_t = Scalar(0);
    bool hasStarted = false;
    int i = 0;
    okvis::Time nexttime;
    for(typename okvis::ImuMeasurementDeque::const_reverse_iterator it = imuMeasurements.rbegin();
          it != imuMeasurements.rend(); ++it) {

      Eigen::Matrix<Scalar,3,1> omega_S_0 = it->measurement.gyroscopes;
      Eigen::Matrix<Scalar,3,1> acc_S_0 = it->measurement.accelerometers;
      Eigen::Matrix<Scalar,3,1> omega_S_1 = (it + 1)->measurement.gyroscopes;
      Eigen::Matrix<Scalar,3,1> acc_S_1 = (it + 1)->measurement.accelerometers;

      // time delta
      if ((it + 1) == imuMeasurements.rend()) {
        nexttime = t_end;
      } else
        nexttime = (it + 1)->timeStamp;
      Scalar dt = (Scalar)(nexttime - time).toSec();

      if (end > nexttime) {
        Scalar interval = (Scalar)(nexttime - it->timeStamp).toSec();
        nexttime = t_end;
        dt = (Scalar)(nexttime - time).toSec();
        if (dt == 0.0)
            break;
        const Scalar r = dt / interval;
        omega_S_1 = ((Scalar(1.0) - r) * omega_S_0 + r * omega_S_1).eval();
        acc_S_1 = ((Scalar(1.0) - r) * acc_S_0 + r * acc_S_1).eval();
      }

      if (dt >= Scalar(0.0)) {
        continue;
      }
      Delta_t += dt;

      if (!hasStarted) {
        hasStarted = true;
        const Scalar r = dt / ((Scalar)(nexttime - it->timeStamp).toSec());
        omega_S_0 = (r * omega_S_0 + (Scalar(1.0) - r) * omega_S_1).eval();
        acc_S_0 = (r * acc_S_0 + (Scalar(1.0) - r) * acc_S_1).eval();
      }

      // actual propagation

      Eigen::Matrix<Scalar,3,1> a_est = Scalar(0.5)*(acc_S_0+acc_S_1) - speedBgBa.template tail<3>();
      Eigen::Matrix<Scalar,3,1> w_est = Scalar(0.5)*(omega_S_0+omega_S_1) - speedBgBa.template segment<3>(3);
      Eigen::Matrix<Scalar,3,1> gW(Scalar(0), Scalar(0), -gravityMag);

      Eigen::Quaternion<Scalar> qb = okvis::kinematics::rvec2quat(-w_est*dt);
      q_new=qb*q_old;

      Eigen::Matrix<Scalar,3,1> vel_inc1=(q_old.conjugate()._transformVector(a_est*dt)+q_new.conjugate()._transformVector(a_est*dt))*Scalar(0.5);
      Eigen::Matrix<Scalar,3,1> vel_inc2=gW*dt;

      v_new= v_old + vel_inc1+vel_inc2;
      r_new= r_old + (v_new+v_old)*dt*Scalar(0.5);

      time = nexttime;
      ++i;

      if (nexttime == t_end)
        break;

      r_old=r_new;
      v_old=v_new;
      q_old=q_new;
    }
    assert(nexttime == t_end);

    T_WS.second = q_new.conjugate();
    T_WS.first = r_new;
    speedBgBa.template head<3>()=v_new;
    return i;
}

// linear interpolation
inline void interpolateInertialData(const okvis::ImuMeasurementDeque &imuMeas,
                                    const okvis::Time &queryTime,
                                    okvis::ImuMeasurement &queryValue) {
  auto iterLeft = imuMeas.begin(), iterRight = imuMeas.end();
  if (iterLeft->timeStamp > queryTime)
    throw std::runtime_error("iterLeft->timeStamp > queryTime: Imu "
                             "measurements has wrong timestamps");
  for (auto iter = imuMeas.begin(); iter != imuMeas.end(); ++iter) {
    if (iter->timeStamp < queryTime) {
      iterLeft = iter;
    } else if (iter->timeStamp == queryTime) {
      queryValue = *iter;
      return;
    } else {
      iterRight = iter;
      break;
    }
  }
  double ratio = (queryTime - iterLeft->timeStamp).toSec() /
                 (iterRight->timeStamp - iterLeft->timeStamp).toSec();
  queryValue.timeStamp = queryTime;
  Eigen::Vector3d omega_S0 =
      (iterRight->measurement.gyroscopes - iterLeft->measurement.gyroscopes) *
          ratio +
      iterLeft->measurement.gyroscopes;
  Eigen::Vector3d acc_S0 = (iterRight->measurement.accelerometers -
                            iterLeft->measurement.accelerometers) *
                               ratio +
                           iterLeft->measurement.accelerometers;
  queryValue.measurement.gyroscopes = omega_S0;
  queryValue.measurement.accelerometers = acc_S0;
}
}  // namespace ode
}  // namespace swift_vio
#endif

