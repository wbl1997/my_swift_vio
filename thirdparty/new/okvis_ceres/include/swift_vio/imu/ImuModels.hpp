#ifndef INCLUDE_SWIFT_VIO_IMU_ERROR_MODELS_HPP_
#define INCLUDE_SWIFT_VIO_IMU_ERROR_MODELS_HPP_

// Generic methods specific to each IMU model is encapsulated in the following classes.
// These models are to be used in constructing ceres::SizedCostFunction or
// ceres::AutoDiffCostFunction. Both require constant parameter block sizes at compile time.
// This is why we cannot use polymorphism for representing IMU models.

// The model parameter data are kept in the ImuModel class which
// is initialized and updated in the estimator.

// The IMU input reference frame or sensor frame denoted by S is affixed to
// the accelerometer triad, and its x-axis aligned to the accelerometer in the x direction.
// Its origin is at the intersection of the three accelerometers.
// Tts y-axis in the plane spanned by the two accelerometers at x and y
// direction while being close to the accelerometer at y-direction.
// The sensor rig body frame denoted by B is used to express the motion of the rig.
// It varies depending on the IMU model.

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <okvis/ModelSwitch.hpp>

namespace swift_vio {
static const int kBgBaDim = 6; // bg ba

template <typename T>
void vectorToLowerTriangularMatrix(const T* data, int startIndex, Eigen::Matrix<T, 3, 3>* mat33) {
  (*mat33)(0, 0) = data[startIndex];
  (*mat33)(0, 1) = 0;
  (*mat33)(0, 2) = 0;
  (*mat33)(1, 0) = data[startIndex + 1];
  (*mat33)(1, 1) = data[startIndex + 2];
  (*mat33)(1, 2) = 0;
  (*mat33)(2, 0) = data[startIndex + 3];
  (*mat33)(2, 1) = data[startIndex + 4];
  (*mat33)(2, 2) = data[startIndex + 5];
}

template <typename T>
void vectorToMatrix(const T* data, int startIndex, Eigen::Matrix<T, 3, 3>* mat33) {
  (*mat33)(0, 0) = data[startIndex];
  (*mat33)(0, 1) = data[startIndex + 1];
  (*mat33)(0, 2) = data[startIndex + 2];
  (*mat33)(1, 0) = data[startIndex + 3];
  (*mat33)(1, 1) = data[startIndex + 4];
  (*mat33)(1, 2) = data[startIndex + 5];
  (*mat33)(2, 0) = data[startIndex + 6];
  (*mat33)(2, 1) = data[startIndex + 7];
  (*mat33)(2, 2) = data[startIndex + 8];
}

template <typename T>
void invertLowerTriangularMatrix(const T* data, int startIndex, Eigen::Matrix<T, 3, 3>* mat33) {
  //  syms a b c d e f positive
  //  g = [a, 0, 0, b, c, 0, d, e, f]
  //  [ a, 0, 0]
  //  [ b, c, 0]
  //  [ d, e, f]
  //  inv(g)
  //  [                 1/a,        0,   0]
  //  [            -b/(a*c),      1/c,   0]
  //  [ (b*e - c*d)/(a*c*f), -e/(c*f), 1/f]
  (*mat33)(0, 0) = 1 / data[startIndex];
  (*mat33)(0, 1) = 0;
  (*mat33)(0, 2) = 0;
  (*mat33)(1, 0) = - data[startIndex + 1] / (data[startIndex] * data[startIndex + 2]);
  (*mat33)(1, 1) = 1 / data[startIndex + 2];
  (*mat33)(1, 2) = 0;
  (*mat33)(2, 0) = (data[startIndex + 1] * data[startIndex + 4] -
      data[startIndex + 2] * data[startIndex + 3]) /
      (data[startIndex] * data[startIndex + 2] * data[startIndex + 5]);
  (*mat33)(2, 1) = - data[startIndex + 4] / (data[startIndex + 2] * data[startIndex + 5]);
  (*mat33)(2, 2) = 1 / data[startIndex + 5];
}

/**
 * @brief The Imu_BG_BA class
 * The body frame is identical to the classic IMU sensor frame which has origin
 * at the accelerometer intersection, and x along x-accelrometer and y in the
 * plane spanned by the x- and y-accelerometer.
 * The accelerometer triad and the gyroscope triad are free of scaling error and misalignment.
 */
class Imu_BG_BA {
 public:
  static const int kModelId = 0;
  static const size_t kGlobalDim = kBgBaDim;
  static const size_t kAugmentedDim = 0;

  /**
   * @brief getAugmentedDim
   * @return dim of all the augmented params.
   */
  static inline int getAugmentedDim() { return kAugmentedDim; }
  /**
   * @brief getMinimalDim
   * @return minimal dim of all the params.
   */
  static inline int getMinimalDim() { return kGlobalDim; }
  /**
   * @brief getAugmentedMinimalDim
   * @return minimal dim of all augmented params.
   */
  static inline int getAugmentedMinimalDim() { return kAugmentedDim; }
  /**
   * get nominal values for augmented params.
   */
  template <typename T>
  static Eigen::Matrix<T, kAugmentedDim, 1> getNominalAugmentedParams() {
    return Eigen::Matrix<T, kAugmentedDim, 1>::Zero();
  }
  /**
   * predict IMU measurement from values in the body frame.
   * This function is used for testing purposes.
   */
  template <typename T>
  static void
  predict(const Eigen::Matrix<T, 3, 1> &bg, const Eigen::Matrix<T, 3, 1> &ba,
          const Eigen::Matrix<T, Eigen::Dynamic, 1> & /*extraParams*/,
          const Eigen::Matrix<T, 3, 1> &w_b, const Eigen::Matrix<T, 3, 1> &a_b,
          Eigen::Matrix<T, 3, 1> *w, Eigen::Matrix<T, 3, 1> *a) {
    *a = a_b + ba;
    *w = w_b + bg;
  }
  /**
   * correct IMU measurement to the body frame.
   * This function is used by the ceres::CostFunction.
   * @param[in] params bg ba and augmented Euclidean params.
   * @param[in] q_gyro_i orientation from the accelerometer triad input reference frame,
   *     i.e., the IMU sensor frame to the gyro triad input reference frame.
   * @param[in] w, a angular velocity and linear acceleration measured by the IMU.
   * @param[out] w_b, a_b angular velocity and linear acceleration in the body frame.
   */
  template <typename T>
  static void
  correct(const Eigen::Matrix<T, 3, 1> &bg, const Eigen::Matrix<T, 3, 1> &ba,
          const Eigen::Matrix<T, Eigen::Dynamic, 1> & /*params*/,
          const Eigen::Matrix<T, 3, 1> &w, const Eigen::Matrix<T, 3, 1> &a,
          Eigen::Matrix<T, 3, 1> *w_b, Eigen::Matrix<T, 3, 1> *a_b) {
    *a_b = a - ba;
    *w_b = w - bg;
  }

  static Eigen::VectorXd computeAugmentedParamsError(
      const Eigen::VectorXd& /*params*/) {
      return Eigen::VectorXd(0);
  }
};

/**
 * @brief The Imu_BG_BA_TG_TS_TA class
 * The body frame is the same as the IMU sensor frame which is
 * defined relative to an external sensor, e.g., the camera. Its 
 * orientation is fixed to the nominal value of R_SC0 and its origin is at 
 * the accelerometer intersection. Thus both accelerometer triad and 
 * gyroscope triad need to account for scaling effect (3), misalignment (3),
 * relative orientation (4, minimal 3) to the body frame.
 * This model also considers the g-sensitivity (9) of the gyroscope triad.
 * In other words, the remaining misalignment between the orthogonal
 *  accelerometer input reference frame (A) and the C frame is
 * absorbed into T_a, the IMU accelerometer misalignment matrix.
 *
 * IMU model
 * w_m = T_g * w_B + T_s * a_B + b_w + n_w
 * a_m = T_a * a_B + b_a + n_a = S * M * R_AB * a_B + b_a + n_a
 *
 * The A frame has origin at the accelerometers intersection and x-axis aligned
 * with accelerometer x.
 */
class Imu_BG_BA_TG_TS_TA {
 public:
  static const int kModelId = 1;
  static const size_t kAugmentedDim = 27;
  static const size_t kGlobalDim = kAugmentedDim + kBgBaDim;

  static inline int getAugmentedDim() { return kAugmentedDim; }
  static inline int getMinimalDim() { return kGlobalDim; }
  static inline int getAugmentedMinimalDim() { return kAugmentedDim; }
  template <typename T>
  static Eigen::Matrix<T, kAugmentedDim, 1> getNominalAugmentedParams() {
    Eigen::Matrix<T, 9, 1> eye;
    eye << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    Eigen::Matrix<T, kAugmentedDim, 1> augmentedParams;
    augmentedParams.template head<9>() = eye;
    augmentedParams.template tail<9>() = eye;
    return augmentedParams;
  }

  template <typename T>
  static void
  predict(const Eigen::Matrix<T, 3, 1> &bg, const Eigen::Matrix<T, 3, 1> &ba,
          const Eigen::Matrix<T, Eigen::Dynamic, 1> &params,
          const Eigen::Matrix<T, 3, 1> &w_b, const Eigen::Matrix<T, 3, 1> &a_b,
          Eigen::Matrix<T, 3, 1> *w, Eigen::Matrix<T, 3, 1> *a) {
    Eigen::Matrix<T, 3, 3> T_g;
    vectorToMatrix<T>(params.data(), 0, &T_g);
    Eigen::Matrix<T, 3, 3> T_s;
    vectorToMatrix<T>(params.data(), 9, &T_s);
    Eigen::Matrix<T, 3, 3> T_a;
    vectorToMatrix<T>(params.data(), 18, &T_a);
    *a = T_a * a_b + ba;
    *w = T_g * w_b + T_s * a_b + bg;
  }

  template <typename T>
  static void
  correct(const Eigen::Matrix<T, 3, 1> &bg, const Eigen::Matrix<T, 3, 1> &ba,
          const Eigen::Matrix<T, Eigen::Dynamic, 1> &params,
          const Eigen::Matrix<T, 3, 1> &w, const Eigen::Matrix<T, 3, 1> &a,
          Eigen::Matrix<T, 3, 1> *w_b, Eigen::Matrix<T, 3, 1> *a_b) {
    Eigen::Matrix<T, 3, 3> T_g;
    vectorToMatrix<T>(params.data(), 0, &T_g);
    Eigen::Matrix<T, 3, 3> T_s;
    vectorToMatrix<T>(params.data(), 9, &T_s);
    Eigen::Matrix<T, 3, 3> T_a;
    vectorToMatrix<T>(params.data(), 18, &T_a);
    Eigen::Matrix<T, 3, 3> inv_T_g = T_g.inverse();
    Eigen::Matrix<T, 3, 3> inv_T_a = T_a.inverse();
    *a_b = inv_T_a * (a - ba);
    *w_b = inv_T_g * (w - bg - T_s * (*a_b));
  }

  static Eigen::VectorXd computeAugmentedParamsError(
      const Eigen::VectorXd& params) {
      Eigen::VectorXd residual = params;
      Eigen::Matrix<double, 9, 1> eye;
      eye << 1, 0, 0, 0, 1, 0, 0, 0, 1;
      residual.head<9>() -= eye;
      residual.tail<9>() -= eye;
      return residual;
  }
};

/**
 * @brief The ScaledMisalignedImu class
 * The body frame is the same as the classic IMU sensor frame. So the gyroscope triad
 * needs to consider scaling effect(3), misalignment(3), and relative 
 * orientation(4, minimal 3) to the IMU sensor frame, and g-sensitivity (9)
 * whereas the accelerometer triad needs to consider scaling effect (3) and
 * misalignment (3). The lever arm(size) effects are ignored.
 * Implemented according to "Extending Kalibr".
 */
class ScaledMisalignedImu {
 public:
  static const int kModelId = 2;
  static const size_t kSMDim = 6;
  static const size_t kSensitivityDim = 9;
  static const size_t kAugmentedDim = kSMDim + kSensitivityDim + kSMDim + 4;
  static const size_t kGlobalDim = kAugmentedDim + kBgBaDim;
  static inline int getAugmentedDim() { return kAugmentedDim; }
  static inline int getMinimalDim() { return kGlobalDim - 1; }
  static inline int getAugmentedMinimalDim() { return kAugmentedDim - 1; }
  template <typename T>
  static Eigen::Matrix<T, kAugmentedDim, 1> getNominalAugmentedParams() {
    Eigen::Matrix<T, kAugmentedDim, 1> nominalValues = Eigen::Matrix<T, kAugmentedDim, 1>::Zero();
    nominalValues[0] = T(1.0);
    nominalValues[2] = T(1.0);
    nominalValues[5] = T(1.0);
    nominalValues[6 + 9] = T(1.0);
    nominalValues[6 + 9 + 2] = T(1.0);
    nominalValues[6 + 9 + 5] = T(1.0);
    nominalValues[kAugmentedDim - 1] = T(1.0);  // quaternion in xyzw format for R_gyro_i.
    return nominalValues;
  }

  /**
   * nearly 1:1 implementation of
   * https://github.com/ethz-asl/kalibr/blob/master/aslam_offline_calibration/kalibr/python/kalibr_imu_camera_calibration/IccSensors.py#L1033-L1049
   * w_b angular velocity in body frame at time tk.
   * w_dot_b angular acceleration in body frame at time tk.
   * a_w linear acceleration at tk.
   * r_b acceleration triad origin, i.e., the sensor frame origin, coordinates expressed in the body frame.
   * params gyro bias, accelerometer bias, gyro Scaling*Misalignment, gyro g-sensitivity, accelerometer Scaling*Misalignment.
   * C_gyro_i the relative orientation from the accelerometer triad frame, i.e., the IMU sensor frame to the gyro triad frame.
   */
  template <typename T>
  static void predictAngularVelocity(
      const Eigen::Matrix<T, 3, 1> &bg, const Eigen::Matrix<T, 3, 1> & /*ba*/,
      const Eigen::Matrix<T, Eigen::Dynamic, 1> &params,
      const Eigen::Quaternion<T> &q_w_b, const Eigen::Matrix<T, 3, 1> &w_b,
      const Eigen::Matrix<T, 3, 1> &a_w, const Eigen::Matrix<T, 3, 1> &g_w,
      Eigen::Matrix<T, 3, 1> *w) {
    Eigen::Matrix<T, 3, 1> w_dot_b = Eigen::Matrix<T, 3, 1>::Zero();
    Eigen::Matrix<T, 3, 3> C_b_w =
        q_w_b.template toRotationMatrix().transpose();
    Eigen::Matrix<T, 3, 1> r_b =
        Eigen::Matrix<T, 3, 1>::Zero(); // Assume the 3 accelerometers are at
                                        // the origin of the body frame.
    Eigen::Matrix<T, 3, 1> a_b =
        C_b_w * (a_w - g_w) + w_dot_b.cross(r_b) + w_b.cross(w_b.cross(r_b));

    Eigen::Matrix<T, 3, 3> C_i_b =
        Eigen::Matrix<T, 3, 3>::Identity(); // The IMU sensor frame coincides
                                            // the accelerometer triad frame.
    Eigen::Map<const Eigen::Quaternion<T>> q_gyro_i(params.data() +
                                                    kAugmentedDim - 4);
    Eigen::Matrix<T, 3, 3> C_gyro_i = q_gyro_i.template toRotationMatrix();
    Eigen::Matrix<T, 3, 3> C_gyro_b = C_gyro_i * C_i_b;

    Eigen::Matrix<T, 3, 3> M_gyro;
    vectorToLowerTriangularMatrix<T>(params.data(), 0, &M_gyro);
    Eigen::Matrix<T, 3, 3> M_accel_gyro;
    vectorToMatrix<T>(params.data(), kSMDim, &M_accel_gyro);
    *w = M_gyro * (C_gyro_b * w_b) + M_accel_gyro * (C_gyro_b * a_b) + bg;
  }

  /**
   * nearly 1:1 implementation of
   * https://github.com/ethz-asl/kalibr/blob/master/aslam_offline_calibration/kalibr/python/kalibr_imu_camera_calibration/IccSensors.py#L989-L1000
   */
  template <typename T>
  static void predictLinearAcceleration(
      const Eigen::Matrix<T, 3, 1> &/*bg*/, const Eigen::Matrix<T, 3, 1> &ba,
      const Eigen::Matrix<T, Eigen::Dynamic, 1> &params,
      const Eigen::Quaternion<T> &q_w_b, const Eigen::Matrix<T, 3, 1> &w_b,
      const Eigen::Matrix<T, 3, 1> &a_w, const Eigen::Matrix<T, 3, 1> &g_w,
      Eigen::Matrix<T, 3, 1> *a) {
    Eigen::Matrix<T, 3, 1> w_dot_b = Eigen::Matrix<T, 3, 1>::Zero();
    Eigen::Matrix<T, 3, 3> C_b_w =
        q_w_b.template toRotationMatrix().transpose();

    Eigen::Matrix<T, 3, 3> M_accel;
    vectorToLowerTriangularMatrix<T>(params.data(), kSMDim + kSensitivityDim,
                                     &M_accel);

    Eigen::Matrix<T, 3, 1> r_b =
        Eigen::Matrix<T, 3, 1>::Zero(); // Assume the 3 accelerometers are at
                                        // the origin of the body frame.
    Eigen::Matrix<T, 3, 1> a_b =
        C_b_w * (a_w - g_w) + w_dot_b.cross(r_b) + w_b.cross(w_b.cross(r_b));

    Eigen::Matrix<T, 3, 3> C_i_b =
        Eigen::Matrix<T, 3, 3>::Identity(); // The IMU sensor frame coincides
                                            // the accelerometer triad frame.
    *a = M_accel * (C_i_b * a_b) + ba;
  }

  template <typename T>
  static void
  predict(const Eigen::Matrix<T, 3, 1> &bg, const Eigen::Matrix<T, 3, 1> &ba,
          const Eigen::Matrix<T, Eigen::Dynamic, 1> &params,
          const Eigen::Matrix<T, 3, 1> &w_b, const Eigen::Matrix<T, 3, 1> &a_b,
          Eigen::Matrix<T, 3, 1> *w, Eigen::Matrix<T, 3, 1> *a) {
    Eigen::Matrix<T, 3, 3> M_accel;
    vectorToLowerTriangularMatrix<T>(params.data(), kSMDim + kSensitivityDim,
                                     &M_accel);
    Eigen::Matrix<T, 3, 3> C_i_b =
        Eigen::Matrix<T, 3, 3>::Identity(); // The IMU sensor frame coincides
                                            // the accelerometer triad frame.
    *a = M_accel * (C_i_b * a_b) + ba;

    Eigen::Map<const Eigen::Quaternion<T>> q_gyro_i(params.data() +
                                                    kAugmentedDim - 4);
    Eigen::Matrix<T, 3, 3> C_gyro_i = q_gyro_i.toRotationMatrix();
    Eigen::Matrix<T, 3, 3> C_gyro_b = C_gyro_i * C_i_b;

    Eigen::Matrix<T, 3, 3> M_gyro;
    vectorToLowerTriangularMatrix<T>(params.data(), 0, &M_gyro);
    Eigen::Matrix<T, 3, 3> M_accel_gyro;
    vectorToMatrix<T>(params.data(), kSMDim, &M_accel_gyro);
    *w = M_gyro * (C_gyro_b * w_b) + M_accel_gyro * (C_gyro_b * a_b) + bg;
  }

  template <typename T>
  static void
  correct(const Eigen::Matrix<T, 3, 1> &bg, const Eigen::Matrix<T, 3, 1> &ba,
          const Eigen::Matrix<T, Eigen::Dynamic, 1> &params,
          const Eigen::Matrix<T, 3, 1> &w, const Eigen::Matrix<T, 3, 1> &a,
          Eigen::Matrix<T, 3, 1> *w_b, Eigen::Matrix<T, 3, 1> *a_b) {
    Eigen::Matrix<T, 3, 3> M_accel_inv;
    invertLowerTriangularMatrix<T>(params.data(), kSMDim + kSensitivityDim,
                                   &M_accel_inv);
    Eigen::Matrix<T, 3, 3> C_b_i =
        Eigen::Matrix<T, 3, 3>::Identity(); // The IMU sensor frame coincides
                                            // the accelerometer triad frame.
    *a_b = C_b_i * M_accel_inv * (a - ba);

    Eigen::Matrix<T, 3, 3> C_i_b = C_b_i.transpose();
    Eigen::Map<const Eigen::Quaternion<T>> q_gyro_i(params.data() +
                                                    kAugmentedDim - 4);
    Eigen::Matrix<T, 3, 3> C_gyro_i = q_gyro_i.toRotationMatrix();
    Eigen::Matrix<T, 3, 3> C_gyro_b = C_gyro_i * C_i_b;

    Eigen::Matrix<T, 3, 3> M_gyro_inv;
    invertLowerTriangularMatrix<T>(params.data(), 0, &M_gyro_inv);
    Eigen::Matrix<T, 3, 3> M_accel_gyro;
    vectorToMatrix<T>(params.data(), kSMDim, &M_accel_gyro);
    *w_b = C_gyro_b.transpose() *
           (M_gyro_inv * (w - bg - M_accel_gyro * (C_gyro_b * (*a_b))));
  }

  static Eigen::VectorXd
  computeAugmentedParamsError(const Eigen::VectorXd &params) {
    Eigen::VectorXd residual(getAugmentedMinimalDim());
    Eigen::VectorXd nominalValues = getNominalAugmentedParams<double>();
    constexpr int kAugmentedEuclideanDim = kAugmentedDim - 4;
    Eigen::Map<const Eigen::Quaterniond> q_g_i(nominalValues.data() +
                                               kAugmentedEuclideanDim);
    residual.head<kAugmentedEuclideanDim>() =
        params.head<kAugmentedEuclideanDim>() -
        nominalValues.head<kAugmentedEuclideanDim>();
    Eigen::Map<const Eigen::Quaterniond> q_g_i_hat(params.data() + kAugmentedEuclideanDim);
    residual.tail<3>() = (q_g_i * q_g_i_hat.conjugate()).coeffs().head<3>() * 2;
    return residual;
  }
};

#ifndef IMU_ERROR_MODEL_CASES
#define IMU_ERROR_MODEL_CASES                                                  \
  IMU_ERROR_MODEL_CASE(Imu_BG_BA)                                              \
  IMU_ERROR_MODEL_CASE(Imu_BG_BA_TG_TS_TA)                                     \
  IMU_ERROR_MODEL_CASE(ScaledMisalignedImu)
#endif

inline int ImuModelGetMinimalDim(int model_id) {
  switch (model_id) {
#define MODEL_CASES IMU_ERROR_MODEL_CASES
#define IMU_ERROR_MODEL_CASE(ImuModel) \
  case ImuModel::kModelId:             \
    return ImuModel::getMinimalDim();

    MODEL_SWITCH_CASES

#undef IMU_ERROR_MODEL_CASE
#undef MODEL_CASES
  }
  return 0;
}

inline int ImuModelGetAugmentedDim(int model_id) {
  switch (model_id) {
#define MODEL_CASES IMU_ERROR_MODEL_CASES
#define IMU_ERROR_MODEL_CASE(ImuModel) \
  case ImuModel::kModelId:             \
    return ImuModel::getAugmentedDim();

    MODEL_SWITCH_CASES

#undef IMU_ERROR_MODEL_CASE
#undef MODEL_CASES
  }
  return 0;
}

inline int ImuModelGetAugmentedMinimalDim(int model_id) {
  switch (model_id) {
#define MODEL_CASES IMU_ERROR_MODEL_CASES
#define IMU_ERROR_MODEL_CASE(ImuModel) \
  case ImuModel::kModelId:             \
    return ImuModel::getAugmentedMinimalDim();

    MODEL_SWITCH_CASES

#undef IMU_ERROR_MODEL_CASE
#undef MODEL_CASES
  }
  return 0;
}

inline void ImuModelToAugmentedDesiredStdevs(const int imu_model,
                                             Eigen::VectorXd *stdevs) {
  int index = 0;
  switch (imu_model) {
  case Imu_BG_BA_TG_TS_TA::kModelId:
    stdevs->resize(27);
    for (int i = 0; i < 9; ++i) {
      (*stdevs)[i] = 4e-3;
    }
    index = 9;
    for (int i = 0; i < 9; ++i) {
      (*stdevs)[i + index] = 1e-3;
    }
    index += 9;
    for (int i = 0; i < 9; ++i) {
      (*stdevs)[i + index] = 5e-3;
    }
    break;
  case ScaledMisalignedImu::kModelId:
    stdevs->resize(24);
    for (int i = 0; i < 6; ++i) {
      (*stdevs)[i] = 4e-3;
    }
    index += 6;
    for (int i = 0; i < 9; ++i) {
      (*stdevs)[i + index] = 1e-3;
    }
    index += 9;
    for (int i = 0; i < 6; ++i) {
      (*stdevs)[i + index] = 5e-3;
    }
    index += 6;
    for (int i = 0; i < 3; ++i) {
      (*stdevs)[i + index] = 5e-3;
    }
    index += 3;
    break;
  case Imu_BG_BA::kModelId:
  default:
    stdevs->resize(0);
    break;
  }
}

inline void
ImuModelToMinimalAugmentedDimensionLabels(const int imu_model,
                                          std::vector<std::string> *labels) {
  std::vector<std::string> extraLabels;
  switch (imu_model) {
  case Imu_BG_BA_TG_TS_TA::kModelId:
    extraLabels = {"Tg_1", "Tg_2", "Tg_3", "Tg_4", "Tg_5", "Tg_6", "Tg_7",
                   "Tg_8", "Tg_9", "Ts_1", "Ts_2", "Ts_3", "Ts_4", "Ts_5",
                   "Ts_6", "Ts_7", "Ts_8", "Ts_9", "Ta_1", "Ta_2", "Ta_3",
                   "Ta_4", "Ta_5", "Ta_6", "Ta_7", "Ta_8", "Ta_9"};
    break;
  case ScaledMisalignedImu::kModelId:
    extraLabels = {"Mg_11", "Mg_21",       "Mg_22",       "Mg_31",      "Mg_32",
                   "Mg_33", "A_11",        "A_12",        "A_13",       "A_21",
                   "A_22",  "A_23",        "A_31",        "A_32",       "A_33",
                   "Ma_11", "Ma_21",       "Ma_22",       "Ma_31",      "Ma_32",
                   "Ma_33", "theta_g_a_x", "theta_g_a_y", "theta_g_a_z"};
    break;
  case Imu_BG_BA::kModelId:
  default:
    break;
  }
  *labels = extraLabels;
}

inline void
ImuModelToAugmentedDimensionLabels(const int imu_model,
                                   std::vector<std::string> *labels) {
  std::vector<std::string> extraLabels;
  switch (imu_model) {
  case Imu_BG_BA_TG_TS_TA::kModelId:
    extraLabels = {"Tg_1", "Tg_2", "Tg_3", "Tg_4", "Tg_5", "Tg_6", "Tg_7",
                   "Tg_8", "Tg_9", "Ts_1", "Ts_2", "Ts_3", "Ts_4", "Ts_5",
                   "Ts_6", "Ts_7", "Ts_8", "Ts_9", "Ta_1", "Ta_2", "Ta_3",
                   "Ta_4", "Ta_5", "Ta_6", "Ta_7", "Ta_8", "Ta_9"};
    break;
  case ScaledMisalignedImu::kModelId:
    extraLabels = {"Mg_11", "Mg_21",   "Mg_22",   "Mg_31",   "Mg_32",
                   "Mg_33", "A_11",    "A_12",    "A_13",    "A_21",
                   "A_22",  "A_23",    "A_31",    "A_32",    "A_33",
                   "Ma_11", "Ma_21",   "Ma_22",   "Ma_31",   "Ma_32",
                   "Ma_33", "q_g_a_x", "q_g_a_y", "q_g_a_z", "q_g_a_w"};
    break;
  case Imu_BG_BA::kModelId:
  default:
    break;
  }
  *labels = extraLabels;
}

inline void ImuModelToDimensionLabels(const int imu_model,
                                      std::vector<std::string> *labels) {
  *labels = {"b_g_x[rad/s]", "b_g_y", "b_g_z",
             "b_a_x[m/s^2]", "b_a_y", "b_a_z"};
  std::vector<std::string> extraLabels;
  ImuModelToAugmentedDimensionLabels(imu_model, &extraLabels);
  labels->insert(labels->end(), extraLabels.begin(), extraLabels.end());
}

inline int ImuModelNameToId(std::string imu_error_model_descrip) {
  std::transform(imu_error_model_descrip.begin(), imu_error_model_descrip.end(),
                 imu_error_model_descrip.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  if (imu_error_model_descrip.compare("SCALEDMISALIGNED") == 0) {
    return ScaledMisalignedImu::kModelId;
  } else if (imu_error_model_descrip.compare("BG_BA_TG_TS_TA") == 0) {
    return Imu_BG_BA_TG_TS_TA::kModelId;
  } else if (imu_error_model_descrip.compare("BG_BA") == 0) {
    return Imu_BG_BA::kModelId;
  } else {
    return Imu_BG_BA_TG_TS_TA::kModelId;
  }
}

inline Eigen::Matrix<double, Eigen::Dynamic, 1> ImuModelNominalAugmentedParams(int model_id) {
  switch (model_id) {
#define MODEL_CASES IMU_ERROR_MODEL_CASES
#define IMU_ERROR_MODEL_CASE(ImuModel) \
    case ImuModel::kModelId:             \
  return ImuModel::getNominalAugmentedParams<double>();

    MODEL_SWITCH_CASES

    #undef IMU_ERROR_MODEL_CASE
    #undef MODEL_CASES
    }
}

inline void ImuModelPredict(int model_id, const Eigen::Vector3d& bg, const Eigen::Vector3d& ba,
                            const Eigen::Matrix<double, Eigen::Dynamic, 1>& params,
                            const Eigen::Matrix<double, 3, 1>& w_b, const Eigen::Matrix<double, 3, 1>& a_b,
                            Eigen::Matrix<double, 3, 1>* w, Eigen::Matrix<double, 3, 1>* a) {
  switch (model_id) {
#define MODEL_CASES IMU_ERROR_MODEL_CASES
#define IMU_ERROR_MODEL_CASE(ImuModel) \
    case ImuModel::kModelId:             \
  return ImuModel::predict<double>(bg, ba, params, w_b, a_b, w, a);

    MODEL_SWITCH_CASES

    #undef IMU_ERROR_MODEL_CASE
    #undef MODEL_CASES
    }
}

inline void ImuModelCorrect(int model_id,
                            const Eigen::Vector3d& bg, const Eigen::Vector3d& ba,
                            const Eigen::Matrix<double, Eigen::Dynamic, 1>& params,
                            const Eigen::Matrix<double, 3, 1>& w, const Eigen::Matrix<double, 3, 1>& a,
                            Eigen::Matrix<double, 3, 1>* w_b, Eigen::Matrix<double, 3, 1>* a_b) {
  switch (model_id) {
#define MODEL_CASES IMU_ERROR_MODEL_CASES
#define IMU_ERROR_MODEL_CASE(ImuModel) \
    case ImuModel::kModelId:             \
  return ImuModel::correct<double>(bg, ba, params, w, a, w_b, a_b);

    MODEL_SWITCH_CASES

    #undef IMU_ERROR_MODEL_CASE
    #undef MODEL_CASES
    }
}

inline Eigen::VectorXd ImuModelComputeAugmentedParamsError(
    int model_id, const Eigen::VectorXd& parameters) {
  switch (model_id) {
#define MODEL_CASES IMU_ERROR_MODEL_CASES
#define IMU_ERROR_MODEL_CASE(ImuModel) \
    case ImuModel::kModelId:             \
  return ImuModel::computeAugmentedParamsError(parameters);

    MODEL_SWITCH_CASES

    #undef IMU_ERROR_MODEL_CASE
    #undef MODEL_CASES
  }
}
}  // namespace swift_vio
#endif  // INCLUDE_SWIFT_VIO_IMU_ERROR_MODELS_HPP_
