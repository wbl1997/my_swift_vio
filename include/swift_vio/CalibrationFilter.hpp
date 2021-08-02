#ifndef INCLUDE_SWIFT_VIO_CALIBRATION_FILTER_HPP_
#define INCLUDE_SWIFT_VIO_CALIBRATION_FILTER_HPP_

#include <swift_vio/HybridFilter.hpp>

namespace swift_vio {

/**
 * @brief The CalibrationFilter class estimates the camer-IMU system parameters using observed known landmarks.

 Frames:
 W: World
 C: Camera
 S: Sensor (IMU)
 B: Body, defined by the IMU model, e.g., Imu_BG_BA, usually defined close to S.
 Its relation to the camera frame is modeled by the extrinsic model, e.g., Extrinsic_p_CB.
 */
class CalibrationFilter : public HybridFilter {
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CalibrationFilter();

  CalibrationFilter(std::shared_ptr<okvis::ceres::Map> mapPtr);

  virtual ~CalibrationFilter();

  /**
   * @brief add a state to the state map
   * @param multiFrame Matched multiFrame.
   * @param imuMeasurements IMU measurements from last state to new one.
   * imuMeasurements covers at least the current state and the last state in
   * time, with an extension on both sides.
   * @param asKeyframe Is this new frame a keyframe?
   * @return True if successful.
   * If it is the first state, initialize it and the covariance matrix. Only one IMU is
   * supported for now.
   */
  bool addStates(okvis::MultiFramePtr multiFrame,
                 const okvis::ImuMeasurementDeque &imuMeasurements,
                 bool asKeyframe) final;

  /**
   * @brief optimize
   * @param numIter
   * @param numThreads
   * @param verbose
   */
  void optimize(size_t numIter, size_t numThreads = 1,
                bool verbose = false) final;

  /**
   * @brief Remove obsolete state and frames.
   * @return True if successful.
   */
  bool applyMarginalizationStrategy(okvis::MapPointVector &removedLandmarks) final;

};

}  // namespace swift_vio
#endif  // INCLUDE_SWIFT_VIO_CALIBRATION_FILTER_HPP_
