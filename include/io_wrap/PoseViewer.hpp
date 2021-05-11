#ifndef OKVIS_POSE_VIEWER_HPP_
#define OKVIS_POSE_VIEWER_HPP_
#include <atomic>

#include <Eigen/Core>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/Time.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

namespace swift_vio {
class PoseViewer {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  constexpr static const double imageSize = 500.0;
  PoseViewer();
  ~PoseViewer();
  // this we can register as a callback
  void publishFullStateAsCallback(
      const okvis::Time & /*t*/, const okvis::kinematics::Transformation &T_WS,
      const Eigen::Matrix<double, 9, 1> &speedAndBiases,
      const Eigen::Matrix<double, 3, 1> & /*omega_S*/);

  void publishLandmarksCallback(const okvis::Time &,
                                const okvis::MapPointVector &landmarks,
                                const okvis::MapPointVector &);
  void display();

 private:
  cv::Point2d convertToImageCoordinates(const cv::Point2d &pointInMeters) const;
  void drawPath();

  cv::Mat _image;
  std::vector<cv::Point2d> _path;
  std::vector<double> _heights;
  double _scale = 1.0;
  double _min_x = -0.5;
  double _min_y = -0.5;
  double _min_z = -0.5;
  double _max_x = 0.5;
  double _max_y = 0.5;
  double _max_z = 0.5;
  const double _frameScale = 0.2;  // [m]
  std::atomic_bool drawing_;
  std::atomic_bool showing_;
};
}  // namespace swift_vio

#endif
