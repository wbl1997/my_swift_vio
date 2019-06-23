
#include "okvis/PoseViewer.hpp"
#include <Eigen/Core>
#include "opencv2/imgproc.hpp"

PoseViewer::PoseViewer() {
  cv::namedWindow("OKVIS Top View");
  _image.create(imageSize, imageSize, CV_8UC3);
  drawing_ = false;
  showing_ = false;
}
PoseViewer::~PoseViewer() { cv::destroyWindow("OKVIS Top View"); }
// this we can register as a callback
void PoseViewer::publishFullStateAsCallback(
    const okvis::Time & /*t*/, const okvis::kinematics::Transformation &T_WS,
    const Eigen::Matrix<double, 9, 1> &speedAndBiases,
    const Eigen::Matrix<double, 3, 1> & /*omega_S*/) {
  // just append the path
  Eigen::Vector3d r = T_WS.r();
  Eigen::Matrix3d C = T_WS.C();
  _path.push_back(cv::Point2d(r[0], r[1]));
  _heights.push_back(r[2]);
  // maintain scaling
  if (r[0] - _frameScale < _min_x) _min_x = r[0] - _frameScale;
  if (r[1] - _frameScale < _min_y) _min_y = r[1] - _frameScale;
  if (r[2] < _min_z) _min_z = r[2];
  if (r[0] + _frameScale > _max_x) _max_x = r[0] + _frameScale;
  if (r[1] + _frameScale > _max_y) _max_y = r[1] + _frameScale;
  if (r[2] > _max_z) _max_z = r[2];
  _scale =
      std::min(imageSize / (_max_x - _min_x), imageSize / (_max_y - _min_y));

  // draw it
  while (showing_) {
  }
  drawing_ = true;
  // erase
  _image.setTo(cv::Scalar(10, 10, 10));
  drawPath();
  // draw axes
  Eigen::Vector3d e_x = C.col(0);
  Eigen::Vector3d e_y = C.col(1);
  Eigen::Vector3d e_z = C.col(2);
  cv::line(_image, convertToImageCoordinates(_path.back()),
           convertToImageCoordinates(_path.back() +
                                     cv::Point2d(e_x[0], e_x[1]) * _frameScale),
           cv::Scalar(0, 0, 255), 1, CV_AA);
  cv::line(_image, convertToImageCoordinates(_path.back()),
           convertToImageCoordinates(_path.back() +
                                     cv::Point2d(e_y[0], e_y[1]) * _frameScale),
           cv::Scalar(0, 255, 0), 1, CV_AA);
  cv::line(_image, convertToImageCoordinates(_path.back()),
           convertToImageCoordinates(_path.back() +
                                     cv::Point2d(e_z[0], e_z[1]) * _frameScale),
           cv::Scalar(255, 0, 0), 1, CV_AA);

  // some text:
  std::stringstream postext;
  postext << "position = [" << r[0] << ", " << r[1] << ", " << r[2] << "]";
  cv::putText(_image, postext.str(), cv::Point(15, 15),
              cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
  std::stringstream veltext;
  veltext << "velocity = [" << speedAndBiases[0] << ", " << speedAndBiases[1]
          << ", " << speedAndBiases[2] << "]";
  cv::putText(_image, veltext.str(), cv::Point(15, 35),
              cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

  drawing_ = false;  // notify
}

void PoseViewer::publishLandmarksCallback(
    const okvis::Time &, const okvis::MapPointVector &landmarks,
    const okvis::MapPointVector &) {
  // LOG(INFO)<<"landmark publishing"<<landmarks.size();
}
void PoseViewer::display() {
  while (drawing_) {
  }
  showing_ = true;
  cv::imshow("OKVIS Top View", _image);
  showing_ = false;
  cv::waitKey(1);
}

cv::Point2d PoseViewer::convertToImageCoordinates(
    const cv::Point2d &pointInMeters) const {
  cv::Point2d pt = (pointInMeters - cv::Point2d(_min_x, _min_y)) * _scale;
  return cv::Point2d(
      pt.x, imageSize - pt.y);  // reverse y for more intuitive top-down plot
}
void PoseViewer::drawPath() {
  for (size_t i = 0; i + 1 < _path.size();) {
    cv::Point2d p0 = convertToImageCoordinates(_path[i]);
    cv::Point2d p1 = convertToImageCoordinates(_path[i + 1]);
    cv::Point2d diff = p1 - p0;
    if (diff.dot(diff) < 2.0) {
      _path.erase(_path.begin() + i + 1);  // clean short segment
      _heights.erase(_heights.begin() + i + 1);
      continue;
    }
    double rel_height = (_heights[i] - _min_z + _heights[i + 1] - _min_z) *
                        0.5 / (_max_z - _min_z);
    rel_height = 1.0;  // huai hack
    cv::line(_image, p0, p1,
             rel_height * cv::Scalar(255, 0, 0) +
                 (1.0 - rel_height) * cv::Scalar(0, 0, 255),
             1, CV_AA);
    i++;
  }
}
