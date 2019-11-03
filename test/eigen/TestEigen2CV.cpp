#include <gtest/gtest.h>

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>

TEST(Eigen, Eigen2CV) {
  double eps = 1e-7;
  Eigen::Matrix2d eigenMat;
  eigenMat << 1, 2, 3, 4;
  cv::Mat cvMat(2, 2, CV_64F);
  cv::eigen2cv(eigenMat, cvMat);
  Eigen::Matrix2d convertedMat;
  cv::cv2eigen(cvMat, convertedMat);
  EXPECT_LT((convertedMat - eigenMat).lpNorm<Eigen::Infinity>(), eps);

  Eigen::Vector3d eigenVec(100, 200, 300);
  cv::Mat cvVec(3, 1, CV_64F);
  cv::eigen2cv(eigenVec, cvVec);
  Eigen::Vector3d convertedVec;
  cv::cv2eigen(cvVec, convertedVec);
  EXPECT_LT((convertedVec - eigenVec).lpNorm<Eigen::Infinity>(), eps);
  // cv2eigen(cvVec, eigenMat); // watch out! wrong result without a warning

  Eigen::Matrix3f eigenRf = Eigen::Matrix3f::Random();
  cv::Matx33f cvRf;
  cv::eigen2cv(eigenRf, cvRf);
  Eigen::Matrix3f convertedRf;
  cv::cv2eigen(cvRf, convertedRf);
  EXPECT_LT((convertedRf - eigenRf).lpNorm<Eigen::Infinity>(), eps);
}
