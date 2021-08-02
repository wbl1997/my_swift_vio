#include "swift_vio/InterpolateImuDataBSplines.h"
#include <gtest/gtest.h>

TEST(SE3Interpolation, CoarseInitialization) {
  // constant linear acceleration case
  const double aing = 1.5;
  const double tEnd = 5;
  const double tStart = 0;
  const double interval = 0.1;

  std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> q02n;
  std::vector<double> times;
  double outputFreq = 100;
  std::vector<Eigen::Matrix<double, 4, 4>,
              Eigen::aligned_allocator<Eigen::Matrix<double, 4, 4>>>
      samplePoses;
  std::vector<Eigen::Matrix<double, 10, 1>,
              Eigen::aligned_allocator<Eigen::Matrix<double, 10, 1>>>
      samples;
  Eigen::Matrix<double, 3, 1> gw = Eigen::Matrix<double, 3, 1>::Zero();
  for (double time = tStart; time <= tEnd; time += interval) {
    times.push_back(time);
    q02n.push_back(
        Sophus::SE3d(Eigen::Quaterniond(1, 0, 0, 0),
                     Eigen::Vector3d(0.5 * aing * time * time, 0, 0)));
  }
  swift_vio::InterpolateIMUData(q02n, times, outputFreq, samplePoses, samples, gw);

  // check the results
  size_t removeEnds = (size_t)(interval * outputFreq);
  for (size_t jack = removeEnds; jack < samples.size() - removeEnds; ++jack) {
    EXPECT_LT(std::fabs(samples[jack][1] - aing), 1e-8);
    EXPECT_LT(samples[jack].segment<5>(2).lpNorm<Eigen::Infinity>(), 1e-8);
  }

  // constant angular rate case
  const double wing = 2;
  const double radius = 5;
  q02n.clear();
  times.clear();

  gw << 0, 0, -9.8;
  for (double time = tStart; time <= tEnd; time += interval) {
    times.push_back(time);
    double theta = wing * time;
    Eigen::Matrix3d Rb2w;
    Rb2w << cos(theta), -sin(theta), 0, sin(theta), cos(theta), 0, 0, 0, 1;
    Eigen::Vector3d tbinw;
    tbinw << radius * cos(theta), radius * sin(theta), 0;
    q02n.push_back(Sophus::SE3d(Rb2w, tbinw));
  }
  swift_vio::InterpolateIMUData(q02n, times, outputFreq, samplePoses, samples, gw);

  // check the results
  for (size_t jack = removeEnds; jack < samples.size() - removeEnds; ++jack) {
    double time = samples[jack][0];
    double theta = wing * time;
//    std::cout << samples[jack].transpose() << std::endl;
    EXPECT_LT(std::fabs(samples[jack][1] + radius * wing * wing), 1e-8);
    EXPECT_LT(std::fabs(samples[jack][2]), 1e-8);
    EXPECT_LT(std::fabs(samples[jack][3] + gw[2]), 1e-8);

    EXPECT_LT((samples[jack].segment<3>(4) - Eigen::Vector3d(0, 0, wing))
               .lpNorm<Eigen::Infinity>(), 1e-8);

    EXPECT_LT(std::fabs(samples[jack][7] + radius * sin(theta) * wing), 1e-8);
    EXPECT_LT(std::fabs(samples[jack][8] - radius * cos(theta) * wing), 1e-8);
    EXPECT_LT(std::fabs(samples[jack][9]), 1e-8);
  }
}
