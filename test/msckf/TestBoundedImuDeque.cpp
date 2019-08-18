#include <gtest/gtest.h>

#include <msckf/BoundedImuDeque.hpp>

okvis::ImuMeasurementDeque createMeasDeque(int start, int end) {
  okvis::ImuMeasurementDeque imu_meas_deque;
  for (int i = start; i < end; ++i) {
    Eigen::Vector3d g(i, i + 1, i + 2);
    Eigen::Vector3d a(i - 1, i, i + 1);
    imu_meas_deque.push_back(
        okvis::ImuMeasurement(okvis::Time(i), okvis::ImuSensorReadings(g, a)));
  }
  return imu_meas_deque;
}

void print(const okvis::ImuMeasurementDeque& a, std::ostream& out = std::cout) {
    for (okvis::ImuMeasurementDeque::const_iterator iter = a.begin(); iter != a.end(); ++iter ) {
        out << iter->timeStamp << " " << iter->measurement.gyroscopes.transpose()
            << " " << iter->measurement.accelerometers.transpose() << std::endl;
    }
}

void checkEqual(const okvis::ImuMeasurementDeque& a,
                const okvis::ImuMeasurementDeque& b) {
//  std::cout << "a:\n";
//  print(a);
//  std::cout << "b:\n";
//  print(b);
//  std::cout << "\n";
  auto iterb = b.begin();
  double eps = 1e-8;
  for (auto iter = a.begin(); iter != a.end(); ++iter, ++iterb) {
    EXPECT_EQ(iter->timeStamp, iterb->timeStamp);
    EXPECT_LT((iter->measurement.gyroscopes - iterb->measurement.gyroscopes)
              .lpNorm<Eigen::Infinity>(),
              eps);
    EXPECT_LT(
        (iter->measurement.accelerometers - iterb->measurement.accelerometers)
            .lpNorm<Eigen::Infinity>(),
        eps);
  }
  EXPECT_TRUE(iterb == b.end());
}


TEST(BoundedImuDeque, PushBack) {
  okvis::ImuMeasurementDeque imu_meas_deque = createMeasDeque(0, 10);
  okvis::BoundedImuDeque bid;
  int res1 = bid.push_back(imu_meas_deque);
  int res2 = bid.push_back(imu_meas_deque);
  EXPECT_EQ(res1, 10);
  EXPECT_EQ(res2, 0);
  checkEqual(imu_meas_deque, imu_meas_deque);
  int res3 = bid.push_back(createMeasDeque(5, 12));
  EXPECT_EQ(res3, 2);
  checkEqual(bid.getAllImuMeasurements(), createMeasDeque(0, 12));
}

TEST(BoundedImuDeque, PopFront) {
  okvis::ImuMeasurementDeque imu_meas_deque = createMeasDeque(0, 10);
  okvis::BoundedImuDeque bid;
  bid.push_back(imu_meas_deque);
  int res1 = bid.pop_front(okvis::Time(3.0));
  EXPECT_EQ(res1, 3);
  int res2 = bid.pop_front(okvis::Time(6.5));
  EXPECT_EQ(res2, 4);
  checkEqual(createMeasDeque(7, 10), bid.getAllImuMeasurements());
  int res3 = bid.pop_front(okvis::Time(12));
  EXPECT_EQ(res3, 3);
  checkEqual(createMeasDeque(0, 0), bid.getAllImuMeasurements());
}

TEST(BoundedImuDeque, Find) {
  okvis::ImuMeasurementDeque imu_meas_deque = createMeasDeque(0, 10);
  okvis::BoundedImuDeque bid;
  bid.push_back(imu_meas_deque);
  okvis::ImuMeasurementDeque imu_meas1 =
      bid.find(okvis::Time(2.0), okvis::Time(2.0));
  okvis::ImuMeasurementDeque imu_meas2 =
      bid.find(okvis::Time(3.5), okvis::Time(6.0));
  okvis::ImuMeasurementDeque imu_meas3 =
      bid.find(okvis::Time(0.0), okvis::Time(11.5));

  checkEqual(imu_meas1, createMeasDeque(2, 3));
  checkEqual(imu_meas2, createMeasDeque(3, 7));
  checkEqual(imu_meas3, createMeasDeque(0, 10));
}
