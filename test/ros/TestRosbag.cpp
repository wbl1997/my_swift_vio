#include "rosbag/bag.h"
#include "rosbag/chunked_file.h"
#include "rosbag/view.h"
#include <ros/ros.h>
#include "sensor_msgs/Imu.h"

#include "glog/logging.h"
#include "gtest/gtest.h"

#include <Eigen/Core>
#include "okvis/Time.hpp"

#include <io_wrap/CommonGflags.hpp>

// Build this test with address sanitizer, add cxx_flags "-fsanitize=address -fno-omit-frame-pointer"
// "-O1 -g" are optional.
// For verbose output, before running, "export ASAN_OPTIONS=fast_unwind_on_malloc=0"
TEST(Rosbag, ClassLoader) {
  std::string bagname = FLAGS_bagname;
  rosbag::Bag bag(bagname, rosbag::bagmode::Read);
  std::string imu_topic = FLAGS_imu_topic;
  rosbag::View view_imu(bag, rosbag::TopicQuery(imu_topic));
  rosbag::View::iterator view_imu_iterator = view_imu.begin();
  LOG(INFO) << "No. IMU messages: " << view_imu.size();
  size_t numImuMsgs = 0u;
  while (1) {
//    ros::spinOnce();
    // check if at the end
    if (view_imu_iterator == view_imu.end()) {
      LOG(INFO) << "Reach end of imu topic. Found #Imu messages " << numImuMsgs;
      EXPECT_EQ(numImuMsgs, view_imu.size());
      break;
    }
    size_t count = 0u;
    do {
      sensor_msgs::ImuConstPtr msg =
          view_imu_iterator->instantiate<sensor_msgs::Imu>();
      Eigen::Vector3d gyr(msg->angular_velocity.x, msg->angular_velocity.y,
                          msg->angular_velocity.z);
      Eigen::Vector3d acc(msg->linear_acceleration.x,
                          msg->linear_acceleration.y,
                          msg->linear_acceleration.z);
      okvis::Time t_imu(msg->header.stamp.sec, msg->header.stamp.nsec);
      ++count;
      view_imu_iterator++;
    } while (view_imu_iterator != view_imu.end() && count < 100);
    numImuMsgs += count;
  }
}
