
#include <gtest/gtest.h>
#include <iostream>
#include <thread>

#include <okvis/Parameters.hpp>
#include "../test/msckf/MockVioInterface.hpp"
#include "io_wrap/Player.hpp"

TEST(Player, removeTrailingSlash) {
  std::string path1 = "/a/b";
  ASSERT_EQ(path1, okvis::removeTrailingSlash(path1));
  std::string path2 = "/a/b/";
  ASSERT_EQ("/a/b", okvis::removeTrailingSlash(path2));
  std::string path3 = "/a\\b\\";
  ASSERT_EQ("/a\\b", okvis::removeTrailingSlash(path3));
  std::string path4 = "/a/b//";
  ASSERT_EQ("/a/b", okvis::removeTrailingSlash(path4));
}

TEST(Player, VioDatasetPlayer) {
  okvis::MockVioInterface mvi;
  okvis::VioParameters parameters;

  okvis::InputData input;
  okvis::InitialState initialState;
  okvis::Optimization optimization;
  okvis::SensorsInformation sensors_information;

  input.videoFile = "";
  input.timeFile = "";
  input.imageFolder = "";
  input.imuFile = "";

  optimization.useMedianFilter = false;

  sensors_information.cameraRate = 20;
  sensors_information.imageDelay = 10;

  parameters.input = input;
  parameters.initialState = initialState;
  parameters.optimization = optimization;
  parameters.sensors_information = sensors_information;

  //  ros::NodeHandle nh("player_node");
  ros::Time::init();
  okvis::Player player(&mvi, parameters);
  std::thread playerThread(&okvis::Player::Run, std::ref(player));
  playerThread.join();

  std::this_thread::sleep_for(std::chrono::seconds(2));
  std::cout << "finished processing the dataset at " << std::endl;
}

TEST(Player, parseCommaSeparatedTopics) {
  {
    std::string topics = "/cam0/image_raw,/cam1/image_raw";
    std::vector<std::string> expected_list{"/cam0/image_raw",
                                           "/cam1/image_raw"};
    std::vector<std::string> topic_list =
        okvis::parseCommaSeparatedTopics(topics);
    for (size_t i = 0; i < expected_list.size(); ++i) {
      EXPECT_EQ(expected_list[i], topic_list[i]);
    }
  }
  {
    std::string topics = "/cam0/image_raw,";
    std::vector<std::string> expected_list{"/cam0/image_raw"};
    std::vector<std::string> topic_list =
        okvis::parseCommaSeparatedTopics(topics);
    for (size_t i = 0; i < expected_list.size(); ++i) {
      EXPECT_EQ(expected_list[i], topic_list[i]);
    }
  }
}
