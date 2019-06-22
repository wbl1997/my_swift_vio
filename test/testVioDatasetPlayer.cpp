
#include <gtest/gtest.h>
#include <iostream>
#include <thread>

#include <okvis/Parameters.hpp>
#include "MockVioInterface.hpp"

#include "okvis/Player.hpp"

TEST(VioCommon, VioDatasetPlayer) {
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
