#include <cstdlib>
#include <iostream>

#include "TimeAndRotationCalibrator.h"

int main(int argc, char **argv) {
  sensor_calib::TimeAndRotationCalibrator example{};
  std::cout << "TimeAndRotationCalibrator.getValue() => " << example.getValue()
            << std::endl;

  return EXIT_SUCCESS;
}
