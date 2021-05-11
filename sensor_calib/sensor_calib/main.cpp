#include <cstdlib>
#include <iostream>

#include "TimeAndRotationCalibrator.h"

int main(int argc, char **argv) {
  swift_vio::TimeAndRotationCalibrator example{};
  std::cout << "TimeAndRotationCalibrator.calibrate() => " << example.calibrate()
            << std::endl;

  return EXIT_SUCCESS;
}
