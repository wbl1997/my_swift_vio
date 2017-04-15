#include <gtest/gtest.h>
#include "glog/logging.h"
#include "BinaryOperators.cpp"
#include "BoostAccumulators.cpp"
#include "EigenMatrixInitialization.cpp"
#include "MapKeyOrder.cpp"
#include "Sizeof.cpp"
#include "StdAccumulate.cpp"
#include "testIMUOdometry.cpp"
#include "testTriangulate.cpp"
#include "testEigenQR.cpp"
//#include "testHybridFilter.cpp"

/// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
