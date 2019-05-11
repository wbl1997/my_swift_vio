#include "../test/testHybridFilter.cpp" //only for testing
#include "../test/testDequeIterator.cpp"
#include "../test/OpenCVDistortion.cpp"

int main(int argc, char **argv)
{
  testHybridFilterSinusoid(std::atoi(argv[1]));
  return 0;
}
