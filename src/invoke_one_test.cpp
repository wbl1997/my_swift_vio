#include "../test/OpenCVDistortion.cpp"
#include "../test/testDequeIterator.cpp"
#include "../test/testHybridFilter.cpp" // only for testing

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <output_dir> <#runs>\n";
    std::cout << "Example: " << argv[0] << " ~/Desktop/temp 10\n";
    return 0;
  }
  testHybridFilterSinusoid(argv[1], std::atoi(argv[2]));
  return 0;
}
