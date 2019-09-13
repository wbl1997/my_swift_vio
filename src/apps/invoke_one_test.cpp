#include "../test/msckf/TestHybridFilter.cpp"
#include "../test/opencv/TestOpenCVDistortion.cpp"
#include "../test/std/TestDequeIterator.cpp"

DECLARE_bool(use_AIDP);

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <output_dir> <#runs>\n";
    std::cout << "Example: " << argv[0] << " ~/Desktop/temp 10\n";
    return 0;
  }
  FLAGS_use_AIDP = true;
  FLAGS_estimator_algorithm = 1;
  testHybridFilterSinusoid(argv[1], std::atoi(argv[2]));

  return 0;
}
