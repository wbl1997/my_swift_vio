#include <iomanip>
#include "io_wrap/PgoPublisher.hpp"

namespace swift_vio {
PgoPublisher::PgoPublisher() {}

PgoPublisher::~PgoPublisher() { csvStream_.close(); }

void PgoPublisher::csvSaveStateAsCallback(
    const okvis::Time& t, const okvis::kinematics::Transformation& T_WB) {
  const char delimiter = ' ';
  const Eigen::Matrix<double, 7, 1>& T_WB_coeffs = T_WB.coeffs();
  csvStream_ << t << std::setprecision(8);
  for (int j = 0; j < T_WB_coeffs.size(); ++j) {
    csvStream_ << delimiter << T_WB_coeffs[j];
  }
  csvStream_ << "\n";
}

bool PgoPublisher::setCsvFile(const std::string& csvFile) {
  csvStream_ = std::ofstream(csvFile.c_str(), std::ios_base::out);
  csvStream_ << "# timestamp tx ty tz qx qy qz qw\n";
  return csvStream_.good();
}
}  // namespace swift_vio
