#include "msckf/StatAccumulator.h"

namespace okvis {

void StatAccumulator::refreshBuffer(int expectedNumEntries) {
  stat.clear();
  stat.reserve(expectedNumEntries);
}

void StatAccumulator::accumulate() {
  if (cumulativeStat.empty()) {
    cumulativeStat = stat;
  } else {
    for (size_t j = 0u; j < cumulativeStat.size(); ++j) {
      cumulativeStat[j].measurement += stat[j].measurement;
    }
  }
  ++numSucceededRuns;
}

void StatAccumulator::computeMean() {
  for (Eigen::AlignedVector<okvis::Measurement<Eigen::VectorXd>>::iterator it =
           cumulativeStat.begin();
       it != cumulativeStat.end(); ++it)
    it->measurement /= numSucceededRuns;
}

void StatAccumulator::computeRootMean() {
  for (Eigen::AlignedVector<okvis::Measurement<Eigen::VectorXd>>::iterator it =
           cumulativeStat.begin();
       it != cumulativeStat.end(); ++it)
    it->measurement = ((it->measurement) / numSucceededRuns).cwiseSqrt();
}

void StatAccumulator::dump(const std::string statFile,
                           const std::string &headerLine) const {
  std::ofstream stream;
  stream.open(statFile, std::ofstream::out);
  stream << headerLine << std::endl;
  for (auto it = cumulativeStat.begin(); it != cumulativeStat.end(); ++it)
    stream << it->timeStamp << " " << it->measurement.transpose() << std::endl;
  stream.close();
}

} // namespace okvis
