#ifndef STATACCUMULATOR_H
#define STATACCUMULATOR_H

#include <swift_vio/memory.h>
#include <okvis/Measurements.hpp>

namespace okvis {
class StatAccumulator {
public:
  StatAccumulator() : numSucceededRuns(0) {}

  void refreshBuffer(int expectedNumEntries);

  void push_back(okvis::Time time, const Eigen::VectorXd &value) {
    stat.emplace_back(time, value);
  }

  void accumulate();

  Eigen::VectorXd lastValue() const { return stat.back().measurement; }

  void computeMean();

  void computeRootMean();

  void dump(const std::string statFile, const std::string &headerLine) const;

  int succeededRuns() const { return numSucceededRuns; }

private:
  // stat for one run, cumulativeStat for multiple runs
  Eigen::AlignedVector<okvis::Measurement<Eigen::VectorXd>> stat, cumulativeStat;
  int numSucceededRuns;
};
}  // namespace okvis


#endif // STATACCUMULATOR_H
