#ifndef INCLUDE_IOWRAP_PGO_PUBLISHER_HPP
#define INCLUDE_IOWRAP_PGO_PUBLISHER_HPP

#include <fstream>
#include <okvis/Time.hpp>
#include <okvis/kinematics/Transformation.hpp>

namespace okvis {
class PgoPublisher {
 public:
  PgoPublisher();

  ~PgoPublisher();

  /**
   * @brief csvSaveStateAsCallback
   * @param t timestamp
   * @param T_WS tx ty tz qx qy qz qw.
   */
  void csvSaveStateAsCallback(const okvis::Time& t,
                              const okvis::kinematics::Transformation& T_WB);

  bool setCsvFile(const std::string& csvFile);

 private:
  std::ofstream csvStream_;
};
}  // namespace okvis

#endif  // INCLUDE_IOWRAP_PGO_PUBLISHER_HPP
