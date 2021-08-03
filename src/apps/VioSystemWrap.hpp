#ifndef APPS_VIO_SYSTEM_WRAP_HPP
#define APPS_VIO_SYSTEM_WRAP_HPP

#include <io_wrap/PgoPublisher.hpp>
#include <io_wrap/Publisher.hpp>

#include <okvis/Parameters.hpp>
#include <okvis/ThreadedKFVio.hpp>

namespace swift_vio {
class VioSystemWrap {
 public:
  /**
   * @brief registerCallbacks
   * @param output_dir
   * @param parameters
   * @param vioSystem
   * @param publisher
   * @param pgoPublisher
   * @return
   */
   static void registerCallbacks(const std::string &output_dir,
                                 const okvis::VioParameters &parameters,
                                 okvis::ThreadedKFVio *vioSystem,
                                 StreamPublisher *publisher,
                                 PgoPublisher *pgoPublisher);
};
}  // namespace swift_vio

#endif  // APPS_VIO_SYSTEM_WRAP_HPP
