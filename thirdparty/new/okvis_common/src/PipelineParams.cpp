#include "okvis/PipelineParams.h"
namespace swift_vio {
PipelineParams::PipelineParams(const std::string& name) : name_(name) {
  CHECK_EQ(kTotalWidth, kNameWidth + kValueWidth)
      << "Make sure these are consistent for pretty printing.";
}
}  // namespace swift_vio
