#include <loop_closure/LoopClosureParameters.hpp>
#include <loop_closure/LoopClosureMethod.hpp>

namespace swift_vio {
LoopClosureParameters::LoopClosureParameters() :
  PipelineParams("Base Loop Closure Parameters") {}

LoopClosureParameters::LoopClosureParameters(const std::string name) :
  PipelineParams(name) {}

LoopClosureParameters::~LoopClosureParameters() {}

bool LoopClosureParameters::parseYAML(const std::string& /*filepath*/) {
  return true;
}

void LoopClosureParameters::print() const {

}

bool LoopClosureParameters::equals(const PipelineParams& /*obj*/) const {
  return true;
}
}  // namespace swift_vio
