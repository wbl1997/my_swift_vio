/**
 * @file LoopClosureParameters.hpp
 * @brief Header file for LoopClosureParameters class which encompasses
 * parameters for loop detection and pose graph optimization.
 */

#ifndef INCLUDE_OKVIS_LOOP_CLOSURE_PARAMETERS_HPP_
#define INCLUDE_OKVIS_LOOP_CLOSURE_PARAMETERS_HPP_

#include <Eigen/Core>
#include <okvis/PipelineParams.h>
#include <okvis/class_macros.hpp>
#include <glog/logging.h>

namespace swift_vio {
class LoopClosureParameters : public PipelineParams {
 public:
  LoopClosureParameters();
  LoopClosureParameters(const std::string name);
  ~LoopClosureParameters();
  bool parseYAML(const std::string& filepath) override;
  void print() const override;
  bool equals(const PipelineParams& obj) const override;
};
}  // namespace swift_vio

#endif  // INCLUDE_OKVIS_LOOP_CLOSURE_PARAMETERS_HPP_
