/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

#include "humanoid_common_mpc/parameter_update/HotUpdateTarget.h"

#include <cstddef>
#include <optional>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/config/model/ModelSettingsFromConfig.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/parameter_update/OcpTermUpdates.h"

namespace ocs2::humanoid {

HotUpdateTarget::HotUpdateTarget(const mpc_config::TaskFile* absl_nonnull task,
                                 SqpSolver* absl_nonnull solver,
                                 SwitchedModelReferenceManager* absl_nullable referenceManager,
                                 const StateInputLayout* absl_nonnull layout,
                                 size_t inputDim,
                                 absl::string_view source)
    : task_(task),
      solver_(solver),
      referenceManager_(referenceManager),
      layout_(layout),
      stateDim_(stateDimension(*layout)),
      inputDim_(inputDim),
      source_(source) {}

const std::optional<ModelSettings::FootConstraintConfig>& HotUpdateTarget::footConstraint() {
  if (!footConstraintConverted_) {
    footConstraintConverted_ = true;
    footConstraint_ =
        convertedOrReported(footConstraintFromConfig(task_->model_settings.foot_constraint), source_, "model_settings.foot_constraint");
  }
  return footConstraint_;
}

const std::optional<scalar_t>& HotUpdateTarget::terminalCostScaling() {
  if (!terminalCostScalingConverted_) {
    terminalCostScalingConverted_ = true;
    terminalCostScaling_ =
        convertedOrReported(terminalCostScalingFromConfig(*task_), source_, "the terminal weights (terminal_cost_scaling)");
  }
  return terminalCostScaling_;
}

void HotUpdateTarget::reportNotApplied(absl::string_view what, const absl::Status& status) const {
  ::ocs2::humanoid::reportNotApplied(source_, what, status);
}

}  // namespace ocs2::humanoid
