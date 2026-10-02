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

#include "humanoid_common_mpc_app/visualization/PolicySnapshot.h"

#include <ocs2_core/misc/LinearInterpolation.h>

namespace ocs2::humanoid::visualization {

void PolicySnapshot::assign(const CommandData& command, const PrimalSolution& solution) {
  time = solution.timeTrajectory_;
  state = solution.stateTrajectory_;
  input = solution.inputTrajectory_;
  modeSchedule = solution.modeSchedule_;
  target = command.mpcTargetTrajectories_;
  initObservation = command.mpcInitObservation_;
}

bool isConsistentPlan(const PolicySnapshot& plan, size_t stateDim, size_t inputDim) {
  if (plan.time.empty() || plan.state.size() != plan.time.size() || plan.input.size() != plan.time.size()) {
    return false;
  }
  for (size_t node = 0; node < plan.time.size(); ++node) {
    if (static_cast<size_t>(plan.state[node].size()) != stateDim || static_cast<size_t>(plan.input[node].size()) != inputDim) {
      return false;
    }
  }
  return true;
}

void samplePlan(const PolicySnapshot& plan, scalar_t time, vector_t* state, vector_t* input) {
  const LinearInterpolation::index_alpha_t indexAlpha = LinearInterpolation::timeSegment(time, plan.time);
  *state = LinearInterpolation::interpolate(indexAlpha, plan.state);
  *input = LinearInterpolation::interpolate(indexAlpha, plan.input);
}

TargetSample sampleTarget(
    const TargetTrajectories& target, scalar_t time, size_t stateDim, size_t inputDim, vector_t* state, vector_t* input) {
  if (target.timeTrajectory.empty() || target.stateTrajectory.size() != target.timeTrajectory.size()) {
    return TargetSample::kNone;
  }
  for (const vector_t& node : target.stateTrajectory) {
    if (static_cast<size_t>(node.size()) != stateDim) {
      return TargetSample::kNone;
    }
  }
  const LinearInterpolation::index_alpha_t indexAlpha = LinearInterpolation::timeSegment(time, target.timeTrajectory);
  *state = LinearInterpolation::interpolate(indexAlpha, target.stateTrajectory);
  if (target.inputTrajectory.size() != target.timeTrajectory.size()) {
    return TargetSample::kState;
  }
  for (const vector_t& node : target.inputTrajectory) {
    if (static_cast<size_t>(node.size()) != inputDim) {
      return TargetSample::kState;
    }
  }
  *input = LinearInterpolation::interpolate(indexAlpha, target.inputTrajectory);
  return TargetSample::kStateAndInput;
}

}  // namespace ocs2::humanoid::visualization
