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

#pragma once

#include "absl/base/nullability.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_mpc/CommandData.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"

namespace ocs2::humanoid::visualization {

/**
 * What the visualization keeps of one MPC solution: the planned trajectories, the mode schedule and the command it was
 * solved for. The controller is left out on purpose: the visualization samples the plan by interpolation, and a linear
 * controller's gains (hundreds of kilobytes) would only make the copy on the solver thread slower.
 */
struct PolicySnapshot {
  scalar_array_t time;
  vector_array_t state;
  vector_array_t input;
  ModeSchedule modeSchedule;
  /** CommandData::mpcTargetTrajectories_: the reference the plan was solved for. */
  TargetTrajectories target;
  /** CommandData::mpcInitObservation_: the observation the plan was solved from. */
  SystemObservation initObservation;

  /**
   * Copies `command` and `solution` into this snapshot. Assignment reuses the storage the snapshot already holds, so a
   * snapshot that held a solution of the same shape takes the next one without allocating.
   */
  void assign(const CommandData& command, const PrimalSolution& solution);
};

/** True when the plan has nodes, as many states and inputs as times, and states of `stateDim` and inputs of `inputDim`. */
bool isConsistentPlan(const PolicySnapshot& plan, size_t stateDim, size_t inputDim);

/**
 * The plan's state and input at `time`, linearly interpolated between its nodes and held at the first or the last node
 * outside the horizon. At an event, where two nodes share a time, the node before the event is taken. Requires a plan
 * for which isConsistentPlan() holds.
 */
void samplePlan(const PolicySnapshot& plan, scalar_t time, vector_t* absl_nonnull state, vector_t* absl_nonnull input);

/** What a TargetTrajectories gives at a time: nothing, a state, or a state and an input. */
enum class TargetSample {
  kNone,
  kState,
  kStateAndInput,
};

/**
 * The reference state (and input, when the target carries inputs of `inputDim`) at `time`, interpolated as
 * samplePlan() interpolates. kNone, with the outputs untouched, when the target has no states of `stateDim`.
 */
TargetSample sampleTarget(const TargetTrajectories& target,
                          scalar_t time,
                          size_t stateDim,
                          size_t inputDim,
                          vector_t* absl_nonnull state,
                          vector_t* absl_nonnull input);

}  // namespace ocs2::humanoid::visualization
