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

#include <cstddef>

#include "absl/base/nullability.h"
#include "ocs2_core/Types.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"

namespace ocs2::humanoid::ipc {

/**
 * The number of leading entries of `timeTrajectory` that a solution cut at `finalTime` keeps: every time up to
 * `finalTime`, and one beyond it when there is one, so that the kept trajectory spans the window. The rule of upstream
 * OCS2 GaussNewtonDDP::getPrimalSolution(); at least one entry of a non-empty trajectory is kept.
 */
size_t solutionWindowLength(const scalar_array_t& timeTrajectory, scalar_t finalTime);

/**
 * Cuts `solution` to the solution time window that ends at `finalTime` (mpc::Settings::solutionTimeWindow_), as upstream
 * OCS2 GaussNewtonDDP::getPrimalSolution() does and as SqpSolver::getPrimalSolution(), which ignores its finalTime, does not:
 * the time, state and input trajectories keep solutionWindowLength() nodes, the post-event indices keep the events
 * among them, and a FeedforwardController or LinearController keeps the nodes of its own time stamps by the same rule.
 * The mode schedule is kept whole. A solution that ends before `finalTime` is left as it is, and so is a controller of
 * another type (policyToProto() refuses those anyway).
 *
 * The MPC node calls it before it sends a policy, so that mpc.solution_time_window bounds the bandwidth whatever the
 * solver (humanoid_nmpc/docs/distributed_runtime/README.md, "Bandwidth").
 */
void trimToSolutionWindow(scalar_t finalTime, PrimalSolution* absl_nonnull solution);

}  // namespace ocs2::humanoid::ipc
