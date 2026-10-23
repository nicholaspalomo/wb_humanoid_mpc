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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <vector>

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_nmpc/humanoid_wb_mpc/test/live_tuning/WholeBodySolverStack.h"

namespace ocs2::humanoid::live_tuning_test {

/**
 * Returns what the first worker's running problem of `stack` evaluates to, after the reference manager has planned the
 * next solve's references (WholeBodySolverStack::replanReferences()), as one vector: every cost, soft constraint and
 * equality constraint term's value and gradient (or linearization) at the points of evaluationPoints(), in the ramps of
 * the swings of the stack's schedule and at a point with every joint inside the band of its limit barrier, the pre-computation's swing-foot
 * coefficients there, the penalty of every soft constraint over a sweep of constraint values from deep inside to far outside its relaxation
 * (a barrier is identically zero away from its boundary), and the SQP settings a reload writes, which the solver iterates by. A field that
 * changes none of it is not live, whatever it writes.
 */
std::vector<scalar_t> runningProblemEffects(WholeBodySolverStack& stack);

/**
 * Returns runningProblemEffects() and what a reload wrote, read back: the parameters of the CppAD foot and joint torque
 * costs and the swing trajectory planner's configuration. Two problems that agree on all of it agree on what a reload
 * writes.
 */
std::vector<scalar_t> runningProblemFingerprint(WholeBodySolverStack& stack);

}  // namespace ocs2::humanoid::live_tuning_test
