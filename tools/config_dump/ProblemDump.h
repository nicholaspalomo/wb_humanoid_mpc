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

#include <cstddef>

#include "absl/strings/string_view.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "tools/config_dump/ValueDump.h"

namespace ocs2::humanoid::config_dump {

/** Where the walking target of dumpProblem() moves the base: the index of the base x position in the MPC's state. */
struct ProblemLayout {
  size_t inputDim = 0;
  Eigen::Index basePositionIndex = 0;
};

/**
 * The optimal control problem `problem` as the solver sees it, with its weights pinned without an accessor of their
 * own: the names of the terms of every collection, and on a walking schedule with a forward-moving target (set on
 * `referenceManager`, the problem's), at three points of the horizon with a fixed perturbed state and input, every cost
 * and soft-constraint term's value and quadratic approximation (the diagonals of its Hessians element by element, the
 * rest as fingerprints) and the linear-quadratic approximation of the whole problem (cost, dynamics, equality
 * constraints), then the same of the final cost. The terms are evaluated on a copy of `problem`.
 */
void dumpProblem(ValueDump& dump,
                 absl::string_view path,
                 const OptimalControlProblem& problem,
                 SwitchedModelReferenceManager& referenceManager,
                 const vector_t& initialState,
                 const ProblemLayout& layout);

}  // namespace ocs2::humanoid::config_dump
