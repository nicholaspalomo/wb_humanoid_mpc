/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include <vector>

#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/MixedIntegerOcpQp.h"
#include "humanoid_common_mpc/contact_planning/problem/ContactPlanningContext.h"
#include "humanoid_common_mpc/contact_planning/problem/ContactPlanningTerm.h"
#include "humanoid_common_mpc/contact_planning/problem/InputBoundsBuilder.h"
#include "humanoid_common_mpc/contact_planning/problem/LayoutBuilder.h"

namespace ocs2::humanoid {

/**
 * A block of the reduced model: the variables it adds to the layout, their dynamics x_{k+1} = A x_k + B u_k + b, the
 * per-node input bounds, the initial state, and how its part of the solution is decoded into the plan.
 */
class LipModelBlock : public ContactPlanningTerm {
 public:
  /** Declares the block's states and inputs, in order (the layout is the concatenation over the blocks). */
  virtual void declareVariables(LayoutBuilder& layout) const = 0;
  /** Writes the block's rows of A, B (and b) of a running node. The stage arrives zero-initialised. */
  virtual void addDynamics(const ContactPlanningContext& ctx, int node, OcpQpStage& stage) const = 0;
  /** Adds the block's input box constraints of a running node, in order. */
  virtual void addInputBounds(const ContactPlanningContext& /*ctx*/, int /*node*/, InputBoundsBuilder& /*bounds*/) const {}
  /** Writes the block's entries of x0 from the input. */
  virtual void setInitialState(const ContactPlanningContext& ctx, vector_t& x0) const = 0;
  /** Fills the block's fields of the plan from the incumbent (trajectories sized N + 1 / N by the caller). */
  virtual void decode(const ContactPlanningContext& ctx, const MiqpResult& result, ContactPlan& plan) const = 0;
  /** Layout indices of the block's inputs that are binaries {0, 1}, in branching order. Empty for most blocks. */
  virtual std::vector<int> binaryInputs() const { return {}; }
};

}  // namespace ocs2::humanoid
