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

#include <string>
#include <vector>

#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/MixedIntegerOcpQp.h"
#include "humanoid_common_mpc/contact_planning/OcpQpHpipm.h"
#include "humanoid_common_mpc/contact_planning/logic/ContactLogicState.h"
#include "humanoid_common_mpc/contact_planning/problem/AssignmentCost.h"
#include "humanoid_common_mpc/contact_planning/problem/ContactLogicRule.h"
#include "humanoid_common_mpc/contact_planning/problem/ContactPlanningContext.h"
#include "humanoid_common_mpc/contact_planning/problem/Layout.h"
#include "humanoid_common_mpc/contact_planning/problem/LipConstraint.h"
#include "humanoid_common_mpc/contact_planning/problem/LipCost.h"
#include "humanoid_common_mpc/contact_planning/problem/LipModelBlock.h"
#include "humanoid_common_mpc/contact_planning/problem/TermCollection.h"

namespace ocs2::humanoid {

/**
 * The contact planner's optimal control problem as a set of named terms, the analogue of ocs2::OptimalControlProblem.
 *
 * The collections are filled by ContactPlanningTermFactory::buildProblem() from the term lists of the configuration
 * (or by hand, in the same order the factory would use). finalize() then composes the variable layout from the model
 * blocks, checks that every term finds the blocks it needs, binds the terms to the layout and configures them.
 * assemble() writes the OCP-QP of one plan, node by node and term by term in collection order, and the logic side
 * (propagate, assignmentCost) is what MixedIntegerOcpQp calls on the binaries. configure() is the hot-reload path:
 * every term re-reads its parameter block; the term lists themselves are the factory's business (re-assembly).
 */
class ContactPlanningProblem {
 public:
  TermCollection<LipModelBlock> model;  // order fixes the variable layout
  TermCollection<LipCost> costs;        // order is the accumulation order of the stage matrices
  TermCollection<LipConstraint> softConstraints;
  TermCollection<LipConstraint> hardConstraints;
  TermCollection<ContactLogicRule> logicRules;  // order is the order within a propagation pass
  TermCollection<AssignmentCost> assignmentCosts;

  /**
   * Composes the layout from the model blocks, checks the required blocks of every term, binds and configures every
   * term. Throws std::invalid_argument on a missing block or a bad parameter. Must be called before assemble().
   */
  void finalize(const ContactPlanningConfig& config);
  bool isFinalized() const { return finalized_; }
  const Layout& layout() const { return layout_; }

  /** Re-reads every term's parameter block (hot reload of the values; the lists stay). */
  void configure(const ContactPlanningConfig& config);

  /** Builds the OCP-QP of one plan from the context: N running stages and the terminal node. */
  OcpQpProblem assemble(const ContactPlanningContext& ctx) const;

  /** The binaries of every node, in branching order (time order, then the blocks' order). */
  std::vector<MiqpBinaryVariable> binaryVariables(int numNodes) const;

  /**
   * Forward logical propagation: runs the listed rules in order to a fixpoint (at most 4 N passes, the shared prefix
   * scan recomputed before every pass). Returns false when the assignment is provably infeasible.
   */
  bool propagate(const ContactLogicState& state, MiqpAssignment& assignment) const;

  /** Sum of the listed assignment costs. */
  scalar_t assignmentCost(const ContactLogicState& state, const MiqpAssignment& assignment) const;

  /** Fills the plan's trajectories from the incumbent through the model blocks. */
  void decode(const ContactPlanningContext& ctx, const MiqpResult& result, ContactPlan& plan) const;

  /** Number of general rows a running node and the terminal node carry, from the terms' node sets. */
  std::pair<int, int> countRows(const ContactPlanningContext& ctx) const;

  /** The start-up print: the layout, every collection with its terms and their one-line descriptions. */
  std::string summary(const ContactPlanningContext* ctx = nullptr) const;

 private:
  Layout layout_;
  bool finalized_ = false;
};

}  // namespace ocs2::humanoid
