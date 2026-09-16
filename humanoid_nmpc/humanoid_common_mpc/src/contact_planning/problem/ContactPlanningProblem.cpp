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

#include "humanoid_common_mpc/contact_planning/problem/ContactPlanningProblem.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

#include "humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h"
#include "humanoid_common_mpc/contact_planning/logic/ContactLogicScan.h"
#include "humanoid_common_mpc/contact_planning/problem/InputBoundsBuilder.h"
#include "humanoid_common_mpc/contact_planning/problem/LayoutBuilder.h"
#include "humanoid_common_mpc/contact_planning/problem/RowBuilder.h"
#include "humanoid_common_mpc/contact_planning/problem/StageAccumulator.h"

namespace ocs2::humanoid {

namespace {

template <typename T>
void checkRequiredBlocks(const TermCollection<T>& collection, const TermCollection<LipModelBlock>& model, const char* what) {
  for (size_t i = 0; i < collection.size(); ++i) {
    for (const std::string& block : collection.at(i).requiredBlocks()) {
      if (!model.has(block)) {
        throw std::invalid_argument(std::string("[ContactPlanningProblem] ") + what + " '" + collection.nameAt(i) +
                                    "' needs the model block '" + block + "', which is not part of the formulation");
      }
    }
  }
}

template <typename T>
void bindAndConfigure(TermCollection<T>& collection, const Layout& layout, const ContactPlanningConfig& config) {
  for (auto& term : collection) {
    term->bind(layout);
    term->configure(config);
  }
}

template <typename T>
void configureAll(TermCollection<T>& collection, const ContactPlanningConfig& config) {
  for (auto& term : collection) term->configure(config);
}

template <typename T>
void describeAll(std::ostream& out, const char* title, const TermCollection<T>& collection) {
  out << title << " (" << collection.size() << "):\n";
  for (size_t i = 0; i < collection.size(); ++i) {
    out << "  - " << collection.nameAt(i) << ": " << collection.at(i).describe() << "\n";
  }
}

}  // namespace

void ContactPlanningProblem::finalize(const ContactPlanningConfig& config) {
  if (model.size() < 2 || !model.has(term::kLipCom) || !model.has(term::kFootholdIntegrator) || model.nameAt(0) != term::kLipCom ||
      model.nameAt(1) != term::kFootholdIntegrator) {
    throw std::invalid_argument(std::string("[ContactPlanningProblem] the model must start with the '") + term::kLipCom + "' and '" +
                                term::kFootholdIntegrator + "' blocks");
  }
  LayoutBuilder builder;
  for (const auto& block : model) block->declareVariables(builder);
  layout_ = builder.build();

  checkRequiredBlocks(costs, model, "cost");
  checkRequiredBlocks(softConstraints, model, "soft constraint");
  checkRequiredBlocks(hardConstraints, model, "hard constraint");
  checkRequiredBlocks(logicRules, model, "logic rule");
  checkRequiredBlocks(assignmentCosts, model, "assignment cost");

  bindAndConfigure(model, layout_, config);
  bindAndConfigure(costs, layout_, config);
  bindAndConfigure(softConstraints, layout_, config);
  bindAndConfigure(hardConstraints, layout_, config);
  bindAndConfigure(logicRules, layout_, config);
  bindAndConfigure(assignmentCosts, layout_, config);
  finalized_ = true;
}

void ContactPlanningProblem::configure(const ContactPlanningConfig& config) {
  if (!finalized_) throw std::logic_error("[ContactPlanningProblem] configure() before finalize()");
  configureAll(model, config);
  configureAll(costs, config);
  configureAll(softConstraints, config);
  configureAll(hardConstraints, config);
  configureAll(logicRules, config);
  configureAll(assignmentCosts, config);
}

OcpQpProblem ContactPlanningProblem::assemble(const ContactPlanningContext& ctx) const {
  if (!finalized_) throw std::logic_error("[ContactPlanningProblem] assemble() before finalize()");
  const int N = ctx.numNodes;
  const int nx = layout_.nx;
  const int nu = layout_.nu;

  OcpQpProblem problem;
  problem.x0 = vector_t::Zero(nx);
  for (const auto& block : model) block->setInitialState(ctx, problem.x0);

  problem.stages.resize(static_cast<size_t>(N) + 1);
  for (int k = 0; k <= N; ++k) {
    const bool terminal = (k == N);
    OcpQpStage& s = problem.stages[static_cast<size_t>(k)];
    s = OcpQpStage::Zero(nx, terminal ? 0 : nu, !terminal);

    // Costs, in collection order (the accumulation order of the stage matrices).
    StageAccumulator accumulator(s);
    for (const auto& cost : costs) {
      if (nodeSetContains(cost->nodeSet(), k, N)) cost->addToStage(ctx, k, accumulator);
    }

    // Dynamics and input bounds of the running nodes, block by block.
    if (!terminal) {
      for (const auto& block : model) block->addDynamics(ctx, k, s);
      InputBoundsBuilder bounds;
      for (const auto& block : model) block->addInputBounds(ctx, k, bounds);
      bounds.writeTo(s);
    }

    // General rows: the hard constraints, then the soft ones with their penalties.
    RowBuilder rows(nx, terminal ? 0 : nu);
    for (const auto& constraint : hardConstraints) {
      if (nodeSetContains(constraint->nodeSet(), k, N)) constraint->addRows(ctx, k, rows);
    }
    for (const auto& constraint : softConstraints) {
      if (nodeSetContains(constraint->nodeSet(), k, N)) constraint->addRows(ctx, k, rows);
    }
    rows.writeTo(s);
  }
  return problem;
}

std::vector<MiqpBinaryVariable> ContactPlanningProblem::binaryVariables(int numNodes) const {
  std::vector<MiqpBinaryVariable> binaries;
  for (int k = 0; k < numNodes; ++k) {
    for (const auto& block : model) {
      for (const int index : block->binaryInputs()) binaries.push_back({k, index});
    }
  }
  return binaries;
}

bool ContactPlanningProblem::propagate(const ContactLogicState& state, MiqpAssignment& a) const {
  // The rules are run to a fixpoint. Every pass recomputes the shared scan of the prefix first, so that a rule sees what
  // the others fixed in the previous pass; within a pass a rule reads the live assignment.
  for (int pass = 0; pass < 4 * state.numNodes; ++pass) {
    bool changed = false;
    const ContactLogicScan scan = ContactLogicScan::compute(state, a);
    for (const auto& rule : logicRules) {
      if (!rule->propagate(state, scan, a, changed)) return false;
    }
    if (!changed) break;
  }
  return true;
}

scalar_t ContactPlanningProblem::assignmentCost(const ContactLogicState& state, const MiqpAssignment& a) const {
  scalar_t total = 0.0;
  for (const auto& cost : assignmentCosts) total += cost->cost(state, a);
  return total;
}

void ContactPlanningProblem::decode(const ContactPlanningContext& ctx, const MiqpResult& result, ContactPlan& plan) const {
  for (const auto& block : model) block->decode(ctx, result, plan);
}

std::pair<int, int> ContactPlanningProblem::countRows(const ContactPlanningContext& ctx) const {
  const OcpQpProblem problem = assemble(ctx);
  return {problem.stages.front().numGeneralConstraints(), problem.stages.back().numGeneralConstraints()};
}

std::string ContactPlanningProblem::summary(const ContactPlanningContext* ctx) const {
  std::ostringstream out;
  out << "layout: " << layout_.describe() << "\n";
  describeAll(out, "dynamics", model);
  describeAll(out, "costs", costs);
  describeAll(out, "soft constraints", softConstraints);
  describeAll(out, "hard constraints", hardConstraints);
  describeAll(out, "logic rules", logicRules);
  describeAll(out, "assignment costs", assignmentCosts);
  if (ctx != nullptr && finalized_) {
    const auto [running, terminal] = countRows(*ctx);
    out << "general rows: " << running << " per running node, " << terminal << " at the terminal node\n";
  }
  return out.str();
}

}  // namespace ocs2::humanoid
