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

#include <functional>
#include <memory>
#include <string>

#include "absl/status/statusor.h"

#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/execution/ExecutionRule.h"
#include "humanoid_common_mpc/contact_planning/problem/ContactPlanningProblem.h"
#include "humanoid_common_mpc/contact_planning/problem/TermCollection.h"
#include "humanoid_common_mpc/contact_planning/search/SearchStage.h"

namespace ocs2::humanoid {

/**
 * Creates the planner's terms by name and assembles the problem and the pipelines from the term lists of the
 * configuration, the analog of HumanoidCostConstraintFactory. Every `make*` returns an InvalidArgument for an unknown
 * name (the message lists the supported ones). Stateless; safe to call from any thread.
 */
class ContactPlanningTermFactory {
 public:
  static absl::StatusOr<std::unique_ptr<LipModelBlock>> makeModelBlock(const std::string& name);
  static absl::StatusOr<std::unique_ptr<LipCost>> makeCost(const std::string& name);
  static absl::StatusOr<std::unique_ptr<LipConstraint>> makeSoftConstraint(const std::string& name);
  static absl::StatusOr<std::unique_ptr<LipConstraint>> makeHardConstraint(const std::string& name);
  static absl::StatusOr<std::unique_ptr<ContactLogicRule>> makeLogicRule(const std::string& name);
  static absl::StatusOr<std::unique_ptr<AssignmentCost>> makeAssignmentCost(const std::string& name);
  static absl::StatusOr<std::unique_ptr<SearchStage>> makeSearchStage(const std::string& name);
  /**
   * The execution rules the core knows (phase resetting, cadence modulation, DCM step adjustment). The planned heading
   * override needs the robot model and lives with the reference manager, which passes its own maker as `extra`. A rule
   * that neither the core nor `extra` builds is an InvalidArgument naming it.
   */
  using ExtraRuleMaker = std::function<std::unique_ptr<ExecutionRule>(const std::string& canonicalName)>;
  static absl::StatusOr<std::unique_ptr<ExecutionRule>> makeExecutionRule(const std::string& name, const ExtraRuleMaker& extra = nullptr);

  /**
   * The problem of the configuration's formulation: the listed terms in list order, finalized (layout composed, terms
   * bound and configured). A formulation the problem cannot be assembled from is returned as the InvalidArgument of
   * ContactPlanningFormulation::validateStatus(), which names the list and the term.
   *
   * The builders configure the terms with the configuration's parameter values as they are: the values are checked by
   * ContactPlanningConfig::validateStatus(), which the caller runs first, and a term's configure() only reads them.
   */
  static absl::StatusOr<ContactPlanningProblem> buildProblemStatus(const ContactPlanningConfig& config);
  /** The listed search stages, configured, in list order; an unknown or repeated name, or a missing block, as a Status. */
  static absl::StatusOr<TermCollection<SearchStage>> buildSearchStagesStatus(const ContactPlanningConfig& config);
  /**
   * The listed execution rules, configured, in list order. Besides what the formulation's validation rejects, a rule
   * that only the reference manager can build (it needs the robot model) and `extra` does not build is an InvalidArgument
   * naming the rule.
   */
  static absl::StatusOr<TermCollection<ExecutionRule>> buildExecutionRulesStatus(const ContactPlanningConfig& config,
                                                                                 const ExtraRuleMaker& extra = nullptr);
};

}  // namespace ocs2::humanoid
