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

#include <functional>
#include <memory>
#include <string>

#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/execution/ExecutionRule.h"
#include "humanoid_common_mpc/contact_planning/problem/ContactPlanningProblem.h"
#include "humanoid_common_mpc/contact_planning/problem/TermCollection.h"
#include "humanoid_common_mpc/contact_planning/search/SearchStage.h"

namespace ocs2::humanoid {

/**
 * Creates the planner's terms by name and assembles the problem and the pipelines from the term lists of the
 * configuration, the analogue of HumanoidCostConstraintFactory. Every `make*` throws std::invalid_argument for an
 * unknown name (the message lists the supported ones).
 */
class ContactPlanningTermFactory {
 public:
  static std::unique_ptr<LipModelBlock> makeModelBlock(const std::string& name);
  static std::unique_ptr<LipCost> makeCost(const std::string& name);
  static std::unique_ptr<LipConstraint> makeSoftConstraint(const std::string& name);
  static std::unique_ptr<LipConstraint> makeHardConstraint(const std::string& name);
  static std::unique_ptr<ContactLogicRule> makeLogicRule(const std::string& name);
  static std::unique_ptr<AssignmentCost> makeAssignmentCost(const std::string& name);
  static std::unique_ptr<SearchStage> makeSearchStage(const std::string& name);
  /**
   * The execution rules the core knows (phase resetting, cadence modulation, DCM step adjustment). The planned heading
   * override needs the robot model and lives with the reference manager, which passes its own maker as `extra`.
   */
  using ExtraRuleMaker = std::function<std::unique_ptr<ExecutionRule>(const std::string& canonicalName)>;
  static std::unique_ptr<ExecutionRule> makeExecutionRule(const std::string& name, const ExtraRuleMaker& extra = nullptr);

  /**
   * The problem of the configuration's formulation: the listed terms in list order, finalized (layout composed, terms
   * bound and configured). Throws std::invalid_argument on a formulation the problem cannot be assembled from.
   */
  static ContactPlanningProblem buildProblem(const ContactPlanningConfig& config);
  /** The listed search stages, configured, in list order. */
  static TermCollection<SearchStage> buildSearchStages(const ContactPlanningConfig& config);
  /** The listed execution rules, configured, in list order. */
  static TermCollection<ExecutionRule> buildExecutionRules(const ContactPlanningConfig& config, const ExtraRuleMaker& extra = nullptr);
};

}  // namespace ocs2::humanoid
