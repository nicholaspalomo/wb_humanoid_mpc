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

#include "humanoid_common_mpc/contact_planning/ContactPlanningTermFactory.h"

#include <stdexcept>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h"
#include "humanoid_common_mpc/contact_planning/constraint/FootMotionInSwingOnlyConstraint.h"
#include "humanoid_common_mpc/contact_planning/constraint/FootSeparationConstraint.h"
#include "humanoid_common_mpc/contact_planning/constraint/FootYawPinnedInContactConstraint.h"
#include "humanoid_common_mpc/contact_planning/constraint/HipYawRangeConstraint.h"
#include "humanoid_common_mpc/contact_planning/constraint/NoFlightConstraint.h"
#include "humanoid_common_mpc/contact_planning/constraint/ReachabilityConstraint.h"
#include "humanoid_common_mpc/contact_planning/constraint/YawTorqueBudgetConstraint.h"
#include "humanoid_common_mpc/contact_planning/constraint/ZmpSupportRegionConstraint.h"
#include "humanoid_common_mpc/contact_planning/cost/FootYawRegularizationCost.h"
#include "humanoid_common_mpc/contact_planning/cost/FootYawTrackingCost.h"
#include "humanoid_common_mpc/contact_planning/cost/FootholdRegularizationCost.h"
#include "humanoid_common_mpc/contact_planning/cost/HeadingRateTrackingCost.h"
#include "humanoid_common_mpc/contact_planning/cost/HeadingTrackingCost.h"
#include "humanoid_common_mpc/contact_planning/cost/PreviousFootholdConsistencyCost.h"
#include "humanoid_common_mpc/contact_planning/cost/RegularizationCost.h"
#include "humanoid_common_mpc/contact_planning/cost/StepLengthCost.h"
#include "humanoid_common_mpc/contact_planning/cost/StepWidthCost.h"
#include "humanoid_common_mpc/contact_planning/cost/TerminalDcmCost.h"
#include "humanoid_common_mpc/contact_planning/cost/VelocityTrackingCost.h"
#include "humanoid_common_mpc/contact_planning/cost/YawTorqueRegularizationCost.h"
#include "humanoid_common_mpc/contact_planning/cost/ZmpRegularizationCost.h"
#include "humanoid_common_mpc/contact_planning/execution/DcmStepAdjustmentRule.h"
#include "humanoid_common_mpc/contact_planning/execution/EnergyCadenceModulationRule.h"
#include "humanoid_common_mpc/contact_planning/execution/PhaseResettingRule.h"
#include "humanoid_common_mpc/contact_planning/logic/AlternatingFeetRule.h"
#include "humanoid_common_mpc/contact_planning/logic/ContactSwitchCost.h"
#include "humanoid_common_mpc/contact_planning/logic/DoubleSupportPenaltyCost.h"
#include "humanoid_common_mpc/contact_planning/logic/MinimumDoubleSupportRule.h"
#include "humanoid_common_mpc/contact_planning/logic/NoFlightRule.h"
#include "humanoid_common_mpc/contact_planning/logic/PhaseDurationsRule.h"
#include "humanoid_common_mpc/contact_planning/logic/PlanConsistencyCost.h"
#include "humanoid_common_mpc/contact_planning/model/FootholdIntegrator.h"
#include "humanoid_common_mpc/contact_planning/model/HeadingDoubleIntegrator.h"
#include "humanoid_common_mpc/contact_planning/model/LipComDynamics.h"
#include "humanoid_common_mpc/contact_planning/search/CadenceStretchStage.h"
#include "humanoid_common_mpc/contact_planning/search/DivingStage.h"
#include "humanoid_common_mpc/contact_planning/search/EventShiftLocalSearchStage.h"
#include "humanoid_common_mpc/contact_planning/search/HeadingRelinearizationStage.h"
#include "humanoid_common_mpc/contact_planning/search/WarmStartPreviousPlanStage.h"

namespace ocs2::humanoid {

namespace {

absl::Status unknownTerm(TermKind kind, const std::string& name) {
  return absl::InvalidArgumentError(absl::StrCat("[ContactPlanningTermFactory] unknown ", termKindName(kind), " term '", name,
                                                 "'; supported: ", absl::StrJoin(knownTermNames(kind), ", ")));
}

[[noreturn]] void unknown(TermKind kind, const std::string& name) {
  throw std::invalid_argument(std::string(unknownTerm(kind, name).message()));
}

absl::Status unavailableExecutionRule(const std::string& canonical) {
  return absl::InvalidArgumentError(absl::StrCat("[ContactPlanningTermFactory] the execution rule '", canonical,
                                                 "' is not available here (it needs the reference manager's robot model)"));
}

absl::Status missingBlock(absl::string_view what, const std::string& name, const std::string& block) {
  return absl::InvalidArgumentError(absl::StrCat("[ContactPlanningTermFactory] ", what, " '", name, "' needs the '", block, "' block"));
}

std::string canonicalOrThrow(TermKind kind, const std::string& name) {
  const std::string canonical = canonicalTermName(kind, name);
  if (canonical.empty()) unknown(kind, name);
  return canonical;
}

}  // namespace

// LINT.IfChange(term_factory)
std::unique_ptr<LipModelBlock> ContactPlanningTermFactory::makeModelBlock(const std::string& name) {
  const std::string canonical = canonicalOrThrow(TermKind::MODEL_BLOCK, name);
  if (canonical == term::kLipCom) return std::make_unique<LipComDynamics>();
  if (canonical == term::kFootholdIntegrator) return std::make_unique<FootholdIntegrator>();
  if (canonical == term::kHeadingDoubleIntegrator) return std::make_unique<HeadingDoubleIntegrator>();
  unknown(TermKind::MODEL_BLOCK, name);
}

std::unique_ptr<LipCost> ContactPlanningTermFactory::makeCost(const std::string& name) {
  const std::string canonical = canonicalOrThrow(TermKind::COST, name);
  if (canonical == term::kRegularization) return std::make_unique<RegularizationCost>();
  if (canonical == term::kPreviousFootholdConsistency) return std::make_unique<PreviousFootholdConsistencyCost>();
  if (canonical == term::kVelocityTracking) return std::make_unique<VelocityTrackingCost>();
  if (canonical == term::kStepWidth) return std::make_unique<StepWidthCost>();
  if (canonical == term::kHeadingRateTracking) return std::make_unique<HeadingRateTrackingCost>();
  if (canonical == term::kHeadingTracking) return std::make_unique<HeadingTrackingCost>();
  if (canonical == term::kFootYawTracking) return std::make_unique<FootYawTrackingCost>();
  if (canonical == term::kYawTorqueRegularization) return std::make_unique<YawTorqueRegularizationCost>();
  if (canonical == term::kFootYawRegularization) return std::make_unique<FootYawRegularizationCost>();
  if (canonical == term::kZmpRegularization) return std::make_unique<ZmpRegularizationCost>();
  if (canonical == term::kFootholdRegularization) return std::make_unique<FootholdRegularizationCost>();
  if (canonical == term::kStepLength) return std::make_unique<StepLengthCost>();
  if (canonical == term::kTerminalDcm) return std::make_unique<TerminalDcmCost>();
  unknown(TermKind::COST, name);
}

std::unique_ptr<LipConstraint> ContactPlanningTermFactory::makeSoftConstraint(const std::string& name) {
  const std::string canonical = canonicalOrThrow(TermKind::SOFT_CONSTRAINT, name);
  if (canonical == term::kZmpSupportRegion) return std::make_unique<ZmpSupportRegionConstraint>();
  if (canonical == term::kReachability) return std::make_unique<ReachabilityConstraint>();
  if (canonical == term::kFootSeparation) return std::make_unique<FootSeparationConstraint>();
  if (canonical == term::kHipYawRange) return std::make_unique<HipYawRangeConstraint>();
  unknown(TermKind::SOFT_CONSTRAINT, name);
}

std::unique_ptr<LipConstraint> ContactPlanningTermFactory::makeHardConstraint(const std::string& name) {
  const std::string canonical = canonicalOrThrow(TermKind::HARD_CONSTRAINT, name);
  if (canonical == term::kNoFlight) return std::make_unique<NoFlightConstraint>();
  if (canonical == term::kFootMotionInSwingOnly) return std::make_unique<FootMotionInSwingOnlyConstraint>();
  if (canonical == term::kYawTorqueBudget) return std::make_unique<YawTorqueBudgetConstraint>();
  if (canonical == term::kFootYawPinnedInContact) return std::make_unique<FootYawPinnedInContactConstraint>();
  unknown(TermKind::HARD_CONSTRAINT, name);
}

std::unique_ptr<ContactLogicRule> ContactPlanningTermFactory::makeLogicRule(const std::string& name) {
  const std::string canonical = canonicalOrThrow(TermKind::LOGIC_RULE, name);
  if (canonical == term::kPhaseDurations) return std::make_unique<PhaseDurationsRule>();
  if (canonical == term::kNoFlight) return std::make_unique<NoFlightRule>();
  if (canonical == term::kMinimumDoubleSupport) return std::make_unique<MinimumDoubleSupportRule>();
  if (canonical == term::kAlternatingFeet) return std::make_unique<AlternatingFeetRule>();
  unknown(TermKind::LOGIC_RULE, name);
}

std::unique_ptr<AssignmentCost> ContactPlanningTermFactory::makeAssignmentCost(const std::string& name) {
  const std::string canonical = canonicalOrThrow(TermKind::ASSIGNMENT_COST, name);
  if (canonical == term::kContactSwitch) return std::make_unique<ContactSwitchCost>();
  if (canonical == term::kPlanConsistency) return std::make_unique<PlanConsistencyCost>();
  if (canonical == term::kDoubleSupportPenalty) return std::make_unique<DoubleSupportPenaltyCost>();
  unknown(TermKind::ASSIGNMENT_COST, name);
}

std::unique_ptr<SearchStage> ContactPlanningTermFactory::makeSearchStage(const std::string& name) {
  const std::string canonical = canonicalOrThrow(TermKind::SEARCH_STAGE, name);
  if (canonical == term::kWarmStartPreviousPlan) return std::make_unique<WarmStartPreviousPlanStage>();
  if (canonical == term::kDiving) return std::make_unique<DivingStage>();
  if (canonical == term::kEventShiftLocalSearch) return std::make_unique<EventShiftLocalSearchStage>();
  if (canonical == term::kHeadingRelinearization) return std::make_unique<HeadingRelinearizationStage>();
  if (canonical == term::kCadenceStretch) return std::make_unique<CadenceStretchStage>();
  unknown(TermKind::SEARCH_STAGE, name);
}

namespace {
/** The execution rule `canonical` names, from the core or else from `extra`; null when neither can build it. */
std::unique_ptr<ExecutionRule> executionRuleOrNull(const std::string& canonical, const ContactPlanningTermFactory::ExtraRuleMaker& extra) {
  if (canonical == term::kPhaseResetting) return std::make_unique<PhaseResettingRule>();
  if (canonical == term::kEnergyCadenceModulation) return std::make_unique<EnergyCadenceModulationRule>();
  if (canonical == term::kDcmStepAdjustment) return std::make_unique<DcmStepAdjustmentRule>();
  if (extra) return extra(canonical);
  return nullptr;
}
}  // namespace

std::unique_ptr<ExecutionRule> ContactPlanningTermFactory::makeExecutionRule(const std::string& name, const ExtraRuleMaker& extra) {
  const std::string canonical = canonicalOrThrow(TermKind::EXECUTION_RULE, name);
  std::unique_ptr<ExecutionRule> rule = executionRuleOrNull(canonical, extra);
  if (rule == nullptr) throw std::invalid_argument(std::string(unavailableExecutionRule(canonical).message()));
  return rule;
}
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/contact_planning/ContactPlanningFormulation.cpp:known_term_names)

absl::StatusOr<ContactPlanningProblem> ContactPlanningTermFactory::buildProblemStatus(const ContactPlanningConfig& config) {
  // The formulation's own validation rejects every list this function could not assemble - an unknown or duplicate
  // name, a term without the block it needs, the mandatory blocks out of place - so nothing below throws once it passed.
  RETURN_IF_ERROR(config.formulation.validateStatus());
  const ContactPlanningFormulation& f = config.formulation;
  ContactPlanningProblem problem;
  for (const std::string& name : f.dynamics) problem.model.add(canonicalTermName(TermKind::MODEL_BLOCK, name), makeModelBlock(name));
  for (const std::string& name : f.costs) problem.costs.add(canonicalTermName(TermKind::COST, name), makeCost(name));
  for (const std::string& name : f.softConstraints) {
    problem.softConstraints.add(canonicalTermName(TermKind::SOFT_CONSTRAINT, name), makeSoftConstraint(name));
  }
  for (const std::string& name : f.hardConstraints) {
    problem.hardConstraints.add(canonicalTermName(TermKind::HARD_CONSTRAINT, name), makeHardConstraint(name));
  }
  for (const std::string& name : f.logicRules) problem.logicRules.add(canonicalTermName(TermKind::LOGIC_RULE, name), makeLogicRule(name));
  for (const std::string& name : f.assignmentCosts) {
    problem.assignmentCosts.add(canonicalTermName(TermKind::ASSIGNMENT_COST, name), makeAssignmentCost(name));
  }
  problem.finalize(config);
  return problem;
}

absl::StatusOr<TermCollection<SearchStage>> ContactPlanningTermFactory::buildSearchStagesStatus(const ContactPlanningConfig& config) {
  TermCollection<SearchStage> stages;
  for (const std::string& name : config.formulation.search) {
    const std::string canonical = canonicalTermName(TermKind::SEARCH_STAGE, name);
    if (canonical.empty()) return unknownTerm(TermKind::SEARCH_STAGE, name);
    if (stages.has(canonical)) {
      return absl::InvalidArgumentError(absl::StrCat("[ContactPlanningTermFactory] search lists '", canonical, "' twice"));
    }
    std::unique_ptr<SearchStage> stage = makeSearchStage(canonical);
    for (const std::string& block : stage->requiredBlocks()) {
      if (!config.formulation.hasDynamics(block)) return missingBlock("search stage", name, block);
    }
    stage->configure(config);
    stages.add(canonical, std::move(stage));
  }
  return stages;
}

absl::StatusOr<TermCollection<ExecutionRule>> ContactPlanningTermFactory::buildExecutionRulesStatus(const ContactPlanningConfig& config,
                                                                                                    const ExtraRuleMaker& extra) {
  RETURN_IF_ERROR(config.formulation.validateStatus());
  TermCollection<ExecutionRule> rules;
  for (const std::string& name : config.formulation.execution) {
    const std::string canonical = canonicalTermName(TermKind::EXECUTION_RULE, name);
    std::unique_ptr<ExecutionRule> rule = executionRuleOrNull(canonical, extra);
    if (rule == nullptr) return unavailableExecutionRule(canonical);
    for (const std::string& block : rule->requiredBlocks()) {
      if (!config.formulation.hasDynamics(block)) return missingBlock("execution rule", name, block);
    }
    rule->configure(config);
    rules.add(canonical, std::move(rule));
  }
  return rules;
}

}  // namespace ocs2::humanoid
