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

#include "humanoid_common_mpc/contact_planning/ContactPlanningTermFactory.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
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

absl::Status unavailableExecutionRule(const std::string& canonical) {
  return absl::InvalidArgumentError(absl::StrCat("[ContactPlanningTermFactory] the execution rule '", canonical,
                                                 "' is not available here (it needs the reference manager's robot model)"));
}

absl::Status missingBlock(absl::string_view what, const std::string& name, const std::string& block) {
  return absl::InvalidArgumentError(absl::StrCat("[ContactPlanningTermFactory] ", what, " '", name, "' needs the '", block, "' block"));
}

/** The canonical spelling of a term name, or the InvalidArgument listing the supported names. */
absl::StatusOr<std::string> canonicalOrError(TermKind kind, const std::string& name) {
  std::string canonical = canonicalTermName(kind, name);
  if (canonical.empty()) return unknownTerm(kind, name);
  return canonical;
}

/** Makes every term `names` lists with `make` and adds it to `collection` under its canonical name, in list order. */
template <typename T>
absl::Status addTerms(const std::vector<std::string>& names,
                      TermKind kind,
                      absl::StatusOr<std::unique_ptr<T>> (*absl_nonnull make)(const std::string&),
                      TermCollection<T>& collection) {
  for (const std::string& name : names) {
    ASSIGN_OR_RETURN(const std::string canonical, canonicalOrError(kind, name));
    ASSIGN_OR_RETURN(std::unique_ptr<T> term, make(name));
    RETURN_IF_ERROR(collection.add(canonical, std::move(term)));
  }
  return absl::OkStatus();
}

}  // namespace

// LINT.IfChange(term_factory)
absl::StatusOr<std::unique_ptr<LipModelBlock>> ContactPlanningTermFactory::makeModelBlock(const std::string& name) {
  ASSIGN_OR_RETURN(const std::string canonical, canonicalOrError(TermKind::kModelBlock, name));
  if (canonical == term::kLipCom) return std::make_unique<LipComDynamics>();
  if (canonical == term::kFootholdIntegrator) return std::make_unique<FootholdIntegrator>();
  if (canonical == term::kHeadingDoubleIntegrator) return std::make_unique<HeadingDoubleIntegrator>();
  return unknownTerm(TermKind::kModelBlock, name);
}

absl::StatusOr<std::unique_ptr<LipCost>> ContactPlanningTermFactory::makeCost(const std::string& name) {
  ASSIGN_OR_RETURN(const std::string canonical, canonicalOrError(TermKind::kCost, name));
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
  return unknownTerm(TermKind::kCost, name);
}

absl::StatusOr<std::unique_ptr<LipConstraint>> ContactPlanningTermFactory::makeSoftConstraint(const std::string& name) {
  ASSIGN_OR_RETURN(const std::string canonical, canonicalOrError(TermKind::kSoftConstraint, name));
  if (canonical == term::kZmpSupportRegion) return std::make_unique<ZmpSupportRegionConstraint>();
  if (canonical == term::kReachability) return std::make_unique<ReachabilityConstraint>();
  if (canonical == term::kFootSeparation) return std::make_unique<FootSeparationConstraint>();
  if (canonical == term::kHipYawRange) return std::make_unique<HipYawRangeConstraint>();
  return unknownTerm(TermKind::kSoftConstraint, name);
}

absl::StatusOr<std::unique_ptr<LipConstraint>> ContactPlanningTermFactory::makeHardConstraint(const std::string& name) {
  ASSIGN_OR_RETURN(const std::string canonical, canonicalOrError(TermKind::kHardConstraint, name));
  if (canonical == term::kNoFlight) return std::make_unique<NoFlightConstraint>();
  if (canonical == term::kFootMotionInSwingOnly) return std::make_unique<FootMotionInSwingOnlyConstraint>();
  if (canonical == term::kYawTorqueBudget) return std::make_unique<YawTorqueBudgetConstraint>();
  if (canonical == term::kFootYawPinnedInContact) return std::make_unique<FootYawPinnedInContactConstraint>();
  return unknownTerm(TermKind::kHardConstraint, name);
}

absl::StatusOr<std::unique_ptr<ContactLogicRule>> ContactPlanningTermFactory::makeLogicRule(const std::string& name) {
  ASSIGN_OR_RETURN(const std::string canonical, canonicalOrError(TermKind::kLogicRule, name));
  if (canonical == term::kPhaseDurations) return std::make_unique<PhaseDurationsRule>();
  if (canonical == term::kNoFlight) return std::make_unique<NoFlightRule>();
  if (canonical == term::kMinimumDoubleSupport) return std::make_unique<MinimumDoubleSupportRule>();
  if (canonical == term::kAlternatingFeet) return std::make_unique<AlternatingFeetRule>();
  return unknownTerm(TermKind::kLogicRule, name);
}

absl::StatusOr<std::unique_ptr<AssignmentCost>> ContactPlanningTermFactory::makeAssignmentCost(const std::string& name) {
  ASSIGN_OR_RETURN(const std::string canonical, canonicalOrError(TermKind::kAssignmentCost, name));
  if (canonical == term::kContactSwitch) return std::make_unique<ContactSwitchCost>();
  if (canonical == term::kPlanConsistency) return std::make_unique<PlanConsistencyCost>();
  if (canonical == term::kDoubleSupportPenalty) return std::make_unique<DoubleSupportPenaltyCost>();
  return unknownTerm(TermKind::kAssignmentCost, name);
}

absl::StatusOr<std::unique_ptr<SearchStage>> ContactPlanningTermFactory::makeSearchStage(const std::string& name) {
  ASSIGN_OR_RETURN(const std::string canonical, canonicalOrError(TermKind::kSearchStage, name));
  if (canonical == term::kWarmStartPreviousPlan) return std::make_unique<WarmStartPreviousPlanStage>();
  if (canonical == term::kDiving) return std::make_unique<DivingStage>();
  if (canonical == term::kEventShiftLocalSearch) return std::make_unique<EventShiftLocalSearchStage>();
  if (canonical == term::kHeadingRelinearization) return std::make_unique<HeadingRelinearizationStage>();
  if (canonical == term::kCadenceStretch) return std::make_unique<CadenceStretchStage>();
  return unknownTerm(TermKind::kSearchStage, name);
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

absl::StatusOr<std::unique_ptr<ExecutionRule>> ContactPlanningTermFactory::makeExecutionRule(const std::string& name,
                                                                                             const ExtraRuleMaker& extra) {
  ASSIGN_OR_RETURN(const std::string canonical, canonicalOrError(TermKind::kExecutionRule, name));
  std::unique_ptr<ExecutionRule> rule = executionRuleOrNull(canonical, extra);
  if (rule == nullptr) return unavailableExecutionRule(canonical);
  return rule;
}
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/contact_planning/ContactPlanningFormulation.cpp:known_term_names)

absl::StatusOr<ContactPlanningProblem> ContactPlanningTermFactory::buildProblemStatus(const ContactPlanningConfig& config) {
  // The formulation's own validation rejects every list this function could not assemble - an unknown or duplicate
  // name, a term without the block it needs, the mandatory blocks out of place - so nothing below fails once it passed;
  // the statuses are still forwarded rather than assumed.
  RETURN_IF_ERROR(config.formulation.validateStatus());
  const ContactPlanningFormulation& f = config.formulation;
  ContactPlanningProblem problem;
  RETURN_IF_ERROR(addTerms(f.dynamics, TermKind::kModelBlock, &makeModelBlock, problem.model));
  RETURN_IF_ERROR(addTerms(f.costs, TermKind::kCost, &makeCost, problem.costs));
  RETURN_IF_ERROR(addTerms(f.softConstraints, TermKind::kSoftConstraint, &makeSoftConstraint, problem.softConstraints));
  RETURN_IF_ERROR(addTerms(f.hardConstraints, TermKind::kHardConstraint, &makeHardConstraint, problem.hardConstraints));
  RETURN_IF_ERROR(addTerms(f.logicRules, TermKind::kLogicRule, &makeLogicRule, problem.logicRules));
  RETURN_IF_ERROR(addTerms(f.assignmentCosts, TermKind::kAssignmentCost, &makeAssignmentCost, problem.assignmentCosts));
  RETURN_IF_ERROR(problem.finalize(config));
  return problem;
}

absl::StatusOr<TermCollection<SearchStage>> ContactPlanningTermFactory::buildSearchStagesStatus(const ContactPlanningConfig& config) {
  TermCollection<SearchStage> stages;
  for (const std::string& name : config.formulation.search) {
    const std::string canonical = canonicalTermName(TermKind::kSearchStage, name);
    if (canonical.empty()) return unknownTerm(TermKind::kSearchStage, name);
    if (stages.has(canonical)) {
      return absl::InvalidArgumentError(absl::StrCat("[ContactPlanningTermFactory] search lists '", canonical, "' twice"));
    }
    ASSIGN_OR_RETURN(std::unique_ptr<SearchStage> stage, makeSearchStage(canonical));
    for (const std::string& block : stage->requiredBlocks()) {
      if (!config.formulation.hasDynamics(block)) return missingBlock("search stage", name, block);
    }
    stage->configure(config);
    RETURN_IF_ERROR(stages.add(canonical, std::move(stage)));
  }
  return stages;
}

absl::StatusOr<TermCollection<ExecutionRule>> ContactPlanningTermFactory::buildExecutionRulesStatus(const ContactPlanningConfig& config,
                                                                                                    const ExtraRuleMaker& extra) {
  RETURN_IF_ERROR(config.formulation.validateStatus());
  TermCollection<ExecutionRule> rules;
  for (const std::string& name : config.formulation.execution) {
    const std::string canonical = canonicalTermName(TermKind::kExecutionRule, name);
    std::unique_ptr<ExecutionRule> rule = executionRuleOrNull(canonical, extra);
    if (rule == nullptr) return unavailableExecutionRule(canonical);
    for (const std::string& block : rule->requiredBlocks()) {
      if (!config.formulation.hasDynamics(block)) return missingBlock("execution rule", name, block);
    }
    rule->configure(config);
    RETURN_IF_ERROR(rules.add(canonical, std::move(rule)));
  }
  return rules;
}

}  // namespace ocs2::humanoid
