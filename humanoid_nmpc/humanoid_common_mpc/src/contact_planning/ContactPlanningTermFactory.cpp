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

#include "humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h"
#include "humanoid_common_mpc/contact_planning/constraint/ContactHeightConstraint.h"
#include "humanoid_common_mpc/contact_planning/constraint/FootMotionInSwingOnlyConstraint.h"
#include "humanoid_common_mpc/contact_planning/constraint/FootSeparationConstraint.h"
#include "humanoid_common_mpc/contact_planning/constraint/FootYawPinnedInContactConstraint.h"
#include "humanoid_common_mpc/contact_planning/constraint/HipYawRangeConstraint.h"
#include "humanoid_common_mpc/contact_planning/constraint/NoFlightConstraint.h"
#include "humanoid_common_mpc/contact_planning/constraint/ReachabilityConstraint.h"
#include "humanoid_common_mpc/contact_planning/constraint/VerticalThrustLimitConstraint.h"
#include "humanoid_common_mpc/contact_planning/constraint/YawTorqueBudgetConstraint.h"
#include "humanoid_common_mpc/contact_planning/constraint/ZmpPinnedInFlightConstraint.h"
#include "humanoid_common_mpc/contact_planning/constraint/ZmpSupportRegionConstraint.h"
#include "humanoid_common_mpc/contact_planning/cost/FootYawRegularizationCost.h"
#include "humanoid_common_mpc/contact_planning/cost/FootYawTrackingCost.h"
#include "humanoid_common_mpc/contact_planning/cost/FootholdRegularizationCost.h"
#include "humanoid_common_mpc/contact_planning/cost/HeadingRateTrackingCost.h"
#include "humanoid_common_mpc/contact_planning/cost/HeadingTrackingCost.h"
#include "humanoid_common_mpc/contact_planning/cost/HeightTrackingCost.h"
#include "humanoid_common_mpc/contact_planning/cost/PreviousFootholdConsistencyCost.h"
#include "humanoid_common_mpc/contact_planning/cost/RegularizationCost.h"
#include "humanoid_common_mpc/contact_planning/cost/StepLengthCost.h"
#include "humanoid_common_mpc/contact_planning/cost/StepWidthCost.h"
#include "humanoid_common_mpc/contact_planning/cost/TerminalDcmCost.h"
#include "humanoid_common_mpc/contact_planning/cost/VelocityTrackingCost.h"
#include "humanoid_common_mpc/contact_planning/cost/VerticalInputRegularizationCost.h"
#include "humanoid_common_mpc/contact_planning/cost/YawTorqueRegularizationCost.h"
#include "humanoid_common_mpc/contact_planning/cost/ZmpRegularizationCost.h"
#include "humanoid_common_mpc/contact_planning/execution/DcmStepAdjustmentRule.h"
#include "humanoid_common_mpc/contact_planning/execution/EnergyCadenceModulationRule.h"
#include "humanoid_common_mpc/contact_planning/execution/PhaseResettingRule.h"
#include "humanoid_common_mpc/contact_planning/logic/AlternatingFeetRule.h"
#include "humanoid_common_mpc/contact_planning/logic/ContactSwitchCost.h"
#include "humanoid_common_mpc/contact_planning/logic/FlightDurationsRule.h"
#include "humanoid_common_mpc/contact_planning/logic/HopOnRequestRule.h"
#include "humanoid_common_mpc/contact_planning/logic/MinimumDoubleSupportRule.h"
#include "humanoid_common_mpc/contact_planning/logic/NoFlightRule.h"
#include "humanoid_common_mpc/contact_planning/logic/PhaseDurationsRule.h"
#include "humanoid_common_mpc/contact_planning/logic/PlanConsistencyCost.h"
#include "humanoid_common_mpc/contact_planning/model/FootholdIntegrator.h"
#include "humanoid_common_mpc/contact_planning/model/HeadingDoubleIntegrator.h"
#include "humanoid_common_mpc/contact_planning/model/LipComDynamics.h"
#include "humanoid_common_mpc/contact_planning/model/VerticalDoubleIntegrator.h"
#include "humanoid_common_mpc/contact_planning/search/DivingStage.h"
#include "humanoid_common_mpc/contact_planning/search/EventShiftLocalSearchStage.h"
#include "humanoid_common_mpc/contact_planning/search/HeadingRelinearisationStage.h"
#include "humanoid_common_mpc/contact_planning/search/WarmStartPreviousPlanStage.h"

namespace ocs2::humanoid {

namespace {

std::string supported(TermKind kind) {
  std::string out;
  for (const std::string& name : knownTermNames(kind)) out += (out.empty() ? "" : ", ") + name;
  return out;
}

[[noreturn]] void unknown(TermKind kind, const std::string& name) {
  throw std::invalid_argument("[ContactPlanningTermFactory] unknown " + termKindName(kind) + " term '" + name +
                              "'; supported: " + supported(kind));
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
  if (canonical == term::kVerticalDoubleIntegrator) return std::make_unique<VerticalDoubleIntegrator>();
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
  if (canonical == term::kHeightTracking) return std::make_unique<HeightTrackingCost>();
  if (canonical == term::kVerticalInputRegularization) return std::make_unique<VerticalInputRegularizationCost>();
  unknown(TermKind::COST, name);
}

std::unique_ptr<LipConstraint> ContactPlanningTermFactory::makeSoftConstraint(const std::string& name) {
  const std::string canonical = canonicalOrThrow(TermKind::SOFT_CONSTRAINT, name);
  if (canonical == term::kZmpSupportRegion) return std::make_unique<ZmpSupportRegionConstraint>();
  if (canonical == term::kReachability) return std::make_unique<ReachabilityConstraint>();
  if (canonical == term::kFootSeparation) return std::make_unique<FootSeparationConstraint>();
  if (canonical == term::kHipYawRange) return std::make_unique<HipYawRangeConstraint>();
  if (canonical == term::kContactHeight) return std::make_unique<ContactHeightConstraint>();
  unknown(TermKind::SOFT_CONSTRAINT, name);
}

std::unique_ptr<LipConstraint> ContactPlanningTermFactory::makeHardConstraint(const std::string& name) {
  const std::string canonical = canonicalOrThrow(TermKind::HARD_CONSTRAINT, name);
  if (canonical == term::kNoFlight) return std::make_unique<NoFlightConstraint>();
  if (canonical == term::kFootMotionInSwingOnly) return std::make_unique<FootMotionInSwingOnlyConstraint>();
  if (canonical == term::kYawTorqueBudget) return std::make_unique<YawTorqueBudgetConstraint>();
  if (canonical == term::kFootYawPinnedInContact) return std::make_unique<FootYawPinnedInContactConstraint>();
  if (canonical == term::kVerticalThrustLimit) return std::make_unique<VerticalThrustLimitConstraint>();
  if (canonical == term::kZmpPinnedInFlight) return std::make_unique<ZmpPinnedInFlightConstraint>();
  unknown(TermKind::HARD_CONSTRAINT, name);
}

std::unique_ptr<ContactLogicRule> ContactPlanningTermFactory::makeLogicRule(const std::string& name) {
  const std::string canonical = canonicalOrThrow(TermKind::LOGIC_RULE, name);
  if (canonical == term::kPhaseDurations) return std::make_unique<PhaseDurationsRule>();
  if (canonical == term::kNoFlight) return std::make_unique<NoFlightRule>();
  if (canonical == term::kFlightDurations) return std::make_unique<FlightDurationsRule>();
  if (canonical == term::kHopOnRequest) return std::make_unique<HopOnRequestRule>();
  if (canonical == term::kMinimumDoubleSupport) return std::make_unique<MinimumDoubleSupportRule>();
  if (canonical == term::kAlternatingFeet) return std::make_unique<AlternatingFeetRule>();
  unknown(TermKind::LOGIC_RULE, name);
}

std::unique_ptr<AssignmentCost> ContactPlanningTermFactory::makeAssignmentCost(const std::string& name) {
  const std::string canonical = canonicalOrThrow(TermKind::ASSIGNMENT_COST, name);
  if (canonical == term::kContactSwitch) return std::make_unique<ContactSwitchCost>();
  if (canonical == term::kPlanConsistency) return std::make_unique<PlanConsistencyCost>();
  unknown(TermKind::ASSIGNMENT_COST, name);
}

std::unique_ptr<SearchStage> ContactPlanningTermFactory::makeSearchStage(const std::string& name) {
  const std::string canonical = canonicalOrThrow(TermKind::SEARCH_STAGE, name);
  if (canonical == term::kWarmStartPreviousPlan) return std::make_unique<WarmStartPreviousPlanStage>();
  if (canonical == term::kDiving) return std::make_unique<DivingStage>();
  if (canonical == term::kEventShiftLocalSearch) return std::make_unique<EventShiftLocalSearchStage>();
  if (canonical == term::kHeadingRelinearisation) return std::make_unique<HeadingRelinearisationStage>();
  unknown(TermKind::SEARCH_STAGE, name);
}

std::unique_ptr<ExecutionRule> ContactPlanningTermFactory::makeExecutionRule(const std::string& name, const ExtraRuleMaker& extra) {
  const std::string canonical = canonicalOrThrow(TermKind::EXECUTION_RULE, name);
  if (canonical == term::kPhaseResetting) return std::make_unique<PhaseResettingRule>();
  if (canonical == term::kEnergyCadenceModulation) return std::make_unique<EnergyCadenceModulationRule>();
  if (canonical == term::kDcmStepAdjustment) return std::make_unique<DcmStepAdjustmentRule>();
  if (extra) {
    std::unique_ptr<ExecutionRule> rule = extra(canonical);
    if (rule != nullptr) return rule;
  }
  throw std::invalid_argument("[ContactPlanningTermFactory] the execution rule '" + canonical +
                              "' is not available here (it needs the reference manager's robot model)");
}
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/contact_planning/ContactPlanningFormulation.cpp:known_term_names)

ContactPlanningProblem ContactPlanningTermFactory::buildProblem(const ContactPlanningConfig& config) {
  config.formulation.validate();
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

TermCollection<SearchStage> ContactPlanningTermFactory::buildSearchStages(const ContactPlanningConfig& config) {
  TermCollection<SearchStage> stages;
  for (const std::string& name : config.formulation.search) {
    std::unique_ptr<SearchStage> stage = makeSearchStage(name);
    for (const std::string& block : stage->requiredBlocks()) {
      if (!config.formulation.hasDynamics(block)) {
        throw std::invalid_argument("[ContactPlanningTermFactory] search stage '" + name + "' needs the '" + block + "' block");
      }
    }
    stage->configure(config);
    stages.add(canonicalTermName(TermKind::SEARCH_STAGE, name), std::move(stage));
  }
  return stages;
}

TermCollection<ExecutionRule> ContactPlanningTermFactory::buildExecutionRules(const ContactPlanningConfig& config,
                                                                              const ExtraRuleMaker& extra) {
  config.formulation.validate();
  TermCollection<ExecutionRule> rules;
  for (const std::string& name : config.formulation.execution) {
    std::unique_ptr<ExecutionRule> rule = makeExecutionRule(name, extra);
    for (const std::string& block : rule->requiredBlocks()) {
      if (!config.formulation.hasDynamics(block)) {
        throw std::invalid_argument("[ContactPlanningTermFactory] execution rule '" + name + "' needs the '" + block + "' block");
      }
    }
    rule->configure(config);
    rules.add(canonicalTermName(TermKind::EXECUTION_RULE, name), std::move(rule));
  }
  return rules;
}

}  // namespace ocs2::humanoid
