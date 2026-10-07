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

#include <string>
#include <vector>

#include "absl/status/status.h"

namespace ocs2::humanoid {

/**
 * Canonical names of the terms the contact planner can be assembled from. They are the entries of the term lists of
 * `contact_planning.textproto` and the names of the parameter block of every term (ContactPlanningConfig), and they are what
 * the start-up print of the assembled problem shows. Lookups normalize a name like the task file's formulation lists
 * (case-insensitive, `_`, `-` and spaces ignored), so `velocityTracking` and `velocity_tracking` are the same term.
 *
 * These strings ARE the names a file writes, so renaming one renames a key of every robot's file. The registry of
 * ContactPlanningFormulation.cpp only refers to them, which is why the directive guards the strings themselves.
 */
// LINT.IfChange(term_names)
namespace term {
// Model blocks (the `dynamics` list). The first two are mandatory and always first: they own the variable layout the
// LipContactPlanner enums and the mixed-integer solver rely on.
inline constexpr char kLipCom[] = "lip_com";
inline constexpr char kFootholdIntegrator[] = "foothold_integrator";
inline constexpr char kHeadingDoubleIntegrator[] = "heading_double_integrator";
// Costs.
inline constexpr char kRegularization[] = "regularization";
inline constexpr char kPreviousFootholdConsistency[] = "previous_foothold_consistency";
inline constexpr char kVelocityTracking[] = "velocity_tracking";
inline constexpr char kStepWidth[] = "step_width";
inline constexpr char kHeadingRateTracking[] = "heading_rate_tracking";
inline constexpr char kHeadingTracking[] = "heading_tracking";
inline constexpr char kFootYawTracking[] = "foot_yaw_tracking";
inline constexpr char kYawTorqueRegularization[] = "yaw_torque_regularization";
inline constexpr char kFootYawRegularization[] = "foot_yaw_regularization";
inline constexpr char kZmpRegularization[] = "zmp_regularization";
inline constexpr char kFootholdRegularization[] = "foothold_regularization";
inline constexpr char kStepLength[] = "step_length";
inline constexpr char kTerminalDcm[] = "terminal_dcm";
// Soft constraints.
inline constexpr char kZmpSupportRegion[] = "zmp_support_region";
inline constexpr char kReachability[] = "reachability";
inline constexpr char kFootSeparation[] = "foot_separation";
inline constexpr char kHipYawRange[] = "hip_yaw_range";
// Hard constraints.
inline constexpr char kNoFlight[] = "no_flight";
inline constexpr char kFootMotionInSwingOnly[] = "foot_motion_in_swing_only";
inline constexpr char kYawTorqueBudget[] = "yaw_torque_budget";
inline constexpr char kFootYawPinnedInContact[] = "foot_yaw_pinned_in_contact";
// Logic rules on the contact binaries (`no_flight` is both a QP row and a logic rule, under the same name).
inline constexpr char kPhaseDurations[] = "phase_durations";
inline constexpr char kMinimumDoubleSupport[] = "minimum_double_support";
inline constexpr char kAlternatingFeet[] = "alternating_feet";
// Assignment costs on the binaries.
inline constexpr char kContactSwitch[] = "contact_switch";
inline constexpr char kPlanConsistency[] = "plan_consistency";
inline constexpr char kDoubleSupportPenalty[] = "double_support_penalty";
// Search stages around the branch-and-bound.
inline constexpr char kWarmStartPreviousPlan[] = "warm_start_previous_plan";
inline constexpr char kDiving[] = "diving";
inline constexpr char kEventShiftLocalSearch[] = "event_shift_local_search";
inline constexpr char kCadenceStretch[] = "cadence_stretch";
inline constexpr char kHeadingRelinearization[] = "heading_relinearization";
// Execution rules of the reference manager, applied between plans.
inline constexpr char kPhaseResetting[] = "phase_resetting";
inline constexpr char kEnergyCadenceModulation[] = "energy_cadence_modulation";
inline constexpr char kDcmStepAdjustment[] = "dcm_step_adjustment";
inline constexpr char kPlannedHeadingOverride[] = "planned_heading_override";
inline constexpr char kPlannedComOverride[] = "planned_com_override";
}  // namespace term
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/contact_planning/ContactPlanningFormulation.cpp:known_term_names, //humanoid_nmpc/docs/README.md:formulation_term_table, //humanoid_nmpc/humanoid_mpc_config/contact_planning_file.proto:term_lists, //humanoid_nmpc/humanoid_mpc_config/contact_planning_file.proto:term_blocks)
// clang-format on

/** Normalizes a term name for comparison: lower case, `_`, `-` and spaces removed. */
std::string normalizeTermName(const std::string& name);
/** True if the two names denote the same term. */
bool sameTermName(const std::string& a, const std::string& b);

/** Which list of the formulation a term belongs to. */
enum class TermKind { kModelBlock, kCost, kSoftConstraint, kHardConstraint, kLogicRule, kAssignmentCost, kSearchStage, kExecutionRule };
/** Every kind, in the order of the configuration file's lists. */
const std::vector<TermKind>& allTermKinds();
/** The configuration key of the list of a kind (`dynamics`, `costs`, ..., `execution`). */
std::string termKindName(TermKind kind);
/** Canonical names of every term of a kind, in the order of the default formulation. */
const std::vector<std::string>& knownTermNames(TermKind kind);
/** Canonical spelling of `name` among the terms of `kind`, or empty if the name is unknown. */
std::string canonicalTermName(TermKind kind, const std::string& name);

/**
 * The model block the term `name` of `kind` needs in the `dynamics` list, or empty when it needs none. This is the one
 * definition of "belongs to the heading model": validateStatus() checks the lists against it and setHeadingModel() adds and
 * removes exactly these terms, where both used to carry a list of their own that nothing kept in step. Each term's
 * ContactPlanningTerm::requiredBlocks() says the same thing a third time, from the term's side; a test builds every
 * term through the factory and holds the two answers equal.
 */
std::string requiredModelBlock(TermKind kind, const std::string& name);

/**
 * The term lists of the planner's formulation: which model blocks, costs, constraints, logic rules, assignment costs,
 * search stages and execution rules the planner is assembled from. Order matters where it is stated:
 *  - `dynamics` fixes the variable layout (blocks append their variables in list order);
 *  - `costs` is the accumulation order of the stage matrices (a sum in floating point is not associative, so the shipped
 *    order reproduces the previous planner bit for bit);
 *  - `execution`: `phase_resetting` must precede `energy_cadence_modulation` when both are listed, because an early
 *    touch-down ends a swing before the cadence rule may re-time it (validateStatus() enforces it).
 * The defaults are the point-mass LIP planner without the heading model and without execution rules. That is the
 * formulation of `planner.type: lip_miqp`, the library default (ContactPlanningConfig.h). It is NOT a working
 * formulation for `planner.type: hlip`, which needs `planned_com_override` in `execution`; the rule is not put into
 * this default because the default is shared with lip_miqp, and ContactPlanningConfig::warnings() reports the
 * omission instead.
 */
struct ContactPlanningFormulation {
  std::vector<std::string> dynamics{term::kLipCom, term::kFootholdIntegrator};
  std::vector<std::string> costs{term::kRegularization,    term::kPreviousFootholdConsistency, term::kVelocityTracking, term::kStepWidth,
                                 term::kZmpRegularization, term::kFootholdRegularization,      term::kTerminalDcm};
  std::vector<std::string> softConstraints{term::kZmpSupportRegion, term::kReachability, term::kFootSeparation};
  std::vector<std::string> hardConstraints{term::kNoFlight, term::kFootMotionInSwingOnly};
  std::vector<std::string> logicRules{term::kPhaseDurations, term::kNoFlight, term::kMinimumDoubleSupport, term::kAlternatingFeet};
  std::vector<std::string> assignmentCosts{term::kContactSwitch, term::kPlanConsistency};
  std::vector<std::string> search{term::kWarmStartPreviousPlan, term::kDiving, term::kEventShiftLocalSearch};
  std::vector<std::string> execution{};

  std::vector<std::string>& list(TermKind kind);
  const std::vector<std::string>& list(TermKind kind) const;

  static bool listed(const std::vector<std::string>& list, const std::string& name);
  /** Adds `name` to the end of the list, or removes it, without duplicates. */
  static void setListed(std::vector<std::string>& list, const std::string& name, bool on);

  bool hasDynamics(const std::string& name) const { return listed(dynamics, name); }
  bool hasCost(const std::string& name) const { return listed(costs, name); }
  bool hasSoftConstraint(const std::string& name) const { return listed(softConstraints, name); }
  bool hasHardConstraint(const std::string& name) const { return listed(hardConstraints, name); }
  bool hasLogicRule(const std::string& name) const { return listed(logicRules, name); }
  bool hasSearchStage(const std::string& name) const { return listed(search, name); }
  bool hasExecutionRule(const std::string& name) const { return listed(execution, name); }

  void setExecutionRule(const std::string& name, bool on) { setListed(execution, name, on); }
  void setLogicRule(const std::string& name, bool on) { setListed(logicRules, name, on); }
  void setSearchStage(const std::string& name, bool on) { setListed(search, name, on); }
  void setCost(const std::string& name, bool on) { setListed(costs, name, on); }

  /** True when the heading model (the ACoM dynamics block) is part of the reduced model. */
  bool usesHeadingModel() const { return hasDynamics(term::kHeadingDoubleIntegrator); }
  /**
   * Adds or removes the heading model as a whole: the block, its costs (inserted before `zmp_regularization`, which
   * keeps the accumulation order of the previous planner), its constraints, the re-linearization stage and the
   * planned-heading override of the reference manager. This is what `useAcomDynamics: true` meant.
   */
  void setHeadingModel(bool on);

  /** True when a listed execution rule compares the measured center of mass with the NMPC's prediction. */
  bool needsPredictedTrajectory() const {
    return hasExecutionRule(term::kEnergyCadenceModulation) || hasExecutionRule(term::kDcmStepAdjustment);
  }

  /**
   * InvalidArgument on an unknown name (the message lists the supported ones), a duplicate, a missing mandatory block,
   * a term whose required block is not listed, or an execution order the rules cannot honor. Every message names the
   * list key (`costs`, `execution`, ...) the operator has to edit.
   */
  absl::Status validateStatus() const;

  /** One line per list, for the start-up print. */
  std::string summary() const;

  bool operator==(const ContactPlanningFormulation& other) const;
  bool operator!=(const ContactPlanningFormulation& other) const { return !(*this == other); }
};

}  // namespace ocs2::humanoid
