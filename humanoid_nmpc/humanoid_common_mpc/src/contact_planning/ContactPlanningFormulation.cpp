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

#include "humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid {

namespace {

absl::Status invalidFormulation(absl::string_view message) {
  return absl::InvalidArgumentError(absl::StrCat("[ContactPlanningFormulation] ", message));
}

std::string joinNames(const std::vector<std::string>& names) {
  return names.empty() ? std::string("(none)") : absl::StrJoin(names, ", ");
}

/**
 * The terms that read the heading model's variables, and so need its block in `dynamics`. The single list that
 * requiredModelBlock(), and through it validateStatus() and setHeadingModel(), reads.
 */
const absl::flat_hash_set<std::string>& headingModelTerms() {
  static const absl::flat_hash_set<std::string>& kTerms = *new absl::flat_hash_set<std::string>{
      term::kHeadingRateTracking,    term::kHeadingTracking,       term::kFootYawTracking, term::kYawTorqueRegularization,
      term::kFootYawRegularization,  term::kHipYawRange,           term::kYawTorqueBudget, term::kFootYawPinnedInContact,
      term::kHeadingRelinearization, term::kPlannedHeadingOverride};
  return kTerms;
}

/** Position of `name` in `list` (matched like every other lookup), or list.size() when it is not listed. */
size_t positionIn(const std::vector<std::string>& list, const std::string& name) {
  for (size_t i = 0; i < list.size(); ++i) {
    if (sameTermName(list[i], name)) return i;
  }
  return list.size();
}

}  // namespace

std::string normalizeTermName(const std::string& name) {
  std::string out;
  out.reserve(name.size());
  for (const char c : name) {
    if (c == '_' || c == '-' || c == ' ') continue;
    out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  return out;
}

bool sameTermName(const std::string& a, const std::string& b) {
  return normalizeTermName(a) == normalizeTermName(b);
}

const std::vector<TermKind>& allTermKinds() {
  static const std::vector<TermKind>& kKinds =
      *new std::vector<TermKind>{TermKind::kModelBlock, TermKind::kCost,           TermKind::kSoftConstraint, TermKind::kHardConstraint,
                                 TermKind::kLogicRule,  TermKind::kAssignmentCost, TermKind::kSearchStage,    TermKind::kExecutionRule};
  return kKinds;
}

std::string termKindName(TermKind kind) {
  switch (kind) {
    case TermKind::kModelBlock:
      return "dynamics";
    case TermKind::kCost:
      return "costs";
    case TermKind::kSoftConstraint:
      return "soft_constraints";
    case TermKind::kHardConstraint:
      return "hard_constraints";
    case TermKind::kLogicRule:
      return "logic_rules";
    case TermKind::kAssignmentCost:
      return "assignment_costs";
    case TermKind::kSearchStage:
      return "search";
    case TermKind::kExecutionRule:
      return "execution";
  }
  return "?";
}

const std::vector<std::string>& knownTermNames(TermKind kind) {
  // LINT.IfChange(known_term_names)
  static const std::vector<std::string>& kBlocks =
      *new std::vector<std::string>{term::kLipCom, term::kFootholdIntegrator, term::kHeadingDoubleIntegrator};
  static const std::vector<std::string>& kCosts =
      *new std::vector<std::string>{term::kRegularization,    term::kPreviousFootholdConsistency, term::kVelocityTracking,
                                    term::kStepWidth,         term::kHeadingRateTracking,         term::kHeadingTracking,
                                    term::kFootYawTracking,   term::kYawTorqueRegularization,     term::kFootYawRegularization,
                                    term::kZmpRegularization, term::kFootholdRegularization,      term::kStepLength,
                                    term::kTerminalDcm};
  static const std::vector<std::string>& kSoft =
      *new std::vector<std::string>{term::kZmpSupportRegion, term::kReachability, term::kFootSeparation, term::kHipYawRange};
  static const std::vector<std::string>& kHard =
      *new std::vector<std::string>{term::kNoFlight, term::kFootMotionInSwingOnly, term::kYawTorqueBudget, term::kFootYawPinnedInContact};
  static const std::vector<std::string>& kLogic =
      *new std::vector<std::string>{term::kPhaseDurations, term::kNoFlight, term::kMinimumDoubleSupport, term::kAlternatingFeet};
  static const std::vector<std::string>& kAssignment =
      *new std::vector<std::string>{term::kContactSwitch, term::kPlanConsistency, term::kDoubleSupportPenalty};
  static const std::vector<std::string>& kSearch = *new std::vector<std::string>{
      term::kWarmStartPreviousPlan, term::kDiving, term::kEventShiftLocalSearch, term::kHeadingRelinearization, term::kCadenceStretch};
  static const std::vector<std::string>& kExecution =
      *new std::vector<std::string>{term::kPhaseResetting, term::kEnergyCadenceModulation, term::kDcmStepAdjustment,
                                    term::kPlannedHeadingOverride, term::kPlannedComOverride};
  // Both robots' files list these names in their tail blocks, and the two READMEs count and tabulate them (section
  // 2.10 of docs/README.md, section 5 of docs/hlip_contact_planner/README.md), so all four are named here.
  // clang-format off
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/contact_planning/ContactPlanningTermFactory.cpp:term_factory, //humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h:term_names, //humanoid_nmpc/docs/README.md:formulation_term_table, //humanoid_nmpc/docs/hlip_contact_planner/README.md:hlip_term_registry_counts, //humanoid_nmpc/humanoid_mpc_config/contact_planning_file.proto:term_blocks)
  // clang-format on
  switch (kind) {
    case TermKind::kModelBlock:
      return kBlocks;
    case TermKind::kCost:
      return kCosts;
    case TermKind::kSoftConstraint:
      return kSoft;
    case TermKind::kHardConstraint:
      return kHard;
    case TermKind::kLogicRule:
      return kLogic;
    case TermKind::kAssignmentCost:
      return kAssignment;
    case TermKind::kSearchStage:
      return kSearch;
    case TermKind::kExecutionRule:
      return kExecution;
  }
  return kBlocks;
}

std::string canonicalTermName(TermKind kind, const std::string& name) {
  for (const std::string& known : knownTermNames(kind)) {
    if (sameTermName(known, name)) return known;
  }
  return std::string();
}

std::string requiredModelBlock(TermKind kind, const std::string& name) {
  if (kind == TermKind::kModelBlock) return std::string();
  const std::string canonical = canonicalTermName(kind, name);
  if (canonical.empty()) return std::string();
  return headingModelTerms().contains(canonical) ? std::string(term::kHeadingDoubleIntegrator) : std::string();
}

std::vector<std::string>& ContactPlanningFormulation::list(TermKind kind) {
  switch (kind) {
    case TermKind::kModelBlock:
      return dynamics;
    case TermKind::kCost:
      return costs;
    case TermKind::kSoftConstraint:
      return softConstraints;
    case TermKind::kHardConstraint:
      return hardConstraints;
    case TermKind::kLogicRule:
      return logicRules;
    case TermKind::kAssignmentCost:
      return assignmentCosts;
    case TermKind::kSearchStage:
      return search;
    case TermKind::kExecutionRule:
      return execution;
  }
  return dynamics;
}

const std::vector<std::string>& ContactPlanningFormulation::list(TermKind kind) const {
  return const_cast<ContactPlanningFormulation*>(this)->list(kind);
}

bool ContactPlanningFormulation::listed(const std::vector<std::string>& list, const std::string& name) {
  return positionIn(list, name) < list.size();
}

void ContactPlanningFormulation::setListed(std::vector<std::string>& list, const std::string& name, bool on) {
  const size_t position = positionIn(list, name);
  if (on && position == list.size()) list.push_back(name);
  if (!on && position < list.size()) list.erase(list.begin() + static_cast<ptrdiff_t>(position));
}

void ContactPlanningFormulation::setHeadingModel(bool on) {
  setListed(dynamics, term::kHeadingDoubleIntegrator, on);
  for (const TermKind kind : allTermKinds()) {
    if (kind == TermKind::kModelBlock) continue;
    // The heading terms of this list, in registry order, from the one definition requiredModelBlock() reads.
    std::vector<std::string> headingTerms;
    for (const std::string& name : knownTermNames(kind)) {
      if (!requiredModelBlock(kind, name).empty()) headingTerms.push_back(name);
    }
    std::vector<std::string>& terms = list(kind);
    if (on && kind == TermKind::kCost) {
      // The heading costs go where the previous planner accumulated them: after the tracking costs on the state and
      // before the running costs, i.e. right before zmp_regularization (appended when that one is not listed).
      for (const std::string& name : headingTerms) setListed(terms, name, /*on=*/false);
      const size_t position = positionIn(terms, term::kZmpRegularization);
      terms.insert(terms.begin() + static_cast<ptrdiff_t>(position), headingTerms.begin(), headingTerms.end());
    } else {
      for (const std::string& name : headingTerms) setListed(terms, name, on);
    }
  }
}

absl::Status ContactPlanningFormulation::validateStatus() const {
  for (const TermKind kind : allTermKinds()) {
    const std::string key = termKindName(kind);
    absl::flat_hash_set<std::string> seen;
    for (const std::string& name : list(kind)) {
      const std::string canonical = canonicalTermName(kind, name);
      if (canonical.empty()) {
        return invalidFormulation(absl::StrCat("unknown ", key, " term '", name, "'; supported: ", joinNames(knownTermNames(kind))));
      }
      if (!seen.insert(canonical).second) return invalidFormulation(absl::StrCat(key, " lists '", canonical, "' twice"));
      const std::string required = requiredModelBlock(kind, canonical);
      if (!required.empty() && !hasDynamics(required)) {
        return invalidFormulation(absl::StrCat(key, " term '", canonical, "' needs the '", required, "' block in the dynamics list"));
      }
    }
  }
  if (dynamics.size() < 2 || !sameTermName(dynamics[0], term::kLipCom) || !sameTermName(dynamics[1], term::kFootholdIntegrator)) {
    return invalidFormulation(absl::StrCat("the dynamics list must start with '", term::kLipCom, "', '", term::kFootholdIntegrator,
                                           "' (they own the layout of the LIP block and the contact binaries)"));
  }
  if (hasExecutionRule(term::kPhaseResetting) && hasExecutionRule(term::kEnergyCadenceModulation) &&
      positionIn(execution, term::kPhaseResetting) > positionIn(execution, term::kEnergyCadenceModulation)) {
    return invalidFormulation(absl::StrCat("'", term::kPhaseResetting, "' must be listed before '", term::kEnergyCadenceModulation,
                                           "' in execution: an early touch-down ends a swing before the cadence rule may re-time it"));
  }
  return absl::OkStatus();
}

std::string ContactPlanningFormulation::summary() const {
  std::string out;
  for (const TermKind kind : allTermKinds()) {
    absl::StrAppend(&out, termKindName(kind), " (", list(kind).size(), "): ", joinNames(list(kind)), "\n");
  }
  return out;
}

bool ContactPlanningFormulation::operator==(const ContactPlanningFormulation& other) const {
  for (const TermKind kind : allTermKinds()) {
    const std::vector<std::string>& mine = list(kind);
    const std::vector<std::string>& theirs = other.list(kind);
    if (mine.size() != theirs.size()) return false;
    for (size_t i = 0; i < mine.size(); ++i) {
      if (!sameTermName(mine[i], theirs[i])) return false;
    }
  }
  return true;
}

}  // namespace ocs2::humanoid
