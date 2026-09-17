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

#include "humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace ocs2::humanoid {

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

std::string termKindName(TermKind kind) {
  switch (kind) {
    case TermKind::MODEL_BLOCK:
      return "dynamics";
    case TermKind::COST:
      return "costs";
    case TermKind::SOFT_CONSTRAINT:
      return "soft_constraints";
    case TermKind::HARD_CONSTRAINT:
      return "hard_constraints";
    case TermKind::LOGIC_RULE:
      return "logic_rules";
    case TermKind::ASSIGNMENT_COST:
      return "assignment_costs";
    case TermKind::SEARCH_STAGE:
      return "search";
    case TermKind::EXECUTION_RULE:
      return "execution";
  }
  return "?";
}

const std::vector<std::string>& knownTermNames(TermKind kind) {
  // LINT.IfChange(known_term_names)
  static const std::vector<std::string> blocks{term::kLipCom, term::kFootholdIntegrator, term::kHeadingDoubleIntegrator,
                                               term::kVerticalDoubleIntegrator};
  static const std::vector<std::string> costs{term::kRegularization,
                                              term::kPreviousFootholdConsistency,
                                              term::kVelocityTracking,
                                              term::kStepWidth,
                                              term::kHeadingRateTracking,
                                              term::kHeadingTracking,
                                              term::kFootYawTracking,
                                              term::kYawTorqueRegularization,
                                              term::kFootYawRegularization,
                                              term::kZmpRegularization,
                                              term::kFootholdRegularization,
                                              term::kStepLength,
                                              term::kTerminalDcm,
                                              term::kHeightTracking,
                                              term::kVerticalInputRegularization};
  static const std::vector<std::string> soft{term::kZmpSupportRegion, term::kReachability, term::kFootSeparation, term::kHipYawRange,
                                             term::kContactHeight};
  static const std::vector<std::string> hard{term::kNoFlight,
                                             term::kFootMotionInSwingOnly,
                                             term::kYawTorqueBudget,
                                             term::kFootYawPinnedInContact,
                                             term::kVerticalThrustLimit,
                                             term::kZmpPinnedInFlight};
  static const std::vector<std::string> logic{term::kPhaseDurations,  term::kNoFlight,        term::kMinimumDoubleSupport,
                                              term::kAlternatingFeet, term::kFlightDurations, term::kHopOnRequest};
  static const std::vector<std::string> assignment{term::kContactSwitch, term::kPlanConsistency};
  static const std::vector<std::string> search{term::kWarmStartPreviousPlan, term::kDiving, term::kEventShiftLocalSearch,
                                               term::kHeadingRelinearisation};
  static const std::vector<std::string> execution{term::kPhaseResetting, term::kEnergyCadenceModulation, term::kDcmStepAdjustment,
                                                  term::kPlannedHeadingOverride, term::kPlannedHeightOverride};
  // clang-format off
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/contact_planning/ContactPlanningTermFactory.cpp:term_factory, //robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/contact_planning.yaml:contact_planning_config)
  // clang-format on
  switch (kind) {
    case TermKind::MODEL_BLOCK:
      return blocks;
    case TermKind::COST:
      return costs;
    case TermKind::SOFT_CONSTRAINT:
      return soft;
    case TermKind::HARD_CONSTRAINT:
      return hard;
    case TermKind::LOGIC_RULE:
      return logic;
    case TermKind::ASSIGNMENT_COST:
      return assignment;
    case TermKind::SEARCH_STAGE:
      return search;
    case TermKind::EXECUTION_RULE:
      return execution;
  }
  return blocks;
}

std::string canonicalTermName(TermKind kind, const std::string& name) {
  for (const std::string& known : knownTermNames(kind)) {
    if (sameTermName(known, name)) return known;
  }
  return std::string();
}

namespace {

std::string joinNames(const std::vector<std::string>& names) {
  std::string out;
  for (size_t i = 0; i < names.size(); ++i) {
    if (i > 0) out += ", ";
    out += names[i];
  }
  return out.empty() ? "(none)" : out;
}

/** Model block a term needs, if any. */
std::string requiredBlockOf(TermKind kind, const std::string& canonical) {
  static const std::set<std::string> heading{term::kHeadingRateTracking,     term::kHeadingTracking,        term::kFootYawTracking,
                                             term::kYawTorqueRegularization, term::kFootYawRegularization,  term::kHipYawRange,
                                             term::kYawTorqueBudget,         term::kFootYawPinnedInContact, term::kHeadingRelinearisation,
                                             term::kPlannedHeadingOverride};
  static const std::set<std::string> vertical{term::kHeightTracking,      term::kVerticalInputRegularization, term::kContactHeight,
                                              term::kVerticalThrustLimit, term::kZmpPinnedInFlight,           term::kPlannedHeightOverride};
  if (kind == TermKind::MODEL_BLOCK) return std::string();
  if (heading.count(canonical)) return term::kHeadingDoubleIntegrator;
  if (vertical.count(canonical)) return term::kVerticalDoubleIntegrator;
  return std::string();
}

}  // namespace

std::vector<std::string>& ContactPlanningFormulation::list(TermKind kind) {
  switch (kind) {
    case TermKind::MODEL_BLOCK:
      return dynamics;
    case TermKind::COST:
      return costs;
    case TermKind::SOFT_CONSTRAINT:
      return softConstraints;
    case TermKind::HARD_CONSTRAINT:
      return hardConstraints;
    case TermKind::LOGIC_RULE:
      return logicRules;
    case TermKind::ASSIGNMENT_COST:
      return assignmentCosts;
    case TermKind::SEARCH_STAGE:
      return search;
    case TermKind::EXECUTION_RULE:
      return execution;
  }
  return dynamics;
}

const std::vector<std::string>& ContactPlanningFormulation::list(TermKind kind) const {
  return const_cast<ContactPlanningFormulation*>(this)->list(kind);
}

bool ContactPlanningFormulation::listed(const std::vector<std::string>& list, const std::string& name) {
  return std::any_of(list.begin(), list.end(), [&](const std::string& entry) { return sameTermName(entry, name); });
}

void ContactPlanningFormulation::setListed(std::vector<std::string>& list, const std::string& name, bool on) {
  const auto it = std::find_if(list.begin(), list.end(), [&](const std::string& entry) { return sameTermName(entry, name); });
  if (on && it == list.end()) list.push_back(name);
  if (!on && it != list.end()) list.erase(it);
}

void ContactPlanningFormulation::setHeadingModel(bool on) {
  static const std::array<const char*, 5> headingCosts{term::kHeadingRateTracking, term::kHeadingTracking, term::kFootYawTracking,
                                                       term::kYawTorqueRegularization, term::kFootYawRegularization};
  if (on) {
    setListed(dynamics, term::kHeadingDoubleIntegrator, true);
    // The heading costs go where the previous planner accumulated them: after the tracking costs on the state and before
    // the running costs, i.e. right before zmp_regularization (appended when that one is not listed).
    for (const char* name : headingCosts) setListed(costs, name, false);
    auto position =
        std::find_if(costs.begin(), costs.end(), [](const std::string& c) { return sameTermName(c, term::kZmpRegularization); });
    costs.insert(position, headingCosts.begin(), headingCosts.end());
    setListed(softConstraints, term::kHipYawRange, true);
    setListed(hardConstraints, term::kYawTorqueBudget, true);
    setListed(hardConstraints, term::kFootYawPinnedInContact, true);
    setListed(search, term::kHeadingRelinearisation, true);
    setListed(execution, term::kPlannedHeadingOverride, true);
  } else {
    setListed(dynamics, term::kHeadingDoubleIntegrator, false);
    for (const char* name : headingCosts) setListed(costs, name, false);
    setListed(softConstraints, term::kHipYawRange, false);
    setListed(hardConstraints, term::kYawTorqueBudget, false);
    setListed(hardConstraints, term::kFootYawPinnedInContact, false);
    setListed(search, term::kHeadingRelinearisation, false);
    setListed(execution, term::kPlannedHeadingOverride, false);
  }
}

void ContactPlanningFormulation::setFlightModel(bool on) {
  // `no_flight` forbids exactly what the flight model allows, so it is swapped in place with its counterpart: the row
  // with vertical_thrust_limit, the rule with flight_durations. In place, so that the accumulation order of the rows and
  // the propagation order of the rules do not depend on which model is listed, and so that the toggle is its own
  // inverse. An `after` entry is inserted right behind the replacement.
  const auto swapInPlace = [](std::vector<std::string>& list, const char* from, const char* to, const char* after = nullptr) {
    auto position = std::find_if(list.begin(), list.end(), [&](const std::string& entry) { return sameTermName(entry, from); });
    if (position == list.end()) {
      setListed(list, to, true);
      if (after != nullptr) setListed(list, after, true);
      return;
    }
    *position = to;
    if (after != nullptr && !listed(list, after)) list.insert(position + 1, after);
  };
  if (on) {
    setListed(dynamics, term::kVerticalDoubleIntegrator, true);
    setListed(costs, term::kHeightTracking, true);
    setListed(costs, term::kVerticalInputRegularization, true);
    setListed(softConstraints, term::kContactHeight, true);
    swapInPlace(hardConstraints, term::kNoFlight, term::kVerticalThrustLimit, term::kZmpPinnedInFlight);
    swapInPlace(logicRules, term::kNoFlight, term::kFlightDurations);
    setListed(logicRules, term::kHopOnRequest, true);
    setListed(execution, term::kPlannedHeightOverride, true);
  } else {
    setListed(dynamics, term::kVerticalDoubleIntegrator, false);
    setListed(costs, term::kHeightTracking, false);
    setListed(costs, term::kVerticalInputRegularization, false);
    setListed(softConstraints, term::kContactHeight, false);
    setListed(hardConstraints, term::kZmpPinnedInFlight, false);
    swapInPlace(hardConstraints, term::kVerticalThrustLimit, term::kNoFlight);
    setListed(logicRules, term::kHopOnRequest, false);
    swapInPlace(logicRules, term::kFlightDurations, term::kNoFlight);
    setListed(execution, term::kPlannedHeightOverride, false);
  }
}

void ContactPlanningFormulation::validate() const {
  const auto fail = [](const std::string& what) { throw std::invalid_argument("[ContactPlanningFormulation] " + what); };
  static const std::array<TermKind, 8> kinds{TermKind::MODEL_BLOCK,     TermKind::COST,          TermKind::SOFT_CONSTRAINT,
                                             TermKind::HARD_CONSTRAINT, TermKind::LOGIC_RULE,    TermKind::ASSIGNMENT_COST,
                                             TermKind::SEARCH_STAGE,    TermKind::EXECUTION_RULE};
  for (const TermKind kind : kinds) {
    std::set<std::string> seen;
    for (const std::string& name : list(kind)) {
      const std::string canonical = canonicalTermName(kind, name);
      if (canonical.empty()) {
        fail("unknown " + termKindName(kind) + " term '" + name + "'; supported: " + joinNames(knownTermNames(kind)));
      }
      if (!seen.insert(canonical).second) fail(termKindName(kind) + " lists '" + canonical + "' twice");
      const std::string required = requiredBlockOf(kind, canonical);
      if (!required.empty() && !hasDynamics(required)) {
        fail(termKindName(kind) + " term '" + canonical + "' needs the '" + required + "' block in the dynamics list");
      }
    }
  }
  if (listed(logicRules, term::kFlightDurations) && (listed(logicRules, term::kNoFlight) || listed(hardConstraints, term::kNoFlight))) {
    fail(std::string("'") + term::kFlightDurations + "' allows flight; remove '" + term::kNoFlight +
         "' from logic_rules and hard_constraints (they forbid it)");
  }
  if (hasFlightModel() && !listed(logicRules, term::kFlightDurations)) {
    fail(std::string("the '") + term::kVerticalDoubleIntegrator + "' block needs '" + term::kFlightDurations + "' in logic_rules");
  }
  if (dynamics.size() < 2 || !sameTermName(dynamics[0], term::kLipCom) || !sameTermName(dynamics[1], term::kFootholdIntegrator)) {
    fail(std::string("the dynamics list must start with '") + term::kLipCom + "', '" + term::kFootholdIntegrator +
         "' (they own the layout of the LIP block and the contact binaries)");
  }
  if (hasExecutionRule(term::kPhaseResetting) && hasExecutionRule(term::kEnergyCadenceModulation)) {
    const auto index = [&](const char* name) {
      return std::find_if(execution.begin(), execution.end(), [&](const std::string& e) { return sameTermName(e, name); }) -
             execution.begin();
    };
    if (index(term::kPhaseResetting) > index(term::kEnergyCadenceModulation)) {
      fail(std::string("'") + term::kPhaseResetting + "' must be listed before '" + term::kEnergyCadenceModulation +
           "' in execution: an early touch-down ends a swing before the cadence rule may re-time it");
    }
  }
}

std::string ContactPlanningFormulation::summary() const {
  std::ostringstream out;
  static const std::array<TermKind, 8> kinds{TermKind::MODEL_BLOCK,     TermKind::COST,          TermKind::SOFT_CONSTRAINT,
                                             TermKind::HARD_CONSTRAINT, TermKind::LOGIC_RULE,    TermKind::ASSIGNMENT_COST,
                                             TermKind::SEARCH_STAGE,    TermKind::EXECUTION_RULE};
  for (const TermKind kind : kinds) {
    out << termKindName(kind) << " (" << list(kind).size() << "): " << joinNames(list(kind)) << "\n";
  }
  return out.str();
}

bool ContactPlanningFormulation::operator==(const ContactPlanningFormulation& other) const {
  const auto same = [](const std::vector<std::string>& a, const std::vector<std::string>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
      if (!sameTermName(a[i], b[i])) return false;
    }
    return true;
  };
  return same(dynamics, other.dynamics) && same(costs, other.costs) && same(softConstraints, other.softConstraints) &&
         same(hardConstraints, other.hardConstraints) && same(logicRules, other.logicRules) &&
         same(assignmentCosts, other.assignmentCosts) && same(search, other.search) && same(execution, other.execution);
}

}  // namespace ocs2::humanoid
