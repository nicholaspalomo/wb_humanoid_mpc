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

#include "humanoid_common_mpc/problem/MpcProblemDefinition.h"

#include <iostream>
#include <string>
#include <vector>

#include <ocs2_core/misc/LoadData.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid {

namespace {

inline constexpr absl::string_view kDefaultPenaltyType =
    "RelaxedBarrierPenalty";
inline constexpr absl::string_view kCostsSection = "costs";
inline constexpr absl::string_view kTerminalCostsSection = "terminal_costs";
inline constexpr absl::string_view kStateSoftConstraintsSection =
    "state_soft_constraints";
inline constexpr absl::string_view kSoftConstraintsSection = "soft_constraints";
inline constexpr absl::string_view kEqualityConstraintsSection =
    "equality_constraints";

absl::StatusOr<std::vector<ProblemTermConfig>> parseSection(
    const loadData::PropertyTree& parentTree,
    absl::string_view sectionName,
    bool verbose) {
  std::vector<ProblemTermConfig> terms;
  const auto optionalSection =
      parentTree.get_child_optional(std::string(sectionName));
  if (!optionalSection.has_value()) {
    return terms;
  }

  const loadData::PropertyTree& sectionTree = optionalSection.value();
  for (const auto& entry : sectionTree) {
    const absl::string_view nodeKey = entry.first;
    const loadData::PropertyTree& termTree = entry.second;

    ProblemTermConfig term;
    if (termTree.count("name") > 0) {
      term.name = termTree.get<std::string>("name");
    } else {
      term.name = std::string(nodeKey);
    }

    if (termTree.count("type") == 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Section '%s', term '%s' is missing required 'type' attribute.",
          sectionName, term.name));
    }
    term.type = termTree.get<std::string>("type");
    term.configPath = termTree.get<std::string>("config_path", "");
    term.penaltyType =
        termTree.get<std::string>("penalty", std::string(kDefaultPenaltyType));
    term.perContact = termTree.get<bool>("per_contact", false);
    term.enabled = termTree.get<bool>("enabled", true);

    if (verbose) {
      std::cout << absl::StrFormat(
          " [MpcProblemDefinition]   - %s (%s): type=%s, enabled=%s, "
          "perContact=%s, configPath=%s\n",
          sectionName, term.name, term.type, term.enabled ? "true" : "false",
          term.perContact ? "true" : "false", term.configPath);
    }

    terms.push_back(std::move(term));
  }

  return terms;
}

}  // namespace

absl::StatusOr<MpcProblemDefinition> loadMpcProblemDefinition(
    absl::string_view filename, absl::string_view fieldName, bool verbose) {
  loadData::PropertyTree pt;
  const std::string filenameStr(filename);
  try {
    loadData::readPropertyTree(filenameStr, pt);
  } catch (const std::exception& e) {
    return absl::NotFoundError(absl::StrFormat(
        "Failed to read property tree from '%s': %s", filename, e.what()));
  }

  const auto problemDefOptional = pt.get_child_optional(std::string(fieldName));
  if (!problemDefOptional.has_value()) {
    return absl::NotFoundError(absl::StrFormat(
        "Field '%s' not found in file '%s'.", fieldName, filename));
  }

  if (verbose) {
    std::cout << absl::StrFormat(
        "\n #### Loading MpcProblemDefinition from '%s' [%s] ####\n", filename,
        fieldName);
  }

  const loadData::PropertyTree& problemTree = problemDefOptional.value();
  MpcProblemDefinition problemDefinition;

  // 1. Costs
  const absl::StatusOr<std::vector<ProblemTermConfig>> costsStatus =
      parseSection(problemTree, kCostsSection, verbose);
  if (!costsStatus.ok()) {
    return costsStatus.status();
  }
  problemDefinition.costs = costsStatus.value();

  // 2. Terminal Costs
  const absl::StatusOr<std::vector<ProblemTermConfig>> terminalCostsStatus =
      parseSection(problemTree, kTerminalCostsSection, verbose);
  if (!terminalCostsStatus.ok()) {
    return terminalCostsStatus.status();
  }
  problemDefinition.terminalCosts = terminalCostsStatus.value();

  // 3. State Soft Constraints
  const absl::StatusOr<std::vector<ProblemTermConfig>>
      stateSoftConstraintsStatus =
          parseSection(problemTree, kStateSoftConstraintsSection, verbose);
  if (!stateSoftConstraintsStatus.ok()) {
    return stateSoftConstraintsStatus.status();
  }
  problemDefinition.stateSoftConstraints = stateSoftConstraintsStatus.value();

  // 4. Soft Constraints (State-Input)
  const absl::StatusOr<std::vector<ProblemTermConfig>> softConstraintsStatus =
      parseSection(problemTree, kSoftConstraintsSection, verbose);
  if (!softConstraintsStatus.ok()) {
    return softConstraintsStatus.status();
  }
  problemDefinition.softConstraints = softConstraintsStatus.value();

  // 5. Equality Constraints
  const absl::StatusOr<std::vector<ProblemTermConfig>>
      equalityConstraintsStatus =
          parseSection(problemTree, kEqualityConstraintsSection, verbose);
  if (!equalityConstraintsStatus.ok()) {
    return equalityConstraintsStatus.status();
  }
  problemDefinition.equalityConstraints = equalityConstraintsStatus.value();

  return problemDefinition;
}

}  // namespace ocs2::humanoid
