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

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid {

/**
 * Configuration for a single cost or constraint term in the Optimal Control Problem.
 */
struct ProblemTermConfig {
  std::string name;
  std::string type;
  std::string configPath;
  std::string penaltyType = "RelaxedBarrierPenalty";
  bool perContact = false;
  bool enabled = true;
};

/**
 * Full declarative definition of an MPC Optimal Control Problem loaded from YAML.
 */
struct MpcProblemDefinition {
  std::vector<ProblemTermConfig> costs;
  std::vector<ProblemTermConfig> terminalCosts;
  std::vector<ProblemTermConfig> stateSoftConstraints;
  std::vector<ProblemTermConfig> softConstraints;
  std::vector<ProblemTermConfig> equalityConstraints;
};

/**
 * Loads and validates an MpcProblemDefinition from a task YAML file.
 *
 * @param [in] filename: Path to the task YAML file.
 * @param [in] fieldName: Top-level YAML block name (default: "problem_definition").
 * @param [in] verbose: Whether to print loaded configuration.
 * @return absl::StatusOr containing the parsed MpcProblemDefinition or an error status.
 */
absl::StatusOr<MpcProblemDefinition> loadMpcProblemDefinition(absl::string_view filename,
                                                              absl::string_view fieldName = "problem_definition",
                                                              bool verbose = true);

}  // namespace ocs2::humanoid
