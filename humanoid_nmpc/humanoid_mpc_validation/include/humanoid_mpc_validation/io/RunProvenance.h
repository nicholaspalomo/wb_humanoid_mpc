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

#include "humanoid_mpc_validation/io/GoldenIo.h"
#include "humanoid_mpc_validation/io/JsonValue.h"

namespace ocs2::humanoid::validation {

/** Where a recording was made: the commit, the state of the worktree and the machine. */
struct RunEnvironment {
  std::string gitCommit = "unknown";
  std::string worktreeState = "unknown";
  std::string machine = "unknown";
};

// LINT.IfChange(provenance_environment)
/**
 * The environment the Make targets pass into the test (WB_VALIDATION_GIT_COMMIT, WB_VALIDATION_WORKTREE_STATE,
 * WB_VALIDATION_MACHINE): a test cannot ask git itself, its sandbox has no checkout. Unset values stay "unknown", and
 * the machine then falls back to the CPU model, the logical cores and the memory read from /proc.
 */
RunEnvironment runEnvironmentFromEnvironment();
// LINT.ThenChange(//Makefile:closed_loop_targets)

/** "<CPU model>, <n> logical cores, <m> GiB": the machine as /proc describes it, "unknown" for what it does not say. */
std::string describeMachine();

/**
 * The provenance object of a metrics or benchmark document: the environment and the SHA-256 of every configuration
 * file. NotFound naming the first file that cannot be read.
 */
absl::StatusOr<JsonValue> makeRunProvenanceJson(const RunEnvironment& environment, const std::vector<std::string>& configurationFiles);

/** The same as the header of a golden file. */
absl::StatusOr<GoldenProvenance> makeRunGoldenProvenance(const RunEnvironment& environment,
                                                         const std::vector<std::string>& configurationFiles);

}  // namespace ocs2::humanoid::validation
