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

#include "humanoid_mpc_validation/io/RunProvenance.h"

#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"

#include "humanoid_mpc_validation/io/Sha256.h"

namespace ocs2::humanoid::validation {
namespace {

std::string environmentOr(const char* absl_nonnull name, const std::string& fallback) {
  const char* absl_nullable value = std::getenv(name);
  return value != nullptr && value[0] != '\0' ? std::string(value) : fallback;
}

/** The value after the first "<key>:" line of `file`, trimmed; empty when there is none. */
std::string firstValue(const std::string& file, const std::string& key) {
  std::ifstream stream(file);
  std::string line;
  while (std::getline(stream, line)) {
    if (!absl::StartsWith(line, key)) continue;
    const size_t colon = line.find(':');
    if (colon == std::string::npos) continue;
    return std::string(absl::StripAsciiWhitespace(line.substr(colon + 1)));
  }
  return "";
}

}  // namespace

std::string describeMachine() {
  const std::string cpu = firstValue("/proc/cpuinfo", "model name");
  const std::string memory = firstValue("/proc/meminfo", "MemTotal");  // "32768000 kB"
  std::string description = cpu.empty() ? "unknown CPU" : cpu;
  absl::StrAppend(&description, ", ", std::thread::hardware_concurrency(), " logical cores");
  const std::vector<absl::string_view> memoryFields = absl::StrSplit(memory, ' ', absl::SkipEmpty());
  double kibibytes = 0.0;
  if (!memoryFields.empty() && absl::SimpleAtod(memoryFields[0], &kibibytes)) {
    absl::StrAppend(&description, ", ", absl::StrFormat("%.1f", kibibytes / (1024.0 * 1024.0)), " GiB");
  }
  return description;
}

RunEnvironment runEnvironmentFromEnvironment() {
  RunEnvironment environment;
  environment.gitCommit = environmentOr("WB_VALIDATION_GIT_COMMIT", environment.gitCommit);
  environment.worktreeState = environmentOr("WB_VALIDATION_WORKTREE_STATE", environment.worktreeState);
  environment.machine = environmentOr("WB_VALIDATION_MACHINE", describeMachine());
  return environment;
}

absl::StatusOr<JsonValue> makeRunProvenanceJson(const RunEnvironment& environment, const std::vector<std::string>& configurationFiles) {
  JsonValue provenance = JsonValue::object();
  provenance.set("git_commit", JsonValue::string(environment.gitCommit));
  provenance.set("worktree_state", JsonValue::string(environment.worktreeState));
  provenance.set("machine", JsonValue::string(environment.machine));
  JsonValue& hashes = provenance.set("configuration_sha256", JsonValue::object());
  for (const std::string& file : configurationFiles) {
    absl::StatusOr<std::string> hash = sha256HexOfFile(file);
    if (!hash.ok()) return hash.status();
    hashes.set(file, JsonValue::string(*std::move(hash)));
  }
  return provenance;
}

absl::StatusOr<GoldenProvenance> makeRunGoldenProvenance(const RunEnvironment& environment,
                                                         const std::vector<std::string>& configurationFiles) {
  absl::StatusOr<GoldenProvenance> provenance = makeGoldenProvenance(environment.gitCommit, environment.worktreeState, configurationFiles);
  if (!provenance.ok()) return provenance;
  provenance->notes.emplace_back("machine", environment.machine);
  return provenance;
}

}  // namespace ocs2::humanoid::validation
