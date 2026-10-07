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

#include "humanoid_common_mpc_app/robot/RobotBackendRegistry.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc_app/robot/MujocoRobotBackend.h"

namespace ocs2::humanoid {

RobotBackendRegistry::RobotBackendRegistry() {
  add(
      std::string(kMujocoBackendName), "the MuJoCo simulator",
      [](const RobotBackendOptions& options) -> absl::StatusOr<std::unique_ptr<RobotBackend>> {
        absl::StatusOr<std::unique_ptr<MujocoRobotBackend>> backend = MujocoRobotBackend::Create(options);
        if (!backend.ok()) return backend.status();
        return std::unique_ptr<RobotBackend>(*std::move(backend));
      },
      [](const RobotBackendOptions& options) { return MujocoRobotBackend::checkOptions(options); });
}

void RobotBackendRegistry::add(const std::string& name, const std::string& description, Factory factory, OptionsCheck checkOptions) {
  entries_.push_back(
      Entry{.name = name, .description = description, .factory = std::move(factory), .checkOptions = std::move(checkOptions)});
}

const RobotBackendRegistry::Entry* absl_nullable RobotBackendRegistry::find(absl::string_view name) const {
  for (const Entry& entry : entries_) {
    if (entry.name == name) return &entry;
  }
  return nullptr;
}

absl::Status RobotBackendRegistry::unknownBackend(absl::string_view name) const {
  return absl::InvalidArgumentError(absl::StrCat("There is no robot backend '", name, "' (--backend). Available: ", availableNames(), "."));
}

bool RobotBackendRegistry::has(absl::string_view name) const {
  for (const Entry& entry : entries_) {
    if (entry.name == name) return true;
  }
  return false;
}

std::vector<std::string> RobotBackendRegistry::names() const {
  std::vector<std::string> names;
  names.reserve(entries_.size());
  for (const Entry& entry : entries_) names.push_back(entry.name);
  return names;
}

std::string RobotBackendRegistry::availableNames() const {
  std::string names;
  for (const Entry& entry : entries_) {
    absl::StrAppend(&names, names.empty() ? "" : ", ", entry.name, " (", entry.description, ")");
  }
  return names;
}

absl::StatusOr<std::unique_ptr<RobotBackend>> RobotBackendRegistry::create(absl::string_view name,
                                                                           const RobotBackendOptions& options) const {
  const Entry* absl_nullable const entry = find(name);
  if (entry == nullptr) return unknownBackend(name);
  return entry->factory(options);
}

absl::Status RobotBackendRegistry::checkOptions(absl::string_view name, const RobotBackendOptions& options) const {
  const Entry* absl_nullable const entry = find(name);
  if (entry == nullptr) return unknownBackend(name);
  if (!entry->checkOptions) return absl::OkStatus();
  return entry->checkOptions(options);
}

}  // namespace ocs2::humanoid
