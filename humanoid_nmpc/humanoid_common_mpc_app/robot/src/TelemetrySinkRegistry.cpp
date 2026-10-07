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

#include "humanoid_common_mpc_app/robot/TelemetrySinkRegistry.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc_app/robot/BusTelemetrySink.h"

namespace ocs2::humanoid {

TelemetrySinkRegistry::TelemetrySinkRegistry() {
  add(std::string(kBusTelemetrySinkName), "publishes robot/state on the IPC bus",
      [](const TelemetrySinkContext& context) -> absl::StatusOr<std::unique_ptr<TelemetrySink>> {
        if (context.bus == nullptr) {
          return absl::FailedPreconditionError("the bus telemetry sink needs the robot process's bus");
        }
        return std::make_unique<BusTelemetrySink>(*context.bus);
      });
}

void TelemetrySinkRegistry::add(const std::string& name, const std::string& description, Factory factory) {
  entries_.push_back(Entry{.name = name, .description = description, .factory = std::move(factory)});
}

bool TelemetrySinkRegistry::has(absl::string_view name) const {
  for (const Entry& entry : entries_) {
    if (entry.name == name) return true;
  }
  return false;
}

absl::StatusOr<std::unique_ptr<TelemetrySink>> TelemetrySinkRegistry::create(absl::string_view name,
                                                                             const TelemetrySinkContext& context) const {
  for (const Entry& entry : entries_) {
    if (entry.name == name) return entry.factory(context);
  }
  return absl::InvalidArgumentError(
      absl::StrCat("There is no telemetry sink '", name, "' (telemetry_sinks). Available: ", availableNames(), "."));
}

std::string TelemetrySinkRegistry::availableNames() const {
  std::string names;
  for (const Entry& entry : entries_) {
    absl::StrAppend(&names, names.empty() ? "" : ", ", entry.name, " (", entry.description, ")");
  }
  return names;
}

std::vector<std::string> TelemetrySinkRegistry::names() const {
  std::vector<std::string> names;
  names.reserve(entries_.size());
  for (const Entry& entry : entries_) {
    names.push_back(entry.name);
  }
  return names;
}

}  // namespace ocs2::humanoid
