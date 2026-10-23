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

#include "humanoid_common_mpc/config/robot/ControllerSideSettingsFromConfig.h"

#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/contact/ContactWrenchGate.h"
#include "humanoid_mpc_config/contact_wrench_gate_config.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid {
namespace {

/** OK for a time of 0 or more; InvalidArgument naming the field of contact_wrench_gate otherwise (NaN included). */
absl::Status nonNegativeTime(absl::string_view field, double value) {
  // Negated, so that NaN is refused too.
  if (!(value >= 0.0)) {
    return absl::InvalidArgumentError(absl::StrCat("contact_wrench_gate.", field, " is ", value, " [s]; it must be a non-negative number"));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<ContactWrenchGate::Config> contactWrenchGateFromConfig(const mpc_config::ContactWrenchGateConfig& config) {
  RETURN_IF_ERROR(nonNegativeTime("debounce_time", config.debounce_time));
  RETURN_IF_ERROR(nonNegativeTime("ramp_time", config.ramp_time));
  ContactWrenchGate::Config gate;
  gate.debounceTime = config.debounce_time;
  gate.rampTime = config.ramp_time;
  // The gate's own check, which refuses NaN too: ContactWrenchGate::setConfig() takes only a configuration it accepts.
  absl::Status valid = ContactWrenchGate::validateConfig(gate);
  if (!valid.ok()) {
    return valid;
  }
  return gate;
}

ControllerSideConfig controllerSideSettingsFromConfig(const mpc_config::TaskFile& task) {
  ControllerSideConfig settings;
  settings.contactEstimator = task.contact_estimator;
  if (task.contact_wrench_gate.has_value()) {
    const absl::StatusOr<ContactWrenchGate::Config> gate = contactWrenchGateFromConfig(*task.contact_wrench_gate);
    if (gate.ok()) {
      settings.contactWrenchGate = *gate;
    } else {
      settings.problems.emplace_back(gate.status().message());
    }
  }
  return settings;
}

std::optional<ContactWrenchGate::Config> wholeFileContactWrenchGate(const ControllerSideConfig& settings) {
  if (settings.contactWrenchGate.has_value()) {
    return settings.contactWrenchGate;
  }
  // No gate: either the file has no contact_wrench_gate block, which is the instantaneous gate as at start-up, or its
  // block was refused, which keeps the gate in use.
  if (settings.problems.empty()) {
    return ContactWrenchGate::Config{};
  }
  return std::nullopt;
}

}  // namespace ocs2::humanoid
