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

#include "humanoid_common_mpc/contact/ContactWrenchGate.h"

#include <algorithm>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/common/StatusMacros.h"

namespace ocs2::humanoid {

absl::StatusOr<ContactWrenchGate> ContactWrenchGate::Create(
    const Config& config) {
  RETURN_IF_ERROR(validateConfig(config));
  return ContactWrenchGate(config);
}

bool ContactWrenchGate::isValidConfig(const Config& config) {
  // Negated, so that NaN is refused too.
  return config.debounceTime >= 0.0 && config.rampTime >= 0.0;
}

absl::Status ContactWrenchGate::validateConfig(const Config& config) {
  if (!isValidConfig(config)) {
    return absl::InvalidArgumentError(
        absl::StrCat("ContactWrenchGate: the debounce time (",
                     config.debounceTime, " s) and the ramp time (",
                     config.rampTime, " s) must be non-negative"));
  }
  return absl::OkStatus();
}

bool ContactWrenchGate::setConfig(const Config& config) {
  if (!isValidConfig(config)) return false;
  config_ = config;
  return true;
}

void ContactWrenchGate::reset() {
  onsetTime_.fill(std::nullopt);
  scales_.fill(1.0);
}

const feet_array_t<scalar_t>& ContactWrenchGate::update(
    scalar_t time, const contact_flag_t& measuredContactFlags) {
  for (size_t i = 0; i < scales_.size(); ++i) {
    if (!measuredContactFlags[i]) {
      onsetTime_[i] = std::nullopt;
      scales_[i] = 0.0;
      continue;
    }
    std::optional<scalar_t>& onsetTime = onsetTime_[i];
    if (!onsetTime.has_value() || time < *onsetTime) onsetTime = time;
    const scalar_t sinceOnset = time - *onsetTime;
    if (sinceOnset < config_.debounceTime) {
      scales_[i] = 0.0;
    } else if (config_.rampTime > 0.0) {
      scales_[i] = std::clamp(
          (sinceOnset - config_.debounceTime) / config_.rampTime, 0.0, 1.0);
    } else {
      scales_[i] = 1.0;
    }
  }
  return scales_;
}

feet_array_t<vector6_t> ContactWrenchGate::apply(
    feet_array_t<vector6_t> plannedWrenchFeetEwrtWorld) const {
  for (size_t i = 0;
       i < plannedWrenchFeetEwrtWorld.size() && i < scales_.size(); ++i) {
    plannedWrenchFeetEwrtWorld[i] *= scales_[i];
  }
  return plannedWrenchFeetEwrtWorld;
}

}  // namespace ocs2::humanoid
