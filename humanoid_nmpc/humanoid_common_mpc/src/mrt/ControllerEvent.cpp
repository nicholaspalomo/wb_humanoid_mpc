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

#include "humanoid_common_mpc/mrt/ControllerEvent.h"

#include <algorithm>
#include <cstring>
#include <string>

#include "absl/base/nullability.h"
#include "absl/strings/str_cat.h"

namespace ocs2::humanoid {

ControllerEvent makeControllerEvent(
    ControllerEventCode code, const char* absl_nonnull controller, double value0, double value1, absl::string_view text) {
  ControllerEvent event;
  event.code = code;
  event.controller = controller;
  event.values = {value0, value1};
  const size_t length = std::min(text.size(), event.text.size() - 1);
  std::memcpy(event.text.data(), text.data(), length);
  event.text[length] = '\0';
  return event;
}

absl::string_view controllerEventText(const ControllerEvent& event) {
  const size_t length = ::strnlen(event.text.data(), event.text.size());
  return absl::string_view(event.text.data(), length);
}

std::string formatControllerEvent(const ControllerEvent& event) {
  const std::string prefix = absl::StrCat("[", event.controller, "] ");
  switch (event.code) {
    case ControllerEventCode::kPolicyDiverged:
      return absl::StrCat(prefix, "MPC policy diverged from actual state (max joint error=", event.values[0],
                          " rad). Resetting the MPC solver and clamping joint targets until a new policy is in use.");
    case ControllerEventCode::kClockRewind:
      return absl::StrCat(prefix, "The observation time went backwards by ", event.values[0], " s, to ", event.values[1],
                          " s. The MPC is reset, and WB_MPC holds the robot with the JOINT_PD action until a policy planned on "
                          "the new clock is in use.");
    case ControllerEventCode::kSafetyEntered:
      return absl::StrCat(prefix, "SAFETY mode entered: holding the measured posture and decaying the joint PD gains to zero with a ",
                          event.values[0], " s time constant (safety_decay_time_constant).");
    case ControllerEventCode::kSafetyDecayComplete:
      return absl::StrCat(prefix, "SAFETY decay complete: commanding zero torque on all joints.");
    case ControllerEventCode::kContactWrenchGateChanged:
      return absl::StrCat(prefix, "contact wrench gate: debounce_time=", event.values[0], " s, ramp_time=", event.values[1], " s.");
    case ControllerEventCode::kContactEstimatorChanged:
      return absl::StrCat(prefix, "measured contact state from ", controllerEventText(event), ".");
    case ControllerEventCode::kNoPolicyWeightCompensation:
      return absl::StrCat(prefix,
                          "WB_MPC without an MPC policy yet: applying the weight-compensating torques until the first one "
                          "arrives.");
    case ControllerEventCode::kContactEstimateRefused:
      return absl::StrCat(prefix, "the contact estimator '", controllerEventText(event), "' reported ", event.values[0],
                          " contact flags, expected ", event.values[1],
                          ": its estimates are refused and the measured contact state stays the last one taken.");
    case ControllerEventCode::kContactWrenchGateRefused:
      return absl::StrCat(prefix, "contact wrench gate refused (debounce_time=", event.values[0], " s, ramp_time=", event.values[1],
                          " s must be non-negative); the gate in use is kept.");
  }
  return absl::StrCat(prefix, "controller event ", static_cast<int>(event.code));
}

bool isWarningControllerEvent(const ControllerEvent& event) {
  switch (event.code) {
    case ControllerEventCode::kPolicyDiverged:
    case ControllerEventCode::kClockRewind:
    case ControllerEventCode::kSafetyEntered:
    case ControllerEventCode::kSafetyDecayComplete:
    case ControllerEventCode::kContactEstimateRefused:
    case ControllerEventCode::kContactWrenchGateRefused:
      return true;
    case ControllerEventCode::kContactWrenchGateChanged:
    case ControllerEventCode::kContactEstimatorChanged:
    case ControllerEventCode::kNoPolicyWeightCompensation:
      return false;
  }
  return false;
}

}  // namespace ocs2::humanoid
