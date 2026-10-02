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

#include <array>
#include <cstdint>
#include <string>

#include "absl/strings/string_view.h"

namespace ocs2::humanoid {

/**
 * What an MRT joint controller (CentroidalMpcMrtJointController, WBMpcMrtJointController) reports from its control
 * thread: computeJointControlAction() and the setters the realtime loop calls. The control thread never logs - a log
 * line formats into a string and writes to a file descriptor, either of which can allocate or block - it posts one of
 * these to its ControllerEventSink instead, a few plain values, and formatControllerEvent() turns it into the line the
 * controller used to log.
 */
enum class ControllerEventCode : std::uint8_t {
  /** The policy diverged from the measured state; the solver is reset. values[0]: the largest joint error [rad]. */
  kPolicyDiverged,
  /** The observation clock ran backwards; the MPC is reset. values: the rewind [s] and the new time [s]. */
  kClockRewind,
  /** SAFETY was entered: the measured posture is held and the gains decay. values[0]: the time constant [s]. */
  kSafetyEntered,
  /** The SAFETY decay has reached zero gains. */
  kSafetyDecayComplete,
  /** The touch-down shaping of the contact wrenches changed. values: debounceTime and rampTime [s]. */
  kContactWrenchGateChanged,
  /** The measured contact state now comes from another contact estimator. text: its name. */
  kContactEstimatorChanged,
  /** WB_MPC without a first policy: the weight-compensating action holds the robot until one arrives. */
  kNoPolicyWeightCompensation,
};

/** One report of a controller: plain data, so that posting it copies a few words and allocates nothing. */
struct ControllerEvent {
  ControllerEventCode code = ControllerEventCode::kPolicyDiverged;
  /** The class that reports it, e.g. "CentroidalMpcMrtJointController": a string literal, never owned. */
  const char* controller = "";
  std::array<double, 2> values{};
  /** A name, NUL-terminated and cut to fit. */
  std::array<char, 32> text{};
};

/** An event of `code` from `controller` with `values` and `text` (cut to fit). */
ControllerEvent makeControllerEvent(
    ControllerEventCode code, const char* controller, double value0 = 0.0, double value1 = 0.0, absl::string_view text = {});

/** The text of `event`, as a string_view of its NUL-terminated buffer. */
absl::string_view controllerEventText(const ControllerEvent& event);

/** The line the communication thread logs for `event`. */
std::string formatControllerEvent(const ControllerEvent& event);

/** Whether formatControllerEvent(event) is logged as a warning (otherwise as information). */
bool isWarningControllerEvent(const ControllerEvent& event);

}  // namespace ocs2::humanoid
