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

#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"

#include "humanoid_common_mpc/contact/ContactWrenchGate.h"
#include "humanoid_mpc_config/contact_wrench_gate_config.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid {

/**
 * What the robot process applies to its MRT joint controller rather than to the MPC, from a task file or from the task
 * file the tuning GUI publishes: settings that reach the controller on the realtime thread, the only thread that uses
 * the estimator and the gate. The robot process (its operator mailbox and its task file watcher) and the lockstep
 * driver read them with the same function, controllerSideSettingsFromConfig().
 */
struct ControllerSideConfig {
  /** contact_estimator: the estimator of the measured contact state, by ContactEstimatorRegistry name. */
  std::string contactEstimator;
  /**
   * contact_wrench_gate: the touch-down shaping of the planned contact wrenches. nullopt when the file has no such
   * block, which is the instantaneous gate (ContactWrenchGate::Config{}), and when the block is invalid: it is then
   * not applied and `problems` says why.
   */
  std::optional<ContactWrenchGate::Config> contactWrenchGate;
  /** What was found wrong, each naming its field: a running consumer keeps what it has, a start-up refuses the file. */
  std::vector<std::string> problems;
};

/**
 * The gate configuration of a contact_wrench_gate block; InvalidArgument naming the field and its unit for a negative or
 * NaN time ("contact_wrench_gate.debounce_time is -1 [s]; it must be a non-negative number").
 */
absl::StatusOr<ContactWrenchGate::Config> contactWrenchGateFromConfig(const mpc_config::ContactWrenchGateConfig& config);

/**
 * The controller-side settings of `task`: its contact estimator and, when it has a valid contact_wrench_gate block, its
 * gate. An invalid gate is left out and reported in `problems`, and the estimator still applies. The robot process
 * and the lockstep driver read the settings through it. Allocates; for the communication thread.
 */
ControllerSideConfig controllerSideSettingsFromConfig(const mpc_config::TaskFile& task);

/**
 * The gate a task file sets when it is applied as a whole file (humanoid_nmpc/humanoid_mpc_config/README.md, "Live
 * updates"): its block's gate; the instantaneous gate (ContactWrenchGate::Config{}) when it has no block, as at start-up;
 * nullopt when its block was refused (one of `settings.problems`), which keeps the gate in use. The robot process's
 * operator mailbox and the validation driver apply a reloaded or published task file through it.
 */
std::optional<ContactWrenchGate::Config> wholeFileContactWrenchGate(const ControllerSideConfig& settings);

}  // namespace ocs2::humanoid
