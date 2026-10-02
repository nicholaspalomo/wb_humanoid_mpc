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
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/contact/ContactWrenchGate.h"

namespace ocs2::humanoid {

/**
 * The keys of a task file, or of the YAML the tuning GUI publishes on operator/mpc_parameters, that the robot process
 * applies to its MRT joint controller rather than the MPC: they reach the controller on the realtime thread, the only
 * thread that uses the estimator and the gate. The MPC node reads the same documents for its own keys.
 */
struct ControllerSideSettings {
  /** `contactEstimator`: the name of the estimator of the measured contact state (ContactEstimatorRegistry). */
  std::optional<std::string> contactEstimator;
  /**
   * `contact_wrench_gate` (debounceTime, rampTime): the touch-down shaping of the planned contact wrenches. A key the
   * block does not carry keeps the library default, as at start-up; a block with a value that does not parse, or a
   * negative one, is not applied (the running gate is kept) and says why in `problems`.
   */
  std::optional<ContactWrenchGate::Config> contactWrenchGate;
  /** What was found wrong and not applied, for the log. */
  std::vector<std::string> problems;

  bool empty() const { return !contactEstimator.has_value() && !contactWrenchGate.has_value(); }
};

/**
 * The controller-side settings of a YAML document, exactly as MpcParameterUpdaterModule::recordControllerSettings()
 * reads them for the in-process sims. A document that is not YAML is InvalidArgument; a document without either key is
 * an empty result. Allocates; for the communication thread.
 */
absl::StatusOr<ControllerSideSettings> parseControllerSideSettings(absl::string_view yamlText);

/** parseControllerSideSettings() of a file; a file that cannot be read is NotFound. */
absl::StatusOr<ControllerSideSettings> loadControllerSideSettings(const std::string& yamlFile);

}  // namespace ocs2::humanoid
