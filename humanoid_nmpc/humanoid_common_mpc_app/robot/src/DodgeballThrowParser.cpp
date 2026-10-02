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

#include "humanoid_common_mpc_app/robot/DodgeballThrowParser.h"

#include <cmath>
#include <exception>
#include <string>

#include <yaml-cpp/yaml.h>

#include "absl/strings/str_cat.h"

namespace ocs2::humanoid {

namespace {

using DodgeballThrow = robot::mujoco_sim_interface::MujocoSimInterface::DodgeballThrow;

absl::StatusOr<double> readNumber(const YAML::Node& ball, absl::string_view key) {
  const YAML::Node node = ball[std::string(key)];
  if (!node || !node.IsScalar()) {
    return absl::InvalidArgumentError(absl::StrCat("Dodgeball throw ignored: the payload has no number at 'dodgeball.", key, "'."));
  }
  double value = 0.0;
  try {
    value = node.as<double>();
  } catch (const std::exception& error) {
    return absl::InvalidArgumentError(absl::StrCat("Dodgeball throw ignored: 'dodgeball.", key, "' is not a number: ", error.what()));
  }
  if (!std::isfinite(value)) {
    return absl::InvalidArgumentError(absl::StrCat("Dodgeball throw ignored: 'dodgeball.", key, "' is ", value, ", not a finite number."));
  }
  return value;
}

absl::Status readVector(const YAML::Node& ball, absl::string_view key, double (&out)[3]) {
  const YAML::Node node = ball[std::string(key)];
  if (!node || !node.IsSequence() || node.size() != 3) {
    return absl::InvalidArgumentError(absl::StrCat("Dodgeball throw ignored: 'dodgeball.", key, "' must be a list of three numbers."));
  }
  for (size_t axis = 0; axis < 3; ++axis) {
    try {
      out[axis] = node[axis].as<double>();
    } catch (const std::exception& error) {
      return absl::InvalidArgumentError(
          absl::StrCat("Dodgeball throw ignored: 'dodgeball.", key, "[", axis, "]' is not a number: ", error.what()));
    }
    if (!std::isfinite(out[axis])) {
      return absl::InvalidArgumentError(
          absl::StrCat("Dodgeball throw ignored: 'dodgeball.", key, "[", axis, "]' is ", out[axis], ", not a finite number."));
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<DodgeballThrow> parseDodgeballThrow(absl::string_view payload) {
  YAML::Node root;
  try {
    root = YAML::Load(std::string(payload));
  } catch (const std::exception& error) {
    return absl::InvalidArgumentError(absl::StrCat("Dodgeball throw ignored: the payload could not be read as YAML: ", error.what()));
  }
  // LINT.IfChange(dodgeball_payload_keys)
  const YAML::Node ball = root.IsMap() ? root["dodgeball"] : YAML::Node();
  if (!ball || !ball.IsMap()) return absl::InvalidArgumentError("Dodgeball throw ignored: the payload has no 'dodgeball' block.");

  DodgeballThrow command;
  absl::Status status = readVector(ball, "spawnOffset", command.spawnOffset);
  if (!status.ok()) return status;
  status = readVector(ball, "launchVelocity", command.launchVelocity);
  if (!status.ok()) return status;
  const absl::StatusOr<double> flightTime = readNumber(ball, "flightTime");
  if (!flightTime.ok()) return flightTime.status();
  const absl::StatusOr<double> mass = readNumber(ball, "mass");
  if (!mass.ok()) return mass.status();
  // LINT.ThenChange(//humanoid_nmpc/remote_control/remote_control/tk_app/dodgeball.py:dodgeball_payload_keys)

  if (*flightTime < 0.0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Dodgeball throw ignored: 'dodgeball.flightTime' is ", *flightTime, " s, before the throw."));
  }
  if (!(*mass > 0.0)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Dodgeball throw ignored: 'dodgeball.mass' is ", *mass, " kg; a ball needs a positive mass."));
  }
  command.flightTime = *flightTime;
  command.mass = *mass;
  return command;
}

}  // namespace ocs2::humanoid
