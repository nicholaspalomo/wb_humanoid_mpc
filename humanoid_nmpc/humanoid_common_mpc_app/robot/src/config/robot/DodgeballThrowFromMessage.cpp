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

#include "humanoid_common_mpc_app/robot/config/robot/DodgeballThrowFromMessage.h"

#include <cmath>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_mpc_msgs/dodgeball_throw.nproto.h"
#include "humanoid_mpc_msgs/vector3.nproto.h"
#include "mujoco_sim_interface/MujocoSimInterface.h"

namespace ocs2::humanoid {
namespace {

using DodgeballThrow = robot::mujoco_sim_interface::MujocoSimInterface::DodgeballThrow;

absl::StatusOr<double> requiredNumber(std::optional<double> value, absl::string_view field) {
  if (!value.has_value()) {
    return absl::InvalidArgumentError(absl::StrCat("Dodgeball throw ignored: the throw has no ", field, "."));
  }
  if (!std::isfinite(*value)) {
    return absl::InvalidArgumentError(absl::StrCat("Dodgeball throw ignored: ", field, " is ", *value, ", not a finite number."));
  }
  return *value;
}

absl::Status requiredVector(std::optional<msgs::Vector3> value, absl::string_view field, double (&out)[3]) {
  if (!value.has_value()) {
    return absl::InvalidArgumentError(absl::StrCat("Dodgeball throw ignored: the throw has no ", field, "."));
  }
  const double components[3] = {value->x, value->y, value->z};
  constexpr absl::string_view kAxes[3] = {"x", "y", "z"};
  for (size_t axis = 0; axis < 3; ++axis) {
    if (!std::isfinite(components[axis])) {
      return absl::InvalidArgumentError(
          absl::StrCat("Dodgeball throw ignored: ", field, ".", kAxes[axis], " is ", components[axis], ", not a finite number."));
    }
    out[axis] = components[axis];
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<DodgeballThrow> dodgeballThrowFromMessage(const msgs::DodgeballThrow& message) {
  DodgeballThrow command;
  absl::Status status = requiredVector(message.spawn_offset, "spawn_offset", command.spawnOffset);
  if (!status.ok()) return status;
  status = requiredVector(message.launch_velocity, "launch_velocity", command.launchVelocity);
  if (!status.ok()) return status;
  const absl::StatusOr<double> flightTime = requiredNumber(message.flight_time, "flight_time");
  if (!flightTime.ok()) return flightTime.status();
  const absl::StatusOr<double> mass = requiredNumber(message.mass, "mass");
  if (!mass.ok()) return mass.status();

  if (*flightTime < 0.0) {
    return absl::InvalidArgumentError(absl::StrCat("Dodgeball throw ignored: flight_time is ", *flightTime, " s, before the throw."));
  }
  if (!(*mass > 0.0)) {
    return absl::InvalidArgumentError(absl::StrCat("Dodgeball throw ignored: mass is ", *mass, " kg; a ball needs a positive mass."));
  }
  command.flightTime = *flightTime;
  command.mass = *mass;
  return command;
}

}  // namespace ocs2::humanoid
