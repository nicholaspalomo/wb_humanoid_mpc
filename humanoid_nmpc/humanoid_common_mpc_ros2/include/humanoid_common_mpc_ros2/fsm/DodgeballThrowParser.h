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

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "mujoco_sim_interface/MujocoSimInterface.h"

namespace ocs2::humanoid {

/**
 * Reads the payload the GUI's Dodgeball tab publishes on /humanoid/dodgeball_throw into a throw the simulator can act
 * on, or explains why it cannot.
 *
 * The geometry was computed by the GUI (remote_control/tk_app/dodgeball.py) and is not recomputed here: this reads
 * the spawn offset, the launch velocity and the flight time it already worked out, in the robot's yaw frame, and the
 * ball's mass. The operator's azimuth, elevation, distance and speed travel in the same payload for the benefit of a
 * recorded bag, and nothing reads them.
 *
 * Every value the simulator uses is REQUIRED, including the mass: a payload that left one out would otherwise be
 * filled in with a default that nobody chose. Rejected, with a message naming the key: a payload that is not YAML, no
 * `dodgeball` block, a vector that is not three numbers, a missing or non-finite value (a NaN typed into the GUI
 * becomes `.nan` in YAML, and a NaN launch velocity makes MuJoCo reset the whole simulation), a negative flight time,
 * or a mass that is not positive. A mass outside the slider's range is NOT rejected here; the simulator clamps it.
 */
absl::StatusOr<robot::mujoco_sim_interface::MujocoSimInterface::DodgeballThrow> parseDodgeballThrow(absl::string_view payload);

}  // namespace ocs2::humanoid
