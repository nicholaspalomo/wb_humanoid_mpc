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

#include "humanoid_mpc_msgs/dodgeball_throw.nproto.h"
#include "mujoco_sim_interface/MujocoSimInterface.h"

namespace ocs2::humanoid {

/**
 * The throw the simulator acts on from a DodgeballThrow of operator/dodgeball_throw, the typed payload of the GUI's
 * Dodgeball tab.
 *
 * The geometry was computed by the GUI (remote_control/tk_app/dodgeball.py) and is not recomputed here: this reads the
 * spawn offset, the launch velocity and the flight time it worked out, in the robot's yaw frame, and the ball's mass;
 * the operator's azimuth, elevation, distance and speed and the documentation fields are not read. Every value the
 * simulator uses is required, including the mass, so that a throw that left one out is not filled in with a default
 * nobody chose. A mass outside the slider's range is not refused here; the simulator clamps it.
 *
 * @return InvalidArgument naming the field when a required one is absent or not finite (a NaN launch velocity makes
 *         MuJoCo reset the whole simulation), when the flight time is negative, or when the mass is not positive.
 */
absl::StatusOr<robot::mujoco_sim_interface::MujocoSimInterface::DodgeballThrow> dodgeballThrowFromMessage(
    const msgs::DodgeballThrow& message);

}  // namespace ocs2::humanoid
