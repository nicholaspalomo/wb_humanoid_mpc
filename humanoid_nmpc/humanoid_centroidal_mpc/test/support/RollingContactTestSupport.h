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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <cstddef>

#include "absl/strings/string_view.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid::rolling_contact_test {

// The tests of the rolling contacts run on the LimX Tron1's shipped
// configuration (robot_models/limx_tron1): two wheels, left (contact 0, mode
// bit 2) and right (contact 1, mode bit 1).
inline constexpr size_t kLeftWheel = 0;
inline constexpr size_t kRightWheel = 1;
inline constexpr size_t kBothWheels = 3;
inline constexpr size_t kLeftWheelOnly = 2;

// [s] A schedule on which the right wheel swings from kSwingStartTime to
// kSwingEndTime, and times in its stance and in its swing.
inline constexpr scalar_t kSwingStartTime = 0.2;
inline constexpr scalar_t kSwingEndTime = 0.6;
inline constexpr scalar_t kStanceTime = 0.1;
inline constexpr scalar_t kMidSwingTime = 0.4;

/** Returns Tron1's shipped configuration; fails the calling test when it does
 * not load. */
CentroidalMpcConfig tron1Config();

/** Returns the schedule on which the right wheel swings between
 * kSwingStartTime and kSwingEndTime and both wheels stand otherwise. */
ModeSchedule rightWheelSwing();

/** Returns the schedule on which both wheels stand throughout. */
ModeSchedule bothWheelsStand();

/**
 * Makes `schedule` the reference manager's, with a standing target at the
 * initial state, the way the MPC loop prepares a solve, and points the problem
 * at that target.
 */
void prepareSolve(CentroidalMpcInterface& mpc, const ModeSchedule& schedule);

/** Returns the world position of the frame `frameName` of the model of `mpc`
 * at the generalized coordinates of `state`. */
vector3_t posWheelRtWorldEwrtWorld(CentroidalMpcInterface& mpc,
                                   const vector_t& state,
                                   absl::string_view frameName);

}  // namespace ocs2::humanoid::rolling_contact_test
