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

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * The planar motion of the base: the velocity of its origin in its heading
 * frame and the rate of its heading. What a velocity command's reference starts
 * from and what the gait switches compare the command with (design defects D1,
 * D1b, D2), in the frame the command is given in: the heading frame is the
 * world frame turned about its z axis by the twist heading psi_h
 * (MpcRobotModelBase::getYawBaseRtWorld()), so (forward, lateral) are the
 * command's (v_x, v_y) axes. Passive data.
 */
struct BasePlanarMotion {
  /** [m/s] (forward, lateral): R_z(psi_h)^T (dp_x, dp_y) / dt of the base
   * origin. */
  vector2_t linvelBaseRtWorldEwrtHeadingXy = vector2_t::Zero();
  /** [rad/s] d psi_h / dt, the rate of the twist heading. */
  scalar_t rateYawBaseRtWorld = 0.0;
};

}  // namespace ocs2::humanoid
