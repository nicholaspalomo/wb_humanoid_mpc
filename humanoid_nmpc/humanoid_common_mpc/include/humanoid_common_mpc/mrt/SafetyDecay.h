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

#include <algorithm>
#include <cmath>

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid::safety_decay {

/**
 * The decay law of the SAFETY mode, shared by the MRT joint controllers.
 *
 * In SAFETY the robot holds the posture it had at mode entry with a joint PD whose gains are both multiplied by
 * alpha(t) = exp(-t / tau), so the commanded torque alpha * (kp * (q_hold - q) - kd * qd) decays smoothly to nothing
 * instead of being cut in one cycle. Once alpha falls below kCutoff the joints are commanded zero gain and zero
 * feedforward, i.e. true zero torque; at the shipped tau that takes about 4 * tau.
 */

/// [s] Guards against a division by zero in alpha(t) when a task file sets the time constant to zero.
inline constexpr scalar_t kMinTimeConstant = 1e-3;

// LINT.IfChange(safety_decay_cutoff)
/// alpha below which the command becomes zero torque.
inline constexpr scalar_t kCutoff = 0.02;
// LINT.ThenChange(//humanoid_nmpc/remote_control/remote_control/humanoid_finite_state_machine.py:safety_decay_cutoff)

/**
 * alpha = exp(-elapsed / tau), snapped to 0 once it falls below kCutoff so that the mode reaches true zero torque in
 * finite time rather than only approaching it. A negative elapsed time (the clock stepped back) counts as zero, so the
 * factor never exceeds one.
 */
inline scalar_t factor(scalar_t elapsedSinceEntry, scalar_t timeConstant) {
  const scalar_t elapsed = std::max(scalar_t(0.0), elapsedSinceEntry);
  const scalar_t alpha = std::exp(-elapsed / std::max(kMinTimeConstant, timeConstant));
  return alpha < kCutoff ? scalar_t(0.0) : alpha;
}

}  // namespace ocs2::humanoid::safety_decay
