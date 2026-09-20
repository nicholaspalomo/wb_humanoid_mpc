/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include <ocs2_core/Types.h>

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"

namespace ocs2::humanoid {

/**
 * The standing / walking blend of arXiv:2502.15630, equations (15) - (17).
 *
 * The paper blends a static standing reference with the H-LIP walking reference so that the robot can come to a
 * complete stop instead of marching in place: with an activity
 *
 *   phi = ||[c_d; v_b]||^2_P,   alpha(phi) = tanh(rho_1 (phi - rho_2)) / 2 + 1/2,
 *
 * P normalises each command and each measured velocity by the largest value it is expected to take, so phi reaches
 * one when a single component is at its threshold. alpha is the weight of the walking reference.
 *
 * Here the same alpha decides whether the planner emits a stepping gait at all, and scales the commanded velocity it
 * plans the footholds for. That keeps the whole standing-to-walking transition in one smooth scalar law with two
 * constants, in place of a stepping trigger with a dead band and a hysteresis of its own.
 */
class HlipStandingBlend {
 public:
  explicit HlipStandingBlend(const HlipBlendParameters& parameters) : parameters_(parameters) {}

  /**
   * phi: the squared norm of the commands and the measured CENTRE-OF-MASS velocity under the threshold metric P.
   *
   * The paper writes v_b for the base velocity, and the thresholds used to be named for it, but what the planner
   * actually measures and passes in here is the centre-of-mass velocity (ContactPlannerInput::comVelocity, filled from
   * ContactPlanningReferenceManager::computeComState). The names follow the quantity rather than the paper.
   */
  scalar_t activity(const vector2_t& velocityCommand, scalar_t yawRateCommand, const vector2_t& comVelocity) const;

  /** alpha in (0, 1): 0 stands still, 1 walks the full H-LIP gait. */
  scalar_t weight(const vector2_t& velocityCommand, scalar_t yawRateCommand, const vector2_t& comVelocity) const;

  /** Whether the blend asks for a stepping gait rather than a standing one. */
  bool isWalking(const vector2_t& velocityCommand, scalar_t yawRateCommand, const vector2_t& comVelocity) const {
    return weight(velocityCommand, yawRateCommand, comVelocity) >= 0.5;
  }

  const HlipBlendParameters& getParameters() const { return parameters_; }

 private:
  HlipBlendParameters parameters_;
};

}  // namespace ocs2::humanoid
