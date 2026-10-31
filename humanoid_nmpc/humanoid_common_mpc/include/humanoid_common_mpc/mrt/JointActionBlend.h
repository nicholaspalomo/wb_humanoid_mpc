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
#include "robot_model/RobotJointAction.h"

namespace ocs2::humanoid {

/**
 * Moves the joint action `to` the fraction 1 - a back towards `from`, so that
 * the torque the drive makes of it, tau(q, qd) = kp (q_des - q) + kd (qd_des -
 * qd) + ff        (robot::model::JointAction::getTotalFeedbackJointTorque()),
 * is the linear blend (1 - a) tau_from(q, qd) + a tau_to(q, qd) at EVERY
 * measured q and qd. Blending each field on its own does not do that: kp q_des
 * is a product of two blended fields, and its torque misses the linear blend by
 * -a (1 - a) (kp_to - kp_from) (q_to - q_from), a quarter of that product at a
 * = 1/2 (likewise for kd and qd_des). Here the gains blend linearly and the
 * targets are the gain-weighted means kp = (1 - a) kp_from + a kp_to,   q_des =
 * ((1 - a) kp_from q_from + a kp_to q_to) / kp, which lie between the two
 * targets for non-negative gains and a in [0, 1]. Where a blended gain is zero
 * its target is blended linearly, and the term's torque, which is then the
 * constant (1 - a) kp_from q_from + a kp_to q_to (zero for non-negative gains),
 * is carried by the feedforward. `from` and `to` may not be the same object.
 * Allocation-free.
 */
inline void blendJointActionLinearInTorque(
    const ::robot::model::JointAction& from,
    scalar_t a,
    ::robot::model::JointAction& to) {
  const scalar_t b = 1.0 - a;
  const scalar_t kp = b * from.kp + a * to.kp;
  const scalar_t kd = b * from.kd + a * to.kd;
  const scalar_t kpTimesTarget =
      b * from.kp * from.q_des + a * to.kp * to.q_des;
  const scalar_t kdTimesTarget =
      b * from.kd * from.qd_des + a * to.kd * to.qd_des;
  scalar_t feedforward =
      b * from.feed_forward_effort + a * to.feed_forward_effort;
  if (kp != 0.0) {
    to.q_des = kpTimesTarget / kp;
  } else {
    to.q_des = b * from.q_des + a * to.q_des;
    feedforward += kpTimesTarget;
  }
  if (kd != 0.0) {
    to.qd_des = kdTimesTarget / kd;
  } else {
    to.qd_des = b * from.qd_des + a * to.qd_des;
    feedforward += kdTimesTarget;
  }
  to.kp = kp;
  to.kd = kd;
  to.feed_forward_effort = feedforward;
}

}  // namespace ocs2::humanoid
