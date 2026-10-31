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

#include <algorithm>
#include <cmath>
#include <random>

#include "gtest/gtest.h"

#include "humanoid_common_mpc/mrt/JointActionBlend.h"
#include "robot_model/RobotJointAction.h"

/*
 * The ramp of an MRT joint controller from a held action into the MPC's
 * (blendJointActionLinearInTorque()): the torque the drive makes of the blended
 * action, kp (q_des - q) + kd (qd_des - qd) + ff, is the linear blend of the
 * two torques at every measured q and qd and every a in [0, 1], for random
 * actions and for the holds the controllers ramp from (GRAVITY_COMP: no
 * position gain; JOINT_PD), and the targets stay between the two. Blending
 * every field on its own, as the centroidal controller did, misses it by a (1 -
 * a) (kp_to - kp_from) (q_to - q_from): the positive control.
 */

namespace ocs2::humanoid {
namespace {

using ::robot::model::JointAction;

constexpr int kSamples = 2000;

JointAction randomAction(std::mt19937& generator,
                         bool zeroPositionGain,
                         bool zeroVelocityGain) {
  std::uniform_real_distribution<scalar_t> gain(0.0, 400.0);
  std::uniform_real_distribution<scalar_t> target(-2.0, 2.0);
  std::uniform_real_distribution<scalar_t> jointTorque(-100.0, 100.0);
  JointAction action;
  action.q_des = target(generator);
  action.qd_des = target(generator);
  action.kp = zeroPositionGain ? 0.0 : gain(generator);
  action.kd = zeroVelocityGain ? 0.0 : gain(generator) * 0.1;
  action.feed_forward_effort = jointTorque(generator);
  return action;
}

JointAction fieldwiseBlend(const JointAction& from,
                           scalar_t a,
                           const JointAction& to) {
  JointAction blended;
  blended.q_des = (1.0 - a) * from.q_des + a * to.q_des;
  blended.qd_des = (1.0 - a) * from.qd_des + a * to.qd_des;
  blended.kp = (1.0 - a) * from.kp + a * to.kp;
  blended.kd = (1.0 - a) * from.kd + a * to.kd;
  blended.feed_forward_effort =
      (1.0 - a) * from.feed_forward_effort + a * to.feed_forward_effort;
  return blended;
}

TEST(JointActionBlend,
     TheTorqueIsTheLinearBlendOfTheTwoTorquesAtEveryMeasurement) {
  std::mt19937 generator(/*sd=*/17);
  std::uniform_real_distribution<scalar_t> unit(0.0, 1.0);
  std::uniform_real_distribution<scalar_t> measurement(-3.0, 3.0);
  scalar_t largestFieldwiseError = 0.0;
  for (int sample = 0; sample < kSamples; ++sample) {
    // Every fourth hold has no position gain (GRAVITY_COMP), every eighth no
    // gain at all, every fifth target none.
    const JointAction from =
        randomAction(generator, /*zeroPositionGain=*/sample % 4 == 0,
                     /*zeroVelocityGain=*/sample % 8 == 0);
    const JointAction to =
        randomAction(generator, /*zeroPositionGain=*/sample % 5 == 0,
                     /*zeroVelocityGain=*/sample % 10 == 0);
    const scalar_t a =
        sample % 7 == 0 ? static_cast<scalar_t>(sample % 2) : unit(generator);
    JointAction blended = to;
    blendJointActionLinearInTorque(from, a, blended);
    for (int probe = 0; probe < 3; ++probe) {
      const scalar_t q = measurement(generator);
      const scalar_t qd = measurement(generator);
      const scalar_t expected =
          (1.0 - a) * from.getTotalFeedbackJointTorque(q, qd) +
          a * to.getTotalFeedbackJointTorque(q, qd);
      const scalar_t scale =
          std::max({1.0, std::abs(from.getTotalFeedbackJointTorque(q, qd)),
                    std::abs(to.getTotalFeedbackJointTorque(q, qd)),
                    from.kp * std::abs(q), to.kp * std::abs(q)});
      EXPECT_NEAR(blended.getTotalFeedbackJointTorque(q, qd), expected,
                  1.0e-12 * scale)
          << "sample " << sample << ", a = " << a;
      largestFieldwiseError = std::max(
          largestFieldwiseError,
          std::abs(
              fieldwiseBlend(from, a, to).getTotalFeedbackJointTorque(q, qd) -
              expected));
    }
    // The gains blend linearly, and the targets stay between the two.
    EXPECT_DOUBLE_EQ(blended.kp, (1.0 - a) * from.kp + a * to.kp);
    EXPECT_DOUBLE_EQ(blended.kd, (1.0 - a) * from.kd + a * to.kd);
    const scalar_t slack = 1.0e-12;
    EXPECT_GE(blended.q_des, std::min(from.q_des, to.q_des) - slack);
    EXPECT_LE(blended.q_des, std::max(from.q_des, to.q_des) + slack);
    EXPECT_GE(blended.qd_des, std::min(from.qd_des, to.qd_des) - slack);
    EXPECT_LE(blended.qd_des, std::max(from.qd_des, to.qd_des) + slack);
  }
  EXPECT_GT(largestFieldwiseError, 1.0)
      << "the positive control: a field-wise blend is not linear in the torque";
}

TEST(JointActionBlend, TheEndsOfTheRampAreTheTwoActions) {
  std::mt19937 generator(/*sd=*/19);
  const JointAction from = randomAction(generator, /*zeroPositionGain=*/false,
                                        /*zeroVelocityGain=*/false);
  const JointAction to = randomAction(generator, /*zeroPositionGain=*/false,
                                      /*zeroVelocityGain=*/false);
  JointAction atStart = to;
  blendJointActionLinearInTorque(from, /*a=*/0.0, atStart);
  EXPECT_DOUBLE_EQ(atStart.kp, from.kp);
  EXPECT_NEAR(atStart.q_des, from.q_des, 1.0e-15);
  EXPECT_DOUBLE_EQ(atStart.feed_forward_effort, from.feed_forward_effort);
  JointAction atEnd = to;
  blendJointActionLinearInTorque(from, /*a=*/1.0, atEnd);
  EXPECT_DOUBLE_EQ(atEnd.kp, to.kp);
  EXPECT_NEAR(atEnd.q_des, to.q_des, 1.0e-15);
  EXPECT_DOUBLE_EQ(atEnd.feed_forward_effort, to.feed_forward_effort);
}

}  // namespace
}  // namespace ocs2::humanoid
