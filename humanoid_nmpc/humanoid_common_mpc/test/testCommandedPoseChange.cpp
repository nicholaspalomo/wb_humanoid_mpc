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

#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

/*
 * The base the reference manager predicts for a touch-down (SwitchedModelReferenceManager::commandedPoseChange(), which
 * predictedBaseAt() carries the measured base forward with): on the arc of a constant body twist, against the closed
 * form int_0^T R_z(omega s) ds v_0 = [[sin, -(1 - cos)], [1 - cos, sin]] v_0 / omega in its half-angle form; the
 * trapezoid, bit for bit, where the command does not turn; and for a command that ramps its body velocity and its yaw
 * rate, off a Simpson quadrature of the motion by the documented leading term, the remainder third order. It was the
 * trapezoid of the two world velocities, short of the arc by dt |v| (sinc(theta / 2) - cos(theta / 2)). Builds no model.
 */

namespace ocs2::humanoid {
namespace {

using PlanarCommandAt = SwitchedModelReferenceManager::PlanarCommandAt;
using PlanarPoseChange = SwitchedModelReferenceManager::PlanarPoseChange;

/** R_z(angle) v. */
vector2_t rotated(scalar_t angle, const vector2_t& vector) {
  return vector2_t(std::cos(angle) * vector(0) - std::sin(angle) * vector(1), std::sin(angle) * vector(0) + std::cos(angle) * vector(1));
}

/** J v, J = R_z(pi / 2). */
vector2_t quarterTurn(const vector2_t& vector) {
  return vector2_t(-vector(1), vector(0));
}

TEST(CommandedPoseChange, IsTheArcOfAConstantBodyTwistInClosedForm) {
  for (const vector2_t& bodyVelocity : std::vector<vector2_t>{vector2_t(0.4, 0.1), vector2_t(-0.2, 0.3)}) {
    for (const scalar_t yawRate : {0.8, -1.3, 1.0e-9, 2.0}) {
      for (const scalar_t startHeading : {0.0, 2.5}) {
        for (const scalar_t duration : {0.3, 1.38}) {
          SCOPED_TRACE(testing::Message() << "v " << bodyVelocity.transpose() << ", w " << yawRate << ", heading " << startHeading
                                          << ", dt " << duration);
          // The command turns its world velocity with the heading: v_1 = R_z(w dt) v_0.
          const vector2_t startVelocity = rotated(startHeading, bodyVelocity);
          const vector2_t endVelocity = rotated(startHeading + yawRate * duration, bodyVelocity);
          const PlanarPoseChange change = SwitchedModelReferenceManager::commandedPoseChange(
              PlanarCommandAt{.velocityWorld = startVelocity, .yawRate = yawRate},
              PlanarCommandAt{.velocityWorld = endVelocity, .yawRate = yawRate}, duration);
          const scalar_t turn = yawRate * duration;
          const scalar_t versine = 2.0 * std::sin(0.5 * turn) * std::sin(0.5 * turn);
          const vector2_t arc = vector2_t(std::sin(turn) * startVelocity(0) - versine * startVelocity(1),
                                          versine * startVelocity(0) + std::sin(turn) * startVelocity(1)) /
                                yawRate;
          EXPECT_LT((change.displacement - arc).cwiseAbs().maxCoeff(), 1.0e-14);
          EXPECT_NEAR(change.yawChange, turn, 1.0e-15);
          // The trapezoid of the two world velocities it replaces, dt cos(theta / 2) R_z(theta / 2) v_0, fell short by
          // dt |v| (sinc(theta / 2) - cos(theta / 2)).
          const vector2_t trapezoid = 0.5 * (startVelocity + endVelocity) * duration;
          const scalar_t half = 0.5 * turn;
          const scalar_t shortfall = duration * bodyVelocity.norm() * (std::sin(half) / half - std::cos(half));
          EXPECT_NEAR((arc - trapezoid).norm(), shortfall, 1.0e-12);
        }
      }
    }
  }
}

TEST(CommandedPoseChange, WithoutATurnIsTheTrapezoidBitForBit) {
  // A world velocity ramping from v_0 to v_1 moves the base by their mean times dt, exactly.
  const vector2_t startVelocity(0.31, -0.12);
  const vector2_t endVelocity(-0.07, 0.43);
  const scalar_t duration = 0.77;
  const PlanarPoseChange change =
      SwitchedModelReferenceManager::commandedPoseChange(PlanarCommandAt{.velocityWorld = startVelocity, .yawRate = 0.0},
                                                         PlanarCommandAt{.velocityWorld = endVelocity, .yawRate = 0.0}, duration);
  EXPECT_EQ(change.displacement, vector2_t(0.5 * (startVelocity + endVelocity) * duration));
  EXPECT_EQ(change.yawChange, 0.0);
  EXPECT_EQ(SwitchedModelReferenceManager::commandedPoseChange(PlanarCommandAt{.velocityWorld = startVelocity, .yawRate = 0.4},
                                                               PlanarCommandAt{.velocityWorld = endVelocity, .yawRate = 0.4},
                                                               /*duration=*/0.0)
                .displacement,
            vector2_t::Zero());
}

/**
 * p(T) - p(0) of a command whose body velocity b and yaw rate w ramp linearly from (b0, w0) to (b1, w1) over T, from the
 * heading psi0: dp/ds = R_z(psi(s)) b(s), psi(s) = psi0 + w0 s + (w1 - w0) s^2 / (2 T); by Simpson's rule, 4000 steps.
 */
vector2_t rampingCommandDisplacement(scalar_t startHeading,
                                     const vector2_t& startBodyVelocity,
                                     const vector2_t& endBodyVelocity,
                                     scalar_t startYawRate,
                                     scalar_t endYawRate,
                                     scalar_t duration) {
  constexpr int kSteps = 4000;
  const scalar_t step = duration / kSteps;
  vector2_t sum = vector2_t::Zero();
  for (int i = 0; i <= kSteps; ++i) {
    const scalar_t s = i * step;
    const scalar_t heading = startHeading + startYawRate * s + (endYawRate - startYawRate) * s * s / (2.0 * duration);
    const vector2_t bodyVelocity = startBodyVelocity + (endBodyVelocity - startBodyVelocity) * (s / duration);
    const scalar_t weight = (i == 0 || i == kSteps) ? 1.0 : (i % 2 == 1 ? 4.0 : 2.0);
    sum += weight * rotated(heading, bodyVelocity);
  }
  return sum * step / 3.0;
}

TEST(CommandedPoseChange, ARampingCommandIsOffByTheDocumentedLeadingTerm) {
  // The body velocity and the yaw rate both ramp: the error is (dt^2 / 12) J (w_bar dv - dw v_bar) + O(dt^3), the
  // remainder an eighth each time dt halves.
  constexpr scalar_t kStartHeading = 0.4;
  const vector2_t startBodyVelocity(0.3, 0.05);
  const vector2_t endBodyVelocity(0.5, -0.1);
  constexpr scalar_t kStartYawRate = 0.6;
  constexpr scalar_t kEndYawRate = 1.0;
  std::vector<scalar_t> remainders;
  for (const scalar_t duration : {0.4, 0.2, 0.1}) {
    SCOPED_TRACE(duration);
    const scalar_t turn = 0.5 * (kStartYawRate + kEndYawRate) * duration;
    const PlanarCommandAt start{.velocityWorld = rotated(kStartHeading, startBodyVelocity), .yawRate = kStartYawRate};
    const PlanarCommandAt end{.velocityWorld = rotated(kStartHeading + turn, endBodyVelocity), .yawRate = kEndYawRate};
    const PlanarPoseChange change = SwitchedModelReferenceManager::commandedPoseChange(start, end, duration);
    EXPECT_NEAR(change.yawChange, turn, 1.0e-15) << "the trapezoid of a yaw rate linear in time is exact";
    const vector2_t error =
        rampingCommandDisplacement(kStartHeading, startBodyVelocity, endBodyVelocity, kStartYawRate, kEndYawRate, duration) -
        change.displacement;
    const vector2_t endInTurningFrame = rotated(-turn, end.velocityWorld);
    const vector2_t velocityChange = endInTurningFrame - start.velocityWorld;
    const vector2_t meanVelocity = 0.5 * (start.velocityWorld + endInTurningFrame);
    const vector2_t leading =
        (duration * duration / 12.0) * quarterTurn(turn / duration * velocityChange - (kEndYawRate - kStartYawRate) * meanVelocity);
    EXPECT_LT((error - leading).norm(), 0.2 * leading.norm()) << error.transpose() << " against " << leading.transpose();
    remainders.push_back((error - leading).norm());
  }
  for (size_t i = 1; i < remainders.size(); ++i) EXPECT_NEAR(remainders[i - 1] / remainders[i], 8.0, 0.5) << i;
}

}  // namespace
}  // namespace ocs2::humanoid
