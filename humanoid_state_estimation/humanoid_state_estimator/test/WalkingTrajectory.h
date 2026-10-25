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

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid::estimation::test_support {

/** test/data/prismatic_biped.urdf, its feet and its joints in robot joint order
 * (their names sorted). */
inline constexpr char kPrismaticBipedUrdf[] =
    "humanoid_state_estimation/humanoid_state_estimator/test/data/"
    "prismatic_biped.urdf";
inline constexpr size_t kNumFeet = 2;
inline constexpr size_t kNumJoints = 6;
std::vector<std::string> prismaticBipedJointNames();
std::vector<std::string> prismaticBipedFootFrames();

/** The world gravity the trajectory's accelerometer reading subtracts [m/s^2].
 */
inline const vector3_t kGravity(0.0, 0.0, -9.81);

/** The true state of the prismatic biped at one time, and what its sensors read
 * then. */
struct WalkingSample {
  scalar_t time = 0.0;
  vector3_t basePosition = vector3_t::Zero();
  /** World frame. */
  vector3_t baseVelocity = vector3_t::Zero();
  vector3_t baseAcceleration = vector3_t::Zero();
  /** Base to world. */
  quaternion_t orientation = quaternion_t::Identity();
  /** Base frame. */
  vector3_t angularVelocity = vector3_t::Zero();
  /** The accelerometer's specific force R^T (a - g), base frame. */
  vector3_t imuLinearAcceleration = vector3_t::Zero();
  std::array<vector3_t, kNumFeet> footPositions{};
  std::array<vector3_t, kNumFeet> footVelocities{};
  std::vector<bool> contactFlags = std::vector<bool>(kNumFeet, true);
  /** Robot joint order: left_x, left_y, left_z, right_x, right_y, right_z. */
  vector_t jointPositions = vector_t::Zero(kNumJoints);
  vector_t jointVelocities = vector_t::Zero(kNumJoints);
};

/**
 * A walk of the prismatic biped, in closed form: the base moves forward at 0.4
 * m/s and sways sideways and up and down, and turns about all three axes (Z-Y-X
 * angles that oscillate); the feet alternate, each standing still on the ground
 * at z = 0 and swinging 0.3 of a 0.8 s gait period to a foothold ahead, with a
 * short double support between swings. The joints are q = R^T (c - p) - o and
 * q_dot = -omega x R^T (c - p) + R^T (c_dot - v), the leg kinematics of the
 * URDF solved exactly, so a perfect estimator reproduces the base.
 */
WalkingSample walkingSample(scalar_t time);

}  // namespace ocs2::humanoid::estimation::test_support
