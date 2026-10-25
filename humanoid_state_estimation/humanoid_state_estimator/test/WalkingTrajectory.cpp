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

#include "humanoid_state_estimation/humanoid_state_estimator/test/WalkingTrajectory.h"

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "Eigen/Geometry"

namespace ocs2::humanoid::estimation::test_support {
namespace {

constexpr scalar_t kPi = 3.14159265358979323846;
constexpr scalar_t kForwardSpeed = 0.4;
constexpr scalar_t kSwayAmplitude = 0.03;
constexpr scalar_t kSwayFrequency = 2.0 * kPi / 0.8;
constexpr scalar_t kHeight = 0.8;
constexpr scalar_t kBobAmplitude = 0.02;
constexpr scalar_t kBobFrequency = 2.0 * kPi / 0.4;
constexpr scalar_t kYawAmplitude = 0.2;
constexpr scalar_t kYawFrequency = 0.5;
constexpr scalar_t kPitchAmplitude = 0.05;
constexpr scalar_t kPitchFrequency = 1.3;
constexpr scalar_t kRollAmplitude = 0.04;
constexpr scalar_t kRollFrequency = 1.7;
constexpr scalar_t kGaitPeriod = 0.8;
constexpr scalar_t kSwingDuration = 0.3 * kGaitPeriod;
constexpr scalar_t kSwingHeight = 0.08;
constexpr scalar_t kFootLead = 0.15;
/** Where in the gait period each foot lifts off: the right at 0.1, the left at
 * 0.6. */
constexpr std::array<scalar_t, kNumFeet> kLiftOffPhase = {0.6 * kGaitPeriod,
                                                          0.1 * kGaitPeriod};
constexpr std::array<scalar_t, kNumFeet> kFootSide = {0.1, -0.1};
/** The legs' origins in the base frame (the URDF's first joint of each leg). */
const std::array<vector3_t, kNumFeet> kLegOrigin = {vector3_t(0.0, 0.1, -0.8),
                                                    vector3_t(0.0, -0.1, -0.8)};

/** A sinusoid a sin(w t) and its first two derivatives. */
struct Wave {
  scalar_t value;
  scalar_t rate;
  scalar_t acceleration;
};

Wave wave(scalar_t amplitude, scalar_t frequency, scalar_t time) {
  return Wave{.value = amplitude * std::sin(frequency * time),
              .rate = amplitude * frequency * std::cos(frequency * time),
              .acceleration = -amplitude * frequency * frequency *
                              std::sin(frequency * time)};
}

/** The foothold a foot stands on after its `swings`-th swing (0: where it
 * starts). */
vector3_t foothold(size_t foot, int swings) {
  const scalar_t touchDown =
      kLiftOffPhase[foot] + (swings - 1) * kGaitPeriod + kSwingDuration;
  const scalar_t x = swings == 0 ? 0.0 : kForwardSpeed * touchDown + kFootLead;
  return vector3_t(x, kFootSide[foot], 0.0);
}

void footState(size_t foot,
               scalar_t time,
               vector3_t& position,
               vector3_t& velocity,
               bool& stance) {
  // The swings that began by `time`, and how far into the newest one it is.
  const scalar_t sinceFirst = time - kLiftOffPhase[foot];
  const int begun =
      sinceFirst < 0.0
          ? 0
          : static_cast<int>(std::floor(sinceFirst / kGaitPeriod)) + 1;
  const scalar_t intoSwing =
      begun == 0 ? -1.0 : sinceFirst - (begun - 1) * kGaitPeriod;
  if (begun == 0 || intoSwing >= kSwingDuration) {
    stance = true;
    position = foothold(foot, begun);
    velocity.setZero();
    return;
  }
  stance = false;
  const vector3_t from = foothold(foot, begun - 1);
  const vector3_t to = foothold(foot, begun);
  const scalar_t phase = intoSwing / kSwingDuration;
  const scalar_t blend = 0.5 * (1.0 - std::cos(kPi * phase));
  const scalar_t blendRate = 0.5 * kPi * std::sin(kPi * phase) / kSwingDuration;
  position = from + blend * (to - from);
  velocity = blendRate * (to - from);
  position.z() = kSwingHeight * std::sin(kPi * phase);
  velocity.z() = kSwingHeight * kPi * std::cos(kPi * phase) / kSwingDuration;
}

}  // namespace

std::vector<std::string> prismaticBipedJointNames() {
  return {"left_x", "left_y", "left_z", "right_x", "right_y", "right_z"};
}

std::vector<std::string> prismaticBipedFootFrames() {
  return {"left_foot", "right_foot"};
}

WalkingSample walkingSample(scalar_t time) {
  WalkingSample sample;
  sample.time = time;
  const Wave sway = wave(kSwayAmplitude, kSwayFrequency, time);
  const Wave bob = wave(kBobAmplitude, kBobFrequency, time);
  sample.basePosition =
      vector3_t(kForwardSpeed * time, sway.value, kHeight + bob.value);
  sample.baseVelocity = vector3_t(kForwardSpeed, sway.rate, bob.rate);
  sample.baseAcceleration = vector3_t(0.0, sway.acceleration, bob.acceleration);

  // R = Rz(yaw) Ry(pitch) Rx(roll); omega_world = yaw' e_z + pitch' Rz e_y +
  // roll' Rz Ry e_x.
  const Wave yaw = wave(kYawAmplitude, kYawFrequency, time);
  const Wave pitch = wave(kPitchAmplitude, kPitchFrequency, time);
  const Wave roll = wave(kRollAmplitude, kRollFrequency, time);
  const matrix3_t rz =
      Eigen::AngleAxisd(yaw.value, vector3_t::UnitZ()).toRotationMatrix();
  const matrix3_t ry =
      Eigen::AngleAxisd(pitch.value, vector3_t::UnitY()).toRotationMatrix();
  const matrix3_t rx =
      Eigen::AngleAxisd(roll.value, vector3_t::UnitX()).toRotationMatrix();
  const matrix3_t rotation = rz * ry * rx;
  sample.orientation = quaternion_t(rotation);
  const vector3_t omegaWorld = yaw.rate * vector3_t::UnitZ() +
                               pitch.rate * (rz * vector3_t::UnitY()) +
                               roll.rate * (rz * ry * vector3_t::UnitX());
  sample.angularVelocity = rotation.transpose() * omegaWorld;
  sample.imuLinearAcceleration =
      rotation.transpose() * (sample.baseAcceleration - kGravity);

  for (size_t foot = 0; foot < kNumFeet; ++foot) {
    bool stance = true;
    footState(foot, time, sample.footPositions[foot],
              sample.footVelocities[foot], stance);
    sample.contactFlags[foot] = stance;
    const vector3_t relative =
        rotation.transpose() *
        (sample.footPositions[foot] - sample.basePosition);
    const vector3_t relativeRate =
        -sample.angularVelocity.cross(relative) +
        rotation.transpose() *
            (sample.footVelocities[foot] - sample.baseVelocity);
    sample.jointPositions.segment<3>(static_cast<Eigen::Index>(3 * foot)) =
        relative - kLegOrigin[foot];
    sample.jointVelocities.segment<3>(static_cast<Eigen::Index>(3 * foot)) =
        relativeRate;
  }
  return sample;
}

}  // namespace ocs2::humanoid::estimation::test_support
