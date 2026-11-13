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

#include "legged_state_estimation/legged_state_estimator/test/WalkingTrajectory.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "Eigen/Geometry"
#include "ocs2_core/Types.h"

#include "legged_common_mpc/common/Types.h"

namespace ocs2::legged::estimation::test_support {
namespace {

constexpr scalar_t kPi = 3.14159265358979323846;
constexpr scalar_t kNomLinvelBaseRtWorldEwrtWorldX = 0.4;
constexpr scalar_t kAmplitudePosBaseRtWorldEwrtWorldY = 0.03;
constexpr scalar_t kSwayFrequency = 2.0 * kPi / 0.8;
constexpr scalar_t kNomPosBaseRtWorldEwrtWorldZ = 0.8;
constexpr scalar_t kAmplitudePosBaseRtWorldEwrtWorldZ = 0.02;
constexpr scalar_t kBobFrequency = 2.0 * kPi / 0.4;
constexpr scalar_t kAmplitudeEulerzyxBaseRtWorldYaw = 0.2;
constexpr scalar_t kYawFrequency = 0.5;
constexpr scalar_t kAmplitudeEulerzyxBaseRtWorldPitch = 0.05;
constexpr scalar_t kPitchFrequency = 1.3;
constexpr scalar_t kAmplitudeEulerzyxBaseRtWorldRoll = 0.04;
constexpr scalar_t kRollFrequency = 1.7;
constexpr scalar_t kGaitPeriod = 0.8;
constexpr scalar_t kSwingDuration = 0.3 * kGaitPeriod;
constexpr scalar_t kMaxSwingPosFootRtWorldEwrtWorldZ = 0.08;
constexpr scalar_t kTouchdownPosFootRtBaseEwrtWorldX = 0.15;
/** Where in the gait period each foot lifts off: the right at 0.1, the left at
 * 0.6. */
constexpr std::array<scalar_t, kNumFeet> kLiftOffPhase = {0.6 * kGaitPeriod,
                                                          0.1 * kGaitPeriod};
constexpr std::array<scalar_t, kNumFeet> kNomPosFeetRtWorldEwrtWorldY = {0.1,
                                                                         -0.1};
/** The legs' origins in the base frame (the URDF's first joint of each leg). */
const std::array<vector3_t, kNumFeet> kPosHipsRtBaseEwrtBase = {
    vector3_t(0.0, 0.1, -0.8), vector3_t(0.0, -0.1, -0.8)};

/** A sinusoid a sin(w t) and its first two derivatives. */
struct Wave {
  scalar_t value;
  scalar_t rate;
  scalar_t secondRate;
};

Wave wave(scalar_t amplitude, scalar_t frequency, scalar_t time) {
  return Wave{.value = amplitude * std::sin(frequency * time),
              .rate = amplitude * frequency * std::cos(frequency * time),
              .secondRate = -amplitude * frequency * frequency *
                            std::sin(frequency * time)};
}

/** The foothold a foot stands on after its
 * `swings`-th swing (0: where it starts). */
vector3_t computeFootholdPosFootRtWorldEwrtWorld(size_t foot, int swings) {
  const scalar_t touchDown =
      kLiftOffPhase[foot] + (swings - 1) * kGaitPeriod + kSwingDuration;
  const scalar_t posFootRtWorldEwrtWorldX =
      swings == 0 ? 0.0
                  : kNomLinvelBaseRtWorldEwrtWorldX * touchDown +
                        kTouchdownPosFootRtBaseEwrtWorldX;
  return vector3_t(posFootRtWorldEwrtWorldX, kNomPosFeetRtWorldEwrtWorldY[foot],
                   0.0);
}

void footState(size_t foot,
               scalar_t time,
               vector3_t& posFootRtWorldEwrtWorld,
               vector3_t& linvelFootRtWorldEwrtWorld,
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
    posFootRtWorldEwrtWorld =
        computeFootholdPosFootRtWorldEwrtWorld(foot, begun);
    linvelFootRtWorldEwrtWorld.setZero();
    return;
  }
  stance = false;
  const vector3_t liftoffPosFootRtWorldEwrtWorld =
      computeFootholdPosFootRtWorldEwrtWorld(foot, begun - 1);
  const vector3_t touchdownPosFootRtWorldEwrtWorld =
      computeFootholdPosFootRtWorldEwrtWorld(foot, begun);
  const scalar_t phase = intoSwing / kSwingDuration;
  const scalar_t blend = 0.5 * (1.0 - std::cos(kPi * phase));
  const scalar_t blendRate = 0.5 * kPi * std::sin(kPi * phase) / kSwingDuration;
  posFootRtWorldEwrtWorld = liftoffPosFootRtWorldEwrtWorld +
                            blend * (touchdownPosFootRtWorldEwrtWorld -
                                     liftoffPosFootRtWorldEwrtWorld);
  linvelFootRtWorldEwrtWorld = blendRate * (touchdownPosFootRtWorldEwrtWorld -
                                            liftoffPosFootRtWorldEwrtWorld);
  posFootRtWorldEwrtWorld.z() =
      kMaxSwingPosFootRtWorldEwrtWorldZ * std::sin(kPi * phase);
  linvelFootRtWorldEwrtWorld.z() = kMaxSwingPosFootRtWorldEwrtWorldZ * kPi *
                                   std::cos(kPi * phase) / kSwingDuration;
}

}  // namespace

std::vector<std::string> prismaticBipedJointNames() {
  return {"left_x", "left_y", "left_z", "right_x", "right_y", "right_z"};
}

std::vector<std::string> prismaticBipedFootLinks() {
  return {"left_foot", "right_foot"};
}

WalkingSample walkingSample(scalar_t time) {
  WalkingSample sample;
  sample.time = time;
  const Wave sway =
      wave(kAmplitudePosBaseRtWorldEwrtWorldY, kSwayFrequency, time);
  const Wave bob =
      wave(kAmplitudePosBaseRtWorldEwrtWorldZ, kBobFrequency, time);
  sample.posBaseRtWorldEwrtWorld =
      vector3_t(kNomLinvelBaseRtWorldEwrtWorldX * time, sway.value,
                kNomPosBaseRtWorldEwrtWorldZ + bob.value);
  sample.linvelBaseRtWorldEwrtWorld =
      vector3_t(kNomLinvelBaseRtWorldEwrtWorldX, sway.rate, bob.rate);
  sample.linaccBaseRtWorldEwrtWorld =
      vector3_t(0.0, sway.secondRate, bob.secondRate);

  // R = Rz(yaw) Ry(pitch) Rx(roll); omega_world = yaw' e_z + pitch' Rz e_y +
  // roll' Rz Ry e_x.
  const Wave eulerzyxBaseRtWorldYaw =
      wave(kAmplitudeEulerzyxBaseRtWorldYaw, kYawFrequency, time);
  const Wave eulerzyxBaseRtWorldPitch =
      wave(kAmplitudeEulerzyxBaseRtWorldPitch, kPitchFrequency, time);
  const Wave eulerzyxBaseRtWorldRoll =
      wave(kAmplitudeEulerzyxBaseRtWorldRoll, kRollFrequency, time);
  const matrix3_t rz =
      Eigen::AngleAxisd(eulerzyxBaseRtWorldYaw.value, vector3_t::UnitZ())
          .toRotationMatrix();
  const matrix3_t ry =
      Eigen::AngleAxisd(eulerzyxBaseRtWorldPitch.value, vector3_t::UnitY())
          .toRotationMatrix();
  const matrix3_t rx =
      Eigen::AngleAxisd(eulerzyxBaseRtWorldRoll.value, vector3_t::UnitX())
          .toRotationMatrix();
  const matrix3_t rotmatBaseRtWorld = rz * ry * rx;
  sample.quatBaseRtWorld = quaternion_t(rotmatBaseRtWorld);
  const vector3_t angvelBaseRtWorldEwrtWorld =
      eulerzyxBaseRtWorldYaw.rate * vector3_t::UnitZ() +
      eulerzyxBaseRtWorldPitch.rate * (rz * vector3_t::UnitY()) +
      eulerzyxBaseRtWorldRoll.rate * (rz * ry * vector3_t::UnitX());
  sample.angvelBaseRtWorldEwrtBase =
      rotmatBaseRtWorld.transpose() * angvelBaseRtWorldEwrtWorld;
  sample.specforceBaseRtWorldEwrtBase =
      rotmatBaseRtWorld.transpose() *
      (sample.linaccBaseRtWorldEwrtWorld - kGravityEwrtWorld);

  for (size_t foot = 0; foot < kNumFeet; ++foot) {
    bool stance = true;
    footState(foot, time, sample.posFeetRtWorldEwrtWorld[foot],
              sample.linvelFeetRtWorldEwrtWorld[foot], stance);
    sample.contactFlags[foot] = stance;
    const vector3_t posFootRtBaseEwrtBase =
        rotmatBaseRtWorld.transpose() *
        (sample.posFeetRtWorldEwrtWorld[foot] - sample.posBaseRtWorldEwrtWorld);
    const vector3_t linvelFootRtBaseEwrtBase =
        -sample.angvelBaseRtWorldEwrtBase.cross(posFootRtBaseEwrtBase) +
        rotmatBaseRtWorld.transpose() *
            (sample.linvelFeetRtWorldEwrtWorld[foot] -
             sample.linvelBaseRtWorldEwrtWorld);
    sample.jointPositions.segment<3>(static_cast<Eigen::Index>(3 * foot)) =
        posFootRtBaseEwrtBase - kPosHipsRtBaseEwrtBase[foot];
    sample.jointVelocities.segment<3>(static_cast<Eigen::Index>(3 * foot)) =
        linvelFootRtBaseEwrtBase;
  }
  return sample;
}

}  // namespace ocs2::legged::estimation::test_support
