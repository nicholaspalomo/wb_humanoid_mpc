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
#include <cstdlib>
#include <filesystem>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/reference/GaitFromConfig.h"
#include "humanoid_common_mpc/gait/ModeSequenceTemplate.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_mpc_config/gait_file.nproto.h"

/**
 * Properties of the swing references of the SwingTrajectoryPlanner over mode
 * schedules whose swings span several phases (the run gaits: a foot's swing is
 * FLY | other foot in stance | FLY). Every swing is one spline over its whole
 * duration, so the references are continuous at every event time, the apex is
 * reached once, at mid-swing, and the short-swing scaling is the one of the
 * whole swing.
 */
namespace ocs2::humanoid {
namespace {

constexpr char kGaitFile[] =
    "humanoid_nmpc/humanoid_common_mpc/config/command/gait.textproto";
constexpr scalar_t kEventEpsilon =
    1.0e-9;  // [s] evaluation offset to either side of an event
constexpr scalar_t kContinuityTolerance = 1.0e-6;
constexpr scalar_t kTol = 1.0e-9;

/** A configuration whose end conditions are all zero, so that every reference
 * is continuous at lift-off and touch-down too. */
SwingTrajectoryPlanner::Config continuousConfig() {
  SwingTrajectoryPlanner::Config config;
  config.swingHeight = 0.1;
  config.swingTimeScale = 0.3;
  config.impactProximityFactorMidPointValue = 0.2;
  return config;
}

/** One swing of one foot: its lift-off and touch-down times. */
struct Swing {
  scalar_t liftOff;
  scalar_t touchDown;
};

/** The swings of `foot` in `schedule`, which begins and ends with the foot in
 * stance. */
std::vector<Swing> swingsOf(const ModeSchedule& schedule, size_t foot) {
  std::vector<Swing> swings;
  for (size_t p = 1; p + 1 < schedule.modeSequence.size(); ++p) {
    const bool inStance = modeNumber2StanceLeg(schedule.modeSequence[p])[foot];
    const bool stanceBefore =
        modeNumber2StanceLeg(schedule.modeSequence[p - 1])[foot];
    if (!inStance && stanceBefore)
      swings.push_back(
          Swing{.liftOff = schedule.eventTimes[p - 1], .touchDown = 0.0});
    const bool stanceAfter =
        modeNumber2StanceLeg(schedule.modeSequence[p + 1])[foot];
    if (!inStance && stanceAfter)
      swings.back().touchDown = schedule.eventTimes[p];
  }
  return swings;
}

/** STANCE | a random sequence of modes with random durations | STANCE. */
ModeSchedule randomSchedule(std::mt19937& generator) {
  std::uniform_int_distribution<int> phaseCount(1, 12);
  std::uniform_int_distribution<size_t> mode(ModeNumber::kFly,
                                             ModeNumber::kStance);
  std::uniform_real_distribution<scalar_t> duration(0.03, 0.4);
  std::vector<size_t> modes = {ModeNumber::kStance};
  std::vector<scalar_t> eventTimes;
  scalar_t time = 0.5;
  const int count = phaseCount(generator);
  for (int i = 0; i < count; ++i) {
    eventTimes.push_back(time);
    modes.push_back(mode(generator));
    time += duration(generator);
  }
  eventTimes.push_back(time);
  modes.push_back(ModeNumber::kStance);
  return ModeSchedule(eventTimes, modes);
}

/** STANCE | `cycles` repetitions of `gait` from time 0.5 | STANCE. */
ModeSchedule repeatedGaitSchedule(const ModeSequenceTemplate& gait,
                                  int cycles) {
  std::vector<size_t> modes = {ModeNumber::kStance};
  std::vector<scalar_t> eventTimes;
  const scalar_t period = gait.switchingTimes.back();
  for (int cycle = 0; cycle < cycles; ++cycle) {
    for (size_t i = 0; i < gait.modeSequence.size(); ++i) {
      eventTimes.push_back(0.5 + cycle * period + gait.switchingTimes[i]);
      modes.push_back(gait.modeSequence[i]);
    }
  }
  eventTimes.push_back(0.5 + cycles * period);
  modes.push_back(ModeNumber::kStance);
  return ModeSchedule(eventTimes, modes);
}

/** Expects every reference of every foot to be continuous at every event of
 * `schedule`. */
void expectContinuousAtEveryEvent(const SwingTrajectoryPlanner& planner,
                                  const ModeSchedule& schedule) {
  for (size_t foot = 0; foot < kNumContacts; ++foot) {
    for (const scalar_t event : schedule.eventTimes) {
      const scalar_t before = event - kEventEpsilon;
      const scalar_t after = event + kEventEpsilon;
      EXPECT_NEAR(planner.getDesPosFootRtWorldEwrtWorldZ(foot, before),
                  planner.getDesPosFootRtWorldEwrtWorldZ(foot, after),
                  kContinuityTolerance)
          << "foot " << foot << " at " << event;
      EXPECT_NEAR(planner.getDesLinvelFootRtWorldEwrtWorldZ(foot, before),
                  planner.getDesLinvelFootRtWorldEwrtWorldZ(foot, after),
                  kContinuityTolerance)
          << "foot " << foot << " at " << event;
      EXPECT_NEAR(planner.getImpactProximityFactor(foot, before),
                  planner.getImpactProximityFactor(foot, after),
                  kContinuityTolerance)
          << "foot " << foot << " at " << event;
      EXPECT_NEAR(planner.getSwingPitchAngle(foot, before),
                  planner.getSwingPitchAngle(foot, after), kContinuityTolerance)
          << "foot " << foot << " at " << event;
    }
  }
}

/** Expects every swing of every foot to reach its scaled apex, at rest, exactly
 * at mid-swing. */
void expectOneApexPerSwing(const SwingTrajectoryPlanner& planner,
                           const ModeSchedule& schedule,
                           const SwingTrajectoryPlanner::Config& config,
                           scalar_t terrainHeight) {
  for (size_t foot = 0; foot < kNumContacts; ++foot) {
    for (const Swing& swing : swingsOf(schedule, foot)) {
      const scalar_t middle = 0.5 * (swing.liftOff + swing.touchDown);
      const scalar_t scaling = std::min(
          1.0, (swing.touchDown - swing.liftOff) / config.swingTimeScale);
      EXPECT_NEAR(planner.getDesPosFootRtWorldEwrtWorldZ(foot, middle),
                  terrainHeight + scaling * config.swingHeight, kTol)
          << "foot " << foot << " swing from " << swing.liftOff;
      EXPECT_NEAR(planner.getDesLinvelFootRtWorldEwrtWorldZ(foot, middle), 0.0,
                  kTol);
      EXPECT_NEAR(planner.getImpactProximityFactor(foot, middle),
                  config.impactProximityFactorMidPointValue, kTol);
    }
  }
}

/** The shipped gait file in this test's runfiles, or empty when it is not
 * there. */
std::string gaitFilePath() {
  std::vector<std::filesystem::path> roots;
  if (const char* absl_nullable srcDir = std::getenv("TEST_SRCDIR")) {
    roots.emplace_back(std::filesystem::path(srcDir) / "_main");
    roots.emplace_back(std::filesystem::path(srcDir) / "wb_humanoid_mpc");
  }
  roots.emplace_back(std::filesystem::current_path());
  for (const std::filesystem::path& root : roots) {
    const std::filesystem::path candidate = root / kGaitFile;
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  return std::string();
}

}  // namespace

TEST(SwingTrajectoryContinuity, RandomMultiPhaseSchedules) {
  std::mt19937 generator(7);
  std::uniform_real_distribution<scalar_t> terrain(-0.2, 0.2);
  const SwingTrajectoryPlanner::Config config = continuousConfig();
  for (int trial = 0; trial < 300; ++trial) {
    const ModeSchedule schedule = randomSchedule(generator);
    const scalar_t terrainHeight = terrain(generator);
    SwingTrajectoryPlanner planner(config, kNumContacts);
    planner.update(schedule, terrainHeight);
    expectContinuousAtEveryEvent(planner, schedule);
    expectOneApexPerSwing(planner, schedule, config, terrainHeight);
  }
}

TEST(SwingTrajectoryContinuity, PitchedSwingsAreContinuous) {
  std::mt19937 generator(11);
  SwingTrajectoryPlanner::Config config = continuousConfig();
  config.swingPitchAngle = 0.2;
  for (int trial = 0; trial < 100; ++trial) {
    const ModeSchedule schedule = randomSchedule(generator);
    SwingTrajectoryPlanner planner(config, kNumContacts);
    planner.update(schedule, /*terrainHeight=*/0.0);
    expectContinuousAtEveryEvent(planner, schedule);
  }
}

TEST(SwingTrajectoryContinuity, ShippedRunGaits) {
  const std::string gaitFile = gaitFilePath();
  ASSERT_FALSE(gaitFile.empty()) << kGaitFile << " is not in the runfiles";
  const absl::StatusOr<mpc_config::GaitFile> file = loadGaitFile(gaitFile);
  ASSERT_TRUE(file.ok()) << file.status();
  const absl::StatusOr<std::map<std::string, ModeSequenceTemplate>> gaits =
      gaitMapFromConfig(*file);
  ASSERT_TRUE(gaits.ok()) << gaits.status();
  for (const std::string& name :
       {std::string("run"), std::string("fast_run")}) {
    ASSERT_TRUE(gaits->contains(name)) << name;
    const ModeSchedule schedule =
        repeatedGaitSchedule(gaits->find(name)->second, /*cycles=*/3);
    // The gait must actually have a swing spanning several phases, or this test
    // checks nothing beyond the single-phase case.
    bool multiPhaseSwing = false;
    for (const Swing& swing : swingsOf(schedule, /*foot=*/0)) {
      for (const scalar_t event : schedule.eventTimes)
        multiPhaseSwing |=
            event > swing.liftOff + kTol && event < swing.touchDown - kTol;
    }
    ASSERT_TRUE(multiPhaseSwing) << name;

    const SwingTrajectoryPlanner::Config config = continuousConfig();
    SwingTrajectoryPlanner planner(config, kNumContacts);
    planner.update(schedule, /*terrainHeight=*/0.05);
    expectContinuousAtEveryEvent(planner, schedule);
    expectOneApexPerSwing(planner, schedule, config, /*terrainHeight=*/0.05);
  }
}

/** With nonzero end velocities the references jump at lift-off and touch-down
 * by design, but never inside a swing. */
TEST(SwingTrajectoryContinuity, InnerEventsAreContinuousWithEndVelocities) {
  std::mt19937 generator(3);
  SwingTrajectoryPlanner::Config config = continuousConfig();
  config.liftoffLinvelFootRtWorldEwrtWorldZ = 0.4;
  config.touchdownLinvelFootRtWorldEwrtWorldZ = -0.3;
  config.impactProximityFactorLiftoffSlope = -2.0;
  config.impactProximityFactorTouchdownSlope = 2.0;
  for (int trial = 0; trial < 300; ++trial) {
    const ModeSchedule schedule = randomSchedule(generator);
    SwingTrajectoryPlanner planner(config, kNumContacts);
    planner.update(schedule, /*terrainHeight=*/0.0);
    for (size_t foot = 0; foot < kNumContacts; ++foot) {
      for (const Swing& swing : swingsOf(schedule, foot)) {
        const scalar_t scaling = std::min(
            1.0, (swing.touchDown - swing.liftOff) / config.swingTimeScale);
        EXPECT_NEAR(planner.getDesLinvelFootRtWorldEwrtWorldZ(
                        foot, swing.liftOff + kEventEpsilon),
                    scaling * config.liftoffLinvelFootRtWorldEwrtWorldZ,
                    kContinuityTolerance);
        EXPECT_NEAR(planner.getDesLinvelFootRtWorldEwrtWorldZ(
                        foot, swing.touchDown - kEventEpsilon),
                    scaling * config.touchdownLinvelFootRtWorldEwrtWorldZ,
                    kContinuityTolerance);
        for (const scalar_t event : schedule.eventTimes) {
          if (event <= swing.liftOff + kTol || event >= swing.touchDown - kTol)
            continue;
          const scalar_t before = event - kEventEpsilon;
          const scalar_t after = event + kEventEpsilon;
          EXPECT_NEAR(planner.getDesPosFootRtWorldEwrtWorldZ(foot, before),
                      planner.getDesPosFootRtWorldEwrtWorldZ(foot, after),
                      kContinuityTolerance);
          EXPECT_NEAR(planner.getDesLinvelFootRtWorldEwrtWorldZ(foot, before),
                      planner.getDesLinvelFootRtWorldEwrtWorldZ(foot, after),
                      kContinuityTolerance);
          EXPECT_NEAR(planner.getImpactProximityFactor(foot, before),
                      planner.getImpactProximityFactor(foot, after),
                      kContinuityTolerance);
        }
      }
    }
  }
}

/** A step up taller than the swing height must still approach the touch-down
 * height from above, never from below. */
TEST(SwingTrajectoryContinuity, StepUpApproachesFromAbove) {
  SwingTrajectoryPlanner::Config config = continuousConfig();
  const ModeSchedule schedule(
      {1.0, 1.4}, {ModeNumber::kStance, ModeNumber::kRf, ModeNumber::kStance});
  for (const scalar_t step : {-0.3, -0.05, 0.0, 0.05, 0.1, 0.25}) {
    feet_array_t<scalar_array_t> liftOff =
        makeFeetArray(scalar_array_t(3, 0.0));
    feet_array_t<scalar_array_t> touchDown =
        makeFeetArray(scalar_array_t(3, step));
    SwingTrajectoryPlanner planner(config, kNumContacts);
    planner.update(schedule, liftOff, touchDown);
    // Foot 0 (left) swings in RF. Its height must stay at or above the
    // touch-down height over the second half.
    scalar_t peak = -1.0;
    for (int i = 0; i <= 400; ++i) {
      const scalar_t time = 1.0 + 0.4 * i / 400.0;
      const scalar_t height =
          planner.getDesPosFootRtWorldEwrtWorldZ(/*leg=*/0, time);
      peak = std::max(peak, height);
      if (time >= 1.2)
        EXPECT_GE(height, step - kTol) << "step " << step << " at " << time;
    }
    EXPECT_NEAR(peak, std::max(0.0, step) + config.swingHeight, kTol)
        << "step " << step;
  }
}

}  // namespace ocs2::humanoid
