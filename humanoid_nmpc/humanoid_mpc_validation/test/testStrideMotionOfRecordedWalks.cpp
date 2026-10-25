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
#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Geometry"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/command/BasePlanarMotion.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/reference/GaitFromConfig.h"
#include "humanoid_common_mpc/gait/ModeSequenceTemplate.h"
#include "humanoid_common_mpc/orientation/BaseOrientation.h"
#include "humanoid_common_mpc/reference_manager/StrideMotionHistory.h"
#include "humanoid_mpc_config/gait_file.nproto.h"
#include "humanoid_mpc_validation/closed_loop/RecordedRobotStates.h"
#include "humanoid_mpc_validation/io/GoldenIo.h"
#include "robot_core/ResourcePaths.h"

/*
 * The measured motion of the gait switches and the velocity command's knots
 * (ProceduralMpcMotionManager) on the robots' recorded MuJoCo walks: straight
 * walks, along which the pelvis's own yaw rate (R w_B)_z, exact, swings past
 * the 0.15 rad/s that raises slow_walk to walk and its sideways velocity past
 * 0.1 m/s, read as neither a turn nor a sideways walk once averaged over the
 * cycle of the gait they walk (StrideMotionHistory). A turn added to the
 * recording reads exactly as that turn. A stall of the solves of up to 1 s,
 * anywhere in the walk, is bridged within the bounds StrideMotionHistory
 * states; a history restarted mid-walk from the instantaneous twist reads the
 * sway as a turn, which is why the gait switches read no restarted mean. Reads
 * data/benchmark/states/ and the shipped gait file.
 */

namespace ocs2::humanoid::validation {
namespace {

/** A recorded straight walk and the gait it walks in. Passive data. */
struct RecordedWalk {
  std::string file;
  /** The gait of the gait file the run walked in, the one whose stride the
   * motion manager averages over. */
  std::string gait;
};

// LINT.IfChange(recorded_walks)
/** The recorded walks; their gaits, from the stride period of each recording
 * (checked below). */
std::vector<RecordedWalk> recordedWalks() {
  return {{.file = "unitree_g1_walk_0p5_states.txt", .gait = "walk"},
          {.file = "engineai_sa01_walk_0p3_states.txt", .gait = "slow_walk"},
          {.file = "unitree_r1_walk_0p3_states.txt", .gait = "slow_walk"},
          {.file = "drc_atlas_walk_0p5_states.txt", .gait = "walk"},
          {.file = "unitree_g1_wb_walk_0p5_states.txt", .gait = "walk"}};
}
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/BUILD.bazel:recorded_walks_test_data)

// [rad/s] Half the smallest turn that raises a gait (slow_walk: 0.2 - 0.05),
// and [m/s] the sideways walk read as none.
constexpr scalar_t kStraightTurnBound = 0.075;
constexpr scalar_t kStraightLateralBound = 0.05;

/** The recording `file` of data/benchmark/states/; an empty recording after a
 * failed expectation. */
RecordedRobotStates loadRecording(const std::string& file) {
  const absl::StatusOr<std::string> path = robot::resolveResourcePath(
      std::string(
          "humanoid_nmpc/humanoid_mpc_validation/data/benchmark/states/") +
      file);
  EXPECT_TRUE(path.ok()) << path.status();
  if (!path.ok()) return RecordedRobotStates();
  const absl::StatusOr<GoldenFile> golden = readGoldenFile(*path);
  EXPECT_TRUE(golden.ok()) << golden.status();
  if (!golden.ok()) return RecordedRobotStates();
  absl::StatusOr<RecordedRobotStates> states = fromGoldenFile(*golden);
  EXPECT_TRUE(states.ok()) << states.status();
  return states.ok() ? *std::move(states) : RecordedRobotStates();
}

/** [s] The cycle of each gait of the shipped gait file; empty after a failed
 * expectation. */
std::map<std::string, scalar_t> gaitCycles() {
  const absl::StatusOr<std::string> path = robot::resolveResourcePath(
      "humanoid_nmpc/humanoid_common_mpc/config/command/gait.textproto");
  EXPECT_TRUE(path.ok()) << path.status();
  if (!path.ok()) return {};
  const absl::StatusOr<mpc_config::GaitFile> file = loadGaitFile(*path);
  EXPECT_TRUE(file.ok()) << file.status();
  if (!file.ok()) return {};
  const absl::StatusOr<std::map<std::string, ModeSequenceTemplate>> gaits =
      gaitMapFromConfig(*file);
  EXPECT_TRUE(gaits.ok()) << gaits.status();
  std::map<std::string, scalar_t> cycles;
  if (!gaits.ok()) return cycles;
  for (const std::pair<const std::string, ModeSequenceTemplate>& gait :
       *gaits) {
    cycles[gait.first] =
        gait.second.switchingTimes.back() - gait.second.switchingTimes.front();
  }
  return cycles;
}

/** The world-frame base twist [v_W, w_W] of a record (its velocities are in the
 * base frame). */
vector6_t worldTwistOf(const RobotStateRecord& record) {
  const Eigen::Quaterniond rotation(record.baseQuaternion);
  vector6_t twist;
  twist << rotation * record.baseLinearVelocityLocal,
      rotation * record.baseAngularVelocityLocal;
  return twist;
}

/** [s] The lag in [0.8, 2.2] s at which the autocorrelation of the pelvis yaw
 * rate peaks: the recording's stride period. */
scalar_t strideOfYawRate(const RecordedRobotStates& recording) {
  // On a uniform 5 ms grid, by linear interpolation of the samples.
  constexpr scalar_t kGrid = 0.005;
  const std::vector<RobotStateRecord>& records = recording.records;
  std::vector<scalar_t> rate;
  size_t next = 1;
  for (scalar_t time = records.front().time; time <= records.back().time;
       time += kGrid) {
    while (next + 1 < records.size() && records[next].time < time) ++next;
    const scalar_t fraction = (time - records[next - 1].time) /
                              (records[next].time - records[next - 1].time);
    rate.push_back((1.0 - fraction) * worldTwistOf(records[next - 1])(5) +
                   fraction * worldTwistOf(records[next])(5));
  }
  scalar_t mean = 0.0;
  for (const scalar_t value : rate)
    mean += value / static_cast<scalar_t>(rate.size());
  scalar_t bestLag = 0.0;
  scalar_t bestCorrelation = -1.0e300;
  for (size_t lag = static_cast<size_t>(0.8 / kGrid);
       lag <= static_cast<size_t>(2.2 / kGrid) && lag < rate.size(); ++lag) {
    scalar_t correlation = 0.0;
    for (size_t i = 0; i + lag < rate.size(); ++i)
      correlation += (rate[i] - mean) * (rate[i + lag] - mean);
    if (correlation > bestCorrelation) {
      bestCorrelation = correlation;
      bestLag = static_cast<scalar_t>(lag) * kGrid;
    }
  }
  return bestLag;
}

/** What the stride means of a recording reach once a whole stride is recorded.
 * Passive data. */
struct StrideMeans {
  scalar_t largestTurn = 0.0;
  scalar_t largestLateralVelocity = 0.0;
  scalar_t smallestForwardVelocity = 1.0e300;
  /** The means, sample by sample from the first whole stride on. */
  std::vector<scalar_t> turns;
};

/**
 * The stride means over `window` of `recording` as the motion manager forms
 * them, its heading turned by `turnRate` [rad/s] about the world z axis since
 * the first sample (the twist heading of q_z(a) (x) xi is psi_h(xi) + a).
 */
StrideMeans strideMeansOf(const RecordedRobotStates& recording,
                          scalar_t window,
                          scalar_t turnRate) {
  StrideMotionHistory history(/*retention=*/2.5);
  StrideMeans means;
  const scalar_t start = recording.records.front().time;
  for (const RobotStateRecord& record : recording.records) {
    const scalar_t addedTurn = turnRate * (record.time - start);
    const Eigen::Quaterniond turned =
        Eigen::Quaterniond(Eigen::AngleAxisd(addedTurn, vector3_t::UnitZ())) *
        Eigen::Quaterniond(record.baseQuaternion);
    // The motion before the recording: its first sample's exact planar motion,
    // as the motion manager seeds a history.
    history.addSample(record.time, vector2_t(record.basePosition.head<2>()),
                      twistHeading(turned.coeffs()), [&record]() {
                        return instantaneousPlanarMotion(record.baseQuaternion,
                                                         worldTwistOf(record));
                      });
    if (record.time < start + window) continue;
    const BasePlanarMotion mean = history.mean(window);
    means.turns.push_back(mean.headingRate);
    means.largestTurn = std::max(means.largestTurn, std::abs(mean.headingRate));
    means.largestLateralVelocity = std::max(
        means.largestLateralVelocity, std::abs(mean.velocityInHeadingFrame(1)));
    means.smallestForwardVelocity =
        std::min(means.smallestForwardVelocity, mean.velocityInHeadingFrame(0));
  }
  return means;
}

TEST(StrideMotionOfRecordedWalks,
     AStraightWalkReadsAsNeitherATurnNorASidewaysWalk) {
  const std::map<std::string, scalar_t> cycles = gaitCycles();
  for (const RecordedWalk& walk : recordedWalks()) {
    SCOPED_TRACE(walk.file);
    const RecordedRobotStates recording = loadRecording(walk.file);
    ASSERT_GT(recording.records.size(), 100);
    ASSERT_TRUE(cycles.contains(walk.gait));
    const scalar_t cycle = cycles.at(walk.gait);
    // The recording walks in that gait: its pelvis yaw rate repeats with the
    // gait's cycle.
    EXPECT_NEAR(strideOfYawRate(recording), cycle, 0.05);
    ASSERT_GT(recording.records.back().time - recording.records.front().time,
              cycle + 1.0)
        << "a stride and a second of samples";

    // The instantaneous motion of the pelvis, exact, sways past the thresholds
    // ...
    scalar_t largestYawRate = 0.0;
    scalar_t largestLateralVelocity = 0.0;
    for (const RobotStateRecord& record : recording.records) {
      const BasePlanarMotion instantaneous = instantaneousPlanarMotion(
          record.baseQuaternion, worldTwistOf(record));
      largestYawRate =
          std::max(largestYawRate, std::abs(worldTwistOf(record)(5)));
      largestLateralVelocity =
          std::max(largestLateralVelocity,
                   std::abs(instantaneous.velocityInHeadingFrame(1)));
    }
    EXPECT_GT(largestYawRate, 0.15)
        << "the pelvis must yaw fast enough to have raised slow_walk";
    EXPECT_GT(largestLateralVelocity, 0.1) << "the pelvis must sway sideways";
    // ... its mean over the gait's cycle does not.
    const StrideMeans means = strideMeansOf(recording, cycle, /*turnRate=*/0.0);
    EXPECT_LT(means.largestTurn, kStraightTurnBound);
    EXPECT_LT(means.largestLateralVelocity, kStraightLateralBound);
    EXPECT_GT(means.smallestForwardVelocity, 0.05) << "it walks forward";
  }
}

TEST(StrideMotionOfRecordedWalks, AnAddedTurnReadsExactlyAsThatTurn) {
  const std::map<std::string, scalar_t> cycles = gaitCycles();
  for (const RecordedWalk& walk : recordedWalks()) {
    SCOPED_TRACE(walk.file);
    const RecordedRobotStates recording = loadRecording(walk.file);
    ASSERT_GT(recording.records.size(), 100);
    ASSERT_TRUE(cycles.contains(walk.gait));
    const scalar_t cycle = cycles.at(walk.gait);
    const StrideMeans straight =
        strideMeansOf(recording, cycle, /*turnRate=*/0.0);
    const StrideMeans turning =
        strideMeansOf(recording, cycle, /*turnRate=*/0.2);
    ASSERT_EQ(turning.turns.size(), straight.turns.size());
    ASSERT_FALSE(turning.turns.empty());
    for (size_t i = 0; i < turning.turns.size(); ++i)
      EXPECT_NEAR(turning.turns[i] - straight.turns[i], 0.2, 1.0e-12) << i;
  }
}

/** The fraction of `recording`'s samples whose instantaneous sideways velocity
 * in the heading frame exceeds `speed`. */
scalar_t fractionMovingSidewaysFasterThan(const RecordedRobotStates& recording,
                                          scalar_t speed) {
  size_t faster = 0;
  for (const RobotStateRecord& record : recording.records) {
    if (std::abs(instantaneousPlanarMotion(record.baseQuaternion,
                                           worldTwistOf(record))
                     .velocityInHeadingFrame(1)) > speed)
      ++faster;
  }
  return static_cast<scalar_t>(faster) /
         static_cast<scalar_t>(recording.records.size());
}

TEST(StrideMotionOfRecordedWalks,
     TheInstantaneousSidewaysSwayPassesTheSlowerGaitThresholdItsMeanDoesNot) {
  // The slower-gait switch out of slow_walk needs |v_y| < 0.05 + 0.05 m/s
  // (design README section 2.9, the linear clauses): the pelvis's instantaneous
  // sideways velocity exceeds it in about half of the samples of four of the
  // five robots, while its stride mean stays below kStraightLateralBound
  // (AStraightWalkReadsAsNeitherATurnNorASidewaysWalk).
  for (const RecordedWalk& walk : recordedWalks()) {
    SCOPED_TRACE(walk.file);
    const RecordedRobotStates recording = loadRecording(walk.file);
    ASSERT_GT(recording.records.size(), 100);
    const scalar_t fraction =
        fractionMovingSidewaysFasterThan(recording, /*speed=*/0.1);
    if (walk.file.starts_with("drc_atlas")) {
      EXPECT_GT(fraction, 0.0)
          << "DRC Atlas sways less, but past the threshold too";
    } else {
      EXPECT_GT(fraction, 0.45);
    }
  }
}

/** The stride means over `window` of `records` from `first` on, the samples
 * strictly inside (gapStart, gapEnd) left out. */
struct BridgedMeans {
  std::vector<scalar_t> times;
  std::vector<BasePlanarMotion> means;
  std::vector<bool> measured;
};

BridgedMeans meansWithAGap(const std::vector<RobotStateRecord>& records,
                           scalar_t window,
                           scalar_t gapStart,
                           scalar_t gapEnd) {
  StrideMotionHistory history(/*retention=*/2.5);
  BridgedMeans result;
  int seeds = 0;
  for (const RobotStateRecord& record : records) {
    if (record.time > gapStart + 1.0e-9 && record.time < gapEnd - 1.0e-9)
      continue;
    history.addSample(record.time, vector2_t(record.basePosition.head<2>()),
                      twistHeading(record.baseQuaternion), [&seeds]() {
                        ++seeds;
                        return BasePlanarMotion{};
                      });
    result.times.push_back(record.time);
    result.means.push_back(history.mean(window));
    result.measured.push_back(history.isMeasuredOver(window));
  }
  EXPECT_EQ(seeds, 1) << "only the first sample is seeded: a bridged gap does "
                         "not restart the history";
  return result;
}

TEST(StrideMotionOfRecordedWalks,
     AStallOfUpToASecondIsBridgedWithoutReadingTheSway) {
  // A stall of the solves at every place in the recording: the history goes on
  // across it without a seed. While the window starts before the stall the
  // heading rate is the dense history's exactly, the velocity off by the chord
  // across the stall only; once it starts inside, both are off by the heading's
  // deviation from the line across the stall, within the bounds
  // StrideMotionHistory states. The mean is never measured while its window
  // holds the stall.
  const std::map<std::string, scalar_t> cycles = gaitCycles();
  for (const RecordedWalk& walk : recordedWalks()) {
    SCOPED_TRACE(walk.file);
    const RecordedRobotStates recording = loadRecording(walk.file);
    ASSERT_GT(recording.records.size(), 100);
    ASSERT_TRUE(cycles.contains(walk.gait));
    const scalar_t cycle = cycles.at(walk.gait);
    const std::vector<RobotStateRecord>& records = recording.records;
    const BridgedMeans dense =
        meansWithAGap(records, cycle, /*gapStart=*/-1.0, /*gapEnd=*/-1.0);
    for (const scalar_t gap : {0.15, 0.5, 1.0}) {
      SCOPED_TRACE(gap);
      int stalls = 0;
      scalar_t largestTurnError = 0.0;
      scalar_t largestVelocityErrorBeforeTheStall = 0.0;
      scalar_t largestVelocityError = 0.0;
      for (size_t start = 0; start < records.size(); start += 7) {
        const scalar_t gapStart = records[start].time;
        if (gapStart - cycle < records.front().time ||
            gapStart + gap + cycle > records.back().time)
          continue;
        ++stalls;
        const BridgedMeans bridged =
            meansWithAGap(records, cycle, gapStart, gapStart + gap);
        size_t denseIndex = 0;
        for (size_t i = 0; i < bridged.times.size(); ++i) {
          while (dense.times[denseIndex] < bridged.times[i]) ++denseIndex;
          const scalar_t time = bridged.times[i];
          if (time <= gapStart) {
            EXPECT_EQ(bridged.means[i].headingRate,
                      dense.means[denseIndex].headingRate)
                << "before the stall";
            continue;
          }
          const scalar_t turnError =
              std::abs(bridged.means[i].headingRate -
                       dense.means[denseIndex].headingRate);
          const scalar_t velocityError =
              (bridged.means[i].velocityInHeadingFrame -
               dense.means[denseIndex].velocityInHeadingFrame)
                  .norm();
          if (time - cycle <= gapStart) {
            EXPECT_LT(turnError, 1.0e-12)
                << "t = " << time << ": the window starts before the stall";
            largestVelocityErrorBeforeTheStall =
                std::max(largestVelocityErrorBeforeTheStall, velocityError);
          }
          largestTurnError = std::max(largestTurnError, turnError);
          largestVelocityError = std::max(largestVelocityError, velocityError);
          if (time - cycle < gapStart + gap - 1.0e-6) {
            EXPECT_FALSE(bridged.measured[i]) << "t = " << time;
          }
        }
      }
      if (stalls == 0)
        continue;  // a recording shorter than two strides and the stall
      EXPECT_LT(largestVelocityErrorBeforeTheStall, 0.0075);
      EXPECT_LT(largestTurnError, 0.045);
      EXPECT_LT(largestVelocityError, 0.051);
    }
  }
}

TEST(StrideMotionOfRecordedWalks,
     ARestartSeededWithTheInstantaneousTwistReadsTheSwayAsATurn) {
  // Why the gait switches read no seed (StrideMotionHistory::isMeasuredOver()):
  // a history restarted mid-walk with the exact instantaneous planar motion of
  // its first sample blends the sway's turn into the mean for a window. On the
  // Unitree G1's and R1's walks most restart instants read a turn above the
  // 0.15 rad/s that raises slow_walk.
  const std::map<std::string, scalar_t> cycles = gaitCycles();
  for (const RecordedWalk& walk : recordedWalks()) {
    if (!walk.file.starts_with("unitree_g1_walk") &&
        !walk.file.starts_with("unitree_r1"))
      continue;
    SCOPED_TRACE(walk.file);
    const RecordedRobotStates recording = loadRecording(walk.file);
    ASSERT_GT(recording.records.size(), 100);
    const scalar_t cycle = cycles.at(walk.gait);
    const std::vector<RobotStateRecord>& records = recording.records;
    int restarts = 0;
    int readAsATurn = 0;
    for (size_t first = 0; first < records.size() &&
                           records[first].time + cycle <= records.back().time;
         ++first) {
      ++restarts;
      StrideMotionHistory history(/*retention=*/2.5);
      bool turned = false;
      for (size_t i = first;
           i < records.size() && records[i].time <= records[first].time + cycle;
           ++i) {
        const RobotStateRecord& seedRecord = records[first];
        history.addSample(
            records[i].time, vector2_t(records[i].basePosition.head<2>()),
            twistHeading(records[i].baseQuaternion), [&seedRecord]() {
              return instantaneousPlanarMotion(seedRecord.baseQuaternion,
                                               worldTwistOf(seedRecord));
            });
        turned = turned || std::abs(history.mean(cycle).headingRate) > 0.15;
        EXPECT_FALSE(history.isMeasuredOver(cycle) &&
                     records[i].time < records[first].time + cycle - 1.0e-9);
      }
      if (turned) ++readAsATurn;
    }
    ASSERT_GT(restarts, 0);
    EXPECT_GT(
        static_cast<scalar_t>(readAsATurn) / static_cast<scalar_t>(restarts),
        0.5);
  }
}

}  // namespace
}  // namespace ocs2::humanoid::validation
