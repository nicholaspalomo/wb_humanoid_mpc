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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_replace.h"
#include "gtest/gtest.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/centroidal.hpp"

#include "humanoid_common_mpc/command/WalkingVelocityCommand.h"
#include "humanoid_common_mpc/config/reference/ReferenceFromConfig.h"
#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"
#include "humanoid_mpc_config/reference_file.nproto.h"
#include "support/AtlasReferenceStack.h"
#include "support/TypedConfigFiles.h"

/*
 * ProceduralMpcMotionManager::Create() and its command-limit reloads, on the DRC Atlas references: a gait file without a
 * gait the velocity command selects and a reference file whose values do not parse are refused by Status, naming the
 * file, and a reloaded limit that is not a number is refused and keeps the running one. The constructor used to throw
 * both, and the reloads threw what the parameter updater then caught.
 *
 * And its gait switches on the centroidal state (design defect D2): they compare the command
 * with the robot's mean motion over a stride, the rate of its twist heading (StrideMotionHistory), which neither the
 * joints' angular momentum nor a roll moves, and the mean velocity of its base in the heading frame, which the gait's
 * sway does not move; and they read it only once its window is measured throughout, not from a restarted history,
 * which starts from the exact base twist of the observation handed over before the solve. They used to read entry 3 of
 * getBaseComVelocity(), which on the centroidal state is the normalized angular momentum about the world x axis, then the joints-locked
 * twist of the momentum, which reads the joints' angular momentum as a turn, and the instantaneous CoM velocity, whose sway dips below and
 * swings past the linear thresholds; and the slower switch tested the command where the measurement was meant.
 */

namespace ocs2::humanoid {
namespace {

/** `text` written to the file `name` in the test's temporary directory; its path. */
std::string writeTempFile(const std::string& name, const std::string& text) {
  const std::string file = (std::filesystem::path(::testing::TempDir()) / name).string();
  EXPECT_TRUE(writeTextFile(file, text).ok()) << file;
  return file;
}

absl::StatusOr<std::unique_ptr<ProceduralMpcMotionManager>> create(AtlasReferenceStack& stack,
                                                                   const std::string& gaitFile,
                                                                   const std::string& referenceFile) {
  CentroidalMpcTargetTrajectoriesCalculator* absl_nonnull calculator = &stack.targetCalculator();
  return ProceduralMpcMotionManager::Create(gaitFile, referenceFile, stack.referenceManagerPtr(), stack.model(),
                                            [calculator](const vector4_t& velocityTarget, scalar_t initTime, scalar_t /*finalTime*/,
                                                         const vector_t& initState, const BasePlanarMotion& measuredMotion) {
                                              return calculator->commandedVelocityToTargetTrajectories(velocityTarget, initTime, initState,
                                                                                                       measuredMotion);
                                            });
}

/**
 * A motion manager on `stack`'s references whose command reaches the gait switches as it is set: no command filter and
 * no ramp, and command limits of 1, so the scaled command is the operator's; null after a failed expectation.
 */
std::unique_ptr<ProceduralMpcMotionManager> unfilteredManager(AtlasReferenceStack& stack) {
  absl::StatusOr<std::unique_ptr<ProceduralMpcMotionManager>> manager = create(stack, stack.gaitFile(), stack.referenceFile());
  EXPECT_TRUE(manager.ok()) << manager.status();
  absl::StatusOr<ReferenceSettings> settings = referenceSettingsFromConfig(stack.config().reference);
  EXPECT_TRUE(settings.ok()) << settings.status();
  if (!manager.ok() || !settings.ok()) return nullptr;
  settings->maxDisplacementVelocityX = 1.0;
  settings->maxDisplacementVelocityY = 1.0;
  settings->maxRotationVelocity = 1.0;
  settings->maxLinearAcceleration = 0.0;
  settings->maxAngularAcceleration = 0.0;
  settings->velocityCommandFilterBreakFrequency = 0.0;
  EXPECT_EQ((*manager)->applyCommandLimits(*settings), absl::OkStatus());
  return *std::move(manager);
}

/**
 * The initial state of `stack` with its base at `position` in the ground plane and at the heading `heading`, its
 * normalized momentum h = A(q) v / m that of the Pinocchio velocity `velocity` ([dp_W, w_B, dq_j]): the momentum a
 * measurement of that motion carries, as the MRT computes it.
 */
vector_t movingAt(const AtlasReferenceStack& stack, const vector2_t& position, scalar_t heading, const vector_t& velocity) {
  const PinocchioInterface::Model& pinocchioModel = stack.pinocchioInterface().getModel();
  PinocchioInterface::Data data(pinocchioModel);
  vector_t state = stack.standingAt(stack.initialState(), position(0), position(1), heading);
  pinocchio::computeCentroidalMap(pinocchioModel, data, stack.model().getGeneralizedCoordinates(state));
  state.head<6>() = data.Ag * velocity / stack.centroidalModelInfo().robotMass;
  return state;
}

/** movingAt() with the base over the origin. */
vector_t movingAt(const AtlasReferenceStack& stack, scalar_t heading, const vector_t& velocity) {
  return movingAt(stack, vector2_t::Zero(), heading, velocity);
}

/** The input of an observation whose joints move at `jointVelocities`, as the MRT writes it; zero contact wrenches. */
vector_t measuredInput(const AtlasReferenceStack& stack, const vector_t& jointVelocities) {
  vector_t input = vector_t::Zero(static_cast<Eigen::Index>(stack.model().getInputDim()));
  vector_t unusedState = stack.initialState();
  stack.model().setJointVelocities(unusedState, input, jointVelocities);
  return input;
}

/**
 * One solve of `manager` at `time`, as the MPC node runs it: the operator turning at `commandedYawRate`, and the robot
 * measured at `state` with the joint velocities of `input`, handed over before the solve (observe()).
 */
void solveAt(ProceduralMpcMotionManager& manager,
             AtlasReferenceStack& stack,
             scalar_t time,
             scalar_t commandedYawRate,
             const vector_t& state,
             const vector_t& input) {
  manager.setAndScaleVelocityCommand(WalkingVelocityCommand(/*v_x=*/0.0, /*v_y=*/0.0, /*desired_pelvis_h=*/0.0, commandedYawRate));
  SystemObservation observation;
  observation.time = time;
  observation.state = state;
  observation.input = input;
  manager.observe(observation);
  manager.preSolverRun(time, time + stack.horizon(), state, stack.referenceManager());
}

/**
 * A Pinocchio velocity [dp_W, w_B, dq_j] of the base turning about the world z axis at `yawRate`, the joints still: a
 * level base's body z axis is the world's, so the rate is w_B,z (entry 5).
 */
vector_t turningBase(const AtlasReferenceStack& stack, scalar_t yawRate) {
  vector_t velocity = vector_t::Zero(stack.pinocchioInterface().getModel().nv);
  velocity(5) = yawRate;
  return velocity;
}

/** A measured robot: its state and the joint velocities of its observation. Passive data. */
struct MeasuredRobot {
  vector_t state;
  vector_t jointVelocities;
};

/**
 * The robot at `heading` with its base turning at `baseYawRate` and its joints swinging so that, locked, the body would
 * turn `lockedYawRateOffset` faster than the base: the joint velocities along the gradient of the yaw momentum, cut to
 * the null space of the linear momentum so that the CoM does not move with them (no linear gait clause fires).
 */
MeasuredRobot swingingJoints(const AtlasReferenceStack& stack, scalar_t heading, scalar_t baseYawRate, scalar_t lockedYawRateOffset) {
  const PinocchioInterface::Model& pinocchioModel = stack.pinocchioInterface().getModel();
  PinocchioInterface::Data data(pinocchioModel);
  const Eigen::Index joints = static_cast<Eigen::Index>(stack.model().getJointDim());
  const vector_t standing = stack.standingAt(stack.initialState(), /*x=*/0.0, /*y=*/0.0, heading);
  pinocchio::computeCentroidalMap(pinocchioModel, data, stack.model().getGeneralizedCoordinates(standing));
  const matrix_t linearJoints = data.Ag.topRows<3>().rightCols(joints);
  const vector_t yawGradient = data.Ag.row(5).tail(joints).transpose();
  const vector_t direction = yawGradient - linearJoints.completeOrthogonalDecomposition().pseudoInverse() * (linearJoints * yawGradient);
  // The locked body's extra yaw rate is linear in the joint velocities: scale the direction to the offset asked for.
  vector_t velocity = turningBase(stack, baseYawRate);
  velocity.tail(joints) = direction;
  const scalar_t unitOffset =
      stack.model().getBaseTwistInWorldFrame(movingAt(stack, heading, velocity), vector_t::Zero(joints))(5) - baseYawRate;
  velocity.tail(joints) = direction * (lockedYawRateOffset / unitOffset);
  return MeasuredRobot{.state = movingAt(stack, heading, velocity), .jointVelocities = velocity.tail(joints)};
}

/**
 * Solves every 10 ms over (from, to] with the robot's base turning at `yawRate` from `heading` (updated), its joints still,
 * and the operator turning at `commandedYawRate`.
 */
void turnFor(ProceduralMpcMotionManager& manager,
             AtlasReferenceStack& stack,
             scalar_t from,
             scalar_t to,
             scalar_t yawRate,
             scalar_t commandedYawRate,
             scalar_t& heading) {
  const vector_t stillJoints = vector_t::Zero(static_cast<Eigen::Index>(stack.model().getJointDim()));
  for (scalar_t time = from + 0.01; time <= to + 1.0e-9; time += 0.01) {
    heading += 0.01 * yawRate;
    solveAt(manager, stack, time, commandedYawRate, movingAt(stack, heading, turningBase(stack, yawRate)),
            measuredInput(stack, stillJoints));
  }
}

/** [s] slow_walk's cycle in the shipped gait file, the window its stride means are over. */
constexpr scalar_t kSlowWalkCycle = 1.7;

/**
 * A robot walking straight along the world x axis from `start` on, from (startX, startY): forward at `meanSpeed` plus a
 * sway `forwardSway` sin(w t), and sideways at `lateralSway` sin(w t), w = 2 pi / kSlowWalkCycle; its base at the
 * integral of that, its momentum that of the motion with the joints still. Passive data.
 */
struct StraightWalk {
  scalar_t start = 0.0;
  scalar_t startX = 0.0;
  scalar_t startY = 0.0;
  scalar_t meanSpeed = 0.0;
  scalar_t forwardSway = 0.0;
  scalar_t lateralSway = 0.0;
};

/** The base of `walk` at `time`: its position (x, y) and its instantaneous velocity (v_x, v_y). Passive data. */
struct WalkingBase {
  vector2_t position = vector2_t::Zero();
  vector2_t velocity = vector2_t::Zero();
};

WalkingBase walkingBaseAt(const StraightWalk& walk, scalar_t time) {
  const scalar_t frequency = 2.0 * M_PI / kSlowWalkCycle;
  const scalar_t swayIntegral = (std::cos(frequency * walk.start) - std::cos(frequency * time)) / frequency;
  return WalkingBase{
      .position = vector2_t(walk.startX + walk.meanSpeed * (time - walk.start) + walk.forwardSway * swayIntegral,
                            walk.startY + walk.lateralSway * swayIntegral),
      .velocity = vector2_t(walk.meanSpeed + walk.forwardSway * std::sin(frequency * time), walk.lateralSway * std::sin(frequency * time))};
}

/** `walk` continued from `time` on at the mean speed `meanSpeed` without sway. */
StraightWalk continuedAt(const StraightWalk& walk, scalar_t time, scalar_t meanSpeed) {
  const WalkingBase base = walkingBaseAt(walk, time);
  return StraightWalk{.start = time, .startX = base.position(0), .startY = base.position(1), .meanSpeed = meanSpeed};
}

/**
 * Solves every 10 ms over (from, to] with the operator commanding `command` ([v_x, v_y, pelvis height, yaw rate]) and
 * the robot on `walk`, each observation handed over before its solve; returns the gait of every solve.
 */
std::vector<std::string> walkFor(ProceduralMpcMotionManager& manager,
                                 AtlasReferenceStack& stack,
                                 scalar_t from,
                                 scalar_t to,
                                 const vector4_t& command,
                                 const StraightWalk& walk) {
  const vector_t stillJoints = vector_t::Zero(static_cast<Eigen::Index>(stack.model().getJointDim()));
  manager.setAndScaleVelocityCommand(WalkingVelocityCommand(command(0), command(1), command(2), command(3)));
  std::vector<std::string> gaits;
  for (scalar_t time = from + 0.01; time <= to + 1.0e-9; time += 0.01) {
    const WalkingBase base = walkingBaseAt(walk, time);
    vector_t velocity = vector_t::Zero(stack.pinocchioInterface().getModel().nv);
    velocity.head<2>() = base.velocity;
    const vector_t state = movingAt(stack, base.position, /*heading=*/0.0, velocity);
    SystemObservation observation;
    observation.time = time;
    observation.state = state;
    observation.input = measuredInput(stack, stillJoints);
    manager.observe(observation);
    manager.preSolverRun(time, time + stack.horizon(), state, stack.referenceManager());
    gaits.push_back(manager.getCurrentGaitCommand());
  }
  return gaits;
}

/** Whether every entry of `gaits` is `gait`. */
bool allAre(const std::vector<std::string>& gaits, const std::string& gait) {
  return std::all_of(gaits.begin(), gaits.end(), [&gait](const std::string& entry) { return entry == gait; });
}

TEST(ProceduralMpcMotionManagerCreate, BuildsTheManagerOfTheShippedFiles) {
  AtlasReferenceStack stack;
  const absl::StatusOr<std::unique_ptr<ProceduralMpcMotionManager>> manager = create(stack, stack.gaitFile(), stack.referenceFile());
  ASSERT_TRUE(manager.ok()) << manager.status();
  EXPECT_EQ((*manager)->getCurrentGaitCommand(), "stance");
}

TEST(ProceduralMpcMotionManagerCreate, AGaitFileWithoutAGaitTheCommandSelectsIsRefusedNamingIt) {
  AtlasReferenceStack stack;
  const std::string stanceOnly = writeTempFile(
      "stance_only_gait.textproto",
      "gait_list: \"stance\"\ngaits { name: \"stance\" mode_sequence: \"STANCE\" switching_times: 0.0 switching_times: 0.5 }\n");
  const absl::StatusOr<std::unique_ptr<ProceduralMpcMotionManager>> manager = create(stack, stanceOnly, stack.referenceFile());
  EXPECT_EQ(manager.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(manager.status().message(), "has no gait")) << manager.status();
  EXPECT_TRUE(absl::StrContains(manager.status().message(), stanceOnly)) << manager.status();
}

TEST(ProceduralMpcMotionManagerCreate, AReferenceFileWhoseLimitDoesNotParseIsRefusedNamingTheFile) {
  AtlasReferenceStack stack;
  mpc_config::ReferenceFile reference = stack.config().reference;
  reference.max_displacement_velocity_x = 0.123;
  const std::string text = referenceFileText(reference);
  const std::string unparsable = absl::StrReplaceAll(text, {{"max_displacement_velocity_x: 0.123", "max_displacement_velocity_x: fast"}});
  ASSERT_NE(unparsable, text) << "the reference file's text no longer writes max_displacement_velocity_x as expected";
  const std::string broken = writeTempFile("broken_limit_reference.textproto", unparsable);
  const absl::StatusOr<std::unique_ptr<ProceduralMpcMotionManager>> manager = create(stack, stack.gaitFile(), broken);
  EXPECT_EQ(manager.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(manager.status().message(), broken)) << manager.status();
}

TEST(ProceduralMpcMotionManagerReload, AnAccelerationLimitThatIsNotANumberIsRefusedAndKeepsTheRamp) {
  AtlasReferenceStack stack;
  const scalar_t before = stack.motionManager().getMaxLinearAcceleration();
  // A reload converts the file first (MpcParameterUpdaterModule), which refuses the value by its field, so that nothing
  // of the file is applied ...
  mpc_config::ReferenceFile reference = stack.config().reference;
  reference.max_linear_acceleration = std::numeric_limits<scalar_t>::quiet_NaN();
  const absl::StatusOr<ReferenceSettings> converted = referenceSettingsFromConfig(reference);
  EXPECT_EQ(converted.status().code(), absl::StatusCode::kInvalidArgument) << converted.status();
  EXPECT_TRUE(absl::StrContains(converted.status().message(), "max_linear_acceleration")) << converted.status();
  // ... and the manager refuses the limit itself when it is handed one, keeping the ramp it runs.
  absl::StatusOr<ReferenceSettings> settings = referenceSettingsFromConfig(stack.config().reference);
  ASSERT_TRUE(settings.ok()) << settings.status();
  settings->maxLinearAcceleration = std::numeric_limits<scalar_t>::quiet_NaN();
  const absl::Status applied = stack.motionManager().applyCommandLimits(*settings);
  EXPECT_EQ(applied.code(), absl::StatusCode::kInvalidArgument) << applied;
  EXPECT_EQ(stack.motionManager().getMaxLinearAcceleration(), before) << "a refused reload switched the ramp off";
}

TEST(ProceduralMpcMotionManagerGaits, AFasterGaitWaitsForTheRobotsMeanTurnOverAStride) {
  AtlasReferenceStack stack;
  const std::unique_ptr<ProceduralMpcMotionManager> manager = unfilteredManager(stack);
  ASSERT_NE(manager, nullptr);
  // Out of stance on a turn command alone, then asked to turn faster than slow_walk allows (0.2 rad/s): the next gait
  // once the robot's mean turn over a stride exceeds 0.2 - 0.05 rad/s, and not before.
  scalar_t heading = 0.0;
  const vector_t stillJoints = vector_t::Zero(static_cast<Eigen::Index>(stack.model().getJointDim()));
  solveAt(*manager, stack, /*time=*/1.0, /*commandedYawRate=*/0.15, stack.initialState(), measuredInput(stack, stillJoints));
  ASSERT_EQ(manager->getCurrentGaitCommand(), "slow_walk");
  turnFor(*manager, stack, /*from=*/1.0, /*to=*/4.0, /*yawRate=*/0.1, /*commandedYawRate=*/0.3, heading);
  EXPECT_EQ(manager->getCurrentGaitCommand(), "slow_walk");
  EXPECT_NEAR(manager->getMeasuredPlanarMotion().headingRate, 0.1, 1.0e-9);
  EXPECT_EQ(manager->getMeasuredMotionWindow(), 1.7) << "slow_walk's cycle, which the schedule runs by now";
  turnFor(*manager, stack, /*from=*/4.0, /*to=*/7.0, /*yawRate=*/0.18, /*commandedYawRate=*/0.3, heading);
  EXPECT_EQ(manager->getCurrentGaitCommand(), "walk");
}

TEST(ProceduralMpcMotionManagerGaits, TheJointsMomentumIsNotATurn) {
  // The joints swing, the base stands still. The centroidal momentum carries the joints'
  // angular momentum, which the twist with the joints locked read as a turn of the base faster than slow_walk's 0.15
  // rad/s; the base's heading does not move, so its mean turn is zero and the gait stays.
  AtlasReferenceStack stack;
  const std::unique_ptr<ProceduralMpcMotionManager> manager = unfilteredManager(stack);
  ASSERT_NE(manager, nullptr);
  const Eigen::Index joints = static_cast<Eigen::Index>(stack.model().getJointDim());
  solveAt(*manager, stack, /*time=*/1.0, /*commandedYawRate=*/0.15, stack.initialState(), measuredInput(stack, vector_t::Zero(joints)));
  ASSERT_EQ(manager->getCurrentGaitCommand(), "slow_walk");
  const MeasuredRobot swinging = swingingJoints(stack, /*heading=*/0.0, /*baseYawRate=*/0.0, /*lockedYawRateOffset=*/0.5);
  ASSERT_NEAR(stack.model().getBaseTwistInWorldFrame(swinging.state, vector_t::Zero(joints))(5), 0.5, 1.0e-9)
      << "locked, the body turns at 0.5 rad/s, which would have raised the gait";
  EXPECT_LT(stack.model().getBaseTwistInWorldFrame(swinging.state, swinging.jointVelocities).norm(), 1.0e-9) << "the base stands still";
  ASSERT_LT(swinging.state.head<3>().norm(), 1.0e-9) << "the CoM stands still";
  for (scalar_t time = 1.01; time <= 5.0; time += 0.01) {
    solveAt(*manager, stack, time, /*commandedYawRate=*/0.3, swinging.state, measuredInput(stack, swinging.jointVelocities));
  }
  EXPECT_EQ(manager->getCurrentGaitCommand(), "slow_walk");
  EXPECT_LT(std::abs(manager->getMeasuredPlanarMotion().headingRate), 1.0e-12);
}

TEST(ProceduralMpcMotionManagerGaits, ARestartedHistoryStartsFromTheObservationsExactTwist) {
  // The first solve after a reset has no history: the motion before it is the base twist of the observation, with its
  // measured joint velocities. The base turns at 0.4 rad/s while the joints swing; locked, the body would turn at 0.9.
  AtlasReferenceStack stack;
  const std::unique_ptr<ProceduralMpcMotionManager> manager = unfilteredManager(stack);
  ASSERT_NE(manager, nullptr);
  const MeasuredRobot swinging = swingingJoints(stack, /*heading=*/0.7, /*baseYawRate=*/0.4, /*lockedYawRateOffset=*/0.5);
  solveAt(*manager, stack, /*time=*/3.0, /*commandedYawRate=*/0.0, swinging.state, measuredInput(stack, swinging.jointVelocities));
  EXPECT_NEAR(manager->getMeasuredPlanarMotion().headingRate, 0.4, 1.0e-9);
  EXPECT_LT(manager->getMeasuredPlanarMotion().velocityInHeadingFrame.norm(), 1.0e-9);
  // The targets read it; the gait switches do not: one instant's twist is no measurement of the stride.
  EXPECT_FALSE(manager->measuredOverTheWholeWindow());

  // Without the observation of the solve the history starts at rest (and warns).
  manager->reset();
  manager->setAndScaleVelocityCommand(WalkingVelocityCommand(/*v_x=*/0.0, /*v_y=*/0.0, /*desired_pelvis_h=*/0.0, /*v_yaw=*/0.0));
  manager->preSolverRun(/*initTime=*/3.0, /*finalTime=*/3.0 + stack.horizon(), swinging.state, stack.referenceManager());
  EXPECT_EQ(manager->getMeasuredPlanarMotion().headingRate, 0.0);
}

TEST(ProceduralMpcMotionManagerGaits, TheGaitSwitchesDoNotReadARestartedHistorysSeed) {
  // Out of stance at t = 1 on the command alone (a turn of 0.3 rad/s, beyond slow_walk's 0.2), the history starting
  // from an observation whose base turns at 0.5 rad/s: one instant of it, for the robot holds its heading from then on.
  // The window blends that seed in (above slow_walk's 0.15 rad/s for a while), and the targets read it; the switches do
  // not, until the window is measured throughout, and then the robot does not turn: the gait stays.
  AtlasReferenceStack stack;
  const std::unique_ptr<ProceduralMpcMotionManager> manager = unfilteredManager(stack);
  ASSERT_NE(manager, nullptr);
  const vector_t stillJoints = vector_t::Zero(static_cast<Eigen::Index>(stack.model().getJointDim()));
  solveAt(*manager, stack, /*time=*/1.0, /*commandedYawRate=*/0.3, movingAt(stack, /*heading=*/0.0, turningBase(stack, /*yawRate=*/0.5)),
          measuredInput(stack, stillJoints));
  ASSERT_EQ(manager->getCurrentGaitCommand(), "slow_walk");
  scalar_t largestUnmeasuredTurn = 0.0;
  scalar_t firstMeasured = -1.0;
  for (scalar_t time = 1.01; time <= 6.0 + 1.0e-9; time += 0.01) {
    solveAt(*manager, stack, time, /*commandedYawRate=*/0.3, stack.initialState(), measuredInput(stack, stillJoints));
    EXPECT_EQ(manager->getCurrentGaitCommand(), "slow_walk") << "t = " << time;
    if (manager->measuredOverTheWholeWindow()) {
      if (firstMeasured < 0.0) firstMeasured = time;
      EXPECT_LT(std::abs(manager->getMeasuredPlanarMotion().headingRate), 1.0e-12) << "t = " << time;
    } else if (time > 1.0 + 0.2) {
      // After the gait-change hold-off, where the seed alone would have decided.
      largestUnmeasuredTurn = std::max(largestUnmeasuredTurn, manager->getMeasuredPlanarMotion().headingRate);
    }
  }
  EXPECT_GT(largestUnmeasuredTurn, 0.15) << "the seed must read as a turn that would have raised slow_walk";
  // stance runs until slow_walk starts: its 0.5 s window is measured from t = 1.5 on.
  EXPECT_NEAR(firstMeasured, 1.5, 0.011);
}

/** [s] When the robots of the tests below set off: after slow_walk, which the first solve's command selects, starts. */
constexpr scalar_t kWalkStart = 2.5;

TEST(ProceduralMpcMotionManagerGaits, ASlowerGaitWaitsForTheMeanSpeedNotForASwayingDip) {
  // The linear clauses of the gait switch (design README section 2.9). Told to stop while the robot still walks forward at 0.2 m/s on the
  // mean, swaying by 0.18 m/s with the cycle: its instantaneous speed dips to 0.02 m/s, below slow_walk's 0.1, every stride, which let the
  // gait down into stance under a walking robot. The stride mean does not dip; once the robot stops, it falls below 0.1 and the gait goes.
  AtlasReferenceStack stack;
  const std::unique_ptr<ProceduralMpcMotionManager> manager = unfilteredManager(stack);
  ASSERT_NE(manager, nullptr);
  const vector4_t walkSlowly(0.2, 0.0, 0.0, 0.0);
  walkFor(*manager, stack, /*from=*/0.99, kWalkStart, walkSlowly, StraightWalk{});
  ASSERT_EQ(manager->getCurrentGaitCommand(), "slow_walk");
  const StraightWalk walking{.start = kWalkStart, .meanSpeed = 0.2, .forwardSway = 0.18};
  scalar_t slowestSpeed = walking.meanSpeed;
  for (scalar_t time = kWalkStart; time <= kWalkStart + kSlowWalkCycle; time += 0.01) {
    slowestSpeed = std::min(slowestSpeed, walkingBaseAt(walking, time).velocity(0));
  }
  ASSERT_LT(slowestSpeed, 0.1) << "the instantaneous speed must dip below slow_walk's threshold";
  walkFor(*manager, stack, kWalkStart, /*to=*/6.0, walkSlowly, walking);
  ASSERT_EQ(manager->getCurrentGaitCommand(), "slow_walk");
  EXPECT_TRUE(allAre(walkFor(*manager, stack, /*from=*/6.0, /*to=*/8.0, vector4_t::Zero(), walking), "slow_walk"))
      << "a walking robot was let down into stance";
  EXPECT_TRUE(manager->measuredOverTheWholeWindow());
  EXPECT_NEAR(manager->getMeasuredPlanarMotion().velocityInHeadingFrame(0), 0.2, 1.0e-6);
  // It stops where it is. A window that still holds part of a stride reads part of its sway, up to 2 (0.18 / w) / W =
  // 0.057 m/s; 0.3 s after the stop the mean is still above 0.2 (1.7 - 0.3) / 1.7 - 0.057 > 0.1.
  const std::vector<std::string> stopping =
      walkFor(*manager, stack, /*from=*/8.0, /*to=*/10.0, vector4_t::Zero(), continuedAt(walking, /*time=*/8.0, /*meanSpeed=*/0.0));
  EXPECT_EQ(stopping.back(), "stance");
  const std::vector<std::string>::const_iterator firstStance = std::find(stopping.begin(), stopping.end(), "stance");
  EXPECT_GE(0.01 * static_cast<scalar_t>(firstStance - stopping.begin()), 0.3);
}

TEST(ProceduralMpcMotionManagerGaits, AFasterGaitWaitsForTheMeanSpeedNotForASwayingPeak) {
  // Asked to walk faster than slow_walk allows while the robot walks forward at 0.2 m/s on the mean, swaying sideways by
  // 0.3 m/s: its instantaneous sideways speed passes slow_walk's 0.3 - 0.05 every stride, which raised the gait. The
  // stride mean is 0.2 m/s forward and none sideways; once the robot speeds up to 0.3 m/s the gait rises.
  AtlasReferenceStack stack;
  const std::unique_ptr<ProceduralMpcMotionManager> manager = unfilteredManager(stack);
  ASSERT_NE(manager, nullptr);
  const vector4_t faster(0.4, 0.0, 0.0, 0.0);
  EXPECT_TRUE(allAre(walkFor(*manager, stack, /*from=*/0.99, kWalkStart, faster, StraightWalk{}), "slow_walk"))
      << "out of stance on the command alone, and no further at rest";
  const StraightWalk swaying{.start = kWalkStart, .meanSpeed = 0.2, .lateralSway = 0.3};
  EXPECT_TRUE(allAre(walkFor(*manager, stack, kWalkStart, /*to=*/6.0, faster, swaying), "slow_walk")) << "the sway raised the gait";
  EXPECT_NEAR(manager->getMeasuredPlanarMotion().velocityInHeadingFrame(1), 0.0, 1.0e-6);
  const std::vector<std::string> speedingUp =
      walkFor(*manager, stack, /*from=*/6.0, /*to=*/9.0, faster, continuedAt(swaying, /*time=*/6.0, /*meanSpeed=*/0.3));
  EXPECT_EQ(speedingUp.back(), "walk");
}

TEST(ProceduralMpcMotionManagerGaits, ARollIsNotATurn) {
  AtlasReferenceStack stack;
  const std::unique_ptr<ProceduralMpcMotionManager> manager = unfilteredManager(stack);
  ASSERT_NE(manager, nullptr);
  const vector_t stillJoints = vector_t::Zero(static_cast<Eigen::Index>(stack.model().getJointDim()));
  solveAt(*manager, stack, /*time=*/1.0, /*commandedYawRate=*/0.15, stack.initialState(), measuredInput(stack, stillJoints));
  ASSERT_EQ(manager->getCurrentGaitCommand(), "slow_walk");
  // Rolling back and forth by 0.1 rad at level heading: the twist heading of a roll at zero pitch stays the heading.
  for (scalar_t time = 1.01; time <= 5.0; time += 0.01) {
    vector_t rolling = stack.initialState();
    stack.model().setBaseOrientationEulerZyx(rolling, vector3_t(0.0, 0.0, 0.1 * std::sin(6.0 * time)));
    solveAt(*manager, stack, time, /*commandedYawRate=*/0.3, rolling, measuredInput(stack, stillJoints));
  }
  EXPECT_EQ(manager->getCurrentGaitCommand(), "slow_walk");
  EXPECT_LT(std::abs(manager->getMeasuredPlanarMotion().headingRate), 1.0e-12);
}

TEST(ProceduralMpcMotionManagerGaits, ASlowerGaitWaitsForTheMeanTurnToStop) {
  AtlasReferenceStack stack;
  const std::unique_ptr<ProceduralMpcMotionManager> manager = unfilteredManager(stack);
  ASSERT_NE(manager, nullptr);
  scalar_t heading = 0.0;
  const vector_t stillJoints = vector_t::Zero(static_cast<Eigen::Index>(stack.model().getJointDim()));
  solveAt(*manager, stack, /*time=*/1.0, /*commandedYawRate=*/0.15, stack.initialState(), measuredInput(stack, stillJoints));
  ASSERT_EQ(manager->getCurrentGaitCommand(), "slow_walk");
  // Told to stop while still turning at 0.3 rad/s: the gait holds until the mean turn over the stride drops below 0.05 +
  // 0.05 rad/s, which takes most of a stride after the turn stops.
  turnFor(*manager, stack, /*from=*/1.0, /*to=*/4.0, /*yawRate=*/0.3, /*commandedYawRate=*/0.0, heading);
  EXPECT_EQ(manager->getCurrentGaitCommand(), "slow_walk");
  turnFor(*manager, stack, /*from=*/4.0, /*to=*/4.5, /*yawRate=*/0.0, /*commandedYawRate=*/0.0, heading);
  EXPECT_EQ(manager->getCurrentGaitCommand(), "slow_walk") << "half a stride after the turn stopped its mean is still above";
  turnFor(*manager, stack, /*from=*/4.5, /*to=*/6.5, /*yawRate=*/0.0, /*commandedYawRate=*/0.0, heading);
  EXPECT_EQ(manager->getCurrentGaitCommand(), "stance");
}

}  // namespace
}  // namespace ocs2::humanoid
