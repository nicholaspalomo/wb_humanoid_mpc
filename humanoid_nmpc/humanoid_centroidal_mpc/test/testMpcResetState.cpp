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
#include <cstdint>
#include <functional>
#include <optional>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_centroidal_model/AccessHelperFunctions.h"

#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerFactory.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "support/AtlasReferenceStack.h"

/*
 * What a reset of the MPC leaves behind in the references, on the DRC Atlas: the reference manager, its gait schedule
 * and latched measurements, the contact planner, the motion manager and the target calculator.
 *
 * A reset used to reset the solver's warm start and nothing else. After a fall, a reset of the simulator or a clock
 * that ran backwards the controller kept the gait it was walking in, events timed on the old clock that held the robot
 * standing until the clock caught up with them, the lift-off positions of the feet where they were before (asking a
 * swing foot for 20 m/s after the robot was put back at the origin), the contact plan and the command ramps.
 *
 * The properties are that a reset stack answers exactly as a stack built afresh, and the behavior that follows from
 * it; none of them depends on a tuned value.
 */

namespace ocs2::humanoid {
namespace {

using ScheduleSource = AtlasReferenceStack::ScheduleSource;

constexpr scalar_t kSolvePeriod = 0.01;  // [s]

/** The robot walking: its state `x` moved on at the commanded speed, which its center of mass also carries. */
vector_t walkingState(const AtlasReferenceStack& stack, const vector_t& state, scalar_t speed, scalar_t dt) {
  vector_t moved = state;
  centroidal_model::getNormalizedMomentum(moved, stack.centroidalModelInfo()).setZero();
  centroidal_model::getNormalizedMomentum(moved, stack.centroidalModelInfo())(0) = speed;
  moved(6) += speed * dt;
  return moved;
}

/** Runs the stack's MPC - the reference manager and the modules before a scripted solve - over [time, time + duration). */
void walk(AtlasReferenceStack& stack, scalar_t& time, vector_t& state, scalar_t speed, scalar_t duration) {
  const scalar_t end = time + duration;
  while (time < end - 1.0e-9) {
    stack.mpc().run(time, state, ModeNumber::kStance);
    state = walkingState(stack, state, speed, kSolvePeriod);
    time += kSolvePeriod;
  }
}

/** Whether the reference manager's mode schedule has a swing anywhere in [from, to]. */
bool scheduleSwingsIn(const AtlasReferenceStack& stack, const SwitchedModelReferenceManager& referenceManager, scalar_t from, scalar_t to) {
  (void)stack;
  for (scalar_t time = from; time <= to + 1.0e-9; time += 0.005) {
    if (!referenceManager.isInStancePhase(time)) return true;
  }
  return false;
}

/** [m/s] The Atlas command limit along x: the raw command that asks for `speed`. */
scalar_t rawCommandFor(scalar_t speed) {
  return speed / 1.2;  // the reference file's max_displacement_velocity_x
}

/** Everything the solver would read from the two stacks, compared. */
void expectSameReferences(AtlasReferenceStack& used, AtlasReferenceStack& fresh, scalar_t time, const std::string& when) {
  SwitchedModelReferenceManager& a = used.referenceManager();
  SwitchedModelReferenceManager& b = fresh.referenceManager();
  EXPECT_EQ(a.getModeSchedule().modeSequence, b.getModeSchedule().modeSequence) << when;
  EXPECT_EQ(a.getModeSchedule().eventTimes, b.getModeSchedule().eventTimes) << when;

  const TargetTrajectories& targetA = a.getTargetTrajectories();
  const TargetTrajectories& targetB = b.getTargetTrajectories();
  ASSERT_EQ(targetA.timeTrajectory, targetB.timeTrajectory) << when;
  for (size_t k = 0; k < targetA.stateTrajectory.size(); ++k) {
    EXPECT_LT((targetA.stateTrajectory[k] - targetB.stateTrajectory[k]).cwiseAbs().maxCoeff(), 1.0e-9) << when << ", knot " << k;
  }

  for (int step = 0; step <= 40; ++step) {
    const scalar_t query = time + 0.05 * step;
    EXPECT_EQ(a.getContactFlags(query), b.getContactFlags(query)) << when << ", t = " << query;
    for (size_t foot = 0; foot < kNumContacts; ++foot) {
      const std::optional<SwingFootReference> referenceA = a.getSwingFootReference(foot, query);
      const std::optional<SwingFootReference> referenceB = b.getSwingFootReference(foot, query);
      ASSERT_EQ(referenceA.has_value(), referenceB.has_value()) << when << ", foot " << foot << ", t = " << query;
      if (!referenceA.has_value() || !referenceB.has_value()) continue;
      EXPECT_LT((referenceA->position - referenceB->position).norm(), 1.0e-9) << when << ", foot " << foot << ", t = " << query;
      EXPECT_LT((referenceA->linearVelocity - referenceB->linearVelocity).norm(), 1.0e-9) << when << ", foot " << foot << ", t = " << query;
    }
  }
  EXPECT_EQ(used.motionManager().getCurrentGaitCommand(), fresh.motionManager().getCurrentGaitCommand()) << when;
  if (used.planningReferenceManager() != nullptr) {
    EXPECT_EQ(used.planningReferenceManager()->hasActivePlan(), fresh.planningReferenceManager()->hasActivePlan()) << when;
    const feet_array_t<TargetContactPose> posesA = used.planningReferenceManager()->getTargetContactPoses();
    const feet_array_t<TargetContactPose> posesB = fresh.planningReferenceManager()->getTargetContactPoses();
    for (size_t foot = 0; foot < kNumContacts; ++foot) {
      EXPECT_EQ(posesA[foot].valid, posesB[foot].valid) << when << ", foot " << foot;
      if (posesA[foot].valid && posesB[foot].valid) {
        EXPECT_LT((posesA[foot].position - posesB[foot].position).norm(), 1.0e-9) << when << ", foot " << foot;
      }
    }
  }
}

/** How a test resets the MPC of a stack. */
using ResetFunction = std::function<void(AtlasReferenceStack&)>;

/** MPC_BASE::reset(), as the controllers run it. */
void resetMpc(AtlasReferenceStack& stack) {
  stack.mpc().reset();
}

/**
 * The core property: a stack that walked and was then reset (by `reset`) answers, solve after solve, exactly as a stack
 * built at that moment, for the same observations and the same command, [m/s] `commandAfterReset` forward. Under the
 * contact planner `plannerType` names the planner (empty: the shipped one).
 */
void resetStackBehavesAsAFreshOne(ScheduleSource source,
                                  const ResetFunction& reset,
                                  scalar_t commandAfterReset = 0.0,
                                  absl::string_view plannerType = "") {
  AtlasReferenceStack used(source, plannerType);
  AtlasReferenceStack fresh(source, plannerType);

  // Walking forward for 6 s.
  scalar_t time = 0.0;
  vector_t state = used.initialState();
  used.referenceManager().setTargetTrajectories(used.resetTarget(time, state));
  used.command(rawCommandFor(1.0));
  walk(used, time, state, /*speed=*/1.0, /*duration=*/6.0);
  ASSERT_TRUE(scheduleSwingsIn(used, used.referenceManager(), time, time + used.horizon()))
      << "positive control: the stack is walking before the reset";

  // The robot is caught and put back elsewhere, standing; the controller resets the MPC from there, as it does at start-up.
  const vector_t held = used.standingAt(used.initialState(), /*x=*/3.0, /*y=*/-1.0, /*yaw=*/0.4);
  reset(used);
  used.referenceManager().setTargetTrajectories(used.resetTarget(time, held));
  fresh.referenceManager().setTargetTrajectories(fresh.resetTarget(time, held));
  used.command(rawCommandFor(commandAfterReset));
  fresh.command(rawCommandFor(commandAfterReset));

  for (int solve = 0; solve < 50; ++solve) {
    used.mpc().run(time, held, ModeNumber::kStance);
    fresh.mpc().run(time, held, ModeNumber::kStance);
    expectSameReferences(used, fresh, time, absl::StrCat("solve ", solve, " after the reset"));
    if (testing::Test::HasFatalFailure()) return;
    time += kSolvePeriod;
  }
}

TEST(MpcResetState, AResetGaitScheduleStackAnswersExactlyAsAFreshOne) {
  resetStackBehavesAsAFreshOne(ScheduleSource::kGaitSchedule, resetMpc);
}

TEST(MpcResetState, AResetContactPlanningStackAnswersExactlyAsAFreshOne) {
  resetStackBehavesAsAFreshOne(ScheduleSource::kContactPlanner, resetMpc);
}

TEST(MpcResetState, AResetStackWalksExactlyAsAFreshOne) {
  // Commanded to walk at once after the reset: the gait decisions, the schedule and, under the contact planner, the steps
  // it places are those of a fresh stack too - also under the mixed-integer planner, which carries its previous plan and
  // assignment from one plan to the next.
  resetStackBehavesAsAFreshOne(ScheduleSource::kGaitSchedule, resetMpc, /*commandAfterReset=*/1.0);
  resetStackBehavesAsAFreshOne(ScheduleSource::kContactPlanner, resetMpc, /*commandAfterReset=*/1.0);
  resetStackBehavesAsAFreshOne(ScheduleSource::kContactPlanner, resetMpc, /*commandAfterReset=*/1.0, planner::kLipMiqp);
}

TEST(MpcResetState, TheContactPlannerIsResetByTheFirstSnapshotOfANewPlanEpoch) {
  // On the robot the planner may plan on a worker thread. A plan of the old epoch that is under way when the MPC is reset
  // leaves its warm start and previous plan in the planner AFTER the module's reset() has run, and the first plan after
  // the reset must still not start from them. So the planner's reset may depend on nothing but the plan epoch the
  // reference manager's reset advances: here the module's reset() is left out altogether, and the stack must still
  // answer as a fresh one. It used to be reset only through a request that the module's reset() raised, which such a
  // plan consumed.
  // The mixed-integer planner, the one that carries state (its previous plan and assignment), commanded to walk after the
  // reset so that it places steps: its previous plan, 6 m of walking away, pulls on them where it survives.
  resetStackBehavesAsAFreshOne(
      ScheduleSource::kContactPlanner,
      [](AtlasReferenceStack& stack) {
        // MPC_BASE::reset() but for ContactPlannerModule::reset().
        stack.referenceManager().reset();
        stack.motionManager().reset();
        stack.mpc().resetSolver();
      },
      /*commandAfterReset=*/1.0, planner::kLipMiqp);
}

TEST(MpcResetState, AfterAResetARewoundClockDoesNotHoldTheRobotStandingUntilItCatchesUp) {
  // Trotting until t = 13 s, then the clock is back at 2 s and the robot at the origin, and the controller resets the MPC
  // (MpcResetSupervisor::observeTime()). The reset used to leave the gait schedule's events at 13 s and beyond: every new
  // gait was inserted after them, and the gait-change hold-off ran until the clock was back at 13 s.
  AtlasReferenceStack stack;
  scalar_t time = 0.0;
  vector_t state = stack.initialState();
  stack.referenceManager().setTargetTrajectories(stack.resetTarget(time, state));
  stack.command(rawCommandFor(1.0));
  walk(stack, time, state, /*speed=*/1.0, /*duration=*/13.0);
  ASSERT_TRUE(scheduleSwingsIn(stack, stack.referenceManager(), time, time + stack.horizon()));

  time = 2.0;
  state = stack.initialState();
  stack.mpc().reset();
  stack.referenceManager().setTargetTrajectories(stack.resetTarget(time, state));
  bool swungWithinASecond = false;
  for (const scalar_t end = time + 1.0; time < end; time += kSolvePeriod) {
    stack.mpc().run(time, state, ModeNumber::kStance);
    state = walkingState(stack, state, /*speed=*/1.0, kSolvePeriod);
    swungWithinASecond = swungWithinASecond || scheduleSwingsIn(stack, stack.referenceManager(), time, time + stack.horizon());
  }
  EXPECT_TRUE(swungWithinASecond) << "the robot, commanded to walk, is held standing after the clock ran backwards";

  // Positive control: a stack built afresh and commanded to walk at 2 s swings within the same second, so the second is
  // enough.
  AtlasReferenceStack fresh;
  time = 2.0;
  state = fresh.initialState();
  fresh.referenceManager().setTargetTrajectories(fresh.resetTarget(time, state));
  fresh.command(rawCommandFor(1.0));
  bool freshSwungWithinASecond = false;
  for (const scalar_t end = time + 1.0; time < end; time += kSolvePeriod) {
    fresh.mpc().run(time, state, ModeNumber::kStance);
    state = walkingState(fresh, state, /*speed=*/1.0, kSolvePeriod);
    freshSwungWithinASecond = freshSwungWithinASecond || scheduleSwingsIn(fresh, fresh.referenceManager(), time, time + fresh.horizon());
  }
  EXPECT_TRUE(freshSwungWithinASecond);
}

TEST(MpcResetState, AfterAResetTheSwingFootStartsWhereTheFootIsNotWhereItWas) {
  // Walked 14 m from the origin, reset and put back at the origin: the first swing after the reset must start from the
  // foot as it stands now. The lift-off positions latched before the reset used to survive it, and the swing reference
  // then asked the foot to cover 14 m in one swing.
  AtlasReferenceStack stack;
  scalar_t time = 0.0;
  vector_t state = stack.initialState();
  stack.referenceManager().setTargetTrajectories(stack.resetTarget(time, state));
  stack.command(rawCommandFor(1.2));
  walk(stack, time, state, /*speed=*/1.2, /*duration=*/12.0);
  ASSERT_GT(state(6), 10.0);

  // The operator's sticks are centered when the robot is caught; the first solve after the reset stands.
  state = stack.initialState();
  stack.mpc().reset();
  stack.referenceManager().setTargetTrajectories(stack.resetTarget(time, state));
  stack.command(0.0);
  stack.mpc().run(time, state, ModeNumber::kStance);
  time += kSolvePeriod;
  EXPECT_FALSE(scheduleSwingsIn(stack, stack.referenceManager(), time, time + stack.horizon())) << "the reset restarts the gait in stance";

  // Walking again from the origin.
  stack.command(rawCommandFor(1.2));
  scalar_t maxSwingSpeed = 0.0;
  size_t swingReferences = 0;
  for (const scalar_t end = time + 4.0; time < end; time += kSolvePeriod) {
    stack.mpc().run(time, state, ModeNumber::kStance);
    for (size_t foot = 0; foot < kNumContacts; ++foot) {
      const std::optional<SwingFootReference> reference = stack.referenceManager().getSwingFootReference(foot, time + 0.05);
      if (!reference.has_value()) continue;
      ++swingReferences;
      maxSwingSpeed = std::max(maxSwingSpeed, reference->linearVelocity.head<2>().norm());
    }
    state = walkingState(stack, state, /*speed=*/1.2, kSolvePeriod);
  }
  ASSERT_GT(swingReferences, 0u) << "the robot never swung a foot after the reset, so nothing was checked";
  EXPECT_LT(maxSwingSpeed, 5.0) << "a swing reference asked for " << maxSwingSpeed << " m/s";
}

TEST(MpcResetState, EachMotionManagerSwitchesGaitsOnItsOwnThresholds) {
  // The thresholds of the current gait were a function-static copy shared by every motion manager in the process. Once
  // one manager trotted, a second one, in stance, judged its command against the trot's thresholds: a slow command
  // then asked it to slow down from stance, and its gait index ran below zero.
  AtlasReferenceStack trotting;
  scalar_t time = 0.0;
  vector_t state = trotting.initialState();
  trotting.referenceManager().setTargetTrajectories(trotting.resetTarget(time, state));
  trotting.command(rawCommandFor(1.2));
  walk(trotting, time, state, /*speed=*/1.2, /*duration=*/3.0);
  ASSERT_EQ(trotting.motionManager().getCurrentGaitCommand(), "trot") << "positive control: the first manager trots";

  AtlasReferenceStack standing;
  scalar_t standingTime = 0.0;
  vector_t standingState = standing.initialState();
  standing.referenceManager().setTargetTrajectories(standing.resetTarget(standingTime, standingState));
  standing.command(rawCommandFor(0.2));
  walk(standing, standingTime, standingState, /*speed=*/0.2, /*duration=*/0.5);
  EXPECT_EQ(standing.motionManager().getCurrentGaitCommand(), "slow_walk") << "a slow command from stance starts the slow walk";

  // After a reset a zero command keeps the manager in stance: there is no gait below it to fall to.
  standing.mpc().reset();
  standing.command(0.0);
  vector_t still = standing.initialState();
  for (int solve = 0; solve < 100; ++solve, standingTime += kSolvePeriod) standing.mpc().run(standingTime, still, ModeNumber::kStance);
  EXPECT_EQ(standing.motionManager().getCurrentGaitCommand(), "stance");
}

/** A plan that stands on both feet for `numNodes` nodes of `dt` from `startTime`. */
ContactPlan standingPlan(scalar_t startTime, scalar_t dt, size_t numNodes) {
  ContactPlan plan;
  plan.valid = true;
  plan.startTime = startTime;
  plan.dt = dt;
  plan.committedUntil = startTime + dt;
  plan.contacts.assign(numNodes, makeFeetArray(true));
  plan.footholds.assign(numNodes + 1, makeFeetArray(vector2_t(vector2_t::Zero())));
  plan.comPosition.assign(numNodes + 1, vector2_t::Zero());
  plan.comVelocity.assign(numNodes + 1, vector2_t::Zero());
  plan.zmp.assign(numNodes, vector2_t::Zero());
  return plan;
}

TEST(MpcResetState, APlanMadeBeforeTheResetIsNeverActivatedAfterIt) {
  // The planner thread may be half-way through a plan made from a snapshot taken before the reset. The reset starts a
  // new plan epoch, and a plan from an earlier one is refused when it is handed over.
  AtlasReferenceStack stack(ScheduleSource::kContactPlanner);
  ContactPlanningReferenceManager& manager = *stack.planningReferenceManager();
  const uint64_t epochBeforeReset = manager.planEpoch();
  const ContactPlan plan = standingPlan(1.0, 0.05, 20);

  ASSERT_TRUE(manager.setContactPlan(plan, epochBeforeReset)) << "positive control: a plan of the current epoch is accepted";
  stack.mpc().reset();
  EXPECT_FALSE(manager.hasPendingPlan()) << "the reset keeps a plan handed over before it";
  EXPECT_NE(manager.planEpoch(), epochBeforeReset);
  EXPECT_FALSE(manager.setContactPlan(plan, epochBeforeReset)) << "a plan made before the reset was accepted after it";
  EXPECT_FALSE(manager.hasPendingPlan());
  EXPECT_TRUE(manager.setContactPlan(plan, manager.planEpoch()));
}

TEST(MpcResetState, TheTargetCalculatorForgetsItsFilters) {
  AtlasReferenceStack used;
  AtlasReferenceStack fresh;
  const vector_t& state = used.initialState();
  // The command filter of the calculator is driven towards a walk.
  for (int step = 0; step < 20; ++step) {
    used.targetCalculator().commandedVelocityToTargetTrajectories(vector4_t(1.0, 0.0, 0.0, 0.0), 0.01 * step, state);
  }
  used.targetCalculator().reset();

  const TargetTrajectories afterReset =
      used.targetCalculator().commandedVelocityToTargetTrajectories(vector4_t::Zero(), /*initTime=*/5.0, state);
  const TargetTrajectories afterConstruction =
      fresh.targetCalculator().commandedVelocityToTargetTrajectories(vector4_t::Zero(), /*initTime=*/5.0, state);
  ASSERT_EQ(afterReset.timeTrajectory, afterConstruction.timeTrajectory);
  for (size_t k = 0; k < afterReset.stateTrajectory.size(); ++k) {
    EXPECT_LT((afterReset.stateTrajectory[k] - afterConstruction.stateTrajectory[k]).cwiseAbs().maxCoeff(), 1.0e-12) << "knot " << k;
  }
}

TEST(MpcResetState, TheJointTargetStartsFromTheCurrentJointsHoweverLateTheResetComes) {
  // A reset comes at whatever time the robot fell, not at t = 0. The joint-state filter of the target must start from
  // the joints the robot has then, as it does after construction; it used to decay over the whole time since its clock
  // was last set, which put it at the nominal joints at once after a late reset.
  AtlasReferenceStack used;
  AtlasReferenceStack fresh;
  vector_t state = used.initialState();
  const Eigen::Index numJoints = state.size() - 12;  // [normalized momentum (6), base pose (6), joints]
  state.tail(numJoints).array() += 0.1;

  const TargetTrajectories afterConstruction =
      fresh.targetCalculator().commandedVelocityToTargetTrajectories(vector4_t::Zero(), /*initTime=*/0.0, state);
  const vector_t nominalJoints = afterConstruction.stateTrajectory.back().tail(numJoints);
  ASSERT_GT((nominalJoints - state.tail(numJoints)).cwiseAbs().maxCoeff(), 0.05)
      << "positive control: the current joints must differ from the nominal ones for the test to tell them apart";
  EXPECT_LT((afterConstruction.stateTrajectory.front().tail(numJoints) - state.tail(numJoints)).cwiseAbs().maxCoeff(), 1.0e-12);

  for (int step = 0; step < 20; ++step) {
    used.targetCalculator().commandedVelocityToTargetTrajectories(vector4_t(1.0, 0.0, 0.0, 0.0), 0.01 * step, used.initialState());
  }
  used.targetCalculator().reset();
  const TargetTrajectories afterLateReset =
      used.targetCalculator().commandedVelocityToTargetTrajectories(vector4_t::Zero(), /*initTime=*/21.0, state);
  EXPECT_LT((afterLateReset.stateTrajectory.front().tail(numJoints) - state.tail(numJoints)).cwiseAbs().maxCoeff(), 1.0e-12)
      << "after a reset at t = 21 s the joint target jumped towards the nominal joints";

  // A gap in the calls (a solver backing off) restarts the filter the same way.
  const TargetTrajectories afterGap =
      used.targetCalculator().commandedVelocityToTargetTrajectories(vector4_t::Zero(), /*initTime=*/23.0, state);
  EXPECT_LT((afterGap.stateTrajectory.front().tail(numJoints) - state.tail(numJoints)).cwiseAbs().maxCoeff(), 1.0e-12);
}

}  // namespace
}  // namespace ocs2::humanoid
