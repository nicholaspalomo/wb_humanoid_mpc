/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <ocs2_core/reference/ModeSchedule.h>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"
#include "humanoid_common_mpc/contact_planning/hlip/HlipContactPlanner.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kStepWidth = 0.25;

ContactPlanningConfig makeConfig() {
  ContactPlanningConfig config;
  config.planner.type = "hlip";
  config.planner.dt = 0.025;
  config.planner.numNodes = 56;
  config.planner.commitTime = 0.05;
  config.shared.comHeight = 0.85;
  config.shared.gaitLimits.minSwingDuration = 0.25;
  config.shared.gaitLimits.maxSwingDuration = 0.35;
  config.shared.gaitLimits.minDoubleSupportDuration = 0.0;
  config.hlip.sspDuration = 0.25;
  config.hlip.dspDuration = 0.05;
  config.hlip.stepWidth = kStepWidth;
  EXPECT_EQ(config.validateStatus(), absl::OkStatus());
  return config;
}

/** A robot standing in double support, with the feet symmetric about the origin. */
ContactPlannerInput makeStandingInput(const vector2_t& velocityCommand) {
  ContactPlannerInput input;
  input.time = 0.0;
  input.comPosition = vector2_t(0.0, 0.0);
  input.comVelocity = vector2_t::Zero();
  input.footPositions[CONTACT_LEFT_INDEX] = vector2_t(0.0, 0.5 * kStepWidth);
  input.footPositions[CONTACT_RIGHT_INDEX] = vector2_t(0.0, -0.5 * kStepWidth);
  input.contacts = makeFeetArray(true);
  input.phaseElapsedTime = makeFeetArray(1.0);
  input.velocityCommand = velocityCommand;
  input.committedUntil = input.time;
  return input;
}

/** The feet the planner placed over the horizon, in the order they land. */
std::vector<size_t> swingOrder(const ContactPlan& plan) {
  std::vector<size_t> order;
  contact_flag_t previous = plan.contacts.front();
  for (const contact_flag_t& contacts : plan.contacts) {
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      if (previous[foot] && !contacts[foot]) order.push_back(foot);
    }
    previous = contacts;
  }
  return order;
}

size_t otherFootOf(size_t foot) {
  return foot == CONTACT_LEFT_INDEX ? CONTACT_RIGHT_INDEX : CONTACT_LEFT_INDEX;
}

Eigen::Matrix<scalar_t, 2, 2> rotationOf(scalar_t yaw) {
  Eigen::Matrix<scalar_t, 2, 2> matrix;
  matrix << std::cos(yaw), -std::sin(yaw), std::sin(yaw), std::cos(yaw);
  return matrix;
}

/** A swing of the plan: which foot, the interval it lifts off and the interval it is down again. */
struct PlannedSwing {
  size_t foot;
  int liftOffInterval;
  int touchDownInterval;
};

/** Every swing of the plan that lifts off and lands inside the horizon. */
std::vector<PlannedSwing> plannedSwings(const ContactPlan& plan) {
  std::vector<PlannedSwing> swings;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    int liftOff = -1;
    for (int interval = 1; interval < plan.numIntervals(); ++interval) {
      const bool wasDown = plan.contacts[static_cast<size_t>(interval - 1)][foot];
      const bool isDown = plan.contacts[static_cast<size_t>(interval)][foot];
      if (wasDown && !isDown) liftOff = interval;
      if (!wasDown && isDown && liftOff >= 0) {
        swings.push_back(PlannedSwing{foot, liftOff, interval});
        liftOff = -1;
      }
    }
  }
  std::sort(swings.begin(), swings.end(),
            [](const PlannedSwing& first, const PlannedSwing& second) { return first.liftOffInterval < second.liftOffInterval; });
  return swings;
}

/**
 * The planner closed around its own reduced model at the MPC rate, with the executed schedule in between kept the way
 * ContactPlanningReferenceManager keeps it. Every cycle builds the planner input from that schedule exactly as
 * makePlannerInput does (commit boundary extended to the touch-down of a swing in flight, committed contacts and their
 * event times sampled with committedContactsForPlanner / committedPhaseStartsForPlanner), plans, merges the plan past
 * the commit boundary with mergeModeSchedules, and flows the H-LIP through the executed contacts to the next cycle,
 * landing each swing where the plan active at its touch-down put it. A foot in flight is reported where its latest
 * plan sends it, as if the swing reference were tracked perfectly. The touch-downs of one plan are therefore off the
 * node grid of the next, as they are at runtime. What it leaves out is the whole-body MPC: the center of mass follows
 * the reduced model exactly, and its double support drifts at constant velocity - except under a standing plan, which
 * asks the MPC to bring the center of mass to rest over the feet, and which the MPC can do with both feet down; there
 * the harness moves the center-of-mass velocity towards the plan's reference with a first-order lag
 * (kStandingTimeConstant). Without that the reduced model would keep sliding sideways at the orbit's 0.165 m/s for ever
 * after the robot stood, which no standing controller allows. The lag is deliberately not zero: a controller that
 * stopped the center of mass instantly would hide the double supports in which the robot is still slowing down from
 * the orbit's sway, and those are the ones the blend has to read as rest (HlipContactPlanner::blendVelocity).
 */
class ReducedModelLoop {
 public:
  struct Event {
    size_t foot;
    scalar_t time;
    bool touchDown;
    vector2_t step = vector2_t::Zero();  // at a touch-down: the landed foot minus the stance foot
    // At a touch-down: the center of mass relative to the stance foot at the impact, and the blend weight and command
    // of the plan the foot was landed from.
    HlipModel::State preImpactX = HlipModel::State::Zero();
    HlipModel::State preImpactY = HlipModel::State::Zero();
    scalar_t alpha = 0.0;
    vector2_t command = vector2_t::Zero();
  };

  ReducedModelLoop(const ContactPlanningConfig& config, scalar_t cyclePeriod)
      : config_(config), planner_(config), cyclePeriod_(cyclePeriod), applied_({}, {ModeNumber::STANCE}) {
    feet_[CONTACT_LEFT_INDEX] = vector2_t(0.0, 0.5 * config.hlip.stepWidth);
    feet_[CONTACT_RIGHT_INDEX] = vector2_t(0.0, -0.5 * config.hlip.stepWidth);
    comPosition_ = vector2_t::Zero();
    comVelocity_ = vector2_t::Zero();
    contacts_ = makeFeetArray(true);
  }

  void setCommand(const vector2_t& command) { command_ = command; }

  /**
   * Builds the executed schedule from the plans' node grid alone, as it was built before plans carried their gait in
   * continuous time (ContactPlan::phaseStartTimes): the positive control of the closed-loop timing tests.
   */
  void useNodeGridSchedule() { useNodeGridSchedule_ = true; }

  /** One MPC cycle. Returns whether the plan made in it was a stepping plan. */
  bool cycle() {
    const ContactPlannerInput input = makeInput();
    ContactPlan plan = planner_.plan(input);
    EXPECT_TRUE(plan.valid);
    if (useNodeGridSchedule_) {
      plan.phaseContacts.clear();
      plan.phaseStartTimes.clear();
    }
    const bool walking = planner_.isWalking(input);
    activeAlpha_ = planner_.blendWeight(input);
    applied_ = mergeModeSchedules(applied_, plan.toModeSchedule(), input.committedUntil, time_ - 2.0, plan.endTime());
    activePlan_ = plan;
    activePlanWalks_ = walking;
    // A foot in flight follows its swing reference to the landing spot of the plan now active.
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      if (contacts_[foot]) continue;
      const std::optional<size_t> touchDown = touchDownEventIndex(applied_, foot, time_);
      if (!touchDown.has_value()) continue;
      const std::optional<vector2_t> target = plan.footholdAtTime(foot, applied_.eventTimes[*touchDown]);
      if (target.has_value()) feet_[foot] = *target;
    }
    advanceTo(cycleStart_ + cyclePeriod_ * static_cast<scalar_t>(++numCycles_));
    return walking;
  }

  scalar_t time() const { return time_; }
  const std::vector<Event>& events() const { return events_; }
  /** The executed schedule; it keeps two seconds of history, as the reference manager keeps one horizon. */
  const ModeSchedule& executedSchedule() const { return applied_; }
  const HlipContactPlanner& planner() const { return planner_; }

  /** The planner input at `time_`, read off the executed schedule as ContactPlanningReferenceManager::makePlannerInput does. */
  ContactPlannerInput makeInput() {
    ContactPlannerInput input;
    input.time = time_;
    input.comPosition = comPosition_;
    input.comVelocity = comVelocity_;
    input.footPositions = feet_;
    input.velocityCommand = command_;
    input.committedUntil = commitBoundaryForSchedule(applied_, time_, config_.planner.commitTime, config_.planner.maxCommitExtension);
    fillPlannerInputFromSchedule(applied_, config_.planner.dt, std::max(0, config_.planner.numNodes - 1), liftOffHistory_, input);
    return input;
  }

 private:
  /** Applies the contact events of the executed schedule at `time_`: lands swings, records lift-offs. */
  void applyEvents() {
    const contact_flag_t now = contactFlagsAtTime(applied_, time_ + kEventTolerance);
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      if (contacts_[foot] == now[foot]) continue;
      Event event{foot, time_, now[foot]};
      if (now[foot]) {
        const std::optional<vector2_t> landing = activePlan_.footholdAtTime(foot, time_);
        EXPECT_TRUE(landing.has_value());
        if (landing.has_value()) feet_[foot] = *landing;
        const vector2_t stance = feet_[otherFootOf(foot)];
        event.step = feet_[foot] - stance;
        event.preImpactX = HlipModel::State(comPosition_.x() - stance.x(), comVelocity_.x());
        event.preImpactY = HlipModel::State(comPosition_.y() - stance.y(), comVelocity_.y());
        event.alpha = activeAlpha_;
        event.command = command_;
      }
      events_.push_back(event);
    }
    contacts_ = now;
  }

  void advanceTo(scalar_t endTime) {
    applyEvents();
    while (time_ < endTime - kEventTolerance) {
      scalar_t next = endTime;
      for (const scalar_t eventTime : applied_.eventTimes) {
        if (eventTime > time_ + kEventTolerance) {
          next = std::min(next, eventTime);
          break;
        }
      }
      flow(next - time_);
      time_ = next;
      applyEvents();
    }
  }

  /** The H-LIP through `duration` of the current contact state: the pendulum in single support, a drift otherwise. */
  void flow(scalar_t duration) {
    const bool left = contacts_[CONTACT_LEFT_INDEX];
    const bool right = contacts_[CONTACT_RIGHT_INDEX];
    if (left == right) {
      const std::optional<vector2_t> reference = activePlanWalks_ ? std::nullopt : activePlan_.comVelocityAtTime(time_);
      if (reference.has_value()) {
        // A first-order approach to the standing reference, integrated exactly over `duration`.
        const scalar_t decay = std::exp(-duration / kStandingTimeConstant);
        const vector2_t offset = comVelocity_ - *reference;
        comPosition_ += duration * *reference + kStandingTimeConstant * (1.0 - decay) * offset;
        comVelocity_ = *reference + decay * offset;
        return;
      }
      comPosition_ += duration * comVelocity_;
      return;
    }
    const vector2_t stance = feet_[left ? CONTACT_LEFT_INDEX : CONTACT_RIGHT_INDEX];
    const HlipModel& model = planner_.getModel();
    const HlipModel::State x = model.flowSingleSupport(HlipModel::State(comPosition_.x() - stance.x(), comVelocity_.x()), duration);
    const HlipModel::State y = model.flowSingleSupport(HlipModel::State(comPosition_.y() - stance.y(), comVelocity_.y()), duration);
    comPosition_ = stance + vector2_t(x(0), y(0));
    comVelocity_ = vector2_t(x(1), y(1));
  }

  // [s] events this close to the current time have happened: the node times of successive plans are sums of dt and
  // land a few ulp either side of each other.
  static constexpr scalar_t kEventTolerance = 1e-9;
  // [s] time constant of the harness's standing controller: how quickly the center of mass approaches a standing plan's
  // velocity reference with both feet down. Deliberately not instantaneous (see the class comment).
  static constexpr scalar_t kStandingTimeConstant = 0.15;

  ContactPlanningConfig config_;
  HlipContactPlanner planner_;
  scalar_t cyclePeriod_;
  scalar_t cycleStart_ = 0.0;
  int numCycles_ = 0;
  ModeSchedule applied_;
  LiftOffHistory liftOffHistory_;
  bool useNodeGridSchedule_ = false;
  ContactPlan activePlan_;
  bool activePlanWalks_ = true;
  scalar_t activeAlpha_ = 0.0;
  scalar_t time_ = 0.0;
  vector2_t command_ = vector2_t::Zero();
  vector2_t comPosition_;
  vector2_t comVelocity_;
  feet_array_t<vector2_t> feet_;
  contact_flag_t contacts_;
  std::vector<Event> events_;
};

// [s] about 58 Hz. A period that shares no small multiple with the planner's 0.025 s grid or the 0.3 s step, so the
// touch-downs of one plan fall at a different fraction of a node of every later plan.
constexpr scalar_t kMpcCyclePeriod = 0.0173;

TEST(HlipContactPlanner, standsAtZeroCommand) {
  HlipContactPlanner planner(makeConfig());
  const ContactPlan plan = planner.plan(makeStandingInput(vector2_t::Zero()));

  ASSERT_TRUE(plan.valid);
  for (const contact_flag_t& contacts : plan.contacts) {
    EXPECT_TRUE(contacts[CONTACT_LEFT_INDEX]);
    EXPECT_TRUE(contacts[CONTACT_RIGHT_INDEX]);
  }
  // Standing keeps the feet exactly where they are: there is no stepping-in-place at rest.
  for (const feet_array_t<vector2_t>& footholds : plan.footholds) {
    EXPECT_NEAR(footholds[CONTACT_LEFT_INDEX].y(), 0.5 * kStepWidth, 1e-12);
    EXPECT_NEAR(footholds[CONTACT_RIGHT_INDEX].y(), -0.5 * kStepWidth, 1e-12);
  }
}

TEST(HlipContactPlanner, standingAsksForACenterOfMassAtRest) {
  // A standing plan must not hand the controller the measured drift as its reference: the reduced model's double
  // support drifts at constant velocity and can never decelerate, so a plan that simply rolled it out would ask the
  // robot to keep sliding. The blend against the static standing reference (the paper's equation 14) is what stops it.
  HlipContactPlanner planner(makeConfig());
  ContactPlannerInput input = makeStandingInput(vector2_t::Zero());
  input.comPosition = vector2_t(0.03, 0.01);
  input.comVelocity = vector2_t(0.02, -0.01);  // a small drift, well below the blend's half point

  const ContactPlan plan = planner.plan(input);
  ASSERT_TRUE(plan.valid);
  ASSERT_FALSE(planner.isWalking(input));

  // The reference is the blend itself: alpha of the rolled-out drift and (1 - alpha) of standing still. The reduced
  // model's double support cannot decelerate, so every bit of the drift that survives is alpha's doing and no more.
  const scalar_t blendWeight = planner.blendWeight(input);
  ASSERT_LT(blendWeight, 0.5);
  const vector2_t expectedVelocity = blendWeight * input.comVelocity;
  EXPECT_NEAR(plan.comVelocity.back().x(), expectedVelocity.x(), 1e-9) << "standing must ask for a stop, not a drift";
  EXPECT_NEAR(plan.comVelocity.back().y(), expectedVelocity.y(), 1e-9) << "standing must ask for a stop, not a drift";

  const vector2_t supportCenter = 0.5 * (input.footPositions[CONTACT_LEFT_INDEX] + input.footPositions[CONTACT_RIGHT_INDEX]);
  EXPECT_LT((plan.comPosition.back() - supportCenter).norm(), 0.02) << "standing must ask for the center of the support";
}

TEST(HlipContactPlanner, walksAtACommandAndAlternatesFeet) {
  HlipContactPlanner planner(makeConfig());
  const ContactPlannerInput input = makeStandingInput(vector2_t(0.5, 0.0));
  ASSERT_TRUE(planner.isWalking(input));
  const ContactPlan plan = planner.plan(input);
  ASSERT_TRUE(plan.valid);

  const std::vector<size_t> order = swingOrder(plan);
  ASSERT_GE(order.size(), 2U);
  for (size_t index = 1; index < order.size(); ++index) {
    EXPECT_NE(order[index], order[index - 1]) << "a foot may not swing twice in a row";
  }
  // Exactly one foot is ever off the ground: the nominal gait has no flight phase.
  for (const contact_flag_t& contacts : plan.contacts) {
    EXPECT_TRUE(contacts[CONTACT_LEFT_INDEX] || contacts[CONTACT_RIGHT_INDEX]);
  }
}

TEST(HlipContactPlanner, singleSupportPhasesLastTheConfiguredDuration) {
  const ContactPlanningConfig config = makeConfig();
  HlipContactPlanner planner(config);
  const std::vector<HlipContactPlanner::GaitPhase> gait = planner.buildGait(makeStandingInput(vector2_t(0.5, 0.0)), /*walking=*/true);

  ASSERT_GE(gait.size(), 2U);
  for (const HlipContactPlanner::GaitPhase& phase : gait) {
    if (phase.isSingleSupport()) {
      EXPECT_NEAR(phase.duration(), config.hlip.sspDuration, 1e-12);
    }
  }
  // The gait is contiguous and covers the horizon, and every phase ends on a node no earlier than the one before.
  for (size_t index = 1; index < gait.size(); ++index) {
    EXPECT_NEAR(gait[index].startTime, gait[index - 1].endTime, 1e-12);
    EXPECT_GE(gait[index].endNode, gait[index - 1].endNode);
  }
  EXPECT_GE(gait.back().endTime, gait.front().startTime + config.horizon() - 1e-12);
  EXPECT_EQ(gait.back().endNode, config.planner.numNodes);
}

TEST(HlipContactPlanner, continuesTheSwingInFlight) {
  const ContactPlanningConfig config = makeConfig();
  HlipContactPlanner planner(config);
  ContactPlannerInput input = makeStandingInput(vector2_t(0.5, 0.0));
  input.contacts[CONTACT_LEFT_INDEX] = false;  // the left foot is in flight, a third of the way through its swing
  input.phaseElapsedTime[CONTACT_LEFT_INDEX] = 0.1;
  input.phaseElapsedTime[CONTACT_RIGHT_INDEX] = 0.1;

  const std::vector<HlipContactPlanner::GaitPhase> gait = planner.buildGait(input, /*walking=*/true);
  ASSERT_FALSE(gait.empty());
  EXPECT_EQ(gait.front().swingFoot, static_cast<int>(CONTACT_LEFT_INDEX));
  EXPECT_NEAR(gait.front().duration(), config.hlip.sspDuration - 0.1, 1e-12);
  // With a non-zero double support the next single support is not the next phase: the hand-over comes first.
  ASSERT_GE(gait.size(), 2U);
  const std::vector<HlipContactPlanner::GaitPhase>::const_iterator nextSingleSupport =
      std::find_if(gait.begin() + 1, gait.end(), [](const HlipContactPlanner::GaitPhase& phase) { return phase.isSingleSupport(); });
  ASSERT_NE(nextSingleSupport, gait.end());
  EXPECT_EQ(nextSingleSupport->swingFoot, static_cast<int>(CONTACT_RIGHT_INDEX)) << "the feet must alternate";
}

TEST(HlipContactPlanner, stepsAdvanceByTheCommandedVelocityOnTheOrbit) {
  // The steady state of the deadbeat law is the period-one orbit, whose step is v * T. Start the reduced model exactly
  // on that orbit and the planner must ask for exactly that step.
  const ContactPlanningConfig config = makeConfig();
  HlipContactPlanner planner(config);
  const HlipModel& model = planner.getModel();
  const scalar_t velocity = 0.5;
  const scalar_t stepDuration = model.stepDuration();
  const scalar_t nominalStep = velocity * stepDuration;
  const HlipModel::State orbit = model.periodOneOrbit(nominalStep);
  const std::pair<HlipModel::State, HlipModel::State> lateralOrbit = model.periodTwoOrbit(config.hlip.stepWidth, -config.hlip.stepWidth);

  // Place the robot at the post-impact state of that orbit, standing on the right foot with the left about to swing.
  ContactPlannerInput input = makeStandingInput(vector2_t(velocity, 0.0));
  input.contacts[CONTACT_LEFT_INDEX] = false;
  input.phaseElapsedTime = makeFeetArray(0.0);
  // The state a single support actually begins from is the post-impact state carried through the double support, which
  // the reduced model drifts at constant velocity.
  const HlipModel::State postImpactX =
      HlipModel::flowDoubleSupport(HlipModel::applyStepTransition(orbit, nominalStep), config.hlip.dspDuration);
  const HlipModel::State postImpactY =
      HlipModel::flowDoubleSupport(HlipModel::applyStepTransition(lateralOrbit.second, -config.hlip.stepWidth), config.hlip.dspDuration);
  const vector2_t stanceFoot = input.footPositions[CONTACT_RIGHT_INDEX];
  input.comPosition = stanceFoot + vector2_t(postImpactX(0), postImpactY(0));
  input.comVelocity = vector2_t(postImpactX(1), postImpactY(1));
  // The blend must not scale the command down, or the step would be shorter than the orbit's.
  ASSERT_GE(planner.blendWeight(input), 0.999);

  const ContactPlan plan = planner.plan(input);
  ASSERT_TRUE(plan.valid);
  EXPECT_EQ(plan.numClippedSteps, 0);
  const std::optional<vector2_t> landing = plan.footholdAtTime(CONTACT_LEFT_INDEX, config.hlip.sspDuration);
  ASSERT_TRUE(landing.has_value());
  EXPECT_NEAR(landing->x() - stanceFoot.x(), nominalStep, 1e-6);
  EXPECT_NEAR(landing->y() - stanceFoot.y(), config.hlip.stepWidth, 1e-6);
}

TEST(HlipContactPlanner, aLateralCommandShiftsBothStepsOfTheLateralOrbitByTheCommandedDrift) {
  // Laterally the nominal gait is the period-two orbit of u_L = +stepWidth + v_y T and u_R = -stepWidth + v_y T, so on
  // that orbit the two steps of a stride sum to exactly 2 v_y T and the robot drifts sideways at the commanded rate,
  // in the direction of the command. Nothing else tested the lateral drift term: every other test commands v_y = 0, or
  // checks only an equivariance that holds whatever its sign.
  const ContactPlanningConfig config = makeConfig();
  HlipContactPlanner planner(config);
  const HlipModel& model = planner.getModel();
  const scalar_t stepDuration = model.stepDuration();
  for (const scalar_t lateralCommand : {0.2, -0.2}) {
    const scalar_t leftStep = config.hlip.stepWidth + lateralCommand * stepDuration;
    const scalar_t rightStep = -config.hlip.stepWidth + lateralCommand * stepDuration;
    const std::pair<HlipModel::State, HlipModel::State> orbit = model.periodTwoOrbit(leftStep, rightStep);

    // On the orbit at the lift-off of the left foot: the right step's impact carried through the double support.
    ContactPlannerInput input = makeStandingInput(vector2_t(0.0, lateralCommand));
    input.contacts[CONTACT_LEFT_INDEX] = false;
    input.phaseElapsedTime = makeFeetArray(0.0);
    const HlipModel::State atLiftOff =
        HlipModel::flowDoubleSupport(HlipModel::applyStepTransition(orbit.second, rightStep), config.hlip.dspDuration);
    const vector2_t stanceFoot = input.footPositions[CONTACT_RIGHT_INDEX];
    input.comPosition = stanceFoot + vector2_t(0.0, atLiftOff(0));
    input.comVelocity = vector2_t(0.0, atLiftOff(1));
    ASSERT_DOUBLE_EQ(planner.blendWeight(input), 1.0) << "the command must not be scaled down, or this is not the commanded orbit";

    const ContactPlan plan = planner.plan(input);
    ASSERT_TRUE(plan.valid);
    EXPECT_EQ(plan.numClippedSteps, 0) << "the commanded orbit must fit inside the step-width clip";
    const std::vector<PlannedSwing> swings = plannedSwings(plan);
    ASSERT_GE(swings.size(), 2U);
    // The first swing is the left foot already in flight, so its touch-down is the first left touch-down.
    const std::optional<vector2_t> left = plan.footholdAtTime(CONTACT_LEFT_INDEX, config.hlip.sspDuration);
    ASSERT_TRUE(left.has_value());
    const scalar_t placedLeft = left->y() - stanceFoot.y();
    EXPECT_NEAR(placedLeft, leftStep, 1e-9) << "v_y = " << lateralCommand;

    // The next swing places the right foot from the left one.
    const PlannedSwing& rightSwing = swings.front().foot == CONTACT_RIGHT_INDEX ? swings.front() : swings[1];
    ASSERT_EQ(rightSwing.foot, CONTACT_RIGHT_INDEX);
    const vector2_t right = plan.footholds[static_cast<size_t>(rightSwing.touchDownInterval)][CONTACT_RIGHT_INDEX];
    const scalar_t placedRight = right.y() - left->y();
    EXPECT_NEAR(placedRight, rightStep, 1e-9) << "v_y = " << lateralCommand;

    EXPECT_NEAR(placedLeft + placedRight, 2.0 * lateralCommand * stepDuration, 1e-9);
    EXPECT_GT((placedLeft + placedRight) * lateralCommand, 0.0) << "the robot must drift the way it is commanded to";
  }
}

TEST(HlipContactPlanner, aYawRateTurnsEveryFootholdByTheIntegratedHeading) {
  // With a yaw-rate command the heading of the gait advances by alpha * rate * t, and each foothold is placed in the
  // frame of the heading at its own touch-down. Nothing exercised this with a non-zero rate: a sign flip of the heading
  // integration turned the robot the wrong way and every test still passed.
  for (const bool headingModel : {false, true}) {
    ContactPlanningConfig config = makeConfig();
    config.setHeadingModel(headingModel);
    HlipContactPlanner planner(config);
    ContactPlannerInput input = makeStandingInput(vector2_t::Zero());
    input.headingRateCommand = 0.6;  // [rad/s] in place, no translation
    const scalar_t alpha = planner.blendWeight(input);
    ASSERT_GE(alpha, 0.5) << "a yaw command alone must start the gait";
    ASSERT_TRUE(planner.isWalking(input));

    const ContactPlan plan = planner.plan(input);
    ASSERT_TRUE(plan.valid);
    const std::vector<PlannedSwing> swings = plannedSwings(plan);
    ASSERT_GE(swings.size(), 3U) << "turning in place is a stepping gait";

    scalar_t previousHeading = input.yaw;
    for (const PlannedSwing& swing : swings) {
      const size_t node = static_cast<size_t>(swing.touchDownInterval);
      const scalar_t touchDown = plan.startTime + plan.dt * static_cast<scalar_t>(node);
      // The phase heading the foot was placed in, within the half node the touch-down is rounded by.
      const scalar_t expectedHeading = input.yaw + alpha * input.headingRateCommand * (touchDown - input.time);
      const vector2_t step = plan.footholds[node][swing.foot] - plan.footholds[node][otherFootOf(swing.foot)];
      const vector2_t stepInHeading = rotationOf(expectedHeading).transpose() * step;
      const scalar_t side = swing.foot == CONTACT_LEFT_INDEX ? 1.0 : -1.0;
      const scalar_t tolerance = alpha * input.headingRateCommand * 0.5 * plan.dt * step.norm() + 1e-9;
      // In its own heading frame the placed foot is on its own side, at least minStepWidth out: a foot placed in the
      // frame of the opposite rotation would be swung round by twice the turn and fail this by a wide margin.
      EXPECT_GE(side * stepInHeading.y(), config.hlip.minStepWidth - tolerance) << "foot " << swing.foot << " at " << touchDown;
      EXPECT_LE(std::abs(stepInHeading.x()), config.hlip.maxStepLength + tolerance) << "foot " << swing.foot << " at " << touchDown;
      EXPECT_GT(expectedHeading, previousHeading) << "a positive rate turns the gait counter-clockwise";
      previousHeading = expectedHeading;
    }
    ASSERT_GT(previousHeading - input.yaw, 0.5) << "the horizon must turn far enough for the frame to matter";

    if (headingModel) {
      ASSERT_EQ(plan.heading.size(), plan.footholds.size());
      for (size_t node = 0; node < plan.heading.size(); ++node) {
        const scalar_t nodeTime = plan.startTime + plan.dt * static_cast<scalar_t>(node);
        EXPECT_NEAR(plan.heading[node], input.yaw + alpha * input.headingRateCommand * (nodeTime - input.time), 1e-12);
        EXPECT_NEAR(plan.headingRate[node], alpha * input.headingRateCommand, 1e-12);
      }
      // A placed foot is yawed to the heading of its touch-down, so the landing yaws advance with the turn.
      for (const PlannedSwing& swing : swings) {
        const size_t node = static_cast<size_t>(swing.touchDownInterval);
        const scalar_t touchDown = plan.startTime + plan.dt * static_cast<scalar_t>(node);
        const scalar_t expectedHeading = input.yaw + alpha * input.headingRateCommand * (touchDown - input.time);
        EXPECT_NEAR(plan.footYaws[node][swing.foot], expectedHeading, alpha * input.headingRateCommand * 0.5 * plan.dt + 1e-12);
      }
    } else {
      EXPECT_TRUE(plan.heading.empty());
    }
  }
}

TEST(HlipContactPlanner, theStrideIsScaledByAlpha) {
  // "Above the half point the commanded velocity used for u* is scaled by alpha, so the stride grows continuously out of
  // standing." On the period-one orbit of alpha * v the step is exactly alpha * v * T, not v * T. Every other stride
  // test runs at alpha = 1, where dropping the scaling changes nothing.
  ContactPlanningConfig config = makeConfig();
  // Normalize the measured velocity away, so that alpha is set by the command alone and the orbit below is not chased
  // by an alpha that depends on it.
  config.hlip.blend.maxComVelocityX = 1.0e3;
  config.hlip.blend.maxComVelocityY = 1.0e3;
  HlipContactPlanner planner(config);
  const HlipModel& model = planner.getModel();
  const scalar_t command = 0.12;  // [m/s] a little above the half point of the 0.7 m/s range

  ContactPlannerInput input = makeStandingInput(vector2_t(command, 0.0));
  input.contacts[CONTACT_LEFT_INDEX] = false;
  input.phaseElapsedTime = makeFeetArray(0.0);
  const vector2_t stanceFoot = input.footPositions[CONTACT_RIGHT_INDEX];
  const std::pair<HlipModel::State, HlipModel::State> lateralOrbit = model.periodTwoOrbit(config.hlip.stepWidth, -config.hlip.stepWidth);
  const HlipModel::State atLiftOffY =
      HlipModel::flowDoubleSupport(HlipModel::applyStepTransition(lateralOrbit.second, -config.hlip.stepWidth), config.hlip.dspDuration);
  scalar_t alpha = planner.getBlend().weight(vector2_t(command, 0.0), /*yawRateCommand=*/0.0, vector2_t::Zero());
  for (int iteration = 0; iteration < 5; ++iteration) {  // the measured velocity still enters alpha, if negligibly
    const scalar_t scaledStep = alpha * command * model.stepDuration();
    const HlipModel::State atLiftOffX =
        HlipModel::flowDoubleSupport(HlipModel::applyStepTransition(model.periodOneOrbit(scaledStep), scaledStep), config.hlip.dspDuration);
    input.comPosition = stanceFoot + vector2_t(atLiftOffX(0), atLiftOffY(0));
    input.comVelocity = vector2_t(atLiftOffX(1), atLiftOffY(1));
    alpha = planner.blendWeight(input);
  }
  ASSERT_GT(alpha, 0.6) << "this test needs a stepping plan with alpha well inside (0.5, 1)";
  ASSERT_LT(alpha, 0.9);

  const ContactPlan plan = planner.plan(input);
  ASSERT_TRUE(plan.valid);
  const std::optional<vector2_t> landing = plan.footholdAtTime(CONTACT_LEFT_INDEX, config.hlip.sspDuration);
  ASSERT_TRUE(landing.has_value());
  const scalar_t step = landing->x() - stanceFoot.x();
  EXPECT_NEAR(step, alpha * command * model.stepDuration(), 1e-9);
  ASSERT_GT(command * model.stepDuration() - step, 0.005) << "the unscaled stride must be distinguishable from the scaled one";
}

TEST(HlipContactPlanner, aYawCommandAloneStartsTheGait) {
  // Turning on the spot is walking: the yaw rate is one of the blend's five terms, and a robot at rest asked only to
  // turn must get a stepping plan, not a standing one.
  HlipContactPlanner planner(makeConfig());
  ContactPlannerInput input = makeStandingInput(vector2_t::Zero());
  ASSERT_FALSE(planner.isWalking(input)) << "positive control: at rest with no command the robot stands";
  input.headingRateCommand = 0.3;
  EXPECT_TRUE(planner.isWalking(input));
  const ContactPlan plan = planner.plan(input);
  EXPECT_FALSE(swingOrder(plan).empty()) << "a yaw command must produce a swing";
}

TEST(HlipContactPlanner, honorsTheCommittedContacts) {
  const ContactPlanningConfig config = makeConfig();
  HlipContactPlanner planner(config);
  ContactPlannerInput input = makeStandingInput(vector2_t(0.5, 0.0));
  input.committedUntil = 0.1;
  input.committedContacts.assign(4, makeFeetArray(true));
  input.committedContacts[2][CONTACT_RIGHT_INDEX] = false;

  const ContactPlan plan = planner.plan(input);
  ASSERT_TRUE(plan.valid);
  for (size_t interval = 0; interval < input.committedContacts.size(); ++interval) {
    EXPECT_EQ(plan.contacts[interval], input.committedContacts[interval]);
  }
}

TEST(HlipContactPlanner, planIsWellFormed) {
  const ContactPlanningConfig config = makeConfig();
  HlipContactPlanner planner(config);
  const ContactPlan plan = planner.plan(makeStandingInput(vector2_t(0.4, 0.1)));

  ASSERT_TRUE(plan.valid);
  EXPECT_EQ(plan.numIntervals(), config.planner.numNodes);
  EXPECT_EQ(plan.footholds.size(), static_cast<size_t>(config.planner.numNodes) + 1);
  EXPECT_EQ(plan.comPosition.size(), plan.footholds.size());
  EXPECT_EQ(plan.zmp.size(), plan.contacts.size());
  EXPECT_NEAR(plan.endTime(), config.horizon(), 1e-12);

  const ModeSchedule schedule = plan.toModeSchedule();
  EXPECT_FALSE(schedule.modeSequence.empty());
  EXPECT_TRUE(std::is_sorted(schedule.eventTimes.begin(), schedule.eventTimes.end()));
}

TEST(HlipContactPlanner, closedLoopTracksTheCommandedVelocity) {
  // Roll the planner and its own reduced model forward: from standing, the deadbeat steps must bring the average
  // velocity of the model to the command within a few steps, which is the property the whole planner rests on.
  const ContactPlanningConfig config = makeConfig();
  HlipContactPlanner planner(config);
  const HlipModel& model = planner.getModel();
  const scalar_t command = 0.5;

  ContactPlannerInput input = makeStandingInput(vector2_t(command, 0.0));
  input.contacts[CONTACT_LEFT_INDEX] = false;
  input.phaseElapsedTime = makeFeetArray(0.0);

  scalar_t previousStanceX = input.footPositions[CONTACT_RIGHT_INDEX].x();
  scalar_t lastStepLength = 0.0;
  for (int step = 0; step < 8; ++step) {
    const size_t swingFoot = input.contacts[CONTACT_LEFT_INDEX] ? CONTACT_RIGHT_INDEX : CONTACT_LEFT_INDEX;
    const size_t stanceFoot = swingFoot == CONTACT_LEFT_INDEX ? CONTACT_RIGHT_INDEX : CONTACT_LEFT_INDEX;
    const ContactPlan plan = planner.plan(input);
    ASSERT_TRUE(plan.valid);
    const std::optional<vector2_t> landing = plan.footholdAtTime(swingFoot, input.time + config.hlip.sspDuration);
    ASSERT_TRUE(landing.has_value());

    // Advance the reduced model to the touch-down of that step.
    HlipModel::State stateX(input.comPosition.x() - input.footPositions[stanceFoot].x(), input.comVelocity.x());
    HlipModel::State stateY(input.comPosition.y() - input.footPositions[stanceFoot].y(), input.comVelocity.y());
    stateX = model.flowSingleSupport(stateX, config.hlip.sspDuration);
    stateY = model.flowSingleSupport(stateY, config.hlip.sspDuration);

    lastStepLength = landing->x() - previousStanceX;
    previousStanceX = landing->x();

    input.time += config.hlip.sspDuration;
    input.footPositions[swingFoot] = *landing;
    input.comPosition = input.footPositions[stanceFoot] + vector2_t(stateX(0), stateY(0));
    input.comVelocity = vector2_t(stateX(1), stateY(1));
    // The double support that follows: the reduced model drifts through it at constant velocity. Leaving it out makes
    // the rolled-out step shorter than the one the planner's own step duration is built on.
    input.comPosition += config.hlip.dspDuration * input.comVelocity;
    input.time += config.hlip.dspDuration;
    input.contacts = makeFeetArray(true);
    input.contacts[stanceFoot] = false;  // the roles swap: the old stance foot swings next
    input.phaseElapsedTime = makeFeetArray(0.0);
    input.lastSwungFoot = static_cast<int>(swingFoot);
    input.committedUntil = input.time;
  }

  EXPECT_NEAR(lastStepLength / model.stepDuration(), command, 1e-3);
}

/**
 * The planner's own lateral start-up out of a standstill, rolled forward step by step as the closed loop does: the
 * lateral distance between the feet after each step, and the clipped steps each plan reported.
 */
struct StartUpRollOut {
  std::vector<scalar_t> widths;
  std::vector<int> numClippedSteps;
};

StartUpRollOut rollOutTheStartUp(const ContactPlanningConfig& config, int numSteps) {
  HlipContactPlanner planner(config);
  const HlipModel& model = planner.getModel();
  ContactPlannerInput input = makeStandingInput(vector2_t(0.3, 0.0));
  input.footPositions[CONTACT_LEFT_INDEX] = vector2_t(0.0, 0.5 * config.hlip.stepWidth);
  input.footPositions[CONTACT_RIGHT_INDEX] = vector2_t(0.0, -0.5 * config.hlip.stepWidth);
  input.contacts[CONTACT_LEFT_INDEX] = false;  // standing on the right, the left about to swing
  input.phaseElapsedTime = makeFeetArray(0.0);

  StartUpRollOut result;
  for (int step = 0; step < numSteps; ++step) {
    const size_t swingFoot = input.contacts[CONTACT_LEFT_INDEX] ? CONTACT_RIGHT_INDEX : CONTACT_LEFT_INDEX;
    const size_t stanceFoot = otherFootOf(swingFoot);
    const ContactPlan plan = planner.plan(input);
    EXPECT_TRUE(plan.valid);
    result.numClippedSteps.push_back(plan.numClippedSteps);
    const std::optional<vector2_t> landing = plan.footholdAtTime(swingFoot, input.time + config.hlip.sspDuration);
    EXPECT_TRUE(landing.has_value());
    if (!landing.has_value()) break;

    HlipModel::State stateX(input.comPosition.x() - input.footPositions[stanceFoot].x(), input.comVelocity.x());
    HlipModel::State stateY(input.comPosition.y() - input.footPositions[stanceFoot].y(), input.comVelocity.y());
    stateX = model.flowSingleSupport(stateX, config.hlip.sspDuration);
    stateY = model.flowSingleSupport(stateY, config.hlip.sspDuration);

    input.time += config.hlip.sspDuration;
    input.footPositions[swingFoot] = *landing;
    input.comPosition = input.footPositions[stanceFoot] + vector2_t(stateX(0), stateY(0));
    input.comVelocity = vector2_t(stateX(1), stateY(1));
    // The double support: the reduced model drifts at constant velocity through it.
    input.comPosition += config.hlip.dspDuration * input.comVelocity;
    input.time += config.hlip.dspDuration;
    input.contacts = makeFeetArray(true);
    input.contacts[stanceFoot] = false;
    input.phaseElapsedTime = makeFeetArray(0.0);
    input.lastSwungFoot = static_cast<int>(swingFoot);
    input.committedUntil = input.time;

    result.widths.push_back(input.footPositions[CONTACT_LEFT_INDEX].y() - input.footPositions[CONTACT_RIGHT_INDEX].y());
  }
  return result;
}

TEST(HlipContactPlanner, theLateralGaitConvergesFromAStandstillWithoutClipping) {
  // The regression this guards. Starting to walk from a standstill demands a first lateral step far wider than the
  // nominal one: the center of mass is half a step width from the stance foot with no lateral velocity, so it falls
  // sideways fast once the foot lifts. If that step does not fit inside maxStepWidth it is clipped, and a clipped step
  // is not the deadbeat step - at some cadences the widths then lock into an alternating wide / narrow limit cycle: the
  // feet come together and the robot walks itself sideways.
  const ContactPlanningConfig config = makeConfig();

  EXPECT_LT(HlipContactPlanner::startUpLateralStep(config), config.hlip.maxStepWidth)
      << "the first step out of a standstill must fit inside the reach, or the gait cannot start; shorten sspDuration";

  // Roll the planner and its own reduced model forward laterally, exactly as the closed loop does. The widths alone
  // cannot see a clipped first step - a gait whose first step is clipped can still settle at the nominal width - so
  // every plan must also report that it clipped nothing.
  const StartUpRollOut rollOut = rollOutTheStartUp(config, /*numSteps=*/6);
  ASSERT_EQ(rollOut.widths.size(), 6U);
  for (size_t step = 0; step < rollOut.numClippedSteps.size(); ++step) {
    EXPECT_EQ(rollOut.numClippedSteps[step], 0) << "the plan made before step " << step << " clipped a step";
  }

  // The deadbeat law reaches the nominal width and stays there; the limit cycle would alternate the two clips forever.
  EXPECT_NEAR(rollOut.widths.back(), config.hlip.stepWidth, 1e-3) << "the lateral gait must settle at the nominal step width";
  EXPECT_NEAR(rollOut.widths[rollOut.widths.size() - 2], config.hlip.stepWidth, 1e-3) << "and stay there, rather than alternating";
}

TEST(HlipContactPlanner, theStartUpWidthsOfTheSummaryAreThePlannersOwn) {
  // formulationSummary decides whether a clipped first step recovers or locks from startUpLateralWidths(), a lateral
  // roll-out of the reduced model under the planner's step law and clip. That verdict is only true of the planner if
  // the roll-out is what the planner itself does, so compare it with the planner closed around its own model, at a
  // cadence that fits, one that clips and recovers, and one that clips and locks.
  struct Case {
    scalar_t sspDuration;
    scalar_t dspDuration;
    scalar_t maxStepWidth;
  };
  for (const Case& testCase : {Case{0.25, 0.05, 0.45}, Case{0.35, 0.0, 0.45}, Case{0.35, 0.05, 0.45}}) {
    ContactPlanningConfig config = makeConfig();
    config.hlip.sspDuration = testCase.sspDuration;
    config.hlip.dspDuration = testCase.dspDuration;
    config.hlip.maxStepWidth = testCase.maxStepWidth;
    const StartUpRollOut planned = rollOutTheStartUp(config, /*numSteps=*/8);
    const std::vector<scalar_t> modeled = HlipContactPlanner::startUpLateralWidths(config, /*numSteps=*/8);
    ASSERT_EQ(planned.widths.size(), modeled.size());
    for (size_t step = 0; step < modeled.size(); ++step) {
      EXPECT_NEAR(planned.widths[step], modeled[step], 1e-9)
          << "ssp " << testCase.sspDuration << " dsp " << testCase.dspDuration << " step " << step;
    }
  }
}

TEST(HlipContactPlanner, theStartUpBannerSaysWhetherTheFirstStepFitsAndWhatAClipDoes) {
  // "DOES NOT FIT" exactly when the first-step demand exceeds maxStepWidth, and "locks" exactly when the planner's own
  // start-up then never settles. The banner used to say that every clipped first step locks the gait, which the
  // library defaults themselves contradict: at 0.35 s with no double support the first step is clipped (0.49 m against
  // 0.45 m) and the gait still settles within three steps, so an operator reading the banner would retune a cadence
  // that works.
  struct Case {
    scalar_t sspDuration;
    scalar_t dspDuration;
    scalar_t maxStepWidth;
  };
  int sawFits = 0;
  int sawRecovers = 0;
  int sawLocks = 0;
  for (const Case& testCase :
       {Case{0.25, 0.05, 0.45}, Case{0.35, 0.0, 0.45}, Case{0.35, 0.05, 0.45}, Case{0.5, 0.05, 0.45}, Case{0.25, 0.05, 0.35}}) {
    ContactPlanningConfig config = makeConfig();
    config.hlip.sspDuration = testCase.sspDuration;
    config.hlip.dspDuration = testCase.dspDuration;
    config.hlip.maxStepWidth = testCase.maxStepWidth;
    const std::string summary = HlipContactPlanner::formulationSummary(config);
    const bool fits = HlipContactPlanner::startUpLateralStep(config) <= config.hlip.maxStepWidth;
    const StartUpRollOut planned = rollOutTheStartUp(config, /*numSteps=*/12);
    const bool settles = std::abs(planned.widths.back() - config.hlip.stepWidth) < 1e-3 &&
                         std::abs(planned.widths[planned.widths.size() - 2] - config.hlip.stepWidth) < 1e-3;
    const std::string label = absl::StrCat("ssp ", testCase.sspDuration, " dsp ", testCase.dspDuration, " max ", testCase.maxStepWidth);
    EXPECT_EQ(summary.find("DOES NOT FIT") == std::string::npos, fits) << label << "\n" << summary;
    EXPECT_EQ(summary.find("locks") != std::string::npos, !settles) << label << "\n" << summary;
    EXPECT_EQ(planned.numClippedSteps.front() > 0, !fits) << label << ": the first plan must count the clipped first step";
    sawFits += fits ? 1 : 0;
    sawRecovers += !fits && settles ? 1 : 0;
    sawLocks += !settles ? 1 : 0;
  }
  // The cases must cover all three outcomes, or the assertions above prove less than they seem to.
  EXPECT_GT(sawFits, 0);
  EXPECT_GT(sawRecovers, 0);
  EXPECT_GT(sawLocks, 0);
}

TEST(HlipContactPlanner, clipsStepsToTheReachableRegion) {
  ContactPlanningConfig config = makeConfig();
  config.hlip.maxStepLength = 0.2;
  HlipContactPlanner planner(config);

  ContactPlannerInput input = makeStandingInput(vector2_t(1.5, 0.0));  // far beyond what one step can deliver
  input.contacts[CONTACT_LEFT_INDEX] = false;
  input.comVelocity = vector2_t(1.5, 0.0);
  input.phaseElapsedTime = makeFeetArray(0.0);

  const ContactPlan plan = planner.plan(input);
  ASSERT_TRUE(plan.valid);
  const std::optional<vector2_t> landing = plan.footholdAtTime(CONTACT_LEFT_INDEX, config.hlip.sspDuration);
  ASSERT_TRUE(landing.has_value());
  const scalar_t step = landing->x() - input.footPositions[CONTACT_RIGHT_INDEX].x();
  EXPECT_LE(step, config.hlip.maxStepLength + 1e-9);
  const scalar_t width = landing->y() - input.footPositions[CONTACT_RIGHT_INDEX].y();
  EXPECT_GE(width, config.hlip.minStepWidth - 1e-9);
  EXPECT_LE(width, config.hlip.maxStepWidth + 1e-9);

  // And the clip is reported: a clipped step is no longer the deadbeat step, which is what the counter and the log
  // line exist to say.
  EXPECT_GT(plan.numClippedSteps, 0);
  EXPECT_NE(plan.describe().find("CLIPPED-STEPS"), std::string::npos) << plan.describe();

  // Positive control: the same input at a command one step can deliver clips nothing.
  ContactPlanningConfig roomy = makeConfig();
  HlipContactPlanner roomyPlanner(roomy);
  ContactPlannerInput reachable = makeStandingInput(vector2_t(0.3, 0.0));
  reachable.contacts[CONTACT_LEFT_INDEX] = false;
  reachable.phaseElapsedTime = makeFeetArray(0.0);
  const std::pair<HlipModel::State, HlipModel::State> lateralOrbit =
      roomyPlanner.getModel().periodTwoOrbit(roomy.hlip.stepWidth, -roomy.hlip.stepWidth);
  const HlipModel::State atLiftOffY =
      HlipModel::flowDoubleSupport(HlipModel::applyStepTransition(lateralOrbit.second, -roomy.hlip.stepWidth), roomy.hlip.dspDuration);
  reachable.comPosition = reachable.footPositions[CONTACT_RIGHT_INDEX] + vector2_t(0.0, atLiftOffY(0));
  reachable.comVelocity = vector2_t(0.3, atLiftOffY(1));
  const ContactPlan reachablePlan = roomyPlanner.plan(reachable);
  EXPECT_EQ(reachablePlan.numClippedSteps, 0);
  EXPECT_EQ(reachablePlan.describe().find("CLIPPED-STEPS"), std::string::npos) << reachablePlan.describe();
}

TEST(HlipContactPlanner, isPlannedInTheHeadingFrame) {
  // Rotating the whole problem must rotate the whole plan: the geometry is planned in the frame of `yaw`, and that
  // frame has to come from `yaw` even when the heading model is off and `heading` is therefore never filled.
  const ContactPlanningConfig config = makeConfig();
  HlipContactPlanner planner(config);
  ASSERT_FALSE(config.usesHeadingModel());

  const scalar_t yaw = 0.5 * M_PI;
  const Eigen::Matrix<scalar_t, 2, 2> world_R_heading = rotationOf(yaw);

  ContactPlannerInput straight = makeStandingInput(vector2_t(0.5, 0.1));
  straight.contacts[CONTACT_LEFT_INDEX] = false;
  straight.phaseElapsedTime = makeFeetArray(0.0);
  straight.comVelocity = vector2_t(0.3, -0.05);

  ContactPlannerInput turned = straight;
  turned.yaw = yaw;
  turned.velocityCommand = world_R_heading * straight.velocityCommand;
  turned.comVelocity = world_R_heading * straight.comVelocity;
  turned.comPosition = world_R_heading * straight.comPosition;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    turned.footPositions[foot] = world_R_heading * straight.footPositions[foot];
  }

  const ContactPlan straightPlan = planner.plan(straight);
  const ContactPlan turnedPlan = planner.plan(turned);
  ASSERT_TRUE(straightPlan.valid && turnedPlan.valid);
  ASSERT_EQ(straightPlan.footholds.size(), turnedPlan.footholds.size());
  for (size_t node = 0; node < straightPlan.footholds.size(); ++node) {
    EXPECT_EQ(straightPlan.contacts[std::min(node, straightPlan.contacts.size() - 1)],
              turnedPlan.contacts[std::min(node, turnedPlan.contacts.size() - 1)]);
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      const vector2_t expected = world_R_heading * straightPlan.footholds[node][foot];
      EXPECT_NEAR(turnedPlan.footholds[node][foot].x(), expected.x(), 1e-9) << "node " << node << " foot " << foot;
      EXPECT_NEAR(turnedPlan.footholds[node][foot].y(), expected.y(), 1e-9) << "node " << node << " foot " << foot;
    }
  }
}

TEST(HlipContactPlanner, summaryNamesThePlannerItsCadenceAndTheExecutionRulesThatStillRun) {
  ContactPlanningConfig config = makeConfig();
  config.formulation.execution = {"planned_com_override"};
  const std::string summary = HlipContactPlanner::formulationSummary(config);
  EXPECT_NE(summary.find("H-LIP"), std::string::npos);
  EXPECT_NE(summary.find("deadbeat"), std::string::npos);
  EXPECT_NE(summary.find("no optimization"), std::string::npos);
  // The execution list is not gated by planner.type, and planned_com_override is required under hlip: the start-up
  // print used to end "no execution rules", which is how an operator comes to empty the list.
  EXPECT_EQ(summary.find("no execution rules"), std::string::npos) << summary;
  EXPECT_NE(summary.find("execution    : planned_com_override"), std::string::npos) << summary;

  config.formulation.execution.clear();
  EXPECT_NE(HlipContactPlanner::formulationSummary(config).find("execution    : none"), std::string::npos);
}

// ---------------------------------------------------------------------------------------------------------------
// The planner closed around its reduced model at the MPC rate (ReducedModelLoop).
// ---------------------------------------------------------------------------------------------------------------

TEST(HlipContactPlanner, aWalkThatIsReleasedComesBackToStanding) {
  // After any walk the robot must be able to stop. At a zero command the deadbeat law regulates the lateral motion onto
  // the period-two orbit at +-stepWidth, which crosses every touch-down at 0.165 m/s at this cadence - three times the
  // blend's half point for the lateral center-of-mass velocity. The blend used to measure that sway raw, so alpha was
  // one through every double support, the next lift-off was planned and committed at every touch-down, and a released
  // stick left the robot marching in place for ever. The start-from-rest tests could not see it. Both phases the blend
  // measures the sway in are exercised: the last cycles of a swing, whose plan decides whether the next lift-off is
  // committed at the touch-down, and the standing double support after it, in which the harness only gradually slows
  // the center of mass from the orbit's sway (ReducedModelLoop), so a blend that read that sway as motion would step
  // again. Removing either phase from HlipContactPlanner::zeroCommandOrbitLateralVelocity fails this test.
  const ContactPlanningConfig config = makeConfig();
  ReducedModelLoop loop(config, kMpcCyclePeriod);
  loop.setCommand(vector2_t(0.3, 0.0));
  while (loop.time() < 3.0) loop.cycle();
  int touchDowns = 0;
  for (const ReducedModelLoop::Event& event : loop.events()) touchDowns += event.touchDown ? 1 : 0;
  ASSERT_GE(touchDowns, 6) << "positive control: the robot must have been walking before the release";

  const scalar_t releasedAt = loop.time();
  loop.setCommand(vector2_t::Zero());
  std::optional<scalar_t> standingSince;
  while (loop.time() < releasedAt + 3.0) {
    const bool walking = loop.cycle();
    if (walking) {
      standingSince.reset();
    } else if (!standingSince.has_value()) {
      standingSince = loop.time();
    }
  }
  ASSERT_TRUE(standingSince.has_value()) << "the planner was still stepping 3 s after the stick was released";
  EXPECT_LT(*standingSince - releasedAt, 1.5) << "the robot must stand within a few steps of the release";

  // Standing means no further lift-off: only the swings already committed when the plans turned to standing run out.
  for (const ReducedModelLoop::Event& event : loop.events()) {
    if (!event.touchDown) {
      EXPECT_LE(event.time, *standingSince + config.planner.commitTime + 1e-9) << "a lift-off at " << event.time << " after standing began";
    }
  }
}

TEST(HlipContactPlanner, everyExecutedStepIsTheDeadbeatStepForItsExecutedTouchDown) {
  // At runtime every touch-down comes from an earlier plan's grid, so it is off the node grid of the plan that has to
  // place it. The deadbeat step of the swing in flight used to be evaluated at the start of the node the executed
  // touch-down falls in - up to a node early, always early - and the double support after it was counted from that node
  // start, so the next lift-off came up to a node early too. The property that must hold instead: the step the robot
  // lands with is the deadbeat step, u = u* + K (x - x*), of the state it is actually in at the impact it executes.
  // Several MPC periods, so that the touch-downs sample many fractions of a node.
  const ContactPlanningConfig config = makeConfig();
  const HlipModel model(config.hlip.sspDuration, config.hlip.dspDuration, config.shared.comHeight, config.shared.gravity);
  const scalar_t stepDuration = model.stepDuration();
  for (const scalar_t cyclePeriod : {0.01, 0.0173, 0.02, 0.021, 0.025, 0.033}) {
    ReducedModelLoop loop(config, cyclePeriod);
    loop.setCommand(vector2_t(0.3, 0.05));
    while (loop.time() < 4.0) ASSERT_TRUE(loop.cycle());

    int checkedSteps = 0;
    int doubleSupports = 0;
    int swings = 0;
    std::optional<scalar_t> lastTouchDown;
    feet_array_t<std::optional<scalar_t>> liftOffs = makeFeetArray(std::optional<scalar_t>());
    for (const ReducedModelLoop::Event& event : loop.events()) {
      if (!event.touchDown) {
        liftOffs[event.foot] = event.time;
        if (lastTouchDown.has_value()) {
          // The executed double support is hlip.dspDuration exactly - the duration the deadbeat gain K = [1, T_dsp +
          // coth(omega T_ssp) / omega] assumes - whatever the MPC period. It used to be the node grid of whichever plan
          // last re-decided the lift-off that set it: anywhere in dspDuration +- dt / 2, and one systematic value at a
          // fixed period (theExecutedDoubleSupportWasRoundedToTheGridWithoutContinuousEvents is the control).
          EXPECT_NEAR(event.time - *lastTouchDown, config.hlip.dspDuration, 1e-9) << "period " << cyclePeriod << ", at " << event.time;
          ++doubleSupports;
        }
        continue;
      }
      lastTouchDown = event.time;
      if (liftOffs[event.foot].has_value()) {
        // And so is every single support, which the gain assumes as well.
        EXPECT_NEAR(event.time - *liftOffs[event.foot], config.hlip.sspDuration, 1e-9) << "period " << cyclePeriod << ", at " << event.time;
        ++swings;
      }
      // The deadbeat step of the planner's own law, computed here from the executed pre-impact state.
      const vector2_t command = event.alpha * event.command;
      const scalar_t nominalX = command.x() * stepDuration;
      const scalar_t expectedX = model.deadbeatStepLength(event.preImpactX, model.periodOneOrbit(nominalX), nominalX);
      const scalar_t leftStep = config.hlip.stepWidth + command.y() * stepDuration;
      const scalar_t rightStep = -config.hlip.stepWidth + command.y() * stepDuration;
      const std::pair<HlipModel::State, HlipModel::State> orbitY = model.periodTwoOrbit(leftStep, rightStep);
      const bool left = event.foot == CONTACT_LEFT_INDEX;
      const scalar_t expectedY =
          model.deadbeatStepLength(event.preImpactY, left ? orbitY.first : orbitY.second, left ? leftStep : rightStep);
      if (std::abs(expectedX) > config.hlip.maxStepLength || (left ? expectedY : -expectedY) < config.hlip.minStepWidth ||
          (left ? expectedY : -expectedY) > config.hlip.maxStepWidth) {
        continue;  // clipped (the start-up): not the deadbeat step by design
      }
      EXPECT_NEAR(event.step.x(), expectedX, 1e-9) << "period " << cyclePeriod << ", touch-down at " << event.time;
      EXPECT_NEAR(event.step.y(), expectedY, 1e-9) << "period " << cyclePeriod << ", touch-down at " << event.time;
      ++checkedSteps;
    }
    EXPECT_GE(checkedSteps, 10) << "period " << cyclePeriod;
    EXPECT_GE(doubleSupports, 10) << "period " << cyclePeriod;
    EXPECT_GE(swings, 10) << "period " << cyclePeriod;
  }
}

/**
 * The positive control of the exact timing above: the same loop with the executed schedule built from the plans' node
 * grid alone, as it was before plans carried their gait in continuous time. At an MPC period that shares no multiple
 * with the planner's grid the double supports it executes are not hlip.dspDuration, though each stays within half a
 * node of it.
 */
TEST(HlipContactPlanner, theExecutedDoubleSupportWasRoundedToTheGridWithoutContinuousEvents) {
  const ContactPlanningConfig config = makeConfig();
  ReducedModelLoop loop(config, kMpcCyclePeriod);
  loop.useNodeGridSchedule();
  loop.setCommand(vector2_t(0.3, 0.05));
  while (loop.time() < 4.0) ASSERT_TRUE(loop.cycle());
  scalar_t largestError = 0.0;
  int doubleSupports = 0;
  std::optional<scalar_t> lastTouchDown;
  for (const ReducedModelLoop::Event& event : loop.events()) {
    if (event.touchDown) {
      lastTouchDown = event.time;
    } else if (lastTouchDown.has_value()) {
      const scalar_t error = std::abs(event.time - *lastTouchDown - config.hlip.dspDuration);
      EXPECT_LE(error, 0.5 * config.planner.dt + 1e-9) << "at " << event.time;
      largestError = std::max(largestError, error);
      ++doubleSupports;
    }
  }
  EXPECT_GE(doubleSupports, 10);
  EXPECT_GT(largestError, 1e-3) << "the node grid alone executed every double support exactly: the exact-timing test proves nothing";
}

/**
 * The last swung foot outlives the executed schedule's history. The schedule keeps its past only one window deep (the
 * reference manager one horizon, this harness two seconds), so a robot that has stood for longer than that has no
 * lift-off left in it. The planner input used to read the last swung foot off the schedule alone, and it became -1:
 * the blend lost the zero-command orbit it measures the lateral velocity against in double support, and the first step
 * of the next walk was chosen by phase timing instead of alternating with the last one.
 */
TEST(HlipContactPlanner, theLastSwungFootOutlivesTheScheduleHistory) {
  const ContactPlanningConfig config = makeConfig();
  ReducedModelLoop loop(config, kMpcCyclePeriod);
  loop.setCommand(vector2_t(0.3, 0.0));
  while (loop.time() < 3.0) loop.cycle();
  loop.setCommand(vector2_t::Zero());
  while (loop.time() < 9.0) loop.cycle();

  int lastLiftOffFoot = -1;
  scalar_t lastLiftOffTime = 0.0;
  for (const ReducedModelLoop::Event& event : loop.events()) {
    if (event.touchDown) continue;
    lastLiftOffFoot = static_cast<int>(event.foot);
    lastLiftOffTime = event.time;
  }
  ASSERT_GE(lastLiftOffFoot, 0) << "positive control: the robot must have walked";
  ASSERT_LT(lastLiftOffTime, loop.time() - 3.0) << "positive control: the robot must have stood for longer than the schedule reaches back";
  // Positive control: the executed schedule alone no longer holds that lift-off.
  LiftOffHistory scheduleOnly;
  scheduleOnly.record(loop.executedSchedule(), loop.time());
  EXPECT_EQ(scheduleOnly.lastSwungFoot(), -1);

  const ContactPlannerInput input = loop.makeInput();
  EXPECT_EQ(input.lastSwungFoot, lastLiftOffFoot);
  // And with it the zero-command orbit the blend reads the lateral velocity against while standing.
  EXPECT_TRUE(loop.planner().zeroCommandOrbitLateralVelocity(input).has_value());
  // The next walk starts with the other foot.
  EXPECT_EQ(static_cast<int>(HlipContactPlanner::nextSwingFoot(input)), 1 - lastLiftOffFoot);
}

/** The history forgets a lift-off later than the time it is asked at: time ran backwards, as it does on a reset. */
TEST(HlipContactPlanner, theLiftOffHistoryForgetsTheFutureAfterAReset) {
  contact_flag_t leftSwinging = makeFeetArray(true);
  leftSwinging[CONTACT_LEFT_INDEX] = false;
  const ModeSchedule walked({5.0, 5.25}, {ModeNumber::STANCE, stanceLeg2ModeNumber(leftSwinging), ModeNumber::STANCE});
  LiftOffHistory history;
  history.record(walked, /*time=*/4.9);
  EXPECT_EQ(history.lastSwungFoot(), -1) << "a lift-off after the query time has not happened";
  history.record(walked, /*time=*/6.0);
  EXPECT_EQ(history.lastSwungFoot(), static_cast<int>(CONTACT_LEFT_INDEX));
  history.record(ModeSchedule({}, {ModeNumber::STANCE}), /*time=*/0.5);
  EXPECT_EQ(history.lastSwungFoot(), -1) << "after the reset the lift-off at 5 s lies in the future";
}

/**
 * A plan records the pendulum it was made on (ContactPlan::omega): the terminal DCM cost takes the plan's DCM, and the
 * robot's, with that omega, so that a reload of shared.comHeight cannot re-interpret a plan already made. And a
 * configuration the H-LIP cannot run is refused by its key, keeping the running one: the model CHECK-fails on it, which
 * on the planner's worker thread used to take the controller down.
 */
TEST(HlipContactPlanner, aPlanCarriesItsPendulumAndARefusedReloadKeepsTheRunningConfiguration) {
  ContactPlanningConfig config = makeConfig();
  config.shared.comHeight = 1.05;
  HlipContactPlanner planner(config);
  const ContactPlan plan = planner.plan(makeStandingInput(vector2_t::Zero()));
  ASSERT_TRUE(plan.valid);
  EXPECT_NEAR(plan.omega, std::sqrt(config.shared.gravity / 1.05), 1e-12) << "a plan must record the pendulum it was made on";

  // A reload that moves the pendulum: the next plan is made, and recorded, on the new one.
  ContactPlanningConfig lower = config;
  lower.shared.comHeight = 0.8;
  ASSERT_EQ(planner.setConfig(lower), absl::OkStatus());
  const scalar_t lowerOmega = std::sqrt(lower.shared.gravity / 0.8);
  ASSERT_GT(std::abs(lowerOmega - plan.omega), 0.3) << "the reload must move the pendulum, or the check below proves nothing";
  EXPECT_NEAR(planner.plan(makeStandingInput(vector2_t::Zero())).omega, lowerOmega, 1e-12);

  ContactPlanningConfig broken = lower;
  broken.shared.comHeight = -1.0;
  const absl::Status refused = planner.setConfig(broken);
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(refused.message(), "shared.comHeight")) << refused;
  EXPECT_DOUBLE_EQ(planner.getConfig().shared.comHeight, 0.8) << "a refused reload must keep the running configuration";
  EXPECT_NEAR(planner.getModel().naturalFrequency(), lowerOmega, 1e-12);
  EXPECT_NEAR(planner.plan(makeStandingInput(vector2_t::Zero())).omega, lowerOmega, 1e-12);
}

}  // namespace
}  // namespace ocs2::humanoid
