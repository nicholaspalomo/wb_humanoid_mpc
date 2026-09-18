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
#include <utility>
#include <vector>

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
  config.shared.gaitLimits.minSwingDuration = 0.3;
  config.shared.gaitLimits.maxSwingDuration = 0.4;
  config.shared.gaitLimits.minDoubleSupportDuration = 0.0;
  config.hlip.sspDuration = 0.35;
  config.hlip.dspDuration = 0.0;
  config.hlip.stepWidth = kStepWidth;
  config.validate();
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

TEST(HlipContactPlanner, standingAsksForACentreOfMassAtRest) {
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
  const scalar_t blendWeight = planner.getBlend().weight(input.velocityCommand, input.headingRateCommand, input.comVelocity);
  ASSERT_LT(blendWeight, 0.5);
  const vector2_t expectedVelocity = blendWeight * input.comVelocity;
  EXPECT_NEAR(plan.comVelocity.back().x(), expectedVelocity.x(), 1e-9) << "standing must ask for a stop, not a drift";
  EXPECT_NEAR(plan.comVelocity.back().y(), expectedVelocity.y(), 1e-9) << "standing must ask for a stop, not a drift";

  const vector2_t supportCentre = 0.5 * (input.footPositions[CONTACT_LEFT_INDEX] + input.footPositions[CONTACT_RIGHT_INDEX]);
  EXPECT_LT((plan.comPosition.back() - supportCentre).norm(), 0.02) << "standing must ask for the centre of the support";
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
  const std::vector<HlipContactPlanner::GaitPhase> gait = planner.buildGait(makeStandingInput(vector2_t(0.5, 0.0)), true);

  ASSERT_GE(gait.size(), 2U);
  for (const HlipContactPlanner::GaitPhase& phase : gait) {
    if (phase.isSingleSupport()) EXPECT_NEAR(phase.duration(), config.hlip.sspDuration, 1e-12);
  }
  // The gait is contiguous and covers the horizon.
  for (size_t index = 1; index < gait.size(); ++index) {
    EXPECT_NEAR(gait[index].startTime, gait[index - 1].endTime, 1e-12);
  }
  EXPECT_GE(gait.back().endTime, gait.front().startTime + config.horizon() - 1e-12);
}

TEST(HlipContactPlanner, continuesTheSwingInFlight) {
  const ContactPlanningConfig config = makeConfig();
  HlipContactPlanner planner(config);
  ContactPlannerInput input = makeStandingInput(vector2_t(0.5, 0.0));
  input.contacts[CONTACT_LEFT_INDEX] = false;  // the left foot is in flight, a third of the way through its swing
  input.phaseElapsedTime[CONTACT_LEFT_INDEX] = 0.1;
  input.phaseElapsedTime[CONTACT_RIGHT_INDEX] = 0.1;

  const std::vector<HlipContactPlanner::GaitPhase> gait = planner.buildGait(input, true);
  ASSERT_FALSE(gait.empty());
  EXPECT_EQ(gait.front().swingFoot, static_cast<int>(CONTACT_LEFT_INDEX));
  EXPECT_NEAR(gait.front().duration(), config.hlip.sspDuration - 0.1, 1e-12);
  ASSERT_GE(gait.size(), 2U);
  EXPECT_EQ(gait[1].swingFoot, static_cast<int>(CONTACT_RIGHT_INDEX));
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
  const HlipModel::State postImpactX = HlipModel::applyStepTransition(orbit, nominalStep);
  const HlipModel::State postImpactY = HlipModel::applyStepTransition(lateralOrbit.second, -config.hlip.stepWidth);
  const vector2_t stanceFoot = input.footPositions[CONTACT_RIGHT_INDEX];
  input.comPosition = stanceFoot + vector2_t(postImpactX(0), postImpactY(0));
  input.comVelocity = vector2_t(postImpactX(1), postImpactY(1));
  // The blend must not scale the command down, or the step would be shorter than the orbit's.
  ASSERT_GE(planner.getBlend().weight(input.velocityCommand, 0.0, input.comVelocity), 0.999);

  const ContactPlan plan = planner.plan(input);
  ASSERT_TRUE(plan.valid);
  const std::optional<vector2_t> landing = plan.footholdAtTime(CONTACT_LEFT_INDEX, config.hlip.sspDuration);
  ASSERT_TRUE(landing.has_value());
  EXPECT_NEAR(landing->x() - stanceFoot.x(), nominalStep, 1e-6);
  EXPECT_NEAR(landing->y() - stanceFoot.y(), config.hlip.stepWidth, 1e-6);
}

TEST(HlipContactPlanner, honoursTheCommittedContacts) {
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
    input.contacts = makeFeetArray(true);
    input.contacts[stanceFoot] = false;  // the roles swap: the old stance foot swings next
    input.phaseElapsedTime = makeFeetArray(0.0);
    input.lastSwungFoot = static_cast<int>(swingFoot);
    input.committedUntil = input.time;
  }

  EXPECT_NEAR(lastStepLength / model.stepDuration(), command, 1e-3);
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
}

TEST(HlipContactPlanner, isPlannedInTheHeadingFrame) {
  // Rotating the whole problem must rotate the whole plan: the geometry is planned in the frame of `yaw`, and that
  // frame has to come from `yaw` even when the heading model is off and `heading` is therefore never filled.
  const ContactPlanningConfig config = makeConfig();
  HlipContactPlanner planner(config);
  ASSERT_FALSE(config.usesHeadingModel());

  const scalar_t yaw = 0.5 * M_PI;
  Eigen::Matrix<scalar_t, 2, 2> world_R_heading;
  world_R_heading << std::cos(yaw), -std::sin(yaw), std::sin(yaw), std::cos(yaw);

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

TEST(HlipContactPlanner, summaryNamesThePlannerAndItsCadence) {
  const std::string summary = HlipContactPlanner::formulationSummary(makeConfig());
  EXPECT_NE(summary.find("H-LIP"), std::string::npos);
  EXPECT_NE(summary.find("deadbeat"), std::string::npos);
  EXPECT_NE(summary.find("no optimization"), std::string::npos);
}

}  // namespace
}  // namespace ocs2::humanoid
