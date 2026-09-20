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
#include <vector>

#include "humanoid_common_mpc/contact_planning/hlip/HlipContactPlanner.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kStepWidth = 0.25;
constexpr scalar_t kDt = 0.025;
constexpr scalar_t kSspDuration = 0.25;
constexpr scalar_t kDspDuration = 0.05;

ContactPlanningConfig makeConfig() {
  ContactPlanningConfig config;
  config.planner.type = "hlip";
  config.planner.dt = kDt;
  config.planner.numNodes = 56;
  config.planner.commitTime = 0.05;
  config.planner.runInBackgroundThread = false;
  config.shared.comHeight = 0.85;
  config.shared.gaitLimits.minSwingDuration = 0.25;
  config.shared.gaitLimits.maxSwingDuration = 0.35;
  config.shared.gaitLimits.minDoubleSupportDuration = 0.0;
  config.hlip.sspDuration = kSspDuration;
  config.hlip.dspDuration = kDspDuration;
  config.hlip.stepWidth = kStepWidth;
  config.validate();
  return config;
}

ContactPlannerInput makeStandingInput(const vector2_t& velocityCommand) {
  ContactPlannerInput input;
  input.time = 0.0;
  input.comPosition = vector2_t(0.0, 0.0);
  input.comVelocity = vector2_t::Zero();
  input.footPositions[CONTACT_LEFT_INDEX] = vector2_t(0.0, 0.5 * kStepWidth);
  input.footPositions[CONTACT_RIGHT_INDEX] = vector2_t(0.0, -0.5 * kStepWidth);
  input.contacts = makeFeetArray(true);
  input.phaseElapsedTime = makeFeetArray(1.0);  // a long stance: the robot has been standing for a while
  input.velocityCommand = velocityCommand;
  input.committedUntil = input.time;
  return input;
}

/** The contact state the gait itself describes at `time`, independent of how the plan was sampled onto nodes. */
contact_flag_t gaitContactsAt(const std::vector<HlipContactPlanner::GaitPhase>& gait, scalar_t time) {
  for (const HlipContactPlanner::GaitPhase& phase : gait) {
    if (time < phase.endTime) return phase.contacts;
  }
  return gait.back().contacts;
}

// ---------------------------------------------------------------------------------------------------------------
// B2 - the gait is built FROM the committed window, so the contacts, the footholds and the centre-of-mass roll-out
// all describe one gait. It used to be built as if the window were free and the window stamped over the contacts at
// the end, leaving the roll-out describing a different gait from the one that was published.
// ---------------------------------------------------------------------------------------------------------------

TEST(HlipPlanConsistency, theGaitStartsFromTheCommittedWindowRatherThanOverwritingIt) {
  HlipContactPlanner planner(makeConfig());
  ContactPlannerInput input = makeStandingInput(vector2_t(0.5, 0.0));
  // The reference manager has committed the first two intervals to double support, which is what it does out of a
  // stance: the commit window covers the planner's own latency.
  input.committedContacts.assign(2, makeFeetArray(true));
  input.committedUntil = input.time + 2.0 * kDt;

  const std::vector<HlipContactPlanner::GaitPhase> gait = planner.buildGait(input, true);

  // The gait's FIRST phase is the committed double support, not a swing starting at node 0. Out of a long stance the
  // nominal cadence used to lift a foot immediately, because `dspDuration - elapsed` was negative.
  ASSERT_FALSE(gait.empty());
  EXPECT_EQ(gait.front().contacts, makeFeetArray(true));
  EXPECT_FALSE(gait.front().isSingleSupport());
  EXPECT_NEAR(gait.front().endTime, input.time + 2.0 * kDt, 1e-9)
      << "the first phase must end where the committed window ends, not where the nominal cadence would";

  // The first single support therefore starts at the end of the committed window.
  const std::vector<HlipContactPlanner::GaitPhase>::const_iterator firstSwing =
      std::find_if(gait.begin(), gait.end(), [](const HlipContactPlanner::GaitPhase& phase) { return phase.isSingleSupport(); });
  ASSERT_NE(firstSwing, gait.end());
  EXPECT_NEAR(firstSwing->startTime, input.time + 2.0 * kDt, 1e-9);
  EXPECT_NEAR(firstSwing->duration(), kSspDuration, 1e-9) << "a whole single support, not one cut short by the window";
}

TEST(HlipPlanConsistency, thePublishedContactsAgreeWithTheGaitTheRollOutUsed) {
  HlipContactPlanner planner(makeConfig());
  ContactPlannerInput input = makeStandingInput(vector2_t(0.4, 0.0));
  input.committedContacts.assign(2, makeFeetArray(true));
  input.committedUntil = input.time + 2.0 * kDt;

  const ContactPlan plan = planner.plan(input);
  const std::vector<HlipContactPlanner::GaitPhase> gait = planner.buildGait(input, true);

  // Every published interval must carry the contact state of the gait the footholds were rolled out against. The two
  // are filled from the same integer phase boundaries, so they cannot drift apart.
  ASSERT_EQ(plan.contacts.size(), static_cast<size_t>(plan.numIntervals()));
  for (int interval = 0; interval < plan.numIntervals(); ++interval) {
    const scalar_t intervalStart = plan.startTime + kDt * static_cast<scalar_t>(interval);
    // The boundary is rounded to the nearest node, so compare against the gait at the interval's own rounded time.
    const contact_flag_t expected = gaitContactsAt(gait, intervalStart + 0.5 * kDt);
    EXPECT_EQ(plan.contacts[interval], expected) << "interval " << interval;
  }
}

TEST(HlipPlanConsistency, theCommittedContactsAreStillHonouredExactly) {
  HlipContactPlanner planner(makeConfig());
  ContactPlannerInput input = makeStandingInput(vector2_t(0.4, 0.0));
  // An arbitrary committed pattern, including a swing the nominal cadence would not have chosen.
  input.committedContacts.assign(4, makeFeetArray(true));
  input.committedContacts[2][CONTACT_RIGHT_INDEX] = false;
  input.committedContacts[3][CONTACT_RIGHT_INDEX] = false;
  input.committedUntil = input.time + 4.0 * kDt;

  const ContactPlan plan = planner.plan(input);
  for (size_t interval = 0; interval < input.committedContacts.size(); ++interval) {
    EXPECT_EQ(plan.contacts[interval], input.committedContacts[interval]) << "interval " << interval;
  }
}

TEST(HlipPlanConsistency, aSwingCommittedInFlightIsContinuedForItsRemainingDurationOnly) {
  HlipContactPlanner planner(makeConfig());
  ContactPlannerInput input = makeStandingInput(vector2_t(0.4, 0.0));
  // The left foot has been in flight for 0.10 s already, and the window commits the next two intervals to that swing.
  input.contacts[CONTACT_LEFT_INDEX] = false;
  input.phaseElapsedTime[CONTACT_LEFT_INDEX] = 0.10;
  input.phaseElapsedTime[CONTACT_RIGHT_INDEX] = 0.40;
  input.committedContacts.assign(2, input.contacts);
  input.committedUntil = input.time + 2.0 * kDt;

  const std::vector<HlipContactPlanner::GaitPhase> gait = planner.buildGait(input, true);

  // One single-support phase, running from the plan start to 0.25 - 0.10 = 0.15 s: the committed window is part of
  // that swing, not an extra phase in front of it, and the swing is not restarted from the end of the window.
  ASSERT_FALSE(gait.empty());
  EXPECT_EQ(gait.front().swingFoot, static_cast<int>(CONTACT_LEFT_INDEX));
  EXPECT_NEAR(gait.front().endTime, input.time + kSspDuration - 0.10, 1e-9)
      << "the elapsed flight time before the plan began must still count against the swing";
}

// ---------------------------------------------------------------------------------------------------------------
// B3 - one rounding rule. Nodes and intervals used to be assigned to phases by two different comparisons.
// ---------------------------------------------------------------------------------------------------------------

TEST(HlipPlanConsistency, theFootholdChangesOnTheSameNodeTheContactDoes) {
  HlipContactPlanner planner(makeConfig());
  ContactPlannerInput input = makeStandingInput(vector2_t(0.4, 0.0));
  input.committedContacts.assign(2, makeFeetArray(true));
  input.committedUntil = input.time + 2.0 * kDt;

  const ContactPlan plan = planner.plan(input);

  // A foot's planned landing spot is written for every node of the swing that places it. So the node at which a foot's
  // foothold starts moving must be the node at which its contact flag says it left the ground - never one node out.
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    for (int interval = 1; interval < plan.numIntervals(); ++interval) {
      const bool liftsOff = plan.contacts[interval - 1][foot] && !plan.contacts[interval][foot];
      if (!liftsOff) continue;
      // At the lift-off interval the landing target is already in place, and it differs from where the foot was.
      const vector2_t before = plan.footholds[static_cast<size_t>(interval - 1)][foot];
      const vector2_t after = plan.footholds[static_cast<size_t>(interval)][foot];
      EXPECT_GT((after - before).norm(), 1e-9) << "foot " << foot << " lifts off at interval " << interval
                                               << " but its foothold had not moved by then";
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// B1 - a stepping plan publishes the centre-of-mass trajectory its own footholds were placed for. The standing blend
// towards the support centre applies only while standing.
// ---------------------------------------------------------------------------------------------------------------

TEST(HlipPlanConsistency, aSteppingPlanPublishesTheUnblendedRollOut) {
  const ContactPlanningConfig config = makeConfig();
  HlipContactPlanner planner(config);
  // A command just above the blend's half point: alpha is close to 0.5, which is exactly where the blend used to halve
  // the lateral sway the deadbeat step had been computed against.
  ContactPlannerInput input = makeStandingInput(vector2_t(0.105, 0.0));
  // The centre of mass must be AWAY from the centre of the support, or the blend towards that centre is the identity
  // and the test proves nothing: makeStandingInput puts both at the origin.
  input.comPosition = vector2_t(0.06, 0.04);
  const vector2_t supportCentre = 0.5 * (input.footPositions[CONTACT_LEFT_INDEX] + input.footPositions[CONTACT_RIGHT_INDEX]);
  ASSERT_GT((input.comPosition - supportCentre).norm(), 0.05) << "the blend would be a no-op at the support centre";
  const scalar_t blendWeight = planner.getBlend().weight(input.velocityCommand, 0.0, input.comVelocity);
  ASSERT_GE(blendWeight, 0.5) << "this test needs a walking plan";
  ASSERT_LT(blendWeight, 0.95) << "this test needs alpha well below 1, where the blend used to bite";

  const ContactPlan plan = planner.plan(input);

  // The roll-out starts at the measured state exactly: no blend towards the support centre is mixed in.
  ASSERT_FALSE(plan.comPosition.empty());
  EXPECT_NEAR(plan.comPosition.front().x(), input.comPosition.x(), 1e-12);
  EXPECT_NEAR(plan.comPosition.front().y(), input.comPosition.y(), 1e-12);
  EXPECT_NEAR(plan.comVelocity.front().x(), input.comVelocity.x(), 1e-12);
  EXPECT_NEAR(plan.comVelocity.front().y(), input.comVelocity.y(), 1e-12);

  // And the published trajectory is the pendulum's own. Compare against what the blend WOULD have produced: every
  // node must be the unblended roll-out, not a mixture of it and the support centre. This is the assertion that fails
  // on the pre-fix code, at every node rather than only the first.
  ASSERT_EQ(plan.comPosition.size(), plan.comVelocity.size());
  scalar_t maxBlendGap = 0.0;
  for (size_t node = 0; node < plan.comPosition.size(); ++node) {
    const vector2_t blended = blendWeight * plan.comPosition[node] + (1.0 - blendWeight) * supportCentre;
    maxBlendGap = std::max(maxBlendGap, (plan.comPosition[node] - blended).norm());
  }
  EXPECT_GT(maxBlendGap, 0.01) << "the published centre of mass is indistinguishable from the blended one, so this "
                                  "test cannot tell the fix from the defect";

  // The lateral sway the period-two orbit requires is present and is NOT scaled down by alpha.
  scalar_t maxLateral = 0.0;
  for (const vector2_t& position : plan.comPosition) {
    maxLateral = std::max(maxLateral, std::abs(position.y()));
  }
  EXPECT_GT(maxLateral, 0.02) << "a stepping plan must ask the controller for the lateral sway its steps assume";
}

TEST(HlipPlanConsistency, aStandingPlanStillAsksForACentreOfMassAtRestOverTheFeet) {
  HlipContactPlanner planner(makeConfig());
  ContactPlannerInput input = makeStandingInput(vector2_t::Zero());
  input.comPosition = vector2_t(0.03, 0.01);
  input.comVelocity = vector2_t(0.02, -0.01);  // a small drift, well below the blend's half point

  const ContactPlan plan = planner.plan(input);
  ASSERT_FALSE(planner.isWalking(input));

  const scalar_t blendWeight = planner.getBlend().weight(input.velocityCommand, input.headingRateCommand, input.comVelocity);
  const vector2_t expectedVelocity = blendWeight * input.comVelocity;
  EXPECT_NEAR(plan.comVelocity.back().x(), expectedVelocity.x(), 1e-9) << "standing must ask for a stop, not a drift";
  EXPECT_NEAR(plan.comVelocity.back().y(), expectedVelocity.y(), 1e-9);
  const vector2_t supportCentre = 0.5 * (input.footPositions[CONTACT_LEFT_INDEX] + input.footPositions[CONTACT_RIGHT_INDEX]);
  EXPECT_LT((plan.comPosition.back() - supportCentre).norm(), 0.02) << "standing must ask for the centre of the support";
}

// ---------------------------------------------------------------------------------------------------------------
// B5 - the reference handed to the whole-body MPC starts where the robot actually is.
// ---------------------------------------------------------------------------------------------------------------

TEST(HlipPlanConsistency, theFirstNodeOfASteppingPlanIsTheMeasuredState) {
  HlipContactPlanner planner(makeConfig());
  ContactPlannerInput input = makeStandingInput(vector2_t(0.5, 0.0));
  input.comPosition = vector2_t(0.07, -0.02);
  input.comVelocity = vector2_t(0.45, 0.06);
  ASSERT_TRUE(planner.isWalking(input));

  const ContactPlan plan = planner.plan(input);
  // planned_com_override writes this straight into the whole-body MPC's reference. A first node that is not the
  // measured state is a step in the reference at the current instant, every cycle.
  EXPECT_NEAR(plan.comPosition.front().x(), input.comPosition.x(), 1e-12);
  EXPECT_NEAR(plan.comPosition.front().y(), input.comPosition.y(), 1e-12);
  EXPECT_NEAR(plan.comVelocity.front().x(), input.comVelocity.x(), 1e-12);
  EXPECT_NEAR(plan.comVelocity.front().y(), input.comVelocity.y(), 1e-12);
}

}  // namespace
}  // namespace ocs2::humanoid
