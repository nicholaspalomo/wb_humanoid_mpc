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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"
#include "humanoid_common_mpc/contact_planning/hlip/HlipContactPlanner.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kStepWidth = 0.25;
constexpr scalar_t kDt = 0.025;
constexpr scalar_t kSspDuration = 0.25;
constexpr scalar_t kDspDuration = 0.05;

using GaitPhase = HlipContactPlanner::GaitPhase;

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
  EXPECT_EQ(config.validateStatus(), absl::OkStatus());
  return config;
}

ContactPlannerInput makeStandingInput(const vector2_t& velocityCommand) {
  ContactPlannerInput input;
  input.time = 0.0;
  input.comPosition = vector2_t(0.0, 0.0);
  input.comVelocity = vector2_t::Zero();
  input.footPositions[kContactLeftIndex] = vector2_t(0.0, 0.5 * kStepWidth);
  input.footPositions[kContactRightIndex] = vector2_t(0.0, -0.5 * kStepWidth);
  input.contacts = makeFeetArray(true);
  input.phaseElapsedTime = makeFeetArray(1.0);  // a long stance: the robot has been standing for a while
  input.velocityCommand = velocityCommand;
  input.committedUntil = input.time;
  return input;
}

size_t modeWithSwinging(size_t foot) {
  contact_flag_t contacts = makeFeetArray(true);
  contacts[foot] = false;
  return stanceLeg2ModeNumber(contacts);
}

/**
 * The executed schedule of a left swing in flight from `liftOff` to `touchDown`, and the planner input the reference
 * manager builds from it at `time` (ContactPlanningReferenceManager::makePlannerInput, through the same
 * fillPlannerInputFromSchedule()): the commit boundary extended to the touch-down, the committed nodes sampled with
 * committedContactsForPlanner - the last one AT the boundary, where the foot is already down - and their event times
 * with committedPhaseStartsForPlanner.
 */
ContactPlannerInput makeInputWithLeftSwingInFlight(scalar_t time,
                                                   scalar_t liftOff,
                                                   scalar_t touchDown,
                                                   const ContactPlanningConfig& config) {
  const ModeSchedule executed({liftOff, touchDown}, {ModeNumber::kStance, modeWithSwinging(kContactLeftIndex), ModeNumber::kStance});
  ContactPlannerInput input = makeStandingInput(vector2_t::Zero());
  input.time = time;
  input.footPositions[kContactLeftIndex] = vector2_t(-0.05, 0.5 * kStepWidth);  // measured in the air, mid-swing
  input.committedUntil = commitBoundaryForSchedule(executed, time, config.planner.commitTime, config.planner.maxCommitExtension);
  LiftOffHistory liftOffHistory;
  fillPlannerInputFromSchedule(executed, config.planner.dt, config.planner.numNodes - 1, liftOffHistory, input);
  EXPECT_EQ(input.lastSwungFoot, static_cast<int>(kContactLeftIndex));
  return input;
}

/** The lateral and sagittal H-LIP state at the lift-off of the left foot, on the orbit of `velocity` along x. */
std::pair<HlipModel::State, HlipModel::State> orbitAtLeftLiftOff(const HlipModel& model, scalar_t velocity) {
  const scalar_t nominalStepX = velocity * model.stepDuration();
  const HlipModel::State orbitX = model.periodOneOrbit(nominalStepX);
  const std::pair<HlipModel::State, HlipModel::State> orbitY = model.periodTwoOrbit(kStepWidth, -kStepWidth);
  // The right step's impact, carried through the double support.
  return {HlipModel::flowDoubleSupport(HlipModel::applyStepTransition(orbitX, nominalStepX), model.dspDuration()),
          HlipModel::flowDoubleSupport(HlipModel::applyStepTransition(orbitY.second, -kStepWidth), model.dspDuration())};
}

/** The contact state the gait itself describes at `time`, independent of how the plan was sampled onto nodes. */
contact_flag_t gaitContactsAt(const std::vector<GaitPhase>& gait, scalar_t time) {
  for (const GaitPhase& phase : gait) {
    if (time < phase.endTime) return phase.contacts;
  }
  return gait.back().contacts;
}

/**
 * The center of mass at every node, rolled out INDEPENDENTLY of the planner: the H-LIP flowed from the measured state
 * through the gait's phases in continuous time, about the stance foot the plan itself places, with the double
 * supports drifting at constant velocity. The pendulum is isotropic, so it is flowed in world axes rather than in any
 * heading frame. Returns (position, velocity) per node.
 */
std::vector<std::pair<vector2_t, vector2_t>> independentRollOut(const HlipModel& model,
                                                                const ContactPlannerInput& input,
                                                                const std::vector<GaitPhase>& gait,
                                                                const ContactPlan& plan) {
  struct Start {
    vector2_t position;
    vector2_t velocity;
    vector2_t stance;
  };
  std::vector<Start> starts;
  vector2_t position = input.comPosition;
  vector2_t velocity = input.comVelocity;
  for (const GaitPhase& phase : gait) {
    Start start{.position = position, .velocity = velocity, .stance = vector2_t::Zero()};
    if (phase.isSingleSupport()) {
      const size_t stanceFoot = phase.swingFoot == static_cast<int>(kContactLeftIndex) ? kContactRightIndex : kContactLeftIndex;
      // The stance foot is down from before this phase until after it, so the node it ends on still holds it.
      start.stance = plan.footholds[static_cast<size_t>(std::min(phase.endNode, plan.numIntervals()))][stanceFoot];
      const HlipModel::State x = model.flowSingleSupport(HlipModel::State(position.x() - start.stance.x(), velocity.x()), phase.duration());
      const HlipModel::State y = model.flowSingleSupport(HlipModel::State(position.y() - start.stance.y(), velocity.y()), phase.duration());
      position = start.stance + vector2_t(x(0), y(0));
      velocity = vector2_t(x(1), y(1));
    } else {
      position += phase.duration() * velocity;
    }
    starts.push_back(start);
  }
  std::vector<std::pair<vector2_t, vector2_t>> nodes;
  for (int node = 0; node <= plan.numIntervals(); ++node) {
    const scalar_t time = plan.startTime + plan.dt * static_cast<scalar_t>(node);
    size_t index = 0;
    while (index + 1 < gait.size() && time >= gait[index].endTime) ++index;
    const Start& start = starts[index];
    const scalar_t elapsed = std::max(0.0, time - gait[index].startTime);
    if (gait[index].isSingleSupport()) {
      const HlipModel::State x =
          model.flowSingleSupport(HlipModel::State(start.position.x() - start.stance.x(), start.velocity.x()), elapsed);
      const HlipModel::State y =
          model.flowSingleSupport(HlipModel::State(start.position.y() - start.stance.y(), start.velocity.y()), elapsed);
      nodes.emplace_back(start.stance + vector2_t(x(0), y(0)), vector2_t(x(1), y(1)));
    } else {
      nodes.emplace_back(start.position + elapsed * start.velocity, start.velocity);
    }
  }
  return nodes;
}

// ---------------------------------------------------------------------------------------------------------------
// B2 - the gait is built FROM the committed window, so the contacts, the footholds and the center-of-mass roll-out
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

  const std::vector<GaitPhase> gait = planner.buildGait(input, /*walking=*/true);

  // The gait's FIRST phase is the committed double support, not a swing starting at node 0. Out of a long stance the
  // nominal cadence used to lift a foot immediately, because `dspDuration - elapsed` was negative.
  ASSERT_FALSE(gait.empty());
  EXPECT_EQ(gait.front().contacts, makeFeetArray(true));
  EXPECT_FALSE(gait.front().isSingleSupport());
  EXPECT_NEAR(gait.front().endTime, input.time + 2.0 * kDt, 1.0e-9)
      << "the first phase must end where the committed window ends, not where the nominal cadence would";
  EXPECT_EQ(gait.front().endNode, 2);

  // The first single support therefore starts at the end of the committed window.
  const std::vector<GaitPhase>::const_iterator firstSwing =
      std::find_if(gait.begin(), gait.end(), [](const GaitPhase& phase) { return phase.isSingleSupport(); });
  ASSERT_NE(firstSwing, gait.end());
  EXPECT_NEAR(firstSwing->startTime, input.time + 2.0 * kDt, 1.0e-9);
  EXPECT_NEAR(firstSwing->duration(), kSspDuration, 1.0e-9) << "a whole single support, not one cut short by the window";
}

TEST(HlipPlanConsistency, theCommittedContactsAreStillHonoredExactly) {
  HlipContactPlanner planner(makeConfig());
  ContactPlannerInput input = makeStandingInput(vector2_t(0.4, 0.0));
  // An arbitrary committed pattern, including a swing the nominal cadence would not have chosen.
  input.committedContacts.assign(4, makeFeetArray(true));
  input.committedContacts[2][kContactRightIndex] = false;
  input.committedContacts[3][kContactRightIndex] = false;
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
  input.contacts[kContactLeftIndex] = false;
  input.phaseElapsedTime[kContactLeftIndex] = 0.10;
  input.phaseElapsedTime[kContactRightIndex] = 0.40;
  input.committedContacts.assign(2, input.contacts);
  input.committedUntil = input.time + 2.0 * kDt;

  const std::vector<GaitPhase> gait = planner.buildGait(input, /*walking=*/true);

  // One single-support phase, running from the plan start to 0.25 - 0.10 = 0.15 s: the committed window is part of
  // that swing, not an extra phase in front of it, and the swing is not restarted from the end of the window.
  ASSERT_FALSE(gait.empty());
  EXPECT_EQ(gait.front().swingFoot, static_cast<int>(kContactLeftIndex));
  EXPECT_NEAR(gait.front().endTime, input.time + kSspDuration - 0.10, 1.0e-9)
      << "the elapsed flight time before the plan began must still count against the swing";
}

// ---------------------------------------------------------------------------------------------------------------
// A48 - the swing in flight is planned for the touch-down the robot executes. The reference manager extends the commit
// boundary to that touch-down and samples the last committed node AT the boundary, where the foot is already down, so
// replaying the committed contacts as whole nodes ended the swing at the START of the node its touch-down falls in and
// evaluated the deadbeat step up to a node early.
// ---------------------------------------------------------------------------------------------------------------

TEST(HlipPlanConsistency, theSwingInFlightIsPlannedForTheTouchDownTheRobotExecutes) {
  const ContactPlanningConfig config = makeConfig();
  HlipContactPlanner planner(config);
  const HlipModel& model = planner.getModel();
  const scalar_t velocity = 0.5;
  const scalar_t nominalStepX = velocity * model.stepDuration();
  const std::pair<HlipModel::State, HlipModel::State> atLiftOff = orbitAtLeftLiftOff(model, velocity);

  // The touch-down on a node (0.150 s ahead), between nodes, and just after and just before one, from two planning
  // instants, one of them off the grid of the other.
  for (const scalar_t planTime : {0.0, 0.013}) {
    for (const scalar_t ahead : {0.150, 0.137, 0.126, 0.149}) {
      const scalar_t touchDown = planTime + ahead;
      const scalar_t liftOff = touchDown - kSspDuration;
      ContactPlannerInput input = makeInputWithLeftSwingInFlight(planTime, liftOff, touchDown, config);
      ASSERT_NEAR(input.committedUntil, touchDown, 1.0e-12) << "the commit boundary must reach the touch-down";
      ASSERT_TRUE(input.committedContacts.back()[kContactLeftIndex]) << "the last committed node is sampled at the touch-down";
      // Exactly on the orbit of the command, the swing in flight for ssp - ahead.
      const vector2_t stance = input.footPositions[kContactRightIndex];
      const HlipModel::State x = model.flowSingleSupport(atLiftOff.first, kSspDuration - ahead);
      const HlipModel::State y = model.flowSingleSupport(atLiftOff.second, kSspDuration - ahead);
      input.comPosition = stance + vector2_t(x(0), y(0));
      input.comVelocity = vector2_t(x(1), y(1));
      input.velocityCommand = vector2_t(velocity, 0.0);
      ASSERT_DOUBLE_EQ(planner.blendWeight(input), 1.0);

      const std::string label = absl::StrCat("plan at ", planTime, ", touch-down ", ahead, " s ahead");
      // The deadbeat law leaves a robot on the orbit alone: it asks for exactly the orbit's step.
      const ContactPlan plan = planner.plan(input);
      ASSERT_TRUE(plan.valid);
      const std::optional<vector2_t> landing = plan.footholdAtTime(kContactLeftIndex, touchDown);
      if (!landing.has_value()) GTEST_FAIL();
      EXPECT_NEAR(landing->x() - stance.x(), nominalStepX, 1.0e-9) << label;
      EXPECT_NEAR(landing->y() - stance.y(), kStepWidth, 1.0e-9) << label;

      // The swing ends at the executed touch-down, and the double support after it lasts dspDuration from there.
      const std::vector<GaitPhase> gait = planner.buildGait(input, /*walking=*/true);
      ASSERT_GE(gait.size(), 3U);
      EXPECT_EQ(gait[0].swingFoot, static_cast<int>(kContactLeftIndex)) << label;
      EXPECT_NEAR(gait[0].endTime, touchDown, 1.0e-12) << label;
      EXPECT_FALSE(gait[1].isSingleSupport()) << label;
      EXPECT_NEAR(gait[1].endTime, touchDown + kDspDuration, 1.0e-12) << label;
      // The grid still carries the committed contacts exactly.
      for (size_t interval = 0; interval < input.committedContacts.size(); ++interval) {
        EXPECT_EQ(plan.contacts[interval], input.committedContacts[interval]) << label << ", interval " << interval;
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// B3 - one rounding rule. Nodes and intervals used to be assigned to phases by two different comparisons. On a grid
// where every phase boundary is a multiple of dt the two agree (the old node rule even had a slack that absorbed the
// floating-point noise at the boundaries), so the fixtures here include an off-grid planning instant with a swing in
// flight, which is the normal case at runtime.
// ---------------------------------------------------------------------------------------------------------------

struct NamedInput {
  std::string name;
  ContactPlannerInput input;
};

std::vector<NamedInput> roundingFixtures(const ContactPlanningConfig& config) {
  std::vector<NamedInput> fixtures;
  ContactPlannerInput onGrid = makeStandingInput(vector2_t(0.4, 0.0));
  onGrid.committedContacts.assign(2, makeFeetArray(true));
  onGrid.committedUntil = onGrid.time + 2.0 * kDt;
  fixtures.push_back({"on-grid stance", onGrid});

  // Planned at 0.013 s with the left foot 0.113 s into its swing: the touch-down at 0.150 s is 5.48 nodes ahead, and
  // every nominal boundary after it is off the grid by the same fraction.
  ContactPlannerInput offGrid = makeInputWithLeftSwingInFlight(/*time=*/0.013, 0.013 - 0.113, 0.013 - 0.113 + kSspDuration, config);
  offGrid.velocityCommand = vector2_t(0.4, 0.0);
  offGrid.comVelocity = vector2_t(0.4, 0.1);
  fixtures.push_back({"off-grid swing in flight", offGrid});

  // Planned in a double support that began off the grid, with the touch-down inside the first node.
  ContactPlannerInput landedOffGrid = makeInputWithLeftSwingInFlight(/*time=*/0.013, 0.013 - 0.244, 0.013 + 0.006, config);
  landedOffGrid.velocityCommand = vector2_t(0.4, 0.0);
  landedOffGrid.comVelocity = vector2_t(0.4, 0.1);
  fixtures.push_back({"off-grid touch-down inside node 0", landedOffGrid});
  return fixtures;
}

TEST(HlipPlanConsistency, thePublishedContactsAgreeWithTheGaitTheRollOutUsed) {
  const ContactPlanningConfig config = makeConfig();
  HlipContactPlanner planner(config);
  for (const NamedInput& fixture : roundingFixtures(config)) {
    const ContactPlannerInput& input = fixture.input;
    ASSERT_TRUE(planner.isWalking(input)) << fixture.name;
    const ContactPlan plan = planner.plan(input);
    const std::vector<GaitPhase> gait = planner.buildGait(input, /*walking=*/true);

    // Every published interval must carry the contact state of the gait the footholds were rolled out against: the
    // committed ones exactly as committed, every later one as the gait has it at the interval's midpoint, which is
    // where the nearest-node rounding puts the boundary.
    ASSERT_EQ(plan.contacts.size(), static_cast<size_t>(plan.numIntervals()));
    const int numCommitted = static_cast<int>(input.committedContacts.size());
    for (int interval = 0; interval < plan.numIntervals(); ++interval) {
      if (interval < numCommitted) {
        EXPECT_EQ(plan.contacts[interval], input.committedContacts[static_cast<size_t>(interval)])
            << fixture.name << ", interval " << interval;
        continue;
      }
      const scalar_t intervalStart = plan.startTime + kDt * static_cast<scalar_t>(interval);
      EXPECT_EQ(plan.contacts[interval], gaitContactsAt(gait, intervalStart + 0.5 * kDt)) << fixture.name << ", interval " << interval;
    }
  }
}

TEST(HlipPlanConsistency, theFootholdChangesOnTheSameNodeTheContactDoes) {
  const ContactPlanningConfig config = makeConfig();
  HlipContactPlanner planner(config);
  for (const NamedInput& fixture : roundingFixtures(config)) {
    const ContactPlan plan = planner.plan(fixture.input);

    // A foot's planned landing spot is written for every node of the swing that places it. So the node at which a
    // foot's foothold starts moving must be the node at which its contact flag says it left the ground - never one
    // node out - and a foot's foothold may not move at any other node.
    int liftOffs = 0;
    for (size_t foot = 0; foot < kNumContacts; ++foot) {
      for (int interval = 1; interval < plan.numIntervals(); ++interval) {
        const bool liftsOff = plan.contacts[interval - 1][foot] && !plan.contacts[interval][foot];
        const vector2_t before = plan.footholds[static_cast<size_t>(interval - 1)][foot];
        const vector2_t after = plan.footholds[static_cast<size_t>(interval)][foot];
        if (liftsOff) {
          ++liftOffs;
          EXPECT_GT((after - before).norm(), 1.0e-9)
              << fixture.name << ": foot " << foot << " lifts off at interval " << interval << " but its foothold had not moved by then";
        } else {
          EXPECT_LT((after - before).norm(), 1.0e-12)
              << fixture.name << ": foot " << foot << " moved at interval " << interval << " without lifting off there";
        }
      }
    }
    EXPECT_GE(liftOffs, 4) << fixture.name << ": the fixture must contain lift-offs for this to test anything";
  }
}

// ---------------------------------------------------------------------------------------------------------------
// B1 - a stepping plan publishes the center-of-mass trajectory its own footholds were placed for. The standing blend
// towards the support center applies only while standing.
// ---------------------------------------------------------------------------------------------------------------

TEST(HlipPlanConsistency, aSteppingPlanPublishesTheUnblendedRollOut) {
  const ContactPlanningConfig config = makeConfig();
  HlipContactPlanner planner(config);
  // A command just above the blend's half point: alpha is close to 0.5, which is exactly where the blend used to halve
  // the lateral sway the deadbeat step had been computed against.
  ContactPlannerInput input = makeStandingInput(vector2_t(0.105, 0.0));
  // The center of mass must be AWAY from the center of the support, or the blend towards that center is the identity
  // and the test proves nothing: makeStandingInput puts both at the origin.
  input.comPosition = vector2_t(0.06, 0.04);
  const vector2_t supportCenter = 0.5 * (input.footPositions[kContactLeftIndex] + input.footPositions[kContactRightIndex]);
  ASSERT_GT((input.comPosition - supportCenter).norm(), 0.05) << "the blend would be a no-op at the support center";
  const scalar_t blendWeight = planner.blendWeight(input);
  ASSERT_GE(blendWeight, 0.5) << "this test needs a walking plan";
  ASSERT_LT(blendWeight, 0.95) << "this test needs alpha well below 1, where the blend used to bite";

  const ContactPlan plan = planner.plan(input);

  // Every node is the H-LIP rolled out from the measured state through the plan's own gait and footholds, computed
  // independently of the planner. This is the assertion that sees a blend at ANY node: blending every node, or every
  // node but the first, moves the published center of mass off the pendulum's own trajectory. (The gap and sway checks
  // this test used to rely on could not: blending the plan's own output again leaves a gap of (1 - a) a |rollout -
  // center|, well above their thresholds, so they passed on the defect.)
  const std::vector<GaitPhase> gait = planner.buildGait(input, /*walking=*/true);
  const std::vector<std::pair<vector2_t, vector2_t>> expected = independentRollOut(planner.getModel(), input, gait, plan);
  ASSERT_EQ(expected.size(), plan.comPosition.size());
  for (size_t node = 0; node < expected.size(); ++node) {
    EXPECT_NEAR((plan.comPosition[node] - expected[node].first).norm(), 0.0, 1.0e-9) << "node " << node;
    EXPECT_NEAR((plan.comVelocity[node] - expected[node].second).norm(), 0.0, 1.0e-9) << "node " << node;
  }
  // The comparison has power: the blended reference would differ from the roll-out by centimeters.
  scalar_t maxBlendGap = 0.0;
  for (size_t node = 0; node < plan.comPosition.size(); ++node) {
    maxBlendGap = std::max(maxBlendGap, (1.0 - blendWeight) * (expected[node].first - supportCenter).norm());
  }
  EXPECT_GT(maxBlendGap, 0.01);
}

TEST(HlipPlanConsistency, aStandingPlanStillAsksForACenterOfMassAtRestOverTheFeet) {
  HlipContactPlanner planner(makeConfig());
  ContactPlannerInput input = makeStandingInput(vector2_t::Zero());
  input.comPosition = vector2_t(0.03, 0.01);
  input.comVelocity = vector2_t(0.02, -0.01);  // a small drift, well below the blend's half point

  const ContactPlan plan = planner.plan(input);
  ASSERT_FALSE(planner.isWalking(input));

  const scalar_t blendWeight = planner.blendWeight(input);
  const vector2_t expectedVelocity = blendWeight * input.comVelocity;
  EXPECT_NEAR(plan.comVelocity.back().x(), expectedVelocity.x(), 1.0e-9) << "standing must ask for a stop, not a drift";
  EXPECT_NEAR(plan.comVelocity.back().y(), expectedVelocity.y(), 1.0e-9);
  const vector2_t supportCenter = 0.5 * (input.footPositions[kContactLeftIndex] + input.footPositions[kContactRightIndex]);
  EXPECT_LT((plan.comPosition.back() - supportCenter).norm(), 0.02) << "standing must ask for the center of the support";
}

TEST(HlipPlanConsistency, aStandingPlanMadeMidSwingCentersOnTheSupportTheRobotWillStandOn) {
  // The stick is released while the left foot is in the air. The standing plan lands that swing (it is committed) and
  // then stands, so the support it asks the center of mass to come to rest over is the one with the left foot at its
  // LANDING spot. Centering on the measured feet instead averaged in a foot still in mid-air, 5 cm behind where it lands.
  const ContactPlanningConfig config = makeConfig();
  HlipContactPlanner planner(config);
  const HlipModel& model = planner.getModel();
  const scalar_t touchDown = 0.137;
  ContactPlannerInput input = makeInputWithLeftSwingInFlight(/*time=*/0.0, touchDown - kSspDuration, touchDown, config);
  const std::pair<HlipModel::State, HlipModel::State> atLiftOff = orbitAtLeftLiftOff(model, /*velocity=*/0.0);
  const vector2_t stance = input.footPositions[kContactRightIndex];
  const HlipModel::State y = model.flowSingleSupport(atLiftOff.second, kSspDuration - touchDown);
  input.comPosition = stance + vector2_t(0.01, y(0));
  input.comVelocity = vector2_t(0.03, y(1));  // stepping in place, with a little forward drift left over
  const scalar_t alpha = planner.blendWeight(input);
  ASSERT_LT(alpha, 0.5) << "this test needs a standing plan";
  ASSERT_GT(alpha, 0.05) << "and one that blends, or the roll-out drops out of the reference";

  const ContactPlan plan = planner.plan(input);
  ASSERT_TRUE(plan.valid);
  const feet_array_t<vector2_t>& standingFeet = plan.footholds.back();
  const vector2_t standingCenter = 0.5 * (standingFeet[kContactLeftIndex] + standingFeet[kContactRightIndex]);
  const vector2_t measuredCenter = 0.5 * (input.footPositions[kContactLeftIndex] + input.footPositions[kContactRightIndex]);
  ASSERT_GT((standingCenter - measuredCenter).norm(), 0.01) << "the fixture must tell the two centers apart";

  // Every node is alpha of the pendulum's own roll-out and (1 - alpha) of the center of the support it stands on.
  const std::vector<GaitPhase> gait = planner.buildGait(input, /*walking=*/false);
  const std::vector<std::pair<vector2_t, vector2_t>> rollOut = independentRollOut(model, input, gait, plan);
  ASSERT_EQ(rollOut.size(), plan.comPosition.size());
  for (size_t node = 0; node < rollOut.size(); ++node) {
    const vector2_t expected = alpha * rollOut[node].first + (1.0 - alpha) * standingCenter;
    EXPECT_NEAR((plan.comPosition[node] - expected).norm(), 0.0, 1.0e-9) << "node " << node;
    EXPECT_NEAR((plan.comVelocity[node] - alpha * rollOut[node].second).norm(), 0.0, 1.0e-9) << "node " << node;
  }
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
  EXPECT_NEAR(plan.comPosition.front().x(), input.comPosition.x(), 1.0e-12);
  EXPECT_NEAR(plan.comPosition.front().y(), input.comPosition.y(), 1.0e-12);
  EXPECT_NEAR(plan.comVelocity.front().x(), input.comVelocity.x(), 1.0e-12);
  EXPECT_NEAR(plan.comVelocity.front().y(), input.comVelocity.y(), 1.0e-12);
}

// ---------------------------------------------------------------------------------------------------------------
// A49 - the blend measures the lateral velocity against the sway of stepping in place.
// ---------------------------------------------------------------------------------------------------------------

TEST(HlipPlanConsistency, steppingInPlaceOnTheZeroCommandOrbitReadsAsRestAtEveryPhase) {
  // At a zero command the deadbeat law settles on the period-two orbit, which sways at 0.165 m/s at every touch-down.
  // Measured raw, that held alpha at one; relative to the orbit it is no motion at all, anywhere in the stride.
  const ContactPlanningConfig config = makeConfig();
  HlipContactPlanner planner(config);
  const HlipModel& model = planner.getModel();
  const std::pair<HlipModel::State, HlipModel::State> atLiftOff = orbitAtLeftLiftOff(model, /*velocity=*/0.0);
  const std::pair<HlipModel::State, HlipModel::State> orbit = model.periodTwoOrbit(kStepWidth, -kStepWidth);

  // Positive control: the raw sway is far above the half point on its own.
  const scalar_t swayAtTouchDown = orbit.first(1);
  ASSERT_GT(std::abs(swayAtTouchDown), 0.1);
  ASSERT_TRUE(planner.getBlend().isWalking(vector2_t::Zero(), /*yawRateCommand=*/0.0, vector2_t(0.0, swayAtTouchDown)));

  // Single support on the right foot, the left in flight, through the whole swing.
  for (int sample = 0; sample <= 10; ++sample) {
    const scalar_t inFlight = kSspDuration * static_cast<scalar_t>(sample) / 10.0;
    ContactPlannerInput input = makeStandingInput(vector2_t::Zero());
    input.contacts[kContactLeftIndex] = false;
    input.phaseElapsedTime = makeFeetArray(inFlight);
    input.lastSwungFoot = static_cast<int>(kContactLeftIndex);
    const HlipModel::State y = model.flowSingleSupport(atLiftOff.second, inFlight);
    input.comVelocity = vector2_t(0.0, y(1));
    EXPECT_NEAR(planner.blendVelocity(input).y(), 0.0, 1.0e-12) << inFlight << " s into the swing";
    EXPECT_FALSE(planner.isWalking(input)) << inFlight << " s into the swing";
  }
  // The double support after the left touch-down, at the orbit's drift and slowing down from it to rest.
  for (const scalar_t fraction : {1.0, 0.7, 0.3, 0.0}) {
    ContactPlannerInput input = makeStandingInput(vector2_t::Zero());
    input.phaseElapsedTime = makeFeetArray(0.02);
    input.lastSwungFoot = static_cast<int>(kContactLeftIndex);
    input.comVelocity = vector2_t(0.0, fraction * orbit.first(1));
    EXPECT_NEAR(planner.blendVelocity(input).y(), 0.0, 1.0e-12) << "at " << fraction << " of the orbit's drift";
    EXPECT_FALSE(planner.isWalking(input)) << "at " << fraction << " of the orbit's drift";
  }
}

TEST(HlipPlanConsistency, aLateralPushBeyondTheOrbitsSwayStillStartsTheGait) {
  // The blend must stay sensitive to real motion: what lies beyond the orbit's own sway, or against it, counts in full,
  // and a robot that has not stepped yet has no orbit to measure against.
  const ContactPlanningConfig config = makeConfig();
  HlipContactPlanner planner(config);
  const std::pair<HlipModel::State, HlipModel::State> orbit = planner.getModel().periodTwoOrbit(kStepWidth, -kStepWidth);
  const scalar_t drift = orbit.first(1);  // towards the left foot that has just landed

  ContactPlannerInput afterLeft = makeStandingInput(vector2_t::Zero());
  afterLeft.phaseElapsedTime = makeFeetArray(0.02);
  afterLeft.lastSwungFoot = static_cast<int>(kContactLeftIndex);
  afterLeft.comVelocity = vector2_t(0.0, drift + 0.1);
  EXPECT_NEAR(planner.blendVelocity(afterLeft).y(), 0.1, 1.0e-12) << "beyond the sway";
  EXPECT_TRUE(planner.isWalking(afterLeft));
  afterLeft.comVelocity = vector2_t(0.0, -0.1);
  EXPECT_NEAR(planner.blendVelocity(afterLeft).y(), -0.1, 1.0e-12) << "against the sway";
  EXPECT_TRUE(planner.isWalking(afterLeft));

  ContactPlannerInput neverStepped = makeStandingInput(vector2_t::Zero());
  neverStepped.comVelocity = vector2_t(0.03, 0.1);
  EXPECT_FALSE(planner.zeroCommandOrbitLateralVelocity(neverStepped).has_value());
  EXPECT_EQ(planner.blendVelocity(neverStepped), neverStepped.comVelocity) << "with no step yet the velocity counts as measured";
  EXPECT_TRUE(planner.isWalking(neverStepped));
}

}  // namespace
}  // namespace ocs2::humanoid
