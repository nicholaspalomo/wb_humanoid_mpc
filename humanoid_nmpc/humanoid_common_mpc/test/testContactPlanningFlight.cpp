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

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "humanoid_common_mpc/contact_planning/ContactPlanningTermFactory.h"
#include "humanoid_common_mpc/contact_planning/LipContactPlanner.h"
#include "humanoid_common_mpc/contact_planning/cost/StepLengthCost.h"
#include "humanoid_common_mpc/contact_planning/logic/AlternatingFeetRule.h"
#include "humanoid_common_mpc/contact_planning/logic/FlightDurationsRule.h"
#include "humanoid_common_mpc/contact_planning/model/LipBlockIndices.h"
#include "humanoid_common_mpc/contact_planning/model/VerticalDoubleIntegrator.h"

namespace ocs2::humanoid {

namespace {

constexpr scalar_t kGravity = 9.81;

/** Walking configuration of the planner tests. */
ContactPlanningConfig walkingConfig() {
  ContactPlanningConfig config;
  config.planner.dt = 0.1;
  config.planner.numNodes = 12;
  config.planner.commitTime = 0.0;
  config.planner.maxBranchAndBoundNodes = 3000;
  config.planner.maxSolveTime = 10.0;
  config.shared.comHeight = 0.85;
  config.shared.gaitLimits.minSwingDuration = 0.3;
  config.shared.gaitLimits.maxSwingDuration = 0.5;
  config.shared.gaitLimits.minContactDuration = 0.15;
  config.shared.gaitLimits.minDoubleSupportDuration = 0.1;
  config.eventShiftLocalSearch.maxTime = 2.0;
  config.validate();
  return config;
}

/** The flight model on top of it: a 0.1 to 0.2 s flight, two feet that can push 2.5 g each, hops of 0.2 s. */
ContactPlanningConfig flightConfig() {
  ContactPlanningConfig config = walkingConfig();
  config.setFlightModel(true);
  config.shared.gaitLimits.minFlightDuration = 0.1;
  config.shared.gaitLimits.maxFlightDuration = 0.2;
  config.shared.gaitLimits.minDoubleSupportDuration = 0.0;
  config.verticalDoubleIntegrator.maxContactAcceleration = 2.5 * kGravity;
  config.hopOnRequest.triggerBaseHeight = 1.0;
  config.hopOnRequest.flightDuration = 0.2;
  config.validate();
  return config;
}

/** Running: short stances while walking, long swings (stance plus two flights), long strides. */
ContactPlanningConfig runningConfig() {
  ContactPlanningConfig config = flightConfig();
  config.shared.gaitLimits.minContactDuration = 0.2;
  config.shared.gaitLimits.maxContactDurationWalking = 0.3;
  config.shared.gaitLimits.walkingSpeedThreshold = 0.1;
  config.shared.gaitLimits.maxSwingDuration = 0.6;
  config.shared.bigM = 2.0;
  config.footSeparation.maxStepLength = 1.4;
  config.reachability.reachX = 0.7;
  config.formulation.setListed(config.formulation.costs, term::kStepLength, true);
  config.stepLength.weight = 5.0;
  config.validate();
  return config;
}

ContactPlannerInput standingInput() {
  ContactPlannerInput input;
  input.time = 3.0;
  input.footPositions[0] = vector2_t(0.0, 0.125);
  input.footPositions[1] = vector2_t(0.0, -0.125);
  input.contacts = {true, true};
  input.phaseElapsedTime = {5.0, 5.0};
  input.yawInertia = 10.0;
  input.comHeight = 0.85;
  input.comHeightRate = 0.0;
  return input;
}

struct Phases {
  int flightNodes = 0;
  int doubleSupportNodes = 0;
  int longestFlight = 0;
  bool alternates = true;  // consecutive single supports are on different feet
};

Phases analyse(const ContactPlan& plan) {
  Phases p;
  int run = 0;
  int lastSingle = -1;
  for (const contact_flag_t& c : plan.contacts) {
    const int sum = (c[0] ? 1 : 0) + (c[1] ? 1 : 0);
    if (sum == 0) {
      ++p.flightNodes;
      p.longestFlight = std::max(p.longestFlight, ++run);
    } else {
      run = 0;
    }
    if (sum == 2) ++p.doubleSupportNodes;
    if (sum == 1) {
      const int foot = c[0] ? 0 : 1;
      if (lastSingle >= 0 && lastSingle == foot && p.doubleSupportNodes == 0) {
        // the same foot again with no double support in between is only fine if nothing happened in between
      }
      lastSingle = foot;
    }
  }
  return p;
}

/** The shipped DRC Atlas planner: the heading model, its weights and limits, plus the flight model. */
ContactPlanningConfig atlasLikeConfig() {
  ContactPlanningConfig config = walkingConfig();
  config.setHeadingModel(true);
  config.yawTorqueBudget.torsionalFrictionTorque = 20.0;  // from the robot model in the real interface
  config.yawTorqueBudget.doubleSupportYawCouple = 40.0;
  config.hipYawRange.lower = {-0.4, -0.6};
  config.hipYawRange.upper = {0.6, 0.4};
  config.shared.bigM = 1.5;
  config.shared.gaitLimits.minDoubleSupportDuration = 0.05;
  config.velocityTracking.weight = 50.0;
  config.stepWidth.weight = 10.0;
  config.zmpRegularization.weight = 0.1;
  config.footholdRegularization.weight = 0.2;
  config.terminalDcm.weight = 50.0;
  config.terminalDcm.trackCommandedVelocity = true;
  config.heightTracking.weight = 400.0;  // the crouch of a hop is a counter-movement, not a squat
  // The height of a foot in contact is the one thing tying the vertical model to the ground; at the shared penalty the
  // planner bought a deep squat with it, which no leg would follow.
  config.contactHeight.slack = SlackPenalty{1.0e5, 1.0e3};
  config.zmpSupportRegion.halfWidthX = 0.08;
  config.zmpSupportRegion.halfWidthY = 0.04;
  config.reachability.reachX = 0.6;
  config.reachability.reachYInner = 0.05;
  config.reachability.reachYOuter = 0.45;
  config.contactSwitch.cost = 0.3;
  config.planConsistency.cost = 0.5;
  config.formulation.setListed(config.formulation.costs, term::kStepLength, true);
  config.stepLength.weight = 20.0;
  // Running: the flight model, a swing that spans the other foot's stance and two flights, and a running stride.
  config.setFlightModel(true);
  config.shared.gaitLimits.minFlightDuration = 0.1;
  config.shared.gaitLimits.maxFlightDuration = 0.2;
  config.shared.gaitLimits.maxSwingDuration = 0.6;
  config.footSeparation.maxStepLength = 1.2;
  config.flightDurations.allowedAboveSpeed = 1.3;
  config.verticalDoubleIntegrator.maxContactAcceleration = 2.5 * kGravity;
  config.validate();
  return config;
}

/** Walking on the left foot at `speed`, the right foot swinging behind, the same speed commanded. */
ContactPlannerInput movingInput(scalar_t speed) {
  ContactPlannerInput input = standingInput();
  input.contacts = {true, false};
  input.phaseElapsedTime = {0.1, 0.2};
  input.comPosition = vector2_t(0.05, 0.0);
  input.comVelocity = vector2_t(speed, 0.0);
  input.velocityCommand = vector2_t(speed, 0.0);
  input.footPositions[0] = vector2_t(0.0, 0.125);
  input.footPositions[1] = vector2_t(-0.25, -0.125);
  input.lastSwungFoot = 1;
  return input;
}

}  // namespace

/*============================================ model and rows ============================================*/

TEST(ContactPlanningFlight, TheVerticalBlockAppendsItsVariablesAndDynamics) {
  const ContactPlanningConfig config = flightConfig();
  LipContactPlanner planner(config);
  const Layout& layout = planner.getLayout();
  ASSERT_TRUE(layout.hasHeight);
  EXPECT_EQ(layout.nx, LIP_STATE_DIM + 2);
  EXPECT_EQ(layout.nu, LIP_INPUT_DIM + 1);
  EXPECT_EQ(layout.state("z"), LIP_STATE_DIM);
  EXPECT_EQ(layout.state("vz"), LIP_STATE_DIM + 1);
  EXPECT_EQ(layout.input("az"), LIP_INPUT_DIM);
  EXPECT_NE(layout.describe().find("z vz"), std::string::npos) << layout.describe();

  const OcpQpProblem qp = planner.buildProblem(standingInput());
  const OcpQpStage& s = qp.stages.front();
  const int z = layout.height, vz = layout.heightRate, az = layout.heightAccel;
  EXPECT_DOUBLE_EQ(s.A(z, z), 1.0);
  EXPECT_DOUBLE_EQ(s.A(z, vz), 0.1);
  EXPECT_DOUBLE_EQ(s.A(vz, vz), 1.0);
  EXPECT_DOUBLE_EQ(s.B(z, az), 0.5 * 0.01);
  EXPECT_DOUBLE_EQ(s.B(vz, az), 0.1);
  // The ground cannot pull; two feet push at most 2 a_max - g. The box bounds are stored on the bounded subset.
  const auto bound = std::find(s.idxbu.begin(), s.idxbu.end(), az);
  ASSERT_NE(bound, s.idxbu.end());
  const Eigen::Index slot = bound - s.idxbu.begin();
  EXPECT_DOUBLE_EQ(s.lbu(slot), -kGravity);
  EXPECT_DOUBLE_EQ(s.ubu(slot), 2.0 * 2.5 * kGravity - kGravity);
  // The initial state carries the measured height.
  EXPECT_DOUBLE_EQ(qp.x0(z), 0.85);
  EXPECT_DOUBLE_EQ(qp.x0(vz), 0.0);
  // Configuration: a foot must at least carry the robot.
  ContactPlanningConfig weak = config;
  weak.verticalDoubleIntegrator.maxContactAcceleration = 0.5 * kGravity;
  EXPECT_THROW(weak.validate(), std::invalid_argument);
}

TEST(ContactPlanningFlight, FlightRowsHangOnTheContactSum) {
  ContactPlanningConfig config = flightConfig();
  config.formulation.costs = {term::kRegularization};
  config.formulation.softConstraints = {term::kContactHeight};
  config.formulation.hardConstraints = {term::kVerticalThrustLimit, term::kZmpPinnedInFlight};
  config.validate();
  LipContactPlanner planner(config);
  const Layout& layout = planner.getLayout();
  const OcpQpProblem qp = planner.buildProblem(standingInput());
  const OcpQpStage& s = qp.stages.front();
  // 1 thrust row + 4 ZMP rows (two axes, two signs) + 4 contact-height rows (two feet, two signs).
  ASSERT_EQ(s.numGeneralConstraints(), 9);
  const scalar_t aMax = 2.5 * kGravity;
  // az - a_max c_L - a_max c_R <= -g
  EXPECT_DOUBLE_EQ(s.D(0, layout.heightAccel), 1.0);
  EXPECT_DOUBLE_EQ(s.D(0, LIP_CL), -aMax);
  EXPECT_DOUBLE_EQ(s.D(0, LIP_CR), -aMax);
  EXPECT_DOUBLE_EQ(s.ug(0), -kGravity);
  // (zmp_x - c_x) - M c_L - M c_R <= 0
  EXPECT_DOUBLE_EQ(s.D(1, LIP_ZX), 1.0);
  EXPECT_DOUBLE_EQ(s.C(1, LIP_CX), -1.0);
  EXPECT_DOUBLE_EQ(s.D(1, LIP_CL), -config.shared.bigM);
  EXPECT_DOUBLE_EQ(s.ug(1), 0.0);
  EXPECT_DOUBLE_EQ(s.D(2, LIP_ZX), -1.0);
  EXPECT_DOUBLE_EQ(s.C(2, LIP_CX), 1.0);
  // z + M c_L <= tol + M + z_nom (soft)
  EXPECT_DOUBLE_EQ(s.C(5, layout.height), 1.0);
  EXPECT_DOUBLE_EQ(s.D(5, LIP_CL), 1.0);
  EXPECT_DOUBLE_EQ(s.ug(5), 0.05 + 1.0 + 0.85);
  EXPECT_EQ(s.softGeneralIndices.size(), 4u);
  // A configuration that forbids flight cannot carry the vertical block.
  ContactPlanningConfig noFlight = config;
  noFlight.shared.gaitLimits.maxFlightDuration = 0.0;
  EXPECT_THROW(noFlight.validate(), std::invalid_argument);
  ContactPlanningConfig contradictory = config;
  contradictory.formulation.logicRules.push_back(term::kNoFlight);
  EXPECT_THROW(contradictory.validate(), std::invalid_argument);
  ContactPlanningConfig orphan = walkingConfig();
  orphan.formulation.costs.push_back(term::kHeightTracking);
  EXPECT_THROW(orphan.validate(), std::invalid_argument);
}

/*============================================ logic ============================================*/

TEST(ContactPlanningFlight, FlightDurationsRuleBoundsAFlightAndForbidsItWhenOff) {
  const ContactPlanningConfig config = flightConfig();  // 1 to 2 flight nodes
  LipContactPlanner planner(config);
  const ContactPlannerInput input = standingInput();
  const auto flight = [&](std::initializer_list<int> nodes) {
    MiqpAssignment a = planner.initialAssignment(input);
    for (const int k : nodes) {
      a[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(k, 0))] = 0;
      a[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(k, 1))] = 0;
    }
    return a;
  };
  // Three flight nodes exceed the maximum of two.
  MiqpAssignment tooLong = flight({3, 4, 5});
  for (int k = 0; k < 3; ++k) {
    tooLong[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(k, 0))] = 1;
    tooLong[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(k, 1))] = 1;
  }
  EXPECT_FALSE(planner.propagate(input, tooLong));
  // Two flight nodes followed by a landing are fine.
  MiqpAssignment hop = flight({3, 4});
  for (int k = 0; k < 3; ++k) {
    hop[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(k, 0))] = 1;
    hop[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(k, 1))] = 1;
  }
  hop[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(5, 0))] = 1;
  hop[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(5, 1))] = 1;
  EXPECT_TRUE(planner.propagate(input, hop));
  // After the maximum the propagation lands the robot: a free foot at node 5 with the other one still in the air.
  MiqpAssignment landing = flight({3, 4});
  for (int k = 0; k < 3; ++k) {
    landing[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(k, 0))] = 1;
    landing[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(k, 1))] = 1;
  }
  landing[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(5, 0))] = 0;
  EXPECT_TRUE(planner.propagate(input, landing));
  EXPECT_EQ(landing[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(5, 1))], 1);
  // A flight cannot start on the last node: it could not last its minimum, and it must land inside the horizon.
  ContactPlanningConfig twoNodes = config;
  twoNodes.shared.gaitLimits.minFlightDuration = 0.2;
  twoNodes.validate();
  LipContactPlanner strict(twoNodes);
  MiqpAssignment late = strict.initialAssignment(input);
  for (int k = 0; k < 10; ++k) {
    late[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(k, 0))] = 1;
    late[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(k, 1))] = 1;
  }
  late[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(10, 0))] = 0;
  late[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(10, 1))] = 0;
  EXPECT_FALSE(strict.propagate(input, late)) << "a two-node flight from node 10 would end on the horizon";
  // On the last interval no flight at all, whatever the limits (the rule alone: the phase-duration rule would already
  // keep a foot from lifting that late).
  {
    FlightDurationsRule rule;
    rule.configure(config);
    const ContactLogicState state = planner.makeLogicState(input);
    MiqpAssignment last = planner.initialAssignment(input);
    for (int k = 0; k < 11; ++k) {
      last[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(k, 0))] = 1;
      last[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(k, 1))] = 1;
    }
    last[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(11, 0))] = 0;
    bool changed = false;
    EXPECT_TRUE(rule.propagate(state, ContactLogicScan::compute(state, last), last, changed));
    EXPECT_EQ(last[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(11, 1))], 1);
    last[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(11, 1))] = 0;
    EXPECT_FALSE(rule.propagate(state, ContactLogicScan::compute(state, last), last, changed));
  }
  // The minimum is propagated: one flight node fixed, the next one follows.
  MiqpAssignment minimum = strict.initialAssignment(input);
  for (int k = 0; k < 3; ++k) {
    minimum[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(k, 0))] = 1;
    minimum[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(k, 1))] = 1;
  }
  minimum[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(3, 0))] = 0;
  minimum[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(3, 1))] = 0;
  EXPECT_TRUE(strict.propagate(input, minimum));
  EXPECT_EQ(minimum[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(4, 0))], 0);
  EXPECT_EQ(minimum[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(4, 1))], 0);

  // With maxFlightDuration at 0 the rule is no_flight.
  ContactPlanningConfig off = walkingConfig();
  FlightDurationsRule rule;
  rule.configure(off);
  EXPECT_NE(rule.describe().find("no flight"), std::string::npos);
  LipContactPlanner walker(off);
  ContactLogicState state = walker.makeLogicState(input);
  EXPECT_EQ(state.nFlightMax, 0);
  MiqpAssignment grounded = walker.initialAssignment(input);
  grounded[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(2, 0))] = 0;
  grounded[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(2, 1))] = 0;
  bool changed = false;
  EXPECT_FALSE(rule.propagate(state, ContactLogicScan::compute(state, grounded), grounded, changed));
}

TEST(ContactPlanningFlight, APureFlightIsNotHeldToTheSwingMinimum) {
  const ContactPlanningConfig config = flightConfig();  // swing minimum 0.3 s = 3 nodes, flight 1 to 2 nodes
  LipContactPlanner planner(config);
  const ContactPlannerInput input = standingInput();
  const auto set = [&](MiqpAssignment& a, int k, int l, int r) {
    a[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(k, 0))] = static_cast<std::int8_t>(l);
    a[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(k, 1))] = static_cast<std::int8_t>(r);
  };
  // Both feet in the air for two nodes, then both down: a hop, shorter than a swing, allowed.
  MiqpAssignment hop = planner.initialAssignment(input);
  for (int k = 0; k < 3; ++k) set(hop, k, 1, 1);
  set(hop, 3, 0, 0);
  set(hop, 4, 0, 0);
  for (int k = 5; k < 12; ++k) set(hop, k, 1, 1);
  EXPECT_TRUE(planner.propagate(input, hop));
  // One foot in the air for two nodes while the other stands: a swing, held to its minimum.
  MiqpAssignment shortSwing = planner.initialAssignment(input);
  for (int k = 0; k < 3; ++k) set(shortSwing, k, 1, 1);
  set(shortSwing, 3, 0, 1);
  set(shortSwing, 4, 0, 1);
  for (int k = 5; k < 12; ++k) set(shortSwing, k, 1, 1);
  EXPECT_FALSE(planner.propagate(input, shortSwing));
  // Running: a foot's air phase is its swing over the other foot's stance plus a flight at each end; it is a swing and
  // is held to the swing minimum against the whole air phase. Left stance, right air (3 nodes), flight, right stance,
  // left air (4 nodes), flight, left stance, right air (3 nodes), double support to the end.
  MiqpAssignment running = planner.initialAssignment(input);
  set(running, 0, 1, 1);
  set(running, 1, 1, 0);  // right lifts
  set(running, 2, 1, 0);
  set(running, 3, 0, 0);  // flight
  set(running, 4, 0, 1);  // right lands, left in the air
  set(running, 5, 0, 1);
  set(running, 6, 0, 0);  // flight
  set(running, 7, 1, 0);  // left lands
  set(running, 8, 1, 0);
  set(running, 9, 1, 1);
  set(running, 10, 1, 1);
  set(running, 11, 1, 1);
  EXPECT_TRUE(planner.propagate(input, running));
  // The same with the right foot landing after only two air nodes: a swing too short.
  MiqpAssignment shortRunning = running;
  set(shortRunning, 1, 1, 1);
  EXPECT_FALSE(planner.propagate(input, shortRunning));
}

// Alternation across a flight. With both feet in the air neither of them is the one that swung most recently: the
// reference manager reads that off the executed schedule and the rule must keep its answer. Reading it off the contact
// flags instead named the same foot after every flight, and the other foot's next swing was rejected as a repeat.
TEST(ContactPlanningFlight, AlternationKeepsTheLastSwungFootThroughAFlight) {
  const ContactPlanningConfig config = flightConfig();
  LipContactPlanner planner(config);
  ContactPlannerInput input = standingInput();
  input.contacts = {false, false};      // airborne at the planning instant
  input.phaseElapsedTime = {0.1, 0.4};  // the left pushed off into this flight, the right has been swinging
  input.lastSwungFoot = 0;
  const ContactLogicState state = planner.makeLogicState(input);
  AlternatingFeetRule rule;
  rule.configure(config);

  const auto swingAfterTheFlight = [&](size_t foot) {
    MiqpAssignment a = planner.initialAssignment(input);
    const auto set = [&](int k, int left, int right) {
      a[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(k, 0))] = static_cast<std::int8_t>(left);
      a[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(k, 1))] = static_cast<std::int8_t>(right);
    };
    set(0, 0, 0);  // the flight ends here
    set(1, 0, 1);  // the right foot lands, the left is still in the air
    set(2, 1, 1);  // the left lands: both down
    set(3, 1, 1);
    for (int k = 4; k < config.planner.numNodes; ++k) set(k, foot == 0 ? 0 : 1, foot == 0 ? 1 : 0);
    bool changed = false;
    return rule.propagate(state, ContactLogicScan::compute(state, a), a, changed);
  };
  EXPECT_TRUE(swingAfterTheFlight(1)) << "the right foot may swing after the left pushed off into the flight";
  EXPECT_FALSE(swingAfterTheFlight(0)) << "the left foot may not swing twice in a row";

  // With one foot in the air the flags do say which foot is swinging, and they still win over a stale input.
  ContactPlannerInput singleSupport = standingInput();
  singleSupport.contacts = {false, true};
  singleSupport.lastSwungFoot = 1;
  const ContactLogicState swinging = planner.makeLogicState(singleSupport);
  MiqpAssignment a = planner.initialAssignment(singleSupport);
  for (int k = 0; k < config.planner.numNodes; ++k) {
    a[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(k, 0))] = k < 2 ? 0 : 1;
    a[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(k, 1))] = 1;
  }
  a[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(5, 0))] = 0;  // the left swings again, right after landing
  bool changed = false;
  EXPECT_FALSE(rule.propagate(swinging, ContactLogicScan::compute(swinging, a), a, changed));
}

// The nominal cadence of the step-length cost counts the flight: a running stride is two stances and two flights, while
// a foot's own air time already contains both flights, so the stride-to-swing ratio is smaller than the walking one.
// With the walking ratio the cost would ask a running gait for steps longer than its cadence produces.
TEST(ContactPlanningFlight, StepLengthUsesTheRunningCadenceWhenFlightIsAllowed) {
  StepLengthCost walking;
  ContactPlanningConfig walkingConfiguration = walkingConfig();
  walkingConfiguration.stepLength.weight = 1.0;
  walking.configure(walkingConfiguration);
  // 2 (T_swing + T_ds) / T_swing with 0.3 s swings and 0.1 s double supports.
  EXPECT_NEAR(walking.strideToSwingRatio(), 2.0 * (0.3 + 0.1) / 0.3, 1e-12);

  StepLengthCost running;
  ContactPlanningConfig runningConfiguration = flightConfig();  // 0.3 s swings, no double support, 0.1 s flights
  runningConfiguration.stepLength.weight = 1.0;
  running.configure(runningConfiguration);
  // 2 (T_swing - T_f) / T_swing: the stance is shorter than the swing by the two flights the swing spans.
  EXPECT_NEAR(running.strideToSwingRatio(), 2.0 * (0.3 - 0.1) / 0.3, 1e-12);
  EXPECT_LT(running.strideToSwingRatio(), walking.strideToSwingRatio());
  EXPECT_NEAR(running.nominalDisplacementPerNode(vector2_t(3.0, 0.0), 0.1).x(), 3.0 * 0.1 * 2.0 * 0.2 / 0.3, 1e-12);

  // A flight that eats the whole swing is not a cadence: the nominal advance degenerates to zero rather than negative.
  ContactPlanningConfig degenerate = runningConfiguration;
  degenerate.shared.gaitLimits.minFlightDuration = 0.5;
  degenerate.shared.gaitLimits.maxFlightDuration = 0.5;
  StepLengthCost none;
  none.configure(degenerate);
  EXPECT_DOUBLE_EQ(none.strideToSwingRatio(), 0.0);
}

/*============================================ hopping ============================================*/

TEST(ContactPlanningFlight, AHopRequestLiftsBothFeetAsSoonAsAllowed) {
  const ContactPlanningConfig config = flightConfig();
  LipContactPlanner planner(config);
  ContactPlannerInput input = standingInput();
  input.hopRequested = true;
  input.hopFlightDuration = 0.2;
  const ContactLogicState state = planner.makeLogicState(input);
  EXPECT_EQ(state.nFlightMin, 2) << "the requested hop raises the minimum flight";
  EXPECT_EQ(state.nFlightMax, 2);
  MiqpAssignment a = planner.initialAssignment(input);
  ASSERT_TRUE(planner.propagate(input, a));
  // Both feet stay down for the push-off (one node at 0.1 s), then the hop: the requested flight, then the landing.
  const int pushOff = 1;
  EXPECT_EQ(a[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(0, 0))], 1) << "the push-off";
  EXPECT_EQ(a[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(0, 1))], 1);
  for (int k = pushOff; k < pushOff + 2; ++k) {
    EXPECT_EQ(a[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(k, 0))], 0) << "node " << k;
    EXPECT_EQ(a[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(k, 1))], 0) << "node " << k;
  }
  // Without a request nothing lifts.
  ContactPlannerInput quiet = standingInput();
  MiqpAssignment b = planner.initialAssignment(quiet);
  ASSERT_TRUE(planner.propagate(quiet, b));
  EXPECT_EQ(b[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(0, 0))], kMiqpFree);
}

TEST(ContactPlanningFlight, AHopPlanLeavesTheGroundAndLandsWhereItLeft) {
  const ContactPlanningConfig config = flightConfig();
  LipContactPlanner planner(config);
  ContactPlannerInput input = standingInput();
  input.hopRequested = true;
  input.hopFlightDuration = 0.2;
  const ContactPlan plan = planner.plan(input);
  std::cout << plan.describe() << std::endl;
  ASSERT_TRUE(plan.valid);
  ASSERT_TRUE(plan.hasHeight());
  const Phases phases = analyse(plan);
  EXPECT_GE(phases.flightNodes, 2) << "at least one hop of 0.2 s in the horizon";
  EXPECT_LE(phases.longestFlight, config.maxFlightNodes());
  EXPECT_NE(plan.describe().find("flight"), std::string::npos);
  // The centre of mass rises above the pendulum height during the flight and is back at it on the ground.
  const scalar_t apex = *std::max_element(plan.comHeight.begin(), plan.comHeight.end());
  EXPECT_GT(apex, config.shared.comHeight + 0.01);
  EXPECT_TRUE(plan.contacts.back()[0] || plan.contacts.back()[1]) << "the plan does not end in flight";
  EXPECT_NEAR(plan.comHeight.back(), config.shared.comHeight, 0.1) << "and the CoM is back near the ground height";
  for (size_t k = 0; k < plan.contacts.size(); ++k) {
    // contact_height is soft: the planner may pay its slack to crouch into a hop, which is what a jumper does. What has
    // to hold is that the height stays somewhere a leg can reach.
    EXPECT_GT(plan.comHeight[k], 0.5) << "node " << k;
    EXPECT_LT(plan.comHeight[k], 1.2) << "node " << k;
    if (!plan.contacts[k][0] && !plan.contacts[k][1]) {
      EXPECT_NEAR(plan.comHeightAccel[k], -kGravity, 1e-6) << "ballistic at node " << k;
      // No horizontal acceleration in flight: the ZMP sits under the CoM.
      EXPECT_NEAR((plan.zmp[k] - plan.comPosition[k]).norm(), 0.0, 1e-6) << "node " << k;
    }
  }
  // A hop in place: the feet come down close to where they left.
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    EXPECT_LT((plan.footholds.back()[foot] - input.footPositions[foot]).norm(), 0.1) << "foot " << foot;
  }
  EXPECT_LT((plan.comPosition.back() - input.comPosition).norm(), 0.15);
}

TEST(ContactPlanningFlight, ThePlanExposesItsHeightTrajectoryInTime) {
  ContactPlan plan;
  plan.valid = true;
  plan.startTime = 2.0;
  plan.dt = 0.1;
  plan.contacts = {{true, true}, {false, false}, {false, false}, {true, true}};
  plan.footholds.assign(5, {vector2_t::Zero(), vector2_t::Zero()});
  plan.comHeight = {0.85, 0.90, 0.90, 0.85, 0.85};
  plan.comHeightRate = {0.98, 0.0, -0.98, 0.0, 0.0};
  plan.comHeightAccel = {-9.81, -9.81, 9.81, 0.0};
  ASSERT_TRUE(plan.hasHeight());
  EXPECT_EQ(plan.numFlightIntervals(), 2);
  EXPECT_NEAR(*plan.heightAtTime(2.0), 0.85, 1e-12);
  EXPECT_NEAR(*plan.heightAtTime(2.05), 0.875, 1e-12) << "linear between nodes";
  EXPECT_NEAR(*plan.heightAtTime(2.1), 0.90, 1e-12);
  EXPECT_NEAR(*plan.heightAtTime(9.0), 0.85, 1e-12) << "clamped to the last node";
  EXPECT_NEAR(*plan.heightRateAtTime(2.2), -0.98, 1e-12);
  EXPECT_NE(plan.describe().find("flight 0.200s total"), std::string::npos) << plan.describe();
  EXPECT_NE(plan.describe().find("z=[0.850..0.900]"), std::string::npos) << plan.describe();
  ContactPlan walking;
  walking.valid = true;
  EXPECT_FALSE(walking.hasHeight());
  EXPECT_FALSE(walking.heightAtTime(0.0).has_value());
}

/*============================================ running ============================================*/

TEST(ContactPlanningFlight, RunningAtSpeedUsesFlightAndNoDoubleSupport) {
  const ContactPlanningConfig config = runningConfig();
  LipContactPlanner planner(config);
  // Already running at 3 m/s on the left foot, the right foot in the air behind, 3 m/s commanded.
  ContactPlannerInput input = standingInput();
  input.contacts = {true, false};
  input.phaseElapsedTime = {0.1, 0.3};
  input.comPosition = vector2_t(0.05, 0.0);
  input.comVelocity = vector2_t(3.0, 0.0);
  input.velocityCommand = vector2_t(3.0, 0.0);
  input.footPositions[0] = vector2_t(0.0, 0.125);
  input.footPositions[1] = vector2_t(-0.3, -0.125);
  input.lastSwungFoot = 1;
  const ContactPlan plan = planner.plan(input);
  std::cout << plan.describe() << std::endl;
  ASSERT_TRUE(plan.valid);
  const Phases phases = analyse(plan);
  EXPECT_GE(phases.flightNodes, 1) << "3 m/s is beyond what the cadence floor allows without flight";
  EXPECT_LE(phases.longestFlight, config.maxFlightNodes());
  EXPECT_EQ(phases.doubleSupportNodes, 0) << "no double support while running";
  // The gait alternates: every single support is on the other foot than the previous one.
  int previousFoot = -1;
  bool alternating = true;
  for (const contact_flag_t& c : plan.contacts) {
    if (c[0] == c[1]) continue;
    const int foot = c[0] ? 0 : 1;
    if (previousFoot == foot) {
      // the same foot in two consecutive single supports is only the same stance continuing
    }
    previousFoot = foot;
  }
  int lastStanceFoot = -1;
  bool changedFoot = false;
  for (size_t k = 0; k < plan.contacts.size(); ++k) {
    const contact_flag_t& c = plan.contacts[k];
    if (c[0] == c[1]) continue;
    const int foot = c[0] ? 0 : 1;
    if (lastStanceFoot >= 0 && foot != lastStanceFoot) changedFoot = true;
    if (lastStanceFoot >= 0 && foot == lastStanceFoot && k > 0 && plan.contacts[k - 1][0] == plan.contacts[k - 1][1]) alternating = false;
    lastStanceFoot = foot;
  }
  EXPECT_TRUE(changedFoot);
  EXPECT_TRUE(alternating) << "after a flight the other foot lands";
  // The CoM keeps its speed through the horizon.
  EXPECT_GT(plan.comVelocity.back().x(), 2.0);
}

/*============================================ the running formulation ============================================*/

// running.enabled keeps a second formulation with the flight model, which the planner module swaps in above the speed
// gate or on a jump request. The walking lists are untouched by it: that is the whole point, because merely listing the
// flight model changes the walk (see WhatTheFlightModelCostsTheWalkingGait).
TEST(ContactPlanningFlight, TheRunningVariantCarriesTheFlightModelAndLeavesTheWalkingOneAlone) {
  ContactPlanningConfig walking = atlasLikeConfig();
  walking.setFlightModel(false);  // the shipped lists: a walk
  walking.shared.gaitLimits.maxFlightDuration = 0.0;
  walking.shared.gaitLimits.maxSwingDuration = 0.5;
  walking.footSeparation.maxStepLength = 0.7;
  walking.planner.maxBranchAndBoundNodes = 200;
  walking.planner.maxSolveTime = 0.1;
  walking.running.enabled = true;
  walking.running.maxFlightDuration = 0.2;
  walking.running.maxSwingDuration = 0.6;
  walking.running.maxStepLength = 1.2;
  walking.running.maxBranchAndBoundNodes = 400;
  walking.running.maxSolveTime = 0.2;
  walking.validate();
  EXPECT_FALSE(walking.usesFlightModel());

  const ContactPlanningConfig running = walking.runningVariant();
  EXPECT_TRUE(running.usesFlightModel());
  EXPECT_TRUE(running.formulation.hasLogicRule(term::kFlightDurations));
  EXPECT_TRUE(running.formulation.hasLogicRule(term::kHopOnRequest));
  EXPECT_TRUE(running.formulation.hasExecutionRule(term::kPlannedHeightOverride));
  EXPECT_FALSE(running.formulation.hasHardConstraint(term::kNoFlight));
  EXPECT_DOUBLE_EQ(running.shared.gaitLimits.maxFlightDuration, 0.2);
  EXPECT_DOUBLE_EQ(running.shared.gaitLimits.maxSwingDuration, 0.6);
  EXPECT_DOUBLE_EQ(running.footSeparation.maxStepLength, 1.2);
  EXPECT_EQ(running.planner.maxBranchAndBoundNodes, 400);
  EXPECT_DOUBLE_EQ(running.planner.maxSolveTime, 0.2);
  EXPECT_FALSE(running.running.enabled) << "the variant is the destination of the switch, not a switch of its own";
  // The walking configuration is untouched by asking for the variant.
  EXPECT_FALSE(walking.usesFlightModel());
  EXPECT_DOUBLE_EQ(walking.shared.gaitLimits.maxSwingDuration, 0.5);
  EXPECT_DOUBLE_EQ(walking.footSeparation.maxStepLength, 0.7);
  EXPECT_EQ(walking.planner.maxBranchAndBoundNodes, 200);

  // Both formulations plan: the walk on the ground, the run with flight above the gate.
  LipContactPlanner walkingPlanner(walking);
  LipContactPlanner runningPlanner(running);
  const ContactPlan walk = walkingPlanner.plan(movingInput(0.8));
  ASSERT_TRUE(walk.valid);
  EXPECT_EQ(analyse(walk).flightNodes, 0);
  const ContactPlan run = runningPlanner.plan(movingInput(2.5));
  ASSERT_TRUE(run.valid);
  EXPECT_GT(analyse(run).flightNodes, 0);

  // A configuration that both lists the flight model and asks for the switch is contradictory.
  ContactPlanningConfig both = running;
  both.running.enabled = true;
  EXPECT_THROW(both.validate(), std::invalid_argument);
  // And the running block is checked on its own terms.
  ContactPlanningConfig shortStride = walking;
  shortStride.running.maxStepLength = walking.shared.bigM + 0.1;
  EXPECT_THROW(shortStride.validate(), std::invalid_argument);
  ContactPlanningConfig noFlight = walking;
  noFlight.running.maxFlightDuration = 0.0;
  EXPECT_THROW(noFlight.validate(), std::invalid_argument);
}

/*============================================ the Atlas gait: walk, then run ============================================*/

// The gait the shipped robot is meant to have: on the ground while the command is a walk, ballistic flight once the
// command asks for a speed the cadence cannot reach on the ground, and a hop whenever the jump button asks for one.
TEST(ContactPlanningFlight, AtlasWalksBelowTheGateAndRunsAboveIt) {
  const ContactPlanningConfig config = atlasLikeConfig();
  LipContactPlanner planner(config);

  const auto planAt = [&](scalar_t speed) {
    const ContactPlan plan = planner.plan(movingInput(speed));
    std::cout << "  " << speed << " m/s: " << plan.describe() << std::endl;
    return plan;
  };

  const ContactPlan walking = planAt(0.8);
  ASSERT_TRUE(walking.valid);
  EXPECT_EQ(analyse(walking).flightNodes, 0) << "a walk stays on the ground";

  const ContactPlan running = planAt(2.5);
  ASSERT_TRUE(running.valid);
  const Phases phases = analyse(running);
  EXPECT_GT(phases.flightNodes, 0) << "above the gate the planner may leave the ground";
  EXPECT_LE(phases.longestFlight, config.maxFlightNodes());
  EXPECT_GT(running.comVelocity.back().x(), 1.5) << "and it keeps the speed up";
}

// What listing the flight model costs the walking gait, at the budget the robot plans with. The speed gate keeps every
// node on the ground below it, so the contact pattern is the walking one either way; what changes is the size of every
// QP, and with it how far the search gets inside maxSolveTime. This is the measurement that decides whether the model
// can simply be listed or has to be switched in by speed.
TEST(ContactPlanningFlight, WhatTheFlightModelCostsTheWalkingGait) {
  ContactPlanningConfig running = atlasLikeConfig();
  running.planner.maxBranchAndBoundNodes = 200;  // the shipped budget, for both
  running.planner.maxSolveTime = 0.1;
  running.validate();
  ContactPlanningConfig walking = running;
  walking.setFlightModel(false);
  walking.shared.gaitLimits.maxFlightDuration = 0.0;
  walking.validate();

  LipContactPlanner runningPlanner(running);
  LipContactPlanner walkingPlanner(walking);
  for (const scalar_t speed : {0.0, 0.8}) {
    const ContactPlannerInput input = speed > 0.0 ? movingInput(speed) : standingInput();
    const ContactPlan withFlight = runningPlanner.plan(input);
    const ContactPlan withoutFlight = walkingPlanner.plan(input);
    std::cout << "  " << speed << " m/s without the flight model: " << withoutFlight.describe() << std::endl;
    std::cout << "  " << speed << " m/s with    the flight model: " << withFlight.describe() << std::endl;
    ASSERT_TRUE(withFlight.valid);
    ASSERT_TRUE(withoutFlight.valid);
    EXPECT_EQ(analyse(withFlight).flightNodes, 0) << "the gate keeps a walk on the ground";
    // The gait itself: the same contact pattern node for node is what "the walking gait is untouched" means.
    ASSERT_EQ(withFlight.contacts.size(), withoutFlight.contacts.size());
    size_t differingIntervals = 0;
    for (size_t k = 0; k < withFlight.contacts.size(); ++k) {
      if (withFlight.contacts[k] != withoutFlight.contacts[k]) ++differingIntervals;
    }
    std::cout << "  " << speed << " m/s: " << differingIntervals << " of " << withFlight.contacts.size()
              << " intervals differ, relaxations " << withoutFlight.numBranchAndBoundNodes << " -> " << withFlight.numBranchAndBoundNodes
              << ", solve " << withoutFlight.solveTime * 1e3 << " -> " << withFlight.solveTime * 1e3 << " ms" << std::endl;
  }
}

// The same gait at the solver budget the robot actually plans with. The search is an anytime one: it returns the best
// plan it has when the budget runs out, and what matters is that a usable plan comes back at every speed.
TEST(ContactPlanningFlight, AtlasFindsAPlanInsideItsSolverBudget) {
  ContactPlanningConfig config = atlasLikeConfig();
  config.planner.maxBranchAndBoundNodes = 200;  // the shipped budget
  config.planner.maxSolveTime = 0.1;            // [s]
  config.validate();
  LipContactPlanner planner(config);
  for (const scalar_t speed : {0.0, 0.8, 1.6, 2.5}) {
    ContactPlannerInput input = speed > 0.0 ? movingInput(speed) : standingInput();
    const ContactPlan plan = planner.plan(input);
    std::cout << "  budget " << speed << " m/s: " << plan.describe() << std::endl;
    EXPECT_TRUE(plan.valid) << "no plan at " << speed << " m/s inside the budget";
    EXPECT_LE(plan.solveTime, 4.0 * config.planner.maxSolveTime) << "the anytime limit is not honoured at " << speed << " m/s";
  }
  ContactPlannerInput hop = standingInput();
  hop.hopRequested = true;
  hop.hopFlightDuration = 0.2;
  const ContactPlan hopPlan = planner.plan(hop);
  std::cout << "  budget hop: " << hopPlan.describe() << std::endl;
  EXPECT_TRUE(hopPlan.valid) << "the jump button must produce a plan inside the budget";
  EXPECT_GT(hopPlan.numFlightIntervals(), 0);
  // The push-off may dip into a counter-movement, but the leg has to be able to follow it.
  const auto [lowest, highest] = std::minmax_element(hopPlan.comHeight.begin(), hopPlan.comHeight.end());
  EXPECT_GT(*lowest, config.shared.comHeight - 0.15) << "the crouch before the hop stays within a leg's travel";
  EXPECT_GT(*highest, config.shared.comHeight) << "and the hop rises above the standing height";
}

// The yaw torque a foot may carry is the ground's, and a foot in the air has none. Reading the double-support share off
// c_L + c_R - 1 made that budget negative with no foot down, so every flight node was infeasible for a robot with the
// heading model: the whole gait above collapses without this.
TEST(ContactPlanningFlight, TheYawTorqueBudgetIsZeroInFlightRatherThanNegative) {
  ContactPlanningConfig config = atlasLikeConfig();
  config.formulation.costs = {term::kRegularization};
  config.formulation.softConstraints = {};
  config.formulation.hardConstraints = {term::kYawTorqueBudget};
  config.validate();
  LipContactPlanner planner(config);
  const Layout& layout = planner.getLayout();
  const OcpQpProblem qp = planner.buildProblem(standingInput());
  const OcpQpStage& stage = qp.stages.front();

  // Every row is +-tau_i - T_t c_i - share c_j <= 0, so with no foot down the budget is exactly zero and tau = 0 is
  // feasible; the row that carries the share of the other foot is the binding one in single support.
  ASSERT_EQ(stage.numGeneralConstraints(), 8) << "two feet, two signs, two rows each";
  for (int row = 0; row < stage.numGeneralConstraints(); ++row) {
    EXPECT_DOUBLE_EQ(stage.ug(row), 0.0) << "row " << row << " must not demand torque where there is no contact";
  }
  const scalar_t share = 0.5 * (config.yawTorqueBudget.doubleSupportYawCouple - config.yawTorqueBudget.torsionalFrictionTorque);
  EXPECT_DOUBLE_EQ(stage.D(0, layout.yawTorque(0)), 1.0);
  EXPECT_DOUBLE_EQ(stage.D(0, LIP_CL), -config.yawTorqueBudget.torsionalFrictionTorque - share) << "j = L folds into c_L";
  EXPECT_DOUBLE_EQ(stage.D(1, LIP_CL), -config.yawTorqueBudget.torsionalFrictionTorque);
  EXPECT_DOUBLE_EQ(stage.D(1, LIP_CR), -share) << "j = R: the binding row in single support";

  // Without the flight model the rows are the ones the walking robot has always had, which the fixtures pin.
  ContactPlanningConfig walking = config;
  walking.setFlightModel(false);
  walking.shared.gaitLimits.maxFlightDuration = 0.0;
  walking.validate();
  const LipContactPlanner walkingPlanner(walking);
  const OcpQpProblem walkingQp = walkingPlanner.buildProblem(standingInput());
  // Four yaw rows again, and the no_flight row that comes back with them.
  EXPECT_EQ(walkingQp.stages.front().numGeneralConstraints(), 5);
  EXPECT_DOUBLE_EQ(walkingQp.stages.front().ug(0), -share);
  EXPECT_DOUBLE_EQ(walkingQp.stages.front().D(0, LIP_CL), -config.yawTorqueBudget.torsionalFrictionTorque - share);
}

// The gate: below allowedAboveSpeed the rule keeps a foot on the ground, so the walking search is the one the robot
// already had, and a hop request opens it at any speed.
TEST(ContactPlanningFlight, TheFlightGateFollowsTheCommandedSpeedAndTheHopRequest) {
  ContactPlanningConfig config = flightConfig();
  config.flightDurations.allowedAboveSpeed = 1.3;
  config.validate();
  LipContactPlanner planner(config);
  FlightDurationsRule rule;
  rule.configure(config);
  EXPECT_NE(rule.describe().find("on the ground below 1.3 m/s"), std::string::npos) << rule.describe();

  const auto flightIsAllowed = [&](const ContactPlannerInput& input) {
    const ContactLogicState state = planner.makeLogicState(input);
    MiqpAssignment a = planner.initialAssignment(input);
    for (int k = 0; k < config.planner.numNodes; ++k) {
      a[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(k, 0))] = (k == 3 || k == 4) ? 0 : 1;
      a[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(k, 1))] = (k == 3 || k == 4) ? 0 : 1;
    }
    bool changed = false;
    return rule.propagate(state, ContactLogicScan::compute(state, a), a, changed);
  };
  EXPECT_FALSE(flightIsAllowed(movingInput(0.8))) << "a walk keeps a foot down";
  EXPECT_TRUE(flightIsAllowed(movingInput(2.5))) << "a run may leave the ground";
  ContactPlannerInput hop = standingInput();
  hop.hopRequested = true;
  hop.hopFlightDuration = 0.2;
  EXPECT_TRUE(flightIsAllowed(hop)) << "the jump button opens the gate at any speed";
  // With the gate open at every speed the walk may leave the ground too.
  ContactPlanningConfig ungated = config;
  ungated.flightDurations.allowedAboveSpeed = 0.0;
  ungated.validate();
  FlightDurationsRule always;
  always.configure(ungated);
  const ContactLogicState state = planner.makeLogicState(movingInput(0.8));
  MiqpAssignment a = planner.initialAssignment(movingInput(0.8));
  for (int k = 0; k < config.planner.numNodes; ++k) {
    a[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(k, 0))] = (k == 3 || k == 4) ? 0 : 1;
    a[static_cast<size_t>(LipContactPlanner::contactBinaryIndex(k, 1))] = (k == 3 || k == 4) ? 0 : 1;
  }
  bool changed = false;
  EXPECT_TRUE(always.propagate(state, ContactLogicScan::compute(state, a), a, changed));
}

}  // namespace ocs2::humanoid
