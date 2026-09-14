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

#include <optional>

#include <tuple>

#include <cmath>
#include <deque>
#include <iostream>
#include <limits>

#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"
#include "humanoid_common_mpc/contact_planning/LipContactPlanner.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::humanoid {

namespace {

constexpr scalar_t kSoftTol = 2e-2;  // the geometric constraints are soft; allow a small violation

ContactPlanningConfig makeConfig() {
  ContactPlanningConfig config;
  config.dt = 0.1;
  config.numNodes = 12;
  config.commitTime = 0.0;
  config.comHeight = 0.85;
  config.minSwingDuration = 0.3;
  config.maxSwingDuration = 0.5;
  config.minContactDuration = 0.15;
  config.maxContactDuration = 0.0;
  config.minDoubleSupportDuration = 0.1;
  config.maxBranchAndBoundNodes = 3000;
  config.maxSolveTime = 10.0;
  config.localSearchMaxTime = 2.0;
  config.verbose = false;
  config.validate();
  return config;
}

ContactPlannerInput makeStandingInput() {
  ContactPlannerInput input;
  input.time = 3.0;
  input.comPosition = vector2_t(0.0, 0.0);
  input.comVelocity = vector2_t::Zero();
  input.yaw = 0.0;
  input.footPositions[0] = vector2_t(0.0, 0.125);
  input.footPositions[1] = vector2_t(0.0, -0.125);
  input.contacts = {true, true};
  input.phaseElapsedTime = {5.0, 5.0};
  input.velocityCommand = vector2_t::Zero();
  return input;
}

struct PhaseStats {
  int numSwings[2] = {0, 0};
  int minSwingNodes = 1000;
  int maxSwingNodes = 0;
  int minContactNodesInterior = 1000;  // contact runs that both start and end inside the horizon
  bool flight = false;
};

PhaseStats analyse(const ContactPlan& plan) {
  PhaseStats stats;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    int run = 0;
    bool current = plan.contacts[0][foot];
    bool runStartedInside = false;
    for (int k = 0; k < plan.numIntervals(); ++k) {
      const bool c = plan.contacts[k][foot];
      if (k > 0 && c != current) {
        if (!current) {
          stats.numSwings[foot]++;
          stats.minSwingNodes = std::min(stats.minSwingNodes, run);
          stats.maxSwingNodes = std::max(stats.maxSwingNodes, run);
        } else if (runStartedInside) {
          stats.minContactNodesInterior = std::min(stats.minContactNodesInterior, run);
        }
        run = 0;
        current = c;
        runStartedInside = true;
      }
      ++run;
    }
  }
  for (const contact_flag_t& c : plan.contacts) {
    if (!c[0] && !c[1]) stats.flight = true;
  }
  return stats;
}

bool zmpInsideSupport(const ContactPlan& plan, const ContactPlanningConfig& config, int k) {
  const vector2_t& z = plan.zmp[k];
  const contact_flag_t c = plan.contacts[k];
  scalar_t xMin = 1e9, xMax = -1e9, yMin = 1e9, yMax = -1e9;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    if (!c[foot]) continue;
    const vector2_t& p = plan.footholds[k][foot];
    xMin = std::min(xMin, p(0) - config.zmpHalfWidthX);
    xMax = std::max(xMax, p(0) + config.zmpHalfWidthX);
    yMin = std::min(yMin, p(1) - config.zmpHalfWidthY);
    yMax = std::max(yMax, p(1) + config.zmpHalfWidthY);
  }
  return z(0) >= xMin - kSoftTol && z(0) <= xMax + kSoftTol && z(1) >= yMin - kSoftTol && z(1) <= yMax + kSoftTol;
}

void printPlan(const ContactPlan& plan) {
  std::cout << "plan valid=" << plan.valid << " objective=" << plan.objective << " nodes=" << plan.numBranchAndBoundNodes
            << " time=" << plan.solveTime << " s optimal=" << plan.optimal << "\n  contacts: ";
  if (!plan.valid) {
    std::cout << "(none)\n";
    return;
  }
  for (const contact_flag_t& c : plan.contacts) std::cout << "[" << c[0] << c[1] << "]";
  std::cout << "\n";
  for (int k = 0; k <= plan.numIntervals(); ++k) {
    std::cout << "  k=" << k << " com=" << plan.comPosition[k].transpose() << " v=" << plan.comVelocity[k].transpose()
              << " pL=" << plan.footholds[k][0].transpose() << " pR=" << plan.footholds[k][1].transpose();
    if (k < plan.numIntervals()) std::cout << " zmp=" << plan.zmp[k].transpose();
    std::cout << "\n";
  }
}

}  // namespace

TEST(LipContactPlannerTest, StandingProducesNoSteps) {
  LipContactPlanner planner(makeConfig());
  const ContactPlannerInput input = makeStandingInput();
  const ContactPlan plan = planner.plan(input);
  printPlan(plan);
  ASSERT_TRUE(plan.valid);
  EXPECT_TRUE(plan.optimal);
  for (const contact_flag_t& c : plan.contacts) {
    EXPECT_TRUE(c[0] && c[1]);
  }
  EXPECT_LT((plan.comPosition.back() - input.comPosition).norm(), 0.02);
  EXPECT_LT((plan.footholds.back()[0] - input.footPositions[0]).norm(), 1e-6);
  EXPECT_LT((plan.footholds.back()[1] - input.footPositions[1]).norm(), 1e-6);
  const ModeSchedule schedule = plan.toModeSchedule();
  EXPECT_TRUE(schedule.eventTimes.empty());
  EXPECT_EQ(schedule.modeSequence.size(), 1u);
  EXPECT_EQ(schedule.modeSequence.front(), static_cast<size_t>(ModeNumber::STANCE));
}

TEST(LipContactPlannerTest, WalkingCommandProducesAlternatingSteps) {
  ContactPlanningConfig config = makeConfig();
  config.verbose = true;
  LipContactPlanner planner(config);
  ContactPlannerInput input = makeStandingInput();
  input.velocityCommand = vector2_t(0.4, 0.0);
  const ContactPlan plan = planner.plan(input);
  printPlan(plan);
  ASSERT_TRUE(plan.valid);

  const PhaseStats stats = analyse(plan);
  EXPECT_FALSE(stats.flight);
  EXPECT_GE(stats.numSwings[0] + stats.numSwings[1], 1) << "walking needs at least one step within 1.2 s";
  EXPECT_GE(stats.minSwingNodes, config.minSwingNodes());
  EXPECT_LE(stats.maxSwingNodes, config.maxSwingNodes());
  EXPECT_GE(stats.minContactNodesInterior, config.minContactNodes());

  // At least one foot moves forward and the CoM follows the command.
  EXPECT_GT(std::max(plan.footholds.back()[0](0), plan.footholds.back()[1](0)), 0.05);
  EXPECT_GT(plan.comPosition.back()(0), 0.15);
  EXPECT_GT(plan.comVelocity.back()(0), 0.2);

  // Geometry (soft constraints, allow a small tolerance).
  for (int k = 0; k <= plan.numIntervals(); ++k) {
    const vector2_t separation = plan.footholds[k][0] - plan.footholds[k][1];
    EXPECT_GE(separation(1), config.minStepWidth - kSoftTol) << "k=" << k;
    EXPECT_LE(separation(1), config.maxStepWidth + kSoftTol) << "k=" << k;
    EXPECT_LE(std::abs(separation(0)), config.maxStepLength + kSoftTol) << "k=" << k;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      const vector2_t rel = plan.footholds[k][foot] - plan.comPosition[k];
      EXPECT_LE(std::abs(rel(0)), config.reachX + kSoftTol) << "k=" << k;
    }
  }
  // A foot in contact does not move.
  for (int k = 0; k < plan.numIntervals(); ++k) {
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      if (plan.contacts[k][foot]) {
        EXPECT_LT((plan.footholds[k + 1][foot] - plan.footholds[k][foot]).norm(), 1e-6) << "k=" << k << " foot=" << foot;
      }
    }
  }
  // The ZMP stays inside the support region (soft constraint): the foot box in single support, the hull in double support.
  for (int k = 0; k < plan.numIntervals(); ++k) {
    EXPECT_TRUE(zmpInsideSupport(plan, config, k)) << "k=" << k << " zmp=" << plan.zmp[k].transpose();
  }
  std::cout << "walking plan solved in " << plan.solveTime * 1e3 << " ms with " << plan.numBranchAndBoundNodes << " relaxations, "
            << planner.getLastStatistics().totalQpIterations << " IPM iterations\n";
}

TEST(LipContactPlannerTest, ForwardPushTriggersRecoveryStep) {
  const ContactPlanningConfig config = makeConfig();
  LipContactPlanner planner(config);
  ContactPlannerInput input = makeStandingInput();
  input.comVelocity = vector2_t(0.7, 0.0);  // DCM 0.2 m ahead of the feet: not capturable without stepping
  const ContactPlan plan = planner.plan(input);
  printPlan(plan);
  ASSERT_TRUE(plan.valid);
  const PhaseStats stats = analyse(plan);
  EXPECT_FALSE(stats.flight);
  EXPECT_GE(stats.numSwings[0] + stats.numSwings[1], 1) << "a push of this size requires a step";
  EXPECT_GE(stats.minSwingNodes, config.minSwingNodes());
  // The plan brings the robot to a capturable state: terminal DCM close to the last ZMP.
  const scalar_t omega = config.omega();
  const vector2_t dcm = plan.comPosition.back() + plan.comVelocity.back() / omega;
  EXPECT_LT((dcm - plan.zmp.back()).norm(), 0.08);
  EXPECT_LT(plan.comVelocity.back().norm(), 0.35);
}

TEST(LipContactPlannerTest, CommittedContactsAreRespected) {
  LipContactPlanner planner(makeConfig());
  ContactPlannerInput input = makeStandingInput();
  input.velocityCommand = vector2_t(0.3, 0.0);
  input.contacts = {true, false};  // right foot already swinging for 0.1 s
  input.phaseElapsedTime = {0.6, 0.1};
  input.footPositions[1] = vector2_t(0.05, -0.125);
  input.committedContacts = {{true, false}, {true, false}, {true, false}};
  const ContactPlan plan = planner.plan(input);
  printPlan(plan);
  ASSERT_TRUE(plan.valid);
  for (size_t k = 0; k < input.committedContacts.size(); ++k) {
    EXPECT_EQ(plan.contacts[k][0], input.committedContacts[k][0]);
    EXPECT_EQ(plan.contacts[k][1], input.committedContacts[k][1]);
  }
  // The right foot must land within maxSwingDuration (0.5 s => 5 nodes) counted from its lift-off 0.1 s ago.
  int firstRightContact = -1;
  for (int k = 0; k < plan.numIntervals(); ++k) {
    if (plan.contacts[k][1]) {
      firstRightContact = k;
      break;
    }
  }
  ASSERT_GE(firstRightContact, 3);
  EXPECT_LE(firstRightContact, 4);
}

TEST(LipContactPlannerTest, PropagationEnforcesDurations) {
  const ContactPlanningConfig config = makeConfig();
  LipContactPlanner planner(config);
  ContactPlannerInput input = makeStandingInput();
  input.contacts = {true, false};
  input.phaseElapsedTime = {1.0, 0.0};  // right foot just lifted off

  // Landing after one node violates the minimum swing duration of 3 nodes.
  MiqpAssignment tooEarly = planner.initialAssignment(input);
  tooEarly[LipContactPlanner::contactBinaryIndex(1, 1)] = 1;
  EXPECT_FALSE(planner.propagate(input, tooEarly));

  // A free assignment gets the first minSwingNodes right-foot nodes fixed to swing and the left foot to contact (no flight).
  MiqpAssignment free = planner.initialAssignment(input);
  ASSERT_TRUE(planner.propagate(input, free));
  for (int k = 0; k < config.minSwingNodes(); ++k) {
    EXPECT_EQ(free[LipContactPlanner::contactBinaryIndex(k, 1)], 0) << "k=" << k;
    EXPECT_EQ(free[LipContactPlanner::contactBinaryIndex(k, 0)], 1) << "k=" << k;
  }
  // Alternation: after the right foot lands it may not lift again before the left foot has swung.
  MiqpAssignment sameFootTwice = planner.initialAssignment(input);
  for (int k = 0; k < config.numNodes; ++k) {
    sameFootTwice[LipContactPlanner::contactBinaryIndex(k, 0)] = 1;
    sameFootTwice[LipContactPlanner::contactBinaryIndex(k, 1)] = (k < 3 || k >= 6) ? 1 : 0;
  }
  EXPECT_FALSE(planner.propagate(input, sameFootTwice));
  MiqpAssignment alternating = planner.initialAssignment(input);
  for (int k = 0; k < config.numNodes; ++k) {
    alternating[LipContactPlanner::contactBinaryIndex(k, 1)] = (k < 3) ? 0 : 1;
    alternating[LipContactPlanner::contactBinaryIndex(k, 0)] = (k >= 5 && k < 8) ? 0 : 1;
  }
  EXPECT_TRUE(planner.propagate(input, alternating));
  // Staying in swing past maxSwingNodes is forbidden.
  MiqpAssignment tooLong = planner.initialAssignment(input);
  for (int k = 0; k <= config.maxSwingNodes(); ++k) {
    tooLong[LipContactPlanner::contactBinaryIndex(k, 1)] = 0;
  }
  EXPECT_FALSE(planner.propagate(input, tooLong));
}

TEST(LipContactPlannerTest, MinimumDoubleSupportAndConsistencyCost) {
  ContactPlanningConfig config = makeConfig();
  config.minDoubleSupportDuration = 0.2;  // 2 nodes
  LipContactPlanner planner(config);
  ContactPlannerInput input = makeStandingInput();
  input.contacts = {true, false};  // right foot swinging
  input.phaseElapsedTime = {1.0, 0.2};

  // Right lands at node 1; the left foot may not lift before node 3.
  MiqpAssignment tooEarly = planner.initialAssignment(input);
  for (int k = 0; k < config.numNodes; ++k) {
    tooEarly[LipContactPlanner::contactBinaryIndex(k, 1)] = (k < 1) ? 0 : 1;
    tooEarly[LipContactPlanner::contactBinaryIndex(k, 0)] = (k == 2) ? 0 : 1;
  }
  EXPECT_FALSE(planner.propagate(input, tooEarly));
  MiqpAssignment lateEnough = planner.initialAssignment(input);
  for (int k = 0; k < config.numNodes; ++k) {
    lateEnough[LipContactPlanner::contactBinaryIndex(k, 1)] = (k < 1) ? 0 : 1;
    lateEnough[LipContactPlanner::contactBinaryIndex(k, 0)] = (k >= 3 && k < 6) ? 0 : 1;
  }
  EXPECT_TRUE(planner.propagate(input, lateEnough));

  // Without a previous plan the assignment cost is the switch cost only.
  const scalar_t switches = planner.assignmentCost(input, lateEnough);
  EXPECT_NEAR(switches, 3.0 * config.contactSwitchCost, 1e-12);  // right touch-down, left lift-off, left touch-down

  // After a plan exists, deviating from it is charged per node.
  const ContactPlan first = planner.plan(input);
  ASSERT_TRUE(first.valid);
  MiqpAssignment same(planner.getLastResult().assignment);
  EXPECT_NEAR(planner.assignmentCost(input, same), planner.getLastResult().incumbentObjective - planner.getLastResult().solution.objective,
              1e-9);
  MiqpAssignment flipped = same;
  const int node = config.numNodes - 1;
  flipped[LipContactPlanner::contactBinaryIndex(node, 0)] = 1 - flipped[LipContactPlanner::contactBinaryIndex(node, 0)];
  EXPECT_GT(planner.assignmentCost(input, flipped), planner.assignmentCost(input, same) + 0.9 * config.planConsistencyCost);
}

TEST(LipContactPlannerTest, MinimumDoubleSupportForbidsASimultaneousSwitch) {
  // Left swings over nodes 1..3 and lands at node 4. If the right foot lifts at that same node there is no double
  // support at all: single support on the right turns into single support on the left in one node, which is exactly
  // the weight transfer minDoubleSupportDuration is meant to forbid. The propagation has to reject it just as it rejects
  // a lift-off one node after the landing.
  ContactPlanningConfig config = makeConfig();
  config.minDoubleSupportDuration = 0.2;  // 2 nodes
  LipContactPlanner planner(config);
  ContactPlannerInput input = makeStandingInput();  // both feet down for a long time

  const auto assignmentWithRightLiftOffAt = [&](int liftOffNode) {
    MiqpAssignment a = planner.initialAssignment(input);
    for (int k = 0; k < config.numNodes; ++k) {
      a[LipContactPlanner::contactBinaryIndex(k, 0)] = (k >= 1 && k < 4) ? 0 : 1;                          // left swing 1..3
      a[LipContactPlanner::contactBinaryIndex(k, 1)] = (k >= liftOffNode && k < liftOffNode + 3) ? 0 : 1;  // right swing
    }
    return a;
  };

  MiqpAssignment simultaneous = assignmentWithRightLiftOffAt(4);
  EXPECT_FALSE(planner.propagate(input, simultaneous)) << "the right foot lifted at the node the left foot landed";
  MiqpAssignment oneNode = assignmentWithRightLiftOffAt(5);
  EXPECT_FALSE(planner.propagate(input, oneNode)) << "one node of double support is less than the two required";
  MiqpAssignment twoNodes = assignmentWithRightLiftOffAt(6);
  EXPECT_TRUE(planner.propagate(input, twoNodes));

  // With the right foot still free at the landing node, propagation has to pin it to contact there and for the
  // following node, not only for the nodes after the landing.
  MiqpAssignment partial = assignmentWithRightLiftOffAt(6);
  for (int k = 4; k < config.numNodes; ++k) partial[LipContactPlanner::contactBinaryIndex(k, 1)] = kMiqpFree;
  ASSERT_TRUE(planner.propagate(input, partial));
  EXPECT_EQ(partial[LipContactPlanner::contactBinaryIndex(4, 1)], 1) << "right foot must stay down at the landing node";
  EXPECT_EQ(partial[LipContactPlanner::contactBinaryIndex(5, 1)], 1) << "and for the second double-support node";
}

TEST(LipContactPlannerTest, ElapsedPhaseTimeIsRoundedConservativelyForBothLimits) {
  // dt 0.1, swing limits [0.3, 0.5]. A swing 0.16 s old used to count as 2 nodes (nearest), so ending it after one more
  // node (0.26 s in total) passed the 0.3 s minimum; a swing 0.44 s old counted as 4 nodes and could go on one more
  // node (0.54 s) against the 0.5 s maximum.
  ContactPlanningConfig config = makeConfig();
  config.minDoubleSupportDuration = 0.0;
  LipContactPlanner planner(config);

  ContactPlannerInput input = makeStandingInput();
  input.contacts = {true, false};  // right foot swinging
  input.phaseElapsedTime = {1.0, 0.16};
  const auto rightLandsAt = [&](int node) {
    MiqpAssignment a = planner.initialAssignment(input);
    for (int k = 0; k < config.numNodes; ++k) {
      a[LipContactPlanner::contactBinaryIndex(k, 0)] = 1;
      a[LipContactPlanner::contactBinaryIndex(k, 1)] = (k < node) ? 0 : 1;
    }
    return a;
  };
  MiqpAssignment tooShort = rightLandsAt(1);  // 0.16 + 0.1 = 0.26 s
  EXPECT_FALSE(planner.propagate(input, tooShort)) << "a 0.26 s swing violates the 0.3 s minimum";
  MiqpAssignment longEnough = rightLandsAt(2);  // 0.16 + 0.2 = 0.36 s
  EXPECT_TRUE(planner.propagate(input, longEnough));

  input.phaseElapsedTime = {1.0, 0.44};
  MiqpAssignment tooLong = rightLandsAt(1);  // 0.44 + 0.1 = 0.54 s
  EXPECT_FALSE(planner.propagate(input, tooLong)) << "a 0.54 s swing violates the 0.5 s maximum";
  MiqpAssignment landsNow = rightLandsAt(0);  // 0.44 s
  EXPECT_TRUE(planner.propagate(input, landsNow));

  // Limits less than a node apart cannot both be honoured on the grid for a mid-node elapsed time; the nearest node
  // decides instead of the horizon becoming infeasible.
  ContactPlanningConfig tight = makeConfig();
  tight.minSwingDuration = 0.3;
  tight.maxSwingDuration = 0.3;
  tight.minDoubleSupportDuration = 0.0;
  LipContactPlanner tightPlanner(tight);
  input.phaseElapsedTime = {1.0, 0.25};
  MiqpAssignment free = tightPlanner.initialAssignment(input);
  EXPECT_TRUE(tightPlanner.propagate(input, free));
}

TEST(LipContactPlannerTest, TouchDownInsideACommittedNodeCountsFromTheExecutedEvent) {
  // The executed schedule lands the right foot at 0.97 s, inside the committed node [0.9, 1.0) of a plan whose grid
  // starts at 0.7 s (the node straddles the commit boundary, which is the touch-down). Sampled at the boundary, the
  // node reports the foot in contact; counted from the node start the contact would be 0.1 s old at 1.0 s when it is
  // really 0.03 s old, and the left foot could lift at 1.0 s with a double support of 0.03 s against the 0.1 s minimum.
  ContactPlanningConfig config = makeConfig();
  config.enforceAlternatingFeet = false;  // so that the right foot's minimum contact duration is what limits its re-lift
  LipContactPlanner planner(config);
  ContactPlannerInput input = makeStandingInput();
  input.time = 0.7;
  input.velocityCommand = vector2_t(0.3, 0.0);
  input.contacts = {true, false};
  input.phaseElapsedTime = {5.0, 0.13};  // the right foot lifted at 0.57
  input.committedUntil = 0.97;
  input.committedContacts = {{true, false}, {true, false}, {true, true}};
  const scalar_t longAgo = -std::numeric_limits<scalar_t>::infinity();
  input.committedPhaseStartTimes = {{longAgo, 0.57}, {longAgo, 0.57}, {longAgo, 0.97}};

  const auto assignment = [&](int leftLiftOff, int rightLiftOff) {
    MiqpAssignment a = planner.initialAssignment(input);
    for (int k = 0; k < config.numNodes; ++k) {
      a[LipContactPlanner::contactBinaryIndex(k, 0)] = (k >= leftLiftOff && k < leftLiftOff + 3) ? 0 : 1;
      if (k >= 3) a[LipContactPlanner::contactBinaryIndex(k, 1)] = (k >= rightLiftOff && k < rightLiftOff + 3) ? 0 : 1;
    }
    return a;
  };
  // Minimum double support 0.1 s after the touch-down at 0.97: the left foot may lift at 1.1 (node 4), not at 1.0.
  MiqpAssignment leftLiftsAtOne = assignment(3, 100);
  EXPECT_FALSE(planner.propagate(input, leftLiftsAtOne)) << "0.03 s of double support";
  MiqpAssignment leftLiftsAtOneOne = assignment(4, 100);
  EXPECT_TRUE(planner.propagate(input, leftLiftsAtOneOne)) << "0.13 s of double support";
  // With the left foot free at those nodes the propagation pins it down at node 3.
  MiqpAssignment partial = assignment(100, 100);
  for (int k = 3; k < config.numNodes; ++k) partial[LipContactPlanner::contactBinaryIndex(k, 0)] = kMiqpFree;
  ASSERT_TRUE(planner.propagate(input, partial));
  EXPECT_EQ(partial[LipContactPlanner::contactBinaryIndex(3, 0)], 1);
  EXPECT_EQ(partial[LipContactPlanner::contactBinaryIndex(4, 0)], kMiqpFree);
  // Minimum contact duration 0.15 s after the touch-down at 0.97: the right foot may lift again at 1.2 (node 5), not 1.1.
  MiqpAssignment rightLiftsAtOneOne = assignment(100, 4);
  EXPECT_FALSE(planner.propagate(input, rightLiftsAtOneOne)) << "0.13 s of contact";
  MiqpAssignment rightLiftsAtOneTwo = assignment(100, 5);
  EXPECT_TRUE(planner.propagate(input, rightLiftsAtOneTwo)) << "0.23 s of contact";

  // Counted from the node start (no executed event times) both of the too-early lift-offs pass: that was the bug.
  input.committedPhaseStartTimes.clear();
  MiqpAssignment fromNodeStart = assignment(3, 100);
  EXPECT_TRUE(planner.propagate(input, fromNodeStart));
  MiqpAssignment rightFromNodeStart = assignment(100, 4);
  EXPECT_TRUE(planner.propagate(input, rightFromNodeStart));

  // A touch-down exactly at the node start is the grid case and counts as a whole node, as before.
  input.committedPhaseStartTimes = {{longAgo, 0.57}, {longAgo, 0.57}, {longAgo, 0.9}};
  MiqpAssignment gridAligned = assignment(3, 100);
  EXPECT_TRUE(planner.propagate(input, gridAligned));
  MiqpAssignment gridAlignedRight = assignment(100, 4);
  EXPECT_TRUE(planner.propagate(input, gridAlignedRight));
}

TEST(LipContactPlannerTest, PlanCarriesThePlanningFrameYaw) {
  // The reference manager clips corrected footholds in the plan's yaw frame, so the plan has to remember the heading
  // the geometry was planned in.
  const ContactPlanningConfig config = makeConfig();
  LipContactPlanner planner(config);
  ContactPlannerInput input = makeStandingInput();
  input.yaw = 0.7;
  const ContactPlan plan = planner.plan(input);
  ASSERT_TRUE(plan.valid);
  EXPECT_DOUBLE_EQ(plan.yaw, 0.7);
  EXPECT_DOUBLE_EQ(plan.startTime, input.time);
  EXPECT_DOUBLE_EQ(plan.committedUntil, input.committedUntil);
}

TEST(LipContactPlannerTest, MaximumContactDurationYieldsToNoFlightAndDoubleSupport) {
  // Both feet have stood far longer than maxContactDuration. Forcing both to lift at the same node contradicts the
  // no-flight rule and made every assignment infeasible: the planner returned no plan at all on a standing start.
  ContactPlanningConfig config = makeConfig();
  config.maxContactDuration = 0.6;        // 6 nodes
  config.minDoubleSupportDuration = 0.1;  // 1 node
  LipContactPlanner planner(config);
  ContactPlannerInput input = makeStandingInput();  // both feet down for 5 s, no velocity command

  MiqpAssignment bothStayDown = planner.initialAssignment(input);
  std::fill(bothStayDown.begin(), bothStayDown.end(), 1);
  EXPECT_FALSE(planner.propagate(input, bothStayDown)) << "the limit still forces a step";
  MiqpAssignment bothLift = planner.initialAssignment(input);
  std::fill(bothLift.begin(), bothLift.end(), 1);
  bothLift[LipContactPlanner::contactBinaryIndex(0, 0)] = 0;
  bothLift[LipContactPlanner::contactBinaryIndex(0, 1)] = 0;
  EXPECT_FALSE(planner.propagate(input, bothLift)) << "no flight";

  // Foot 0 lifts now (swing 0..2, lands at 3, stands 3..7 which is the 0.6 s limit, lifts again at 8). Foot 1 yields
  // while it cannot lift (foot 0 in the air, then the one-node double support at 3) and lifts at the first node where
  // it can, node 4.
  const auto sequence = [&](int secondLiftOff) {
    MiqpAssignment a = planner.initialAssignment(input);
    for (int k = 0; k < config.numNodes; ++k) {
      a[LipContactPlanner::contactBinaryIndex(k, 0)] = (k < 3 || (k >= 8 && k < 11)) ? 0 : 1;
      a[LipContactPlanner::contactBinaryIndex(k, 1)] = (k >= secondLiftOff && k < secondLiftOff + 3) ? 0 : 1;
    }
    return a;
  };
  MiqpAssignment liftsWhenAllowed = sequence(4);
  EXPECT_TRUE(planner.propagate(input, liftsWhenAllowed));
  MiqpAssignment liftsInsideDoubleSupport = sequence(3);
  EXPECT_FALSE(planner.propagate(input, liftsInsideDoubleSupport)) << "the double support after the landing holds it";
  MiqpAssignment staysTooLong = sequence(5);
  EXPECT_FALSE(planner.propagate(input, staysTooLong)) << "overdue and able to lift at node 4, so it must";

  // Propagation on a free assignment lifts exactly one foot at the first free node and leaves the other down.
  MiqpAssignment free = planner.initialAssignment(input);
  ASSERT_TRUE(planner.propagate(input, free));
  const std::int8_t first = free[LipContactPlanner::contactBinaryIndex(0, 0)];
  const std::int8_t second = free[LipContactPlanner::contactBinaryIndex(0, 1)];
  EXPECT_EQ(static_cast<int>(first) + static_cast<int>(second), 1) << "one foot lifts, the other supports";

  // And the planner produces a valid stepping plan from that state.
  const ContactPlan plan = planner.plan(input);
  ASSERT_TRUE(plan.valid);
  EXPECT_FALSE(analyse(plan).flight);
  EXPECT_GT(analyse(plan).numSwings[0] + analyse(plan).numSwings[1], 0) << "the limit forces stepping";
}

namespace {
/** Calls `visit(foot, liftOff, touchDown)` for every swing of `schedule` with a lift-off event. */
template <typename Visitor>
void forEachSwing(const ModeSchedule& schedule, Visitor&& visit) {
  for (size_t f = 0; f < N_CONTACTS; ++f) {
    for (size_t i = 0; i + 1 < schedule.modeSequence.size(); ++i) {
      const bool liftOff = modeNumber2StanceLeg(schedule.modeSequence[i])[f] && !modeNumber2StanceLeg(schedule.modeSequence[i + 1])[f];
      if (!liftOff) continue;
      for (size_t j = i + 1; j < schedule.eventTimes.size(); ++j) {
        if (modeNumber2StanceLeg(schedule.modeSequence[j + 1])[f]) {
          visit(f, schedule.eventTimes[i], schedule.eventTimes[j]);
          break;
        }
      }
    }
  }
}
}  // namespace

namespace {
/**
 * Drives the planner the way ContactPlanningReferenceManager does (same commit boundary, same committed-node sampling,
 * same activation and merge policy) for 3 s of walking at 0.4 m/s. A plan computed at tick i is activated
 * `latencyTicks` ticks later, like a plan the background planner hands over one or more solves later: it is merged at
 * max(now, plan.committedUntil), and dropped if that boundary has already passed or if it contradicts a swing in flight
 * there. Returns the executed schedule with its full history.
 */
ModeSchedule walkRecedingHorizon(const ContactPlanningConfig& c, int latencyTicks, scalar_t tick, int steps) {
  LipContactPlanner planner(c);
  ContactPlannerInput in = makeStandingInput();
  in.velocityCommand = vector2_t(0.4, 0.0);
  ModeSchedule applied({-1.0, 20.0}, {ModeNumber::STANCE, ModeNumber::STANCE, ModeNumber::STANCE});
  std::deque<std::pair<int, ContactPlan>> pending;  // (tick at which the plan is applied, plan)

  const auto checkFutureSwings = [&](const ModeSchedule& schedule, scalar_t from, const char* label, scalar_t t) {
    forEachSwing(schedule, [&](size_t f, scalar_t liftOff, scalar_t touchDown) {
      if (liftOff < from - 1e-9) return;
      EXPECT_GE(touchDown - liftOff, c.minSwingDuration - 1e-9)
          << label << " at t=" << t << ": foot " << f << " swing [" << liftOff << ", " << touchDown << ")";
    });
  };

  // The reference manager's activation: merge at the plan's own boundary; a stale plan or one that contradicts a swing in
  // flight at the merge point is dropped.
  const auto activateDuePlans = [&](int step, scalar_t t) {
    while (!pending.empty() && pending.front().first <= step) {
      const ContactPlan& due = pending.front().second;
      const scalar_t commitTime = std::max(t, due.committedUntil);
      const bool fresh = due.committedUntil >= t - 1e-6;
      if (fresh && planAgreesWithSwingsInFlight(applied, due, commitTime)) {
        const ModeSchedule planSchedule = due.toModeSchedule();
        applied = mergeModeSchedules(applied, planSchedule, commitTime, -10.0, t + 2.4);
        checkFutureSwings(planSchedule, commitTime, "plan", t);
        checkFutureSwings(applied, t, "merged", t);
      }
      pending.pop_front();
    }
  };

  for (int step = 0; step < steps; ++step) {
    const scalar_t t = tick * step;
    activateDuePlans(step, t);

    // One plan in flight at a time, like the planner module (it drops its snapshot until the previous plan is applied,
    // so every plan is made from a schedule that already contains the plan before it).
    if (pending.empty()) {
      in.time = t;
      in.contacts = contactFlagsAtTime(applied, t);
      in.committedUntil = commitBoundaryForSchedule(applied, t, c.commitTime);
      in.committedContacts = committedContactsForPlanner(applied, t, c.dt, c.numNodes - 1, in.committedUntil);
      in.committedPhaseStartTimes = committedPhaseStartsForPlanner(applied, t, c.dt, c.numNodes - 1, in.committedUntil);
      for (size_t f = 0; f < N_CONTACTS; ++f) {
        size_t idx = modeIndexAtTime(applied, t);
        scalar_t phaseStart = t - 10.0;
        while (idx > 0 && modeNumber2StanceLeg(applied.modeSequence[idx - 1])[f] == in.contacts[f]) --idx;
        if (idx > 0) phaseStart = applied.eventTimes[idx - 1];
        in.phaseElapsedTime[f] = std::max(0.0, t - phaseStart);
      }
      // Most recently swung foot, as the reference manager reports it (the alternation constraint spans plans).
      in.lastSwungFoot = -1;
      for (size_t i = modeIndexAtTime(applied, t); i > 0 && in.lastSwungFoot < 0; --i) {
        const contact_flag_t before = modeNumber2StanceLeg(applied.modeSequence[i - 1]);
        const contact_flag_t after = modeNumber2StanceLeg(applied.modeSequence[i]);
        for (size_t f = 0; f < N_CONTACTS; ++f) {
          if (before[f] && !after[f]) in.lastSwungFoot = static_cast<int>(f);
        }
      }
      ContactPlan plan = planner.plan(in);
      EXPECT_TRUE(plan.valid) << "step " << step;
      if (plan.valid) pending.emplace_back(step + latencyTicks, std::move(plan));
      if (latencyTicks == 0) activateDuePlans(step, t);
    }
    for (size_t m : applied.modeSequence) {
      const contact_flag_t flags = modeNumber2StanceLeg(m);
      bool anyContact = false;
      for (size_t i = 0; i < N_CONTACTS; ++i) anyContact = anyContact || flags[i];
      EXPECT_TRUE(anyContact) << "flight phase at step " << step;
    }
  }
  return applied;
}

void expectExecutedSwingsRespectTheMinimum(const ModeSchedule& applied, const ContactPlanningConfig& c, scalar_t walkEnd) {
  int executedSwings = 0;
  forEachSwing(applied, [&](size_t f, scalar_t liftOff, scalar_t touchDown) {
    if (liftOff < 0.0 || liftOff > walkEnd) return;
    ++executedSwings;
    EXPECT_GE(touchDown - liftOff, c.minSwingDuration - 1e-9)
        << "executed swing of foot " << f << " [" << liftOff << ", " << touchDown << ")";
  });
  EXPECT_GE(executedSwings, 3) << "the walk command must produce steps; schedule: " << applied;
}

/**
 * The executed steps alternate between the feet and every foot stands at least minContactDuration between two of its
 * swings: a foot that has just landed is never lifted again at once.
 */
void expectExecutedStepsAlternateWithFullStances(const ModeSchedule& applied, const ContactPlanningConfig& c, scalar_t walkEnd) {
  std::vector<std::tuple<scalar_t, scalar_t, size_t>> swings;  // (liftOff, touchDown, foot)
  forEachSwing(applied, [&](size_t f, scalar_t liftOff, scalar_t touchDown) {
    if (liftOff < 0.0 || liftOff > walkEnd) return;
    swings.emplace_back(liftOff, touchDown, f);
  });
  std::sort(swings.begin(), swings.end());
  for (size_t i = 1; i < swings.size(); ++i) {
    const auto& [liftOff, touchDown, foot] = swings[i];
    const auto& [previousLiftOff, previousTouchDown, previousFoot] = swings[i - 1];
    if (c.enforceAlternatingFeet) {
      EXPECT_NE(foot, previousFoot) << "foot " << foot << " swings twice in a row: [" << previousLiftOff << ", " << previousTouchDown
                                    << ") then [" << liftOff << ", " << touchDown << "); schedule: " << applied;
    }
  }
  for (size_t f = 0; f < N_CONTACTS; ++f) {
    std::optional<scalar_t> lastTouchDown;
    for (const auto& [liftOff, touchDown, foot] : swings) {
      if (foot != f) continue;
      if (lastTouchDown.has_value()) {
        EXPECT_GE(liftOff - *lastTouchDown, c.minContactDuration - 1e-9)
            << "foot " << f << " lifted again " << (liftOff - *lastTouchDown) << " s after landing at " << *lastTouchDown
            << "; schedule: " << applied;
      }
      lastTouchDown = touchDown;
    }
  }
}

ContactPlanningConfig recedingHorizonConfig() {
  ContactPlanningConfig c = makeConfig();
  c.commitTime = 0.3;
  c.minSwingDuration = 0.3;
  c.maxSwingDuration = 0.6;
  c.minContactDuration = 0.15;
  c.minDoubleSupportDuration = 0.1;
  c.validate();
  return c;
}
}  // namespace

TEST(LipContactPlannerTest, RecedingHorizonNeverExecutesASwingShorterThanTheMinimum) {
  // Plans applied as soon as they are computed. The planner used to lift a foot in the last nodes of its horizon where
  // a minimum-length swing does not fit; toModeSchedule() then closed the plan with a touch-down at endTime() and the
  // plan carried a 0.1 s swing at its tail.
  const ContactPlanningConfig c = recedingHorizonConfig();
  const ModeSchedule applied = walkRecedingHorizon(c, 0, 0.05, 60);
  expectExecutedSwingsRespectTheMinimum(applied, c, 0.05 * 60);
  expectExecutedStepsAlternateWithFullStances(applied, c, 0.05 * 60);
}

TEST(LipContactPlannerTest, RecedingHorizonWithPlannerLatencyNeverExecutesASwingShorterThanTheMinimum) {
  // Plans activated one to three solves after they were computed, as with the background planner. The reference manager
  // merges a plan at its own commit boundary and drops a plan whose boundary has passed or that contradicts a swing in
  // flight there. Two earlier policies failed this walk: merging at the boundary of the activating solve cut a lift-off
  // placed between the two boundaries short by the plan's age (0.28 s swings against a 0.3 s minimum), and shifting the
  // plan forward onto that boundary instead delayed a plan made just before a touch-down by a whole swing, so that the
  // foot that had just landed was lifted again at once and the robot was thrown.
  const ContactPlanningConfig c = recedingHorizonConfig();
  for (int latency : {1, 2, 3}) {
    SCOPED_TRACE("latency ticks: " + std::to_string(latency));
    const ModeSchedule applied = walkRecedingHorizon(c, latency, 0.05, 60);
    expectExecutedSwingsRespectTheMinimum(applied, c, 0.05 * 60);
    expectExecutedStepsAlternateWithFullStances(applied, c, 0.05 * 60);
  }
}

namespace {
ContactPlanningConfig headingConfig() {
  ContactPlanningConfig c = makeConfig();
  c.useAcomDynamics = true;
  // Model parameters, as deriveContactPlanningModelParameters() would set them for a ~100 kg biped.
  c.torsionalFrictionTorque = 50.0;
  c.doubleSupportYawCouple = 80.0;
  c.setSymmetricFootYawOffset(0.5);
  c.minDoubleSupportDuration = 0.1;
  c.validate();
  return c;
}

ContactPlannerInput turnInPlaceInput(scalar_t yawRateCommand) {
  ContactPlannerInput in = makeStandingInput();
  in.velocityCommand = vector2_t::Zero();
  in.heading = 0.0;
  in.yaw = 0.0;
  in.headingRate = 0.0;
  in.headingRateCommand = yawRateCommand;
  in.footYaws = makeFeetArray(0.0);
  in.yawInertia = 15.0;  // from the robot model in the reference manager
  return in;
}
}  // namespace

TEST(LipContactPlannerHeading, LayoutAppendsTheHeadingBlockAndLipModeIsUnchanged) {
  const LipContactPlanner::Layout lip = LipContactPlanner::makeLayout(makeConfig());
  EXPECT_FALSE(lip.hasHeading);
  EXPECT_EQ(lip.nx, LipContactPlanner::STATE_DIM);
  EXPECT_EQ(lip.nu, LipContactPlanner::INPUT_DIM);

  const LipContactPlanner::Layout heading = LipContactPlanner::makeLayout(headingConfig());
  EXPECT_TRUE(heading.hasHeading);
  EXPECT_EQ(heading.nx, LipContactPlanner::STATE_DIM + 2 + static_cast<int>(N_CONTACTS));
  EXPECT_EQ(heading.nu, LipContactPlanner::INPUT_DIM + 2 * static_cast<int>(N_CONTACTS));
  // The binaries keep their indices: the heading block is appended after the LIP inputs.
  EXPECT_EQ(heading.yawTorque0, LipContactPlanner::INPUT_DIM);
  EXPECT_GT(heading.yawTorque0, LipContactPlanner::CR);
  EXPECT_EQ(heading.footYaw(N_CONTACTS - 1), heading.nx - 1);
  EXPECT_EQ(heading.footYawDelta(N_CONTACTS - 1), heading.nu - 1);

  // A LIP plan carries no heading; the heading model needs the yaw inertia from the model and the derived parameters.
  LipContactPlanner lipPlanner(makeConfig());
  const ContactPlan lipPlan = lipPlanner.plan(makeStandingInput());
  ASSERT_TRUE(lipPlan.valid);
  EXPECT_FALSE(lipPlan.hasHeading());
  EXPECT_FALSE(lipPlan.headingAtTime(0.5).has_value());
  LipContactPlanner planner(headingConfig());
  ContactPlannerInput in = turnInPlaceInput(0.0);
  in.yawInertia = 0.0;
  EXPECT_FALSE(planner.plan(in).valid) << "no inertia: no plan, not a crash";
  in.yawInertia = 12.0;
  EXPECT_TRUE(planner.plan(in).valid);
  ContactPlanningConfig noParameters = makeConfig();
  noParameters.useAcomDynamics = true;
  EXPECT_FALSE(noParameters.hasModelParameters());
  LipContactPlanner unprepared(noParameters);
  EXPECT_FALSE(unprepared.plan(in).valid) << "without the model-derived limits the heading model refuses to plan";
}

TEST(LipContactPlannerHeading, TurnsInPlaceByStepping) {
  // Standing, no linear command, a yaw rate command: the LIP alone would stand still. With the heading model the
  // heading follows the command, every foot steps to keep up with the rotating frame, foot yaws stay pinned while the
  // foot stands and end up near the heading, and the yaw torques respect the ground limits.
  const ContactPlanningConfig c = headingConfig();
  LipContactPlanner planner(c);
  const ContactPlannerInput in = turnInPlaceInput(0.6);
  const ContactPlan plan = planner.plan(in);
  ASSERT_TRUE(plan.valid);
  ASSERT_TRUE(plan.hasHeading());
  ASSERT_EQ(plan.heading.size(), static_cast<size_t>(c.numNodes + 1));

  const PhaseStats analysis = analyse(plan);
  EXPECT_FALSE(analysis.flight);
  EXPECT_GT(analysis.numSwings[0] + analysis.numSwings[1], 0) << "turning has to be stepped";
  EXPECT_GT(plan.heading.back(), 0.25) << "the heading follows the commanded rate over the horizon";
  for (size_t k = 1; k < plan.heading.size(); ++k) {
    EXPECT_GE(plan.heading[k], plan.heading[k - 1] - 1e-6) << "monotone turn at node " << k;
  }
  // Foot yaw is pinned in contact and lands near the heading; every foot stays within hip range (soft, with margin).
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    for (int k = 0; k < c.numNodes; ++k) {
      if (plan.contacts[k][foot]) {
        EXPECT_NEAR(plan.footYaws[k + 1][foot], plan.footYaws[k][foot], 1e-6)
            << "foot " << foot << " yaw moved while in contact at node " << k;
      }
      EXPECT_LE(std::abs(plan.footYaws[k][foot] - plan.heading[k]), c.footYawOffsetBounds(foot).second + 0.05)
          << "foot " << foot << " node " << k;
    }
    EXPECT_GT(plan.footYaws.back()[foot], 0.1) << "foot " << foot << " turned with the heading";
  }
  // Ground torque limits from the solution: the QP inputs beyond the LIP block are the torques.
  const LipContactPlanner::Layout& L = planner.getLayout();
  const OcpQpSolution& sol = planner.getLastResult().solution;
  for (int k = 0; k < c.numNodes; ++k) {
    scalar_t total = 0.0;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      const scalar_t tau = sol.u[k](L.yawTorque(foot));
      const scalar_t limit = (plan.contacts[k][foot] ? c.torsionalFrictionTorque : 0.0) +
                             ((plan.contacts[k][0] && plan.contacts[k][1]) ? 0.5 * c.doubleSupportYawCouple : 0.0);
      EXPECT_LE(std::abs(tau), limit + 1e-6) << "foot " << foot << " node " << k;
      total += tau;
    }
    // The dynamics: omega_{k+1} - omega_k = dt * sum(tau) / I.
    EXPECT_NEAR(plan.headingRate[k + 1] - plan.headingRate[k], c.dt * total / in.yawInertia, 1e-6);
    EXPECT_NEAR(plan.heading[k + 1] - plan.heading[k], c.dt * plan.headingRate[k] + 0.5 * c.dt * c.dt * total / in.yawInertia, 1e-6);
  }
  EXPECT_GE(planner.getLastStatistics().numHeadingRelinearizations, 1);
  EXPECT_NEAR(plan.yaw, in.heading, 1e-12) << "the planning frame is the heading";
}

TEST(LipContactPlannerHeading, CannotTurnWithoutGroundTorque) {
  ContactPlanningConfig c = headingConfig();
  c.torsionalFrictionTorque = 0.0;
  c.doubleSupportYawCouple = 0.0;
  LipContactPlanner planner(c);
  const ContactPlan plan = planner.plan(turnInPlaceInput(0.6));
  ASSERT_TRUE(plan.valid);
  for (size_t k = 0; k < plan.heading.size(); ++k) {
    EXPECT_NEAR(plan.heading[k], 0.0, 1e-6) << "no ground torque, no turn, node " << k;
    EXPECT_NEAR(plan.headingRate[k], 0.0, 1e-6);
  }
}

TEST(LipContactPlannerHeading, RelinearisedFrameMatchesThePlannedHeading) {
  // After the re-linearisation the frame the step width is measured in is the plan's own heading: the lateral foot
  // separation in that frame sits near the nominal step width whenever both feet stand.
  ContactPlanningConfig c = headingConfig();
  c.headingLinearizationPasses = 2;
  LipContactPlanner planner(c);
  const ContactPlan plan = planner.plan(turnInPlaceInput(0.5));
  ASSERT_TRUE(plan.valid);
  EXPECT_EQ(planner.getLastStatistics().numHeadingRelinearizations, 2);
  for (int k = 0; k <= c.numNodes; ++k) {
    const bool bothDown =
        (k == c.numNodes) ? (plan.contacts[k - 1][0] && plan.contacts[k - 1][1]) : (plan.contacts[k][0] && plan.contacts[k][1]);
    if (!bothDown) continue;
    const vector2_t ey(-std::sin(plan.heading[k]), std::cos(plan.heading[k]));
    const scalar_t width = ey.dot(plan.footholds[k][0] - plan.footholds[k][1]);
    EXPECT_GE(width, c.minStepWidth - 0.02) << "node " << k;
    EXPECT_LE(width, c.maxStepWidth + 0.02) << "node " << k;
  }
  // The heading trajectory seeds the next plan's nominal frame (warm start), and shifting keeps it consistent.
  ContactPlannerInput next = turnInPlaceInput(0.5);
  next.time = 0.1;
  next.heading = plan.heading[1];
  next.headingRate = plan.headingRate[1];
  next.footYaws = plan.footYaws[1];
  next.contacts = plan.contacts[1];
  next.phaseElapsedTime = makeFeetArray(0.1);
  const LipContactPlanner::HeadingNominal nominal = planner.defaultNominal(next);
  ASSERT_EQ(nominal.heading.size(), static_cast<size_t>(c.numNodes + 1));
  EXPECT_NEAR(nominal.heading[0], plan.heading[1], 1e-12) << "the previous plan, shifted by one node, is the nominal";
}

TEST(LipContactPlannerTest, RecedingHorizonWarmStartKeepsPlanConsistent) {
  const ContactPlanningConfig config = makeConfig();
  LipContactPlanner planner(config);
  ContactPlannerInput input = makeStandingInput();
  input.velocityCommand = vector2_t(0.4, 0.0);
  const ContactPlan first = planner.plan(input);
  ASSERT_TRUE(first.valid);

  // Advance one node along the first plan and re-plan from there.
  ContactPlannerInput next = input;
  next.time += config.dt;
  next.comPosition = first.comPosition[1];
  next.comVelocity = first.comVelocity[1];
  next.footPositions[0] = first.footholds[1][0];
  next.footPositions[1] = first.footholds[1][1];
  next.contacts = first.contacts[1];
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    next.phaseElapsedTime[foot] = (first.contacts[1][foot] == first.contacts[0][foot]) ? input.phaseElapsedTime[foot] + config.dt : 0.0;
  }
  const ContactPlan second = planner.plan(next);
  printPlan(second);
  ASSERT_TRUE(second.valid);
  EXPECT_LE(second.objective, first.objective + 1e-6) << "the shifted incumbent bounds the re-planned objective";
  std::cout << "re-plan solved in " << second.solveTime * 1e3 << " ms with " << second.numBranchAndBoundNodes << " relaxations\n";
}

TEST(ContactPlanTest, ModeScheduleConversionAndMerge) {
  ContactPlan plan;
  plan.valid = true;
  plan.startTime = 1.0;
  plan.dt = 0.1;
  plan.contacts = {{true, true}, {true, false}, {true, false}, {true, true}, {false, true}, {false, true}};
  const ModeSchedule schedule = plan.toModeSchedule();
  ASSERT_EQ(schedule.modeSequence.size(), 5u);
  EXPECT_EQ(schedule.modeSequence[0], static_cast<size_t>(ModeNumber::STANCE));
  EXPECT_EQ(schedule.modeSequence[1], static_cast<size_t>(ModeNumber::LF));
  EXPECT_EQ(schedule.modeSequence[2], static_cast<size_t>(ModeNumber::STANCE));
  EXPECT_EQ(schedule.modeSequence[3], static_cast<size_t>(ModeNumber::RF));
  EXPECT_EQ(schedule.modeSequence[4], static_cast<size_t>(ModeNumber::STANCE));
  ASSERT_EQ(schedule.eventTimes.size(), 4u);
  EXPECT_NEAR(schedule.eventTimes[0], 1.1, 1e-9);
  EXPECT_NEAR(schedule.eventTimes[1], 1.3, 1e-9);
  EXPECT_NEAR(schedule.eventTimes[2], 1.4, 1e-9);
  EXPECT_NEAR(schedule.eventTimes[3], 1.6, 1e-9);
  EXPECT_EQ(schedule.modeAtTime(1.25), static_cast<size_t>(ModeNumber::LF));

  // Applied schedule: RF swing from 0.9 to 1.15, then stance. Merge at commit time 1.05: keep the applied RF phase, then
  // follow the plan from 1.05 on (which says LF from 1.1).
  const ModeSchedule applied({0.9, 1.15}, {ModeNumber::STANCE, ModeNumber::RF, ModeNumber::STANCE});
  const ModeSchedule merged = mergeModeSchedules(applied, schedule, 1.05, 0.0, 2.5);
  EXPECT_EQ(merged.modeAtTime(0.5), static_cast<size_t>(ModeNumber::STANCE));
  EXPECT_EQ(merged.modeAtTime(1.0), static_cast<size_t>(ModeNumber::RF));
  EXPECT_EQ(merged.modeAtTime(1.07), static_cast<size_t>(ModeNumber::STANCE));  // plan mode at the commit time
  EXPECT_EQ(merged.modeAtTime(1.2), static_cast<size_t>(ModeNumber::LF));
  EXPECT_EQ(merged.modeAtTime(1.5), static_cast<size_t>(ModeNumber::RF));
  EXPECT_EQ(merged.modeAtTime(2.4), static_cast<size_t>(ModeNumber::STANCE));
  EXPECT_EQ(merged.modeSequence.front(), static_cast<size_t>(ModeNumber::STANCE));
  EXPECT_EQ(merged.modeSequence.back(), static_cast<size_t>(ModeNumber::STANCE));
  for (size_t i = 1; i < merged.eventTimes.size(); ++i) {
    EXPECT_GT(merged.eventTimes[i], merged.eventTimes[i - 1]);
  }
  for (size_t i = 1; i < merged.modeSequence.size(); ++i) {
    EXPECT_NE(merged.modeSequence[i], merged.modeSequence[i - 1]);
  }
}

TEST(ContactPlanTest, AllStanceMergeKeepsOneEvent) {
  ContactPlan plan;
  plan.valid = true;
  plan.startTime = 1.0;
  plan.dt = 0.1;
  plan.contacts.assign(12, contact_flag_t{true, true});
  const ModeSchedule applied({0.5}, {ModeNumber::STANCE, ModeNumber::STANCE});
  const ModeSchedule merged = mergeModeSchedules(applied, plan.toModeSchedule(), 1.1, 0.0, 3.0);
  ASSERT_FALSE(merged.eventTimes.empty());
  ASSERT_EQ(merged.modeSequence.size(), merged.eventTimes.size() + 1);
  for (const size_t mode : merged.modeSequence) {
    EXPECT_EQ(mode, static_cast<size_t>(ModeNumber::STANCE));
  }
}

}  // namespace ocs2::humanoid
