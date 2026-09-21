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

#include <array>

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

#include "absl/log/log.h"

namespace ocs2::humanoid {

namespace {

constexpr scalar_t kSoftTol = 2e-2;  // the geometric constraints are soft; allow a small violation

ContactPlanningConfig makeConfig() {
  ContactPlanningConfig config;
  config.planner.dt = 0.1;
  config.planner.numNodes = 12;
  config.planner.commitTime = 0.0;
  config.shared.comHeight = 0.85;
  config.shared.gaitLimits.minSwingDuration = 0.3;
  config.shared.gaitLimits.maxSwingDuration = 0.5;
  config.shared.gaitLimits.minContactDuration = 0.15;
  config.shared.gaitLimits.maxContactDuration = 0.0;
  config.shared.gaitLimits.minDoubleSupportDuration = 0.1;
  config.planner.maxBranchAndBoundNodes = 3000;
  config.planner.maxSolveTime = 10.0;
  config.eventShiftLocalSearch.maxTime = 2.0;
  config.planner.verbose = false;
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
    xMin = std::min(xMin, p(0) - config.zmpSupportRegion.halfWidthX);
    xMax = std::max(xMax, p(0) + config.zmpSupportRegion.halfWidthX);
    yMin = std::min(yMin, p(1) - config.zmpSupportRegion.halfWidthY);
    yMax = std::max(yMax, p(1) + config.zmpSupportRegion.halfWidthY);
  }
  return z(0) >= xMin - kSoftTol && z(0) <= xMax + kSoftTol && z(1) >= yMin - kSoftTol && z(1) <= yMax + kSoftTol;
}

void printPlan(const ContactPlan& plan) {
  LOG(INFO) << "plan valid=" << plan.valid << " objective=" << plan.objective << " nodes=" << plan.numBranchAndBoundNodes
            << " time=" << plan.solveTime << " s optimal=" << plan.optimal << "\n  contacts: ";
  if (!plan.valid) {
    LOG(INFO) << "(none)\n";
    return;
  }
  for (const contact_flag_t& c : plan.contacts) LOG(INFO) << "[" << c[0] << c[1] << "]";
  LOG(INFO) << "\n";
  for (int k = 0; k <= plan.numIntervals(); ++k) {
    LOG(INFO) << "  k=" << k << " com=" << plan.comPosition[k].transpose() << " v=" << plan.comVelocity[k].transpose()
              << " pL=" << plan.footholds[k][0].transpose() << " pR=" << plan.footholds[k][1].transpose();
    if (k < plan.numIntervals()) LOG(INFO) << " zmp=" << plan.zmp[k].transpose();
    LOG(INFO) << "\n";
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
  config.planner.verbose = true;
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
    EXPECT_GE(separation(1), config.footSeparation.minStepWidth - kSoftTol) << "k=" << k;
    EXPECT_LE(separation(1), config.footSeparation.maxStepWidth + kSoftTol) << "k=" << k;
    EXPECT_LE(std::abs(separation(0)), config.footSeparation.maxStepLength + kSoftTol) << "k=" << k;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      const vector2_t rel = plan.footholds[k][foot] - plan.comPosition[k];
      EXPECT_LE(std::abs(rel(0)), config.reachability.reachX + kSoftTol) << "k=" << k;
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
  LOG(INFO) << "walking plan solved in " << plan.solveTime * 1e3 << " ms with " << plan.numBranchAndBoundNodes << " relaxations, "
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
  for (int k = 0; k < config.planner.numNodes; ++k) {
    sameFootTwice[LipContactPlanner::contactBinaryIndex(k, 0)] = 1;
    sameFootTwice[LipContactPlanner::contactBinaryIndex(k, 1)] = (k < 3 || k >= 6) ? 1 : 0;
  }
  EXPECT_FALSE(planner.propagate(input, sameFootTwice));
  MiqpAssignment alternating = planner.initialAssignment(input);
  for (int k = 0; k < config.planner.numNodes; ++k) {
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
  config.shared.gaitLimits.minDoubleSupportDuration = 0.2;  // 2 nodes
  LipContactPlanner planner(config);
  ContactPlannerInput input = makeStandingInput();
  input.contacts = {true, false};  // right foot swinging
  input.phaseElapsedTime = {1.0, 0.2};

  // Right lands at node 1; the left foot may not lift before node 3.
  MiqpAssignment tooEarly = planner.initialAssignment(input);
  for (int k = 0; k < config.planner.numNodes; ++k) {
    tooEarly[LipContactPlanner::contactBinaryIndex(k, 1)] = (k < 1) ? 0 : 1;
    tooEarly[LipContactPlanner::contactBinaryIndex(k, 0)] = (k == 2) ? 0 : 1;
  }
  EXPECT_FALSE(planner.propagate(input, tooEarly));
  MiqpAssignment lateEnough = planner.initialAssignment(input);
  for (int k = 0; k < config.planner.numNodes; ++k) {
    lateEnough[LipContactPlanner::contactBinaryIndex(k, 1)] = (k < 1) ? 0 : 1;
    lateEnough[LipContactPlanner::contactBinaryIndex(k, 0)] = (k >= 3 && k < 6) ? 0 : 1;
  }
  EXPECT_TRUE(planner.propagate(input, lateEnough));

  // Without a previous plan the assignment cost is the switch cost only.
  const scalar_t switches = planner.assignmentCost(input, lateEnough);
  EXPECT_NEAR(switches, 3.0 * config.contactSwitch.cost, 1e-12);  // right touch-down, left lift-off, left touch-down

  // After a plan exists, deviating from it is charged per node.
  const ContactPlan first = planner.plan(input);
  ASSERT_TRUE(first.valid);
  MiqpAssignment same(planner.getLastResult().assignment);
  EXPECT_NEAR(planner.assignmentCost(input, same), planner.getLastResult().incumbentObjective - planner.getLastResult().solution.objective,
              1e-9);
  MiqpAssignment flipped = same;
  const int node = config.planner.numNodes - 1;
  flipped[LipContactPlanner::contactBinaryIndex(node, 0)] = 1 - flipped[LipContactPlanner::contactBinaryIndex(node, 0)];
  EXPECT_GT(planner.assignmentCost(input, flipped), planner.assignmentCost(input, same) + 0.9 * config.planConsistency.cost);
}

TEST(LipContactPlannerTest, MinimumDoubleSupportForbidsASimultaneousSwitch) {
  // Left swings over nodes 1..3 and lands at node 4. If the right foot lifts at that same node there is no double
  // support at all: single support on the right turns into single support on the left in one node, which is exactly
  // the weight transfer minDoubleSupportDuration is meant to forbid. The propagation has to reject it just as it rejects
  // a lift-off one node after the landing.
  ContactPlanningConfig config = makeConfig();
  config.shared.gaitLimits.minDoubleSupportDuration = 0.2;  // 2 nodes
  LipContactPlanner planner(config);
  ContactPlannerInput input = makeStandingInput();  // both feet down for a long time

  const auto assignmentWithRightLiftOffAt = [&](int liftOffNode) {
    MiqpAssignment a = planner.initialAssignment(input);
    for (int k = 0; k < config.planner.numNodes; ++k) {
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
  for (int k = 4; k < config.planner.numNodes; ++k) partial[LipContactPlanner::contactBinaryIndex(k, 1)] = kMiqpFree;
  ASSERT_TRUE(planner.propagate(input, partial));
  EXPECT_EQ(partial[LipContactPlanner::contactBinaryIndex(4, 1)], 1) << "right foot must stay down at the landing node";
  EXPECT_EQ(partial[LipContactPlanner::contactBinaryIndex(5, 1)], 1) << "and for the second double-support node";
}

TEST(LipContactPlannerTest, ElapsedPhaseTimeIsRoundedConservativelyForBothLimits) {
  // dt 0.1, swing limits [0.3, 0.5]. A swing 0.16 s old used to count as 2 nodes (nearest), so ending it after one more
  // node (0.26 s in total) passed the 0.3 s minimum; a swing 0.44 s old counted as 4 nodes and could go on one more
  // node (0.54 s) against the 0.5 s maximum.
  ContactPlanningConfig config = makeConfig();
  config.shared.gaitLimits.minDoubleSupportDuration = 0.0;
  LipContactPlanner planner(config);

  ContactPlannerInput input = makeStandingInput();
  input.contacts = {true, false};  // right foot swinging
  input.phaseElapsedTime = {1.0, 0.16};
  const auto rightLandsAt = [&](int node) {
    MiqpAssignment a = planner.initialAssignment(input);
    for (int k = 0; k < config.planner.numNodes; ++k) {
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
  tight.shared.gaitLimits.minSwingDuration = 0.3;
  tight.shared.gaitLimits.maxSwingDuration = 0.3;
  tight.shared.gaitLimits.minDoubleSupportDuration = 0.0;
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
  config.formulation.setLogicRule(term::kAlternatingFeet,
                                  false);  // so that the right foot's minimum contact duration is what limits its re-lift
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
    for (int k = 0; k < config.planner.numNodes; ++k) {
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
  for (int k = 3; k < config.planner.numNodes; ++k) partial[LipContactPlanner::contactBinaryIndex(k, 0)] = kMiqpFree;
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
  config.shared.gaitLimits.maxContactDuration = 0.6;        // 6 nodes
  config.shared.gaitLimits.minDoubleSupportDuration = 0.1;  // 1 node
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
    for (int k = 0; k < config.planner.numNodes; ++k) {
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
      EXPECT_GE(touchDown - liftOff, c.shared.gaitLimits.minSwingDuration - 1e-9)
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
      in.committedUntil = commitBoundaryForSchedule(applied, t, c.planner.commitTime);
      in.committedContacts = committedContactsForPlanner(applied, t, c.planner.dt, c.planner.numNodes - 1, in.committedUntil);
      in.committedPhaseStartTimes = committedPhaseStartsForPlanner(applied, t, c.planner.dt, c.planner.numNodes - 1, in.committedUntil);
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
    EXPECT_GE(touchDown - liftOff, c.shared.gaitLimits.minSwingDuration - 1e-9)
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
    if (c.formulation.hasLogicRule(term::kAlternatingFeet)) {
      EXPECT_NE(foot, previousFoot) << "foot " << foot << " swings twice in a row: [" << previousLiftOff << ", " << previousTouchDown
                                    << ") then [" << liftOff << ", " << touchDown << "); schedule: " << applied;
    }
  }
  for (size_t f = 0; f < N_CONTACTS; ++f) {
    std::optional<scalar_t> lastTouchDown;
    for (const auto& [liftOff, touchDown, foot] : swings) {
      if (foot != f) continue;
      if (lastTouchDown.has_value()) {
        EXPECT_GE(liftOff - *lastTouchDown, c.shared.gaitLimits.minContactDuration - 1e-9)
            << "foot " << f << " lifted again " << (liftOff - *lastTouchDown) << " s after landing at " << *lastTouchDown
            << "; schedule: " << applied;
      }
      lastTouchDown = touchDown;
    }
  }
}

ContactPlanningConfig recedingHorizonConfig() {
  ContactPlanningConfig c = makeConfig();
  c.planner.commitTime = 0.3;
  c.shared.gaitLimits.minSwingDuration = 0.3;
  c.shared.gaitLimits.maxSwingDuration = 0.6;
  c.shared.gaitLimits.minContactDuration = 0.15;
  c.shared.gaitLimits.minDoubleSupportDuration = 0.1;
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
  c.setHeadingModel(true);
  // Model parameters, as deriveContactPlanningModelParameters() would set them for a ~100 kg biped.
  c.yawTorqueBudget.torsionalFrictionTorque = 50.0;
  c.yawTorqueBudget.doubleSupportYawCouple = 80.0;
  c.setSymmetricFootYawOffset(0.5);
  c.shared.gaitLimits.minDoubleSupportDuration = 0.1;
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
  noParameters.setHeadingModel(true);
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
  ASSERT_EQ(plan.heading.size(), static_cast<size_t>(c.planner.numNodes + 1));

  const PhaseStats analysis = analyse(plan);
  EXPECT_FALSE(analysis.flight);
  EXPECT_GT(analysis.numSwings[0] + analysis.numSwings[1], 0) << "turning has to be stepped";
  EXPECT_GT(plan.heading.back(), 0.25) << "the heading follows the commanded rate over the horizon";
  for (size_t k = 1; k < plan.heading.size(); ++k) {
    EXPECT_GE(plan.heading[k], plan.heading[k - 1] - 1e-6) << "monotone turn at node " << k;
  }
  // Foot yaw is pinned in contact and lands near the heading; every foot stays within hip range (soft, with margin).
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    for (int k = 0; k < c.planner.numNodes; ++k) {
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
  for (int k = 0; k < c.planner.numNodes; ++k) {
    scalar_t total = 0.0;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      const scalar_t tau = sol.u[k](L.yawTorque(foot));
      const bool bothDown = plan.contacts[k][0] && plan.contacts[k][1];
      const scalar_t limit = !plan.contacts[k][foot]
                                 ? 0.0
                                 : (bothDown ? 0.5 * (c.yawTorqueBudget.torsionalFrictionTorque + c.yawTorqueBudget.doubleSupportYawCouple)
                                             : c.yawTorqueBudget.torsionalFrictionTorque);
      EXPECT_LE(std::abs(tau), limit + 1e-6) << "foot " << foot << " node " << k;
      total += tau;
    }
    // The dynamics: omega_{k+1} - omega_k = dt * sum(tau) / I.
    EXPECT_NEAR(plan.headingRate[k + 1] - plan.headingRate[k], c.planner.dt * total / in.yawInertia, 1e-6);
    EXPECT_NEAR(plan.heading[k + 1] - plan.heading[k],
                c.planner.dt * plan.headingRate[k] + 0.5 * c.planner.dt * c.planner.dt * total / in.yawInertia, 1e-6);
  }
  EXPECT_GE(planner.getLastStatistics().numHeadingRelinearizations, 1);
  EXPECT_NEAR(plan.yaw, in.heading, 1e-12) << "the planning frame is the heading";
}

TEST(LipContactPlannerHeading, CannotTurnWithoutGroundTorque) {
  ContactPlanningConfig c = headingConfig();
  c.yawTorqueBudget.torsionalFrictionTorque = 0.0;
  c.yawTorqueBudget.doubleSupportYawCouple = 0.0;
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
  c.headingRelinearisation.passes = 2;
  LipContactPlanner planner(c);
  const ContactPlan plan = planner.plan(turnInPlaceInput(0.5));
  ASSERT_TRUE(plan.valid);
  EXPECT_EQ(planner.getLastStatistics().numHeadingRelinearizations, 2);
  for (int k = 0; k <= c.planner.numNodes; ++k) {
    const bool bothDown =
        (k == c.planner.numNodes) ? (plan.contacts[k - 1][0] && plan.contacts[k - 1][1]) : (plan.contacts[k][0] && plan.contacts[k][1]);
    if (!bothDown) continue;
    const vector2_t ey(-std::sin(plan.heading[k]), std::cos(plan.heading[k]));
    const scalar_t width = ey.dot(plan.footholds[k][0] - plan.footholds[k][1]);
    EXPECT_GE(width, c.footSeparation.minStepWidth - 0.02) << "node " << k;
    EXPECT_LE(width, c.footSeparation.maxStepWidth + 0.02) << "node " << k;
  }
  // The heading trajectory seeds the next plan's nominal frame (warm start): one node later the nominal is the previous
  // plan shifted by one node, its last node held. (Node 0 of the nominal is always the input heading, so the shift is
  // only visible from node 1 on, and only when the next input is timed on the plan's grid.)
  ContactPlannerInput next = turnInPlaceInput(0.5);
  next.time = plan.startTime + c.planner.dt;
  next.heading = plan.heading[1];
  next.headingRate = plan.headingRate[1];
  next.footYaws = plan.footYaws[1];
  next.contacts = plan.contacts[1];
  next.phaseElapsedTime = makeFeetArray(0.1);
  const LipContactPlanner::HeadingNominal nominal = planner.defaultNominal(next);
  ASSERT_EQ(nominal.heading.size(), static_cast<size_t>(c.planner.numNodes + 1));
  for (int k = 0; k < c.planner.numNodes; ++k) {
    EXPECT_NEAR(nominal.heading[k], plan.heading[k + 1], 1e-12) << "node " << k << ": the previous plan shifted by one node";
    EXPECT_NEAR((nominal.feet[k][0] - plan.footholds[k + 1][0]).norm(), 0.0, 1e-12) << "node " << k;
  }
  EXPECT_NEAR(nominal.heading[c.planner.numNodes], plan.heading[c.planner.numNodes], 1e-12) << "the last node is held";
  EXPECT_GT(std::abs(nominal.heading[c.planner.numNodes / 2] - (next.heading + next.headingRateCommand * 0.5 * c.horizon())), 1e-3)
      << "the warm start must differ from the commanded ramp, or this test could not tell the two apart";
  // Beyond the previous plan's horizon (or after a reset) the nominal falls back to the commanded ramp from the input
  // heading; an input between two nodes is rounded to the nearest node of the previous plan, not dropped.
  next.time = plan.startTime + c.horizon() + c.planner.dt;
  const LipContactPlanner::HeadingNominal ramp = planner.defaultNominal(next);
  for (int k = 0; k <= c.planner.numNodes; ++k) {
    EXPECT_NEAR(ramp.heading[k], next.heading + next.headingRateCommand * static_cast<scalar_t>(k) * c.planner.dt, 1e-12) << "node " << k;
  }
}

/**
 * The measured heading is wrapped to [-pi, pi]; the previous plan's heading trajectory lives on whatever branch that plan
 * started on. Crossing +-pi between two plans, the warm-started nominal must follow the measurement's branch: on the old
 * branch every first-order frame term g (theta - theta_n) was worth g 2 pi and the foot yaw tracking aimed a full turn
 * away, so the plan after the crossing was garbage.
 */
TEST(LipContactPlannerHeading, WarmStartNominalFollowsTheBranchOfTheMeasuredHeadingAcrossTheWrap) {
  const ContactPlanningConfig c = headingConfig();
  LipContactPlanner planner(c);
  // Turning left just below +pi.
  ContactPlannerInput first = turnInPlaceInput(0.6);
  first.heading = M_PI - 0.05;
  first.yaw = first.heading;
  first.footYaws = makeFeetArray(first.heading);
  const ContactPlan before = planner.plan(first);
  ASSERT_TRUE(before.valid);
  ASSERT_GT(before.heading.back(), M_PI) << "the first plan turns through +pi on its own branch";

  // One node later the measurement has wrapped to just above -pi.
  ContactPlannerInput second = first;
  second.time = first.time + c.planner.dt;
  second.heading = -M_PI + 0.02;
  second.yaw = second.heading;
  second.footYaws = makeFeetArray(second.heading);
  const LipContactPlanner::HeadingNominal nominal = planner.defaultNominal(second);
  for (int k = 0; k <= c.planner.numNodes; ++k) {
    EXPECT_LT(std::abs(nominal.heading[k] - second.heading), M_PI) << "node " << k << " on the measurement's branch";
    if (k > 0) EXPECT_LT(std::abs(nominal.heading[k] - nominal.heading[k - 1]), 0.5) << "node " << k << " continuous";
  }
  EXPECT_NEAR(std::remainder(nominal.heading[0] - before.heading[1], 2.0 * M_PI), 0.0, 1e-9) << "the same trajectory, shifted by 2 pi";

  const ContactPlan after = planner.plan(second);
  ASSERT_TRUE(after.valid);
  EXPECT_NEAR(after.heading.front(), second.heading, 1e-9);
  for (int k = 0; k <= c.planner.numNodes; ++k) {
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      EXPECT_LE(std::abs(after.footYaws[k][foot] - after.heading[k]), c.footYawOffsetBounds(foot).second + 0.05)
          << "node " << k << " foot " << foot << ": the feet stay within hip range of the heading, no full turn";
    }
  }
  // The plan still turns the commanded way and continues the plan before the wrap (shifted by 2 pi, one node on).
  EXPECT_GT(after.heading.back() - after.heading.front(), 0.2);
  for (int k = 0; k <= c.planner.numNodes; ++k) {
    EXPECT_NEAR(after.heading[k], before.heading[std::min(k + 1, c.planner.numNodes)] - 2.0 * M_PI, 0.3) << "node " << k;
  }
  // Its geometry is as good as a fresh solve of the same input and as the plan before the wrap: the step-width rows are
  // soft and a turn in place bends them by a few centimetres, but a nominal on the wrong branch bent them by a metre.
  const auto worstWidthViolation = [&](const ContactPlan& plan) {
    scalar_t worst = 0.0;
    for (int k = 0; k <= c.planner.numNodes; ++k) {
      const bool bothDown =
          (k == c.planner.numNodes) ? (plan.contacts[k - 1][0] && plan.contacts[k - 1][1]) : (plan.contacts[k][0] && plan.contacts[k][1]);
      if (!bothDown) continue;
      const vector2_t ey(-std::sin(plan.heading[k]), std::cos(plan.heading[k]));
      const scalar_t width = ey.dot(plan.footholds[k][0] - plan.footholds[k][1]);
      worst = std::max({worst, c.footSeparation.minStepWidth - width, width - c.footSeparation.maxStepWidth});
    }
    return worst;
  };
  LipContactPlanner fresh(c);
  const ContactPlan freshAfter = fresh.plan(second);
  ASSERT_TRUE(freshAfter.valid);
  EXPECT_LE(worstWidthViolation(after), worstWidthViolation(freshAfter) + 0.01) << "no worse than a solve without the warm start";
  EXPECT_LE(worstWidthViolation(after), worstWidthViolation(before) + 0.02) << "no worse than the plan before the wrap";
  EXPECT_LT(worstWidthViolation(after), 0.1) << "and nowhere near the metre a wrong-branch frame term produced";
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
  next.time += config.planner.dt;
  next.comPosition = first.comPosition[1];
  next.comVelocity = first.comVelocity[1];
  next.footPositions[0] = first.footholds[1][0];
  next.footPositions[1] = first.footholds[1][1];
  next.contacts = first.contacts[1];
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    next.phaseElapsedTime[foot] =
        (first.contacts[1][foot] == first.contacts[0][foot]) ? input.phaseElapsedTime[foot] + config.planner.dt : 0.0;
  }
  const ContactPlan second = planner.plan(next);
  printPlan(second);
  ASSERT_TRUE(second.valid);
  // From the state the first plan predicted, the re-plan is the first plan shifted by one node: the same contacts over
  // the part of the horizon both plans cover, and the footholds where the feet stand within a few centimetres (the
  // horizon gained a node and the pull towards the previous footholds is soft). The QP objectives are not comparable:
  // the residual constants are dropped and the second problem carries previous-foothold terms the first lacks.
  for (int k = 0; k + 1 < config.planner.numNodes; ++k) {
    EXPECT_EQ(second.contacts[k], first.contacts[k + 1]) << "node " << k;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      if (!first.contacts[k + 1][foot]) continue;
      EXPECT_NEAR((second.footholds[k][foot] - first.footholds[k + 1][foot]).norm(), 0.0, 0.05) << "node " << k << " foot " << foot;
    }
  }
  LOG(INFO) << "re-plan solved in " << second.solveTime * 1e3 << " ms with " << second.numBranchAndBoundNodes << " relaxations\n";
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

/*
 * Physics of the ACoM-LIP model, pinned to closed forms computed independently of the planner: the exact zero-order-hold
 * solution of the linear inverted pendulum, the capture-point propagation the terminal cost relies on, the orbital
 * energy invariant, the heading double integrator with its torque budget, and the heading frame the footholds live in.
 */
namespace {

/// Exact solution of xddot = omega^2 (x - z) with z held constant over dt, from x0, v0.
std::pair<scalar_t, scalar_t> lipClosedForm(scalar_t x0, scalar_t v0, scalar_t z, scalar_t omega, scalar_t dt) {
  const scalar_t ch = std::cosh(omega * dt);
  const scalar_t sh = std::sinh(omega * dt);
  return {z + (x0 - z) * ch + (v0 / omega) * sh, (x0 - z) * omega * sh + v0 * ch};
}

scalar_t orbitalEnergy(scalar_t x, scalar_t v, scalar_t z, scalar_t omega) {
  return 0.5 * v * v - 0.5 * omega * omega * (x - z) * (x - z);
}

}  // namespace

TEST(LipContactPlannerModel, LipTransitionIsTheExactZeroOrderHoldSolution) {
  const ContactPlanningConfig c = makeConfig();
  const scalar_t omega = c.omega();
  EXPECT_NEAR(omega, std::sqrt(c.shared.gravity / c.shared.comHeight), 1e-12);
  const scalar_t ch = std::cosh(omega * c.planner.dt);
  const scalar_t sh = std::sinh(omega * c.planner.dt);

  LipContactPlanner planner(c);
  const OcpQpProblem problem = planner.buildProblem(makeStandingInput());
  ASSERT_EQ(problem.stages.size(), static_cast<size_t>(c.planner.numNodes + 1));
  using P = LipContactPlanner;
  for (int k = 0; k < c.planner.numNodes; ++k) {
    const OcpQpStage& s = problem.stages[k];
    for (int axis = 0; axis < 2; ++axis) {
      EXPECT_NEAR(s.A(P::CX + axis, P::CX + axis), ch, 1e-12) << "node " << k;
      EXPECT_NEAR(s.A(P::CX + axis, P::VX + axis), sh / omega, 1e-12);
      EXPECT_NEAR(s.A(P::VX + axis, P::CX + axis), omega * sh, 1e-12);
      EXPECT_NEAR(s.A(P::VX + axis, P::VX + axis), ch, 1e-12);
      EXPECT_NEAR(s.B(P::CX + axis, P::ZX + axis), 1.0 - ch, 1e-12);
      EXPECT_NEAR(s.B(P::VX + axis, P::ZX + axis), -omega * sh, 1e-12);
      // No cross-axis coupling and no coupling into the foothold states.
      EXPECT_NEAR(s.A(P::CX + axis, P::CX + 1 - axis), 0.0, 1e-12);
      EXPECT_NEAR(s.A(P::PLX + axis, P::CX + axis), 0.0, 1e-12);
    }
  }
  // The matrices reproduce the analytic solution of the pendulum for arbitrary states, not just a linearisation of it.
  const OcpQpStage& s0 = problem.stages[0];
  const std::array<std::array<scalar_t, 3>, 3> cases{{{0.03, 0.4, -0.05}, {-0.2, -0.1, 0.1}, {0.0, 0.0, 0.08}}};
  for (const auto& [x0, v0, z] : cases) {
    const auto [x1, v1] = lipClosedForm(x0, v0, z, omega, c.planner.dt);
    EXPECT_NEAR(s0.A(P::CX, P::CX) * x0 + s0.A(P::CX, P::VX) * v0 + s0.B(P::CX, P::ZX) * z, x1, 1e-12);
    EXPECT_NEAR(s0.A(P::VX, P::CX) * x0 + s0.A(P::VX, P::VX) * v0 + s0.B(P::VX, P::ZX) * z, v1, 1e-12);
  }
}

TEST(LipContactPlannerModel, DecodedPlanObeysTheLipDynamicsDcmPropagationAndEnergyInvariant) {
  const ContactPlanningConfig c = makeConfig();
  const scalar_t omega = c.omega();
  LipContactPlanner planner(c);
  ContactPlannerInput input = makeStandingInput();
  input.velocityCommand = vector2_t(0.4, 0.0);
  const ContactPlan plan = planner.plan(input);
  ASSERT_TRUE(plan.valid);
  ASSERT_EQ(plan.comPosition.size(), static_cast<size_t>(c.planner.numNodes + 1));
  ASSERT_GT(analyse(plan).numSwings[0] + analyse(plan).numSwings[1], 0) << "a walking plan, so the dynamics are exercised";

  for (int k = 0; k < c.planner.numNodes; ++k) {
    for (int axis = 0; axis < 2; ++axis) {
      const scalar_t x0 = plan.comPosition[k](axis), v0 = plan.comVelocity[k](axis), z = plan.zmp[k](axis);
      const auto [x1, v1] = lipClosedForm(x0, v0, z, omega, c.planner.dt);
      EXPECT_NEAR(plan.comPosition[k + 1](axis), x1, 1e-6) << "node " << k << " axis " << axis;
      EXPECT_NEAR(plan.comVelocity[k + 1](axis), v1, 1e-6) << "node " << k << " axis " << axis;
      // Capture point xi = x + v / omega: its offset from the (constant) ZMP grows exactly by e^{omega dt} per node,
      // which is the identity the terminal capturability residual and its exp(2 omega dt) weight scaling rest on.
      const scalar_t xi0 = x0 + v0 / omega, xi1 = plan.comPosition[k + 1](axis) + plan.comVelocity[k + 1](axis) / omega;
      EXPECT_NEAR(xi1 - z, std::exp(omega * c.planner.dt) * (xi0 - z), 1e-6) << "node " << k << " axis " << axis;
      // Orbital energy is conserved over a node: the ZMP does no work while it is held.
      EXPECT_NEAR(orbitalEnergy(plan.comPosition[k + 1](axis), plan.comVelocity[k + 1](axis), z, omega), orbitalEnergy(x0, v0, z, omega),
                  1e-6)
          << "node " << k << " axis " << axis;
    }
  }
}

TEST(LipContactPlannerModel, HeadingBlockIsTheExactDoubleIntegratorScaledByTheYawInertia) {
  const ContactPlanningConfig c = headingConfig();
  LipContactPlanner planner(c);
  const LipContactPlanner::Layout& L = planner.getLayout();
  for (const scalar_t inertia : {15.0, 30.0}) {
    ContactPlannerInput in = turnInPlaceInput(0.6);
    in.yawInertia = inertia;
    const OcpQpProblem problem = planner.buildProblem(in);
    for (int k = 0; k < c.planner.numNodes; ++k) {
      const OcpQpStage& s = problem.stages[k];
      // theta_{k+1} = theta_k + dt * rate_k + 0.5 dt^2 tau / I ; rate_{k+1} = rate_k + dt tau / I  (zero-order hold).
      EXPECT_NEAR(s.A(L.heading, L.heading), 1.0, 1e-12);
      EXPECT_NEAR(s.A(L.heading, L.headingRate), c.planner.dt, 1e-12);
      EXPECT_NEAR(s.A(L.headingRate, L.headingRate), 1.0, 1e-12);
      EXPECT_NEAR(s.A(L.headingRate, L.heading), 0.0, 1e-12);
      for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
        EXPECT_NEAR(s.B(L.heading, L.yawTorque(foot)), 0.5 * c.planner.dt * c.planner.dt / inertia, 1e-12) << "I=" << inertia;
        EXPECT_NEAR(s.B(L.headingRate, L.yawTorque(foot)), c.planner.dt / inertia, 1e-12) << "I=" << inertia;
        // A foot's yaw is a pure integrator of its own displacement input.
        EXPECT_NEAR(s.B(L.footYaw(foot), L.footYawDelta(foot)), 1.0, 1e-12);
      }
      // The heading is not coupled to the translational LIP state.
      EXPECT_NEAR(s.A(L.heading, LipContactPlanner::CX), 0.0, 1e-12);
      EXPECT_NEAR(s.A(LipContactPlanner::VX, L.headingRate), 0.0, 1e-12);
      // Yaw torque box: the most one foot ever carries, the whole weight's torsion alone or half the double-support budget.
      const scalar_t share = std::max(c.yawTorqueBudget.torsionalFrictionTorque,
                                      0.5 * (c.yawTorqueBudget.torsionalFrictionTorque + c.yawTorqueBudget.doubleSupportYawCouple));
      for (size_t i = 0; i < s.idxbu.size(); ++i) {
        for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
          if (s.idxbu[i] != L.yawTorque(foot)) continue;
          EXPECT_NEAR(s.lbu(i), -share, 1e-12);
          EXPECT_NEAR(s.ubu(i), share, 1e-12);
        }
      }
    }
  }
}

/**
 * The yaw torque one foot may carry under every contact state, read off the planner's own constraint rows: with the
 * torques at zero and the contact binaries set, the slack of the torque rows is the budget. T_t is the torsional
 * friction of the whole weight on one foot, T_c the friction couple of two feet: alone a foot has T_t, in double support
 * each foot has (T_t + T_c) / 2 so that the pair has T_t + T_c, and a swinging foot has nothing. The pair's budget was
 * 2 T_t + T_c before, i.e. both feet carrying the whole weight at once.
 */
TEST(LipContactPlannerModel, YawTorqueBudgetPerContactStateIsTheGroundsTorsionAndCouple) {
  const ContactPlanningConfig c = headingConfig();
  LipContactPlanner planner(c);
  const LipContactPlanner::Layout& L = planner.getLayout();
  const OcpQpProblem problem = planner.buildProblem(turnInPlaceInput(0.6));
  const OcpQpStage& s = problem.stages[0];
  const scalar_t Tt = c.yawTorqueBudget.torsionalFrictionTorque, Tc = c.yawTorqueBudget.doubleSupportYawCouple;
  ASSERT_GT(Tc, Tt) << "the fixture must tell the double-support share (T_t + T_c) / 2 from the old T_t + T_c / 2";

  // Budget of `foot` given the contact state: the tightest upper bound the torque rows leave for +tau (the -tau rows are
  // the mirror image).
  const auto budget = [&](size_t foot, bool left, bool right) {
    vector_t u = vector_t::Zero(s.numInputs());
    u(LipContactPlanner::CL) = left ? 1.0 : 0.0;
    u(LipContactPlanner::CR) = right ? 1.0 : 0.0;
    scalar_t bound = std::numeric_limits<scalar_t>::infinity();
    for (int i = 0; i < s.numGeneralConstraints(); ++i) {
      const scalar_t coefficient = s.D(i, L.yawTorque(foot));
      if (coefficient <= 0.0) continue;  // not a +tau row of this foot
      bound = std::min(bound, (s.ug(i) - s.D.row(i).dot(u)) / coefficient);
    }
    return bound;
  };
  EXPECT_NEAR(budget(0, true, false), Tt, 1e-9) << "alone: the whole weight's torsion";
  EXPECT_NEAR(budget(1, false, true), Tt, 1e-9);
  EXPECT_NEAR(budget(0, false, true), 0.0, 1e-9) << "swinging: nothing";
  EXPECT_NEAR(budget(1, true, false), 0.0, 1e-9);
  EXPECT_NEAR(budget(0, true, true), 0.5 * (Tt + Tc), 1e-9) << "double support: half the weight's torsion and half the couple";
  EXPECT_NEAR(budget(1, true, true), 0.5 * (Tt + Tc), 1e-9);
  EXPECT_NEAR(budget(0, true, true) + budget(1, true, true), Tt + Tc, 1e-9) << "the pair has the ground's budget, not 2 T_t + T_c";
  // A relaxed double support (both binaries at one half) is bounded by the single-support share of half a foot.
  {
    vector_t u = vector_t::Zero(s.numInputs());
    u(LipContactPlanner::CL) = 0.5;
    u(LipContactPlanner::CR) = 0.5;
    scalar_t bound = std::numeric_limits<scalar_t>::infinity();
    for (int i = 0; i < s.numGeneralConstraints(); ++i) {
      if (s.D(i, L.yawTorque(0)) <= 0.0) continue;
      bound = std::min(bound, (s.ug(i) - s.D.row(i).dot(u)) / s.D(i, L.yawTorque(0)));
    }
    EXPECT_NEAR(bound, 0.5 * Tt, 1e-9);
  }
}

TEST(LipContactPlannerModel, YawTorqueIsCarriedOnlyByStanceFeetWithinTheGroundBudget) {
  const ContactPlanningConfig c = headingConfig();
  LipContactPlanner planner(c);
  const ContactPlannerInput in = turnInPlaceInput(0.6);
  const ContactPlan plan = planner.plan(in);
  ASSERT_TRUE(plan.valid);
  const LipContactPlanner::Layout& L = planner.getLayout();
  const OcpQpSolution& sol = planner.getLastResult().solution;
  int singleSupportNodes = 0, doubleSupportNodes = 0;
  for (int k = 0; k < c.planner.numNodes; ++k) {
    const bool both = plan.contacts[k][0] && plan.contacts[k][1];
    scalar_t sumAbs = 0.0;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      const scalar_t tau = std::abs(sol.u[k](L.yawTorque(foot)));
      sumAbs += tau;
      if (!plan.contacts[k][foot]) {
        EXPECT_LE(tau, 1e-9) << "a swinging foot carries no ground torque, node " << k << " foot " << foot;
      } else if (both) {
        EXPECT_LE(tau, 0.5 * (c.yawTorqueBudget.torsionalFrictionTorque + c.yawTorqueBudget.doubleSupportYawCouple) + 1e-9) << "node " << k;
      } else {
        EXPECT_LE(tau, c.yawTorqueBudget.torsionalFrictionTorque + 1e-9) << "single support, node " << k;
      }
    }
    if (both) {
      ++doubleSupportNodes;
      EXPECT_LE(sumAbs, c.yawTorqueBudget.torsionalFrictionTorque + c.yawTorqueBudget.doubleSupportYawCouple + 1e-9) << "node " << k;
    } else {
      ++singleSupportNodes;
    }
    // The heading rate integrates exactly the torque the feet carry.
    scalar_t total = 0.0;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) total += sol.u[k](L.yawTorque(foot));
    EXPECT_NEAR(plan.headingRate[k + 1], plan.headingRate[k] + c.planner.dt * total / in.yawInertia, 1e-6);
  }
  EXPECT_GT(singleSupportNodes, 0) << "turning in place must step, so both regimes are exercised";
  EXPECT_GT(doubleSupportNodes, 0);
}

TEST(LipContactPlannerModel, FootholdGeometryLivesInTheHeadingFrame) {
  // Facing +y (heading pi/2) and commanded to walk along it: the step width must be realised along world -x, the
  // footholds must advance along world +y, and the ZMP must stay in the support box measured along the heading.
  const ContactPlanningConfig c = headingConfig();
  LipContactPlanner planner(c);
  const scalar_t heading = M_PI / 2.0;
  const vector2_t ex(std::cos(heading), std::sin(heading));   // (0, 1)
  const vector2_t ey(-std::sin(heading), std::cos(heading));  // (-1, 0)
  ContactPlannerInput in = turnInPlaceInput(0.0);
  in.heading = heading;
  in.yaw = heading;
  in.footYaws = makeFeetArray(heading);
  in.footPositions[0] = 0.5 * c.stepWidth.nominalStepWidth * ey;   // left foot on the heading's left
  in.footPositions[1] = -0.5 * c.stepWidth.nominalStepWidth * ey;  // right foot on its right
  in.velocityCommand = 0.4 * ex;                                   // forward along the heading
  const ContactPlan plan = planner.plan(in);
  ASSERT_TRUE(plan.valid);
  ASSERT_TRUE(plan.hasHeading());
  ASSERT_GT(analyse(plan).numSwings[0] + analyse(plan).numSwings[1], 0) << "walking, so footholds move";

  for (int k = 0; k <= c.planner.numNodes; ++k) {
    EXPECT_NEAR(plan.heading[k], heading, 0.05) << "no yaw command: the heading holds, node " << k;
    const bool bothDown =
        (k == c.planner.numNodes) ? (plan.contacts[k - 1][0] && plan.contacts[k - 1][1]) : (plan.contacts[k][0] && plan.contacts[k][1]);
    const vector2_t d = plan.footholds[k][0] - plan.footholds[k][1];
    if (bothDown) {
      EXPECT_GE(ey.dot(d), c.footSeparation.minStepWidth - kSoftTol) << "node " << k;
      EXPECT_LE(ey.dot(d), c.footSeparation.maxStepWidth + kSoftTol) << "node " << k;
      EXPECT_LE(std::abs(ex.dot(d)), c.footSeparation.maxStepLength + kSoftTol) << "node " << k;
    }
    // The lateral separation shows up on world x, the forward progress on world y: the frame really is rotated.
    EXPECT_LT(d(0), 0.0) << "left foot at smaller world x when facing +y, node " << k;
  }
  const scalar_t forwardStart = ex.dot(0.5 * (plan.footholds.front()[0] + plan.footholds.front()[1]));
  const scalar_t forwardEnd = ex.dot(0.5 * (plan.footholds.back()[0] + plan.footholds.back()[1]));
  EXPECT_GT(forwardEnd - forwardStart, 0.1) << "the feet advance along the heading";
  EXPECT_GT(ex.dot(plan.comVelocity.back()), 0.2) << "the CoM moves along the heading";
  for (int k = 0; k < c.planner.numNodes; ++k) {
    // ZMP inside the box measured along the heading around the midpoint of the stance feet.
    vector2_t mid = vector2_t::Zero();
    int n = 0;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      if (plan.contacts[k][foot]) {
        mid += plan.footholds[k][foot];
        ++n;
      }
    }
    ASSERT_GT(n, 0);
    mid /= static_cast<scalar_t>(n);
    EXPECT_LE(std::abs(ex.dot(plan.zmp[k] - mid)), c.zmpSupportRegion.halfWidthX + kSoftTol) << "node " << k;
  }
}

/**
 * The foot yaw tracking term is quadratic in the foot yaw alone, with the nominal heading as its target: a stance foot's
 * yaw is pinned, so a term in (psi_i - theta) was a penalty on the heading state that pulled it back towards the stance
 * feet. With equal weights and both feet down the per-node minimiser was (theta_cmd + psi_L + psi_R) / 3.
 */
TEST(LipContactPlannerModel, FootYawTrackingTargetsTheNominalHeadingAndLeavesTheHeadingStateAlone) {
  const ContactPlanningConfig c = headingConfig();
  ASSERT_GT(c.footYawTracking.weight, 0.0);
  LipContactPlanner planner(c);
  const LipContactPlanner::Layout& L = planner.getLayout();
  const ContactPlannerInput in = turnInPlaceInput(0.6);

  // With a hand-set nominal the linear term on every foot yaw is the nominal heading of that node, nothing else.
  LipContactPlanner::HeadingNominal nominal = planner.defaultNominal(in);
  for (int k = 0; k <= c.planner.numNodes; ++k) nominal.heading[k] = 0.3 + 0.01 * static_cast<scalar_t>(k);
  const OcpQpProblem problem = planner.buildProblem(in, nominal);
  for (int k = 0; k <= c.planner.numNodes; ++k) {
    const OcpQpStage& s = problem.stages[k];
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      EXPECT_NEAR(s.Q(L.footYaw(foot), L.heading), 0.0, 1e-12) << "node " << k << ": no coupling of the foot yaw into the heading";
      EXPECT_NEAR(s.Q(L.heading, L.footYaw(foot)), 0.0, 1e-12) << "node " << k;
      // The diagonal carries the tracking weight (plus the solver's 1e-8 state regularisation).
      EXPECT_NEAR(s.Q(L.footYaw(foot), L.footYaw(foot)), 2.0 * c.footYawTracking.weight, 1e-6) << "node " << k;
      EXPECT_NEAR(s.q(L.footYaw(foot)), -2.0 * c.footYawTracking.weight * nominal.heading[k], 1e-12) << "node " << k;
    }
    // (The heading's own diagonal is not the tracking weight alone: the linearised step-width and reach rows put their
    // first-order heading terms there as well, which is expected.)
  }

  // Turning in place with an ample torque budget: the heading gets close to the commanded ramp over the horizon. Under
  // the old coupling the stance feet, pinned at zero yaw, held the heading back well below it.
  const ContactPlan plan = planner.plan(in);
  ASSERT_TRUE(plan.valid);
  const scalar_t commanded = in.heading + in.headingRateCommand * c.horizon();
  EXPECT_GT(plan.heading.back(), 0.85 * commanded) << "commanded " << commanded << " reached " << plan.heading.back();
  // Whatever the heading did, the feet never step further than the hip range allows (soft rows, with margin).
  for (int k = 0; k <= c.planner.numNodes; ++k) {
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      EXPECT_LE(std::abs(plan.footYaws[k][foot] - plan.heading[k]), c.footYawOffsetBounds(foot).second + 0.05) << "node " << k;
    }
  }
}

/**
 * The ZMP support rows evaluated on hand-set states: in single support the ZMP is confined to the box of the stance
 * foot (the other foot's box is switched off by the big M), in double support to the box of half-width r_x around the
 * midpoint along the heading and to the lateral hull between the feet. The rows are soft; here their violation is read
 * directly.
 */
TEST(LipContactPlannerModel, ZmpRowsSelectTheSupportOfTheContactState) {
  ContactPlanningConfig c = makeConfig();
  c.zmpSupportRegion.halfWidthX = 0.08;
  c.zmpSupportRegion.halfWidthY = 0.05;
  c.validate();
  LipContactPlanner planner(c);
  const ContactPlannerInput in = makeStandingInput();  // yaw 0: the heading frame is the world frame
  const OcpQpProblem problem = planner.buildProblem(in);
  const OcpQpStage& s = problem.stages[0];
  using P = LipContactPlanner;
  const vector2_t pL = in.footPositions[0], pR = in.footPositions[1];
  ASSERT_GT(pL(1), pR(1)) << "left foot on the left";

  // Largest violation over the soft rows that involve the ZMP for the given ZMP and contact state.
  const auto zmpViolation = [&](const vector2_t& zmp, bool left, bool right) {
    vector_t x = vector_t::Zero(s.numStates());
    x.segment<2>(P::PLX) = pL;
    x.segment<2>(P::PRX) = pR;
    vector_t u = vector_t::Zero(s.numInputs());
    u.segment<2>(P::ZX) = zmp;
    u(P::CL) = left ? 1.0 : 0.0;
    u(P::CR) = right ? 1.0 : 0.0;
    const vector_t value = s.C * x + s.D * u;
    scalar_t violation = -std::numeric_limits<scalar_t>::infinity();
    for (const int i : s.softGeneralIndices) {
      if (s.D(i, P::ZX) == 0.0 && s.D(i, P::ZX + 1) == 0.0) continue;
      violation = std::max({violation, value(i) - s.ug(i), s.lg(i) - value(i)});
    }
    return violation;
  };
  const scalar_t rx = c.zmpSupportRegion.halfWidthX, ry = c.zmpSupportRegion.halfWidthY;

  // Single support on the left foot: the ZMP may reach the edge of the left box, not beyond, and not the right foot.
  EXPECT_NEAR(zmpViolation(pL + vector2_t(rx, 0.0), true, false), 0.0, 1e-12) << "on the edge of the stance box";
  EXPECT_NEAR(zmpViolation(pL + vector2_t(rx + 0.01, 0.0), true, false), 0.01, 1e-12) << "past the edge along the heading";
  EXPECT_NEAR(zmpViolation(pL + vector2_t(0.0, -ry - 0.02), true, false), 0.02, 1e-12) << "past the edge laterally";
  EXPECT_NEAR(zmpViolation(pR, true, false), (pL(1) - pR(1)) - ry, 1e-12) << "under the swinging foot";
  // And mirrored on the right.
  EXPECT_NEAR(zmpViolation(pR + vector2_t(-rx, 0.0), false, true), 0.0, 1e-12);
  EXPECT_NEAR(zmpViolation(pL, false, true), (pL(1) - pR(1)) - ry, 1e-12);

  // Double support: a box of half-width r_x around the midpoint along the heading, the hull between the feet laterally.
  const vector2_t mid = 0.5 * (pL + pR);
  // At the midpoint the heading box binds first: the lateral hull leaves half the step width plus r_y on either side.
  EXPECT_NEAR(zmpViolation(mid, true, true), -rx, 1e-12) << "the midpoint has the full margin of the heading box";
  EXPECT_GT(0.5 * (pL(1) - pR(1)) + ry, rx) << "(sanity: the fixture's lateral margin is the larger one)";
  EXPECT_NEAR(zmpViolation(mid + vector2_t(rx + 0.01, 0.0), true, true), 0.01, 1e-12);
  EXPECT_LE(zmpViolation(pL, true, true), 1e-12) << "under the left foot: inside the lateral hull";
  EXPECT_LE(zmpViolation(pR, true, true), 1e-12) << "under the right foot: inside the lateral hull";
  EXPECT_NEAR(zmpViolation(vector2_t(0.0, pL(1) + ry + 0.03), true, true), 0.03, 1e-12) << "beyond the left foot's edge";
  EXPECT_NEAR(zmpViolation(vector2_t(0.0, pR(1) - ry - 0.03), true, true), 0.03, 1e-12) << "beyond the right foot's edge";
  // The single-support boxes are switched off in double support: a ZMP under the right foot is fine although it is
  // outside the left box (and vice versa), which the previous two checks already rely on.
  EXPECT_GT(zmpViolation(pR, true, false), 0.1);
}

/**
 * The kinematic rows (reachability, foot separation, hip range) involve the state alone and must bind at the terminal
 * node too: the foothold of a swing that ends at the horizon is a state of node N (its last displacement is an input of
 * node N-1), and it was left without any constraint when the terminal node was skipped along with its inputs.
 */
TEST(LipContactPlannerModel, TerminalNodeCarriesTheKinematicRows) {
  using P = LipContactPlanner;
  for (const bool heading : {false, true}) {
    const ContactPlanningConfig c = heading ? headingConfig() : makeConfig();
    LipContactPlanner planner(c);
    const ContactPlannerInput in = heading ? turnInPlaceInput(0.0) : makeStandingInput();  // yaw 0: the frame is the world frame
    const OcpQpProblem problem = planner.buildProblem(in);
    const OcpQpStage& terminal = problem.stages.back();
    const OcpQpStage& running = problem.stages[problem.numStages() - 1];
    ASSERT_EQ(terminal.numInputs(), 0);
    // Reachability (2 feet x 2 axes) and separation (2 axes), plus the hip range of both feet with the heading model.
    const int expected = 4 + 2 + (heading ? 2 : 0);
    EXPECT_EQ(terminal.numGeneralConstraints(), expected) << "heading " << heading;
    EXPECT_EQ(static_cast<int>(terminal.softGeneralIndices.size()), expected) << "all of them soft";
    // They are exactly the state-only rows of the last running node: same state coefficients, same bounds.
    for (int i = 0; i < terminal.numGeneralConstraints(); ++i) {
      bool found = false;
      for (int j = 0; j < running.numGeneralConstraints() && !found; ++j) {
        found = running.D.row(j).isZero() && running.C.row(j).isApprox(terminal.C.row(i), 1e-12) &&
                std::abs(running.lg(j) - terminal.lg(i)) < 1e-12 && std::abs(running.ug(j) - terminal.ug(i)) < 1e-12;
      }
      EXPECT_TRUE(found) << "terminal row " << i << " has no state-only twin on the last running node (heading " << heading << ")";
    }
    // Evaluated on hand-set terminal states.
    const auto violation = [&](const vector_t& x) {
      const vector_t value = terminal.C * x;
      scalar_t worst = -std::numeric_limits<scalar_t>::infinity();
      for (int i = 0; i < terminal.numGeneralConstraints(); ++i) {
        worst = std::max({worst, value(i) - terminal.ug(i), terminal.lg(i) - value(i)});
      }
      return worst;
    };
    vector_t x = vector_t::Zero(terminal.numStates());
    x.segment<2>(P::PLX) = vector2_t(0.0, 0.5 * c.stepWidth.nominalStepWidth);
    x.segment<2>(P::PRX) = vector2_t(0.0, -0.5 * c.stepWidth.nominalStepWidth);
    EXPECT_LE(violation(x), 1e-12) << "the nominal stance is feasible";
    x(P::PLX) = c.reachability.reachX + 0.1;
    EXPECT_NEAR(violation(x), std::max(0.1, c.reachability.reachX + 0.1 - c.footSeparation.maxStepLength), 1e-12)
        << "a landing beyond reach is caught at node N";
    x(P::PLX) = 0.0;
    // Feet 3 cm inside the self-collision margin, symmetric about the CoM (which may also put them inside reachYInner).
    const scalar_t half = 0.5 * (c.footSeparation.minStepWidth - 0.03);
    x(P::PLY) = half;
    x(P::PRY) = -half;
    EXPECT_NEAR(violation(x), std::max(0.03, c.reachability.reachYInner - half), 1e-12) << "feet too close are caught at node N";
  }

  // On a real plan the terminal footholds obey the bounds (soft rows, with margin), whichever foot is in the air at the end.
  ContactPlanningConfig c = makeConfig();
  c.reachability.reachX = 0.35;
  c.footSeparation.maxStepLength = 0.6;
  c.validate();
  LipContactPlanner planner(c);
  ContactPlannerInput in = makeStandingInput();
  in.velocityCommand = vector2_t(0.6, 0.0);
  const ContactPlan plan = planner.plan(in);
  ASSERT_TRUE(plan.valid);
  const int N = c.planner.numNodes;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    EXPECT_LE(std::abs(plan.footholds[N][foot](0) - plan.comPosition[N](0)), c.reachability.reachX + kSoftTol) << "foot " << foot;
  }
  EXPECT_LE(std::abs(plan.footholds[N][0](0) - plan.footholds[N][1](0)), c.footSeparation.maxStepLength + kSoftTol);
  EXPECT_GE(plan.footholds[N][0](1) - plan.footholds[N][1](1), c.footSeparation.minStepWidth - kSoftTol);
}

/**
 * The duration limits in whole planner nodes, pinned for the DRC Atlas grid (0.1 s nodes): the quotients 0.7 / 0.1,
 * 0.15 / 0.1 and 0.3 / 0.1 are not exact in floating point and the epsilons must absorb that.
 */
TEST(LipContactPlannerModel, NodeConversionsRoundConservativelyOnTheLiveGrid) {
  ContactPlanningConfig c = makeConfig();
  c.planner.dt = 0.1;
  c.shared.gaitLimits.minSwingDuration = 0.4;
  c.shared.gaitLimits.maxSwingDuration = 0.7;
  c.shared.gaitLimits.minContactDuration = 0.15;
  c.shared.gaitLimits.maxContactDuration = 0.0;
  c.shared.gaitLimits.minDoubleSupportDuration = 0.1;
  c.planner.commitTime = 0.3;
  c.validate();
  EXPECT_EQ(c.minSwingNodes(), 4);
  EXPECT_EQ(c.maxSwingNodes(), 7);
  EXPECT_EQ(c.minContactNodes(), 2) << "0.15 s is rounded up to two whole nodes";
  EXPECT_EQ(c.maxContactNodes(), 0) << "0 disables the maximum";
  EXPECT_EQ(c.minDoubleSupportNodes(), 1);
  EXPECT_EQ(c.commitNodes(), 3);
  EXPECT_NEAR(c.horizon(), 1.2, 1e-12);
  // A maximum below one node is raised to the minimum; a zero minimum double support disables it.
  c.shared.gaitLimits.maxSwingDuration = 0.05;
  EXPECT_EQ(c.maxSwingNodes(), c.minSwingNodes());
  c.shared.gaitLimits.minDoubleSupportDuration = 0.0;
  EXPECT_EQ(c.minDoubleSupportNodes(), 0);
}

/**
 * The warm start survives a change of weights or limits (a hot reload of the task file while the operator tunes) and
 * is dropped only when the grid or the decision variables change. Dropping it on every reload made the next plan start
 * from scratch and move footholds and timing abruptly at every edit.
 */
TEST(LipContactPlannerTest, ChangingWeightsKeepsTheWarmStartChangingTheGridDropsIt) {
  const ContactPlanningConfig c = headingConfig();
  LipContactPlanner planner(c);
  const ContactPlan plan = planner.plan(turnInPlaceInput(0.5));
  ASSERT_TRUE(plan.valid);
  ContactPlannerInput next = turnInPlaceInput(0.5);
  next.time = plan.startTime + c.planner.dt;
  next.heading = plan.heading[1];
  next.headingRate = plan.headingRate[1];
  next.footYaws = plan.footYaws[1];
  next.contacts = plan.contacts[1];
  next.phaseElapsedTime = makeFeetArray(0.1);
  const scalar_t ramp = next.heading + next.headingRateCommand * c.planner.dt;
  ASSERT_GT(std::abs(plan.heading[2] - ramp), 1e-6) << "the warm start must be distinguishable from the commanded ramp";

  ContactPlanningConfig retuned = c;
  retuned.headingTracking.weight *= 2.0;
  retuned.footholdRegularization.weight *= 0.5;
  planner.setConfig(retuned);
  EXPECT_NEAR(planner.defaultNominal(next).heading[1], plan.heading[2], 1e-12) << "the previous plan still seeds the nominal";

  ContactPlanningConfig regridded = retuned;
  regridded.planner.numNodes += 1;
  regridded.validate();
  planner.setConfig(regridded);
  EXPECT_NEAR(planner.defaultNominal(next).heading[1], ramp, 1e-12) << "a new grid starts from the commanded ramp";
}

TEST(LipContactPlannerTest, APlanWithoutAnIncumbentIsNotReportedOptimal) {
  const ContactPlanningConfig c = makeConfig();
  LipContactPlanner planner(c);
  ContactPlannerInput in = makeStandingInput();
  // A committed prefix with both feet in the air breaks the no-flight rule: the root propagation is infeasible, the
  // search is "exhausted" at once, and there is no plan to call optimal.
  in.committedUntil = in.time + c.planner.dt;
  in.committedContacts = {makeFeetArray(false)};
  const ContactPlan plan = planner.plan(in);
  EXPECT_FALSE(plan.valid);
  EXPECT_FALSE(plan.optimal);
}

/**
 * The alternation rule constrains the plan's own decisions, not the executed history it is handed as the committed
 * prefix: a foot that lifted twice there (a re-timed event, an earlier plan) used to make every plan infeasible,
 * silently, until the node left the window. From the prefix on the rule applies: the next foot to lift is the other one.
 */
TEST(LipContactPlannerTest, ARepeatedLiftOffInTheCommittedPrefixDoesNotMakeThePlanInfeasible) {
  const ContactPlanningConfig c = makeConfig();
  LipContactPlanner planner(c);
  ContactPlannerInput in = makeStandingInput();
  in.velocityCommand = vector2_t(0.3, 0.0);
  in.contacts = {false, true};  // the left foot is in the air now
  in.phaseElapsedTime = {0.05, 5.0};
  // Left: in the air over nodes 0-2, down over nodes 3-4, in the air again from node 5, the right foot down throughout.
  const contact_flag_t leftUp{false, true}, bothDown{true, true};
  in.committedContacts = {leftUp, leftUp, leftUp, bothDown, bothDown, leftUp, leftUp};
  in.committedUntil = in.time + 7.0 * c.planner.dt;
  const ContactPlan plan = planner.plan(in);
  ASSERT_TRUE(plan.valid) << "the committed prefix is not the plan's to judge";
  for (int k = 0; k < 7; ++k) EXPECT_EQ(plan.contacts[k], in.committedContacts[k]) << "node " << k;
  int lastSwung = 0;  // the left foot lifted last inside the prefix
  for (int k = 7; k < c.planner.numNodes; ++k) {
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      if (plan.contacts[k - 1][foot] && !plan.contacts[k][foot]) {
        EXPECT_NE(static_cast<int>(foot), lastSwung) << "foot " << foot << " lifted twice in a row at node " << k;
        lastSwung = static_cast<int>(foot);
      }
    }
  }
}

/**
 * `double_support_penalty` is the incentive that `minDoubleSupportDuration: 0` only permits. With the hold at zero and
 * the term absent the planner still keeps a double support while walking, because it buys the ZMP the freedom the
 * support-region rows charge for; the term prices that freedom and the plan exchanges support in a single node
 * instead (a touch-down at the very node the other foot lifts).
 */
TEST(LipContactPlannerTest, DoubleSupportPenaltyRemovesTheTransitionDoubleSupport) {
  const auto planAt = [](scalar_t doubleSupportCost) {
    ContactPlanningConfig config = makeConfig();
    config.shared.gaitLimits.minContactDuration = 0.1;
    config.shared.gaitLimits.minDoubleSupportDuration = 0.0;  // the exchange is admissible
    config.contactSwitch.cost = 0.3;
    config.doubleSupportPenalty.cost = doubleSupportCost;
    config.formulation.assignmentCosts = {term::kContactSwitch, term::kPlanConsistency, term::kDoubleSupportPenalty};
    config.validate();
    LipContactPlanner planner(config);
    ContactPlannerInput input = makeStandingInput();
    input.velocityCommand = vector2_t(0.6, 0.0);
    input.comVelocity = vector2_t(0.6, 0.0);
    return planner.plan(input);
  };
  // Double-support nodes, and exchanges of support that happen in a single node (single support on one foot becomes
  // single support on the other with no double-support node between them).
  const auto count = [](const ContactPlan& plan) {
    int doubleSupportNodes = 0, exchanges = 0;
    for (const contact_flag_t& c : plan.contacts) {
      if (c[0] && c[1]) ++doubleSupportNodes;
    }
    for (int k = 1; k < plan.numIntervals(); ++k) {
      const contact_flag_t& previous = plan.contacts[k - 1];
      const contact_flag_t& current = plan.contacts[k];
      const bool bothSingle = previous[0] != previous[1] && current[0] != current[1];
      if (bothSingle && previous[0] != current[0]) ++exchanges;
    }
    return std::make_pair(doubleSupportNodes, exchanges);
  };

  const ContactPlan without = planAt(0.0);
  ASSERT_TRUE(without.valid);
  const auto [doubleSupportWithout, exchangesWithout] = count(without);
  EXPECT_EQ(exchangesWithout, 0) << "permission alone does not buy the exchange: " << without.describe();
  EXPECT_GE(doubleSupportWithout, 2) << without.describe();

  const ContactPlan with = planAt(0.1);
  ASSERT_TRUE(with.valid);
  const auto [doubleSupportWith, exchangesWith] = count(with);
  EXPECT_GT(exchangesWith, 0) << "the penalty should buy at least one single-node exchange: " << with.describe();
  EXPECT_LT(doubleSupportWith, doubleSupportWithout) << with.describe();
}

/**
 * The penalty cannot tell the weight transfer of a step apart from standing on two feet, so it prices standing too: a
 * foot lifted and put back down costs only the switch cost, while standing pays the penalty at every node of the
 * horizon. Above roughly 0.2 with these gait limits that trade flips and the robot marches at a zero velocity command
 * (at 5.0 it never puts both feet down at all). This pins the shipped default on the right side of it; the cost is
 * taken from the configuration default rather than written here, so that raising that default fails this test.
 */
TEST(LipContactPlannerTest, DoubleSupportPenaltyLeavesStandingAlone) {
  ContactPlanningConfig config = makeConfig();
  config.shared.gaitLimits.minContactDuration = 0.1;
  config.shared.gaitLimits.minDoubleSupportDuration = 0.0;
  config.contactSwitch.cost = 0.3;
  config.formulation.assignmentCosts = {term::kContactSwitch, term::kPlanConsistency, term::kDoubleSupportPenalty};
  config.validate();
  ASSERT_LT(config.doubleSupportPenalty.cost, 0.2) << "the shipped default is above the measured marching threshold";

  LipContactPlanner planner(config);
  ContactPlannerInput input = makeStandingInput();  // both feet down for a long time, no velocity command
  const ContactPlan plan = planner.plan(input);
  ASSERT_TRUE(plan.valid);

  for (int k = 0; k < plan.numIntervals(); ++k) {
    EXPECT_TRUE(plan.contacts[k][0] && plan.contacts[k][1])
        << "a standing robot stepped at node " << k << " to dodge the double support penalty: " << plan.describe();
  }
}

}  // namespace ocs2::humanoid
