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

#include <cmath>
#include <iostream>

#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
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
