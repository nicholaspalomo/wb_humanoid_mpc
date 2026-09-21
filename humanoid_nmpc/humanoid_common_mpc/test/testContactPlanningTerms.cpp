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

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningTermFactory.h"
#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"
#include "humanoid_common_mpc/contact_planning/LipContactPlanner.h"
#include "humanoid_common_mpc/contact_planning/MixedIntegerOcpQp.h"
#include "humanoid_common_mpc/contact_planning/OcpQpHpipm.h"
#include "humanoid_common_mpc/contact_planning/cost/PreviousFootholdConsistencyCost.h"
#include "humanoid_common_mpc/contact_planning/cost/StepLengthCost.h"
#include "humanoid_common_mpc/contact_planning/cost/StepWidthCost.h"
#include "humanoid_common_mpc/contact_planning/cost/TerminalDcmCost.h"
#include "humanoid_common_mpc/contact_planning/cost/VelocityTrackingCost.h"
#include "humanoid_common_mpc/contact_planning/execution/ExecutionContext.h"
#include "humanoid_common_mpc/contact_planning/execution/ScheduleAdaptationPipeline.h"
#include "humanoid_common_mpc/contact_planning/logic/ContactLogicState.h"
#include "humanoid_common_mpc/contact_planning/logic/ContactSwitchCost.h"
#include "humanoid_common_mpc/contact_planning/logic/DoubleSupportPenaltyCost.h"
#include "humanoid_common_mpc/contact_planning/logic/PlanConsistencyCost.h"
#include "humanoid_common_mpc/contact_planning/model/LipBlockIndices.h"
#include "humanoid_common_mpc/contact_planning/problem/AssignmentCost.h"
#include "humanoid_common_mpc/contact_planning/problem/ContactPlanningContext.h"
#include "humanoid_common_mpc/contact_planning/problem/ContactPlanningTerm.h"
#include "humanoid_common_mpc/contact_planning/problem/Layout.h"
#include "humanoid_common_mpc/contact_planning/problem/LipCost.h"
#include "humanoid_common_mpc/contact_planning/problem/StageAccumulator.h"
#include "humanoid_common_mpc/contact_planning/search/CadenceStretchStage.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::humanoid {

namespace {

ContactPlanningConfig makeConfig() {
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

ContactPlanningConfig makeHeadingConfig() {
  ContactPlanningConfig config = makeConfig();
  config.setHeadingModel(true);
  config.yawTorqueBudget.torsionalFrictionTorque = 20.0;
  config.yawTorqueBudget.doubleSupportYawCouple = 40.0;
  config.setSymmetricFootYawOffset(0.5);
  config.validate();
  return config;
}

ContactPlannerInput makeStandingInput() {
  ContactPlannerInput input;
  input.time = 3.0;
  input.footPositions[0] = vector2_t(0.0, 0.125);
  input.footPositions[1] = vector2_t(0.0, -0.125);
  input.contacts = {true, true};
  input.phaseElapsedTime = {5.0, 5.0};
  input.yawInertia = 10.0;
  return input;
}

ContactPlannerInput makeWalkingInput() {
  ContactPlannerInput input = makeStandingInput();
  input.velocityCommand = vector2_t(0.4, 0.0);
  return input;
}

}  // namespace

/*============================================ layout and rows ============================================*/

TEST(ContactPlanningTerms, LayoutIsComposedFromTheBlocksInOrder) {
  const Layout lip = LipContactPlanner::makeLayout(makeConfig());
  EXPECT_EQ(lip.nx, LIP_STATE_DIM);
  EXPECT_EQ(lip.nu, LIP_INPUT_DIM);
  EXPECT_FALSE(lip.hasHeading);
  EXPECT_EQ(lip.state(var::kComX), LIP_CX);
  EXPECT_EQ(lip.state(var::footX(1)), LIP_PRX);
  EXPECT_EQ(lip.input(var::contact(0)), LIP_CL);
  EXPECT_EQ(lip.input(var::contact(1)), LIP_CR);
  EXPECT_NE(lip.describe().find("x = [c_x c_y v_x v_y p_Lx p_Ly p_Rx p_Ry] (8)"), std::string::npos) << lip.describe();

  const Layout heading = LipContactPlanner::makeLayout(makeHeadingConfig());
  EXPECT_EQ(heading.nx, LIP_STATE_DIM + 2 + static_cast<int>(N_CONTACTS));
  EXPECT_EQ(heading.nu, LIP_INPUT_DIM + 2 * static_cast<int>(N_CONTACTS));
  EXPECT_TRUE(heading.hasHeading);
  EXPECT_EQ(heading.heading, LIP_STATE_DIM) << "the heading block appends, the LIP indices are unchanged";
  EXPECT_EQ(heading.state(var::kComX), LIP_CX);
  EXPECT_EQ(heading.input(var::contact(1)), LIP_CR);
  EXPECT_EQ(heading.yawTorque(0), LIP_INPUT_DIM);
  EXPECT_EQ(heading.footYawDelta(1), LIP_INPUT_DIM + 3);
}

TEST(ContactPlanningTerms, RowCountsPerNodeAreTheDocumentedOnes) {
  {
    LipContactPlanner planner(makeConfig());
    const OcpQpProblem problem = planner.buildProblem(makeStandingInput());
    EXPECT_EQ(problem.stages.front().numGeneralConstraints(), 27) << "1 no-flight + 12 support + 8 foot motion + 4 reach + 2 separation";
    EXPECT_EQ(problem.stages.back().numGeneralConstraints(), 6) << "the terminal node carries the kinematic rows only";
    EXPECT_EQ(static_cast<int>(problem.stages.front().softGeneralIndices.size()), 18);
    EXPECT_EQ(static_cast<int>(problem.stages.front().idxbu.size()), 6);
  }
  {
    LipContactPlanner planner(makeHeadingConfig());
    const OcpQpProblem problem = planner.buildProblem(makeStandingInput());
    EXPECT_EQ(problem.stages.front().numGeneralConstraints(), 37) << "+ 4 torque budget + 4 yaw pin + 2 hip range";
    EXPECT_EQ(problem.stages.back().numGeneralConstraints(), 8);
    EXPECT_EQ(static_cast<int>(problem.stages.front().idxbu.size()), 10);
  }
}

TEST(ContactPlanningTerms, TermsAreAssembledInListOrderAndOnTheirNodeSets) {
  ContactPlanningConfig config = makeConfig();
  config.formulation.costs = {term::kRegularization, term::kVelocityTracking};
  config.formulation.softConstraints = {};
  config.formulation.hardConstraints = {term::kNoFlight};
  config.validate();
  LipContactPlanner planner(config);
  const ContactPlanningProblem& problem = planner.getProblem();
  ASSERT_EQ(problem.costs.size(), 2u);
  EXPECT_EQ(problem.costs.nameAt(0), term::kRegularization);
  EXPECT_EQ(problem.costs.nameAt(1), term::kVelocityTracking);
  EXPECT_EQ(problem.costs.get(term::kVelocityTracking).nodeSet(), NodeSet::ALL);
  const OcpQpProblem qp = planner.buildProblem(makeWalkingInput());
  EXPECT_EQ(qp.stages.front().numGeneralConstraints(), 1);
  EXPECT_EQ(qp.stages.back().numGeneralConstraints(), 0);
  // The velocity tracking cost acts on the terminal node too: q_N carries -2 w v_cmd on the velocity entries.
  EXPECT_NEAR(qp.stages.back().q(LIP_VX), -2.0 * config.velocityTracking.weight * 0.4, 1e-12);
  EXPECT_NEAR(qp.stages.back().Q(LIP_CX, LIP_CX), config.regularization.state, 1e-15) << "only the regularisation touches the CoM";
}

// step_length: w ||dp - d_nom (1 - c)||^2 per foot and axis on the running nodes. The residual is affine in dp and the
// relaxed binary c, so the stage carries the cross terms of the two: R(dp, dp) = 2w, R(dp, c) = 2w d, R(c, c) = 2w d^2,
// r(dp) = -2w d, r(c) = -2w d^2 (0.5 u'Ru + r'u convention), summed over both axes for the c entries.
TEST(ContactPlanningTerms, StepLengthCostTiesTheSwingDisplacementToTheCommandedSpeed) {
  ContactPlanningConfig config = makeConfig();
  config.formulation.costs = {term::kStepLength};
  config.formulation.softConstraints = {};
  config.formulation.hardConstraints = {};
  config.stepLength.weight = 20.0;
  config.validate();
  LipContactPlanner planner(config);
  const auto& cost = planner.getProblem().costs.get<StepLengthCost>(term::kStepLength);
  EXPECT_EQ(cost.nodeSet(), NodeSet::RUNNING);
  // T_stride / T_swing = 2 (0.3 + 0.1) / 0.3
  EXPECT_NEAR(cost.strideToSwingRatio(), 8.0 / 3.0, 1e-12);
  const vector2_t d = cost.nominalDisplacementPerNode(vector2_t(0.4, 0.0), 0.1);
  EXPECT_NEAR(d.x(), 0.4 * 0.1 * 8.0 / 3.0, 1e-12);
  EXPECT_NEAR(d.y(), 0.0, 1e-12);

  const scalar_t w = config.stepLength.weight;
  const OcpQpProblem qp = planner.buildProblem(makeWalkingInput());
  const OcpQpStage& stage = qp.stages.front();
  EXPECT_NEAR(stage.R(LIP_DLX, LIP_DLX), 2.0 * w, 1e-12);
  EXPECT_NEAR(stage.R(LIP_DLY, LIP_DLY), 2.0 * w, 1e-12);
  EXPECT_NEAR(stage.R(LIP_DLX, LIP_CL), 2.0 * w * d.x(), 1e-12);
  EXPECT_NEAR(stage.R(LIP_CL, LIP_DLX), 2.0 * w * d.x(), 1e-12);
  EXPECT_NEAR(stage.R(LIP_DLY, LIP_CL), 0.0, 1e-12) << "no lateral command, no lateral nominal";
  EXPECT_NEAR(stage.R(LIP_CL, LIP_CL), 2.0 * w * d.squaredNorm(), 1e-12);
  EXPECT_NEAR(stage.r(LIP_DLX), -2.0 * w * d.x(), 1e-12);
  EXPECT_NEAR(stage.r(LIP_CL), -2.0 * w * d.squaredNorm(), 1e-12);
  EXPECT_NEAR(stage.R(LIP_DRX, LIP_CR), 2.0 * w * d.x(), 1e-12) << "both feet";
  EXPECT_NEAR(stage.R(LIP_DLX, LIP_CR), 0.0, 1e-12) << "no coupling across feet";
  EXPECT_TRUE(stage.Q.isZero()) << "the term touches no state";
  EXPECT_EQ(qp.stages.back().numInputs(), 0) << "not on the terminal node";

  // Standing still: the nominal is zero and the term is a plain regulariser of the foot displacement.
  const OcpQpProblem standing = planner.buildProblem(makeStandingInput());
  EXPECT_NEAR(standing.stages.front().R(LIP_DLX, LIP_DLX), 2.0 * w, 1e-12);
  EXPECT_NEAR(standing.stages.front().R(LIP_DLX, LIP_CL), 0.0, 1e-12);
  EXPECT_NEAR(standing.stages.front().r(LIP_CL), 0.0, 1e-12);
  EXPECT_NE(cost.describe().find("d_nom = v_cmd dt T_stride / T_swing"), std::string::npos);
}

// terminal_dcm.trackCommandedVelocity moves the target of the DCM from the last ZMP (rest) to zmp + v_cmd / omega (a CoM
// over the foot that keeps moving at the commanded velocity): only the linear term of the last running node changes.
TEST(ContactPlanningTerms, TerminalDcmCanTrackTheCommandedVelocityInsteadOfComingToRest) {
  ContactPlanningConfig rest = makeConfig();
  rest.formulation.costs = {term::kTerminalDcm};
  rest.formulation.softConstraints = {};
  rest.formulation.hardConstraints = {};
  rest.validate();
  ContactPlanningConfig walking = rest;
  walking.terminalDcm.trackCommandedVelocity = true;
  LipContactPlanner restPlanner(rest);
  LipContactPlanner walkingPlanner(walking);
  EXPECT_FALSE(restPlanner.getProblem().costs.get<TerminalDcmCost>(term::kTerminalDcm).tracksCommandedVelocity());
  EXPECT_TRUE(walkingPlanner.getProblem().costs.get<TerminalDcmCost>(term::kTerminalDcm).tracksCommandedVelocity());
  EXPECT_NE(walkingPlanner.getProblem().costs.get(term::kTerminalDcm).describe().find("v_cmd / omega"), std::string::npos);

  const ContactPlannerInput input = makeWalkingInput();  // v_cmd = (0.4, 0)
  const OcpQpProblem qpRest = restPlanner.buildProblem(input);
  const OcpQpProblem qpWalking = walkingPlanner.buildProblem(input);
  // ContactPlanningProblem::assemble allocates numNodes + 1 stages: the running nodes 0 ... numNodes - 1, which carry
  // inputs and dynamics, and the terminal stage numNodes, which carries neither. `last` below is therefore the index of
  // the last RUNNING node - the single node NodeSet::LAST_RUNNING selects - and stages.back() is the terminal stage.
  // The assertion pins that convention, so that a change of the stage allocation cannot silently turn the indices used
  // in the rest of this test into something else.
  ASSERT_EQ(qpWalking.stages.size(), static_cast<size_t>(rest.planner.numNodes) + 1u);
  const size_t last = static_cast<size_t>(rest.planner.numNodes - 1);
  const scalar_t omega = std::sqrt(rest.shared.gravity / rest.shared.comHeight);
  const scalar_t gain = std::exp(2.0 * omega * rest.planner.dt);
  const scalar_t w = rest.terminalDcm.weight;
  // Quadratic terms are identical, the linear ones differ by 2 w gain offset per coefficient, offset = -v_cmd / omega.
  EXPECT_TRUE(qpWalking.stages[last].Q.isApprox(qpRest.stages[last].Q));
  EXPECT_TRUE(qpWalking.stages[last].R.isApprox(qpRest.stages[last].R));
  EXPECT_TRUE(qpRest.stages[last].q.isZero());
  EXPECT_NEAR(qpWalking.stages[last].q(LIP_CX), -2.0 * w * gain * 0.4 / omega, 1e-9);
  EXPECT_NEAR(qpWalking.stages[last].q(LIP_VX), -2.0 * w * gain * 0.4 / omega / omega, 1e-9);
  EXPECT_NEAR(qpWalking.stages[last].r(LIP_ZX), 2.0 * w * gain * 0.4 / omega, 1e-9);
  EXPECT_NEAR(qpWalking.stages[last].q(LIP_CY), 0.0, 1e-12);
  // Every other stage of the horizon has to stay untouched. The bound of this loop used to be `k + 1 < last`, which
  // stopped one node too early: with the twelve-node configuration of makeConfig() it inspected the nodes 0 ... 9 only,
  // so node 10 - the one immediately before the last running node - was never looked at, and the terminal stage was not
  // looked at either, while the failure message already claimed "only the last running node". A widening of
  // NodeSet::LAST_RUNNING that let the terminal capturability cost spill onto the second-to-last running node as well
  // (which would double the terminal capturability weight and shorten the last steps of every plan) therefore slipped
  // through unnoticed. The bound is now `k < last`, and the terminal stage - which has no inputs and no ZMP, so the
  // residual of this cost is not even defined there - is checked on its own afterwards.
  for (size_t k = 0; k < last; ++k) EXPECT_TRUE(qpWalking.stages[k].q.isZero()) << "only the last running node, k = " << k;
  EXPECT_TRUE(qpWalking.stages.back().q.isZero()) << "not on the terminal node";
}

// nodeSetContains is the one place that decides, for every stage of the assembled QP, whether a term is invoked on it
// (ContactPlanningProblem::assemble consults it for every cost and every constraint). Until now it had no direct test
// anywhere in the repository: its four cases were pinned only indirectly, and only as far as the stages the terms of
// the other tests in this file happen to be written on. That is a thin thread to hang the semantics of the whole
// formulation on, so the four node sets are checked here against the convention they promise - N running nodes
// 0 ... N - 1 that carry inputs and dynamics, plus the terminal stage N that carries neither - over a complete horizon,
// one entry per stage, together with the names printed in the start-up summary of the problem.
TEST(ContactPlanningTerms, NodeSetsSelectExactlyTheStagesTheirNamesPromise) {
  constexpr int kNumNodes = 4;
  struct Expectation {
    NodeSet set;
    const char* name;
    std::array<bool, 5> contains;  // one entry per stage, 0 ... kNumNodes, the terminal stage last
  };
  const std::vector<Expectation> expectations = {
      {NodeSet::RUNNING, "running nodes", {{true, true, true, true, false}}},
      {NodeSet::TERMINAL, "terminal node", {{false, false, false, false, true}}},
      {NodeSet::ALL, "all nodes", {{true, true, true, true, true}}},
      {NodeSet::LAST_RUNNING, "last running node", {{false, false, false, true, false}}},
  };
  for (const Expectation& expectation : expectations) {
    EXPECT_EQ(nodeSetName(expectation.set), expectation.name);
    for (int node = 0; node <= kNumNodes; ++node) {
      EXPECT_EQ(nodeSetContains(expectation.set, node, kNumNodes), expectation.contains[static_cast<size_t>(node)])
          << expectation.name << ", node " << node << " of " << kNumNodes;
    }
  }
}

// planner.logPlans prints one line per plan: search statistics, phase sequence with durations, step lengths.
TEST(ContactPlanningTerms, APlanDescribesItsPhasesAndSteps) {
  ContactPlan plan;
  plan.valid = true;
  plan.startTime = 2.0;
  plan.dt = 0.1;
  plan.objective = 1.5;
  plan.numBranchAndBoundNodes = 7;
  plan.solveTime = 0.012;
  plan.optimal = true;
  // Left: contact 0.2 s, swing 0.3 s (a 0.4 m step forward), contact 0.1 s. Right: contact throughout.
  plan.contacts = {{true, true}, {true, true}, {false, true}, {false, true}, {false, true}, {true, true}};
  plan.footholds.assign(7, {vector2_t(0.0, 0.1), vector2_t(0.0, -0.1)});
  for (size_t k = 5; k < 7; ++k) plan.footholds[k][0] = vector2_t(0.4, 0.1);
  plan.comVelocity = {vector2_t(0.3, 0.0), vector2_t(0.5, 0.0)};
  const std::string line = plan.describe();
  EXPECT_NE(line.find("plan t=2.000 valid J=1.500 relaxations=7 solve=12.000ms optimal"), std::string::npos) << line;
  EXPECT_NE(line.find("v0=[0.300 0.000] vN=[0.500 0.000]"), std::string::npos) << line;
  EXPECT_NE(line.find("| L: C0.200 S0.300(0.400,0.000) C0.100"), std::string::npos) << line;
  EXPECT_NE(line.find("| R: C0.600"), std::string::npos) << line;
  plan.valid = false;
  plan.timeLimitHit = true;
  plan.optimal = false;
  EXPECT_NE(plan.describe().find("INVALID"), std::string::npos);
  EXPECT_NE(plan.describe().find("TIME-LIMIT"), std::string::npos);
}

/*============================================ stage accumulation =========================================*/
namespace {

// The functional StepLengthCost documents, w sum_i ||dp_i - d_nom (1 - c_i)||^2 summed over the running nodes, written
// out from the input trajectory alone. It deliberately restates the formula of the term instead of reading anything the
// assembly produced, so that it is an independent statement of what the assembled objective has to evaluate to.
scalar_t stepLengthFunctional(scalar_t weight, const vector2_t& nominalDisplacement, const std::vector<vector_t>& inputs) {
  const std::array<std::array<int, 2>, N_CONTACTS> footDelta{{{LIP_DLX, LIP_DLY}, {LIP_DRX, LIP_DRY}}};
  const std::array<int, N_CONTACTS> contact{LIP_CL, LIP_CR};
  scalar_t total = 0.0;
  for (const vector_t& input : inputs) {
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      for (int axis = 0; axis < 2; ++axis) {
        const scalar_t residual =
            input(footDelta[foot][static_cast<size_t>(axis)]) - nominalDisplacement(axis) * (1.0 - input(contact[foot]));
        total += weight * residual * residual;
      }
    }
  }
  return total;
}

}  // namespace

// StageAccumulator::addQuadraticResidual expands w (l_x' x + l_u' u + c)^2 into the stage, and the expansion has three
// parts: the quadratic one, the linear one, and the constant w c^2. Only the first two used to be written, because the
// third is invisible to the QP solver - a constant moves neither the minimiser nor any comparison between two solutions
// of the SAME assembled problem - so the objective every consumer of the planner saw was the documented functional
// minus a per-problem constant. This pins the whole expansion: the stage has to evaluate to the squared residuals
// themselves, at trajectories where the residual is zero as well as at trajectories where it is not. The all-zero
// trial matters most: there the entire cost IS the dropped constant, and the old code reported exactly 0.
TEST(ContactPlanningTerms, AnAccumulatedStageEvaluatesToTheFullSquaredResidual) {
  constexpr int kNumStates = 2;
  constexpr int kNumInputs = 2;
  OcpQpProblem problem;
  problem.x0 = vector_t::Zero(kNumStates);
  problem.stages.resize(2);
  problem.stages[0] = OcpQpStage::Zero(kNumStates, kNumInputs, true);
  problem.stages[1] = OcpQpStage::Zero(kNumStates, 0, false);

  // Two residuals on the running stage: one over both states and both inputs, one over a single state. Both offsets are
  // non-zero, which is the whole point - with a zero offset the assembled stage is right either way.
  const scalar_t firstWeight = 3.0;
  const scalar_t firstOffset = 0.75;
  const scalar_t secondWeight = 0.5;
  const scalar_t secondOffset = -1.25;
  StageAccumulator accumulator(problem.stages[0]);
  accumulator.addQuadraticResidual({{0, 1.5}, {1, -0.5}}, {{0, 2.0}, {1, 0.25}}, firstOffset, firstWeight);
  accumulator.addQuadraticResidual({{1, 1.0}}, {}, secondOffset, secondWeight);

  const vector_t terminalState = vector_t::Zero(kNumStates);
  const std::array<std::array<scalar_t, 4>, 4> trials{
      {{0.0, 0.0, 0.0, 0.0}, {0.3, -0.7, 0.9, -0.2}, {-1.1, 2.0, -0.4, 1.3}, {2.5, 0.125, -1.75, 0.0}}};
  for (const std::array<scalar_t, 4>& trial : trials) {
    vector_t state = vector_t::Zero(kNumStates);
    state << trial[0], trial[1];
    vector_t input = vector_t::Zero(kNumInputs);
    input << trial[2], trial[3];
    const scalar_t firstResidual = 1.5 * state(0) - 0.5 * state(1) + 2.0 * input(0) + 0.25 * input(1) + firstOffset;
    const scalar_t secondResidual = state(1) + secondOffset;
    const scalar_t expected = firstWeight * firstResidual * firstResidual + secondWeight * secondResidual * secondResidual;
    const std::vector<vector_t> stateTrajectory = {state, terminalState};
    const std::vector<vector_t> inputTrajectory = {input};
    EXPECT_NEAR(evaluateOcpQpObjective(problem, stateTrajectory, inputTrajectory), expected, 1e-12)
        << "x = " << state.transpose() << ", u = " << input.transpose()
        << ": the stage must evaluate to the squared residuals, the constant part of the expansion included";
  }
}

// The reason the dropped constant is not a cosmetic issue: CadenceStretchStage::afterSearch re-assembles the problem on
// a stretched node grid (s * dt) and scores the result against the incumbent assembled at dt, and the residual offsets
// of the shipped terms depend on dt - StepLengthCost's nominal displacement is v_cmd dt T_stride / T_swing. The
// constant therefore differs between the two problems, and while it was dropped, the stretched candidate's reported
// objective was depressed relative to the incumbent's by an amount that grows with the stretch, so the stage kept
// accepting stretches whose true cost was higher. This assembles the same single-term formulation on two grids and
// demands that both report their own documented functional, on an arbitrary trajectory and on the zero trajectory,
// where the entire cost is the constant and the old code called the two grids equally good at 0.
TEST(ContactPlanningTerms, ObjectivesOfProblemsAssembledOnDifferentNodeGridsAreComparable) {
  constexpr scalar_t kStretch = 1.25;
  ContactPlanningConfig config = makeConfig();
  config.formulation.costs = {term::kStepLength};
  config.formulation.softConstraints = {};
  config.formulation.hardConstraints = {};
  config.stepLength.weight = 20.0;
  config.validate();
  ContactPlanningConfig stretchedConfig = config;
  stretchedConfig.planner.dt = kStretch * config.planner.dt;
  stretchedConfig.validate();
  LipContactPlanner planner(config);
  LipContactPlanner stretchedPlanner(stretchedConfig);

  ContactPlannerInput input = makeStandingInput();
  input.velocityCommand = vector2_t(0.4, -0.25);  // both axes, so that neither of the two offsets per foot is zero

  const scalar_t weight = config.stepLength.weight;
  const scalar_t strideToSwingRatio = 8.0 / 3.0;  // 2 (minSwingDuration + minDoubleSupportDuration) / minSwingDuration
  const vector2_t nominalDisplacement = input.velocityCommand * (config.planner.dt * strideToSwingRatio);
  const vector2_t stretchedDisplacement = input.velocityCommand * (stretchedConfig.planner.dt * strideToSwingRatio);

  const OcpQpProblem qp = planner.buildProblem(input);
  const OcpQpProblem stretchedQp = stretchedPlanner.buildProblem(input);
  // Anchor d_nom to what was actually assembled, so that a change of the nominal displacement shows up as a failure
  // here rather than silently moving both sides of the comparisons below.
  ASSERT_NEAR(qp.stages.front().R(LIP_DLX, LIP_CL), 2.0 * weight * nominalDisplacement.x(), 1e-12);
  ASSERT_NEAR(stretchedQp.stages.front().R(LIP_DRY, LIP_CR), 2.0 * weight * stretchedDisplacement.y(), 1e-12);
  ASSERT_EQ(stretchedQp.stages.size(), qp.stages.size()) << "the stretch changes the node duration, not the node count";

  const int numStates = qp.stages.front().numStates();
  const int numInputs = qp.stages.front().numInputs();
  const int numRunningNodes = qp.numStages();
  std::vector<vector_t> stateTrajectory;
  std::vector<vector_t> inputTrajectory;
  for (int k = 0; k <= numRunningNodes; ++k) {
    vector_t stateAtNode = vector_t::Zero(numStates);
    for (int i = 0; i < numStates; ++i) stateAtNode(i) = 0.2 * static_cast<scalar_t>(i) - 0.03 * static_cast<scalar_t>(k);
    stateTrajectory.push_back(stateAtNode);
    if (k == numRunningNodes) break;  // the terminal stage carries no inputs
    vector_t inputAtNode = vector_t::Zero(numInputs);
    for (int i = 0; i < numInputs; ++i) inputAtNode(i) = 0.05 * static_cast<scalar_t>(i + 1) - 0.01 * static_cast<scalar_t>(k);
    inputTrajectory.push_back(inputAtNode);
  }

  const scalar_t nominalOnTrajectory = evaluateOcpQpObjective(qp, stateTrajectory, inputTrajectory);
  const scalar_t stretchedOnTrajectory = evaluateOcpQpObjective(stretchedQp, stateTrajectory, inputTrajectory);
  EXPECT_NEAR(nominalOnTrajectory, stepLengthFunctional(weight, nominalDisplacement, inputTrajectory), 1e-9);
  EXPECT_NEAR(stretchedOnTrajectory, stepLengthFunctional(weight, stretchedDisplacement, inputTrajectory), 1e-9);
  // The two problems are not merely shifted by a constant each: on this trajectory the dropped constants reversed the
  // ORDER of the two grids. The stretched problem is the genuinely worse of the two here - 35.50 against 33.90 in the
  // functional the terms document - while an objective that leaves the constants out reports 23.63 against 26.30 and
  // makes the stretch look like an improvement of 2.7. That is exactly the comparison CadenceStretchStage makes, so
  // the ordering, and not only the two values, is part of what this pins.
  EXPECT_GT(stretchedOnTrajectory, nominalOnTrajectory)
      << "the grid whose documented cost is the higher one must report the higher objective";

  // Feet that do not move while no foot is in contact: every residual is exactly -d_nom, so the entire cost of each
  // problem is its own dropped constant and the two grids differ by the factor s^2 of the stretched nominal step.
  const vector_t zeroState = vector_t::Zero(numStates);
  const vector_t zeroInput = vector_t::Zero(numInputs);
  const std::vector<vector_t> zeroStates(qp.stages.size(), zeroState);
  const std::vector<vector_t> zeroInputs(static_cast<size_t>(numRunningNodes), zeroInput);
  const scalar_t expectedNominal = stepLengthFunctional(weight, nominalDisplacement, zeroInputs);
  const scalar_t nominalObjective = evaluateOcpQpObjective(qp, zeroStates, zeroInputs);
  const scalar_t stretchedObjective = evaluateOcpQpObjective(stretchedQp, zeroStates, zeroInputs);
  ASSERT_GT(expectedNominal, 1.0) << "the trial is only meaningful while the dropped constant is a large number";
  EXPECT_NEAR(nominalObjective, expectedNominal, 1e-9);
  EXPECT_NEAR(stretchedObjective, kStretch * kStretch * expectedNominal, 1e-9);
  EXPECT_NEAR(stretchedObjective - nominalObjective, (kStretch * kStretch - 1.0) * expectedNominal, 1e-9)
      << "the stretched grid is strictly worse here; reporting a tie is what let the cadence stretch overshoot";
}

/*============================================ required blocks and names ==================================*/

TEST(ContactPlanningTerms, AHeadingTermWithoutTheHeadingBlockIsRejected) {
  ContactPlanningConfig config = makeConfig();
  config.formulation.costs.push_back(term::kHeadingTracking);
  EXPECT_THROW(config.validate(), std::invalid_argument);
  // The problem checks it too, for a problem assembled by hand.
  ContactPlanningProblem problem;
  problem.model.add(term::kLipCom, ContactPlanningTermFactory::makeModelBlock(term::kLipCom));
  problem.model.add(term::kFootholdIntegrator, ContactPlanningTermFactory::makeModelBlock(term::kFootholdIntegrator));
  problem.costs.add(term::kHeadingTracking, ContactPlanningTermFactory::makeCost(term::kHeadingTracking));
  EXPECT_THROW(problem.finalize(makeConfig()), std::invalid_argument);
}

TEST(ContactPlanningTerms, TheFactoryNamesTheSupportedTermsInItsErrors) {
  try {
    ContactPlanningTermFactory::makeCost("gravity_compensation");
    FAIL() << "an unknown cost must throw";
  } catch (const std::invalid_argument& e) {
    const std::string what = e.what();
    EXPECT_NE(what.find("gravity_compensation"), std::string::npos);
    EXPECT_NE(what.find(term::kVelocityTracking), std::string::npos) << "the supported names are listed";
  }
  EXPECT_THROW(ContactPlanningTermFactory::makeExecutionRule(term::kPlannedHeadingOverride), std::invalid_argument)
      << "the heading override needs the reference manager's model";
  EXPECT_NO_THROW(ContactPlanningTermFactory::makeExecutionRule("phaseResetting"));
}

TEST(ContactPlanningTerms, EveryKnownTermCanBeBuiltAndDescribed) {
  const ContactPlanningConfig config = makeHeadingConfig();
  for (const std::string& name : knownTermNames(TermKind::COST)) {
    auto cost = ContactPlanningTermFactory::makeCost(name);
    cost->configure(config);
    EXPECT_FALSE(cost->describe().empty()) << name;
  }
  for (const std::string& name : knownTermNames(TermKind::SOFT_CONSTRAINT)) {
    auto constraint = ContactPlanningTermFactory::makeSoftConstraint(name);
    constraint->configure(config);
    EXPECT_EQ(constraint->softness(), Softness::SOFT) << name;
  }
  for (const std::string& name : knownTermNames(TermKind::HARD_CONSTRAINT)) {
    auto constraint = ContactPlanningTermFactory::makeHardConstraint(name);
    constraint->configure(config);
    EXPECT_EQ(constraint->softness(), Softness::HARD) << name;
  }
  for (const std::string& name : knownTermNames(TermKind::LOGIC_RULE))
    EXPECT_NO_THROW(ContactPlanningTermFactory::makeLogicRule(name)) << name;
  for (const std::string& name : knownTermNames(TermKind::ASSIGNMENT_COST)) {
    EXPECT_NO_THROW(ContactPlanningTermFactory::makeAssignmentCost(name)) << name;
  }
  for (const std::string& name : knownTermNames(TermKind::SEARCH_STAGE))
    EXPECT_NO_THROW(ContactPlanningTermFactory::makeSearchStage(name)) << name;
  const std::string summary = LipContactPlanner::formulationSummary(config);
  for (const std::string& name : config.formulation.costs) EXPECT_NE(summary.find(name), std::string::npos) << summary;
  EXPECT_NE(summary.find("x = [c_x c_y v_x v_y p_Lx p_Ly p_Rx p_Ry theta omega psi_L psi_R] (12)"), std::string::npos) << summary;
}

/*============================================ hot reload =================================================*/

TEST(ContactPlanningTerms, ParametersReloadByTermNameWithoutReassembly) {
  LipContactPlanner planner(makeConfig());
  const ContactPlannerInput input = makeWalkingInput();
  const OcpQpProblem before = planner.buildProblem(input);
  ContactPlanningConfig retuned = planner.getConfig();
  retuned.velocityTracking.weight *= 3.0;
  retuned.zmpSupportRegion.slack = SlackPenalty{123.0, 4.0};
  planner.setConfig(retuned);
  EXPECT_NEAR(planner.getProblem().costs.get<VelocityTrackingCost>(term::kVelocityTracking).weight(), retuned.velocityTracking.weight,
              1e-12);
  const OcpQpProblem after = planner.buildProblem(input);
  EXPECT_NEAR(after.stages.front().Q(LIP_VX, LIP_VX), 3.0 * before.stages.front().Q(LIP_VX, LIP_VX) - 2.0 * retuned.regularization.state,
              1e-9);
  // The ZMP rows are the first 12 soft rows of a running node; their penalty is the term's own, the rest keep the shared one.
  EXPECT_NEAR(after.stages.front().Zl(0), 123.0, 1e-12);
  EXPECT_NEAR(after.stages.front().zl(0), 4.0, 1e-12);
  EXPECT_NEAR(after.stages.front().Zl(12), retuned.shared.slackPenalty.quadratic, 1e-12);
  EXPECT_NEAR(before.stages.front().Zl(0), planner.getConfig().shared.slackPenalty.quadratic, 1e-12);
}

TEST(ContactPlanningTerms, ChangingATermListReassemblesTheProblemAndKeepsTheWarmStartWhileTheLayoutHolds) {
  LipContactPlanner planner(makeConfig());
  const ContactPlannerInput input = makeWalkingInput();
  const ContactPlan first = planner.plan(input);
  ASSERT_TRUE(first.valid);
  ContactPlannerInput next = input;
  next.time += 0.1;
  next.contacts = first.contactsAtTime(next.time);
  ASSERT_GE(planner.defaultNominal(next).com.size(), 1u);
  const bool warmBefore = planner.makeContext(next, planner.defaultNominal(next)).previousPlanShift >= 0;
  EXPECT_TRUE(warmBefore);

  ContactPlanningConfig fewerCosts = planner.getConfig();
  fewerCosts.formulation.setCost(term::kStepWidth, false);
  planner.setConfig(fewerCosts);
  EXPECT_FALSE(planner.getProblem().costs.has(term::kStepWidth));
  EXPECT_GE(planner.makeContext(next, planner.defaultNominal(next)).previousPlanShift, 0)
      << "same grid and layout: the warm start survives";

  ContactPlanningConfig heading = makeHeadingConfig();
  planner.setConfig(heading);
  EXPECT_TRUE(planner.getLayout().hasHeading);
  EXPECT_LT(planner.makeContext(next, planner.defaultNominal(next)).previousPlanShift, 0) << "a new layout drops the warm start";
}

TEST(ContactPlanningTerms, SearchStagesFollowTheList) {
  ContactPlanningConfig config = makeConfig();
  config.formulation.search = {term::kWarmStartPreviousPlan};
  LipContactPlanner planner(config);
  EXPECT_EQ(planner.getSearchStages().size(), 1u);
  const ContactPlan plan = planner.plan(makeWalkingInput());
  EXPECT_TRUE(plan.valid);
  EXPECT_EQ(planner.getLastStatistics().numLocalSearchQps, 0) << "no local search stage";
  config.formulation.search = {term::kHeadingRelinearisation};
  EXPECT_THROW(LipContactPlanner{config}, std::invalid_argument) << "a stage that needs the heading block";
}

/*============================================ execution rules ============================================*/

TEST(ContactPlanningTerms, ExecutionRulesAreBuiltFromTheListAndReportWhatTheyNeed) {
  ContactPlanningConfig config = makeConfig();
  config.formulation.execution = {term::kPhaseResetting, term::kEnergyCadenceModulation, term::kDcmStepAdjustment};
  config.validate();
  const TermCollection<ExecutionRule> rules = ContactPlanningTermFactory::buildExecutionRules(config);
  ASSERT_EQ(rules.size(), 3u);
  EXPECT_FALSE(rules.get(term::kPhaseResetting).needsPredictedTrajectory());
  EXPECT_TRUE(rules.get(term::kEnergyCadenceModulation).needsPredictedTrajectory());
  EXPECT_TRUE(rules.get(term::kDcmStepAdjustment).needsPredictedTrajectory());
  config.formulation.execution.push_back(term::kPlannedHeadingOverride);
  EXPECT_THROW(ContactPlanningTermFactory::buildExecutionRules(config), std::invalid_argument) << "needs the heading block and the model";
}

TEST(ContactPlanningTerms, WithoutRulesTheScheduleIsLeftAloneAndTheLatchTracksTheSwing) {
  // A swing of the left foot from 1.0 to 1.4 s, the right foot down throughout.
  ModeSchedule schedule({1.0, 1.4}, {ModeNumber::STANCE, ModeNumber::RF, ModeNumber::STANCE});
  feet_array_t<SwingTimingLatch> latches = makeFeetArray(SwingTimingLatch{});
  const ContactPlanningConfig config = makeConfig();
  const feet_array_t<scalar_t> noCadence = makeFeetArray(0.0);
  contact_flag_t measured = {true, true};  // the left foot reports contact mid-swing
  const auto reports = adaptScheduleToContactEvents(schedule, 1.2, measured, noCadence, config, latches);
  EXPECT_EQ(reports[0].type, ContactEventReport::Type::NONE) << "no phase_resetting rule listed";
  EXPECT_TRUE(latches[0].active);
  EXPECT_NEAR(latches[0].nominalTouchDownTime, 1.4, 1e-12);
  EXPECT_NEAR(schedule.eventTimes[1], 1.4, 1e-12);
  measured = {false, true};  // the foot is still in the air at its scheduled touch-down: nothing extends it either
  const auto late = adaptScheduleToContactEvents(schedule, 1.45, measured, noCadence, config, latches);
  EXPECT_EQ(late[0].type, ContactEventReport::Type::NONE);
  EXPECT_FALSE(latches[0].active) << "the contact phase proceeds and the latch is released";
}

/** The double support penalty, on a hand-built assignment of three nodes. Binaries are node-major: 2 * node + foot. */
TEST(ContactPlanningTerms, DoubleSupportPenaltyChargesEveryNodeWithBothFeetDown) {
  ContactPlanningConfig config = makeConfig();
  config.formulation.assignmentCosts = {term::kDoubleSupportPenalty};
  config.doubleSupportPenalty.cost = 5.0;
  config.validate();
  LipContactPlanner planner(config);
  const auto& cost = planner.getProblem().assignmentCosts.get<DoubleSupportPenaltyCost>(term::kDoubleSupportPenalty);

  ContactLogicState state;
  state.numNodes = 3;
  MiqpAssignment a(3 * N_CONTACTS);
  const auto set = [&a](int node, std::int8_t left, std::int8_t right) {
    a[static_cast<size_t>(ContactLogicState::contactBinaryIndex(node, 0))] = left;
    a[static_cast<size_t>(ContactLogicState::contactBinaryIndex(node, 1))] = right;
  };

  set(0, 0, 1);  // left swinging, right in contact
  set(1, 1, 1);  // double support
  set(2, 1, 0);  // left in contact, right swinging
  EXPECT_NEAR(cost.cost(state, a), 5.0, 1e-12) << "one double-support node out of three";

  set(2, 1, 1);
  EXPECT_NEAR(cost.cost(state, a), 10.0, 1e-12) << "two double-support nodes";

  set(0, 1, 1);
  set(1, 1, 1);
  set(2, 1, 1);
  EXPECT_NEAR(cost.cost(state, a), 15.0, 1e-12) << "standing costs the whole horizon";

  config.doubleSupportPenalty.cost = 0.0;
  LipContactPlanner disabled(config);
  EXPECT_NEAR(disabled.getProblem().assignmentCosts.get<DoubleSupportPenaltyCost>(term::kDoubleSupportPenalty).cost(state, a), 0.0, 1e-12)
      << "a zero cost is free whatever the assignment";
}

/**
 * The branch-and-bound prunes on the assignment cost of a *partial* assignment, so the term has to be a lower bound
 * there and exact only once every binary is decided. A free node may still become a double support, so charging it
 * would overestimate the partial assignment and could discard the optimum.
 */
TEST(ContactPlanningTerms, DoubleSupportPenaltyIsALowerBoundOnAPartialAssignment) {
  ContactPlanningConfig config = makeConfig();
  config.formulation.assignmentCosts = {term::kDoubleSupportPenalty};
  config.doubleSupportPenalty.cost = 5.0;
  config.validate();
  LipContactPlanner planner(config);
  const auto& cost = planner.getProblem().assignmentCosts.get<DoubleSupportPenaltyCost>(term::kDoubleSupportPenalty);

  ContactLogicState state;
  state.numNodes = 3;
  MiqpAssignment partial(3 * N_CONTACTS, kMiqpFree);
  EXPECT_NEAR(cost.cost(state, partial), 0.0, 1e-12) << "nothing decided, nothing charged";

  // One foot down at node 0, the other still free: not yet a double support.
  partial[static_cast<size_t>(ContactLogicState::contactBinaryIndex(0, 0))] = 1;
  EXPECT_NEAR(cost.cost(state, partial), 0.0, 1e-12) << "a half-decided node is not charged";

  // Deciding the second foot completes the double support and the cost may only rise.
  partial[static_cast<size_t>(ContactLogicState::contactBinaryIndex(0, 1))] = 1;
  EXPECT_NEAR(cost.cost(state, partial), 5.0, 1e-12);

  // Completing the rest of the horizon can only add to it, never take away.
  MiqpAssignment complete = partial;
  for (int k = 1; k < state.numNodes; ++k) {
    complete[static_cast<size_t>(ContactLogicState::contactBinaryIndex(k, 0))] = 1;
    complete[static_cast<size_t>(ContactLogicState::contactBinaryIndex(k, 1))] = 0;
  }
  EXPECT_GE(cost.cost(state, complete), cost.cost(state, partial)) << "the bound must not decrease when a node is decided";
}

/**
 * In a zero-double-support gait the exchange is a single event: one foot lands at the very instant the other lifts.
 * Ending a swing early holds the landing foot down from the measured contact until its scheduled touch-down, so an
 * early touch-down there opens a double support exactly as long as the foot was early. Since a landing within a few
 * milliseconds of the plan is the common case, that put a sliver of double support - shorter than the MPC's own time
 * step - on nearly every step, at both ends of every swing (each exchange is one foot's touch-down and the other's
 * lift-off). earlyTouchdownMinAdvance executes a near-on-time touch-down as planned instead.
 */
TEST(ContactPlanningTerms, CadenceStretchIsBoundedByTheGaitLimits) {
  // The stretch re-times every phase of the incumbent together, so the phase that is closest to its limit is what
  // bounds it. A five-node swing at dt 0.1 is 0.5 s, already at maxSwingDuration, so nothing may be stretched; a
  // four-node swing is 0.4 s and leaves room for exactly 0.5 / 0.4.
  ContactPlanningConfig config = makeConfig();
  config.planner.numNodes = 12;
  config.cadenceStretch.samples = 4;
  config.cadenceStretch.maxStretch = 2.0;
  config.validate();

  const auto assignmentWithSwing = [&config](int swingNodes) {
    MiqpAssignment assignment(static_cast<size_t>(ContactLogicState::kBinariesPerNode * config.planner.numNodes), 1);
    for (int node = 0; node < swingNodes; ++node) {
      assignment[static_cast<size_t>(ContactLogicState::contactBinaryIndex(node, CONTACT_LEFT_INDEX))] = 0;
    }
    return assignment;
  };

  EXPECT_NEAR(CadenceStretchStage::admissibleStretch(config, assignmentWithSwing(4), config.planner.numNodes, 2.0), 0.5 / 0.4, 1e-9);
  EXPECT_NEAR(CadenceStretchStage::admissibleStretch(config, assignmentWithSwing(5), config.planner.numNodes, 2.0), 1.0, 1e-9);
  // maxStretch is the other bound, and it is the binding one for a short swing.
  EXPECT_NEAR(CadenceStretchStage::admissibleStretch(config, assignmentWithSwing(3), config.planner.numNodes, 1.2), 1.2, 1e-9);
  // A swing that already exceeds the limit cannot be repaired by stretching, and must not push the bound below 1.
  EXPECT_NEAR(CadenceStretchStage::admissibleStretch(config, assignmentWithSwing(8), config.planner.numNodes, 2.0), 1.0, 1e-9);
}

TEST(ContactPlanningTerms, CadenceStretchIsDisabledWithoutSamples) {
  ContactPlanningConfig config = makeConfig();
  config.cadenceStretch.samples = 0;
  config.validate();
  std::unique_ptr<SearchStage> stage = ContactPlanningTermFactory::makeSearchStage(term::kCadenceStretch);
  ASSERT_NE(stage, nullptr);
  stage->configure(config);
  EXPECT_NE(stage->describe().find("0 stretch"), std::string::npos);
}

TEST(ContactPlanningTerms, ANearlyOnTimeTouchDownDoesNotSplitAZeroDoubleSupportExchange) {
  // Left swings 1.0 -> 1.4 and lands at the instant the right lifts; right then swings 1.4 -> 1.8.
  const ModeSchedule nominal({1.0, 1.4, 1.8}, {ModeNumber::STANCE, ModeNumber::RF, ModeNumber::LF, ModeNumber::STANCE});
  ContactPlanningConfig config = makeConfig();
  config.formulation.execution = {term::kPhaseResetting};
  config.validate();
  ASSERT_GT(config.phaseResetting.earlyTouchdownMinAdvance, config.phaseResetting.earlyTouchdownMinContactDuration)
      << "otherwise the debounce alone pushes every truncation inside the guard window";
  const feet_array_t<scalar_t> noCadence = makeFeetArray(0.0);
  const contact_flag_t bothDown = {true, true};

  // The shortest double support anywhere in the schedule, ignoring the leading and trailing stance phases.
  const auto shortestDoubleSupport = [](const ModeSchedule& schedule) {
    scalar_t shortest = std::numeric_limits<scalar_t>::infinity();
    for (size_t i = 0; i + 1 < schedule.eventTimes.size(); ++i) {
      if (schedule.modeSequence[i + 1] == ModeNumber::STANCE) {
        shortest = std::min(shortest, schedule.eventTimes[i + 1] - schedule.eventTimes[i]);
      }
    }
    return shortest;
  };

  {  // Landing 30 ms early: inside the guard, the exchange is left exactly as planned.
    ModeSchedule schedule = nominal;
    feet_array_t<SwingTimingLatch> latches = makeFeetArray(SwingTimingLatch{});
    adaptScheduleToContactEvents(schedule, 1.37, bothDown, noCadence, config, latches);
    const auto reports = adaptScheduleToContactEvents(schedule, 1.392, bothDown, noCadence, config, latches);
    EXPECT_EQ(reports[0].type, ContactEventReport::Type::NONE) << "a near-on-time touch-down is not an early one";
    EXPECT_EQ(schedule.eventTimes, nominal.eventTimes) << "the schedule was re-timed";
    EXPECT_EQ(schedule.modeSequence, nominal.modeSequence) << "the exchange was split by a sliver of double support";
  }
  {  // Landing 200 ms early: a real early touch-down, still reset, and the double support it opens spans the guard.
    ModeSchedule schedule = nominal;
    feet_array_t<SwingTimingLatch> latches = makeFeetArray(SwingTimingLatch{});
    adaptScheduleToContactEvents(schedule, 1.20, bothDown, noCadence, config, latches);
    const auto reports = adaptScheduleToContactEvents(schedule, 1.222, bothDown, noCadence, config, latches);
    EXPECT_EQ(reports[0].type, ContactEventReport::Type::EARLY_TOUCH_DOWN);
    EXPECT_NEAR(reports[0].touchDownTime, 1.222, 1e-9);
    EXPECT_GE(shortestDoubleSupport(schedule), config.phaseResetting.earlyTouchdownMinAdvance - 1e-9)
        << "the rule may not create a double support shorter than its own guard";
  }
  {  // With the guard off the sliver comes back: this is the behaviour the guard exists to remove.
    ContactPlanningConfig unguarded = config;
    unguarded.phaseResetting.earlyTouchdownMinAdvance = 0.0;
    unguarded.validate();
    ModeSchedule schedule = nominal;
    feet_array_t<SwingTimingLatch> latches = makeFeetArray(SwingTimingLatch{});
    adaptScheduleToContactEvents(schedule, 1.37, bothDown, noCadence, unguarded, latches);
    const auto reports = adaptScheduleToContactEvents(schedule, 1.392, bothDown, noCadence, unguarded, latches);
    EXPECT_EQ(reports[0].type, ContactEventReport::Type::EARLY_TOUCH_DOWN);
    EXPECT_NEAR(shortestDoubleSupport(schedule), 0.008, 1e-9) << "8 ms of double support, below the MPC time step";
  }
}

/*=================================== assignment costs: the lower-bound contract ==========================*/
namespace {

/**
 * Every assignment of `numBinaries` binaries with three states per entry - free, 0 and 1 - enumerated in base three.
 *
 * The lower-bound contract of AssignmentCost is a statement about EVERY partial assignment, not about a handful of
 * hand-picked ones, and the branch-and-bound really does evaluate arbitrary partial assignments: MixedIntegerOcpQp
 * calls the assignment cost on a node whose fixings are whatever branching and propagation have produced so far, which
 * is an arbitrary mixture of decided and free entries and not necessarily a prefix in time order (the propagation hook
 * fixes late variables from early ones). Enumerating all 3^6 patterns of a three-node horizon is cheap and is the only
 * way to state the contract as the search relies on it.
 */
std::vector<MiqpAssignment> allPartialAssignments(int numBinaries) {
  int numCodes = 1;
  for (int i = 0; i < numBinaries; ++i) numCodes *= 3;
  std::vector<MiqpAssignment> assignments;
  assignments.reserve(static_cast<size_t>(numCodes));
  for (int code = 0; code < numCodes; ++code) {
    MiqpAssignment assignment(static_cast<size_t>(numBinaries), kMiqpFree);
    int rest = code;
    for (int i = 0; i < numBinaries; ++i) {
      assignment[static_cast<size_t>(i)] = static_cast<std::int8_t>(rest % 3 - 1);
      rest /= 3;
    }
    assignments.push_back(assignment);
  }
  return assignments;
}

/** Every way of deciding the free entries of `partial`; the singleton {partial} when it is already complete. */
std::vector<MiqpAssignment> allCompletionsOf(const MiqpAssignment& partial) {
  std::vector<size_t> freeIndices;
  for (size_t i = 0; i < partial.size(); ++i) {
    if (partial[i] == kMiqpFree) freeIndices.push_back(i);
  }
  const int numCompletions = 1 << static_cast<int>(freeIndices.size());
  std::vector<MiqpAssignment> completions;
  completions.reserve(static_cast<size_t>(numCompletions));
  for (int mask = 0; mask < numCompletions; ++mask) {
    MiqpAssignment complete = partial;
    for (size_t j = 0; j < freeIndices.size(); ++j) {
      complete[freeIndices[j]] = static_cast<std::int8_t>((mask >> j) & 1);
    }
    completions.push_back(complete);
  }
  return completions;
}

/** "10 0. .." - one group per node, left foot then right, '.' for a free entry. For the failure messages below. */
std::string printAssignment(const MiqpAssignment& assignment) {
  std::string out;
  for (size_t i = 0; i < assignment.size(); ++i) {
    if (i > 0 && i % N_CONTACTS == 0) out += " ";
    out += (assignment[i] == kMiqpFree) ? "." : (assignment[i] == 1 ? "1" : "0");
  }
  return out;
}

/** Writes both feet of one node of an assignment. Binaries are node-major: N_CONTACTS * node + foot. */
void setContacts(MiqpAssignment& assignment, int node, std::int8_t left, std::int8_t right) {
  assignment[static_cast<size_t>(ContactLogicState::contactBinaryIndex(node, 0))] = left;
  assignment[static_cast<size_t>(ContactLogicState::contactBinaryIndex(node, 1))] = right;
}

}  // namespace

/**
 * AssignmentCost.h states the contract of every cost on the binaries that is not part of the QP: "exact for a complete
 * assignment, a lower bound for a partial one", and MiqpAssignmentCostFn repeats it. It is not a stylistic preference.
 * MixedIntegerOcpQp adds the assignment cost of a node's PARTIAL assignment to the bound of its QP relaxation and
 * prunes the node when the sum is within absoluteGap of the incumbent, so a term that overcharges a partial assignment
 * by even one event can cut away the subtree that contains the optimum, and the search would then return a worse gait
 * while still reporting `optimal` - the one failure mode of a branch-and-bound that leaves no trace anywhere.
 *
 * ContactSwitchCost had no test of that contract at all; only its sibling DoubleSupportPenaltyCost did (above). The
 * mechanism that makes it a bound is subtle enough to deserve one: the term skips free entries and carries `previous`
 * across them, so what it counts is the number of alternations of the SUBSEQUENCE of decided nodes, prefixed by the
 * foot's measured contact state. That is a lower bound because two decided nodes that differ force at least one switch
 * somewhere between them, while two decided nodes that agree force none - but any completion may insert a lift-off and
 * a touch-down between them. Charging a free node, or resetting `previous` at one, would break it in either direction.
 */
TEST(ContactPlanningTerms, ContactSwitchCostIsExactWhenCompleteAndALowerBoundOnEveryPartialAssignment) {
  constexpr int kNumNodes = 3;
  ContactPlanningConfig config = makeConfig();
  config.contactSwitch.cost = 0.25;
  config.validate();
  std::unique_ptr<AssignmentCost> term = ContactPlanningTermFactory::makeAssignmentCost(term::kContactSwitch);
  ASSERT_NE(term, nullptr);
  term->configure(config);
  EXPECT_NE(term->describe().find("per lift-off / touch-down event"), std::string::npos) << term->describe();

  const ContactPlannerInput standing = makeStandingInput();  // both feet in contact at planning time
  ContactLogicState state;
  state.input = &standing;
  state.numNodes = kNumNodes;

  // The measured contact state is the node before the horizon: an assignment that keeps both feet down over the whole
  // horizon costs nothing, and the same assignment costs one event when a foot is in the air at planning time.
  MiqpAssignment allDown(static_cast<size_t>(kNumNodes) * N_CONTACTS, static_cast<std::int8_t>(1));
  EXPECT_NEAR(term->cost(state, allDown), 0.0, 1e-12) << "both feet were already down, nothing happens";
  ContactPlannerInput airborneLeft = standing;
  airborneLeft.contacts = {false, true};
  ContactLogicState airborneState = state;
  airborneState.input = &airborneLeft;
  EXPECT_NEAR(term->cost(airborneState, allDown), 0.25, 1e-12) << "the left foot touches down at node 0";

  // One full swing of the left foot inside the horizon: lift-off at node 0 and touch-down at node 2 is two events, the
  // right foot never moves. 2 * 0.25 = 0.5.
  MiqpAssignment oneSwing(static_cast<size_t>(kNumNodes) * N_CONTACTS, static_cast<std::int8_t>(1));
  setContacts(oneSwing, 0, 0, 1);
  setContacts(oneSwing, 1, 0, 1);
  setContacts(oneSwing, 2, 1, 1);
  EXPECT_NEAR(term->cost(state, oneSwing), 0.5, 1e-12) << "one lift-off and one touch-down";

  // Both feet stepping on every node is the most expensive pattern of this horizon: three alternations per foot.
  MiqpAssignment chattering(static_cast<size_t>(kNumNodes) * N_CONTACTS, static_cast<std::int8_t>(0));
  setContacts(chattering, 0, 0, 0);
  setContacts(chattering, 1, 1, 1);
  setContacts(chattering, 2, 0, 0);
  EXPECT_NEAR(term->cost(state, chattering), 6.0 * 0.25, 1e-12) << "three events per foot";

  // The subsequence argument, made concrete. Only the left foot at node 1 is decided, to the value the foot already
  // has, so nothing is charged - while the completion that lifts the foot at node 0 and puts it back down at node 1
  // costs two events. A term that charged the free nodes, or that treated a free node as a swing, would report more
  // than zero here and the search could prune the completion away.
  MiqpAssignment onlyNodeOne(static_cast<size_t>(kNumNodes) * N_CONTACTS, kMiqpFree);
  onlyNodeOne[static_cast<size_t>(ContactLogicState::contactBinaryIndex(1, 0))] = 1;
  EXPECT_NEAR(term->cost(state, onlyNodeOne), 0.0, 1e-12) << "the only decided node agrees with the measured state";
  MiqpAssignment completedWithASwing = onlyNodeOne;  // node 1 already has the left foot down
  setContacts(completedWithASwing, 0, 0, 1);         // the left foot lifts before that decided node
  completedWithASwing[static_cast<size_t>(ContactLogicState::contactBinaryIndex(1, 1))] = 1;
  setContacts(completedWithASwing, 2, 1, 1);
  EXPECT_NEAR(term->cost(state, completedWithASwing), 0.5, 1e-12)
      << "the completion pays for the lift-off and the touch-down the free nodes were hiding";

  // The contract itself, over every partial assignment of the horizon and every completion of each.
  const std::vector<MiqpAssignment> partials = allPartialAssignments(kNumNodes * static_cast<int>(N_CONTACTS));
  ASSERT_EQ(partials.size(), 729u) << "3^6 patterns of free / 0 / 1";
  int numStrictBounds = 0;
  for (const MiqpAssignment& partial : partials) {
    const scalar_t bound = term->cost(state, partial);
    EXPECT_GE(bound, 0.0) << printAssignment(partial);
    const std::vector<MiqpAssignment> completions = allCompletionsOf(partial);
    for (const MiqpAssignment& complete : completions) {
      const scalar_t exact = term->cost(state, complete);
      ASSERT_LE(bound, exact + 1e-12) << "the partial assignment " << printAssignment(partial) << " is charged " << bound
                                      << ", more than its completion " << printAssignment(complete) << " at " << exact
                                      << "; the branch-and-bound would prune the completion away on that bound";
      if (exact > bound + 1e-12) ++numStrictBounds;
    }
  }
  EXPECT_GT(numStrictBounds, 0) << "a bound that never actually undercharges would mean the enumeration is degenerate";

  // Exactness on a complete assignment is the other half of the contract: there the bound has to be reached, i.e. the
  // partial assignment that IS complete evaluates to the same number as itself. That is trivially true here, so the
  // statement worth pinning is that the horizon's contribution is the event count and nothing else - a zero price is
  // free whatever happens.
  ContactPlanningConfig freePrice = config;
  freePrice.contactSwitch.cost = 0.0;
  freePrice.validate();
  std::unique_ptr<AssignmentCost> freeTerm = ContactPlanningTermFactory::makeAssignmentCost(term::kContactSwitch);
  freeTerm->configure(freePrice);
  EXPECT_NEAR(freeTerm->cost(state, chattering), 0.0, 1e-12) << "a zero price charges nothing for any pattern";
}

/**
 * PlanConsistencyCost is the hysteresis that stops the planner from re-deciding the contact pattern every cycle, and it
 * is under the same lower-bound obligation as ContactSwitchCost (see the test above for why a violation is invisible).
 * Its arithmetic is the part that had no test: the previous plan is indexed at `k + previousPlanShift` - the node of
 * the previous plan that is live at the current node, since the horizon has moved forward by that many nodes - and the
 * index is CLAMPED at the previous plan's last node, because the last `shift` nodes of the current horizon reach past
 * the end of the previous one. The clamp is what makes the term compare the tail of the new horizon against the
 * previous plan's terminal pattern instead of running off the end of the vector.
 */
TEST(ContactPlanningTerms, PlanConsistencyCostShiftsIntoThePreviousPlanClampsAtItsLastNodeAndBoundsEveryPartial) {
  constexpr int kNumNodes = 3;
  ContactPlanningConfig config = makeConfig();
  config.planConsistency.cost = 0.5;
  config.validate();
  std::unique_ptr<AssignmentCost> term = ContactPlanningTermFactory::makeAssignmentCost(term::kPlanConsistency);
  ASSERT_NE(term, nullptr);
  term->configure(config);
  EXPECT_NE(term->describe().find("per decided node whose contact differs from the previous plan"), std::string::npos) << term->describe();

  const ContactPlannerInput standing = makeStandingInput();
  MiqpAssignment previous(static_cast<size_t>(kNumNodes) * N_CONTACTS, static_cast<std::int8_t>(0));
  setContacts(previous, 0, 1, 0);  // left down, right swinging
  setContacts(previous, 1, 0, 1);  // the exchange
  setContacts(previous, 2, 1, 1);  // double support

  ContactLogicState state;
  state.input = &standing;
  state.numNodes = kNumNodes;
  state.previousAssignment = &previous;

  // The assignment every case below is scored against: all four feet-nodes in the air is deliberately far from the
  // previous plan, so that each shift produces a different, checkable number.
  MiqpAssignment allUp(static_cast<size_t>(kNumNodes) * N_CONTACTS, static_cast<std::int8_t>(0));

  // shift 0 - the previous plan has not moved, node k is compared with node k. The previous plan has four ones in it
  // (node 0 left, node 1 right, node 2 both), so four of the six entries differ.
  state.previousPlanShift = 0;
  EXPECT_NEAR(term->cost(state, allUp), 4.0 * 0.5, 1e-12) << "one charge per previous contact that is now a swing";

  // shift 1 - node 0 reads the previous node 1 (one contact), node 1 reads the previous node 2 (two contacts), and
  // node 2 would read the previous node 3, which does not exist and is clamped back to node 2 (two contacts).
  state.previousPlanShift = 1;
  EXPECT_NEAR(term->cost(state, allUp), 5.0 * 0.5, 1e-12) << "1 + 2 + 2 charges, the last node clamped";

  // shift 2 - every node of the horizon is at or past the end of the previous plan, so all three read its last node.
  // This is the clamp on its own: three nodes times the two contacts of the previous terminal pattern.
  state.previousPlanShift = 2;
  EXPECT_NEAR(term->cost(state, allUp), 6.0 * 0.5, 1e-12) << "every node clamped to the previous plan's last node";

  // The clamp is a clamp and not a wrap: with shift 2 the previous plan's node 0 and node 1 are unreachable, so
  // rewriting them cannot change the charge, while rewriting its last node changes all three of them at once.
  MiqpAssignment rewrittenHead = previous;
  setContacts(rewrittenHead, 0, 0, 0);
  setContacts(rewrittenHead, 1, 0, 0);
  ContactLogicState headState = state;
  headState.previousAssignment = &rewrittenHead;
  EXPECT_NEAR(term->cost(headState, allUp), 6.0 * 0.5, 1e-12) << "the nodes before the clamp are never read at shift 2";
  MiqpAssignment rewrittenTail = previous;
  setContacts(rewrittenTail, 2, 0, 0);
  ContactLogicState tailState = state;
  tailState.previousAssignment = &rewrittenTail;
  EXPECT_NEAR(term->cost(tailState, allUp), 0.0, 1e-12) << "every node of the horizon reads that one node";

  // An exact, hand-counted value at the shipped shift of one cycle: node 0 matches the previous node 1, node 1 matches
  // the previous node 2, and node 2 differs from the (clamped) previous node 2 in the right foot alone.
  MiqpAssignment nearlyConsistent(static_cast<size_t>(kNumNodes) * N_CONTACTS, static_cast<std::int8_t>(0));
  setContacts(nearlyConsistent, 0, 0, 1);
  setContacts(nearlyConsistent, 1, 1, 1);
  setContacts(nearlyConsistent, 2, 1, 0);
  state.previousPlanShift = 1;
  EXPECT_NEAR(term->cost(state, nearlyConsistent), 0.5, 1e-12) << "exactly one node-foot pair differs";

  // No usable previous plan: the reference manager leaves previousPlanShift at -1 on the first cycle, after a reset and
  // whenever the stored plan no longer matches the node count, and the term has to contribute nothing then rather than
  // index into a plan that is not there.
  ContactLogicState noShift = state;
  noShift.previousPlanShift = -1;
  EXPECT_NEAR(term->cost(noShift, allUp), 0.0, 1e-12) << "no usable previous plan, no hysteresis";
  ContactLogicState noPlan = state;
  noPlan.previousAssignment = nullptr;
  EXPECT_NEAR(term->cost(noPlan, allUp), 0.0, 1e-12) << "a shift without a plan must not be dereferenced";
  ContactLogicState defaulted;
  defaulted.input = &standing;
  defaulted.numNodes = kNumNodes;
  EXPECT_NEAR(term->cost(defaulted, allUp), 0.0, 1e-12) << "the defaults of ContactLogicState mean 'no previous plan'";

  // The lower-bound contract, exhaustively, at the shift where the clamp is active for the last node.
  state.previousPlanShift = 1;
  const std::vector<MiqpAssignment> partials = allPartialAssignments(kNumNodes * static_cast<int>(N_CONTACTS));
  ASSERT_EQ(partials.size(), 729u);
  int numStrictBounds = 0;
  for (const MiqpAssignment& partial : partials) {
    const scalar_t bound = term->cost(state, partial);
    EXPECT_GE(bound, 0.0) << printAssignment(partial);
    const std::vector<MiqpAssignment> completions = allCompletionsOf(partial);
    for (const MiqpAssignment& complete : completions) {
      const scalar_t exact = term->cost(state, complete);
      ASSERT_LE(bound, exact + 1e-12) << "the partial assignment " << printAssignment(partial) << " is charged " << bound
                                      << ", more than its completion " << printAssignment(complete) << " at " << exact;
      if (exact > bound + 1e-12) ++numStrictBounds;
    }
  }
  EXPECT_GT(numStrictBounds, 0);

  // Exactness on a complete assignment: the charge is the number of disagreeing node-foot pairs, counted here by an
  // independent walk over the assignment that applies the shift and the clamp itself.
  for (const MiqpAssignment& partial : partials) {
    if (std::find(partial.begin(), partial.end(), kMiqpFree) != partial.end()) continue;
    int expected = 0;
    for (int node = 0; node < kNumNodes; ++node) {
      const int source = std::min(node + 1, kNumNodes - 1);
      for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
        const size_t here = static_cast<size_t>(ContactLogicState::contactBinaryIndex(node, foot));
        const size_t there = static_cast<size_t>(ContactLogicState::contactBinaryIndex(source, foot));
        if (partial[here] != previous[there]) ++expected;
      }
    }
    ASSERT_NEAR(term->cost(state, partial), 0.5 * static_cast<scalar_t>(expected), 1e-12) << printAssignment(partial);
  }
}

/*================================ previous foothold consistency and step width ===========================*/
namespace {

/**
 * A previous plan whose footholds are a distinct, easily recognisable number per node, foot and axis, so that reading
 * the wrong node shows up as a wrong number rather than as a near miss of two neighbouring footholds.
 */
ContactPlan makePreviousPlan(int numNodes) {
  ContactPlan plan;
  plan.valid = true;
  plan.startTime = 0.0;
  plan.dt = 0.1;
  plan.footholds.resize(static_cast<size_t>(numNodes) + 1u);
  for (int node = 0; node <= numNodes; ++node) {
    const scalar_t s = static_cast<scalar_t>(node);
    plan.footholds[static_cast<size_t>(node)][0] = vector2_t(0.10 * s, 0.125 + 0.01 * s);
    plan.footholds[static_cast<size_t>(node)][1] = vector2_t(0.10 * s + 0.05, -0.125 - 0.01 * s);
  }
  return plan;
}

/** A nominal trajectory over numNodes + 1 nodes with a turning heading, for the frame terms of the heading model. */
HeadingNominal makeTurningNominal(int numNodes, scalar_t headingAtStart, scalar_t headingPerNode) {
  HeadingNominal nominal;
  nominal.heading.resize(static_cast<size_t>(numNodes) + 1u);
  nominal.com.resize(static_cast<size_t>(numNodes) + 1u);
  nominal.feet.resize(static_cast<size_t>(numNodes) + 1u);
  for (int node = 0; node <= numNodes; ++node) {
    const scalar_t s = static_cast<scalar_t>(node);
    nominal.heading[static_cast<size_t>(node)] = headingAtStart + headingPerNode * s;
    nominal.com[static_cast<size_t>(node)] = vector2_t(0.05 * s, 0.0);
    nominal.feet[static_cast<size_t>(node)][0] = vector2_t(0.30 + 0.02 * s, 0.13 + 0.01 * s);
    nominal.feet[static_cast<size_t>(node)][1] = vector2_t(0.26 + 0.02 * s, -0.11 - 0.01 * s);
  }
  return nominal;
}

}  // namespace

/**
 * PreviousFootholdConsistencyCost is the foothold half of the hysteresis between two planning cycles: it pulls every
 * node's footholds towards the place the previous plan put them, at the node of the previous plan that is live now.
 * Nothing pinned its node arithmetic. The shift is the number of nodes the horizon has advanced since the previous
 * plan started, so the current node k corresponds to the previous plan's node k + shift; the last `shift` nodes of the
 * current horizon reach past the end of the previous plan and are clamped to its final node, which is what makes the
 * tail of the horizon hold still rather than be released the moment it runs off the end.
 *
 * The residual is per foot and per axis, so the assembled stage is diagonal in the four foothold states, and the check
 * below reads the whole expansion - the 2w on the diagonal of Q, the -2 w p_prev in q and the w p_prev^2 in the stage
 * constant - which together are the statement "this node's cost is w ||p - p_prev||^2" and nothing else.
 */
TEST(ContactPlanningTerms, PreviousFootholdConsistencyCostPullsEachNodeToTheShiftedAndClampedPreviousFoothold) {
  constexpr scalar_t kWeight = 3.0;
  constexpr int kShift = 4;
  ContactPlanningConfig config = makeConfig();
  config.previousFootholdConsistency.weight = kWeight;
  config.validate();
  const Layout layout = LipContactPlanner::makeLayout(config);
  ASSERT_FALSE(layout.hasHeading) << "the flat planner, where the term is a plain quadratic on the foothold states";
  std::unique_ptr<LipCost> term = ContactPlanningTermFactory::makeCost(term::kPreviousFootholdConsistency);
  ASSERT_NE(term, nullptr);
  term->bind(layout);
  term->configure(config);
  EXPECT_EQ(term->nodeSet(), NodeSet::ALL);
  EXPECT_NE(term->describe().find("p_{i,prev}(k + shift)"), std::string::npos) << term->describe();

  const int numNodes = config.planner.numNodes;
  ASSERT_GT(numNodes, kShift) << "otherwise the unclamped half of the horizon below is empty";
  const ContactPlannerInput input = makeStandingInput();
  const ContactPlan previousPlan = makePreviousPlan(numNodes);
  ASSERT_EQ(previousPlan.footholds.size(), static_cast<size_t>(numNodes) + 1u);
  const HeadingNominal nominal = makeTurningNominal(numNodes, 0.0, 0.0);

  ContactPlanningContext ctx;
  ctx.input = &input;
  ctx.layout = &layout;
  ctx.nominal = &nominal;
  ctx.config = &config;
  ctx.previousPlan = &previousPlan;
  ctx.previousPlanShift = kShift;
  ctx.dt = config.planner.dt;
  ctx.numNodes = numNodes;
  ctx.omega = config.omega();
  ctx.bigM = config.shared.bigM;
  ctx.computeAxes();

  for (int node = 0; node <= numNodes; ++node) {
    const bool running = node < numNodes;
    OcpQpStage stage = OcpQpStage::Zero(layout.nx, running ? layout.nu : 0, running);
    StageAccumulator accumulator(stage);
    term->addToStage(ctx, node, accumulator);

    const int source = std::min(node + kShift, numNodes);
    scalar_t expectedConstant = 0.0;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      for (int axis = 0; axis < 2; ++axis) {
        const int index = layout.state(axis == 0 ? var::footX(foot) : var::footY(foot));
        const scalar_t target = previousPlan.footholds[static_cast<size_t>(source)][foot](axis);
        EXPECT_NEAR(stage.Q(index, index), 2.0 * kWeight, 1e-12) << "node " << node << ", foot " << foot << ", axis " << axis;
        EXPECT_NEAR(stage.q(index), -2.0 * kWeight * target, 1e-12)
            << "node " << node << " must be pulled to the previous plan's node " << source << ", foot " << foot << ", axis " << axis;
        expectedConstant += kWeight * target * target;
      }
    }
    EXPECT_NEAR(stage.constant, expectedConstant, 1e-12) << "node " << node << ": the full squared residual, offset included";
    EXPECT_NEAR(stage.Q(LIP_CX, LIP_CX), 0.0, 1e-12) << "node " << node << ": the term touches only the foothold states";
    EXPECT_NEAR(stage.Q(LIP_VY, LIP_VY), 0.0, 1e-12) << "node " << node;
    EXPECT_NEAR(stage.Q(LIP_PLX, LIP_PRX), 0.0, 1e-12) << "node " << node << ": the feet are not coupled";
    if (running) {
      EXPECT_TRUE(stage.R.isZero()) << "node " << node << ": the term touches no input";
      EXPECT_TRUE(stage.r.isZero()) << "node " << node;
      EXPECT_TRUE(stage.S.isZero()) << "node " << node;
    }
  }

  // The clamp, stated on its own rather than through the loop's own arithmetic: every node from numNodes - shift on is
  // pulled to the previous plan's final foothold, and the nodes before it are not.
  const std::vector<int> clampedNodes = {numNodes - kShift, numNodes - 1, numNodes};
  const scalar_t lastLeftX = previousPlan.footholds.back()[0].x();
  for (const int node : clampedNodes) {
    OcpQpStage stage = OcpQpStage::Zero(layout.nx, 0, false);
    StageAccumulator accumulator(stage);
    term->addToStage(ctx, node, accumulator);
    EXPECT_NEAR(stage.q(LIP_PLX), -2.0 * kWeight * lastLeftX, 1e-12) << "node " << node << " reaches past the previous plan";
  }
  {
    OcpQpStage stage = OcpQpStage::Zero(layout.nx, 0, false);
    StageAccumulator accumulator(stage);
    term->addToStage(ctx, numNodes - kShift - 1, accumulator);
    EXPECT_GT(std::abs(stage.q(LIP_PLX) + 2.0 * kWeight * lastLeftX), 1e-6)
        << "the last node before the clamp must still read a node of its own";
  }

  // No usable previous plan. previousPlanShift is -1 on the first cycle, after reset() and whenever the stored plan no
  // longer matches the node count, and previousPlan is then null. The term has to emit nothing at all - not a pull
  // towards the origin, which is what a missing guard would produce and which would drag both feet to (0, 0).
  {
    ContactPlanningContext withoutPlan = ctx;
    withoutPlan.previousPlan = nullptr;
    withoutPlan.previousPlanShift = -1;
    OcpQpStage stage = OcpQpStage::Zero(layout.nx, layout.nu, true);
    StageAccumulator accumulator(stage);
    term->addToStage(withoutPlan, 0, accumulator);
    EXPECT_TRUE(stage.Q.isZero()) << "without a previous plan the term contributes nothing";
    EXPECT_TRUE(stage.q.isZero());
    EXPECT_NEAR(stage.constant, 0.0, 1e-15);
  }
  {  // A negative shift with a plan still present is the same "not usable" state and must not index at a negative node.
    ContactPlanningContext staleShift = ctx;
    staleShift.previousPlanShift = -1;
    OcpQpStage stage = OcpQpStage::Zero(layout.nx, layout.nu, true);
    StageAccumulator accumulator(stage);
    term->addToStage(staleShift, 0, accumulator);
    EXPECT_TRUE(stage.Q.isZero());
    EXPECT_TRUE(stage.q.isZero());
  }
  {  // A zero weight switches the term off without removing it from the formulation.
    ContactPlanningConfig unweighted = config;
    unweighted.previousFootholdConsistency.weight = 0.0;
    unweighted.validate();
    std::unique_ptr<LipCost> disabled = ContactPlanningTermFactory::makeCost(term::kPreviousFootholdConsistency);
    disabled->bind(layout);
    disabled->configure(unweighted);
    OcpQpStage stage = OcpQpStage::Zero(layout.nx, layout.nu, true);
    StageAccumulator accumulator(stage);
    disabled->addToStage(ctx, 0, accumulator);
    EXPECT_TRUE(stage.Q.isZero());
    EXPECT_TRUE(stage.q.isZero());
    EXPECT_NEAR(stage.constant, 0.0, 1e-15);
  }
}

/**
 * StepWidthCost appeared in no test anywhere in the repository, which is remarkable for a term that is in the default
 * cost list of every shipped formulation and that is the only thing holding the two feet apart laterally once the
 * heading model is on. It is also the most intricate of the foothold costs, because it is the one that is linearised:
 * the lateral separation e_y(theta) . (p_L - p_R) is not linear in the heading, so the term keeps the axis e_y at the
 * NOMINAL heading of the node and adds the first-order heading term g (theta - theta_n) with
 * g = d e_y / d theta . d_n = -e_x(theta_n) . (p_L,n - p_R,n), which couples five variables - four footholds and the
 * heading - into a single residual.
 *
 * Everything about that is easy to get wrong in a way no smoke test would notice: the sign of the derivative of e_y
 * (it is -e_x, not +e_x), which axis of the frame is differentiated, whether the offset carries -g theta_n so that the
 * term is an expansion AROUND the nominal rather than a term proportional to the absolute heading, and which node's
 * nominal is read. The residual and its gradient are therefore checked here against numbers written out by hand at a
 * heading of 0.5 rad, where every one of those mistakes changes the answer.
 */
TEST(ContactPlanningTerms, StepWidthCostCouplesTheFourFootholdsAndTheHeadingAtANonZeroHeading) {
  constexpr scalar_t kWeight = 7.0;
  constexpr scalar_t kNominalWidth = 0.24;
  constexpr int kNode = 3;
  ContactPlanningConfig config = makeHeadingConfig();
  config.stepWidth.weight = kWeight;
  config.stepWidth.nominalStepWidth = kNominalWidth;
  config.validate();
  const Layout layout = LipContactPlanner::makeLayout(config);
  ASSERT_TRUE(layout.hasHeading);
  std::unique_ptr<LipCost> term = ContactPlanningTermFactory::makeCost(term::kStepWidth);
  ASSERT_NE(term, nullptr);
  term->bind(layout);
  term->configure(config);
  EXPECT_EQ(term->nodeSet(), NodeSet::ALL);
  EXPECT_NE(term->describe().find("e_y . (p_L - p_R)"), std::string::npos) << term->describe();

  const int numNodes = config.planner.numNodes;
  const ContactPlannerInput input = makeStandingInput();
  // heading 0.2 at node 0 and +0.1 per node, so node 3 sits at exactly 0.5 rad and no two nodes share a frame.
  const HeadingNominal nominal = makeTurningNominal(numNodes, 0.2, 0.1);
  ContactPlanningContext ctx;
  ctx.input = &input;
  ctx.layout = &layout;
  ctx.nominal = &nominal;
  ctx.config = &config;
  ctx.previousPlan = nullptr;
  ctx.previousPlanShift = -1;
  ctx.yawInertia = input.yawInertia;
  ctx.dt = config.planner.dt;
  ctx.numNodes = numNodes;
  ctx.omega = config.omega();
  ctx.bigM = config.shared.bigM;
  ctx.computeAxes();

  // The hand computation. theta_n = 0.5, d_n = p_L,n - p_R,n = (0.36, 0.16) - (0.32, -0.14) = (0.04, 0.30).
  const scalar_t theta = nominal.heading[static_cast<size_t>(kNode)];
  ASSERT_NEAR(theta, 0.5, 1e-12);
  const vector2_t nominalSeparation = nominal.feet[static_cast<size_t>(kNode)][0] - nominal.feet[static_cast<size_t>(kNode)][1];
  ASSERT_NEAR(nominalSeparation.x(), 0.04, 1e-12);
  ASSERT_NEAR(nominalSeparation.y(), 0.30, 1e-12);
  const vector2_t ey(-std::sin(theta), std::cos(theta));
  // g = (d e_y / d theta) . d_n = -e_x(theta_n) . d_n = -(cos 0.5 * 0.04 + sin 0.5 * 0.30).
  const scalar_t g = -(std::cos(theta) * nominalSeparation.x() + std::sin(theta) * nominalSeparation.y());
  ASSERT_NEAR(g, -0.178930964056875812, 1e-12) << "-(0.8775825618903728 * 0.04 + 0.4794255386042030 * 0.30)";
  // The residual is e_y . (p_L - p_R) + g (theta - theta_n) - w_nom, so the constant part is -w_nom - g theta_n.
  const scalar_t offset = -kNominalWidth - g * theta;
  ASSERT_NEAR(offset, -0.150534517971562094, 1e-12) << "-0.24 + 0.178930964056875812 * 0.5";

  vector_t coefficients = vector_t::Zero(layout.nx);
  coefficients(layout.state(var::footX(0))) = ey.x();
  coefficients(layout.state(var::footY(0))) = ey.y();
  coefficients(layout.state(var::footX(1))) = -ey.x();
  coefficients(layout.state(var::footY(1))) = -ey.y();
  coefficients(layout.heading) = g;

  OcpQpStage stage = OcpQpStage::Zero(layout.nx, layout.nu, true);
  StageAccumulator accumulator(stage);
  term->addToStage(ctx, kNode, accumulator);

  // The expansion, entry by entry. Q = 2 w l l', q = 2 w c l, constant = w c^2.
  const matrix_t expectedQ = 2.0 * kWeight * coefficients * coefficients.transpose();
  const vector_t expectedq = 2.0 * kWeight * offset * coefficients;
  EXPECT_TRUE(stage.Q.isApprox(expectedQ, 1e-12)) << "Q =\n" << stage.Q << "\nexpected\n" << expectedQ;
  EXPECT_TRUE(stage.q.isApprox(expectedq, 1e-12)) << "q = " << stage.q.transpose() << ", expected " << expectedq.transpose();
  EXPECT_NEAR(stage.constant, kWeight * offset * offset, 1e-12);
  EXPECT_NEAR(stage.Q(LIP_PLY, LIP_PLY), 2.0 * kWeight * ey.y() * ey.y(), 1e-12);
  EXPECT_NEAR(stage.Q(LIP_PLY, LIP_PRY), -2.0 * kWeight * ey.y() * ey.y(), 1e-12) << "the two feet enter with opposite signs";
  EXPECT_NEAR(stage.Q(LIP_PLX, layout.heading), 2.0 * kWeight * ey.x() * g, 1e-12) << "the heading is coupled to every foothold";
  EXPECT_NEAR(stage.Q(layout.heading, layout.heading), 2.0 * kWeight * g * g, 1e-12);
  EXPECT_NEAR(stage.q(layout.heading), 2.0 * kWeight * offset * g, 1e-12);
  EXPECT_NEAR(stage.Q(LIP_CX, LIP_CX), 0.0, 1e-12) << "the centre of mass is not part of the step width";
  EXPECT_NEAR(stage.Q(layout.headingRate, layout.headingRate), 0.0, 1e-12) << "nor is the heading rate";
  EXPECT_TRUE(stage.R.isZero()) << "the term touches no input";
  EXPECT_TRUE(stage.r.isZero());
  EXPECT_TRUE(stage.S.isZero());

  // The residual and its gradient at a state that is not the nominal, which is the statement the entries above add up
  // to: the stage evaluates to w r^2 and its gradient is 2 w r l, with r written out from the formula of the term.
  vector_t state = vector_t::Zero(layout.nx);
  state(layout.state(var::footX(0))) = 0.31;
  state(layout.state(var::footY(0))) = 0.14;
  state(layout.state(var::footX(1))) = 0.27;
  state(layout.state(var::footY(1))) = -0.12;
  state(layout.heading) = 0.55;
  const vector2_t separation(state(layout.state(var::footX(0))) - state(layout.state(var::footX(1))),
                             state(layout.state(var::footY(0))) - state(layout.state(var::footY(1))));
  const scalar_t residual = ey.dot(separation) + g * (state(layout.heading) - theta) - kNominalWidth;
  ASSERT_GT(std::abs(residual), 1e-3) << "the trial is only meaningful away from the minimum of the term";
  EXPECT_NEAR(0.5 * state.dot(stage.Q * state) + stage.q.dot(state) + stage.constant, kWeight * residual * residual, 1e-12)
      << "the stage must evaluate to w (e_y . (p_L - p_R) + g (theta - theta_n) - w_nom)^2";
  const vector_t gradient = stage.Q * state + stage.q;
  const vector_t expectedGradient = 2.0 * kWeight * residual * coefficients;
  EXPECT_TRUE(gradient.isApprox(expectedGradient, 1e-9))
      << "gradient = " << gradient.transpose() << ", expected 2 w r l = " << expectedGradient.transpose();

  // Each node carries its own frame and its own nominal separation. Node 0 sits at 0.2 rad with d_n = (0.04, 0.24), so
  // reading the wrong node's nominal - a plausible off-by-one in a term that indexes ctx.nominal by node - changes both
  // the axis and g. Checking a second node also pins that the term is not accidentally constant over the horizon.
  const scalar_t thetaZero = nominal.heading[0];
  ASSERT_NEAR(thetaZero, 0.2, 1e-12);
  const vector2_t nominalSeparationZero = nominal.feet[0][0] - nominal.feet[0][1];
  ASSERT_NEAR(nominalSeparationZero.y(), 0.24, 1e-12);
  const vector2_t eyZero(-std::sin(thetaZero), std::cos(thetaZero));
  const scalar_t gZero = -(std::cos(thetaZero) * nominalSeparationZero.x() + std::sin(thetaZero) * nominalSeparationZero.y());
  OcpQpStage stageZero = OcpQpStage::Zero(layout.nx, layout.nu, true);
  StageAccumulator accumulatorZero(stageZero);
  term->addToStage(ctx, 0, accumulatorZero);
  EXPECT_NEAR(stageZero.Q(LIP_PLY, LIP_PLY), 2.0 * kWeight * eyZero.y() * eyZero.y(), 1e-12);
  EXPECT_NEAR(stageZero.Q(layout.heading, layout.heading), 2.0 * kWeight * gZero * gZero, 1e-12);
  EXPECT_NEAR(stageZero.q(layout.heading), 2.0 * kWeight * (-kNominalWidth - gZero * thetaZero) * gZero, 1e-12);
  EXPECT_GT(std::abs(stageZero.Q(layout.heading, layout.heading) - stage.Q(layout.heading, layout.heading)), 1e-6)
      << "two nodes with different nominal headings must not produce the same coupling";

  // Without the heading block the term degenerates to the plain lateral separation in the frame of the measured base
  // yaw, with no heading column and no expansion offset: e_y(yaw) . (p_L - p_R) - w_nom.
  ContactPlanningConfig flatConfig = makeConfig();
  flatConfig.stepWidth.weight = kWeight;
  flatConfig.stepWidth.nominalStepWidth = kNominalWidth;
  flatConfig.validate();
  const Layout flatLayout = LipContactPlanner::makeLayout(flatConfig);
  ASSERT_FALSE(flatLayout.hasHeading);
  ASSERT_EQ(flatLayout.heading, -1);
  std::unique_ptr<LipCost> flatTerm = ContactPlanningTermFactory::makeCost(term::kStepWidth);
  flatTerm->bind(flatLayout);
  flatTerm->configure(flatConfig);
  ContactPlannerInput yawedInput = makeStandingInput();
  yawedInput.yaw = 0.4;
  ContactPlanningContext flatCtx;
  flatCtx.input = &yawedInput;
  flatCtx.layout = &flatLayout;
  flatCtx.nominal = &nominal;
  flatCtx.config = &flatConfig;
  flatCtx.dt = flatConfig.planner.dt;
  flatCtx.numNodes = flatConfig.planner.numNodes;
  flatCtx.omega = flatConfig.omega();
  flatCtx.bigM = flatConfig.shared.bigM;
  flatCtx.computeAxes();
  OcpQpStage flatStage = OcpQpStage::Zero(flatLayout.nx, flatLayout.nu, true);
  StageAccumulator flatAccumulator(flatStage);
  flatTerm->addToStage(flatCtx, kNode, flatAccumulator);
  const vector2_t flatEy(-std::sin(yawedInput.yaw), std::cos(yawedInput.yaw));
  EXPECT_NEAR(flatStage.Q(LIP_PLY, LIP_PLY), 2.0 * kWeight * flatEy.y() * flatEy.y(), 1e-12) << "the frame is the base yaw";
  EXPECT_NEAR(flatStage.Q(LIP_PLX, LIP_PLY), 2.0 * kWeight * flatEy.x() * flatEy.y(), 1e-12);
  EXPECT_NEAR(flatStage.q(LIP_PLY), -2.0 * kWeight * kNominalWidth * flatEy.y(), 1e-12) << "the offset is -w_nom alone";
  EXPECT_NEAR(flatStage.constant, kWeight * kNominalWidth * kNominalWidth, 1e-12);
  EXPECT_EQ(flatLayout.nx, LIP_STATE_DIM) << "no heading column exists for the term to couple to";
}

}  // namespace ocs2::humanoid
