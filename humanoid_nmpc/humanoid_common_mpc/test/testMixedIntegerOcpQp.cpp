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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/LipContactPlanner.h"
#include "humanoid_common_mpc/contact_planning/MixedIntegerOcpQp.h"
#include "humanoid_common_mpc/contact_planning/OcpQpHpipm.h"
#include "humanoid_common_mpc/contact_planning/logic/ContactLogicState.h"
#include "humanoid_common_mpc/contact_planning/problem/Layout.h"

namespace ocs2::humanoid {

namespace {

constexpr int kNumStages = 4;
constexpr scalar_t kStepGain = 0.7;
constexpr scalar_t kDrift = -0.25;
constexpr scalar_t kSwitchCost = 0.05;
/** The QP's own price for switching a decoupled binary on: far smaller than the idle price below, so ones win. */
constexpr scalar_t kIdleTemptation = 0.01;

/**
 * Scalar state x, inputs u = [a (continuous), b (binary)]:
 *   x_{k+1} = x_k + 0.1 a_k + kStepGain b_k + kDrift
 * Cost: 0.5 x_k^2 + 0.5 a_k^2 + kSwitchCost b_k, terminal 5 (x_N - target)^2.
 * Continuous input bounded |a| <= 0.5, so the binaries decide the coarse motion.
 */
OcpQpProblem makeProblem(scalar_t target) {
  OcpQpProblem problem;
  problem.x0 = vector_t::Zero(1);
  problem.stages.resize(kNumStages + 1);
  for (int k = 0; k <= kNumStages; ++k) {
    const bool terminal = (k == kNumStages);
    OcpQpStage& s = problem.stages[k];
    s = OcpQpStage::Zero(1, terminal ? 0 : 2, !terminal);
    s.Q = matrix_t::Constant(1, 1, 1.0);
    if (!terminal) {
      s.R = (matrix_t(2, 2) << 1.0, 0.0, 0.0, 1e-6).finished();
      s.r = (vector_t(2) << 0.0, kSwitchCost).finished();
      s.A = matrix_t::Constant(1, 1, 1.0);
      s.B = (matrix_t(1, 2) << 0.1, kStepGain).finished();
      s.b = vector_t::Constant(1, kDrift);
      s.idxbu = {0, 1};
      s.lbu = (vector_t(2) << -0.5, 0.0).finished();
      s.ubu = (vector_t(2) << 0.5, 1.0).finished();
    } else {
      s.Q = matrix_t::Constant(1, 1, 10.0);
      s.q = vector_t::Constant(1, -10.0 * target);
    }
  }
  return problem;
}

std::vector<MiqpBinaryVariable> makeBinaries() {
  std::vector<MiqpBinaryVariable> binaries;
  for (int k = 0; k < kNumStages; ++k) {
    binaries.push_back({k, 1});
  }
  return binaries;
}

/** Brute force over all 2^N binary assignments using fixed-bound QPs. */
scalar_t bruteForceOptimum(OcpQpProblem problem, MiqpAssignment& bestAssignment, const MiqpPropagateFn& propagate = nullptr) {
  OcpQpHpipmSolver solver;
  scalar_t best = std::numeric_limits<scalar_t>::infinity();
  for (int code = 0; code < (1 << kNumStages); ++code) {
    MiqpAssignment assignment(kNumStages);
    for (int k = 0; k < kNumStages; ++k) {
      assignment[k] = static_cast<std::int8_t>((code >> k) & 1);
    }
    if (propagate) {
      MiqpAssignment copy = assignment;
      if (!propagate(copy)) continue;
    }
    for (int k = 0; k < kNumStages; ++k) {
      problem.stages[k].lbu(1) = assignment[k];
      problem.stages[k].ubu(1) = assignment[k];
    }
    const OcpQpSolution solution = solver.solve(problem);
    if (solution.success() && solution.objective < best) {
      best = solution.objective;
      bestAssignment = assignment;
    }
  }
  return best;
}

/**
 * Brute force including a logical cost on the binaries, which is what the branch-and-bound has to match.
 *
 * The QP cannot see MiqpAssignmentCostFn, so the total objective is the QP objective plus that cost. Any search that
 * prunes or fences on the QP objective alone is wrong exactly here.
 */
scalar_t bruteForceOptimumWithAssignmentCost(OcpQpProblem problem,
                                             const MiqpAssignmentCostFn& assignmentCost,
                                             MiqpAssignment& bestAssignment) {
  OcpQpHpipmSolver solver;
  scalar_t best = std::numeric_limits<scalar_t>::infinity();
  for (int code = 0; code < (1 << kNumStages); ++code) {
    MiqpAssignment assignment(kNumStages);
    for (int k = 0; k < kNumStages; ++k) {
      assignment[k] = static_cast<std::int8_t>((code >> k) & 1);
    }
    for (int k = 0; k < kNumStages; ++k) {
      problem.stages[k].lbu(1) = assignment[k];
      problem.stages[k].ubu(1) = assignment[k];
    }
    const OcpQpSolution solution = solver.solve(problem);
    if (!solution.success()) continue;
    const scalar_t total = solution.objective + assignmentCost(assignment);
    if (total < best) {
      best = total;
      bestAssignment = assignment;
    }
  }
  return best;
}

/**
 * A problem whose binaries are INVISIBLE to the QP except for a small price on switching them on.
 *
 * The binary column of B is zero, so the binaries do not move the state at all, and their only QP cost is the linear
 * `kIdleTemptation` per stage. The relaxation therefore drives every binary to its lower bound and is INTEGRAL at the
 * root, which is precisely the node shape that used to fence off its own subtree. Everything that distinguishes the
 * completions then lives in the assignment cost, where the QP cannot see it.
 */
OcpQpProblem makeDecoupledBinaryProblem(scalar_t target) {
  OcpQpProblem problem;
  problem.x0 = vector_t::Zero(1);
  problem.stages.resize(kNumStages + 1);
  for (int k = 0; k <= kNumStages; ++k) {
    const bool terminal = (k == kNumStages);
    OcpQpStage& s = problem.stages[k];
    s = OcpQpStage::Zero(1, terminal ? 0 : 2, !terminal);
    s.Q = matrix_t::Constant(1, 1, 1.0);
    if (!terminal) {
      s.R = (matrix_t(2, 2) << 1.0, 0.0, 0.0, 1e-6).finished();
      s.r = (vector_t(2) << 0.0, kIdleTemptation).finished();
      s.A = matrix_t::Constant(1, 1, 1.0);
      s.B = (matrix_t(1, 2) << 0.1, 0.0).finished();  // the binary column is zero: it cannot move the state
      s.b = vector_t::Constant(1, kDrift);
      s.idxbu = {0, 1};
      s.lbu = (vector_t(2) << -0.5, 0.0).finished();
      s.ubu = (vector_t(2) << 0.5, 1.0).finished();
    } else {
      s.Q = matrix_t::Constant(1, 1, 10.0);
      s.q = vector_t::Constant(1, -10.0 * target);
    }
  }
  return problem;
}

/**
 * A non-negative price on every binary left at 0, heavier at the later stages.
 *
 * Non-negativity is what makes "count only what is decided" a valid LOWER bound on a partial assignment, which is the
 * contract MixedIntegerOcpQp.h states. The weights rise with the stage index so that two completions of the same node
 * differ in logical cost - the situation the fence used to discard.
 */
scalar_t idlePricePerStage(const MiqpAssignment& assignment) {
  scalar_t cost = 0.0;
  for (std::size_t i = 0; i < assignment.size(); ++i) {
    if (assignment[i] == 0) {
      cost += 0.35 * static_cast<scalar_t>(i + 1);
    }
  }
  return cost;
}

/**
 * A propagation that completes any assignment whose first binary is fixed: every later binary alternates with its
 * predecessor, and a fixing that contradicts the alternation is rejected.
 *
 * This is the shape the contact planner's own rules have - alternating_feet and phase_durations fix long runs of later
 * binaries from a few early ones - and it is what makes the distinction below observable: the assignment `solveFixed`
 * prices and applies is the one the propagation produced, not the partial one the caller handed in.
 */
bool alternateFromTheFirstBinary(MiqpAssignment& assignment) {
  if (assignment.empty() || assignment[0] == kMiqpFree) {
    return false;
  }
  for (std::size_t k = 1; k < assignment.size(); ++k) {
    const std::int8_t implied = static_cast<std::int8_t>(1 - assignment[k - 1]);
    if (assignment[k] == kMiqpFree) {
      assignment[k] = implied;
    } else if (assignment[k] != implied) {
      return false;
    }
  }
  return true;
}

/**
 * The configuration the binary-layout checks below run the planner on.
 *
 * It mirrors the fixture of the term tests rather than inventing a formulation of its own, so that what is pinned is
 * the layout of the shipped point-mass planner: the mandatory lip_com and foothold_integrator blocks, in that order.
 */
ContactPlanningConfig makePlannerConfig() {
  ContactPlanningConfig config;
  config.planner.dt = 0.1;
  config.planner.numNodes = 12;
  config.planner.commitTime = 0.0;
  config.shared.comHeight = 0.85;
  config.shared.gaitLimits.minSwingDuration = 0.3;
  config.shared.gaitLimits.maxSwingDuration = 0.5;
  config.shared.gaitLimits.minContactDuration = 0.15;
  config.shared.gaitLimits.minDoubleSupportDuration = 0.1;
  config.validate();
  return config;
}

/**
 * The same formulation with the heading model, which appends a third model block with four more inputs per node.
 *
 * The heading block declares no binaries, so the binary list must come out exactly as it does without it. That is the
 * half of the agreement a new model block could break without any other test noticing: `contactBinaryIndex` hard-codes
 * N_CONTACTS binaries per node in the feet's own order, while `binaryVariables` concatenates whatever the blocks
 * declare, in the blocks' order.
 */
ContactPlanningConfig makeHeadingPlannerConfig() {
  ContactPlanningConfig config = makePlannerConfig();
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

/** The foot a layout input index belongs to, or -1 when the index is not a contact binary of any foot. */
int footOfContactInput(const Layout& layout, int inputIndex) {
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    if (layout.input(var::contact(foot)) == inputIndex) {
      return static_cast<int>(foot);
    }
  }
  return -1;
}

/**
 * Checks, for one configuration, that the two independent namings of a contact binary denote the same variable.
 *
 * `ContactPlanningProblem::binaryVariables()` builds the branching list the mixed-integer solver indexes an assignment
 * by: node by node, and within a node block by block over `LipModelBlock::binaryInputs()`. Every logic rule, every
 * assignment cost, the committed prefix of `initialAssignment` and the event-shift local search instead address the
 * same assignment through `ContactLogicState::contactBinaryIndex(node, foot)`, which assumes N_CONTACTS entries per
 * node laid out in the feet's order. Nothing in the code connects the two, so this states the connection.
 */
void expectBinaryLayoutAgreesWithContactBinaryIndex(const ContactPlanningConfig& config) {
  const LipContactPlanner planner(config);
  const Layout& layout = planner.getLayout();
  const std::vector<MiqpBinaryVariable> binaries = planner.binaryVariables();
  const int numNodes = config.planner.numNodes;

  ASSERT_EQ(binaries.size(), static_cast<std::size_t>(ContactLogicState::kBinariesPerNode) * static_cast<std::size_t>(numNodes))
      << "the branching list must hold exactly the contact binaries contactBinaryIndex addresses";

  // Forward: the index contactBinaryIndex hands out is the position of that node's and that foot's contact input.
  for (int k = 0; k < numNodes; ++k) {
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      const int index = ContactLogicState::contactBinaryIndex(k, foot);
      ASSERT_GE(index, 0);
      ASSERT_LT(static_cast<std::size_t>(index), binaries.size());
      const MiqpBinaryVariable& binary = binaries[static_cast<std::size_t>(index)];
      EXPECT_EQ(binary.stage, k) << "node " << k << ", foot " << foot;
      EXPECT_EQ(binary.inputIndex, layout.input(var::contact(foot))) << "node " << k << ", foot " << foot;
    }
  }

  // Backward: no entry of the branching list is anything but the contact input contactBinaryIndex would put there.
  for (std::size_t i = 0; i < binaries.size(); ++i) {
    const MiqpBinaryVariable& binary = binaries[i];
    const int foot = footOfContactInput(layout, binary.inputIndex);
    ASSERT_GE(foot, 0) << "binary " << i << " is input " << binary.inputIndex << ", which is not a contact of any foot";
    EXPECT_EQ(static_cast<int>(i), ContactLogicState::contactBinaryIndex(binary.stage, static_cast<size_t>(foot)))
        << "binary " << i << " sits at stage " << binary.stage << " of foot " << foot;
  }

  // Every listed binary must also carry an input box constraint at its stage, which is what MixedIntegerOcpQp needs to
  // fix it (locateBinaries throws otherwise, and the plan is then lost for that frame).
  const ContactPlannerInput input = makeStandingInput();
  const OcpQpProblem qp = planner.buildProblem(input);
  ASSERT_EQ(qp.numStages(), numNodes);
  for (const MiqpBinaryVariable& binary : binaries) {
    const OcpQpStage& stage = qp.stages[static_cast<std::size_t>(binary.stage)];
    EXPECT_TRUE(std::find(stage.idxbu.begin(), stage.idxbu.end(), binary.inputIndex) != stage.idxbu.end())
        << "input " << binary.inputIndex << " of stage " << binary.stage << " has no box constraint to fix";
  }

  // The logic state sizes the assignment the rules read; it must be the same assignment the solver branches on.
  const ContactLogicState logicState = planner.makeLogicState(input);
  EXPECT_EQ(logicState.numBinaries(), static_cast<int>(binaries.size()));
}

}  // namespace

TEST(MixedIntegerOcpQpTest, FindsBruteForceOptimum) {
  for (const scalar_t target : {0.0, 0.6, 1.3, 2.5}) {
    OcpQpProblem problem = makeProblem(target);
    MiqpAssignment bruteForceAssignment;
    const scalar_t bruteForce = bruteForceOptimum(problem, bruteForceAssignment);
    ASSERT_TRUE(std::isfinite(bruteForce));

    MixedIntegerOcpQp miqp{OcpQpHpipmSolver::Settings{}, MiqpSettings{}};
    const MiqpAssignment initial(kNumStages, kMiqpFree);
    const MiqpResult result = miqp.solve(problem, makeBinaries(), initial, nullptr);
    ASSERT_TRUE(result.hasIncumbent) << "target " << target;
    EXPECT_TRUE(result.optimal);
    EXPECT_NEAR(result.incumbentObjective, bruteForce, 1e-5) << "target " << target;
    EXPECT_LE(result.rootBound, result.incumbentObjective + 1e-6);
    for (int k = 0; k < kNumStages; ++k) {
      EXPECT_NEAR(result.solution.u[k](1), static_cast<scalar_t>(result.assignment[k]), 1e-4);
    }
  }
}

TEST(MixedIntegerOcpQpTest, RespectsInitialFixingsAndPropagation) {
  OcpQpProblem problem = makeProblem(1.3);
  // Propagation rule: no two consecutive steps (b_k = 1 implies b_{k+1} = 0), and b_0 is fixed to 1 for the whole search.
  const MiqpPropagateFn noConsecutive = [](MiqpAssignment& a) {
    for (std::size_t k = 0; k + 1 < a.size(); ++k) {
      if (a[k] == 1) {
        if (a[k + 1] == 1) return false;
        a[k + 1] = 0;
      }
    }
    return true;
  };
  MiqpAssignment initial(kNumStages, kMiqpFree);
  initial[0] = 1;

  const MiqpPropagateFn bruteForceRule = [&](MiqpAssignment& a) { return a[0] == 1 && noConsecutive(a); };
  MiqpAssignment bruteForceAssignment;
  const scalar_t bruteForce = bruteForceOptimum(problem, bruteForceAssignment, bruteForceRule);
  ASSERT_TRUE(std::isfinite(bruteForce));

  MixedIntegerOcpQp miqp{OcpQpHpipmSolver::Settings{}, MiqpSettings{}};
  const MiqpResult result = miqp.solve(problem, makeBinaries(), initial, noConsecutive);
  ASSERT_TRUE(result.hasIncumbent);
  EXPECT_TRUE(result.optimal);
  EXPECT_NEAR(result.incumbentObjective, bruteForce, 1e-5);
  EXPECT_EQ(result.assignment[0], 1);
  for (int k = 0; k + 1 < kNumStages; ++k) {
    EXPECT_FALSE(result.assignment[k] == 1 && result.assignment[k + 1] == 1);
  }
}

TEST(MixedIntegerOcpQpTest, WarmStartProvidesIncumbentEvenWithZeroNodeBudget) {
  OcpQpProblem problem = makeProblem(1.3);
  MiqpSettings settings;
  settings.maxNodes = 1;  // the warm-start solve consumes the whole budget
  MixedIntegerOcpQp miqp{OcpQpHpipmSolver::Settings{}, settings};
  const MiqpAssignment initial(kNumStages, kMiqpFree);
  const MiqpAssignment warm = {1, 0, 1, 0};
  const MiqpResult result = miqp.solve(problem, makeBinaries(), initial, nullptr, &warm);
  ASSERT_TRUE(result.hasIncumbent);
  EXPECT_TRUE(result.nodeLimitHit);
  EXPECT_FALSE(result.optimal);
  EXPECT_EQ(result.assignment, warm);
}

TEST(MixedIntegerOcpQpTest, InfeasibleRootPropagationReturnsNoIncumbent) {
  OcpQpProblem problem = makeProblem(1.3);
  MixedIntegerOcpQp miqp{OcpQpHpipmSolver::Settings{}, MiqpSettings{}};
  const MiqpAssignment initial(kNumStages, kMiqpFree);
  const MiqpResult result = miqp.solve(problem, makeBinaries(), initial, [](MiqpAssignment&) { return false; });
  EXPECT_FALSE(result.hasIncumbent);
  EXPECT_EQ(result.numNodes, 0);
}

/**
 * A relaxation the QP solver does not solve (iteration limit) is dropped with its subtree, but nothing about it is
 * proven: it is not an infeasible node, and a search that dropped one cannot claim to have exhausted the tree.
 */
TEST(MixedIntegerOcpQpTest, FailedRelaxationsAreNotReportedOptimal) {
  OcpQpProblem problem = makeProblem(1.3);
  OcpQpHpipmSolver::Settings starved;
  starved.iterMax = 1;  // no relaxation converges in a single interior point iteration
  MixedIntegerOcpQp miqp{starved, MiqpSettings{}};
  const MiqpAssignment initial(kNumStages, kMiqpFree);
  const MiqpResult result = miqp.solve(problem, makeBinaries(), initial, [](MiqpAssignment&) { return true; });
  EXPECT_GT(result.numFailedRelaxations, 0);
  EXPECT_EQ(result.numInfeasible, 0) << "an unsolved relaxation is not an infeasible node";
  EXPECT_FALSE(result.optimal);
  EXPECT_FALSE(result.hasIncumbent);
}

/**
 * An integral relaxation must not fence off its own subtree when a logical cost is in play.
 *
 * With R(1,1) = 1e-6 the binaries enter the relaxation almost linearly, so the relaxation sits on a vertex of the box
 * and is integral at the root itself. The search used to take that rounding as an incumbent and `break`, closing the
 * whole tree. That is sound only when the completion attains the node's lower bound - and the node's bound carries
 * logicalCost(PARTIAL), which the contract makes a lower bound on logicalCost(complete). A sibling with a worse QP
 * objective and a smaller idle price is then strictly better and was never looked at, while the result still claimed
 * to be optimal.
 */
TEST(MixedIntegerOcpQpTest, AnIntegralRelaxationDoesNotFenceOffABetterCompletion) {
  const MiqpAssignmentCostFn assignmentCost = &idlePricePerStage;

  for (const scalar_t target : {0.0, 0.6, 1.3, 2.5}) {
    OcpQpProblem problem = makeDecoupledBinaryProblem(target);
    MiqpAssignment bruteForceAssignment;
    const scalar_t bruteForce = bruteForceOptimumWithAssignmentCost(problem, assignmentCost, bruteForceAssignment);
    ASSERT_TRUE(std::isfinite(bruteForce)) << "target " << target;
    // The idle price dwarfs the QP's own price for a one, so the optimum is every binary ON - the exact opposite of
    // what the integral relaxation hands the search.
    for (int k = 0; k < kNumStages; ++k) {
      ASSERT_EQ(bruteForceAssignment[k], 1) << "target " << target << ", stage " << k;
    }

    MixedIntegerOcpQp miqp{OcpQpHpipmSolver::Settings{}, MiqpSettings{}};
    const MiqpAssignment initial(kNumStages, kMiqpFree);
    const MiqpResult result = miqp.solve(problem, makeBinaries(), initial, nullptr, nullptr, assignmentCost);

    ASSERT_TRUE(result.hasIncumbent) << "target " << target;
    EXPECT_NEAR(result.incumbentObjective, bruteForce, 1e-5)
        << "target " << target << ": the search settled for a completion the assignment cost makes suboptimal";
    for (int k = 0; k < kNumStages; ++k) {
      EXPECT_EQ(result.assignment[k], 1) << "target " << target << ", stage " << k;
    }
    EXPECT_TRUE(result.optimal) << "target " << target;
    // The reported objective must be the TOTAL, not the QP part alone.
    EXPECT_NEAR(result.incumbentObjective - idlePricePerStage(result.assignment), result.solution.objective, 1e-5) << "target " << target;
  }
}

/** Without an assignment cost the fence is sound, and the fast path must be preserved exactly. */
TEST(MixedIntegerOcpQpTest, AnIntegralRelaxationStillClosesItsSubtreeWithoutAnAssignmentCost) {
  OcpQpProblem problem = makeProblem(1.3);
  MiqpAssignment bruteForceAssignment;
  const scalar_t bruteForce = bruteForceOptimum(problem, bruteForceAssignment);

  MixedIntegerOcpQp miqp{OcpQpHpipmSolver::Settings{}, MiqpSettings{}};
  const MiqpAssignment initial(kNumStages, kMiqpFree);
  const MiqpResult result = miqp.solve(problem, makeBinaries(), initial, nullptr);
  ASSERT_TRUE(result.hasIncumbent);
  EXPECT_TRUE(result.optimal);
  EXPECT_NEAR(result.incumbentObjective, bruteForce, 1e-5);
  // The fence still closes subtrees: the search settles in a handful of relaxations instead of walking the 2^4 tree.
  // The exact count is an implementation detail, so this pins only that the fast path did not turn into an exhaustive
  // enumeration; AZeroAssignmentCostMatchesNoAssignmentCost below is what pins it exactly.
  EXPECT_LT(result.numNodes, 1 << kNumStages) << "the integral-relaxation fast path should still close subtrees";
}

/** A zero assignment cost must behave exactly like no assignment cost at all. */
TEST(MixedIntegerOcpQpTest, AZeroAssignmentCostMatchesNoAssignmentCost) {
  OcpQpProblem withCost = makeProblem(1.3);
  OcpQpProblem without = makeProblem(1.3);
  const MiqpAssignment initial(kNumStages, kMiqpFree);

  MixedIntegerOcpQp a{OcpQpHpipmSolver::Settings{}, MiqpSettings{}};
  const MiqpResult costed = a.solve(withCost, makeBinaries(), initial, nullptr, nullptr, [](const MiqpAssignment&) { return 0.0; });
  MixedIntegerOcpQp b{OcpQpHpipmSolver::Settings{}, MiqpSettings{}};
  const MiqpResult plain = b.solve(without, makeBinaries(), initial, nullptr);

  ASSERT_TRUE(costed.hasIncumbent);
  ASSERT_TRUE(plain.hasIncumbent);
  EXPECT_NEAR(costed.incumbentObjective, plain.incumbentObjective, 1e-9);
  EXPECT_EQ(costed.assignment, plain.assignment);
  EXPECT_EQ(costed.numNodes, plain.numNodes);
}

/*=========================== solveFixed: the contract the afterSearch stages decide on ====================*/

/**
 * `solveFixed` is the single QP entry point of every stage that runs after the branch-and-bound - the event-shift local
 * search, the cadence stretch and the heading re-linearisation - and all three of them take its `objective` and compare
 * it against `MiqpResult::incumbentObjective`, which is the QP objective PLUS the assignment cost of the incumbent. A
 * `solveFixed` that reported the QP objective alone would therefore look cheaper than the incumbent by exactly the
 * incumbent's logical cost, and the stages would accept every candidate they evaluated: the local search would walk the
 * contact events wherever it looked last, the cadence stretch would always stretch, and the re-linearisation would
 * "improve" the objective on a problem it did not change. Nothing in the three stages re-adds the cost, so it has to be
 * in here, and it has to be the cost of the assignment that was actually solved.
 */
TEST(MixedIntegerOcpQpSolveFixedTest, ReportsTheQpObjectivePlusTheAssignmentCost) {
  const MiqpAssignment assignment = {1, 0, 1, 0};
  const MiqpAssignmentCostFn assignmentCost = &idlePricePerStage;
  MixedIntegerOcpQp miqp{OcpQpHpipmSolver::Settings{}, MiqpSettings{}};

  OcpQpProblem plainProblem = makeProblem(1.3);
  OcpQpSolution plainSolution;
  scalar_t plainObjective = -1.0;
  ASSERT_TRUE(miqp.solveFixed(plainProblem, makeBinaries(), assignment, nullptr, nullptr, plainSolution, plainObjective));
  EXPECT_EQ(plainObjective, plainSolution.objective) << "without a logical cost the reported objective is the QP's own";

  OcpQpProblem costedProblem = makeProblem(1.3);
  OcpQpSolution costedSolution;
  scalar_t costedObjective = -1.0;
  ASSERT_TRUE(miqp.solveFixed(costedProblem, makeBinaries(), assignment, nullptr, assignmentCost, costedSolution, costedObjective));
  // The QP never sees the assignment cost, so the continuous part of the two solves must be identical...
  EXPECT_NEAR(costedSolution.objective, plainSolution.objective, 1e-6);
  // ...and the whole difference between the two reported objectives is the logical cost.
  const scalar_t logicalCost = idlePricePerStage(assignment);
  ASSERT_GT(logicalCost, 0.0) << "the fixture has to charge something, or the test cannot tell the two apart";
  EXPECT_NEAR(costedObjective, costedSolution.objective + logicalCost, 1e-12);
  EXPECT_NEAR(costedObjective - plainObjective, logicalCost, 1e-6);

  // The binaries the caller listed were fixed to the assignment, both in the problem handed in and in the solution.
  for (int k = 0; k < kNumStages; ++k) {
    EXPECT_EQ(plainProblem.stages[k].lbu(1), static_cast<scalar_t>(assignment[k])) << "stage " << k;
    EXPECT_EQ(plainProblem.stages[k].ubu(1), static_cast<scalar_t>(assignment[k])) << "stage " << k;
    EXPECT_NEAR(plainSolution.u[k](1), static_cast<scalar_t>(assignment[k]), 1e-6) << "stage " << k;
  }
}

/**
 * Every path that returns false must leave `objective` at +infinity rather than at whatever the caller had there.
 *
 * The three afterSearch stages declare their own `scalar_t objective = 0.0` next to the call and only look at it after
 * a true return, so today they are safe by construction; but zero is the most natural value to initialise it with and
 * it is smaller than any real objective of these problems, so a failure path that left the out-parameter untouched
 * would hand a caller who forgot the return value a candidate that beats every incumbent. Stating it here is what makes
 * the out-parameter safe to read unconditionally, which is the only reason a bool-plus-out-parameter signature is
 * tolerable at all.
 */
TEST(MixedIntegerOcpQpSolveFixedTest, EveryFailurePathLeavesTheObjectiveAtInfinity) {
  const scalar_t infinity = std::numeric_limits<scalar_t>::infinity();
  const MiqpAssignment complete = {1, 0, 1, 0};
  const MiqpAssignmentCostFn assignmentCost = &idlePricePerStage;

  // (1) The propagation rejects the assignment: a candidate the logic rules prove infeasible never reaches the QP.
  {
    MixedIntegerOcpQp miqp{OcpQpHpipmSolver::Settings{}, MiqpSettings{}};
    OcpQpProblem problem = makeProblem(1.3);
    OcpQpSolution solution;
    scalar_t objective = -1.0;
    EXPECT_FALSE(
        miqp.solveFixed(problem, makeBinaries(), complete, [](MiqpAssignment&) { return false; }, assignmentCost, solution, objective));
    EXPECT_EQ(objective, infinity);
  }

  // (2) The assignment is still incomplete after the propagation. `solveFixed` fixes every binary's box bounds to its
  // assigned value, so a free entry would be solved as a relaxation - a fractional contact - and priced as if it were
  // a schedule. Refusing it is the only correct answer.
  {
    MixedIntegerOcpQp miqp{OcpQpHpipmSolver::Settings{}, MiqpSettings{}};
    OcpQpProblem problem = makeProblem(1.3);
    const MiqpAssignment partial = {1, kMiqpFree, 1, 0};
    OcpQpSolution solution;
    scalar_t objective = -1.0;
    EXPECT_FALSE(miqp.solveFixed(problem, makeBinaries(), partial, nullptr, assignmentCost, solution, objective))
        << "without a propagation there is nothing to fill the hole";
    EXPECT_EQ(objective, infinity);

    objective = -1.0;
    EXPECT_FALSE(miqp.solveFixed(
        problem, makeBinaries(), partial, [](MiqpAssignment&) { return true; }, assignmentCost, solution, objective))
        << "a propagation that accepts the assignment without completing it does not make it solvable either";
    EXPECT_EQ(objective, infinity);
  }

  // (3) The QP solver does not solve the relaxation. HPIPM does not certify infeasibility, so this is the same code
  // path as a genuinely infeasible fixed schedule, and the caller must not be able to mistake it for a cheap candidate.
  {
    OcpQpHpipmSolver::Settings starved;
    starved.iterMax = 1;  // no QP of this family converges in a single interior point iteration

    OcpQpProblem fixedProblem = makeProblem(1.3);
    for (int k = 0; k < kNumStages; ++k) {
      fixedProblem.stages[k].lbu(1) = static_cast<scalar_t>(complete[k]);
      fixedProblem.stages[k].ubu(1) = static_cast<scalar_t>(complete[k]);
    }
    OcpQpHpipmSolver starvedSolver(starved);
    ASSERT_FALSE(starvedSolver.solve(fixedProblem).success()) << "precondition of this case: the starved solve really does fail";

    MixedIntegerOcpQp miqp{starved, MiqpSettings{}};
    OcpQpProblem problem = makeProblem(1.3);
    OcpQpSolution solution;
    scalar_t objective = -1.0;
    EXPECT_FALSE(miqp.solveFixed(problem, makeBinaries(), complete, nullptr, assignmentCost, solution, objective));
    EXPECT_EQ(objective, infinity);
  }
}

/**
 * An assignment of the wrong length is a programming error, not a candidate to reject.
 *
 * Silently truncating or padding it would fix the wrong binaries: the stages build candidates by copying an incumbent
 * whose length came from `binaryVariables()`, so a mismatch means the problem and the assignment were built for
 * different horizons, and every index from `contactBinaryIndex` onwards points at the wrong node. Throwing is what
 * turns that into a caught exception and a dropped plan (LipContactPlanner::plan catches) instead of a schedule that
 * silently belongs to another horizon.
 */
TEST(MixedIntegerOcpQpSolveFixedTest, ThrowsWhenTheAssignmentDoesNotMatchTheBinaries) {
  MixedIntegerOcpQp miqp{OcpQpHpipmSolver::Settings{}, MiqpSettings{}};
  OcpQpProblem problem = makeProblem(1.3);
  OcpQpSolution solution;
  scalar_t objective = 0.0;

  const MiqpAssignment tooShort(static_cast<std::size_t>(kNumStages - 1), static_cast<std::int8_t>(1));
  const MiqpAssignment tooLong(static_cast<std::size_t>(kNumStages + 1), static_cast<std::int8_t>(1));
  EXPECT_THROW(miqp.solveFixed(problem, makeBinaries(), tooShort, nullptr, nullptr, solution, objective), std::invalid_argument);
  EXPECT_THROW(miqp.solveFixed(problem, makeBinaries(), tooLong, nullptr, nullptr, solution, objective), std::invalid_argument);
  EXPECT_TRUE(std::isinf(objective)) << "the out-parameter is invalidated before the length is checked, so a caller that "
                                        "swallows the exception cannot read a stale objective either";
}

/**
 * A partial assignment the propagation completes is accepted, and it is the COMPLETED assignment that is solved and
 * priced.
 *
 * The event-shift local search hands `solveFixed` a candidate it has already propagated, so the two agree there; the
 * cadence stretch and the heading re-linearisation hand it the incumbent, which is complete. The contract is
 * nevertheless the wider one - `solveFixed` runs the propagation itself and only then demands completeness - and the
 * distinction is not academic: the assignment cost of a partial assignment is only a lower bound (MixedIntegerOcpQp.h),
 * so pricing the caller's partial assignment instead of the propagated one would under-charge every candidate whose
 * logical cost the propagation's own fixings create.
 */
TEST(MixedIntegerOcpQpSolveFixedTest, APartialAssignmentThePropagationCompletesIsSolvedAndPricedAsCompleted) {
  const MiqpPropagateFn propagate = &alternateFromTheFirstBinary;
  const MiqpAssignmentCostFn assignmentCost = &idlePricePerStage;
  const MiqpAssignment partial = {1, kMiqpFree, kMiqpFree, kMiqpFree};
  const MiqpAssignment completed = {1, 0, 1, 0};  // what the alternation makes of it

  MixedIntegerOcpQp miqp{OcpQpHpipmSolver::Settings{}, MiqpSettings{}};
  OcpQpProblem partialProblem = makeProblem(1.3);
  OcpQpSolution partialSolution;
  scalar_t partialObjective = -1.0;
  ASSERT_TRUE(miqp.solveFixed(partialProblem, makeBinaries(), partial, propagate, assignmentCost, partialSolution, partialObjective));

  OcpQpProblem completeProblem = makeProblem(1.3);
  OcpQpSolution completeSolution;
  scalar_t completeObjective = -1.0;
  ASSERT_TRUE(miqp.solveFixed(completeProblem, makeBinaries(), completed, propagate, assignmentCost, completeSolution, completeObjective));

  EXPECT_NEAR(partialObjective, completeObjective, 1e-6) << "the two calls describe the same schedule";
  EXPECT_NEAR(partialObjective, partialSolution.objective + idlePricePerStage(completed), 1e-12);
  // The price of the partial assignment is strictly smaller - it counts only what was decided - so this distinguishes
  // "priced after the propagation" from "priced as handed in".
  ASSERT_LT(idlePricePerStage(partial), idlePricePerStage(completed));
  EXPECT_GT(partialObjective, partialSolution.objective + idlePricePerStage(partial));

  for (int k = 0; k < kNumStages; ++k) {
    EXPECT_EQ(partialProblem.stages[k].lbu(1), static_cast<scalar_t>(completed[k])) << "stage " << k;
    EXPECT_NEAR(partialSolution.u[k](1), static_cast<scalar_t>(completed[k]), 1e-6) << "stage " << k;
  }
}

/**
 * Re-solving the branch-and-bound's own incumbent through `solveFixed` reproduces `incumbentObjective` exactly.
 *
 * This is the comparison the three afterSearch stages make on every candidate, and it only means anything if the two
 * numbers are computed the same way. The fixture is the decoupled-binary problem, where the assignment cost carries
 * most of the objective, so an accounting that dropped or double-counted the logical part cannot hide behind the QP.
 */
TEST(MixedIntegerOcpQpSolveFixedTest, ReproducesTheBranchAndBoundIncumbentObjective) {
  const MiqpAssignmentCostFn assignmentCost = &idlePricePerStage;
  MixedIntegerOcpQp miqp{OcpQpHpipmSolver::Settings{}, MiqpSettings{}};

  OcpQpProblem searched = makeDecoupledBinaryProblem(1.3);
  const MiqpAssignment initial(kNumStages, kMiqpFree);
  const MiqpResult result = miqp.solve(searched, makeBinaries(), initial, nullptr, nullptr, assignmentCost);
  ASSERT_TRUE(result.hasIncumbent);

  OcpQpProblem fresh = makeDecoupledBinaryProblem(1.3);
  OcpQpSolution solution;
  scalar_t objective = -1.0;
  ASSERT_TRUE(miqp.solveFixed(fresh, makeBinaries(), result.assignment, nullptr, assignmentCost, solution, objective));
  EXPECT_NEAR(objective, result.incumbentObjective, 1e-5)
      << "solveFixed and the search must put the same schedule at the same objective, or the local search around the "
         "incumbent compares two different functionals";
  EXPECT_NEAR(solution.objective, result.solution.objective, 1e-5);
}

/**
 * The assignment is indexed by the POSITION in the caller's branching list, not by the stage.
 *
 * The list is the caller's to order - the contact planner builds it in time order so that early contact decisions are
 * settled first - and `MiqpAssignment` is a bare vector of bytes with nothing in it that says which variable an entry
 * belongs to. The whole contact planner rests on that pairing: `contactBinaryIndex` computes a position in this list
 * and nothing checks the result. A list deliberately in the reverse of the natural order is the cheapest way to state
 * that entry i belongs to `binaries[i]` and not to stage i, because under either reading the values are the same
 * multiset and only their placement differs.
 */
TEST(MixedIntegerOcpQpSolveFixedTest, TheAssignmentIsIndexedByThePositionInTheBinaryList) {
  std::vector<MiqpBinaryVariable> reversed = makeBinaries();
  std::reverse(reversed.begin(), reversed.end());
  // Reversed, this assignment fixes the stages to 1, 0, 1, 0 in stage order; read as "entry k belongs to stage k" it
  // would fix them to 0, 1, 0, 1 instead.
  const MiqpAssignment assignment = {0, 1, 0, 1};

  MixedIntegerOcpQp miqp{OcpQpHpipmSolver::Settings{}, MiqpSettings{}};
  OcpQpProblem problem = makeProblem(1.3);
  OcpQpSolution solution;
  scalar_t objective = -1.0;
  ASSERT_TRUE(miqp.solveFixed(problem, reversed, assignment, nullptr, nullptr, solution, objective));

  for (std::size_t i = 0; i < reversed.size(); ++i) {
    const int stage = reversed[i].stage;
    const scalar_t expected = static_cast<scalar_t>(assignment[i]);
    EXPECT_EQ(problem.stages[stage].lbu(1), expected) << "binary " << i << " lives at stage " << stage;
    EXPECT_EQ(problem.stages[stage].ubu(1), expected) << "binary " << i << " lives at stage " << stage;
    EXPECT_NEAR(solution.u[stage](1), expected, 1e-6) << "binary " << i << " lives at stage " << stage;
  }
}

/*======================= the planner's binary layout against contactBinaryIndex ==========================*/

/**
 * The branching list the solver indexes and the index the rest of the planner computes must name the same variable.
 *
 * `binaryVariables()` walks the nodes and, inside a node, the model blocks, appending whatever each block declares as
 * a binary input; `contactBinaryIndex(k, foot)` returns `N_CONTACTS * k + foot`. The two agree only because exactly one
 * block (foothold_integrator) declares binaries and declares them one per foot in the feet's order. Should the loops in
 * `binaryVariables` ever be exchanged, should a second block declare a binary, or should a block be listed before the
 * foothold integrator, every logic rule, every assignment cost, the committed prefix and the event-shift local search
 * would go on addressing the assignment as if nothing had changed - and would silently read and write another node's or
 * another foot's contact. Nothing in the two files refers to the other, so this is where they are tied together.
 */
TEST(ContactPlannerBinaryLayoutTest, BinaryVariablesAgreeWithContactBinaryIndex) {
  {
    SCOPED_TRACE("the point-mass formulation");
    expectBinaryLayoutAgreesWithContactBinaryIndex(makePlannerConfig());
  }
  {
    SCOPED_TRACE("the formulation with the heading model");
    expectBinaryLayoutAgreesWithContactBinaryIndex(makeHeadingPlannerConfig());
  }
}

/**
 * The committed prefix lands on the binaries of its own node and foot.
 *
 * `initialAssignment` writes the executed schedule into the assignment through `contactBinaryIndex`, and those entries
 * are the fixings that hold for the whole search. Getting the mapping wrong here is the worst version of the
 * disagreement above, because the committed window is precisely the part of the plan the robot is already executing:
 * the planner would fix the wrong foot's contact for the next quarter second and then hand the result to the reference
 * manager as the schedule it promised to honour. The foot each entry belongs to is read back out of the layout rather
 * than assumed, so this test still says something if the feet's order in the layout ever changes.
 */
TEST(ContactPlannerBinaryLayoutTest, TheCommittedPrefixLandsOnTheBinariesOfItsOwnNodeAndFoot) {
  const ContactPlanningConfig config = makePlannerConfig();
  const LipContactPlanner planner(config);
  const Layout& layout = planner.getLayout();

  // An asymmetric prefix: without it a wrong foot or a wrong node cannot be told from the right one.
  ContactPlannerInput input = makeStandingInput();
  input.committedContacts.resize(2);
  input.committedContacts[0] = {true, false};
  input.committedContacts[1] = {false, true};

  const std::vector<MiqpBinaryVariable> binaries = planner.binaryVariables();
  const MiqpAssignment initial = planner.initialAssignment(input);
  ASSERT_EQ(initial.size(), binaries.size());

  const int numCommittedNodes = static_cast<int>(input.committedContacts.size());
  for (std::size_t i = 0; i < binaries.size(); ++i) {
    const MiqpBinaryVariable& binary = binaries[i];
    const int foot = footOfContactInput(layout, binary.inputIndex);
    ASSERT_GE(foot, 0) << "binary " << i << " is not a contact input";
    if (binary.stage < numCommittedNodes) {
      const bool committed = input.committedContacts[static_cast<std::size_t>(binary.stage)][static_cast<std::size_t>(foot)];
      EXPECT_EQ(static_cast<int>(initial[i]), committed ? 1 : 0) << "node " << binary.stage << ", foot " << foot;
    } else {
      EXPECT_EQ(static_cast<int>(initial[i]), static_cast<int>(kMiqpFree)) << "node " << binary.stage << " is beyond the commit window";
    }
  }
}

}  // namespace ocs2::humanoid
