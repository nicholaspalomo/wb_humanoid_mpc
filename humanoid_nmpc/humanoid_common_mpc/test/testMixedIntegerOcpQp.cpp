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
#include <limits>

#include "humanoid_common_mpc/contact_planning/MixedIntegerOcpQp.h"

namespace ocs2::humanoid {

namespace {

constexpr int kNumStages = 4;
constexpr scalar_t kStepGain = 0.7;
constexpr scalar_t kDrift = -0.25;
constexpr scalar_t kSwitchCost = 0.05;

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

}  // namespace ocs2::humanoid
