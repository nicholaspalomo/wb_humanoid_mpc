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

#include <Eigen/Dense>
#include <cmath>

#include "humanoid_common_mpc/contact_planning/OcpQpHpipm.h"

namespace ocs2::humanoid {

namespace {

constexpr int kNumStages = 10;
constexpr scalar_t kDt = 0.1;

/** Double integrator x = [p, v], u = a, with quadratic tracking of the origin. */
OcpQpProblem makeDoubleIntegratorProblem(scalar_t p0, scalar_t v0) {
  OcpQpProblem problem;
  problem.x0 = (vector_t(2) << p0, v0).finished();
  problem.stages.resize(kNumStages + 1);
  for (int k = 0; k <= kNumStages; ++k) {
    const bool terminal = (k == kNumStages);
    OcpQpStage& s = problem.stages[k];
    s = OcpQpStage::Zero(2, terminal ? 0 : 1, !terminal);
    s.Q = (matrix_t(2, 2) << 10.0, 0.0, 0.0, 1.0).finished();
    if (!terminal) {
      s.R = matrix_t::Constant(1, 1, 0.1);
      s.A = (matrix_t(2, 2) << 1.0, kDt, 0.0, 1.0).finished();
      s.B = (matrix_t(2, 1) << 0.5 * kDt * kDt, kDt).finished();
    } else {
      s.Q *= 10.0;
    }
  }
  return problem;
}

/** Dense reference solution of the unconstrained problem via the KKT system. */
scalar_t solveUnconstrainedDense(const OcpQpProblem& problem, std::vector<vector_t>& x, std::vector<vector_t>& u) {
  const int N = problem.numStages();
  const int nx = 2;
  const int nu = 1;
  const int numVars = (N + 1) * nx + N * nu;  // [x0..xN, u0..uN-1]
  const int numEq = (N + 1) * nx;             // x0 fixed + dynamics
  matrix_t H = matrix_t::Zero(numVars, numVars);
  vector_t g = vector_t::Zero(numVars);
  matrix_t Aeq = matrix_t::Zero(numEq, numVars);
  vector_t beq = vector_t::Zero(numEq);
  const auto xi = [&](int k) { return k * nx; };
  const auto ui = [&](int k) { return (N + 1) * nx + k * nu; };
  for (int k = 0; k <= N; ++k) {
    H.block(xi(k), xi(k), nx, nx) = problem.stages[k].Q;
    if (k < N) {
      H.block(ui(k), ui(k), nu, nu) = problem.stages[k].R;
    }
  }
  Aeq.block(0, 0, nx, nx).setIdentity();
  beq.head(nx) = problem.x0;
  for (int k = 0; k < N; ++k) {
    Aeq.block(xi(k + 1), xi(k), nx, nx) = problem.stages[k].A;
    Aeq.block(xi(k + 1), ui(k), nx, nu) = problem.stages[k].B;
    Aeq.block(xi(k + 1), xi(k + 1), nx, nx) = -matrix_t::Identity(nx, nx);
    beq.segment(xi(k + 1), nx) = -problem.stages[k].b;
  }
  matrix_t KKT = matrix_t::Zero(numVars + numEq, numVars + numEq);
  KKT.topLeftCorner(numVars, numVars) = H;
  KKT.topRightCorner(numVars, numEq) = Aeq.transpose();
  KKT.bottomLeftCorner(numEq, numVars) = Aeq;
  vector_t rhs(numVars + numEq);
  rhs << -g, beq;
  const vector_t sol = KKT.fullPivLu().solve(rhs);
  x.resize(N + 1);
  u.resize(N);
  for (int k = 0; k <= N; ++k) x[k] = sol.segment(xi(k), nx);
  for (int k = 0; k < N; ++k) u[k] = sol.segment(ui(k), nu);
  return evaluateOcpQpObjective(problem, x, u);
}

}  // namespace

TEST(OcpQpHpipmTest, UnconstrainedMatchesDenseKkt) {
  OcpQpProblem problem = makeDoubleIntegratorProblem(1.0, -0.5);
  OcpQpHpipmSolver solver;
  const OcpQpSolution solution = solver.solve(problem);
  ASSERT_TRUE(solution.success());

  std::vector<vector_t> xRef, uRef;
  const scalar_t objRef = solveUnconstrainedDense(problem, xRef, uRef);
  EXPECT_NEAR(solution.objective, objRef, 1e-6);
  for (int k = 0; k <= kNumStages; ++k) {
    EXPECT_LT((solution.x[k] - xRef[k]).norm(), 1e-5) << "state mismatch at k=" << k;
  }
  EXPECT_LT(evaluateOcpQpMaxHardViolation(problem, solution.x, solution.u), 1e-6);
}

TEST(OcpQpHpipmTest, InputBoxConstraintIsRespected) {
  OcpQpProblem problem = makeDoubleIntegratorProblem(1.0, 0.0);
  for (int k = 0; k < kNumStages; ++k) {
    OcpQpStage& s = problem.stages[k];
    s.idxbu = {0};
    s.lbu = vector_t::Constant(1, -2.0);
    s.ubu = vector_t::Constant(1, 2.0);
  }
  OcpQpHpipmSolver solver;
  const OcpQpSolution solution = solver.solve(problem);
  ASSERT_TRUE(solution.success());
  for (int k = 0; k < kNumStages; ++k) {
    EXPECT_LE(std::abs(solution.u[k](0)), 2.0 + 1e-6);
  }
  // The unconstrained solution uses larger accelerations, so the bound must be active somewhere.
  scalar_t maxAbsInput = 0.0;
  for (int k = 0; k < kNumStages; ++k) maxAbsInput = std::max(maxAbsInput, std::abs(solution.u[k](0)));
  EXPECT_NEAR(maxAbsInput, 2.0, 1e-4);
  EXPECT_LT(evaluateOcpQpMaxHardViolation(problem, solution.x, solution.u), 1e-6);
}

TEST(OcpQpHpipmTest, GeneralStateInputConstraintIsRespected) {
  OcpQpProblem problem = makeDoubleIntegratorProblem(0.0, 0.0);
  // Ask the terminal state to sit at p = 1 while keeping v + 0.5 u <= 0.3 along the way.
  problem.stages[kNumStages].q = (vector_t(2) << -20.0, 0.0).finished();
  for (int k = 0; k < kNumStages; ++k) {
    OcpQpStage& s = problem.stages[k];
    s.C = (matrix_t(1, 2) << 0.0, 1.0).finished();
    s.D = matrix_t::Constant(1, 1, 0.5);
    s.lg = vector_t::Constant(1, -1e3);
    s.ug = vector_t::Constant(1, 0.3);
  }
  OcpQpHpipmSolver solver;
  const OcpQpSolution solution = solver.solve(problem);
  ASSERT_TRUE(solution.success());
  EXPECT_LT(evaluateOcpQpMaxHardViolation(problem, solution.x, solution.u), 1e-6);
  EXPECT_GT(solution.x[kNumStages](0), 0.05);  // it did move towards the target
}

TEST(OcpQpHpipmTest, SoftConstraintKeepsProblemFeasible) {
  OcpQpProblem hard = makeDoubleIntegratorProblem(0.0, 0.0);
  // Contradictory hard constraints: u >= 1 (box) but v <= 0.01 (general) after several steps.
  for (int k = 0; k < kNumStages; ++k) {
    OcpQpStage& s = hard.stages[k];
    s.idxbu = {0};
    s.lbu = vector_t::Constant(1, 1.0);
    s.ubu = vector_t::Constant(1, 2.0);
    s.C = (matrix_t(1, 2) << 0.0, 1.0).finished();
    s.D = matrix_t::Zero(1, 1);
    s.lg = vector_t::Constant(1, -1e3);
    s.ug = vector_t::Constant(1, 0.01);
  }
  OcpQpHpipmSolver solver;
  const OcpQpSolution hardSolution = solver.solve(hard);
  EXPECT_FALSE(hardSolution.success());

  OcpQpProblem soft = hard;
  for (int k = 0; k < kNumStages; ++k) {
    OcpQpStage& s = soft.stages[k];
    s.softGeneralIndices = {0};
    s.Zl = vector_t::Constant(1, 100.0);
    s.Zu = vector_t::Constant(1, 100.0);
    s.zl = vector_t::Constant(1, 1.0);
    s.zu = vector_t::Constant(1, 1.0);
  }
  const OcpQpSolution softSolution = solver.solve(soft);
  ASSERT_TRUE(softSolution.success());
  EXPECT_LT(evaluateOcpQpMaxHardViolation(soft, softSolution.x, softSolution.u), 1e-6);
  // The velocity constraint is violated (that is the point of the slack), and the objective accounts for it.
  EXPECT_GT(softSolution.x[kNumStages](1), 0.01);
  EXPECT_NEAR(softSolution.objective, evaluateOcpQpObjective(soft, softSolution.x, softSolution.u), 1e-9);
}

TEST(OcpQpHpipmTest, RepeatedSolvesWithChangedBoundsReuseMemory) {
  OcpQpProblem problem = makeDoubleIntegratorProblem(1.0, 0.0);
  for (int k = 0; k < kNumStages; ++k) {
    OcpQpStage& s = problem.stages[k];
    s.idxbu = {0};
    s.lbu = vector_t::Constant(1, -10.0);
    s.ubu = vector_t::Constant(1, 10.0);
  }
  OcpQpHpipmSolver solver;
  const scalar_t looseObjective = solver.solve(problem).objective;
  for (int k = 0; k < kNumStages; ++k) {
    problem.stages[k].lbu(0) = -0.5;
    problem.stages[k].ubu(0) = 0.5;
  }
  const OcpQpSolution tight = solver.solve(problem);
  ASSERT_TRUE(tight.success());
  EXPECT_GT(tight.objective, looseObjective);
  for (int k = 0; k < kNumStages; ++k) {
    EXPECT_LE(std::abs(tight.u[k](0)), 0.5 + 1e-6);
  }
}

}  // namespace ocs2::humanoid
