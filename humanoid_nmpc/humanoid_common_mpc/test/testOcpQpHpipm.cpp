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
    const OcpQpStage& s = problem.stages[k];
    H.block(xi(k), xi(k), nx, nx) = s.Q;
    g.segment(xi(k), nx) = s.q;
    if (k < N) {
      H.block(ui(k), ui(k), nu, nu) = s.R;
      g.segment(ui(k), nu) = s.r;
      // The cross term u'Sx, S being nu x nx, is the off-diagonal block of the symmetric Hessian.
      H.block(ui(k), xi(k), nu, nx) = s.S;
      H.block(xi(k), ui(k), nx, nu) = s.S.transpose();
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

constexpr int kSoftSelectionNumStages = 2;
constexpr int kSoftSelectionStage = 1;

/**
 * Single integrator x_{k+1} = x_k + u_k over two stages plus a terminal node, whose stage-1 input is pulled upwards
 * by a linear cost and held back by two general rows: u <= 1 (row 0) and u <= 2 (row 1). Exactly one of the two rows
 * is declared soft, so the row that stays hard is what decides the answer - with row 0 soft the input climbs to the
 * hard bound of row 1 (u = 2, paying the slack penalty on row 0), and with row 1 soft it has to stop at the hard
 * bound of row 0 (u = 1, with no slack needed at all).
 *
 * The two variants have identical HPIPM dimensions - the same N, nx, nu, nbx, nbu, ng and nsg - and differ only in
 * the content of softGeneralIndices. That is precisely the case in which OcpQpHpipmSolver reuses its allocation, so
 * it is the case in which a soft-constraint mapping left over from the previous solve can survive into the next one.
 */
OcpQpProblem makeSoftRowSelectionProblem(int softRow) {
  OcpQpProblem problem;
  problem.x0 = vector_t::Zero(1);
  problem.stages.resize(kSoftSelectionNumStages + 1);
  for (int k = 0; k <= kSoftSelectionNumStages; ++k) {
    const bool terminal = (k == kSoftSelectionNumStages);
    OcpQpStage& s = problem.stages[k];
    s = OcpQpStage::Zero(1, terminal ? 0 : 1, !terminal);
    s.Q = matrix_t::Constant(1, 1, 1.0);
    if (!terminal) {
      s.R = matrix_t::Constant(1, 1, 1.0);
      s.A = matrix_t::Constant(1, 1, 1.0);
      s.B = matrix_t::Constant(1, 1, 1.0);
    }
  }

  OcpQpStage& constrained = problem.stages[kSoftSelectionStage];
  constrained.r = vector_t::Constant(1, -10.0);  // pulls the input well past both upper bounds
  constrained.C = matrix_t::Zero(2, 1);
  constrained.D = (matrix_t(2, 1) << 1.0, 1.0).finished();
  constrained.lg = vector_t::Constant(2, -1.0e3);  // finite, so that no bound mask is involved here
  constrained.ug = (vector_t(2) << 1.0, 2.0).finished();
  constrained.softGeneralIndices = {softRow};
  constrained.Zl = vector_t::Constant(1, 1.0);
  constrained.Zu = vector_t::Constant(1, 1.0);
  constrained.zl = vector_t::Constant(1, 1.0);
  constrained.zu = vector_t::Constant(1, 1.0);
  return problem;
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

/**
 * The state-input cross term S (nu x nx, cost u'Sx) is what the contact planner's terminal capturability and ZMP
 * regularisation residuals produce; a wrong transpose or a dropped block would go unnoticed by every S = 0 problem.
 */
TEST(OcpQpHpipmTest, StateInputCrossTermMatchesDenseKkt) {
  OcpQpProblem problem = makeDoubleIntegratorProblem(1.0, -0.5);
  OcpQpProblem noCrossTerm = problem;
  for (int k = 0; k < kNumStages; ++k) {
    // Small enough to keep [Q S'; S R] positive definite (Q - S' R^-1 S stays so).
    problem.stages[k].S = (matrix_t(1, 2) << 0.1, -0.05).finished();
  }
  OcpQpHpipmSolver solver;
  const OcpQpSolution solution = solver.solve(problem);
  ASSERT_TRUE(solution.success());

  std::vector<vector_t> xRef, uRef;
  const scalar_t objRef = solveUnconstrainedDense(problem, xRef, uRef);
  EXPECT_NEAR(solution.objective, objRef, 1e-6);
  EXPECT_NEAR(evaluateOcpQpObjective(problem, solution.x, solution.u), objRef, 1e-6);
  for (int k = 0; k <= kNumStages; ++k) {
    EXPECT_LT((solution.x[k] - xRef[k]).norm(), 1e-5) << "state mismatch at k=" << k;
  }
  for (int k = 0; k < kNumStages; ++k) {
    EXPECT_LT((solution.u[k] - uRef[k]).norm(), 1e-5) << "input mismatch at k=" << k;
  }
  // The cross term actually changes the solution, so an S that was silently ignored could not pass this test.
  const OcpQpSolution without = solver.solve(noCrossTerm);
  ASSERT_TRUE(without.success());
  scalar_t difference = 0.0;
  for (int k = 0; k < kNumStages; ++k) difference = std::max(difference, (solution.u[k] - without.u[k]).norm());
  EXPECT_GT(difference, 1e-3);
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

/**
 * Which general rows are soft must be decided by the problem handed to solve(), never by the problem handed to the
 * previous solve() on the same solver instance.
 *
 * HPIPM keeps the row-to-slack mapping in qp.idxs_rev, which is initialised to "all rows hard" only when the QP is
 * created and which d_ocp_qp_set_all then updates sparsely, writing the soft rows of the current problem and clearing
 * nothing. The solver reuses its HPIPM allocation whenever the problem dimensions are unchanged, and the dimensions
 * are pure counts: they cannot distinguish {row 0 soft} from {row 1 soft}. Before the fix the second solve below
 * therefore inherited the first solve's entry, so row 0 stayed attached to a slack pair - and shared it with row 1,
 * which had just become soft. The hard bound u <= 1 was then relaxed by that shared slack and the input ran away to
 * roughly 3.75 instead of stopping at 1.0, a hard-constraint violation of about 2.75 on a problem that is perfectly
 * feasible. This test therefore fails resoundingly on the unfixed wrapper.
 *
 * Both orders are exercised, since the stale entry corrupts the second solve whichever row was soft in the first.
 */
TEST(OcpQpHpipmTest, SoftRowSelectionIsNotInheritedFromThePreviousSolve) {
  const OcpQpProblem firstRowSoft = makeSoftRowSelectionProblem(0);
  const OcpQpProblem secondRowSoft = makeSoftRowSelectionProblem(1);

  // References from solvers that have never seen the other variant, i.e. with a freshly created (all hard) mapping.
  OcpQpHpipmSolver referenceSolverA;
  const OcpQpSolution referenceFirstRowSoft = referenceSolverA.solve(firstRowSoft);
  ASSERT_TRUE(referenceFirstRowSoft.success());
  OcpQpHpipmSolver referenceSolverB;
  const OcpQpSolution referenceSecondRowSoft = referenceSolverB.solve(secondRowSoft);
  ASSERT_TRUE(referenceSecondRowSoft.success());

  // The two variants really do have different answers, otherwise the test below could not tell them apart.
  EXPECT_NEAR(referenceFirstRowSoft.u[kSoftSelectionStage](0), 2.0, 1e-4);
  EXPECT_NEAR(referenceSecondRowSoft.u[kSoftSelectionStage](0), 1.0, 1e-4);

  // Row 0 soft, then row 1 soft, on one solver instance: the stale entry would leave row 0 soft as well.
  OcpQpHpipmSolver forwardSolver;
  ASSERT_TRUE(forwardSolver.solve(firstRowSoft).success());
  const OcpQpSolution reusedSecondRowSoft = forwardSolver.solve(secondRowSoft);
  EXPECT_TRUE(reusedSecondRowSoft.success());
  EXPECT_LT(evaluateOcpQpMaxHardViolation(secondRowSoft, reusedSecondRowSoft.x, reusedSecondRowSoft.u), 1e-6);
  EXPECT_NEAR(reusedSecondRowSoft.u[kSoftSelectionStage](0), referenceSecondRowSoft.u[kSoftSelectionStage](0), 1e-6);
  EXPECT_NEAR(reusedSecondRowSoft.objective, referenceSecondRowSoft.objective, 1e-6);

  // Row 1 soft, then row 0 soft, on one solver instance: the stale entry would leave row 1 soft as well.
  OcpQpHpipmSolver reverseSolver;
  ASSERT_TRUE(reverseSolver.solve(secondRowSoft).success());
  const OcpQpSolution reusedFirstRowSoft = reverseSolver.solve(firstRowSoft);
  EXPECT_TRUE(reusedFirstRowSoft.success());
  EXPECT_LT(evaluateOcpQpMaxHardViolation(firstRowSoft, reusedFirstRowSoft.x, reusedFirstRowSoft.u), 1e-6);
  EXPECT_NEAR(reusedFirstRowSoft.u[kSoftSelectionStage](0), referenceFirstRowSoft.u[kSoftSelectionStage](0), 1e-6);
  EXPECT_NEAR(reusedFirstRowSoft.objective, referenceFirstRowSoft.objective, 1e-6);
}

/**
 * Companion coverage for the other branch of the allocation logic: consecutive solves whose dimensions differ, so
 * that the solver has to throw away its HPIPM objects and create new ones, in both directions (small after large and
 * large after small, the former being the case in which the memory blocks are larger than the problem needs). This
 * path was correct before the idxs_rev fix as well - creation reinitialises the whole QP - but nothing exercised it,
 * so a future change to allocate() could have broken it unnoticed.
 */
TEST(OcpQpHpipmTest, ConsecutiveSolvesWithDifferentDimensionsReallocate) {
  const OcpQpProblem small = makeSoftRowSelectionProblem(1);
  const OcpQpProblem large = makeDoubleIntegratorProblem(1.0, -0.5);

  OcpQpHpipmSolver referenceSmallSolver;
  const OcpQpSolution smallReference = referenceSmallSolver.solve(small);
  ASSERT_TRUE(smallReference.success());
  OcpQpHpipmSolver referenceLargeSolver;
  const OcpQpSolution largeReference = referenceLargeSolver.solve(large);
  ASSERT_TRUE(largeReference.success());

  OcpQpHpipmSolver solver;
  ASSERT_TRUE(solver.solve(small).success());
  const OcpQpSolution largeAfterSmall = solver.solve(large);
  ASSERT_TRUE(largeAfterSmall.success());
  const OcpQpSolution smallAfterLarge = solver.solve(small);
  ASSERT_TRUE(smallAfterLarge.success());

  EXPECT_NEAR(largeAfterSmall.objective, largeReference.objective, 1e-6);
  for (int k = 0; k <= kNumStages; ++k) {
    EXPECT_LT((largeAfterSmall.x[k] - largeReference.x[k]).norm(), 1e-6) << "state mismatch at k=" << k;
  }
  EXPECT_LT(evaluateOcpQpMaxHardViolation(large, largeAfterSmall.x, largeAfterSmall.u), 1e-6);

  EXPECT_NEAR(smallAfterLarge.objective, smallReference.objective, 1e-6);
  EXPECT_NEAR(smallAfterLarge.u[kSoftSelectionStage](0), smallReference.u[kSoftSelectionStage](0), 1e-6);
  EXPECT_LT(evaluateOcpQpMaxHardViolation(small, smallAfterLarge.x, smallAfterLarge.u), 1e-6);
}

}  // namespace ocs2::humanoid
