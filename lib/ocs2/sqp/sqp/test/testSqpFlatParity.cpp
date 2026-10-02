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

/*
 * Bitwise parity of the flat SQP path.
 *
 * The manifold support of the SQP (OptimalControlProblem::stateManifoldPtr) must leave every problem without a manifold
 * exactly as it was: same operations, same order, same bits. This test solves a nonlinear toy problem in three
 * configurations (RK4 with the state-input constraint projected and a feedback policy; RK2 with the constraint handed to
 * HPIPM; RK4 without constraints and with a feedforward policy), three times each as a real-time MPC would (cold start,
 * warm start, warm start across the event), and prints everything the solver produces at 17 significant digits: the
 * iteration log, the primal solution, the per-node metrics and the policy evaluated on and off the nominal trajectory.
 *
 * The printout was first recorded from the solver BEFORE the manifold support was added, against the colcon build of
 * HPIPM (sqp_flat_parity_golden_colcon_hpipm.txt). Since the ROS removal bazel/system_libs.bzl builds the same HPIPM and
 * BLASFEO commits for TARGET=GENERIC (the colcon HPIPM was built for TARGET=AVX), so that the robot image also builds
 * for aarch64. Two configurations do not notice; rk2_hpipm_constrained_feedback, which hands its equality constraint
 * to HPIPM's interior point method, does: its primal solution, metrics and performance agree to 1e-9 relative (its
 * cost to 1e-12), but its feedback gains, which grow with the barrier terms of the IPM's last iterate, do not (its
 * policy outputs differ by up to 2 %). The golden the solver is compared with bit for bit (sqp_flat_parity_golden.txt)
 * is therefore the printout of this toolchain. It was recorded on the main line after the merge of the manifold
 * support, and the solver of the main line before that merge, without the manifold support, reproduces it line for line
 * on the same toolchain, so the comparison is one before and after the manifold support for every configuration, the
 * HPIPM-sensitive gains included. TheRecordingIsTheOriginalOneButForTheHpipmBuildSensitiveGains pins that it differs
 * from the original recording only there, which the HPIPM build alone explains. Each file's '#' header states where and
 * when it was recorded. The comparison is line by line, so a difference in any bit fails the test and names the first
 * line that differs. Printing at 17 significant digits round-trips every double, and also tells -0 from +0.
 *
 * The hash of the golden's data is pinned below (kGoldenDataHash). A re-record rewrites the file, so it cannot pass
 * by accident: the test fails until the new hash is written into this source file, a change a reviewer sees. Re-record
 * only when the toy problem itself changes or HPIPM is built differently (bazel/system_libs.bzl), never to absorb a
 * solver difference, and from a solver without the manifold support where one can be built (as this one was verified);
 * after a re-record for the HPIPM build, the comparison with the original recording must still pass:
 *   OCS2_RECORD_SQP_FLAT_PARITY="<solver state, e.g. commit and worktree>" bazel run //lib/ocs2:test_sqp_flat_parity
 * which writes the golden file into the source tree through BUILD_WORKSPACE_DIRECTORY, with that provenance in its
 * header, and then fails, printing the hash to pin.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <ocs2_core/constraint/LinearStateConstraint.h>
#include <ocs2_core/constraint/LinearStateInputConstraint.h>
#include <ocs2_core/cost/QuadraticStateCost.h>
#include <ocs2_core/cost/QuadraticStateInputCost.h>
#include <ocs2_core/dynamics/SystemDynamicsBase.h>
#include <ocs2_core/initialization/DefaultInitializer.h>
#include <ocs2_core/misc/LinearInterpolation.h>
#include <ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>
#include <ocs2_core/soft_constraint/StateSoftConstraint.h>
#include <ocs2_oc/oc_data/PrimalSolution.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>

#include "ocs2_sqp/SqpSolver.h"

namespace ocs2 {
namespace {

// LINT.IfChange(golden_path)
constexpr char kGoldenRelativePath[] = "lib/ocs2/sqp/sqp/test/data/sqp_flat_parity_golden.txt";
constexpr char kOriginalGoldenRelativePath[] = "lib/ocs2/sqp/sqp/test/data/sqp_flat_parity_golden_colcon_hpipm.txt";
// LINT.ThenChange(//lib/ocs2/BUILD.bazel:sqp_flat_parity_data)

// FNV-1a (64 bit) of the golden's data lines, each followed by '\n'; the '#' provenance lines are not hashed. The value
// of the printout recorded on 2026-10-02 on the main line, with HPIPM built for TARGET=GENERIC, which the solver before
// the manifold support reproduces on that build.
constexpr uint64_t kGoldenDataHash = 0x131bb9bff9606ddfULL;
// The same of the printout recorded on 2026-10-01 before the manifold support, against the colcon HPIPM.
constexpr uint64_t kOriginalGoldenDataHash = 0xdc834eb10ee062c7ULL;

// LINT.IfChange(hpipm_build_sensitive)
/** The configuration whose feedback gains depend on how HPIPM is built (see the file comment). */
constexpr char kHpipmBuildSensitiveConfiguration[] = "rk2_hpipm_constrained_feedback";
// LINT.ThenChange(//bazel/system_libs.bzl:hpipm_target)
/** How closely that configuration's primal solution, metrics and performance agree across the two HPIPM builds. */
constexpr double kHpipmBuildRelativeTolerance = 1e-9;
constexpr double kHpipmBuildAbsoluteTolerance = 1e-15;  ///< below this a difference is rounding (violations are ~1e-25)

/** FNV-1a (64 bit) of the lines, each followed by '\n'. */
uint64_t hashLines(const std::vector<std::string>& lines) {
  uint64_t hash = 0xcbf29ce484222325ULL;
  const uint64_t prime = 0x100000001b3ULL;
  for (const std::string& line : lines) {
    for (const char c : line) {
      hash = (hash ^ static_cast<uint64_t>(static_cast<unsigned char>(c))) * prime;
    }
    hash = (hash ^ static_cast<uint64_t>('\n')) * prime;
  }
  return hash;
}

std::string hexHash(uint64_t hash) {
  char buffer[19];
  std::snprintf(buffer, sizeof(buffer), "0x%016llx", static_cast<unsigned long long>(hash));
  return buffer;
}

constexpr size_t kStateDim = 4;
constexpr size_t kInputDim = 2;
constexpr scalar_t kEventTime = 0.35;

/**
 * A pendulum on a cart with friction and a nonlinear coupling term, x = [p, theta, v, omega], u = [a, tau]:
 *   p' = v, theta' = omega, v' = a - 0.1 v + 0.3 sin(theta) omega^2, omega' = -9.81 sin(theta) + tau cos(theta) - 0.2 omega.
 * The jump map damps the velocities and couples the angle into the angular rate, so the event node is nonlinear too.
 */
class NonlinearToyDynamics final : public SystemDynamicsBase {
 public:
  NonlinearToyDynamics() = default;
  ~NonlinearToyDynamics() override = default;
  NonlinearToyDynamics* clone() const override { return new NonlinearToyDynamics(*this); }

  vector_t computeFlowMap(scalar_t /*t*/, const vector_t& x, const vector_t& u, const PreComputation& /*preComp*/) override {
    vector_t dxdt(kStateDim);
    dxdt(0) = x(2);
    dxdt(1) = x(3);
    dxdt(2) = u(0) - 0.1 * x(2) + 0.3 * std::sin(x(1)) * x(3) * x(3);
    dxdt(3) = -9.81 * std::sin(x(1)) + u(1) * std::cos(x(1)) - 0.2 * x(3);
    return dxdt;
  }

  VectorFunctionLinearApproximation linearApproximation(scalar_t t,
                                                        const vector_t& x,
                                                        const vector_t& u,
                                                        const PreComputation& preComp) override {
    VectorFunctionLinearApproximation approximation;
    approximation.f = computeFlowMap(t, x, u, preComp);
    approximation.dfdx.setZero(kStateDim, kStateDim);
    approximation.dfdx(0, 2) = 1.0;
    approximation.dfdx(1, 3) = 1.0;
    approximation.dfdx(2, 1) = 0.3 * std::cos(x(1)) * x(3) * x(3);
    approximation.dfdx(2, 2) = -0.1;
    approximation.dfdx(2, 3) = 0.6 * std::sin(x(1)) * x(3);
    approximation.dfdx(3, 1) = -9.81 * std::cos(x(1)) - u(1) * std::sin(x(1));
    approximation.dfdx(3, 3) = -0.2;
    approximation.dfdu.setZero(kStateDim, kInputDim);
    approximation.dfdu(2, 0) = 1.0;
    approximation.dfdu(3, 1) = std::cos(x(1));
    return approximation;
  }

  vector_t computeJumpMap(scalar_t /*t*/, const vector_t& x, const PreComputation& /*preComp*/) override {
    vector_t xNext = x;
    xNext(2) = 0.7 * x(2);
    xNext(3) = -0.5 * x(3) + 0.1 * std::sin(x(1));
    return xNext;
  }

  VectorFunctionLinearApproximation jumpMapLinearApproximation(scalar_t t, const vector_t& x, const PreComputation& preComp) override {
    VectorFunctionLinearApproximation approximation;
    approximation.f = computeJumpMap(t, x, preComp);
    approximation.dfdx.setIdentity(kStateDim, kStateDim);
    approximation.dfdx(2, 2) = 0.7;
    approximation.dfdx(3, 3) = -0.5;
    approximation.dfdx(3, 1) = 0.1 * std::cos(x(1));
    approximation.dfdu.setZero(kStateDim, 0);
    return approximation;
  }

 private:
  NonlinearToyDynamics(const NonlinearToyDynamics& other) = default;
};

/** One way of setting up the solver; each is solved from scratch in its own solver instance. */
struct ParityConfiguration {
  std::string name;
  SensitivityIntegratorType integratorType;
  bool withEqualityConstraint;
  bool projectStateInputEqualityConstraints;
  bool useFeedbackPolicy;
};

std::vector<ParityConfiguration> parityConfigurations() {
  return {
      {"rk4_projected_feedback", SensitivityIntegratorType::RK4, /*withEqualityConstraint=*/true,
       /*projectStateInputEqualityConstraints=*/true, /*useFeedbackPolicy=*/true},
      {"rk2_hpipm_constrained_feedback", SensitivityIntegratorType::RK2, /*withEqualityConstraint=*/true,
       /*projectStateInputEqualityConstraints=*/false, /*useFeedbackPolicy=*/true},
      {"rk4_unconstrained_feedforward", SensitivityIntegratorType::RK4, /*withEqualityConstraint=*/false,
       /*projectStateInputEqualityConstraints=*/true, /*useFeedbackPolicy=*/false},
  };
}

matrix_t diagonal(const std::vector<scalar_t>& entries) {
  matrix_t m = matrix_t::Zero(entries.size(), entries.size());
  for (size_t i = 0; i < entries.size(); ++i) {
    m(i, i) = entries[i];
  }
  return m;
}

OptimalControlProblem makeProblem(const ParityConfiguration& configuration) {
  OptimalControlProblem problem;
  problem.dynamicsPtr = std::make_unique<NonlinearToyDynamics>();

  const matrix_t Q = diagonal({1.0, 5.0, 0.5, 0.2});
  const matrix_t R = diagonal({0.1, 0.05});
  problem.costPtr->add("trackingCost", std::make_unique<QuadraticStateInputCost>(Q, R));
  problem.preJumpCostPtr->add("preJumpCost", std::make_unique<QuadraticStateCost>(0.5 * Q));
  problem.finalCostPtr->add("finalCost", std::make_unique<QuadraticStateCost>(10.0 * Q));

  // Soft input bound tau <= 2 (h = 2 - tau >= 0), and a soft position bound p <= 1.2 at the event and at the end.
  vector_t inputBoundOffset(1);
  inputBoundOffset << 2.0;
  matrix_t inputBoundStateFactor = matrix_t::Zero(1, kStateDim);
  matrix_t inputBoundInputFactor(1, kInputDim);
  inputBoundInputFactor << 0.0, -1.0;
  problem.softConstraintPtr->add(
      "inputBound", std::make_unique<StateInputSoftConstraint>(
                        std::make_unique<LinearStateInputConstraint>(inputBoundOffset, inputBoundStateFactor, inputBoundInputFactor),
                        std::make_unique<RelaxedBarrierPenalty>(RelaxedBarrierPenalty::Config(/*muParam=*/0.1, /*deltaParam=*/0.5))));
  vector_t positionBoundOffset(1);
  positionBoundOffset << 1.2;
  matrix_t positionBoundFactor = matrix_t::Zero(1, kStateDim);
  positionBoundFactor(0, 0) = -1.0;
  problem.preJumpSoftConstraintPtr->add(
      "positionBound", std::make_unique<StateSoftConstraint>(
                           std::make_unique<LinearStateConstraint>(positionBoundOffset, positionBoundFactor),
                           std::make_unique<RelaxedBarrierPenalty>(RelaxedBarrierPenalty::Config(/*muParam=*/0.2, /*deltaParam=*/0.1))));
  problem.finalSoftConstraintPtr->add(
      "positionBound", std::make_unique<StateSoftConstraint>(
                           std::make_unique<LinearStateConstraint>(positionBoundOffset, positionBoundFactor),
                           std::make_unique<RelaxedBarrierPenalty>(RelaxedBarrierPenalty::Config(/*muParam=*/0.2, /*deltaParam=*/0.1))));

  if (configuration.withEqualityConstraint) {
    // a + 0.5 tau - 0.2 p - 0.1 = 0
    vector_t e(1);
    e << -0.1;
    matrix_t C = matrix_t::Zero(1, kStateDim);
    C(0, 0) = -0.2;
    matrix_t D(1, kInputDim);
    D << 1.0, 0.5;
    problem.equalityConstraintPtr->add("inputCoupling", std::make_unique<LinearStateInputConstraint>(e, C, D));
  }
  return problem;
}

sqp::Settings makeSettings(const ParityConfiguration& configuration) {
  sqp::Settings settings;
  settings.dt = 0.05;
  settings.sqpIteration = 6;
  settings.integratorType = configuration.integratorType;
  settings.projectStateInputEqualityConstraints = configuration.projectStateInputEqualityConstraints;
  settings.useFeedbackPolicy = configuration.useFeedbackPolicy;
  settings.createValueFunction = false;
  settings.printSolverStatus = false;
  settings.printSolverStatistics = false;
  settings.printLinesearch = false;
  settings.enableLogging = false;
  settings.nThreads = 1;  // the per-worker accumulation order is then fixed, as bitwise parity needs
  return settings;
}

/** Appends `label value value ...` with every value at 17 significant digits. */
void printLine(std::ostringstream& out, const std::string& label, const vector_t& values) {
  out << label;
  char buffer[64];
  for (Eigen::Index i = 0; i < values.size(); ++i) {
    std::snprintf(buffer, sizeof(buffer), " %.17g", values(i));
    out << buffer;
  }
  out << '\n';
}

void printScalar(std::ostringstream& out, const std::string& label, scalar_t value) {
  vector_t values(1);
  values << value;
  printLine(out, label, values);
}

void printMatrix(std::ostringstream& out, const std::string& label, const matrix_t& matrix) {
  const vector_t values = Eigen::Map<const vector_t>(matrix.data(), matrix.size());
  printLine(out, label + " [" + std::to_string(matrix.rows()) + "x" + std::to_string(matrix.cols()) + "]", values);
}

void printPerformance(std::ostringstream& out, const std::string& label, const PerformanceIndex& performance) {
  vector_t values(8);
  values << performance.merit, performance.cost, performance.dualFeasibilitiesSSE, performance.dynamicsViolationSSE,
      performance.equalityConstraintsSSE, performance.inequalityConstraintsSSE, performance.equalityLagrangian,
      performance.inequalityLagrangian;
  printLine(out, label, values);
}

void printMetrics(std::ostringstream& out, const std::string& label, const Metrics& metrics) {
  printScalar(out, label + " cost", metrics.cost);
  printLine(out, label + " dynamicsViolation", metrics.dynamicsViolation);
  for (size_t i = 0; i < metrics.stateInputEqConstraint.size(); ++i) {
    printLine(out, label + " stateInputEq" + std::to_string(i), metrics.stateInputEqConstraint[i]);
  }
}

/** Solves the three-solve MPC sequence of one configuration and prints everything it produced. */
std::string solveAndPrint(const ParityConfiguration& configuration) {
  const OptimalControlProblem problem = makeProblem(configuration);
  const DefaultInitializer initializer(kInputDim);

  vector_t targetStart = vector_t::Zero(kStateDim);
  vector_t targetEnd(kStateDim);
  targetEnd << 1.0, 0.5, 0.0, 0.0;
  const TargetTrajectories targetTrajectories({0.0, 1.5}, {targetStart, targetEnd}, {vector_t::Zero(kInputDim), vector_t::Zero(kInputDim)});
  const ModeSchedule modeSchedule({kEventTime}, {0, 1});
  std::shared_ptr<ReferenceManager> referenceManagerPtr = std::make_shared<ReferenceManager>(targetTrajectories, modeSchedule);

  SqpSolver solver(makeSettings(configuration), problem, initializer);
  solver.setReferenceManager(referenceManagerPtr);

  std::ostringstream out;
  out << "configuration " << configuration.name << '\n';

  vector_t initState(kStateDim);
  initState << 0.1, 0.3, 0.0, 0.0;
  const std::vector<scalar_t> initTimes = {0.0, 0.05, 0.4};
  vector_t perturbation(kStateDim);
  perturbation << 0.01, -0.02, 0.03, -0.01;
  for (size_t solve = 0; solve < initTimes.size(); ++solve) {
    const scalar_t initTime = initTimes[solve];
    const scalar_t finalTime = initTime + 1.0;
    const std::string prefix = configuration.name + " solve" + std::to_string(solve);

    solver.run(initTime, initState, /*initMode=*/0, finalTime);

    const std::vector<PerformanceIndex>& iterations = solver.getIterationsLog();
    printScalar(out, prefix + " iterations", static_cast<scalar_t>(iterations.size()));
    for (size_t k = 0; k < iterations.size(); ++k) {
      printPerformance(out, prefix + " performance" + std::to_string(k), iterations[k]);
    }

    const PrimalSolution primalSolution = solver.primalSolution(finalTime);
    const vector_t times = Eigen::Map<const vector_t>(primalSolution.timeTrajectory_.data(), primalSolution.timeTrajectory_.size());
    printLine(out, prefix + " time", times);
    for (size_t k = 0; k < primalSolution.stateTrajectory_.size(); ++k) {
      printLine(out, prefix + " state" + std::to_string(k), primalSolution.stateTrajectory_[k]);
      printLine(out, prefix + " input" + std::to_string(k), primalSolution.inputTrajectory_[k]);
    }

    const ProblemMetrics& metrics = solver.getSolutionMetrics();
    for (size_t k = 0; k < metrics.intermediates.size(); ++k) {
      printMetrics(out, prefix + " intermediate" + std::to_string(k), metrics.intermediates[k]);
    }
    for (size_t k = 0; k < metrics.preJumps.size(); ++k) {
      printMetrics(out, prefix + " preJump" + std::to_string(k), metrics.preJumps[k]);
    }
    printMetrics(out, prefix + " final", metrics.final);

    // The policy at and between the nodes, on and off the nominal trajectory.
    for (size_t k = 0; k + 1 < primalSolution.timeTrajectory_.size(); ++k) {
      for (const scalar_t fraction : {0.0, 0.3}) {
        const scalar_t t =
            primalSolution.timeTrajectory_[k] + fraction * (primalSolution.timeTrajectory_[k + 1] - primalSolution.timeTrajectory_[k]);
        const vector_t nominalState = LinearInterpolation::interpolate(t, primalSolution.timeTrajectory_, primalSolution.stateTrajectory_);
        const std::string label = prefix + " policy" + std::to_string(k) + (fraction == 0.0 ? "a" : "b");
        printLine(out, label + " nominal", primalSolution.controllerPtr_->computeInput(t, nominalState));
        printLine(out, label + " perturbed", primalSolution.controllerPtr_->computeInput(t, nominalState + perturbation));
      }
    }

    // The next solve starts a little off the plan, as a measured state would.
    const scalar_t nextTime = (solve + 1 < initTimes.size()) ? initTimes[solve + 1] : finalTime;
    initState =
        LinearInterpolation::interpolate(nextTime, primalSolution.timeTrajectory_, primalSolution.stateTrajectory_) + 0.5 * perturbation;
  }
  return out.str();
}

std::string solveAllConfigurations() {
  std::string printout;
  for (const ParityConfiguration& configuration : parityConfigurations()) {
    printout += solveAndPrint(configuration);
  }
  return printout;
}

std::vector<std::string> splitLines(const std::string& text) {
  std::vector<std::string> lines;
  std::istringstream stream(text);
  std::string line;
  while (std::getline(stream, line)) {
    lines.push_back(line);
  }
  return lines;
}

/** The data lines of a golden file: every line but the '#' provenance header. */
std::vector<std::string> dataLines(const std::vector<std::string>& lines) {
  std::vector<std::string> data;
  for (const std::string& line : lines) {
    if (line.empty() || line.front() != '#') {
      data.push_back(line);
    }
  }
  return data;
}

/** The data lines of the golden file at `relativePath`; empty when it cannot be read. */
std::vector<std::string> readGoldenDataLines(const std::string& relativePath) {
  std::ifstream file(relativePath);
  if (!file.good()) return {};
  std::stringstream golden;
  golden << file.rdbuf();
  return dataLines(splitLines(golden.str()));
}

/** The whitespace-separated tokens of a line. */
std::vector<std::string> splitTokens(const std::string& line) {
  std::vector<std::string> tokens;
  std::istringstream stream(line);
  std::string token;
  while (stream >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

/** `token` as a double when it is one in full; nothing otherwise. */
std::optional<double> parseNumber(const std::string& token) {
  char* end = nullptr;
  const double value = std::strtod(token.c_str(), &end);
  if (end == token.c_str() || *end != '\0') return std::nullopt;
  return value;
}

/** Whether `a` and `b` have the same words and numbers that agree within the HPIPM build tolerance. */
bool agreesWithinHpipmBuildTolerance(const std::string& a, const std::string& b) {
  const std::vector<std::string> tokensA = splitTokens(a);
  const std::vector<std::string> tokensB = splitTokens(b);
  if (tokensA.size() != tokensB.size()) return false;
  for (size_t k = 0; k < tokensA.size(); ++k) {
    if (tokensA[k] == tokensB[k]) continue;
    const std::optional<double> x = parseNumber(tokensA[k]);
    const std::optional<double> y = parseNumber(tokensB[k]);
    if (!x.has_value() || !y.has_value() || !std::isfinite(*x) || !std::isfinite(*y)) return false;
    const double scale = std::max(std::abs(*x), std::abs(*y));
    if (std::abs(*x - *y) > kHpipmBuildRelativeTolerance * scale + kHpipmBuildAbsoluteTolerance) return false;
  }
  return true;
}

TEST(SqpFlatParity, SolveWithoutManifoldIsBitwiseIdenticalToTheRecordedSolve) {
  const std::string printout = solveAllConfigurations();

  const char* provenance = std::getenv("OCS2_RECORD_SQP_FLAT_PARITY");
  if (provenance != nullptr) {
    ASSERT_GT(std::string(provenance).size(), 1u) << "Set OCS2_RECORD_SQP_FLAT_PARITY to the solver state being recorded.";
    const char* workspace = std::getenv("BUILD_WORKSPACE_DIRECTORY");
    ASSERT_NE(workspace, nullptr) << "Record with `bazel run`, which sets BUILD_WORKSPACE_DIRECTORY.";
    const std::string path = std::string(workspace) + "/" + kGoldenRelativePath;
    const std::string hash = hexHash(hashLines(splitLines(printout)));
    std::ofstream file(path);
    ASSERT_TRUE(file.good()) << "Cannot write " << path;
    file << "# Golden printout of testSqpFlatParity.cpp. Lines starting with '#' are provenance and are not compared.\n"
         << "# Recorded from: " << provenance << "\n"
         << "# Data hash (FNV-1a 64 of the non-comment lines): " << hash << "\n"
         << printout;
    FAIL() << "Recorded " << splitLines(printout).size() << " lines to " << path << ". The test fails until kGoldenDataHash in "
           << "testSqpFlatParity.cpp is set to " << hash << ", after a review that only the toy problem changed.";
  }

  const std::vector<std::string> expected = readGoldenDataLines(kGoldenRelativePath);
  const std::vector<std::string> actual = splitLines(printout);
  ASSERT_GT(expected.size(), 100u) << "The golden file looks truncated.";
  ASSERT_EQ(hexHash(hashLines(expected)), hexHash(kGoldenDataHash))
      << "The golden data is not the recorded one: " << kGoldenRelativePath << " was rewritten (re-recorded or edited).";
  const size_t common = std::min(expected.size(), actual.size());
  for (size_t i = 0; i < common; ++i) {
    ASSERT_EQ(actual[i], expected[i]) << "First difference at line " << i + 1 << " of " << kGoldenRelativePath;
  }
  ASSERT_EQ(actual.size(), expected.size());
}

// The golden of this toolchain against the recording made before the manifold support: the same line for line, but for
// the feedback policy of the configuration whose gains depend on the HPIPM build, and that configuration's other lines
// within rounding. So the re-recording absorbed the HPIPM build and nothing else.
TEST(SqpFlatParity, TheRecordingIsTheOriginalOneButForTheHpipmBuildSensitiveGains) {
  const std::vector<std::string> golden = readGoldenDataLines(kGoldenRelativePath);
  const std::vector<std::string> original = readGoldenDataLines(kOriginalGoldenRelativePath);
  ASSERT_GT(original.size(), 100u) << "Missing or truncated " << kOriginalGoldenRelativePath;
  ASSERT_EQ(hexHash(hashLines(golden)), hexHash(kGoldenDataHash)) << kGoldenRelativePath << " is not the pinned recording.";
  ASSERT_EQ(hexHash(hashLines(original)), hexHash(kOriginalGoldenDataHash)) << kOriginalGoldenRelativePath << " is not the original.";
  ASSERT_EQ(golden.size(), original.size());

  const std::string sensitivePrefix = std::string(kHpipmBuildSensitiveConfiguration) + " ";
  size_t policyLines = 0;
  size_t toleratedLines = 0;
  for (size_t i = 0; i < golden.size(); ++i) {
    const std::vector<std::string> tokens = splitTokens(original[i]);
    const bool sensitive = original[i].rfind(sensitivePrefix, 0) == 0;
    if (!sensitive) {
      ASSERT_EQ(golden[i], original[i]) << "line " << i + 1 << " of a configuration HPIPM's build does not reach";
    } else if (tokens.size() > 2 && tokens[2].rfind("policy", 0) == 0) {
      // The gains: compared nowhere across the builds, only the labels.
      const std::vector<std::string> goldenTokens = splitTokens(golden[i]);
      ASSERT_GE(goldenTokens.size(), 4u) << "line " << i + 1;
      ASSERT_EQ(std::vector<std::string>(goldenTokens.begin(), goldenTokens.begin() + 4),
                std::vector<std::string>(tokens.begin(), tokens.begin() + 4))
          << "line " << i + 1;
      ++policyLines;
    } else {
      ASSERT_TRUE(agreesWithinHpipmBuildTolerance(golden[i], original[i])) << "line " << i + 1 << ":\n  golden:   " << golden[i]
                                                                           << "\n  original: " << original[i];
      ++toleratedLines;
    }
  }
  EXPECT_GT(policyLines, 0u);
  EXPECT_GT(toleratedLines, policyLines) << "the primal solution of the HPIPM-build-sensitive configuration was not compared";
}

// getInitialStateGap() is the delta_x0 of the first iteration: zero for a cold start, the observation against the warm
// start otherwise, and nothing before the first solve or after a reset. The lockstep closed loop's initial-state gap
// metric reads it, so that a reset served before the solve (which discards the warm start) is measured as the SQP sees it.
TEST(SqpFlatParity, RecordsTheInitialStateGapOfTheFirstIteration) {
  const ParityConfiguration configuration = parityConfigurations().front();
  const OptimalControlProblem problem = makeProblem(configuration);
  const DefaultInitializer initializer(kInputDim);
  const TargetTrajectories targetTrajectories({0.0}, {vector_t::Zero(kStateDim)}, {vector_t::Zero(kInputDim)});
  SqpSolver solver(makeSettings(configuration), problem, initializer);
  solver.setReferenceManager(std::make_shared<ReferenceManager>(targetTrajectories, ModeSchedule({kEventTime}, {0, 1})));
  EXPECT_EQ(solver.getInitialStateGap().size(), 0);

  vector_t initState(kStateDim);
  initState << 0.1, 0.3, 0.0, 0.0;
  solver.run(/*initTime=*/0.0, initState, /*initMode=*/0, /*finalTime=*/1.0);
  ASSERT_EQ(static_cast<size_t>(solver.getInitialStateGap().size()), kStateDim);
  EXPECT_TRUE(solver.getInitialStateGap().isZero(0.0)) << solver.getInitialStateGap().transpose();

  PrimalSolution previous;
  solver.getPrimalSolution(/*finalTime=*/1.0, &previous);
  const scalar_t nextTime = 0.05;
  vector_t observed = LinearInterpolation::interpolate(nextTime, previous.timeTrajectory_, previous.stateTrajectory_);
  observed(1) += 0.02;
  const vector_t expected = observed - LinearInterpolation::interpolate(nextTime, previous.timeTrajectory_, previous.stateTrajectory_);
  solver.run(nextTime, observed, /*initMode=*/0, nextTime + 1.0);
  EXPECT_LE((solver.getInitialStateGap() - expected).cwiseAbs().maxCoeff(), 1e-15) << solver.getInitialStateGap().transpose();

  // A reset discards the warm start: the next solve starts from the observation itself.
  solver.reset();
  EXPECT_EQ(solver.getInitialStateGap().size(), 0);
  solver.run(nextTime, observed, /*initMode=*/0, nextTime + 1.0);
  EXPECT_TRUE(solver.getInitialStateGap().isZero(0.0)) << solver.getInitialStateGap().transpose();
}

// Two solves in the same process give the same bits, so the comparison above is not at the mercy of run-to-run noise.
TEST(SqpFlatParity, SolveIsDeterministic) {
  EXPECT_EQ(solveAllConfigurations(), solveAllConfigurations());
}

}  // namespace
}  // namespace ocs2
