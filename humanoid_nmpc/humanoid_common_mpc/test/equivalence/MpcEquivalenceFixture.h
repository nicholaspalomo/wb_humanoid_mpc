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

#pragma once

// Pinocchio forward declarations must be included first.
#include "pinocchio/fwd.hpp"

#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"
#include "ocs2_oc/oc_data/ProblemMetrics.h"
#include "ocs2_oc/oc_data/TimeDiscretization.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "ocs2_sqp/SqpSettings.h"
#include "ocs2_sqp/SqpSolverStatus.h"

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_mpc_validation/chart/EulerChart.h"
#include "humanoid_mpc_validation/closed_loop/RobotConfiguration.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/equivalence/EulerMpcOracle.h"
#include "humanoid_wb_mpc/WBMpcInterface.h"

/*
 * What the MPC-level tests
 * (testMpc{Dynamics,Rollout,OrientationCost,SqpStep}Equivalence.cpp,
 * testMpcSqpConvergence.cpp and
 * testQuaternionRootTerm{Derivatives,Hessians}.cpp) share: the shipped MPC of
 * one configuration on the quaternion layout (its interface, optimal control
 * problem, state manifold, robot model and Pinocchio model, one real-time SQP
 * iteration on a given manifold, an SQP run with the solver's records of its
 * iterations, and a sequence of real-time solves), the Euler side of it
 * (EulerMpcOracle on the Euler-root twin of the MPC's own Pinocchio model, and
 * the Euler-chart manifold EulerChartStateManifold), the chart between the two
 * (humanoid_mpc_validation/chart/EulerChart.h: Phi, dPhi / dx_e, P_e), random
 * Euler states and inputs, and the walking reference and resting warm start the
 * SQP and term tests evaluate on, built from the configuration's shipped files:
 * no recorded data.
 *
 * Chart identities the tests use (design section 2.12, derived in
 * EulerChart.h): with x = Phi(x_e), (E1) values             f_a(Phi(x_e), u) =
 * dPhi(x_e) f_e(x_e, u), the ambient flow of the quaternion state; (E2)
 * derivatives        a function g of the state has d (g o Phi) / dx_e = (dg /
 * dx) dPhi = (dg / dx) E(x) P_e(x_e), because dPhi = E P_e block for block (on
 * the angles 1/2 G(xi) T_B(Theta) = E(xi) T_B(Theta)); (E3) Gauss-Newton H_e =
 * dPhi' H dPhi for a term whose Hessian is J' W J; an exact Hessian adds the
 * chart's curvature, sum_i (dg / dx_i) d^2 Phi_i / dx_e^2, which only the angle
 * (and, on the whole body, angle-rate) block of the Euler state has; (E4) node
 * maps          A_t P_e(x_e) = P_e(F_e(x_e)) A_e and B_t = P_e(F_e(x_e)) B_e,
 * A_t and B_t the SQP's tangent linearization of one shooting interval, A_e and
 * B_e the Euler RK4 map's, up to RK4's dependence on the coordinates
 * (testMpcDynamicsEquivalence derives the bound).
 *
 * The configuration a test binary runs on is the one
 * kMpcEquivalenceConfigurationVariable names (each per-robot Bazel target sets
 * it, so that a binary generates one robot's CppAD libraries); the first
 * interface of a binary generates them, the others load them.
 */

namespace ocs2::humanoid {

/** The environment variable naming the configuration an equivalence test binary
 * runs on (BUILD.bazel). */
inline constexpr char kMpcEquivalenceConfigurationVariable[] =
    "MPC_EQUIVALENCE_CONFIGURATION";

/** The configurations the tests are instantiated for: the one the variable
 * names, else every RobotConfiguration. */
std::vector<std::string> mpcEquivalenceConfigurationNames();

/** The name of a test of the configuration `info.param`. */
std::string mpcEquivalenceTestName(
    const ::testing::TestParamInfo<std::string>& info);

/** [s] The walking reference's schedule
 * (MpcEquivalenceFixture::walkingReference()): the first lift-off, ... */
inline constexpr scalar_t kWalkingFirstLiftOff = 0.2;
/** [s] ... the duration of each single support ... */
inline constexpr scalar_t kWalkingSingleSupport = 0.35;
/** [s] ... and of each double support between two of them. */
inline constexpr scalar_t kWalkingDoubleSupport = 0.1;

/**
 * The reference a test evaluates the problem against, as the MPC loop sets it
 * before a solve: installed on the reference manager (reset, mode schedule,
 * target trajectories, preSolverRun at `initTime` from `initState`). Passive
 * data.
 */
struct EquivalenceReference {
  ModeSchedule modeSchedule;
  TargetTrajectories targetTrajectories;
  scalar_t initTime = 0.0;
  /** The state the reference manager takes as measured. */
  vector_t initState;
  size_t initMode = ModeNumber::kStance;
};

/** One SQP solve from a fixed warm start: `warmStartState` and `warmStartInput`
 * held over the horizon. Passive data. */
struct SqpCase {
  /** The measured state the solve starts from. */
  vector_t initState;
  vector_t warmStartState;
  vector_t warmStartInput;
};

/** One real-time iteration of the SQP, from a warm start held over the horizon.
 * Passive data. */
struct RealTimeIteration {
  /** The applied step at every node, difference(warm start, x_k) in the tangent
   * of the iteration's manifold (nodes x ndx). */
  matrix_t stateStep;
  /** u_k - u_warm at every node (nodes x nu); NaN at a node without an input.
   */
  matrix_t inputStep;
  /** The feedback gain at the first node, on the same tangent (nu x ndx). */
  matrix_t firstGain;
  /** The line search's step size alpha: stateStep and inputStep are alpha times
   * the QP's solution. */
  scalar_t stepSize = 0.0;
};

/** How sqpSolution() and sqpRun() run the SQP. Passive data. */
struct SqpRunOptions {
  /** The SQP's iteration limit (sqp_iteration). */
  size_t iterations = 1;
  /** Its deltaTol and costTol, when set; the task file's otherwise. */
  std::optional<scalar_t> convergenceTolerance;
};

/** What one SQP solve did. Passive data. */
struct SqpRun {
  PrimalSolution solution;
  /** Every iteration, as the solver recorded it
   * (SqpSolver::getIterationRecords()). */
  std::vector<sqp::IterationRecord> iterations;
  /** The cost, dynamics and constraint values of every node of the solution
   * (SolverBase::getSolutionMetrics()). */
  ProblemMetrics metrics;
};

/** One solve of MpcEquivalenceFixture::realTimeSolves(). Passive data. */
struct RealTimeSolve {
  /** [s] Where the solve's horizon starts. */
  scalar_t initTime = 0.0;
  /** The solve's one iteration (sqp_iteration 1). */
  sqp::IterationRecord iteration;
  /** The values of every node of its plan (SolverBase::getSolutionMetrics()).
   */
  ProblemMetrics metrics;
};

/** Where random Euler states and inputs are drawn. Passive data. */
struct EulerSamplingBounds {
  /** [rad] |pitch| and |roll|; |yaw| <= pi always. */
  scalar_t maxAbsPitch = 1.2;
  scalar_t maxAbsRoll = 1.2;
  /** [m/s] |c_dot| per axis (centroidal) and |pd_W| per axis (whole body). */
  scalar_t maxAbsLinearVelocity = 0.5;
  /** [m^2/s] |L_G / m| per axis (centroidal). */
  scalar_t maxAbsNormalizedAngularMomentum = 0.05;
  /** [rad/s] |Euler rate| per axis (whole body). */
  scalar_t maxAbsEulerRate = 1.0;
  /** [rad/s] |joint velocity| per joint (the whole body's state, the centroidal
   * input). */
  scalar_t maxAbsJointVelocity = 1.0;
  /** [rad] |joint angle - midpoint of its limits| per joint, at most the
   * limits'. */
  scalar_t maxAbsJointOffset = 0.5;
};

/**
 * One configuration's shipped MPC on the quaternion layout and its Euler side.
 * Build it with Create(). Not thread-safe: it installs references on the
 * interface's reference manager.
 */
class MpcEquivalenceFixture {
 public:
  /**
   * The MPC of the configuration `robotName`
   * (validation::findRobotConfiguration()) built from its shipped files, and
   * its oracle. The errors of the interface's and the oracle's Create();
   * InvalidArgument when the task file's tuning layout is not the Euler state
   * the oracle evaluates.
   */
  static absl::StatusOr<std::unique_ptr<MpcEquivalenceFixture>> Create(
      absl::string_view robotName);

  MpcEquivalenceFixture(const MpcEquivalenceFixture&) = delete;
  MpcEquivalenceFixture& operator=(const MpcEquivalenceFixture&) = delete;
  MpcEquivalenceFixture(MpcEquivalenceFixture&&) = delete;
  MpcEquivalenceFixture& operator=(MpcEquivalenceFixture&&) = delete;
  ~MpcEquivalenceFixture() = default;

  const validation::RobotConfiguration& robot() const { return robot_; }
  bool isWholeBody() const { return wholeBody_ != nullptr; }
  const ModelSettings& modelSettings() const;
  /** The interface's effective robot model: the layout of every state and
   * input. */
  const MpcRobotModelBase<scalar_t>& model() const;
  /** The interface's Pinocchio model of the MPC's joints, on the quaternion
   * root. */
  const PinocchioInterface& pinocchioInterface() const;
  /**
   * The MPC's optimal control problem. Its members are owning pointers, so its
   * dynamics and terms can be evaluated (and OCS2's Collection::get() called)
   * through the const problem.
   */
  const OptimalControlProblem& problem() const;
  const StateManifold& manifold() const { return *manifold_; }
  const std::shared_ptr<const StateManifold>& manifoldPtr() const {
    return manifold_;
  }
  /** The task file's Euler state layout, which is the oracle's state. */
  const validation::EulerStateLayout& eulerLayout() const {
    return eulerLayout_;
  }
  /** A new Euler-chart manifold of the MPC (EulerChartStateManifold over
   * manifoldPtr()), for the chart's SQP. */
  std::shared_ptr<const StateManifold> eulerChartPtr() const;
  /** The oracle under the world's gravity. */
  const EulerMpcOracle& oracle() const { return *oracle_; }
  /** An oracle of the same model under the gravity `gravity`. */
  absl::StatusOr<std::unique_ptr<EulerMpcOracle>> oracleUnderGravity(
      const vector3_t& gravity) const;
  /** [s] The SQP's node spacing: one shooting interval. */
  scalar_t nodeTimeStep() const;
  /** [s] The MPC horizon. */
  scalar_t timeHorizon() const;
  /** The MPC's SQP settings: its integrator and node spacing. */
  const sqp::Settings& sqpSettings() const;

  /** Phi(x_e), the quaternion state of an Euler state (EulerChart); the sizes
   * are a precondition (ABSL_CHECK). */
  vector_t quaternionStateOf(const vector_t& eulerState) const;
  /** dPhi / dx_e at x_e, nx x nx_e. */
  matrix_t chartJacobian(const vector_t& eulerState) const;
  /** P_e(x_e), ndx x nx_e (square). */
  matrix_t physicalTangentJacobian(const vector_t& eulerState) const;
  /** Phi^-1(x), the Euler state of a quaternion state (EulerChart, the angles
   * wrapped); the sizes are a precondition. */
  vector_t eulerStateOf(const vector_t& quaternionState) const;

  /**
   * How a rotation g of the world acts on the quaternion state (ambient, nx x
   * nx): the world-frame blocks (the momentum, the base position and linear
   * velocity) turn by R(g), the base orientation is left-multiplied, xi -> g
   * (x) xi, and the body angular velocity and the joints stay. The dynamics of
   * the turned state under the turned gravity is the turned dynamics: f(M x, N
   * u; R(g) gravity) = M f(x, u; gravity).
   */
  matrix_t ambientWorldRotation(const vector4_t& worldRotation) const;
  /** The same on tangent vectors (ndx x ndx): the body rotation perturbation
   * stays, d(g (x) xi Exp(d)) = g (x) d(xi Exp(d)). */
  matrix_t tangentWorldRotation(const vector4_t& worldRotation) const;
  /** The same on inputs (nu x nu): world-frame wrenches turn, local basis
   * scalings and the joint entries stay. */
  matrix_t inputWorldRotation(const vector4_t& worldRotation) const;

  /** A random Euler state within `bounds`; deterministic for a given generator
   * state. */
  vector_t randomEulerState(const EulerSamplingBounds& bounds,
                            std::mt19937& generator) const;
  /**
   * A random input: contact entries around the weight carried on both feet
   * (forces within +-50 % of half the weight in the vertical, +-25 % of it
   * horizontally, moments within +-20 N m; basis scalings uniform in [0, 2]
   * times the scaling that carries half the weight on a level foot), joint
   * entries within +-`maxAbsJointEntry`.
   */
  vector_t randomInput(scalar_t maxAbsJointEntry,
                       std::mt19937& generator) const;

  /**
   * `input` with its contact entries corrected, by the least-norm change, to
   * exert no moment on the base at `eulerState`: the centroidal momentum's
   * angular rate (O1) or the whole body's angular base acceleration (O3) of the
   * oracle vanishes, so that a robot at rest under the input does not turn.
   * Linear in the moment removed.
   */
  vector_t torqueFreeInput(const vector_t& eulerState,
                           const vector_t& input) const;

  /** The task file's initial_state: the robot standing level at rest, as the
   * interface loads it. */
  const vector_t& initialState() const;

  /**
   * The input that carries the robot's weight evenly on the feet `contactFlags`
   * at `state`, vertical in the world (weightCompensatingInput(), the MRTs' and
   * reset targets' weight compensation), in the effective model's inputs.
   */
  vector_t weightCompensatingInput(const vector_t& state,
                                   const contact_flag_t& contactFlags) const;

  /**
   * The walking reference of the SQP and term tests, from initialState() at 0
   * s: a double support until kWalkingFirstLiftOff, then four single supports
   * of kWalkingSingleSupport (the left foot in stance first, then the right, in
   * turn) separated by double supports of kWalkingDoubleSupport, and a double
   * support after them; the target the initial state at 0 s and the same 1 m
   * ahead along the world x axis at 2 s, with zero inputs; the stance mode at
   * the start.
   */
  EquivalenceReference walkingReference() const;

  /** The resting warm start: initialState() held with the input that carries
   * the weight on both feet, from itself. */
  SqpCase restingWarmStart() const;

  /** Installs `reference` on the MPC's reference manager: its schedule, its
   * targets, and preSolverRun() over the horizon. */
  void installReference(const EquivalenceReference& reference) const;
  /** The MPC's reference manager, which the problem's terms read. Shared with
   * the interface. */
  SwitchedModelReferenceManager& referenceManager() const;

  /**
   * One real-time iteration (sqp_iteration 1, one thread, the feedback policy
   * on) of the MPC's SQP under `reference`, on the state manifold `manifold` in
   * place of the MPC's own: the quaternion manifold, or the Euler chart
   * (EulerChartStateManifold) for the Euler formulation's iteration. Installs
   * the reference on the MPC's reference manager first. The solver's errors (it
   * throws, which a test reports as a failure).
   *
   * From `sqpCase`: its warm start held over the horizon, from its initial
   * state. Or from `warmStart`, a primal solution the solver reads its nodes
   * from, whose state at each node of sqpNodeGrid() is `warmStartNodes` (in
   * order: the steps are taken from them) and whose input is `warmStartInput`
   * throughout, from `initState`.
   */
  RealTimeIteration realTimeIteration(
      std::shared_ptr<const StateManifold> manifold,
      const EquivalenceReference& reference,
      const SqpCase& sqpCase) const;
  RealTimeIteration realTimeIteration(
      std::shared_ptr<const StateManifold> manifold,
      const EquivalenceReference& reference,
      const vector_t& initState,
      const PrimalSolution& warmStart,
      const vector_array_t& warmStartNodes,
      const vector_t& warmStartInput) const;

  /**
   * The SQP's node grid under `reference`: the node spacing cut at the events
   * of the schedule the reference manager has after installing the reference,
   * as SqpSolver discretizes the horizon. Installs the reference.
   */
  std::vector<AnnotatedTime> sqpNodeGrid(
      const EquivalenceReference& reference) const;

  /**
   * The SQP's solution on `manifold` under `reference` from `initState` and
   * `warmStart` (its mode schedule replaced by the reference manager's), after
   * `options.iterations` iterations or the solver's own convergence test, one
   * thread, the feedback policy on. The solver's errors, as
   * realTimeIteration().
   */
  PrimalSolution sqpSolution(std::shared_ptr<const StateManifold> manifold,
                             const EquivalenceReference& reference,
                             const vector_t& initState,
                             PrimalSolution warmStart,
                             const SqpRunOptions& options) const;
  /** The same solve, with what each of its iterations did and the values of its
   * solution. */
  SqpRun sqpRun(std::shared_ptr<const StateManifold> manifold,
                const EquivalenceReference& reference,
                const vector_t& initState,
                PrimalSolution warmStart,
                const SqpRunOptions& options) const;

  /**
   * `solves` real-time solves of one SQP on the MPC's manifold under
   * `reference`, as the MPC runs them (sqp_iteration 1, the task file's other
   * settings, one thread): each at the time of node 1 of the last plan (one
   * node later), from the last plan's state there (a plant that follows the
   * plan exactly; its post-event node at an event), the first after `plan` (a
   * solution under `reference`, sqpRun()'s) and warm-started from it, every
   * later one warm-started by the solver itself from its own last solution
   * (trajectorySpread and the interpolation at its new nodes). The reference is
   * installed once; the solver's preRun() updates the reference manager per
   * solve, as in the MPC. The solver's errors, as realTimeIteration().
   */
  std::vector<RealTimeSolve> realTimeSolves(
      const EquivalenceReference& reference,
      PrimalSolution plan,
      size_t solves) const;

 private:
  /** Where worldAction() acts. */
  enum class WorldActionSpace {
    kAmbient,
    kTangent,
  };

  MpcEquivalenceFixture() = default;

  matrix_t worldAction(const vector4_t& worldRotation,
                       WorldActionSpace space) const;

  /** The settings of the fixture's solves: the task file's, with `options`, one
   * thread, the feedback policy on, no output. */
  sqp::Settings solveSettings(const SqpRunOptions& options) const;

  validation::RobotConfiguration robot_;
  // Exactly one of the two is set; the extractor and the oracle read it.
  std::unique_ptr<CentroidalMpcInterface> centroidal_;
  std::unique_ptr<WBMpcInterface> wholeBody_;
  // Shared because OCS2 shares it (OptimalControlProblem::stateManifoldPtr).
  std::shared_ptr<const StateManifold> manifold_;
  validation::EulerStateLayout eulerLayout_;
  EulerMpcOracleOptions oracleOptions_;
  std::unique_ptr<EulerMpcOracle> oracle_;
};

/**
 * The parameterized test of the MPC-level suite: the fixture of the
 * configuration GetParam() names, built in SetUp() (a fatal failure with its
 * status when it cannot be). Each test file derives its own suite from it.
 */
class MpcEquivalenceTest : public ::testing::TestWithParam<std::string> {
 protected:
  void SetUp() override;

  MpcEquivalenceFixture& mpc() { return *mpc_; }

 private:
  std::unique_ptr<MpcEquivalenceFixture> mpc_;
};

/**
 * |actual - expected| <= bound entry by entry (same shapes), with the worst
 * entry relative to its bound in the message; a NaN fails.
 */
::testing::AssertionResult isWithinEntrywiseBound(const matrix_t& actual,
                                                  const matrix_t& expected,
                                                  const matrix_t& bound);

/** max |actual - expected| <= relativeTolerance * max(floor, max |expected|),
 * with the worst entry in the message. */
::testing::AssertionResult isCloseToScale(const matrix_t& actual,
                                          const matrix_t& expected,
                                          scalar_t relativeTolerance,
                                          scalar_t floor);

}  // namespace ocs2::humanoid
