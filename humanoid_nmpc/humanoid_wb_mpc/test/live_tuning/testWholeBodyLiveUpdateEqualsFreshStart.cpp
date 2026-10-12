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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/log_severity.h"
#include "absl/base/nullability.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"
#include "ocs2_sqp/SqpSolver.h"

#include "humanoid_common_mpc/common/CostTermNames.h"
#include "humanoid_common_mpc/cost/InputQuadraticCost.h"
#include "humanoid_common_mpc/cost/TuningQuadraticStateCost.h"
#include "humanoid_common_mpc/cost/TuningQuadraticStateInputCost.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_mpc_config/yaw_pitch_roll.nproto.h"
#include "humanoid_nmpc/humanoid_wb_mpc/test/live_tuning/HotLeafPerturbation.h"
#include "humanoid_nmpc/humanoid_wb_mpc/test/live_tuning/RunningProblemFingerprint.h"
#include "humanoid_nmpc/humanoid_wb_mpc/test/live_tuning/WholeBodyLiveTuningFixture.h"
#include "humanoid_nmpc/humanoid_wb_mpc/test/live_tuning/WholeBodySolverStack.h"
#include "humanoid_wb_mpc/config/costs/EndEffectorDynamicsWeightsFromConfig.h"
#include "humanoid_wb_mpc/parameter_update/WholeBodyHotFieldAppliers.h"

/*
 * A live update of the whole-body MPC is a fresh start on the edited file: the MPC of a task file, handed an edit of the
 * file's RELOAD_HOT fields through its parameter updater before its first solve, solves exactly as the MPC built from
 * the edited file, block by block, on the shipped G1 file and on a variant whose problem carries the terms the shipped
 * one does not (variantTaskFile()), and on the shipped file with every orientation row weighted, the yaw included, which
 * the block cases leave at the shipped 0 (LeafStep::kScaled keeps a zero a zero). Every field the updater applies changes
 * what the running problem evaluates to, an updater fed nothing changes nothing, and a reload of the file the MPC started
 * from reproduces its problem. Every case runs on the quaternion layout of the whole-body state (expectQuaternionLayout()),
 * and the orientation case's solves keep a unit quaternion. The ground
 * (terrain_height) is the one hot field left out: its live change lifts the target in use once, which a fresh start does
 * not (testBaseHeightFollowsTerrain); its stance and swing feet are testWholeBodyStanceFootOnTheGround's.
 */
namespace ocs2::humanoid::live_tuning_test {
namespace {

using ::testing::_;
using ::testing::Contains;
using ::testing::HasSubstr;

// Standing, the right foot's swing and touch-down, and the start of the left foot's (WholeBodySolverStack).
constexpr size_t kNumSolves = 30;
// The policies of a live update and a fresh start agree to this, relative to max(1, |value|); expected bitwise.
constexpr scalar_t kPolicyTolerance = 1.0e-12;

/** Which task file a case runs on. */
enum class TaskFileKind {
  kShipped,
  kVariant,
};

mpc_config::TaskFile taskFileOf(TaskFileKind kind) {
  switch (kind) {
    case TaskFileKind::kShipped:
      return shippedTaskFile();
    case TaskFileKind::kVariant:
      return variantTaskFile();
  }
  ADD_FAILURE() << "unknown task file kind " << static_cast<int>(kind);
  return mpc_config::TaskFile{};
}

/** One case of the block-by-block comparison: a task file and the top-level block of its edit. */
struct BlockCase {
  TaskFileKind file = TaskFileKind::kShipped;
  std::string block;
};

std::string caseName(const testing::TestParamInfo<BlockCase>& info) {
  return absl::StrCat(info.param.file == TaskFileKind::kShipped ? "shipped_" : "variant_", info.param.block);
}

std::vector<BlockCase> blockCases() {
  // The top-level blocks of the whole-body MPC's applied fields (wholeBodyHotFieldNames()), terrain_height left out.
  const std::vector<std::string> blocks = {
      "state_weights", "input_weights",        "final_state_weights", "terminal_cost_scaling",   "task_space_foot_cost", "contacts",
      "joint_limits",  "collision_constraint", "model_settings",      "swing_trajectory_config", "multiple_shooting"};
  std::vector<BlockCase> cases;
  for (const std::string& block : blocks) {
    cases.push_back(BlockCase{.file = TaskFileKind::kShipped, .block = block});
    cases.push_back(BlockCase{.file = TaskFileKind::kVariant, .block = block});
  }
  // The shipped file lists no joint_torque_cost.
  cases.push_back(BlockCase{.file = TaskFileKind::kVariant, .block = "joint_torque_weights"});
  return cases;
}

class LiveUpdateEqualsFreshStart : public testing::TestWithParam<BlockCase> {};

TEST_P(LiveUpdateEqualsFreshStart, AnUpdateOfTheBlockSolvesAsAFreshStartOnTheEditedFile) {
  const mpc_config::TaskFile file = taskFileOf(GetParam().file);
  const mpc_config::TaskFile edited = withMovedLeaves(file, GetParam().block, LeafStep::kScaled);
  ASSERT_NE(edited, file) << "the file sets no applied leaf of " << GetParam().block;

  WholeBodySolverStack live(file, Updater::kRegistered);
  ASSERT_TRUE(live.ok());
  live.applyNow(edited);
  WholeBodySolverStack fresh(edited, Updater::kRegistered);
  ASSERT_TRUE(fresh.ok());
  live.expectQuaternionLayout();
  fresh.expectQuaternionLayout();

  const std::vector<PrimalSolution> freshPolicies = fresh.solve(kNumSolves);
  const std::vector<PrimalSolution> livePolicies = live.solve(kNumSolves);
  const scalar_t difference = policyDifference(freshPolicies, livePolicies);
  EXPECT_LE(difference, kPolicyTolerance);
  RecordProperty("policy_difference", absl::StrCat(difference));
  EXPECT_EQ(runningProblemFingerprint(fresh), runningProblemFingerprint(live)) << "the running problems differ after the solves";
}

INSTANTIATE_TEST_SUITE_P(WholeBodyBlocks, LiveUpdateEqualsFreshStart, testing::ValuesIn(blockCases()), caseName);

// The weights of the rows the robot model's tuning layout maps from Euler angles (orientation/TuningLayoutDeviation.h):
// the yaw, pitch and roll of the base orientation and of the base angular velocity, running and terminal. Each is moved
// whatever it ships with, so that a zero a future tuning ships cannot drop its row from the equality.
constexpr std::array<absl::string_view, 4> kOrientationWeightBlocks = {
    "state_weights.base_orientation", "final_state_weights.base_orientation", "state_weights.base_angular_velocity",
    "final_state_weights.base_angular_velocity"};

/** `task` with every weight of kOrientationWeightBlocks moved by LeafStep::kMoved: each nonzero, a yaw of 0 made 0.1. */
mpc_config::TaskFile withOrientationWeightsMoved(const mpc_config::TaskFile& task) {
  mpc_config::TaskFile moved = task;
  for (const absl::string_view block : kOrientationWeightBlocks) moved = withMovedLeaves(moved, block, LeafStep::kMoved);
  return moved;
}

/** Expects each of yaw, pitch and roll of `weights`, the block `block`, to be weighted. */
void expectEveryRowWeighted(const mpc_config::YawPitchRoll& weights, absl::string_view block) {
  EXPECT_NE(weights.yaw, 0.0) << block << ".yaw";
  EXPECT_NE(weights.pitch, 0.0) << block << ".pitch";
  EXPECT_NE(weights.roll, 0.0) << block << ".roll";
}

/** The weights of the running quadratic costs of every worker of a solver, with the name of each. */
struct RunningQuadraticWeights {
  std::vector<std::string> names;
  std::vector<matrix_t> weights;
};

/**
 * Returns the weights the running quadratic costs of every worker of `stack`'s solver hold: Q and R of the Tuning*
 * state-input costs, R of the input cost and Q_final of the terminal cost, in a fixed order.
 */
RunningQuadraticWeights runningQuadraticWeights(WholeBodySolverStack& stack) {
  RunningQuadraticWeights running;
  std::vector<OptimalControlProblem>& workers = stack.mpc().getSolverPtr()->getOcpDefinitions();
  for (size_t worker = 0; worker < workers.size(); ++worker) {
    OptimalControlProblem& problem = workers[worker];
    for (const absl::string_view term : {absl::string_view(kStateInputQuadraticCostTerm), absl::string_view(kStateQuadraticCostTerm)}) {
      const std::string name(term);
      if (!problem.costPtr->getTermNameMap().contains(name)) continue;
      const TuningQuadraticStateInputCost& cost = problem.costPtr->get<TuningQuadraticStateInputCost>(name);
      running.names.push_back(absl::StrCat("worker ", worker, " ", name, " Q"));
      running.weights.push_back(cost.getStateWeights());
      running.names.push_back(absl::StrCat("worker ", worker, " ", name, " R"));
      running.weights.push_back(cost.getInputWeights());
    }
    if (problem.costPtr->getTermNameMap().contains(kInputQuadraticCostTerm)) {
      running.names.push_back(absl::StrCat("worker ", worker, " ", kInputQuadraticCostTerm, " R"));
      running.weights.push_back(problem.costPtr->get<InputQuadraticCost>(kInputQuadraticCostTerm).getInputWeights());
    }
    if (problem.finalCostPtr->getTermNameMap().contains(kTerminalCostTerm)) {
      running.names.push_back(absl::StrCat("worker ", worker, " ", kTerminalCostTerm, " Q_final"));
      running.weights.push_back(problem.finalCostPtr->get<TuningQuadraticStateCost>(kTerminalCostTerm).getStateWeights());
    }
  }
  return running;
}

/** Returns the largest |(|xi| - 1)| of the base orientation over the planned states of `policies`. */
scalar_t largestQuaternionNormDeviation(const WBMpcInterface& interface, const std::vector<PrimalSolution>& policies) {
  const Eigen::Index start = static_cast<Eigen::Index>(interface.getMpcRobotModel().getBaseOrientationStartIndex());
  scalar_t largest = 0.0;
  for (const PrimalSolution& policy : policies) {
    for (const vector_t& state : policy.stateTrajectory_) largest = std::max(largest, std::abs(state.segment<4>(start).norm() - 1.0));
  }
  return largest;
}

/** The weight of the first row of the block `block` of the tuning layout in `weights` (Q or Q_final). */
scalar_t firstRowWeight(const WBMpcInterface& interface, const matrix_t& weights, absl::string_view block) {
  const absl::StatusOr<size_t> row = tuningSegmentStart(interface.getMpcRobotModel().getTuningLayout(), block);
  EXPECT_TRUE(row.ok()) << row.status();
  if (!row.ok() || *row >= static_cast<size_t>(weights.rows())) return 0.0;
  const Eigen::Index index = static_cast<Eigen::Index>(*row);
  return weights(index, index);
}

TEST(WholeBodyLiveUpdate, AnUpdateOfEveryOrientationRowSolvesAsAFreshStartOnTheEditedFile) {
  // LeafStep::kScaled keeps every zero a zero, and the shipped yaw weights are 0: here every orientation row is weighted,
  // so that the live and the fresh MPC map each of them through the robot model's tuning layout.
  const mpc_config::TaskFile file = shippedTaskFile();
  const mpc_config::TaskFile edited = withOrientationWeightsMoved(file);
  // A block the file leaves out weighs nothing, and fails here.
  const mpc_config::YawPitchRoll unweighted;
  expectEveryRowWeighted(edited.state_weights.base_orientation.value_or(unweighted), kOrientationWeightBlocks[0]);
  expectEveryRowWeighted(edited.final_state_weights.base_orientation.value_or(unweighted), kOrientationWeightBlocks[1]);
  expectEveryRowWeighted(edited.state_weights.base_angular_velocity.value_or(unweighted), kOrientationWeightBlocks[2]);
  expectEveryRowWeighted(edited.final_state_weights.base_angular_velocity.value_or(unweighted), kOrientationWeightBlocks[3]);

  WholeBodySolverStack live(file, Updater::kRegistered);
  ASSERT_TRUE(live.ok());
  const RunningQuadraticWeights started = runningQuadraticWeights(live);
  live.applyNow(edited);
  WholeBodySolverStack fresh(edited, Updater::kRegistered);
  ASSERT_TRUE(fresh.ok());
  // On the quaternion layout: the orientation rows map onto the heading-tilt residual of the base's quaternion
  // (orientation/TuningLayoutDeviation.h) and the angular-velocity rows onto its body rate.
  live.expectQuaternionLayout();
  fresh.expectQuaternionLayout();

  // The reload reached the running terms, and they weigh every orientation row, the yaw (the first row) included.
  const RunningQuadraticWeights reloaded = runningQuadraticWeights(live);
  const RunningQuadraticWeights freshStart = runningQuadraticWeights(fresh);
  ASSERT_EQ(reloaded.names, freshStart.names);
  ASSERT_EQ(reloaded.names, started.names);
  EXPECT_THAT(reloaded.names, Contains(HasSubstr(absl::StrCat(kStateQuadraticCostTerm, " Q"))));
  EXPECT_THAT(reloaded.names, Contains(HasSubstr(absl::StrCat(kTerminalCostTerm, " Q_final"))));
  for (size_t i = 0; i < reloaded.names.size(); ++i) {
    // Bitwise: both went through the same weight helper and the same setters (StateInputWeightsFromConfig.h).
    EXPECT_TRUE(reloaded.weights[i] == freshStart.weights[i]) << reloaded.names[i] << " differs from a fresh start's";
    if (absl::EndsWith(reloaded.names[i], " Q") || absl::EndsWith(reloaded.names[i], " Q_final")) {
      EXPECT_FALSE(reloaded.weights[i] == started.weights[i]) << reloaded.names[i] << " was not reloaded";
      EXPECT_NE(firstRowWeight(live.interface(), reloaded.weights[i], "base_orientation"), 0.0) << reloaded.names[i];
      EXPECT_NE(firstRowWeight(live.interface(), reloaded.weights[i], "base_angular_velocity"), 0.0) << reloaded.names[i];
    }
  }

  const std::vector<PrimalSolution> freshPolicies = fresh.solve(kNumSolves);
  const std::vector<PrimalSolution> livePolicies = live.solve(kNumSolves);
  const scalar_t difference = policyDifference(freshPolicies, livePolicies);
  EXPECT_LE(difference, kPolicyTolerance);
  RecordProperty("policy_difference", absl::StrCat(difference));
  EXPECT_EQ(runningProblemFingerprint(fresh), runningProblemFingerprint(live)) << "the running problems differ after the solves";
  // The solver stepped on the manifold: every planned orientation is a unit quaternion.
  const scalar_t normDeviation = std::max(largestQuaternionNormDeviation(live.interface(), livePolicies),
                                          largestQuaternionNormDeviation(fresh.interface(), freshPolicies));
  EXPECT_LE(normDeviation, 1.0e-12);
  RecordProperty("quaternion_norm_deviation", absl::StrCat(normDeviation));
}

TEST(WholeBodyLiveUpdate, EveryAppliedLeafChangesWhatTheProblemEvaluatesTo) {
  // A leaf counts as live when it changes what the running problem of one of the two files evaluates to (its effects,
  // not what the reload wrote): each file carries terms the other does not (the hard or soft foot constraints, the
  // friction or the contact wrench cone, the joint torque cost; the variant's swing pitch, which tilts the plane the
  // swing foot's orientation_z weighs the tilt from). A weight the whole-body foot cost would multiply by a zero error is
  // refused at start-up and on a reload alike (wholeBodyFootCostWeightsFromConfig()), and changes nothing.
  const std::set<std::string> refusedAsInert = {"task_space_foot_cost.weights.pos_x", "task_space_foot_cost.weights.pos_y",
                                                "task_space_foot_cost.weights.pos_z"};
  std::map<std::string, bool> live;
  std::set<std::string> refused;
  for (const TaskFileKind kind : {TaskFileKind::kShipped, TaskFileKind::kVariant}) {
    const mpc_config::TaskFile file = taskFileOf(kind);
    WholeBodySolverStack stack(file, Updater::kRegistered);
    ASSERT_TRUE(stack.ok());
    stack.expectQuaternionLayout();
    // One solve plans the swing trajectories the swing foot's coefficients are derived from.
    stack.solve(/*count=*/1);
    const std::vector<scalar_t> original = runningProblemEffects(stack);
    for (const std::string& leaf : appliedNumericLeaves(file)) {
      const mpc_config::TaskFile moved = withMovedLeaves(file, /*prefix=*/"", LeafStep::kMoved, leaf);
      const bool isRefused = !wholeBodyFootCostWeightsFromConfig(moved.task_space_foot_cost).ok();
      stack.applyNow(moved);
      const bool changed = runningProblemEffects(stack) != original;
      if (isRefused) {
        refused.insert(leaf);
        EXPECT_FALSE(changed) << "a refused edit of " << leaf << " changed the running problem";
      } else {
        bool& isLive = live[leaf];
        isLive = isLive || changed;
      }
      stack.applyNow(file);
      EXPECT_EQ(runningProblemEffects(stack), original) << "a reload of the file after an edit of " << leaf << " did not restore it";
    }
  }
  EXPECT_EQ(refused, refusedAsInert) << "the refused leaves are not the inert weights of the foot cost";
  // Every weight of the state, input and terminal costs alone is over a hundred leaves.
  ASSERT_GT(live.size(), 100u);
  RecordProperty("applied_leaves", static_cast<int>(live.size()));
  for (const std::pair<const std::string, bool>& leaf : live) {
    EXPECT_TRUE(leaf.second) << leaf.first << " is applied by the whole-body updater but changes nothing the problem evaluates to";
  }
}

TEST(WholeBodyLiveUpdate, AnUpdaterFedNothingSolvesAsNoUpdater) {
  for (const TaskFileKind kind : {TaskFileKind::kShipped, TaskFileKind::kVariant}) {
    const mpc_config::TaskFile file = taskFileOf(kind);
    WholeBodySolverStack withUpdater(file, Updater::kRegistered);
    WholeBodySolverStack without(file, Updater::kNone);
    ASSERT_TRUE(withUpdater.ok() && without.ok());
    withUpdater.expectQuaternionLayout();
    without.expectQuaternionLayout();
    EXPECT_EQ(policyDifference(without.solve(kNumSolves), withUpdater.solve(kNumSolves)), 0.0);
  }
}

TEST(WholeBodyLiveUpdate, AReloadOfTheShippedFileReproducesTheStartUpProblemAndReportsNothing) {
  for (const TaskFileKind kind : {TaskFileKind::kShipped, TaskFileKind::kVariant}) {
    const mpc_config::TaskFile file = taskFileOf(kind);
    WholeBodySolverStack reloaded(file, Updater::kRegistered);
    WholeBodySolverStack started(file, Updater::kNone);
    ASSERT_TRUE(reloaded.ok() && started.ok());
    reloaded.expectQuaternionLayout();
    started.expectQuaternionLayout();
    {
      // Nothing is refused, stopped or reported as taking effect at the next start.
      absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
      EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, _, HasSubstr("[MpcParameterUpdaterModule]"))).Times(0);
      EXPECT_CALL(log, Log(absl::LogSeverity::kError, _, _)).Times(0);
      log.StartCapturingLogs();
      reloaded.applyNow(file);
      log.StopCapturingLogs();
    }
    EXPECT_EQ(policyDifference(started.solve(kNumSolves), reloaded.solve(kNumSolves)), 0.0);
  }
}

TEST(WholeBodyLiveUpdate, TheUpdaterAppliesTheWholeBodyFieldsAndTheVariantCarriesEveryTermTheyWrite) {
  WholeBodySolverStack stack(variantTaskFile(), Updater::kRegistered);
  ASSERT_TRUE(stack.ok());
  stack.expectQuaternionLayout();
  ASSERT_NE(stack.updater(), nullptr);
  EXPECT_EQ(stack.updater()->appliedFields(), wholeBodyHotFieldNames());
  // Positive control of the variant: the terms the shipped file leaves out are built.
  const std::vector<std::string> leaves = appliedNumericLeaves(variantTaskFile());
  for (const char* absl_nonnull const block : {"joint_torque_weights.", "contacts.contact_wrench_cone_soft_constraint.mu",
                                               "model_settings.foot_constraint.soft_constraint_weight"}) {
    bool found = false;
    for (const std::string& leaf : leaves) found = found || absl::StartsWith(leaf, block);
    EXPECT_TRUE(found) << "the variant sets no applied leaf " << block;
  }
}

}  // namespace
}  // namespace ocs2::humanoid::live_tuning_test
