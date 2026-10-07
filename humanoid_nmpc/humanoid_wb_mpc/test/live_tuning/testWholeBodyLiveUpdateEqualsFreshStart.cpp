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

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/log_severity.h"
#include "absl/base/nullability.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"

#include "humanoid_mpc_config/task_file.nproto.h"
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
 * one does not (variantTaskFile()). Every field the updater applies changes what the running problem evaluates to, an
 * updater fed nothing changes nothing, and a reload of the file the MPC started from reproduces its problem. The ground
 * (terrain_height) is the one hot field left out: its live change lifts the target in use once, which a fresh start does
 * not (testBaseHeightFollowsTerrain); its stance and swing feet are testWholeBodyStanceFootOnTheGround's.
 */
namespace ocs2::humanoid::live_tuning_test {
namespace {

using ::testing::_;
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

  const std::vector<PrimalSolution> freshPolicies = fresh.solve(kNumSolves);
  const std::vector<PrimalSolution> livePolicies = live.solve(kNumSolves);
  const scalar_t difference = policyDifference(freshPolicies, livePolicies);
  EXPECT_LE(difference, kPolicyTolerance);
  RecordProperty("policy_difference", absl::StrCat(difference));
  EXPECT_EQ(runningProblemFingerprint(fresh), runningProblemFingerprint(live)) << "the running problems differ after the solves";
}

INSTANTIATE_TEST_SUITE_P(WholeBodyBlocks, LiveUpdateEqualsFreshStart, testing::ValuesIn(blockCases()), caseName);

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
    EXPECT_EQ(policyDifference(without.solve(kNumSolves), withUpdater.solve(kNumSolves)), 0.0);
  }
}

TEST(WholeBodyLiveUpdate, AReloadOfTheShippedFileReproducesTheStartUpProblemAndReportsNothing) {
  for (const TaskFileKind kind : {TaskFileKind::kShipped, TaskFileKind::kVariant}) {
    const mpc_config::TaskFile file = taskFileOf(kind);
    WholeBodySolverStack reloaded(file, Updater::kRegistered);
    WholeBodySolverStack started(file, Updater::kNone);
    ASSERT_TRUE(reloaded.ok() && started.ok());
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
