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

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_mpc_config/task_file.pb.h"
#include "humanoid_wb_mpc/config/costs/EndEffectorDynamicsWeightsFromConfig.h"
#include "humanoid_wb_mpc/cost/EndEffectorDynamicsCostHelpers.h"
#include "nproto/Textproto.h"

/**
 * The whole-body swing-foot cost weights as the task file gives them, task_space_foot_cost.weights, read by the strict
 * parser and the weights' conversion (endEffectorDynamicsWeightsFromConfig(), with which wholeBodyFootCostWeightsFromConfig()
 * ends after refusing what the foot cost does not have). The loader they replaced (EndEffectorDynamicsWeights::getWeights())
 * once wrote the lin_acceleration_* and ang_acceleration_* weights into the VELOCITY weights, over the lin_velocity_* and
 * ang_velocity_* weights it had just read, and left the acceleration weights at their defaults. Each field lands in its
 * own weight, and the G1 whole-body task file was rewritten with the values that were in effect, so the cost the robot ran
 * was unchanged by the fix (shown on frozen copies of the block before and after, not on the live file, which is tuned).
 */
namespace ocs2::humanoid {
namespace {

using ::google::protobuf::Descriptor;
using ::google::protobuf::FieldDescriptor;
using ::google::protobuf::Message;
using ::google::protobuf::Reflection;

// The weight groups and their axes, in the order of EndEffectorDynamicsWeights::toVector().
constexpr std::array<absl::string_view, 6> kWeightGroups = {"pos",          "orientation",      "lin_velocity",
                                                            "ang_velocity", "lin_acceleration", "ang_acceleration"};
constexpr std::array<absl::string_view, 3> kAxes = {"x", "y", "z"};
constexpr size_t kNumWeights = kWeightGroups.size() * kAxes.size();

/** The field of weight `index` of EndEffectorDynamicsWeights::toVector(), e.g. "lin_velocity_y" for 7. */
std::string weightField(size_t index) {
  return absl::StrCat(kWeightGroups[index / kAxes.size()], "_", kAxes[index % kAxes.size()]);
}

std::string runfilePath(absl::string_view relativePath) {
  std::vector<std::filesystem::path> roots;
  if (const char* absl_nullable srcDir = std::getenv("TEST_SRCDIR")) {
    roots.emplace_back(std::filesystem::path(srcDir) / "_main");
  }
  roots.emplace_back(std::filesystem::current_path());
  for (const std::filesystem::path& root : roots) {
    const std::filesystem::path candidate = root / std::string(relativePath);
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  return std::string();
}

/**
 * The foot cost weights of the task file `contents`, written to a file of its own under the test's temporary directory
 * and read as the stack reads a task file (loadTaskFile(), endEffectorDynamicsWeightsFromConfig()); a test failure and
 * the default weights when it does not read or convert.
 */
EndEffectorDynamicsWeights footWeightsOf(absl::string_view name, absl::string_view contents) {
  const std::string path =
      (std::filesystem::path(testing::TempDir()) / absl::StrCat("testFootWeightsLoader_", name, ".textproto")).string();
  std::ofstream(path) << contents;
  const absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(path);
  EXPECT_TRUE(task.ok()) << task.status();
  if (!task.ok()) return EndEffectorDynamicsWeights();
  const absl::StatusOr<EndEffectorDynamicsWeights> weights =
      endEffectorDynamicsWeightsFromConfig(task->task_space_foot_cost.weights, "task_space_foot_cost.weights");
  EXPECT_TRUE(weights.ok()) << weights.status();
  return weights.ok() ? *weights : EndEffectorDynamicsWeights();
}

/**
 * What the loader applied before the fix to a block whose weights `read` holds, each in its own entry: the same 18
 * weights read, the acceleration weights written into the velocity weights, and the acceleration weights left at their
 * defaults.
 */
EndEffectorDynamicsWeights legacyApplied(const EndEffectorDynamicsWeights& read) {
  const VECTOR18_T<scalar_t> values = read.toVector();
  EndEffectorDynamicsWeights weights;
  weights.contactPositionErrorWeight = values.segment<3>(0);
  weights.contactOrientationErrorWeight = values.segment<3>(3);
  weights.contactLinearVelocityErrorWeight = values.segment<3>(6);
  weights.contactAngularVelocityErrorWeight = values.segment<3>(9);
  weights.contactLinearVelocityErrorWeight = values.segment<3>(12);
  weights.contactAngularVelocityErrorWeight = values.segment<3>(15);
  return weights;
}

// The parity record of the loader fix, as two frozen copies of the G1 whole-body block: as it shipped before the fix,
// and as the fix rewrote it. Read by the legacy loader the first gives the weights the robot ran with; the second must
// give the same weights as each field is read now. Both are history, so neither follows later tuning of the task file
// (which TheShippedG1FileNamesEveryWeight checks only for properties).
constexpr absl::string_view kLegacyShippedBlock = R"pb(
  task_space_foot_cost {
    weights {
      pos_x: 0.0
      pos_y: 0.0
      pos_z: 0.0
      orientation_x: 10000.0
      orientation_y: 10000.0
      orientation_z: 0.0
      lin_velocity_x: 50.0
      lin_velocity_y: 50.0
      lin_velocity_z: 0.0
      ang_velocity_x: 100.0
      ang_velocity_y: 100.0
      ang_velocity_z: 100.0
      lin_acceleration_x: 5.0
      lin_acceleration_y: 5.0
      lin_acceleration_z: 0.0
      ang_acceleration_x: 2.0
      ang_acceleration_y: 2.0
      ang_acceleration_z: 2.0
    }
  }
)pb";

constexpr absl::string_view kFixedShippedBlock = R"pb(
  task_space_foot_cost {
    weights {
      pos_x: 0.0
      pos_y: 0.0
      pos_z: 0.0
      orientation_x: 10000.0
      orientation_y: 10000.0
      orientation_z: 0.0
      lin_velocity_x: 5.0
      lin_velocity_y: 5.0
      lin_velocity_z: 0.0
      ang_velocity_x: 2.0
      ang_velocity_y: 2.0
      ang_velocity_z: 2.0
      lin_acceleration_x: 0.01
      lin_acceleration_y: 0.01
      lin_acceleration_z: 0.01
      ang_acceleration_x: 0.01
      ang_acceleration_y: 0.01
      ang_acceleration_z: 0.01
    }
  }
)pb";

TEST(FootWeightsLoader, EachFieldLandsInItsOwnWeight) {
  // Every field a different value, so that a field read into another field's weight cannot go unnoticed.
  std::string contents = "task_space_foot_cost {\n  weights {\n";
  for (size_t i = 0; i < kNumWeights; ++i) {
    absl::StrAppend(&contents, "    ", weightField(i), ": ", i + 1, ".5\n");
  }
  absl::StrAppend(&contents, "  }\n}\n");
  const EndEffectorDynamicsWeights read = footWeightsOf("distinct", contents);
  const VECTOR18_T<scalar_t> weights = read.toVector();
  for (size_t i = 0; i < kNumWeights; ++i) {
    EXPECT_EQ(weights(i), static_cast<scalar_t>(i + 1) + 0.5) << weightField(i);
  }
  // Positive control: the legacy loader lost the velocity weights and the acceleration fields on the same block.
  const VECTOR18_T<scalar_t> legacy = legacyApplied(read).toVector();
  EXPECT_EQ(legacy(6), 13.5) << "the legacy loader wrote lin_acceleration_x into the linear velocity weight";
  EXPECT_EQ(legacy(12), EndEffectorDynamicsWeights().contactLinearAccelerationErrorWeight(0));
}

TEST(FootWeightsLoader, TheRewrittenBlockAppliesTheWeightsTheLegacyLoaderApplied) {
  const EndEffectorDynamicsWeights legacyBlock = footWeightsOf("legacy_shipped", kLegacyShippedBlock);
  const VECTOR18_T<scalar_t> appliedBefore = legacyApplied(legacyBlock).toVector();
  const VECTOR18_T<scalar_t> appliedAfter = footWeightsOf("fixed_shipped", kFixedShippedBlock).toVector();
  // Bit for bit: the cost takes the square roots of these as CppAD parameters, so the problem was unchanged exactly.
  for (size_t i = 0; i < kNumWeights; ++i) {
    EXPECT_EQ(appliedAfter(i), appliedBefore(i)) << weightField(i);
  }
  // Positive control: the legacy block read field by field is NOT what ran, so the values had to move in the file.
  EXPECT_NE(legacyBlock.toVector(), appliedBefore);
}

TEST(FootWeightsLoader, TheShippedG1FileNamesEveryWeight) {
  // Properties only, so that tuning the block stays a tuning: every weight of the schema is given (an absent one would
  // silently be 0), and every weight is finite and non-negative.
  // LINT.IfChange(robot_files)
  const std::string shippedTaskFile = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto");
  // LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc/BUILD.bazel:foot_weights_loader_test_data)
  ASSERT_FALSE(shippedTaskFile.empty()) << "the G1 whole-body task file is not in the runfiles";
  const absl::StatusOr<humanoid_mpc_config::TaskFile> message = nproto::ParseTextprotoFile<humanoid_mpc_config::TaskFile>(shippedTaskFile);
  ASSERT_TRUE(message.ok()) << message.status();
  const Message& weightsMessage = message->task_space_foot_cost().weights();
  const Descriptor* absl_nonnull descriptor = weightsMessage.GetDescriptor();
  const Reflection* absl_nonnull reflection = weightsMessage.GetReflection();
  ASSERT_EQ(static_cast<size_t>(descriptor->field_count()), kNumWeights) << "a weight of the schema is not one of this test's";
  for (int index = 0; index < descriptor->field_count(); ++index) {
    const FieldDescriptor* absl_nonnull field = descriptor->field(index);
    EXPECT_TRUE(reflection->HasField(weightsMessage, field)) << "task_space_foot_cost.weights." << field->name();
  }
  const absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(shippedTaskFile);
  ASSERT_TRUE(task.ok()) << task.status();
  const absl::StatusOr<EndEffectorDynamicsWeights> weights = wholeBodyFootCostWeightsFromConfig(task->task_space_foot_cost);
  ASSERT_TRUE(weights.ok()) << weights.status();
  EXPECT_TRUE(weights->toVector().allFinite());
  EXPECT_GE(weights->toVector().minCoeff(), 0.0);
}

}  // namespace
}  // namespace ocs2::humanoid
