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

#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/misc/PropertyTree.h>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_wb_mpc/cost/EndEffectorDynamicsCostHelpers.h"

/**
 * EndEffectorDynamicsWeights::getWeights(), which loads the whole-body swing-foot cost weights. It used to write the
 * lin_acceleration_* and ang_acceleration_* keys into the VELOCITY weights, over the lin_velocity_* and ang_velocity_*
 * keys it had just read, and to leave the acceleration weights at their defaults. Each key now lands in its own field,
 * and the G1 whole-body task file was rewritten with the values that were in effect, so the cost the robot ran was
 * unchanged by the fix (shown on frozen copies of the block before and after, not on the live file, which is tuned).
 */
namespace ocs2::humanoid {
namespace {

constexpr absl::string_view kPrefix = "task_space_foot_cost_weights.";

// The weight groups and their axes, in the order of EndEffectorDynamicsWeights::toVector().
constexpr std::array<absl::string_view, 6> kWeightGroups = {"pos",          "orientation",      "lin_velocity",
                                                            "ang_velocity", "lin_acceleration", "ang_acceleration"};
constexpr std::array<absl::string_view, 3> kAxes = {"x", "y", "z"};
constexpr size_t kNumWeights = kWeightGroups.size() * kAxes.size();

/** The task-file key of weight `index` of EndEffectorDynamicsWeights::toVector(), e.g. "lin_velocity_y" for 7. */
std::string weightKey(size_t index) {
  return absl::StrCat(kWeightGroups[index / kAxes.size()], "_", kAxes[index % kAxes.size()]);
}

std::string runfilePath(absl::string_view relativePath) {
  std::vector<std::filesystem::path> roots;
  if (const char* srcDir = std::getenv("TEST_SRCDIR")) {
    roots.emplace_back(std::filesystem::path(srcDir) / "_main");
  }
  roots.emplace_back(std::filesystem::current_path());
  for (const std::filesystem::path& root : roots) {
    const std::filesystem::path candidate = root / std::string(relativePath);
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  return std::string();
}

/** Writes `contents` to a file of its own under the test's temporary directory and returns its path. */
std::string writeTaskFile(absl::string_view name, absl::string_view contents) {
  const std::string path = (std::filesystem::path(testing::TempDir()) / absl::StrCat("testFootWeightsLoader_", name, ".yaml")).string();
  std::ofstream out(path);
  out << contents;
  return path;
}

/**
 * The loader as it was before the fix, read for read: the same 18 keys, the acceleration keys written into the
 * velocity weights, and the acceleration weights left at their defaults.
 */
EndEffectorDynamicsWeights legacyGetWeights(const std::string& taskFile, const std::string& prefix) {
  PropertyTree pt;
  loadData::readPropertyTree(taskFile, pt);
  std::vector<scalar_t> values(kNumWeights, 0.0);
  for (size_t i = 0; i < kNumWeights; ++i) {
    loadData::loadPtreeValue(pt, values[i], absl::StrCat(prefix, weightKey(i)), /*verbose=*/false);
  }
  EndEffectorDynamicsWeights weights;
  weights.contactPositionErrorWeight = vector3_t(values[0], values[1], values[2]);
  weights.contactOrientationErrorWeight = vector3_t(values[3], values[4], values[5]);
  weights.contactLinearVelocityErrorWeight = vector3_t(values[6], values[7], values[8]);
  weights.contactAngularVelocityErrorWeight = vector3_t(values[9], values[10], values[11]);
  weights.contactLinearVelocityErrorWeight = vector3_t(values[12], values[13], values[14]);
  weights.contactAngularVelocityErrorWeight = vector3_t(values[15], values[16], values[17]);
  return weights;
}

// The parity record of the loader fix, as two frozen copies of the G1 whole-body block: as it shipped before the fix,
// and as the fix rewrote it. Loaded by the legacy loader the first gives the weights the robot ran with; the second
// must give the same weights through the fixed loader. Both are history, so neither follows later tuning of the task
// file (which TheShippedG1FileNamesEveryWeight checks only for properties).
constexpr absl::string_view kLegacyShippedBlock = R"yaml(
task_space_foot_cost_weights:
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
)yaml";

constexpr absl::string_view kFixedShippedBlock = R"yaml(
task_space_foot_cost_weights:
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
)yaml";

TEST(FootWeightsLoader, EachKeyLandsInItsOwnField) {
  // Every key a different value, so that a key read into another key's field cannot go unnoticed.
  std::string contents = "task_space_foot_cost_weights:\n";
  for (size_t i = 0; i < kNumWeights; ++i) {
    absl::StrAppend(&contents, "  ", weightKey(i), ": ", i + 1, ".5\n");
  }
  const std::string taskFile = writeTaskFile("distinct", contents);
  const VECTOR18_T<scalar_t> weights = EndEffectorDynamicsWeights::getWeights(taskFile, std::string(kPrefix), /*verbose=*/false).toVector();
  for (size_t i = 0; i < kNumWeights; ++i) {
    EXPECT_EQ(weights(i), static_cast<scalar_t>(i + 1) + 0.5) << weightKey(i);
  }
  // Positive control: the legacy loader lost the velocity keys and the acceleration fields on the same file.
  const VECTOR18_T<scalar_t> legacy = legacyGetWeights(taskFile, std::string(kPrefix)).toVector();
  EXPECT_EQ(legacy(6), 13.5) << "the legacy loader wrote lin_acceleration_x into the linear velocity weight";
  EXPECT_EQ(legacy(12), EndEffectorDynamicsWeights().contactLinearAccelerationErrorWeight(0));
}

TEST(FootWeightsLoader, TheRewrittenBlockAppliesTheWeightsTheLegacyLoaderApplied) {
  const std::string legacyTaskFile = writeTaskFile("legacy_shipped", kLegacyShippedBlock);
  const std::string fixedTaskFile = writeTaskFile("fixed_shipped", kFixedShippedBlock);
  const VECTOR18_T<scalar_t> appliedBefore = legacyGetWeights(legacyTaskFile, std::string(kPrefix)).toVector();
  const VECTOR18_T<scalar_t> appliedAfter =
      EndEffectorDynamicsWeights::getWeights(fixedTaskFile, std::string(kPrefix), /*verbose=*/false).toVector();
  // Bit for bit: the cost takes the square roots of these as CppAD parameters, so the problem was unchanged exactly.
  for (size_t i = 0; i < kNumWeights; ++i) {
    EXPECT_EQ(appliedAfter(i), appliedBefore(i)) << weightKey(i);
  }
  // Positive control: the legacy block read by the fixed loader is NOT what ran, so the values had to move in the file.
  const VECTOR18_T<scalar_t> legacyBlockFixedLoader =
      EndEffectorDynamicsWeights::getWeights(legacyTaskFile, std::string(kPrefix), /*verbose=*/false).toVector();
  EXPECT_NE(legacyBlockFixedLoader, appliedBefore);
}

TEST(FootWeightsLoader, TheShippedG1FileNamesEveryWeight) {
  // Properties only, so that tuning the block stays a tuning: every key the loader reads is present (a misspelled key
  // would silently keep its default), and every weight is finite and non-negative.
  // LINT.IfChange(robot_files)
  const std::string shippedTaskFile = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml");
  // LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc/BUILD.bazel:foot_weights_loader_test_data)
  ASSERT_FALSE(shippedTaskFile.empty()) << "the G1 whole-body task file is not in the runfiles";
  PropertyTree pt;
  loadData::readPropertyTree(shippedTaskFile, pt);
  for (size_t i = 0; i < kNumWeights; ++i) {
    EXPECT_NE(pt.findChild(absl::StrCat(kPrefix, weightKey(i))), nullptr) << weightKey(i);
  }
  const VECTOR18_T<scalar_t> weights =
      EndEffectorDynamicsWeights::getWeights(shippedTaskFile, std::string(kPrefix), /*verbose=*/false).toVector();
  EXPECT_TRUE(weights.allFinite());
  EXPECT_GE(weights.minCoeff(), 0.0);
}

}  // namespace
}  // namespace ocs2::humanoid
