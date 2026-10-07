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

#include <string>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/config/ConfigReload.h"
#include "humanoid_common_mpc/config/reference/ReferenceFromConfig.h"
#include "humanoid_mpc_config/contact_planning_file.pb.h"
#include "humanoid_mpc_config/reference_file.pb.h"
#include "humanoid_mpc_config/task_file.pb.h"

/**
 * The reload classes of the configuration schemas as the C++ side reads them (ConfigReload.h): the inheritance of the
 * (humanoid_mpc_config.tuning) options, the fields a reload leaves for the next start, and that the fields of the
 * reference file and the contact planner's file the schema marks RELOAD_HOT are the ones their reloads apply. The task
 * file's half of test T8 of humanoid_nmpc/humanoid_mpc_config/README.md, per formulation, is
 * humanoid_nmpc/humanoid_mpc_validation/test/testHotFieldCoverage.cpp.
 */
namespace ocs2::humanoid {
namespace {

/** The leaf of `leaves` at `path`; a missing one fails the test and is a default leaf. */
ConfigLeaf leafAt(const std::vector<ConfigLeaf>& leaves, absl::string_view path) {
  for (const ConfigLeaf& leaf : leaves) {
    if (leaf.path == path) return leaf;
  }
  ADD_FAILURE() << path << " is not a field of the schema";
  return ConfigLeaf{};
}

/** The paths of what changedStartUpFields() found between `running` and `reloaded`. */
std::vector<std::string> changedPaths(const google::protobuf::Message& running, const google::protobuf::Message& reloaded) {
  std::vector<std::string> paths;
  for (const ConfigChange& change : changedStartUpFields(running, reloaded)) paths.push_back(change.path);
  return paths;
}

/** Whether the declared path `declared` stands for `path`: it is the path, or a block above it. */
bool covers(absl::string_view declared, absl::string_view path) {
  return path == declared || absl::StartsWith(path, absl::StrCat(declared, ".")) || absl::StartsWith(path, absl::StrCat(declared, "["));
}

TEST(ConfigReload, ABlocksOptionsApplyBelowItUnlessAFieldSetsItsOwn) {
  const std::vector<ConfigLeaf> leaves = configLeaves(*humanoid_mpc_config::TaskFile::descriptor());
  // A HOT block makes every weight below it hot, through a shared message (Xyz) and a repeated block (JointValue).
  EXPECT_EQ(leafAt(leaves, "state_weights.base_position.x").reload, ConfigReload::kHot);
  EXPECT_EQ(leafAt(leaves, "state_weights.joint_positions[*].value").reload, ConfigReload::kHot);
  // A START_UP block's own HOT field is hot; its other fields are not.
  EXPECT_EQ(leafAt(leaves, "contacts.basis_scaling_regularization").reload, ConfigReload::kHot);
  EXPECT_EQ(leafAt(leaves, "contacts.contact_rectangle.x_min").reload, ConfigReload::kStartUp);
  EXPECT_EQ(leafAt(leaves, "multiple_shooting.sqp_iteration").reload, ConfigReload::kHot);
  EXPECT_EQ(leafAt(leaves, "multiple_shooting.dt").reload, ConfigReload::kStartUp);
  // Every element of a repeated block, written "[*]" as remote_control/config_schema.py writes it.
  EXPECT_EQ(leafAt(leaves, "task_space_costs[*].weights.pos_x").reload, ConfigReload::kHot);
  // What the GUI renders: numbers, bools, name lists and registry names, never excluded fields or plain text.
  EXPECT_TRUE(leafAt(leaves, "state_weights.scaling").tunable);
  EXPECT_TRUE(leafAt(leaves, "costs").tunable);
  EXPECT_TRUE(leafAt(leaves, "contact_estimator").tunable) << "a registry name";
  EXPECT_FALSE(leafAt(leaves, "state_weights.joint_positions[*].joint").tunable) << "a key is text";
  EXPECT_FALSE(leafAt(leaves, "multiple_shooting.n_threads").tunable) << "excluded with a reason";
  // A field without an option takes its block's (the registry being each field's own).
  EXPECT_FALSE(leafAt(leaves, "model_settings.robot_name").tunable);
  EXPECT_EQ(leafAt(leaves, "model_settings.robot_name").reload, ConfigReload::kStartUp);
}

TEST(ConfigReload, TheConsumerAndTheFormulationsApplyBelowABlockUnlessAFieldSetsItsOwn) {
  const std::vector<ConfigLeaf> leaves = configLeaves(*humanoid_mpc_config::TaskFile::descriptor());
  // The robot reads the controller-side settings, the MPC everything with no consumer.
  EXPECT_EQ(leafAt(leaves, "telemetry_frequency").consumer, "robot");
  EXPECT_EQ(leafAt(leaves, "state_weights.base_position.x").consumer, "");
  // A block one formulation reads passes its formulation on, through a shared message.
  EXPECT_THAT(leafAt(leaves, "com_weights.x").formulations, testing::ElementsAre(kCentroidalFormulation));
  EXPECT_THAT(leafAt(leaves, "joint_torque_weights.joints[*].value").formulations, testing::ElementsAre(kWholeBodyFormulation));
  // A field of a block every formulation reads names its own.
  EXPECT_THAT(leafAt(leaves, "model_settings.foot_constraint.linear_acceleration_error_gain_z").formulations,
              testing::ElementsAre(kWholeBodyFormulation));
  EXPECT_THAT(leafAt(leaves, "model_settings.foot_constraint.position_error_gain_z").formulations, testing::IsEmpty());
  EXPECT_THAT(leafAt(leaves, "state_weights.base_position.x").formulations, testing::IsEmpty()) << "every formulation reads it";
}

TEST(ConfigReload, TheFieldsAReloadLeavesForTheNextStartAreTheNonHotOnesThatDiffer) {
  humanoid_mpc_config::TaskFile running;
  running.add_costs("state_quadratic_cost");
  running.mutable_initial_state()->add_joint_positions()->set_value(1.0);
  EXPECT_THAT(changedPaths(running, running), testing::IsEmpty());

  humanoid_mpc_config::TaskFile reloaded = running;
  // Hot fields are applied, so they are not reported.
  reloaded.mutable_state_weights()->set_scaling(3.0);
  reloaded.set_terrain_height(0.1);
  reloaded.mutable_multiple_shooting()->set_sqp_iteration(7);
  EXPECT_THAT(changedPaths(running, reloaded), testing::IsEmpty());

  // Start-up fields that differ are, by their path: a value, an element of a repeated block by its index, a list whose
  // size differs, and an optional value set on one side only.
  reloaded.set_contact_schedule_source("contact_planner");
  reloaded.mutable_initial_state()->mutable_joint_positions(0)->set_value(2.0);
  reloaded.add_costs("input_quadratic_cost");
  reloaded.set_telemetry_frequency(50.0);
  reloaded.mutable_multiple_shooting()->set_dt(0.05);
  EXPECT_THAT(changedPaths(running, reloaded),
              testing::UnorderedElementsAre("contact_schedule_source", "initial_state.joint_positions[0].value", "costs",
                                            "telemetry_frequency", "multiple_shooting.dt"));

  // Two messages of different files have nothing in common: the file message is named.
  EXPECT_THAT(changedPaths(running, humanoid_mpc_config::ReferenceFile()), testing::ElementsAre("humanoid_mpc_config.TaskFile"));
}

TEST(ConfigReload, TheStructureOfAHotRepeatedBlockIsLeftForTheNextStart) {
  // task_space_costs is RELOAD_HOT for its weights, but an entry's name and link_name decide which link costs the
  // problem was assembled with.
  humanoid_mpc_config::TaskFile running;
  humanoid_mpc_config::TaskSpaceCostConfig& torso = *running.add_task_space_costs();
  torso.set_name("torso");
  torso.set_link_name("torso_link");
  torso.mutable_weights()->set_pos_x(1.0);

  humanoid_mpc_config::TaskFile reweighted = running;
  reweighted.mutable_task_space_costs(0)->mutable_weights()->set_pos_x(2.0);
  EXPECT_THAT(changedPaths(running, reweighted), testing::IsEmpty()) << "a weight is applied";

  humanoid_mpc_config::TaskFile retargeted = running;
  retargeted.mutable_task_space_costs(0)->set_link_name("pelvis");
  EXPECT_THAT(changedPaths(running, retargeted), testing::ElementsAre("task_space_costs[0].link_name"));

  humanoid_mpc_config::TaskFile added = running;
  humanoid_mpc_config::TaskSpaceCostConfig& head = *added.add_task_space_costs();
  head.set_name("head");
  head.set_link_name("head_link");
  head.mutable_weights()->set_pos_z(5.0);
  EXPECT_THAT(changedPaths(running, added), testing::ElementsAre("task_space_costs[1].name", "task_space_costs[1].link_name"));
  EXPECT_THAT(changedPaths(added, running), testing::ElementsAre("task_space_costs[1].name", "task_space_costs[1].link_name"))
      << "a removed entry is named as an added one";
}

TEST(ConfigReload, AChangeCarriesWhoReadsTheField) {
  humanoid_mpc_config::TaskFile running;
  humanoid_mpc_config::TaskFile reloaded = running;
  reloaded.set_telemetry_frequency(50.0);
  reloaded.mutable_multiple_shooting()->set_dt(0.05);
  const std::vector<ConfigChange> changes = changedStartUpFields(running, reloaded);
  ASSERT_EQ(changes.size(), 2u);
  for (const ConfigChange& change : changes) {
    EXPECT_EQ(change.reload, ConfigReload::kStartUp) << change.path;
    EXPECT_EQ(isReadByTheMpc(change.consumer), change.path == "multiple_shooting.dt") << change.path;
  }
  EXPECT_TRUE(isReadByFormulation(/*formulations=*/{}, kWholeBodyFormulation)) << "no formulation named: every one reads it";
  const std::vector<std::string> centroidal = {kCentroidalFormulation};
  EXPECT_TRUE(isReadByFormulation(centroidal, kCentroidalFormulation));
  EXPECT_FALSE(isReadByFormulation(centroidal, kWholeBodyFormulation));
}

TEST(ConfigReload, AReloadOfTheReferenceFileAppliesExactlyTheFieldsTheSchemaMarksHot) {
  // The command limits' reloaders (TargetTrajectoriesCalculatorBase and ProceduralMpcMotionManager::applyCommandLimits)
  // are what makes a field of the reference file hot.
  const absl::Span<const absl::string_view> declared = hotReferenceFileFields();
  std::vector<bool> used;
  used.resize(declared.size());
  for (const ConfigLeaf& leaf : configLeaves(*humanoid_mpc_config::ReferenceFile::descriptor())) {
    bool isDeclared = false;
    for (size_t i = 0; i < declared.size(); ++i) {
      if (covers(declared[i], leaf.path)) {
        isDeclared = true;
        used[i] = true;
      }
    }
    if (!leaf.tunable) continue;
    EXPECT_EQ(leaf.reload == ConfigReload::kHot, isDeclared)
        << leaf.path
        << (isDeclared ? " is applied by a reload, but the schema does not mark it RELOAD_HOT"
                       : " is marked RELOAD_HOT, but no reloader applies it (hotReferenceFileFields())");
  }
  for (size_t i = 0; i < declared.size(); ++i) {
    EXPECT_TRUE(used[i]) << declared[i] << " names no field of the reference file";
  }
}

TEST(ConfigReload, TheContactPlannersFileIsAppliedWhole) {
  // MpcParameterUpdaterModule hands the whole file to ContactPlannerModule::setConfig(): every field the GUI renders is hot.
  for (const ConfigLeaf& leaf : configLeaves(*humanoid_mpc_config::ContactPlanningFile::descriptor())) {
    if (!leaf.tunable) continue;
    EXPECT_EQ(leaf.reload, ConfigReload::kHot) << leaf.path << " is applied by a reload of the planner's file";
  }
}

}  // namespace
}  // namespace ocs2::humanoid
