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
#include <string>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_centroidal_mpc/mrt/CentroidalMpcParameterUpdater.h"
#include "humanoid_common_mpc/config/ConfigReload.h"
#include "humanoid_mpc_config/task_file.pb.h"
#include "humanoid_wb_mpc/parameter_update/WholeBodyHotFieldAppliers.h"

/*
 * The reload classes of the task file's schema against what each formulation's parameter updater applies (T8,
 * humanoid_nmpc/humanoid_mpc_config/README.md): for the centroidal and the whole-body MPC, a tunable field the formulation
 * reads is RELOAD_HOT exactly when its updater applies it; a field another formulation reads alone, and a field the
 * robot process or the GUI reads, is applied by neither; and every name of the updaters' lists is a field of the schema.
 * It compares the schema's annotations with the appliers' static field lists (centroidalHotFieldNames(),
 * wholeBodyHotFieldNames()) and builds no MPC; the node tests check that the updater they build applies those lists.
 */
namespace ocs2::humanoid {
namespace {

/** A formulation and the fields its updater applies. */
struct Formulation {
  std::string name;
  std::vector<std::string> appliedFields;
};

std::vector<Formulation> formulations() {
  return {Formulation{.name = kCentroidalFormulation, .appliedFields = centroidalHotFieldNames()},
          Formulation{.name = kWholeBodyFormulation, .appliedFields = wholeBodyHotFieldNames()}};
}

/** Whether the field path `name` stands for the leaf at `path`: the leaf itself, or a block above it. */
bool covers(absl::string_view name, absl::string_view path) {
  return path == name || absl::StartsWith(path, absl::StrCat(name, ".")) || absl::StartsWith(path, absl::StrCat(name, "["));
}

bool coveredBy(const std::vector<std::string>& names, absl::string_view path) {
  return std::any_of(names.begin(), names.end(), [path](const std::string& name) { return covers(name, path); });
}

std::vector<ConfigLeaf> taskFileLeaves() {
  return configLeaves(*humanoid_mpc_config::TaskFile::descriptor());
}

TEST(HotFieldCoverage, EachFormulationAppliesExactlyTheHotFieldsItReads) {
  const std::vector<ConfigLeaf> leaves = taskFileLeaves();
  ASSERT_FALSE(leaves.empty());
  for (const Formulation& formulation : formulations()) {
    SCOPED_TRACE(formulation.name);
    ASSERT_FALSE(formulation.appliedFields.empty());
    for (const ConfigLeaf& leaf : leaves) {
      if (!leaf.tunable || !isReadByTheMpc(leaf.consumer) || !isReadByFormulation(leaf.formulations, formulation.name)) continue;
      const bool applied = coveredBy(formulation.appliedFields, leaf.path);
      if (leaf.reload == ConfigReload::kHot) {
        EXPECT_TRUE(applied) << leaf.path << " is RELOAD_HOT and read by the " << formulation.name
                             << " MPC, whose parameter updater does not apply it: apply it, or mark it RELOAD_START_UP";
      } else {
        EXPECT_FALSE(applied) << leaf.path << " is applied by the " << formulation.name
                              << " MPC's parameter updater but not marked RELOAD_HOT";
      }
    }
  }
}

TEST(HotFieldCoverage, NoFormulationAppliesAFieldItDoesNotRead) {
  for (const Formulation& formulation : formulations()) {
    for (const ConfigLeaf& leaf : taskFileLeaves()) {
      if (isReadByFormulation(leaf.formulations, formulation.name)) continue;
      EXPECT_FALSE(coveredBy(formulation.appliedFields, leaf.path))
          << leaf.path << " is read by another formulation only, yet the " << formulation.name << " MPC's parameter updater applies it";
    }
  }
}

TEST(HotFieldCoverage, TheRobotsAndTheGuisFieldsAreAppliedByNoMpc) {
  for (const ConfigLeaf& leaf : taskFileLeaves()) {
    if (isReadByTheMpc(leaf.consumer)) continue;
    for (const Formulation& formulation : formulations()) {
      EXPECT_FALSE(coveredBy(formulation.appliedFields, leaf.path))
          << leaf.path << " is read by the " << leaf.consumer << ", yet the " << formulation.name << " MPC's parameter updater applies it";
    }
    // What the robot process applies while it runs (the controller-side settings) is RELOAD_HOT as well.
    if (leaf.consumer == kRobotConsumer && leaf.tunable) {
      EXPECT_NE(leaf.reload, ConfigReload::kUnspecified) << leaf.path << " says neither when it applies";
    }
  }
}

TEST(HotFieldCoverage, EveryAppliedNameIsAFieldOfTheSchema) {
  const std::vector<ConfigLeaf> leaves = taskFileLeaves();
  for (const Formulation& formulation : formulations()) {
    for (const std::string& name : formulation.appliedFields) {
      const bool named = std::any_of(leaves.begin(), leaves.end(), [&name](const ConfigLeaf& leaf) { return covers(name, leaf.path); });
      EXPECT_TRUE(named) << "the " << formulation.name << " MPC's parameter updater applies " << name
                         << ", which no field of the task file is";
    }
  }
}

TEST(HotFieldCoverage, EveryFormulationNameOfTheSchemaIsAFormulation) {
  const std::vector<Formulation> known = formulations();
  for (const ConfigLeaf& leaf : taskFileLeaves()) {
    for (const std::string& name : leaf.formulations) {
      const bool isKnown =
          std::any_of(known.begin(), known.end(), [&name](const Formulation& formulation) { return formulation.name == name; });
      EXPECT_TRUE(isKnown) << leaf.path << " names the formulation \"" << name << "\", which is not \"" << kCentroidalFormulation
                           << "\" or \"" << kWholeBodyFormulation << "\"";
    }
  }
}

TEST(HotFieldCoverage, TheFormulationsDifferInTheFieldsOnlyOneReads) {
  // Positive controls of the annotations: a block each formulation reads alone is in its list only.
  const std::vector<std::string> centroidal = centroidalHotFieldNames();
  const std::vector<std::string> wholeBody = wholeBodyHotFieldNames();
  EXPECT_TRUE(coveredBy(centroidal, "com_weights.x"));
  EXPECT_FALSE(coveredBy(wholeBody, "com_weights.x"));
  EXPECT_TRUE(coveredBy(wholeBody, "joint_torque_weights.scaling"));
  EXPECT_FALSE(coveredBy(centroidal, "joint_torque_weights.scaling"));
  EXPECT_TRUE(coveredBy(wholeBody, "model_settings.foot_constraint.linear_acceleration_error_gain_z"));
  EXPECT_FALSE(coveredBy(centroidal, "model_settings.foot_constraint.linear_acceleration_error_gain_z"));
  EXPECT_TRUE(coveredBy(centroidal, "state_weights.scaling") && coveredBy(wholeBody, "state_weights.scaling"));
}

}  // namespace
}  // namespace ocs2::humanoid
