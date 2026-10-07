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

#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/ContactInputParameterization.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/costs/ContactsFromConfig.h"
#include "humanoid_common_mpc/config/model/MpcFormulationFromConfig.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_mpc_config/task_file.pb.h"
#include "nproto/Textproto.h"

/**
 * The task-file side of the contact input parameterization: the registry that resolves `contact_input_parameterization`
 * (findings A79/A90), the refusal of the retired `useContactBasisVectorInputs` boolean, the one conversion of the
 * contacts.contact_wrench_cone_soft_constraint block every consumer shares (AC5, A86), and the bases a task file
 * configures through contacts.basis_generator_set. Nothing here builds a CppAD model.
 */
namespace ocs2::humanoid {
namespace {

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

void expectInvalidArgumentNaming(const absl::Status& status, const std::vector<std::string>& phrases) {
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
  for (const std::string& phrase : phrases) {
    EXPECT_TRUE(absl::StrContains(status.message(), phrase)) << status << "\n does not name '" << phrase << "'";
  }
}

constexpr absl::string_view kAtlasTaskFile = "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto";
constexpr absl::string_view kAtlasUrdfFile = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";
constexpr absl::string_view kConeBlock = "contacts.contact_wrench_cone_soft_constraint";

/** A complete wrench-cone block, with values that are none of Config's library defaults (0.7, 0.05, 5, 0, 4). */
mpc_config::ContactsConfig completeCone() {
  mpc_config::ContactsConfig contacts;
  mpc_config::ContactsConfig::WrenchCone& cone = contacts.contact_wrench_cone_soft_constraint;
  cone.friction_coefficient = 0.45;
  cone.torsional_friction_coefficient = 0.03;
  cone.min_normal_force = 7.0;
  cone.gripper_force = 1.5;
  cone.num_basis_vectors = 6;
  cone.mu = 0.2;
  cone.delta = 5.0;
  return contacts;
}

}  // namespace

// ==================== The registry ====================

TEST(ContactInputParameterizationTest, TheRegistryListsEveryNameAndEachRoundTrips) {
  const std::vector<std::string> names = contactInputParameterizationNames();
  ASSERT_EQ(names.size(), 2u);
  EXPECT_EQ(names[0], kWrenchContactInputParameterization);
  EXPECT_EQ(names[1], kBasisVectorsContactInputParameterization);
  for (const std::string& name : names) {
    const absl::StatusOr<ContactInputParameterization> parameterization = contactInputParameterizationFromName(name);
    ASSERT_TRUE(parameterization.ok()) << parameterization.status();
    EXPECT_EQ(contactInputParameterizationName(*parameterization), name);
  }
  // Every enumerator is registered under a name of its own.
  EXPECT_NE(contactInputParameterizationName(ContactInputParameterization::kWrench),
            contactInputParameterizationName(ContactInputParameterization::kBasisVectors));
  EXPECT_EQ(kDefaultContactInputParameterization, ContactInputParameterization::kWrench) << "the default is what every robot used to run";
}

TEST(ContactInputParameterizationTest, AnUnknownNameIsRefusedListingTheValidOnes) {
  for (const std::string& unknown : {std::string("basis"), std::string("true"), std::string("Wrench"), std::string()}) {
    SCOPED_TRACE(unknown);
    expectInvalidArgumentNaming(contactInputParameterizationFromName(unknown).status(),
                                {std::string(kContactInputParameterizationKey), std::string(kWrenchContactInputParameterization),
                                 std::string(kBasisVectorsContactInputParameterization)});
    mpc_config::TaskFile task;
    task.contact_input_parameterization = unknown;
    expectInvalidArgumentNaming(contactInputParameterizationFromConfig(task).status(),
                                {"contact_input_parameterization", std::string(kBasisVectorsContactInputParameterization)});
  }
}

// ==================== The task file ====================

TEST(ContactInputParameterizationTest, AFileWithoutTheFieldRunsTheWrenchParameterization) {
  const absl::StatusOr<ContactInputParameterization> absent = contactInputParameterizationFromConfig(mpc_config::TaskFile{});
  ASSERT_TRUE(absent.ok()) << absent.status();
  EXPECT_EQ(*absent, ContactInputParameterization::kWrench);
  // Positive control: the field, when set, is what decides.
  mpc_config::TaskFile named;
  named.contact_input_parameterization = kBasisVectorsContactInputParameterization;
  const absl::StatusOr<ContactInputParameterization> basis = contactInputParameterizationFromConfig(named);
  ASSERT_TRUE(basis.ok()) << basis.status();
  EXPECT_EQ(*basis, ContactInputParameterization::kBasisVectors);
}

TEST(ContactInputParameterizationTest, TheRetiredBooleanIsRefusedWhateverItsValueNamingItsReplacement) {
  for (const std::string& value : {std::string("true"), std::string("false")}) {
    SCOPED_TRACE(value);
    // Refused even beside a valid new field: the file was written against the old loader and has to be looked at.
    const absl::StatusOr<humanoid_mpc_config::TaskFile> parsed = nproto::ParseTextproto<humanoid_mpc_config::TaskFile>(
        absl::StrCat(kRetiredContactBasisVectorInputsKey, ": ", value, "\ncontact_input_parameterization: \"wrench\"\n"), "task.textproto");
    expectInvalidArgumentNaming(parsed.status(),
                                {std::string(kRetiredContactBasisVectorInputsKey), "is retired",
                                 absl::StrCat("contact_input_parameterization: \"", kBasisVectorsContactInputParameterization, "\""),
                                 absl::StrCat("\"", kWrenchContactInputParameterization, "\"")});
  }
}

TEST(ContactInputParameterizationTest, TheShippedRobotsSelectTheParameterizationTheyRanBefore) {
  // Atlas and SA01 shipped the boolean set true; G1 and R1 never set it. The rename must not move any of them, and no
  // shipped file may still carry the retired key (the parser would refuse it at start-up).
  const std::vector<std::pair<std::string, ContactInputParameterization>> shipped = {
      {std::string(kAtlasTaskFile), ContactInputParameterization::kBasisVectors},
      {"robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.textproto", ContactInputParameterization::kBasisVectors},
      {"robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto", ContactInputParameterization::kWrench},
      {"robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.textproto", ContactInputParameterization::kWrench},
      {"robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto", ContactInputParameterization::kWrench},
  };
  for (const std::pair<std::string, ContactInputParameterization>& robot : shipped) {
    SCOPED_TRACE(robot.first);
    const std::string taskFile = runfilePath(robot.first);
    ASSERT_FALSE(taskFile.empty()) << "not in the runfiles";
    const absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(taskFile);
    ASSERT_TRUE(task.ok()) << task.status();
    const absl::StatusOr<ContactInputParameterization> parameterization = contactInputParameterizationFromConfig(*task);
    ASSERT_TRUE(parameterization.ok()) << parameterization.status();
    EXPECT_EQ(*parameterization, robot.second);
  }
}

// ==================== The wrench-cone block: one conversion, every geometry field required ====================

TEST(ContactWrenchConeConfigTest, ACompleteBlockConvertsEveryValue) {
  const absl::StatusOr<ContactWrenchConeConstraint::Config> config = contactWrenchConeConfigFromConfig(completeCone());
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_DOUBLE_EQ(config->frictionCoefficient, 0.45);
  EXPECT_DOUBLE_EQ(config->torsionalFrictionCoefficient, 0.03);
  EXPECT_DOUBLE_EQ(config->minNormalForce, 7.0);
  EXPECT_DOUBLE_EQ(config->gripperForce, 1.5);
  EXPECT_EQ(config->numBasisVectors, 6u);
}

TEST(ContactWrenchConeConfigTest, EveryMissingFieldIsRefusedByName) {
  using WrenchCone = mpc_config::ContactsConfig::WrenchCone;
  const std::vector<std::pair<std::string, void (*absl_nonnull)(WrenchCone&)>> fields = {
      {"friction_coefficient", [](WrenchCone& cone) { cone.friction_coefficient.reset(); }},
      {"torsional_friction_coefficient", [](WrenchCone& cone) { cone.torsional_friction_coefficient.reset(); }},
      {"min_normal_force", [](WrenchCone& cone) { cone.min_normal_force.reset(); }},
      {"gripper_force", [](WrenchCone& cone) { cone.gripper_force.reset(); }},
      {"num_basis_vectors", [](WrenchCone& cone) { cone.num_basis_vectors.reset(); }},
  };
  for (const std::pair<std::string, void (*absl_nonnull)(WrenchCone&)>& field : fields) {
    SCOPED_TRACE(field.first);
    mpc_config::ContactsConfig contacts = completeCone();
    field.second(contacts.contact_wrench_cone_soft_constraint);
    expectInvalidArgumentNaming(contactWrenchConeConfigFromConfig(contacts).status(),
                                {absl::StrCat(kConeBlock, ".", field.first), "missing"});
  }
  // The whole block absent, as in the G1 and R1 task files: refused, not defaulted to mu 0.7.
  expectInvalidArgumentNaming(contactWrenchConeConfigFromConfig(mpc_config::ContactsConfig{}).status(),
                              {absl::StrCat(kConeBlock, ".friction_coefficient")});
}

TEST(ContactWrenchConeConfigTest, AValueOutOfRangeIsRefusedByName) {
  using WrenchCone = mpc_config::ContactsConfig::WrenchCone;
  const std::vector<std::pair<std::string, void (*absl_nonnull)(WrenchCone&)>> bad = {
      {"friction_coefficient", [](WrenchCone& cone) { cone.friction_coefficient = 0.0; }},
      {"friction_coefficient", [](WrenchCone& cone) { cone.friction_coefficient = std::numeric_limits<double>::quiet_NaN(); }},
      {"torsional_friction_coefficient", [](WrenchCone& cone) { cone.torsional_friction_coefficient = -0.03; }},
      {"min_normal_force", [](WrenchCone& cone) { cone.min_normal_force = -1.0; }},
      {"gripper_force", [](WrenchCone& cone) { cone.gripper_force = -1.5; }},
      {"num_basis_vectors", [](WrenchCone& cone) { cone.num_basis_vectors = 2; }},
  };
  for (const std::pair<std::string, void (*absl_nonnull)(WrenchCone&)>& edit : bad) {
    SCOPED_TRACE(edit.first);
    mpc_config::ContactsConfig contacts = completeCone();
    edit.second(contacts.contact_wrench_cone_soft_constraint);
    expectInvalidArgumentNaming(contactWrenchConeConfigFromConfig(contacts).status(), {absl::StrCat(kConeBlock, ".", edit.first)});
  }
  // A count that is not an integer is the parser's to refuse, at its line.
  const absl::StatusOr<humanoid_mpc_config::TaskFile> parsed = nproto::ParseTextproto<humanoid_mpc_config::TaskFile>(
      "contacts {\n  contact_wrench_cone_soft_constraint {\n    num_basis_vectors: 4.5\n  }\n}\n", "task.textproto");
  expectInvalidArgumentNaming(parsed.status(), {"task.textproto:3:"});
}

// ==================== The bases a task file configures ====================

class ContactWrenchConeBasesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string taskFile = runfilePath(kAtlasTaskFile);
    urdfFile_ = runfilePath(kAtlasUrdfFile);
    ASSERT_FALSE(taskFile.empty() || urdfFile_.empty()) << "the DRC Atlas files are not in the runfiles";
    absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(taskFile);
    ASSERT_TRUE(task.ok()) << task.status();
    task_ = *std::move(task);
    modelSettings_ = std::make_unique<ModelSettings>(ModelSettings::Create(task_, urdfFile_, "centroidal_mpc_", /*verbose=*/false).value());
  }

  std::string urdfFile_;
  mpc_config::TaskFile task_;
  std::unique_ptr<ModelSettings> modelSettings_;
};

TEST_F(ContactWrenchConeBasesTest, TheShippedFileBuildsTheConservativeSetFromItsConeBlock) {
  const absl::StatusOr<feet_array_t<ContactWrenchConeBasisMatrix>> bases =
      contactWrenchConeBasesFromConfig(task_.contacts, *modelSettings_);
  ASSERT_TRUE(bases.ok()) << bases.status();
  const absl::StatusOr<ContactWrenchConeConstraint::Config> cone = contactWrenchConeConfigFromConfig(task_.contacts);
  ASSERT_TRUE(cone.ok()) << cone.status();
  for (const ContactWrenchConeBasisMatrix& basis : *bases) {
    EXPECT_EQ(basis.generatorSet(), kConservativeInnerApproximationGeneratorSet) << "the shipped generator set moved";
    EXPECT_EQ(basis.numBasis(), cone->numBasisVectors + 7);
  }
}

TEST_F(ContactWrenchConeBasesTest, TheGeneratorSetIsTheOneTheTaskFileNames) {
  mpc_config::TaskFile exact = task_;
  exact.contacts.basis_generator_set = kExactWrenchConeGeneratorSet;
  const absl::StatusOr<feet_array_t<ContactWrenchConeBasisMatrix>> bases =
      contactWrenchConeBasesFromConfig(exact.contacts, *modelSettings_);
  ASSERT_TRUE(bases.ok()) << bases.status();
  const absl::StatusOr<ContactWrenchConeConstraint::Config> cone = contactWrenchConeConfigFromConfig(exact.contacts);
  ASSERT_TRUE(cone.ok());
  for (const ContactWrenchConeBasisMatrix& basis : *bases) {
    EXPECT_EQ(basis.generatorSet(), kExactWrenchConeGeneratorSet);
    EXPECT_EQ(basis.numBasis(), 8 * cone->numBasisVectors);
  }

  // Without the field the default is used, which is the shipped set.
  const absl::StatusOr<humanoid_mpc_config::TaskFile> unset =
      nproto::ParseTextproto<humanoid_mpc_config::TaskFile>(/*text=*/"", "task.textproto");
  ASSERT_TRUE(unset.ok()) << unset.status();
  EXPECT_EQ(unset->contacts().basis_generator_set(), kDefaultBasisGeneratorSet) << "the schema's default is the library's";
}

TEST_F(ContactWrenchConeBasesTest, AnUnknownSetOrAMissingConeFieldIsRefusedByName) {
  mpc_config::TaskFile unknown = task_;
  unknown.contacts.basis_generator_set = "exact";
  std::vector<std::string> phrases = {std::string(kBasisGeneratorSetKey), "'exact'"};
  for (const std::string& name : basisGeneratorSetNames()) phrases.push_back(name);
  expectInvalidArgumentNaming(contactWrenchConeBasesFromConfig(unknown.contacts, *modelSettings_).status(), phrases);

  mpc_config::TaskFile noFriction = task_;
  noFriction.contacts.contact_wrench_cone_soft_constraint.friction_coefficient.reset();
  expectInvalidArgumentNaming(contactWrenchConeBasesFromConfig(noFriction.contacts, *modelSettings_).status(),
                              {absl::StrCat(kConeBlock, ".friction_coefficient")});
}

TEST_F(ContactWrenchConeBasesTest, TheCppAdFolderIsDerivedAndTheDeadFolderKeyIsRefused) {
  // Finding A81: model_settings.modelFolderCppAd was carried by every task file and read by nothing. The folder is
  // derived from the MPC and robot names, and a file that still carries the key is refused, saying so.
  EXPECT_EQ(modelSettings_->modelFolderCppAd, "cppad_code_gen/cppad_centroidal_mpc_atlas");
  const absl::StatusOr<humanoid_mpc_config::TaskFile> withKey = nproto::ParseTextproto<humanoid_mpc_config::TaskFile>(
      "model_settings {\n  model_folder_cpp_ad: \"build/somewhere_else\"\n}\n", "task.textproto");
  expectInvalidArgumentNaming(withKey.status(), {"task.textproto:2:", "'model_folder_cpp_ad' is retired", "derived from the robot"});
}

}  // namespace ocs2::humanoid
