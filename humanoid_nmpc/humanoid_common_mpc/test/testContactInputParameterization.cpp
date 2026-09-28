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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/ContactInputParameterization.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"

/**
 * The task-file side of the contact input parameterization: the registry that resolves `contactInputParameterization`
 * (findings A79/A90), the refusal of the retired `useContactBasisVectorInputs` boolean, the one loader of the
 * contacts.contactWrenchConeSoftConstraint block every consumer shares (AC5, A86), and the bases a task file configures
 * through contacts.basisGeneratorSet. Nothing here builds a CppAD model.
 */
namespace ocs2::humanoid {
namespace {

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

std::string readFile(const std::string& path) {
  std::ifstream in(path);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string writeTempFile(absl::string_view name, const std::string& content) {
  const std::string path = (std::filesystem::path(testing::TempDir()) / absl::StrCat("testContactInputParameterization_", name)).string();
  std::ofstream out(path);
  out << content;
  return path;
}

/** `content` with `from` replaced by `to`; fails the test unless `from` occurs exactly once. */
std::string replacedOnce(const std::string& content, const std::string& from, const std::string& to) {
  const std::string::size_type position = content.find(from);
  EXPECT_NE(position, std::string::npos) << "'" << from << "' not found";
  EXPECT_EQ(content.find(from, position + 1), std::string::npos) << "'" << from << "' occurs more than once";
  if (position == std::string::npos) return content;
  std::string result = content;
  result.replace(position, from.size(), to);
  return result;
}

void expectInvalidArgumentNaming(const absl::Status& status, const std::vector<std::string>& phrases) {
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
  for (const std::string& phrase : phrases) {
    EXPECT_TRUE(absl::StrContains(status.message(), phrase)) << status << "\n does not name '" << phrase << "'";
  }
}

constexpr absl::string_view kAtlasTaskFile = "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml";
constexpr absl::string_view kAtlasUrdfFile = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";

/** The five geometry keys of the wrench-cone block, every one of which is required. */
const std::vector<std::string>& coneGeometryKeys() {
  static const std::vector<std::string> keys = {"frictionCoefficient", "torsionalFrictionCoefficient", "minNormalForce", "gripperForce",
                                                "numBasisVectors"};
  return keys;
}

const std::string kCompleteConeBlock =
    "contacts:\n"
    "  contactWrenchConeSoftConstraint:\n"
    "    frictionCoefficient: 0.45\n"
    "    torsionalFrictionCoefficient: 0.03\n"
    "    minNormalForce: 7\n"
    "    gripperForce: 1.5\n"
    "    numBasisVectors: 6\n"
    "    mu: 0.2\n"
    "    delta: 5\n";

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
  }
}

// ==================== The loader ====================

TEST(ContactInputParameterizationTest, AFileWithoutTheKeyRunsTheWrenchParameterization) {
  const absl::StatusOr<ContactInputParameterization> absent = loadContactInputParameterization(writeTempFile("absent.yaml", "other: 1\n"));
  ASSERT_TRUE(absent.ok()) << absent.status();
  EXPECT_EQ(*absent, ContactInputParameterization::kWrench);
  // Positive control: the key, when present, is what decides.
  const absl::StatusOr<ContactInputParameterization> named =
      loadContactInputParameterization(writeTempFile("named.yaml", "other: 1\ncontactInputParameterization: basis_vectors\n"));
  ASSERT_TRUE(named.ok()) << named.status();
  EXPECT_EQ(*named, ContactInputParameterization::kBasisVectors);
}

TEST(ContactInputParameterizationTest, TheRetiredBooleanIsRefusedWhateverItsValueNamingItsReplacement) {
  for (const std::string& value : {std::string("true"), std::string("false")}) {
    SCOPED_TRACE(value);
    // Refused even beside a valid new key: the file was written against the old loader and has to be looked at.
    const std::string file =
        writeTempFile(absl::StrCat("retired_", value, ".yaml"),
                      absl::StrCat("useContactBasisVectorInputs: ", value, "\ncontactInputParameterization: wrench\n"));
    expectInvalidArgumentNaming(loadContactInputParameterization(file).status(),
                                {std::string(kRetiredContactBasisVectorInputsKey),
                                 absl::StrCat(kContactInputParameterizationKey, ": ", kBasisVectorsContactInputParameterization),
                                 absl::StrCat(kContactInputParameterizationKey, ": ", kWrenchContactInputParameterization)});
  }
}

TEST(ContactInputParameterizationTest, AValueThatIsNotOneValidNameIsRefused) {
  expectInvalidArgumentNaming(
      loadContactInputParameterization(writeTempFile("list.yaml", "contactInputParameterization: [wrench]\n")).status(),
      {std::string(kContactInputParameterizationKey), std::string(kBasisVectorsContactInputParameterization)});
  // A boolean written where the name belongs, the likeliest slip after the migration.
  expectInvalidArgumentNaming(loadContactInputParameterization(writeTempFile("bool.yaml", "contactInputParameterization: true\n")).status(),
                              {std::string(kContactInputParameterizationKey), std::string(kWrenchContactInputParameterization)});
  EXPECT_EQ(loadContactInputParameterization(absl::StrCat(testing::TempDir(), "/no_such_task_file.yaml")).status().code(),
            absl::StatusCode::kNotFound);
  // A file that is not a map at all: a key lookup on it throws in yaml-cpp, which must not escape the StatusOr.
  expectInvalidArgumentNaming(loadContactInputParameterization(writeTempFile("scalar.yaml", "just_a_scalar\n")).status(),
                              {std::string(kContactInputParameterizationKey)});
  expectInvalidArgumentNaming(loadContactInputParameterization(writeTempFile("sequence.yaml", "- wrench\n")).status(),
                              {std::string(kContactInputParameterizationKey)});
  // Positive control: an empty file names nothing, so it runs the default.
  const absl::StatusOr<ContactInputParameterization> empty = loadContactInputParameterization(writeTempFile("empty.yaml", ""));
  ASSERT_TRUE(empty.ok()) << empty.status();
  EXPECT_EQ(*empty, kDefaultContactInputParameterization);
}

TEST(ContactInputParameterizationTest, TheShippedRobotsSelectTheParameterizationTheyRanBefore) {
  // Atlas and SA01 shipped the boolean set true; G1 and R1 never set it. The rename must not move any of them, and no
  // shipped file may still carry the retired key (the loader would refuse it at start-up).
  const std::vector<std::pair<std::string, ContactInputParameterization>> shipped = {
      {std::string(kAtlasTaskFile), ContactInputParameterization::kBasisVectors},
      {"robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml", ContactInputParameterization::kBasisVectors},
      {"robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.yaml", ContactInputParameterization::kWrench},
      {"robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.yaml", ContactInputParameterization::kWrench},
      {"robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml", ContactInputParameterization::kWrench},
  };
  for (const std::pair<std::string, ContactInputParameterization>& robot : shipped) {
    SCOPED_TRACE(robot.first);
    const std::string taskFile = runfilePath(robot.first);
    ASSERT_FALSE(taskFile.empty()) << "not in the runfiles";
    const absl::StatusOr<ContactInputParameterization> parameterization = loadContactInputParameterization(taskFile);
    ASSERT_TRUE(parameterization.ok()) << parameterization.status();
    EXPECT_EQ(*parameterization, robot.second);
    EXPECT_FALSE(absl::StrContains(readFile(taskFile), absl::StrCat("\n", kRetiredContactBasisVectorInputsKey, ":")));
  }
}

// ==================== The wrench-cone block: one loader, every key required ====================

TEST(ContactWrenchConeConfigTest, ACompleteBlockLoadsEveryValue) {
  const absl::StatusOr<ContactWrenchConeConstraint::Config> config =
      ContactWrenchConeConstraint::loadConfig(writeTempFile("cone_complete.yaml", kCompleteConeBlock));
  ASSERT_TRUE(config.ok()) << config.status();
  // Values that are none of Config's library defaults (0.7, 0.05, 5, 0, 4), so a key that was silently defaulted shows.
  EXPECT_DOUBLE_EQ(config->frictionCoefficient, 0.45);
  EXPECT_DOUBLE_EQ(config->torsionalFrictionCoefficient, 0.03);
  EXPECT_DOUBLE_EQ(config->minNormalForce, 7.0);
  EXPECT_DOUBLE_EQ(config->gripperForce, 1.5);
  EXPECT_EQ(config->numBasisVectors, 6u);
}

TEST(ContactWrenchConeConfigTest, EveryMissingKeyIsRefusedByName) {
  for (const std::string& key : coneGeometryKeys()) {
    SCOPED_TRACE(key);
    // Drop exactly the line of this key.
    const std::string::size_type line = kCompleteConeBlock.find(absl::StrCat("    ", key, ":"));
    ASSERT_NE(line, std::string::npos);
    std::string content = kCompleteConeBlock;
    content.erase(line, content.find('\n', line) + 1 - line);
    const absl::Status status =
        ContactWrenchConeConstraint::loadConfig(writeTempFile(absl::StrCat("cone_without_", key, ".yaml"), content)).status();
    expectInvalidArgumentNaming(status, {absl::StrCat("contacts.contactWrenchConeSoftConstraint.", key), "missing"});
  }
  // The whole block absent, as in the G1 and R1 task files: refused, not defaulted to mu 0.7.
  expectInvalidArgumentNaming(
      ContactWrenchConeConstraint::loadConfig(
          writeTempFile("cone_absent.yaml", "contacts:\n  frictionForceConeSoftConstraint:\n    frictionCoefficient: 0.4\n"))
          .status(),
      {"contacts.contactWrenchConeSoftConstraint.frictionCoefficient"});
}

TEST(ContactWrenchConeConfigTest, AValueOfTheWrongTypeOrOutOfRangeIsRefusedByName) {
  const std::vector<std::pair<std::string, std::string>> bad = {
      {"    frictionCoefficient: 0.45", "    frictionCoefficient: slippery"},
      {"    frictionCoefficient: 0.45", "    frictionCoefficient: 0"},
      {"    torsionalFrictionCoefficient: 0.03", "    torsionalFrictionCoefficient: -0.03"},
      {"    minNormalForce: 7", "    minNormalForce: -1"},
      {"    gripperForce: 1.5", "    gripperForce: -1.5"},
      {"    numBasisVectors: 6", "    numBasisVectors: 2"},
      {"    numBasisVectors: 6", "    numBasisVectors: 4.5"},
  };
  for (const std::pair<std::string, std::string>& edit : bad) {
    SCOPED_TRACE(edit.second);
    const std::string key(absl::StripAsciiWhitespace(edit.first.substr(0, edit.first.find(':'))));
    const absl::Status status =
        ContactWrenchConeConstraint::loadConfig(writeTempFile("cone_bad.yaml", replacedOnce(kCompleteConeBlock, edit.first, edit.second)))
            .status();
    expectInvalidArgumentNaming(status, {absl::StrCat("contacts.contactWrenchConeSoftConstraint.", key)});
  }
}

// ==================== The bases a task file configures ====================

class ContactWrenchConeBasesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    taskFile_ = runfilePath(kAtlasTaskFile);
    urdfFile_ = runfilePath(kAtlasUrdfFile);
    ASSERT_FALSE(taskFile_.empty() || urdfFile_.empty()) << "the DRC Atlas files are not in the runfiles";
    shipped_ = readFile(taskFile_);
    modelSettings_ = std::make_unique<ModelSettings>(taskFile_, urdfFile_, "centroidal_mpc_", /*verbose=*/false);
  }

  std::string taskFile_;
  std::string urdfFile_;
  std::string shipped_;
  std::unique_ptr<ModelSettings> modelSettings_;
};

TEST_F(ContactWrenchConeBasesTest, TheShippedFileBuildsTheConservativeSetFromItsConeBlock) {
  const absl::StatusOr<feet_array_t<ContactWrenchConeBasisMatrix>> bases = loadContactWrenchConeBases(taskFile_, *modelSettings_);
  ASSERT_TRUE(bases.ok()) << bases.status();
  const absl::StatusOr<ContactWrenchConeConstraint::Config> cone = ContactWrenchConeConstraint::loadConfig(taskFile_);
  ASSERT_TRUE(cone.ok()) << cone.status();
  for (const ContactWrenchConeBasisMatrix& basis : *bases) {
    EXPECT_EQ(basis.generatorSet(), kConservativeInnerApproximationGeneratorSet) << "the shipped generator set moved";
    EXPECT_EQ(basis.numBasis(), cone->numBasisVectors + 7);
  }
}

TEST_F(ContactWrenchConeBasesTest, TheGeneratorSetIsTheOneTheTaskFileNames) {
  const std::string exact = writeTempFile(
      "exact.yaml", replacedOnce(shipped_, "basisGeneratorSet: conservative_inner_approximation", "basisGeneratorSet: exact_wrench_cone"));
  const absl::StatusOr<feet_array_t<ContactWrenchConeBasisMatrix>> bases = loadContactWrenchConeBases(exact, *modelSettings_);
  ASSERT_TRUE(bases.ok()) << bases.status();
  const absl::StatusOr<ContactWrenchConeConstraint::Config> cone = ContactWrenchConeConstraint::loadConfig(exact);
  ASSERT_TRUE(cone.ok());
  for (const ContactWrenchConeBasisMatrix& basis : *bases) {
    EXPECT_EQ(basis.generatorSet(), kExactWrenchConeGeneratorSet);
    EXPECT_EQ(basis.numBasis(), 8 * cone->numBasisVectors);
  }

  // Without the key the default is used, which is the shipped set.
  const std::string absent =
      writeTempFile("absent_set.yaml", replacedOnce(shipped_, "basisGeneratorSet: conservative_inner_approximation", "otherKey: 0"));
  const absl::StatusOr<feet_array_t<ContactWrenchConeBasisMatrix>> byDefault = loadContactWrenchConeBases(absent, *modelSettings_);
  ASSERT_TRUE(byDefault.ok()) << byDefault.status();
  EXPECT_EQ((*byDefault)[0].generatorSet(), kDefaultBasisGeneratorSet);
}

TEST_F(ContactWrenchConeBasesTest, AnUnknownSetOrAMissingConeKeyIsRefusedByName) {
  const std::string unknown = writeTempFile(
      "unknown_set.yaml", replacedOnce(shipped_, "basisGeneratorSet: conservative_inner_approximation", "basisGeneratorSet: exact"));
  std::vector<std::string> phrases = {std::string(kBasisGeneratorSetKey), "'exact'"};
  for (const std::string& name : basisGeneratorSetNames()) phrases.push_back(name);
  expectInvalidArgumentNaming(loadContactWrenchConeBases(unknown, *modelSettings_).status(), phrases);

  const std::string noFriction =
      writeTempFile("no_friction.yaml", replacedOnce(shipped_, "    frictionCoefficient: 0.5\n    torsional", "    torsional"));
  expectInvalidArgumentNaming(loadContactWrenchConeBases(noFriction, *modelSettings_).status(),
                              {"contacts.contactWrenchConeSoftConstraint.frictionCoefficient"});
}

TEST_F(ContactWrenchConeBasesTest, TheCppAdFolderIsDerivedAndADeadFolderKeyIsReported) {
  // Finding A81: model_settings.modelFolderCppAd was carried by every task file and read by nothing. The shipped files
  // no longer carry it, the folder is derived from the MPC and robot names, and a file that still carries it is told so.
  EXPECT_FALSE(absl::StrContains(shipped_, "modelFolderCppAd:")) << "the shipped Atlas file still carries the dead key";
  EXPECT_EQ(modelSettings_->modelFolderCppAd, "cppad_code_gen/cppad_centroidal_mpc_atlas");

  const std::string withKey =
      writeTempFile("dead_folder_key.yaml", replacedOnce(shipped_, "  recompileLibrariesCppAd: false\n",
                                                         "  recompileLibrariesCppAd: false\n  modelFolderCppAd: build/somewhere_else\n"));
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_, testing::HasSubstr("model_settings.modelFolderCppAd is not read")))
      .Times(1);
  log.StartCapturingLogs();
  const ModelSettings settings(withKey, urdfFile_, "centroidal_mpc_", /*verbose=*/false);
  log.StopCapturingLogs();
  EXPECT_EQ(settings.modelFolderCppAd, modelSettings_->modelFolderCppAd) << "the dead key moved the folder";
}

}  // namespace ocs2::humanoid
