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

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_common_mpc/common/BasisInputsCostTransform.h"
#include "humanoid_common_mpc/common/ContactInputParameterization.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"

/**
 * CentroidalMpcInterface::Create() refuses every contact-input configuration error with an InvalidArgument that names
 * the task-file key to change (findings A79/A90, AC5, and the basis wiring of the generator set and the regularization),
 * and does so before any CppAD model is built. Each case edits one line of the shipped DRC Atlas task file, so every
 * other key is the configuration the robot runs; a refusal reached here is that line's doing.
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

class ContactInputParameterizationWiringTest : public ::testing::Test {
 protected:
  void SetUp() override {
    taskFile_ = runfilePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml");
    contactPlanningFile_ = runfilePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/contact_planning.yaml");
    referenceFile_ = runfilePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/reference.yaml");
    urdfFile_ = runfilePath("robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf");
    ASSERT_FALSE(taskFile_.empty() || contactPlanningFile_.empty() || referenceFile_.empty() || urdfFile_.empty())
        << "the DRC Atlas files are not in the runfiles";
    shipped_ = readFile(taskFile_);
    // The positive control of every case below: the shipped file selects basis vectors, with the default generator set
    // and regularization written out, so that each edit is the only difference.
    ASSERT_TRUE(absl::StrContains(shipped_, "\ncontactInputParameterization: basis_vectors\n"));
  }

  /** Writes `content` as the task file of a directory of its own, beside a copy of the planner's configuration. */
  std::string writeVariant(absl::string_view name, const std::string& content) const {
    const std::filesystem::path directory = std::filesystem::path(testing::TempDir()) / absl::StrCat("wiring_", name);
    std::filesystem::create_directories(directory);
    std::filesystem::copy_file(contactPlanningFile_, directory / "contact_planning.yaml",
                               std::filesystem::copy_options::overwrite_existing);
    const std::filesystem::path taskFile = directory / "task.yaml";
    std::ofstream out(taskFile);
    out << content;
    return taskFile.string();
  }

  /** Create()'s status for a variant, which must be an InvalidArgument naming every phrase. */
  void expectRefusedNaming(absl::string_view name, const std::string& content, const std::vector<std::string>& phrases) const {
    SCOPED_TRACE(name);
    const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> created =
        CentroidalMpcInterface::Create(writeVariant(name, content), urdfFile_, referenceFile_);
    ASSERT_FALSE(created.ok()) << "Create() accepted the variant";
    EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument) << created.status();
    for (const std::string& phrase : phrases) {
      EXPECT_TRUE(absl::StrContains(created.status().message(), phrase)) << created.status() << "\n does not name '" << phrase << "'";
    }
  }

  std::string taskFile_;
  std::string contactPlanningFile_;
  std::string referenceFile_;
  std::string urdfFile_;
  std::string shipped_;
};

}  // namespace

TEST_F(ContactInputParameterizationWiringTest, TheRetiredBooleanIsRefusedAtStartUpNamingItsReplacement) {
  for (const std::string& value : {std::string("true"), std::string("false")}) {
    expectRefusedNaming(absl::StrCat("retired_", value),
                        replacedOnce(shipped_, "\ncontactInputParameterization: basis_vectors\n",
                                     absl::StrCat("\nuseContactBasisVectorInputs: ", value, "\n")),
                        {std::string(kRetiredContactBasisVectorInputsKey), std::string(kContactInputParameterizationKey)});
  }
}

TEST_F(ContactInputParameterizationWiringTest, AnUnknownParameterizationIsRefusedListingTheValidNames) {
  expectRefusedNaming("unknown_parameterization",
                      replacedOnce(shipped_, "\ncontactInputParameterization: basis_vectors\n", "\ncontactInputParameterization: basis\n"),
                      {std::string(kContactInputParameterizationKey), std::string(kWrenchContactInputParameterization),
                       std::string(kBasisVectorsContactInputParameterization)});
}

TEST_F(ContactInputParameterizationWiringTest, TheGeneratorSetIsReadFromTheTaskFile) {
  // The key reaches the builder: an unknown name is refused listing the registered sets.
  std::vector<std::string> phrases = {std::string(kBasisGeneratorSetKey), "'exact'"};
  for (const std::string& set : basisGeneratorSetNames()) phrases.push_back(set);
  expectRefusedNaming("unknown_generator_set",
                      replacedOnce(shipped_, "basisGeneratorSet: conservative_inner_approximation", "basisGeneratorSet: exact"), phrases);
}

TEST_F(ContactInputParameterizationWiringTest, TheRegularizationIsReadFromTheTaskFileAndValidated) {
  std::vector<std::string> phrases = {std::string(kBasisRegularizationKey), "'diagonal'"};
  for (const std::string& name : basisRegularizationNames()) phrases.push_back(name);
  expectRefusedNaming("unknown_regularization",
                      replacedOnce(shipped_, "basisRegularization: full_diagonal", "basisRegularization: diagonal"), phrases);
  expectRefusedNaming("negative_regularization",
                      replacedOnce(shipped_, "basisScalingRegularization: 0.0001", "basisScalingRegularization: -1.0"),
                      {std::string(kBasisScalingRegularizationKey), "non-negative"});
  // Zero passes the range check, but leaves the lambda block of M^T R M singular: the QP would have no unique input.
  expectRefusedNaming("zero_regularization",
                      replacedOnce(shipped_, "basisScalingRegularization: 0.0001", "basisScalingRegularization: 0.0"),
                      {std::string(kBasisScalingRegularizationKey), "not positive definite"});
  expectRefusedNaming("unparsable_regularization",
                      replacedOnce(shipped_, "basisScalingRegularization: 0.0001", "basisScalingRegularization: small"),
                      {std::string(kBasisScalingRegularizationKey), "'small'"});
}

TEST_F(ContactInputParameterizationWiringTest, TheNonNegativityBarrierIsReadByNameBeforeTheProblemIsBuilt) {
  // The lambda >= 0 barrier's parameters used to be read with loadPtreeValue inside the loop that builds the terms, after
  // the dynamics had been compiled: a value that is not a number threw out of Create() without naming its key.
  expectRefusedNaming(
      "unparsable_barrier_mu",
      replacedOnce(shipped_, "  basisNonNegativityBarrier:\n    mu: 0.01\n", "  basisNonNegativityBarrier:\n    mu: stiff\n"),
      {"contacts.basisNonNegativityBarrier.mu", "'stiff'"});
  expectRefusedNaming("unparsable_barrier_delta",
                      replacedOnce(shipped_, "    mu: 0.01\n    delta: 0.001\n", "    mu: 0.01\n    delta: narrow\n"),
                      {"contacts.basisNonNegativityBarrier.delta", "'narrow'"});
}

TEST_F(ContactInputParameterizationWiringTest, TheBasisIsBuiltFromAConeBlockThatNamesEveryKey) {
  // AC5: the basis used to be built from the library defaults (mu 0.7) when the block or a key was missing.
  expectRefusedNaming("no_friction_coefficient", replacedOnce(shipped_, "    frictionCoefficient: 0.5\n    torsional", "    torsional"),
                      {"contacts.contactWrenchConeSoftConstraint.frictionCoefficient", "missing"});
  expectRefusedNaming("two_facets", replacedOnce(shipped_, "    numBasisVectors: 4", "    numBasisVectors: 2"),
                      {"contacts.contactWrenchConeSoftConstraint.numBasisVectors"});
}

TEST_F(ContactInputParameterizationWiringTest, TheContactPlannerDerivesItsGroundFromAConeBlockThatNamesEveryKey) {
  // AC5, the planner's half: its friction and torsion bounds came from the same block, with the same silent defaults.
  // The planner has to be the only reader of the block for this to test it: the wrench parameterization keeps the basis
  // builder out, and friction_force_cone in place of contact_wrench_cone keeps the wrench cone term out - both read the
  // block through the same loader and would refuse the file on the planner's behalf, later.
  std::string content =
      replacedOnce(shipped_, "\ncontactInputParameterization: basis_vectors\n", "\ncontactInputParameterization: wrench\n");
  content = replacedOnce(content, "\n  - contact_wrench_cone ", "\n  - friction_force_cone ");
  content = replacedOnce(content, "\ncontactScheduleSource: gait_schedule\n", "\ncontactScheduleSource: contact_planner\n");
  content = replacedOnce(content, "    torsionalFrictionCoefficient: 0.05\n", "");
  expectRefusedNaming("planner_without_torsion", content, {"contacts.contactWrenchConeSoftConstraint.torsionalFrictionCoefficient"});
}

}  // namespace ocs2::humanoid
