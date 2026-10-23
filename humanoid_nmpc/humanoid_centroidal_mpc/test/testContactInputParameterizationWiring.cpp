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

#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_common_mpc/common/BasisInputsCostTransform.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "support/TypedConfigFiles.h"

/**
 * CentroidalMpcInterface::Create() refuses every contact-input configuration error with an InvalidArgument that names
 * the field to change (findings A79/A90, AC5, and the basis wiring of the generator set and the regularization), and
 * does so before any CppAD model is built. Each case edits one field of the shipped DRC Atlas configuration, so every
 * other value is the configuration the robot runs; a refusal reached here is that field's doing.
 */
namespace ocs2::humanoid {
namespace {

class ContactInputParameterizationWiringTest : public ::testing::Test {
 protected:
  void SetUp() override {
    files_ = atlasFiles();
    absl::StatusOr<CentroidalMpcConfig> config = loadConfigOf(files_);
    ASSERT_TRUE(config.ok()) << config.status();
    shipped_ = *std::move(config);
    // The positive control of every case below: the shipped file selects basis vectors, with the default generator set
    // and regularization written out, so that each edit is the only difference.
    ASSERT_EQ(shipped_.task.contact_input_parameterization, kBasisVectorsContactInputParameterization);
  }

  /** Create()'s status for the shipped configuration with `edit` applied, which must be an InvalidArgument naming every phrase. */
  void expectRefusedNaming(absl::string_view name,
                           const std::function<void(mpc_config::TaskFile&)>& edit,
                           const std::vector<std::string>& phrases) const {
    SCOPED_TRACE(name);
    CentroidalMpcConfig config = shipped_;
    edit(config.task);
    const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> created = CentroidalMpcInterface::Create(config, files_.urdfFile);
    expectRefusal(created.status(), phrases);
  }

  static void expectRefusal(const absl::Status& status, const std::vector<std::string>& phrases) {
    ASSERT_FALSE(status.ok()) << "Create() accepted the variant";
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
    for (const std::string& phrase : phrases) {
      EXPECT_TRUE(absl::StrContains(status.message(), phrase)) << status << "\n does not name '" << phrase << "'";
    }
  }

  CentroidalRobotFiles files_;
  CentroidalMpcConfig shipped_;
};

}  // namespace

TEST_F(ContactInputParameterizationWiringTest, TheRetiredBooleanIsRefusedByTheParserNamingItsReplacement) {
  // A task file still carrying the retired boolean does not parse: the parser names what replaced it, with the position.
  const std::string directory = absl::StrCat(testing::TempDir(), "/wiring_retired");
  absl::StatusOr<CentroidalRobotFiles> files = writeConfig(directory, shipped_, files_.urdfFile);
  ASSERT_TRUE(files.ok()) << files.status();
  ASSERT_TRUE(writeTextFile(files->taskFile, absl::StrCat(taskFileText(shipped_.task), "useContactBasisVectorInputs: true\n")).ok());
  const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> created =
      CentroidalMpcInterface::Create(files->taskFile, files->urdfFile, files->referenceFile);
  expectRefusal(created.status(), {"task.textproto:", "is retired", "contact_input_parameterization"});
}

TEST_F(ContactInputParameterizationWiringTest, AnUnknownParameterizationIsRefusedListingTheValidNames) {
  expectRefusedNaming("unknown_parameterization", [](mpc_config::TaskFile& task) { task.contact_input_parameterization = "basis"; },
                      {"contact_input_parameterization", std::string(kWrenchContactInputParameterization),
                       std::string(kBasisVectorsContactInputParameterization)});
}

TEST_F(ContactInputParameterizationWiringTest, TheGeneratorSetIsReadFromTheTaskFile) {
  // The field reaches the builder: an unknown name is refused listing the registered sets.
  std::vector<std::string> phrases = {"'exact'"};
  for (const std::string& set : basisGeneratorSetNames()) phrases.push_back(set);
  expectRefusedNaming("unknown_generator_set", [](mpc_config::TaskFile& task) { task.contacts.basis_generator_set = "exact"; }, phrases);
}

TEST_F(ContactInputParameterizationWiringTest, TheRegularizationIsReadFromTheTaskFileAndValidated) {
  std::vector<std::string> phrases = {"'diagonal'"};
  for (const std::string& name : basisRegularizationNames()) phrases.push_back(name);
  expectRefusedNaming(
      "unknown_regularization", [](mpc_config::TaskFile& task) { task.contacts.basis_regularization = "diagonal"; }, phrases);
  expectRefusedNaming("negative_regularization", [](mpc_config::TaskFile& task) { task.contacts.basis_scaling_regularization = -1.0; },
                      {"non-negative"});
  // Zero passes the range check, but leaves the lambda block of M^T R M singular: the QP would have no unique input.
  expectRefusedNaming("zero_regularization", [](mpc_config::TaskFile& task) { task.contacts.basis_scaling_regularization = 0.0; },
                      {"not positive definite"});
  expectRefusedNaming(
      "infinite_regularization",
      [](mpc_config::TaskFile& task) { task.contacts.basis_scaling_regularization = std::numeric_limits<double>::infinity(); },
      {"contacts.basis_scaling_regularization"});
}

TEST_F(ContactInputParameterizationWiringTest, TheNonNegativityBarrierIsConvertedByNameBeforeTheProblemIsBuilt) {
  expectRefusedNaming(
      "infinite_barrier_mu",
      [](mpc_config::TaskFile& task) { task.contacts.basis_non_negativity_barrier.mu = std::numeric_limits<double>::infinity(); },
      {"contacts.basis_non_negativity_barrier.mu"});
  expectRefusedNaming(
      "nan_barrier_delta",
      [](mpc_config::TaskFile& task) { task.contacts.basis_non_negativity_barrier.delta = std::numeric_limits<double>::quiet_NaN(); },
      {"contacts.basis_non_negativity_barrier.delta"});
}

TEST_F(ContactInputParameterizationWiringTest, TheBasisIsBuiltFromAConeBlockThatNamesEveryGroundField) {
  // AC5: the basis used to be built from the library defaults (mu 0.7) when the block or a key was missing.
  expectRefusedNaming("no_friction_coefficient",
                      [](mpc_config::TaskFile& task) { task.contacts.contact_wrench_cone_soft_constraint.friction_coefficient.reset(); },
                      {"contacts.contact_wrench_cone_soft_constraint.friction_coefficient"});
  expectRefusedNaming("two_facets",
                      [](mpc_config::TaskFile& task) { task.contacts.contact_wrench_cone_soft_constraint.num_basis_vectors = 2; },
                      {"num_basis_vectors"});
}

TEST_F(ContactInputParameterizationWiringTest, TheContactPlannerDerivesItsGroundFromAConeBlockThatNamesEveryGroundField) {
  // AC5, the planner's half: its friction and torsion bounds come from the same block, with the same refusal. The planner
  // has to be the only reader of the block for this to test it: the wrench parameterization keeps the basis builder out,
  // and friction_force_cone in place of contact_wrench_cone keeps the wrench cone term out - both read the block through
  // the same conversion and would refuse the file on the planner's behalf, later.
  expectRefusedNaming("planner_without_torsion",
                      [](mpc_config::TaskFile& task) {
                        task.contact_input_parameterization = kWrenchContactInputParameterization;
                        for (std::string& constraint : task.soft_constraints) {
                          if (constraint == "contact_wrench_cone") constraint = "friction_force_cone";
                        }
                        task.contact_schedule_source = "contact_planner";
                        task.contacts.contact_wrench_cone_soft_constraint.torsional_friction_coefficient.reset();
                      },
                      {"contacts.contact_wrench_cone_soft_constraint.torsional_friction_coefficient"});
}

}  // namespace ocs2::humanoid
