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
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_core/PreComputation.h"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_centroidal_mpc/constraint/JointMimicKinematicConstraint.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "support/TypedConfigFiles.h"

/**
 * JointMimicKinematicConstraint::Create(): the two joints are resolved by name among the MPC joints, and a name that is
 * not one of them, or a position gain that is not positive, is refused with a Status. The joint indices used to be
 * looked up by a Config constructor that threw. Built on the DRC Atlas centroidal robot model, without CppAD.
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

class JointMimicKinematicConstraintTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // LINT.IfChange(robot_files)
    const std::string taskFile = runfilePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto");
    const std::string referenceFile = runfilePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/reference.textproto");
    const std::string urdfFile = runfilePath("robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf");
    // LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc/BUILD.bazel:joint_mimic_test_data)
    ASSERT_FALSE(taskFile.empty() || referenceFile.empty() || urdfFile.empty()) << "the DRC Atlas files are not in the runfiles";
    const absl::StatusOr<CentroidalMpcConfig> config = loadCentroidalMpcConfig(taskFile, referenceFile);
    ASSERT_TRUE(config.ok()) << config.status();
    absl::StatusOr<ModelSettings> settings =
        ModelSettings::Create(config->task, urdfFile, "testJointMimicKinematicConstraint_", /*verbose=*/false);
    ASSERT_TRUE(settings.ok()) << settings.status();
    modelSettings_ = std::make_unique<ModelSettings>(*std::move(settings));
    ASSERT_GE(modelSettings_->mpcModelJointNames.size(), 2U);
    absl::StatusOr<PinocchioInterface> pinocchioInterface =
        loadCustomPinocchioInterface(config->task, urdfFile, *modelSettings_, /*scaleTotalMass=*/false);
    ASSERT_TRUE(pinocchioInterface.ok()) << pinocchioInterface.status();
    pinocchioInterface_ = std::make_unique<PinocchioInterface>(*std::move(pinocchioInterface));
    absl::StatusOr<CentroidalModelInfo> info = centroidalModelInfoOf(*config, *pinocchioInterface_, *modelSettings_);
    ASSERT_TRUE(info.ok()) << info.status();
    info_ = *std::move(info);
    model_ = std::make_unique<CentroidalMpcRobotModel<scalar_t>>(*modelSettings_, *pinocchioInterface_, info_);
  }

  const std::string& jointName(size_t index) const { return modelSettings_->mpcModelJointNames[index]; }

  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  CentroidalModelInfo info_;
  std::unique_ptr<CentroidalMpcRobotModel<scalar_t>> model_;
};

TEST_F(JointMimicKinematicConstraintTest, ResolvesTheJointsByName) {
  const absl::StatusOr<std::unique_ptr<JointMimicKinematicConstraint>> constraint =
      JointMimicKinematicConstraint::Create(*model_, jointName(0), jointName(1), /*multiplier=*/0.5, /*positionGain=*/20.0);
  ASSERT_TRUE(constraint.ok()) << constraint.status();
  const JointMimicKinematicConstraint::Config& config = (*constraint)->getConfig();
  EXPECT_EQ(config.parentJointIndex, model_->findJointIndex(jointName(0)).value());
  EXPECT_EQ(config.childJointIndex, model_->findJointIndex(jointName(1)).value());

  // The child joint at half the parent's angle and velocity: the constraint is met.
  vector_t state = vector_t::Zero(model_->getStateDim());
  const vector_t input = vector_t::Zero(model_->getInputDim());
  vector_t jointAngles = vector_t::Zero(model_->getJointDim());
  jointAngles[config.parentJointIndex] = 0.4;
  jointAngles[config.childJointIndex] = 0.2;
  model_->setJointAngles(state, jointAngles);
  const PreComputation preComputation;
  EXPECT_NEAR((*constraint)->getValue(/*time=*/0.0, state, input, preComputation)[0], 0.0, 1.0e-12);
  // And an error of the child's angle is weighted by the position gain.
  jointAngles[config.childJointIndex] = 0.1;
  model_->setJointAngles(state, jointAngles);
  EXPECT_NEAR((*constraint)->getValue(/*time=*/0.0, state, input, preComputation)[0], 20.0 * 0.1, 1.0e-12);
}

TEST_F(JointMimicKinematicConstraintTest, AJointThatIsNotAnMpcJointIsNotFoundNamingIt) {
  const absl::StatusOr<std::unique_ptr<JointMimicKinematicConstraint>> parent =
      JointMimicKinematicConstraint::Create(*model_, "no_such_knee", jointName(1), /*multiplier=*/0.5, /*positionGain=*/20.0);
  EXPECT_EQ(parent.status().code(), absl::StatusCode::kNotFound);
  EXPECT_TRUE(absl::StrContains(parent.status().message(), "no_such_knee")) << parent.status();
  const absl::StatusOr<std::unique_ptr<JointMimicKinematicConstraint>> child =
      JointMimicKinematicConstraint::Create(*model_, jointName(0), "no_such_knee", /*multiplier=*/0.5, /*positionGain=*/20.0);
  EXPECT_EQ(child.status().code(), absl::StatusCode::kNotFound);
}

TEST_F(JointMimicKinematicConstraintTest, APositionGainThatIsNotPositiveIsRefused) {
  for (const scalar_t positionGain : {0.0, -1.0, std::numeric_limits<scalar_t>::quiet_NaN()}) {
    const absl::StatusOr<std::unique_ptr<JointMimicKinematicConstraint>> constraint =
        JointMimicKinematicConstraint::Create(*model_, jointName(0), jointName(1), /*multiplier=*/0.5, positionGain);
    EXPECT_EQ(constraint.status().code(), absl::StatusCode::kInvalidArgument) << positionGain;
  }
}

}  // namespace
}  // namespace ocs2::humanoid
