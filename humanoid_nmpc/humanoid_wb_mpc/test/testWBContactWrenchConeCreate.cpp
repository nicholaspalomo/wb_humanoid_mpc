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

#include <cmath>
#include <cstdlib>
#include <filesystem>
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

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"

/**
 * ContactWrenchConeConstraint::Create() on a real wrench-space model, the G1 whole-body one. A Config built in code
 * never passes through loadConfig()'s checks, and Create() used to hand it to the constructor, whose range checks were
 * assert()s that the optimized build compiles out: a cone with two friction facets, a negative friction coefficient or
 * a NaN offset was built and enforced. Create() now refuses it with the InvalidArgument of
 * ContactWrenchConeConstraint::validateConfig(), naming the task-file key to change (testContactWrenchConeBasisMatrix
 * pins validateConfig() itself).
 */
namespace ocs2::humanoid {
namespace {

constexpr absl::string_view kConfigBlock = "contacts.contactWrenchConeSoftConstraint.";

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

class WBContactWrenchConeCreateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    taskFile_ = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml");
    const std::string urdfFile = runfilePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf");
    const std::string referenceFile = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/command/reference.yaml");
    ASSERT_FALSE(taskFile_.empty() || urdfFile.empty() || referenceFile.empty()) << "the G1 whole-body files are not in the runfiles";
    modelSettings_ = std::make_unique<ModelSettings>(taskFile_, urdfFile, "wb_mpc_", /*verbose=*/false);
    absl::StatusOr<PinocchioInterface> pinocchioInterface = loadCustomPinocchioInterface(taskFile_, urdfFile, *modelSettings_);
    ASSERT_TRUE(pinocchioInterface.ok()) << pinocchioInterface.status();
    pinocchioInterface_ = std::make_unique<PinocchioInterface>(*std::move(pinocchioInterface));
    model_ = std::make_unique<WBAccelMpcRobotModel<scalar_t>>(*modelSettings_);
    referenceManager_ = std::make_unique<SwitchedModelReferenceManager>(
        GaitSchedule::loadGaitSchedule(referenceFile, *modelSettings_, /*verbose=*/false),
        std::make_unique<SwingTrajectoryPlanner>(loadSwingTrajectorySettings(taskFile_, "swing_trajectory_config", /*verbose=*/false),
                                                 N_CONTACTS),
        *pinocchioInterface_, *model_);
  }

  absl::StatusOr<std::unique_ptr<ContactWrenchConeConstraint>> create(const ContactWrenchConeConstraint::Config& config) const {
    return ContactWrenchConeConstraint::Create(
        *referenceManager_, ContactRectangle::loadContactRectangle(taskFile_, *modelSettings_, static_cast<int>(CONTACT_LEFT_INDEX)),
        CONTACT_LEFT_INDEX, *pinocchioInterface_, *model_, config);
  }

  std::string taskFile_;
  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  std::unique_ptr<WBAccelMpcRobotModel<scalar_t>> model_;
  std::unique_ptr<SwitchedModelReferenceManager> referenceManager_;
};

}  // namespace

TEST_F(WBContactWrenchConeCreateTest, aConfigOutOfRangeIsRefusedNamingItsKey) {
  // Positive control: a Config in range - the boundary values included - builds the cone on this model, so each
  // refusal below is the value's.
  const ContactWrenchConeConstraint::Config valid;
  const absl::StatusOr<std::unique_ptr<ContactWrenchConeConstraint>> built = create(valid);
  ASSERT_TRUE(built.ok()) << built.status();
  ASSERT_NE(*built, nullptr);
  ContactWrenchConeConstraint::Config boundary(/*numBasisVectorsParam=*/3);
  boundary.torsionalFrictionCoefficient = 0.0;
  boundary.minNormalForce = 0.0;
  boundary.gripperForce = 0.0;
  EXPECT_TRUE(create(boundary).ok()) << create(boundary).status();

  std::vector<std::pair<std::string, ContactWrenchConeConstraint::Config>> refused;
  refused.emplace_back("numBasisVectors", ContactWrenchConeConstraint::Config(/*numBasisVectorsParam=*/2));
  ContactWrenchConeConstraint::Config friction = valid;
  friction.frictionCoefficient = -0.7;
  refused.emplace_back("frictionCoefficient", friction);
  ContactWrenchConeConstraint::Config torsion = valid;
  torsion.torsionalFrictionCoefficient = std::numeric_limits<scalar_t>::quiet_NaN();
  refused.emplace_back("torsionalFrictionCoefficient", torsion);
  ContactWrenchConeConstraint::Config minForce = valid;
  minForce.minNormalForce = -5.0;
  refused.emplace_back("minNormalForce", minForce);
  ContactWrenchConeConstraint::Config gripper = valid;
  gripper.gripperForce = std::numeric_limits<scalar_t>::infinity();
  refused.emplace_back("gripperForce", gripper);
  for (const std::pair<std::string, ContactWrenchConeConstraint::Config>& entry : refused) {
    SCOPED_TRACE(entry.first);
    const absl::StatusOr<std::unique_ptr<ContactWrenchConeConstraint>> cone = create(entry.second);
    ASSERT_FALSE(cone.ok()) << "Create() built a cone from an out-of-range " << entry.first;
    EXPECT_EQ(cone.status().code(), absl::StatusCode::kInvalidArgument) << cone.status();
    EXPECT_TRUE(absl::StrContains(cone.status().message(), absl::StrCat(kConfigBlock, entry.first))) << cone.status();
  }

  ContactWrenchConeConstraint::Config offPatch = valid;
  offPatch.patchOffset = vector3_t(0.0, std::numeric_limits<scalar_t>::quiet_NaN(), 0.0);
  EXPECT_EQ(create(offPatch).status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace ocs2::humanoid
