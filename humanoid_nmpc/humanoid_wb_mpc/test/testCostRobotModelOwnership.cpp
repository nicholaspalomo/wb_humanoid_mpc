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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_nmpc/humanoid_wb_mpc/test/AffineEndEffectorDynamics.h"
#include "humanoid_wb_mpc/WBMpcInterface.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "humanoid_wb_mpc/cost/EndEffectorDynamicsCostHelpers.h"
#include "humanoid_wb_mpc/cost/EndEffectorDynamicsFootCost.h"
#include "humanoid_wb_mpc/cost/JointTorqueCostCppAd.h"

/*
 * The CppAD costs of the whole-body MPC that clone the robot model - JointTorqueCostCppAd and
 * EndEffectorDynamicsFootCost - own their copy and release it: the cost and every clone of it, of which the solver
 * makes one per worker thread. Both used to keep the clone in a raw pointer that nothing deleted, so every cost and
 * every copy leaked a model. A robot model that counts its live instances stands in for the G1's. Each test tapes and
 * compiles the CppAD library of one cost on the G1 whole-body model, in a working directory made for the run.
 */

namespace ocs2::humanoid {
namespace {

using test_support::AffineEndEffectorDynamics;

/** The whole-body robot model of the G1, counting its live instances, clones included. */
class CountingRobotModel final : public WBAccelMpcRobotModel<ad_scalar_t> {
 public:
  explicit CountingRobotModel(const ModelSettings& modelSettings) : WBAccelMpcRobotModel<ad_scalar_t>(modelSettings) { ++liveInstances; }
  ~CountingRobotModel() override { --liveInstances; }
  CountingRobotModel& operator=(const CountingRobotModel&) = delete;
  CountingRobotModel(CountingRobotModel&&) = delete;
  CountingRobotModel& operator=(CountingRobotModel&&) = delete;
  CountingRobotModel* absl_nonnull clone() const override { return new CountingRobotModel(*this); }

  /** The instances alive in the test program. */
  inline static int liveInstances = 0;

 private:
  CountingRobotModel(const CountingRobotModel& other) : WBAccelMpcRobotModel<ad_scalar_t>(other) { ++liveInstances; }
};

std::string runfilePath(absl::string_view relativePath) {
  std::vector<std::filesystem::path> roots;
  if (const char* absl_nullable srcDir = std::getenv("TEST_SRCDIR")) {
    roots.emplace_back(std::filesystem::path(srcDir) / "_main");
  }
  roots.emplace_back(std::filesystem::current_path());
  for (const std::filesystem::path& root : roots) {
    const std::filesystem::path candidate = root / std::string(relativePath);
    if (std::filesystem::exists(candidate)) return std::filesystem::absolute(candidate).string();
  }
  return std::string();
}

/**
 * The G1 controller models (no optimal control problem), in a working directory made for this run: the CppAD libraries
 * go to a folder relative to it (ModelSettings::modelFolderCppAd), and the shipped file does not force a recompile, so a
 * library left by an earlier unsandboxed run would be loaded instead of taped.
 */
std::unique_ptr<WBMpcInterface> g1ControllerModelsInAFreshDirectory(absl::string_view testName) {
  const std::string taskFile = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto");
  const std::string referenceFile = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/command/reference.textproto");
  const std::string urdfFile = runfilePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf");
  EXPECT_FALSE(taskFile.empty() || referenceFile.empty() || urdfFile.empty()) << "the G1 whole-body files are not in the runfiles";
  if (taskFile.empty() || referenceFile.empty() || urdfFile.empty()) return nullptr;
  std::string freshDirectory = (std::filesystem::path(testing::TempDir()) / absl::StrCat(testName, "_XXXXXX")).string();
  if (mkdtemp(freshDirectory.data()) == nullptr) {
    ADD_FAILURE() << "cannot make a fresh working directory from " << freshDirectory;
    return nullptr;
  }
  std::filesystem::current_path(freshDirectory);
  absl::StatusOr<std::unique_ptr<WBMpcInterface>> created = WBMpcInterface::CreateControllerModels(taskFile, urdfFile, referenceFile);
  EXPECT_TRUE(created.ok()) << created.status();
  return created.ok() ? *std::move(created) : nullptr;
}

TEST(CostRobotModelOwnership, TheJointTorqueCostReleasesTheRobotModelsItClones) {
  const std::unique_ptr<WBMpcInterface> interface = g1ControllerModelsInAFreshDirectory("jointTorqueCostOwnership");
  ASSERT_NE(interface, nullptr);
  {
    const CountingRobotModel model(interface->modelSettings());
    ASSERT_EQ(CountingRobotModel::liveInstances, 1);
    {
      const JointTorqueCostCppAd cost(vector_t::Ones(static_cast<Eigen::Index>(model.getJointDim())), interface->getPinocchioInterface(),
                                      model, "jointTorqueCostOwnership", interface->modelSettings());
      EXPECT_EQ(CountingRobotModel::liveInstances, 2) << "the cost holds a copy of the model of its own";
      {
        const std::unique_ptr<JointTorqueCostCppAd> costCopy(cost.clone());
        EXPECT_EQ(CountingRobotModel::liveInstances, 3) << "every copy of the cost holds a copy of its own";
      }
      EXPECT_EQ(CountingRobotModel::liveInstances, 2) << "a copy of the cost released its model";
    }
    EXPECT_EQ(CountingRobotModel::liveInstances, 1) << "the cost released its model";
  }
  EXPECT_EQ(CountingRobotModel::liveInstances, 0);
}

TEST(CostRobotModelOwnership, TheSwingFootCostReleasesTheRobotModelsItClones) {
  const std::unique_ptr<WBMpcInterface> interface = g1ControllerModelsInAFreshDirectory("footCostOwnership");
  ASSERT_NE(interface, nullptr);
  // The cost keeps a clone of the end-effector dynamics and never evaluates it while taping: a stand-in does.
  const AffineEndEffectorDynamics footDynamics;
  {
    const CountingRobotModel model(interface->modelSettings());
    ASSERT_EQ(CountingRobotModel::liveInstances, 1);
    {
      const EndEffectorDynamicsFootCost cost(*interface->getSwitchedModelReferenceManagerPtr(), EndEffectorDynamicsWeights(),
                                             interface->getPinocchioInterface(), footDynamics, model, /*contactIndex=*/0,
                                             "footCostOwnership", interface->modelSettings());
      EXPECT_EQ(CountingRobotModel::liveInstances, 2) << "the cost holds a copy of the model of its own";
      {
        const std::unique_ptr<EndEffectorDynamicsFootCost> costCopy(cost.clone());
        EXPECT_EQ(CountingRobotModel::liveInstances, 3) << "every copy of the cost holds a copy of its own";
      }
      EXPECT_EQ(CountingRobotModel::liveInstances, 2) << "a copy of the cost released its model";
    }
    EXPECT_EQ(CountingRobotModel::liveInstances, 1) << "the cost released its model";
  }
  EXPECT_EQ(CountingRobotModel::liveInstances, 0);
}

}  // namespace
}  // namespace ocs2::humanoid
