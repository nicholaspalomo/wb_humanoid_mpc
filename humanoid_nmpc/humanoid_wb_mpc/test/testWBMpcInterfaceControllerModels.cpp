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
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_mpc_config/reference_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_mpc_config/xyz.nproto.h"
#include "humanoid_wb_mpc/WBMpcInterface.h"

/*
 * WBMpcInterface::CreateControllerModels(): what the robot process of a remote MPC builds, the models of the MRT joint
 * controller and nothing else. No optimal control problem: nothing taped or loaded with CppAD.
 */

namespace ocs2::humanoid {
namespace {

constexpr char kTask[] = "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto";
constexpr char kUrdf[] = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf";
constexpr char kReference[] = "robot_models/unitree_g1/g1_wb_mpc/config/command/reference.textproto";

TEST(WBMpcInterfaceControllerModels, BuildsTheControllersModelsAndNoProblem) {
  absl::StatusOr<std::unique_ptr<WBMpcInterface>> created = WBMpcInterface::CreateControllerModels(kTask, kUrdf, kReference);
  ASSERT_TRUE(created.ok()) << created.status();
  const WBMpcInterface& interface = **created;
  EXPECT_FALSE(interface.hasOptimalControlProblem());
  const WBAccelMpcRobotModel<scalar_t>& model = interface.getMpcRobotModel();
  EXPECT_EQ(interface.getInitialState().size(), static_cast<Eigen::Index>(model.getStateDim()));
  EXPECT_EQ(model.getJointDim(), interface.modelSettings().mpcModelJointNames.size());
  EXPECT_GT(interface.mpcSettings().mrtDesiredFrequency_, 0.0) << "the solver settings are read: the control rate";
}

TEST(WBMpcInterfaceControllerModels, TheTypedFilesBuildWhatTheirPathsBuild) {
  // The path form loads the files and builds the typed form, so the two agree on everything the controller reads.
  const absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(kTask);
  ASSERT_TRUE(task.ok()) << task.status();
  const absl::StatusOr<mpc_config::ReferenceFile> reference = loadReferenceFile(kReference);
  ASSERT_TRUE(reference.ok()) << reference.status();
  const absl::StatusOr<std::unique_ptr<WBMpcInterface>> typed = WBMpcInterface::CreateControllerModels(*task, kUrdf, *reference);
  ASSERT_TRUE(typed.ok()) << typed.status();
  const absl::StatusOr<std::unique_ptr<WBMpcInterface>> byPath = WBMpcInterface::CreateControllerModels(kTask, kUrdf, kReference);
  ASSERT_TRUE(byPath.ok()) << byPath.status();
  EXPECT_TRUE((*typed)->getInitialState() == (*byPath)->getInitialState());
  EXPECT_EQ((*typed)->modelSettings().mpcModelJointNames, (*byPath)->modelSettings().mpcModelJointNames);
  EXPECT_EQ((*typed)->mpcSettings().mrtDesiredFrequency_, (*byPath)->mpcSettings().mrtDesiredFrequency_);
  EXPECT_EQ((*typed)->sqpSettings().dt, (*byPath)->sqpSettings().dt);
  EXPECT_EQ((*typed)->rolloutSettings().timeStep, (*byPath)->rolloutSettings().timeStep);
}

TEST(WBMpcInterfaceControllerModels, TheInitialStateIsTheFilesByName) {
  // The whole-body state is base pose, joint positions, base velocities, joint velocities: the initial state's base
  // height and a joint position land where the robot model reads them.
  const absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(kTask);
  ASSERT_TRUE(task.ok()) << task.status();
  mpc_config::TaskFile raised = *task;
  mpc_config::Xyz basePosition = raised.initial_state.base_position.value_or(mpc_config::Xyz{});
  basePosition.z = 1.25;
  raised.initial_state.base_position = basePosition;
  ASSERT_FALSE(raised.initial_state.joint_positions.empty());
  raised.initial_state.joint_positions.front().value = 0.375;
  const std::string joint = raised.initial_state.joint_positions.front().joint;
  const absl::StatusOr<mpc_config::ReferenceFile> reference = loadReferenceFile(kReference);
  ASSERT_TRUE(reference.ok()) << reference.status();
  const absl::StatusOr<std::unique_ptr<WBMpcInterface>> created = WBMpcInterface::CreateControllerModels(raised, kUrdf, *reference);
  ASSERT_TRUE(created.ok()) << created.status();
  const WBAccelMpcRobotModel<scalar_t>& model = (*created)->getMpcRobotModel();
  const vector_t& state = (*created)->getInitialState();
  EXPECT_EQ(model.getBasePose(state)(2), 1.25);
  const std::vector<std::string>& joints = (*created)->modelSettings().mpcModelJointNames;
  const std::vector<std::string>::const_iterator found = std::find(joints.begin(), joints.end(), joint);
  ASSERT_NE(found, joints.end()) << joint;
  EXPECT_EQ(model.getJointAngles(state)(found - joints.begin()), 0.375) << joint;
}

TEST(WBMpcInterfaceControllerModels, RefusesAMissingFileByName) {
  const absl::StatusOr<std::unique_ptr<WBMpcInterface>> created =
      WBMpcInterface::CreateControllerModels("no/such/task.textproto", kUrdf, kReference);
  EXPECT_EQ(created.status().code(), absl::StatusCode::kNotFound);
  EXPECT_NE(created.status().message().find("no/such/task.textproto"), std::string::npos);
}

}  // namespace
}  // namespace ocs2::humanoid
