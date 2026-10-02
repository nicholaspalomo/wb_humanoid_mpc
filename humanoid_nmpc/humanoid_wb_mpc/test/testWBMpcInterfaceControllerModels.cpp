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

#include <memory>
#include <string>

#include "absl/status/statusor.h"

#include "humanoid_wb_mpc/WBMpcInterface.h"

/*
 * WBMpcInterface::CreateControllerModels(): what the robot process of a remote MPC builds, the models of the MRT joint
 * controller and nothing else. No optimal control problem: nothing taped or loaded with CppAD.
 */

namespace ocs2::humanoid {
namespace {

constexpr const char* kTask = "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml";
constexpr const char* kUrdf = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf";
constexpr const char* kReference = "robot_models/unitree_g1/g1_wb_mpc/config/command/reference.yaml";

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

TEST(WBMpcInterfaceControllerModels, RefusesAMissingFileByName) {
  const absl::StatusOr<std::unique_ptr<WBMpcInterface>> created =
      WBMpcInterface::CreateControllerModels("no/such/task.yaml", kUrdf, kReference);
  EXPECT_EQ(created.status().code(), absl::StatusCode::kNotFound);
  EXPECT_NE(created.status().message().find("no/such/task.yaml"), std::string::npos);
}

}  // namespace
}  // namespace ocs2::humanoid
