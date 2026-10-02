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

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"

/*
 * CentroidalMpcInterface::CreateControllerModels(): what the robot process of a remote MPC builds, the models of the MRT
 * joint controller and nothing else. No reference manager and no optimal control problem (nothing taped or loaded with
 * CppAD); the effective model follows the contact input parameterization exactly as Create()'s does.
 */

namespace ocs2::humanoid {
namespace {

struct RobotFiles {
  std::string task;
  std::string urdf;
  std::string reference;
};

const RobotFiles kAtlas{"robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml",
                        "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf",
                        "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/reference.yaml"};

TEST(CentroidalMpcInterfaceControllerModels, BuildsTheControllersModelsAndNoProblem) {
  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> created =
      CentroidalMpcInterface::CreateControllerModels(kAtlas.task, kAtlas.urdf, kAtlas.reference);
  ASSERT_TRUE(created.ok()) << created.status();
  const CentroidalMpcInterface& interface = **created;
  EXPECT_FALSE(interface.hasOptimalControlProblem());
  EXPECT_EQ(interface.getReferenceManagerPtr(), nullptr) << "the reference manager is the MPC's";
  EXPECT_EQ(interface.getContactPlannerModulePtr(), nullptr);

  const CentroidalMpcRobotModel<scalar_t>& model = interface.getMpcRobotModel();
  const MpcRobotModelBase<scalar_t>& effective = interface.getEffectiveMpcRobotModel();
  EXPECT_EQ(interface.getInitialState().size(), static_cast<Eigen::Index>(model.getStateDim()));
  EXPECT_EQ(effective.getStateDim(), model.getStateDim());
  EXPECT_EQ(model.getJointDim(), interface.modelSettings().mpcModelJointNames.size());
  // The input layout of the problem the MPC node builds from the same file: [lambda, joint velocities] under basis
  // vectors, [six-dimensional wrenches, joint velocities] otherwise.
  if (interface.usesContactBasisVectorInputs()) {
    EXPECT_EQ(effective.getInputDim(), interface.getNumBasisInputs() + model.getJointDim());
    EXPECT_NE(interface.getBasisDecoratorPtr(), nullptr);
  } else {
    EXPECT_EQ(effective.getInputDim(), model.getInputDim());
  }
  EXPECT_GT(interface.mpcSettings().mrtDesiredFrequency_, 0.0) << "the solver settings are read: the control rate";
  EXPECT_GT(interface.getNominalComHeight(), 0.0);
}

TEST(CentroidalMpcInterfaceControllerModels, RefusesAMissingFileByName) {
  const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> created =
      CentroidalMpcInterface::CreateControllerModels(kAtlas.task, "no/such/robot.urdf", kAtlas.reference);
  EXPECT_EQ(created.status().code(), absl::StatusCode::kNotFound);
  EXPECT_NE(created.status().message().find("no/such/robot.urdf"), std::string::npos);
}

}  // namespace
}  // namespace ocs2::humanoid
