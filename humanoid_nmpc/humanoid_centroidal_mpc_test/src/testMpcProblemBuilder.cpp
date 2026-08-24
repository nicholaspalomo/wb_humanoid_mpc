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

#include <gtest/gtest.h>
#include <string>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include <ocs2_core/constraint/LinearStateInputConstraint.h>
#include <ocs2_core/cost/QuadraticStateCost.h>
#include <ocs2_core/cost/QuadraticStateInputCost.h>
#include <ocs2_oc/oc_problem/OptimalControlProblem.h>

#include <humanoid_centroidal_mpc_test/CentroidalTestingModelInterface.h>
#include <humanoid_common_mpc/HumanoidCostConstraintFactory.h>
#include <humanoid_common_mpc/problem/MpcProblemBuilder.h>
#include <humanoid_common_mpc/problem/MpcProblemDefinition.h>
#include <humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h>

namespace ocs2::humanoid {

namespace {

std::string getG1TaskFile() {
  const std::string packagePath = ament_index_cpp::get_package_share_directory("g1_centroidal_mpc");
  return absl::StrCat(packagePath, "/config/mpc/task.yaml");
}

std::string getAtlasTaskFile() {
  const std::string packagePath = ament_index_cpp::get_package_share_directory("drc_atlas_centroidal_mpc");
  return absl::StrCat(packagePath, "/config/mpc/task.yaml");
}

std::string getR1TaskFile() {
  const std::string packagePath = ament_index_cpp::get_package_share_directory("unitree_r1_centroidal_mpc");
  return absl::StrCat(packagePath, "/config/mpc/task.yaml");
}

}  // namespace

TEST(TestMpcProblemBuilder, LoadG1ProblemDefinition) {
  const std::string taskFile = getG1TaskFile();
  const absl::StatusOr<MpcProblemDefinition> problemDefStatus = loadMpcProblemDefinition(taskFile, "problem_definition", false);

  ASSERT_TRUE(problemDefStatus.ok()) << problemDefStatus.status().ToString();
  const MpcProblemDefinition& problemDef = problemDefStatus.value();

  EXPECT_FALSE(problemDef.costs.empty());
  EXPECT_FALSE(problemDef.terminalCosts.empty());
  EXPECT_FALSE(problemDef.stateSoftConstraints.empty());
  EXPECT_FALSE(problemDef.softConstraints.empty());
  EXPECT_FALSE(problemDef.equalityConstraints.empty());

  bool hasStateInputCost = false;
  for (const ProblemTermConfig& term : problemDef.costs) {
    if (term.type == "StateInputQuadraticCost" && term.enabled) {
      hasStateInputCost = true;
    }
  }
  EXPECT_TRUE(hasStateInputCost);

  bool hasJointMimic = false;
  for (const ProblemTermConfig& term : problemDef.equalityConstraints) {
    if (term.type == "JointMimicConstraint" && term.enabled) {
      hasJointMimic = true;
    }
  }
  EXPECT_TRUE(hasJointMimic);
}

TEST(TestMpcProblemBuilder, LoadAtlasProblemDefinition) {
  const std::string taskFile = getAtlasTaskFile();
  const absl::StatusOr<MpcProblemDefinition> problemDefStatus = loadMpcProblemDefinition(taskFile, "problem_definition", false);

  ASSERT_TRUE(problemDefStatus.ok()) << problemDefStatus.status().ToString();
  const MpcProblemDefinition& problemDef = problemDefStatus.value();

  bool hasContactWrenchCone = false;
  for (const ProblemTermConfig& term : problemDef.softConstraints) {
    if (term.type == "ContactWrenchConeConstraint" && term.enabled) {
      hasContactWrenchCone = true;
    }
  }
  EXPECT_TRUE(hasContactWrenchCone);
}

TEST(TestMpcProblemBuilder, LoadR1ProblemDefinition) {
  const std::string taskFile = getR1TaskFile();
  const absl::StatusOr<MpcProblemDefinition> problemDefStatus = loadMpcProblemDefinition(taskFile, "problem_definition", false);

  ASSERT_TRUE(problemDefStatus.ok()) << problemDefStatus.status().ToString();
  const MpcProblemDefinition& problemDef = problemDefStatus.value();

  EXPECT_FALSE(problemDef.costs.empty());
  EXPECT_FALSE(problemDef.equalityConstraints.empty());

  bool hasFootCollisionCbf = false;
  for (const ProblemTermConfig& term : problemDef.softConstraints) {
    if (term.type == "FootCollisionCbfConstraint" && term.enabled) {
      hasFootCollisionCbf = true;
    }
  }
  EXPECT_TRUE(hasFootCollisionCbf);
}

TEST(TestMpcProblemBuilder, NonExistentFileReturnsError) {
  const absl::StatusOr<MpcProblemDefinition> problemDefStatus =
      loadMpcProblemDefinition("/non/existent/path/task.yaml", "problem_definition", false);

  EXPECT_FALSE(problemDefStatus.ok());
  EXPECT_EQ(problemDefStatus.status().code(), absl::StatusCode::kNotFound);
}

TEST(TestMpcProblemBuilder, MissingBlockReturnsError) {
  const std::string taskFile = getG1TaskFile();
  const absl::StatusOr<MpcProblemDefinition> problemDefStatus = loadMpcProblemDefinition(taskFile, "non_existent_problem_block", false);

  EXPECT_FALSE(problemDefStatus.ok());
  EXPECT_EQ(problemDefStatus.status().code(), absl::StatusCode::kNotFound);
}

TEST(TestMpcProblemBuilder, BuildProblemWithCustomBuilders) {
  CentroidalTestingModelInterface testingModel;
  ModeSchedule initModeSchedule({0.0, 1.0}, {3});
  ModeSequenceTemplate initModeSequenceTemplate({0.5, 0.5}, {3, 3});
  const std::shared_ptr<GaitSchedule> gaitSchedule =
      std::make_shared<GaitSchedule>(initModeSchedule, initModeSequenceTemplate, testingModel.getModelSettings().phaseTransitionStanceTime);
  const std::shared_ptr<SwingTrajectoryPlanner> swingPlanner =
      std::make_shared<SwingTrajectoryPlanner>(SwingTrajectoryPlanner::Config{}, testingModel.getModelSettings().contactNames.size());
  SwitchedModelReferenceManager referenceManager(gaitSchedule, swingPlanner, testingModel.getPinocchioInterface(),
                                                 testingModel.getMpcRobotModel());

  const HumanoidCostConstraintFactory factory(testingModel.taskFile, testingModel.referenceFile, referenceManager,
                                              testingModel.getPinocchioInterface(), testingModel.getMpcRobotModel(),
                                              testingModel.getMpcRobotModelAD(), testingModel.getModelSettings(), false);

  MpcProblemDefinition problemDef;
  // Add cost terms
  ProblemTermConfig stateCost;
  stateCost.name = "state_input_cost";
  stateCost.type = "StateInputQuadraticCost";
  stateCost.configPath = "stateInputCost";
  stateCost.enabled = true;
  problemDef.costs.push_back(stateCost);

  ProblemTermConfig customFootCost;
  customFootCost.name = "task_space_foot_cost";
  customFootCost.type = "FootTrackingCost";
  customFootCost.perContact = true;
  customFootCost.enabled = true;
  problemDef.costs.push_back(customFootCost);

  // Add terminal cost
  ProblemTermConfig termCost;
  termCost.name = "terminal_cost";
  termCost.type = "TerminalCost";
  termCost.configPath = "terminalCost";
  termCost.enabled = true;
  problemDef.terminalCosts.push_back(termCost);

  // Add soft constraints
  ProblemTermConfig frictionCone;
  frictionCone.name = "friction_force_cone";
  frictionCone.type = "FrictionForceConeConstraint";
  frictionCone.configPath = "contacts.frictionForceConeSoftConstraint";
  frictionCone.perContact = true;
  frictionCone.enabled = true;
  problemDef.softConstraints.push_back(frictionCone);

  // Add equality constraints
  ProblemTermConfig zeroWrench;
  zeroWrench.name = "zero_wrench";
  zeroWrench.type = "ZeroWrenchConstraint";
  zeroWrench.perContact = true;
  zeroWrench.enabled = true;
  problemDef.equalityConstraints.push_back(zeroWrench);

  ProblemTermConfig customStance;
  customStance.name = "zero_velocity";
  customStance.type = "ZeroVelocityConstraint";
  customStance.perContact = true;
  customStance.enabled = true;
  problemDef.equalityConstraints.push_back(customStance);

  // Custom builders
  MpcProblemBuilder::CustomBuilders customBuilders;
  const size_t stateDim = testingModel.getMpcRobotModel().getStateDim();
  const size_t inputDim = testingModel.getMpcRobotModel().getInputDim();

  customBuilders.footTrackingCostBuilder =
      [stateDim, inputDim](size_t /*contactIndex*/, absl::string_view /*name*/) -> absl::StatusOr<std::unique_ptr<StateInputCost>> {
    return std::unique_ptr<StateInputCost>(
        new QuadraticStateInputCost(matrix_t::Identity(stateDim, stateDim), matrix_t::Identity(inputDim, inputDim)));
  };

  customBuilders.stanceFootConstraintBuilder =
      [stateDim, inputDim](size_t /*contactIndex*/, absl::string_view /*name*/) -> absl::StatusOr<std::unique_ptr<StateInputConstraint>> {
    return std::unique_ptr<StateInputConstraint>(
        new LinearStateInputConstraint(vector_t::Zero(3), matrix_t::Zero(3, stateDim), matrix_t::Identity(3, inputDim)));
  };

  OptimalControlProblem problem;
  const MpcProblemBuilder problemBuilder(problemDef, factory, testingModel.getModelSettings(), std::move(customBuilders));
  const absl::Status buildStatus = problemBuilder.buildProblem(problem);

  ASSERT_TRUE(buildStatus.ok()) << buildStatus.ToString();

  // Verify terms exist in problem
  EXPECT_FALSE(problem.costPtr->empty());
  EXPECT_FALSE(problem.finalCostPtr->empty());
  EXPECT_FALSE(problem.softConstraintPtr->empty());
  EXPECT_FALSE(problem.equalityConstraintPtr->empty());
}

}  // namespace ocs2::humanoid
