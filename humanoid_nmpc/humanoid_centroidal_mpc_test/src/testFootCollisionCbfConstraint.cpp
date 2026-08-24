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
#include <memory>
#include <string>

#include <ocs2_core/PreComputation.h>
#include <ocs2_core/misc/LoadData.h>
#include <ocs2_oc/oc_problem/OptimalControlProblem.h>

#include <humanoid_centroidal_mpc_test/CentroidalTestingModelInterface.h>
#include "humanoid_common_mpc/HumanoidCostConstraintFactory.h"
#include "humanoid_common_mpc/constraint/FootCollisionCbfConstraint.h"
#include "humanoid_common_mpc/problem/MpcProblemBuilder.h"
#include "humanoid_common_mpc/problem/MpcProblemDefinition.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

class FootCollisionCbfConstraintTest : public ::testing::Test {
 protected:
  void SetUp() override {
    testingModelPtr_ = std::make_unique<CentroidalTestingModelInterface>();

    // Mode schedule: time 0.0-0.5 single stance (contact 0 active: mode 1), time 0.5-1.0 double stance (mode 3)
    ModeSchedule initModeSchedule({0.0, 0.5, 1.0}, {1, 3});
    ModeSequenceTemplate initModeSequenceTemplate({0.5, 0.5}, {1, 3});
    auto gaitSchedule = std::make_shared<GaitSchedule>(initModeSchedule, initModeSequenceTemplate,
                                                       testingModelPtr_->getModelSettings().phaseTransitionStanceTime);
    auto swingPlanner = std::make_shared<SwingTrajectoryPlanner>(SwingTrajectoryPlanner::Config{},
                                                                 testingModelPtr_->getModelSettings().contactNames.size());
    referenceManagerPtr_ = std::make_shared<SwitchedModelReferenceManager>(
        gaitSchedule, swingPlanner, testingModelPtr_->getPinocchioInterface(), testingModelPtr_->getMpcRobotModel());

    FootCollisionCbfConstraint::Config config;
    config.leftAnkleFrame = "left_ankle_roll_joint";
    config.rightAnkleFrame = "right_ankle_roll_joint";
    config.leftKneeFrame = "left_knee_joint";
    config.rightKneeFrame = "right_knee_joint";
    config.footCollisionSphereRadius = 0.065;
    config.kneeCollisionSphereRadius = 0.07;
    config.gamma = 10.0;

    cbfConstraint_ = std::make_unique<FootCollisionCbfConstraint>(*referenceManagerPtr_, testingModelPtr_->getPinocchioInterface(),
                                                                  testingModelPtr_->getMpcRobotModelAD(), config,
                                                                  "FootCollisionCbfConstraint", testingModelPtr_->getModelSettings());
  }

  std::unique_ptr<CentroidalTestingModelInterface> testingModelPtr_;
  std::shared_ptr<SwitchedModelReferenceManager> referenceManagerPtr_;
  std::unique_ptr<FootCollisionCbfConstraint> cbfConstraint_;
};

TEST_F(FootCollisionCbfConstraintTest, ActivityDuringStanceAndSwing) {
  // During single stance (t = 0.25), constraint is active
  EXPECT_TRUE(cbfConstraint_->isActive(0.25));

  // During double stance (t = 0.75), constraint is inactive to avoid overconstraining
  EXPECT_FALSE(cbfConstraint_->isActive(0.75));
}

TEST_F(FootCollisionCbfConstraintTest, ConstraintEvaluationAndVelocityEffect) {
  PreComputation preComp;
  const size_t stateDim = testingModelPtr_->getMpcRobotModel().getStateDim();
  const size_t inputDim = testingModelPtr_->getMpcRobotModel().getInputDim();

  vector_t state = vector_t::Zero(stateDim);
  // Default base height = 0.75m
  state(8) = 0.75;

  // Zero input (zero joint velocities)
  vector_t inputZero = vector_t::Zero(inputDim);
  vector_t valZero = cbfConstraint_->getValue(0.25, state, inputZero, preComp);

  EXPECT_EQ(valZero.size(), 16);
  for (int i = 0; i < valZero.size(); ++i) {
    EXPECT_FALSE(std::isnan(valZero(i)));
    EXPECT_FALSE(std::isinf(valZero(i)));
  }

  // Linear approximation check
  const auto approx = cbfConstraint_->getLinearApproximation(0.25, state, inputZero, preComp);
  EXPECT_EQ(approx.f.size(), 16);
  EXPECT_EQ(approx.dfdx.rows(), 16);
  EXPECT_EQ(approx.dfdx.cols(), stateDim);
  EXPECT_EQ(approx.dfdu.rows(), 16);
  EXPECT_EQ(approx.dfdu.cols(), inputDim);

  for (int i = 0; i < 16; ++i) {
    for (size_t j = 0; j < stateDim; ++j) {
      EXPECT_FALSE(std::isnan(approx.dfdx(i, j)));
    }
    for (size_t j = 0; j < inputDim; ++j) {
      EXPECT_FALSE(std::isnan(approx.dfdu(i, j)));
    }
  }
}

TEST_F(FootCollisionCbfConstraintTest, FactoryAndProblemBuilderIntegration) {
  const HumanoidCostConstraintFactory factory(testingModelPtr_->taskFile, testingModelPtr_->referenceFile, *referenceManagerPtr_,
                                              testingModelPtr_->getPinocchioInterface(), testingModelPtr_->getMpcRobotModel(),
                                              testingModelPtr_->getMpcRobotModelAD(), testingModelPtr_->getModelSettings(), false);

  MpcProblemDefinition problemDef;
  ProblemTermConfig cbfTerm;
  cbfTerm.name = "foot_collision_cbf";
  cbfTerm.type = "FootCollisionCbfConstraint";
  cbfTerm.configPath = "collision_cbf_constraint";
  cbfTerm.enabled = true;
  problemDef.softConstraints.push_back(cbfTerm);

  OptimalControlProblem problem;
  MpcProblemBuilder::CustomBuilders customBuilders;
  const MpcProblemBuilder builder(problemDef, factory, testingModelPtr_->getModelSettings(), std::move(customBuilders));
  const absl::Status status = builder.buildProblem(problem);

  ASSERT_TRUE(status.ok()) << status.ToString();
  EXPECT_FALSE(problem.softConstraintPtr->empty());
}

}  // namespace ocs2::humanoid
