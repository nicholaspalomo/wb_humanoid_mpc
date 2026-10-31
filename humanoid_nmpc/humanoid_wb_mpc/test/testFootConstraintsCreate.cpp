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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"
#include "ocs2_core/PreComputation.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_nmpc/humanoid_wb_mpc/test/AffineEndEffectorDynamics.h"
#include "humanoid_wb_mpc/WBMpcInterface.h"
#include "humanoid_wb_mpc/constraint/SwingLegVerticalConstraintCppAd.h"
#include "humanoid_wb_mpc/constraint/ZeroAccelerationConstraintCppAd.h"

/*
 * The foot constraints of the whole-body MPC on the G1 reference manager:
 * ZeroAccelerationConstraintCppAd::Create() and
 * SwingLegVerticalConstraintCppAd::Create() hand back the refusal of the
 * single-end-effector constraint they are built on, where their constructors
 * used to throw it, and make the constraint on one end effector, active in
 * stance and in swing respectively; the stance constraint with the shipped
 * gains acts on the pose rates of its dynamics (the rate and acceleration of
 * the orientation error), not on its linear velocity and acceleration, and
 * servoes the height row to the ground. An affine stand-in plays the
 * end-effector dynamics. Builds no CppAD model.
 */

namespace ocs2::humanoid {
namespace {

using test_support::AffineEndEffectorDynamics;

constexpr char kTask[] =
    "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto";
constexpr char kUrdf[] =
    "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf";
constexpr char kReference[] =
    "robot_models/unitree_g1/g1_wb_mpc/config/command/reference.textproto";
constexpr size_t kContactIndex = 0;

std::unique_ptr<WBMpcInterface> controllerModels() {
  absl::StatusOr<std::unique_ptr<WBMpcInterface>> created =
      WBMpcInterface::CreateControllerModels(kTask, kUrdf, kReference);
  EXPECT_TRUE(created.ok()) << created.status();
  return created.ok() ? *std::move(created) : nullptr;
}

TEST(FootConstraintsCreate,
     TheStanceFootConstraintRefusesDynamicsOfTwoEndEffectors) {
  const std::unique_ptr<WBMpcInterface> interface = controllerModels();
  ASSERT_NE(interface, nullptr);
  const SwitchedModelReferenceManager& referenceManager =
      *interface->getSwitchedModelReferenceManagerPtr();

  const AffineEndEffectorDynamics twoFeet(
      std::vector<std::string>{"left_foot", "right_foot"});
  const absl::StatusOr<std::unique_ptr<ZeroAccelerationConstraintCppAd>>
      refused = ZeroAccelerationConstraintCppAd::Create(referenceManager,
                                                        twoFeet, kContactIndex);
  ASSERT_FALSE(refused.ok());
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(refused.status().message(),
                                "EndEffectorDynamicsAccelerationsConstraint"))
      << refused.status();

  const AffineEndEffectorDynamics oneFoot;
  const absl::StatusOr<std::unique_ptr<ZeroAccelerationConstraintCppAd>>
      created = ZeroAccelerationConstraintCppAd::Create(referenceManager,
                                                        oneFoot, kContactIndex);
  ASSERT_TRUE(created.ok()) << created.status();
  EXPECT_EQ((*created)->getNumConstraints(/*time=*/0.0), 6u);
  EXPECT_EQ((*created)->isActive(/*time=*/0.0),
            referenceManager.isInContact(/*time=*/0.0, kContactIndex));
}

TEST(FootConstraintsCreate,
     TheStanceFootConstraintActsOnThePoseRatesOfItsDynamics) {
  const std::unique_ptr<WBMpcInterface> interface = controllerModels();
  ASSERT_NE(interface, nullptr);
  SwitchedModelReferenceManager& referenceManager =
      *interface->getSwitchedModelReferenceManagerPtr();
  // The swing planner's trajectories, whose stance height the height row
  // servoes to.
  referenceManager.preSolverRun(/*initTime=*/-0.5, /*finalTime=*/1.0,
                                interface->getInitialState(),
                                ModeNumber::kStance);
  const ModelSettings::FootConstraintConfig& gains =
      interface->modelSettings().footConstraintConfig;
  const AffineEndEffectorDynamics dynamics;
  const absl::StatusOr<std::unique_ptr<ZeroAccelerationConstraintCppAd>>
      created = ZeroAccelerationConstraintCppAd::Create(
          referenceManager, dynamics, kContactIndex,
          stanceFootAccelerationConstraintConfig(gains));
  ASSERT_TRUE(created.ok()) << created.status();
  const vector_t state =
      (vector_t(test_support::kAffineStateDim) << 0.3, -0.7, 1.2, 0.4)
          .finished();
  const vector_t input =
      (vector_t(test_support::kAffineInputDim) << -0.5, 0.9, 0.2, -1.1, 0.6)
          .finished();

  vector6_t pose;
  pose << dynamics.position(state), dynamics.orientationErrorWrtPlane(state);
  const vector6_t rate = dynamics.poseWrtPlaneRate(state, input);
  const vector6_t acceleration =
      dynamics.poseWrtPlaneAcceleration(state, input);
  vector6_t expected;
  for (Eigen::Index i = 0; i < 2; ++i) {
    expected(i) = gains.linearVelocityErrorGain_xy * rate(i) +
                  gains.linearAccelerationErrorGain_xy * acceleration(i);
  }
  const scalar_t groundHeight =
      referenceManager.getSwingTrajectoryPlanner()
          ->getDesPosFootRtWorldEwrtWorldZ(kContactIndex, /*time=*/0.0);
  expected(2) = gains.positionErrorGain_z * (pose(2) - groundHeight) +
                gains.linearVelocityErrorGain_z * rate(2) +
                gains.linearAccelerationErrorGain_z * acceleration(2);
  for (Eigen::Index i = 3; i < 6; ++i) {
    expected(i) = gains.orientationErrorGain * pose(i) +
                  gains.angularVelocityErrorGain * rate(i) +
                  gains.angularAccelerationErrorGain * acceleration(i);
  }
  const vector_t value =
      (*created)->getValue(/*time=*/0.0, state, input, PreComputation());
  EXPECT_TRUE(value.isApprox(expected, /*prec=*/1.0e-12))
      << value.transpose() << "\nvs\n"
      << expected.transpose();
  EXPECT_TRUE(
      (*created)
          ->getLinearApproximation(/*time=*/0.0, state, input, PreComputation())
          .f.isApprox(value, /*prec=*/1.0e-12));

  // Positive control: on the stand-in's linear velocity and acceleration, which
  // the swing foot's constraint reads, the linear rows would be other values.
  const vector3_t velocity = dynamics.velocity(state, input);
  const vector3_t linearAcceleration =
      dynamics.linearAcceleration(state, input);
  vector3_t onTheVelocity;
  for (Eigen::Index i = 0; i < 2; ++i) {
    onTheVelocity(i) =
        gains.linearVelocityErrorGain_xy * velocity(i) +
        gains.linearAccelerationErrorGain_xy * linearAcceleration(i);
  }
  onTheVelocity(2) =
      gains.positionErrorGain_z * (pose(2) - groundHeight) +
      gains.linearVelocityErrorGain_z * velocity(2) +
      gains.linearAccelerationErrorGain_z * linearAcceleration(2);
  EXPECT_GT((value.head<3>() - onTheVelocity).cwiseAbs().maxCoeff(), 1.0)
      << "the case under test: rates other than the velocity";
}

TEST(FootConstraintsCreate,
     TheSwingFootConstraintRefusesDynamicsOfTwoEndEffectors) {
  const std::unique_ptr<WBMpcInterface> interface = controllerModels();
  ASSERT_NE(interface, nullptr);
  const SwitchedModelReferenceManager& referenceManager =
      *interface->getSwitchedModelReferenceManagerPtr();

  const AffineEndEffectorDynamics twoFeet(
      std::vector<std::string>{"left_foot", "right_foot"});
  const absl::StatusOr<std::unique_ptr<SwingLegVerticalConstraintCppAd>>
      refused = SwingLegVerticalConstraintCppAd::Create(referenceManager,
                                                        twoFeet, kContactIndex);
  ASSERT_FALSE(refused.ok());
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(refused.status().message(),
                                "EndEffectorDynamicsLinearAccConstraint"))
      << refused.status();

  const AffineEndEffectorDynamics oneFoot;
  const absl::StatusOr<std::unique_ptr<SwingLegVerticalConstraintCppAd>>
      created = SwingLegVerticalConstraintCppAd::Create(referenceManager,
                                                        oneFoot, kContactIndex);
  ASSERT_TRUE(created.ok()) << created.status();
  EXPECT_EQ((*created)->getNumConstraints(/*time=*/0.0), 1u);
  EXPECT_EQ((*created)->isActive(/*time=*/0.0),
            !referenceManager.isInContact(/*time=*/0.0, kContactIndex));
}

}  // namespace
}  // namespace ocs2::humanoid
