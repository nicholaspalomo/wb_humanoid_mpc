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

#include "humanoid_nmpc/humanoid_wb_mpc/test/AffineEndEffectorDynamics.h"
#include "humanoid_wb_mpc/WBMpcInterface.h"
#include "humanoid_wb_mpc/constraint/SwingLegVerticalConstraintCppAd.h"
#include "humanoid_wb_mpc/constraint/ZeroAccelerationConstraintCppAd.h"

/*
 * The foot constraints of the whole-body MPC on the G1 reference manager: ZeroAccelerationConstraintCppAd::Create() and
 * SwingLegVerticalConstraintCppAd::Create() hand back the refusal of the single-end-effector constraint they are built
 * on, where their constructors used to throw it, and make the constraint on one end effector, active in stance and in
 * swing respectively. An affine stand-in plays the end-effector dynamics. Builds no CppAD model.
 */

namespace ocs2::humanoid {
namespace {

using test_support::AffineEndEffectorDynamics;

constexpr char kTask[] = "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto";
constexpr char kUrdf[] = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf";
constexpr char kReference[] = "robot_models/unitree_g1/g1_wb_mpc/config/command/reference.textproto";
constexpr size_t kContactIndex = 0;

std::unique_ptr<WBMpcInterface> controllerModels() {
  absl::StatusOr<std::unique_ptr<WBMpcInterface>> created = WBMpcInterface::CreateControllerModels(kTask, kUrdf, kReference);
  EXPECT_TRUE(created.ok()) << created.status();
  return created.ok() ? *std::move(created) : nullptr;
}

TEST(FootConstraintsCreate, TheStanceFootConstraintRefusesDynamicsOfTwoEndEffectors) {
  const std::unique_ptr<WBMpcInterface> interface = controllerModels();
  ASSERT_NE(interface, nullptr);
  const SwitchedModelReferenceManager& referenceManager = *interface->getSwitchedModelReferenceManagerPtr();

  const AffineEndEffectorDynamics twoFeet(std::vector<std::string>{"left_foot", "right_foot"});
  const absl::StatusOr<std::unique_ptr<ZeroAccelerationConstraintCppAd>> refused =
      ZeroAccelerationConstraintCppAd::Create(referenceManager, twoFeet, kContactIndex);
  ASSERT_FALSE(refused.ok());
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(refused.status().message(), "EndEffectorDynamicsAccelerationsConstraint")) << refused.status();

  const AffineEndEffectorDynamics oneFoot;
  const absl::StatusOr<std::unique_ptr<ZeroAccelerationConstraintCppAd>> created =
      ZeroAccelerationConstraintCppAd::Create(referenceManager, oneFoot, kContactIndex);
  ASSERT_TRUE(created.ok()) << created.status();
  EXPECT_EQ((*created)->getNumConstraints(/*time=*/0.0), 6u);
  EXPECT_EQ((*created)->isActive(/*time=*/0.0), referenceManager.isInContact(/*time=*/0.0, kContactIndex));
}

TEST(FootConstraintsCreate, TheSwingFootConstraintRefusesDynamicsOfTwoEndEffectors) {
  const std::unique_ptr<WBMpcInterface> interface = controllerModels();
  ASSERT_NE(interface, nullptr);
  const SwitchedModelReferenceManager& referenceManager = *interface->getSwitchedModelReferenceManagerPtr();

  const AffineEndEffectorDynamics twoFeet(std::vector<std::string>{"left_foot", "right_foot"});
  const absl::StatusOr<std::unique_ptr<SwingLegVerticalConstraintCppAd>> refused =
      SwingLegVerticalConstraintCppAd::Create(referenceManager, twoFeet, kContactIndex);
  ASSERT_FALSE(refused.ok());
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(refused.status().message(), "EndEffectorDynamicsLinearAccConstraint")) << refused.status();

  const AffineEndEffectorDynamics oneFoot;
  const absl::StatusOr<std::unique_ptr<SwingLegVerticalConstraintCppAd>> created =
      SwingLegVerticalConstraintCppAd::Create(referenceManager, oneFoot, kContactIndex);
  ASSERT_TRUE(created.ok()) << created.status();
  EXPECT_EQ((*created)->getNumConstraints(/*time=*/0.0), 1u);
  EXPECT_EQ((*created)->isActive(/*time=*/0.0), !referenceManager.isInContact(/*time=*/0.0, kContactIndex));
}

}  // namespace
}  // namespace ocs2::humanoid
