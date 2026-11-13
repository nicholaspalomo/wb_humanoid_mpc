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

// The MPC model of a robot on rolling contacts, the LimX Tron1 on its two
// wheels: the continuous wheel joints keep one coordinate and no position
// bound, the joint-limit cost has no row for them, and the contact positions
// are the wheels' lowest points.

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/penalties/penalties/PieceWisePolynomialBarrierPenalty.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/multibody/data.hpp"  // IWYU pragma: keep
#include "pinocchio/multibody/fwd.hpp"
#include "pinocchio/multibody/model.hpp"  // IWYU pragma: keep

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/constraint/JointLimitsSoftConstraint.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/pinocchio_model/pinocchioUtils.h"
#include "humanoid_mpc_config/model_settings_config.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/QuaternionLayoutRobotModel.h"

namespace ocs2::humanoid {
namespace {

constexpr char kTron1UrdfFile[] =
    "robot_models/limx_tron1/tron1_description/urdf/tron1.urdf";
constexpr char kLeftWheel[] = "wheel_L_contact";
constexpr char kRightWheel[] = "wheel_R_contact";
constexpr char kLeftWheelJoint[] = "wheel_L_Joint";
constexpr char kRightWheelJoint[] = "wheel_R_Joint";
// [m] The wheel radius, and the wheel center's offset along the wheel joint's
// axis.
constexpr scalar_t kWheelRadius = 0.127;
constexpr scalar_t kPosWheelRtLinkEwrtLinkY = 0.00707;
// The Tron1 joints, in the MPC's order: abad, hip, knee, wheel of the left
// leg, then of the right.
constexpr size_t kNumJoints = 8;
constexpr size_t kLeftWheelJointIndex = 3;
constexpr size_t kRightWheelJointIndex = 7;
constexpr scalar_t kTolerance = 1.0e-12;

std::string runfilesPath(absl::string_view relativePath) {
  const char* absl_nullable directory = std::getenv("TEST_SRCDIR");
  return directory == nullptr
             ? std::string(relativePath)
             : absl::StrCat(directory, "/_main/", relativePath);
}

mpc_config::ModelSettingsConfig::RollingContact wheel(absl::string_view name,
                                                      absl::string_view joint,
                                                      scalar_t side) {
  mpc_config::ModelSettingsConfig::RollingContact contact;
  contact.name = std::string(name);
  contact.parent_joint_name = std::string(joint);
  contact.pos_wheel_rt_link_ewrt_link.y = side * kPosWheelRtLinkEwrtLinkY;
  contact.wheel_radius = kWheelRadius;
  return contact;
}

mpc_config::ModelSettingsConfig::Leg leg(absl::string_view name,
                                         absl::string_view contact) {
  mpc_config::ModelSettingsConfig::Leg entry;
  entry.name = std::string(name);
  entry.contacts = {std::string(contact)};
  return entry;
}

/** Tron1 on its two wheels, legs "left" and "right". */
mpc_config::TaskFile tron1Task() {
  mpc_config::TaskFile task;
  task.model_settings.robot_name = "tron1";
  task.model_settings.contacts_rolling = {
      wheel(kLeftWheel, kLeftWheelJoint, /*side=*/1.0),
      wheel(kRightWheel, kRightWheelJoint, /*side=*/-1.0)};
  task.model_settings.legs = {leg("left", kLeftWheel),
                              leg("right", kRightWheel)};
  return task;
}

/** Tron1's settings, its MPC model and a robot model of the centroidal
 * quaternion layout; the settings outlive the models that keep them. */
class Tron1 {
 public:
  static absl::StatusOr<std::unique_ptr<Tron1>> Create() {
    const mpc_config::TaskFile task = tron1Task();
    const std::string urdf = runfilesPath(kTron1UrdfFile);
    absl::StatusOr<ModelSettings> settings =
        ModelSettings::Create(task, urdf, "testRollingContactModel_",
                              /*verbose=*/false);
    if (!settings.ok()) return settings.status();
    std::unique_ptr<ModelSettings> owned =
        std::make_unique<ModelSettings>(*std::move(settings));
    absl::StatusOr<PinocchioInterface> interface =
        loadCustomPinocchioInterface(task, urdf, *owned);
    if (!interface.ok()) return interface.status();
    return absl::WrapUnique(new Tron1(std::move(owned), *std::move(interface)));
  }

  Tron1(const Tron1&) = delete;
  Tron1& operator=(const Tron1&) = delete;
  Tron1(Tron1&&) = delete;
  Tron1& operator=(Tron1&&) = delete;
  ~Tron1() = default;

  const ModelSettings& settings() const { return *settings_; }
  PinocchioInterface& pinocchioInterface() { return pinocchioInterface_; }
  const test::QuaternionLayoutRobotModel& robotModel() const {
    return robotModel_;
  }

  /** The generalized coordinates of the default posture, the base 0.7485 m up
   * and both wheels turned by `wheelAngle` [rad]. */
  vector_t standingCoordinates(scalar_t wheelAngle) const {
    vector_t state = robotModel_.getNeutralState();
    robotModel_.setPosBaseRtWorldEwrtWorld(state, vector3_t(0.0, 0.0, 0.7485));
    vector_t joints = vector_t::Zero(kNumJoints);
    joints << 0.0, 0.35, 0.8, wheelAngle, 0.0, -0.35, -0.8, wheelAngle;
    robotModel_.setJointAngles(state, joints);
    return robotModel_.getGeneralizedCoordinates(state);
  }

 private:
  Tron1(std::unique_ptr<ModelSettings> settings, PinocchioInterface interface)
      : settings_(std::move(settings)),
        pinocchioInterface_(std::move(interface)),
        robotModel_(*settings_, StateInputLayout::Mpc::kCentroidal) {}

  std::unique_ptr<ModelSettings> settings_;
  PinocchioInterface pinocchioInterface_;
  test::QuaternionLayoutRobotModel robotModel_;
};

TEST(RollingContactModel, AWheelJointHasOneCoordinateAndNoPositionBound) {
  absl::StatusOr<std::unique_ptr<Tron1>> tron1 = Tron1::Create();
  ASSERT_TRUE(tron1.ok()) << tron1.status();
  const pinocchio::Model& model = (*tron1)->pinocchioInterface().getModel();
  // The floating base's (x, y, z, w) quaternion is the one coordinate more.
  EXPECT_EQ(model.nq, model.nv + 1);
  for (pinocchio::JointIndex joint = kFirstJointAfterFloatingBase;
       joint < static_cast<pinocchio::JointIndex>(model.njoints); ++joint) {
    SCOPED_TRACE(model.names[joint]);
    EXPECT_EQ(model.nqs[joint], 1);
    EXPECT_EQ(model.nvs[joint], 1);
  }
  const JointPositionLimits limits = readPinocchioJointLimits(
      (*tron1)->pinocchioInterface(), (*tron1)->settings(), Verbosity::kQuiet);
  ASSERT_EQ(limits.lower.size(), static_cast<Eigen::Index>(kNumJoints));
  for (const size_t wheelJoint :
       {kLeftWheelJointIndex, kRightWheelJointIndex}) {
    EXPECT_EQ((*tron1)->settings().mpcModelJointNames[wheelJoint],
              wheelJoint == kLeftWheelJointIndex ? kLeftWheelJoint
                                                 : kRightWheelJoint);
    EXPECT_EQ(limits.lower[wheelJoint],
              -std::numeric_limits<scalar_t>::infinity());
    EXPECT_EQ(limits.upper[wheelJoint],
              std::numeric_limits<scalar_t>::infinity());
  }
  // The legs keep their URDF limits: abad_L_Joint's.
  EXPECT_EQ(limits.lower[0], -0.38397);
  EXPECT_EQ(limits.upper[0], 1.39626);
}

TEST(RollingContactModel, TheContactFrameIsTheWheelCenterOnTheWheelJoint) {
  absl::StatusOr<std::unique_ptr<Tron1>> tron1 = Tron1::Create();
  ASSERT_TRUE(tron1.ok()) << tron1.status();
  const pinocchio::Model& model = (*tron1)->pinocchioInterface().getModel();
  ASSERT_TRUE(model.existFrame(kLeftWheel));
  const pinocchio::Frame& frame = model.frames[model.getFrameId(kLeftWheel)];
  EXPECT_EQ(frame.parentJoint, model.getJointId(kLeftWheelJoint));
  EXPECT_TRUE(frame.placement.translation().isApprox(
      vector3_t(0.0, kPosWheelRtLinkEwrtLinkY, 0.0), kTolerance));
}

TEST(RollingContactModel, TheContactPositionsAreTheWheelsLowestPoints) {
  absl::StatusOr<std::unique_ptr<Tron1>> tron1 = Tron1::Create();
  ASSERT_TRUE(tron1.ok()) << tron1.status();
  PinocchioInterface& interface = (*tron1)->pinocchioInterface();
  const pinocchio::Model& model = interface.getModel();
  const vector_t q = (*tron1)->standingCoordinates(/*wheelAngle=*/0.0);
  const std::vector<vector3_t> posContactsRtWorldEwrtWorld =
      computePosContactsRtWorldEwrtWorld<scalar_t>(q, interface,
                                                   (*tron1)->robotModel());
  ASSERT_EQ(posContactsRtWorldEwrtWorld.size(), 2U);
  for (size_t i = 0; i < posContactsRtWorldEwrtWorld.size(); ++i) {
    SCOPED_TRACE(i);
    const vector3_t posWheelRtWorldEwrtWorld =
        interface.getData()
            .oMf[model.getFrameId((*tron1)->settings().contactNames[i])]
            .translation();
    EXPECT_EQ(posContactsRtWorldEwrtWorld[i],
              vector3_t(posWheelRtWorldEwrtWorld -
                        kWheelRadius * vector3_t::UnitZ()));
    // The default posture puts the wheel centers a radius above the ground.
    EXPECT_NEAR(posContactsRtWorldEwrtWorld[i].z(), 0.0, 1.0e-3);
  }

  // Turning the wheels (to the final positions) moves their material, not the
  // point they touch the ground at.
  const std::vector<vector3_t> finalPosContactsRtWorldEwrtWorld =
      computePosContactsRtWorldEwrtWorld<scalar_t>(
          (*tron1)->standingCoordinates(/*wheelAngle=*/2.5), interface,
          (*tron1)->robotModel());
  for (size_t i = 0; i < posContactsRtWorldEwrtWorld.size(); ++i) {
    EXPECT_TRUE(finalPosContactsRtWorldEwrtWorld[i].isApprox(
        posContactsRtWorldEwrtWorld[i], kTolerance))
        << i;
  }
}

TEST(RollingContactModel, TheHeightsAboveTheContactsAreAboveTheWheelBottoms) {
  absl::StatusOr<std::unique_ptr<Tron1>> tron1 = Tron1::Create();
  ASSERT_TRUE(tron1.ok()) << tron1.status();
  PinocchioInterface& interface = (*tron1)->pinocchioInterface();
  const vector_t q = (*tron1)->standingCoordinates(/*wheelAngle=*/0.0);
  const std::vector<vector3_t> posContactsRtWorldEwrtWorld =
      computePosContactsRtWorldEwrtWorld<scalar_t>(q, interface,
                                                   (*tron1)->robotModel());
  pinocchio::centerOfMass(interface.getModel(), interface.getData(), q,
                          /*computeSubtreeComs=*/false);
  const scalar_t posComRtWorldEwrtWorldZ = interface.getData().com[0].z();
  const scalar_t meanZ = 0.5 * (posContactsRtWorldEwrtWorld[0].z() +
                                posContactsRtWorldEwrtWorld[1].z());
  const scalar_t lowestZ = std::min(posContactsRtWorldEwrtWorld[0].z(),
                                    posContactsRtWorldEwrtWorld[1].z());
  EXPECT_NEAR(computeProxyPosComRtFeetmeanEwrtWorldZ(q, interface,
                                                     (*tron1)->robotModel()),
              posComRtWorldEwrtWorldZ - meanZ, kTolerance);
  EXPECT_NEAR(
      computePosComRtLowcontactEwrtWorldZ(q, interface, (*tron1)->robotModel()),
      posComRtWorldEwrtWorldZ - lowestZ, kTolerance);
  // Tron1's center of mass stands 0.446 m above the ground.
  EXPECT_NEAR(posComRtWorldEwrtWorldZ - meanZ, 0.446, 5.0e-3);
}

/** The barrier sums of JointLimitsSoftConstraint as they were before a limit
 * could be infinite: every offset penalized. */
scalar_t valueOfEveryOffset(const JointPositionLimits& limits,
                            const vector_t& jointPositions,
                            const PieceWisePolynomialBarrierPenalty& penalty) {
  const vector_t upper = limits.upper - jointPositions;
  const vector_t lower = jointPositions - limits.lower;
  return upper
             .unaryExpr(
                 [&](scalar_t hi) { return penalty.getValue(/*t=*/0.0, hi); })
             .sum() +
         lower
             .unaryExpr(
                 [&](scalar_t hi) { return penalty.getValue(/*t=*/0.0, hi); })
             .sum();
}

TEST(RollingContactModel, TheJointLimitCostHasNoRowForAWheel) {
  absl::StatusOr<std::unique_ptr<Tron1>> tron1 = Tron1::Create();
  ASSERT_TRUE(tron1.ok()) << tron1.status();
  const JointPositionLimits limits = readPinocchioJointLimits(
      (*tron1)->pinocchioInterface(), (*tron1)->settings(), Verbosity::kQuiet);
  const PieceWisePolynomialBarrierPenalty::Config barrier(/*muParam=*/0.1,
                                                          /*deltaParam=*/0.1);
  const JointLimitsSoftConstraint constraint(limits, barrier,
                                             (*tron1)->robotModel());
  // A leg near its limit, so that the legs' rows are not zero.
  vector_t jointPositions = vector_t::Zero(kNumJoints);
  jointPositions[0] = -0.35;
  const scalar_t value = constraint.getValue(jointPositions);
  EXPECT_TRUE(std::isfinite(value));
  EXPECT_GT(value, 0.0);
  // However far the wheels have turned.
  jointPositions[kLeftWheelJointIndex] = 1.0e6;
  jointPositions[kRightWheelJointIndex] = -1.0e6;
  EXPECT_EQ(constraint.getValue(jointPositions), value);
  const ScalarFunctionQuadraticApproximation approximation =
      constraint.getQuadraticApproximation(jointPositions);
  EXPECT_EQ(approximation.f, value);
  const size_t jointStart = (*tron1)->robotModel().getJointStartindex();
  for (const size_t wheelJoint :
       {kLeftWheelJointIndex, kRightWheelJointIndex}) {
    const Eigen::Index row = static_cast<Eigen::Index>(jointStart + wheelJoint);
    EXPECT_EQ(approximation.dfdx[row], 0.0);
    EXPECT_EQ(approximation.dfdxx(row, row), 0.0);
  }
  EXPECT_NE(approximation.dfdx[static_cast<Eigen::Index>(jointStart)], 0.0);
}

TEST(RollingContactModel, FiniteLimitsGiveTheSumOfEveryOffsetBitForBit) {
  absl::StatusOr<std::unique_ptr<Tron1>> tron1 = Tron1::Create();
  ASSERT_TRUE(tron1.ok()) << tron1.status();
  JointPositionLimits limits = readPinocchioJointLimits(
      (*tron1)->pinocchioInterface(), (*tron1)->settings(), Verbosity::kQuiet);
  // Bound the wheels as a revolute joint would be.
  for (const size_t wheelJoint :
       {kLeftWheelJointIndex, kRightWheelJointIndex}) {
    limits.lower[wheelJoint] = -3.0;
    limits.upper[wheelJoint] = 3.0;
  }
  const PieceWisePolynomialBarrierPenalty::Config barrier(/*muParam=*/0.1,
                                                          /*deltaParam=*/0.1);
  const JointLimitsSoftConstraint constraint(limits, barrier,
                                             (*tron1)->robotModel());
  const PieceWisePolynomialBarrierPenalty penalty(barrier);
  vector_t jointPositions(kNumJoints);
  jointPositions << -0.35, 1.3, 0.1, 2.95, -1.35, 0.95, -0.8, -2.97;
  EXPECT_EQ(constraint.getValue(jointPositions),
            valueOfEveryOffset(limits, jointPositions, penalty));
  EXPECT_EQ(constraint.getQuadraticApproximation(jointPositions).f,
            valueOfEveryOffset(limits, jointPositions, penalty));
}

}  // namespace
}  // namespace ocs2::humanoid
