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

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"

#include "humanoid_common_mpc/acom/AngularCenterOfMass.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningReferenceManager.h"
#include "humanoid_common_mpc/contact_planning/execution/ExecutionContext.h"
#include "humanoid_common_mpc/contact_planning/execution/PlannedHeadingOverride.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/orientation/BaseOrientation.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/QuaternionLayoutRobotModel.h"
#include "robot_core/ResourcePaths.h"

/**
 * The two readers of the Angular Center of Mass heading, on the shipped DRC
 * Atlas (the one robot whose contact planner runs the ACoM as its heading
 * model): the planner's input heading
 * (ContactPlanningReferenceManager::computeHeading), the twist heading
 * psi_h(xi_a) of the ACoM attitude xi_a = xi (x) Exp(Delta_theta(q_j)); and
 * planned_heading_override, which sets a target knot's yaw so that the knot's
 * ACoM heading is the planned one. The ACoM cost compares those attitudes, so
 * the two must agree with it and with each other. The states are the centroidal
 * MPC's quaternion layout (test/support/QuaternionLayoutRobotModel.h), whose
 * base orientation is four slots xi read through xi / |xi|.
 */
namespace ocs2::humanoid {
namespace {

constexpr char kTaskFile[] =
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto";
constexpr char kReferenceFile[] =
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/"
    "reference.textproto";
constexpr char kUrdfFile[] =
    "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";

/** `angle` wrapped to (-pi, pi]. */
scalar_t wrapped(scalar_t angle) {
  return moduloAngleWithReference(angle, /*reference=*/0.0);
}

/** A plan over [0, 1] s in ten nodes whose heading goes linearly from
 * `startHeading` to `endHeading`. */
ContactPlan headingPlan(scalar_t startHeading, scalar_t endHeading) {
  constexpr size_t kNodes = 10;
  ContactPlan plan;
  plan.valid = true;
  plan.startTime = 0.0;
  plan.dt = 0.1;
  plan.committedUntil = plan.dt;
  plan.contacts.assign(kNodes, makeFeetArray(true));
  plan.footholds.assign(kNodes + 1,
                        makeFeetArray(vector2_t(vector2_t::Zero())));
  plan.comPosition.assign(kNodes + 1, vector2_t::Zero());
  plan.comVelocity.assign(kNodes + 1, vector2_t::Zero());
  plan.zmp.assign(kNodes, vector2_t::Zero());
  plan.heading.resize(kNodes + 1);
  for (size_t node = 0; node <= kNodes; ++node) {
    plan.heading[node] = startHeading + (endHeading - startHeading) *
                                            static_cast<scalar_t>(node) /
                                            static_cast<scalar_t>(kNodes);
  }
  return plan;
}

/** The shipped DRC Atlas's settings, Pinocchio model, its centroidal
 * quaternion-layout robot model and its ACoM network. */
class PlannedHeadingOverrideAcomTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const absl::StatusOr<std::string> taskFile =
        robot::resolveResourcePath(kTaskFile);
    const absl::StatusOr<std::string> referenceFile =
        robot::resolveResourcePath(kReferenceFile);
    const absl::StatusOr<std::string> urdfFile =
        robot::resolveResourcePath(kUrdfFile);
    ASSERT_TRUE(taskFile.ok() && referenceFile.ok() && urdfFile.ok())
        << "the shipped DRC Atlas files are not in the runfiles";
    referenceFile_ = *referenceFile;
    absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(*taskFile);
    ASSERT_TRUE(task.ok()) << task.status();
    absl::StatusOr<ModelSettings> settings = ModelSettings::Create(
        *task, *urdfFile, "testPlannedHeadingOverrideAcom", /*verbose=*/false);
    ASSERT_TRUE(settings.ok()) << settings.status();
    settings_ = std::make_unique<ModelSettings>(*std::move(settings));
    absl::StatusOr<PinocchioInterface> pinocchioInterface =
        loadCustomPinocchioInterface(*task, *urdfFile, *settings_);
    ASSERT_TRUE(pinocchioInterface.ok()) << pinocchioInterface.status();
    pinocchioInterface_ =
        std::make_unique<PinocchioInterface>(*std::move(pinocchioInterface));
    model_ = std::make_unique<test::QuaternionLayoutRobotModel>(
        *settings_, StateInputLayout::Mpc::kCentroidal);
    absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> acom =
        AngularCenterOfMass::Create(settings_->robotName,
                                    settings_->mpcModelJointNames);
    ASSERT_TRUE(acom.ok()) << acom.status();
    acom_ = *std::move(acom);

    // Bent joints, so that the network's rotation is far from the identity.
    const Eigen::Index jointDim =
        static_cast<Eigen::Index>(model_->getJointDim());
    joints_ = vector_t(jointDim);
    for (Eigen::Index joint = 0; joint < jointDim; ++joint)
      joints_(joint) = 0.3 * std::sin(1.7 * static_cast<scalar_t>(joint + 1));
    ASSERT_GT(acom_->computeJointOrientationOffset(joints_).norm(), 0.02)
        << "the joints must rotate the ACoM for these tests to mean anything";
  }

  /** A state of the fixture's robot with the base at the ZYX Euler angles
   * `eulerZyx` and the fixture's bent joints. */
  vector_t stateAt(const vector3_t& eulerZyx) const {
    vector_t state = model_->getNeutralState();
    model_->setPosBaseRtWorldEwrtWorld(state, vector3_t(0.2, -0.1, 0.85));
    model_->setEulerzyxBaseRtWorld(state, eulerZyx);
    model_->setJointAngles(state, joints_);
    return state;
  }

  /** The twist heading of the ACoM attitude of `state`. */
  scalar_t acomHeadingOf(const vector_t& state) const {
    return yawOf(acom_->acomAttitude(model_->getQuatBaseRtWorldXyzw(state),
                                     model_->getJointAngles(state)));
  }

  /** A fresh ACoM evaluator of the fixture's robot, for a manager to own; null
   * after a failed expectation. */
  std::unique_ptr<AngularCenterOfMass> makeAcom() const {
    absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> acom =
        AngularCenterOfMass::Create(settings_->robotName,
                                    settings_->mpcModelJointNames);
    EXPECT_TRUE(acom.ok()) << acom.status();
    return acom.ok() ? *std::move(acom) : nullptr;
  }

  /** The contact planner's reference manager of the fixture's robot, with the
   * heading model on, or null after a failed expectation. */
  std::shared_ptr<ContactPlanningReferenceManager> headingReferenceManager()
      const {
    ContactPlanningConfig config;
    config.shared.comHeight = 0.85;  // no model derives the pendulum here
                                     // (unset means "from the model")
    config.setHeadingModel(true);
    EXPECT_EQ(config.validateStatus(), absl::OkStatus());
    absl::StatusOr<std::shared_ptr<GaitSchedule>> gaitSchedule =
        GaitSchedule::Create(referenceFile_, *settings_);
    EXPECT_TRUE(gaitSchedule.ok()) << gaitSchedule.status();
    if (!gaitSchedule.ok()) return nullptr;
    absl::StatusOr<std::shared_ptr<ContactPlanningReferenceManager>> manager =
        ContactPlanningReferenceManager::Create(
            *std::move(gaitSchedule),
            std::make_unique<SwingTrajectoryPlanner>(
                SwingTrajectoryPlanner::Config(), /*numFeet=*/2),
            *pinocchioInterface_, *model_, config);
    EXPECT_TRUE(manager.ok()) << manager.status();
    return manager.ok() ? *std::move(manager) : nullptr;
  }

  std::string referenceFile_;
  std::unique_ptr<ModelSettings> settings_;
  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  std::unique_ptr<test::QuaternionLayoutRobotModel> model_;
  std::unique_ptr<AngularCenterOfMass> acom_;
  vector_t joints_;
};

/**
 * The override's offset is the twist heading of the network's rotation, and a
 * rotation about the world z axis adds to the twist: the ACoM heading of a
 * level base at yaw psi is psi plus the offset, modulo 2 pi, exactly - on both
 * sides of the +-pi seam.
 */
TEST_F(PlannedHeadingOverrideAcomTest,
       TheOffsetIsTheTwistOfTheJointRotationAndAddsToALevelBasesYaw) {
  const scalar_t offset = acomHeadingOffset(*acom_, joints_);
  const vector4_t identity(0.0, 0.0, 0.0, 1.0);
  EXPECT_NEAR(offset, yawOf(acom_->acomAttitude(identity, joints_)), 1.0e-15);
  EXPECT_GT(std::abs(offset), 1.0e-3)
      << "the joints must turn the ACoM for this to tell the ACoM heading from "
         "the base's";

  const std::array<scalar_t, 5> yaws = {0.0, 0.7, -2.0, 3.1, -3.1};
  for (const scalar_t yaw : yaws) {
    EXPECT_LT(
        std::abs(wrapped(acomHeadingOf(stateAt(vector3_t(yaw, 0.0, 0.0))) -
                         (yaw + offset))),
        1.0e-12)
        << "yaw " << yaw;
  }
}

/**
 * The override sets each level knot inside the plan so that its ACoM heading is
 * the planned heading at the knot's time
 * - exactly, modulo 2 pi - and keeps its pitch and roll (to round-off: they are
 * written back through the knot's quaternion). Without an evaluator it is the
 * base yaw that becomes the planned heading. A knot past the plan's end keeps
 * the operator's yaw. The quaternion layout stores no branch of the yaw (the
 * knots' Euler yaws read back in
 * (-pi, pi]), so the re-branching next to the knot's own yaw
 * (moduloAngleWithReference) leaves no trace in the state.
 */
TEST_F(PlannedHeadingOverrideAcomTest,
       OverriddenLevelKnotsCarryThePlannedAcomHeading) {
  const ContactPlan plan =
      headingPlan(/*startHeading=*/2.8, /*endHeading=*/3.6);
  ContactPlanningConfig config;
  ExecutionContext context(config);
  context.activePlan = &plan;
  const vector_t input =
      vector_t::Zero(static_cast<Eigen::Index>(model_->getInputDim()));
  const scalar_array_t times = {0.0, 0.45, 0.9, 1.5};
  const vector_array_t knots = {
      stateAt(vector3_t(2.7, 0.0, 0.0)), stateAt(vector3_t(-3.0, 0.0, 0.0)),
      stateAt(vector3_t(9.6, 0.0, 0.0)), stateAt(vector3_t(0.4, 0.0, 0.0))};

  const std::unique_ptr<AngularCenterOfMass> noAcom;
  const std::array<const std::unique_ptr<AngularCenterOfMass>* absl_nonnull, 2>
      slots = {&acom_, &noAcom};
  for (const std::unique_ptr<AngularCenterOfMass>* absl_nonnull const slot :
       slots) {
    const PlannedHeadingOverride rule(*model_, slot);
    TargetTrajectories target(times, knots,
                              vector_array_t(times.size(), input));
    rule.overrideTarget(context, target);
    for (size_t knot = 0; knot + 1 < times.size(); ++knot) {
      const vector_t& rewritten = target.stateTrajectory[knot];
      const std::optional<scalar_t> planned = plan.headingAtTime(times[knot]);
      // An if, not ASSERT_TRUE: bugprone-unchecked-optional-access sees the
      // check only through an if.
      if (!planned.has_value())
        FAIL() << "the plan has no heading at knot " << knot;
      const scalar_t heading =
          (*slot != nullptr) ? acomHeadingOf(rewritten)
                             : model_->getEulerzyxBaseRtWorld(rewritten)(0);
      EXPECT_LT(std::abs(wrapped(heading - *planned)), 1.0e-12)
          << "knot " << knot << (*slot != nullptr ? " (ACoM)" : " (base)");
      const vector3_t before = model_->getEulerzyxBaseRtWorld(knots[knot]);
      const vector3_t after = model_->getEulerzyxBaseRtWorld(rewritten);
      EXPECT_LT((vector2_t(after.tail<2>()) - vector2_t(before.tail<2>()))
                    .cwiseAbs()
                    .maxCoeff(),
                1.0e-12)
          << "knot " << knot << ": the override changed the pitch or the roll";
      EXPECT_EQ(model_->getPosBaseRtWorldEwrtWorld(rewritten),
                model_->getPosBaseRtWorldEwrtWorld(knots[knot]))
          << "knot " << knot;
      EXPECT_EQ(model_->getJointAngles(rewritten),
                model_->getJointAngles(knots[knot]))
          << "knot " << knot;
    }
    EXPECT_EQ(target.stateTrajectory.back(), knots.back())
        << "a knot past the plan's end must keep the operator's yaw";
  }
}

/** A tilted knot keeps its tilt: the override replaces the ZYX yaw alone,
 * whatever the evaluator. */
TEST_F(PlannedHeadingOverrideAcomTest, ATiltedKnotKeepsItsPitchAndRoll) {
  const ContactPlan plan =
      headingPlan(/*startHeading=*/0.5, /*endHeading=*/0.5);
  ContactPlanningConfig config;
  ExecutionContext context(config);
  context.activePlan = &plan;
  const vector_t knot = stateAt(vector3_t(0.1, 0.12, -0.07));
  TargetTrajectories target(
      {0.5}, {knot},
      {vector_t::Zero(static_cast<Eigen::Index>(model_->getInputDim()))});
  PlannedHeadingOverride(*model_, &acom_).overrideTarget(context, target);
  const vector3_t after =
      model_->getEulerzyxBaseRtWorld(target.stateTrajectory.front());
  EXPECT_LT((vector2_t(after.tail<2>()) - vector2_t(0.12, -0.07))
                .cwiseAbs()
                .maxCoeff(),
            1.0e-12)
      << after.transpose();
  EXPECT_NEAR(after(0), 0.5 - acomHeadingOffset(*acom_, joints_), 1.0e-12);
}

/**
 * With an evaluator the planner's heading is the twist heading of the ACoM
 * attitude of the state, in (-pi, pi] whatever the stored yaw; without one it
 * is the robot model's base heading. At a tilted, turned state with bent joints
 * the two differ, so the check can fail.
 */
TEST_F(PlannedHeadingOverrideAcomTest,
       ThePlannersHeadingIsTheTwistHeadingOfTheAcomAttitude) {
  const std::shared_ptr<ContactPlanningReferenceManager> manager =
      headingReferenceManager();
  ASSERT_NE(manager, nullptr);
  const vector_t state = stateAt(vector3_t(3.5, 0.15, -0.1));
  EXPECT_EQ(manager->computeHeading(state), model_->getYawBaseRtWorld(state))
      << "without an evaluator";

  manager->setAngularCenterOfMass(makeAcom());
  const scalar_t heading = manager->computeHeading(state);
  EXPECT_EQ(heading, acomHeadingOf(state));
  EXPECT_GT(heading, -std::numbers::pi);
  EXPECT_LE(heading, std::numbers::pi);
  EXPECT_GT(std::abs(wrapped(heading - model_->getYawBaseRtWorld(state))),
            1.0e-3)
      << "the ACoM heading must differ from the base's here";
  EXPECT_EQ(
      manager->makePlannerInput(/*initTime=*/0.0, state, vector2_t::Zero())
          .heading,
      heading);
}

/**
 * On the quaternion layout the planner's heading reads the stored quaternion
 * through xi / |xi|: the same attitude stored at another length or as its
 * negative -xi gives the same heading, with the ACoM and without it. It is
 * defined where the ZYX yaw is not - at a pitch of exactly 90 degrees, where
 * Euler angles leave only yaw - roll - and it is continuous across the +-pi
 * seam, where the stored quaternion's w changes sign.
 */
TEST_F(
    PlannedHeadingOverrideAcomTest,
    OnTheQuaternionLayoutThePlannersHeadingReadsTheAttitudeAndNotItsCoordinates) {
  const std::shared_ptr<ContactPlanningReferenceManager> manager =
      headingReferenceManager();
  ASSERT_NE(manager, nullptr);
  const Eigen::Index orientation =
      static_cast<Eigen::Index>(model_->getBaseOrientationStartIndex());
  const vector_t state = stateAt(vector3_t(2.4, 0.2, -0.3));
  for (const bool withAcom : {false, true}) {
    manager->setAngularCenterOfMass(withAcom ? makeAcom() : nullptr);
    const scalar_t heading = manager->computeHeading(state);
    for (const scalar_t scale : {-1.0, 0.25, -3.0}) {
      vector_t scaled = state;
      scaled.segment<4>(orientation) *= scale;
      EXPECT_LT(std::abs(wrapped(manager->computeHeading(scaled) - heading)),
                1.0e-12)
          << "scale " << scale << (withAcom ? " (ACoM)" : " (base)");
    }

    // Pitched up by exactly 90 degrees about the body y axis of a base at yaw
    // 0.8: the twist heading is finite.
    vector_t vertical = stateAt(vector3_t(0.8, 0.0, 0.0));
    model_->setQuatBaseRtWorldXyzw(
        vertical, quaternionProduct(model_->getQuatBaseRtWorldXyzw(vertical),
                                    quaternionExp(vector3_t(
                                        0.0, 0.5 * std::numbers::pi, 0.0))));
    const scalar_t verticalHeading = manager->computeHeading(vertical);
    EXPECT_TRUE(std::isfinite(verticalHeading)) << (withAcom ? "ACoM" : "base");
    EXPECT_GT(verticalHeading, -std::numbers::pi);
    EXPECT_LE(verticalHeading, std::numbers::pi);

    // Across the seam: the yaw of a level base 1e-6 on either side of pi.
    const scalar_t below = manager->computeHeading(
        stateAt(vector3_t(std::numbers::pi - 1.0e-6, 0.0, 0.0)));
    const scalar_t above = manager->computeHeading(
        stateAt(vector3_t(std::numbers::pi + 1.0e-6, 0.0, 0.0)));
    EXPECT_LT(std::abs(wrapped(above - below)), 3.0e-6)
        << (withAcom ? "ACoM" : "base");
  }
  manager->setAngularCenterOfMass(nullptr);
  EXPECT_EQ(manager->computeHeading(stateAt(vector3_t(0.8, 0.0, 0.0))),
            model_->getYawBaseRtWorld(stateAt(vector3_t(0.8, 0.0, 0.0))));
  EXPECT_NEAR(model_->getYawBaseRtWorld(stateAt(vector3_t(0.8, 0.0, 0.0))), 0.8,
              1.0e-12);
}

}  // namespace
}  // namespace ocs2::humanoid
