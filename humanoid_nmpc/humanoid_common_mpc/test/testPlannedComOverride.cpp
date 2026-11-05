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

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "Eigen/Geometry"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/TargetTrajectories.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/execution/ExecutionContext.h"
#include "humanoid_common_mpc/contact_planning/execution/PlannedComOverride.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/QuaternionLayoutRobotModel.h"
#include "robot_core/ResourcePaths.h"

/**
 * planned_com_override writes the planned center of mass into a target knot as
 * the base position that puts the knot's center of mass there. The horizontal
 * offset from the base to the center of mass is measured once per cycle in
 * world axes; it is fixed in the base for a fixed posture, so a knot whose
 * reference heading has turned by dpsi from the measured one carries the
 * offset turned by dpsi. The states are the centroidal MPC's quaternion layout
 * of the shipped DRC Atlas (test/support/QuaternionLayoutRobotModel.h).
 */
namespace ocs2::humanoid {
namespace {

constexpr char kTaskFile[] =
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto";
constexpr char kUrdfFile[] =
    "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";

// The planned center of mass of every node [m], and the base-to-CoM offset of
// the posture in the base's heading frame [m]: a pelvis whose CoM sits 5 cm
// ahead of the base origin and 1 cm to its left.
const vector2_t kPlannedPosComRtWorldEwrtWorldXy(1.0, 0.5);
const vector2_t kPosComRtBaseEwrtHeadingXy(0.05, 0.01);
constexpr scalar_t kTurnRate = 1.0;  // [rad/s], the commanded yaw rate

/** A standing plan over [0, 1] s in ten nodes whose center of mass rests at
 * kPlannedPosComRtWorldEwrtWorldXy. */
ContactPlan restingComPlan() {
  constexpr size_t kNodes = 10;
  ContactPlan plan;
  plan.valid = true;
  plan.startTime = 0.0;
  plan.dt = 0.1;
  plan.committedUntil = plan.dt;
  plan.contacts.assign(kNodes, makeFeetArray(true));
  plan.desPosFeetRtWorldEwrtWorldXyTraj.assign(
      kNodes + 1, makeFeetArray(vector2_t(vector2_t::Zero())));
  plan.posComRtWorldEwrtWorldXyTraj.assign(kNodes + 1,
                                           kPlannedPosComRtWorldEwrtWorldXy);
  plan.linvelComRtWorldEwrtWorldXyTraj.assign(kNodes + 1, vector2_t::Zero());
  plan.posZmpRtWorldEwrtWorldXyTraj.assign(kNodes, vector2_t::Zero());
  return plan;
}

/** The shipped DRC Atlas's settings and its centroidal quaternion-layout robot
 * model. */
class PlannedComOverrideTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const absl::StatusOr<std::string> taskFile =
        robot::resolveResourcePath(kTaskFile);
    const absl::StatusOr<std::string> urdfFile =
        robot::resolveResourcePath(kUrdfFile);
    ASSERT_TRUE(taskFile.ok() && urdfFile.ok())
        << "the shipped DRC Atlas files are not in the runfiles";
    absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(*taskFile);
    ASSERT_TRUE(task.ok()) << task.status();
    absl::StatusOr<ModelSettings> settings = ModelSettings::Create(
        *task, *urdfFile, "testPlannedComOverride", /*verbose=*/false);
    ASSERT_TRUE(settings.ok()) << settings.status();
    settings_ = std::make_unique<ModelSettings>(*std::move(settings));
    model_ = std::make_unique<test::QuaternionLayoutRobotModel>(
        *settings_, StateInputLayout::Mpc::kCentroidal);
  }

  /** A level state with the base at `posBaseRtWorldEwrtWorldXy` and heading
   * `yawBaseRtWorld`. */
  vector_t stateAt(const vector2_t& posBaseRtWorldEwrtWorldXy,
                   scalar_t yawBaseRtWorld) const {
    vector_t state = model_->getNeutralState();
    model_->setPosBaseRtWorldEwrtWorld(
        state, vector3_t(posBaseRtWorldEwrtWorldXy.x(),
                         posBaseRtWorldEwrtWorldXy.y(), 0.85));
    model_->setEulerzyxBaseRtWorld(state, vector3_t(yawBaseRtWorld, 0.0, 0.0));
    return state;
  }

  /**
   * Runs the override on a robot measured at heading `measYawBaseRtWorld`
   * whose reference turns at kTurnRate, and checks that every knot's base sits
   * where the posture puts the planned center of mass: c_plan - R_z(psi_ref)
   * r_heading. `setMeasuredYaw` says whether the context is told the measured
   * heading (a context left at its default reads heading 0).
   */
  void expectKnotsPlaceThePlannedCom(scalar_t measYawBaseRtWorld,
                                     bool setMeasuredYaw) const {
    const ContactPlan plan = restingComPlan();
    ContactPlanningConfig config;
    ExecutionContext context(config);
    context.activePlan = &plan;
    const vector2_t measPosBaseRtWorldEwrtWorldXy(0.2, -0.1);
    context.posBaseRtWorldEwrtWorldXy = measPosBaseRtWorldEwrtWorldXy;
    context.posComRtWorldEwrtWorldXy =
        measPosBaseRtWorldEwrtWorldXy +
        Eigen::Rotation2D<scalar_t>(measYawBaseRtWorld) *
            kPosComRtBaseEwrtHeadingXy;
    if (setMeasuredYaw) context.yawBaseRtWorld = measYawBaseRtWorld;

    const scalar_array_t times = {0.0, 0.5, 1.0};
    vector_array_t knots;
    for (const scalar_t time : times) {
      knots.push_back(stateAt(measPosBaseRtWorldEwrtWorldXy,
                              measYawBaseRtWorld + kTurnRate * time));
    }
    const vector_t input =
        vector_t::Zero(static_cast<Eigen::Index>(model_->getInputDim()));
    TargetTrajectories target(times, knots,
                              vector_array_t(times.size(), input));
    PlannedComOverride(*model_).overrideTarget(context, target);

    for (size_t knot = 0; knot < times.size(); ++knot) {
      const scalar_t refYawBaseRtWorld =
          measYawBaseRtWorld + kTurnRate * times[knot];
      const vector2_t expectedPosBaseRtWorldEwrtWorldXy =
          kPlannedPosComRtWorldEwrtWorldXy -
          Eigen::Rotation2D<scalar_t>(refYawBaseRtWorld) *
              kPosComRtBaseEwrtHeadingXy;
      const vector2_t posBaseRtWorldEwrtWorldXy =
          model_->getPosBaseRtWorldEwrtWorld(target.stateTrajectory[knot])
              .head<2>();
      EXPECT_LT((posBaseRtWorldEwrtWorldXy - expectedPosBaseRtWorldEwrtWorldXy)
                    .norm(),
                1.0e-12)
          << "knot " << knot << " at " << times[knot] << " s: base at "
          << posBaseRtWorldEwrtWorldXy.transpose() << ", expected "
          << expectedPosBaseRtWorldEwrtWorldXy.transpose();
    }
  }

  std::unique_ptr<ModelSettings> settings_;
  std::unique_ptr<test::QuaternionLayoutRobotModel> model_;
};

/** Turning in place from heading 0 at 1 rad/s over a 1 s plan: on the old code
 * the last knot's base was 2 |r| sin(0.5) = 0.05 m off. */
TEST_F(PlannedComOverrideTest, TheOffsetTurnsWithTheKnotsReferenceHeading) {
  expectKnotsPlaceThePlannedCom(/*measYawBaseRtWorld=*/0.0,
                                /*setMeasuredYaw=*/false);
}

/** The same turn from a measured heading next to the +-pi seam: the knots'
 * headings wrap, the turn from the measured heading does not. */
TEST_F(PlannedComOverrideTest, TheOffsetTurnsAcrossTheSeam) {
  expectKnotsPlaceThePlannedCom(/*measYawBaseRtWorld=*/2.8,
                                /*setMeasuredYaw=*/true);
}

/** Without a turn every knot gets the measured offset bit for bit. */
TEST_F(PlannedComOverrideTest, WithoutATurnTheMeasuredOffsetIsUsedAsIs) {
  const ContactPlan plan = restingComPlan();
  ContactPlanningConfig config;
  ExecutionContext context(config);
  context.activePlan = &plan;
  constexpr scalar_t kYawBaseRtWorld = 0.7;
  context.posBaseRtWorldEwrtWorldXy = vector2_t(0.2, -0.1);
  context.posComRtWorldEwrtWorldXy = vector2_t(0.23, -0.06);
  const vector_t knot = stateAt(vector2_t(0.2, -0.1), kYawBaseRtWorld);
  // The knot's heading as the model reads it back, so that the turn is zero.
  context.yawBaseRtWorld = model_->getYawBaseRtWorld(knot);
  TargetTrajectories target(
      {0.5}, {knot},
      {vector_t::Zero(static_cast<Eigen::Index>(model_->getInputDim()))});
  PlannedComOverride(*model_).overrideTarget(context, target);
  const vector2_t expected =
      kPlannedPosComRtWorldEwrtWorldXy -
      (context.posComRtWorldEwrtWorldXy - context.posBaseRtWorldEwrtWorldXy);
  EXPECT_EQ(model_->getPosBaseRtWorldEwrtWorld(target.stateTrajectory.front())
                .head<2>(),
            expected);
}

}  // namespace
}  // namespace ocs2::humanoid
