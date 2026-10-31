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
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "ocs2_core/reference/TargetTrajectories.h"

#include "humanoid_centroidal_mpc/command/CentroidalMpcTargetTrajectoriesCalculator.h"
#include "humanoid_common_mpc/command/BasePlanarMotion.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "support/ShippedCentroidalRobot.h"
#include "support/TypedConfigFiles.h"

/*
 * A world turn of the robot turns the centroidal command references with it,
 * across a heading of +-pi: the knots of a velocity command and of a pose
 * command from a state at heading 3.1 and from the same state turned by 0.2
 * about the world z axis (heading -2.98, where the Euler yaw of the knots jumps
 * by 2 pi) are each other's turn - the base position about the world origin,
 * the quaternion as a rotation, the normalized linear and angular momentum
 * (world-frame, so that a turning knot's L_G = R_z(dpsi) I_G e_z psidot turns
 * too) - and so is the model's interpolation of them between the knots
 * (MpcRobotModelBase::getDesiredState(), the quaternion slerped), to 1e-12. On
 * every shipped configuration; builds no CppAD model.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kInitTime = 0.7;
constexpr scalar_t kTurn = 0.2;
constexpr scalar_t kTolerance = 1.0e-12;

/** R_z(angle). */
matrix3_t yawRotation(scalar_t angle) {
  return quaternionRotationMatrix<scalar_t>(
      quaternionFromEulerZyx(vector3_t(angle, 0.0, 0.0)));
}

/**
 * The centroidal `state` turned by `angle` about the world z axis through the
 * origin: the normalized momentum's linear and angular blocks and the base
 * position turn, the orientation is left-multiplied, the joints stay.
 */
vector_t turnedAboutWorldZ(const CentroidalMpcRobotModel<scalar_t>& model,
                           const vector_t& state,
                           scalar_t angle) {
  const matrix3_t turn = yawRotation(angle);
  vector_t turned = state;
  // [m c_dot; L_G] / m, both in the world frame.
  turned.head<3>() = turn * state.head<3>();
  turned.segment<3>(3) = turn * state.segment<3>(3);
  model.setPosBaseRtWorldEwrtWorld(
      turned, turn * model.getPosBaseRtWorldEwrtWorld(state));
  model.setQuatBaseRtWorldXyzw(
      turned,
      quaternionProduct(quaternionFromEulerZyx(vector3_t(angle, 0.0, 0.0)),
                        model.getQuatBaseRtWorldXyzw(state)));
  return turned;
}

/** Expects `turned` to be `state` turned by `angle` (turnedAboutWorldZ()), its
 * orientation as a rotation. */
void expectTurned(const CentroidalMpcRobotModel<scalar_t>& model,
                  const vector_t& state,
                  const vector_t& turned,
                  scalar_t angle) {
  const vector_t expected = turnedAboutWorldZ(model, state, angle);
  const Eigen::Index orientation =
      static_cast<Eigen::Index>(model.getBaseOrientationStartIndex());
  const scalar_t scale = std::max(1.0, state.cwiseAbs().maxCoeff());
  for (Eigen::Index i = 0; i < state.size(); ++i) {
    if (i >= orientation && i < orientation + 4) continue;
    EXPECT_NEAR(turned(i), expected(i), kTolerance * scale)
        << "state entry " << i;
  }
  const matrix3_t rotation =
      quaternionRotationMatrix<scalar_t>(model.getQuatBaseRtWorldXyzw(turned));
  const matrix3_t expectedRotation =
      yawRotation(angle) *
      quaternionRotationMatrix<scalar_t>(model.getQuatBaseRtWorldXyzw(state));
  EXPECT_LE((rotation - expectedRotation).cwiseAbs().maxCoeff(), kTolerance)
      << "the orientation, as a rotation";
}

/** The kind of command a reference is computed from. */
enum class CommandKind {
  kVelocity,
  kPose,
};

class CentroidalCommandYawEquivariance
    : public ::testing::TestWithParam<ShippedCentroidalRobot> {};

TEST_P(CentroidalCommandYawEquivariance,
       TheCommandReferencesTurnWithTheRobotAcrossPi) {
  absl::StatusOr<LoadedCentroidalRobot> robot =
      loadShippedCentroidalRobot(GetParam(), /*centroidalModel=*/"");
  ASSERT_TRUE(robot.ok()) << robot.status();
  const CentroidalMpcRobotModel<scalar_t>& model = *robot->model;
  absl::StatusOr<std::unique_ptr<CentroidalMpcTargetTrajectoriesCalculator>>
      calculator = CentroidalMpcTargetTrajectoriesCalculator::Create(
          robot->config.reference, model, *robot->pinocchioInterface,
          robot->config.task.mpc.time_horizon);
  ASSERT_TRUE(calculator.ok()) << calculator.status();
  const absl::StatusOr<vector_t> joints =
      defaultJointStateOf(robot->config.reference, *robot->modelSettings);
  ASSERT_TRUE(joints.ok()) << joints.status();

  // A standing robot at heading 3.1, moving a little (so that the momentum's
  // turn is visible), and the same turned by 0.2.
  vector_t state =
      vector_t::Zero(static_cast<Eigen::Index>(model.getStateDim()));
  state.head<6>() << 0.05, -0.02, 0.01, 0.003, -0.002, 0.004;
  model.setPosBaseRtWorldEwrtWorld(state, vector3_t(1.0, 2.0, 0.75));
  model.setEulerzyxBaseRtWorld(state, vector3_t(3.1, 0.0, 0.0));
  model.setJointAngles(state, *joints);
  const vector_t turnedState = turnedAboutWorldZ(model, state, kTurn);
  ASSERT_LT(model.getEulerzyxBaseRtWorld(turnedState)(0), -2.9)
      << "the turned heading is across the cut";
  const vector4_t command(0.4, -0.2, 0.0, 0.5);
  for (const CommandKind kind : {CommandKind::kVelocity, CommandKind::kPose}) {
    SCOPED_TRACE(kind == CommandKind::kPose ? "pose command"
                                            : "velocity command");
    std::vector<TargetTrajectories> targets;
    for (const vector_t& start : {state, turnedState}) {
      (*calculator)->reset();
      targets.push_back(
          kind == CommandKind::kPose
              ? (*calculator)
                    ->poseCommandToTargetTrajectories(command, kInitTime, start)
              : (*calculator)
                    ->velocityCommandToTargetTrajectories(
                        command, kInitTime, start,
                        /*measuredMotion=*/BasePlanarMotion{}));
    }
    const TargetTrajectories& target = targets[0];
    const TargetTrajectories& turnedTarget = targets[1];
    // The knot times are the command's, up to the round-off of the turned
    // arithmetic (a pose command's subdivision).
    ASSERT_EQ(turnedTarget.timeTrajectory.size(), target.timeTrajectory.size());
    for (size_t k = 0; k < target.timeTrajectory.size(); ++k) {
      EXPECT_NEAR(turnedTarget.timeTrajectory[k], target.timeTrajectory[k],
                  1.0e-12);
    }
    ASSERT_EQ(turnedTarget.stateTrajectory.size(),
              target.stateTrajectory.size());
    for (size_t k = 0; k < target.stateTrajectory.size(); ++k) {
      SCOPED_TRACE(absl::StrCat("knot ", k));
      expectTurned(model, target.stateTrajectory[k],
                   turnedTarget.stateTrajectory[k], kTurn);
    }
    for (scalar_t time = kInitTime; time <= target.timeTrajectory.back();
         time += 0.05) {
      SCOPED_TRACE(absl::StrCat("reference at ", time));
      expectTurned(model, model.getDesiredState(target, time),
                   model.getDesiredState(turnedTarget, time), kTurn);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    ShippedRobots,
    CentroidalCommandYawEquivariance,
    ::testing::Values(ShippedCentroidalRobot::kDrcAtlas,
                      ShippedCentroidalRobot::kEngineAiSa01,
                      ShippedCentroidalRobot::kUnitreeG1,
                      ShippedCentroidalRobot::kUnitreeR1),
    shippedCentroidalRobotTestName);

}  // namespace
}  // namespace ocs2::humanoid
