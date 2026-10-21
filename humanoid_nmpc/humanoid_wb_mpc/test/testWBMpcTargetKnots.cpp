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
#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_core/reference/TargetTrajectories.h"

#include "humanoid_common_mpc/command/BasePlanarMotion.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/reference/ReferenceFromConfig.h"
#include "humanoid_mpc_config/reference_file.nproto.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"
#include "humanoid_wb_mpc/command/WBMpcTargetTrajectoriesCalculator.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "robot_core/ResourcePaths.h"

/*
 * The knots of the whole-body target calculator, written through the robot model's layout-agnostic accessors
 * (setBasePoseEulerZyx(), setBaseComLinearVelocity(), setBaseAngularVelocityFromWorldYawRate()) and read back through
 * them: a velocity command's knots move the base at the command in each knot's own heading frame and turn it
 * about the world z axis at the commanded yaw rate, consistently with how their poses advance along the arc of that
 * twist (design README section 2.9), with the joints at the reference file's default posture and still; a pose command's knots hold
 * everything still. Builds no Pinocchio or CppAD model.
 */
namespace ocs2::humanoid {
namespace {

constexpr scalar_t kHorizon = 1.0;   // [s]
constexpr scalar_t kInitTime = 2.0;  // [s]
constexpr scalar_t kTolerance = 1.0e-12;
// The share of the command that reaches the knots: all of it, since the calculator does not filter the command
// (ProceduralMpcMotionManager does, on the solver time).
constexpr scalar_t kFirstCommandFactor = 1.0;
// The intermediate knot's share of the horizon.
constexpr scalar_t kIntermediateShare = 0.7;

/** The G1 whole-body model and target calculator of the shipped files. Not thread-safe. */
class G1Calculator {
 public:
  G1Calculator() {
    // LINT.IfChange(robot_files)
    const std::string taskFile = checkedValue(robot::resolveResourcePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto"));
    const std::string urdfFile = checkedValue(robot::resolveResourcePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"));
    const std::string referenceFile =
        checkedValue(robot::resolveResourcePath("robot_models/unitree_g1/g1_wb_mpc/config/command/reference.textproto"));
    // LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc/BUILD.bazel:target_knots_test_data)
    modelSettings_ = std::make_unique<ModelSettings>(checkedValue(ModelSettings::Create(taskFile, urdfFile, "wb_mpc_", /*verbose=*/false)));
    model_ = std::make_unique<WBAccelMpcRobotModel<scalar_t>>(*modelSettings_);
    calculator_ = checkedValue(WBMpcTargetTrajectoriesCalculator::Create(referenceFile, *model_, kHorizon));
    defaultJointState_ = checkedValue(defaultJointStateFromConfig(checkedValue(loadReferenceFile(referenceFile)),
                                                                  modelSettings_->mpcModelJointNames, modelSettings_->fixedJointNames));
  }
  ~G1Calculator() = default;
  G1Calculator(const G1Calculator&) = delete;
  G1Calculator& operator=(const G1Calculator&) = delete;

  const WBAccelMpcRobotModel<scalar_t>& model() const { return *model_; }
  WBMpcTargetTrajectoriesCalculator& calculator() { return *calculator_; }
  const vector_t& defaultJointState() const { return defaultJointState_; }

  /** The robot at rest, level, at the base pose `basePoseEulerZyx` ([x, y, z, yaw, pitch, roll]), the joints at zero. */
  vector_t restingState(const vector6_t& basePoseEulerZyx) const {
    vector_t state = model_->getNeutralState();
    model_->setBasePoseEulerZyx(state, basePoseEulerZyx);
    return state;
  }

 private:
  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<WBAccelMpcRobotModel<scalar_t>> model_;
  std::unique_ptr<WBMpcTargetTrajectoriesCalculator> calculator_;
  vector_t defaultJointState_;
};

/** Expects the joints of `knot` at `jointAngles` and still. */
void expectJointsStillAt(const WBAccelMpcRobotModel<scalar_t>& model, const vector_t& knot, const vector_t& jointAngles) {
  EXPECT_TRUE(model.getJointAngles(knot) == jointAngles);
  EXPECT_TRUE(model.getJointVelocities(knot, vector_t::Zero(model.getInputDim())).isZero(0.0));
}

TEST(WBMpcTargetKnots, AVelocityCommandsKnotsMoveAndTurnTheBaseAtTheFilteredCommandWithTheJointsStill) {
  G1Calculator g1;
  const WBAccelMpcRobotModel<scalar_t>& model = g1.model();
  // Turned past the heading of +-pi/2, so that the pelvis-frame command and the world frame differ in sign.
  const scalar_t heading = 2.5;
  const vector_t initState = g1.restingState((vector6_t() << 0.3, -0.2, 0.75, heading, 0.0, 0.0).finished());
  // The command the calculator used to see after its per-call filter (1 - 0.8 of it), so that the knots stay unsubdivided.
  const vector4_t command(0.08, 0.02, 0.0, 0.1);  // [v_x, v_y in the pelvis frame, dz, yaw rate]
  const TargetTrajectories targets =
      g1.calculator().commandedVelocityToTargetTrajectories(command, kInitTime, initState, /*measuredMotion=*/BasePlanarMotion{});

  ASSERT_EQ(targets.timeTrajectory.size(), 3);
  ASSERT_EQ(targets.stateTrajectory.size(), 3);
  EXPECT_DOUBLE_EQ(targets.timeTrajectory[1], kInitTime + kIntermediateShare * kHorizon);
  EXPECT_DOUBLE_EQ(targets.timeTrajectory[2], kInitTime + kHorizon);

  const scalar_t forward = kFirstCommandFactor * command(0);
  const scalar_t left = kFirstCommandFactor * command(1);
  const scalar_t yawRate = kFirstCommandFactor * command(3);
  for (size_t i = 0; i < targets.stateTrajectory.size(); ++i) {
    SCOPED_TRACE(i);
    const vector_t& knot = targets.stateTrajectory[i];
    ASSERT_EQ(static_cast<size_t>(knot.size()), model.getStateDim());
    // The command in the knot's own heading frame (design README section 2.9): (forward, left) turned by the knot's yaw.
    const scalar_t knotYaw = model.getBaseOrientationEulerZyx(knot)(0);
    const vector3_t linearVelocityWorld(std::cos(knotYaw) * forward - std::sin(knotYaw) * left,
                                        std::sin(knotYaw) * forward + std::cos(knotYaw) * left, 0.0);
    EXPECT_LT((model.getBaseComLinearVelocity(knot) - linearVelocityWorld).norm(), kTolerance);
    EXPECT_EQ(model.getBaseComLinearVelocity(knot)(2), 0.0) << "the base moves in the plane";
    // Level knots: a turn about the world z axis is one about the base's own z axis.
    const vector3_t eulerZyx = model.getBaseOrientationEulerZyx(knot);
    EXPECT_EQ(eulerZyx(1), 0.0) << "pitch";
    EXPECT_EQ(eulerZyx(2), 0.0) << "roll";
    EXPECT_LT((model.getBaseAngularVelocityLocal(knot) - vector3_t(0.0, 0.0, yawRate)).norm(), kTolerance);
    // The base velocity in the tuning order of state_weights (x, y, z; then about z, y, x).
    const vector6_t tuningOrderVelocity = model.getBaseComVelocity(knot);
    EXPECT_LT(std::abs(tuningOrderVelocity(3) - yawRate), kTolerance);
    EXPECT_EQ(tuningOrderVelocity(4), 0.0);
    EXPECT_EQ(tuningOrderVelocity(5), 0.0);
    expectJointsStillAt(model, knot, g1.defaultJointState());
  }

  // From the intermediate knot on, the base moves along the arc of the knots' twist: its chord is T sinc(theta / 2)
  // R_z(psi_1 + theta / 2) v, theta = omega T, the velocity v of the intermediate knot.
  const scalar_t finalStretch = targets.timeTrajectory[2] - targets.timeTrajectory[1];
  const vector6_t intermediate = model.getBasePoseEulerZyx(targets.stateTrajectory[1]);
  const vector6_t last = model.getBasePoseEulerZyx(targets.stateTrajectory[2]);
  const scalar_t theta = yawRate * finalStretch;
  const scalar_t chordHeading = intermediate(3) + 0.5 * theta;
  const scalar_t chordLength = finalStretch * std::sin(0.5 * theta) / (0.5 * theta);
  const vector2_t chord(chordLength * (std::cos(chordHeading) * forward - std::sin(chordHeading) * left),
                        chordLength * (std::sin(chordHeading) * forward + std::cos(chordHeading) * left));
  EXPECT_LT((last.head<2>() - intermediate.head<2>() - chord).norm(), kTolerance);
  EXPECT_LT(std::abs(last(3) - intermediate(3) - finalStretch * yawRate), kTolerance);
}

TEST(WBMpcTargetKnots, APoseCommandsKnotsHoldTheBaseAndTheJointsStill) {
  G1Calculator g1;
  const WBAccelMpcRobotModel<scalar_t>& model = g1.model();
  const vector6_t startPose = (vector6_t() << 0.3, -0.2, 0.75, 2.5, 0.0, 0.0).finished();
  const vector_t initState = g1.restingState(startPose);
  const vector4_t command(0.5, 0.2, 0.0, -45.0);  // [dx, dy in the pelvis frame, dz, dyaw in degrees]
  const TargetTrajectories targets = g1.calculator().commandedPositionToTargetTrajectories(command, kInitTime, initState);

  ASSERT_EQ(targets.stateTrajectory.size(), 2);
  // The first knot is where the robot stands, level.
  EXPECT_LT((model.getBasePoseEulerZyx(targets.stateTrajectory[0]) - startPose).norm(), kTolerance);
  EXPECT_LT(std::abs(model.getBaseOrientationEulerZyx(targets.stateTrajectory[1])(0) - (startPose(3) - M_PI / 4.0)), kTolerance);
  for (size_t i = 0; i < targets.stateTrajectory.size(); ++i) {
    SCOPED_TRACE(i);
    const vector_t& knot = targets.stateTrajectory[i];
    ASSERT_EQ(static_cast<size_t>(knot.size()), model.getStateDim());
    EXPECT_TRUE(model.getBaseComLinearVelocity(knot).isZero(0.0));
    EXPECT_TRUE(model.getBaseAngularVelocityLocal(knot).isZero(0.0));
    expectJointsStillAt(model, knot, g1.defaultJointState());
  }
}

}  // namespace
}  // namespace ocs2::humanoid
