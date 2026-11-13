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
#include <numbers>
#include <string>

#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/TargetTrajectories.h"

#include "legged_common_mpc/command/BasePlanarMotion.h"
#include "legged_common_mpc/common/ModelSettings.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/config/ConfigFiles.h"
#include "legged_common_mpc/config/reference/ReferenceFromConfig.h"
#include "legged_nmpc/legged_common_mpc/test/support/StatusTestMacros.h"
#include "legged_wb_mpc/command/WBMpcTargetTrajectoriesCalculator.h"
#include "legged_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "robot_core/ResourcePaths.h"

/*
 * The knots of the whole-body target calculator, written through the robot
 * model's layout-agnostic accessors (setPoseeulerzyxBaseRtWorld(),
 * setLinvelBaseRtWorldEwrtWorld(), setTurnAngvelBaseRtWorldEwrtBase()) and
 * read back through them: a velocity command's knots move the base at the
 * command in each knot's own heading frame and turn it about the world z axis
 * at the commanded yaw rate, consistently with how their poses advance along
 * the arc of that twist (design README section 2.9), with the joints at the
 * reference file's default posture and still; a pose command's knots hold
 * everything still. Builds no Pinocchio or CppAD model.
 */
namespace ocs2::legged {
namespace {

constexpr scalar_t kHorizon = 1.0;   // [s]
constexpr scalar_t kInitTime = 2.0;  // [s]
constexpr scalar_t kTolerance = 1.0e-12;
// The share of the command that reaches the knots: all of it, since the
// calculator does not filter the command (ProceduralMpcMotionManager does, on
// the solver time).
constexpr scalar_t kFirstCommandFactor = 1.0;
// The intermediate knot's share of the horizon.
constexpr scalar_t kIntermediateShare = 0.7;

/** The G1 whole-body model and target calculator of the shipped files. Not
 * thread-safe. */
class G1Calculator {
 public:
  G1Calculator() {
    // LINT.IfChange(robot_files)
    const std::string taskFile = checkedValue(robot::resolveResourcePath(
        "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto"));
    const std::string urdfFile = checkedValue(robot::resolveResourcePath(
        "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"));
    const std::string referenceFile = checkedValue(
        robot::resolveResourcePath("robot_models/unitree_g1/g1_wb_mpc/config/"
                                   "command/reference.textproto"));
    // LINT.ThenChange(//legged_nmpc/legged_wb_mpc/BUILD.bazel:target_knots_test_data)
    modelSettings_ =
        std::make_unique<ModelSettings>(checkedValue(ModelSettings::Create(
            taskFile, urdfFile, "wb_mpc_", /*verbose=*/false)));
    model_ = std::make_unique<WBAccelMpcRobotModel<scalar_t>>(*modelSettings_);
    calculator_ = checkedValue(WBMpcTargetTrajectoriesCalculator::Create(
        referenceFile, *model_, kHorizon));
    defaultJointState_ = checkedValue(defaultJointStateFromConfig(
        checkedValue(loadReferenceFile(referenceFile)),
        modelSettings_->mpcModelJointNames, modelSettings_->fixedJointNames));
  }
  ~G1Calculator() = default;
  G1Calculator(const G1Calculator&) = delete;
  G1Calculator& operator=(const G1Calculator&) = delete;

  const WBAccelMpcRobotModel<scalar_t>& model() const { return *model_; }
  WBMpcTargetTrajectoriesCalculator& calculator() { return *calculator_; }
  const vector_t& defaultJointState() const { return defaultJointState_; }

  /** The robot at rest, level, at the base pose `poseeulerzyxBaseRtWorld` ([x,
   * y, z, yaw, pitch, roll]), the joints at zero. */
  vector_t restingState(const vector6_t& poseeulerzyxBaseRtWorld) const {
    vector_t state = model_->getNeutralState();
    model_->setPoseeulerzyxBaseRtWorld(state, poseeulerzyxBaseRtWorld);
    return state;
  }

 private:
  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<WBAccelMpcRobotModel<scalar_t>> model_;
  std::unique_ptr<WBMpcTargetTrajectoriesCalculator> calculator_;
  vector_t defaultJointState_;
};

/** Expects the joints of `knot` at `jointAngles` and still. */
void expectJointsStillAt(const WBAccelMpcRobotModel<scalar_t>& model,
                         const vector_t& knot,
                         const vector_t& jointAngles) {
  EXPECT_TRUE(model.getJointAngles(knot) == jointAngles);
  EXPECT_TRUE(
      model.getJointVelocities(knot, vector_t::Zero(model.getInputDim()))
          .isZero(0.0));
}

TEST(
    WBMpcTargetKnots,
    AVelocityCommandsKnotsMoveAndTurnTheBaseAtTheFilteredCommandWithTheJointsStill) {
  G1Calculator g1;
  const WBAccelMpcRobotModel<scalar_t>& model = g1.model();
  // Turned past the heading of +-pi/2, so that the heading-frame command and
  // the world frame differ in sign.
  const scalar_t initYawBaseRtWorld = 2.5;
  const vector_t initState = g1.restingState(
      (vector6_t() << 0.3, -0.2, 0.75, initYawBaseRtWorld, 0.0, 0.0)
          .finished());
  // The command the calculator used to see after its per-call filter (1 - 0.8
  // of it), so that the knots stay unsubdivided.
  const vector4_t command(
      0.08, 0.02, 0.0,
      0.1);  // [v_x, v_y in the heading frame, dz, yaw rate]
  const TargetTrajectories targets =
      g1.calculator().velocityCommandToTargetTrajectories(
          command, kInitTime, initState, /*measuredMotion=*/BasePlanarMotion{});

  ASSERT_EQ(targets.timeTrajectory.size(), 3);
  ASSERT_EQ(targets.stateTrajectory.size(), 3);
  EXPECT_DOUBLE_EQ(targets.timeTrajectory[1],
                   kInitTime + kIntermediateShare * kHorizon);
  EXPECT_DOUBLE_EQ(targets.timeTrajectory[2], kInitTime + kHorizon);

  const scalar_t cmdLinvelBaseRtWorldEwrtHeadingX =
      kFirstCommandFactor * command(0);
  const scalar_t cmdLinvelBaseRtWorldEwrtHeadingY =
      kFirstCommandFactor * command(1);
  const scalar_t cmdRateYawBaseRtWorld = kFirstCommandFactor * command(3);
  for (size_t i = 0; i < targets.stateTrajectory.size(); ++i) {
    SCOPED_TRACE(i);
    const vector_t& knot = targets.stateTrajectory[i];
    ASSERT_EQ(static_cast<size_t>(knot.size()), model.getStateDim());
    // The command in the knot's own heading frame (design README section 2.9):
    // the commanded velocity turned by the knot's yaw.
    const scalar_t refEulerzyxBaseRtWorldYaw =
        model.getEulerzyxBaseRtWorld(knot)(0);
    const vector3_t cmdLinvelBaseRtWorldEwrtWorld(
        std::cos(refEulerzyxBaseRtWorldYaw) * cmdLinvelBaseRtWorldEwrtHeadingX -
            std::sin(refEulerzyxBaseRtWorldYaw) *
                cmdLinvelBaseRtWorldEwrtHeadingY,
        std::sin(refEulerzyxBaseRtWorldYaw) * cmdLinvelBaseRtWorldEwrtHeadingX +
            std::cos(refEulerzyxBaseRtWorldYaw) *
                cmdLinvelBaseRtWorldEwrtHeadingY,
        0.0);
    EXPECT_LT((model.getLinvelBaseRtWorldEwrtWorld(knot) -
               cmdLinvelBaseRtWorldEwrtWorld)
                  .norm(),
              kTolerance);
    EXPECT_EQ(model.getLinvelBaseRtWorldEwrtWorld(knot)(2), 0.0)
        << "the base moves in the plane";
    // Level knots: a turn about the world z axis is one about the base's own z
    // axis.
    const vector3_t eulerzyxBaseRtWorld = model.getEulerzyxBaseRtWorld(knot);
    EXPECT_EQ(eulerzyxBaseRtWorld(1), 0.0) << "pitch";
    EXPECT_EQ(eulerzyxBaseRtWorld(2), 0.0) << "roll";
    EXPECT_LT((model.getAngvelBaseRtWorldEwrtBase(knot) -
               vector3_t(0.0, 0.0, cmdRateYawBaseRtWorld))
                  .norm(),
              kTolerance);
    // The base velocity in the tuning order of state_weights (x, y, z; then
    // about z, y, x).
    const vector6_t tuningOrderBaseRates = model.getTuningLayoutBaseRates(knot);
    EXPECT_LT(std::abs(tuningOrderBaseRates(3) - cmdRateYawBaseRtWorld),
              kTolerance);
    EXPECT_EQ(tuningOrderBaseRates(4), 0.0);
    EXPECT_EQ(tuningOrderBaseRates(5), 0.0);
    expectJointsStillAt(model, knot, g1.defaultJointState());
  }

  // From the intermediate knot on, the base moves along the arc of the knots'
  // twist: its chord, the last knot's planar base position minus the
  // intermediate one's (errPosBaseRtWorldEwrtWorldXy), is
  // T sinc(theta / 2) R_z(psi_1 + theta / 2) v, theta = omega T, the velocity v
  // of the intermediate knot.
  const scalar_t finalStretch =
      targets.timeTrajectory[2] - targets.timeTrajectory[1];
  const vector6_t refPoseeulerzyxBaseRtWorld =
      model.getPoseeulerzyxBaseRtWorld(targets.stateTrajectory[1]);
  const vector6_t finalRefPoseeulerzyxBaseRtWorld =
      model.getPoseeulerzyxBaseRtWorld(targets.stateTrajectory[2]);
  const scalar_t theta = cmdRateYawBaseRtWorld * finalStretch;
  const scalar_t chordYawBaseRtWorld =
      refPoseeulerzyxBaseRtWorld(3) + 0.5 * theta;
  const scalar_t chordLength =
      finalStretch * std::sin(0.5 * theta) / (0.5 * theta);
  const vector2_t errPosBaseRtWorldEwrtWorldXy(
      chordLength *
          (std::cos(chordYawBaseRtWorld) * cmdLinvelBaseRtWorldEwrtHeadingX -
           std::sin(chordYawBaseRtWorld) * cmdLinvelBaseRtWorldEwrtHeadingY),
      chordLength *
          (std::sin(chordYawBaseRtWorld) * cmdLinvelBaseRtWorldEwrtHeadingX +
           std::cos(chordYawBaseRtWorld) * cmdLinvelBaseRtWorldEwrtHeadingY));
  EXPECT_LT(
      (finalRefPoseeulerzyxBaseRtWorld.head<2>() -
       refPoseeulerzyxBaseRtWorld.head<2>() - errPosBaseRtWorldEwrtWorldXy)
          .norm(),
      kTolerance);
  EXPECT_LT(std::abs(finalRefPoseeulerzyxBaseRtWorld(3) -
                     refPoseeulerzyxBaseRtWorld(3) -
                     finalStretch * cmdRateYawBaseRtWorld),
            kTolerance);
}

TEST(WBMpcTargetKnots, APoseCommandsKnotsHoldTheBaseAndTheJointsStill) {
  G1Calculator g1;
  const WBAccelMpcRobotModel<scalar_t>& model = g1.model();
  const vector6_t initPoseeulerzyxBaseRtWorld =
      (vector6_t() << 0.3, -0.2, 0.75, 2.5, 0.0, 0.0).finished();
  const vector_t initState = g1.restingState(initPoseeulerzyxBaseRtWorld);
  const vector4_t command(
      0.5, 0.2, 0.0,
      -45.0);  // [dx, dy in the heading frame, dz, dyaw in degrees]
  const TargetTrajectories targets =
      g1.calculator().poseCommandToTargetTrajectories(command, kInitTime,
                                                      initState);

  ASSERT_EQ(targets.stateTrajectory.size(), 2);
  // The first knot is where the robot stands, level.
  EXPECT_LT((model.getPoseeulerzyxBaseRtWorld(targets.stateTrajectory[0]) -
             initPoseeulerzyxBaseRtWorld)
                .norm(),
            kTolerance);
  EXPECT_LT(
      std::abs(model.getEulerzyxBaseRtWorld(targets.stateTrajectory[1])(0) -
               (initPoseeulerzyxBaseRtWorld(3) - std::numbers::pi / 4.0)),
      kTolerance);
  for (size_t i = 0; i < targets.stateTrajectory.size(); ++i) {
    SCOPED_TRACE(i);
    const vector_t& knot = targets.stateTrajectory[i];
    ASSERT_EQ(static_cast<size_t>(knot.size()), model.getStateDim());
    EXPECT_TRUE(model.getLinvelBaseRtWorldEwrtWorld(knot).isZero(0.0));
    EXPECT_TRUE(model.getAngvelBaseRtWorldEwrtBase(knot).isZero(0.0));
    expectJointsStillAt(model, knot, g1.defaultJointState());
  }
}

}  // namespace
}  // namespace ocs2::legged
