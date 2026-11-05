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
#include <cstddef>
#include <numbers>

#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/TargetTrajectories.h"

#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_common_mpc/command/TargetTrajectoriesCalculatorBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "support/AtlasReferenceStack.h"

/*
 * The base pose command of the centroidal MPC on the quaternion layout (design
 * README section 5(c), TargetTrajectoriesCalculatorBase::subdivideYawKnots()):
 * a commanded turn of 270 degrees turns the reference +270 degrees, not the -90
 * degrees the shortest-path slerp between its two end knots would; consecutive
 * knots are at most a quarter turn apart; and the reference the solver reads
 * (the robot model's slerp of the knots) is, at every time of the MPC's node
 * grid and between, the linear interpolation of the two end poses in the Euler
 * angles that the pose command produced on the Euler layout, to 1e-12. On the
 * DRC Atlas references; builds no CppAD model.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kInitTime = 2.0;
/** [s] The grid the reference is read on: every node of the Atlas's 0.02 s grid
 * from kInitTime on, and the times between. */
constexpr scalar_t kGridStep = 0.01;

/** `angle` wrapped into [-pi, pi]. */
scalar_t wrapped(scalar_t angle) {
  return std::remainder(angle, 2.0 * std::numbers::pi);
}

/** The pose command's two end poses [x, y, z, yaw, pitch, roll], the yaw
 * unbounded, and the times they are reached at. */
struct EndPoses {
  scalar_t startTime = 0.0;
  vector6_t initPoseeulerzyxBaseRtWorld = vector6_t::Zero();
  scalar_t targetTime = 0.0;
  vector6_t finalPoseeulerzyxBaseRtWorld = vector6_t::Zero();
};

/** The linear interpolation of `ends` in every coordinate at `time`, held
 * before and after them: the Euler layout's reference. */
vector6_t interpolateLinearPoseeulerzyxBaseRtWorld(const EndPoses& ends,
                                                   scalar_t time) {
  const scalar_t share = std::clamp(
      (time - ends.startTime) / (ends.targetTime - ends.startTime), 0.0, 1.0);
  return ends.initPoseeulerzyxBaseRtWorld +
         share * (ends.finalPoseeulerzyxBaseRtWorld -
                  ends.initPoseeulerzyxBaseRtWorld);
}

/**
 * The pose command (dx, dy, dz in the heading frame [m], dyaw [deg]) from the
 * robot standing at (0.3, -0.2) with ZYX yaw `eulerzyxBaseRtWorldYaw`, its
 * target, and the end poses it moves between: from the robot's position and
 * heading, level, to the commanded displacement turned into the world, at the
 * height of the last knot, turned by the commanded yaw.
 */
struct PoseCommand {
  TargetTrajectories targets;
  EndPoses ends;
};

PoseCommand makePoseCommand(AtlasReferenceStack& stack,
                            scalar_t eulerzyxBaseRtWorldYaw,
                            const vector4_t& command) {
  const CentroidalMpcRobotModel<scalar_t>& model = stack.model();
  const vector_t initState = stack.standingAt(
      stack.initialState(), /*posBaseRtWorldEwrtWorldX=*/0.3,
      /*posBaseRtWorldEwrtWorldY=*/-0.2, eulerzyxBaseRtWorldYaw);
  PoseCommand poseCommand;
  poseCommand.targets =
      stack.targetCalculator().poseCommandToTargetTrajectories(
          command, kInitTime, initState);
  poseCommand.ends.startTime = poseCommand.targets.timeTrajectory.front();
  poseCommand.ends.targetTime = poseCommand.targets.timeTrajectory.back();
  const vector3_t posBaseRtWorldEwrtWorld =
      model.getPosBaseRtWorldEwrtWorld(initState);
  poseCommand.ends.initPoseeulerzyxBaseRtWorld << posBaseRtWorldEwrtWorld,
      model.getYawBaseRtWorld(initState), 0.0, 0.0;
  const vector2_t errPosBaseRtWorldEwrtWorldXy(
      std::cos(eulerzyxBaseRtWorldYaw) * command(0) -
          std::sin(eulerzyxBaseRtWorldYaw) * command(1),
      std::sin(eulerzyxBaseRtWorldYaw) * command(0) +
          std::cos(eulerzyxBaseRtWorldYaw) * command(1));
  poseCommand.ends.finalPoseeulerzyxBaseRtWorld
      << posBaseRtWorldEwrtWorld.head<2>() + errPosBaseRtWorldEwrtWorldXy,
      model.getPosBaseRtWorldEwrtWorld(
          poseCommand.targets.stateTrajectory.back())(2),
      poseCommand.ends.initPoseeulerzyxBaseRtWorld(3) +
          command(3) * std::numbers::pi / 180.0,
      0.0, 0.0;
  return poseCommand;
}

/**
 * Expects the knots of `poseCommand` to be at most a quarter turn apart and to
 * turn by the commanded yaw in total, and the reference at every time of the
 * grid to be the linear Euler interpolation of its end poses.
 */
void expectTheCommandedTurnOnTheLinearEulerReference(
    const AtlasReferenceStack& stack,
    const PoseCommand& poseCommand,
    scalar_t cmdErrYawBaseRtWorld) {
  const CentroidalMpcRobotModel<scalar_t>& model = stack.model();
  const TargetTrajectories& targets = poseCommand.targets;
  ASSERT_GE(targets.size(), 2U);
  EXPECT_EQ(targets.timeTrajectory.front(), kInitTime);
  // The parts of subdivideYawKnots(): ceil(|turn| / (pi / 2)) of them, one more
  // where the round-off of the turn the calculator adds to the heading puts a
  // turn of exactly k quarter turns an ulp above k.
  const size_t parts = static_cast<size_t>(std::ceil(
      std::abs(cmdErrYawBaseRtWorld) /
          TargetTrajectoriesCalculatorBase::kKnotMaxErrYawBaseRtWorld -
      1.0e-12));
  EXPECT_GE(targets.size(), parts + 1);
  EXPECT_LE(targets.size(), parts + 2);

  // Read back from the knots' quaternions, whose headings wrap: the steps
  // between knots at most a quarter turn apart unwrap without ambiguity, and
  // they add up to the commanded turn, in its direction.
  scalar_t errYawBaseRtWorld = 0.0;
  for (size_t i = 1; i < targets.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_GT(targets.timeTrajectory[i], targets.timeTrajectory[i - 1]);
    const scalar_t step =
        wrapped(model.getYawBaseRtWorld(targets.stateTrajectory[i]) -
                model.getYawBaseRtWorld(targets.stateTrajectory[i - 1]));
    EXPECT_LE(
        std::abs(step),
        TargetTrajectoriesCalculatorBase::kKnotMaxErrYawBaseRtWorld + 1.0e-12);
    EXPECT_GT(step * cmdErrYawBaseRtWorld, 0.0)
        << "every knot turns the commanded way";
    errYawBaseRtWorld += step;
  }
  EXPECT_NEAR(errYawBaseRtWorld, cmdErrYawBaseRtWorld, 1.0e-12);

  // The reference the solver reads, on the node grid and between, before the
  // start and after the target is reached.
  scalar_t maxNormErrYawBaseRtWorld = 0.0;
  scalar_t maxErrPosBaseRtWorldEwrtWorld = 0.0;
  // Of the pitch and roll; the yaw is maxNormErrYawBaseRtWorld's.
  scalar_t maxNormEulerzyxBaseRtWorld = 0.0;
  for (scalar_t time = kInitTime - 0.1;
       time <= poseCommand.ends.targetTime + 0.1 + 1.0e-9; time += kGridStep) {
    const vector_t reference = model.getDesiredState(targets, time);
    const vector6_t expected =
        interpolateLinearPoseeulerzyxBaseRtWorld(poseCommand.ends, time);
    const vector6_t read = model.getPoseeulerzyxBaseRtWorld(reference);
    maxNormErrYawBaseRtWorld = std::max(
        maxNormErrYawBaseRtWorld,
        std::abs(wrapped(model.getYawBaseRtWorld(reference) - expected(3))));
    maxErrPosBaseRtWorldEwrtWorld =
        std::max(maxErrPosBaseRtWorldEwrtWorld,
                 (read.head<3>() - expected.head<3>()).cwiseAbs().maxCoeff());
    maxNormEulerzyxBaseRtWorld = std::max(maxNormEulerzyxBaseRtWorld,
                                          read.tail<2>().cwiseAbs().maxCoeff());
    EXPECT_NEAR(model.getQuatBaseRtWorldXyzw(reference).norm(), 1.0, 1.0e-15)
        << "t = " << time;
  }
  EXPECT_LE(maxNormErrYawBaseRtWorld, 1.0e-12);
  EXPECT_LE(maxErrPosBaseRtWorldEwrtWorld, 1.0e-12);
  EXPECT_LE(maxNormEulerzyxBaseRtWorld, 1.0e-12);
}

TEST(PoseCommandSubdivision, ATurnOf270DegreesTurnsPlus270Degrees) {
  AtlasReferenceStack stack;
  const PoseCommand poseCommand = makePoseCommand(
      stack, /*eulerzyxBaseRtWorldYaw=*/2.8, vector4_t(0.4, 0.1, 0.0, 270.0));
  expectTheCommandedTurnOnTheLinearEulerReference(
      stack, poseCommand,
      /*cmdErrYawBaseRtWorld=*/1.5 * std::numbers::pi);
  // Positive control: the two end knots alone are a quarter turn apart the
  // other way round, which their slerp would take.
  const CentroidalMpcRobotModel<scalar_t>& model = stack.model();
  EXPECT_NEAR(
      wrapped(
          model.getYawBaseRtWorld(poseCommand.targets.stateTrajectory.back()) -
          model.getYawBaseRtWorld(poseCommand.targets.stateTrajectory.front())),
      -0.5 * std::numbers::pi, 1.0e-12);
}

TEST(PoseCommandSubdivision, ATurnOfMinus270DegreesTurnsMinus270Degrees) {
  AtlasReferenceStack stack;
  const PoseCommand poseCommand =
      makePoseCommand(stack, /*eulerzyxBaseRtWorldYaw=*/-3.0,
                      vector4_t(-0.2, 0.3, 0.0, -270.0));
  expectTheCommandedTurnOnTheLinearEulerReference(
      stack, poseCommand, /*cmdErrYawBaseRtWorld=*/-1.5 * std::numbers::pi);
}

TEST(PoseCommandSubdivision, ATurnOf720DegreesTurnsTwiceAround) {
  // Whose end knots are the same attitude: without the knots between them the
  // reference would not turn at all.
  AtlasReferenceStack stack;
  const PoseCommand poseCommand = makePoseCommand(
      stack, /*eulerzyxBaseRtWorldYaw=*/0.4, vector4_t(0.0, 0.0, 0.0, 720.0));
  expectTheCommandedTurnOnTheLinearEulerReference(
      stack, poseCommand,
      /*cmdErrYawBaseRtWorld=*/4.0 * std::numbers::pi);
  const CentroidalMpcRobotModel<scalar_t>& model = stack.model();
  EXPECT_NEAR(
      wrapped(
          model.getYawBaseRtWorld(poseCommand.targets.stateTrajectory.back()) -
          model.getYawBaseRtWorld(poseCommand.targets.stateTrajectory.front())),
      0.0, 1.0e-12);
}

TEST(PoseCommandSubdivision, ATurnOfAQuarterTurnOrLessIsTheTwoEndKnots) {
  AtlasReferenceStack stack;
  const PoseCommand poseCommand = makePoseCommand(
      stack, /*eulerzyxBaseRtWorldYaw=*/1.0, vector4_t(0.5, 0.0, 0.0, 60.0));
  EXPECT_EQ(poseCommand.targets.size(), 2U);
  expectTheCommandedTurnOnTheLinearEulerReference(
      stack, poseCommand,
      /*cmdErrYawBaseRtWorld=*/std::numbers::pi / 3.0);
}

}  // namespace
}  // namespace ocs2::humanoid
