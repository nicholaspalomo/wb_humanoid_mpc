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
#include <vector>

#include "gtest/gtest.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/centroidal.hpp"

#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_common_mpc/command/BasePlanarMotion.h"
#include "humanoid_common_mpc/command/TargetTrajectoriesCalculatorBase.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "support/AtlasReferenceStack.h"

/*
 * The knots of the centroidal MPC's velocity command (TargetTrajectoriesCalculatorBase's class comment): over the first
 * 70 % of the horizon the base moves at the mean of the measured and the commanded planar twist (design defect D1b), the
 * measurement being the robot's stride-mean motion handed in (design README section 2.9), then at the command; every
 * knot lies on the arc of its twist, checked against an RK4 integration of the unicycle; the momentum of every
 * knot is the command in that knot's heading frame; between the knots the interpolated reference stays within
 * the tolerances of the arc; and a fast turn is cut into knots at most pi / 2 apart, any measured spin included. On the
 * DRC Atlas references; builds no CppAD model.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kInitTime = 2.0;
// The share of the horizon of the first segment, and the share of the command the filter lets through on the first call
// after construction (1 - 0.8).
constexpr scalar_t kFirstSegmentShare = 0.7;
constexpr scalar_t kFirstCommandFactor = 0.2;

/** R_z(angle) v. */
vector2_t rotated(scalar_t angle, const vector2_t& vector) {
  return vector2_t(std::cos(angle) * vector(0) - std::sin(angle) * vector(1), std::sin(angle) * vector(0) + std::cos(angle) * vector(1));
}

/** A planar pose: the position of the base in the ground plane and its heading. Passive data. */
struct PlanarPose {
  vector2_t position = vector2_t::Zero();
  scalar_t heading = 0.0;
};

/**
 * `start` moved for `duration` by RK4 (2000 steps) of the unicycle dp/dt = R_z(psi) v, dpsi/dt = omega, at the
 * constant body planar twist (`velocityInHeadingFrame`, `yawRate`): independent of the calculator's closed form.
 */
PlanarPose integrated(const PlanarPose& start, const vector2_t& velocityInHeadingFrame, scalar_t yawRate, scalar_t duration) {
  constexpr int kSteps = 2000;
  const scalar_t step = duration / kSteps;
  PlanarPose pose = start;
  for (int i = 0; i < kSteps; ++i) {
    const vector2_t k1 = rotated(pose.heading, velocityInHeadingFrame);
    const vector2_t k23 = rotated(pose.heading + 0.5 * step * yawRate, velocityInHeadingFrame);
    const vector2_t k4 = rotated(pose.heading + step * yawRate, velocityInHeadingFrame);
    pose.position += step / 6.0 * (k1 + 4.0 * k23 + k4);
    pose.heading += step * yawRate;
  }
  return pose;
}

/** The planar pose of a knot: its heading read from the knot's quaternion, in (-pi, pi]. */
PlanarPose planarPoseOf(const MpcRobotModelBase<scalar_t>& model, const vector_t& knot) {
  const vector6_t pose = model.getBasePoseEulerZyx(knot);
  return PlanarPose{.position = vector2_t(pose(0), pose(1)), .heading = pose(3)};
}

/** `angle` wrapped into [-pi, pi]: the difference of two headings read from quaternions. */
scalar_t wrapped(scalar_t angle) {
  return std::remainder(angle, 2.0 * M_PI);
}

/**
 * The parts velocityCommandKnots() cuts a segment into: knotSubdivisionCount() of the segment that moves at
 * `velocityInHeadingFrame` and turns at `yawRate` for `duration`, its knots carrying the velocity target `targetVelocity`.
 */
size_t segmentParts(const vector2_t& velocityInHeadingFrame, scalar_t yawRate, scalar_t duration, const vector2_t& targetVelocity) {
  return TargetTrajectoriesCalculatorBase::knotSubdivisionCount(TargetTrajectoriesCalculatorBase::KnotSegmentExtent{
      .yawChange = yawRate * duration, .arcLength = velocityInHeadingFrame.norm() * duration, .targetSpeed = targetVelocity.norm()});
}

/** The initial state moved to the planar pose `pose`, still. */
vector_t standingAt(const AtlasReferenceStack& stack, const PlanarPose& pose) {
  return stack.standingAt(stack.initialState(), pose.position(0), pose.position(1), pose.heading);
}

/** The knots' planar poses expected from `start`: the mean twist for 0.7 T, then the command (`command` in the heading frame). */
std::vector<PlanarPose> expectedEnds(const PlanarPose& start,
                                     const BasePlanarMotion& measured,
                                     const vector2_t& commandVelocity,
                                     scalar_t commandYawRate,
                                     scalar_t horizon) {
  const PlanarPose intermediate = integrated(start, 0.5 * (measured.velocityInHeadingFrame + commandVelocity),
                                             0.5 * (measured.headingRate + commandYawRate), kFirstSegmentShare * horizon);
  const PlanarPose last = integrated(intermediate, commandVelocity, commandYawRate, (1.0 - kFirstSegmentShare) * horizon);
  return {start, intermediate, last};
}

TEST(IntermediateKnotMeasuredRate, OnAStraightLineTheFirstSegmentIsTheTrapezoidOfTheMeasuredAndTheCommandedVelocity) {
  AtlasReferenceStack stack;
  // Heading 1.0, the robot walking at (0.2, -0.1) m/s in its heading frame, not turning; the command not turning either.
  const PlanarPose start{.position = vector2_t(0.5, -0.3), .heading = 1.0};
  const BasePlanarMotion measured{.velocityInHeadingFrame = vector2_t(0.2, -0.1), .headingRate = 0.0};
  const vector4_t command(0.3, 0.1, 0.0, 0.0);
  const TargetTrajectories targets =
      stack.targetCalculator().commandedVelocityToTargetTrajectories(command, kInitTime, standingAt(stack, start), measured);
  ASSERT_EQ(targets.stateTrajectory.size(), 3);
  // p(0.7 T) - p0 = 0.7 T (v_m + v_c) / 2 in the world: the integral of a velocity ramping linearly from one to the other.
  const vector2_t commandVelocity = kFirstCommandFactor * vector2_t(command(0), command(1));
  const vector2_t expected =
      kFirstSegmentShare * stack.horizon() * rotated(/*angle=*/1.0, 0.5 * (measured.velocityInHeadingFrame + commandVelocity));
  const vector2_t moved = planarPoseOf(stack.model(), targets.stateTrajectory[1]).position - start.position;
  EXPECT_LT((moved - expected).norm(), 1.0e-14) << moved.transpose();
  EXPECT_NEAR(targets.timeTrajectory[1] - kInitTime, kFirstSegmentShare * stack.horizon(), 1.0e-15);
}

TEST(IntermediateKnotMeasuredRate, TheKnotsLieOnTheArcOfTheirTwist) {
  AtlasReferenceStack stack;
  const PlanarPose start{.position = vector2_t(-1.0, 2.0), .heading = -2.5};
  const BasePlanarMotion measured{.velocityInHeadingFrame = vector2_t(0.25, -0.05), .headingRate = 0.6};
  const vector4_t command(0.5, 0.2, 0.0, 0.9);
  const TargetTrajectories targets =
      stack.targetCalculator().commandedVelocityToTargetTrajectories(command, kInitTime, standingAt(stack, start), measured);
  ASSERT_EQ(targets.stateTrajectory.size(), 3);
  const std::vector<PlanarPose> expected = expectedEnds(start, measured, kFirstCommandFactor * vector2_t(command(0), command(1)),
                                                        kFirstCommandFactor * command(3), stack.horizon());
  for (size_t i = 0; i < 3; ++i) {
    SCOPED_TRACE(i);
    const PlanarPose knot = planarPoseOf(stack.model(), targets.stateTrajectory[i]);
    EXPECT_LT((knot.position - expected[i].position).norm(), 1.0e-12) << knot.position.transpose();
    EXPECT_NEAR(wrapped(knot.heading - expected[i].heading), 0.0, 1.0e-12);
  }
  // The straight chord at the starting heading the knots used to take is off the arc by a few centimeters.
  const vector2_t chord =
      start.position +
      kFirstSegmentShare * stack.horizon() *
          rotated(start.heading, 0.5 * (measured.velocityInHeadingFrame + kFirstCommandFactor * vector2_t(command(0), command(1))));
  EXPECT_GT((chord - expected[1].position).norm(), 1.0e-2);
}

TEST(IntermediateKnotMeasuredRate, EveryKnotsMomentumIsTheCommandInItsHeadingFrame) {
  AtlasReferenceStack stack;
  const PlanarPose start{.position = vector2_t(0.0, 0.0), .heading = 0.4};
  const BasePlanarMotion measured{.velocityInHeadingFrame = vector2_t(0.1, 0.0), .headingRate = 0.8};
  const vector4_t command(0.4, -0.2, 0.0, 1.0);
  const vector_t initState = standingAt(stack, start);
  const TargetTrajectories targets =
      stack.targetCalculator().commandedVelocityToTargetTrajectories(command, kInitTime, initState, measured);
  ASSERT_EQ(targets.stateTrajectory.size(), 3);
  // The momentum of the measured body turning at the command: L = I_G e_z omega, I_G the composite inertia of ccrba().
  const PinocchioInterface::Model& pinocchioModel = stack.pinocchioInterface().getModel();
  PinocchioInterface::Data data(pinocchioModel);
  pinocchio::ccrba(pinocchioModel, data, stack.model().getGeneralizedCoordinates(initState), vector_t::Zero(pinocchioModel.nv));
  const matrix3_t lockedInertia = data.Ig.inertia().matrix();
  const scalar_t mass = stack.centroidalModelInfo().robotMass;
  const scalar_t yawRate = kFirstCommandFactor * command(3);
  const vector3_t angularMomentum = lockedInertia.col(2) * yawRate;
  ASSERT_GT(angularMomentum.head<2>().norm(), 1.0e-4 * std::abs(angularMomentum(2))) << "the products of inertia must show the turn";
  const vector2_t commandVelocity = kFirstCommandFactor * vector2_t(command(0), command(1));
  for (size_t i = 0; i < 3; ++i) {
    SCOPED_TRACE(i);
    const vector_t& knot = targets.stateTrajectory[i];
    const scalar_t heading = planarPoseOf(stack.model(), knot).heading;
    // The CoM velocity: the command turned into the world at the knot's own heading.
    EXPECT_LT((vector2_t(knot.head<2>()) - rotated(heading, commandVelocity)).norm(), 1.0e-14);
    EXPECT_EQ(knot(2), 0.0);
    // The angular momentum of the measured body turned about the world z axis to the knot's heading: R_z L.
    const vector2_t turnedMomentum = rotated(heading - start.heading, vector2_t(angularMomentum.head<2>()));
    EXPECT_LT((mass * vector3_t(knot.segment<3>(3)) - vector3_t(turnedMomentum(0), turnedMomentum(1), angularMomentum(2))).norm(),
              1.0e-12 * angularMomentum.norm());
    // The commanded yaw rate reads back from every knot's momentum alike (D3).
    EXPECT_NEAR(stack.model().getTargetPlanarCommand(knot, lockedInertia(2, 2))(2), yawRate, 1.0e-12);
  }
  EXPECT_GT(std::abs(planarPoseOf(stack.model(), targets.stateTrajectory[2]).heading - start.heading), 0.3) << "the knots must turn";
}

TEST(IntermediateKnotMeasuredRate, AFastMeasuredTurnIsCutIntoKnotsAtMostAQuarterTurnApart) {
  AtlasReferenceStack stack;
  // A spin of 9 rad/s (a fall): the first segment turns by 0.7 T (9 + 0.2) / 2 = 3.22 rad, past pi, where the
  // shortest-path slerp of a quaternion layout would turn the reference the other way. Cut into at least three parts for
  // the yaw spacing, six here, where the chord of the turning velocity target needs more.
  const PlanarPose start{.position = vector2_t(0.2, 0.1), .heading = 0.3};
  const BasePlanarMotion measured{.velocityInHeadingFrame = vector2_t(0.3, 0.1), .headingRate = 9.0};
  const vector4_t command(0.5, 0.0, 0.0, 1.0);
  const TargetTrajectories targets =
      stack.targetCalculator().commandedVelocityToTargetTrajectories(command, kInitTime, standingAt(stack, start), measured);
  const scalar_t horizon = stack.horizon();
  const vector2_t commandVelocity = kFirstCommandFactor * vector2_t(command(0), command(1));
  const scalar_t commandYawRate = kFirstCommandFactor * command(3);
  const vector2_t meanVelocity = 0.5 * (measured.velocityInHeadingFrame + commandVelocity);
  const scalar_t meanYawRate = 0.5 * (measured.headingRate + commandYawRate);
  ASSERT_GT(kFirstSegmentShare * horizon * meanYawRate, M_PI);
  const size_t firstParts = segmentParts(meanVelocity, meanYawRate, kFirstSegmentShare * horizon, commandVelocity);
  const size_t secondParts = segmentParts(commandVelocity, commandYawRate, (1.0 - kFirstSegmentShare) * horizon, commandVelocity);
  ASSERT_GE(firstParts, 3) << "at least the three parts of the yaw spacing";
  ASSERT_EQ(targets.stateTrajectory.size(), 1 + firstParts + secondParts);
  EXPECT_EQ(targets.timeTrajectory.front(), kInitTime);
  EXPECT_NEAR(targets.timeTrajectory[firstParts], kInitTime + kFirstSegmentShare * horizon, 1.0e-15)
      << "the intermediate knot is still there";
  EXPECT_NEAR(targets.timeTrajectory.back(), kInitTime + horizon, 1.0e-15);
  for (size_t i = 1; i < targets.stateTrajectory.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_GT(targets.timeTrajectory[i], targets.timeTrajectory[i - 1]);
    const PlanarPose previous = planarPoseOf(stack.model(), targets.stateTrajectory[i - 1]);
    const PlanarPose knot = planarPoseOf(stack.model(), targets.stateTrajectory[i]);
    // The knots' quaternions are at most a quarter turn apart, so the shortest-path slerp between them turns the way
    // the arc does.
    EXPECT_LE(std::abs(wrapped(knot.heading - previous.heading)), 0.5 * M_PI + 1.0e-12);
    // Every knot of the first segment on its one arc: the heading linear in time, so that the slerp between the knots is
    // the one between the segment's ends.
    if (i <= firstParts) {
      const PlanarPose onTheArc = integrated(start, meanVelocity, meanYawRate, targets.timeTrajectory[i] - kInitTime);
      EXPECT_LT((knot.position - onTheArc.position).norm(), 1.0e-12);
      EXPECT_NEAR(wrapped(knot.heading - onTheArc.heading), 0.0, 1.0e-12);
    }
  }
}

TEST(IntermediateKnotMeasuredRate, AnyMeasuredSpinKeepsTheKnotsAQuarterTurnApart) {
  // The cap of 64 parts cuts a segment into quarter turns up to a turn of kMaxSegmentYawChange (100.5 rad); a
  // faster spin, 600 rad/s, as the seed of a restarted stride history can read near an upside-down base (its heading
  // rate is |w_B| / rho there), has its yaw rate clamped to that turn.
  AtlasReferenceStack stack;
  const PlanarPose start{.position = vector2_t(0.0, 0.0), .heading = -0.2};
  const BasePlanarMotion measured{.velocityInHeadingFrame = vector2_t(0.1, 0.0), .headingRate = 600.0};
  const vector4_t command(0.5, 0.0, 0.0, 1.0);
  const TargetTrajectories targets =
      stack.targetCalculator().commandedVelocityToTargetTrajectories(command, kInitTime, standingAt(stack, start), measured);
  const size_t firstParts = TargetTrajectoriesCalculatorBase::kMaxKnotSubdivisions;
  ASSERT_GT(targets.stateTrajectory.size(), 1 + firstParts);
  EXPECT_NEAR(targets.timeTrajectory[firstParts], kInitTime + kFirstSegmentShare * stack.horizon(), 1.0e-15);
  // The knots' quaternions read their headings wrapped; at most a quarter turn apart, the steps between them unwrap
  // without ambiguity, and the first segment turns by the clamped turn.
  scalar_t turned = 0.0;
  for (size_t i = 1; i < targets.stateTrajectory.size(); ++i) {
    SCOPED_TRACE(i);
    const scalar_t step = wrapped(planarPoseOf(stack.model(), targets.stateTrajectory[i]).heading -
                                  planarPoseOf(stack.model(), targets.stateTrajectory[i - 1]).heading);
    EXPECT_LE(std::abs(step), TargetTrajectoriesCalculatorBase::kMaxKnotYawSpacing + 1.0e-12);
    if (i <= firstParts) turned += step;
  }
  EXPECT_NEAR(turned, TargetTrajectoriesCalculatorBase::kMaxSegmentYawChange, 1.0e-9);
}

TEST(IntermediateKnotMeasuredRate, BetweenTheKnotsTheInterpolatedReferenceStaysOnTheArc) {
  // The solver interpolates the knots through the robot model (MpcRobotModelBase::getDesiredState(): the position
  // and the momentum linearly, the quaternion by slerp). At the DRC Atlas's command
  // limits, 1.2 m/s forward, 0.25 m/s to the side and 1 rad/s, in steady state (the measured motion the command), the
  // three knots alone would put the reference 0.074 m and 0.074 m/s off the arc between them; cut, it stays within
  // kMaxKnotPositionDeviation and kMaxKnotVelocityDeviation of it at every time of the horizon.
  AtlasReferenceStack stack;
  const PlanarPose start{.position = vector2_t(0.4, -0.7), .heading = -1.1};
  const vector2_t velocity(1.2, 0.25);
  const scalar_t yawRate = 1.0;
  const BasePlanarMotion measured{.velocityInHeadingFrame = velocity, .headingRate = yawRate};
  // The filter lets kFirstCommandFactor of the command through on the first call.
  const vector4_t command(velocity(0) / kFirstCommandFactor, velocity(1) / kFirstCommandFactor, 0.0, yawRate / kFirstCommandFactor);
  const TargetTrajectories targets =
      stack.targetCalculator().commandedVelocityToTargetTrajectories(command, kInitTime, standingAt(stack, start), measured);
  const scalar_t horizon = stack.horizon();
  const size_t parts = segmentParts(velocity, yawRate, kFirstSegmentShare * horizon, velocity) +
                       segmentParts(velocity, yawRate, (1.0 - kFirstSegmentShare) * horizon, velocity);
  ASSERT_GT(parts, 2);
  ASSERT_EQ(targets.stateTrajectory.size(), 1 + parts);
  scalar_t worstPosition = 0.0;
  scalar_t worstVelocity = 0.0;
  constexpr int kSamples = 2000;
  for (int sample = 0; sample <= kSamples; ++sample) {
    const scalar_t elapsed = horizon * static_cast<scalar_t>(sample) / kSamples;
    const vector_t reference = stack.model().getDesiredState(targets, kInitTime + elapsed);
    // Both segments hold the same twist in steady state: one arc over the horizon.
    const PlanarPose onTheArc = integrated(start, velocity, yawRate, elapsed);
    const PlanarPose pose = planarPoseOf(stack.model(), reference);
    EXPECT_NEAR(wrapped(pose.heading - onTheArc.heading), 0.0, 1.0e-9) << "the slerped heading is linear in time";
    worstPosition = std::max(worstPosition, (pose.position - onTheArc.position).norm());
    // The CoM velocity target: the command in the arc's heading frame.
    worstVelocity = std::max(worstVelocity, (vector2_t(reference.head<2>()) - rotated(onTheArc.heading, velocity)).norm());
  }
  EXPECT_LE(worstPosition, TargetTrajectoriesCalculatorBase::kMaxKnotPositionDeviation + 1.0e-9);
  EXPECT_LE(worstVelocity, TargetTrajectoriesCalculatorBase::kMaxKnotVelocityDeviation + 1.0e-9);
  EXPECT_GT(worstPosition, 0.5 * TargetTrajectoriesCalculatorBase::kMaxKnotPositionDeviation) << "the samples are between the knots";
  // Between the three knots alone the chord of the first segment is off the arc by rho (1 - cos(0.35)), rho = |v| / omega.
  EXPECT_GT(velocity.norm() / yawRate * (1.0 - std::cos(0.5 * kFirstSegmentShare * horizon * yawRate)), 0.07);
}

TEST(IntermediateKnotMeasuredRate, AtRestTheKnotsAreTheCommandsAlone) {
  // At rest: no measured motion, the first segment at half the command.
  AtlasReferenceStack stack;
  const PlanarPose start{.position = vector2_t(0.0, 0.0), .heading = 1.3};
  const vector4_t command(0.5, 0.0, 0.0, 0.0);
  const TargetTrajectories targets = stack.targetCalculator().commandedVelocityToTargetTrajectories(
      command, kInitTime, standingAt(stack, start), /*measuredMotion=*/BasePlanarMotion{});
  ASSERT_EQ(targets.stateTrajectory.size(), 3);
  const scalar_t forward = kFirstCommandFactor * command(0);
  const vector2_t intermediate = planarPoseOf(stack.model(), targets.stateTrajectory[1]).position;
  const vector2_t last = planarPoseOf(stack.model(), targets.stateTrajectory[2]).position;
  EXPECT_LT((intermediate - rotated(/*angle=*/1.3, vector2_t(0.5 * forward * kFirstSegmentShare * stack.horizon(), 0.0))).norm(), 1.0e-15);
  EXPECT_LT((last - intermediate - rotated(/*angle=*/1.3, vector2_t(forward * (1.0 - kFirstSegmentShare) * stack.horizon(), 0.0))).norm(),
            1.0e-15);
}

}  // namespace
}  // namespace ocs2::humanoid
