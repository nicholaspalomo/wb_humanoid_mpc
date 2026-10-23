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
#include <vector>

#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_core/reference/TargetTrajectories.h"

#include "humanoid_common_mpc/command/BasePlanarMotion.h"
#include "humanoid_common_mpc/command/TargetTrajectoriesCalculatorBase.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_wb_mpc/command/WBMpcTargetTrajectoriesCalculator.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "robot_core/ResourcePaths.h"

/*
 * The knots of the whole-body MPC's velocity command (TargetTrajectoriesCalculatorBase's class comment): over the first
 * 70 % of the horizon the base moves at the mean of the measured and the commanded planar twist (design defect D1), the
 * measurement being the robot's stride-mean motion handed in (design README section 2.9), then at the command; every
 * knot lies on the arc of its twist, checked against an RK4 integration of the unicycle, and moves at the command
 * in its own heading frame; a fast turn is cut into knots at most pi / 2 apart. On the shipped G1
 * whole-body files; builds no Pinocchio or CppAD model.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kInitTime = 1.5;
constexpr scalar_t kHorizon = 1.0;
// The share of the horizon of the first segment, and the share of the command that reaches the knots: all of it, since
// the calculator does not filter the command (ProceduralMpcMotionManager does, on the solver time).
constexpr scalar_t kFirstSegmentShare = 0.7;
constexpr scalar_t kFirstCommandFactor = 1.0;

/** The G1 whole-body model and target calculator of the shipped files, and the task file's initial state. */
struct WholeBodyCalculator {
  std::unique_ptr<ModelSettings> modelSettings;
  std::unique_ptr<WBAccelMpcRobotModel<scalar_t>> model;
  std::unique_ptr<WBMpcTargetTrajectoriesCalculator> calculator;
  vector_t initialState;
};

absl::StatusOr<WholeBodyCalculator> loadG1() {
  // LINT.IfChange(robot_files)
  ASSIGN_OR_RETURN(const std::string taskFile, robot::resolveResourcePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto"));
  ASSIGN_OR_RETURN(const std::string urdfFile, robot::resolveResourcePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"));
  ASSIGN_OR_RETURN(const std::string referenceFile,
                   robot::resolveResourcePath("robot_models/unitree_g1/g1_wb_mpc/config/command/reference.textproto"));
  // LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc/BUILD.bazel:intermediate_knot_test_data)
  WholeBodyCalculator loaded;
  ASSIGN_OR_RETURN(const mpc_config::TaskFile task, loadTaskFile(taskFile));
  ASSIGN_OR_RETURN(ModelSettings modelSettings, ModelSettings::Create(task, urdfFile, "wb_mpc_", /*verbose=*/false));
  loaded.modelSettings = std::make_unique<ModelSettings>(std::move(modelSettings));
  loaded.model = std::make_unique<WBAccelMpcRobotModel<scalar_t>>(*loaded.modelSettings);
  ASSIGN_OR_RETURN(loaded.calculator, WBMpcTargetTrajectoriesCalculator::Create(referenceFile, *loaded.model, kHorizon));
  ASSIGN_OR_RETURN(const vector_t tuningState,
                   stateValuesFromConfig(task.initial_state, stateInputLayout(*loaded.modelSettings, StateInputLayout::Mpc::kWholeBody),
                                         "initial_state"));
  ASSIGN_OR_RETURN(loaded.initialState, loaded.model->stateFromTuning(tuningState));
  return loaded;
}

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

/**
 * `heading` - `reference` wrapped into [-pi, pi]: a knot's heading is read back from its quaternion in (-pi, pi]
 * (getBasePoseEulerZyx()), the integrated one is unbounded.
 */
scalar_t headingDifference(scalar_t heading, scalar_t reference) {
  return std::remainder(heading - reference, 2.0 * M_PI);
}

/** The planar pose of a knot. */
PlanarPose planarPoseOf(const MpcRobotModelBase<scalar_t>& model, const vector_t& knot) {
  const vector6_t pose = model.getBasePoseEulerZyx(knot);
  return PlanarPose{.position = vector2_t(pose(0), pose(1)), .heading = pose(3)};
}

/** The initial state at the planar pose `pose`, level and still. */
vector_t standingAt(const WholeBodyCalculator& g1, const PlanarPose& pose) {
  vector_t state = g1.initialState;
  g1.model->setBasePoseEulerZyx(
      state, (vector6_t() << pose.position(0), pose.position(1), g1.model->getBasePosition(state)(2), pose.heading, 0.0, 0.0).finished());
  return state;
}

TEST(WBIntermediateKnotMeasuredRate, TheKnotsLieOnTheArcOfTheirTwistAndMoveAtTheCommandInTheirHeadingFrame) {
  absl::StatusOr<WholeBodyCalculator> g1 = loadG1();
  ASSERT_TRUE(g1.ok()) << g1.status();
  const PlanarPose start{.position = vector2_t(-1.0, 2.0), .heading = 2.5};
  const BasePlanarMotion measured{.velocityInHeadingFrame = vector2_t(0.25, -0.05), .headingRate = 0.6};
  // The command the calculator used to see after its per-call filter (1 - 0.8 of it), so that the knots stay unsubdivided.
  const vector4_t command(0.1, 0.04, 0.0, 0.18);
  const TargetTrajectories targets =
      g1->calculator->commandedVelocityToTargetTrajectories(command, kInitTime, standingAt(*g1, start), measured);
  ASSERT_EQ(targets.stateTrajectory.size(), 3);
  const vector2_t commandVelocity = kFirstCommandFactor * vector2_t(command(0), command(1));
  const scalar_t commandYawRate = kFirstCommandFactor * command(3);
  const PlanarPose intermediate = integrated(start, 0.5 * (measured.velocityInHeadingFrame + commandVelocity),
                                             0.5 * (measured.headingRate + commandYawRate), kFirstSegmentShare * kHorizon);
  const PlanarPose last = integrated(intermediate, commandVelocity, commandYawRate, (1.0 - kFirstSegmentShare) * kHorizon);
  const std::vector<PlanarPose> expected = {start, intermediate, last};
  for (size_t i = 0; i < 3; ++i) {
    SCOPED_TRACE(i);
    const vector_t& knot = targets.stateTrajectory[i];
    const PlanarPose pose = planarPoseOf(*g1->model, knot);
    EXPECT_LT((pose.position - expected[i].position).norm(), 1.0e-12) << pose.position.transpose();
    EXPECT_NEAR(headingDifference(pose.heading, expected[i].heading), 0.0, 1.0e-12);
    // The base velocity: the command turned into the world at the knot's heading, turning at the commanded yaw rate.
    const vector3_t command3 = g1->model->getTargetPlanarCommand(knot, /*compositeYawInertia=*/0.0);
    EXPECT_LT((vector2_t(command3.head<2>()) - rotated(pose.heading, commandVelocity)).norm(), 1.0e-14);
    EXPECT_NEAR(command3(2), commandYawRate, 1.0e-14);
  }
}

TEST(WBIntermediateKnotMeasuredRate, OnAStraightLineTheFirstSegmentIsTheTrapezoidOfTheMeasuredAndTheCommandedVelocity) {
  absl::StatusOr<WholeBodyCalculator> g1 = loadG1();
  ASSERT_TRUE(g1.ok()) << g1.status();
  const PlanarPose start{.position = vector2_t(0.5, -0.3), .heading = 1.0};
  const BasePlanarMotion measured{.velocityInHeadingFrame = vector2_t(0.2, -0.1), .headingRate = 0.0};
  const vector4_t command(0.3, 0.1, 0.0, 0.0);
  const TargetTrajectories targets =
      g1->calculator->commandedVelocityToTargetTrajectories(command, kInitTime, standingAt(*g1, start), measured);
  ASSERT_EQ(targets.stateTrajectory.size(), 3);
  const vector2_t commandVelocity = kFirstCommandFactor * vector2_t(command(0), command(1));
  const vector2_t expected =
      kFirstSegmentShare * kHorizon * rotated(/*angle=*/1.0, 0.5 * (measured.velocityInHeadingFrame + commandVelocity));
  EXPECT_LT((planarPoseOf(*g1->model, targets.stateTrajectory[1]).position - start.position - expected).norm(), 1.0e-14);
}

TEST(WBIntermediateKnotMeasuredRate, AFastMeasuredTurnIsCutIntoKnotsAtMostAQuarterTurnApart) {
  absl::StatusOr<WholeBodyCalculator> g1 = loadG1();
  ASSERT_TRUE(g1.ok()) << g1.status();
  // A spin of 9 rad/s: the first segment turns by 0.7 T (9 + 1) / 2 = 3.5 rad, past pi. Cut into at least the three
  // parts of the yaw spacing, more for the chord of the turning velocity target (knotSubdivisionCount()).
  const PlanarPose start{.position = vector2_t(0.2, 0.1), .heading = 0.3};
  const BasePlanarMotion measured{.velocityInHeadingFrame = vector2_t(0.3, 0.1), .headingRate = 9.0};
  const vector4_t command(0.5, 0.0, 0.0, 1.0);
  const TargetTrajectories targets =
      g1->calculator->commandedVelocityToTargetTrajectories(command, kInitTime, standingAt(*g1, start), measured);
  const vector2_t commandVelocity = kFirstCommandFactor * vector2_t(command(0), command(1));
  const scalar_t commandYawRate = kFirstCommandFactor * command(3);
  const vector2_t meanVelocity = 0.5 * (measured.velocityInHeadingFrame + commandVelocity);
  const scalar_t meanYawRate = 0.5 * (measured.headingRate + commandYawRate);
  using Extent = TargetTrajectoriesCalculatorBase::KnotSegmentExtent;
  const size_t firstParts =
      TargetTrajectoriesCalculatorBase::knotSubdivisionCount(Extent{.yawChange = meanYawRate * kFirstSegmentShare * kHorizon,
                                                                    .arcLength = meanVelocity.norm() * kFirstSegmentShare * kHorizon,
                                                                    .targetSpeed = commandVelocity.norm()});
  const size_t secondParts = TargetTrajectoriesCalculatorBase::knotSubdivisionCount(
      Extent{.yawChange = commandYawRate * (1.0 - kFirstSegmentShare) * kHorizon,
             .arcLength = commandVelocity.norm() * (1.0 - kFirstSegmentShare) * kHorizon,
             .targetSpeed = commandVelocity.norm()});
  ASSERT_GE(firstParts, 3);
  ASSERT_EQ(targets.stateTrajectory.size(), 1 + firstParts + secondParts);
  EXPECT_NEAR(targets.timeTrajectory[firstParts], kInitTime + kFirstSegmentShare * kHorizon, 1.0e-15);
  EXPECT_NEAR(targets.timeTrajectory.back(), kInitTime + kHorizon, 1.0e-15);
  for (size_t i = 1; i < targets.stateTrajectory.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_GT(targets.timeTrajectory[i], targets.timeTrajectory[i - 1]);
    const PlanarPose knot = planarPoseOf(*g1->model, targets.stateTrajectory[i]);
    EXPECT_LE(std::abs(headingDifference(knot.heading, planarPoseOf(*g1->model, targets.stateTrajectory[i - 1]).heading)),
              0.5 * M_PI + 1.0e-12);
    if (i <= firstParts) {
      const PlanarPose onTheArc = integrated(start, meanVelocity, meanYawRate, targets.timeTrajectory[i] - kInitTime);
      EXPECT_LT((knot.position - onTheArc.position).norm(), 1.0e-12);
      EXPECT_NEAR(headingDifference(knot.heading, onTheArc.heading), 0.0, 1.0e-12);
    }
  }
}

}  // namespace
}  // namespace ocs2::humanoid
