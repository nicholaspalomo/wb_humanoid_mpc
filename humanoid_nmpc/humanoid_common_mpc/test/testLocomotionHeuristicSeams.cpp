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

#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "Eigen/Geometry"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/locomotion_heuristics/FootholdHeuristic.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicLayer.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicModelParameters.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

/*
 * Properties of the reference manager's seams into the locomotion heuristics
 * that need no robot model:
 *
 *  - the foothold anchor
 * (SwitchedModelReferenceManager::computePosHipRtWorldEwrtWorldXy()) keeps the
 * feet with the body under a sustained command, sideways above all, when the
 * listed heuristics' leads are summed on top of it - simulated step by step,
 * landing each foot on its target, against a body that moves at the command;
 *  - the angular velocity the base-pose seam gives a shaped attitude
 * (computeShapedAngvelBaseRtWorldEwrtBase(),
 * computeOffsetAngvelBaseRtWorldEwrtWorld()) is the derivative of that
 * attitude, against a central difference of rotation matrices built from
 * elementary rotations, independent of the Euler-boundary code under test.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kNomPosLfootRtRfootEwrtHeadingY = 0.25;  // [m]
constexpr scalar_t kStepDuration =
    0.5;  // [s] alternating single support, so each stance lasts one step

/** The foothold layer of `foothold`, with Raibert's half-stance leads on both
 * axes. */
std::unique_ptr<LocomotionHeuristicLayer> raibertLayer() {
  LocomotionHeuristicConfig config;
  config.formulation.footholdHeuristics = {"translational_stepping"};
  config.translationalStepping.forwardStanceFraction = 0.5;
  config.translationalStepping.lateralStanceFraction = 0.5;
  LocomotionHeuristicModelParameters model;
  model.totalMass = 40.0;
  model.totalWeight = model.totalMass * model.gravity;
  model.nomPosComRtFeetmeanEwrtWorldZ = 0.6;
  LocomotionHeuristicEnvironment environment;
  environment.nomPosLfootRtRfootEwrtHeadingY = kNomPosLfootRtRfootEwrtHeadingY;
  absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
      LocomotionHeuristicLayer::Create(config, model, environment);
  EXPECT_TRUE(layer.ok()) << layer.status();
  return layer.ok() ? *std::move(layer) : nullptr;
}

/** R_z(yaw)' v: a world vector in the heading frame. */
vector2_t inHeadingFrame(scalar_t yawHeadingRtWorld,
                         const vector2_t& worldComponents) {
  return vector2_t(std::cos(yawHeadingRtWorld) * worldComponents.x() +
                       std::sin(yawHeadingRtWorld) * worldComponents.y(),
                   -std::sin(yawHeadingRtWorld) * worldComponents.x() +
                       std::cos(yawHeadingRtWorld) * worldComponents.y());
}

/** How a landing target is anchored: the reference manager's
 * computePosHipRtWorldEwrtWorldXy(), or the stance-foot anchor it replaced. */
enum class Anchor { kFootholdAnchor, kStanceFootPlusStepWidth };

/**
 * The midpoint of the feet minus the base, in the heading frame, after each of
 * `numSteps` steps of a robot that walks at the world velocity
 * `cmdLinvelBaseRtWorldEwrtWorldXy` at
 * heading `yawBaseRtWorld` and lands every foot exactly on its target. The base
 * is where the command puts it (perfect tracking), so the prediction at each
 * touch-down is exact.
 */
std::vector<vector2_t> computePosFeetmeanRtBaseEwrtHeadingXyTraj(
    const LocomotionHeuristicLayer& layer,
    const vector2_t& cmdLinvelBaseRtWorldEwrtWorldXy,
    scalar_t yawBaseRtWorld,
    int numSteps,
    Anchor anchor) {
  feet_array_t<vector2_t> posFeetRtWorldEwrtWorldXy =
      makeFeetArray(vector2_t(vector2_t::Zero()));
  const vector2_t leftDirHeadingEwrtWorldXy(-std::sin(yawBaseRtWorld),
                                            std::cos(yawBaseRtWorld));
  posFeetRtWorldEwrtWorldXy[kContactLeftIndex] =
      0.5 * kNomPosLfootRtRfootEwrtHeadingY * leftDirHeadingEwrtWorldXy;
  posFeetRtWorldEwrtWorldXy[kContactRightIndex] =
      -0.5 * kNomPosLfootRtRfootEwrtHeadingY * leftDirHeadingEwrtWorldXy;
  std::vector<vector2_t> posFeetmeanRtBaseEwrtHeadingXyTraj;
  for (int step = 1; step <= numSteps; ++step) {
    const size_t swing =
        (step % 2 == 1) ? kContactRightIndex : kContactLeftIndex;
    const size_t stance =
        (swing == kContactLeftIndex) ? kContactRightIndex : kContactLeftIndex;
    const scalar_t touchDown = static_cast<scalar_t>(step) * kStepDuration;
    FootholdHeuristicContext context;
    context.contactIndex = swing;
    context.side = (swing == kContactLeftIndex) ? 1.0 : -1.0;
    context.posBaseRtWorldEwrtWorldXy =
        cmdLinvelBaseRtWorldEwrtWorldXy * touchDown;
    context.yawBaseRtWorld = yawBaseRtWorld;
    context.cmdLinvelBaseRtWorldEwrtWorldXy = cmdLinvelBaseRtWorldEwrtWorldXy;
    context.prevCmdLinvelBaseRtWorldEwrtWorldXy =
        cmdLinvelBaseRtWorldEwrtWorldXy;
    context.measProxyLinvelComRtWorldEwrtWorldXy =
        cmdLinvelBaseRtWorldEwrtWorldXy;
    context.stanceDuration = kStepDuration;
    const vector2_t offsetPosFootRtWorldEwrtWorldXy =
        layer.computeOffsetPosFootRtWorldEwrtWorldXy(context);
    if (anchor == Anchor::kFootholdAnchor) {
      posFeetRtWorldEwrtWorldXy[swing] =
          SwitchedModelReferenceManager::computePosHipRtWorldEwrtWorldXy(
              context.posBaseRtWorldEwrtWorldXy, yawBaseRtWorld, context.side,
              kNomPosLfootRtRfootEwrtHeadingY) +
          offsetPosFootRtWorldEwrtWorldXy;
    } else {
      // The anchor this replaced: the predicted base along the heading, the
      // stance foot plus a whole step width across it.
      const vector2_t dirHeadingEwrtWorldXy(std::cos(yawBaseRtWorld),
                                            std::sin(yawBaseRtWorld));
      posFeetRtWorldEwrtWorldXy[swing] =
          dirHeadingEwrtWorldXy *
              dirHeadingEwrtWorldXy.dot(context.posBaseRtWorldEwrtWorldXy) +
          leftDirHeadingEwrtWorldXy *
              (leftDirHeadingEwrtWorldXy.dot(
                   posFeetRtWorldEwrtWorldXy[stance]) +
               context.side * kNomPosLfootRtRfootEwrtHeadingY) +
          offsetPosFootRtWorldEwrtWorldXy;
    }
    posFeetmeanRtBaseEwrtHeadingXyTraj.push_back(inHeadingFrame(
        yawBaseRtWorld, 0.5 * (posFeetRtWorldEwrtWorldXy[kContactLeftIndex] +
                               posFeetRtWorldEwrtWorldXy[kContactRightIndex]) -
                            context.posBaseRtWorldEwrtWorldXy));
  }
  return posFeetmeanRtBaseEwrtHeadingXyTraj;
}

TEST(LocomotionHeuristicSeams, TheFeetStayWithTheBodyUnderASustainedSidestep) {
  // The property of LH-1: a constant lateral command (0, 0.3) m/s must leave
  // the stance where the body is. With every foot landed on its target, the
  // midpoint of the feet relative to the base settles to a constant after the
  // first two steps - set by the lead of the heuristics, the same at every step
  // - at any heading and any command. The anchor it replaced (the stance foot
  // plus a whole step width) lets the midpoint fall behind by v T_step - k T_st
  // v at EVERY step, which is checked as the control: 0.3 * 0.5 - 0.25 * 0.3 =
  // 0.075 m a step, 2.85 m after 40 steps.
  const std::unique_ptr<LocomotionHeuristicLayer> layer = raibertLayer();
  ASSERT_NE(layer, nullptr);
  const int numSteps = 40;
  for (const vector2_t& cmdLinvelBaseRtWorldEwrtHeadingXy :
       {vector2_t(0.0, 0.3), vector2_t(0.0, -0.45), vector2_t(0.6, 0.2),
        vector2_t(-0.3, 0.0)}) {
    for (const scalar_t yawBaseRtWorld : {0.0, 0.7, -2.6}) {
      const vector2_t cmdLinvelBaseRtWorldEwrtWorldXy(
          std::cos(yawBaseRtWorld) * cmdLinvelBaseRtWorldEwrtHeadingXy.x() -
              std::sin(yawBaseRtWorld) * cmdLinvelBaseRtWorldEwrtHeadingXy.y(),
          std::sin(yawBaseRtWorld) * cmdLinvelBaseRtWorldEwrtHeadingXy.x() +
              std::cos(yawBaseRtWorld) * cmdLinvelBaseRtWorldEwrtHeadingXy.y());
      const std::vector<vector2_t> posFeetmeanRtBaseEwrtHeadingXyTraj =
          computePosFeetmeanRtBaseEwrtHeadingXyTraj(
              *layer, cmdLinvelBaseRtWorldEwrtWorldXy, yawBaseRtWorld, numSteps,
              Anchor::kFootholdAnchor);
      // From the second step on, both feet have been placed by the anchor: the
      // midpoint minus the base is the mean of the two feet's leads, constant
      // step after step. Bounded is not enough, it must not move at all.
      for (int step = 2; step < numSteps; ++step) {
        EXPECT_LT((posFeetmeanRtBaseEwrtHeadingXyTraj[step] -
                   posFeetmeanRtBaseEwrtHeadingXyTraj[1])
                      .norm(),
                  1.0e-12)
            << "command " << cmdLinvelBaseRtWorldEwrtHeadingXy.transpose()
            << ", yaw " << yawBaseRtWorld << ", step " << step + 1 << ": "
            << posFeetmeanRtBaseEwrtHeadingXyTraj[step].transpose();
      }
      // And the constant is what Raibert's rule predicts: the foot just landed
      // leads its hip by k T_st v, the other one landed a step earlier and has
      // since fallen T_step v behind the base, so the midpoint sits k T_st v -
      // T_step v / 2 from the base - zero for a half-stance lead on a gait
      // whose stance is one step. The feet are not left behind and not thrown
      // ahead.
      const vector2_t expectedPosFeetmeanRtBaseEwrtHeadingXy =
          0.5 * kStepDuration * cmdLinvelBaseRtWorldEwrtHeadingXy -
          0.5 * kStepDuration * cmdLinvelBaseRtWorldEwrtHeadingXy;
      EXPECT_LT((posFeetmeanRtBaseEwrtHeadingXyTraj[1] -
                 expectedPosFeetmeanRtBaseEwrtHeadingXy)
                    .norm(),
                1.0e-12)
          << "command " << cmdLinvelBaseRtWorldEwrtHeadingXy.transpose()
          << ", yaw " << yawBaseRtWorld;

      // Control: the stance-foot anchor drifts under any lateral command,
      // without bound.
      const std::vector<vector2_t> driftingPosFeetmeanRtBaseEwrtHeadingXyTraj =
          computePosFeetmeanRtBaseEwrtHeadingXyTraj(
              *layer, cmdLinvelBaseRtWorldEwrtWorldXy, yawBaseRtWorld, numSteps,
              Anchor::kStanceFootPlusStepWidth);
      if (std::abs(cmdLinvelBaseRtWorldEwrtHeadingXy.y()) > 0.0) {
        const scalar_t perStep =
            (kStepDuration - 0.25) *
            std::abs(cmdLinvelBaseRtWorldEwrtHeadingXy.y());
        EXPECT_GT(
            std::abs(driftingPosFeetmeanRtBaseEwrtHeadingXyTraj.back().y() -
                     driftingPosFeetmeanRtBaseEwrtHeadingXyTraj[1].y()),
            0.9 * perStep * (numSteps - 2))
            << "the control did not drift";
      }
    }
  }
}

TEST(LocomotionHeuristicSeams,
     TheAnchorIsHalfAStepWidthToTheFootsSideAcrossThePredictedHeading) {
  // The two hips at a touch-down are a step width apart and centered on the
  // base, across the heading: rotating the heading rotates them about the base,
  // and the side sign mirrors them.
  for (const scalar_t yawBaseRtWorld : {0.0, 0.4, 2.0, -1.3}) {
    const vector2_t posBaseRtWorldEwrtWorldXy(1.2, -0.7);
    const vector2_t posLhipRtWorldEwrtWorldXy =
        SwitchedModelReferenceManager::computePosHipRtWorldEwrtWorldXy(
            posBaseRtWorldEwrtWorldXy, yawBaseRtWorld, /*side=*/1.0,
            kNomPosLfootRtRfootEwrtHeadingY);
    const vector2_t posRhipRtWorldEwrtWorldXy =
        SwitchedModelReferenceManager::computePosHipRtWorldEwrtWorldXy(
            posBaseRtWorldEwrtWorldXy, yawBaseRtWorld, /*side=*/-1.0,
            kNomPosLfootRtRfootEwrtHeadingY);
    EXPECT_LT((0.5 * (posLhipRtWorldEwrtWorldXy + posRhipRtWorldEwrtWorldXy) -
               posBaseRtWorldEwrtWorldXy)
                  .norm(),
              1.0e-15)
        << "yaw " << yawBaseRtWorld;
    EXPECT_NEAR((posLhipRtWorldEwrtWorldXy - posRhipRtWorldEwrtWorldXy).norm(),
                kNomPosLfootRtRfootEwrtHeadingY, 1.0e-15)
        << "yaw " << yawBaseRtWorld;
    const vector2_t posLhipRtBaseEwrtHeadingXy = inHeadingFrame(
        yawBaseRtWorld, posLhipRtWorldEwrtWorldXy - posBaseRtWorldEwrtWorldXy);
    EXPECT_NEAR(posLhipRtBaseEwrtHeadingXy.x(), 0.0, 1.0e-15)
        << "yaw " << yawBaseRtWorld;
    EXPECT_NEAR(posLhipRtBaseEwrtHeadingXy.y(),
                0.5 * kNomPosLfootRtRfootEwrtHeadingY, 1.0e-15)
        << "the left hip is to the robot's left, yaw " << yawBaseRtWorld;
  }
}

/** R = R_z(yaw) R_y(pitch) R_x(roll) of ZYX Euler angles (yaw, pitch, roll),
 * from elementary rotations. */
matrix3_t rotmatFromEulerZyx(const vector3_t& eulerZyx) {
  return (Eigen::AngleAxis<scalar_t>(eulerZyx(0), vector3_t::UnitZ()) *
          Eigen::AngleAxis<scalar_t>(eulerZyx(1), vector3_t::UnitY()) *
          Eigen::AngleAxis<scalar_t>(eulerZyx(2), vector3_t::UnitX()))
      .toRotationMatrix();
}

/** The vector of a skew-symmetric matrix. */
vector3_t vee(const matrix3_t& skew) {
  return vector3_t(0.5 * (skew(2, 1) - skew(1, 2)),
                   0.5 * (skew(0, 2) - skew(2, 0)),
                   0.5 * (skew(1, 0) - skew(0, 1)));
}

/** An attitude trajectory and its rate, as ZYX Euler angles: Theta(t) = a + b t
 * + c sin(w t). */
struct EulerTrajectory {
  vector3_t constant;
  vector3_t linear;
  vector3_t amplitude;
  scalar_t frequency = 0.0;

  vector3_t at(scalar_t time) const {
    return constant + linear * time + amplitude * std::sin(frequency * time);
  }
  vector3_t rateAt(scalar_t time) const {
    return linear + amplitude * frequency * std::cos(frequency * time);
  }
};

TEST(LocomotionHeuristicSeams,
     TheShapedAngularVelocityIsTheDerivativeOfTheShapedAttitude) {
  // The nominal attitude Theta_n(t) turns, pitches and rolls; the offset
  // delta(t) tilts it in pitch and roll at gait frequency. The shaped attitude
  // Theta_n + delta must rotate at w_B = vee(R' dR/dt), measured by a central
  // difference of R_z R_y R_x - and its world rate must exceed the nominal's by
  // computeOffsetAngvelBaseRtWorldEwrtWorld() wherever the nominal does not
  // roll.
  const scalar_t step = 1.0e-6;
  const std::vector<EulerTrajectory> nominals = {
      {.constant = vector3_t(0.3, 0.05, -0.04),
       .linear = vector3_t(0.8, 0.0, 0.0),
       .amplitude = vector3_t::Zero(),
       .frequency = 0.0},
      {.constant = vector3_t(-2.1, 0.1, 0.2),
       .linear = vector3_t(-0.5, 0.03, 0.0),
       .amplitude = vector3_t(0.1, 0.05, 0.0),
       .frequency = 3.0},
      {.constant = vector3_t(1.0, -0.2, 0.1),
       .linear = vector3_t(0.2, 0.0, 0.3),
       .amplitude = vector3_t(0.0, 0.1, 0.1),
       .frequency = 5.0},
  };
  const EulerTrajectory offset{.constant = vector3_t(0.0, 0.02, -0.01),
                               .linear = vector3_t::Zero(),
                               .amplitude = vector3_t(0.0, 0.03, 0.05),
                               .frequency = 7.0};
  for (size_t index = 0; index < nominals.size(); ++index) {
    const EulerTrajectory& nominal = nominals[index];
    for (const scalar_t time : {0.0, 0.37, 1.1}) {
      const vector3_t refEulerzyxBaseRtWorld = nominal.at(time);
      const vector3_t shapedEulerzyxBaseRtWorld =
          refEulerzyxBaseRtWorld + offset.at(time);
      const vector3_t refAngvelBaseRtWorldEwrtBase = bodyRatesFromEulerZyxRates(
          refEulerzyxBaseRtWorld, nominal.rateAt(time));
      const vector3_t shapedAngvelBaseRtWorldEwrtBase =
          SwitchedModelReferenceManager::computeShapedAngvelBaseRtWorldEwrtBase(
              refEulerzyxBaseRtWorld, refAngvelBaseRtWorldEwrtBase,
              shapedEulerzyxBaseRtWorld, offset.rateAt(time));

      const matrix3_t rotmatBaseRtWorld =
          rotmatFromEulerZyx(shapedEulerzyxBaseRtWorld);
      const matrix3_t rateRotmatBaseRtWorld =
          (rotmatFromEulerZyx(nominal.at(time + step) +
                              offset.at(time + step)) -
           rotmatFromEulerZyx(nominal.at(time - step) +
                              offset.at(time - step))) /
          (2.0 * step);
      const vector3_t fdShapedAngvelBaseRtWorldEwrtBase =
          vee(rotmatBaseRtWorld.transpose() * rateRotmatBaseRtWorld);
      EXPECT_LT(
          (shapedAngvelBaseRtWorldEwrtBase - fdShapedAngvelBaseRtWorldEwrtBase)
              .norm(),
          1.0e-8)
          << "nominal " << index << ", t " << time << ": "
          << shapedAngvelBaseRtWorldEwrtBase.transpose() << " vs "
          << fdShapedAngvelBaseRtWorldEwrtBase.transpose();

      // The world rate the tilt adds, where the nominal does not roll (its roll
      // rate is zero on the first trajectory).
      if (std::abs(nominal.rateAt(time)(2)) == 0.0) {
        const matrix3_t nomRateRotmatBaseRtWorld =
            (rotmatFromEulerZyx(nominal.at(time + step)) -
             rotmatFromEulerZyx(nominal.at(time - step))) /
            (2.0 * step);
        const vector3_t nomAngvelBaseRtWorldEwrtWorld =
            vee(nomRateRotmatBaseRtWorld *
                rotmatFromEulerZyx(refEulerzyxBaseRtWorld).transpose());
        const vector3_t shapedAngvelBaseRtWorldEwrtWorld =
            vee(rateRotmatBaseRtWorld * rotmatBaseRtWorld.transpose());
        const vector3_t offsetAngvelBaseRtWorldEwrtWorld =
            SwitchedModelReferenceManager::
                computeOffsetAngvelBaseRtWorldEwrtWorld(
                    shapedEulerzyxBaseRtWorld, offset.rateAt(time));
        EXPECT_LT(
            (offsetAngvelBaseRtWorldEwrtWorld -
             (shapedAngvelBaseRtWorldEwrtWorld - nomAngvelBaseRtWorldEwrtWorld))
                .norm(),
            1.0e-8)
            << "nominal " << index << ", t " << time;
      }
    }
  }
}

TEST(LocomotionHeuristicSeams, AConstantTiltKeepsTheNominalsTurnInTheWorld) {
  // No offset rate, a nominal that only turns: the shaped attitude turns about
  // the world z axis at the nominal's rate, which is what the seam promised
  // before it carried tilt rates, so a constant offset changes nothing about
  // the turn.
  const vector3_t refEulerzyxBaseRtWorld(0.9, 0.0, 0.0);
  const scalar_t rateYawBaseRtWorld = 0.6;
  const vector3_t refAngvelBaseRtWorldEwrtBase = bodyRatesFromEulerZyxRates(
      refEulerzyxBaseRtWorld, vector3_t(rateYawBaseRtWorld, 0.0, 0.0));
  const vector3_t shapedEulerzyxBaseRtWorld(0.9, 0.08, -0.05);
  const vector3_t shapedAngvelBaseRtWorldEwrtBase =
      SwitchedModelReferenceManager::computeShapedAngvelBaseRtWorldEwrtBase(
          refEulerzyxBaseRtWorld, refAngvelBaseRtWorldEwrtBase,
          shapedEulerzyxBaseRtWorld, vector3_t::Zero());
  EXPECT_LT((rotmatFromEulerZyx(shapedEulerzyxBaseRtWorld) *
                 shapedAngvelBaseRtWorldEwrtBase -
             vector3_t(0.0, 0.0, rateYawBaseRtWorld))
                .norm(),
            1.0e-15);
  EXPECT_TRUE(
      SwitchedModelReferenceManager::computeOffsetAngvelBaseRtWorldEwrtWorld(
          shapedEulerzyxBaseRtWorld, vector3_t::Zero())
          .isZero(0.0));
}

}  // namespace
}  // namespace ocs2::humanoid
