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

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "Eigen/Geometry"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicLayer.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicModelParameters.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

/*
 * Properties of the reference manager's seams into the locomotion heuristics that need no robot model:
 *
 *  - the foothold anchor (SwitchedModelReferenceManager::footholdAnchor()) keeps the feet with the body under a
 *    sustained command, sideways above all, when the listed heuristics' leads are summed on top of it - simulated
 *    step by step, landing each foot on its target, against a body that moves at the command;
 *  - the angular velocity the base-pose seam gives a shaped attitude (shapedBodyAngularVelocity(),
 *    tiltAngularVelocityInWorld()) is the derivative of that attitude, against a central difference of rotation
 *    matrices built from elementary rotations, independent of the Euler-boundary code under test.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kStepWidth = 0.25;    // [m]
constexpr scalar_t kStepDuration = 0.5;  // [s] alternating single support, so each stance lasts one step

/** The foothold layer of `foothold`, with Raibert's half-stance leads on both axes. */
std::unique_ptr<LocomotionHeuristicLayer> raibertLayer() {
  LocomotionHeuristicConfig config;
  config.formulation.foothold = {"translational_stepping"};
  config.translationalStepping.forwardStanceFraction = 0.5;
  config.translationalStepping.lateralStanceFraction = 0.5;
  LocomotionHeuristicModelParameters model;
  model.totalMass = 40.0;
  model.totalWeight = model.totalMass * model.gravity;
  model.nominalComHeight = 0.6;
  LocomotionHeuristicEnvironment environment;
  environment.nominalStepWidth = kStepWidth;
  absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer = LocomotionHeuristicLayer::Create(config, model, environment);
  EXPECT_TRUE(layer.ok()) << layer.status();
  return layer.ok() ? *std::move(layer) : nullptr;
}

/** R_z(yaw)' v: a world vector in the heading frame. */
vector2_t inHeadingFrame(scalar_t yaw, const vector2_t& world) {
  return vector2_t(std::cos(yaw) * world.x() + std::sin(yaw) * world.y(), -std::sin(yaw) * world.x() + std::cos(yaw) * world.y());
}

/** How a landing target is anchored: the reference manager's footholdAnchor(), or the stance-foot anchor it replaced. */
enum class Anchor { kFootholdAnchor, kStanceFootPlusStepWidth };

/**
 * The midpoint of the feet minus the base, in the heading frame, after each of `numSteps` steps of a robot that walks
 * at the world velocity `command` at heading `yaw` and lands every foot exactly on its target. The base is where the
 * command puts it (perfect tracking), so the prediction at each touch-down is exact.
 */
std::vector<vector2_t> feetMinusBase(
    const LocomotionHeuristicLayer& layer, const vector2_t& command, scalar_t yaw, int numSteps, Anchor anchor) {
  feet_array_t<vector2_t> feet = makeFeetArray(vector2_t(vector2_t::Zero()));
  const vector2_t leftward(-std::sin(yaw), std::cos(yaw));
  feet[kContactLeftIndex] = 0.5 * kStepWidth * leftward;
  feet[kContactRightIndex] = -0.5 * kStepWidth * leftward;
  std::vector<vector2_t> history;
  for (int step = 1; step <= numSteps; ++step) {
    const size_t swing = (step % 2 == 1) ? kContactRightIndex : kContactLeftIndex;
    const size_t stance = (swing == kContactLeftIndex) ? kContactRightIndex : kContactLeftIndex;
    const scalar_t touchDown = static_cast<scalar_t>(step) * kStepDuration;
    FootholdHeuristicContext context;
    context.contactIndex = swing;
    context.side = (swing == kContactLeftIndex) ? 1.0 : -1.0;
    context.basePosition = command * touchDown;
    context.baseYaw = yaw;
    context.commandedVelocity = command;
    context.commandedVelocityAtMeasurement = command;
    context.measuredVelocity = command;
    context.stanceDuration = kStepDuration;
    const vector2_t offsets = layer.footholdOffset(context);
    if (anchor == Anchor::kFootholdAnchor) {
      feet[swing] = SwitchedModelReferenceManager::footholdAnchor(context.basePosition, yaw, context.side, kStepWidth) + offsets;
    } else {
      // The anchor this replaced: the predicted base along the heading, the stance foot plus a whole step width across it.
      const vector2_t forward(std::cos(yaw), std::sin(yaw));
      feet[swing] =
          forward * forward.dot(context.basePosition) + leftward * (leftward.dot(feet[stance]) + context.side * kStepWidth) + offsets;
    }
    history.push_back(inHeadingFrame(yaw, 0.5 * (feet[kContactLeftIndex] + feet[kContactRightIndex]) - context.basePosition));
  }
  return history;
}

TEST(LocomotionHeuristicSeams, TheFeetStayWithTheBodyUnderASustainedSidestep) {
  // The property of LH-1: a constant lateral command (0, 0.3) m/s must leave the stance where the body is. With every
  // foot landed on its target, the midpoint of the feet relative to the base settles to a constant after the first
  // two steps - set by the lead of the heuristics, the same at every step - at any heading and any command. The anchor it
  // replaced (the stance foot plus a whole step width) lets the midpoint fall behind by v T_step - k T_st v at EVERY
  // step, which is checked as the control: 0.3 * 0.5 - 0.25 * 0.3 = 0.075 m a step, 2.85 m after 40 steps.
  const std::unique_ptr<LocomotionHeuristicLayer> layer = raibertLayer();
  ASSERT_NE(layer, nullptr);
  const int numSteps = 40;
  for (const vector2_t& commandInHeading : {vector2_t(0.0, 0.3), vector2_t(0.0, -0.45), vector2_t(0.6, 0.2), vector2_t(-0.3, 0.0)}) {
    for (const scalar_t yaw : {0.0, 0.7, -2.6}) {
      const vector2_t command(std::cos(yaw) * commandInHeading.x() - std::sin(yaw) * commandInHeading.y(),
                              std::sin(yaw) * commandInHeading.x() + std::cos(yaw) * commandInHeading.y());
      const std::vector<vector2_t> midpoints = feetMinusBase(*layer, command, yaw, numSteps, Anchor::kFootholdAnchor);
      // From the second step on, both feet have been placed by the anchor: the midpoint minus the base is the mean of
      // the two feet's leads, constant step after step. Bounded is not enough, it must not move at all.
      for (int step = 2; step < numSteps; ++step) {
        EXPECT_LT((midpoints[step] - midpoints[1]).norm(), 1.0e-12) << "command " << commandInHeading.transpose() << ", yaw " << yaw
                                                                    << ", step " << step + 1 << ": " << midpoints[step].transpose();
      }
      // And the constant is what Raibert's rule predicts: the foot just landed leads its hip by k T_st v, the other one
      // landed a step earlier and has since fallen T_step v behind the base, so the midpoint sits k T_st v - T_step v / 2
      // from the base - zero for a half-stance lead on a gait whose stance is one step. The feet are not left behind and
      // not thrown ahead.
      const vector2_t expectedMidpoint = 0.5 * kStepDuration * commandInHeading - 0.5 * kStepDuration * commandInHeading;
      EXPECT_LT((midpoints[1] - expectedMidpoint).norm(), 1.0e-12) << "command " << commandInHeading.transpose() << ", yaw " << yaw;

      // Control: the stance-foot anchor drifts under any lateral command, without bound.
      const std::vector<vector2_t> drifting = feetMinusBase(*layer, command, yaw, numSteps, Anchor::kStanceFootPlusStepWidth);
      if (std::abs(commandInHeading.y()) > 0.0) {
        const scalar_t perStep = (kStepDuration - 0.25) * std::abs(commandInHeading.y());
        EXPECT_GT(std::abs(drifting.back().y() - drifting[1].y()), 0.9 * perStep * (numSteps - 2)) << "the control did not drift";
      }
    }
  }
}

TEST(LocomotionHeuristicSeams, TheAnchorIsHalfAStepWidthToTheFootsSideAcrossThePredictedHeading) {
  // The two hips at a touch-down are a step width apart and centered on the base, across the heading: rotating the
  // heading rotates them about the base, and the side sign mirrors them.
  for (const scalar_t yaw : {0.0, 0.4, 2.0, -1.3}) {
    const vector2_t base(1.2, -0.7);
    const vector2_t left = SwitchedModelReferenceManager::footholdAnchor(base, yaw, /*side=*/1.0, kStepWidth);
    const vector2_t right = SwitchedModelReferenceManager::footholdAnchor(base, yaw, /*side=*/-1.0, kStepWidth);
    EXPECT_LT((0.5 * (left + right) - base).norm(), 1.0e-15) << "yaw " << yaw;
    EXPECT_NEAR((left - right).norm(), kStepWidth, 1.0e-15) << "yaw " << yaw;
    const vector2_t leftInHeading = inHeadingFrame(yaw, left - base);
    EXPECT_NEAR(leftInHeading.x(), 0.0, 1.0e-15) << "yaw " << yaw;
    EXPECT_NEAR(leftInHeading.y(), 0.5 * kStepWidth, 1.0e-15) << "the left hip is to the robot's left, yaw " << yaw;
  }
}

/** R = R_z(yaw) R_y(pitch) R_x(roll) of ZYX Euler angles (yaw, pitch, roll), from elementary rotations. */
matrix3_t rotationOf(const vector3_t& eulerZyx) {
  return (Eigen::AngleAxis<scalar_t>(eulerZyx(0), vector3_t::UnitZ()) * Eigen::AngleAxis<scalar_t>(eulerZyx(1), vector3_t::UnitY()) *
          Eigen::AngleAxis<scalar_t>(eulerZyx(2), vector3_t::UnitX()))
      .toRotationMatrix();
}

/** The vector of a skew-symmetric matrix. */
vector3_t vee(const matrix3_t& skew) {
  return vector3_t(0.5 * (skew(2, 1) - skew(1, 2)), 0.5 * (skew(0, 2) - skew(2, 0)), 0.5 * (skew(1, 0) - skew(0, 1)));
}

/** An attitude trajectory and its rate, as ZYX Euler angles: Theta(t) = a + b t + c sin(w t). */
struct EulerTrajectory {
  vector3_t constant;
  vector3_t linear;
  vector3_t amplitude;
  scalar_t frequency = 0.0;

  vector3_t at(scalar_t time) const { return constant + linear * time + amplitude * std::sin(frequency * time); }
  vector3_t rateAt(scalar_t time) const { return linear + amplitude * frequency * std::cos(frequency * time); }
};

TEST(LocomotionHeuristicSeams, TheShapedAngularVelocityIsTheDerivativeOfTheShapedAttitude) {
  // The nominal attitude Theta_n(t) turns, pitches and rolls; the offset delta(t) tilts it in pitch and roll at gait
  // frequency. The shaped attitude Theta_n + delta must rotate at w_B = vee(R' dR/dt), measured by a central difference
  // of R_z R_y R_x - and its world rate must exceed the nominal's by tiltAngularVelocityInWorld() wherever the nominal
  // does not roll.
  const scalar_t step = 1.0e-6;
  const std::vector<EulerTrajectory> nominals = {
      {.constant = vector3_t(0.3, 0.05, -0.04), .linear = vector3_t(0.8, 0.0, 0.0), .amplitude = vector3_t::Zero(), .frequency = 0.0},
      {.constant = vector3_t(-2.1, 0.1, 0.2),
       .linear = vector3_t(-0.5, 0.03, 0.0),
       .amplitude = vector3_t(0.1, 0.05, 0.0),
       .frequency = 3.0},
      {.constant = vector3_t(1.0, -0.2, 0.1), .linear = vector3_t(0.2, 0.0, 0.3), .amplitude = vector3_t(0.0, 0.1, 0.1), .frequency = 5.0},
  };
  const EulerTrajectory offset{
      .constant = vector3_t(0.0, 0.02, -0.01), .linear = vector3_t::Zero(), .amplitude = vector3_t(0.0, 0.03, 0.05), .frequency = 7.0};
  for (size_t index = 0; index < nominals.size(); ++index) {
    const EulerTrajectory& nominal = nominals[index];
    for (const scalar_t time : {0.0, 0.37, 1.1}) {
      const vector3_t nominalAngles = nominal.at(time);
      const vector3_t shapedAngles = nominalAngles + offset.at(time);
      const vector3_t nominalBodyRate = localAngularVelocityFromEulerZyxRates(nominalAngles, nominal.rateAt(time));
      const vector3_t shaped =
          SwitchedModelReferenceManager::shapedBodyAngularVelocity(nominalAngles, nominalBodyRate, shapedAngles, offset.rateAt(time));

      const matrix3_t rotation = rotationOf(shapedAngles);
      const matrix3_t rotationRate =
          (rotationOf(nominal.at(time + step) + offset.at(time + step)) - rotationOf(nominal.at(time - step) + offset.at(time - step))) /
          (2.0 * step);
      const vector3_t expected = vee(rotation.transpose() * rotationRate);
      EXPECT_LT((shaped - expected).norm(), 1.0e-8)
          << "nominal " << index << ", t " << time << ": " << shaped.transpose() << " vs " << expected.transpose();

      // The world rate the tilt adds, where the nominal does not roll (its roll rate is zero on the first trajectory).
      if (std::abs(nominal.rateAt(time)(2)) == 0.0) {
        const matrix3_t nominalRotationRate = (rotationOf(nominal.at(time + step)) - rotationOf(nominal.at(time - step))) / (2.0 * step);
        const vector3_t nominalWorldRate = vee(nominalRotationRate * rotationOf(nominalAngles).transpose());
        const vector3_t shapedWorldRate = vee(rotationRate * rotation.transpose());
        const vector3_t tilt = SwitchedModelReferenceManager::tiltAngularVelocityInWorld(shapedAngles, offset.rateAt(time));
        EXPECT_LT((tilt - (shapedWorldRate - nominalWorldRate)).norm(), 1.0e-8) << "nominal " << index << ", t " << time;
      }
    }
  }
}

TEST(LocomotionHeuristicSeams, AConstantTiltKeepsTheNominalsTurnInTheWorld) {
  // No offset rate, a nominal that only turns: the shaped attitude turns about the world z axis at the nominal's rate,
  // which is what the seam promised before it carried tilt rates, so a constant offset changes nothing about the turn.
  const vector3_t nominalAngles(0.9, 0.0, 0.0);
  const scalar_t yawRate = 0.6;
  const vector3_t nominalBodyRate = localAngularVelocityFromEulerZyxRates(nominalAngles, vector3_t(yawRate, 0.0, 0.0));
  const vector3_t shapedAngles(0.9, 0.08, -0.05);
  const vector3_t shaped =
      SwitchedModelReferenceManager::shapedBodyAngularVelocity(nominalAngles, nominalBodyRate, shapedAngles, vector3_t::Zero());
  EXPECT_LT((rotationOf(shapedAngles) * shaped - vector3_t(0.0, 0.0, yawRate)).norm(), 1.0e-15);
  EXPECT_TRUE(SwitchedModelReferenceManager::tiltAngularVelocityInWorld(shapedAngles, vector3_t::Zero()).isZero(0.0));
}

}  // namespace
}  // namespace ocs2::humanoid
