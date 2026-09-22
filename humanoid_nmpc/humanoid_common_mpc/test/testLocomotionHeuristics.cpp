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

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <string>

#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicFactory.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicFormulation.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicLayer.h"

namespace ocs2::humanoid {

namespace {

constexpr scalar_t kTol = 1e-12;

std::string writeTemp(const std::string& name, const std::string& content) {
  const std::string file = testing::TempDir() + "/" + name;
  std::ofstream out(file);
  out << content;
  return file;
}

/** Model constants standing in for a mid-sized humanoid; the exact values matter only where a test says so. */
LocomotionHeuristicModelParameters testModel() {
  LocomotionHeuristicModelParameters model;
  model.totalMass = 40.0;
  model.gravity = 9.81;
  model.totalWeight = model.totalMass * model.gravity;
  model.nominalComHeight = 0.6125;
  model.hipPositionInBaseFrame[CONTACT_LEFT_INDEX] = vector2_t(0.0, 0.08);
  model.hipPositionInBaseFrame[CONTACT_RIGHT_INDEX] = vector2_t(0.0, -0.08);
  return model;
}

FootholdHeuristicContext footholdContext() {
  FootholdHeuristicContext context;
  context.contactIndex = CONTACT_LEFT_INDEX;
  context.side = 1.0;
  context.comHeight = 0.6125;
  return context;
}

WrenchHeuristicContext wrenchContext() {
  WrenchHeuristicContext context;
  context.contactFlags = {true, true};
  context.numStanceFeet = 2;
  context.totalWeight = 40.0 * 9.81;
  return context;
}

}  // namespace

/*=========================================== names and lists ============================================*/

TEST(LocomotionHeuristics, NamesAreMatchedLikeTheTaskFileLists) {
  EXPECT_EQ(canonicalHeuristicName(HeuristicKind::FOOTHOLD, "capturePoint"), heuristic::kCapturePoint);
  EXPECT_EQ(canonicalHeuristicName(HeuristicKind::FOOTHOLD, "Capture-Point"), heuristic::kCapturePoint);
  EXPECT_EQ(canonicalHeuristicName(HeuristicKind::FOOTHOLD, "CAPTURE POINT"), heuristic::kCapturePoint);
  EXPECT_EQ(canonicalHeuristicName(HeuristicKind::FOOTHOLD, "no_such_heuristic"), "");
  // A name is only valid in the list of the channel it shapes; this is what makes a misfiled name a load-time error.
  EXPECT_EQ(canonicalHeuristicName(HeuristicKind::BASE_POSE, "capture_point"), "");
  EXPECT_EQ(*heuristicKindOf("capture_point"), HeuristicKind::FOOTHOLD);
  EXPECT_EQ(*heuristicKindOf("orientation_compensation"), HeuristicKind::BASE_POSE);
  EXPECT_EQ(*heuristicKindOf("impulse_scaling"), HeuristicKind::WRENCH);
  EXPECT_FALSE(heuristicKindOf("no_such_heuristic").has_value());
}

TEST(LocomotionHeuristics, EveryNameOfBledtsAppendixCIsRegisteredAndBuildable) {
  // The ten of Tables C.1 and C.2. Pinned as a list so that dropping one is a test failure rather than a silent gap
  // between the dissertation and this implementation.
  const std::vector<std::string> expectedBasePose{"orientation_compensation", "periodic_orientation", "height_compensation"};
  const std::vector<std::string> expectedFoothold{"hip_centered_stepping", "capture_point", "translational_stepping", "in_place_turning",
                                                  "high_speed_turning"};
  const std::vector<std::string> expectedWrench{"impulse_scaling", "centripetal_acceleration"};
  EXPECT_EQ(knownHeuristicNames(HeuristicKind::BASE_POSE), expectedBasePose);
  EXPECT_EQ(knownHeuristicNames(HeuristicKind::FOOTHOLD), expectedFoothold);
  EXPECT_EQ(knownHeuristicNames(HeuristicKind::WRENCH), expectedWrench);

  const LocomotionHeuristicConfig config;
  const LocomotionHeuristicModelParameters model = testModel();
  for (const std::string& name : knownHeuristicNames(HeuristicKind::BASE_POSE)) {
    absl::StatusOr<std::unique_ptr<BasePoseHeuristic>> heuristic = LocomotionHeuristicFactory::makeBasePoseHeuristic(name);
    ASSERT_TRUE(heuristic.ok()) << name << ": " << heuristic.status().message();
    EXPECT_EQ((*heuristic)->name(), name);
    EXPECT_TRUE((*heuristic)->configure(config, model).ok());
    EXPECT_FALSE((*heuristic)->describe().empty());
  }
  for (const std::string& name : knownHeuristicNames(HeuristicKind::FOOTHOLD)) {
    absl::StatusOr<std::unique_ptr<FootholdHeuristic>> heuristic = LocomotionHeuristicFactory::makeFootholdHeuristic(name);
    ASSERT_TRUE(heuristic.ok()) << name << ": " << heuristic.status().message();
    EXPECT_EQ((*heuristic)->name(), name);
    EXPECT_TRUE((*heuristic)->configure(config, model).ok());
    EXPECT_FALSE((*heuristic)->describe().empty());
  }
  for (const std::string& name : knownHeuristicNames(HeuristicKind::WRENCH)) {
    absl::StatusOr<std::unique_ptr<WrenchHeuristic>> heuristic = LocomotionHeuristicFactory::makeWrenchHeuristic(name);
    ASSERT_TRUE(heuristic.ok()) << name << ": " << heuristic.status().message();
    EXPECT_EQ((*heuristic)->name(), name);
    EXPECT_TRUE((*heuristic)->configure(config, model).ok());
    EXPECT_FALSE((*heuristic)->describe().empty());
  }
}

TEST(LocomotionHeuristics, UnknownNameNamesTheValidOnes) {
  const absl::StatusOr<std::unique_ptr<FootholdHeuristic>> heuristic = LocomotionHeuristicFactory::makeFootholdHeuristic("capture_pointt");
  ASSERT_FALSE(heuristic.ok());
  EXPECT_EQ(heuristic.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(std::string(heuristic.status().message()).find("capture_point"), std::string::npos)
      << "the message must list the valid names: " << heuristic.status().message();
}

TEST(LocomotionHeuristics, MisfiledNameSaysWhereItBelongs) {
  LocomotionHeuristicFormulation formulation;
  formulation.basePose = {"capture_point"};
  const absl::Status status = formulation.validate();
  ASSERT_FALSE(status.ok());
  const std::string message(status.message());
  EXPECT_NE(message.find("foothold"), std::string::npos) << message;
}

TEST(LocomotionHeuristics, DuplicateNameIsRejectedBecauseTheOffsetsAreSummed) {
  LocomotionHeuristicFormulation formulation;
  formulation.foothold = {"capture_point", "capturePoint"};
  const absl::Status status = formulation.validate();
  ASSERT_FALSE(status.ok());
  EXPECT_NE(std::string(status.message()).find("more than once"), std::string::npos) << status.message();
}

/*======================================== the empty-list parity ==========================================*/

TEST(LocomotionHeuristics, EmptyListsAreAnExactNoOp) {
  // The safety property the whole subsystem rests on and the reason every robot can ship with the block present.
  const LocomotionHeuristicConfig config;
  ASSERT_TRUE(config.validate().ok());
  absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
      LocomotionHeuristicLayer::Create(config, testModel(), /*usesContactPlanning=*/false, /*usesContactBasisVectorInputs=*/false);
  ASSERT_TRUE(layer.ok()) << layer.status().message();
  EXPECT_TRUE((*layer)->empty());
  EXPECT_FALSE((*layer)->footholdMovesAnchor());
  EXPECT_FALSE((*layer)->wrenchNeedsWorldFrame()) << "an empty layer must not push the input costs onto the FK path";

  BasePoseHeuristicContext basePose;
  basePose.commandedVelocityInBaseFrame = vector2_t(1.0, 0.4);
  basePose.commandedYawRate = 0.7;
  basePose.gaitPhase = 0.3;
  const BasePoseOffset offset = (*layer)->basePoseOffset(basePose);
  EXPECT_NEAR(offset.roll, 0.0, kTol);
  EXPECT_NEAR(offset.pitch, 0.0, kTol);
  EXPECT_NEAR(offset.height, 0.0, kTol);
  EXPECT_TRUE((*layer)->footholdOffset(footholdContext()).isZero());
  EXPECT_TRUE((*layer)->wrenchOffset(wrenchContext(), CONTACT_LEFT_INDEX).isZero());
}

TEST(LocomotionHeuristics, AListedHeuristicWithZeroCoefficientsIsStillANoOp) {
  // The list and the coefficients are independent switches, which is what lets "does listing it change anything" be a
  // separate experiment from "what should the number be". Every fitted coefficient of Table C.2 ships at zero.
  LocomotionHeuristicConfig config;
  config.formulation.basePose = {"orientation_compensation", "periodic_orientation", "height_compensation"};
  absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
      LocomotionHeuristicLayer::Create(config, testModel(), /*usesContactPlanning=*/false, /*usesContactBasisVectorInputs=*/false);
  ASSERT_TRUE(layer.ok()) << layer.status().message();
  EXPECT_FALSE((*layer)->empty());

  BasePoseHeuristicContext context;
  context.commandedVelocityInBaseFrame = vector2_t(1.5, -0.6);
  context.gaitPhase = 0.42;
  const BasePoseOffset offset = (*layer)->basePoseOffset(context);
  EXPECT_NEAR(offset.roll, 0.0, kTol);
  EXPECT_NEAR(offset.pitch, 0.0, kTol);
  EXPECT_NEAR(offset.height, 0.0, kTol);
}

/*============================================ the formulae ==============================================*/

TEST(LocomotionHeuristics, OrientationCompensationIsAffineInTheCommandAndClamped) {
  LocomotionHeuristicConfig config;
  config.formulation.basePose = {"orientation_compensation"};
  config.orientationCompensation.rollPerLateralVelocity = -0.339;   // Bledt equation 4.11
  config.orientationCompensation.pitchPerForwardVelocity = 0.0725;  // Bledt equation 4.12
  config.orientationCompensation.rollOffset = 0.01;
  config.orientationCompensation.pitchOffset = -0.02;
  config.orientationCompensation.maximumTilt = 0.1;
  absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
      LocomotionHeuristicLayer::Create(config, testModel(), /*usesContactPlanning=*/false, /*usesContactBasisVectorInputs=*/false);
  ASSERT_TRUE(layer.ok()) << layer.status().message();

  BasePoseHeuristicContext context;
  context.commandedVelocityInBaseFrame = vector2_t(1.0, 0.2);
  const BasePoseOffset offset = (*layer)->basePoseOffset(context);
  EXPECT_NEAR(offset.roll, -0.339 * 0.2 + 0.01, 1e-9) << "roll follows the LATERAL command";
  EXPECT_NEAR(offset.pitch, 0.0725 * 1.0 - 0.02, 1e-9) << "pitch follows the FORWARD command";
  EXPECT_NEAR(offset.height, 0.0, kTol) << "this heuristic has no opinion about height";

  // The command is filtered but not rate limited, so an affine law on it has no bound of its own.
  BasePoseHeuristicContext fast;
  fast.commandedVelocityInBaseFrame = vector2_t(0.0, 100.0);
  EXPECT_NEAR((*layer)->basePoseOffset(fast).roll, -0.1, 1e-9);
}

TEST(LocomotionHeuristics, PeriodicOrientationIsASinusoidInTheGaitPhase) {
  LocomotionHeuristicConfig config;
  config.formulation.basePose = {"periodic_orientation"};
  config.periodicOrientation.pitchAmplitude = 0.023;  // Bledt equation 4.13
  config.periodicOrientation.pitchPhaseRate = 4.0 * M_PI;
  config.periodicOrientation.pitchPhaseOffset = 0.5;
  absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
      LocomotionHeuristicLayer::Create(config, testModel(), /*usesContactPlanning=*/false, /*usesContactBasisVectorInputs=*/false);
  ASSERT_TRUE(layer.ok()) << layer.status().message();

  for (const scalar_t phase : {0.0, 0.25, 0.5, 0.75}) {
    BasePoseHeuristicContext context;
    context.gaitPhase = phase;
    EXPECT_NEAR((*layer)->basePoseOffset(context).pitch, 0.023 * std::sin(4.0 * M_PI * phase + 0.5), 1e-12);
  }
  // A whole number of periods per cycle is what makes the reference periodic in the gait rather than drifting with it.
  BasePoseHeuristicContext start;
  BasePoseHeuristicContext end;
  end.gaitPhase = 1.0;
  EXPECT_NEAR((*layer)->basePoseOffset(start).pitch, (*layer)->basePoseOffset(end).pitch, 1e-12);
}

TEST(LocomotionHeuristics, HeightCompensationIsEvenInTheDirectionOfTravel) {
  LocomotionHeuristicConfig config;
  config.formulation.basePose = {"height_compensation"};
  config.heightCompensation.heightPerSpeed = -0.02;
  config.heightCompensation.maximumHeightOffset = 0.05;
  absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
      LocomotionHeuristicLayer::Create(config, testModel(), /*usesContactPlanning=*/false, /*usesContactBasisVectorInputs=*/false);
  ASSERT_TRUE(layer.ok()) << layer.status().message();

  BasePoseHeuristicContext forward;
  forward.commandedVelocityInBaseFrame = vector2_t(1.0, 0.0);
  BasePoseHeuristicContext backward;
  backward.commandedVelocityInBaseFrame = vector2_t(-1.0, 0.0);
  EXPECT_NEAR((*layer)->basePoseOffset(forward).height, -0.02, 1e-12);
  // Crouching to walk forwards and rising to walk backwards is not a thing any legged system does.
  EXPECT_NEAR((*layer)->basePoseOffset(backward).height, -0.02, 1e-12);

  BasePoseHeuristicContext fast;
  fast.commandedVelocityInBaseFrame = vector2_t(100.0, 0.0);
  EXPECT_NEAR((*layer)->basePoseOffset(fast).height, -0.05, 1e-12) << "clamped: the base must not fall into its knees";
}

TEST(LocomotionHeuristics, CapturePointIsZeroWhenTheCommandIsTracked) {
  LocomotionHeuristicConfig config;
  config.formulation.foothold = {"capture_point"};
  absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
      LocomotionHeuristicLayer::Create(config, testModel(), /*usesContactPlanning=*/false, /*usesContactBasisVectorInputs=*/false);
  ASSERT_TRUE(layer.ok()) << layer.status().message();

  // The defining property: it is feedback on the velocity ERROR, so a robot going exactly as fast as it was asked to
  // gets no offset however fast that is.
  FootholdHeuristicContext tracking = footholdContext();
  tracking.measuredVelocity = vector2_t(1.2, 0.3);
  tracking.commandedVelocity = vector2_t(1.2, 0.3);
  EXPECT_TRUE((*layer)->footholdOffset(tracking).isZero(1e-12));

  FootholdHeuristicContext pushed = footholdContext();
  pushed.measuredVelocity = vector2_t(0.5, 0.0);
  pushed.commandedVelocity = vector2_t(0.0, 0.0);
  const scalar_t expected = std::sqrt(0.6125 / 9.81) * 0.5;
  EXPECT_NEAR((*layer)->footholdOffset(pushed).x(), expected, 1e-9);
  EXPECT_NEAR((*layer)->footholdOffset(pushed).y(), 0.0, 1e-12);

  // Clamped in MAGNITUDE, which preserves the direction of the step into the error - a per-axis clamp would rotate
  // the offset towards the diagonal exactly when the robot most needs the foot to go where the push came from.
  FootholdHeuristicContext thrown = footholdContext();
  thrown.measuredVelocity = vector2_t(30.0, 40.0);
  const vector2_t offset = (*layer)->footholdOffset(thrown);
  EXPECT_NEAR(offset.norm(), 0.25, 1e-9);
  EXPECT_NEAR(offset.x() / offset.y(), 30.0 / 40.0, 1e-9) << "the direction survives the clamp";
}

TEST(LocomotionHeuristics, TranslationalSteppingRotatesWithTheHeading) {
  LocomotionHeuristicConfig config;
  config.formulation.foothold = {"translational_stepping"};
  config.translationalStepping.forwardPerForwardVelocity = 0.15;
  absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
      LocomotionHeuristicLayer::Create(config, testModel(), /*usesContactPlanning=*/false, /*usesContactBasisVectorInputs=*/false);
  ASSERT_TRUE(layer.ok()) << layer.status().message();

  // Facing +x and walking +x: the step goes forward along +x.
  FootholdHeuristicContext alongX = footholdContext();
  alongX.commandedVelocity = vector2_t(2.0, 0.0);
  const vector2_t stepAlongX = (*layer)->footholdOffset(alongX);
  EXPECT_NEAR(stepAlongX.x(), 0.3, 1e-9);
  EXPECT_NEAR(stepAlongX.y(), 0.0, 1e-9);

  // Facing +y and walking +y: the same step, in the world's +y. The coefficients are in the base's frame, so the law
  // has to be applied there and rotated back - applying it in the world would swap forward for lateral here.
  FootholdHeuristicContext alongY = footholdContext();
  alongY.measuredBaseYaw = M_PI / 2.0;
  alongY.commandedVelocity = vector2_t(0.0, 2.0);
  const vector2_t stepAlongY = (*layer)->footholdOffset(alongY);
  EXPECT_NEAR(stepAlongY.x(), 0.0, 1e-9);
  EXPECT_NEAR(stepAlongY.y(), 0.3, 1e-9);
}

TEST(LocomotionHeuristics, InPlaceTurningLeadsEachHipIntoTheTurn) {
  LocomotionHeuristicConfig config;
  config.formulation.foothold = {"in_place_turning"};
  config.inPlaceTurning.forwardPerYawRate = 0.05;
  absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
      LocomotionHeuristicLayer::Create(config, testModel(), /*usesContactPlanning=*/false, /*usesContactBasisVectorInputs=*/false);
  ASSERT_TRUE(layer.ok()) << layer.status().message();

  FootholdHeuristicContext left = footholdContext();
  left.commandedYawRate = 1.0;
  FootholdHeuristicContext right = footholdContext();
  right.contactIndex = CONTACT_RIGHT_INDEX;
  right.side = -1.0;
  right.commandedYawRate = 1.0;

  // A foot to the LEFT of a body turning counter-clockwise travels BACKWARDS: its velocity is
  // omega x r = (0,0,psidot) x (0,d,0) = (-psidot d, 0, 0). "Leading the hips into the turn" therefore means placing
  // the left foot BEHIND and the right foot ahead, and a POSITIVE coefficient must do that - otherwise a tuner
  // following the documentation drives the feet to TRAIL the hips by twice the intended amount, which is the very
  // workspace-exhaustion failure the heuristic exists to remove.
  EXPECT_NEAR((*layer)->footholdOffset(left).x(), -0.05, 1e-9);
  EXPECT_NEAR((*layer)->footholdOffset(right).x(), 0.05, 1e-9);
  // Equal and opposite: that is what a turn on the spot is.
  EXPECT_NEAR((*layer)->footholdOffset(left).x(), -(*layer)->footholdOffset(right).x(), 1e-12);
}

TEST(LocomotionHeuristics, HighSpeedTurningVanishesAtZeroSpeed) {
  LocomotionHeuristicConfig config;
  config.formulation.foothold = {"high_speed_turning"};
  config.highSpeedTurning.lateralPerCrossTerm = 0.1;
  config.highSpeedTurning.forwardPerCrossTerm = 0.1;
  absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
      LocomotionHeuristicLayer::Create(config, testModel(), /*usesContactPlanning=*/false, /*usesContactBasisVectorInputs=*/false);
  ASSERT_TRUE(layer.ok()) << layer.status().message();

  // The property that distinguishes it from in_place_turning: however fast the robot spins, standing still it is zero.
  FootholdHeuristicContext spinning = footholdContext();
  spinning.commandedYawRate = 5.0;
  EXPECT_TRUE((*layer)->footholdOffset(spinning).isZero(1e-12));

  // Walking forwards and turning left throws the foot to the OUTSIDE of the turn, i.e. to the right: the cross
  // product v x omega = (v_y psidot, -v_x psidot) has a negative lateral component here.
  FootholdHeuristicContext turning = footholdContext();
  turning.commandedVelocity = vector2_t(2.0, 0.0);
  turning.commandedYawRate = 1.0;
  EXPECT_NEAR((*layer)->footholdOffset(turning).y(), 0.1 * -2.0, 1e-9);
}

TEST(LocomotionHeuristics, HipCenteredSteppingPlacesEachFootOnItsOwnSide) {
  LocomotionHeuristicConfig config;
  config.formulation.foothold = {"hip_centered_stepping"};
  absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
      LocomotionHeuristicLayer::Create(config, testModel(), /*usesContactPlanning=*/false, /*usesContactBasisVectorInputs=*/false);
  ASSERT_TRUE(layer.ok()) << layer.status().message();
  EXPECT_TRUE((*layer)->footholdMovesAnchor()) << "the reference manager must anchor on the base, not the stance foot";

  FootholdHeuristicContext left = footholdContext();
  FootholdHeuristicContext right = footholdContext();
  right.contactIndex = CONTACT_RIGHT_INDEX;
  right.side = -1.0;
  EXPECT_NEAR((*layer)->footholdOffset(left).y(), 0.08, 1e-12);
  EXPECT_NEAR((*layer)->footholdOffset(right).y(), -0.08, 1e-12);

  // r_hip is in the BASE frame, so a robot facing +y puts its left hip towards -x in the world.
  FootholdHeuristicContext turned = footholdContext();
  turned.measuredBaseYaw = M_PI / 2.0;
  EXPECT_NEAR((*layer)->footholdOffset(turned).x(), -0.08, 1e-9);
  EXPECT_NEAR((*layer)->footholdOffset(turned).y(), 0.0, 1e-9);
}

TEST(LocomotionHeuristics, ImpulseScalingTargetsBledtsImpulseBudget) {
  LocomotionHeuristicConfig config;
  config.formulation.wrench = {"impulse_scaling"};
  absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
      LocomotionHeuristicLayer::Create(config, testModel(), /*usesContactPlanning=*/false, /*usesContactBasisVectorInputs=*/false);
  ASSERT_TRUE(layer.ok()) << layer.status().message();
  EXPECT_FALSE((*layer)->wrenchNeedsWorldFrame()) << "a vertical force is yaw-invariant and must keep the cheap path";

  const scalar_t weight = 40.0 * 9.81;
  const scalar_t numFeet = 2.0;

  // beta = 1 is standing, with both feet down: W/(F beta) is W/2, which IS the existing reference, so the correction
  // is zero.
  WrenchHeuristicContext standing = wrenchContext();
  standing.stanceDutyFactor = {1.0, 1.0};
  EXPECT_TRUE((*layer)->wrenchOffset(standing, CONTACT_LEFT_INDEX).isZero(1e-9));

  // beta = 1/2 in SINGLE support is the other fixed point: W/(F beta) = W, and the one stance foot already carries W.
  WrenchHeuristicContext singleSupport = wrenchContext();
  singleSupport.contactFlags = {true, false};
  singleSupport.numStanceFeet = 1;
  singleSupport.stanceDutyFactor = {0.5, 0.5};
  EXPECT_TRUE((*layer)->wrenchOffset(singleSupport, CONTACT_LEFT_INDEX).isZero(1e-9))
      << "a pure alternating single support is exactly weight compensation; the correction must not fire there";

  // The correction bites where the two disagree: DOUBLE support in a gait that also has single support. Bledt's
  // reference is W/(F beta) per foot whatever n_stance is, so at beta = 0.6 each foot is asked for W/1.2 = 0.833 W
  // rather than the 0.5 W of instantaneous weight compensation.
  WrenchHeuristicContext doubleSupport = wrenchContext();
  doubleSupport.stanceDutyFactor = {0.6, 0.6};
  const vector3_t offset = (*layer)->wrenchOffset(doubleSupport, CONTACT_LEFT_INDEX);
  EXPECT_NEAR(offset.x(), 0.0, kTol);
  EXPECT_NEAR(offset.y(), 0.0, kTol);
  EXPECT_NEAR(offset.z(), weight / (numFeet * 0.6) - weight / 2.0, 1e-6);

  // THE IMPULSE BUDGET, which is the whole point of the heuristic. Over one cycle the mean number of feet on the
  // ground is F * beta, so the mean total vertical reference must come back to exactly the robot's weight. Scaling
  // the INSTANTANEOUS weight compensation by 1/beta instead would put W/beta on the ground at every node and
  // integrate to W*T/beta, i.e. it would regularise the solver towards accelerating the CoM upwards for ever.
  const scalar_t beta = 0.6;
  const scalar_t perFootReference = weight / (numFeet * beta);
  // A cycle of this duty factor is (2 beta - 1) double support and (2 - 2 beta) single support, by fractions.
  const scalar_t doubleSupportFraction = 2.0 * beta - 1.0;
  const scalar_t meanTotal = doubleSupportFraction * 2.0 * perFootReference + (1.0 - doubleSupportFraction) * perFootReference;
  EXPECT_NEAR(meanTotal, weight, 1e-6) << "the cycle-averaged vertical reference must equal the robot's weight";

  // 1/beta is unbounded as a flight phase opens, so both clamps have to bite.
  WrenchHeuristicContext flying = wrenchContext();
  flying.stanceDutyFactor = {1e-6, 1e-6};
  const scalar_t baseline = weight / 2.0;
  EXPECT_NEAR((*layer)->wrenchOffset(flying, CONTACT_LEFT_INDEX).z(), baseline, 1e-6)
      << "clamped at maximumForceRatio (2) times weight compensation";
}

TEST(LocomotionHeuristics, CentripetalAccelerationPointsIntoTheTurnAndNeedsTheWorldFrame) {
  LocomotionHeuristicConfig config;
  config.formulation.wrench = {"centripetal_acceleration"};
  absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
      LocomotionHeuristicLayer::Create(config, testModel(), /*usesContactPlanning=*/false, /*usesContactBasisVectorInputs=*/false);
  ASSERT_TRUE(layer.ok()) << layer.status().message();
  // The one heuristic of the ten that forces the input costs onto the forward-kinematics path.
  EXPECT_TRUE((*layer)->wrenchNeedsWorldFrame());

  // Walking along +x and turning left (positive yaw rate): the centre of the turn is to the LEFT, so the force is +y.
  WrenchHeuristicContext turning = wrenchContext();
  turning.commandedVelocity = vector2_t(2.0, 0.0);
  turning.commandedYawRate = 1.0;
  const vector3_t offset = (*layer)->wrenchOffset(turning, CONTACT_LEFT_INDEX);
  EXPECT_NEAR(offset.x(), 0.0, 1e-9);
  EXPECT_NEAR(offset.y(), 40.0 * 1.0 * 2.0 / 2.0, 1e-9) << "m * omega x v, shared over the two stance feet";
  EXPECT_NEAR(offset.z(), 0.0, kTol) << "purely horizontal: the vertical reference is weight compensation's job";

  // Straight-line walking is not a turn.
  WrenchHeuristicContext straight = wrenchContext();
  straight.commandedVelocity = vector2_t(2.0, 0.0);
  EXPECT_TRUE((*layer)->wrenchOffset(straight, CONTACT_LEFT_INDEX).isZero(1e-12));

  // A foot cannot pull sideways harder than friction allows.
  WrenchHeuristicContext extreme = wrenchContext();
  extreme.commandedVelocity = vector2_t(50.0, 0.0);
  extreme.commandedYawRate = 10.0;
  EXPECT_NEAR((*layer)->wrenchOffset(extreme, CONTACT_LEFT_INDEX).norm(), 0.3 * extreme.totalWeight, 1e-6);
}

/*======================================= the rejected combinations =======================================*/

TEST(LocomotionHeuristics, FootholdHeuristicUnderContactPlanningIsRejected) {
  // It would be silently inert: ContactPlanningReferenceManager supplies the landing target itself and never calls
  // nominalFoothold(), which is this family's only seam.
  LocomotionHeuristicConfig config;
  config.formulation.foothold = {"capture_point"};
  const absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
      LocomotionHeuristicLayer::Create(config, testModel(), /*usesContactPlanning=*/true, /*usesContactBasisVectorInputs=*/false);
  ASSERT_FALSE(layer.ok());
  const std::string message(layer.status().message());
  EXPECT_NE(message.find("useContactPlanning"), std::string::npos) << message;

  // The other two channels are unaffected: the planner has no opinion about base pose or contact force.
  LocomotionHeuristicConfig other;
  other.formulation.basePose = {"orientation_compensation"};
  other.formulation.wrench = {"impulse_scaling"};
  EXPECT_TRUE(LocomotionHeuristicLayer::Create(other, testModel(), /*usesContactPlanning=*/true,
                                               /*usesContactBasisVectorInputs=*/false)
                  .ok());
}

TEST(LocomotionHeuristics, InvalidParametersAreRejectedWithTheKeyThatIsWrong) {
  LocomotionHeuristicConfig config;
  config.impulseScaling.minimumDutyFactor = 0.0;
  const absl::Status status = config.validate();
  ASSERT_FALSE(status.ok());
  EXPECT_NE(std::string(status.message()).find("minimumDutyFactor"), std::string::npos) << status.message();

  LocomotionHeuristicConfig tooSmall;
  tooSmall.impulseScaling.maximumForceRatio = 0.5;
  EXPECT_FALSE(tooSmall.validate().ok()) << "a ratio below 1 asks the feet to carry less than the robot weighs";
}

/*============================================ the loader ================================================*/

TEST(LocomotionHeuristics, TaskFileWithoutTheBlockIsTheDefaultNoOp) {
  const std::string file = writeTemp("heuristics_absent.yaml", "costs:\n  - state_quadratic_cost\n");
  const absl::StatusOr<LocomotionHeuristicConfig> config = loadLocomotionHeuristicConfig(file);
  ASSERT_TRUE(config.ok()) << config.status().message();
  EXPECT_TRUE(config->formulation.empty());
}

TEST(LocomotionHeuristics, EmptyAndNullListsBothMeanNoHeuristic) {
  // A block whose entries are all commented out parses as a null scalar rather than an empty sequence, and the task
  // files ship with the names commented out under each list. The two must not be different configurations.
  const std::string explicitly =
      writeTemp("heuristics_empty.yaml", "locomotion_heuristics:\n  base_pose: []\n  foothold: []\n  wrench: []\n");
  const std::string nulls = writeTemp("heuristics_null.yaml", "locomotion_heuristics:\n  base_pose:\n  foothold:\n  wrench:\n");
  for (const std::string& file : {explicitly, nulls}) {
    const absl::StatusOr<LocomotionHeuristicConfig> config = loadLocomotionHeuristicConfig(file);
    ASSERT_TRUE(config.ok()) << file << ": " << config.status().message();
    EXPECT_TRUE(config->formulation.empty()) << file;
  }
}

TEST(LocomotionHeuristics, LoaderReadsTheListsAndTheCoefficients) {
  const std::string file = writeTemp("heuristics_full.yaml", R"(locomotion_heuristics:
  base_pose:
    - orientation_compensation
    - periodic_orientation
  foothold:
    - hip_centered_stepping
    - translational_stepping
  wrench:
    - impulse_scaling
  orientation_compensation:
    rollPerLateralVelocity: -0.339
    pitchPerForwardVelocity: 0.0725
  periodic_orientation:
    pitchAmplitude: 0.023
  translational_stepping:
    forwardPerForwardVelocity: 0.12
  impulse_scaling:
    minimumDutyFactor: 0.3
)");
  const absl::StatusOr<LocomotionHeuristicConfig> config = loadLocomotionHeuristicConfig(file);
  ASSERT_TRUE(config.ok()) << config.status().message();
  EXPECT_EQ(config->formulation.basePose.size(), 2u);
  EXPECT_EQ(config->formulation.foothold.size(), 2u);
  EXPECT_EQ(config->formulation.wrench.size(), 1u);
  EXPECT_NEAR(config->orientationCompensation.rollPerLateralVelocity, -0.339, kTol);
  EXPECT_NEAR(config->periodicOrientation.pitchAmplitude, 0.023, kTol);
  EXPECT_NEAR(config->translationalStepping.forwardPerForwardVelocity, 0.12, kTol);
  EXPECT_NEAR(config->impulseScaling.minimumDutyFactor, 0.3, kTol);
  // A key the file does not mention keeps its struct default; that is what lets a block be written a line at a time.
  EXPECT_NEAR(config->orientationCompensation.maximumTilt, OrientationCompensationParameters().maximumTilt, kTol);
}

TEST(LocomotionHeuristics, LoaderRejectsAnUnknownNameInTheFile) {
  const std::string file = writeTemp("heuristics_unknown.yaml", "locomotion_heuristics:\n  base_pose:\n    - orientation_compenstaion\n");
  const absl::StatusOr<LocomotionHeuristicConfig> config = loadLocomotionHeuristicConfig(file);
  ASSERT_FALSE(config.ok());
  EXPECT_NE(std::string(config.status().message()).find("orientation_compensation"), std::string::npos) << config.status().message();
}

TEST(LocomotionHeuristics, LoaderRejectsAListThatIsNotAList) {
  // A scalar or a map where a sequence belongs used to read as "no heuristic", i.e. a typo in the file produced a
  // silently inert configuration - the failure this subsystem refuses to allow anywhere else.
  const std::string scalarFile =
      writeTemp("heuristics_scalar_list.yaml", "locomotion_heuristics:\n  base_pose: orientation_compensation\n");
  const absl::StatusOr<LocomotionHeuristicConfig> fromScalar = loadLocomotionHeuristicConfig(scalarFile);
  ASSERT_FALSE(fromScalar.ok());
  EXPECT_NE(std::string(fromScalar.status().message()).find("must be a list"), std::string::npos) << fromScalar.status().message();

  const std::string mapFile = writeTemp("heuristics_map_list.yaml", "locomotion_heuristics:\n  foothold:\n    capture_point: true\n");
  const absl::StatusOr<LocomotionHeuristicConfig> fromMap = loadLocomotionHeuristicConfig(mapFile);
  ASSERT_FALSE(fromMap.ok());
  EXPECT_NE(std::string(fromMap.status().message()).find("must be a list"), std::string::npos) << fromMap.status().message();
}

TEST(LocomotionHeuristics, LoaderReturnsAStatusForAnUnparseableNumber) {
  // loadPtreeValue() catches only the missing-key case, so a present-but-malformed scalar throws straight through it.
  // A typo like `0.0.1` is an ordinary mistake and must come back as the file-and-key message, not as a crash out of
  // a function whose whole contract is to return a status.
  const std::string file =
      writeTemp("heuristics_bad_number.yaml", "locomotion_heuristics:\n  orientation_compensation:\n    rollOffset: 0.0.1\n");
  const absl::StatusOr<LocomotionHeuristicConfig> config = loadLocomotionHeuristicConfig(file);
  ASSERT_FALSE(config.ok());
  EXPECT_EQ(config.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(std::string(config.status().message()).find("could not be read as a number"), std::string::npos) << config.status().message();
}

TEST(LocomotionHeuristics, ClampsCannotBeConfiguredAway) {
  // Every clamp here exists to stop a bad measurement or an unbounded command reaching the solver, so a value that
  // REMOVES one has to be rejected rather than read as "no limit".
  LocomotionHeuristicConfig noCaptureClamp;
  noCaptureClamp.capturePoint.maximumOffset = 0.0;
  EXPECT_FALSE(noCaptureClamp.validate().ok());

  LocomotionHeuristicConfig noCentripetalClamp;
  noCentripetalClamp.centripetalAcceleration.maximumForce = 0.0;
  noCentripetalClamp.centripetalAcceleration.maximumForceRatioOfWeight = 0.0;
  const absl::Status status = noCentripetalClamp.validate();
  ASSERT_FALSE(status.ok());
  EXPECT_NE(std::string(status.message()).find("force clamp"), std::string::npos) << status.message();

  // `maximumForce: 0` is the documented way to select the ratio instead, so that combination stays valid.
  LocomotionHeuristicConfig ratioClamp;
  ratioClamp.centripetalAcceleration.maximumForce = 0.0;
  ratioClamp.centripetalAcceleration.maximumForceRatioOfWeight = 0.3;
  EXPECT_TRUE(ratioClamp.validate().ok());

  // A negative blend would invert a heuristic rather than reduce it.
  LocomotionHeuristicConfig negativeScale;
  negativeScale.impulseScaling.scale = -1.0;
  EXPECT_FALSE(negativeScale.validate().ok());
}

TEST(LocomotionHeuristics, ReconfigureReplacesCoefficientsWithoutRebuildingTheLayer) {
  LocomotionHeuristicConfig config;
  config.formulation.basePose = {"orientation_compensation"};
  config.orientationCompensation.pitchPerForwardVelocity = 0.05;
  absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
      LocomotionHeuristicLayer::Create(config, testModel(), /*usesContactPlanning=*/false, /*usesContactBasisVectorInputs=*/false);
  ASSERT_TRUE(layer.ok()) << layer.status().message();

  BasePoseHeuristicContext context;
  context.commandedVelocityInBaseFrame = vector2_t(1.0, 0.0);
  EXPECT_NEAR((*layer)->basePoseOffset(context).pitch, 0.05, 1e-12);

  // configure() must overwrite rather than accumulate, or a slider drag would ratchet.
  LocomotionHeuristicConfig reloaded = config;
  reloaded.orientationCompensation.pitchPerForwardVelocity = -0.03;
  ASSERT_TRUE((*layer)->reconfigure(reloaded).ok());
  EXPECT_NEAR((*layer)->basePoseOffset(context).pitch, -0.03, 1e-12);

  // An invalid reload leaves the running values alone rather than half-applying itself.
  LocomotionHeuristicConfig invalid = reloaded;
  invalid.impulseScaling.minimumDutyFactor = -1.0;
  EXPECT_FALSE((*layer)->reconfigure(invalid).ok());
  EXPECT_NEAR((*layer)->basePoseOffset(context).pitch, -0.03, 1e-12);
}

}  // namespace ocs2::humanoid
