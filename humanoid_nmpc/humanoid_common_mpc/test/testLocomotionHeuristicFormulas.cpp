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

// Unit tests of the locomotion-heuristic FORMULAE, the layer that sums them, and the loader that reads their
// coefficients - all with hand-built contexts and in-memory configurations, so that nothing here needs a robot model.
//
// Every expected number is derived by hand (the arithmetic is written out beside it) rather than by calling the code
// under test, and each case is chosen so that the specific regression named in its comment changes the result by far
// more than the tolerance. The evaluation points are deliberately generic - headings off the axes, velocities with
// both components non-zero, feet with different hips, coefficients that all differ - because the degenerate points
// (yaw 0, speed 1, symmetric feet, equal coefficients) are exactly where transposed rotations, dropped factors and
// swapped keys hide.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/log_severity.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicFactory.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicFormulation.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicLayer.h"

namespace ocs2::humanoid {

namespace {

constexpr scalar_t kTol = 1e-12;
constexpr scalar_t kMass = 40.0;
constexpr scalar_t kGravity = 9.81;
constexpr scalar_t kWeight = kMass * kGravity;  // 392.4 N

std::string writeTemp(const std::string& name, const std::string& content) {
  const std::string file = testing::TempDir() + "/" + name;
  std::ofstream out(file);
  out << content;
  return file;
}

/**
 * Model constants of a 40 kg robot. The two hips are deliberately NOT mirror images of each other, so that reading
 * the other foot's hip, or the right hip mirrored from the left, is a visible error rather than a coincidence.
 */
LocomotionHeuristicModelParameters testModel() {
  LocomotionHeuristicModelParameters model;
  model.totalMass = kMass;
  model.gravity = kGravity;
  model.totalWeight = kWeight;
  model.nominalComHeight = 0.6125;
  model.hipPositionInBaseFrame[CONTACT_LEFT_INDEX] = vector2_t(0.03, 0.08);
  model.hipPositionInBaseFrame[CONTACT_RIGHT_INDEX] = vector2_t(0.01, -0.09);
  return model;
}

/** An environment in which every listed heuristic can act: a positive step width and nothing that makes one inert. */
LocomotionHeuristicEnvironment steppingEnvironment() {
  LocomotionHeuristicEnvironment environment;
  environment.nominalStepWidth = 0.2;
  return environment;
}

FootholdHeuristicContext leftFoot() {
  FootholdHeuristicContext context;
  context.contactIndex = CONTACT_LEFT_INDEX;
  context.side = 1.0;
  context.comHeight = 0.6125;
  return context;
}

FootholdHeuristicContext rightFoot() {
  FootholdHeuristicContext context = leftFoot();
  context.contactIndex = CONTACT_RIGHT_INDEX;
  context.side = -1.0;
  return context;
}

WrenchHeuristicContext doubleSupport(scalar_t leftDutyFactor, scalar_t rightDutyFactor) {
  WrenchHeuristicContext context;
  context.contactFlags = {true, true};
  context.numStanceFeet = 2;
  context.stanceDutyFactor = {leftDutyFactor, rightDutyFactor};
  context.totalWeight = kWeight;
  return context;
}

WrenchHeuristicContext leftSingleSupport(scalar_t leftDutyFactor) {
  WrenchHeuristicContext context = doubleSupport(leftDutyFactor, leftDutyFactor);
  context.contactFlags = {true, false};
  context.numStanceFeet = 1;
  return context;
}

/*
 * One heuristic, built by the factory and configured, evaluated once. The formula tests go through these rather than
 * the layer so that a failure points at the formula; the layer's own contract (summing, flags) is tested separately.
 * A heuristic that cannot be built or configured is a test failure, and the NaN it returns then fails every
 * comparison made with it.
 */
BasePoseOffset basePoseOffsetOf(absl::string_view name, const LocomotionHeuristicConfig& config, const BasePoseHeuristicContext& context) {
  BasePoseOffset nan;
  nan.roll = nan.pitch = nan.height = std::numeric_limits<scalar_t>::quiet_NaN();
  absl::StatusOr<std::unique_ptr<BasePoseHeuristic>> heuristic = LocomotionHeuristicFactory::makeBasePoseHeuristic(name);
  if (!heuristic.ok()) {
    ADD_FAILURE() << heuristic.status().message();
    return nan;
  }
  const absl::Status configured = (*heuristic)->configure(config, testModel());
  if (!configured.ok()) {
    ADD_FAILURE() << configured.message();
    return nan;
  }
  return (*heuristic)->offset(context);
}

vector2_t footholdOffsetOf(absl::string_view name,
                           const LocomotionHeuristicConfig& config,
                           const LocomotionHeuristicModelParameters& model,
                           const FootholdHeuristicContext& context) {
  const vector2_t nan = vector2_t::Constant(std::numeric_limits<scalar_t>::quiet_NaN());
  absl::StatusOr<std::unique_ptr<FootholdHeuristic>> heuristic = LocomotionHeuristicFactory::makeFootholdHeuristic(name);
  if (!heuristic.ok()) {
    ADD_FAILURE() << heuristic.status().message();
    return nan;
  }
  const absl::Status configured = (*heuristic)->configure(config, model);
  if (!configured.ok()) {
    ADD_FAILURE() << configured.message();
    return nan;
  }
  return (*heuristic)->offset(context);
}

vector2_t footholdOffsetOf(absl::string_view name, const LocomotionHeuristicConfig& config, const FootholdHeuristicContext& context) {
  return footholdOffsetOf(name, config, testModel(), context);
}

vector3_t wrenchOffsetOf(absl::string_view name,
                         const LocomotionHeuristicConfig& config,
                         const WrenchHeuristicContext& context,
                         size_t contactIndex) {
  const vector3_t nan = vector3_t::Constant(std::numeric_limits<scalar_t>::quiet_NaN());
  absl::StatusOr<std::unique_ptr<WrenchHeuristic>> heuristic = LocomotionHeuristicFactory::makeWrenchHeuristic(name);
  if (!heuristic.ok()) {
    ADD_FAILURE() << heuristic.status().message();
    return nan;
  }
  const absl::Status configured = (*heuristic)->configure(config, testModel());
  if (!configured.ok()) {
    ADD_FAILURE() << configured.message();
    return nan;
  }
  return (*heuristic)->forceOffset(context, contactIndex);
}

/** Every LOG(WARNING) issued on any thread while it is alive, in order. */
class WarningLog {
 public:
  WarningLog() {
    EXPECT_CALL(log_, Log(testing::_, testing::_, testing::_))
        .WillRepeatedly([this](absl::LogSeverity severity, const std::string& /*filePath*/, const std::string& message) {
          if (severity == absl::LogSeverity::kWarning) warnings_.push_back(message);
        });
    log_.StartCapturingLogs();
  }

  /** How many of the captured warnings contain `needle`. */
  size_t count(absl::string_view needle) const {
    return static_cast<size_t>(std::count_if(warnings_.begin(), warnings_.end(),
                                             [needle](const std::string& warning) { return absl::StrContains(warning, needle); }));
  }

  const std::vector<std::string>& warnings() const { return warnings_; }

  /** For failure messages: every warning captured so far. */
  std::string all() const { return absl::StrCat("captured warnings:\n", absl::StrJoin(warnings_, "\n---\n")); }

 private:
  // Declared before the mock so that it outlives it: the mock stops capturing in its destructor.
  std::vector<std::string> warnings_;
  absl::ScopedMockLog log_;
};

}  // namespace

/*=============================== the layer: several heuristics per channel (H32) ===============================*/

TEST(LocomotionHeuristicFormulas, BasePoseOffsetsOfEveryListedHeuristicAreSummedOnEveryChannel) {
  // Every other formula test lists ONE heuristic, so a layer that kept only the last offset (`total = offset(...)`)
  // or an operator+= that dropped or crossed a channel would pass all of them. Here each of the three heuristics
  // contributes to a different mix of channels, and the three totals are each a sum of two non-zero parts.
  LocomotionHeuristicConfig config;
  config.formulation.basePose = {"orientation_compensation", "periodic_orientation", "height_compensation"};
  config.orientationCompensation.rollPerLateralVelocity = -0.1;
  config.orientationCompensation.rollOffset = 0.01;
  config.orientationCompensation.pitchPerForwardVelocity = 0.04;
  config.orientationCompensation.pitchOffset = 0.005;
  config.periodicOrientation.rollAmplitude = 0.03;
  config.periodicOrientation.rollPhaseRate = 2.0 * M_PI;
  config.periodicOrientation.pitchAmplitude = 0.02;
  config.periodicOrientation.pitchPhaseRate = 4.0 * M_PI;
  config.heightCompensation.heightPerSpeedSquared = -0.01;
  config.heightCompensation.heightPerSpeed = -0.02;
  config.heightCompensation.heightOffset = 0.005;

  BasePoseHeuristicContext context;
  context.commandedVelocityInBaseFrame = vector2_t(1.0, 0.2);
  context.gaitPhase = 0.125;
  // orientation:  roll  = -0.1 * 0.2 + 0.01 = -0.01          pitch = 0.04 * 1.0 + 0.005 = 0.045
  // periodic:     roll  = 0.03 sin(2 pi / 8) = 0.0212132034   pitch = 0.02 sin(4 pi / 8) = 0.02
  // height:       |v| = sqrt(1.04);  dz = -0.01 * 1.04 - 0.02 * sqrt(1.04) + 0.005 = -0.0257960781
  const scalar_t expectedRoll = -0.01 + 0.03 * std::sqrt(0.5);
  const scalar_t expectedPitch = 0.065;
  const scalar_t expectedHeight = -0.025796078054371143;

  // Addition is commutative, which is what makes the order of the task-file list documentary: both orders must agree.
  for (const std::vector<std::string>& order :
       {config.formulation.basePose, std::vector<std::string>{"height_compensation", "periodic_orientation", "orientation_compensation"}}) {
    LocomotionHeuristicConfig ordered = config;
    ordered.formulation.basePose = order;
    absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
        LocomotionHeuristicLayer::Create(ordered, testModel(), steppingEnvironment());
    ASSERT_TRUE(layer.ok()) << layer.status().message();
    const BasePoseOffset offset = (*layer)->basePoseOffset(context);
    const std::string listed = absl::StrJoin(order, ", ");
    EXPECT_NEAR(offset.roll, expectedRoll, 1e-12) << listed;
    EXPECT_NEAR(offset.pitch, expectedPitch, 1e-12) << listed;
    EXPECT_NEAR(offset.height, expectedHeight, 1e-12) << listed;
  }
}

TEST(LocomotionHeuristicFormulas, FootholdOffsetsOfEveryListedHeuristicAreSummed) {
  LocomotionHeuristicConfig config;
  config.formulation.foothold = {"hip_centered_stepping", "translational_stepping", "capture_point"};
  config.translationalStepping.forwardPerForwardVelocity = 0.2;

  FootholdHeuristicContext context = leftFoot();
  context.commandedVelocity = vector2_t(1.0, 0.0);
  context.measuredVelocity = vector2_t(1.1, 0.0);
  // hip:            the left hip, (0.03, 0.08), at yaw 0
  // translational:  0.2 * 1.0 = 0.2 forward
  // capture point:  sqrt(0.6125 / 9.81) * (1.1 - 1.0) = 0.0249872547 forward
  const vector2_t expected(0.03 + 0.2 + 0.024987254651223631, 0.08);

  for (const std::vector<std::string>& order :
       {config.formulation.foothold, std::vector<std::string>{"capture_point", "translational_stepping", "hip_centered_stepping"}}) {
    LocomotionHeuristicConfig ordered = config;
    ordered.formulation.foothold = order;
    absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
        LocomotionHeuristicLayer::Create(ordered, testModel(), steppingEnvironment());
    ASSERT_TRUE(layer.ok()) << layer.status().message();
    const vector2_t offset = (*layer)->footholdOffset(context);
    EXPECT_NEAR(offset.x(), expected.x(), 1e-12) << absl::StrJoin(order, ", ");
    EXPECT_NEAR(offset.y(), expected.y(), 1e-12) << absl::StrJoin(order, ", ");
  }
}

TEST(LocomotionHeuristicFormulas, WrenchOffsetsOfEveryListedHeuristicAreSummed) {
  LocomotionHeuristicConfig config;
  config.formulation.wrench = {"impulse_scaling", "centripetal_acceleration"};

  WrenchHeuristicContext context = doubleSupport(0.6, 0.6);
  context.commandedVelocity = vector2_t(2.0, 0.0);
  context.commandedYawRate = 0.5;
  // impulse scaling:  W / (2 * 0.6) - W / 2 = 327 - 196.2 = 130.8 N, vertical
  // centripetal:      m psidot v_x / n = 40 * 0.5 * 2 / 2 = 20 N along +y (the clamp, 0.3 W = 117.72 N, does not bind)
  for (const std::vector<std::string>& order :
       {config.formulation.wrench, std::vector<std::string>{"centripetal_acceleration", "impulse_scaling"}}) {
    LocomotionHeuristicConfig ordered = config;
    ordered.formulation.wrench = order;
    absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
        LocomotionHeuristicLayer::Create(ordered, testModel(), steppingEnvironment());
    ASSERT_TRUE(layer.ok()) << layer.status().message();
    const vector3_t offset = (*layer)->wrenchOffset(context, CONTACT_LEFT_INDEX);
    EXPECT_NEAR(offset.x(), 0.0, 1e-9) << absl::StrJoin(order, ", ");
    EXPECT_NEAR(offset.y(), 20.0, 1e-9) << absl::StrJoin(order, ", ");
    EXPECT_NEAR(offset.z(), 130.8, 1e-9) << absl::StrJoin(order, ", ");
  }
}

TEST(LocomotionHeuristicFormulas, AnchorFlagIsSetWhereverTheHipHeuristicSitsInTheList) {
  // An overwrite (`footholdMovesAnchor_ = heuristic->movesAnchor()`) instead of an OR leaves the flag false whenever
  // hip_centered_stepping is not LAST, and the reference manager then adds a hip offset measured from the base to the
  // stance-foot anchor - landing the foot a hip's width beside where it should.
  const std::vector<std::vector<std::string>> withHip{{"hip_centered_stepping", "capture_point"},
                                                      {"capture_point", "hip_centered_stepping"},
                                                      {"translational_stepping", "hip_centered_stepping", "capture_point"}};
  for (const std::vector<std::string>& foothold : withHip) {
    LocomotionHeuristicConfig config;
    config.formulation.foothold = foothold;
    absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
        LocomotionHeuristicLayer::Create(config, testModel(), steppingEnvironment());
    ASSERT_TRUE(layer.ok()) << layer.status().message();
    EXPECT_TRUE((*layer)->footholdMovesAnchor()) << absl::StrJoin(foothold, ", ");
  }

  // And not set by anything else, or the flag would be vacuously true.
  LocomotionHeuristicConfig withoutHip;
  withoutHip.formulation.foothold = {"capture_point", "translational_stepping", "in_place_turning", "high_speed_turning"};
  absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
      LocomotionHeuristicLayer::Create(withoutHip, testModel(), steppingEnvironment());
  ASSERT_TRUE(layer.ok()) << layer.status().message();
  EXPECT_FALSE((*layer)->footholdMovesAnchor());
}

TEST(LocomotionHeuristicFormulas, WorldFrameFlagIsSetWhereverTheCentripetalHeuristicSitsInTheList) {
  // The same overwrite bug for wrenchNeedsWorldFrame_ would send a horizontal force down the input-only, vertical-only
  // path and hand it to the basis-vector parameterization in the wrong frame.
  for (const std::vector<std::string>& wrench : {std::vector<std::string>{"centripetal_acceleration", "impulse_scaling"},
                                                 std::vector<std::string>{"impulse_scaling", "centripetal_acceleration"}}) {
    LocomotionHeuristicConfig config;
    config.formulation.wrench = wrench;
    absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
        LocomotionHeuristicLayer::Create(config, testModel(), steppingEnvironment());
    ASSERT_TRUE(layer.ok()) << layer.status().message();
    EXPECT_TRUE((*layer)->wrenchNeedsWorldFrame()) << absl::StrJoin(wrench, ", ");
  }

  LocomotionHeuristicConfig verticalOnly;
  verticalOnly.formulation.wrench = {"impulse_scaling"};
  absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
      LocomotionHeuristicLayer::Create(verticalOnly, testModel(), steppingEnvironment());
  ASSERT_TRUE(layer.ok()) << layer.status().message();
  EXPECT_FALSE((*layer)->wrenchNeedsWorldFrame()) << "a vertical force must keep the cheap path";
}

TEST(LocomotionHeuristicFormulas, FormulationWarnsAboutEitherHalfOfBledtsFootPlacementOnItsOwn) {
  // hip_centered_stepping is the base term the others are summed onto (figure 4-8). Listing it alone, or listing the
  // others without it, is legal but is not the sum the dissertation validates, so each is reported - and the full sum
  // is not.
  LocomotionHeuristicFormulation hipAlone;
  hipAlone.foothold = {"hip_centered_stepping"};
  const std::vector<std::string> hipAloneWarnings = hipAlone.warnings();
  ASSERT_EQ(hipAloneWarnings.size(), 1u) << absl::StrJoin(hipAloneWarnings, "\n");
  EXPECT_TRUE(absl::StrContains(hipAloneWarnings.front(), "lists only 'hip_centered_stepping'")) << hipAloneWarnings.front();
  EXPECT_TRUE(absl::StrContains(hipAloneWarnings.front(), "translational_stepping")) << "the warning must say what to add";

  LocomotionHeuristicFormulation withoutHip;
  withoutHip.foothold = {"capture_point"};
  const std::vector<std::string> withoutHipWarnings = withoutHip.warnings();
  ASSERT_EQ(withoutHipWarnings.size(), 1u) << absl::StrJoin(withoutHipWarnings, "\n");
  EXPECT_TRUE(absl::StrContains(withoutHipWarnings.front(), "does not list 'hip_centered_stepping'")) << withoutHipWarnings.front();

  LocomotionHeuristicFormulation both;
  both.foothold = {"hip_centered_stepping", "translational_stepping"};
  EXPECT_TRUE(both.warnings().empty()) << absl::StrJoin(both.warnings(), "\n");

  // The other two channels have no such pairing, and an empty foothold list has nothing to warn about.
  LocomotionHeuristicFormulation noFoothold;
  noFoothold.basePose = {"orientation_compensation"};
  noFoothold.wrench = {"impulse_scaling"};
  EXPECT_TRUE(noFoothold.warnings().empty()) << absl::StrJoin(noFoothold.warnings(), "\n");
}

/*========================================= foothold formulae (H34, H35) ========================================*/

TEST(LocomotionHeuristicFormulas, TranslationalSteppingAppliesItsLawInTheBaseFrameOffTheAxes) {
  // At yaw pi/2 walking along the heading - the only rotated case the older test has - rotating by +yaw then -yaw
  // instead of -yaw then +yaw gives the same answer. Off the axes, with a velocity that is NOT along the heading and
  // different forward and lateral coefficients, it does not.
  LocomotionHeuristicConfig config;
  config.translationalStepping.forwardPerForwardVelocity = 0.25;
  config.translationalStepping.forwardOffset = 0.02;
  config.translationalStepping.lateralPerLateralVelocity = 0.1;
  config.translationalStepping.lateralOffset = 0.05;

  // Base-frame velocity R(-pi/3) (0.4, 0.9) = (0.9794228634, 0.1035898385).
  //   left:  base offset (0.25 * 0.97942 + 0.02, 0.1 * 0.10359 + 0.05) -> world (0.0801554446, 0.2595512702)
  //   right: base offset (0.25 * 0.97942 + 0.02, 0.1 * 0.10359 - 0.05) -> world (0.1667579849, 0.2095512702)
  // The transposed-rotation mutant gives (0.0498, 0.1729) for the left foot; dropping `side` from the lateral constant
  // gives the right foot the left foot's answer; putting `side` on the velocity term moves the right foot to
  // (0.1847, 0.1992).
  FootholdHeuristicContext left = leftFoot();
  left.baseYaw = M_PI / 3.0;
  left.commandedVelocity = vector2_t(0.4, 0.9);
  FootholdHeuristicContext right = rightFoot();
  right.baseYaw = M_PI / 3.0;
  right.commandedVelocity = vector2_t(0.4, 0.9);

  const vector2_t leftOffset = footholdOffsetOf("translational_stepping", config, left);
  const vector2_t rightOffset = footholdOffsetOf("translational_stepping", config, right);
  EXPECT_NEAR(leftOffset.x(), 0.080155444566227696, 1e-9);
  EXPECT_NEAR(leftOffset.y(), 0.25955127018922197, 1e-9);
  EXPECT_NEAR(rightOffset.x(), 0.16675798494467156, 1e-9);
  EXPECT_NEAR(rightOffset.y(), 0.20955127018922193, 1e-9);
}

TEST(LocomotionHeuristicFormulas, TranslationalSteppingLeadGrowsWithTheStanceAboutToBegin) {
  // a1 = Bledt's constant + fraction * T_stance, separately for the two axes. The fractions differ from each other
  // (0.5 forward, 0.25 lateral) so that reading one for the other is visible, and the heading is off the axes.
  LocomotionHeuristicConfig config;
  config.translationalStepping.forwardPerForwardVelocity = 0.1;
  config.translationalStepping.forwardStanceFraction = 0.5;
  config.translationalStepping.forwardOffset = 0.01;
  config.translationalStepping.lateralPerLateralVelocity = 0.05;
  config.translationalStepping.lateralStanceFraction = 0.25;
  config.translationalStepping.lateralOffset = 0.03;

  FootholdHeuristicContext left = leftFoot();
  left.baseYaw = M_PI / 6.0;
  left.commandedVelocity = vector2_t(0.8, 0.3);
  FootholdHeuristicContext right = rightFoot();
  right.baseYaw = M_PI / 6.0;
  right.commandedVelocity = vector2_t(0.8, 0.3);

  // T_stance = 0.6 s: forward gain 0.1 + 0.5 * 0.6 = 0.4, lateral gain 0.05 + 0.25 * 0.6 = 0.2, applied to the
  // base-frame velocity R(-pi/6) (0.8, 0.3) and rotated back by pi/6.
  left.stanceDuration = 0.6;
  right.stanceDuration = 0.6;
  const vector2_t leftLong = footholdOffsetOf("translational_stepping", config, left);
  const vector2_t rightLong = footholdOffsetOf("translational_stepping", config, right);
  EXPECT_NEAR(leftLong.x(), 0.29964101615137761, 1e-9);
  EXPECT_NEAR(leftLong.y(), 0.17526279441628825, 1e-9);
  EXPECT_NEAR(rightLong.x(), 0.32964101615137759, 1e-9);
  EXPECT_NEAR(rightLong.y(), 0.12330127018922195, 1e-9);

  // T_stance = 0 - what the context carries when the schedule does not say - is exactly Bledt's constant form: gains
  // 0.1 and 0.05.
  left.stanceDuration = 0.0;
  right.stanceDuration = 0.0;
  const vector2_t leftConstant = footholdOffsetOf("translational_stepping", config, left);
  const vector2_t rightConstant = footholdOffsetOf("translational_stepping", config, right);
  EXPECT_NEAR(leftConstant.x(), 0.070155444566227701, 1e-9);
  EXPECT_NEAR(leftConstant.y(), 0.06705127018922194, 1e-9);
  EXPECT_NEAR(rightConstant.x(), 0.10015544456622769, 1e-9);
  EXPECT_NEAR(rightConstant.y(), 0.015089745962155614, 1e-9);

  // ...and the stance fractions are the ONLY thing the stance duration reaches: with them at zero, a long stance gives
  // the constant form too.
  LocomotionHeuristicConfig bledt = config;
  bledt.translationalStepping.forwardStanceFraction = 0.0;
  bledt.translationalStepping.lateralStanceFraction = 0.0;
  left.stanceDuration = 0.6;
  const vector2_t leftBledt = footholdOffsetOf("translational_stepping", bledt, left);
  EXPECT_NEAR(leftBledt.x(), 0.070155444566227701, 1e-9);
  EXPECT_NEAR(leftBledt.y(), 0.06705127018922194, 1e-9);
}

TEST(LocomotionHeuristicFormulas, InPlaceTurningRotatesItsBaseFrameOffsetIntoTheWorld) {
  // Evaluated at yaw pi/2, where dropping the final rotation applies the fore-aft lead along the world's x axis.
  LocomotionHeuristicConfig config;
  config.inPlaceTurning.forwardPerYawRate = 0.05;
  config.inPlaceTurning.lateralPerYawRate = 0.02;
  config.inPlaceTurning.lateralOffset = 0.03;

  FootholdHeuristicContext left = leftFoot();
  left.baseYaw = M_PI / 2.0;
  left.commandedYawRate = 1.0;
  FootholdHeuristicContext right = rightFoot();
  right.baseYaw = M_PI / 2.0;
  right.commandedYawRate = 1.0;

  // left:  base (-0.05 * 1, 0.02 * 1 + 0.03) = (-0.05, 0.05)  -> world R(pi/2) = (-0.05, -0.05)
  // right: base (+0.05 * 1, 0.02 * 1 - 0.03) = (0.05, -0.01)  -> world (0.01, 0.05)
  // Without the rotation the left foot would get (-0.05, 0.05); an unsigned lateral constant would give the right foot
  // (-0.05, 0.05); a signed lateral RATE would give it (0.05, 0.05) rather than (0.01, 0.05).
  const vector2_t leftOffset = footholdOffsetOf("in_place_turning", config, left);
  const vector2_t rightOffset = footholdOffsetOf("in_place_turning", config, right);
  EXPECT_NEAR(leftOffset.x(), -0.05, 1e-9);
  EXPECT_NEAR(leftOffset.y(), -0.05, 1e-9);
  EXPECT_NEAR(rightOffset.x(), 0.01, 1e-9);
  EXPECT_NEAR(rightOffset.y(), 0.05, 1e-9);
}

TEST(LocomotionHeuristicFormulas, InPlaceTurningLeverGrowsWithTheStanceAboutToBegin) {
  // forward = -side (a1 + lever * T_stance) psidot + a0. The constant a0 is NOT signed per foot.
  LocomotionHeuristicConfig config;
  config.inPlaceTurning.forwardPerYawRate = 0.05;
  config.inPlaceTurning.forwardStanceLever = 0.1;
  config.inPlaceTurning.forwardOffset = 0.01;
  config.inPlaceTurning.lateralPerYawRate = 0.02;
  config.inPlaceTurning.lateralOffset = 0.03;

  FootholdHeuristicContext left = leftFoot();
  left.baseYaw = M_PI / 2.0;
  left.commandedYawRate = 1.0;
  FootholdHeuristicContext right = rightFoot();
  right.baseYaw = M_PI / 2.0;
  right.commandedYawRate = 1.0;

  // T_stance = 0.5 s: gain 0.05 + 0.1 * 0.5 = 0.1.
  //   left:  base (-0.1 + 0.01, 0.02 + 0.03) = (-0.09, 0.05) -> world (-0.05, -0.09)
  //   right: base ( 0.1 + 0.01, 0.02 - 0.03) = (0.11, -0.01) -> world (0.01, 0.11)
  left.stanceDuration = 0.5;
  right.stanceDuration = 0.5;
  const vector2_t leftLong = footholdOffsetOf("in_place_turning", config, left);
  const vector2_t rightLong = footholdOffsetOf("in_place_turning", config, right);
  EXPECT_NEAR(leftLong.x(), -0.05, 1e-9);
  EXPECT_NEAR(leftLong.y(), -0.09, 1e-9);
  EXPECT_NEAR(rightLong.x(), 0.01, 1e-9);
  EXPECT_NEAR(rightLong.y(), 0.11, 1e-9);

  // T_stance = 0 is Bledt's constant form, gain 0.05:
  //   left:  base (-0.05 + 0.01, 0.05) = (-0.04, 0.05) -> world (-0.05, -0.04)
  //   right: base ( 0.05 + 0.01, -0.01) = (0.06, -0.01) -> world (0.01, 0.06)
  left.stanceDuration = 0.0;
  right.stanceDuration = 0.0;
  const vector2_t leftConstant = footholdOffsetOf("in_place_turning", config, left);
  const vector2_t rightConstant = footholdOffsetOf("in_place_turning", config, right);
  EXPECT_NEAR(leftConstant.x(), -0.05, 1e-9);
  EXPECT_NEAR(leftConstant.y(), -0.04, 1e-9);
  EXPECT_NEAR(rightConstant.x(), 0.01, 1e-9);
  EXPECT_NEAR(rightConstant.y(), 0.06, 1e-9);
}

TEST(LocomotionHeuristicFormulas, HighSpeedTurningUsesBothCrossTermsAtAnyHeading) {
  // Different forward and lateral coefficients, so a missing rotation cannot hide behind the rotation-equivariance of
  // an isotropic cross product.
  LocomotionHeuristicConfig config;
  config.highSpeedTurning.forwardPerCrossTerm = 0.1;
  config.highSpeedTurning.forwardOffset = 0.01;
  config.highSpeedTurning.lateralPerCrossTerm = 0.2;
  config.highSpeedTurning.lateralOffset = 0.03;

  // Facing +y (yaw pi/2), walking +y at 2 m/s and turning left at 1 rad/s. In the base frame v = (2, 0), so
  // v x omega = (0, -2):
  //   left:  base (0.01, 0.2 * -2 + 0.03) = (0.01, -0.37) -> world (0.37, 0.01)
  //   right: base (0.01, 0.2 * -2 - 0.03) = (0.01, -0.43) -> world (0.43, 0.01)
  // Both are thrown to world +x: OUTSIDE a counter-clockwise turn centered to the robot's left, i.e. at -x.
  FootholdHeuristicContext left = leftFoot();
  left.baseYaw = M_PI / 2.0;
  left.commandedVelocity = vector2_t(0.0, 2.0);
  left.commandedYawRate = 1.0;
  FootholdHeuristicContext right = rightFoot();
  right.baseYaw = M_PI / 2.0;
  right.commandedVelocity = vector2_t(0.0, 2.0);
  right.commandedYawRate = 1.0;
  EXPECT_NEAR(footholdOffsetOf("high_speed_turning", config, left).x(), 0.37, 1e-9);
  EXPECT_NEAR(footholdOffsetOf("high_speed_turning", config, left).y(), 0.01, 1e-9);
  EXPECT_NEAR(footholdOffsetOf("high_speed_turning", config, right).x(), 0.43, 1e-9);
  EXPECT_NEAR(footholdOffsetOf("high_speed_turning", config, right).y(), 0.01, 1e-9);

  // The FORWARD cross term, v_y psidot, is zero whenever the velocity is along the heading. Sidestepping left at
  // 1.5 m/s while turning left at 0.8 rad/s it is +1.2, so forward = 0.1 * 1.2 + 0.01 = 0.13; flipping its sign gives
  // -0.11.
  FootholdHeuristicContext sidestepping = leftFoot();
  sidestepping.commandedVelocity = vector2_t(0.0, 1.5);
  sidestepping.commandedYawRate = 0.8;
  const vector2_t sidestep = footholdOffsetOf("high_speed_turning", config, sidestepping);
  EXPECT_NEAR(sidestep.x(), 0.13, 1e-9);
  EXPECT_NEAR(sidestep.y(), 0.03, 1e-9);

  // Everything at once, off the axes: yaw 0.4, v = (1.0, 0.5), psidot 0.7.
  FootholdHeuristicContext generalLeft = leftFoot();
  generalLeft.baseYaw = 0.4;
  generalLeft.commandedVelocity = vector2_t(1.0, 0.5);
  generalLeft.commandedYawRate = 0.7;
  FootholdHeuristicContext generalRight = rightFoot();
  generalRight.baseYaw = 0.4;
  generalRight.commandedVelocity = vector2_t(1.0, 0.5);
  generalRight.commandedYawRate = 0.7;
  const vector2_t generalLeftOffset = footholdOffsetOf("high_speed_turning", config, generalLeft);
  const vector2_t generalRightOffset = footholdOffsetOf("high_speed_turning", config, generalRight);
  EXPECT_NEAR(generalLeftOffset.x(), 0.062943155438677234, 1e-9);
  EXPECT_NEAR(generalLeftOffset.y(), -0.11041245317471936, 1e-9);
  EXPECT_NEAR(generalRightOffset.x(), 0.086308255977196269, 1e-9);
  EXPECT_NEAR(generalRightOffset.y(), -0.16567611281489247, 1e-9);
}

TEST(LocomotionHeuristicFormulas, HipCenteredSteppingAppliesEachScaleToItsOwnAxisAndFoot) {
  // Atlas ships lateralScale 2.528, so ignoring it, or swapping it with the longitudinal one, changes its stance by
  // centimeters. With both scales at 1 and hip x = 0 - the older test - neither would be visible.
  LocomotionHeuristicConfig config;
  config.hipCenteredStepping.lateralScale = 2.0;
  config.hipCenteredStepping.longitudinalScale = 0.5;

  // left hip (0.03, 0.08) -> (0.5 * 0.03, 2.0 * 0.08) = (0.015, 0.16)
  // right hip (0.01, -0.09) -> (0.005, -0.18); the swapped-scale mutant gives (0.06, 0.04) for the left foot.
  const vector2_t left = footholdOffsetOf("hip_centered_stepping", config, leftFoot());
  const vector2_t right = footholdOffsetOf("hip_centered_stepping", config, rightFoot());
  EXPECT_NEAR(left.x(), 0.015, 1e-12);
  EXPECT_NEAR(left.y(), 0.16, 1e-12);
  EXPECT_NEAR(right.x(), 0.005, 1e-12);
  EXPECT_NEAR(right.y(), -0.18, 1e-12);

  // At yaw pi/3 the scaled hip is rotated into the world, and the predicted base position - which the reference
  // manager adds back as the anchor - must not leak into the offset.
  FootholdHeuristicContext turnedLeft = leftFoot();
  turnedLeft.baseYaw = M_PI / 3.0;
  turnedLeft.basePosition = vector2_t(5.0, -3.0);
  FootholdHeuristicContext turnedRight = rightFoot();
  turnedRight.baseYaw = M_PI / 3.0;
  turnedRight.basePosition = vector2_t(5.0, -3.0);
  const vector2_t turnedLeftOffset = footholdOffsetOf("hip_centered_stepping", config, turnedLeft);
  const vector2_t turnedRightOffset = footholdOffsetOf("hip_centered_stepping", config, turnedRight);
  // R(pi/3) (0.015, 0.16) = (0.0075 - 0.13856406, 0.01299038 + 0.08)
  EXPECT_NEAR(turnedLeftOffset.x(), -0.13106406460551018, 1e-9);
  EXPECT_NEAR(turnedLeftOffset.y(), 0.09299038105676659, 1e-9);
  // R(pi/3) (0.005, -0.18)
  EXPECT_NEAR(turnedRightOffset.x(), 0.15838457268119893, 1e-9);
  EXPECT_NEAR(turnedRightOffset.y(), -0.08566987298107781, 1e-9);
}

/*============================================ capture point (H41) ============================================*/

TEST(LocomotionHeuristicFormulas, CapturePointPrefersTheOverrideThenTheMeasurementThenTheNominalHeight) {
  // The older fixture has the measured and the nominal height both at 0.6125, gain 1 and no override, so "always use
  // the nominal", "ignore the override" and "drop the gain" all pass it. Here each of the four is distinguishable:
  // measured 0.9, override 0.4, nominal 0.6125, gain 0.5, gravity 3.
  const scalar_t error = 0.5;  // measured 0.5 m/s forward, commanded 0
  FootholdHeuristicContext pushed = leftFoot();
  pushed.measuredVelocity = vector2_t(error, 0.0);
  pushed.commandedVelocity = vector2_t::Zero();
  pushed.comHeight = 0.9;

  LocomotionHeuristicConfig config;
  // (a) the measured height: sqrt(0.9 / 9.81) * 0.5 = 0.1514456332
  EXPECT_NEAR(footholdOffsetOf("capture_point", config, pushed).x(), 0.15144563320384566, 1e-9);
  EXPECT_NEAR(footholdOffsetOf("capture_point", config, pushed).y(), 0.0, kTol);

  // (b) a positive override wins over the measurement: sqrt(0.4 / 9.81) * 0.5 = 0.1009637555
  LocomotionHeuristicConfig overridden = config;
  overridden.capturePoint.comHeightOverride = 0.4;
  EXPECT_NEAR(footholdOffsetOf("capture_point", overridden, pushed).x(), 0.10096375546923045, 1e-9);

  // (c) a non-positive measurement falls back to the model's nominal: sqrt(0.6125 / 9.81) * 0.5 = 0.1249362733. Both
  // zero and a NEGATIVE measurement: trusting -0.3 would take the square root of a negative height and put a NaN
  // foothold in front of the solver, and returning zero for it instead of falling back would leave the push unanswered.
  for (const scalar_t measured : {0.0, -0.3}) {
    FootholdHeuristicContext unmeasured = pushed;
    unmeasured.comHeight = measured;
    EXPECT_NEAR(footholdOffsetOf("capture_point", config, unmeasured).x(), 0.12493627325611814, 1e-9) << "measured " << measured;
  }

  // (d) the gain multiplies: 0.5 * 0.1514456332 = 0.0757228166
  LocomotionHeuristicConfig halfGain = config;
  halfGain.capturePoint.gain = 0.5;
  EXPECT_NEAR(footholdOffsetOf("capture_point", halfGain, pushed).x(), 0.07572281660192283, 1e-9);

  // gravity is the configured one: sqrt(0.9 / 3) * 0.5 = 0.2738612788, with the 0.25 m clamp opened so it cannot bind
  LocomotionHeuristicConfig lowGravity = config;
  lowGravity.capturePoint.gravity = 3.0;
  lowGravity.capturePoint.maximumOffset = 1.0;
  EXPECT_NEAR(footholdOffsetOf("capture_point", lowGravity, pushed).x(), 0.27386127875258304, 1e-9);
}

TEST(LocomotionHeuristicFormulas, CapturePointIsSilentWithNoUsablePendulumHeight) {
  // Neither the measurement (-1) nor the nominal is usable, so there is no pendulum time constant and the heuristic
  // must say nothing. A nominal of exactly 0 cannot catch a missing guard - sqrt(0 / g) is a time constant of 0, and
  // the offset is zero with or without it - so the nominal here is one the guard is actually needed for: negative, or
  // NaN (a model that never set it). Without the guard, sqrt of either is NaN and a NaN foothold reaches the solver,
  // which fails the comparisons below.
  const LocomotionHeuristicConfig config;
  FootholdHeuristicContext pushed = leftFoot();
  pushed.measuredVelocity = vector2_t(0.5, 0.0);
  for (const scalar_t nominal : {-0.2, std::numeric_limits<scalar_t>::quiet_NaN()}) {
    LocomotionHeuristicModelParameters noNominal = testModel();
    noNominal.nominalComHeight = nominal;
    pushed.comHeight = -1.0;
    const vector2_t offset = footholdOffsetOf("capture_point", config, noNominal, pushed);
    EXPECT_NEAR(offset.x(), 0.0, kTol) << "nominal " << nominal;
    EXPECT_NEAR(offset.y(), 0.0, kTol) << "nominal " << nominal;

    // Positive control: the same push with a usable measured height does step, sqrt(0.9 / 9.81) * 0.5.
    pushed.comHeight = 0.9;
    EXPECT_NEAR(footholdOffsetOf("capture_point", config, noNominal, pushed).x(), 0.15144563320384566, 1e-9) << "nominal " << nominal;
  }
}

/*========================================= contact-wrench formulae (H36, H40) ========================================*/

TEST(LocomotionHeuristicFormulas, CentripetalForceInSingleSupportPointsIntoTheTurnAlongX) {
  // The older test walks along +x, where the x component -m psidot v_y is zero whatever its sign, and always has two
  // stance feet, where dividing by the foot COUNT instead of the stance count is invisible.
  const LocomotionHeuristicConfig config;
  WrenchHeuristicContext context = leftSingleSupport(0.5);
  context.commandedVelocity = vector2_t(0.0, 1.5);
  context.commandedYawRate = 0.8;
  // omega x v = (0, 0, 0.8) x (0, 1.5, 0) = (-1.2, 0, 0); times m = 40 over one stance foot: (-48, 0, 0). The center of
  // a left turn taken while moving +y is at -x.
  const vector3_t force = wrenchOffsetOf("centripetal_acceleration", config, context, CONTACT_LEFT_INDEX);
  EXPECT_NEAR(force.x(), -48.0, 1e-9);
  EXPECT_NEAR(force.y(), 0.0, 1e-12);
  EXPECT_NEAR(force.z(), 0.0, 1e-12);

  // `scale` blends it: 0.5 * -48 = -24.
  LocomotionHeuristicConfig halved = config;
  halved.centripetalAcceleration.scale = 0.5;
  EXPECT_NEAR(wrenchOffsetOf("centripetal_acceleration", halved, context, CONTACT_LEFT_INDEX).x(), -24.0, 1e-9);
}

TEST(LocomotionHeuristicFormulas, CentripetalAbsoluteClampReplacesTheWeightRatio) {
  WrenchHeuristicContext context = leftSingleSupport(0.5);
  context.commandedVelocity = vector2_t(0.0, 1.5);
  context.commandedYawRate = 0.8;

  // maximumForce 30 N clamps the 48 N demand.
  LocomotionHeuristicConfig absolute;
  absolute.centripetalAcceleration.maximumForce = 30.0;
  EXPECT_NEAR(wrenchOffsetOf("centripetal_acceleration", absolute, context, CONTACT_LEFT_INDEX).x(), -30.0, 1e-9);

  // A demand of m psidot v = 40 * 2.5 * 1.5 = 150 N sits between the ratio clamp (0.3 W = 117.72 N) and a 200 N
  // absolute clamp. With the absolute clamp set, the ratio must play no part...
  WrenchHeuristicContext fast = context;
  fast.commandedYawRate = 2.5;
  LocomotionHeuristicConfig generous;
  generous.centripetalAcceleration.maximumForce = 200.0;
  EXPECT_NEAR(wrenchOffsetOf("centripetal_acceleration", generous, fast, CONTACT_LEFT_INDEX).x(), -150.0, 1e-9);
  // ...and with it at 0 the ratio is the clamp.
  const LocomotionHeuristicConfig ratio;
  EXPECT_NEAR(wrenchOffsetOf("centripetal_acceleration", ratio, fast, CONTACT_LEFT_INDEX).x(), -0.3 * kWeight, 1e-9);
}

TEST(LocomotionHeuristicFormulas, ImpulseScalingMeetsTheImpulseBudgetOverACycleOfItsDutyFactor) {
  // The property the heuristic exists for, evaluated THROUGH THE CODE: over a cycle of duty factor beta the mean
  // number of feet down is 2 beta, and the mean total vertical reference must come back to exactly W. A gait of
  // beta = 0.6 is 20 % double support and 80 % single support.
  LocomotionHeuristicConfig config;
  config.formulation.wrench = {"impulse_scaling"};
  absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
      LocomotionHeuristicLayer::Create(config, testModel(), steppingEnvironment());
  ASSERT_TRUE(layer.ok()) << layer.status().message();

  const scalar_t doubleSupportOffset = (*layer)->wrenchOffset(doubleSupport(0.6, 0.6), CONTACT_LEFT_INDEX).z();
  const scalar_t singleSupportOffset = (*layer)->wrenchOffset(leftSingleSupport(0.6), CONTACT_LEFT_INDEX).z();
  // Both feet are asked for W / (2 * 0.6) = 327 N whatever the stance count: +130.8 on W/2, -65.4 on W.
  EXPECT_NEAR(doubleSupportOffset, 130.8, 1e-9);
  EXPECT_NEAR(singleSupportOffset, -65.4, 1e-9);
  const scalar_t meanTotal = 0.2 * 2.0 * (kWeight / 2.0 + doubleSupportOffset) + 0.8 * (kWeight + singleSupportOffset);
  // The rejected form, (W / n) / beta, gives 654 N here: it would regularize the CoM towards accelerating upwards.
  EXPECT_NEAR(meanTotal, kWeight, 1e-9) << "the cycle-averaged vertical reference must equal the robot's weight";

  // `scale` blends linearly between weight compensation and Bledt's value: half of 130.8.
  LocomotionHeuristicConfig halved;
  halved.impulseScaling.scale = 0.5;
  EXPECT_NEAR(wrenchOffsetOf("impulse_scaling", halved, doubleSupport(0.6, 0.6), CONTACT_LEFT_INDEX).z(), 65.4, 1e-9);
}

TEST(LocomotionHeuristicFormulas, ImpulseScalingFloorsTheDutyFactorBeforeTakingTheReciprocal) {
  // With the force-ratio clamp opened to 10, the duty-factor floor is the only thing standing between 1/beta and an
  // unbounded reference. beta 0.1 floored to 0.4: W / (2 * 0.4) - W / 2 = 490.5 - 196.2 = 294.3 N. Without the floor it
  // would be W / 0.2 - W / 2 = 1765.8 N - and with the default ratio of 2 both would clamp to the same 196.2, which is
  // why the older test could not tell.
  LocomotionHeuristicConfig config;
  config.impulseScaling.maximumForceRatio = 10.0;
  config.impulseScaling.minimumDutyFactor = 0.4;
  EXPECT_NEAR(wrenchOffsetOf("impulse_scaling", config, doubleSupport(0.1, 0.1), CONTACT_LEFT_INDEX).z(), 294.3, 1e-9);
}

TEST(LocomotionHeuristicFormulas, ImpulseScalingNeverAsksAFootToPullDown) {
  // scale 3 is admissible (only a negative one is rejected). In single support at beta 0.9 the blend is
  // W + 3 * (W / 1.8 - W) = 392.4 - 523.2 = -130.8 N: a foot pulling the robot through the floor. The lower clamp at
  // zero turns that into "no force", an offset of -W; without it the offset would be -523.2.
  LocomotionHeuristicConfig config;
  config.impulseScaling.scale = 3.0;
  const vector3_t offset = wrenchOffsetOf("impulse_scaling", config, leftSingleSupport(0.9), CONTACT_LEFT_INDEX);
  EXPECT_NEAR(offset.z(), -kWeight, 1e-9);
  EXPECT_NEAR(kWeight + offset.z(), 0.0, 1e-9) << "the reference itself is clamped to zero, not below";
}

TEST(LocomotionHeuristicFormulas, ImpulseScalingReadsEachFootsOwnDutyFactor) {
  // Every older case has both feet at the same beta and queries only the left foot, so reading stanceDutyFactor[0]
  // for every foot passes. Here: left W / 1.4 - W / 2 = 84.0857 N, right W / 1.1 - W / 2 = 160.5273 N.
  const LocomotionHeuristicConfig config;
  const WrenchHeuristicContext context = doubleSupport(0.7, 0.55);
  EXPECT_NEAR(wrenchOffsetOf("impulse_scaling", config, context, CONTACT_LEFT_INDEX).z(), 84.08571428571432, 1e-9);
  EXPECT_NEAR(wrenchOffsetOf("impulse_scaling", config, context, CONTACT_RIGHT_INDEX).z(), 160.52727272727273, 1e-9);
}

TEST(LocomotionHeuristicFormulas, ImpulseScalingDescribesTheFormulaItImplements) {
  // The banner used to print `f_z *= s/beta + 1-s`, the instantaneous form the implementation explicitly rejects
  // because it puts W/beta on the ground at every instant. The operator reads this line to know what is running.
  absl::StatusOr<std::unique_ptr<WrenchHeuristic>> heuristic = LocomotionHeuristicFactory::makeWrenchHeuristic("impulse_scaling");
  ASSERT_TRUE(heuristic.ok()) << heuristic.status().message();
  LocomotionHeuristicConfig config;
  config.impulseScaling.scale = 0.75;
  config.impulseScaling.minimumDutyFactor = 0.3;
  config.impulseScaling.maximumForceRatio = 1.5;
  ASSERT_TRUE((*heuristic)->configure(config, testModel()).ok());

  const std::string description = (*heuristic)->describe();
  EXPECT_TRUE(absl::StrContains(description, "f_z = W/n + 0.75 * (W/(F max(beta, 0.3)) - W/n)")) << description;
  EXPECT_TRUE(absl::StrContains(description, "[0, 1.5 * W/n]")) << description;
  EXPECT_FALSE(absl::StrContains(description, "*=")) << description;
  EXPECT_FALSE(absl::StrContains(description, "s/beta")) << description;
}

TEST(LocomotionHeuristicFormulas, SteppingDescriptionsNameTheStanceTerm) {
  // The stance terms are the part of these two formulas that is not Bledt's, and a banner that omitted them would show
  // an operator the constant gain while the controller runs a larger one. translational_stepping has one per axis, set
  // to different values, so that a banner dropping the lateral one, or printing the forward fraction in its place,
  // fails as well.
  LocomotionHeuristicConfig config;
  config.translationalStepping.forwardStanceFraction = 0.375;
  config.translationalStepping.lateralStanceFraction = 0.4375;
  config.inPlaceTurning.forwardStanceLever = 0.0625;
  for (const std::pair<const char*, const char*>& entry :
       {std::pair<const char*, const char*>{"translational_stepping", "0.375 * T_stance"},
        std::pair<const char*, const char*>{"translational_stepping", "0.4375 * T_stance"},
        std::pair<const char*, const char*>{"in_place_turning", "0.0625 * T_stance"}}) {
    absl::StatusOr<std::unique_ptr<FootholdHeuristic>> heuristic = LocomotionHeuristicFactory::makeFootholdHeuristic(entry.first);
    ASSERT_TRUE(heuristic.ok()) << heuristic.status().message();
    ASSERT_TRUE((*heuristic)->configure(config, testModel()).ok());
    EXPECT_TRUE(absl::StrContains((*heuristic)->describe(), entry.second)) << (*heuristic)->describe();
  }
}

/*=============================================== base-pose formulae (H42) ==============================================*/

TEST(LocomotionHeuristicFormulas, HeightCompensationIsAQuadraticInTheSpeedMagnitude) {
  // At |v| = 1 along x, v, v^2 and |v_x| are all the same number; at v = (1.2, 0.9) they are 1.5, 2.25 and 1.2.
  LocomotionHeuristicConfig config;
  config.heightCompensation.heightPerSpeedSquared = -0.01;
  config.heightCompensation.heightPerSpeed = -0.02;
  config.heightCompensation.heightOffset = 0.005;
  config.heightCompensation.maximumHeightOffset = 0.05;

  BasePoseHeuristicContext context;
  context.commandedVelocityInBaseFrame = vector2_t(1.2, 0.9);
  // -0.01 * 2.25 - 0.02 * 1.5 + 0.005 = -0.0475; linear-in-a2 gives -0.04, |v_x| gives -0.0334.
  const BasePoseOffset offset = basePoseOffsetOf("height_compensation", config, context);
  EXPECT_NEAR(offset.height, -0.0475, 1e-12);
  EXPECT_NEAR(offset.roll, 0.0, kTol);
  EXPECT_NEAR(offset.pitch, 0.0, kTol);

  // The POSITIVE side of the clamp: 0.2 - 0.0525 = 0.1475 -> 0.05.
  LocomotionHeuristicConfig rising = config;
  rising.heightCompensation.heightOffset = 0.2;
  EXPECT_NEAR(basePoseOffsetOf("height_compensation", rising, context).height, 0.05, 1e-12);
}

TEST(LocomotionHeuristicFormulas, OrientationCompensationClampsPitchAsWellAsRoll) {
  // The older test saturates only roll, so deleting the pitch clamp passes it.
  LocomotionHeuristicConfig config;
  config.orientationCompensation.pitchPerForwardVelocity = 0.0725;
  config.orientationCompensation.rollOffset = 0.01;
  config.orientationCompensation.maximumTilt = 0.1;

  BasePoseHeuristicContext forward;
  forward.commandedVelocityInBaseFrame = vector2_t(100.0, 0.0);
  const BasePoseOffset leaning = basePoseOffsetOf("orientation_compensation", config, forward);
  EXPECT_NEAR(leaning.pitch, 0.1, 1e-12);
  EXPECT_NEAR(leaning.roll, 0.01, 1e-12) << "roll follows the lateral command only";

  BasePoseHeuristicContext backward;
  backward.commandedVelocityInBaseFrame = vector2_t(-100.0, 0.0);
  EXPECT_NEAR(basePoseOffsetOf("orientation_compensation", config, backward).pitch, -0.1, 1e-12);
}

TEST(LocomotionHeuristicFormulas, PeriodicOrientationRollRunsAtItsOwnRateAndPhase) {
  // The shipped Atlas block uses the ROLL channel (pelvic obliquity at stride frequency); the older test sets only
  // pitch. Roll at 2 pi per cycle and pitch at 4 pi per cycle, so a roll computed with the pitch rate vanishes at
  // phase 0.25 instead of peaking.
  LocomotionHeuristicConfig config;
  config.periodicOrientation.rollAmplitude = 0.03;
  config.periodicOrientation.rollPhaseRate = 2.0 * M_PI;
  config.periodicOrientation.rollPhaseOffset = 0.0;
  config.periodicOrientation.pitchAmplitude = 0.02;
  config.periodicOrientation.pitchPhaseRate = 4.0 * M_PI;

  BasePoseHeuristicContext context;
  context.gaitPhase = 0.25;  // roll 0.03 sin(pi/2) = 0.03, pitch 0.02 sin(pi) = 0
  EXPECT_NEAR(basePoseOffsetOf("periodic_orientation", config, context).roll, 0.03, 1e-12);
  EXPECT_NEAR(basePoseOffsetOf("periodic_orientation", config, context).pitch, 0.0, 1e-12);
  context.gaitPhase = 0.75;  // roll 0.03 sin(3 pi/2) = -0.03
  EXPECT_NEAR(basePoseOffsetOf("periodic_orientation", config, context).roll, -0.03, 1e-12);
  context.gaitPhase = 0.125;  // roll 0.03 sin(pi/4), pitch 0.02 sin(pi/2) = 0.02
  EXPECT_NEAR(basePoseOffsetOf("periodic_orientation", config, context).roll, 0.03 * std::sqrt(0.5), 1e-12);
  EXPECT_NEAR(basePoseOffsetOf("periodic_orientation", config, context).pitch, 0.02, 1e-12);
  EXPECT_NEAR(basePoseOffsetOf("periodic_orientation", config, context).height, 0.0, kTol);

  // The roll phase offset is its own: pi/2 at phase 0 is a peak.
  LocomotionHeuristicConfig shifted = config;
  shifted.periodicOrientation.rollPhaseOffset = M_PI / 2.0;
  context.gaitPhase = 0.0;
  EXPECT_NEAR(basePoseOffsetOf("periodic_orientation", shifted, context).roll, 0.03, 1e-12);
  EXPECT_NEAR(basePoseOffsetOf("periodic_orientation", shifted, context).pitch, 0.0, 1e-12);
}

/*========================================== Create(): refusals and warnings ==========================================*/

TEST(LocomotionHeuristicFormulas, CreateRefusesAFootholdListThatNothingKeepsApart) {
  // With the step width at zero and no hip_centered_stepping, every other foothold heuristic corrects a target on the
  // stance foot's own lateral line: the swing foot is aimed at the stance foot. SA01, G1 and R1 ship stepWidth 0.
  LocomotionHeuristicConfig config;
  config.formulation.foothold = {"capture_point", "translational_stepping"};
  for (const scalar_t stepWidth : {0.0, -0.1, std::numeric_limits<scalar_t>::quiet_NaN()}) {
    LocomotionHeuristicEnvironment environment;
    environment.nominalStepWidth = stepWidth;
    const absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
        LocomotionHeuristicLayer::Create(config, testModel(), environment);
    ASSERT_FALSE(layer.ok()) << "stepWidth " << stepWidth;
    EXPECT_EQ(layer.status().code(), absl::StatusCode::kInvalidArgument);
    const std::string message(layer.status().message());
    EXPECT_TRUE(absl::StrContains(message, "model_settings.nominal_foothold.stepWidth")) << message;
    EXPECT_TRUE(absl::StrContains(message, "hip_centered_stepping")) << message;
  }
}

TEST(LocomotionHeuristicFormulas, CreateAcceptsAFootholdListKeptApartByTheHipsOrByTheStepWidth) {
  LocomotionHeuristicEnvironment zeroWidth;
  zeroWidth.nominalStepWidth = 0.0;

  // The hips keep the feet apart.
  LocomotionHeuristicConfig anchored;
  anchored.formulation.foothold = {"capture_point", "hip_centered_stepping"};
  const absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> withHips =
      LocomotionHeuristicLayer::Create(anchored, testModel(), zeroWidth);
  EXPECT_TRUE(withHips.ok()) << withHips.status().message();

  // The step width keeps the feet apart.
  LocomotionHeuristicConfig unanchored;
  unanchored.formulation.foothold = {"capture_point"};
  const absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> withWidth =
      LocomotionHeuristicLayer::Create(unanchored, testModel(), steppingEnvironment());
  EXPECT_TRUE(withWidth.ok()) << withWidth.status().message();

  // The check is about the FOOTHOLD channel: the other two do not place feet and load at any step width.
  LocomotionHeuristicConfig noFoothold;
  noFoothold.formulation.basePose = {"orientation_compensation"};
  noFoothold.formulation.wrench = {"impulse_scaling"};
  const absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> otherChannels =
      LocomotionHeuristicLayer::Create(noFoothold, testModel(), zeroWidth);
  EXPECT_TRUE(otherChannels.ok()) << otherChannels.status().message();
}

TEST(LocomotionHeuristicFormulas, CreateWarnsThatBasePoseHeuristicsDoNothingUnderAcomTracking) {
  // The cost factory zeroes Q's base-pose block under ACoM tracking, so the shaped reference reaches no cost.
  LocomotionHeuristicConfig config;
  config.formulation.basePose = {"orientation_compensation", "height_compensation"};
  LocomotionHeuristicEnvironment environment = steppingEnvironment();
  environment.listsComAndAcomTrackingCost = true;

  WarningLog log;
  ASSERT_TRUE(LocomotionHeuristicLayer::Create(config, testModel(), environment).ok());
  EXPECT_EQ(log.count("com_and_acom_tracking_cost"), 1u) << log.all();
  EXPECT_EQ(log.count("locomotion_heuristics.base_pose lists orientation_compensation, height_compensation"), 1u) << log.all();
}

TEST(LocomotionHeuristicFormulas, CreateWarnsThatFootholdHeuristicsDoNothingWithTheFootPositionUntracked) {
  LocomotionHeuristicConfig config;
  config.formulation.foothold = {"hip_centered_stepping", "translational_stepping"};
  LocomotionHeuristicEnvironment environment = steppingEnvironment();
  environment.footPositionIsUntracked = true;

  WarningLog log;
  ASSERT_TRUE(LocomotionHeuristicLayer::Create(config, testModel(), environment).ok());
  EXPECT_EQ(log.count("task_space_foot_cost_weights"), 1u) << log.all();
  EXPECT_EQ(log.count("locomotion_heuristics.foothold lists hip_centered_stepping, translational_stepping"), 1u) << log.all();
}

TEST(LocomotionHeuristicFormulas, CreateWarnsAboutTheForwardKinematicsCostOfAHorizontalForceUnderBasisVectors) {
  LocomotionHeuristicEnvironment environment = steppingEnvironment();
  environment.usesContactBasisVectorInputs = true;

  LocomotionHeuristicConfig horizontal;
  horizontal.formulation.wrench = {"impulse_scaling", "centripetal_acceleration"};
  {
    WarningLog log;
    ASSERT_TRUE(LocomotionHeuristicLayer::Create(horizontal, testModel(), environment).ok());
    EXPECT_EQ(log.count("contactInputParameterization is basis_vectors"), 1u) << log.all();
  }

  // A vertical-only wrench list never takes the forward-kinematics path, so there is nothing to warn about.
  LocomotionHeuristicConfig vertical;
  vertical.formulation.wrench = {"impulse_scaling"};
  {
    WarningLog log;
    ASSERT_TRUE(LocomotionHeuristicLayer::Create(vertical, testModel(), environment).ok());
    EXPECT_EQ(log.count("contactInputParameterization is basis_vectors"), 0u) << log.all();
  }
}

TEST(LocomotionHeuristicFormulas, CreateIsSilentWhenEveryListedHeuristicCanAct) {
  // The warnings above are only useful if they are specific: one that fired on every configuration would be noise the
  // operator learns to ignore. The same lists as above, in an environment where each channel reaches a cost.
  LocomotionHeuristicConfig config;
  config.formulation.basePose = {"orientation_compensation", "height_compensation"};
  config.formulation.foothold = {"hip_centered_stepping", "translational_stepping"};
  config.formulation.wrench = {"impulse_scaling", "centripetal_acceleration"};
  {
    WarningLog log;
    ASSERT_TRUE(LocomotionHeuristicLayer::Create(config, testModel(), steppingEnvironment()).ok());
    EXPECT_TRUE(log.warnings().empty()) << log.all();
  }

  // And an environment that WOULD make a channel inert is not warned about when that channel lists nothing.
  LocomotionHeuristicEnvironment hostile = steppingEnvironment();
  hostile.listsComAndAcomTrackingCost = true;
  hostile.footPositionIsUntracked = true;
  hostile.usesContactBasisVectorInputs = true;
  const LocomotionHeuristicConfig empty;
  {
    WarningLog log;
    ASSERT_TRUE(LocomotionHeuristicLayer::Create(empty, testModel(), hostile).ok());
    EXPECT_TRUE(log.warnings().empty()) << log.all();
  }

  // Positive control: the listed configuration in that environment does warn, once per inert channel.
  {
    WarningLog log;
    ASSERT_TRUE(LocomotionHeuristicLayer::Create(config, testModel(), hostile).ok());
    EXPECT_EQ(log.count("com_and_acom_tracking_cost"), 1u) << log.all();
    EXPECT_EQ(log.count("task_space_foot_cost_weights"), 1u) << log.all();
    EXPECT_EQ(log.count("contactInputParameterization is basis_vectors"), 1u) << log.all();
  }
}

TEST(LocomotionHeuristicFormulas, CreateLogsTheFormulationWarnings) {
  LocomotionHeuristicConfig withoutHip;
  withoutHip.formulation.foothold = {"capture_point"};
  WarningLog log;
  ASSERT_TRUE(LocomotionHeuristicLayer::Create(withoutHip, testModel(), steppingEnvironment()).ok());
  EXPECT_EQ(log.count("does not list 'hip_centered_stepping'"), 1u) << log.all();
}

/*===================================================== reconfigure (H44) =====================================================*/

TEST(LocomotionHeuristicFormulas, ReconfigureAppliesAllOfAReloadOrNoneOfIt) {
  LocomotionHeuristicConfig config;
  config.formulation.basePose = {"orientation_compensation"};
  config.formulation.foothold = {"translational_stepping"};
  config.formulation.wrench = {"impulse_scaling"};
  config.orientationCompensation.pitchPerForwardVelocity = 0.05;
  config.translationalStepping.forwardPerForwardVelocity = 0.2;
  absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
      LocomotionHeuristicLayer::Create(config, testModel(), steppingEnvironment());
  ASSERT_TRUE(layer.ok()) << layer.status().message();

  BasePoseHeuristicContext basePose;
  basePose.commandedVelocityInBaseFrame = vector2_t(1.0, 0.0);
  FootholdHeuristicContext foothold = leftFoot();
  foothold.commandedVelocity = vector2_t(1.0, 0.0);
  const WrenchHeuristicContext wrench = doubleSupport(0.6, 0.6);
  EXPECT_NEAR((*layer)->basePoseOffset(basePose).pitch, 0.05, 1e-12);
  EXPECT_NEAR((*layer)->footholdOffset(foothold).x(), 0.2, 1e-12);
  EXPECT_NEAR((*layer)->wrenchOffset(wrench, CONTACT_LEFT_INDEX).z(), 130.8, 1e-9);

  // A reload that changes a coefficient in EVERY channel and is invalid elsewhere must change none of them. Copying the
  // running values into the invalid reload, as the older test did, cannot tell a rejected reload from a half-applied one.
  LocomotionHeuristicConfig invalid = config;
  invalid.orientationCompensation.pitchPerForwardVelocity = -0.03;
  invalid.translationalStepping.forwardPerForwardVelocity = 0.3;
  invalid.impulseScaling.scale = 0.5;
  invalid.capturePoint.maximumOffset = 0.0;  // invalid, and in a heuristic that is not even listed
  EXPECT_FALSE((*layer)->reconfigure(invalid).ok());
  EXPECT_NEAR((*layer)->basePoseOffset(basePose).pitch, 0.05, 1e-12);
  EXPECT_NEAR((*layer)->footholdOffset(foothold).x(), 0.2, 1e-12);
  EXPECT_NEAR((*layer)->wrenchOffset(wrench, CONTACT_LEFT_INDEX).z(), 130.8, 1e-9);

  // Positive control: the same reload made valid applies all three.
  LocomotionHeuristicConfig valid = invalid;
  valid.capturePoint.maximumOffset = 0.25;
  ASSERT_TRUE((*layer)->reconfigure(valid).ok());
  EXPECT_NEAR((*layer)->basePoseOffset(basePose).pitch, -0.03, 1e-12);
  EXPECT_NEAR((*layer)->footholdOffset(foothold).x(), 0.3, 1e-12);
  EXPECT_NEAR((*layer)->wrenchOffset(wrench, CONTACT_LEFT_INDEX).z(), 65.4, 1e-9);
}

TEST(LocomotionHeuristicFormulas, ReconfigureIgnoresAListChangedOnDiskAndReportsEachEditOnce) {
  LocomotionHeuristicConfig config;
  config.formulation.basePose = {"orientation_compensation"};
  config.orientationCompensation.pitchPerForwardVelocity = 0.05;
  absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
      LocomotionHeuristicLayer::Create(config, testModel(), steppingEnvironment());
  ASSERT_TRUE(layer.ok()) << layer.status().message();

  BasePoseHeuristicContext context;
  context.commandedVelocityInBaseFrame = vector2_t(1.0, 0.0);
  FootholdHeuristicContext pushed = leftFoot();
  pushed.measuredVelocity = vector2_t(0.5, 0.0);

  WarningLog log;
  // An edit that adds height_compensation on disk, next to a retuned pitch. The list is structural and is NOT picked up
  // - the height stays zero although the new heightOffset is 0.02 - while the coefficient of the running list IS.
  LocomotionHeuristicConfig edited = config;
  edited.formulation.basePose = {"orientation_compensation", "height_compensation"};
  edited.heightCompensation.heightOffset = 0.02;
  edited.orientationCompensation.pitchPerForwardVelocity = -0.03;
  ASSERT_TRUE((*layer)->reconfigure(edited).ok());
  EXPECT_NEAR((*layer)->basePoseOffset(context).height, 0.0, kTol);
  EXPECT_NEAR((*layer)->basePoseOffset(context).pitch, -0.03, 1e-12) << "the coefficient must still be applied";
  EXPECT_EQ(log.count("base_pose list has changed on disk"), 1u) << log.all();

  // A slider drag re-reads the same file: the same edit is not reported again.
  edited.orientationCompensation.pitchPerForwardVelocity = -0.04;
  ASSERT_TRUE((*layer)->reconfigure(edited).ok());
  EXPECT_NEAR((*layer)->basePoseOffset(context).pitch, -0.04, 1e-12);
  EXPECT_EQ(log.count("changed on disk"), 1u) << log.all();

  // A further, DIFFERENT edit is reported, naming the list it touched - and a foothold heuristic added on disk is not
  // instantiated either (capture_point would answer this push).
  LocomotionHeuristicConfig editedAgain = edited;
  editedAgain.formulation.foothold = {"capture_point"};
  ASSERT_TRUE((*layer)->reconfigure(editedAgain).ok());
  EXPECT_EQ(log.count("foothold list has changed on disk"), 1u) << log.all();
  EXPECT_TRUE((*layer)->footholdEmpty());
  EXPECT_TRUE((*layer)->footholdOffset(pushed).isZero(kTol));

  // Reverting the file to the running lists is not a change at all.
  const size_t reported = log.count("changed on disk");
  LocomotionHeuristicConfig reverted = config;
  reverted.orientationCompensation.pitchPerForwardVelocity = 0.07;
  ASSERT_TRUE((*layer)->reconfigure(reverted).ok());
  EXPECT_NEAR((*layer)->basePoseOffset(context).pitch, 0.07, 1e-12);
  EXPECT_EQ(log.count("changed on disk"), reported) << log.all();

  // ...but it ends the edit it reverted: making that SAME edit again is news. The operator re-adds the heuristics
  // after a revert and hot-reloads; if the layer still remembered having reported them before the revert, the reload
  // would be silent and the operator would believe they were running. It must be the most recently reported edit
  // (editedAgain) that is re-applied, since re-applying an older one differs from what was last reported anyway.
  const size_t basePoseReported = log.count("base_pose list has changed on disk");
  const size_t footholdReported = log.count("foothold list has changed on disk");
  ASSERT_TRUE((*layer)->reconfigure(editedAgain).ok());
  EXPECT_EQ(log.count("base_pose list has changed on disk"), basePoseReported + 1) << log.all();
  EXPECT_EQ(log.count("foothold list has changed on disk"), footholdReported + 1) << log.all();
  EXPECT_NEAR((*layer)->basePoseOffset(context).pitch, -0.04, 1e-12) << "the coefficient must still be applied";
  EXPECT_TRUE((*layer)->footholdEmpty()) << "and the list still must not be";
}

TEST(LocomotionHeuristicFormulas, ReconfigureReportsAnEditThatEmptiesEveryList) {
  // The layer remembers which on-disk lists it has already reported, so that a slider drag does not repeat the warning.
  // If that memory starts out as the EMPTY formulation rather than the running one, an edit that comments out every
  // heuristic looks "already reported" from the first reload on, and the operator who disabled the heuristics on disk
  // is never told they are still running.
  LocomotionHeuristicConfig config;
  config.formulation.basePose = {"orientation_compensation"};
  config.orientationCompensation.pitchPerForwardVelocity = 0.05;
  absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
      LocomotionHeuristicLayer::Create(config, testModel(), steppingEnvironment());
  ASSERT_TRUE(layer.ok()) << layer.status().message();

  BasePoseHeuristicContext context;
  context.commandedVelocityInBaseFrame = vector2_t(1.0, 0.0);
  ASSERT_NEAR((*layer)->basePoseOffset(context).pitch, 0.05, 1e-12);

  WarningLog log;
  LocomotionHeuristicConfig emptied;  // every list empty, as after commenting out every name
  emptied.orientationCompensation.pitchPerForwardVelocity = -0.03;
  ASSERT_TRUE((*layer)->reconfigure(emptied).ok());
  EXPECT_EQ(log.count("changed on disk"), 1u) << log.all();
  EXPECT_EQ(log.count("base_pose list has changed on disk"), 1u) << log.all();
  // The running list is kept, and its coefficient follows the file: -0.03 * 1.0.
  EXPECT_FALSE((*layer)->empty());
  EXPECT_NEAR((*layer)->basePoseOffset(context).pitch, -0.03, 1e-12);

  // Reported once: the same emptied file re-read by the next slider drag is not reported again.
  emptied.orientationCompensation.pitchPerForwardVelocity = -0.02;
  ASSERT_TRUE((*layer)->reconfigure(emptied).ok());
  EXPECT_EQ(log.count("changed on disk"), 1u) << log.all();
  EXPECT_NEAR((*layer)->basePoseOffset(context).pitch, -0.02, 1e-12);
}

/*======================================================= the loader =======================================================*/

TEST(LocomotionHeuristicFormulas, LoaderRejectsAListItemThatIsAMapOrANestedList) {
  // `- capture_point:` (a trailing colon left over from uncommenting) is the one-entry map {capture_point: null}, and
  // `- [capture_point]` a nested list. Both used to be skipped, starting the controller with the heuristic silently off.
  const std::vector<std::string> malformed{"locomotion_heuristics:\n  foothold:\n    - capture_point:\n",
                                           "locomotion_heuristics:\n  foothold:\n    - [capture_point]\n",
                                           "locomotion_heuristics:\n  foothold:\n    - {name: capture_point}\n"};
  for (size_t i = 0; i < malformed.size(); ++i) {
    const std::string file = writeTemp(absl::StrCat("heuristics_malformed_item_", i, ".yaml"), malformed[i]);
    const absl::StatusOr<LocomotionHeuristicConfig> config = loadLocomotionHeuristicConfig(file);
    ASSERT_FALSE(config.ok()) << malformed[i];
    EXPECT_EQ(config.status().code(), absl::StatusCode::kInvalidArgument);
    const std::string message(config.status().message());
    EXPECT_TRUE(absl::StrContains(message, "locomotion_heuristics.foothold[0]")) << message;
    EXPECT_TRUE(absl::StrContains(message, "must be a heuristic name")) << message;
  }
  // The trailing-colon case names the heuristic it was meant to be.
  const std::string trailingColon = writeTemp("heuristics_trailing_colon.yaml", malformed.front());
  EXPECT_TRUE(absl::StrContains(loadLocomotionHeuristicConfig(trailingColon).status().message(), "capture_point"));
}

TEST(LocomotionHeuristicFormulas, LoaderStillAcceptsADashFollowedOnlyByAComment) {
  // A dash with nothing but a comment after it is a null item: it names nothing, and it is not an error.
  const std::string bare = writeTemp("heuristics_bare_dash.yaml", "locomotion_heuristics:\n  foothold:\n    - # capture_point\n");
  const absl::StatusOr<LocomotionHeuristicConfig> empty = loadLocomotionHeuristicConfig(bare);
  ASSERT_TRUE(empty.ok()) << empty.status().message();
  EXPECT_TRUE(empty->formulation.foothold.empty());

  // Positive control: the same null item next to a real name keeps the name.
  const std::string mixed = writeTemp("heuristics_bare_dash_and_name.yaml",
                                      "locomotion_heuristics:\n  foothold:\n    - # translational_stepping\n    - capture_point\n");
  const absl::StatusOr<LocomotionHeuristicConfig> one = loadLocomotionHeuristicConfig(mixed);
  ASSERT_TRUE(one.ok()) << one.status().message();
  EXPECT_EQ(one->formulation.foothold, std::vector<std::string>{"capture_point"});
}

TEST(LocomotionHeuristicFormulas, LoaderRoundTripsEveryCoefficientKey) {
  // Every key the loader reads, written with a value distinct from its default and from every other key's, and read
  // back into the member written out by hand. A misspelled key string in the loader leaves the default in place; two
  // keys wired to each other's member swap two distinct numbers. Both fail here.
  const std::vector<std::string> expectedKeys{"orientation_compensation.rollPerLateralVelocity",
                                              "orientation_compensation.rollOffset",
                                              "orientation_compensation.pitchPerForwardVelocity",
                                              "orientation_compensation.pitchOffset",
                                              "orientation_compensation.maximumTilt",
                                              "periodic_orientation.rollAmplitude",
                                              "periodic_orientation.rollPhaseRate",
                                              "periodic_orientation.rollPhaseOffset",
                                              "periodic_orientation.pitchAmplitude",
                                              "periodic_orientation.pitchPhaseRate",
                                              "periodic_orientation.pitchPhaseOffset",
                                              "height_compensation.heightPerSpeedSquared",
                                              "height_compensation.heightPerSpeed",
                                              "height_compensation.heightOffset",
                                              "height_compensation.maximumHeightOffset",
                                              "hip_centered_stepping.lateralScale",
                                              "hip_centered_stepping.longitudinalScale",
                                              "capture_point.gain",
                                              "capture_point.comHeightOverride",
                                              "capture_point.gravity",
                                              "capture_point.maximumOffset",
                                              "translational_stepping.forwardPerForwardVelocity",
                                              "translational_stepping.forwardStanceFraction",
                                              "translational_stepping.forwardOffset",
                                              "translational_stepping.lateralPerLateralVelocity",
                                              "translational_stepping.lateralStanceFraction",
                                              "translational_stepping.lateralOffset",
                                              "in_place_turning.forwardPerYawRate",
                                              "in_place_turning.forwardStanceLever",
                                              "in_place_turning.forwardOffset",
                                              "in_place_turning.lateralPerYawRate",
                                              "in_place_turning.lateralOffset",
                                              "high_speed_turning.forwardPerCrossTerm",
                                              "high_speed_turning.forwardOffset",
                                              "high_speed_turning.lateralPerCrossTerm",
                                              "high_speed_turning.lateralOffset",
                                              "impulse_scaling.scale",
                                              "impulse_scaling.minimumDutyFactor",
                                              "impulse_scaling.maximumForceRatio",
                                              "centripetal_acceleration.scale",
                                              "centripetal_acceleration.maximumForce",
                                              "centripetal_acceleration.maximumForceRatioOfWeight"};
  ASSERT_EQ(expectedKeys.size(), 42u);
  std::vector<std::string> readKeys = locomotionHeuristicCoefficientKeys();
  std::vector<std::string> sortedExpected = expectedKeys;
  std::sort(readKeys.begin(), readKeys.end());
  std::sort(sortedExpected.begin(), sortedExpected.end());
  EXPECT_EQ(readKeys, sortedExpected) << "the key query must list exactly the keys written below, each once";

  const std::string file = writeTemp("heuristics_every_key.yaml", R"(locomotion_heuristics:
  orientation_compensation:
    rollPerLateralVelocity: -0.101
    rollOffset: 0.102
    pitchPerForwardVelocity: 0.103
    pitchOffset: -0.104
    maximumTilt: 0.105
  periodic_orientation:
    rollAmplitude: 0.106
    rollPhaseRate: 0.107
    rollPhaseOffset: 0.108
    pitchAmplitude: 0.109
    pitchPhaseRate: 0.110
    pitchPhaseOffset: 0.111
  height_compensation:
    heightPerSpeedSquared: 0.112
    heightPerSpeed: -0.113
    heightOffset: 0.114
    maximumHeightOffset: 0.115
  hip_centered_stepping:
    lateralScale: 0.116
    longitudinalScale: 0.117
  capture_point:
    gain: 0.118
    comHeightOverride: 0.119
    gravity: 0.120
    maximumOffset: 0.121
  translational_stepping:
    forwardPerForwardVelocity: 0.122
    forwardStanceFraction: 0.123
    forwardOffset: 0.124
    lateralPerLateralVelocity: 0.125
    lateralStanceFraction: 0.126
    lateralOffset: -0.127
  in_place_turning:
    forwardPerYawRate: 0.128
    forwardStanceLever: 0.129
    forwardOffset: 0.130
    lateralPerYawRate: 0.131
    lateralOffset: 0.132
  high_speed_turning:
    forwardPerCrossTerm: 0.133
    forwardOffset: 0.134
    lateralPerCrossTerm: 0.135
    lateralOffset: 0.136
  impulse_scaling:
    scale: 0.137
    minimumDutyFactor: 0.138
    maximumForceRatio: 1.139
  centripetal_acceleration:
    scale: 0.140
    maximumForce: 0.141
    maximumForceRatioOfWeight: 0.142
)");
  const absl::StatusOr<LocomotionHeuristicConfig> loaded = loadLocomotionHeuristicConfig(file);
  ASSERT_TRUE(loaded.ok()) << loaded.status().message();
  const LocomotionHeuristicConfig& config = *loaded;

  EXPECT_DOUBLE_EQ(config.orientationCompensation.rollPerLateralVelocity, -0.101);
  EXPECT_DOUBLE_EQ(config.orientationCompensation.rollOffset, 0.102);
  EXPECT_DOUBLE_EQ(config.orientationCompensation.pitchPerForwardVelocity, 0.103);
  EXPECT_DOUBLE_EQ(config.orientationCompensation.pitchOffset, -0.104);
  EXPECT_DOUBLE_EQ(config.orientationCompensation.maximumTilt, 0.105);

  EXPECT_DOUBLE_EQ(config.periodicOrientation.rollAmplitude, 0.106);
  EXPECT_DOUBLE_EQ(config.periodicOrientation.rollPhaseRate, 0.107);
  EXPECT_DOUBLE_EQ(config.periodicOrientation.rollPhaseOffset, 0.108);
  EXPECT_DOUBLE_EQ(config.periodicOrientation.pitchAmplitude, 0.109);
  EXPECT_DOUBLE_EQ(config.periodicOrientation.pitchPhaseRate, 0.110);
  EXPECT_DOUBLE_EQ(config.periodicOrientation.pitchPhaseOffset, 0.111);

  EXPECT_DOUBLE_EQ(config.heightCompensation.heightPerSpeedSquared, 0.112);
  EXPECT_DOUBLE_EQ(config.heightCompensation.heightPerSpeed, -0.113);
  EXPECT_DOUBLE_EQ(config.heightCompensation.heightOffset, 0.114);
  EXPECT_DOUBLE_EQ(config.heightCompensation.maximumHeightOffset, 0.115);

  EXPECT_DOUBLE_EQ(config.hipCenteredStepping.lateralScale, 0.116);
  EXPECT_DOUBLE_EQ(config.hipCenteredStepping.longitudinalScale, 0.117);

  EXPECT_DOUBLE_EQ(config.capturePoint.gain, 0.118);
  EXPECT_DOUBLE_EQ(config.capturePoint.comHeightOverride, 0.119);
  EXPECT_DOUBLE_EQ(config.capturePoint.gravity, 0.120);
  EXPECT_DOUBLE_EQ(config.capturePoint.maximumOffset, 0.121);

  EXPECT_DOUBLE_EQ(config.translationalStepping.forwardPerForwardVelocity, 0.122);
  EXPECT_DOUBLE_EQ(config.translationalStepping.forwardStanceFraction, 0.123);
  EXPECT_DOUBLE_EQ(config.translationalStepping.forwardOffset, 0.124);
  EXPECT_DOUBLE_EQ(config.translationalStepping.lateralPerLateralVelocity, 0.125);
  EXPECT_DOUBLE_EQ(config.translationalStepping.lateralStanceFraction, 0.126);
  EXPECT_DOUBLE_EQ(config.translationalStepping.lateralOffset, -0.127);

  EXPECT_DOUBLE_EQ(config.inPlaceTurning.forwardPerYawRate, 0.128);
  EXPECT_DOUBLE_EQ(config.inPlaceTurning.forwardStanceLever, 0.129);
  EXPECT_DOUBLE_EQ(config.inPlaceTurning.forwardOffset, 0.130);
  EXPECT_DOUBLE_EQ(config.inPlaceTurning.lateralPerYawRate, 0.131);
  EXPECT_DOUBLE_EQ(config.inPlaceTurning.lateralOffset, 0.132);

  EXPECT_DOUBLE_EQ(config.highSpeedTurning.forwardPerCrossTerm, 0.133);
  EXPECT_DOUBLE_EQ(config.highSpeedTurning.forwardOffset, 0.134);
  EXPECT_DOUBLE_EQ(config.highSpeedTurning.lateralPerCrossTerm, 0.135);
  EXPECT_DOUBLE_EQ(config.highSpeedTurning.lateralOffset, 0.136);

  EXPECT_DOUBLE_EQ(config.impulseScaling.scale, 0.137);
  EXPECT_DOUBLE_EQ(config.impulseScaling.minimumDutyFactor, 0.138);
  EXPECT_DOUBLE_EQ(config.impulseScaling.maximumForceRatio, 1.139);

  EXPECT_DOUBLE_EQ(config.centripetalAcceleration.scale, 0.140);
  EXPECT_DOUBLE_EQ(config.centripetalAcceleration.maximumForce, 0.141);
  EXPECT_DOUBLE_EQ(config.centripetalAcceleration.maximumForceRatioOfWeight, 0.142);
}

namespace {

/** One way of putting one key out of its admissible range, and the key the rejection has to name. */
struct OutOfRange {
  const char* key;
  void (*breakIt)(LocomotionHeuristicConfig&);
};

}  // namespace

TEST(LocomotionHeuristicFormulas, ValidateRejectsEveryOutOfRangeKeyByName) {
  // One case per branch of validate(), each asserting the message names the key the operator has to change.
  const std::vector<OutOfRange> cases{
      {"locomotion_heuristics.base_pose", [](LocomotionHeuristicConfig& c) { c.formulation.basePose = {"no_such_heuristic"}; }},
      {"locomotion_heuristics.foothold", [](LocomotionHeuristicConfig& c) { c.formulation.foothold = {"capture_point", "capture_point"}; }},
      {"locomotion_heuristics.orientation_compensation.maximumTilt",
       [](LocomotionHeuristicConfig& c) { c.orientationCompensation.maximumTilt = 0.0; }},
      {"locomotion_heuristics.orientation_compensation.maximumTilt",
       [](LocomotionHeuristicConfig& c) { c.orientationCompensation.maximumTilt = -0.1; }},
      {"locomotion_heuristics.height_compensation.maximumHeightOffset",
       [](LocomotionHeuristicConfig& c) { c.heightCompensation.maximumHeightOffset = -0.01; }},
      {"locomotion_heuristics.capture_point.comHeightOverride",
       [](LocomotionHeuristicConfig& c) { c.capturePoint.comHeightOverride = -0.5; }},
      {"locomotion_heuristics.capture_point.gravity", [](LocomotionHeuristicConfig& c) { c.capturePoint.gravity = 0.0; }},
      {"locomotion_heuristics.capture_point.gravity", [](LocomotionHeuristicConfig& c) { c.capturePoint.gravity = -9.81; }},
      {"locomotion_heuristics.capture_point.maximumOffset", [](LocomotionHeuristicConfig& c) { c.capturePoint.maximumOffset = 0.0; }},
      {"locomotion_heuristics.capture_point.gain", [](LocomotionHeuristicConfig& c) { c.capturePoint.gain = -1.0; }},
      {"locomotion_heuristics.hip_centered_stepping.lateralScale",
       [](LocomotionHeuristicConfig& c) { c.hipCenteredStepping.lateralScale = -1.0; }},
      {"locomotion_heuristics.hip_centered_stepping.longitudinalScale",
       [](LocomotionHeuristicConfig& c) { c.hipCenteredStepping.longitudinalScale = -1.0; }},
      {"locomotion_heuristics.impulse_scaling.scale", [](LocomotionHeuristicConfig& c) { c.impulseScaling.scale = -0.5; }},
      {"locomotion_heuristics.impulse_scaling.minimumDutyFactor",
       [](LocomotionHeuristicConfig& c) { c.impulseScaling.minimumDutyFactor = 0.0; }},
      {"locomotion_heuristics.impulse_scaling.minimumDutyFactor",
       [](LocomotionHeuristicConfig& c) { c.impulseScaling.minimumDutyFactor = 1.5; }},
      {"locomotion_heuristics.impulse_scaling.maximumForceRatio",
       [](LocomotionHeuristicConfig& c) { c.impulseScaling.maximumForceRatio = 0.99; }},
      {"locomotion_heuristics.centripetal_acceleration.scale",
       [](LocomotionHeuristicConfig& c) { c.centripetalAcceleration.scale = -0.5; }},
      {"locomotion_heuristics.centripetal_acceleration.maximumForce",
       [](LocomotionHeuristicConfig& c) { c.centripetalAcceleration.maximumForce = -10.0; }},
      {"locomotion_heuristics.centripetal_acceleration.maximumForceRatioOfWeight",
       [](LocomotionHeuristicConfig& c) { c.centripetalAcceleration.maximumForceRatioOfWeight = -0.3; }},
      {"locomotion_heuristics.centripetal_acceleration",
       [](LocomotionHeuristicConfig& c) {
         c.centripetalAcceleration.maximumForce = 0.0;
         c.centripetalAcceleration.maximumForceRatioOfWeight = 0.0;
       }},
  };

  // Positive control: every case starts from a configuration that validates.
  ASSERT_TRUE(LocomotionHeuristicConfig().validate().ok());
  for (const OutOfRange& entry : cases) {
    LocomotionHeuristicConfig config;
    entry.breakIt(config);
    const absl::Status status = config.validate();
    ASSERT_FALSE(status.ok()) << entry.key;
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << entry.key;
    EXPECT_TRUE(absl::StrContains(status.message(), entry.key)) << "expected '" << entry.key << "' in: " << status.message();
  }
}

TEST(LocomotionHeuristicFormulas, ValidateAcceptsEachRangeBoundary) {
  // The boundaries each check is documented to allow. A `<=` where a `<` belongs would refuse the legitimate no-op
  // (a zero scale, a zero override meaning "measured") or the extreme a robot may genuinely want (beta floor 1).
  const std::vector<OutOfRange> boundaries{
      {"height_compensation.maximumHeightOffset = 0", [](LocomotionHeuristicConfig& c) { c.heightCompensation.maximumHeightOffset = 0.0; }},
      {"capture_point.comHeightOverride = 0", [](LocomotionHeuristicConfig& c) { c.capturePoint.comHeightOverride = 0.0; }},
      {"capture_point.gain = 0", [](LocomotionHeuristicConfig& c) { c.capturePoint.gain = 0.0; }},
      {"hip_centered_stepping.lateralScale = 0", [](LocomotionHeuristicConfig& c) { c.hipCenteredStepping.lateralScale = 0.0; }},
      {"hip_centered_stepping.longitudinalScale = 0", [](LocomotionHeuristicConfig& c) { c.hipCenteredStepping.longitudinalScale = 0.0; }},
      {"impulse_scaling.scale = 0", [](LocomotionHeuristicConfig& c) { c.impulseScaling.scale = 0.0; }},
      {"impulse_scaling.minimumDutyFactor = 1", [](LocomotionHeuristicConfig& c) { c.impulseScaling.minimumDutyFactor = 1.0; }},
      {"impulse_scaling.maximumForceRatio = 1", [](LocomotionHeuristicConfig& c) { c.impulseScaling.maximumForceRatio = 1.0; }},
      {"centripetal_acceleration.scale = 0", [](LocomotionHeuristicConfig& c) { c.centripetalAcceleration.scale = 0.0; }},
      {"centripetal_acceleration.maximumForceRatioOfWeight = 0 with maximumForce set",
       [](LocomotionHeuristicConfig& c) {
         c.centripetalAcceleration.maximumForce = 50.0;
         c.centripetalAcceleration.maximumForceRatioOfWeight = 0.0;
       }},
      {"fitted coefficients of either sign",
       [](LocomotionHeuristicConfig& c) {
         c.orientationCompensation.pitchPerForwardVelocity = -0.2;
         c.translationalStepping.forwardStanceFraction = -0.1;
         c.inPlaceTurning.forwardStanceLever = -0.1;
       }},
  };
  for (const OutOfRange& entry : boundaries) {
    LocomotionHeuristicConfig config;
    entry.breakIt(config);
    const absl::Status status = config.validate();
    EXPECT_TRUE(status.ok()) << entry.key << ": " << status.message();
  }
}

TEST(LocomotionHeuristicFormulas, LoaderRejectsAnOutOfRangeValueReadFromTheFileByName) {
  // validate() runs on what was READ, so a negative gain in the file is refused at start-up rather than stepping the
  // robot against the velocity error.
  const std::string file = writeTemp("heuristics_negative_gain.yaml", "locomotion_heuristics:\n  capture_point:\n    gain: -1.0\n");
  const absl::StatusOr<LocomotionHeuristicConfig> config = loadLocomotionHeuristicConfig(file);
  ASSERT_FALSE(config.ok());
  EXPECT_EQ(config.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(config.status().message(), "locomotion_heuristics.capture_point.gain")) << config.status().message();

  // Positive control: the same key at an admissible value loads.
  const std::string good = writeTemp("heuristics_positive_gain.yaml", "locomotion_heuristics:\n  capture_point:\n    gain: 0.5\n");
  const absl::StatusOr<LocomotionHeuristicConfig> loaded = loadLocomotionHeuristicConfig(good);
  ASSERT_TRUE(loaded.ok()) << loaded.status().message();
  EXPECT_DOUBLE_EQ(loaded->capturePoint.gain, 0.5);
}

TEST(LocomotionHeuristicFormulas, LoaderNamesTheKeyTheFileAndTheTextOfAnUnparseableNumber) {
  // Each key is read on its own so that the error names it; a blanket catch around all forty-two could not.
  const std::string file =
      writeTemp("heuristics_bad_ratio.yaml", "locomotion_heuristics:\n  impulse_scaling:\n    scale: 0.5\n    maximumForceRatio: two\n");
  const absl::StatusOr<LocomotionHeuristicConfig> config = loadLocomotionHeuristicConfig(file);
  ASSERT_FALSE(config.ok());
  EXPECT_EQ(config.status().code(), absl::StatusCode::kInvalidArgument);
  const std::string expected =
      absl::StrCat("locomotion_heuristics.impulse_scaling.maximumForceRatio in '", file, "' is 'two', which is not a number.");
  EXPECT_TRUE(absl::StrContains(config.status().message(), expected)) << "expected '" << expected << "' in: " << config.status().message();
  EXPECT_FALSE(absl::StrContains(config.status().message(), "impulse_scaling.scale")) << "the error must name the bad key only";
}

TEST(LocomotionHeuristicFormulas, LoaderReportsAnUnreadableFileAsNotFound) {
  const std::string missing = testing::TempDir() + "/no_such_directory/no_such_task.yaml";
  const absl::StatusOr<LocomotionHeuristicConfig> config = loadLocomotionHeuristicConfig(missing);
  ASSERT_FALSE(config.ok());
  EXPECT_EQ(config.status().code(), absl::StatusCode::kNotFound) << config.status().message();
  EXPECT_TRUE(absl::StrContains(config.status().message(), missing)) << config.status().message();
}

}  // namespace ocs2::humanoid
