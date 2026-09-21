/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/hlip/HlipStandingBlend.h"

// The standing / walking blend is a five-line scalar law, and until now nothing tested it at all. It is nevertheless
// the switch that decides whether the default H-LIP planner emits a stepping gait or a standing one, and it scales the
// commanded velocity every foothold is then planned from: HlipContactPlanner::isWalking forwards straight to it, and
// HlipContactPlanner::plan takes the weight, compares it with a half to pick the gait and multiplies the commanded
// velocity by it. A silent change here is therefore the difference between a robot that walks when the stick moves and
// one that either marches in place at rest or refuses to start. The assertions below are deliberately written against the
// CLAIMS IN THE DOCUMENTATION of HlipStandingBlend.h and of HlipBlendParameters in ContactPlanningConfig.h - each test
// quotes the sentence it pins - so that a future edit which changes the law has to change the prose with it.

namespace ocs2::humanoid {
namespace {

/** The five terms of the activity metric P, in the order HlipStandingBlend::activity sums them. */
enum class BlendComponent {
  kCommandedVelocityX,
  kCommandedVelocityY,
  kCommandedYawRate,
  kComVelocityX,
  kComVelocityY,
};

constexpr std::array<BlendComponent, 5> kAllComponents = {BlendComponent::kCommandedVelocityX, BlendComponent::kCommandedVelocityY,
                                                          BlendComponent::kCommandedYawRate, BlendComponent::kComVelocityX,
                                                          BlendComponent::kComVelocityY};

const char* componentName(BlendComponent component) {
  switch (component) {
    case BlendComponent::kCommandedVelocityX:
      return "commanded velocity x";
    case BlendComponent::kCommandedVelocityY:
      return "commanded velocity y";
    case BlendComponent::kCommandedYawRate:
      return "commanded yaw rate";
    case BlendComponent::kComVelocityX:
      return "measured centre of mass velocity x";
    case BlendComponent::kComVelocityY:
      return "measured centre of mass velocity y";
  }
  return "unknown component";
}

/**
 * Blend parameters chosen so that every assertion in this file is EXACT in binary floating point, and so that no two
 * components share a maximum.
 *
 * The distinct maxima are the point: if `activity` normalised, say, the measured centre-of-mass velocity by the
 * COMMANDED velocity's maximum - the kind of copy-and-paste slip a five-term sum invites, and one that would still
 * produce a plausible-looking blend - every component would still be "normalised by something" and a test written with
 * one shared maximum could not see it. Here each of the five ratios below is a different number, so mixing two of them
 * up changes the activity and the test fails.
 *
 * The values are all exact binary fractions (1, 1/2, 2, 1/4, 4) and the test ratios are too, so value / maximum is
 * exact, the square is exact, and the sum is exact. That lets the tests use EXPECT_DOUBLE_EQ on the activity rather
 * than a tolerance that would hide a small systematic error.
 */
HlipBlendParameters makeParameters() {
  HlipBlendParameters parameters;
  parameters.sharpness = 4.0;
  parameters.threshold = 0.25;
  parameters.maxCommandedVelocityX = 1.0;
  parameters.maxCommandedVelocityY = 0.5;
  parameters.maxCommandedYawRate = 2.0;
  parameters.maxComVelocityX = 0.25;
  parameters.maxComVelocityY = 4.0;
  return parameters;
}

scalar_t componentMaximum(const HlipBlendParameters& parameters, BlendComponent component) {
  switch (component) {
    case BlendComponent::kCommandedVelocityX:
      return parameters.maxCommandedVelocityX;
    case BlendComponent::kCommandedVelocityY:
      return parameters.maxCommandedVelocityY;
    case BlendComponent::kCommandedYawRate:
      return parameters.maxCommandedYawRate;
    case BlendComponent::kComVelocityX:
      return parameters.maxComVelocityX;
    case BlendComponent::kComVelocityY:
      return parameters.maxComVelocityY;
  }
  return 0.0;
}

/** The three arguments of the blend, so that one component can be driven while the other four stay at exactly zero. */
struct BlendInput {
  vector2_t velocityCommand = vector2_t::Zero();
  scalar_t yawRateCommand = 0.0;
  vector2_t comVelocity = vector2_t::Zero();
};

BlendInput makeInput(BlendComponent component, scalar_t value) {
  BlendInput input;
  switch (component) {
    case BlendComponent::kCommandedVelocityX:
      input.velocityCommand.x() = value;
      break;
    case BlendComponent::kCommandedVelocityY:
      input.velocityCommand.y() = value;
      break;
    case BlendComponent::kCommandedYawRate:
      input.yawRateCommand = value;
      break;
    case BlendComponent::kComVelocityX:
      input.comVelocity.x() = value;
      break;
    case BlendComponent::kComVelocityY:
      input.comVelocity.y() = value;
      break;
  }
  return input;
}

scalar_t activityOf(const HlipStandingBlend& blend, const BlendInput& input) {
  return blend.activity(input.velocityCommand, input.yawRateCommand, input.comVelocity);
}

scalar_t weightOf(const HlipStandingBlend& blend, const BlendInput& input) {
  return blend.weight(input.velocityCommand, input.yawRateCommand, input.comVelocity);
}

bool isWalkingOf(const HlipStandingBlend& blend, const BlendInput& input) {
  return blend.isWalking(input.velocityCommand, input.yawRateCommand, input.comVelocity);
}

/**
 * A contact planning configuration that validates, so that the degenerate-blend tests below change exactly one key and
 * can attribute the rejection to it. The cadence is the one testHlipContactPlanner.cpp uses for the H-LIP planner.
 */
ContactPlanningConfig makeValidConfig() {
  ContactPlanningConfig config;
  config.planner.type = "hlip";
  config.planner.dt = 0.025;
  config.planner.numNodes = 56;
  config.planner.commitTime = 0.05;
  config.shared.comHeight = 0.85;
  config.shared.gaitLimits.minSwingDuration = 0.25;
  config.shared.gaitLimits.maxSwingDuration = 0.35;
  config.shared.gaitLimits.minDoubleSupportDuration = 0.0;
  config.hlip.sspDuration = 0.25;
  config.hlip.dspDuration = 0.05;
  config.hlip.stepWidth = 0.25;
  return config;
}

ContactPlanningConfig makeConfigWithMaximum(BlendComponent component, scalar_t value) {
  ContactPlanningConfig config = makeValidConfig();
  switch (component) {
    case BlendComponent::kCommandedVelocityX:
      config.hlip.blend.maxCommandedVelocityX = value;
      break;
    case BlendComponent::kCommandedVelocityY:
      config.hlip.blend.maxCommandedVelocityY = value;
      break;
    case BlendComponent::kCommandedYawRate:
      config.hlip.blend.maxCommandedYawRate = value;
      break;
    case BlendComponent::kComVelocityX:
      config.hlip.blend.maxComVelocityX = value;
      break;
    case BlendComponent::kComVelocityY:
      config.hlip.blend.maxComVelocityY = value;
      break;
  }
  return config;
}

TEST(HlipStandingBlend, phiIsTheSquaredRatioOfWhicheverComponentIsDrivenAlone) {
  // HlipStandingBlend.h: "P normalises each command and each measured velocity by the largest value it is expected to
  // take, so phi reaches one when a single component is at its threshold." Drive each of the five components alone
  // from zero past its own maximum and the activity must be exactly (value / maximum)^2 - which is one at the maximum,
  // a quarter at half of it, and four at twice it. The four components left at zero must contribute exactly nothing,
  // which is what makes this a test of INDEPENDENT normalisation and not merely of the total.
  const HlipStandingBlend blend(makeParameters());
  const std::array<scalar_t, 6> ratios = {0.0, 0.25, 0.5, 1.0, 1.5, 2.0};
  for (const BlendComponent component : kAllComponents) {
    const scalar_t maximum = componentMaximum(blend.getParameters(), component);
    for (const scalar_t ratio : ratios) {
      const BlendInput input = makeInput(component, ratio * maximum);
      EXPECT_DOUBLE_EQ(activityOf(blend, input), ratio * ratio)
          << componentName(component) << " driven alone at " << ratio << " of its maximum (" << maximum << ")";
    }
    // Spelled out on its own, because it is the sentence the header actually makes: at its maximum, one component
    // alone carries the whole of phi = 1.
    EXPECT_DOUBLE_EQ(activityOf(blend, makeInput(component, maximum)), 1.0) << componentName(component) << " at its maximum";
  }
}

TEST(HlipStandingBlend, phiIsTheSumOfTheIndependentlyNormalisedSquaredRatios) {
  // The activity is a squared norm under a diagonal metric, so driving all five components at once must give the sum
  // of the five single-component activities: no cross term, and no component that quietly saturates or clips.
  const HlipStandingBlend blend(makeParameters());
  const HlipBlendParameters& parameters = blend.getParameters();
  const std::array<scalar_t, 5> ratios = {0.25, 0.5, 0.75, 1.0, 1.5};  // one per component, all different

  const vector2_t velocityCommand(ratios[0] * parameters.maxCommandedVelocityX, ratios[1] * parameters.maxCommandedVelocityY);
  const scalar_t yawRateCommand = ratios[2] * parameters.maxCommandedYawRate;
  const vector2_t comVelocity(ratios[3] * parameters.maxComVelocityX, ratios[4] * parameters.maxComVelocityY);
  const scalar_t combined = blend.activity(velocityCommand, yawRateCommand, comVelocity);

  scalar_t expected = 0.0;
  scalar_t sumOfParts = 0.0;
  for (size_t index = 0; index < kAllComponents.size(); ++index) {
    expected += ratios[index] * ratios[index];
    sumOfParts += activityOf(blend, makeInput(kAllComponents[index], ratios[index] * componentMaximum(parameters, kAllComponents[index])));
  }
  EXPECT_DOUBLE_EQ(combined, expected);
  EXPECT_DOUBLE_EQ(combined, sumOfParts);
}

TEST(HlipStandingBlend, phiAndAlphaAreEvenInTheSignOfEveryComponent) {
  // phi is a sum of SQUARED ratios, so walking backwards, sidestepping to the right or turning clockwise has to blend
  // exactly like the mirror-image command. There is no asymmetry to tune and none to accidentally introduce: a law
  // that answered differently to -0.3 m/s than to +0.3 m/s would start stepping later in one direction than the other.
  const HlipStandingBlend blend(makeParameters());
  for (const BlendComponent component : kAllComponents) {
    const scalar_t maximum = componentMaximum(blend.getParameters(), component);
    const std::array<scalar_t, 3> magnitudes = {0.25 * maximum, 0.75 * maximum, 1.5 * maximum};
    for (const scalar_t magnitude : magnitudes) {
      const BlendInput forward = makeInput(component, magnitude);
      const BlendInput backward = makeInput(component, -magnitude);
      EXPECT_DOUBLE_EQ(activityOf(blend, backward), activityOf(blend, forward)) << componentName(component) << " at " << magnitude;
      EXPECT_DOUBLE_EQ(weightOf(blend, backward), weightOf(blend, forward)) << componentName(component) << " at " << magnitude;
      EXPECT_EQ(isWalkingOf(blend, backward), isWalkingOf(blend, forward)) << componentName(component) << " at " << magnitude;
    }
  }
}

TEST(HlipStandingBlend, alphaIsTheDocumentedSigmoidInSharpnessAndThreshold) {
  // HlipStandingBlend.h: "alpha(phi) = tanh(rho_1 (phi - rho_2)) / 2 + 1/2", with rho_1 the sharpness and rho_2 the
  // threshold. Every phi below is an exact square, and the forward command's maximum is one, so the command that
  // realises a given phi is exactly its square root and the closed form can be compared to the last bit.
  const std::array<scalar_t, 6> phiValues = {0.0, 0.0625, 0.25, 0.5625, 1.0, 4.0};
  const std::array<scalar_t, 3> sharpnessValues = {0.5, 4.0, 40.0};
  for (const scalar_t sharpness : sharpnessValues) {
    HlipBlendParameters parameters = makeParameters();
    parameters.sharpness = sharpness;
    const HlipStandingBlend blend(parameters);
    for (const scalar_t phi : phiValues) {
      const BlendInput input = makeInput(BlendComponent::kCommandedVelocityX, std::sqrt(phi));
      ASSERT_DOUBLE_EQ(activityOf(blend, input), phi) << "the test itself must realise phi exactly";
      const scalar_t expected = 0.5 * std::tanh(sharpness * (phi - parameters.threshold)) + 0.5;
      EXPECT_NEAR(weightOf(blend, input), expected, 1e-15) << "phi = " << phi << ", sharpness = " << sharpness;
    }
  }
}

TEST(HlipStandingBlend, sharpnessSteepensTheTransitionWithoutMovingTheHalfPoint) {
  // rho_1 is what the ContactPlanningConfig.h comment tunes when it says the defaults "saturate by a fifth" of the
  // maximum command: it does not move where the planner starts walking, only how quickly alpha runs from standing to
  // the full gait once it has. Above the threshold a sharper blend must be closer to one, below it closer to zero, and
  // at the threshold every sharpness must agree on exactly a half.
  HlipBlendParameters gentle = makeParameters();
  gentle.sharpness = 2.0;
  HlipBlendParameters sharp = makeParameters();
  sharp.sharpness = 20.0;
  const HlipStandingBlend gentleBlend(gentle);
  const HlipStandingBlend sharpBlend(sharp);

  // phi = 0.5625 is above the threshold of 0.25, phi = 0.0625 below it, and phi = 0.25 sits exactly on it.
  const BlendInput above = makeInput(BlendComponent::kCommandedVelocityX, 0.75);
  const BlendInput below = makeInput(BlendComponent::kCommandedVelocityX, 0.25);
  const BlendInput at = makeInput(BlendComponent::kCommandedVelocityX, 0.5);
  EXPECT_GT(weightOf(sharpBlend, above), weightOf(gentleBlend, above));
  EXPECT_LT(weightOf(sharpBlend, below), weightOf(gentleBlend, below));
  EXPECT_DOUBLE_EQ(weightOf(sharpBlend, at), 0.5);
  EXPECT_DOUBLE_EQ(weightOf(gentleBlend, at), 0.5);
}

TEST(HlipStandingBlend, alphaIsMonotoneInPhiAndNeverLeavesTheUnitInterval) {
  // "alpha in (0, 1): 0 stands still, 1 walks the full H-LIP gait." A weight outside that interval would be handed
  // straight to HlipContactPlanner, which uses it as the convex weight of the walking reference against the standing
  // one (plan.comPosition = alpha * rolled-out + (1 - alpha) * support centre), so a value above one or below zero
  // would extrapolate the reference rather than interpolate it. Monotonicity is the other half of the contract: more
  // command, or more measured motion, may never make the planner LESS willing to walk.
  const HlipStandingBlend blend(makeParameters());
  const scalar_t maximum = blend.getParameters().maxCommandedVelocityX;
  scalar_t previousWeight = -1.0;
  for (int step = 0; step <= 100; ++step) {
    const scalar_t command = 0.05 * static_cast<scalar_t>(step) * maximum;  // zero up to five times the maximum command
    const scalar_t weight = weightOf(blend, makeInput(BlendComponent::kCommandedVelocityX, command));
    EXPECT_GE(weight, 0.0) << "command " << command;
    EXPECT_LE(weight, 1.0) << "command " << command;
    EXPECT_GE(weight, previousWeight) << "alpha fell as the activity grew, at command " << command;
    previousWeight = weight;
  }

  // Strictly increasing, checked only where tanh has not yet saturated in double precision. Past roughly
  // sharpness * (phi - threshold) = 19 the library's tanh returns exactly 1.0 and neighbouring commands necessarily
  // tie, which is saturation rather than a violation of the law, so the strict sweep stops well short of it.
  previousWeight = weightOf(blend, makeInput(BlendComponent::kCommandedVelocityX, 0.0));
  for (int step = 1; step <= 30; ++step) {
    const scalar_t command = 0.05 * static_cast<scalar_t>(step) * maximum;
    const scalar_t weight = weightOf(blend, makeInput(BlendComponent::kCommandedVelocityX, command));
    EXPECT_GT(weight, previousWeight) << "alpha did not strictly increase at command " << command;
    previousWeight = weight;
  }
}

TEST(HlipStandingBlend, alphaLandsOnBothOfItsAsymptotes) {
  // The two ends of the sigmoid are the two behaviours the blend exists to produce: a robot that is asked for nothing
  // and is measured to be doing nothing must be given the STANDING reference with no walking in it at all, and a robot
  // at full stick must be given the whole H-LIP gait with no standing in it. A law that only ever reached, say, 0.9
  // would permanently shorten every step by a tenth.
  //
  // Note on the header's "alpha in (0, 1)": the interval is open in exact arithmetic, because tanh never actually
  // reaches +-1. In double precision it does - tanh(-20) rounds to exactly -1 - so the assertions below pin the
  // asymptotes with a tolerance and pin the bounds inclusively. That is the honest statement of the shipped law.
  HlipBlendParameters parameters = makeParameters();
  parameters.sharpness = 40.0;
  parameters.threshold = 0.5;  // the half point sits far above rest, so rest is out on the standing asymptote
  const HlipStandingBlend blend(parameters);

  const BlendInput rest{};  // every command and every measured velocity exactly zero
  EXPECT_NEAR(weightOf(blend, rest), 0.0, 1e-12) << "a robot at rest with no command must be given the standing reference";
  EXPECT_GE(weightOf(blend, rest), 0.0);
  EXPECT_FALSE(isWalkingOf(blend, rest));

  const BlendInput fullStick = makeInput(BlendComponent::kCommandedVelocityX, 2.0 * parameters.maxCommandedVelocityX);
  EXPECT_NEAR(weightOf(blend, fullStick), 1.0, 1e-12) << "a saturated command must be given the full walking gait";
  EXPECT_LE(weightOf(blend, fullStick), 1.0);
  EXPECT_TRUE(isWalkingOf(blend, fullStick));
}

TEST(HlipStandingBlend, theHalfPointSitsExactlyAtTheThresholdAndIsInclusive) {
  // ContactPlanningConfig.h: "The planner stands while alpha is below a half and steps above it." alpha is a half
  // exactly when phi equals rho_2, so the stepping decision is phi >= threshold and nothing else - there is no dead
  // band and no hysteresis, which is the whole point of replacing a stepping trigger with this one scalar law.
  // `isWalking` compares with >=, so the boundary itself counts as walking; that is pinned here so that a later edit
  // to the comparison has to be deliberate.
  const std::array<scalar_t, 2> thresholds = {0.25, 1.0};  // exact squares, so the crossing command is exact too
  for (const scalar_t threshold : thresholds) {
    HlipBlendParameters parameters = makeParameters();
    parameters.threshold = threshold;
    const HlipStandingBlend blend(parameters);

    const scalar_t crossing = std::sqrt(threshold) * parameters.maxCommandedVelocityX;
    const BlendInput atCrossing = makeInput(BlendComponent::kCommandedVelocityX, crossing);
    ASSERT_DOUBLE_EQ(activityOf(blend, atCrossing), threshold);
    EXPECT_DOUBLE_EQ(weightOf(blend, atCrossing), 0.5) << "phi = rho_2 must give exactly the half point";
    EXPECT_TRUE(isWalkingOf(blend, atCrossing)) << "the comparison at the half point is inclusive";

    // A per-cent either side of the crossing, which is far enough out that no rounding of the ratio can flip it.
    EXPECT_FALSE(isWalkingOf(blend, makeInput(BlendComponent::kCommandedVelocityX, 0.99 * crossing)));
    EXPECT_TRUE(isWalkingOf(blend, makeInput(BlendComponent::kCommandedVelocityX, 1.01 * crossing)));
    EXPECT_LT(weightOf(blend, makeInput(BlendComponent::kCommandedVelocityX, 0.99 * crossing)), 0.5);
    EXPECT_GT(weightOf(blend, makeInput(BlendComponent::kCommandedVelocityX, 1.01 * crossing)), 0.5);
  }
}

TEST(HlipStandingBlend, aZeroThresholdLeavesTheRobotNoWayToStand) {
  // A consequence of the two facts above that is worth stating once, because it is the one setting of
  // `hlip.blend.threshold` that breaks the law rather than tuning it: phi is a sum of squares, so it is never
  // negative, and the half point is inclusive. With rho_2 = 0 the robot is therefore declared to be walking at rest,
  // with no command and no measured motion at all, and the planner marches in place forever.
  //
  // ContactPlanningConfig::validate() now rejects a non-positive threshold, so no loaded configuration can reach this
  // state; the check used to cover the sharpness and the five maxima but not its neighbour rho_2. This test builds the
  // blend DIRECTLY, bypassing validate(), so it goes on recording what the value does to the law itself - which is
  // what makes the rejection in validate() worth having rather than arbitrary.
  HlipBlendParameters parameters = makeParameters();
  parameters.threshold = 0.0;
  const HlipStandingBlend blend(parameters);

  const BlendInput rest{};
  EXPECT_DOUBLE_EQ(activityOf(blend, rest), 0.0);
  EXPECT_DOUBLE_EQ(weightOf(blend, rest), 0.5);
  EXPECT_TRUE(isWalkingOf(blend, rest)) << "a zero threshold makes standing unreachable; keep hlip.blend.threshold positive";
}

TEST(HlipStandingBlend, everyComponentAloneCanCrossTheHalfPoint) {
  // Each of the five terms must be able to start the gait ON ITS OWN. This is not a formality: the yaw rate was
  // omitted from the blend's input once, and ContactPlanningReferenceManager.cpp records what that cost - "`alpha`
  // never crossed its half point on yaw alone and the robot would not start" turning on the spot. The measured
  // centre-of-mass velocity matters for the same reason in the other direction: a robot that has been pushed is
  // moving without any command, and the blend has to let it step to catch itself.
  //
  // With the threshold at 0.25 the crossing of each component sits at exactly half its maximum, and half of each of
  // the maxima above is again an exact binary fraction, so the activity at the crossing is exactly the threshold.
  const HlipStandingBlend blend(makeParameters());
  const HlipBlendParameters& parameters = blend.getParameters();
  for (const BlendComponent component : kAllComponents) {
    const scalar_t crossing = 0.5 * componentMaximum(parameters, component);
    EXPECT_DOUBLE_EQ(activityOf(blend, makeInput(component, crossing)), parameters.threshold) << componentName(component);
    EXPECT_TRUE(isWalkingOf(blend, makeInput(component, crossing))) << componentName(component) << " alone must be able to start the gait";
    EXPECT_TRUE(isWalkingOf(blend, makeInput(component, -crossing))) << componentName(component) << " must behave the same in reverse";
    EXPECT_TRUE(isWalkingOf(blend, makeInput(component, 2.0 * crossing))) << componentName(component);
    EXPECT_FALSE(isWalkingOf(blend, makeInput(component, 0.5 * crossing))) << componentName(component) << " must not start the gait alone";

    // And the other four components stay out of it: with this component just under its crossing the blend stands, and
    // it is this component alone that takes it over.
    const BlendInput justUnder = makeInput(component, 0.99 * crossing);
    EXPECT_FALSE(isWalkingOf(blend, justUnder)) << componentName(component);
  }
}

TEST(HlipStandingBlend, theShippedDefaultsPutTheHalfPointAtATenthOfAMetrePerSecond) {
  // ContactPlanningConfig.h, on the defaults: "The defaults below put the half point at about a tenth of the maximum
  // command (0.1 m/s forward) and saturate by a fifth of it, which is what 'walk when asked to walk, stand when asked
  // to stand' means for a humanoid." That worked example is the only statement in the repository about where the
  // shipped robots actually start walking, so it is pinned here; a retune of `hlip.blend.sharpness` or
  // `hlip.blend.threshold` has to update the sentence and this test together.
  const HlipBlendParameters defaults{};
  const HlipStandingBlend blend(defaults);

  const vector2_t atRest = vector2_t::Zero();
  EXPECT_FALSE(blend.isWalking(atRest, 0.0, atRest)) << "no command and no motion must stand";
  EXPECT_FALSE(blend.isWalking(vector2_t(0.09, 0.0), 0.0, atRest)) << "0.09 m/s is below the documented half point";
  EXPECT_TRUE(blend.isWalking(vector2_t(0.10, 0.0), 0.0, atRest)) << "0.10 m/s is above the documented half point";

  // The half point itself, phi = rho_2, is at maxCommandedVelocityX * sqrt(threshold) = 0.7 * sqrt(0.02) m/s.
  const scalar_t halfPoint = defaults.maxCommandedVelocityX * std::sqrt(defaults.threshold);
  EXPECT_NEAR(halfPoint, 0.099, 1e-3);
  EXPECT_NEAR(blend.weight(vector2_t(halfPoint, 0.0), 0.0, atRest), 0.5, 1e-12);

  // "saturate by a fifth of it": at a fifth of the maximum forward command the blend is already most of the way to the
  // full gait, and at the maximum command it is on the walking asymptote.
  EXPECT_GT(blend.weight(vector2_t(0.2 * defaults.maxCommandedVelocityX, 0.0), 0.0, atRest), 0.8);
  EXPECT_NEAR(blend.weight(vector2_t(defaults.maxCommandedVelocityX, 0.0), 0.0, atRest), 1.0, 1e-12);
}

TEST(HlipStandingBlend, theConfigurationRejectsANonPositiveMaximumSoTheBlendNeverDividesByZero) {
  // HlipStandingBlend::activity divides each component by its maximum with no guard of its own: a zero maximum would
  // make phi infinite for any non-zero value of that component and NOT-A-NUMBER for a zero one, and a NaN phi makes
  // alpha NaN, `isWalking` false whatever the operator does with the stick, and every blended reference NaN from
  // there on. A negative maximum is just as wrong and is silently swallowed by the squaring, so it would never show
  // up as a sign error - it would simply normalise the command by the wrong number.
  //
  // Strictly positive maxima are therefore a PRECONDITION of this class, and ContactPlanningConfig::validate() is
  // where it is enforced, before any planner is built from the configuration. This test pins that enforcement for each
  // of the five keys individually, so that adding a sixth term to phi without extending the check fails here.
  EXPECT_NO_THROW(makeValidConfig().validate()) << "the baseline configuration must be valid, or the checks below prove nothing";

  for (const BlendComponent component : kAllComponents) {
    EXPECT_THROW(makeConfigWithMaximum(component, 0.0).validate(), std::invalid_argument)
        << "a zero maximum for " << componentName(component) << " would divide by zero in the blend";
    EXPECT_THROW(makeConfigWithMaximum(component, -1.0).validate(), std::invalid_argument)
        << "a negative maximum for " << componentName(component) << " normalises by the wrong number";
  }

  // The message has to name the block the operator must edit, not merely report that something is invalid.
  try {
    makeConfigWithMaximum(BlendComponent::kComVelocityY, 0.0).validate();
    ADD_FAILURE() << "a zero hlip.blend.maxComVelocityY must be rejected";
  } catch (const std::invalid_argument& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("hlip.blend"), std::string::npos) << "the rejection must name the configuration block: " << message;
  }

  // The sharpness is the other divisor-like constant of the law: at zero, alpha would be exactly a half everywhere,
  // `isWalking` would be true at rest and the robot would march in place; negative, the whole law would run backwards
  // and a released stick would ask for the full gait.
  ContactPlanningConfig zeroSharpness = makeValidConfig();
  zeroSharpness.hlip.blend.sharpness = 0.0;
  EXPECT_THROW(zeroSharpness.validate(), std::invalid_argument);
  ContactPlanningConfig negativeSharpness = makeValidConfig();
  negativeSharpness.hlip.blend.sharpness = -40.0;
  EXPECT_THROW(negativeSharpness.validate(), std::invalid_argument);
}

}  // namespace
}  // namespace ocs2::humanoid
