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

#include <cmath>
#include <utility>

#include "humanoid_common_mpc/contact_planning/hlip/HlipModel.h"

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kTol = 1e-9;

/** The cadence and height of the paper's experiments (arXiv:2502.15630, Table I). */
HlipModel makePaperModel() {
  return HlipModel(0.35, 0.0, 0.62, 9.81);
}

/** A cadence with a non-zero double support, which changes every matrix in the step-to-step map. */
HlipModel makeDoubleSupportModel() {
  return HlipModel(0.3, 0.1, 0.85, 9.81);
}

TEST(HlipModel, naturalFrequencyAndStepDuration) {
  const HlipModel model = makeDoubleSupportModel();
  EXPECT_NEAR(model.naturalFrequency(), std::sqrt(9.81 / 0.85), kTol);
  EXPECT_NEAR(model.stepDuration(), 0.4, kTol);
}

TEST(HlipModel, stepToStepMapComposesTheFlows) {
  const HlipModel model = makeDoubleSupportModel();
  const HlipModel::State state(0.05, 0.3);
  const scalar_t stepLength = 0.2;

  // The map is the impact, then the double support drift, then the single support flow.
  const HlipModel::State afterImpact = HlipModel::applyStepTransition(state, stepLength);
  const HlipModel::State afterDrift = HlipModel::flowDoubleSupport(afterImpact, model.dspDuration());
  const HlipModel::State expected = model.flowSingleSupport(afterDrift, model.sspDuration());

  const HlipModel::State actual = model.applyStepToStep(state, stepLength);
  EXPECT_NEAR(actual(0), expected(0), kTol);
  EXPECT_NEAR(actual(1), expected(1), kTol);
}

TEST(HlipModel, deadbeatGainIsNilpotent) {
  // The whole point of the H-LIP step controller: A + B K has both eigenvalues at zero, so the error is gone after two
  // steps without a single tuned gain.
  for (const HlipModel& model : {makePaperModel(), makeDoubleSupportModel()}) {
    const HlipModel::Matrix2 closedLoop = model.stepToStepA() + model.stepToStepB() * model.deadbeatGain().transpose();
    EXPECT_NEAR(closedLoop.trace(), 0.0, 1e-9);
    EXPECT_NEAR(closedLoop.determinant(), 0.0, 1e-9);
    EXPECT_NEAR((closedLoop * closedLoop).norm(), 0.0, 1e-9);
  }
}

TEST(HlipModel, periodOneOrbitIsAFixedPoint) {
  const HlipModel model = makePaperModel();
  for (const scalar_t stepLength : {0.0, 0.15, -0.2}) {
    const HlipModel::State orbit = model.periodOneOrbit(stepLength);
    const HlipModel::State next = model.applyStepToStep(orbit, stepLength);
    EXPECT_NEAR(next(0), orbit(0), 1e-9);
    EXPECT_NEAR(next(1), orbit(1), 1e-9);
  }
}

TEST(HlipModel, periodOneOrbitAdvancesByTheStepEveryStep) {
  // The orbit of a commanded velocity covers exactly one step length per step: integrating the velocity the continuous
  // flow produces over one whole step must give back the step length the orbit was built for.
  const HlipModel model = makePaperModel();
  const scalar_t velocity = 0.4;
  const scalar_t stepLength = velocity * model.stepDuration();
  const HlipModel::State orbit = model.periodOneOrbit(stepLength);
  const HlipModel::State postImpact = HlipModel::applyStepTransition(orbit, stepLength);

  constexpr int kSamples = 20000;
  const scalar_t sampleTime = model.sspDuration() / static_cast<scalar_t>(kSamples);
  scalar_t displacement = model.dspDuration() * postImpact(1);
  const HlipModel::State afterDrift = HlipModel::flowDoubleSupport(postImpact, model.dspDuration());
  for (int sample = 0; sample < kSamples; ++sample) {
    const scalar_t start = sampleTime * static_cast<scalar_t>(sample);
    const scalar_t end = start + sampleTime;
    displacement += 0.5 * sampleTime * (model.flowSingleSupport(afterDrift, start)(1) + model.flowSingleSupport(afterDrift, end)(1));
  }
  EXPECT_NEAR(displacement, stepLength, 1e-6);
  EXPECT_NEAR(displacement / model.stepDuration(), velocity, 1e-6);
}

TEST(HlipModel, periodTwoOrbitAlternates) {
  const HlipModel model = makePaperModel();
  const scalar_t leftStep = 0.25;
  const scalar_t rightStep = -0.25;
  const std::pair<HlipModel::State, HlipModel::State> orbit = model.periodTwoOrbit(leftStep, rightStep);

  const HlipModel::State afterLeft = model.applyStepToStep(orbit.first, leftStep);
  EXPECT_NEAR(afterLeft(0), orbit.second(0), 1e-9);
  EXPECT_NEAR(afterLeft(1), orbit.second(1), 1e-9);
  const HlipModel::State afterRight = model.applyStepToStep(orbit.second, rightStep);
  EXPECT_NEAR(afterRight(0), orbit.first(0), 1e-9);
  EXPECT_NEAR(afterRight(1), orbit.first(1), 1e-9);

  // A symmetric lateral orbit stands still on average: the pre-impact positions mirror each other.
  EXPECT_NEAR(orbit.first(0), -orbit.second(0), 1e-9);
}

TEST(HlipModel, deadbeatControllerRecoversInTwoSteps) {
  const HlipModel model = makeDoubleSupportModel();
  const scalar_t stepLength = 0.3 * model.stepDuration();
  const HlipModel::State orbit = model.periodOneOrbit(stepLength);

  HlipModel::State state(orbit(0) + 0.08, orbit(1) - 0.25);  // a shove, well away from the orbit
  for (int step = 0; step < 2; ++step) {
    const scalar_t command = model.deadbeatStepLength(state, orbit, stepLength);
    state = model.applyStepToStep(state, command);
  }
  EXPECT_NEAR(state(0), orbit(0), 1e-9);
  EXPECT_NEAR(state(1), orbit(1), 1e-9);
}

TEST(HlipModel, deadbeatControllerLeavesTheOrbitAlone) {
  const HlipModel model = makePaperModel();
  const scalar_t stepLength = 0.18;
  const HlipModel::State orbit = model.periodOneOrbit(stepLength);
  EXPECT_NEAR(model.deadbeatStepLength(orbit, orbit, stepLength), stepLength, kTol);
}

}  // namespace
}  // namespace ocs2::humanoid
