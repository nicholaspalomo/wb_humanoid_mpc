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

// The continuation and the thresholding of the contact-invariant planner
// (design sections 4.2 and 9) on a toy problem: a ball released at rest 0.4 m
// above a floor must be 0.5 m further on at the same height at the final
// time. Only a contact with the floor can stop its fall, and it is not told
// so: the first solve, with forces almost free, lets it float there (the
// paper's ghost motion); the second, with the force gate raised, makes it
// fall to the floor, push off while its contact variable claims the contact,
// and fly to the goal; the thresholding finds that one contact phase.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_core/PhysicalConstants.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"
#include "ocs2_oc/synchronized_module/ReferenceManager.h"
#include "ocs2_sqp/SqpSettings.h"

#include "humanoid_multi_contact/planner/contact_invariant/ContactFeatureRows.h"
#include "humanoid_multi_contact/planner/contact_invariant/ContactInvariantContinuation.h"
#include "humanoid_multi_contact/planner/contact_invariant/StanceThresholding.h"
#include "humanoid_nmpc/humanoid_multi_contact/planner/test/contact_invariant/BallProblem.h"

namespace ocs2::humanoid::multi_contact {
namespace {

constexpr double kHorizon = 1.6;  // [s]
constexpr double kDt = 0.02;      // [s]
constexpr double kStartHeight = 0.5;
constexpr double kGoalAdvance = 0.5;

vector_t ballStart() {
  vector_t x = vector_t::Zero(kBallStateDim);
  x(2) = kStartHeight;
  return x;
}

/** The goal: a point, passed through at the final time (a ball in the air
 * cannot be at rest there). */
vector_t ballGoal() {
  vector_t goal = ballStart().head<3>();
  goal(0) = kGoalAdvance;
  return goal;
}

/** The guess: hovering where it starts, held up by a force, c = 1/2. */
GuessTrajectory hoveringGuess() {
  GuessTrajectory guess;
  const int nodes = static_cast<int>(std::lround(kHorizon / kDt)) + 1;
  for (int n = 0; n < nodes; ++n) {
    guess.times.push_back(kDt * n);
    vector_t x(ballLayout().stateDim());
    x << ballStart(), 0.5;
    guess.states.push_back(x);
    vector_t u = vector_t::Zero(ballLayout().inputDim());
    u(2) = kBallMass * kGravityMagnitude;
    guess.inputs.push_back(u);
  }
  // It starts in the air: c = 0.
  guess.states.front()(ballLayout().activationIndex(0)) = 0.0;
  return guess;
}

sqp::Settings solverSettings() {
  sqp::Settings settings;
  settings.dt = kDt;
  settings.nThreads = 1;
  settings.enableLogging = false;
  settings.useFeedbackPolicy = false;
  settings.integratorType = SensitivityIntegratorType::RK4;
  return settings;
}

std::vector<ContinuationStage> stages() {
  const double weight = kBallMass * kGravityMagnitude;
  ContinuationStage ghost;
  ghost.name = "ghost";
  ghost.weights.contactInvariant = {
      .weight = 1.0e-2, .slipWeight = 0.1, .tiltWeight = 0.0};
  ghost.weights.forceGate = {
      .k0 = 1.0e-3, .k1 = 1.0e-2, .forceScale = weight, .momentArm = 1.0};
  ghost.weights.activationRateWeight = 1.0;
  ghost.sqpIterations = 30;
  ContinuationStage gate = ghost;
  gate.name = "gate";
  gate.weights.contactInvariant.weight = 1.0e2;
  gate.weights.forceGate.k0 = 0.1;
  gate.sqpIterations = 100;
  ContinuationStage contact = gate;
  contact.name = "contact";
  contact.weights.contactInvariant.weight = 1.0e4;
  contact.weights.forceGate.k0 = 10.0;
  return {ghost, gate, contact};
}

/** The ball's distance to the floor and its force at every node. */
struct BallTrace {
  std::vector<double> distances;
  std::vector<double> forces;  // over the ball's weight
  std::vector<double> activations;
};

BallTrace traceOf(const GuessTrajectory& trajectory) {
  BallTrace trace;
  const double weight = kBallMass * kGravityMagnitude;
  for (size_t n = 0; n < trajectory.times.size(); ++n) {
    trace.distances.push_back(trajectory.states[n](2) - kBallRadius);
    trace.forces.push_back(trajectory.inputs[n].head<3>().norm() / weight);
    trace.activations.push_back(
        trajectory.states[n](ballLayout().activationIndex(0)));
  }
  return trace;
}

TEST(ContactInvariantToy, TheBallDiscoversTheFloorToReachItsGoal) {
  const std::string folder =
      (std::filesystem::path(::testing::TempDir()) / "cppad").string();
  absl::StatusOr<std::unique_ptr<ContactFeatureFunction>> features =
      ballFeatures(ballFloor(), "ball_toy_features", folder);
  ASSERT_TRUE(features.ok()) << features.status();
  const OptimalControlProblem physical =
      ballPhysicalProblem(ballGoal(), /*goalWeight=*/1000.0);
  ContinuationProblem problem{
      .physical = &physical,
      .physicalReferenceManager = std::make_shared<ReferenceManager>(
          TargetTrajectories({0.0}, {ballStart()},
                             {vector_t::Zero(kBallInputDim)}),
          ModeSchedule()),
      .features = features->get(),
      .layout = ballLayout(),
      .augmentedManifold = nullptr,
      .barriers = {},
      .settings = solverSettings(),
      .initTime = 0.0,
      .finalTime = kHorizon,
      .initMode = 0};
  problem.barriers.activationBounds = {/*muParam=*/1.0e3, /*deltaParam=*/0.0};
  problem.barriers.frictionCone = {/*muParam=*/1.0e-2, /*deltaParam=*/1.0e-2};
  problem.barriers.padClearance = {/*muParam=*/0.1, /*deltaParam=*/1.0e-3};
  const absl::StatusOr<std::vector<StageSolution>> solutions =
      solveContinuation(problem, hoveringGuess(), stages());
  ASSERT_TRUE(solutions.ok()) << solutions.status();
  ASSERT_EQ(solutions->size(), 3);

  // The ghost: forces are nearly free, the ball floats.
  const BallTrace ghost = traceOf((*solutions)[0].trajectory);
  EXPECT_GT(*std::min_element(ghost.distances.begin(), ghost.distances.end()),
            0.1);

  // With the gate: the ball falls to the floor, claims the contact there and
  // pushes off; far from the floor it does not push. The contact stays soft:
  // c rises with its rate cost as the ball nears the floor, and the force
  // with it (solve 3 of the planner is the hard-contact problem).
  const GuessTrajectory& final = solutions->back().trajectory;
  const BallTrace trace = traceOf(final);
  const size_t lowest =
      std::min_element(trace.distances.begin(), trace.distances.end()) -
      trace.distances.begin();
  EXPECT_LT(trace.distances[lowest], 0.01);
  EXPECT_GT(trace.distances[lowest], -0.03) << "it sinks into the floor";
  EXPECT_GT(trace.activations[lowest], 0.5);
  double impulse = 0.0;
  double impulseInTheAir = 0.0;
  for (size_t n = 0; n + 1 < trace.distances.size(); ++n) {
    impulse += trace.forces[n];
    if (trace.distances[n] > 0.25) impulseInTheAir += trace.forces[n];
  }
  EXPECT_LT(impulseInTheAir, 0.1 * impulse);
  const vector_t& end = final.states.back();
  EXPECT_LT((end.head<3>() - ballGoal().head<3>()).norm(), 0.05)
      << end.transpose();

  // One contact phase, at the floor, around the lowest node.
  std::vector<std::vector<double>> activations;
  for (const double c : trace.activations) activations.push_back({c});
  const absl::StatusOr<StanceSchedule> schedule =
      thresholdActivations(final.times, activations, {false}, {false},
                           // A bounce is shorter than a robot's dwell.
                           ThresholdingOptions{.minimumDwell = 0.1});
  ASSERT_TRUE(schedule.ok()) << schedule.status();
  int phases = 0;
  for (size_t n = 0; n < schedule->active.size(); ++n) {
    const bool active = schedule->active[n][0];
    if (active && (n == 0 || !schedule->active[n - 1][0])) ++phases;
    if (active) {
      EXPECT_LT(trace.distances[n], 0.25) << "t = " << final.times[n];
    }
  }
  EXPECT_EQ(phases, 1);
  EXPECT_TRUE(schedule->active[lowest][0]);
}

}  // namespace
}  // namespace ocs2::humanoid::multi_contact
