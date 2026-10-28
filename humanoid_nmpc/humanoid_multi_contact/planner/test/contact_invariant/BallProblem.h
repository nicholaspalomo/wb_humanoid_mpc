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

#pragma once

#include <memory>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/cost/StateCost.h"
#include "ocs2_core/dynamics/SystemDynamicsBase.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"

#include "humanoid_multi_contact/planner/contact_invariant/ContactFeatureFunction.h"
#include "humanoid_multi_contact/planner/contact_invariant/ContactInvariantLayout.h"
#include "humanoid_multi_contact/terrain/TerrainModel.h"

namespace ocs2::humanoid::multi_contact {

/**
 * The toy problem of the contact-invariant tests (design section 9): a ball of
 * mass kBallMass and radius kBallRadius above a floor at z = 0, its state the
 * position and velocity of its center, its input the force at its lowest
 * point, its one contact candidate that point. Gravity pulls it down; only a
 * contact with the floor can hold it up.
 */
inline constexpr double kBallMass = 1.0;              // [kg]
inline constexpr double kBallRadius = 0.1;            // [m]
inline constexpr double kBallFriction = 0.8;          // the floor's
inline constexpr double kBallConeSmoothing = 1.0e-3;  // eps of the cone
inline constexpr int kBallStateDim = 6;               // position, velocity
inline constexpr int kBallInputDim = 3;               // the force

/** The ball's layout: one point candidate, its force the whole input. */
ContactInvariantLayout ballLayout();

/** The floor: a box whose top face is z = 0. */
TerrainModel ballFloor();

/** The ball's contact features on `terrain` (ContactFeatureRows.h), taped as
 * `libraryName` in `folder`. */
absl::StatusOr<std::unique_ptr<ContactFeatureFunction>> ballFeatures(
    const TerrainModel& terrain,
    const std::string& libraryName,
    const std::string& folder);

/** x_dot = (v, f / m + g). */
class BallDynamics final : public SystemDynamicsBase {
 public:
  BallDynamics() = default;
  ~BallDynamics() override = default;
  BallDynamics* absl_nonnull clone() const override {
    return new BallDynamics(*this);
  }
  BallDynamics& operator=(const BallDynamics&) = delete;
  BallDynamics(BallDynamics&&) = delete;
  BallDynamics& operator=(BallDynamics&&) = delete;

  vector_t computeFlowMap(scalar_t time,
                          const vector_t& state,
                          const vector_t& input,
                          const PreComputation& preComp) override;
  VectorFunctionLinearApproximation linearApproximation(
      scalar_t time,
      const vector_t& state,
      const vector_t& input,
      const PreComputation& preComp) override;

 private:
  BallDynamics(const BallDynamics& other) = default;
};

/** weight |x - goal|^2 on the ball's state. */
class BallGoalCost final : public StateCost {
 public:
  BallGoalCost(vector_t goal, scalar_t weight);
  ~BallGoalCost() override = default;
  BallGoalCost* absl_nonnull clone() const override {
    return new BallGoalCost(*this);
  }
  BallGoalCost& operator=(const BallGoalCost&) = delete;
  BallGoalCost(BallGoalCost&&) = delete;
  BallGoalCost& operator=(BallGoalCost&&) = delete;

  scalar_t getValue(scalar_t time,
                    const vector_t& state,
                    const TargetTrajectories& targetTrajectories,
                    const PreComputation& preComp) const override;
  ScalarFunctionQuadraticApproximation getQuadraticApproximation(
      scalar_t time,
      const vector_t& state,
      const TargetTrajectories& targetTrajectories,
      const PreComputation& preComp) const override;

 private:
  BallGoalCost(const BallGoalCost& other) = default;

  vector_t goal_;
  scalar_t weight_ = 0.0;
};

/** The physical problem of the ball: its dynamics and a final cost of
 * `goalWeight` to `goal`. */
OptimalControlProblem ballPhysicalProblem(const vector_t& goal,
                                          scalar_t goalWeight);

}  // namespace ocs2::humanoid::multi_contact
