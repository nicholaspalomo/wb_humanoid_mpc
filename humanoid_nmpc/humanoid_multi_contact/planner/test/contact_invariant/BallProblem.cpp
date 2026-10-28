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

#include "humanoid_nmpc/humanoid_multi_contact/planner/test/contact_invariant/BallProblem.h"

#include <memory>
#include <string>
#include <utility>

#include "Eigen/Core"
#include "absl/status/statusor.h"
#include "ocs2_core/PhysicalConstants.h"
#include "ocs2_core/automatic_differentiation/Types.h"

#include "humanoid_multi_contact/planner/contact_invariant/ContactFeatureRows.h"
#include "humanoid_multi_contact/terrain/terrain_scene.nproto.h"

namespace ocs2::humanoid::multi_contact {
namespace {

using AdVector3 = Eigen::Matrix<ad_scalar_t, kPositionDim, 1>;

}  // namespace

ContactInvariantLayout ballLayout() {
  ContactInvariantLayout layout;
  layout.physicalStateDim = kBallStateDim;
  layout.physicalInputDim = kBallInputDim;
  layout.candidates.push_back(ActivationCandidate{
      .name = "ball", .forceOffset = 0, .wrenchDim = kForceDim});
  return layout;
}

TerrainModel ballFloor() {
  config::TerrainScene::Box box;
  box.center = Eigen::Vector3d(0.0, 0.0, -0.5);
  box.half_extents = Eigen::Vector3d(10.0, 10.0, 0.5);
  box.orientation_xyzw = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
  config::TerrainScene::Primitive floor;
  floor.name = "floor";
  floor.friction = kBallFriction;
  floor.shape = box;
  config::TerrainScene scene;
  scene.name = "ball_floor";
  scene.primitives.push_back(floor);
  return *TerrainModel::Create(scene);
}

absl::StatusOr<std::unique_ptr<ContactFeatureFunction>> ballFeatures(
    const TerrainModel& terrain,
    const std::string& libraryName,
    const std::string& folder) {
  const ContactFeatureFunction::Function function =
      [terrain](const ad_vector_t& state,
                const ad_vector_t& input) -> ad_vector_t {
    const AdVector3 center = state.head<kPositionDim>();
    const AdVector3 velocity = state.segment<kPositionDim>(kPositionDim);
    const AdVector3 force = input.head<kPositionDim>();
    const ad_scalar_t weight(kBallMass * kGravitationalAcceleration);
    const AdVector3 pad =
        center - ad_scalar_t(kBallRadius) * AdVector3::UnitZ();
    const AdVector3 normal = terrain.normal<ad_scalar_t>(pad);
    ad_vector_t features = ad_vector_t::Zero(numFeatureRows(1));
    features(featureRow(/*candidate=*/0, kDistanceFeature)) =
        terrain.signedDistance<ad_scalar_t>(pad);
    features.segment<kPositionDim>(featureRow(/*candidate=*/0, kSlipFeature)) =
        velocity - velocity.dot(normal) * normal;
    const AdVector3 scaled = force / weight;
    const ad_scalar_t normalForce = scaled.dot(normal);
    const AdVector3 tangential = scaled - normalForce * normal;
    features(featureRow(/*candidate=*/0, kConeFeature)) =
        ad_scalar_t(kBallFriction) * normalForce -
        CppAD::sqrt(tangential.squaredNorm() +
                    ad_scalar_t(kBallConeSmoothing * kBallConeSmoothing));
    features(featureRow(/*candidate=*/0, kClearanceFeature)) =
        terrain.sphereDistance<ad_scalar_t>(center, ad_scalar_t(kBallRadius));
    features.segment<kPositionDim>(momentumRateRow(1)) =
        force / ad_scalar_t(kBallMass) -
        ad_scalar_t(kGravitationalAcceleration) * AdVector3::UnitZ();
    return features;
  };
  return ContactFeatureFunction::Create(
      function, kBallStateDim, kBallInputDim, /*numCandidates=*/1,
      FeatureLibrarySettings{
          .name = libraryName, .folder = folder, .recompile = true});
}

vector_t BallDynamics::computeFlowMap(scalar_t /*time*/,
                                      const vector_t& state,
                                      const vector_t& input,
                                      const PreComputation& /*preComp*/) {
  vector_t flow(kBallStateDim);
  flow.head<kPositionDim>() = state.tail<kPositionDim>();
  flow.tail<kPositionDim>() = input / kBallMass;
  flow(kBallStateDim - 1) -= kGravitationalAcceleration;
  return flow;
}

VectorFunctionLinearApproximation BallDynamics::linearApproximation(
    scalar_t time,
    const vector_t& state,
    const vector_t& input,
    const PreComputation& preComp) {
  VectorFunctionLinearApproximation result;
  result.setZero(kBallStateDim, kBallStateDim, kBallInputDim);
  result.f = computeFlowMap(time, state, input, preComp);
  result.dfdx.topRightCorner<kPositionDim, kPositionDim>().setIdentity();
  result.dfdu.bottomRows<kPositionDim>().diagonal().setConstant(1.0 /
                                                                kBallMass);
  return result;
}

BallGoalCost::BallGoalCost(vector_t goal, scalar_t weight)
    : goal_(std::move(goal)), weight_(weight) {}

scalar_t BallGoalCost::getValue(scalar_t /*time*/,
                                const vector_t& state,
                                const TargetTrajectories& /*targets*/,
                                const PreComputation& /*preComp*/) const {
  return weight_ * (state.head(goal_.size()) - goal_).squaredNorm();
}

ScalarFunctionQuadraticApproximation BallGoalCost::getQuadraticApproximation(
    scalar_t /*time*/,
    const vector_t& state,
    const TargetTrajectories& /*targets*/,
    const PreComputation& /*preComp*/) const {
  const vector_t error = state.head(goal_.size()) - goal_;
  ScalarFunctionQuadraticApproximation result;
  result.setZero(static_cast<int>(state.size()));
  result.f = weight_ * error.squaredNorm();
  result.dfdx.head(goal_.size()) = 2.0 * weight_ * error;
  result.dfdxx.topLeftCorner(goal_.size(), goal_.size())
      .diagonal()
      .setConstant(2.0 * weight_);
  return result;
}

OptimalControlProblem ballPhysicalProblem(const vector_t& goal,
                                          scalar_t goalWeight) {
  OptimalControlProblem problem;
  problem.dynamicsPtr = std::make_unique<BallDynamics>();
  problem.finalCostPtr->add("goal",
                            std::make_unique<BallGoalCost>(goal, goalWeight));
  return problem;
}

}  // namespace ocs2::humanoid::multi_contact
