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

// The terms planner A adds on the DRC Atlas's multi-contact model (design
// sections 4.2 and 4.3) against finite differences of their values and
// against the stance model of the certifier: the contact features (distance,
// slip, tilt, cone, clearance and momentum rate of the pads), the collision
// spheres' terrain clearance, the quasi-static joint torques and the support
// polygon constraint.

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_core/PhysicalConstants.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/Types.h"

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_multi_contact/collision/CollisionModel.h"
#include "humanoid_multi_contact/planner/contact_invariant/ContactFeatureRows.h"
#include "humanoid_multi_contact/planner/contact_invariant/HardContactRefinement.h"
#include "humanoid_multi_contact/planner/contact_invariant/PlanSeeding.h"
#include "humanoid_multi_contact/planner/contact_invariant/QuasiStaticTorqueConstraint.h"
#include "humanoid_multi_contact/planner/contact_invariant/RobotContactFeatures.h"
#include "humanoid_multi_contact/planner/contact_invariant/SupportPolygonConstraint.h"
#include "humanoid_multi_contact/planner/contact_invariant/TerrainClearanceConstraint.h"
#include "humanoid_multi_contact/scenarios/Scenario.h"
#include "humanoid_multi_contact/stability/JointLimits.h"
#include "humanoid_multi_contact/stability/RobotStanceModel.h"

namespace ocs2::humanoid::multi_contact {
namespace {

constexpr char kStairs[] =
    "humanoid_nmpc/humanoid_multi_contact/scenarios/config/stairs.textproto";
constexpr double kStep = 1.0e-6;
constexpr double kJacobianTolerance = 1.0e-5;

/** The stairs scenario and the MPC's models of planner A. */
struct AtlasModels {
  LoadedScenario scenario;
  std::unique_ptr<CentroidalMpcInterface> models;
  std::unique_ptr<RobotStanceModel> stanceModel;
};

AtlasModels atlasModels() {
  absl::StatusOr<LoadedScenario> scenario = loadScenario(kStairs);
  EXPECT_TRUE(scenario.ok()) << scenario.status();
  absl::StatusOr<CentroidalMpcConfig> config = plannerMpcConfigOf(*scenario);
  EXPECT_TRUE(config.ok()) << config.status();
  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> models =
      CentroidalMpcInterface::CreateControllerModels(*config,
                                                     scenario->urdfPath);
  EXPECT_TRUE(models.ok()) << models.status();
  absl::StatusOr<std::unique_ptr<RobotStanceModel>> stanceModel =
      RobotStanceModel::Create(scenario->urdfPath, scenario->contactSet);
  EXPECT_TRUE(stanceModel.ok()) << stanceModel.status();
  return AtlasModels{.scenario = *std::move(scenario),
                     .models = *std::move(models),
                     .stanceModel = *std::move(stanceModel)};
}

std::string cppAdFolder() {
  return (std::filesystem::path(::testing::TempDir()) / "cppad").string();
}

/** The MPC state of the stairs' initial pose, perturbed so that no term sits
 * at a symmetric point. */
vector_t perturbedState(const AtlasModels& atlas) {
  const MpcRobotModelBase<scalar_t>& model = atlas.models->getMpcRobotModel();
  const absl::StatusOr<RobotConfiguration> initial =
      configurationOf(atlas.scenario.scenario.initial_pose);
  EXPECT_TRUE(initial.ok()) << initial.status();
  vector_t state = mpcStateOf(model, *initial);
  for (Eigen::Index i = 0; i < state.size(); ++i) {
    state(i) += 0.01 * std::sin(1.0 + 0.7 * static_cast<double>(i));
  }
  state.segment<4>(model.getBaseOrientationStartIndex()).normalize();
  return state;
}

/** An input with forces on the feet and some joint velocities. */
vector_t someInput(const AtlasModels& atlas) {
  const MpcRobotModelBase<scalar_t>& model = atlas.models->getMpcRobotModel();
  vector_t input = vector_t::Zero(model.getInputDim());
  for (Eigen::Index i = 0; i < input.size(); ++i) {
    input(i) = 0.1 * std::cos(0.3 * static_cast<double>(i));
  }
  const std::vector<std::string>& names =
      atlas.models->modelSettings().contactNames;
  const double weight = atlas.models->getCentroidalModelInfo().robotMass *
                        kGravitationalAcceleration;
  for (size_t i = 0; i < names.size(); ++i) {
    if (names[i] == "foot_l_contact" || names[i] == "foot_r_contact") {
      model.setContactForce(input, Eigen::Vector3d(10.0, -5.0, 0.5 * weight),
                            i);
    }
  }
  return input;
}

/** someInput() with the forces of the feet only. */
vector_t feetOnlyInput(const AtlasModels& atlas) {
  const MpcRobotModelBase<scalar_t>& model = atlas.models->getMpcRobotModel();
  vector_t input = someInput(atlas);
  const std::vector<std::string>& names =
      atlas.models->modelSettings().contactNames;
  for (size_t i = 0; i < names.size(); ++i) {
    if (names[i] != "foot_l_contact" && names[i] != "foot_r_contact") {
      input
          .segment(model.getContactWrenchStartIndices(i),
                   model.getContactInputDim(i))
          .setZero();
    } else {
      model.setContactMoment(input, Eigen::Vector3d::Zero(), i);
    }
  }
  return input;
}

/** Central differences of `value` in `point`'s entries. */
matrix_t numericJacobian(const std::function<vector_t(const vector_t&)>& value,
                         const vector_t& point) {
  const vector_t center = value(point);
  matrix_t jacobian(center.size(), point.size());
  for (Eigen::Index i = 0; i < point.size(); ++i) {
    vector_t plus = point;
    vector_t minus = point;
    plus(i) += kStep;
    minus(i) -= kStep;
    jacobian.col(i) = (value(plus) - value(minus)) / (2.0 * kStep);
  }
  return jacobian;
}

/** Expects `analytic` to match `numeric`, entry by entry, relative to its
 * scale. */
void expectJacobian(const matrix_t& analytic, const matrix_t& numeric) {
  ASSERT_EQ(analytic.rows(), numeric.rows());
  ASSERT_EQ(analytic.cols(), numeric.cols());
  const double scale = 1.0 + numeric.cwiseAbs().maxCoeff();
  EXPECT_LT((analytic - numeric).cwiseAbs().maxCoeff(),
            kJacobianTolerance * scale);
}

TEST(RobotContactInvariantTerms, TheInitialStateStandsOnTheFloor) {
  const AtlasModels atlas = atlasModels();
  const MpcRobotModelBase<scalar_t>& model = atlas.models->getMpcRobotModel();
  const absl::StatusOr<RobotConfiguration> initial =
      configurationOf(atlas.scenario.scenario.initial_pose);
  ASSERT_TRUE(initial.ok()) << initial.status();
  const vector_t state = mpcStateOf(model, *initial);
  RobotConfiguration configuration;
  configuration.basePosition = model.getBasePosition(state);
  configuration.baseOrientation.coeffs() = model.getBaseOrientation(state);
  configuration.jointNames = model.modelSettings.mpcModelJointNames;
  configuration.jointPositions = model.getJointAngles(state);
  ASSERT_TRUE(atlas.stanceModel->setConfiguration(configuration).ok());
  for (const char* absl_nonnull name : {"foot_l_contact", "foot_r_contact"}) {
    const Eigen::Vector3d pad = atlas.stanceModel->padPosition(
        *atlas.scenario.contactSet.indexOf(name));
    EXPECT_NEAR(pad.z(), 0.0, 1.0e-3)
        << name << " " << pad.transpose() << "\nstate " << state.transpose();
    EXPECT_LT(std::abs(pad.x()), 0.1) << name;
  }
}

TEST(RobotContactInvariantTerms, TheFeaturesMatchFiniteDifferences) {
  const AtlasModels atlas = atlasModels();
  const std::vector<std::string>& names =
      atlas.models->modelSettings().contactNames;
  std::vector<RobotFeatureCandidate> candidates;
  for (const char* absl_nonnull name : {"foot_l_contact", "foot_r_contact"}) {
    const int index = *atlas.scenario.contactSet.indexOf(name);
    const ContactCandidate& candidate =
        atlas.scenario.contactSet.candidate(index);
    candidates.push_back(RobotFeatureCandidate{
        .frameName = name,
        .mpcContactIndex = static_cast<size_t>(
            std::find(names.begin(), names.end(), name) - names.begin()),
        .kind = candidate.kind,
        .padRotation = candidate.padRotation,
        .geomRadius = candidate.geomRadius});
  }
  absl::StatusOr<std::unique_ptr<ContactFeatureFunction>> features =
      robotContactFeatures(
          atlas.models->getPinocchioInterface(),
          atlas.models->getCentroidalModelInfo(), candidates,
          atlas.scenario.terrain,
          RobotFeatureSettings{
              .friction = 0.6, .coneSmoothing = 0.01, .maxPointTilt = 0.5},
          FeatureLibrarySettings{.name = "test_atlas_features",
                                 .folder = cppAdFolder(),
                                 .recompile = true});
  ASSERT_TRUE(features.ok()) << features.status();
  const vector_t state = perturbedState(atlas);
  const vector_t input = someInput(atlas);
  const VectorFunctionLinearApproximation linear =
      (*features)->linearApproximation(state, input);
  expectJacobian(linear.dfdx, numericJacobian(
                                  [&](const vector_t& x) {
                                    return (*features)->value(x, input);
                                  },
                                  state));
  expectJacobian(linear.dfdu, numericJacobian(
                                  [&](const vector_t& u) {
                                    return (*features)->value(state, u);
                                  },
                                  input));
  // Against the stance model: the pads' distance to the floor.
  const MpcRobotModelBase<scalar_t>& model = atlas.models->getMpcRobotModel();
  RobotConfiguration configuration;
  configuration.basePosition = model.getBasePosition(state);
  configuration.baseOrientation.coeffs() = model.getBaseOrientation(state);
  configuration.jointNames = model.modelSettings.mpcModelJointNames;
  configuration.jointPositions = model.getJointAngles(state);
  ASSERT_TRUE(atlas.stanceModel->setConfiguration(configuration).ok());
  for (int i = 0; i < 2; ++i) {
    const int index =
        *atlas.scenario.contactSet.indexOf(candidates[i].frameName);
    EXPECT_NEAR(linear.f(featureRow(i, kDistanceFeature)),
                atlas.scenario.terrain.signedDistance<double>(
                    atlas.stanceModel->padPosition(index)),
                1.0e-6);
  }
}

TEST(RobotContactInvariantTerms, TheMomentumRateIsTheForcesOverTheMass) {
  const AtlasModels atlas = atlasModels();
  absl::StatusOr<std::unique_ptr<ContactFeatureFunction>> features =
      robotContactFeatures(
          atlas.models->getPinocchioInterface(),
          atlas.models->getCentroidalModelInfo(), /*candidates=*/{},
          atlas.scenario.terrain,
          RobotFeatureSettings{
              .friction = 0.6, .coneSmoothing = 0.01, .maxPointTilt = 0.5},
          FeatureLibrarySettings{.name = "test_atlas_momentum_rate",
                                 .folder = cppAdFolder(),
                                 .recompile = true});
  ASSERT_TRUE(features.ok()) << features.status();
  const vector_t state = perturbedState(atlas);
  const vector_t input = feetOnlyInput(atlas);
  const vector_t value = (*features)->value(state, input);
  const double mass = atlas.models->getCentroidalModelInfo().robotMass;
  const Eigen::Vector3d linear = value.segment<3>(momentumRateRow(0));
  // Two feet of (10, -5, w / 2) each hold the weight: (20, -10, 0) / m.
  EXPECT_NEAR(linear.x(), 20.0 / mass, 1.0e-9);
  EXPECT_NEAR(linear.y(), -10.0 / mass, 1.0e-9);
  EXPECT_NEAR(linear.z(), 0.0, 1.0e-9);
}

TEST(RobotContactInvariantTerms, TheSphereClearanceIsTheStanceModelsSpheres) {
  const AtlasModels atlas = atlasModels();
  absl::StatusOr<CollisionModel> collision = CollisionModel::Create(
      *atlas.scenario.collisionModelConfig, *atlas.stanceModel);
  ASSERT_TRUE(collision.ok()) << collision.status();
  absl::StatusOr<std::vector<ClearanceSphere>> spheres = clearanceSpheresOf(
      *collision, *atlas.stanceModel, atlas.models->getPinocchioInterface());
  ASSERT_TRUE(spheres.ok()) << spheres.status();
  const TerrainClearanceConstraint constraint(
      atlas.models->getPinocchioInterface(),
      atlas.models->getCentroidalModelInfo(), *spheres, atlas.scenario.terrain,
      /*margin=*/0.0, "test_atlas_clearance", cppAdFolder(),
      /*recompile=*/true, /*verbose=*/false);
  const vector_t state = perturbedState(atlas);
  const PreComputation pre;
  const VectorFunctionLinearApproximation linear =
      constraint.getLinearApproximation(/*time=*/0.0, state, pre);
  ASSERT_EQ(linear.f.size(), static_cast<Eigen::Index>(spheres->size()));
  expectJacobian(linear.dfdx, numericJacobian(
                                  [&](const vector_t& x) {
                                    return constraint.getValue(/*time=*/0.0, x,
                                                               pre);
                                  },
                                  state));
  const MpcRobotModelBase<scalar_t>& model = atlas.models->getMpcRobotModel();
  RobotConfiguration configuration;
  configuration.basePosition = model.getBasePosition(state);
  configuration.baseOrientation.coeffs() = model.getBaseOrientation(state);
  configuration.jointNames = model.modelSettings.mpcModelJointNames;
  configuration.jointPositions = model.getJointAngles(state);
  ASSERT_TRUE(atlas.stanceModel->setConfiguration(configuration).ok());
  for (size_t k = 0; k < spheres->size(); ++k) {
    const double expected = atlas.scenario.terrain.sphereDistance<double>(
        collision->worldCenter(*atlas.stanceModel, static_cast<int>(k)),
        collision->spheres()[k].radius);
    EXPECT_NEAR(linear.f(k), expected, 1.0e-6) << "sphere " << k;
  }
}

TEST(RobotContactInvariantTerms, TheTorquesAreGravityLessTheContactWrenches) {
  const AtlasModels atlas = atlasModels();
  const pinocchio::Model& mpcModel =
      atlas.models->getPinocchioInterface().getModel();
  const ModelSettings& settings = atlas.models->modelSettings();
  std::vector<JointEffortLimit> limits;
  std::vector<std::string> limitedJoints;
  for (const std::string& name : settings.mpcModelJointNames) {
    const int index = mpcModel.joints[mpcModel.getJointId(name)].idx_v();
    if (mpcModel.effortLimit(index) > 0.0) {
      limits.push_back(
          JointEffortLimit{.velocityIndex = static_cast<size_t>(index),
                           .effort = mpcModel.effortLimit(index)});
      limitedJoints.push_back(name);
    }
  }
  ASSERT_FALSE(limits.empty());
  const QuasiStaticTorqueConstraint constraint(
      atlas.models->getPinocchioInterface(),
      atlas.models->getCentroidalModelInfo(), settings.contactNames, limits,
      /*margin=*/0.9, "test_atlas_torques", cppAdFolder(),
      /*recompile=*/true, /*verbose=*/false);
  const vector_t state = perturbedState(atlas);
  const vector_t input = someInput(atlas);
  const PreComputation pre;
  const VectorFunctionLinearApproximation linear =
      constraint.getLinearApproximation(/*time=*/0.0, state, input, pre);
  expectJacobian(linear.dfdx, numericJacobian(
                                  [&](const vector_t& x) {
                                    return constraint.getValue(/*time=*/0.0, x,
                                                               input, pre);
                                  },
                                  state));
  expectJacobian(linear.dfdu, numericJacobian(
                                  [&](const vector_t& u) {
                                    return constraint.getValue(/*time=*/0.0,
                                                               state, u, pre);
                                  },
                                  input));
  // Against the stance model: g(q) - J_p' f - J_w' tau at the feet.
  const MpcRobotModelBase<scalar_t>& model = atlas.models->getMpcRobotModel();
  RobotConfiguration configuration;
  configuration.basePosition = model.getBasePosition(state);
  configuration.baseOrientation.coeffs() = model.getBaseOrientation(state);
  configuration.jointNames = model.modelSettings.mpcModelJointNames;
  configuration.jointPositions = model.getJointAngles(state);
  RobotStanceModel& stance = *atlas.stanceModel;
  ASSERT_TRUE(stance.setConfiguration(configuration).ok());
  Eigen::VectorXd tau = stance.generalizedGravity();
  for (size_t i = 0; i < settings.contactNames.size(); ++i) {
    const int index =
        *atlas.scenario.contactSet.indexOf(settings.contactNames[i]);
    const Eigen::Vector3d point = stance.padPosition(index);
    tau -= stance.pointJacobian(index, point).transpose() *
           model.getContactForce(input, i);
    if (atlas.scenario.contactSet.candidate(index).kind ==
        ContactKind::kSurface) {
      tau -= stance.jointAngularJacobian(stance.padJoint(index)).transpose() *
             model.getContactMoment(input, i);
    }
  }
  const vector_t torques = constraint.torques(state, input);
  for (size_t j = 0; j < limits.size(); ++j) {
    const int joint = static_cast<int>(*stance.jointIndex(limitedJoints[j]));
    const int velocity = stance.model().joints[joint].idx_v();
    EXPECT_NEAR(torques(j), tau(velocity),
                1.0e-6 * (1.0 + std::abs(tau(velocity))))
        << limitedJoints[j];
    EXPECT_NEAR(linear.f(j), 0.9 - torques(j) / limits[j].effort, 1.0e-9);
    EXPECT_NEAR(linear.f(limits.size() + j),
                0.9 + torques(j) / limits[j].effort, 1.0e-9);
  }
}

TEST(RobotContactInvariantTerms, SolveThreeConstrainsTheEffortLimitsByName) {
  const AtlasModels atlas = atlasModels();
  const pinocchio::Model& mpcModel =
      atlas.models->getPinocchioInterface().getModel();
  const ModelSettings& settings = atlas.models->modelSettings();
  const std::vector<JointEffortLimit> enforced = constrainedEffortLimits(
      JointLimits::kPositionAndEffort, mpcModel, settings);
  // Every joint of the MPC with an effort limit, in the MPC's order.
  std::vector<size_t> expected;
  for (const std::string& name : settings.mpcModelJointNames) {
    const int index = mpcModel.joints[mpcModel.getJointId(name)].idx_v();
    if (mpcModel.effortLimit(index) > 0.0) {
      expected.push_back(static_cast<size_t>(index));
    }
  }
  ASSERT_EQ(enforced.size(), expected.size());
  ASSERT_FALSE(enforced.empty());
  for (size_t j = 0; j < enforced.size(); ++j) {
    EXPECT_EQ(enforced[j].velocityIndex, expected[j]);
    EXPECT_EQ(enforced[j].effort,
              mpcModel.effortLimit(static_cast<int>(expected[j])));
  }
  // Under "position_only" solve 3 adds no torque constraint.
  EXPECT_TRUE(
      constrainedEffortLimits(JointLimits::kPositionOnly, mpcModel, settings)
          .empty());
}

TEST(RobotContactInvariantTerms, ThePolygonRowsAreTheEdgesLessTheMargin) {
  const AtlasModels atlas = atlasModels();
  // A unit square about the origin, counterclockwise.
  const absl::StatusOr<TimedPolygon> square =
      polygonOf({Eigen::Vector2d(-1.0, -1.0), Eigen::Vector2d(1.0, -1.0),
                 Eigen::Vector2d(1.0, 1.0), Eigen::Vector2d(-1.0, 1.0)},
                /*startTime=*/0.0, /*endTime=*/1.0);
  ASSERT_TRUE(square.ok()) << square.status();
  EXPECT_FALSE(
      polygonOf({Eigen::Vector2d(-1.0, -1.0), Eigen::Vector2d(-1.0, 1.0),
                 Eigen::Vector2d(1.0, 1.0), Eigen::Vector2d(1.0, -1.0)},
                /*startTime=*/0.0, /*endTime=*/1.0)
          .ok())
      << "a clockwise polygon";
  const SupportPolygonConstraint constraint(
      atlas.models->getPinocchioInterface(),
      atlas.models->getCentroidalModelInfo(), {*square}, /*margin=*/0.1,
      "test_atlas_polygon", cppAdFolder(), /*recompile=*/true,
      /*verbose=*/false);
  EXPECT_TRUE(constraint.isActive(0.5));
  EXPECT_FALSE(constraint.isActive(1.5));
  const vector_t state = perturbedState(atlas);
  const PreComputation pre;
  const VectorFunctionLinearApproximation linear =
      constraint.getLinearApproximation(/*time=*/0.5, state, pre);
  ASSERT_EQ(linear.f.size(), kMaxPolygonEdges);
  expectJacobian(linear.dfdx, numericJacobian(
                                  [&](const vector_t& x) {
                                    return constraint.getValue(/*time=*/0.5, x,
                                                               pre);
                                  },
                                  state));
  const MpcRobotModelBase<scalar_t>& model = atlas.models->getMpcRobotModel();
  RobotConfiguration configuration;
  configuration.basePosition = model.getBasePosition(state);
  configuration.baseOrientation.coeffs() = model.getBaseOrientation(state);
  configuration.jointNames = model.modelSettings.mpcModelJointNames;
  configuration.jointPositions = model.getJointAngles(state);
  ASSERT_TRUE(atlas.stanceModel->setConfiguration(configuration).ok());
  const Eigen::Vector3d com = atlas.stanceModel->centerOfMass();
  // The edges: +y at y = -1 turned..., each b - margin - a . c.
  EXPECT_NEAR(linear.f(0), 1.0 - 0.1 + com.y(), 1.0e-6);  // bottom, a = (0, -1)
  EXPECT_NEAR(linear.f(1), 1.0 - 0.1 - com.x(), 1.0e-6);  // right, a = (1, 0)
  EXPECT_NEAR(linear.f(2), 1.0 - 0.1 - com.y(), 1.0e-6);  // top, a = (0, 1)
  EXPECT_NEAR(linear.f(3), 1.0 - 0.1 + com.x(), 1.0e-6);  // left, a = (-1, 0)
  for (int k = 4; k < kMaxPolygonEdges; ++k) {
    EXPECT_NEAR(linear.f(k), 1.0 - 0.1, 1.0e-12) << "an unused row";
  }
}

/** Eleven knots a second apart: both feet, the right foot from knot 3, both
 * feet again from knot 7. */
config::MultiContactPlan threeStancePlan() {
  constexpr ContactMask kBothFeet = 3;
  constexpr ContactMask kRightFoot = 1;
  config::MultiContactPlan plan;
  for (int k = 0; k <= 10; ++k) {
    config::MultiContactPlan::Knot knot;
    knot.time = static_cast<double>(k);
    knot.active_contacts = k >= 3 && k < 7 ? kRightFoot : kBothFeet;
    plan.knots.push_back(knot);
  }
  return plan;
}

TEST(RobotContactInvariantTerms, RetimingRoundsTheChangesToTheQuantum) {
  // Runs of 3, 4 and 3 s scaled to 1.17 s: the changes at 0.351 s and
  // 0.819 s, rounded to 0.4 s and 0.8 s by a 0.1 s quantum.
  const config::MultiContactPlan plan = threeStancePlan();
  const absl::StatusOr<config::MultiContactPlan> exact = retimedPlan(
      plan, /*duration=*/1.17, /*minimumRun=*/0.1, /*changeQuantum=*/0.0);
  ASSERT_TRUE(exact.ok()) << exact.status();
  EXPECT_NEAR(exact->knots[3].time, 0.351, 1.0e-6);
  EXPECT_NEAR(exact->knots[7].time, 0.819, 1.0e-6);
  const absl::StatusOr<config::MultiContactPlan> quantized = retimedPlan(
      plan, /*duration=*/1.17, /*minimumRun=*/0.1, /*changeQuantum=*/0.1);
  ASSERT_TRUE(quantized.ok()) << quantized.status();
  EXPECT_NEAR(quantized->knots[3].time, 0.4, 1.0e-9);
  EXPECT_NEAR(quantized->knots[7].time, 0.8, 1.0e-9);
  EXPECT_NEAR(quantized->knots.back().time, 1.17, 1.0e-6);
  // Within a run the knots keep their shares: knot 5 halves 0.4 to 0.8 s.
  EXPECT_NEAR(quantized->knots[5].time, 0.6, 1.0e-9);
}

}  // namespace
}  // namespace ocs2::humanoid::multi_contact
